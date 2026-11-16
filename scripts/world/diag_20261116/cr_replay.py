# 절차 2: cr+replay (12.13 위에), 설계도 docs/HANDOVER.md 0.3절.
#   12.13 구성 그대로 (더하는 Q, 난수 키, 주소 극 {0, 8}, 키 128, 손 선호 판정, 시드 1, 세계 256개).
#   σ², R, q는 12.13 기록 (eb_process_noise.log)에서 읽고 다시 맞추지 않는다 (순전파만).
#   세션 저장소 φ = unit(ReLU(W_φ x)), W_φ ∈ R^{1024×256} ~ N(0, 1/256) 고정, S ∈ R^{8×1024}, 0에서 시작.
#   시냅스는 같은 칼만 이득으로 M_r (r)과 M_φ (φ)를 함께 쓰고, 시냅스 몫 M_t = M_r − S M_φ.
#   합친 예측 ŷ = M_r z + S (φ − M_φ z). 깨어 있을 때 S ← S + η_a (r − Sφ) φᵀ.
#   잠: K스텝마다 (C 도착까지) + 교사 세션 끝 (C 도착 뒤, 판정 전) 한 번. 버퍼 (C 도착까지의 x_j, φ_j)를 무작위 순서로 n번,
#       S ← S + η_s (M_t z_j) φ_jᵀ (M_t z_j는 현재 S로 매번 다시 계산).
# 구현 메모 (결과는 설계도와 같음):
#   - 도착 칸 (SLOT)만 계산한다. 칸끼리 독립이고 판정·기록은 도착 칸만 읽는다 (설계도가 허용).
#   - 잠 안에서는 S만 바뀌므로 M_r z_j, M_φ z_j를 잠 시작에 행렬곱으로 한 번 계산하고, 순차 갱신은 S에만.
#   - K스텝 묶음 단위로 scan하고 묶음 끝에 잠 (vmap 아래 lax.cond는 두 갈래를 다 실행하므로). C 도착 뒤의 묶음 잠은 η = 0.
# 환경변수 STORE=0: 저장소 없음 (S를 만들지 않음, 읽기 iii-a = 12.13 재현, 84.4%가 나와야 함).
import sys, json, os; sys.path.insert(0, '/workspace/RL-TDP')
import jax, jax.numpy as jnp, numpy as np
from rltdp.world import errand as ER, errand_learned as EL, errand_prec as EP, errand_reuse as RU
jax.config.update("jax_default_matmul_precision", "highest")
ER.PUBLIC = False
We, ybar = EL.emb_fixed(0)
p, vmax = EP.train("random", seed=0, D=256, iters_k=0, log=lambda m: None)
feat, wm, key0 = EP.parts(p, We, ybar, "random")
HL = [0.0, 8.0]
AL = jnp.array([0.0 if x == 0 else 2.0 ** (-1 / x) for x in HL])
KD = EP.DK
DZ = len(HL) * KD
SL = EP.SLOT


def key(o, a):
    k = key0(o, a)[:KD]
    return k / (jnp.linalg.norm(k) + 1e-8)


itm = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
gB = itm[1] - itm.mean(0)

STORE = os.environ.get('STORE', '1') == '1'
DF, ETA_A, ETA_S, K, NPASS = 1024, 0.05, 0.05, 64, 5
if not STORE:
    ETA_A = ETA_S = 0.0
H1213 = 'docs/world/results/errand_diag_20261115b/eb_process_noise.log'
prev = json.loads(open('/workspace/RL-TDP/' + H1213).read().strip().splitlines()[-1])
SG = float(np.sqrt(prev["sigma2"][SL])); RP = float(prev["R"][SL]); QP = float(prev["q"][SL])
WPHI = jax.random.normal(jax.random.PRNGKey(20261116), (DF, DZ)) / np.sqrt(DZ)


def phif(x):
    v = jax.nn.relu(WPHI @ x)
    return v / (jnp.linalg.norm(v) + 1e-8)


def session(sess, plans, ai_all, rk):
    O, s_end, _ = EL.observe_stream(sess)
    T = sess["acts"].shape[0]
    t_c = RU.last_arrival(O)
    acts = sess["acts"]

    def fx(e, inp):
        o, a = inp
        e = AL[:, None] * e + key(o, a)
        return e, e.reshape(-1)
    e_end, X = jax.lax.scan(fx, jnp.zeros((len(HL), KD)), (O[:-1], acts))
    Phi = jax.vmap(phif)(X)
    Z = SG * X
    mu = jax.vmap(lambda o, a: wm(o, a)[0][SL])(O[:-1], acts)
    Rr = jax.vmap(lambda o: feat(o)[SL])(O[1:]) - mu                     # 잔차 r (T, 8)
    blank = feat(ER.TEX[jnp.full(12, ER.BLANK)].reshape(12, -1))[SL]
    arrived = ~jnp.all(jnp.isclose(jax.vmap(lambda o: feat(o)[SL])(O[1:]), blank, atol=1e-5), -1)
    tt = jnp.arange(T)
    written = tt <= t_c

    NC = -(-T // K)                                                      # 묶음 수 (끝은 패딩)
    TP = NC * K
    padT = lambda A: jnp.concatenate([A, jnp.repeat(A[-1:], TP - T, 0)])
    Zp, Phip, Rp_, Wp = padT(Z), padT(Phi), padT(Rr), jnp.concatenate([written, jnp.zeros(TP - T, bool)])
    bufmask0 = jnp.concatenate([written, jnp.zeros(TP - T, bool)])      # 버퍼에 드는 스텝 (C 도착까지)
    arrp = jnp.concatenate([arrived & written, jnp.zeros(TP - T, bool)])

    def shares(Mr, Mphi, S, upto):
        m = arrp & (jnp.arange(TP) < upto)
        Mt_z = Zp @ Mr.T - (Zp @ Mphi.T) @ S.T                           # (TP, 8)
        Sphi = Phip @ S.T
        n = jnp.maximum(m.sum(), 1)
        return jnp.stack([jnp.sum(jnp.where(m, jnp.linalg.norm(Mt_z, axis=-1), 0)) / n,
                          jnp.sum(jnp.where(m, jnp.linalg.norm(Sphi, axis=-1), 0)) / n, m.sum().astype(jnp.float32)])

    def sleep(S, Mr, Mphi, upto, eta, k):
        mask = bufmask0 & (jnp.arange(TP) < upto)
        U = Zp @ Mr.T                                                     # M_r z_j (TP, 8)
        V = Zp @ Mphi.T                                                   # M_φ z_j (TP, 1024)

        def one_pass(S, kk):
            perm = jax.random.permutation(kk, TP)

            def st(S, j):
                err = U[j] - S @ V[j]                                     # M_t z_j, 현재 S로
                return S + jnp.where(mask[j], eta, 0.0) * jnp.outer(err, Phip[j]), None
            return jax.lax.scan(st, S, perm)[0], None
        return jax.lax.scan(one_pass, S, jax.random.split(k, NPASS))[0]

    def awake(c, inp):
        Mr, P, Mphi, S = c
        z, ph, r, w = inp
        P = P + QP * jnp.eye(DZ)
        Pz = P @ z; Sv = RP + z @ Pz; g = Pz / Sv
        Mr2 = Mr + jnp.outer(r - Mr @ z, g)
        Mphi2 = Mphi + jnp.outer(ph - Mphi @ z, g)
        P2 = P - jnp.outer(Pz, Pz) / Sv
        S2 = S + ETA_A * jnp.outer(r - S @ ph, ph)
        yh = Mr2 @ z + S2 @ (ph - Mphi2 @ z)                              # 쓴 뒤 합친 예측 (도착 순간 쓰인 몫)
        got = yh @ r / (r @ r + 1e-12)
        sel = lambda a, b: jnp.where(w, a, b)
        return (sel(Mr2, Mr), sel(P2, P), sel(Mphi2, Mphi), sel(S2, S)), got

    def chunk(c, ci):
        sl = lambda A: jax.lax.dynamic_slice_in_dim(A, ci * K, K)
        c, got = jax.lax.scan(awake, c, (sl(Zp), sl(Phip), sl(Rp_), sl(Wp)))
        Mr, P, Mphi, S = c
        t_end = ci * K + K - 1
        eta = jnp.where(t_end <= t_c, ETA_S, 0.0)
        before = shares(Mr, Mphi, S, t_end + 1)
        S = sleep(S, Mr, Mphi, t_end + 1, eta, jax.random.fold_in(rk, ci))
        after = shares(Mr, Mphi, S, t_end + 1)
        return (Mr, P, Mphi, S), (got, jnp.stack([before, after]), t_end <= t_c)

    c0 = (jnp.zeros((EP.H, DZ)), jnp.eye(DZ), jnp.zeros((DF, DZ)), jnp.zeros((EP.H, DF)))
    (Mr, P, Mphi, S), (got, sh, active) = jax.lax.scan(chunk, c0, jnp.arange(NC))
    got = got.reshape(-1)
    got_mean = jnp.sum(jnp.where(arrp, got, 0)) / jnp.maximum(arrp.sum(), 1)
    fin_before = shares(Mr, Mphi, S, t_c + 1)
    S = sleep(S, Mr, Mphi, t_c + 1, ETA_S, jax.random.fold_in(rk, 10_000))   # 세션 끝 잠
    fin_after = shares(Mr, Mphi, S, t_c + 1)

    def sim(plan, ai):
        def f(c, a):
            s, o, e = c
            e = AL[:, None] * e + key(o, a)
            x = e.reshape(-1); z = SG * x; ph = phif(x)
            Mz = Mr @ z; Mpz = Mphi @ z
            out = jnp.stack([Mz + S @ (ph - Mpz), S @ ph, Mz - S @ Mpz, Mz])  # (i) 합침, (ii) 저장소, (iii-b) 시냅스 몫, M_r z
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (s2, o2, e), out
        _, YH = jax.lax.scan(f, (s_end, O[-1], e_end), plan)
        yh = YH[ai]
        return (yh @ gB) / (jnp.linalg.norm(yh, axis=-1) * jnp.linalg.norm(gB) + 1e-8), jnp.linalg.norm(yh, axis=-1)
    sc, nrm = jax.vmap(sim)(plans, ai_all)                               # (5, 4)
    return sc, nrm, got_mean, sh, active, jnp.stack([fin_before, fin_after])


sessions, stack = EL.stack_sessions(range(1000, 1256), 256)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])
run = jax.jit(jax.vmap(session))
BS = 32
outs = []
for i in range(0, 256, BS):
    sb = jax.tree_util.tree_map(lambda x: x[i:i + BS], stack)
    rks = jax.vmap(lambda j: jax.random.fold_in(jax.random.PRNGKey(1), j))(jnp.arange(i, i + BS))
    outs.append([np.array(v) for v in run(sb, plans[i:i + BS], arrive[i:i + BS], rks)])
    print(f"batch {i // BS + 1}/{256 // BS}", flush=True)
sc, nrm, got, sh, active, fin = [np.concatenate([o[k] for o in outs]) for k in range(6)]

names = ["i_combined", "ii_store", "iiib_synapse_share", "Mr_z"]
res = dict(store=STORE, DF=DF, eta_a=ETA_A, eta_s=ETA_S, K=K, n=NPASS, sigma2=SG ** 2, R=RP, q=QP,
           written=float(got.mean()))
for k, nm in enumerate(names):
    x = sc[:, :, k]
    best = (x + 1e-9 * np.random.default_rng(0).random(x.shape)).argmax(1)
    res[nm] = dict(b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)),
                   other=float(np.mean((best != b1) & (best != b0))), read_norm=float(nrm[:, :, k].mean()))
# 잠마다 도착 칸 몫 크기 (세계 평균, 그 잠이 든 세계만): [잠 전/후][시냅스 몫 |M_t z|, 저장소 몫 |Sφ|]
per = []
for ci in range(sh.shape[1]):
    m = active[:, ci]
    if m.sum() == 0:
        continue
    per.append(dict(sleep=ci, t_end=ci * K + K - 1, n_worlds=int(m.sum()),
                    before=dict(synapse=float(sh[m, ci, 0, 0].mean()), store=float(sh[m, ci, 0, 1].mean())),
                    after=dict(synapse=float(sh[m, ci, 1, 0].mean()), store=float(sh[m, ci, 1, 1].mean()))))
res["per_sleep"] = per
res["final_sleep"] = dict(before=dict(synapse=float(fin[:, 0, 0].mean()), store=float(fin[:, 0, 1].mean())),
                          after=dict(synapse=float(fin[:, 1, 0].mean()), store=float(fin[:, 1, 1].mean())),
                          n_arrivals=float(fin[:, 0, 2].mean()))
for nm in names:
    print(f"{nm:20s} b1 {res[nm]['b1']:.4f}  b0 {res[nm]['b0']:.4f}  other {res[nm]['other']:.4f}  |ŷ| {res[nm]['read_norm']:.4f}")
print("written", res["written"])
print("final sleep", res["final_sleep"])
print(json.dumps(res), flush=True)
