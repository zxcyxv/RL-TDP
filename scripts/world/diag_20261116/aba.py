# A → B → A 복귀 시험. 설계와 사전 예측: docs/world/aba_predictions.md (실행 전 고정).
#   한 세션 = 구역 A → 구역 B₁ … B_m (서로 다른 errand 세계) → A로 복귀. 기억 (시냅스, 저장소, 흔적)은 이어지고 구역 경계를 넘는 전이는 쓰지 않는다.
#   각 구역에서 교사가 관찰 2바퀴 (채집처 5곳 무작위 순서로 가서 상호작용 → 보급소로 돌아오며 도착까지 대기). 의뢰·변경 없음.
#   복귀 판정: A 보급소에서 채집처마다 "가서 하고 돌아오기"를 흉내 내 도착 칸 예측과 손 선호 g_C의 코사인으로 점수,
#   첫 보상까지 시도 수 = 점수 내림차순에서 C 채집처의 순위 (기억 없음 = 3.0).
#   기억은 12.16 (cr_replay.py, 더하는 Q) 그대로: σ², R, q는 12.13 도착 칸 값, φ 1024, η_a = η_s = 0.05, K = 64, n = 5, 판정 직전 잠 한 번. 도착 칸만 계산.
# argv: m (B 구역 수). 환경변수 STORE=0: 저장소·잠 없음 (시냅스만, m 정하기용).
# 환경변수 SMODE (진단, 용량 대 망각):
#   belief (기본) = 12.16 그대로, 잠의 목표 = 시냅스 몫 M_t z_j
#   raw   = 날것 재생: 버퍼에 r_j도 두고 잠에서 S ← S + η_s (r_j − S φ_j) φ_jᵀ (토이의 raw+replay, 시냅스 cr은 그대로)
#   ridge = 용량 상한: 깨어 있을 때·잠의 S 학습 없이, 세션 전체 (φ_j, r_j)로 S* = Rᵀ Φ (ΦᵀΦ + λI)⁻¹를 한 번에 풀어 읽음.
#           λ ∈ RIDGE_L 모두 잰다 (상한이라 가장 좋은 λ를 본다)
import sys, json, os; sys.path.insert(0, '/workspace/RL-TDP')
import jax, jax.numpy as jnp, numpy as np
from rltdp.world import errand as ER, errand_learned as EL, errand_prec as EP
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
M_B = int(sys.argv[1])
STORE = os.environ.get('STORE', '1') == '1'
ROUNDS = 2
DF, ETA_A, ETA_S, K, NPASS = 1024, 0.05, 0.05, 64, 5
SMODE = os.environ.get('SMODE', 'belief')
RIDGE_L = [1e-3, 1e-2, 1e-1, 1.0]
if not STORE:
    ETA_A = ETA_S = 0.0
if SMODE == 'ridge':
    ETA_A = 0.0
prev = json.loads(open('/workspace/RL-TDP/docs/world/results/errand_diag_20261115b/eb_process_noise.log').read().strip().splitlines()[-1])
SG = float(np.sqrt(prev["sigma2"][SL])); RP = float(prev["R"][SL]); QP = float(prev["q"][SL])
WPHI = jax.random.normal(jax.random.PRNGKey(20261116), (DF, DZ)) / np.sqrt(DZ)
itm = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
gC = itm[2] - itm.mean(0)
BLANK_F = feat(ER.TEX[jnp.full(12, ER.BLANK)].reshape(12, -1))[SL]


def key(o, a):
    k = key0(o, a)[:KD]
    return k / (jnp.linalg.norm(k) + 1e-8)


def phif(x):
    v = jax.nn.relu(WPHI @ x)
    return v / (jnp.linalg.norm(v) + 1e-8)


# ---------- 구역 (numpy) ----------
def area_session(seed):
    rng = np.random.default_rng(seed)
    tiles, depot, sites, prod, roles = ER.gen_world(rng)
    acts, pos = [], np.array(depot)
    for _ in range(ROUNDS):
        for s in rng.permutation(ER.NSITE):
            acts += ER.path(pos, sites[s]); pos = np.array(sites[s])
            acts.append(ER.INTERACT); t0 = len(acts)
            acts += ER.path(pos, depot); pos = np.array(depot)
            while len(acts) - t0 < ER.D_DELAY + 1:
                acts.append(ER.WAIT)
    acts = np.array(acts)
    return dict(tiles=tiles, depot=np.array(depot), sites=np.array(sites), prod=prod, b0=roles["b0"],
                acts=acts, req=np.full(len(acts), -1), rid=np.zeros(len(acts), int), change_t=-1)


def stack(sessions):
    L = max(len(s["acts"]) for s in sessions)
    out = {}
    for k in ("tiles", "depot", "sites", "prod", "b0", "acts", "req", "rid", "change_t"):
        v = []
        for s in sessions:
            x = np.asarray(s[k])
            if k in ("acts", "req", "rid"):
                fill = ER.WAIT if k == "acts" else x[-1]
                x = np.concatenate([x, np.full(L - len(x), fill)])
            v.append(x)
        out[k] = jnp.array(np.stack(v))
    return out


@jax.jit
@jax.vmap
def area_feats(sess):
    O, _, _ = EL.observe_stream(sess)
    acts = sess["acts"]
    Kk = jax.vmap(key)(O[:-1], acts)
    mu = jax.vmap(lambda o, a: wm(o, a)[0][SL])(O[:-1], acts)
    Y = jax.vmap(lambda o: feat(o)[SL])(O[1:])
    arr = ~jnp.all(jnp.isclose(Y, BLANK_F, atol=1e-5), -1)
    return Kk, Y - mu, arr


# ---------- 기억 (cr_replay.py와 같은 식) ----------
def memory_and_judge(Kk, Rr, arr, sessA, plans, ai_all, rk):
    T = Kk.shape[0]

    def fx(e, k):
        e = AL[:, None] * e + k
        return e, e.reshape(-1)
    e_end, X = jax.lax.scan(fx, jnp.zeros((len(HL), KD)), Kk)
    Phi = jax.vmap(phif)(X)
    Z = SG * X
    NC = -(-T // K)
    TP = NC * K
    pad = lambda A: jnp.concatenate([A, jnp.repeat(A[-1:], TP - T, 0)])
    Zp, Phip, Rp_ = pad(Z), pad(Phi), pad(Rr)
    Wp = jnp.arange(TP) < T
    arrp = jnp.concatenate([arr, jnp.zeros(TP - T, bool)])

    def shares(Mr, Mphi, S, upto):
        m = arrp & (jnp.arange(TP) < upto)
        Mt_z = Zp @ Mr.T - (Zp @ Mphi.T) @ S.T
        n = jnp.maximum(m.sum(), 1)
        return jnp.stack([jnp.sum(jnp.where(m, jnp.linalg.norm(Mt_z, axis=-1), 0)) / n,
                          jnp.sum(jnp.where(m, jnp.linalg.norm(Phip @ S.T, axis=-1), 0)) / n])

    def sleep(S, Mr, Mphi, upto, k):
        mask = Wp & (jnp.arange(TP) < upto)
        U = Zp @ Mr.T
        V = Zp @ Mphi.T

        def one_pass(S, kk):
            perm = jax.random.permutation(kk, TP)

            def st(S, j):
                err = (Rp_[j] - S @ Phip[j]) if SMODE == 'raw' else (U[j] - S @ V[j])
                return S + jnp.where(mask[j], ETA_S, 0.0) * jnp.outer(err, Phip[j]), None
            return jax.lax.scan(st, S, perm)[0], None
        return jax.lax.scan(one_pass, S, jax.random.split(k, NPASS))[0]

    def awake(c, inp):
        Mr, P, Mphi, S = c
        z, ph, r, w = inp
        P = P + QP * jnp.eye(DZ)
        Pz = P @ z; Sv = RP + z @ Pz; g = Pz / Sv
        Mr2 = Mr + jnp.outer(r - Mr @ z, g)
        Mphi2 = Mphi + jnp.outer(ph - Mphi @ z, g) if STORE else Mphi
        P2 = P - jnp.outer(Pz, Pz) / Sv
        S2 = S + ETA_A * jnp.outer(r - S @ ph, ph)
        sel = lambda a, b: jnp.where(w, a, b)
        return (sel(Mr2, Mr), sel(P2, c[1]), sel(Mphi2, Mphi), sel(S2, S)), None

    def chunk(c, ci):
        sl = lambda A: jax.lax.dynamic_slice_in_dim(A, ci * K, K)
        c, _ = jax.lax.scan(awake, c, (sl(Zp), sl(Phip), sl(Rp_), sl(Wp)))
        if STORE and SMODE != 'ridge':
            Mr, P, Mphi, S = c
            S = sleep(S, Mr, Mphi, ci * K + K, jax.random.fold_in(rk, ci))
            c = (Mr, P, Mphi, S)
        return c, None

    c0 = (jnp.zeros((EP.H, DZ)), jnp.eye(DZ), jnp.zeros((DF, DZ)), jnp.zeros((EP.H, DF)))
    (Mr, P, Mphi, S), _ = jax.lax.scan(chunk, c0, jnp.arange(NC))
    sh_before = shares(Mr, Mphi, S, TP)
    if STORE and SMODE != 'ridge':
        S = sleep(S, Mr, Mphi, TP, jax.random.fold_in(rk, 10_000))       # 복귀 판정 직전 잠
    sh_after = shares(Mr, Mphi, S, TP)
    if SMODE == 'ridge':
        W = Wp[:, None]
        Pw, Rw = Phip * W, Rp_ * W
        G = Pw.T @ Pw
        Ss = jnp.stack([jnp.linalg.solve(G + l * jnp.eye(DF), Pw.T @ Rw).T for l in RIDGE_L])   # (L, 8, DF)
    else:
        Ss = S[None]

    s0, o0 = EL.env0(sessA)

    def sim(plan, ai):
        def f(c, a):
            s, o, e = c
            e = AL[:, None] * e + key(o, a)
            x = e.reshape(-1); z = SG * x; ph = phif(x)
            Mz = Mr @ z; Mpz = Mphi @ z
            out = jnp.concatenate([Mz[None], jnp.einsum('lhd,d->lh', Ss, ph - Mpz) + Mz, jnp.einsum('lhd,d->lh', Ss, ph)])   # 시냅스만, 합친 기억 ×L, 저장소만 ×L
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(2), jnp.array(0), jnp.array(False), sessA["b0"])
            return (s2, o2, e), out
        _, YH = jax.lax.scan(f, (s0, o0, e_end), plan)
        yh = YH[ai]
        return (yh @ gC) / (jnp.linalg.norm(yh, axis=-1) * jnp.linalg.norm(gC) + 1e-8)
    sc = jax.vmap(sim)(plans, ai_all)                                    # (5, 1 + 2L)
    return sc, jnp.stack([sh_before, sh_after])


run = jax.jit(jax.vmap(memory_and_judge))

NW, BS = 256, 32
names = (["synapse_only"] + [f"combined_l{l:g}" for l in RIDGE_L] + [f"store_only_l{l:g}" for l in RIDGE_L]
         if SMODE == 'ridge' else ["synapse_only", "combined", "store_only"])
scores, shs, cidx, Tlen = [], [], [], []
for i in range(0, NW, BS):
    idx = range(i, i + BS)
    A = [area_session(1000 + w) for w in idx]
    areas = [A] + [[area_session(200_000 + 1000 * j + w) for w in idx] for j in range(M_B)]
    feats = [area_feats(stack(a)) for a in areas]
    Kk = jnp.concatenate([f[0] for f in feats], 1)
    Rr = jnp.concatenate([f[1] for f in feats], 1)
    arr = jnp.concatenate([f[2] for f in feats], 1)
    sA = stack(A)
    plans, ai = EL.plans_for(A)
    rks = jax.vmap(lambda j: jax.random.fold_in(jax.random.PRNGKey(1), j))(jnp.arange(i, i + BS))
    sc, sh = run(Kk, Rr, arr, sA, plans, ai, rks)
    scores.append(np.array(sc)); shs.append(np.array(sh))
    cidx += [int(np.flatnonzero(a["prod"] == 2)[0]) for a in A]
    Tlen.append(Kk.shape[1])
    print(f"batch {i // BS + 1}/{NW // BS}  T {Kk.shape[1]}", flush=True)
sc = np.concatenate(scores); sh = np.concatenate(shs); cidx = np.array(cidx)

res = dict(m=M_B, store=STORE, smode=SMODE, rounds=ROUNDS, T=Tlen, sigma2=SG ** 2, R=RP, q=QP, DF=DF, eta_a=ETA_A, eta_s=ETA_S, K=K, n=NPASS)
rng = np.random.default_rng(0)
for k, nm in enumerate(names):
    x = sc[:, :, k] + 1e-9 * rng.random(sc[:, :, 0].shape)
    order = np.argsort(-x, 1)
    rank = np.argmax(order == cidx[:, None], 1) + 1
    res[nm] = dict(tries=float(rank.mean()), first=float(np.mean(rank == 1)), tries_hist=np.bincount(rank, minlength=6)[1:].tolist())
res["shares"] = dict(before_final_sleep=dict(synapse=float(sh[:, 0, 0].mean()), store=float(sh[:, 0, 1].mean())),
                     after_final_sleep=dict(synapse=float(sh[:, 1, 0].mean()), store=float(sh[:, 1, 1].mean())))
for nm in names:
    print(f"m {M_B} store {int(STORE)} {SMODE}  {nm:18s} tries {res[nm]['tries']:.3f}  first {res[nm]['first']:.3f}  hist {res[nm]['tries_hist']}")
print(json.dumps(res), flush=True)
