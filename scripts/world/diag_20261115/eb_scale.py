# 쓰기 스케일을 표준형 (베이즈 선형 회귀 / 칼만)으로: 칸 p마다 y_p − μ_p = M_p z + ε,
#   z = σ_p [e; k] (사전 M ~ N(0, I) → 기억 몫의 사전 분산 σ_p² |[e; k]|²), ε ~ N(0, R_p).
#   σ_p², R_p는 칸별 상수, 학습 세계의 교사 세션에서 순차 예측 우도 (= 주변 우도, 경험적 베이즈)를 최대로 맞춤.
#   12.7과 같은 판정 (난수 키, 주소 [e; k], C 도착 뒤 쓰기 없음, 시드 1, 세계 256개).
# 환경변수 POLES (흔적 극의 반감기, 쉼표, 0 = 지금 키; 기본 0,8 = 12.7의 [k; e]), KD (키 차원, 기본 128; 앞 KD성분을 다시 정규화).
# argv: [mode]  const = 칸별 상수 σ_p (기본) / wm = σ_p(o, a) = c_p √v_p(o, a) (월드 모델 분산 × 칸별 상수)
import sys, json; sys.path.insert(0, '/workspace/RL-TDP')
import jax, jax.numpy as jnp, numpy as np, optax
from rltdp.world import errand as ER, errand_learned as EL, errand_prec as EP, errand_reuse as RU
MODE = sys.argv[1] if len(sys.argv) > 1 else "const"
ER.PUBLIC = False
We, ybar = EL.emb_fixed(0)
p, vmax = EP.train("random", seed=0, D=256, iters_k=0, log=lambda m: None)
feat, wm, key = EP.parts(p, We, ybar, "random")
import os
HL = [float(x) for x in os.environ.get('POLES', '0,8').split(',')]
AL = jnp.array([0.0 if x == 0 else 2.0 ** (-1 / x) for x in HL])
KD = int(os.environ.get('KD', EP.DK))
DZ = len(HL) * KD
key0 = key
def key(o, a):
    k = key0(o, a)[:KD]
    return k / (jnp.linalg.norm(k) + 1e-8)
phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
gB = phi[1] - phi.mean(0)


def sig(h, o, a):
    if MODE == "wm":
        _, lv = wm(o, a)
        return jnp.exp(0.5 * (h["ls"] + lv))
    return jnp.exp(0.5 * h["ls"])


FORGET = os.environ.get('FORGET', 'add')   # add = 12.13 (P ← P + qI) / mult = 곱하는 망각 (HANDOVER 0.2절)


def write_mult(h, O, acts, stop_t):
    # 정보형 곱하는 망각, 칸별 λ_p: H ← λH + zzᵀ/R, G ← λG + r zᵀ/R, M = G (H + I)⁻¹. 사전 N(0, I)는 감쇠하지 않음.
    # 예측 (주변 우도)은 감쇠 뒤·쓰기 전: 평균 G' u, 분산 R + zᵀu, u = (λH + I)⁻¹ z. 모두 누적합 꼴 (구현은 순차 scan).
    # 구현: 제곱근 공분산형, float32. P = (H + I)⁻¹ = S Sᵀ (S 정사각)와 M을 들고 감.
    #   감쇠 (H ← λH, G ← λG): C = λI + (1−λ)SᵀS (고유값 ∈ [λ, 1]), C = LLᵀ, S ← S L⁻ᵀ, M ← M − (1−λ) M S C⁻¹ Sᵀ (= λ M (λI + (1−λ)P)⁻¹, Woodbury).
    #   쓰기 (H += zzᵀ/R, G += rzᵀ/R): Potter. f = Sᵀz, α = 1/(R + |f|²), γ = 1/(1 + √(αR)), K = α S f, M += err Kᵀ, S ← S − γ K fᵀ.
    #   예측: 평균 M z, 분산 R + |Sᵀz|² (구조상 > 0).
    #   이유: 곱하는 망각은 P에 바닥이 없어 (정상 상태 분산 ≈ (1−λ)R/|z|², 지도 칸 ~1e-8) float32 정밀도 밑으로 내려감.
    #   정보형·공분산형 모두 it 100에서 S < 0 → NaN (P 최소 고유값 음수, 학습 세계 50/256). QR 기반 제곱근형 (SRIF)은 정확하지만 GPU에서 너무 느림.
    R = jnp.exp(h["lr"])
    lam = jax.nn.sigmoid(h["lt"])                   # λ_p ∈ (0, 1)
    I = jnp.eye(DZ)

    @jax.checkpoint
    def f(c, x):
        M, S, e = c
        o, a, o2, t = x
        k = key(o, a)
        e = AL[:, None] * e + k
        mu, _ = wm(o, a)
        s = sig(h, o, a)
        r = feat(o2) - mu

        def pp(Mp, Sp, sp, rp, Rp, lp):
            z = sp * e.reshape(-1)
            Mp, Sp, err, Sv = mult_step(Mp, Sp, z, rp, Rp, lp)
            got = (Mp @ z) @ rp / (rp @ rp + 1e-12)
            return Mp, Sp, 0.5 * (err @ err / Sv + EP.H * jnp.log(Sv)), got
        M2, S2, nll, got = jax.vmap(pp)(M, S, s, r, R, lam)
        keep = t > stop_t
        return (jnp.where(keep, M, M2), jnp.where(keep, S, S2), e), (jnp.where(keep, 0.0, nll.sum()), got[EP.SLOT])
    c0 = (jnp.zeros((EP.NP, EP.H, DZ)), jnp.tile(I, (EP.NP, 1, 1)), jnp.zeros((len(HL), KD)))
    with jax.default_matmul_precision("highest"):
        (M, S, e), ys = jax.lax.scan(f, c0, (O[:-1], acts, O[1:], jnp.arange(acts.shape[0])))
        P = jnp.einsum('pij,pkj->pik', S, S)
    return (M, P, e), ys


def mult_step(M, S, z, r, Rp, lp):
    """곱하는 망각 한 스텝 (감쇠 뒤 예측, 그다음 쓰기). 반환: M2, S2, 예측 오차, 예측 분산."""
    D = S.shape[0]
    C = lp * jnp.eye(D) + (1 - lp) * (S.T @ S)
    L = jnp.linalg.cholesky(C)
    Ci_St = jax.scipy.linalg.cho_solve((L, True), S.T)                 # C⁻¹ Sᵀ
    M = M - (1 - lp) * (M @ S) @ Ci_St
    S = jax.scipy.linalg.solve_triangular(L, S.T, lower=True).T         # S L⁻ᵀ
    fv = S.T @ z; Sv = Rp + fv @ fv
    err = r - M @ z
    al = 1.0 / Sv; ga = 1.0 / (1.0 + jnp.sqrt(al * Rp))
    K = al * (S @ fv)
    return M + jnp.outer(err, K), S - ga * jnp.outer(K, fv), err, Sv


def write(h, O, acts, stop_t):
    if FORGET == 'mult':
        return write_mult(h, O, acts, stop_t)
    R = jnp.exp(h["lr"])
    Q = jnp.exp(h["lq"]) if "lq" in h else jnp.zeros(EP.NP)          # 과정 잡음 (칸별 q_p I): 매 스텝 P ← P + q_p I

    @jax.checkpoint
    def f(c, x):
        M, P, e = c
        o, a, o2, t = x
        k = key(o, a)
        e = AL[:, None] * e + k
        mu, _ = wm(o, a)
        s = sig(h, o, a)
        r = feat(o2) - mu

        def pp(Mp, Pp, sp, rp, Rp, Qp):
            Pp = Pp + Qp * jnp.eye(Pp.shape[0])
            z = sp * e.reshape(-1)
            err = rp - Mp @ z; Pz = Pp @ z; S = Rp + z @ Pz
            M2 = Mp + jnp.outer(err, Pz) / S
            got = (M2 @ z) @ rp / (rp @ rp + 1e-12)
            return M2, Pp - jnp.outer(Pz, Pz) / S, 0.5 * (err @ err / S + EP.H * jnp.log(S)), got
        M2, P2, nll, got = jax.vmap(pp)(M, P, s, r, R, Q)
        keep = t > stop_t
        return (jnp.where(keep, M, M2), jnp.where(keep, P, P2), e), (jnp.where(keep, 0.0, nll.sum()), got[EP.SLOT])
    c0 = (jnp.zeros((EP.NP, EP.H, DZ)), jnp.tile(jnp.eye(DZ), (EP.NP, 1, 1)), jnp.zeros((len(HL), KD)))
    return jax.lax.scan(f, c0, (O[:-1], acts, O[1:], jnp.arange(acts.shape[0])))


# ---- 경험적 베이즈: 학습 세계의 교사 세션 전체의 순차 예측 NLL
_, tr = EL.stack_sessions(range(50_000, 50_000 + 256), 256)
h = dict(ls=jnp.zeros(EP.NP), lr=jnp.full(EP.NP, jnp.log(0.01)))
if os.environ.get('QN', '0') == '1':
    h['lq'] = jnp.full(EP.NP, jnp.log(1e-3))
if FORGET == 'mult':
    h['lt'] = jnp.full(EP.NP, jnp.log((1 - 1e-3) / 1e-3))   # λ_p = sigmoid(lt), 초기 1 − λ = 1e-3 (q 초기값 자리 그대로)
# HFROM = 기록 파일 (마지막 줄 JSON)의 σ², R을 그대로 쓰고 맞추지 않음 (12.13 분해: q = 0)
HFROM = os.environ.get('HFROM')
if HFROM:
    prev = json.loads(open(HFROM).read().strip().splitlines()[-1])
    h = dict(ls=jnp.log(jnp.array(prev["sigma2"])), lr=jnp.log(jnp.array(prev["R"])))
opt = optax.adam(0.05)
st = opt.init(h)


@jax.jit
def upd(h, st, idx):
    def lf(h):
        sb = jax.tree_util.tree_map(lambda x: x[idx], tr)
        def one(sess):
            O, _, _ = EL.observe_stream(sess)
            _, (nll, _) = write(h, O, sess["acts"], 10 ** 9)
            return nll.sum()
        return jax.vmap(one)(sb).mean()
    l, g = jax.value_and_grad(lf)(h)
    g, st = opt.update(g, st, h)
    return optax.apply_updates(h, g), st, l


rk = jax.random.PRNGKey(5)
for it in range(0 if HFROM else 200):
    rk, k = jax.random.split(rk)
    h, st, l = upd(h, st, jax.random.randint(k, (8,), 0, 256))
    if it % 25 == 0 or it == 199:
        print(f"[EB {MODE}] it {it} nll {float(l):.1f}", flush=True)
        if not np.isfinite(float(l)):
            print("nan: stop", {k: np.array(v).round(3).tolist() for k, v in h.items()}, flush=True); sys.exit(1)
print("sigma^2_p", np.round(np.exp(np.array(h["ls"])), 4).tolist())
print("q_p      ", np.round(np.exp(np.array(h["lq"])), 6).tolist() if "lq" in h else None)
print("R_p      ", np.round(np.exp(np.array(h["lr"])), 5).tolist(), flush=True)
if "lt" in h:
    print("1-lambda_p", np.array(jax.nn.sigmoid(-h["lt"])).tolist())
    print("window    ", np.round(1 / np.array(jax.nn.sigmoid(-h["lt"])), 1).tolist(), flush=True)

# ---- 판정 (12.7과 같음)
sessions, stack = EL.stack_sessions(range(1000, 1256), 256)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])


def one(sess, plans, ai_all):
    O, s_end, _ = EL.observe_stream(sess)
    t_c = RU.last_arrival(O)
    Y = jax.vmap(feat)(O[1:])[:, EP.SLOT]
    blank = feat(ER.TEX[jnp.full(12, ER.BLANK)].reshape(12, -1))[EP.SLOT]
    arrived = ~jnp.all(jnp.isclose(Y, blank, atol=1e-5), -1)
    (M, P, e), (_, got) = write(h, O, sess["acts"], t_c)
    got_mean = jnp.sum(jnp.where(arrived, got, 0)) / jnp.maximum(arrived.sum(), 1)

    def sim(plan, ai):
        def f(c, a):
            s, o, e = c
            k = key(o, a)
            e = AL[:, None] * e + k
            yh = M[EP.SLOT] @ (sig(h, o, a)[EP.SLOT] * e.reshape(-1))
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (s2, o2, e), yh
        _, YH = jax.lax.scan(f, (s_end, O[-1], e), plan)
        yh = YH[ai]
        return (yh @ gB) / (jnp.linalg.norm(yh) * jnp.linalg.norm(gB) + 1e-8), jnp.linalg.norm(yh)
    sc, nrm = jax.vmap(sim)(plans, ai_all)
    return sc, nrm, got_mean


sc, nrm, got = jax.jit(jax.vmap(one))(stack, plans, arrive)
x = np.array(sc)
best = (x + 1e-9 * np.random.default_rng(0).random(x.shape)).argmax(1)
res = dict(mode=MODE, poles=HL, kd=KD, sigma2=np.exp(np.array(h["ls"])).tolist(), R=np.exp(np.array(h["lr"])).tolist(), q=(np.exp(np.array(h["lq"])).tolist() if "lq" in h else None),
           forget=FORGET, hfrom=HFROM,
           one_minus_lam=(np.array(jax.nn.sigmoid(-h["lt"])).tolist() if "lt" in h else None),
           window=((1 / np.array(jax.nn.sigmoid(-h["lt"]))).tolist() if "lt" in h else None),
           b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)), other=float(np.mean((best != b1) & (best != b0))),
           written=float(np.mean(got)), read_norm=float(np.mean(nrm)))
print(json.dumps(res), flush=True)
