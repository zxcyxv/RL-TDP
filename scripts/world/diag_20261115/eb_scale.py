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


def write(h, O, acts, stop_t):
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
for it in range(200):
    rk, k = jax.random.split(rk)
    h, st, l = upd(h, st, jax.random.randint(k, (8,), 0, 256))
    if it % 25 == 0 or it == 199:
        print(f"[EB {MODE}] it {it} nll {float(l):.1f}", flush=True)
print("sigma^2_p", np.round(np.exp(np.array(h["ls"])), 4).tolist())
print("q_p      ", np.round(np.exp(np.array(h["lq"])), 6).tolist() if "lq" in h else None)
print("R_p      ", np.round(np.exp(np.array(h["lr"])), 5).tolist(), flush=True)

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
           b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)), other=float(np.mean((best != b1) & (best != b0))),
           written=float(np.mean(got)), read_norm=float(np.mean(nrm)))
print(json.dumps(res), flush=True)
