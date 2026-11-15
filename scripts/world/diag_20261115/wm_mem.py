# 시냅스를 월드 모델의 상태 자리에 (DreamerV3의 GRU h 자리). 한 스텝:
#   흔적 e ← α_i e + k (극 POLES, 기본 0,2,4,8 반감기), zb = flatten(e)
#   읽기 m = A zb (칸별 [12, 8])
#   월드 모델 (o, a, m, e) → 다음 장면 칸별 평균 μ, 로그 분산 lv   (m, e 입력 가중치는 0에서 시작 = 기억 없는 월드 모델에서 출발)
#   쓰기 (칸별 RLS): z = σ_p zb,  σ_p² = c_p · exp(lv_p)  (이번 스텝 기억 몫의 사전 분산 = 월드 모델이 말하는 불확실성),
#        목표 y = 다음 장면 특징, 잡음 R_p.  A ← A + (y − A z)(P z)ᵀ / S
#   느린 학습: 월드 모델·c_p·R_p를 학습 세계 교사 세션 전체의 칸별 NLL (μ, lv 아래)로, 세션 전체 역전파.
# 판정: 12.7과 같음 (난수 키, C 도착 뒤 쓰기 없음, 시드 1, 세계 256개), 점수 = 도착 스텝 μ_도착칸의 B 방향 코사인 (손 선호).
#   대조: 판정 때 m = 0 (흔적은 보지만 시냅스 내용은 못 봄).
import sys, os, json; sys.path.insert(0, '/workspace/RL-TDP')
import jax, jax.numpy as jnp, numpy as np, optax
from rltdp.world import errand as ER, errand_learned as EL, errand_prec as EP, errand_reuse as RU
ER.PUBLIC = False
ITERS = int(os.environ.get('ITERS', 300))
We, ybar = EL.emb_fixed(0)
p, vmax = EP.train("random", seed=0, D=256, iters_k=0, log=lambda m: None)
feat, _, key0 = EP.parts(p, We, ybar, "random")
HL = [float(x) for x in os.environ.get('POLES', '0,2,4,8').split(',')]
AL = jnp.array([0.0 if x == 0 else 2.0 ** (-1 / x) for x in HL])
KD = int(os.environ.get('KD', 64))
NL, DZ, NP, H, SLOT = len(HL), len(HL) * KD, EP.NP, EP.H, EP.SLOT


def key(o, a):
    k = key0(o, a)[:KD]
    return k / (jnp.linalg.norm(k) + 1e-8)


phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
gB = phi[1] - phi.mean(0)

q0 = {k: p[k] for k in ("wm1", "wb1", "wm2", "wb2")}
q0["wmx"] = jnp.zeros((NP * H + DZ, q0["wm1"].shape[1]))
q0["ls"] = jnp.zeros(NP)
q0["lr"] = jnp.full(NP, jnp.log(1e-3))


def wmm(q, o, a, m, zb):
    x = jnp.concatenate([feat(o).reshape(-1), jax.nn.one_hot(a, ER.NA)])
    rms = lambda v: v / jnp.sqrt(jnp.mean(v * v) + 1e-8)              # 새 입력의 크기만 맞춤 (시냅스 읽기는 칸별, 흔적은 전체에 하나: 극 사이 비율 = 경과 시간은 남음)
    hd = jax.nn.relu(x @ q["wm1"] + jnp.concatenate([jax.vmap(rms)(m).reshape(-1), rms(zb)]) @ q["wmx"] + q["wb1"])
    out = hd @ q["wm2"] + q["wb2"]
    return out[:NP * H].reshape(NP, H), out[NP * H:]


def step(q, c, o, a, o2, write_on, zero_m=False):
    A, P, e = c
    e = AL[:, None] * e + key(o, a)
    zb = e.reshape(-1)
    m = jnp.einsum("phd,d->ph", A, zb)
    m = jnp.where(zero_m, 0.0, m)
    mu, lv = wmm(q, o, a, m, zb)
    y = feat(o2)
    nll = 0.5 * (((y - mu) ** 2).sum(-1) * jnp.exp(-lv) + H * lv)
    sig = jnp.exp(0.5 * (q["ls"] + lv))
    R = jnp.exp(q["lr"])

    def pp(Ap, Pp, sp, yp, Rp):
        z = sp * zb
        err = yp - Ap @ z; Pz = Pp @ z; S = Rp + z @ Pz
        A2 = Ap + jnp.outer(err, Pz) / S
        return A2, Pp - jnp.outer(Pz, Pz) / S, (z @ Pz) / S
    A2, P2, gain = jax.vmap(pp)(A, P, sig, y, R)
    A = jnp.where(write_on, A2, A); P = jnp.where(write_on, P2, P)
    return (A, P, e), (nll, mu, lv, gain)


def c_init():
    return (jnp.zeros((NP, H, DZ)), jnp.tile(jnp.eye(DZ), (NP, 1, 1)), jnp.zeros((NL, KD)))


def session(q, O, acts, stop_t):
    @jax.checkpoint
    def f(c, x):
        o, a, o2, t = x
        c, out = step(q, c, o, a, o2, t <= stop_t)
        return c, out
    return jax.lax.scan(f, c_init(), (O[:-1], acts, O[1:], jnp.arange(acts.shape[0])))


_, tr = EL.stack_sessions(range(50_000, 50_000 + 256), 256)
LR = float(os.environ.get('LR', 1e-3))
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.linear_schedule(0.0, LR, 50)))   # 처음 50번 선형 워밍업
st = opt.init(q0)


@jax.jit
def upd(q, st, idx):
    def lf(q):
        sb = jax.tree_util.tree_map(lambda x: x[idx], tr)
        def one(sess):
            O, _, _ = EL.observe_stream(sess)
            _, (nll, _, _, _) = session(q, O, sess["acts"], 10 ** 9)
            return nll.sum(-1).mean()
        return jax.vmap(one)(sb).mean()
    l, g = jax.value_and_grad(lf)(q)
    g, st = opt.update(g, st, q)
    return optax.apply_updates(q, g), st, l


q = q0
rk = jax.random.PRNGKey(5)
for it in range(ITERS):
    rk, k = jax.random.split(rk)
    q, st, l = upd(q, st, jax.random.randint(k, (8,), 0, 256))
    if it % 25 == 0 or it == ITERS - 1:
        print(f"[wm+mem] it {it} nll/step {float(l):.3f}", flush=True)

# ---- 판정
sessions, stack = EL.stack_sessions(range(1000, 1256), 256)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])
blankf = feat(ER.TEX[jnp.full(12, ER.BLANK)].reshape(12, -1))[SLOT]


def one(qq, sess, plans, ai_all, zero_m):
    O, s_end, _ = EL.observe_stream(sess)
    t_c = RU.last_arrival(O)
    c, (_, _, lv, gain) = session(qq, O, sess["acts"], t_c)
    Y = jax.vmap(feat)(O[1:])[:, SLOT]
    arrived = ~jnp.all(jnp.isclose(Y, blankf, atol=1e-5), -1)
    g_arr = jnp.sum(jnp.where(arrived, gain[:, SLOT], 0)) / jnp.maximum(arrived.sum(), 1)
    g_non = jnp.sum(jnp.where(arrived, 0, gain[:, SLOT])) / jnp.maximum((~arrived).sum(), 1)
    v_arr = jnp.sum(jnp.where(arrived, jnp.exp(lv[:, SLOT]), 0)) / jnp.maximum(arrived.sum(), 1)
    v_non = jnp.sum(jnp.where(arrived, 0, jnp.exp(lv[:, SLOT]))) / jnp.maximum((~arrived).sum(), 1)

    def sim(plan, ai):
        def f(cc, a):
            s, o, cm = cc
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            cm2, (_, mu, _, _) = step(qq, cm, o, a, o2, False, zero_m)
            return (s2, o2, cm2), mu[SLOT]
        _, MU = jax.lax.scan(f, (s_end, O[-1], c), plan)
        d = MU[ai] - phi.mean(0)
        return d @ gB / (jnp.linalg.norm(d) * jnp.linalg.norm(gB) + 1e-8)
    return jax.vmap(sim)(plans, ai_all), jnp.stack([g_arr, g_non, v_arr, v_non])


res = dict(lr=LR, poles=HL, kd=KD, iters=ITERS, c=np.exp(np.array(q["ls"])).tolist(), R=np.exp(np.array(q["lr"])).tolist())
for zm in (False, True):
    sc, diag = jax.jit(jax.vmap(lambda s, pl, ai: one(q, s, pl, ai, zm)))(stack, plans, arrive)
    x = np.array(sc)
    best = (x + 1e-9 * np.random.default_rng(0).random(x.shape)).argmax(1)
    d = np.array(diag).mean(0)
    r = dict(b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)), other=float(np.mean((best != b1) & (best != b0))),
             gain_arrival=float(d[0]), gain_blank=float(d[1]), var_arrival=float(d[2]), var_blank=float(d[3]))
    res["m0" if zm else "full"] = r
    print(("m = 0  " if zm else "full   "), json.dumps(r), flush=True)
print(json.dumps(res), flush=True)

# ---- 진단: 교사 세션 안 도착 스텝에서 도착 칸 예측이 실제 산출물을 맞히는가 (시냅스 읽기 있음 / 0), 처음 도착 대 같은 채집처 재방문
import pickle
pickle.dump(jax.tree_util.tree_map(np.array, q), open(os.environ.get('SAVE', '/tmp/claude-0/-workspace/f74b0329-95de-4049-a17a-9ad7c26c1ed8/scratchpad/wm_mem_q.pkl'), 'wb'))


def teach(qq, sess, zero_m):
    O, _, _ = EL.observe_stream(sess)
    def f(c, x):
        o, a, o2 = x
        c, (_, mu, _, _) = step(qq, c, o, a, o2, True, zero_m)
        return c, mu[SLOT]
    _, MU = jax.lax.scan(f, c_init(), (O[:-1], sess["acts"], O[1:]))
    Y = jax.vmap(feat)(O[1:])[:, SLOT]
    arrived = ~jnp.all(jnp.isclose(Y, blankf, atol=1e-5), -1)
    item = jnp.argmax((Y - phi.mean(0)) @ (phi - phi.mean(0)).T, -1)                    # 실제 도착 산출물
    pred = jnp.argmax((MU - phi.mean(0)) @ (phi - phi.mean(0)).T, -1)                  # 예측이 가장 닮은 산출물
    n_arr = jnp.cumsum(arrived)
    return arrived, pred == item, n_arr


for zm in (False, True):
    arr, ok, n = jax.jit(jax.vmap(lambda s: teach(q, s, zm)))(stack)
    arr, ok, n = np.array(arr), np.array(ok), np.array(n)
    obs = arr & (n <= 5); later = arr & (n > 5)                                        # 관찰 단계 5번 (처음 봄) 대 그 뒤 (이미 본 채집처)
    print(("m = 0  " if zm else "full   "), "teacher arrivals: correct item  first-5 %.3f  later %.3f" % (ok[obs].mean(), ok[later].mean()), flush=True)
