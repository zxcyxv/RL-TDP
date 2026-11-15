# 시냅스 읽기 뒤에 블록 (잔차 연결 + 사후 정규화) 을 붙인다. 쓰기는 12.7 / eb_scale const 그대로:
#   흔적 [k; e] (극 0, 8), 난수 키 128, 칸별 RLS, 목표 = 기억 없는 월드 모델 μ⁰ 기준 잔차, z = σ_p [k; e], 잡음 R_p (경험적 베이즈).
# 읽기 (예측):
#   m = M z                                   (시냅스 읽기, 칸별 [12, 8])
#   u = W_in [feat(o), onehot(a)]             (d = 256)
#   h = RMSNorm(u + W_m m)                    (읽기를 잔차 흐름에 더함, 사후 정규화)
#   h = RMSNorm(h + W_2 ReLU(W_1 h))          (MLP, 잔차 + 사후 정규화)
#   ŷ = μ⁰ + W_μ h,  log v = log v⁰ + W_v h  (머리는 0에서 시작 = 기억 없는 월드 모델에서 출발)
# 쓰기가 블록에 의존하지 않으므로 블록은 학습 세계 교사 세션의 (o, a, m, y)로 칸별 NLL 지도 학습 (역전파는 세션을 거치지 않음).
# 판정: 12.7과 같음. 점수 = 도착 스텝 ŷ_도착칸의 B 방향 코사인 (손 선호). 대조: 같은 쓰기의 시냅스 직접 읽기 (12.7 방식), 블록에 m = 0.
import sys, os, json; sys.path.insert(0, '/workspace/RL-TDP')
import jax, jax.numpy as jnp, numpy as np, optax
from rltdp.world import errand as ER, errand_learned as EL, errand_prec as EP, errand_reuse as RU
ER.PUBLIC = False
We, ybar = EL.emb_fixed(0)
p, vmax = EP.train("random", seed=0, D=256, iters_k=0, log=lambda m: None)
feat, wm, key = EP.parts(p, We, ybar, "random")
NP, H, SLOT, DK = EP.NP, EP.H, EP.SLOT, EP.DK
DZ = 2 * DK
phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
gB = phi[1] - phi.mean(0)


def wstep(h, c, o, a, o2, write_on):
    """쓰기 한 스텝 (12.7 / eb const). 반환: 새 상태, 쓰기 전 읽기 m [12, 8]."""
    M, P, e = c
    k = key(o, a)
    e = EP.ALPHA * e + k
    zb = jnp.concatenate([e, k])
    s = jnp.exp(0.5 * h["ls"]); R = jnp.exp(h["lr"])
    mu, _ = wm(o, a)
    r = feat(o2) - mu
    m = jnp.einsum("phd,pd->ph", M, s[:, None] * zb[None])

    def pp(Mp, Pp, sp, rp, Rp):
        z = sp * zb
        err = rp - Mp @ z; Pz = Pp @ z; S = Rp + z @ Pz
        return Mp + jnp.outer(err, Pz) / S, Pp - jnp.outer(Pz, Pz) / S, 0.5 * (err @ err / S + H * jnp.log(S))
    M2, P2, nll = jax.vmap(pp)(M, P, s, r, R)
    M = jnp.where(write_on, M2, M); P = jnp.where(write_on, P2, P)
    return (M, P, e), m, nll.sum()


def c_init():
    return (jnp.zeros((NP, H, DZ)), jnp.tile(jnp.eye(DZ), (NP, 1, 1)), jnp.zeros(DK))


def stream(h, O, acts, stop_t, ckpt=False):
    def f(c, x):
        o, a, o2, t = x
        c, m, nll = wstep(h, c, o, a, o2, t <= stop_t)
        return c, (m, nll)
    f = jax.checkpoint(f) if ckpt else f
    return jax.lax.scan(f, c_init(), (O[:-1], acts, O[1:], jnp.arange(acts.shape[0])))


# ---- 1. 쓰기의 칸별 σ², R: 경험적 베이즈 (eb_scale const와 같음)
_, tr = EL.stack_sessions(range(50_000, 50_000 + 256), 256)
h = dict(ls=jnp.zeros(NP), lr=jnp.full(NP, jnp.log(0.01)))
opt = optax.adam(0.05); st = opt.init(h)


@jax.jit
def upd_eb(h, st, idx):
    def lf(h):
        sb = jax.tree_util.tree_map(lambda x: x[idx], tr)
        def one(sess):
            O, _, _ = EL.observe_stream(sess)
            _, (_, nll) = stream(h, O, sess["acts"], 10 ** 9, ckpt=True)
            return nll.sum()
        return jax.vmap(one)(sb).mean()
    l, g = jax.value_and_grad(lf)(h)
    g, st = opt.update(g, st, h)
    return optax.apply_updates(h, g), st, l


rk = jax.random.PRNGKey(5)
for it in range(200):
    rk, k = jax.random.split(rk)
    h, st, l = upd_eb(h, st, jax.random.randint(k, (8,), 0, 256))
print("EB sigma2", np.round(np.exp(np.array(h["ls"])), 5).tolist(), "R", np.round(np.exp(np.array(h["lr"])), 6).tolist(), flush=True)

# ---- 2. 학습 세계의 (o, a, m, y) 모으기
def collect(sess):
    O, _, _ = EL.observe_stream(sess)
    _, (m, _) = stream(h, O, sess["acts"], 10 ** 9)
    return O[:-1], sess["acts"], m, jax.vmap(feat)(O[1:])
Ot, At, Mt, Yt = jax.jit(jax.vmap(collect))(tr)
NV = 32                                                                              # 검증 세계 (블록 학습에 안 씀)
Ov, Av, Mv, Yv = (x[-NV:].reshape(-1, *x.shape[2:]) for x in (Ot, At, Mt, Yt))
Ot, At, Mt, Yt = (x[:-NV].reshape(-1, *x.shape[2:]) for x in (Ot, At, Mt, Yt))
N = Ot.shape[0]
print("train steps", N, flush=True)

# ---- 3. 블록
D = 256
rms = lambda x: x / jnp.sqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
kk = jax.random.split(jax.random.PRNGKey(3), 4)
init = lambda k, i, o: jax.random.normal(k, (i, o)) / jnp.sqrt(i)
bp = dict(Win=init(kk[0], NP * H + ER.NA, D), Wm=init(kk[1], NP * H, D), W1=init(kk[2], D, 4 * D), W2=jnp.zeros((4 * D, D)),
          Wmu=jnp.zeros((D, NP * H)), Wv=jnp.zeros((D, NP)))
GAIN = float(os.environ.get('MGAIN', 1.0))


def block(bp, o, a, m):
    mu0, lv0 = wm(o, a)
    u = jnp.concatenate([feat(o).reshape(-1), jax.nn.one_hot(a, ER.NA)]) @ bp["Win"]
    x = rms(u + (GAIN * m.reshape(-1)) @ bp["Wm"])
    x = rms(x + jax.nn.relu(x @ bp["W1"]) @ bp["W2"])
    return mu0 + (x @ bp["Wmu"]).reshape(NP, H), lv0 + x @ bp["Wv"]


def nll_b(bp, o, a, m, y):
    mu, lv = block(bp, o, a, m)
    return (0.5 * (((y - mu) ** 2).sum(-1) * jnp.exp(-lv) + H * lv)).sum()


ITERS = int(os.environ.get('ITERS', 4000))
optb = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.warmup_cosine_decay_schedule(0.0, 1e-3, 200, ITERS)))
sb_ = optb.init(bp)


@jax.jit
def upd_b(bp, st, idx):
    l, g = jax.value_and_grad(lambda b: jax.vmap(lambda o, a, m, y: nll_b(b, o, a, m, y))(Ot[idx], At[idx], Mt[idx], Yt[idx]).mean())(bp)
    g, st = optb.update(g, st, bp)
    return optax.apply_updates(bp, g), st, l


def nll_slot(b, o, a, m, y):
    mu, lv = block(b, o, a, m)
    return 0.5 * (((y[SLOT] - mu[SLOT]) ** 2).sum() * jnp.exp(-lv[SLOT]) + H * lv[SLOT])


@jax.jit
def val(b):
    tot = jax.vmap(lambda o, a, m, y: nll_b(b, o, a, m, y))(Ov, Av, Mv, Yv).mean()
    sl = jax.vmap(lambda o, a, m, y: nll_slot(b, o, a, m, y))(Ov, Av, Mv, Yv).mean()
    return tot, sl


rk = jax.random.PRNGKey(9)
best_v, best_bp, best_it = 1e30, bp, -1
for it in range(ITERS):
    rk, k = jax.random.split(rk)
    bp, sb_, l = upd_b(bp, sb_, jax.random.randint(k, (2048,), 0, N))
    if it % 250 == 0 or it == ITERS - 1:
        vt, vs = val(bp)
        if float(vt) < best_v:
            best_v, best_bp, best_it = float(vt), bp, it
        print(f"[block] it {it} train {float(l):.3f} | val total {float(vt):.3f} val slot {float(vs):.4f}", flush=True)
bp = best_bp
print("best val it", best_it, flush=True)
def nll_wm(o, a, y):
    mu, lv = wm(o, a)
    return (0.5 * (((y - mu) ** 2).sum(-1) * jnp.exp(-lv) + H * lv)).sum()
base = jax.vmap(nll_wm)(Ot[:20000], At[:20000], Yt[:20000]).mean()
full = jax.vmap(lambda o, a, m, y: nll_b(bp, o, a, m, y))(Ot[:20000], At[:20000], Mt[:20000], Yt[:20000]).mean()
nom = jax.vmap(lambda o, a, y: nll_b(bp, o, a, jnp.zeros((NP, H)), y))(Ot[:20000], At[:20000], Yt[:20000]).mean()
print("train nll/step: memoryless wm %.3f | block m=0 %.3f | block full %.3f" % (float(base), float(nom), float(full)), flush=True)

# ---- 4. 판정
sessions, stack = EL.stack_sessions(range(1000, 1256), 256)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])


def one(sess, plans, ai_all):
    O, s_end, _ = EL.observe_stream(sess)
    c, _ = stream(h, O, sess["acts"], RU.last_arrival(O))

    def sim(plan, ai):
        def f(cc, a):
            s, o, cm = cc
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            cm2, m, _ = wstep(h, cm, o, a, o2, False)
            mu_f, _ = block(bp, o, a, m)
            mu_0, _ = block(bp, o, a, jnp.zeros_like(m))
            return (s2, o2, cm2), (m[SLOT], mu_f[SLOT], mu_0[SLOT])
        _, (MS, MF, M0) = jax.lax.scan(f, (s_end, O[-1], c), plan)
        cos = lambda d: d @ gB / (jnp.linalg.norm(d) * jnp.linalg.norm(gB) + 1e-8)
        return jnp.stack([cos(MS[ai]), cos(MF[ai] - phi.mean(0)), cos(M0[ai] - phi.mean(0))])
    return jax.vmap(sim)(plans, ai_all)


sc = np.array(jax.jit(jax.vmap(one))(stack, plans, arrive))
res = {}
for j, nm in enumerate(["synapse direct (12.7)", "block", "block m=0"]):
    x = sc[:, :, j]
    best = (x + 1e-9 * np.random.default_rng(0).random(x.shape)).argmax(1)
    res[nm] = dict(b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)), other=float(np.mean((best != b1) & (best != b0))))
    print("%-22s b1 %.3f b0 %.3f other %.3f" % (nm, res[nm]["b1"], res[nm]["b0"], res[nm]["other"]), flush=True)
print(json.dumps(res), flush=True)

# ---- 5. 선형 읽기 한계: 시험 세계 교사 세션의 도착 칸 예측 (선형 μ⁰ + M z 대 블록), 쓰기는 같음
blankf = feat(ER.TEX[jnp.full(12, ER.BLANK)].reshape(12, -1))[SLOT]
ctr = phi - phi.mean(0)


def measure(sess):
    O, _, _ = EL.observe_stream(sess)
    acts = sess["acts"]
    def f(c, x):
        o, a, o2 = x
        M, P, e = c
        k = key(o, a); e2 = EP.ALPHA * e + k; zb = jnp.concatenate([e2, k])
        z = jnp.exp(0.5 * h["ls"][SLOT]) * zb
        S = jnp.exp(h["lr"][SLOT]) + z @ P[SLOT] @ z
        c2, m, _ = wstep(h, c, o, a, o2, True)
        mu0, _ = wm(o, a)
        y = feat(o2)[SLOT]
        lin = mu0[SLOT] + m[SLOT]
        mb, lvb = block(bp, o, a, m)
        nl_lin = 0.5 * (((y - lin) ** 2).sum() / S + H * jnp.log(S))
        nl_blk = 0.5 * (((y - mb[SLOT]) ** 2).sum() * jnp.exp(-lvb[SLOT]) + H * lvb[SLOT])
        return c2, (y, lin, mb[SLOT], nl_lin, nl_blk)
    _, (Y, L, B, NL, NB) = jax.lax.scan(f, c_init(), (O[:-1], acts, O[1:]))
    arrived = ~jnp.all(jnp.isclose(Y, blankf, atol=1e-5), -1)
    item = jnp.argmax((Y - phi.mean(0)) @ ctr.T, -1)
    okL = jnp.argmax((L - phi.mean(0)) @ ctr.T, -1) == item
    okB = jnp.argmax((B - phi.mean(0)) @ ctr.T, -1) == item
    n = jnp.cumsum(arrived)
    last = jnp.arange(arrived.shape[0]) == RU.last_arrival(O)
    return arrived, n, last, okL, okB, NL, NB


arr, n, last, okL, okB, NL, NB = (np.array(x) for x in jax.jit(jax.vmap(measure))(stack))
first, later = arr & (n <= 5), arr & (n > 5) & ~last
out = dict(nll_arrival_linear=float(NL[arr].mean()), nll_arrival_block=float(NB[arr].mean()),
           nll_blank_linear=float(NL[~arr].mean()), nll_blank_block=float(NB[~arr].mean()),
           acc_first5=[float(okL[first].mean()), float(okB[first].mean())], acc_revisit=[float(okL[later].mean()), float(okB[later].mean())],
           acc_b0_C=[float(okL[last].mean()), float(okB[last].mean())])
print("arrival slot, test worlds (linear vs block):", json.dumps(out), flush=True)
