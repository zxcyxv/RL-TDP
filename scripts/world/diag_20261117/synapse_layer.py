# 명세 1 (docs/world/synapse_layer_spec_basic.md): 12.13 구성에서 키 경로만 학습.
#   u = w + MLP(w) (마지막 층 0), k̂ = W_k[a] u (초기 = 12.13의 Wr[a]), k = exp(ℓ) k̂/|k̂| (ℓ = 선형([w; onehot a]), 초기 0),
#   e = α e + k (α = sigmoid(θ_α), 초기 2^(-1/8)), x = [k; e], z_p = σ_p x. 쓰기는 eb_scale.py write() (QN=1) 그대로.
#   목적: 학습 세계 교사 세션의 순차 예측 NLL (경험적 베이즈). 키 경로 lr 1e-3, 칼만 매개변수 lr 0.05, 배치 8.
# 환경변수: ITERS (최대 반복, 기본 2000; 0 = 학습 전 판정만), PATIENCE (기본 500), OUT (결과 JSON 경로)
import sys, os, json, time; sys.path.insert(0, '/workspace/RL-TDP')
import jax, jax.numpy as jnp, numpy as np, optax
from rltdp.world import errand as ER, errand_learned as EL, errand_prec as EP, errand_reuse as RU

ER.PUBLIC = False
We, ybar = EL.emb_fixed(0)
p, vmax = EP.train("random", seed=0, D=256, iters_k=0, log=lambda m: None)
feat, wm, _ = EP.parts(p, We, ybar, "random")
KD, NPOLE = EP.DK, 2
DZ = NPOLE * KD
DW = 9 * EP.H
phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
gB = phi[1] - phi.mean(0)
LOG1213 = '/workspace/RL-TDP/docs/world/results/errand_diag_20261115b/eb_process_noise.log'
p13 = json.loads(open(LOG1213).read().strip().splitlines()[-1])

# ---- 매개변수: 키 경로 (kp)와 칼만 (kf)
kr = jax.random.split(jax.random.PRNGKey(11), 2)
kp = dict(W1=jax.random.normal(kr[0], (DW, 128)) / np.sqrt(DW), b1=jnp.zeros(128),
          W2=jnp.zeros((128, DW)), b2=jnp.zeros(DW),
          Wk=jnp.array(p["Wr"]),                                   # [NA, 128, 72], 12.13의 난수 키
          lw=jnp.zeros(DW + ER.NA), lb=jnp.zeros(()),
          ta=jnp.array(np.log(2.0 ** (-1 / 8) / (1 - 2.0 ** (-1 / 8)))))
kf = dict(ls=jnp.log(jnp.array(p13["sigma2"])), lr=jnp.log(jnp.array(p13["R"])), lq=jnp.log(jnp.array(p13["q"])))
th = dict(kp=kp, kf=kf)


def key_fn(kp, o, a):
    w = feat(o)[:9].reshape(-1)
    u = w + jax.nn.silu(w @ kp["W1"] + kp["b1"]) @ kp["W2"] + kp["b2"]
    kh = kp["Wk"][a] @ u
    ell = jnp.concatenate([w, jax.nn.one_hot(a, ER.NA)]) @ kp["lw"] + kp["lb"]
    return jnp.exp(ell) * kh / (jnp.linalg.norm(kh) + 1e-8), ell


def alpha(kp):
    return jax.nn.sigmoid(kp["ta"])


def step_e(kp, e, o, a):
    k, ell = key_fn(kp, o, a)
    e = jnp.stack([k, alpha(kp) * e[1] + k])                    # 극 {0, 반감기 학습}: [지금 키; 흔적]
    return e, ell


def write(th, O, acts, stop_t):
    kp, kf = th["kp"], th["kf"]
    R, Q, sg = jnp.exp(kf["lr"]), jnp.exp(kf["lq"]), jnp.exp(0.5 * kf["ls"])

    @jax.checkpoint
    def f(c, x):
        M, P, e = c
        o, a, o2, t = x
        e, ell = step_e(kp, e, o, a)
        mu, _ = wm(o, a)
        r = feat(o2) - mu

        def pp(Mp, Pp, sp, rp, Rp, Qp):
            Pp = Pp + Qp * jnp.eye(DZ)
            z = sp * e.reshape(-1)
            err = rp - Mp @ z; Pz = Pp @ z; S = Rp + z @ Pz
            M2 = Mp + jnp.outer(err, Pz) / S
            got = (M2 @ z) @ rp / (rp @ rp + 1e-12)
            return M2, Pp - jnp.outer(Pz, Pz) / S, 0.5 * (err @ err / S + EP.H * jnp.log(S)), got
        M2, P2, nll, got = jax.vmap(pp)(M, P, sg, r, R, Q)
        keep = t > stop_t
        return (jnp.where(keep, M, M2), jnp.where(keep, P, P2), e), (jnp.where(keep, 0.0, nll.sum()), got[EP.SLOT], ell, e.reshape(-1))
    c0 = (jnp.zeros((EP.NP, EP.H, DZ)), jnp.tile(jnp.eye(DZ), (EP.NP, 1, 1)), jnp.zeros((NPOLE, KD)))
    return jax.lax.scan(f, c0, (O[:-1], acts, O[1:], jnp.arange(acts.shape[0])))


def sess_nll(th, sess):
    O, _, _ = EL.observe_stream(sess)
    _, (nll, _, _, _) = write(th, O, sess["acts"], 10 ** 9)
    return nll.sum()


# ---- 자료
_, tr = EL.stack_sessions(range(50_000, 51_024), 256)
_, va = EL.stack_sessions(range(51_024, 51_152), 256)
print("lengths: train", tr["acts"].shape, "val", va["acts"].shape, flush=True)


@jax.jit
def val_nll(th):
    def chunk(c, idx):
        sb = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, idx, 16), va)
        return c + jax.vmap(lambda s: sess_nll(th, s))(sb).sum(), None
    tot, _ = jax.lax.scan(chunk, 0.0, jnp.arange(0, 128, 16))
    return tot / 128


# ---- 판정 (eb_scale.py와 같음, 키·흔적만 학습된 식)
sessions, stack = EL.stack_sessions(range(1000, 1256), 256)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])


def judge_one(th, sess, plans, ai_all):
    kp, kf = th["kp"], th["kf"]
    O, s_end, _ = EL.observe_stream(sess)
    t_c = RU.last_arrival(O)
    Y = jax.vmap(feat)(O[1:])[:, EP.SLOT]
    blank = feat(ER.TEX[jnp.full(12, ER.BLANK)].reshape(12, -1))[EP.SLOT]
    arrived = ~jnp.all(jnp.isclose(Y, blank, atol=1e-5), -1)
    (M, P, e), (_, got, ell, X) = write(th, O, sess["acts"], t_c)
    got_mean = jnp.sum(jnp.where(arrived, got, 0)) / jnp.maximum(arrived.sum(), 1)
    T = sess["acts"].shape[0]
    live = jnp.arange(T) <= t_c
    a_ = sess["acts"]
    def mean_where(v, m):
        return jnp.sum(jnp.where(m, v, 0)) / jnp.maximum(m.sum(), 1)
    ksz = jnp.exp(ell)
    k_int = mean_where(ksz, live & (a_ == ER.INTERACT))
    k_mov = mean_where(ksz, live & (a_ < 4))
    k_wait = mean_where(ksz, live & (a_ == ER.WAIT))
    Xn = X / (jnp.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)
    En = X[:, KD:] / (jnp.linalg.norm(X[:, KD:], axis=-1, keepdims=True) + 1e-12)
    l2 = live[1:] & live[:-1]
    cos_x = mean_where(jnp.sum(Xn[1:] * Xn[:-1], -1), l2)
    cos_e = mean_where(jnp.sum(En[1:] * En[:-1], -1), l2)
    sgs = jnp.exp(0.5 * kf["ls"][EP.SLOT])

    def sim(plan, ai):
        def f(c, a):
            s, o, e = c
            e, _ = step_e(kp, e, o, a)
            yh = M[EP.SLOT] @ (sgs * e.reshape(-1))
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (s2, o2, e), yh
        _, YH = jax.lax.scan(f, (s_end, O[-1], e), plan)
        yh = YH[ai]
        return (yh @ gB) / (jnp.linalg.norm(yh) * jnp.linalg.norm(gB) + 1e-8), jnp.linalg.norm(yh)
    sc, nrm = jax.vmap(sim)(plans, ai_all)
    return sc, nrm, got_mean, k_int, k_mov, k_wait, cos_x, cos_e


@jax.jit
def judge_chunk(th, sb, pl, ar):
    return jax.vmap(lambda s, a, b: judge_one(th, s, a, b))(sb, pl, ar)


def judge(th):
    outs = []
    for i in range(0, 256, 32):
        sl = lambda x: x[i:i + 32]
        outs.append(judge_chunk(th, jax.tree_util.tree_map(sl, stack), plans[i:i + 32], arrive[i:i + 32]))
    sc, nrm, got, ki, km, kw, cx, ce = [np.concatenate([np.array(o[j]) for o in outs]) for j in range(8)]
    best = (sc + 1e-9 * np.random.default_rng(0).random(sc.shape)).argmax(1)
    return dict(b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)),
                other=float(np.mean((best != b1) & (best != b0))),
                written=float(np.mean(got)), read_norm=float(np.mean(nrm)),
                ksize_interact=float(np.mean(ki)), ksize_move=float(np.mean(km)), ksize_wait=float(np.mean(kw)),
                cos_addr_neighbor=float(np.mean(cx)), cos_trace_neighbor=float(np.mean(ce)),
                per_world_b1=(best == b1).astype(int).tolist())


def params_summary(th):
    a = float(alpha(th["kp"]))
    kf = th["kf"]
    return dict(alpha=a, half_life=float(-np.log(2) / np.log(a)),
                sigma2=np.exp(np.array(kf["ls"])).tolist(), R=np.exp(np.array(kf["lr"])).tolist(),
                q=np.exp(np.array(kf["lq"])).tolist())


t0 = time.time()
v0 = float(val_nll(th))
j0 = judge(th)
print(f"[start = 12.13 (무작위 키 + 12.13 경험적 베이즈 σ²·R·q)] val_nll {v0:.1f}  b1 {j0['b1']:.4f} b0 {j0['b0']:.4f} other {j0['other']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
res = dict(spec="synapse_layer_spec_basic", before=dict(val_nll=v0, **j0, **params_summary(th)))
OUT = os.environ.get('OUT', '/workspace/RL-TDP/docs/world/results/synapse_layer_20261117/spec1.json')

ITERS = int(os.environ.get('ITERS', 2000))
PAT = int(os.environ.get('PATIENCE', 500))
JUDGE_EVERY = int(os.environ.get('JUDGE_EVERY', 500))   # 이 반복마다 판정 (b1/b0/다른 곳)
opt = optax.multi_transform({"kp": optax.adam(1e-3), "kf": optax.adam(0.05)}, {"kp": "kp", "kf": "kf"})
st = opt.init(th)


@jax.jit
def upd(th, st, idx):
    def lf(th):
        sb = jax.tree_util.tree_map(lambda x: x[idx], tr)
        return jax.vmap(lambda s: sess_nll(th, s))(sb).mean()
    l, g = jax.value_and_grad(lf)(th)
    g, st = opt.update(g, st, th)
    return optax.apply_updates(th, g), st, l


curve = [dict(it=0, val_nll=v0)]
best_v, best_th, best_it = v0, th, 0
rk = jax.random.PRNGKey(5)
for it in range(1, ITERS + 1):
    rk, k = jax.random.split(rk)
    th, st, l = upd(th, st, jax.random.randint(k, (8,), 0, 1024))
    if not np.isfinite(float(l)):
        print("nan at", it, flush=True); break
    if it % 100 == 0:
        v = float(val_nll(th))
        curve.append(dict(it=it, val_nll=v, train_nll=float(l)))
        if v < best_v:
            best_v, best_th, best_it = v, th, it
        print(f"it {it} train {float(l):.1f} val {v:.1f} best {best_v:.1f}@{best_it} half_life {params_summary(th)['half_life']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if it % JUDGE_EVERY == 0:
            jj = judge(th)
            curve[-1].update(b1=jj["b1"], b0=jj["b0"], other=jj["other"])
            print(f"  [judge it {it}] b1 {jj['b1']:.4f} b0 {jj['b0']:.4f} other {jj['other']:.4f}", flush=True)
        if it - best_it >= PAT:
            print("early stop", flush=True); break

j1 = judge(best_th)
res.update(after=dict(val_nll=best_v, best_it=best_it, **j1, **params_summary(best_th)), curve=curve,
           note="train 50000-51023, val 51024-51151, judge 1000-1255; padding with WAIT as stack_sessions")
print(f"[after] val_nll {best_v:.1f}@{best_it}  b1 {j1['b1']:.4f} b0 {j1['b0']:.4f} other {j1['other']:.4f}", flush=True)
print(json.dumps({k: v for k, v in res["after"].items() if k != "per_world_b1"}), flush=True)
json.dump(res, open(OUT, 'w'))
np.savez(OUT.replace('.json', '_params.npz'), **{f"{a}/{b}": np.array(v) for a in best_th for b, v in best_th[a].items()})
