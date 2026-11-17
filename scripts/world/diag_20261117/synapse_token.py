# 칸 토큰판 (명세 1 후속, 2026-11-17): 토큰 = 칸 12개, 키는 토큰마다, 시냅스는 하나를 공유.
#   k_{t,p} = exp(ℓ) f̂/|f̂|, f̂ = f([w_t; onehot a_t; onehot p]) (공유 MLP 98→128→128), ℓ = 선형(같은 입력)
#   e_{t,p} = α e_{t−1,p} + k_{t,p} (칸 위치별 흔적, α 공유·학습), x_{t,p} = [k; e], z = σ_p x.
#   공유 M ∈ R^{8×256}, P ∈ R^{256×256}: 스텝마다 P ← P + qI 한 번, 칸 12개 배치 칼만 쓰기 (12×12 풀이, 결합 우도 정확). R_p, σ_p 칸별, q 하나.
#   목적·자료·lr·판정은 명세 1과 같음.
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

# ---- 매개변수
DIN = DW + ER.NA + EP.NP
kr = jax.random.split(jax.random.PRNGKey(11), 2)
kp = dict(W1=jax.random.normal(kr[0], (DIN, 128)) / np.sqrt(DIN), b1=jnp.zeros(128),
          W2=jax.random.normal(kr[1], (128, KD)) / np.sqrt(128.0), b2=jnp.zeros(KD),
          lw=jnp.zeros(DIN), lb=jnp.zeros(()),
          ta=jnp.array(np.log(2.0 ** (-1 / 8) / (1 - 2.0 ** (-1 / 8)))))
kf = dict(ls=jnp.log(jnp.array(p13["sigma2"])), lr=jnp.log(jnp.array(p13["R"])), lq=jnp.log(jnp.array(np.mean(p13["q"]))))
th = dict(kp=kp, kf=kf)
POS = jnp.eye(EP.NP)


def key_fn(kp, o, a):
    """칸 12개의 키 [12, KD]와 ℓ [12]."""
    w = feat(o)[:9].reshape(-1)
    ctx = jnp.concatenate([w, jax.nn.one_hot(a, ER.NA)])
    inp = jnp.concatenate([jnp.tile(ctx, (EP.NP, 1)), POS], -1)          # [12, DIN]
    kh = jax.nn.silu(inp @ kp["W1"] + kp["b1"]) @ kp["W2"] + kp["b2"]
    ell = inp @ kp["lw"] + kp["lb"]
    return jnp.exp(ell)[:, None] * kh / (jnp.linalg.norm(kh, axis=-1, keepdims=True) + 1e-8), ell


def alpha(kp):
    return jax.nn.sigmoid(kp["ta"])


def step_e(kp, e, o, a):
    """e: [12, KD] 칸별 흔적. 반환: 새 흔적, 주소 x [12, 2KD], ℓ [12]."""
    k, ell = key_fn(kp, o, a)
    e = alpha(kp) * e + k
    return e, jnp.concatenate([k, e], -1), ell


def write(th, O, acts, stop_t):
    kp, kf = th["kp"], th["kf"]
    R, q, sg = jnp.exp(kf["lr"]), jnp.exp(kf["lq"]), jnp.exp(0.5 * kf["ls"])

    @jax.checkpoint
    def f(c, x):
        M, P, e = c
        o, a, o2, t = x
        e, X, ell = step_e(kp, e, o, a)
        mu, _ = wm(o, a)
        r = feat(o2) - mu
        Z = sg[:, None] * X                                                 # [12, DZ]
        P0 = P + q * jnp.eye(DZ)

        # 배치 칼만: 칸 12개를 한 번에 (관측 잡음 독립 → 순차 12번과 같은 사후·결합 우도)
        E = r - Z @ M.T                                                     # [12, 8] 쓰기 전 오차
        PZ = Z @ P0                                                         # [12, DZ] (P 대칭)
        S = PZ @ Z.T + jnp.diag(R)                                          # [12, 12]
        L = jnp.linalg.cholesky(S)
        SiPZ = jax.scipy.linalg.cho_solve((L, True), PZ)                    # S⁻¹ Z P
        M2 = M + jax.scipy.linalg.cho_solve((L, True), E).T @ PZ            # M + Eᵀ S⁻¹ Z P
        P2 = P0 - PZ.T @ SiPZ
        P2 = 0.5 * (P2 + P2.T)
        logdet = 2 * jnp.sum(jnp.log(jnp.diag(L)))
        nll = 0.5 * (jnp.sum(E * jax.scipy.linalg.cho_solve((L, True), E)) + EP.H * logdet)
        zs = Z[EP.SLOT]; rs = r[EP.SLOT]
        got = (M2 @ zs) @ rs / (rs @ rs + 1e-12)
        got = jnp.full(EP.NP, got)
        keep = t > stop_t
        return (jnp.where(keep, M, M2), jnp.where(keep, P, P2), e), (jnp.where(keep, 0.0, nll.sum()), got[EP.SLOT], ell[EP.SLOT], X[EP.SLOT])
    c0 = (jnp.zeros((EP.H, DZ)), jnp.eye(DZ), jnp.zeros((EP.NP, KD)))
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
            e, X, _ = step_e(kp, e, o, a)
            yh = M @ (sgs * X[EP.SLOT])
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
                q=float(np.exp(np.array(kf["lq"]))))


t0 = time.time()
v0 = float(val_nll(th))
j0 = judge(th)
print(f"[start = 무작위 초기화 키 (12.13 아님)] val_nll {v0:.1f}  b1 {j0['b1']:.4f} b0 {j0['b0']:.4f} other {j0['other']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
res = dict(spec="token_cells", before=dict(val_nll=v0, **j0, **params_summary(th)))
OUT = os.environ.get('OUT', '/workspace/RL-TDP/docs/world/results/synapse_layer_20261117/token.json')

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
