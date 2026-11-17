# 월드 모델 + 시냅스 동시 학습 (사전학습 없음), 2026-11-17. 명세: docs/world/spec_wm_joint.md
#   월드 모델 예측 ŷ = F₀(o, a) + m + G(o, a, m)
#     F₀: 기억 없는 예측 (무작위 초기화, 함께 학습). 쓰기 기준: r = feat(o') − F₀(o, a)
#     m: 시냅스 읽기 (칸마다 M_p z_p, 쓰기 전), 예측에 그대로 더함 (지름길)
#     G: 기억을 입력으로 받아 해석을 고치는 망, 마지막 층 0 (처음에는 ŷ = F₀ + m)
#   흔적은 월드 모델에 넣지 않음 (주소에만). 시냅스 쓰기·키·흔적은 synapse_cellkey.py (칸별 키, 칸별 칼만) 그대로, R₀ 8비트.
#   목적: 순차 NLL Σ ½[‖r − m − G‖²/S + 8 log S], S = R₀ + R_p + zᵀPz. 칼만 쓰기는 r − m (G 없음, 쓰기 규칙 그대로).
# 환경변수: ITERS (기본 10000), JUDGE_EVERY (500), OUT
import sys, os, json, time, functools; sys.path.insert(0, '/workspace/RL-TDP')
import jax, jax.numpy as jnp, numpy as np, optax
from rltdp.world import errand as ER, errand_learned as EL, errand_prec as EP, errand_reuse as RU

ER.PUBLIC = False
DIAGP = os.environ.get('DIAGP', '0') == '1'   # 대각 공분산 (KDN식)
We, ybar = EL.emb_fixed(0)
p0 = EP.init(jax.random.PRNGKey(0))                       # 사전학습 없음. 쓰는 것은 고정 난수 키 초기값 Wr뿐 (12.13과 같은 시드)
NP, H, KD, NPOLE = EP.NP, EP.H, EP.DK, 2
DZ, DW, HY = NPOLE * KD, 9 * EP.H, EP.NP * EP.H
R0 = float(np.mean(np.sum(np.array(We) ** 2, 0))) / 255.0 ** 2 / 12.0
print("R0", R0, flush=True)


def feat(o):
    return ((o @ We) - ybar).reshape(NP, H)


phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
gB = phi[1] - phi.mean(0)
p13 = json.loads(open('/workspace/RL-TDP/docs/world/results/errand_diag_20261115b/eb_process_noise.log').read().strip().splitlines()[-1])

ks = jax.random.split(jax.random.PRNGKey(11), 6)
nrm = lambda k, i, o, s=1.0: jax.random.normal(k, (i, o)) * s / np.sqrt(i)
kp = dict(W1=nrm(ks[0], DW + NP, 128), b1=jnp.zeros(128), W2=jnp.zeros((128, DW)), b2=jnp.zeros(DW),
          Wk=jnp.array(p0["Wr"]), lw=jnp.zeros(DW + ER.NA + NP), lb=jnp.zeros(()),
          ta=jnp.array(np.log(2.0 ** (-1 / 8) / (1 - 2.0 ** (-1 / 8)))))
wp = dict(F1=nrm(ks[1], HY + ER.NA, 256), Fb1=jnp.zeros(256), F2=nrm(ks[2], 256, HY, 0.1), Fb2=jnp.zeros(HY),     # F₀ (EP.init의 월드 모델 꼴)
          G1=nrm(ks[3], 2 * HY + ER.NA, 256), Gb1=jnp.zeros(256), G2=jnp.zeros((256, HY)), Gb2=jnp.zeros(HY))       # G, 마지막 층 0
kf = dict(ls=jnp.log(jnp.array(p13["sigma2"])), lr=jnp.log(jnp.array(p13["R"])), lq=jnp.log(jnp.array(p13["q"])))
th = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), dict(kp=kp, wp=wp, kf=kf))
POS = jnp.eye(NP, dtype=jnp.float32)


def F0(wp, o, a):
    x = jnp.concatenate([feat(o).reshape(-1), jax.nn.one_hot(a, ER.NA)])
    return (jax.nn.relu(x @ wp["F1"] + wp["Fb1"]) @ wp["F2"] + wp["Fb2"]).reshape(NP, H)


def G(wp, o, a, m):
    x = jnp.concatenate([feat(o).reshape(-1), jax.nn.one_hot(a, ER.NA), m.reshape(-1)])
    return (jax.nn.silu(x @ wp["G1"] + wp["Gb1"]) @ wp["G2"] + wp["Gb2"]).reshape(NP, H)


def key_fn(kp, o, a):
    w = feat(o)[:9].reshape(-1)
    wpos = jnp.concatenate([jnp.tile(w, (NP, 1)), POS], -1)
    u = w + jax.nn.silu(wpos @ kp["W1"] + kp["b1"]) @ kp["W2"] + kp["b2"]
    kh = u @ kp["Wk"][a].T
    li = jnp.concatenate([jnp.tile(jnp.concatenate([w, jax.nn.one_hot(a, ER.NA)]), (NP, 1)), POS], -1)
    ell = li @ kp["lw"] + kp["lb"]
    return jnp.exp(ell)[:, None] * kh / (jnp.linalg.norm(kh, axis=-1, keepdims=True) + 1e-8), ell


def step_e(kp, e, o, a):
    k, ell = key_fn(kp, o, a)
    return jnp.stack([k, jax.nn.sigmoid(kp["ta"]) * e[:, 1] + k], 1), ell


def write(th, O, acts, stop_t):
    kp, wp, kf = th["kp"], th["wp"], th["kf"]
    R, Q, sg = jnp.exp(kf["lr"]), jnp.exp(kf["lq"]), jnp.exp(0.5 * kf["ls"])

    @jax.checkpoint
    def f(c, x):
        M, P, e = c
        o, a, o2, t = x
        e, ell = step_e(kp, e, o, a)
        r = feat(o2) - F0(wp, o, a)                                       # 쓰기 기준: 기억 없는 예측의 잔차
        Z = sg[:, None] * e.reshape(NP, -1)                               # [12, DZ]
        Pq = (P + Q[:, None]) if DIAGP else (P + Q[:, None, None] * jnp.eye(DZ))
        m = jnp.einsum('phd,pd->ph', M, Z)                                # 쓰기 전 읽기 [12, 8]
        g_ = G(wp, o, a, m)
        # 대칭화 (Pq + Pqᵀ)/2는 역전파에서 P 덩어리 전치 복사 (12GB)를 만들어 OOM → 쓰지 않음
        PZ = (Pq * Z) if DIAGP else jnp.einsum('pij,pj->pi', Pq, Z, precision='highest'); S = R + R0 + jnp.sum(Z * PZ, -1)
        err_w = r - m                                                     # 칼만 쓰기 (G 없음)
        err = r - m - g_                                                  # 월드 모델 예측 오차
        nll = 0.5 * (jnp.sum(err ** 2, -1) / S + H * jnp.log(S))
        M2 = M + jnp.einsum('ph,pd->phd', err_w, PZ / S[:, None])
        P2 = (Pq / (1.0 + Pq * Z * Z / (R + R0)[:, None])) if DIAGP else (Pq - jnp.einsum('pi,pj->pij', PZ, PZ) / S[:, None, None])   # 대각: KDN식, 늘 양수
        keep = t > stop_t
        return (jnp.where(keep, M, M2), jnp.where(keep, P, P2), e), (jnp.where(keep, 0.0, nll.sum()), ell[EP.SLOT])
    c0 = (jnp.zeros((NP, H, DZ)), jnp.ones((NP, DZ)) if DIAGP else jnp.tile(jnp.eye(DZ), (NP, 1, 1)), jnp.zeros((NP, NPOLE, KD)))
    return jax.lax.scan(f, c0, (O[:-1], acts, O[1:], jnp.arange(acts.shape[0])))


def sess_nll(th, sess):
    O, _, _ = EL.observe_stream(sess)
    _, (nll, _) = write(th, O, sess["acts"], 10 ** 9)
    return nll.sum()


_, tr = EL.stack_sessions(range(50_000, 51_024), 256)
_, va = EL.stack_sessions(range(51_024, 51_152), 256)
print("lengths: train", tr["acts"].shape, "val", va["acts"].shape, flush=True)


DIAG = os.environ.get('DIAG')          # P 진단: 체크포인트로 학습 세계 64개 순전파, 칸별 zᵀPz·S·P 최소 고유값
if DIAG:
    _z = np.load(DIAG); thD = {a: {b: jnp.array(_z[f"{a}/{b}"]) for b in th[a]} for a in th}
    def diag_one(sess):
        kp, wp, kf = thD["kp"], thD["wp"], thD["kf"]
        R, Q, sg = jnp.exp(kf["lr"]), jnp.exp(kf["lq"]), jnp.exp(0.5 * kf["ls"])
        O, _, _ = EL.observe_stream(sess)
        def f(c, x):
            M, P, e = c
            o, a, o2 = x
            e, _ = step_e(kp, e, o, a)
            r = feat(o2) - F0(wp, o, a)
            Z = sg[:, None] * e.reshape(NP, -1)
            Pq = P + Q[:, None, None] * jnp.eye(DZ)
            m = jnp.einsum('phd,pd->ph', M, Z)
            PZ = jnp.einsum('pij,pj->pi', Pq, Z); zPz = jnp.sum(Z * PZ, -1); S = R + R0 + zPz
            M2 = M + jnp.einsum('ph,pd->phd', r - m, PZ / S[:, None])
            P2 = Pq - jnp.einsum('pi,pj->pij', PZ, PZ) / S[:, None, None]
            return (M2, P2, e), (zPz, S, jnp.sum((r - m) ** 2, -1) / S, jnp.sum(Z * Z, -1))
        c0 = (jnp.zeros((NP, H, DZ)), jnp.tile(jnp.eye(DZ), (NP, 1, 1)), jnp.zeros((NP, NPOLE, KD)))
        (M, P, _), (zPz, S, e2, zz) = jax.lax.scan(f, c0, (O[:-1], sess["acts"], O[1:]))
        eig = jax.vmap(lambda A: jnp.linalg.eigvalsh(0.5 * (A + A.T))[0])(P)
        return zPz.min(0), (zPz < 0).mean(0), S.min(0), e2.max(0), eig, zz.max(0), jnp.all(jnp.isfinite(P))
    dj = jax.jit(jax.vmap(diag_one))
    outs = [dj(jax.tree_util.tree_map(lambda x: x[i:i + 16], tr)) for i in range(0, 64, 16)]
    zmin, zneg, smin, e2max, eig, zzmax, fin = [np.concatenate([np.array(o[j]) for o in outs]) for j in range(7)]
    R_ = np.exp(np.array(thD["kf"]["lr"]))
    rep = dict(ckpt=DIAG, R0=R0, R=R_.tolist(), zPz_min=zmin.min(0).tolist(), frac_steps_zPz_negative=zneg.mean(0).tolist(),
               sessions_with_negative_zPz=int((zneg.max(1) > 0).sum()), S_min=smin.min(0).tolist(),
               max_err2_over_S=e2max.max(0).tolist(), P_eigmin_end=eig.min(0).tolist(), max_z_sq=zzmax.max(0).tolist(),
               P_all_finite=bool(fin.all()))
    for k, v in rep.items():
        print(k, ["%.3g" % x for x in v] if isinstance(v, list) else v, flush=True)
    json.dump(rep, open(os.environ['OUT'], 'w'), indent=1)
    sys.exit(0)


@jax.jit
def val_nll(th):
    def chunk(c, idx):
        sb = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, idx, 16), va)
        return c + jax.vmap(lambda s: sess_nll(th, s))(sb).sum(), None
    tot, _ = jax.lax.scan(chunk, 0.0, jnp.arange(0, 128, 16))
    return tot / 128


sessions, stack = EL.stack_sessions(range(1000, 1256), 256)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])


def judge_one(th, sess, plans, ai_all):
    """세 가지 읽기로 판정: syn = 시냅스 읽기 m (기존 판정과 같음), wm = m + G (월드 모델의 기억 몫), wm0 = G(m=0) (기억 끔)."""
    kp, wp, kf = th["kp"], th["wp"], th["kf"]
    sg = jnp.exp(0.5 * kf["ls"])
    O, s_end, _ = EL.observe_stream(sess)
    t_c = RU.last_arrival(O)
    (M, P, e), _ = write(th, O, sess["acts"], t_c)

    def sim(plan, ai):
        def f(c, a):
            s, o, e = c
            e, _ = step_e(kp, e, o, a)
            Z = sg[:, None] * e.reshape(NP, -1)
            m = jnp.einsum('phd,pd->ph', M, Z)
            out = jnp.stack([m[EP.SLOT], (m + G(wp, o, a, m))[EP.SLOT], G(wp, o, a, jnp.zeros_like(m))[EP.SLOT]])
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (s2, o2, e), out
        _, YH = jax.lax.scan(f, (s_end, O[-1], e), plan)
        yh = YH[ai]                                                        # [3, 8]
        return (yh @ gB) / (jnp.linalg.norm(yh, axis=-1) * jnp.linalg.norm(gB) + 1e-8)
    return jax.vmap(sim)(plans, ai_all)                                     # [NSITE, 3]


@jax.jit
def judge_chunk(th, sb, pl, ar):
    return jax.vmap(lambda s, a, b: judge_one(th, s, a, b))(sb, pl, ar)


def judge(th):
    sc = np.concatenate([np.array(judge_chunk(th, jax.tree_util.tree_map(lambda x: x[i:i + 32], stack), plans[i:i + 32], arrive[i:i + 32]))
                         for i in range(0, 256, 32)])                       # [256, NSITE, 3]
    out = {}
    for j, name in enumerate(["syn", "wm", "wm_memoff"]):
        x = sc[:, :, j]
        best = (x + 1e-9 * np.random.default_rng(0).random(x.shape)).argmax(1)
        out[name] = dict(b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)),
                         other=float(np.mean((best != b1) & (best != b0))), per_world_b1=(best == b1).astype(int).tolist())
    return out


def fmt(j):
    return "  ".join(f"{k}: b1 {v['b1']:.4f} b0 {v['b0']:.4f} other {v['other']:.4f}" for k, v in j.items())


OUT = os.environ.get('OUT', '/workspace/RL-TDP/docs/world/results/synapse_layer_20261117/wm_joint.json')
t0 = time.time()
v0 = float(val_nll(th)); j0 = judge(th)
print(f"[start = 무작위 월드 모델 + 12.13 칼만 값 + 무작위 키] val_nll {v0:.1f}  {fmt(j0)}  ({time.time()-t0:.0f}s)", flush=True)
ITERS = int(os.environ.get('ITERS', 10000)); JE = int(os.environ.get('JUDGE_EVERY', 500))
opt = optax.multi_transform({"kp": optax.adam(1e-3), "wp": optax.adam(1e-3), "kf": optax.adam(0.05)}, {"kp": "kp", "wp": "wp", "kf": "kf"})
st = opt.init(th)


@jax.jit
def upd(th, st, idx):
    def lf(th):
        sb = jax.tree_util.tree_map(lambda x: x[idx], tr)
        return jax.vmap(lambda s: sess_nll(th, s))(sb).mean()
    l, g = jax.value_and_grad(lf)(th)
    g, st = opt.update(g, st, th)
    return optax.apply_updates(th, g), st, l


def save(path, t_):
    np.savez(path, **{f"{a}/{b}": np.array(v) for a in t_ for b, v in t_[a].items()})


curve = [dict(it=0, val_nll=v0)]
best_v, best_th, best_it = v0, th, 0
rk = jax.random.PRNGKey(5)
for it in range(1, ITERS + 1):
    rk, k = jax.random.split(rk)
    th, st, l = upd(th, st, jax.random.randint(k, (8,), 0, 1024))
    if it <= 3 or it == 10:
        print(f"  [time] it {it} loss {float(l):.1f} ({time.time()-t0:.1f}s)", flush=True)
    if not np.isfinite(float(l)):
        print("nan at", it, flush=True); break
    if it % 100 == 0:
        v = float(val_nll(th))
        curve.append(dict(it=it, val_nll=v, train_nll=float(l)))
        if v < best_v:
            best_v, best_th, best_it = v, th, it
            save(OUT.replace('.json', '_best_params.npz'), best_th)
        print(f"it {it} train {float(l):.1f} val {v:.1f} best {best_v:.1f}@{best_it} half_life {float(-np.log(2)/np.log(jax.nn.sigmoid(th['kp']['ta']))):.2f} ({time.time()-t0:.0f}s)", flush=True)
        if it % JE == 0:
            jj = judge(th)
            curve[-1].update(judge={k: {kk: vv for kk, vv in v_.items() if kk != 'per_world_b1'} for k, v_ in jj.items()})
            print(f"  [judge it {it}] {fmt(jj)}", flush=True)

j1 = judge(best_th)
res = dict(spec="wm_joint", R0=R0, before=dict(val_nll=v0, judge=j0), after=dict(val_nll=best_v, best_it=best_it, judge=j1,
           sigma2=np.exp(np.array(best_th["kf"]["ls"])).tolist(), R=np.exp(np.array(best_th["kf"]["lr"])).tolist(),
           q=np.exp(np.array(best_th["kf"]["lq"])).tolist()), curve=curve)
print(f"[after] val_nll {best_v:.1f}@{best_it}  {fmt(j1)}", flush=True)
json.dump(res, open(OUT, 'w'))
save(OUT.replace('.json', '_params.npz'), best_th)
