# 단계 A · 기억을 입력받는 월드 모델 (명세 docs/world/spec_memory_wm_agent.md 4절), 2026-11-18
#   ŷ_t(a) = m_t(a) + f_θ(x_t, a, m_t(a)),  m_p = M_p z_p (쓰기 전 읽기)
#   쓰기: v = y − ŷ (최종 오차), P⁻ = P + qI, S = R₀ + R_p + zᵀP⁻z, g = P⁻z/S, M ← M + v gᵀ, P ← P⁻ − S g gᵀ
#   학습: Σ ½(‖v_p‖²/S_p + 8 log S_p), f_θ·키·흔적·칼만을 쓰기 재귀를 통해 함께.
#   MEMIN=1 본판 (f가 m을 입력으로), MEMIN=0 대조 (f(x, a, 0), 바깥 +m 경로는 같음).
# 출발점 (두 판 같음): f_θ의 (x, a) 가중치 = 고정 월드 모델 wm_q8.npz (ReLU 256, 평균 96만), 기억 입력 열 0.
#   키·흔적·칼만 = q8_base_frozen_1k_best_params.npz (11-17 기준선 70.7%). 따라서 0반복 판정은 70.7%여야 한다 (구현 확인).
#   명세의 SiLU 대신 ReLU: wm_q8을 그대로 쓰기 위함.
# 판정 (500반복마다): 다섯 후보 경로의 도착 예측에서
#   mem  = cos(ŷ(m) − ŷ(0), g_B)   (예측의 기억 몫; MEMIN=0이면 m과 같아 기존 판정과 같음)
#   full = cos(ŷ − φ̄, g_B)          (전체 예측)
#   b1 경로 도착 칸 오차 ‖ŷ − y‖², b1과 차순위 점수 차, |f(m) − f(0)|, 주소 일치 (stage0과 같은 정의), P 최소 고유값.
# 환경변수: MEMIN (1), ITERS (2000), JUDGE_EVERY (500), OUT
import sys, os, json, time
sys.path.insert(0, '/workspace/RL-TDP')
import numpy as np, jax, jax.numpy as jnp, optax
from rltdp.world import errand as ER, errand_learned as EL, errand_prec as EP, errand_reuse as RU

ROOT = '/workspace/RL-TDP'
RES17 = f'{ROOT}/docs/world/results/synapse_layer_20261117'
MEMIN = os.environ.get('MEMIN', '1') == '1'
ITERS = int(os.environ.get('ITERS', 2000)); JE = int(os.environ.get('JUDGE_EVERY', 500))
OUT = os.environ.get('OUT', f'{ROOT}/docs/world/results/stageA_20261118/A_memin{int(MEMIN)}.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
ER.PUBLIC = False

We, ybar = EL.emb_fixed(0)
NP, H, KD, NPOLE = EP.NP, EP.H, EP.DK, 2
DZ, DW, HY, SLOT = NPOLE * KD, 9 * EP.H, EP.NP * EP.H, EP.SLOT
R0 = float(np.mean(np.sum(np.array(We) ** 2, 0))) / 255.0 ** 2 / 12.0
phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
gB, phibar = phi[1] - phi.mean(0), phi.mean(0)
POS = jnp.eye(NP, dtype=jnp.float32)
print("MEMIN", MEMIN, "R0", R0, flush=True)

# ---- 매개변수
wq = np.load(f'{RES17}/wm_q8.npz')
bz = np.load(f'{RES17}/q8_base_frozen_1k_best_params.npz')
fp = dict(W1=jnp.concatenate([jnp.array(wq['wm1']), jnp.zeros((HY, 256))]), b1=jnp.array(wq['wb1']),   # 입력 [x 96; a 6; m 96], m 열 0
          W2=jnp.array(wq['wm2'][:, :HY]), b2=jnp.array(wq['wb2'][:HY]))
kp = {k: jnp.array(bz[f'kp/{k}']) for k in ('W1', 'b1', 'W2', 'b2', 'Wk', 'lw', 'lb', 'ta')}
kf = {k: jnp.array(bz[f'kf/{k}']) for k in ('ls', 'lr', 'lq')}
th = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), dict(fp=fp, kp=kp, kf=kf))


def feat(o):
    return ((o @ We) - ybar).reshape(NP, H)


def f_net(fp, o, a, m):
    x = jnp.concatenate([feat(o).reshape(-1), jax.nn.one_hot(a, ER.NA), m.reshape(-1)])
    return (jax.nn.relu(x @ fp["W1"] + fp["b1"]) @ fp["W2"] + fp["b2"]).reshape(NP, H)


def predict(fp, o, a, m):
    """ŷ = m + f(x, a, m) (본판) 또는 m + f(x, a, 0) (대조)."""
    return m + f_net(fp, o, a, m if MEMIN else jnp.zeros_like(m))


def key_fn(kp, o, a):
    w = feat(o)[:9].reshape(-1)
    u = w + jax.nn.silu(jnp.concatenate([jnp.tile(w, (NP, 1)), POS], -1) @ kp["W1"] + kp["b1"]) @ kp["W2"] + kp["b2"]
    kh = u @ kp["Wk"][a].T
    ell = jnp.concatenate([jnp.tile(jnp.concatenate([w, jax.nn.one_hot(a, ER.NA)]), (NP, 1)), POS], -1) @ kp["lw"] + kp["lb"]
    return jnp.exp(ell)[:, None] * kh / (jnp.linalg.norm(kh, axis=-1, keepdims=True) + 1e-8)


def step_e(kp, e, o, a):
    k = key_fn(kp, o, a)
    return jnp.stack([k, jax.nn.sigmoid(kp["ta"]) * e[:, 1] + k], 1)


def write(th, O, acts, stop_t):
    fp, kp, kf = th["fp"], th["kp"], th["kf"]
    R, Q, sg = jnp.exp(kf["lr"]), jnp.exp(kf["lq"]), jnp.exp(0.5 * kf["ls"])

    @jax.checkpoint
    def f(c, x):
        M, P, e = c
        o, a, o2, t = x
        e = step_e(kp, e, o, a)
        Z = sg[:, None] * e.reshape(NP, -1)
        m = jnp.einsum('phd,pd->ph', M, Z)
        v = feat(o2) - predict(fp, o, a, m)                               # 최종 오차: 쓰기와 학습에 같이
        Pq = P + Q[:, None, None] * jnp.eye(DZ)
        PZ = jnp.einsum('pij,pj->pi', Pq, Z, precision='highest')
        S = R + R0 + jnp.sum(Z * PZ, -1)
        nll = 0.5 * (jnp.sum(v ** 2, -1) / S + H * jnp.log(S))
        M2 = M + jnp.einsum('ph,pd->phd', v, PZ / S[:, None])
        P2 = Pq - jnp.einsum('pi,pj->pij', PZ, PZ) / S[:, None, None]
        keep = t > stop_t
        return (jnp.where(keep, M, M2), jnp.where(keep, P, P2), e), (jnp.where(keep, 0.0, nll.sum()), Z[SLOT], S.min())
    c0 = (jnp.zeros((NP, H, DZ)), jnp.tile(jnp.eye(DZ), (NP, 1, 1)), jnp.zeros((NP, NPOLE, KD)))
    return jax.lax.scan(f, c0, (O[:-1], acts, O[1:], jnp.arange(acts.shape[0])))


def sess_nll(th, sess):
    O, _, _ = EL.observe_stream(sess)
    _, (nll, _, _) = write(th, O, sess["acts"], 10 ** 9)
    return nll.sum()


_, tr = EL.stack_sessions(range(50_000, 51_024), 256)
_, va = EL.stack_sessions(range(51_024, 51_152), 256)
sessions, stack = EL.stack_sessions(range(1000, 1256), 256)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])
pw = json.load(open(f'{ROOT}/docs/world/results/stage0_20261118/stage0_per_world.json'))['init']
t_arr_b1 = jnp.array([r['t_arrive_b1'][0] for r in pw])                   # 교사 때 b1 도착 스텝 (stage0, 교사 기록은 모델과 무관)
assert all(r['b1'] == int(b1[i]) for i, r in enumerate(pw))
print("lengths: train", tr["acts"].shape, "val", va["acts"].shape, flush=True)


@jax.jit
def val_nll(th):
    def chunk(c, idx):
        sb = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, idx, 16), va)
        return c + jax.vmap(lambda s: sess_nll(th, s))(sb).sum(), None
    tot, _ = jax.lax.scan(chunk, 0.0, jnp.arange(0, 128, 16))
    return tot / 128


def cosv(y, g):
    return (y @ g) / (jnp.linalg.norm(y, axis=-1) * jnp.linalg.norm(g) + 1e-8)


def judge_one(th, sess, plans, ai_all, tb1):
    fp, kp, kf = th["fp"], th["kp"], th["kf"]
    sg = jnp.exp(0.5 * kf["ls"])
    O, s_end, _ = EL.observe_stream(sess)
    t_c = RU.last_arrival(O)
    (M, P, e), (_, Zs, Smin) = write(th, O, sess["acts"], t_c)
    z_b1 = Zs[tb1]

    def sim(plan, ai):
        def f(c, a):
            s, o, e = c
            e = step_e(kp, e, o, a)
            Z = sg[:, None] * e.reshape(NP, -1)
            m = jnp.einsum('phd,pd->ph', M, Z)
            y_m = predict(fp, o, a, m)
            y_0 = predict(fp, o, a, jnp.zeros_like(m))
            fd = f_net(fp, o, a, m) - f_net(fp, o, a, jnp.zeros_like(m))
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (s2, o2, e), (y_m[SLOT], y_0[SLOT], fd[SLOT], m[SLOT], Z[SLOT], feat(o2)[SLOT])
        _, out = jax.lax.scan(f, (s_end, O[-1], e), plan)
        y_m, y_0, fd, m, Z, y = (x[ai] for x in out)
        return (cosv(y_m - y_0, gB), cosv(y_m - phibar, gB), jnp.sum((y_m - y) ** 2), jnp.sum((y_0 - y) ** 2),
                jnp.linalg.norm(fd), jnp.linalg.norm(m), cosv(Z, z_b1))
    res = jax.vmap(sim)(plans, ai_all)
    eig = jnp.linalg.eigvalsh(0.5 * (P[SLOT] + P[SLOT].T))[0]
    return res + (eig, Smin.min(), jnp.all(jnp.isfinite(M)) & jnp.all(jnp.isfinite(P)))


judge_chunk = jax.jit(jax.vmap(judge_one, in_axes=(None, 0, 0, 0, 0)))


def judge(th):
    outs = [jax.tree_util.tree_map(np.array, judge_chunk(th, jax.tree_util.tree_map(lambda x: x[i:i + 32], stack),
                                                          plans[i:i + 32], arrive[i:i + 32], t_arr_b1[i:i + 32])) for i in range(0, 256, 32)]
    mem, full, mse_m, mse_0, fdn, mn, ac, eig, smin, fin = [np.concatenate([o[j] for o in outs]) for j in range(10)]
    W = np.arange(256)
    rnd = 1e-9 * np.random.default_rng(0).random(mem.shape)
    out = {}
    for name, sc in (('mem', mem), ('full', full)):
        best = (sc + rnd).argmax(1)
        oth = np.where(np.arange(5)[None] == b1[:, None], -np.inf, sc).max(1)
        out[name] = dict(b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)), other=float(np.mean((best != b1) & (best != b0))),
                         margin_b1_mean=float(np.mean(sc[W, b1] - oth)), per_world_b1=(best == b1).astype(int).tolist())
    out['b1_arrival_mse'] = dict(with_mem=float(mse_m[W, b1].mean()), mem_zero=float(mse_0[W, b1].mean()),
                                 per_world_with_mem=mse_m[W, b1].tolist(), per_world_mem_zero=mse_0[W, b1].tolist())
    out['f_mem_response_b1'] = float(fdn[W, b1].mean())        # |f(x,a,m) − f(x,a,0)|
    out['m_norm_b1'] = float(mn[W, b1].mean())
    oth_ac = np.where(np.arange(5)[None] == b1[:, None], -np.inf, ac).max(1)
    out['addr'] = dict(b1_best=float(np.mean(ac.argmax(1) == b1)), cos_b1_median=float(np.median(ac[W, b1])), cos_other_max_median=float(np.median(oth_ac)))
    out['num'] = dict(P_slot_eigmin_min=float(eig.min()), S_min=float(smin.min()), all_finite=bool(fin.all()))
    return out


def fmt(j):
    return (f"mem b1 {j['mem']['b1']:.4f} b0 {j['mem']['b0']:.4f} other {j['mem']['other']:.4f} | full b1 {j['full']['b1']:.4f} b0 {j['full']['b0']:.4f} | "
            f"b1 mse {j['b1_arrival_mse']['with_mem']:.4f} (m=0 {j['b1_arrival_mse']['mem_zero']:.4f}) |f(m)-f(0)| {j['f_mem_response_b1']:.4f} "
            f"| addr b1best {j['addr']['b1_best']:.3f} ({j['addr']['cos_b1_median']:.3f}/{j['addr']['cos_other_max_median']:.3f}) "
            f"| Pmin {j['num']['P_slot_eigmin_min']:.2e} Smin {j['num']['S_min']:.2e} fin {j['num']['all_finite']}")


def strip(j):
    return {k: ({kk: vv for kk, vv in v.items() if not kk.startswith('per_world')} if isinstance(v, dict) else v) for k, v in j.items()}


def save(path, t_):
    np.savez(path, **{f"{a}/{b}": np.array(v) for a in t_ for b, v in t_[a].items()})


t0 = time.time()
v0 = float(val_nll(th)); j0 = judge(th)
print(f"[it 0] val_nll {v0:.1f}  {fmt(j0)}  ({time.time()-t0:.0f}s)", flush=True)
opt = optax.multi_transform({"fp": optax.adam(1e-3), "kp": optax.adam(1e-3), "kf": optax.adam(0.05)}, {"fp": "fp", "kp": "kp", "kf": "kf"})
st = opt.init(th)


@jax.jit
def upd(th, st, idx):
    def lf(th):
        sb = jax.tree_util.tree_map(lambda x: x[idx], tr)
        return jax.vmap(lambda s: sess_nll(th, s))(sb).mean()
    l, g = jax.value_and_grad(lf)(th)
    g, st = opt.update(g, st, th)
    return optax.apply_updates(th, g), st, l


curve = [dict(it=0, val_nll=v0, judge=strip(j0))]
best_v, best_th, best_it = v0, th, 0
rk = jax.random.PRNGKey(5)
for it in range(1, ITERS + 1):
    rk, k = jax.random.split(rk)
    th, st, l = upd(th, st, jax.random.randint(k, (8,), 0, 1024))
    if it <= 3 or it == 10:
        print(f"  [time] it {it} loss {float(l):.1f} ({time.time()-t0:.1f}s)", flush=True)
    if not np.isfinite(float(l)):
        print("nan at", it, flush=True); curve.append(dict(it=it, nan=True)); break
    if it % 100 == 0:
        v = float(val_nll(th))
        curve.append(dict(it=it, val_nll=v, train_nll=float(l)))
        if v < best_v:
            best_v, best_th, best_it = v, th, it
            save(OUT.replace('.json', '_best_params.npz'), best_th)
        print(f"it {it} train {float(l):.1f} val {v:.1f} best {best_v:.1f}@{best_it} ({time.time()-t0:.0f}s)", flush=True)
        if it % JE == 0:
            jj = judge(th)
            curve[-1].update(judge=strip(jj))
            save(OUT.replace('.json', f'_it{it}_params.npz'), th)
            print(f"  [judge it {it}] {fmt(jj)}", flush=True)
            json.dump(dict(spec='stageA', memin=MEMIN, curve=curve), open(OUT.replace('.json', '_curve.json'), 'w'))

j1 = judge(best_th)
res = dict(spec='stageA', memin=MEMIN, R0=R0, act='relu (wm_q8)', start=dict(val_nll=v0, judge=j0),
           best=dict(val_nll=best_v, best_it=best_it, judge=j1, sigma2=np.exp(np.array(best_th["kf"]["ls"])).tolist(),
                     R=np.exp(np.array(best_th["kf"]["lr"])).tolist(), q=np.exp(np.array(best_th["kf"]["lq"])).tolist(),
                     half_life=float(-np.log(2) / np.log(jax.nn.sigmoid(best_th['kp']['ta'])))), curve=curve)
print(f"[best] val_nll {best_v:.1f}@{best_it}  {fmt(j1)}", flush=True)
json.dump(res, open(OUT, 'w'))
