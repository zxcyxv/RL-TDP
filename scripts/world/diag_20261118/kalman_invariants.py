# 도착 칸 칼만의 게이지 불변량 (2026-11-18).
# 게이지: z = σ·exp(ℓ)·[k̂; e]이고 P₀ = I, P⁻ = P + qI이므로 (z → c z, P → P/c², q → q/c²)나 σ ↔ exp(ℓ)는 같은 모델이다.
# q, R, σ², 키 크기 각각의 값은 의미가 없고, 아래 조합만 본다 (모두 순전파 값).
#   κ_t = z_tᵀ P⁻_t z_t / S_t                (쓰기 t의 덮어쓰기 비율, 0–1: 그 주소 방향의 기존 내용을 얼마나 갈아치우나)
#   ρ_t = q |z_t|² / (R + R₀)                (스텝당 망각을 관측 잡음 단위로; 쓰기 주소 기준)
#   판정 읽기 주소 z_q 기준: ρ_q = q |z_q|² / (R + R₀),  π_q = z_qᵀ P_T z_q / (R + R₀) (읽는 방향의 남은 불확실성)
#   w_t: 판정 읽기 m = Σ_t w_t r_t 의 쓰기별 가중치 (정확 분해), 쓰기 나이 (t_c − t)별로 묶음.
# 모델 코드는 stageA.py (MEMIN=1). q8_learned (f 고정) 체크포인트는 fp = 출발점 (wm_q8, m 열 0)으로 같은 코드에서 기준선과 정확히 같다.
#   대조판 (MEMIN=0)은 f의 m 열이 학습되지 않아 0이므로 같은 코드로 같은 순전파.
# 환경변수: CKPTS (이름=경로, 쉼표; 경로 'start'), OUT
import sys, os, json, time
import numpy as np

ROOT = '/workspace/RL-TDP'
SRC = f'{ROOT}/scripts/world/diag_20261118/stageA.py'
OUT = os.environ['OUT']
os.environ['MEMIN'] = '1'
t0 = time.time()
src = open(SRC).read()
ns = {'__name__': 'stageA_defs'}
exec(compile(src[:src.index("t0 = time.time()\nv0 = float(val_nll(th))")], SRC, 'exec'), ns)
jax, jnp = ns['jax'], ns['jnp']
ER, EL, EP, RU = ns['ER'], ns['EL'], ns['EP'], ns['RU']
th0, stack, plans, arrive, b1, b0 = ns['th'], ns['stack'], ns['plans'], ns['arrive'], ns['b1'], ns['b0']
feat, f_net, step_e, R0 = ns['feat'], ns['f_net'], ns['step_e'], ns['R0']
NP, H, DZ, KD, NPOLE, SLOT = ns['NP'], ns['H'], ns['DZ'], ns['KD'], ns['NPOLE'], ns['SLOT']
JCH = 32
CATS = ["b1_B", "b0_B", "b0_C", "other_site", "non_arrival"]
AGE_EDGES = [0, 1, 8, 32, 128, 1024]
AGE_NAMES = ["0", "1-7", "8-31", "32-127", "128+"]
print(f"defs loaded ({time.time()-t0:.0f}s)", flush=True)


def one(th, sess, plans_w, ai_all):
    fp, kp, kf = th["fp"], th["kp"], th["kf"]
    R, Q, sg = jnp.exp(kf["lr"]), jnp.exp(kf["lq"]), jnp.exp(0.5 * kf["ls"])
    O, s_end, _ = EL.observe_stream(sess)
    t_c = RU.last_arrival(O)
    T = sess["acts"].shape[0]
    s0, _ = EL.env0(sess)
    change = jnp.arange(T) == sess["change_t"]

    def fenv(c, x):
        st, cur = c
        a, rq, ri, ch = x
        st2, _, _, site = ER.env_step(st, a, rq, ri, ch, sess["b0"])
        return (st2, jnp.where(site >= 0, site, cur)), jnp.where(site >= 0, site, cur)
    _, cur_site = jax.lax.scan(fenv, (s0, jnp.array(-1)), (sess["acts"], sess["req"], sess["rid"], change))
    arrived = ~jnp.all(jnp.isclose(O[1:, SLOT], ER.TEX[ER.BLANK].reshape(-1)), -1)
    items = ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1)
    item = jnp.argmin(jnp.sum((O[1:, SLOT][:, None] - items[None]) ** 2, -1), -1)
    cat = jnp.stack([arrived & (cur_site == sess["b1"]), arrived & (cur_site == sess["b0"]) & (item == 1),
                     arrived & (cur_site == sess["b0"]) & (item == 2), arrived & (cur_site != sess["b0"]) & (cur_site != sess["b1"]),
                     ~arrived]).astype(jnp.float32)
    Rs, Qs = R[SLOT] + R0, Q[SLOT]

    def fw(c, x):
        M, P, e = c
        o, a, o2, t = x
        e = step_e(kp, e, o, a)
        Z = sg[:, None] * e.reshape(NP, -1)
        m = jnp.einsum('phd,pd->ph', M, Z)
        r = feat(o2) - f_net(fp, o, a, m)
        Pq = P + Q[:, None, None] * jnp.eye(DZ)
        PZ = jnp.einsum('pij,pj->pi', Pq, Z, precision='highest')
        zPz = jnp.sum(Z * PZ, -1)
        S = R + R0 + zPz
        g = PZ / S[:, None]
        on = t <= t_c
        M2 = jnp.where(on, M + jnp.einsum('ph,pd->phd', r - m, g), M)
        P2 = jnp.where(on, Pq - jnp.einsum('pi,pj->pij', PZ, PZ) / S[:, None, None], P)
        z = Z[SLOT]
        return (M2, P2, e), (z, jnp.where(on, 1.0, 0.0) * g[SLOT], zPz[SLOT] / S[SLOT], Qs * (z @ z) / Rs)
    c0 = (jnp.zeros((NP, H, DZ)), jnp.tile(jnp.eye(DZ), (NP, 1, 1)), jnp.zeros((NP, NPOLE, KD)))
    (MT, PT, e_end), (Z, G, kap, rho) = jax.lax.scan(fw, c0, (O[:-1], sess["acts"], O[1:], jnp.arange(T)))

    def probe(plan, ai):
        def f(c, a):
            st, o, e = c
            e = step_e(kp, e, o, a)
            st2, o2, _, _ = ER.env_step(st, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (st2, o2, e), (sg[SLOT] * e[SLOT].reshape(-1))
        _, Zs = jax.lax.scan(f, (s_end, O[-1], e_end), plan)
        return Zs[ai]
    Zq = jax.vmap(probe)(plans_w, ai_all)                                     # [5, DZ]

    def weights(zq):
        def fb(u, x):
            z, g = x
            w = g @ u
            return u - z * w, w
        _, Wt = jax.lax.scan(fb, zq, (Z, G), reverse=True)
        return Wt
    W = jax.vmap(weights)(Zq)                                                 # [5, T]
    age = t_c - jnp.arange(T)
    live = age >= 0
    bins = jnp.stack([(age >= lo) & (age < hi) & live for lo, hi in zip(AGE_EDGES[:-1], AGE_EDGES[1:])]).astype(jnp.float32)   # [nb, T]
    arr = arrived.astype(jnp.float32)
    w_age_arr = W @ (bins * arr).T; w_age_non = W @ (bins * (1 - arr)).T      # [5, nb]
    absw_age = jnp.abs(W) @ bins.T
    t_last = jnp.max(jnp.where(arrived & live, jnp.arange(T), -1))
    t_b1 = jnp.min(jnp.where(cat[0] > 0, jnp.arange(T), T))
    PTs = PT[SLOT]
    rho_q = Qs * jnp.sum(Zq * Zq, -1) / Rs
    pi_q = jnp.einsum('sd,de,se->s', Zq, PTs, Zq) / Rs
    livef = live.astype(jnp.float32)
    kap_cat = (cat * livef) @ kap / jnp.maximum((cat * livef).sum(1), 1)    # [5c]
    rho_cat = (cat * livef) @ rho / jnp.maximum((cat * livef).sum(1), 1)
    return dict(kap_cat=kap_cat, rho_cat=rho_cat, w_age_arr=w_age_arr, w_age_non=w_age_non, absw_age=absw_age,
                w_last=W[:, t_last], w_b1=W[:, t_b1], rho_q=rho_q, pi_q=pi_q, gap=(t_c - t_b1).astype(jnp.float32))


one_j = jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0)))
Wd = np.arange(256)


def run(th):
    rs = [jax.tree_util.tree_map(np.array, one_j(th, jax.tree_util.tree_map(lambda x: x[i:i + JCH], stack), plans[i:i + JCH], arrive[i:i + JCH]))
          for i in range(0, 256, JCH)]
    F = {k: np.concatenate([r[k] for r in rs]) for k in rs[0]}
    oth = (np.arange(5)[None] != b1[:, None]) & (np.arange(5)[None] != b0[:, None])
    def by_probe(x):                                                           # x [256, 5, ...] → b1 후보, 다른 곳 후보 평균
        o = (x * oth.reshape(oth.shape + (1,) * (x.ndim - 2))).sum(1) / oth.sum(1).reshape((-1,) + (1,) * (x.ndim - 2))
        return dict(probe_b1=np.round(x[Wd, b1].mean(0), 4).tolist(), probe_other=np.round(o.mean(0), 4).tolist())
    return dict(
        overwrite_kappa_by_event=dict(zip(CATS, np.round(F['kap_cat'].mean(0), 4).tolist())),
        forget_rho_by_event=dict(zip(CATS, [float('%.3g' % v) for v in F['rho_cat'].mean(0)])),
        read_rho_q=by_probe(F['rho_q']), read_pi_q=by_probe(F['pi_q']),
        gap_b1_to_stop_median=float(np.median(F['gap'])),
        forget_over_gap_rho_q_x_gap=dict(probe_b1=float(np.median(F['rho_q'][Wd, b1] * F['gap']))),
        w_on_b1_write=by_probe(F['w_b1']), w_on_last_arrival_write=by_probe(F['w_last']),
        age_bins=AGE_NAMES, w_by_age_arrivals=by_probe(F['w_age_arr']), w_by_age_nonarrivals=by_probe(F['w_age_non']),
        absw_by_age=by_probe(F['absw_age']))


rep = {}
for item in os.environ['CKPTS'].split(','):
    name, path = item.split('=')
    if path == 'start':
        th = th0
    else:
        z = np.load(path)
        th = {a: {b: (jnp.array(z[f"{a}/{b}"]) if f"{a}/{b}" in z.files else th0[a][b]) for b in th0[a]} for a in th0}
        if th["fp"]["W2"].shape[1] > th0["fp"]["W2"].shape[1]:                # 분할판 f₀ (평균 96 + 로그 분산 12): 평균 부분만 (순전파 같음)
            th["fp"]["W2"], th["fp"]["b2"] = th["fp"]["W2"][:, :th0["fp"]["W2"].shape[1]], th["fp"]["b2"][:th0["fp"]["b2"].shape[0]]
        if not any(k.startswith('fp/') for k in z.files):
            print(f"  {name}: fp from start (f fixed)", flush=True)
    rep[name] = run(th)
    print(f"== {name} ({time.time()-t0:.0f}s)\n" + json.dumps(rep[name], ensure_ascii=False), flush=True)
    json.dump(rep, open(OUT, 'w'), ensure_ascii=False, indent=1)
