# 단계 A 체크포인트의 읽기 분해 (2026-11-18).
# 판정 읽기 (MEMIN=1) = m + Δf,  m = M_SLOT z_q,  Δf = f(x, a, m) − f(x, a, 0) (도착 칸).
#   m: 쓰기 재귀 M_{t+1} = M_t (I − z_t g_tᵀ) + r_t g_tᵀ,  r_t = y_t − f(x_t, a_t, m_t)  →  m = Σ_t w_t r_t (정확; f가 m에 비선형이어도 r_t는 그때의 값)
#      w_t = g_tᵀ u_{t+1}, u_t = (I − g_t z_tᵀ)ᵀ… 즉 u_t = u_{t+1} − z_t (g_tᵀ u_{t+1}), u_T = z_q   (synapse_cellkey.decomp_one과 같은 역방향 재귀)
#   Δf: m에 비선형이라 쓰기별로 나눌 수 없다. 대신 판정 점수에서 m과 Δf 각각의 크기·B 방향과, 각각만으로 매긴 판정을 기록.
# 사건 (교사 세션 도착 칸 쓰기): b1 도착 (B), b0 도착 B, b0 도착 C, 다른 채집처 도착, 도착 없음.
#   후보 b1, b0은 위 사건별로. 그 밖의 후보 ("다른 곳", A 판정의 주된 오답)는 [자기 채집처 도착, b1, b0 B, b0 C, 나머지 도착, 도착 없음].
# 환경변수: CKPTS (쉼표, 'start' 허용), OUT. MEMIN=1로 stageA 정의를 불러 쓴다.
import sys, os, json, time
import numpy as np

ROOT = '/workspace/RL-TDP'
SRC = f'{ROOT}/scripts/world/diag_20261118/stageA.py'
OUT = os.environ['OUT']
os.environ['MEMIN'] = '1'
t0 = time.time()
src = open(SRC).read()
cut = src.index("t0 = time.time()\nv0 = float(val_nll(th))")
ns = {'__name__': 'stageA_defs'}
exec(compile(src[:cut], SRC, 'exec'), ns)
jax, jnp = ns['jax'], ns['jnp']
ER, EL, EP, RU = ns['ER'], ns['EL'], ns['EP'], ns['RU']
th0, stack, plans, arrive, b1, b0 = ns['th'], ns['stack'], ns['plans'], ns['arrive'], ns['b1'], ns['b0']
feat, f_net, predict, step_e, gB, R0 = ns['feat'], ns['f_net'], ns['predict'], ns['step_e'], ns['gB'], ns['R0']
NP, H, DZ, KD, NPOLE, SLOT = ns['NP'], ns['H'], ns['DZ'], ns['KD'], ns['NPOLE'], ns['SLOT']
JCH = 32
print(f"defs loaded ({time.time()-t0:.0f}s)", flush=True)
CATS = ["b1_B", "b0_B", "b0_C", "other_site", "non_arrival"]
CATS_O = ["own_site", "b1_B", "b0_B", "b0_C", "rest_arrivals", "non_arrival"]
gBn = gB / jnp.linalg.norm(gB)


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
        cur = jnp.where(site >= 0, site, cur)
        return (st2, cur), cur
    _, cur_site = jax.lax.scan(fenv, (s0, jnp.array(-1)), (sess["acts"], sess["req"], sess["rid"], change))
    blank = ER.TEX[ER.BLANK].reshape(-1)
    arrived = ~jnp.all(jnp.isclose(O[1:, SLOT], blank), -1)
    items = ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1)
    item = jnp.argmin(jnp.sum((O[1:, SLOT][:, None] - items[None]) ** 2, -1), -1)

    # stageA.write와 같은 쓰기 (전 칸; f가 전 칸 m을 입력으로 받으므로), 도착 칸의 z, g, r 기록
    def fw(c, x):
        M, P, e = c
        o, a, o2, t = x
        e = step_e(kp, e, o, a)
        Z = sg[:, None] * e.reshape(NP, -1)
        m = jnp.einsum('phd,pd->ph', M, Z)
        fo = f_net(fp, o, a, m)
        r = feat(o2) - fo                                                 # 쓰기 목표 r_t (v = r − m)
        Pq = P + Q[:, None, None] * jnp.eye(DZ)
        PZ = jnp.einsum('pij,pj->pi', Pq, Z, precision='highest')
        S = R + R0 + jnp.sum(Z * PZ, -1)
        g = PZ / S[:, None]
        on = t <= t_c
        M2 = jnp.where(on, M + jnp.einsum('ph,pd->phd', r - m, g), M)
        P2 = jnp.where(on, Pq - jnp.einsum('pi,pj->pij', PZ, PZ) / S[:, None, None], P)
        return (M2, P2, e), (Z[SLOT], jnp.where(on, 1.0, 0.0) * g[SLOT], r[SLOT])
    c0 = (jnp.zeros((NP, H, DZ)), jnp.tile(jnp.eye(DZ), (NP, 1, 1)), jnp.zeros((NP, NPOLE, KD)))
    (MT, _, e_end), (Z, G, Rr) = jax.lax.scan(fw, c0, (O[:-1], sess["acts"], O[1:], jnp.arange(T)))

    def probe(plan, ai):
        def f(c, a):
            st, o, e = c
            e = step_e(kp, e, o, a)
            Zf = sg[:, None] * e.reshape(NP, -1)
            m = jnp.einsum('phd,pd->ph', MT, Zf)
            df = f_net(fp, o, a, m) - f_net(fp, o, a, jnp.zeros_like(m))
            st2, o2, _, _ = ER.env_step(st, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (st2, o2, e), (Zf[SLOT], m[SLOT], df[SLOT])
        _, (Zs, ms, dfs) = jax.lax.scan(f, (s_end, O[-1], e_end), plan)
        return Zs[ai], ms[ai], dfs[ai]
    Zq, mq, dfq = jax.vmap(probe)(plans_w, ai_all)                          # [5, DZ], [5, 8], [5, 8]

    def weights(zq):
        def fb(u, x):
            z, g = x
            w = g @ u
            return u - z * w, w
        _, Wt = jax.lax.scan(fb, zq, (Z, G), reverse=True)
        return Wt
    W = jax.vmap(weights)(Zq)                                                # [5, T]
    recon = jnp.max(jnp.abs(W @ Rr - mq))
    bp = Rr @ gBn                                                            # 쓰기 목표의 B 방향 성분 [T]
    cat = jnp.stack([arrived & (cur_site == sess["b1"]), arrived & (cur_site == sess["b0"]) & (item == 1),
                     arrived & (cur_site == sess["b0"]) & (item == 2),
                     arrived & (cur_site != sess["b0"]) & (cur_site != sess["b1"]), ~arrived]).astype(jnp.float32)   # [5c, T]
    wsum = W @ cat.T; bproj = (W * bp[None]) @ cat.T                         # [5 probe, 5c]
    # 후보별 자기 채집처 기준 범주
    sites = jnp.arange(5)
    own = arrived[None] & (cur_site[None] == sites[:, None])                 # [5, T]
    c_b1 = arrived[None] & (cur_site[None] == sess["b1"]) & ~own
    c_b0B = (arrived & (cur_site == sess["b0"]) & (item == 1))[None] & ~own
    c_b0C = (arrived & (cur_site == sess["b0"]) & (item == 2))[None] & ~own
    rest = arrived[None] & ~own & ~c_b1 & ~c_b0B & ~c_b0C
    non = jnp.broadcast_to(~arrived, own.shape)
    catO = jnp.stack([own, c_b1, c_b0B, c_b0C, rest, non], 1).astype(jnp.float32)    # [5, 6, T]
    wsumO = jnp.einsum('pt,pct->pc', W, catO); bprojO = jnp.einsum('pt,pct,t->pc', W, catO, bp)
    cosg = lambda y: (y @ gB) / (jnp.linalg.norm(y, axis=-1) * jnp.linalg.norm(gB) + 1e-8)
    sc = jnp.stack([cosg(mq + dfq), cosg(mq), cosg(dfq)], 1)                 # [5, 3] (판정 mem, m만, Δf만)
    return dict(wsum=wsum, bproj=bproj, wsumO=wsumO, bprojO=bprojO, recon=recon, sc=sc,
                nm=jnp.linalg.norm(mq, axis=-1), ndf=jnp.linalg.norm(dfq, axis=-1), mB=mq @ gBn, dfB=dfq @ gBn,
                cos_m_df=(jnp.sum(mq * dfq, -1) / (jnp.linalg.norm(mq, axis=-1) * jnp.linalg.norm(dfq, axis=-1) + 1e-12)),
                n_cat=cat.sum(1))


one_j = jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0)))
Wd = np.arange(256)


def run(th):
    rs = [jax.tree_util.tree_map(np.array, one_j(th, jax.tree_util.tree_map(lambda x: x[i:i + JCH], stack), plans[i:i + JCH], arrive[i:i + JCH]))
          for i in range(0, 256, JCH)]
    F = {k: np.concatenate([r[k] for r in rs]) for k in rs[0]}
    rnd = 1e-9 * np.random.default_rng(0).random((256, 5))
    out = dict(max_abs_recon_err=float(F['recon'].max()), events_per_world=dict(zip(CATS, F['n_cat'].mean(0).round(2).tolist())))
    best_all = {}
    for j, name in enumerate(("judge_mem(m+df)", "m_only", "df_only")):
        best = (F['sc'][:, :, j] + rnd).argmax(1); best_all[name] = best
        out[name] = dict(b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)), other=float(np.mean((best != b1) & (best != b0))))
    best = best_all["judge_mem(m+df)"]
    oth = (np.arange(5)[None] != b1[:, None]) & (np.arange(5)[None] != b0[:, None])
    def avg_oth(x):                                                            # 다른 곳 후보 평균 [256, ...]
        return (x * oth.reshape(oth.shape + (1,) * (x.ndim - 2))).sum(1) / oth.sum(1).reshape((-1,) + (1,) * (x.ndim - 2))
    groups = (("all", np.ones(256, bool)), ("judged_b1", best == b1), ("judged_b0", best == b0), ("judged_other", (best != b1) & (best != b0)))
    for grp, msk in groups:
        if msk.sum() == 0:
            continue
        g = dict(n=int(msk.sum()))
        for site, idx in (("probe_b1", b1), ("probe_b0", b0)):
            g[f"{site}/weight"] = dict(zip(CATS, F['wsum'][Wd, idx][msk].mean(0).round(4).tolist()))
            g[f"{site}/Bproj"] = dict(zip(CATS, F['bproj'][Wd, idx][msk].mean(0).round(5).tolist()))
        g["probe_other/weight"] = dict(zip(CATS_O, avg_oth(F['wsumO'])[msk].mean(0).round(4).tolist()))
        g["probe_other/Bproj"] = dict(zip(CATS_O, avg_oth(F['bprojO'])[msk].mean(0).round(5).tolist()))
        for nm_, key in (("|m|", 'nm'), ("|df|", 'ndf'), ("m_B", 'mB'), ("df_B", 'dfB'), ("cos(m,df)", 'cos_m_df')):
            g[f"score_parts/{nm_}"] = dict(b1=float(F[key][Wd, b1][msk].mean()), b0=float(F[key][Wd, b0][msk].mean()), other=float(avg_oth(F[key][:, :, None])[msk].mean()))
        out[grp] = g
    return out


rep = {}
for ck in os.environ['CKPTS'].split(','):
    if ck == 'start':
        th = th0
    else:
        z = np.load(ck); th = {a: {b: jnp.array(z[f"{a}/{b}"]) for b in th0[a]} for a in th0}
    rep[ck] = run(th)
    print(f"== {ck} ({time.time()-t0:.0f}s)\n" + json.dumps(rep[ck], ensure_ascii=False), flush=True)
    json.dump(rep, open(OUT, 'w'), ensure_ascii=False, indent=1)
