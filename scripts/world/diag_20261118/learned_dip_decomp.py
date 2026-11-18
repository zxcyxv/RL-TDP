# 키 학습의 하락·회복 분해 (2026-11-18). 대상: q8_learned 재현 (synapse_cellkey.py, WM_LOAD=wm_q8, KF_INIT=q8_base_frozen_1k)의 판정 시점 체크포인트.
#   기록 (q8_learned_5k): b1 70.7 → 54.7 (500) → 53.9 (1000) → 57.8 (2000) → 71.5 (3000) → 74.6 (5000), 오답은 거의 모두 b0.
# 체크포인트마다:
#   judge  : 판정 (재현 확인; synapse_cellkey.judge)
#   read   : 도착 칸 읽기를 과거 쓰기별 가중치로 정확 분해 (synapse_cellkey.decomp_one). b1·b0 후보 읽기의 사건별 가중치와 B 방향 기여
#   addr   : 판정 때 후보 도착 주소 z = σ[k; e]와 교사 때 b1·b0 도착 주소의 코사인, 전체 / 지금 키 부분 / 흔적 부분
#   nll    : 도착 칸 (SLOT) 순차 NLL을 사건별로 (교사 세션 전체 쓰기, 판정 세계)
#   kf     : 도착 칸 σ², R, q, 반감기, 행동별 키 크기 exp(ℓ)
# 환경변수: CKPTS (쉼표, 'start' 허용), OUT, WM_LOAD, KF_INIT (재현 실행과 같게)
import sys, os, json, time
import numpy as np

ROOT = '/workspace/RL-TDP'
SRC = f'{ROOT}/scripts/world/diag_20261117/synapse_cellkey.py'
OUT = os.environ['OUT']
t0 = time.time()
src = open(SRC).read()
cut = src.index("t0 = time.time()\nv0 = float(val_nll(th))")
ns = {'__name__': 'cellkey_defs'}
exec(compile(src[:cut], SRC, 'exec'), ns)
jax, jnp = ns['jax'], ns['jnp']
ER, EL, EP, RU = ns['ER'], ns['EL'], ns['EP'], ns['RU']
th0, stack, plans, arrive, b1, b0 = ns['th'], ns['stack'], ns['plans'], ns['arrive'], ns['b1'], ns['b0']
feat, wm, step_e, gB, R0 = ns['feat'], ns['wm'], ns['step_e'], ns['gB'], ns['R0']
KD, DZ, NPOLE, SLOT, H = ns['KD'], ns['DZ'], ns['NPOLE'], EP.SLOT, EP.H
JCH = 32
print(f"defs loaded ({time.time()-t0:.0f}s)", flush=True)
CATS = ["b1_B", "b0_B", "b0_C", "other_site", "non_arrival"]


def slot_scan(th, sess, plans_w, ai_all):
    kp, kf = th["kp"], th["kf"]
    R, Q, sg = jnp.exp(kf["lr"][SLOT]), jnp.exp(kf["lq"][SLOT]), jnp.exp(0.5 * kf["ls"][SLOT])
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
    cat = jnp.stack([arrived & (cur_site == sess["b1"]), arrived & (cur_site == sess["b0"]) & (item == 1),
                     arrived & (cur_site == sess["b0"]) & (item == 2), arrived & (cur_site != sess["b0"]) & (cur_site != sess["b1"]), ~arrived])

    def fw(c, x):
        Mp, Pp, e = c
        o, a, o2, t = x
        e, ell = step_e(kp, e, o, a)
        mu, _ = wm(o, a)
        r = (feat(o2) - mu)[SLOT]
        Pp = Pp + Q * jnp.eye(DZ)
        z = sg * e[SLOT].reshape(-1)
        err = r - Mp @ z; Pz = Pp @ z; Sv = R + R0 + z @ Pz
        nll = 0.5 * (err @ err / Sv + H * jnp.log(Sv))
        keep = t > t_c
        Mn = jnp.where(keep, Mp, Mp + jnp.outer(err, Pz / Sv)); Pn = jnp.where(keep, Pp, Pp - jnp.outer(Pz, Pz) / Sv)
        return (Mn, Pn, e), (z, jnp.where(keep, 0.0, nll), jnp.exp(ell[SLOT]), err @ err, Sv)
    (_, _, e_end), (Z, NLL, KS, E2, SV) = jax.lax.scan(fw, (jnp.zeros((H, DZ)), jnp.eye(DZ), jnp.zeros((EP.NP, NPOLE, KD))),
                                                       (O[:-1], sess["acts"], O[1:], jnp.arange(T)))

    def probe_z(plan, ai):
        def f(c, a):
            st, o, e = c
            e, _ = step_e(kp, e, o, a)
            st2, o2, _, _ = ER.env_step(st, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (st2, o2, e), sg * e[SLOT].reshape(-1)
        _, Zs = jax.lax.scan(f, (s_end, O[-1], e_end), plan)
        return Zs[ai]
    Zq = jax.vmap(probe_z)(plans_w, ai_all)                                   # [5, DZ]
    # 교사 때 b1 도착 주소, b0 도착 주소 (첫 B, 마지막 = C)
    idx = jnp.arange(T)
    t_b1 = jnp.min(jnp.where(cat[0], idx, T)); t_b0B = jnp.min(jnp.where(cat[1], idx, T)); t_b0C = jnp.max(jnp.where(cat[2], idx, -1))
    refs = jnp.stack([Z[t_b1], Z[t_b0B], Z[t_b0C]])                          # [3, DZ]

    def cos(a, b):
        return (a @ b.T) / (jnp.linalg.norm(a, axis=-1)[:, None] * jnp.linalg.norm(b, axis=-1)[None] + 1e-12)
    c_all = cos(Zq, refs); c_k = cos(Zq[:, :KD], refs[:, :KD]); c_e = cos(Zq[:, KD:], refs[:, KD:])   # [5, 3]
    nrm_k = jnp.linalg.norm(Zq[:, :KD], axis=-1); nrm_e = jnp.linalg.norm(Zq[:, KD:], axis=-1)
    nll_cat = cat.astype(jnp.float32) @ NLL                                   # [5] 사건별 NLL 합
    a_ = sess["acts"]; live = idx <= t_c
    ks = jnp.stack([jnp.sum(jnp.where(live & m, KS, 0)) / jnp.maximum(jnp.sum(live & m), 1)
                    for m in (a_ == ER.INTERACT, a_ < 4, a_ == ER.WAIT)])
    return c_all, c_k, c_e, nrm_k, nrm_e, nll_cat, cat.sum(1), ks


scan_j = jax.jit(jax.vmap(slot_scan, in_axes=(None, 0, 0, 0)))
decomp_j = jax.jit(jax.vmap(lambda s, a, b, t_: ns['decomp_one'](t_, s, a, b), in_axes=(0, 0, 0, None)))
W = np.arange(256)


def run(th):
    # 판정
    jj = ns['judge'](th)
    ok = np.array(jj['per_world_b1']).astype(bool)
    # 읽기 분해
    res_ = [decomp_j(jax.tree_util.tree_map(lambda x: x[i:i + JCH], stack), plans[i:i + JCH], arrive[i:i + JCH], th) for i in range(0, 256, JCH)]
    wsum, bproj, ncnt, err, yh = [np.concatenate([np.array(r[j]) for r in res_]) for j in range(5)]
    # 주소·NLL
    res2 = [scan_j(th, jax.tree_util.tree_map(lambda x: x[i:i + JCH], stack), plans[i:i + JCH], arrive[i:i + JCH]) for i in range(0, 256, JCH)]
    c_all, c_k, c_e, nk, ne, nllc, cnt, ks = [np.concatenate([np.array(r[j]) for r in res2]) for j in range(8)]
    out = dict(judge={k: jj[k] for k in ('b1', 'b0', 'other')}, max_abs_recon_err=float(err.max()))
    for grp, m in (("all", np.ones(256, bool)), ("correct", ok), ("wrong", ~ok)):
        g = dict(n=int(m.sum()))
        for site, sidx in (("probe_b1", b1), ("probe_b0", b0)):
            g[f"{site}/weight"] = dict(zip(CATS, wsum[W, sidx][m].mean(0).round(4).tolist()))
            g[f"{site}/Bproj"] = dict(zip(CATS, bproj[W, sidx][m].mean(0).round(5).tolist()))
        out[f"read/{grp}"] = g
    # 주소: 행 = 판정 후보 (b1, b0, 다른 곳 평균), 열 = 교사 도착 (b1 B, b0 B, b0 C)
    oth = (np.arange(5)[None] != b1[:, None]) & (np.arange(5)[None] != b0[:, None])     # [256, 5]
    def rows(C):
        return dict(probe_b1=C[W, b1].mean(0).round(4).tolist(), probe_b0=C[W, b0].mean(0).round(4).tolist(),
                    probe_other=(C * oth[:, :, None]).sum(1).__truediv__(oth.sum(1)[:, None]).mean(0).round(4).tolist())
    out['addr'] = dict(columns=["teacher_b1_B", "teacher_b0_B", "teacher_b0_C"], full=rows(c_all), key_part=rows(c_k), trace_part=rows(c_e),
                       probe_b1_closest_to_teacher_b1=float(np.mean(c_all[:, :, 0].argmax(1) == b1)),
                       norm_key_part=float(nk.mean()), norm_trace_part=float(ne.mean()))
    out['nll_slot_mean_per_event'] = dict(zip(CATS, (nllc.sum(0) / cnt.sum(0)).round(3).tolist()))
    out['nll_slot_total_by_cat'] = dict(zip(CATS, (nllc.mean(0)).round(2).tolist()))
    ps = ns['params_summary'](th)
    out['kf_slot'] = dict(sigma2=ps['sigma2'][SLOT], R=ps['R'][SLOT], q=ps['q'][SLOT], half_life=ps['half_life'],
                          window_steps=1.0 / max(ps['q'][SLOT], 1e-30))
    out['key_size_slot'] = dict(zip(("interact", "move", "wait"), ks.mean(0).round(4).tolist()))
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
