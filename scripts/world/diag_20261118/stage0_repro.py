# 단계 0 · 기존 코드 재현 (명세 docs/world/spec_memory_wm_agent.md 3절), 2026-11-18
#   (a) 기존 synapse_wm_joint.py의 함수를 그대로 불러, 기록된 판정 (시작점, 900반복 best 체크포인트)과 세계별로 일치하는지
#   (b) 쓰기·다섯 후보 읽기·g_B 코사인을 독립 재구현해 기존 judge_chunk 점수와 대조 (작은 묶음)
#   (c) 누출 점검: 쓰기 없음 (M = 0) → syn 다섯 후보 동점 → 동점 무작위로 b1 약 20%
#   (d) b1 증거 기록: b1 단서가 보인 칸·시점, 직접 관측인지, 주소 일치, 읽기 전후 순서
# 환경변수: OUT (json), NSMALL (b의 세계 수, 기본 16)
import sys, os, json, time
import numpy as np

ROOT = '/workspace/RL-TDP'
SRC = f'{ROOT}/scripts/world/diag_20261117/synapse_wm_joint.py'
RES = f'{ROOT}/docs/world/results/synapse_layer_20261117'
OUT = os.environ.get('OUT', f'{ROOT}/docs/world/results/stage0_20261118/stage0.json')
NSMALL = int(os.environ.get('NSMALL', 16))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
t0 = time.time()

# ---- 기존 스크립트의 정의부만 실행 (학습 루프 직전까지) ----
src = open(SRC).read()
cut = src.index("OUT = os.environ.get('OUT'")
old = {'__name__': 'old_wm_joint'}
exec(compile(src[:cut], SRC, 'exec'), old)
jax, jnp = old['jax'], old['jnp']
ER, EL, EP, RU = old['ER'], old['EL'], old['EP'], old['RU']
print(f"old definitions loaded ({time.time()-t0:.0f}s)", flush=True)


def load(path, like):
    z = np.load(path)
    return {a: {b: jnp.array(z[f"{a}/{b}"]) for b in like[a]} for a in like}


th_init = old['th']
th900 = load(f'{RES}/wm_joint_10k_best_params.npz', th_init)
logged = json.load(open(f'{RES}/wm_joint_10k.json'))
rep = dict(spec='stage0', src=SRC, ckpt=f'{RES}/wm_joint_10k_best_params.npz', logged_best_it=logged['after']['best_it'])

# ---- (a) 재현 ----
rep['a'] = {}
for tag, th, lg in [('init', th_init, logged['before']['judge']), ('it900', th900, logged['after']['judge'])]:
    j = old['judge'](th)
    row = {}
    for k in ('syn', 'wm', 'wm_memoff'):
        same = j[k]['per_world_b1'] == lg[k]['per_world_b1']
        row[k] = dict(b1=j[k]['b1'], b0=j[k]['b0'], other=j[k]['other'], logged_b1=lg[k]['b1'], logged_b0=lg[k]['b0'],
                      per_world_identical=bool(same), n_world_diff=int(np.sum(np.array(j[k]['per_world_b1']) != np.array(lg[k]['per_world_b1']))))
    rep['a'][tag] = row
    print(f"(a) {tag}: " + "  ".join(f"{k} b1 {v['b1']:.4f} (log {v['logged_b1']:.4f}) b0 {v['b0']:.4f} (log {v['logged_b0']:.4f}) same={v['per_world_identical']}"
                                    for k, v in row.items()), f"({time.time()-t0:.0f}s)", flush=True)

# ---- (b) 독립 재구현 ----
NP, H, DZ, SLOT, R0 = old['NP'], old['H'], old['DZ'], EP.SLOT, old['R0']
We, ybar, gB, TEX = old['We'], old['ybar'], old['gB'], ER.TEX


def my_feat(o):
    return ((o @ We) - ybar).reshape(NP, H)


def my_F0(wp, o, a):
    x = jnp.concatenate([my_feat(o).reshape(-1), jax.nn.one_hot(a, ER.NA)])
    return (jax.nn.relu(x @ wp["F1"] + wp["Fb1"]) @ wp["F2"] + wp["Fb2"]).reshape(NP, H)


def my_G(wp, o, a, m):
    x = jnp.concatenate([my_feat(o).reshape(-1), jax.nn.one_hot(a, ER.NA), m.reshape(-1)])
    return (jax.nn.silu(x @ wp["G1"] + wp["Gb1"]) @ wp["G2"] + wp["Gb2"]).reshape(NP, H)


def my_key(kp, o, a):
    w = my_feat(o)[:9].reshape(-1)
    pos = jnp.eye(NP)
    u = w + jax.nn.silu(jnp.concatenate([jnp.tile(w, (NP, 1)), pos], -1) @ kp["W1"] + kp["b1"]) @ kp["W2"] + kp["b2"]
    kh = u @ kp["Wk"][a].T
    ell = jnp.concatenate([jnp.tile(jnp.concatenate([w, jax.nn.one_hot(a, ER.NA)]), (NP, 1)), pos], -1) @ kp["lw"] + kp["lb"]
    return jnp.exp(ell)[:, None] * kh / (jnp.linalg.norm(kh, axis=-1, keepdims=True) + 1e-8)


def my_trace(kp, e, o, a):
    k = my_key(kp, o, a)
    return jnp.stack([k, jax.nn.sigmoid(kp["ta"]) * e[:, 1] + k], 1)


def teacher_stream(sess):
    """관측 [T+1], 스텝별 작업 시작 채집처 (−1 없음), 스텝별 도착 산출물 (−1 없음; o_{t+1}에 보임)."""
    s, o0 = EL.env0(sess)
    change = jnp.arange(sess["acts"].shape[0]) == sess["change_t"]

    def f(c, x):
        s, o = c
        a, rq, ri, ch = x
        s2, o2, r, start = ER.env_step(s, a, rq, ri, ch, sess["b0"])
        arr = jnp.where((s["pending"] >= 0) & (s2["pending"] < 0), s["pending"], -1)
        return (s2, o2), (o2, start, arr, r)
    (s, _), (O, start, arr, r) = jax.lax.scan(f, (s, o0), (sess["acts"], sess["req"], sess["rid"], change))
    return jnp.concatenate([o0[None], O]), s, start, arr, r


def my_write(th, O, acts, stop_t):
    kp, wp, kf = th["kp"], th["wp"], th["kf"]
    R, Q, sg = jnp.exp(kf["lr"]), jnp.exp(kf["lq"]), jnp.exp(0.5 * kf["ls"])

    def f(c, x):
        M, P, e = c
        o, a, o2, t = x
        e = my_trace(kp, e, o, a)
        Z = sg[:, None] * e.reshape(NP, -1)
        Pq = P + Q[:, None, None] * jnp.eye(DZ)
        m = jnp.einsum('phd,pd->ph', M, Z)
        PZ = jnp.einsum('pij,pj->pi', Pq, Z, precision='highest')
        S = R + R0 + jnp.sum(Z * PZ, -1)
        v = my_feat(o2) - my_F0(wp, o, a) - m
        dM = jnp.einsum('ph,pd->phd', v, PZ / S[:, None])
        on = t <= stop_t
        M2 = jnp.where(on, M + dM, M)
        P2 = jnp.where(on, Pq - jnp.einsum('pi,pj->pij', PZ, PZ) / S[:, None, None], P)
        rec = dict(z=Z[SLOT], m=m[SLOT], v=v[SLOT], dMn=jnp.where(on, jnp.linalg.norm(dM[SLOT]), 0.0), S=S[SLOT])
        return (M2, P2, e), rec
    c0 = (jnp.zeros((NP, H, DZ)), jnp.tile(jnp.eye(DZ), (NP, 1, 1)), jnp.zeros((NP, 2, old['KD'])))
    return jax.lax.scan(f, c0, (O[:-1], acts, O[1:], jnp.arange(acts.shape[0])))


def my_candidates(th, M, e, s_end, o_end, sess, plans, ai_all):
    """다섯 후보 경로 재생 (쓰기 없음). 도착 예측 스텝의 SLOT 읽기 셋과 주소."""
    kp, wp, kf = th["kp"], th["wp"], th["kf"]
    sg = jnp.exp(0.5 * kf["ls"])

    def sim(plan, ai):
        def f(c, a):
            s, o, e = c
            e = my_trace(kp, e, o, a)
            Z = sg[:, None] * e.reshape(NP, -1)
            m = jnp.einsum('phd,pd->ph', M, Z)
            reads = jnp.stack([m[SLOT], (m + my_G(wp, o, a, m))[SLOT], my_G(wp, o, a, jnp.zeros_like(m))[SLOT]])
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (s2, o2, e), (reads, Z[SLOT], o2[SLOT])
        _, (Y, Zs, Os) = jax.lax.scan(f, (s_end, o_end, e), plan)
        y = Y[ai]
        cos = (y @ gB) / (jnp.linalg.norm(y, axis=-1) * jnp.linalg.norm(gB) + 1e-8)
        return cos, Zs[ai], Os[ai]
    return jax.vmap(sim)(plans, ai_all)


def my_one(th, sess, plans, ai_all, nowrite):
    O, s_end, start, arr, rw = teacher_stream(sess)
    t_c = RU.last_arrival(O)
    (M, P, e), rec = my_write(th, O, sess["acts"], jnp.where(nowrite, -1, t_c))
    cos, Zj, Oj = my_candidates(th, M, e, s_end, O[-1], sess, plans, ai_all)
    return dict(cos=cos, Zj=Zj, Oj=Oj, rec=rec, start=start, arr=arr, rw=rw, t_c=t_c, Mslot=M[SLOT])


my_chunk = jax.jit(jax.vmap(my_one, in_axes=(None, 0, 0, 0, None)))
old_chunk = old['judge_chunk']
stack, plans, arrive = old['stack'], old['plans'], old['arrive']
b1, b0 = old['b1'], old['b0']
sl = lambda tree, i, n: jax.tree_util.tree_map(lambda x: x[i:i + n], tree)

rep['b'] = {}
for tag, th in [('init', th_init), ('it900', th900)]:
    ref = np.array(old_chunk(th, sl(stack, 0, NSMALL), plans[:NSMALL], arrive[:NSMALL]))      # [n, 5, 3]
    mine = np.array(my_chunk(th, sl(stack, 0, NSMALL), plans[:NSMALL], arrive[:NSMALL], False)['cos'])
    d = np.abs(ref - mine)
    rep['b'][tag] = dict(n_worlds=NSMALL, max_abs_diff=float(d.max()), argmax_agree=float(np.mean(ref.argmax(1) == mine.argmax(1))))
    print(f"(b) {tag}: max |old - mine| {d.max():.2e}, argmax agree {rep['b'][tag]['argmax_agree']:.3f}", flush=True)


def tally(sc):
    best = (sc + 1e-9 * np.random.default_rng(0).random(sc.shape)).argmax(1)
    return dict(b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)), other=float(np.mean((best != b1) & (best != b0))))


# ---- (c), (d): 판정 세계 256개 전체 ----
def full(th, nowrite):
    outs = [jax.tree_util.tree_map(np.array, my_chunk(th, sl(stack, i, 32), plans[i:i + 32], arrive[i:i + 32], nowrite)) for i in range(0, 256, 32)]
    return {k: (np.concatenate([o[k] for o in outs]) if not isinstance(outs[0][k], dict)
                else {kk: np.concatenate([o[k][kk] for o in outs]) for kk in outs[0][k]}) for k in outs[0]}


rep['c'] = {}
for tag, th in [('init', th_init), ('it900', th900)]:
    F = full(th, True)
    rep['c'][tag] = {n: tally(F['cos'][:, :, j]) for j, n in enumerate(('syn', 'wm', 'wm_memoff'))}
    rep['c'][tag]['syn_all_tied'] = float(np.mean(np.ptp(F['cos'][:, :, 0], 1) < 1e-6))
    print(f"(c) {tag} no-write: " + "  ".join(f"{n} b1 {v['b1']:.4f}" for n, v in rep['c'][tag].items() if isinstance(v, dict)),
          f"syn tied {rep['c'][tag]['syn_all_tied']:.3f}", flush=True)

itemB = np.array(TEX[ER.ITEM0 + 1]).reshape(-1)
rep['d'] = {}
per_world = {}
for tag, th in [('init', th_init), ('it900', th900)]:
    F = full(th, False)
    st, ar, rec = F['start'], F['arr'], F['rec']
    rows = []
    for w in range(256):
        sess = old['sessions'][w]
        t_int = np.where(st[w] == b1[w])[0]                        # b1에서 작업 시작한 스텝 (o_{t+1}부터 대기)
        # 도착 스텝: 작업 시작 뒤 첫 도착 (같은 시기에 작업은 하나뿐)
        t_arr = [int(np.where((ar[w] >= 0) & (np.arange(len(ar[w])) > t))[0][0]) for t in t_int]
        tb = t_arr[0]
        z_t = rec['z'][w, tb]                                        # 교사 쓰기 때 b1 도착 칸 주소
        Zj = F['Zj'][w]                                              # 판정 때 다섯 후보의 도착 예측 주소 [5, DZ]
        cz = (Zj @ z_t) / (np.linalg.norm(Zj, axis=1) * np.linalg.norm(z_t) + 1e-12)
        vB = rec['v'][w, tb]
        rows.append(dict(
            world=w, b1=int(b1[w]), b0=int(b0[w]),
            n_b1_visits=len(t_int), t_interact_b1=t_int.tolist(), t_arrive_b1=t_arr, arrived_item_at_b1=[int(ar[w][t]) for t in t_arr],
            b1_order_in_observation=int(np.searchsorted(np.sort(np.where(st[w] >= 0)[0][:5]), t_int[0])),
            stop_t=int(F['t_c'][w]), T=int(len(ar[w])),
            read_before_write_cos_gB=float(rec['m'][w, tb] @ np.array(gB) / (np.linalg.norm(rec['m'][w, tb]) * np.linalg.norm(gB) + 1e-12)),
            resid_cos_gB=float(vB @ np.array(gB) / (np.linalg.norm(vB) * np.linalg.norm(gB) + 1e-12)),
            dM_slot_at_b1=float(rec['dMn'][w, tb]), dM_slot_median=float(np.median(rec['dMn'][w][rec['dMn'][w] > 0])),
            addr_cos_judge_vs_teacher=cz.tolist(), addr_b1_is_best=bool(cz.argmax() == b1[w]),
            judge_slot_shows_B=[bool(np.allclose(F['Oj'][w, i], itemB)) for i in range(5)],
            syn_score=F['cos'][w, :, 0].tolist()))
    per_world[tag] = rows
    n_vis = np.array([r['n_b1_visits'] for r in rows])
    rep['d'][tag] = dict(
        b1_visits_hist={int(k): int(v) for k, v in zip(*np.unique(n_vis, return_counts=True))},
        b1_item_was_B=float(np.mean([r['arrived_item_at_b1'][0] == 1 for r in rows])),
        t_arrive_b1_median=float(np.median([r['t_arrive_b1'][0] for r in rows])), stop_t_median=float(np.median([r['stop_t'] for r in rows])),
        gap_stop_minus_b1_median=float(np.median([r['stop_t'] - r['t_arrive_b1'][0] for r in rows])),
        b1_written_before_stop=float(np.mean([r['t_arrive_b1'][0] <= r['stop_t'] for r in rows])),
        resid_cos_gB_median=float(np.median([r['resid_cos_gB'] for r in rows])),
        dM_ratio_b1_over_median=float(np.median([r['dM_slot_at_b1'] / r['dM_slot_median'] for r in rows])),
        addr_b1_best_frac=float(np.mean([r['addr_b1_is_best'] for r in rows])),
        addr_cos_b1_median=float(np.median([r['addr_cos_judge_vs_teacher'][r['b1']] for r in rows])),
        addr_cos_other_max_median=float(np.median([max(c for i, c in enumerate(r['addr_cos_judge_vs_teacher']) if i != r['b1']) for r in rows])),
        note='b1 단서 = 관찰 단계에서 b1 작업 뒤 D스텝에 HUD 도착 칸(SLOT=10)에 B 그림이 직접 보임 (출처 표시 없음). 채집처와의 연결은 흔적 주소로만 (간접). 쓰기는 그 스텝의 예측 읽기 뒤. 판정 읽기는 쓰기 종료 뒤, 정답 특징을 보지 않음.')
    print(f"(d) {tag}: " + json.dumps(rep['d'][tag]), flush=True)

json.dump(rep, open(OUT, 'w'), indent=1)
json.dump(per_world, open(OUT.replace('.json', '_per_world.json'), 'w'))
print(f"saved {OUT} ({time.time()-t0:.0f}s)", flush=True)
