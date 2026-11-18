# f↔M 분담 가설 확인 (2026-11-18): f를 함께 학습하면 f가 HUD 칸 예측을 기억에 넘겨 잔차가 시간적으로 지속되고, 그래서 "직전 값 복사"가 NLL 최적이 되는가.
# 순전파 값만 (교사 세션 전체 쓰기, 판정 세계 256개):
#   r_t = y_t − f(x_t, a_t, m_t)  (기억의 쓰기 목표 = f가 남긴 몫),  v_t = r_t − m_t (최종 오차),  S_t (예측 분산)
#   지속성 = E‖r_t − r_{t−1}‖² / E‖r_t‖²  (< 1이면 직전 값 복사가 0 예측보다 나음)
#   칸: 카드 9, 도착 10, 도장 11, 시야 0–8 평균. 도착 칸은 도착 스텝 / 도착 없는 스텝으로 나눔.
# 환경변수: CKPTS (이름=경로, 쉼표; 'start'), OUT
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
ER, EL, RU = ns['ER'], ns['EL'], ns['RU']
th0, stack = ns['th'], ns['stack']
feat, f_net, step_e, R0 = ns['feat'], ns['f_net'], ns['step_e'], ns['R0']
NP, H, DZ, KD, NPOLE, SLOT = ns['NP'], ns['H'], ns['DZ'], ns['KD'], ns['NPOLE'], ns['SLOT']
JCH = 32


def one(th, sess):
    fp, kp, kf = th["fp"], th["kp"], th["kf"]
    R, Q, sg = jnp.exp(kf["lr"]), jnp.exp(kf["lq"]), jnp.exp(0.5 * kf["ls"])
    O, _, _ = EL.observe_stream(sess)
    t_c = RU.last_arrival(O)
    T = sess["acts"].shape[0]
    arrived = ~jnp.all(jnp.isclose(O[1:, SLOT], ER.TEX[ER.BLANK].reshape(-1)), -1)

    def fw(c, x):
        M, P, e = c
        o, a, o2, t = x
        e = step_e(kp, e, o, a)
        Z = sg[:, None] * e.reshape(NP, -1)
        m = jnp.einsum('phd,pd->ph', M, Z)
        r = feat(o2) - f_net(fp, o, a, m)
        Pq = P + Q[:, None, None] * jnp.eye(DZ)
        PZ = jnp.einsum('pij,pj->pi', Pq, Z, precision='highest')
        S = R + R0 + jnp.sum(Z * PZ, -1)
        v = r - m
        on = t <= t_c
        M2 = jnp.where(on, M + jnp.einsum('ph,pd->phd', v, PZ / S[:, None]), M)
        P2 = jnp.where(on, Pq - jnp.einsum('pi,pj->pij', PZ, PZ) / S[:, None, None], P)
        nll = 0.5 * (jnp.sum(v ** 2, -1) / S + H * jnp.log(S))
        return (M2, P2, e), (r, v, S, nll)
    c0 = (jnp.zeros((NP, H, DZ)), jnp.tile(jnp.eye(DZ), (NP, 1, 1)), jnp.zeros((NP, NPOLE, KD)))
    _, (r, v, S, nll) = jax.lax.scan(fw, c0, (O[:-1], sess["acts"], O[1:], jnp.arange(T)))
    live = (jnp.arange(T) <= t_c)
    r2 = jnp.sum(r ** 2, -1); v2 = jnp.sum(v ** 2, -1)                       # [T, 12]
    d2 = jnp.sum((r[1:] - r[:-1]) ** 2, -1)                                    # [T-1, 12]
    live2 = live[1:] & live[:-1]
    masks = dict(all=live, arrival=live & arrived, non_arrival=live & ~arrived)
    masks2 = dict(all=live2, arrival=live2 & arrived[1:], non_arrival=live2 & ~arrived[1:])
    out = {}
    for k in masks:
        mk = masks[k].astype(jnp.float32); mk2 = masks2[k].astype(jnp.float32)
        out[k] = dict(r2=mk @ r2, v2=mk @ v2, S=mk @ S, nll=mk @ nll, n=mk.sum(), d2=mk2 @ d2, r2p=mk2 @ r2[1:], n2=mk2.sum())
    return out


one_j = jax.jit(jax.vmap(one, in_axes=(None, 0)))
CELLS = dict(card=[9], arrival_slot=[10], stamp=[11], view=list(range(9)))


def run(th):
    rs = [jax.tree_util.tree_map(np.array, one_j(th, jax.tree_util.tree_map(lambda x: x[i:i + JCH], stack))) for i in range(0, 256, JCH)]
    F = jax.tree_util.tree_map(lambda *x: np.concatenate(x), *rs)
    rep = {}
    for cname, cl in CELLS.items():
        for k in (('all', 'arrival', 'non_arrival') if cname == 'arrival_slot' else ('all',)):
            g = F[k]
            n = g['n'].sum(); n2 = g['n2'].sum()
            tot = lambda a: a[:, cl].sum() / len(cl)
            rep[f"{cname}/{k}"] = dict(
                f_leftover_r2=float(tot(g['r2']) / n), final_err_v2=float(tot(g['v2']) / n),
                persistence=float(tot(g['d2']) / max(tot(g['r2p']), 1e-30)),
                mean_S=float(tot(g['S']) / n), nll_per_step=float(tot(g['nll']) / n), steps_per_world=float(n / 256))
    return rep


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
    rep[name] = run(th)
    print(f"== {name} ({time.time()-t0:.0f}s)\n" + json.dumps(rep[name]), flush=True)
    json.dump(rep, open(OUT, 'w'), indent=1)
