"""지연 채집 과제: 학습된 월드 모델 + 학습된 키 위의 사실 시냅스 (docs/world/README.md 12.1절).

느린 학습 (다른 세계의 교사 세션, 보상 쓰지 않음):
  관측 특징 y = 12패치 × 고정 임베딩 8 (평균을 뺌, 96차원).
  월드 모델 (장면, 행동) → 다음 y의 평균 μ⁰ [96]와 로그 분산 (스칼라). 가우시안 음의 로그우도.
  키 k(o, a) = unit(SwiGLU([flatten(emb(o)), onehot(a)])) [128]. 시냅스가 잔차 r = y − μ⁰를 예측한 오차로 학습 (64스텝 조각, 조각 사이 시냅스는 상수).
빠른 시냅스 (세션 안, 경사 없음): 흔적 e ← α e + k (α = 2^(−1/8)), z = s e (s = √(v / v̄)), 정밀도 가중 쓰기 (RLS, R = 0.01, P₀ = I).
기억 내용 판정: 교사 구간 끝의 기억을 복제하고, 채집처마다 그곳에 가서 상호작용하고 돌아오는 경로를 시뮬레이션 (쓰지 않음).
  산출물이 도착하는 관측을 예측하는 순간의 읽기 ŷ = s M e에서 HUD 도착 칸 성분을 꺼내 cos(ŷ_칸, φ_B − φ̄)를 점수로. 최대인 채집처가 b1이면 정답.
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax

from . import errand as ER

H, DK, SWI = 8, 128, 128
HY = 12 * H
ALPHA = 2.0 ** (-1 / 8)
SLOT = 10                                         # HUD 도착 칸 = 패치 10 (9 세계 + 카드 9, 도착 10, 도장 11)


def emb_fixed(seed=0):
    We = jax.random.normal(jax.random.PRNGKey(900 + seed), (147, H)) / jnp.sqrt(147.0)
    ybar = (ER.TEX.reshape(ER.TEX.shape[0], -1) @ We).mean(0)
    return We, ybar


def init(key):
    ks = jax.random.split(key, 4)
    d = lambda k, i, o, s=1.0: jax.random.normal(k, (i, o)) * s / jnp.sqrt(i)
    return dict(wm1=d(ks[0], HY + ER.NA, 256), wb1=jnp.zeros(256), wm2=d(ks[1], 256, HY + 1, 0.1), wb2=jnp.zeros(HY + 1),
                Ku=d(ks[2], HY + ER.NA, 2 * SWI), Kd=d(ks[3], SWI, DK))


def parts(p, We, ybar, kind):
    def feat(o):
        return ((o @ We) - ybar).reshape(-1)

    def wm(o, a):
        x = jnp.concatenate([feat(o), jax.nn.one_hot(a, ER.NA)])
        out = jax.nn.relu(x @ p["wm1"] + p["wb1"]) @ p["wm2"] + p["wb2"]
        return out[:HY], out[HY]

    def key(o, a):
        if kind == "learned":
            x = jnp.concatenate([feat(o), jax.nn.one_hot(a, ER.NA)])
            g, u = jnp.split(x @ p["Ku"], 2, -1)
            k = (jax.nn.silu(g) * u) @ p["Kd"]
        else:
            k = ER.fixed(0)[0](o)[a]
        return k / (jnp.linalg.norm(k) + 1e-8)
    return feat, wm, key


def stack_sessions(seeds, D):
    sessions = [ER.make_session(s, D) for s in seeds]
    L = max(len(s["acts"]) for s in sessions)
    for s in sessions:
        pad = L - len(s["acts"])
        s["acts"] = np.concatenate([s["acts"], np.full(pad, ER.WAIT)])
        s["req"] = np.concatenate([s["req"], np.full(pad, s["req"][-1])])
        s["rid"] = np.concatenate([s["rid"], np.full(pad, s["rid"][-1])])
    return sessions, {k: jnp.array(np.stack([s[k] for s in sessions])) for k in ("tiles", "depot", "sites", "prod", "b0", "b1", "acts", "req", "rid", "change_t")}


def env0(sess):
    s = dict(tiles=sess["tiles"], pos=sess["depot"], sites=sess["sites"], prod=sess["prod"], pending=jnp.array(-1), due=jnp.array(-1),
             t=jnp.array(0), stamp=jnp.array(0), req=jnp.array(-1), rid=jnp.array(0))
    return s, ER.obs(s["tiles"], s["pos"], ER.BLANK, ER.BLANK, ER.BLANK)


def observe_stream(sess):
    """교사 세션의 관측 [T+1, 12, 147]과 행동 [T]."""
    s, o0 = env0(sess)
    change = jnp.arange(sess["acts"].shape[0]) == sess["change_t"]

    def f(c, x):
        s, o = c
        a, rq, ri, ch = x
        s2, o2, r, _ = ER.env_step(s, a, rq, ri, ch, sess["b0"])
        return (s2, o2), o2
    (s, _), O = jax.lax.scan(f, (s, o0), (sess["acts"], sess["req"], sess["rid"], change))
    return jnp.concatenate([o0[None], O]), s


def gauss_logpdf(e, var):
    return -0.5 * (jnp.sum(e * e) / var + e.shape[0] * jnp.log(2 * jnp.pi * var))


def write_scan(p, We, ybar, kind, use_wm, vmax, O, acts, c0, evidence=False, q0=1.0):
    """evidence=True: 7절의 증거 망각. 관측마다 '그대로' N(Mz, R + zᵀPz) 대 '바뀜' N(0, R + q₀|z|²)의 우도비로 변화 확률 χ,
    지금 주소 방향만 P += χ q₀ ẑẑᵀ. 사건 위험률 h ~ Beta(a, b) 온라인 (사전 Beta(1, 1)). c = (M, P, e, ha, hb)."""
    feat, wm, key = parts(p, We, ybar, kind)

    def f(c, x):
        M, P, e, ha, hb = c
        o, a, o2 = x
        k = key(o, a)
        e = ALPHA * e + k
        if use_wm:
            mu, lv = wm(o, a)
            s = jnp.sqrt(jnp.exp(lv) / vmax)
        else:
            mu, s = jnp.zeros(HY), 1.0
        z = s * e
        r = feat(o2) - mu
        err = jnp.sum((r - M @ z) ** 2)
        if evidence:
            h = ha / (ha + hb)
            L0 = gauss_logpdf(r - M @ z, 0.01 + z @ P @ z)
            L1 = gauss_logpdf(r, 0.01 + q0 * (z @ z))
            chi = jax.nn.sigmoid(jnp.log(h) + L1 - jnp.log1p(-h) - L0)
            zh = z / (jnp.linalg.norm(z) + 1e-12)
            P = P + chi * q0 * jnp.outer(zh, zh)
            ha, hb = ha + chi, hb + 1 - chi
        Pz = P @ z
        S = 0.01 + z @ Pz
        M = M + jnp.outer(r - M @ z, Pz) / S
        P = P - jnp.outer(Pz, Pz) / S
        return (M, P, e, ha, hb), err
    return jax.lax.scan(f, c0, (O[:-1], acts, O[1:]))


def c_init():
    return (jnp.zeros((HY, DK)), jnp.eye(DK), jnp.zeros(DK), jnp.array(1.0), jnp.array(1.0))


def train(kind, seed=0, D=256, n_train=512, iters_wm=400, iters_k=600, B=32, L=64, log=print):
    We, ybar = emb_fixed(seed)
    p = init(jax.random.PRNGKey(seed))
    _, tr = stack_sessions(range(50_000, 50_000 + n_train), D)
    O_all, _ = jax.jit(jax.vmap(observe_stream))(tr)
    A_all = tr["acts"]
    feat, _, _ = parts(p, We, ybar, kind)
    Y_all = jax.vmap(jax.vmap(feat))(O_all[:, 1:])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))

    wmkeys = ("wm1", "wb1", "wm2", "wb2")
    q = {k: p[k] for k in wmkeys}
    st = opt.init(q)

    @jax.jit
    def upd_wm(q, st, idx):
        def nll(q):
            pp = dict(p, **q)
            _, wm, _ = parts(pp, We, ybar, kind)
            mu, lv = jax.vmap(jax.vmap(wm))(O_all[idx, :-1], A_all[idx])
            return jnp.mean(0.5 * (((Y_all[idx] - mu) ** 2).sum(-1) * jnp.exp(-lv) + HY * lv))
        l, g = jax.value_and_grad(nll)(q)
        g, st = opt.update(g, st, q)
        return optax.apply_updates(q, g), st, l
    key = jax.random.PRNGKey(7)
    for it in range(iters_wm):
        key, k = jax.random.split(key)
        q, st, l = upd_wm(q, st, jax.random.randint(k, (B,), 0, n_train))
        if it % 200 == 0 or it == iters_wm - 1:
            log(f"  [world model] it {it} nll {float(l):.2f}")
    p.update(q)
    _, wm, _ = parts(p, We, ybar, kind)
    _, lv = jax.vmap(jax.vmap(wm))(O_all[:64, :-1], A_all[:64])
    vmax = float(jnp.exp(lv).max())
    if kind != "learned":
        return p, vmax

    kkeys = ("Ku", "Kd")
    q = {k: p[k] for k in kkeys}
    st = opt.init(q)
    T = A_all.shape[1]

    @jax.jit
    def upd_k(q, st, idx, t0, carry):
        def lf(q):
            pp = dict(p, **q)
            O = jax.lax.dynamic_slice_in_dim(O_all[idx], t0, L + 1, axis=1)
            Ac = jax.lax.dynamic_slice_in_dim(A_all[idx], t0, L, axis=1)
            c2, err = jax.vmap(lambda o, a, c: write_scan(pp, We, ybar, kind, True, vmax, o, a, c))(O, Ac, carry)
            return err.mean(), c2
        (l, c2), g = jax.value_and_grad(lf, has_aux=True)(q)
        g, st = opt.update(g, st, q)
        return optax.apply_updates(q, g), st, jax.lax.stop_gradient(c2), l
    for it in range(iters_k):
        key, k = jax.random.split(key)
        idx = jax.random.randint(k, (B,), 0, n_train)
        carry = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), c_init())
        for t0 in range(0, T - L + 1, L):
            q, st, carry, l = upd_k(q, st, idx, t0, carry)
        if it % 200 == 0 or it == iters_k - 1:
            log(f"  [keys] it {it} memory prediction error {float(l):.3f}")
    p.update(q)
    return p, vmax


def probe(p, vmax, kind, use_wm, stack, seed=0, evidence=False):
    """교사 구간 끝의 기억으로 채집처마다 '가서 하고 돌아오기'를 시뮬레이션, 도착 예측의 B 점수. 반환: 점수 [NW, NSITE]."""
    We, ybar = emb_fixed(seed)
    feat, wm, key = parts(p, We, ybar, kind)
    phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
    gB = phi[1] - phi.mean(0)

    def one(sess, plans, arrive_idx):
        O, s_end = observe_stream(sess)
        (M, P, e, _, _), _ = write_scan(p, We, ybar, kind, use_wm, vmax, O, sess["acts"], c_init(), evidence=evidence)

        def sim(plan, ai):
            def f(c, a):
                s, o, e = c
                k = key(o, a)
                e = ALPHA * e + k
                if use_wm:
                    _, lv = wm(o, a)
                    sv = jnp.sqrt(jnp.exp(lv) / vmax)
                else:
                    sv = 1.0
                yh = (M @ (sv * e)).reshape(12, H)[SLOT]
                s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
                return (s2, o2, e), yh
            _, YH = jax.lax.scan(f, (s_end, O[-1], e), plan)
            yh = YH[ai]
            return (yh @ gB) / (jnp.linalg.norm(yh) * jnp.linalg.norm(gB) + 1e-8)
        return jax.vmap(sim)(plans, arrive_idx)
    return one


def plans_for(sessions):
    """채집처마다: 보급소 → 채집처, 상호작용, 보급소로 돌아오며 대기 (도착까지). 반환: 행동 [NW, NSITE, Lp], 도착 예측 스텝 [NW, NSITE]."""
    allp, alla = [], []
    for s in sessions:
        ps, ai = [], []
        for i in range(ER.NSITE):
            go = ER.path(s["depot"], s["sites"][i])
            back = ER.path(s["sites"][i], s["depot"])
            plan = go + [ER.INTERACT] + back
            j = len(go)                                          # 상호작용 스텝
            while len(plan) < j + ER.D_DELAY + 1:
                plan.append(ER.WAIT)
            ps.append(plan); ai.append(j + ER.D_DELAY - 1)       # 이 스텝의 예측이 도착 관측
        allp.append(ps); alla.append(ai)
    Lp = max(len(x) for ps in allp for x in ps)
    arr = np.full((len(sessions), ER.NSITE, Lp), ER.WAIT)
    for i, ps in enumerate(allp):
        for j, x in enumerate(ps):
            arr[i, j, :len(x)] = x
    return jnp.array(arr), jnp.array(alla)
