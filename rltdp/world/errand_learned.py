"""지연 채집 과제: 학습된 월드 모델 + 학습된 키 위의 사실 시냅스 (docs/world/README.md 12.1절).

느린 학습 (다른 세계의 교사 세션):
  손실 = 머리별 척도 (DreamerV3식): 세계 시야 9칸 오차 평균 + HUD 3칸 오차 평균, 같은 무게. 시냅스는 관측만 쓴다.
  (보상 머리는 넣지 않는다: 의뢰가 비공개라 카드로는 무엇이 보상인지 알 수 없어 산출물 정체를 담게 하는 데 도움이 안 된다. reward_head는 판정에서 쓰지 않는다.)
  관측 특징 y = 12패치 × 고정 임베딩 8 (평균을 뺌, 96차원).
  월드 모델 (장면, 행동) → 다음 y의 평균 μ⁰ [96]와 로그 분산 (스칼라). 가우시안 음의 로그우도.
  키 k(o, a) = SwiGLU([flatten(emb(o)), onehot(a)]) [128], 크기 정규화 없음 (12.1절 첫 판은 정규화해서 인코더가 키 크기로 흔적을 조절할 수 없었다). 시냅스가 잔차 r = y − μ⁰를 예측한 오차로 학습 (64스텝 조각, 조각 사이 시냅스는 상수).
빠른 시냅스 (세션 안, 경사 없음): 흔적 e ← α e + k (α = 2^(−1/8)), z = s e (s = √(v / v̄)), 정밀도 가중 쓰기 (RLS, R = 0.01, P₀ = I).
기억 내용 판정: 교사 구간 끝의 기억을 복제하고, 채집처마다 그곳에 가서 상호작용하고 돌아오는 경로를 시뮬레이션 (쓰지 않음).
  산출물이 도착하는 관측을 예측하는 순간의 읽기 ŷ = s M e에서 HUD 도착 칸 성분을 꺼내 cos(ŷ_칸, φ_B − φ̄)를 점수로. 최대인 채집처가 b1이면 정답.
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax

from . import errand as ER

H, DK, SWI = int(__import__("os").environ.get("ERRAND_H", 8)), 128, 128          # ERRAND_H: 픽셀 임베딩 차원 (12.8)
HY = 12 * H
HO = HY                                          # 시냅스·월드 모델의 출력 = 관측 특징만 (보상은 느린 보상 머리가 맡는다)
CARD = 9
ALPHA = 2.0 ** (-1 / 8)
SLOT = 10                                         # HUD 도착 칸 = 패치 10 (9 세계 + 카드 9, 도착 10, 도장 11)


def emb_fixed(seed=0):
    We = jax.random.normal(jax.random.PRNGKey(900 + seed), (147, H)) / jnp.sqrt(147.0)
    ybar = (ER.TEX.reshape(ER.TEX.shape[0], -1) @ We).mean(0)
    return We, ybar


def init(key):
    ks = jax.random.split(key, 4)
    d = lambda k, i, o, s=1.0: jax.random.normal(k, (i, o)) * s / jnp.sqrt(i)
    k5, k6 = jax.random.split(ks[3])
    return dict(wm1=d(ks[0], HY + ER.NA, 256), wb1=jnp.zeros(256), wm2=d(ks[1], 256, HO + 1, 0.1), wb2=jnp.zeros(HO + 1),
                Ku=d(ks[2], HY + ER.NA, 2 * SWI), Kd=d(k5, SWI, DK),
                Rh1=d(k6, 2 * H, 32), rb1=jnp.zeros(32), Rh2=jnp.zeros((32, 1)), rb2=jnp.zeros(1))      # 느린 보상 머리


def parts(p, We, ybar, kind):
    def feat(o):
        return ((o @ We) - ybar).reshape(-1)

    def wm(o, a):
        x = jnp.concatenate([feat(o), jax.nn.one_hot(a, ER.NA)])
        out = jax.nn.relu(x @ p["wm1"] + p["wb1"]) @ p["wm2"] + p["wb2"]
        return out[:HO], out[HO]

    def key(o, a):
        if kind == "learned":                                   # 크기를 정규화하지 않는다: 무엇을 얼마나 흔적에 넣을지 인코더가 정한다 (8절과 같음)
            x = jnp.concatenate([feat(o), jax.nn.one_hot(a, ER.NA)])
            g, u = jnp.split(x @ p["Ku"], 2, -1)
            return (jax.nn.silu(g) * u) @ p["Kd"]
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
        return (s2, o2), (o2, r)
    (s, _), (O, Rw) = jax.lax.scan(f, (s, o0), (sess["acts"], sess["req"], sess["rid"], change))
    return jnp.concatenate([o0[None], O]), s, Rw


def gauss_logpdf(e, var):
    return -0.5 * (jnp.sum(e * e) / var + e.shape[0] * jnp.log(2 * jnp.pi * var))


def target(feat, o2, r):
    return feat(o2)


def reward_head(p, yslot, card):
    """느린 보상 머리: 시냅스가 예측한 도착 칸 [8]과 지금 의뢰 카드 [8] → 보상. 과제와 무관한 일반 규칙 (도착할 것과 의뢰가 맞으면 보상)."""
    x = jnp.concatenate([yslot, card])
    return (jax.nn.relu(x @ p["Rh1"] + p["rb1"]) @ p["Rh2"] + p["rb2"])[0]


def loss_terms(err_vec):
    """머리별 척도 (DreamerV3식): 세계 시야 9칸 오차 평균 + HUD 3칸 오차 평균, 같은 무게.
    (평균 없이 96성분을 더하면 손실의 대부분이 지도 = 새로 보이는 칸이라 키가 드문 산출물 도착을 희생했다: 12.1절)"""
    e = (err_vec ** 2).reshape(12, H)
    return e[:9].mean() + e[9:].mean()


def write_scan(p, We, ybar, kind, use_wm, vmax, O, acts, c0, evidence=False, vbar=None, Rw=None):
    """evidence=True: 7절의 증거 망각. 관측마다 '그대로' N(Mz, R + zᵀPz) 대 '바뀜'의 우도비로 변화 확률 χ,
    지금 주소 방향만 P를 키움. 사건 위험률 h ~ Beta(a, b) 온라인 (사전 Beta(1, 1)). c = (M, P, e, ha, hb).
    '바뀜' = 이 주소의 내용이 사전분포에서 새로 뽑혔다: 예측 분산 R + v̄ (v̄ = 내용의 세계 간 분산, 경험적 베이즈),
    P += χ (v̄ / |z|²) ẑẑᵀ (그 방향으로 zᵀPz가 v̄만큼 는다). vbar=None이면 예전 판 (q₀ = 1: '바뀜' 분산 R + |z|², 12.1절에서 χ = 0)."""
    feat, wm, key = parts(p, We, ybar, kind)

    def f(c, x):
        M, P, e, ha, hb = c
        o, a, o2, rw = x
        k = key(o, a)
        e = ALPHA * e + k
        if use_wm:
            mu, lv = wm(o, a)
            s = jnp.sqrt(jnp.exp(lv) / vmax)
        else:
            mu, s = jnp.zeros(HO), 1.0
        z = s * e
        r = target(feat, o2, rw) - mu
        err = loss_terms(r - M @ z)
        if evidence:
            h = ha / (ha + hb)
            L0 = gauss_logpdf(r - M @ z, 0.01 + z @ P @ z)
            vb = (z @ z) if vbar is None else vbar
            L1 = gauss_logpdf(r, 0.01 + vb)
            chi = jax.nn.sigmoid(jnp.log(h) + L1 - jnp.log1p(-h) - L0)
            zh = z / (jnp.linalg.norm(z) + 1e-12)
            P = P + chi * (vb / (z @ z + 1e-12)) * jnp.outer(zh, zh)
            ha, hb = ha + chi, hb + 1 - chi
        Pz = P @ z
        S = 0.01 + z @ Pz
        M = M + jnp.outer(r - M @ z, Pz) / S
        P = P - jnp.outer(Pz, Pz) / S
        return (M, P, e, ha, hb), err
    Rw = jnp.zeros(acts.shape[0]) if Rw is None else Rw
    return jax.lax.scan(f, c0, (O[:-1], acts, O[1:], Rw))


def c_init():
    return (jnp.zeros((HO, DK)), jnp.eye(DK), jnp.zeros(DK), jnp.array(1.0), jnp.array(1.0))


def train(kind, seed=0, D=256, n_train=512, iters_wm=400, iters_k=600, B=32, L=64, log=print):
    We, ybar = emb_fixed(seed)
    p = init(jax.random.PRNGKey(seed))
    _, tr = stack_sessions(range(50_000, 50_000 + n_train), D)
    O_all, _, R_all = jax.jit(jax.vmap(observe_stream))(tr)
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
            return jnp.mean(0.5 * (((Y_all[idx] - mu) ** 2).sum(-1) * jnp.exp(-lv) + HO * lv))
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
            Rc = jax.lax.dynamic_slice_in_dim(R_all[idx], t0, L, axis=1)
            c2, err = jax.vmap(lambda o, a, c, rw: write_scan(pp, We, ybar, kind, True, vmax, o, a, c, Rw=rw))(O, Ac, carry, Rc)
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


def content_var(p, vmax, kind, use_wm, seed=0, n=128, D=256):
    """v̄: 산출물이 도착한 관측에서 잔차 (y − μ⁰)의 성분당 평균 제곱 (학습용 세션, 경험적 베이즈)."""
    We, ybar = emb_fixed(seed)
    feat, wm, _ = parts(p, We, ybar, kind)
    _, tr = stack_sessions(range(60_000, 60_000 + n), D)
    O, _, Rw = jax.jit(jax.vmap(observe_stream))(tr)
    blank = ER.TEX[ER.BLANK].reshape(-1)
    arrived = ~jnp.all(jnp.isclose(O[:, 1:, SLOT], blank), -1)                # [n, T]
    def res(o, a, o2, rw):
        mu = wm(o, a)[0] if use_wm else jnp.zeros(HO)
        return target(feat, o2, rw) - mu
    R = jax.vmap(jax.vmap(res))(O[:, :-1], tr["acts"], O[:, 1:], Rw)
    return float(((R ** 2).mean(-1) * arrived).sum() / arrived.sum())


def probe(p, vmax, kind, use_wm, stack, seed=0, evidence=False, vbar=None):
    """교사 구간 끝의 기억으로 채집처마다 '가서 하고 돌아오기'를 시뮬레이션, 도착 예측의 B 점수. 반환: 점수 [NW, NSITE]."""
    We, ybar = emb_fixed(seed)
    feat, wm, key = parts(p, We, ybar, kind)
    phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
    gB = phi[1] - phi.mean(0)

    def one(sess, plans, arrive_idx):
        O, s_end, Rw = observe_stream(sess)
        (M, P, e, _, _), _ = write_scan(p, We, ybar, kind, use_wm, vmax, O, sess["acts"], c_init(), evidence=evidence, vbar=vbar, Rw=Rw)

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
                out = M @ (sv * e)
                yh = out.reshape(12, H)[SLOT]
                rh = reward_head(p, yh, feat(o).reshape(12, H)[CARD])
                s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
                return (s2, o2, e), (yh, rh)
            _, (YH, RH) = jax.lax.scan(f, (s_end, O[-1], e), plan)
            yh = YH[ai]
            return jnp.stack([(yh @ gB) / (jnp.linalg.norm(yh) * jnp.linalg.norm(gB) + 1e-8), RH[ai]])
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
