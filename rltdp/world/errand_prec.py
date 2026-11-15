"""지연 채집 과제: 칸별 정밀도 (docs/world/README.md 12.3절).

12.2까지의 문제: 키 학습 손실이 정밀도 없는 제곱 오차라, 줄어들 수 없는 오차 (처음 보는 칸의 무늬)가 손실을 지배했다.
월드 모델의 분산도 스텝마다 스칼라 하나라 "새로 들어온 칸은 모른다, 밀려 온 칸은 안다"를 칸별로 구별하지 못했다.

  월드 모델 (장면, 행동) → 칸 p마다 다음 특징의 평균 μ⁰_p [8]와 로그 분산 log v_p. 칸별 가우시안 음의 로그우도.
  시냅스 (칸마다): y_p = μ⁰_p + s_p M_p e + ε,  s_p = √(v_p / v̄)  (v̄ = 학습 자료의 v 최대).
    칸마다 정밀도 가중 쓰기 (RLS, R = 0.01, P_p₀ = I): 회귀 입력 z_p = s_p e. 흔적 e ← α e + k (α = 2^(−1/8)).
  키 학습 손실 = 모형의 예측 분포 아래 칸별 음의 로그우도 (F의 정확도 항):
    σ²_p = R + z_pᵀ P_p z_p,  Σ_p [|y_p − μ⁰_p − M_p z_p|² / σ²_p + H log σ²_p] / 2.
    처음 보는 칸: 월드 모델 분산도 시냅스 불확실성도 커서 무게가 작다. 이미 겪은 채집처의 산출물: 시냅스가 아는 주소라 무게가 크다.
  키 k(o, a) = SwiGLU([flatten(emb(o)), onehot(a)]) [128], 크기 정규화 없음.
판정: 12.1과 같은 기억 내용 판정 (채집처마다 가서 하고 돌아오기를 시뮬레이션, 도착 예측의 B 점수 최대가 b1인가).
"""
import jax
import jax.numpy as jnp
import optax

from . import errand as ER
from . import errand_learned as EL

H, DK, SWI, NP = 8, 128, 128, 12
HY = NP * H
ALPHA, R_OBS = 2.0 ** (-1 / 8), 0.01
SLOT = EL.SLOT


def init(key):
    ks = jax.random.split(key, 4)
    d = lambda k, i, o, s=1.0: jax.random.normal(k, (i, o)) * s / jnp.sqrt(i)
    k5, _ = jax.random.split(ks[3])
    return dict(wm1=d(ks[0], HY + ER.NA, 256), wb1=jnp.zeros(256), wm2=d(ks[1], 256, HY + NP, 0.1), wb2=jnp.zeros(HY + NP),
                Ku=d(ks[2], 9 * H + ER.NA, 2 * SWI), Kd=d(ks[3], SWI, DK),
                Wr=jax.random.normal(k5, (ER.NA, DK, 9 * H)) / jnp.sqrt(9.0 * H))            # 고정 난수 키 (시야 9칸만)


def parts(p, We, ybar, kind):
    def feat(o):
        return ((o @ We) - ybar).reshape(NP, H)

    def wm(o, a):
        x = jnp.concatenate([feat(o).reshape(-1), jax.nn.one_hot(a, ER.NA)])
        out = jax.nn.relu(x @ p["wm1"] + p["wb1"]) @ p["wm2"] + p["wb2"]
        return out[:HY].reshape(NP, H), out[HY:]                                   # 평균 [12, 8], 로그 분산 [12]

    def key(o, a):                                       # 주소는 시야 9칸과 행동만 (HUD = 의뢰 카드·도착·도장은 주소에 넣지 않는다, 12.4)
        w = feat(o)[:9].reshape(-1)
        if kind == "learned":
            x = jnp.concatenate([w, jax.nn.one_hot(a, ER.NA)])
            g, u = jnp.split(x @ p["Ku"], 2, -1)
            return (jax.nn.silu(g) * u) @ p["Kd"]
        k = p["Wr"][a] @ w
        return k / (jnp.linalg.norm(k) + 1e-8)
    return feat, wm, key


def c_init():
    return (jnp.zeros((NP, H, DK)), jnp.tile(jnp.eye(DK), (NP, 1, 1)), jnp.zeros(DK), jnp.ones(NP), jnp.ones(NP))


def gauss_logpdf(e, var):
    return -0.5 * (jnp.sum(e * e) / var + e.shape[0] * jnp.log(2 * jnp.pi * var))


def write_scan(p, We, ybar, kind, vmax, O, acts, c0, evidence=False, stop_t=None):
    """evidence=True: 칸별 증거 망각. 칸 p마다 '그대로' N(M_p z, R + zᵀP_p z) 대 '바뀜' N(0, R + v_p)
    (그 칸의 내용이 사전분포에서 새로 뽑혔다, v_p = 월드 모델이 예측한 그 칸의 세계 간 분산)의 우도비로 변화 확률 χ_p,
    P_p += χ_p (v_p / |z|²) ẑẑᵀ (그 방향으로 예측 분산이 v_p만큼 는다). 칸별 사건 위험률 h_p ~ Beta(a_p, b_p) 온라인, 사전 Beta(1, 1)."""
    """stop_t: 이 스텝 뒤로는 M, P에 쓰지 않는다 (흔적 e는 계속 진행). 판정을 'C 도착 직후의 기억'으로 맞추기 위함 (12.4)."""
    feat, wm, key = parts(p, We, ybar, kind)
    T = acts.shape[0]
    stop_t = T if stop_t is None else stop_t

    def f(c, x):
        M, P, e, ha, hb = c
        o, a, o2, t = x
        M_in, P_in = M, P
        e = ALPHA * e + key(o, a)
        mu, lv = wm(o, a)
        s = jnp.sqrt(jnp.exp(lv) / vmax)                                             # [12]
        r = feat(o2) - mu                                                             # [12, 8]

        v = jnp.exp(lv)

        def per_patch(Mp, Pp, sp, rp, vp, hap, hbp):
            z = sp * e
            err = rp - Mp @ z
            S0 = R_OBS + z @ Pp @ z
            if evidence:
                h = hap / (hap + hbp)
                chi = jax.nn.sigmoid(jnp.log(h) + gauss_logpdf(rp, R_OBS + vp) - jnp.log1p(-h) - gauss_logpdf(err, S0))
                zh = z / (jnp.linalg.norm(z) + 1e-12)
                Pp = Pp + chi * (vp / (z @ z + 1e-12)) * jnp.outer(zh, zh)
                hap, hbp = hap + chi, hbp + 1 - chi
            Pz = Pp @ z
            S = R_OBS + z @ Pz                                                        # 예측 분산
            nll = 0.5 * (err @ err / S + H * jnp.log(S))
            Mp = Mp + jnp.outer(err, Pz) / S
            Pp = Pp - jnp.outer(Pz, Pz) / S
            return Mp, Pp, nll, hap, hbp
        M, P, nll, ha, hb = jax.vmap(per_patch)(M, P, s, r, v, ha, hb)
        keep = t > stop_t
        M = jnp.where(keep, M_in, M)
        P = jnp.where(keep, P_in, P)
        return (M, P, e, ha, hb), nll.sum()
    return jax.lax.scan(f, c0, (O[:-1], acts, O[1:], jnp.arange(T)))


def train(kind, seed=0, D=256, n_train=512, iters_wm=400, iters_k=600, B=32, L=64, log=print):
    We, ybar = EL.emb_fixed(seed)
    p = init(jax.random.PRNGKey(seed))
    _, tr = EL.stack_sessions(range(50_000, 50_000 + n_train), D)
    O_all, _, _ = jax.jit(jax.vmap(EL.observe_stream))(tr)
    A_all = tr["acts"]
    feat, _, _ = parts(p, We, ybar, kind)
    Y_all = jax.vmap(jax.vmap(feat))(O_all[:, 1:])                                   # [n, T, 12, 8]
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))

    q = {k: p[k] for k in ("wm1", "wb1", "wm2", "wb2")}
    st = opt.init(q)

    @jax.jit
    def upd_wm(q, st, idx):
        def nll(q):
            _, wm, _ = parts(dict(p, **q), We, ybar, kind)
            mu, lv = jax.vmap(jax.vmap(wm))(O_all[idx, :-1], A_all[idx])
            return jnp.mean(0.5 * (((Y_all[idx] - mu) ** 2).sum(-1) * jnp.exp(-lv) + H * lv).sum(-1))
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

    q = {k: p[k] for k in ("Ku", "Kd")}
    st = opt.init(q)
    T = A_all.shape[1]

    @jax.jit
    def upd_k(q, st, idx, t0, carry):
        def lf(q):
            pp = dict(p, **q)
            O = jax.lax.dynamic_slice_in_dim(O_all[idx], t0, L + 1, axis=1)
            Ac = jax.lax.dynamic_slice_in_dim(A_all[idx], t0, L, axis=1)
            c2, nll = jax.vmap(lambda o, a, c: write_scan(pp, We, ybar, kind, vmax, o, a, c))(O, Ac, carry)
            return nll.mean(), c2
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
            log(f"  [keys] it {it} precision-weighted nll {float(l):.2f}")
    p.update(q)
    return p, vmax


def probe(p, vmax, kind, seed=0, evidence=False):
    We, ybar = EL.emb_fixed(seed)
    feat, wm, key = parts(p, We, ybar, kind)
    phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
    gB = phi[1] - phi.mean(0)

    def one(sess, plans, arrive_idx):
        O, s_end, _ = EL.observe_stream(sess)
        blank = ER.TEX[ER.BLANK].reshape(-1)
        arrived = ~jnp.all(jnp.isclose(O[1:, SLOT], blank), -1)                     # 각 스텝 뒤 관측에 산출물 도착
        t_c = jnp.max(jnp.where(arrived, jnp.arange(arrived.shape[0]), -1))         # 마지막 도착 = b0 재방문의 C
        (M, P, e, _, _), _ = write_scan(p, We, ybar, kind, vmax, O, sess["acts"], c_init(), evidence=evidence, stop_t=t_c)

        def sim(plan, ai):
            def f(c, a):
                s, o, e = c
                e = ALPHA * e + key(o, a)
                _, lv = wm(o, a)
                sv = jnp.sqrt(jnp.exp(lv) / vmax)
                yh = M[SLOT] @ (sv[SLOT] * e)
                s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
                return (s2, o2, e), yh
            _, YH = jax.lax.scan(f, (s_end, O[-1], e), plan)
            yh = YH[ai]
            return (yh @ gB) / (jnp.linalg.norm(yh) * jnp.linalg.norm(gB) + 1e-8)
        return jax.vmap(sim)(plans, arrive_idx)
    return one
