"""갈림길 과제: 학습된 인코더가 시냅스 입력 (키)을 만든다 (docs/world/README.md 8절).

시냅스는 1단계 ours 그대로: 키 선행 흔적 e ← α e + k_t (α = 2^(−1/8)), 정밀도 가중 쓰기 (RLS, R = 0.01, P₀ = I), 세션마다 0.
쓰기 세기 s_t = 느린 모델의 놀라움 (1단계와 같은 이상적 느린 모델의 세계 간 분산 크기). 흔적 입력에 손으로 무게를 주지 않는다.
바꾸는 것은 키 하나:
  random    k_t = unit(Wk[a_t] flatten(emb(o_t)))      (고정 난수, 1단계 ours)
  surp      random 키에 손 편향 u_t = −log p(a_t)를 곱함 (1단계 ours_surp, 비교용)
  learned   k_t = MLP([flatten(emb(o_t)), onehot(a_t)]) (느린 가중치, 크기 정규화 없음: 무엇을 얼마나 흔적에 넣을지 스스로 배운다)
학습: 같은 세계에서 대기 뒤 무작위 문을 다시 방문하는 세션 (D ∈ {8, 16, 32}). 재방문 갈림길의 HUD에는 무작위 자원.
     손실 = 재방문 결과의 편차를 시냅스로 예측한 오차 (그때 M은 노출에서 쓴 것). 세션 전체 BPTT.
평가 (원래 과제): 후보 a마다 그 문으로 걸어 들어갈 때 생길 흔적 e(a)로 읽는다: ŷ(a) = M (s* e(a)), 점수 cos(ŷ(a), φ − μ̄).
     복도 장면은 두 문이 같고 결정적이라 느린 모델이 안다고 둔다.
"""
import jax
import jax.numpy as jnp
import optax

from . import gate as G
from . import gate_memory as GM

DK, HID, R_OBS = 32, 64, 0.01


def revisit_world(key, d, D):
    w = G.make_world(key, d, D)
    k1, k2 = jax.random.split(jax.random.fold_in(key, 1))
    types = jnp.stack([w.ids[d, 3, 3], w.ids[2 * d + 1, 3, 3]])
    first_left = w.act[0] == G.LEFT
    door = jax.random.randint(k2, (), 0, 2)
    res = jnp.where((door == 0) == first_left, types[0], types[1])
    hud = G.RES0 + jax.random.randint(k1, (), 0, G.N_RES)
    J = G.scene(G.BG["junc"], G.JUNC, hud)
    corr = [G.scene(G.BG[f"corr{j}"], G.CORR, G.HUD_BLANK) for j in range(d - 1)]
    ids = jnp.concatenate([w.ids[:-1], jnp.stack([J, *corr, G.scene(G.BG[f"corr{d - 1}"], res, G.HUD_BLANK)])])
    act = jnp.concatenate([w.act, jnp.stack([jnp.where(door == 0, G.LEFT, G.RIGHT)] + [jnp.array(G.FWD)] * (d - 1))])
    return ids, act


def slow_stats(enc, ids_batch, act_batch):
    """이상적 느린 모델: 스텝별 μ⁰, s (세계 간 분산 크기), 행동 빈도 pa."""
    c = ids_batch[:, 1:, 3, 3]
    Y = G.TEX[c].reshape(*c.shape, -1) @ enc["We"]
    mu = Y.mean(0)
    v = ((Y - mu) ** 2).mean((0, 2))
    pa = jax.nn.one_hot(act_batch, G.A).mean(0)
    return mu, jnp.sqrt(v / (v.max() + 1e-12)), pa


def init(key):
    k1, k2 = jax.random.split(key)
    return dict(W1=jax.random.normal(k1, (49 * GM.H + G.A, HID)) / jnp.sqrt(49.0 * GM.H), b1=jnp.zeros(HID),
                W2=jax.random.normal(k2, (HID, DK)) / jnp.sqrt(HID), b2=jnp.zeros(DK))


def keys_fn(kind, enc):
    def k(p, ids, a):
        e = GM.emb(enc, ids)
        if kind == "learned":
            x = jnp.concatenate([e.reshape(-1), jax.nn.one_hot(a, G.A)])
            return jax.nn.relu(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
        return GM.key(enc, e, a)
    return k


def run(p, kf, enc, ids, act, mu, s, u):
    """세션 하나. 반환: 최종 (M, P, e), 스텝별 예측 오차 제곱, 스텝별 키 노름."""
    Y = GM.emb(enc, ids[1:])[:, G.CENTER]

    def f(c, x):
        M, P, e = c
        o, a, y, m, sv, uv = x
        k = kf(p, o, a)
        e = GM.ALPHA * e + uv * k
        z = sv * e
        r = y - m
        err = jnp.sum((r - M @ z) ** 2)
        Pz = P @ z
        S = R_OBS + z @ Pz
        M = M + jnp.outer(r - M @ z, Pz) / S
        P = P - jnp.outer(Pz, Pz) / S
        return (M, P, e), (err, jnp.linalg.norm(k))
    c0 = (jnp.zeros((GM.H, DK)), jnp.eye(DK), jnp.zeros(DK))
    return jax.lax.scan(f, c0, (ids[:-1], act, Y, mu, s, u))


def train(seed, kind="learned", d=4, iters=2000, B=64, lr=1e-3, log=print):
    enc = GM.encoder(seed)
    kf = keys_fn(kind, enc)
    p = init(jax.random.PRNGKey(seed))
    if kind != "learned":
        return p, enc, kf
    stats = {}
    for D in (8, 16, 32):
        ids, act = jax.vmap(lambda k: revisit_world(k, d, D))(jax.random.split(jax.random.PRNGKey(555 + D), 2048))
        stats[D] = slow_stats(enc, ids, act)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    st = opt.init(p)

    def loss_fn(p, ids, act, D):
        mu, s, _ = stats[D]
        u = jnp.ones(act.shape[1])
        _, (err, _) = jax.vmap(lambda i, a: run(p, kf, enc, i, a, mu, s, u))(ids, act)
        return err[:, -1].mean()

    def make_upd(D):
        @jax.jit
        def upd(p, st, ids, act):
            l, g = jax.value_and_grad(loss_fn)(p, ids, act, D)
            g, st = opt.update(g, st, p)
            return optax.apply_updates(p, g), st, l
        return upd
    upd = {D: make_upd(D) for D in stats}
    key = jax.random.PRNGKey(1000 + seed)
    for it in range(iters):
        key, k = jax.random.split(key)
        D = (8, 16, 32)[it % 3]
        ids, act = jax.vmap(lambda kk: revisit_world(kk, d, D))(jax.random.split(k, B))
        p, st, l = upd[D](p, st, ids, act)
        if it % 500 == 0 or it == iters - 1:
            log(f"  [{kind} seed {seed}] it {it} revisit loss {float(l):.4f}")
    return p, enc, kf


def evaluate(p, enc, kf, kind, seed, d, D, n=1024):
    """S_mem, 그리고 노출 중 스텝 종류별 평균 키 노름 (갈림길 / 복도 / 방)."""
    ws = G.make_worlds(jax.random.PRNGKey(10_000 + seed), n, d, D)
    mu, s, pa = GM.slow_model(enc, d, D)
    mbar = (mu * s[:, None]).sum(0) / s.sum()
    s_star = s.max()
    corr = [G.scene(G.BG[f"corr{j}"], G.CORR, G.HUD_BLANK) for j in range(d - 1)]

    def one(w):
        u = -jnp.log(jnp.take_along_axis(pa, w.act[:, None], 1)[:, 0] + 1e-12) if kind == "surp" else jnp.ones(w.act.shape[0])
        (M, P, e), (_, kn) = run(p, kf, enc, w.ids, w.act, mu, s, u)

        def walk(a):                                                             # 그 문으로 들어갈 때의 흔적 (현재 흔적에서 이어서)
            ua = -jnp.log(pa[0, a] + 1e-12) if kind == "surp" else 1.0
            ee = GM.ALPHA * e + ua * kf(p, w.ids[-1], a)
            for c in corr:
                uc = -jnp.log(pa[1, G.FWD] + 1e-12) if kind == "surp" else 1.0
                ee = GM.ALPHA * ee + uc * kf(p, c, G.FWD)
            return s_star * ee
        Z = jnp.stack([walk(G.LEFT), walk(G.RIGHT)])
        return M, Z, GM.emb(enc, w.ids[-1])[G.HUD], kn
    M, Z, phi, kn = jax.lax.map(one, ws, batch_size=64)
    a, _ = jax.vmap(lambda m, z, ph: GM.choose(m, z, ph, mbar))(M, Z, phi)
    acc = float(jnp.mean(a == ws.answer))
    kn = kn.mean(0)
    norms = dict(junction=float(kn[0]), corridor=float(kn[1:d].mean()), room=float(kn[2 * d + 2:].mean()) if D > 0 else None)
    return 2 * acc - 1, norms
