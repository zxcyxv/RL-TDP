"""갈림길 과제의 기억들 (학습 없음, 1단계). docs/world/README.md 6절.

공통 (느린 부분 = 고정 난수, 시드별):
  패치 임베딩 We [147 → h=8], 위치와 무관 (모든 칸에 같은 사상).
  주소 k_t = unit(Wk[a_t] · flatten(emb(o_t))), dk = 32. 결정 시점 정보 (장면, 행동)만. 행동마다 사영이 달라 곱으로 묶인다.
  결과 y_{t+1} = emb(o_{t+1})의 가운데 칸 (그 행동 뒤에 "여기 무엇이 있나"). 보상·자원 정답·좌표는 쓰지 않는다.
  읽기: 검사 장면에서 후보 a ∈ {LEFT, RIGHT}의 주소로 ŷ(a) = M k(a) (세계별 편차의 예측), 필요 φ = emb(HUD 칸).
  점수 = cos(ŷ(a), φ − μ̄): 예측한 편차와 필요한 편차의 비교. μ̄ = 느린 모델이 아는 결과의 평균 (s > 0인 스텝의 μ⁰ 평균).
  (원래 φ와의 내적은 자원 무늬 임베딩끼리의 상관 때문에 노름이 판정을 좌우했다: README 6절.)
느린 모델 (1단계: 이상적인 것): 세계에 걸친 정확한 조건부 평균 μ⁰_t와 분산 v_t (세션 구조 = 스텝 번호를 안다).
  4,096개 학습용 세계에서 평균해 얻는다. 빠른 기억은 느린 모델이 예측하지 못하는 세계별 편차 r = y − μ⁰만 맡는다.
  생성 모델: y_{t+1} = μ⁰_t + s_t wᵀ e_t + ε,  s_t = √(v_t / v̄) (느린 모델이 세계마다 다르다고 아는 만큼만 빠른 항이 기여).
  그래서 결정적인 결과 (복도 다음 복도)는 빠른 기억에 제약을 걸지 않는다. c = 1 그대로, 문턱 없음.
  (이 항이 없으면 잔차 0인 결정적 스텝이 흔적 위 회귀를 통해 갈림길 연합을 지운다: README 6절.)
기억 (빠른 상태, 세션마다 0에서 시작, 매 스텝 갱신, 감쇠 없음):
  ours        키 선행 ET e ← α e + k (α = 2^(−1/8)), 정밀도 가중 쓰기 (RLS = 조건부 선형 가우시안 사후, R = 0.01, P₀ = I, c = 1)
  ours_noet   같은 것, α = 0 (ET 없음)
  hebb_et     같은 ET, 모두 쓰는 Hebb M += y eᵀ (정밀도 없음)
  product     TDP band_product를 토큰 하나·실수 극점 하나로: E_K ← z E_K + γ k, E_V ← z E_V + γ y, M = E_V E_Kᵀ (γ = √(1 − z²))
              z = 2^(−1/8) (짧은 창 = 짧은 보존), product_long은 z = 1 − 1/4096 (긴 보존 = 긴 대칭 창). 토큰 하나라 계수 1이다 (TDP의 계수는 토큰에서 온다)
"""
import jax
import jax.numpy as jnp

from . import gate as G

ALPHA = 2.0 ** (-1 / 8)
H, DK = 8, 32                                                       # DK는 encoder(seed, dk)로 바꿀 수 있다 (진단)


def encoder(seed, dk=DK):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return dict(We=jax.random.normal(k1, (147, H)) / jnp.sqrt(147.0),
                Wk=jax.random.normal(k2, (G.A, dk, 49 * H)) / jnp.sqrt(49.0 * H))


def emb(enc, ids):
    return G.patches(ids) @ enc["We"]                                   # [..., 49, H]


def key(enc, e, a):
    k = jnp.einsum("akh,h->ak", enc["Wk"], e.reshape(-1))[a]
    return k / (jnp.linalg.norm(k) + 1e-8)


def stream(enc, w):
    """세계 하나 → 주소 [T−1, DK], 결과 [T−1, H], 검사 후보 주소 [2, DK], 필요 φ [H]."""
    E = emb(enc, w.ids)                                                  # [T, 49, H]
    K = jax.vmap(lambda e, a: key(enc, e, a))(E[:-1], w.act)
    Y = E[1:, G.CENTER]
    Kt = jnp.stack([key(enc, E[-1], G.LEFT), key(enc, E[-1], G.RIGHT)])
    return K, Y, Kt, E[-1, G.HUD]


def write_rls(K, Y, S_, alpha, R=0.01):
    def f(c, x):
        M, P, e = c
        k, y, s = x
        e = alpha * e + k
        x_ = s * e                                                       # 이 스텝에서 빠른 항의 회귀 입력
        a = P @ x_
        S = R + x_ @ a
        M = M + jnp.outer(y - M @ x_, a) / S
        P = P - jnp.outer(a, a) / S
        return (M, P, e), None
    dk = K.shape[-1]
    (M, _, _), _ = jax.lax.scan(f, (jnp.zeros((H, dk)), jnp.eye(dk), jnp.zeros(dk)), (K, Y, S_))
    return M


def write_hebb(K, Y, S_, alpha):
    def f(c, x):
        M, e = c
        k, y, s = x
        e = alpha * e + k
        return (M + s * jnp.outer(y, e), e), None
    dk = K.shape[-1]
    (M, _), _ = jax.lax.scan(f, (jnp.zeros((H, dk)), jnp.zeros(dk)), (K, Y, S_))
    return M


def write_product(K, Y, S_, z):
    g = jnp.sqrt(1 - z * z)
    def f(c, x):
        ek, ev = c
        k, y, s = x
        return (z * ek + g * k, z * ev + g * s * y), None
    (ek, ev), _ = jax.lax.scan(f, (jnp.zeros(K.shape[-1]), jnp.zeros(H)), (K, Y, S_))
    return jnp.outer(ev, ek)


MEMORIES = dict(
    ours=lambda K, Y, S: write_rls(K, Y, S, ALPHA),
    ours_noet=lambda K, Y, S: write_rls(K, Y, S, 0.0),
    hebb_et=lambda K, Y, S: write_hebb(K, Y, S, ALPHA),
    product=lambda K, Y, S: write_product(K, Y, S, ALPHA),
    product_long=lambda K, Y, S: write_product(K, Y, S, 1 - 1 / 4096),
)


def choose(M, Kt, phi, mbar):
    yh = Kt @ M.T                                                        # [2, H]
    g = phi - mbar
    score = (yh @ g) / (jnp.linalg.norm(yh, axis=-1) * jnp.linalg.norm(g) + 1e-8)
    return jnp.where(score[0] >= score[1], G.LEFT, G.RIGHT), score


def slow_model(enc, d, D, n=4096, gap=0):
    """이상적인 느린 모델: 스텝별 세계 간 평균 μ⁰ [T−1, H]와 분산 기반 크기 s [T−1]."""
    ws = G.make_worlds(jax.random.PRNGKey(777), n, d, D, gap)
    c = ws.ids[:, 1:, 3, 3]
    Y = G.TEX[c].reshape(*c.shape, -1) @ enc["We"]                     # 가운데 칸만 [n, T−1, H]
    mu = Y.mean(0)
    v = ((Y - mu) ** 2).mean((0, 2))
    return mu, jnp.sqrt(v / (v.max() + 1e-12))


def evaluate(name, seed, d, D, n=1024, swap=False, gap=0, dk=DK):
    """정확도 A, S_mem = (A − 0.5) / (1 − 0.5). swap=True: 다른 세계의 M으로 읽는다 (기억 교환 개입)."""
    enc = encoder(seed, dk)
    mu, s = slow_model(enc, d, D, gap=gap)
    ws = G.make_worlds(jax.random.PRNGKey(10_000 + seed), n, d, D, gap)

    def one(w):                                                          # 세계 하나씩 (긴 대기에서 메모리 절약)
        K, Y, Kt, phi = stream(enc, w)
        return MEMORIES[name](K, Y - mu, s), Kt, phi
    M, Kt, phi = jax.lax.map(one, ws, batch_size=64)
    if swap:
        M = jnp.roll(M, 1, axis=0)
    mbar = (mu * s[:, None]).sum(0) / s.sum()
    a, _ = jax.vmap(lambda m, k, p: choose(m, k, p, mbar))(M, Kt, phi)
    acc = jnp.mean(a == ws.answer)
    return float(acc), float(2 * acc - 1)
