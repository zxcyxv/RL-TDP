"""갈림길 과제를 초기화 없이 이어 붙인 흐름: 일정 망각 대 증거 망각 (docs/world/README.md 7절).

세계 W개를 차례로 겪는다. 기억 (M, P, 흔적 e)은 세계 경계에서 비우지 않는다. 경계 신호도 주지 않는다.
세계마다 끝의 검사에서 그때의 M으로 읽어 정답 여부를 기록한다 (읽기는 쓰지 않는다).
느린 모델은 gate_memory의 이상적인 것 (세계 안 스텝별 μ⁰, s, 행동 빈도)을 그대로 쓴다.

망각 (관측 기회 = s > 0인 스텝, 세계별 결과가 나오는 곳):
  none      Q = 0. 정확한 정적 베이즈.
  const     매 관측 기회마다 P ← P + h q₀ I (모든 방향 균일).
  evidence  두 가설 비교. 그대로: ε ~ N(0, S₀), S₀ = R + xᵀPx. 바뀜 (그 주소의 내용이 새 사전에서 다시 뽑힘): ε' ~ N(0, R + q₀ xᵀx)
            (ε' = 사전 평균 0 기준의 잔차 y). 변화 확률 χ = h L₁ / (h L₁ + (1 − h) L₀).
            P ← P + χ q₀ x̂ x̂ᵀ (지금 관측한 주소 방향만), 그 뒤 보통의 정밀도 가중 쓰기. 바뀌었다고 판단한 방향만 다시 배우기 쉬워진다.
h = 1/1024, q₀ = 1 (plasticity_design.md 7.3절에서 실행 전에 고정한 값).
scale="eb": 사전 척도를 경험적 베이즈로. 빠른 가중치 사전 P₀ = p₀ I, 바뀜 가설 q₀ = p₀, p₀ = v̄ / E|x|².
  v̄ = 관측 기회에서 결과 편차의 성분당 세계 간 분산 (느린 모델), E|x|² = 학습용 흐름 (세계 20개)의 관측 기회에서 회귀 입력 크기의 평균.
  "예측되는 내용의 분산 = 세계마다 실제로 달라지는 분산"이 되게 한다. 손으로 고르는 값이 없다.
forget="evidence_h": evidence에 더해 사건 위험률 h를 흐름 안에서 온라인 추정. h ~ Beta(a, b), 사전 Beta(1, 1),
  관측 기회마다 a += χ, b += 1 − χ, h = a / (a + b). 정답 변화율을 주지 않는다.
"""
import jax
import jax.numpy as jnp

from . import gate as G
from . import gate_memory as GM

H_HAZ, Q0, R = 1 / 1024, 1.0, 0.01


def gauss_logpdf(e, var):
    return -0.5 * (jnp.sum(e * e) / var + e.shape[0] * jnp.log(2 * jnp.pi * var))


def step_fn(forget, alpha, surp, q0=Q0):
    def f(c, x):
        M, P, e, ha, hb = c
        k, y, s, u = x
        e = alpha * e + (u if surp else 1.0) * k
        xv = s * e
        obs = s > 1e-3
        if forget == "const":
            P = P + jnp.where(obs, H_HAZ * q0, 0.0) * jnp.eye(P.shape[0])
        elif forget in ("evidence", "evidence_h"):
            h = ha / (ha + hb) if forget == "evidence_h" else H_HAZ
            S0 = R + xv @ P @ xv
            eps0 = y - M @ xv
            L0 = gauss_logpdf(eps0, S0)
            L1 = gauss_logpdf(y, R + q0 * (xv @ xv))
            chi = jax.nn.sigmoid(jnp.log(h) + L1 - jnp.log1p(-h) - L0)
            ha = ha + jnp.where(obs, chi, 0.0)
            hb = hb + jnp.where(obs, 1.0 - chi, 0.0)
            xh = xv / (jnp.linalg.norm(xv) + 1e-12)
            P = P + jnp.where(obs, chi, 0.0) * q0 * jnp.outer(xh, xh)
        a = P @ xv
        Sx = R + xv @ a
        M = M + jnp.outer(y - M @ xv, a) / Sx
        P = P - jnp.outer(a, a) / Sx
        return (M, P, e, ha, hb), None
    return f


def prior_scale(enc, mu, s, pa, d, D, surp, n_worlds=20):
    """p₀ = v̄ / E|x|² (학습용 흐름 하나, 첫 세계 제외)."""
    ws = G.make_worlds(jax.random.PRNGKey(31_337), n_worlds, d, D)
    obs = s > 1e-3

    def world(e, w):
        K, Y, _, _ = GM.stream(enc, w)
        U = -jnp.log(jnp.take_along_axis(pa, w.act[:, None], 1)[:, 0] + 1e-12)
        def g(e, x):
            k, u, sv = x
            e = GM.ALPHA * e + (u if surp else 1.0) * k
            return e, jnp.sum((sv * e) ** 2)
        e, x2 = jax.lax.scan(g, e, (K, U, s))
        return e, ((Y - mu) ** 2).mean(-1), x2
    def body(e, w):
        e, v, x2 = world(e, w)
        return e, (v, x2)
    _, (v, x2) = jax.lax.scan(body, jnp.zeros(GM.DK), ws)
    vbar = v[1:, obs].mean()
    return vbar / x2[1:, obs].mean()


def run_stream(forget, seed, n_worlds=200, n_streams=256, d=4, D=32, surp=False, scale="fixed"):
    """세계 순번별 정확도 [n_worlds] (흐름 n_streams개 평균)."""
    enc = GM.encoder(seed)
    mu, s, pa = GM.slow_model(enc, d, D)
    mbar = (mu * s[:, None]).sum(0) / s.sum()
    p0 = prior_scale(enc, mu, s, pa, d, D, surp) if scale == "eb" else 1.0
    f = step_fn(forget, GM.ALPHA, surp, q0=p0)

    def one_stream(key):
        ws = G.make_worlds(key, n_worlds, d, D)

        def world(c, w):
            K, Y, Kt, phi = GM.stream(enc, w)
            U = -jnp.log(jnp.take_along_axis(pa, w.act[:, None], 1)[:, 0] + 1e-12)
            c, _ = jax.lax.scan(f, c, (K, Y - mu, s, U))
            a, _ = GM.choose(c[0], Kt, phi, mbar)
            return c, a == w.answer
        c0 = (jnp.zeros((GM.H, GM.DK)), p0 * jnp.eye(GM.DK), jnp.zeros(GM.DK), jnp.array(1.0), jnp.array(1.0))
        _, ok = jax.lax.scan(world, c0, ws)
        return ok

    ok = jax.lax.map(one_stream, jax.random.split(jax.random.PRNGKey(20_000 + seed), n_streams), batch_size=32)
    return ok.mean(0)


def hazard_trace(seed, n_worlds=200, d=4, D=32):
    """흐름 하나에서 세계 끝마다의 추정 h (evidence_h, 놀라움 가중, eb)."""
    enc = GM.encoder(seed)
    mu, s, pa = GM.slow_model(enc, d, D)
    p0 = prior_scale(enc, mu, s, pa, d, D, True)
    f = step_fn("evidence_h", GM.ALPHA, True, q0=p0)
    ws = G.make_worlds(jax.random.PRNGKey(99), n_worlds, d, D)

    def world(c, w):
        K, Y, _, _ = GM.stream(enc, w)
        U = -jnp.log(jnp.take_along_axis(pa, w.act[:, None], 1)[:, 0] + 1e-12)
        c, _ = jax.lax.scan(f, c, (K, Y - mu, s, U))
        return c, c[3] / (c[3] + c[4])
    c0 = (jnp.zeros((GM.H, GM.DK)), p0 * jnp.eye(GM.DK), jnp.zeros(GM.DK), jnp.array(1.0), jnp.array(1.0))
    return jax.lax.scan(world, c0, ws)[1]
