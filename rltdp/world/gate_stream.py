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
"""
import jax
import jax.numpy as jnp

from . import gate as G
from . import gate_memory as GM

H_HAZ, Q0, R = 1 / 1024, 1.0, 0.01


def gauss_logpdf(e, var):
    return -0.5 * (jnp.sum(e * e) / var + e.shape[0] * jnp.log(2 * jnp.pi * var))


def step_fn(forget, alpha, surp):
    def f(c, x):
        M, P, e = c
        k, y, s, u = x
        e = alpha * e + (u if surp else 1.0) * k
        xv = s * e
        obs = s > 1e-3
        if forget == "const":
            P = P + jnp.where(obs, H_HAZ * Q0, 0.0) * jnp.eye(P.shape[0])
        elif forget == "evidence":
            S0 = R + xv @ P @ xv
            eps0 = y - M @ xv
            L0 = gauss_logpdf(eps0, S0)
            L1 = gauss_logpdf(y, R + Q0 * (xv @ xv))
            chi = jax.nn.sigmoid(jnp.log(H_HAZ) + L1 - jnp.log1p(-H_HAZ) - L0)
            xh = xv / (jnp.linalg.norm(xv) + 1e-12)
            P = P + jnp.where(obs, chi, 0.0) * Q0 * jnp.outer(xh, xh)
        a = P @ xv
        Sx = R + xv @ a
        M = M + jnp.outer(y - M @ xv, a) / Sx
        P = P - jnp.outer(a, a) / Sx
        return (M, P, e), None
    return f


def run_stream(forget, seed, n_worlds=200, n_streams=256, d=4, D=32, surp=False):
    """세계 순번별 정확도 [n_worlds] (흐름 n_streams개 평균)."""
    enc = GM.encoder(seed)
    mu, s, pa = GM.slow_model(enc, d, D)
    mbar = (mu * s[:, None]).sum(0) / s.sum()
    f = step_fn(forget, GM.ALPHA, surp)

    def one_stream(key):
        ws = G.make_worlds(key, n_worlds, d, D)

        def world(c, w):
            K, Y, Kt, phi = GM.stream(enc, w)
            U = -jnp.log(jnp.take_along_axis(pa, w.act[:, None], 1)[:, 0] + 1e-12)
            c, _ = jax.lax.scan(f, c, (K, Y - mu, s, U))
            a, _ = GM.choose(c[0], Kt, phi, mbar)
            return c, a == w.answer
        c0 = (jnp.zeros((GM.H, GM.DK)), jnp.eye(GM.DK), jnp.zeros(GM.DK))
        _, ok = jax.lax.scan(world, c0, ws)
        return ok

    ok = jax.lax.map(one_stream, jax.random.split(jax.random.PRNGKey(20_000 + seed), n_streams), batch_size=32)
    return ok.mean(0)
