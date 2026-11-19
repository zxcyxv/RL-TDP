"""설계 초안 1 (docs/design_draft1.md)의 공통 부품. 환경·DreamerV3 연결과 무관한 부분만.

- 빠른 선호 u: r_{t+1} ≈ uᵀφ_{t+1}를 역전파 없이 갱신한다. 갱신식은 열린 항목이라 델타 규칙과 칼만을 둘 다 둔다.
- 후속 특징 Ψ_θ(s, a): 느린 머리. 행위자 E를 따른다고 가정한 TD 목표로 학습한다 (Ψ^E).
- 행동: Q(a) = uᵀΨ(s, a), π(a) ∝ E(a | s) · exp[κ Q(a)].
- 증류: 느린 보상 머리가 sg(uᵀφ)를 맞추는 손실.

φ는 RSSM의 다음 잠재 z_{t+1}을 펼친 벡터로 받는다 (design_draft1.md 정한 것). 여기서는 길이 dφ의 벡터로만 다룬다.
"""
import jax
import jax.numpy as jnp


# ---------- 빠른 선호 u (역전파 없음) ----------

def pref_init(dphi):
    return jnp.zeros(dphi)


def pref_update_delta(u, phi, r, lr):
    """델타 규칙: u ← u + lr · (r − uᵀφ) φ."""
    return u + lr * (r - u @ phi) * phi


def pref_kalman_init(dphi, p0):
    return dict(u=jnp.zeros(dphi), P=p0 * jnp.eye(dphi))


def pref_update_kalman(st, phi, r, R, Q=0.0):
    """칼만 (정밀도 가중) 갱신. r = uᵀφ + 잡음(R), u는 스텝마다 과정 잡음 Q·I로 퍼진다. Joseph 꼴."""
    u, P = st["u"], st["P"] + Q * jnp.eye(phi.shape[0])
    Pphi = P @ phi
    S = R + phi @ Pphi
    g = Pphi / S
    u = u + (r - u @ phi) * g
    A = jnp.eye(phi.shape[0]) - jnp.outer(g, phi)
    return dict(u=u, P=A @ P @ A.T + R * jnp.outer(g, g))


# ---------- 후속 특징 Ψ (느린 가중치) ----------

def mlp_init(key, sizes):
    ks = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (a, b)) / jnp.sqrt(a), jnp.zeros(b)) for k, a, b in zip(ks, sizes[:-1], sizes[1:])]


def mlp(p, x):
    for W, b in p[:-1]:
        x = jax.nn.silu(x @ W + b)
    W, b = p[-1]
    return x @ W + b


def sf_init(key, ds, na, dphi, hidden=256):
    return dict(net=mlp_init(key, [ds, hidden, hidden, na * dphi]), na=na, dphi=dphi)


def sf_apply(p, s):
    """Ψ(s, ·): [..., na, dφ]."""
    out = mlp(p["net"], s)
    return out.reshape(out.shape[:-1] + (p["na"], p["dphi"]))


def sf_td_target(p_target, s_next, phi_next, cont, pi_next, gamma):
    """Ψ^E의 TD 목표: φ_{t+1} + γ c_t Σ_b E(b | s_{t+1}) Ψ(s_{t+1}, b). pi_next는 다음 상태의 행위자 확률 [..., na]."""
    psi_next = jnp.einsum("...b,...bd->...d", pi_next, sf_apply(p_target, s_next))
    return phi_next + gamma * cont[..., None] * psi_next


def sf_loss(p, p_target, s, a, s_next, phi_next, cont, pi_next, gamma):
    """제곱 TD 오차 (목표는 정지 기울기)."""
    psi = jnp.take_along_axis(sf_apply(p, s), a[..., None, None], axis=-2)[..., 0, :]
    y = jax.lax.stop_gradient(sf_td_target(p_target, s_next, phi_next, cont, pi_next, gamma))
    return jnp.mean(jnp.sum((y - psi) ** 2, -1))


# ---------- 행동 ----------

def q_values(u, psi):
    """Q(a) = uᵀΨ(s, a). psi [..., na, dφ]."""
    return jnp.einsum("...ad,d->...a", psi, u)


def policy_logits(actor_logits, q, kappa):
    """π(a) ∝ E(a | s) · exp[κ Q(a)]의 로짓."""
    return jax.nn.log_softmax(actor_logits, -1) + kappa * q


# ---------- 증류 ----------

def distill_loss(reward_head, phi, u):
    """느린 보상 머리 r_θ(φ)가 sg(uᵀφ)를 맞추는 손실. reward_head(phi) -> [...]."""
    target = jax.lax.stop_gradient(phi @ u)
    return jnp.mean((reward_head(phi) - target) ** 2)
