"""rltdp/agent/sf.py 단위 시험 (합성 데이터)."""
import numpy as np
import jax
import jax.numpy as jnp
import optax

from rltdp.agent import sf


def test_pref_kalman_recovers_linear_reward():
    rng = np.random.default_rng(0)
    d = 8
    u_true = rng.normal(size=d)
    st = sf.pref_kalman_init(d, p0=1.0)
    for _ in range(200):
        phi = rng.normal(size=d)
        st = sf.pref_update_kalman(st, jnp.asarray(phi), float(u_true @ phi), R=1e-4)
    assert np.allclose(np.asarray(st["u"]), u_true, atol=1e-2)
    assert np.all(np.linalg.eigvalsh(np.asarray(st["P"])) > 0)


def test_pref_delta_recovers_linear_reward():
    rng = np.random.default_rng(1)
    d = 8
    u_true = rng.normal(size=d)
    u = sf.pref_init(d)
    for _ in range(3000):
        phi = rng.normal(size=d) / np.sqrt(d)
        u = sf.pref_update_delta(u, jnp.asarray(phi), float(u_true @ phi), lr=0.5)
    assert np.allclose(np.asarray(u), u_true, atol=1e-2)


def test_sf_td_matches_analytic_successor_features():
    # 상태 4개, 행동 2개의 표 형태 MDP. 상태는 원-핫, φ_{t+1} = 다음 상태의 원-핫.
    ns, na, gamma = 4, 2, 0.8
    rng = np.random.default_rng(2)
    T = rng.dirichlet(np.ones(ns), size=(ns, na))            # T[s, a, s']
    E = rng.dirichlet(np.ones(na), size=ns)                   # 행위자 E(a | s)
    # 해석해: Ψ(s, a) = Σ_s' T[s,a,s'] (e_s' + γ Σ_b E(b|s') Ψ(s', b)) = T (I + γ M), M[s'] = Σ_b E Ψ(s', b)
    P_pi = np.einsum("sb,sbt->st", E, T)                      # 상태 전이 (E를 따를 때)
    M = P_pi @ np.linalg.inv(np.eye(ns) - gamma * P_pi)      # M[s'] = Σ_b E(b|s') Ψ(s', b)
    psi_true = np.einsum("sat,tu->sau", T, np.eye(ns) + gamma * M)

    p = sf.sf_init(jax.random.PRNGKey(0), ns, na, ns, hidden=64)
    opt = optax.adam(3e-3)
    st = opt.init(p["net"])
    S = jnp.eye(ns)

    @jax.jit
    def step(net, st, tgt_net, key):
        k1, k2, k3 = jax.random.split(key, 3)
        s = jax.random.randint(k1, (256,), 0, ns)
        a = jax.random.randint(k2, (256,), 0, na)
        s2 = jax.vmap(lambda kk, ss, aa: jax.random.choice(kk, ns, p=jnp.asarray(T)[ss, aa]))(jax.random.split(k3, 256), s, a)
        pp, pt = dict(p, net=net), dict(p, net=tgt_net)
        loss, g = jax.value_and_grad(lambda n: sf.sf_loss(dict(p, net=n), pt, S[s], a, S[s2], S[s2], jnp.ones(256),
                                                          jnp.asarray(E)[s2], gamma))(net)
        u, st = opt.update(g, st, net)
        return optax.apply_updates(net, u), st, loss

    net, tgt = p["net"], p["net"]
    key = jax.random.PRNGKey(1)
    for it in range(4000):
        key, k = jax.random.split(key)
        net, st, _ = step(net, st, tgt, k)
        if it % 100 == 0:
            tgt = net
    psi = np.asarray(sf.sf_apply(dict(p, net=net), S))
    assert np.abs(psi - psi_true).max() < 0.1, np.abs(psi - psi_true).max()


def test_q_and_policy_shapes():
    psi = jnp.ones((3, 2, 5))
    u = jnp.arange(5.0)
    q = sf.q_values(u, psi)
    assert q.shape == (3, 2) and np.allclose(np.asarray(q), 10.0)
    lg = sf.policy_logits(jnp.zeros((3, 2)), q, kappa=1.0)
    assert lg.shape == (3, 2)


def test_distill_loss_zero_when_head_matches():
    u = jnp.array([1.0, -2.0, 0.5])
    phi = jnp.asarray(np.random.default_rng(3).normal(size=(10, 3)))
    assert float(sf.distill_loss(lambda x: x @ u, phi, u)) < 1e-10
