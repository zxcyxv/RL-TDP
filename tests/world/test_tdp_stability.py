"""State bounds and differentiability of the projected TDP delta rule."""
import jax
import jax.numpy as jnp
import numpy as np

from rltdp.world import env as E
from rltdp.world.tdp_agent import make_tdp


def test_long_rollout_and_gradient():
    cfg = E.EnvCfg(N=3, V=3, P=2, T=512)
    ag = make_tdp(cfg, tex=E.textures(cfg), d=16, heads=2)
    p = ag['init'](jax.random.PRNGKey(0))
    c = ag['carry0'](jax.random.PRNGKey(1))
    # Exercise large learned gains and value projections, not just initialization.
    p['log_kappa'] = jnp.array(20.)
    p['mem']['Wv'] *= 100
    p['mem']['c'] *= 100
    p['mem']['pole'] = p['mem']['pole'].at[0].set(-30.)
    c, out = jax.jit(lambda p, c: jax.lax.scan(
        lambda c, k: ag['step'](p, c, k), c,
        jax.random.split(jax.random.PRNGKey(2), 512)))(p, c)
    for x in jax.tree.leaves((c, out)):
        assert np.isfinite(np.asarray(x)).all()
    assert float(out['Wn'].max()) <= 1.00001
    assert float(jnp.linalg.norm(c['ek'], axis=-1).max()) <= 1.00001
    assert float(jnp.linalg.norm(c['ev'], axis=-1).max()) <= np.sqrt(8) + 1e-5
    # Loss gradients include recurrent writes and the long-memory projection.
    def objective(p):
        _, outs = jax.lax.scan(lambda c, k: ag['step'](p, c, k), c,
                              jax.random.split(jax.random.PRNGKey(3), 4))
        return sum(jnp.mean(outs[k]) for k in ('obs', 'rew', 'v'))
    grad = jax.jit(jax.grad(objective))(p)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad))


def test_no_long_memory():
    cfg = E.EnvCfg(N=3, V=3, P=2)
    ag = make_tdp(cfg, tex=E.textures(cfg), d=16, heads=2, gate=False)
    p = ag['init'](jax.random.PRNGKey(0))
    c = ag['carry0'](jax.random.PRNGKey(1))
    c, _ = jax.jit(ag['step'])(p, c, jax.random.PRNGKey(2))
    np.testing.assert_array_equal(c['W'], 0)


def test_projection_is_active_for_oversized_state():
    cfg = E.EnvCfg(N=3, V=3, P=2)
    ag = make_tdp(cfg, tex=E.textures(cfg), d=16, heads=2)
    p = ag['init'](jax.random.PRNGKey(0))
    c = ag['carry0'](jax.random.PRNGKey(1))
    c['W'] = jnp.full_like(c['W'], 1000.)
    c, metrics = jax.jit(ag['step'])(p, c, jax.random.PRNGKey(2))
    assert float(metrics['Wclip']) == 1.
    assert float(jnp.linalg.norm(c['W'], axis=(-2, -1)).max()) <= 8.00001
