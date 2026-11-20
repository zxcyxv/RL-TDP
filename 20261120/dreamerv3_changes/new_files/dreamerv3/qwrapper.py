"""Runs the Q-path policy and applies its shared-weight updates.

The per-environment Q state (dsyn, traces, ...) stays stacked on the device
and is not split into per-environment lists like the rest of the policy
carry. The jitted policy returns the new W_g (with the dsyn of every finished
episode transferred in) and W_o (with the summed online updates); this wrapper
writes them into the train params, the policy params and a pending policy
sync. Separate copies are stored because the jax agent deletes old policy
buffers when it swaps in a sync.
"""

import collections
import json

import elements
import jax
import jax.numpy as jnp
import numpy as np
from embodied.jax import internal


class QAgentWrapper:

  def __init__(self, agent, logdir, log_every=500, stats_every=10):
    self.agent = agent
    self.path = elements.Path(logdir) / 'qstats.jsonl'
    self.log_every = log_every
    self.stats_every = stats_every
    self.calls = 0
    self.pending_stats = []
    self.qc = None

  def __getattr__(self, name):
    return getattr(self.agent, name)

  def init_policy(self, batch_size):
    carry = self.agent.init_policy(batch_size)
    a = self.agent
    with jax.transfer_guard('allow'):
      stacked = a._stack(carry[4])
      self.qc = internal.to_global(stacked, a.policy_sharded)
    return carry[:4]

  def _set(self, Wg, Wo):
    a = self.agent
    a.params['q/Wg'] = jnp.array(Wg, copy=True)
    a.params['q/Wo'] = jnp.array(Wo, copy=True)
    a.policy_params['q/Wg'] = jnp.array(Wg, copy=True)
    a.policy_params['q/Wo'] = jnp.array(Wo, copy=True)
    if a.pending_sync:
      a.pending_sync['q/Wg'] = jnp.array(Wg, copy=True)
      a.pending_sync['q/Wo'] = jnp.array(Wo, copy=True)

  @elements.timer.section('qwrapper_policy')
  def policy(self, carry, obs, mode='train'):
    a = self.agent
    for key, space in a.obs_space.items():
      assert np.isfinite(obs[key]).all(), (obs[key], key, space)
    with a.policy_lock:
      dobs = internal.device_put(obs, a.policy_sharded)
      with a.n_actions.lock:
        counter = a.n_actions.value
        a.n_actions.value += 1
      seed = a._seeds(counter, a.policy_mirrored)
      gcarry = internal.to_global(a._stack(carry), a.policy_sharded)
      gcarry, acts, outs = a._policy(
          a.policy_params, seed, (*gcarry, self.qc), dobs, mode)
      if a.pending_sync:
        old = a.policy_params
        a.policy_params = a.pending_sync
        jax.tree.map(lambda x: x.delete(), old)
        a.pending_sync = None
    self.qc = gcarry[4]
    with jax.transfer_guard('allow'):
      self._set(self.qc['Wg_new'], self.qc['Wo_new'])
    acts, outs = a._take_outs(internal.fetch_async((acts, outs)))
    carry = a._split(internal.to_local(gcarry[:4]))
    finite = outs.pop('finite', {})
    for key, fin in finite.items():
      assert all(x.all() for x in jax.tree.leaves(fin)), str(finite)
    assert (acts['action'] >= 0).all(), acts

    self.calls += 1
    if self.calls % self.stats_every == 0:
      self.pending_stats.append(self.qc['stats'])
    if self.calls % self.log_every == 0:
      with jax.transfer_guard('allow'):
        stats = jax.device_get(self.pending_stats)
        Wo, Wg = jax.device_get((self.qc['Wo_new'], self.qc['Wg_new']))
      rec = {k: float(np.mean([s[k][0] for s in stats])) for k in stats[0]}
      rec['transfers'] = float(stats[-1]['transfers'][0])  # cumulative
      # cumulative signed projections of online W_o updates onto W_o
      rec['dWo_pos_cum'] = float(stats[-1]['dWo_pos_cum'][0])
      rec['dWo_zero_cum'] = float(stats[-1]['dWo_zero_cum'][0])
      rec['dWo_zero_disc_cum'] = float(stats[-1]['dWo_zero_disc_cum'][0])
      rec['dWo_zero_chg_cum'] = float(stats[-1]['dWo_zero_chg_cum'][0])
      rec['calls'] = self.calls
      rec['Wo_norm'] = float(np.linalg.norm(Wo))
      rec['Wg_norm'] = float(np.linalg.norm(Wg))
      with self.path.open('a') as f:
        f.write(json.dumps(rec) + '\n')
      self.pending_stats = []
    return carry, acts, outs

  def train(self, carry, data):
    out = self.agent.train(carry, data)
    with jax.transfer_guard('allow'):
      self._set(self.agent.params['q/Wg'], self.agent.params['q/Wo'])
    return out
