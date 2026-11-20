"""DreamerV3 world model + locally learned Q path (RL-TDP, 2026-11-20 draft).

World model (encoder, RSSM, decoder, reward and continue heads) is trained
exactly as in the official agent. The DreamerV3 actor, critic and replay value
loss are removed. Behavior comes from a Q path that reads the world model state
through stop-gradient and is learned with local rules only:

  Q(s, a | c) = sum_i Wo_i * c_i * silu(u_i),  u_i = (Wg + dsyn)_i . x(s, a)
  c = h_t (GRU state),  x(s, a) = h_{t+1}(a) = GRU(h_t, z_t, a)

  Wg   : initialized randomly once, afterwards only changed by transfer from
         dsyn (online episodes) or from the per-rollout fast weights of
         imagined rollouts.
  Wo   : initialized to zero, slow local TD rule.
  dsyn : fast synapse, per environment, cleared at every episode start and
         transferred into Wg at the episode end (handled by QAgentWrapper).

Online TD(lambda) runs inside policy(). Imagined TD(lambda) runs inside
train() on world-model rollouts with predicted reward (phase 1 only).
"""

import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from . import rssm
from .agent import Agent as DreamerAgent

f32 = jnp.float32
sg = jax.lax.stop_gradient


def dsilu(u):
  s = jax.nn.sigmoid(u)
  return s * (1 + u * (1 - s))


def qvalues(Wg, Wo, dsyn, c, X):
  """c: [B, n], X: [B, A, d], dsyn: [B, n, d] or None. Returns Q [B, A]."""
  if dsyn is None:
    u = jnp.einsum('nd,bad->ban', Wg, X)
  else:
    u = jnp.einsum('bnd,bad->ban', Wg[None] + dsyn, X)
  s = jax.nn.silu(u)
  Q = jnp.einsum('n,bn,ban->ba', Wo, c, s)
  return Q, u, s


def take(x, a):
  """x: [B, A, ...], a: [B] -> [B, ...]."""
  return jnp.take_along_axis(
      x, a.reshape((-1,) + (1,) * (x.ndim - 1)), 1)[:, 0]


def softmax_value(Q, tau):
  pi = jax.nn.softmax(Q / tau, -1)
  return (pi * Q).sum(-1), pi


class QHead(nj.Module):

  def __init__(self, n, d, gscale, seed, wo_std=0.0):
    self.n, self.d, self.gscale, self.seed = n, d, gscale, seed
    self.wo_std = wo_std

  def __call__(self):
    def init_wg():
      key = jax.random.PRNGKey(self.seed)
      return self.gscale / np.sqrt(self.d) * jax.random.normal(
          key, (self.n, self.d), f32)
    Wg = self.value('Wg', init_wg)
    def init_wo():
      # zero, or random once (initialized once, never resampled)
      if not self.wo_std:
        return jnp.zeros((self.n,), f32)
      key = jax.random.PRNGKey(self.seed + 1)
      return self.wo_std * jax.random.normal(key, (self.n,), f32)
    Wo = self.value('Wo', init_wo)
    return Wg, Wo

  def update(self, dWg, dWo):
    Wg, Wo = self()
    self.write('Wg', Wg + dWg)
    self.write('Wo', Wo + dWo)


class QAgent(DreamerAgent):

  banner = ['--- DreamerV3 world model + local Q path ---']

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    assert list(act_space.keys()) == ['action'], act_space
    assert act_space['action'].discrete
    self.nact = int(np.asarray(act_space['action'].classes).flatten()[0])

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = rssm.Encoder(enc_space, **config.enc.simple, name='enc')
    self.dyn = rssm.RSSM(act_space, **config.dyn.rssm, name='dyn')
    self.dec = rssm.Decoder(dec_space, **config.dec.simple, name='dec')
    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)
    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    q = config.q
    self.qcfg = q
    deter = config.dyn.rssm.deter
    self.qhead = QHead(deter, deter, q.gscale, q.seed, q.wo_std, name='q')

    self.modules = [self.dyn, self.enc, self.dec, self.rew, self.con]
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')
    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    for k in ('policy', 'value', 'repval'):
      scales.pop(k)
    self.scales = scales

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|q)/'

  # ------------------------------------------------------------------ carry

  def _qcarry(self, B):
    n = d = self.config.dyn.rssm.deter
    return dict(
        dsyn=jnp.zeros((B, n, d), f32),
        e_d=jnp.zeros((B, n, d), f32),
        e_o=jnp.zeros((B, n), f32),
        gn_d=jnp.zeros((B,), f32),
        gn_o=jnp.zeros((B,), f32),
        q_prev=jnp.zeros((B,), f32),
        valid=jnp.zeros((B,), bool),
        tau_s=jnp.zeros((1,), f32),
        Wo_new=jnp.zeros((n,), f32),
        Wg_new=jnp.zeros((n, d), f32),
        stats=dict(dWo_zero_disc_cum=jnp.zeros((1,), f32),
                   dWo_zero_chg_cum=jnp.zeros((1,), f32),
                   zero_q_prev=jnp.zeros((1,), f32),
                   zero_V=jnp.zeros((1,), f32),
                   dWo_pos_cum=jnp.zeros((1,), f32),
                   dWo_zero_cum=jnp.zeros((1,), f32),
                   delta_abs=jnp.zeros((1,), f32), delta_max=jnp.zeros((1,), f32),
                   dsyn_norm=jnp.zeros((1,), f32), q_sel=jnp.zeros((1,), f32),
                   u_std=jnp.zeros((1,), f32), transfers=jnp.zeros((1,), f32),
                   tau=jnp.zeros((1,), f32), pi_maxprob=jnp.zeros((1,), f32)),
    )

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space),
        self._qcarry(batch_size))

  def init_train(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_report(self, batch_size):
    return self.init_train(batch_size)

  # ------------------------------------------------------------- one step

  def _candidates(self, deter, stoch, raw=False):
    """x(s, a) = h_{t+1}(a) for every action, plus the prior stoch."""
    B, A = deter.shape[0], self.nact
    rep = lambda x: jnp.repeat(x, A, 0)
    carry = dict(deter=rep(deter), stoch=rep(stoch))
    acts = {'action': jnp.tile(jnp.arange(A, dtype=jnp.int32), B)}
    _, (feat, _) = self.dyn.imagine(carry, acts, 1, False, single=True)
    unrep = lambda x: x.reshape((B, A, *x.shape[1:]))
    Draw = unrep(feat['deter'])
    X = f32(Draw)
    if self.qcfg.xnorm:
      X = X / jnp.sqrt((X ** 2).mean(-1, keepdims=True) + 1e-8)
    if raw:
      return X, unrep(feat['stoch']), Draw
    return X, unrep(feat['stoch'])

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact, qc) = carry
    q = self.qcfg
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}

    Wg, Wo = self.qhead()
    c = f32(feat['deter'])
    X, _ = self._candidates(feat['deter'], feat['stoch'])

    # Episode start: clear the fast synapse and the traces.
    keep = (~reset).astype(f32)
    qc = dict(qc)
    qc['dsyn'] = qc['dsyn'] * keep[:, None, None]
    qc['e_d'] = qc['e_d'] * keep[:, None, None]
    qc['e_o'] = qc['e_o'] * keep[:, None]
    valid = qc['valid'] & ~reset

    # TD error for the previous transition.
    Q, _, _ = qvalues(Wg, Wo, qc['dsyn'], c, X)
    if q.tau_rel:
      # temperature = tau_rel * running mean of the across-action std of Q
      spread = Q.std(-1).mean()
      tau_s = (1 - q.tau_rate) * qc['tau_s'][0] + q.tau_rate * spread
      tau = q.tau_rel * tau_s + 1e-8
    else:
      tau_s = qc['tau_s'][0]
      tau = q.tau
    V, _ = softmax_value(Q, tau)
    term = obs['is_terminal'].astype(f32)
    r = obs['reward'].astype(f32)
    q_prev_old = qc['q_prev']
    delta = r + q.gamma * (1 - term) * V - q_prev_old
    delta = jnp.where(valid, delta, 0.0)
    # Joint normalization over (W_o, dsyn): one NLMS step on the whole
    # parameter vector, with per-group rates beta (fast) and eta (slow).
    if q.trace_norm:
      # normalize by the trace norm, so the change of Q(s_t) per step is at
      # most rate * delta even when the trace has accumulated
      gn = (qc['e_o'] ** 2).sum(-1) + (qc['e_d'] ** 2).sum((1, 2)) + q.eps
    else:
      gn = qc['gn_d'] + qc['gn_o'] + q.eps
    step_d = q.beta * delta / gn
    step_o = q.eta * delta / gn
    dsyn = qc['dsyn'] + step_d[:, None, None] * qc['e_d']
    dWo = step_o[:, None] * qc['e_o']

    # Act with the updated fast synapse.
    Q, u, s = qvalues(Wg, Wo, dsyn, c, X)
    logits = Q / tau
    qc['tau_s'] = tau_s.reshape((1,))
    a = jax.random.categorical(nj.seed(), logits, -1).astype(jnp.int32)
    act = {'action': a}

    # Gradients of Q(s, a) for the chosen action and trace update.
    s_a, u_a, x_a = take(s, a), take(u, a), take(X, a)
    g_o = c * s_a
    coef = Wo[None] * c * dsilu(u_a)
    g_d = coef[:, :, None] * x_a[:, None, :]
    gl = q.gamma * q.lam
    qc['e_o'] = gl * qc['e_o'] + g_o
    qc['e_d'] = gl * qc['e_d'] + g_d
    qc['gn_o'] = (g_o ** 2).sum(-1)
    qc['gn_d'] = (coef ** 2).sum(-1) * (x_a ** 2).sum(-1)
    qc['q_prev'] = take(Q, a)
    qc['valid'] = jnp.ones_like(valid)
    qc['dsyn'] = dsyn
    # Shared-weight side, computed here so the wrapper only swaps arrays in:
    # summed online W_o updates, and transfer of dsyn into W_g for every
    # environment whose episode ends at this step (dsyn is cleared at the
    # next episode start).
    last = obs['is_last'].astype(f32)
    qc['Wo_new'] = Wo + (dWo.mean(0) if q.env_mean else dWo.sum(0))
    qc['Wg_new'] = Wg + jnp.einsum('b,bnd->nd', last, dsyn)
    nv = jnp.maximum(valid.sum(), 1)
    wdir = Wo / (jnp.linalg.norm(Wo) + 1e-8)
    proj = dWo @ wdir                                   # [B] signed growth
    rpos = (r > 0).astype(f32)
    agg = (lambda x: x.mean(0)) if q.env_mean else (lambda x: x.sum(0))
    # Exact split of delta (linear in dWo):
    #   delta = r + (g' - 1) * q_prev + g' * (V - q_prev),  g' = gamma*(1-term)
    # discount term (g' - 1) * q_prev, change term g' * (V - q_prev).
    gp = q.gamma * (1 - term)
    d_disc = jnp.where(valid, (gp - 1) * q_prev_old, 0.0)
    d_chg = jnp.where(valid, gp * (V - q_prev_old), 0.0)
    per_delta = jnp.where(jnp.abs(delta) > 0, proj / jnp.where(
        jnp.abs(delta) > 0, delta, 1.0), 0.0)        # projection per unit delta
    zero = valid.astype(f32) * (1 - rpos)
    nz = jnp.maximum(zero.sum(), 1)
    qc['stats'] = dict(
        dWo_zero_disc_cum=qc['stats']['dWo_zero_disc_cum'][0] + agg(
            per_delta * d_disc * zero),
        dWo_zero_chg_cum=qc['stats']['dWo_zero_chg_cum'][0] + agg(
            per_delta * d_chg * zero),
        zero_q_prev=(q_prev_old * zero).sum() / nz,
        zero_V=(V * zero).sum() / nz,
        dWo_pos_cum=qc['stats']['dWo_pos_cum'][0] + agg(proj * rpos),
        dWo_zero_cum=qc['stats']['dWo_zero_cum'][0] + agg(proj * (1 - rpos)),
        delta_abs=jnp.abs(delta).sum() / nv,
        delta_max=jnp.abs(delta).max(),
        dsyn_norm=jnp.sqrt((dsyn ** 2).sum((1, 2))).mean(),
        q_sel=take(Q, a).mean(),
        u_std=u_a.std(),
        transfers=qc['stats']['transfers'][0] + last.sum(),
        tau=jnp.asarray(tau, f32),
        pi_maxprob=jax.nn.softmax(logits, -1).max(-1).mean(),
    )
    # Rank-1 so the batch sharding of the policy carry applies.
    qc['stats'] = jax.tree.map(lambda x: x.reshape((1,)), qc['stats'])

    out = {}
    if q.debug_outs:
      out['qdbg/K'] = c[:, None, :] * s       # features for every action
      out['qdbg/Xn'] = X                       # normalized x for every action
      out['qdbg/c'] = c
      out['qdbg/Q'] = Q
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, tokens=tokens, feat=feat, act=act, Q=Q)))
    carry = (enc_carry, dyn_carry, dec_carry, act, qc)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  # ---------------------------------------------------------------- train

  def train(self, carry, data):
    carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True)
    metrics.update(mets)
    if self.qcfg.imag_updates:
      metrics.update(self._imagined_q_update(entries[1]))
    else:
      self.qhead()  # make sure the Q variables exist
    outs = {}
    if self.config.replay_context:
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
      outs['replay'] = updates
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, training):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    losses = {}
    metrics = {}
    enc_carry, enc_entries, tokens = self.enc(
        enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
        dyn_carry, tokens, prevact, reset, training)
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(
        dec_carry, repfeat, reset, training)
    inp = self.feat2tensor(repfeat)
    inp = inp if self.config.reward_grad else sg(inp)
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      target = f32(value) / 255 if (
          space.dtype == np.uint8 and len(space.shape) == 3) else value
      losses[key] = recon.loss(sg(target))
    B, T = reset.shape
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])
    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return loss, (carry, entries, outs, metrics)

  def _imagined_q_update(self, dyn_entries):
    q = self.qcfg
    H = q.imag_length
    Wg, Wo = self.qhead()
    starts = jax.tree.map(
        lambda x: sg(x.reshape((-1, *x.shape[2:]))), dyn_entries)
    starts = {k: starts[k] for k in ('deter', 'stoch')}
    disc = 1 - 1 / self.config.horizon if self.config.contdisc else 1.0

    def step(carry, _):
      X, S, Draw = self._candidates(carry['deter'], carry['stoch'], raw=True)
      c = f32(carry['deter'])
      Q, u, s = qvalues(Wg, Wo, None, c, X)
      tau = (q.tau_rel * sg(Q.std(-1).mean()) + 1e-8) if q.tau_rel else q.tau
      V, _ = softmax_value(Q, tau)
      a = jax.random.categorical(nj.seed(), Q / tau, -1).astype(jnp.int32)
      nxt = dict(deter=nn.cast(take(Draw, a)), stoch=take(S, a))
      rec = dict(c=c, x=take(X, a), Qa=take(Q, a), V=V,
                 ua=take(u, a), sa=take(s, a), nxt=nxt)
      return nxt, rec

    unroll = H if self.config.dyn.rssm.unroll else 1
    last, traj = nj.scan(step, starts, (), H, unroll=unroll, axis=0)
    traj = jax.tree.map(sg, traj)
    # Reward and continuation of all imagined next states in one batch.
    ft = self.feat2tensor(traj.pop('nxt'))           # [H, N, F]
    traj['r'] = sg(f32(self.rew(ft, 2).pred()))
    traj['cont'] = sg(jnp.clip(f32(self.con(ft, 2).prob(1)) / disc, 0, 1))
    Xl, _ = self._candidates(last['deter'], last['stoch'])
    Ql, _, _ = qvalues(Wg, Wo, None, f32(last['deter']), Xl)
    tau_l = (q.tau_rel * sg(Ql.std(-1).mean()) + 1e-8) if q.tau_rel else q.tau
    Vlast, _ = softmax_value(Ql, tau_l)
    Vnext = jnp.concatenate([traj['V'][1:], Vlast[None]], 0)  # [H, N]

    def lam_step(G, inp):
      r, cont, vn = inp
      G = r + q.gamma * cont * ((1 - q.lam) * vn + q.lam * G)
      return G, G
    _, G = jax.lax.scan(
        lam_step, Vlast, (traj['r'], traj['cont'], Vnext), reverse=True)
    G = sg(G)
    delta = G - traj['Qa']  # [H, N]

    g_o = traj['c'] * traj['sa']                     # [H, N, n]
    coef = Wo * traj['c'] * dsilu(traj['ua'])       # [H, N, n]
    gn = ((g_o ** 2).sum(-1) + (coef ** 2).sum(-1) * (
        traj['x'] ** 2).sum(-1)).mean() + q.eps
    M = delta.size
    dWo = q.imag_eta * jnp.einsum('hb,hbn->n', delta, g_o) / M / gn
    dWg = q.imag_beta * jnp.einsum(
        'hb,hbn,hbd->nd', delta, coef, traj['x']) / M / gn
    self.qhead.update(dWg, dWo)
    wdir = Wo / (jnp.linalg.norm(Wo) + 1e-8)
    return {
        'qimag/dWo_proj': dWo @ wdir,
        'qimag/delta_abs': jnp.abs(delta).mean(),
        'qimag/G': G.mean(),
        'qimag/Q': traj['Qa'].mean(),
        'qimag/rhat': traj['r'].mean(),
        'qimag/rhat_max': traj['r'].max(),
        'qimag/cont': traj['cont'].mean(),
        'qimag/u_std': traj['ua'].std(),
        'qimag/c_rms': jnp.sqrt((traj['c'] ** 2).mean()),
        'qimag/x_rms_raw': jnp.sqrt((f32(last['deter']) ** 2).mean()),
        'qimag/pi_maxprob': jax.nn.softmax(Ql / tau_l, -1).max(-1).mean(),
        'qimag/tau': jnp.asarray(tau_l, f32),
        'qimag/x_std_across_actions': Xl.std(1).mean(),
        'qimag/Wo_norm': jnp.linalg.norm(Wo),
        'qimag/Wg_norm': jnp.linalg.norm(Wg),
    }

  def report(self, carry, data):
    return carry, {}
