"""Offline sweep of the Q-path local rule on fixed features (diagnostic).

Uses features dumped by diag_features.py --dump (world model frozen, recorded
actions). Runs the same online rule as qagent.policy over the training
streams (per-stream dsyn cleared at episode start and transferred into W_g at
episode end, shared W_g / W_o), for several hyperparameter settings at once,
then evaluates the slow weights (dsyn = 0) on held-out streams against the
Monte-Carlo return of the recorded behavior.

Differences to the agent, on purpose: the TD target bootstraps with the
recorded next action (SARSA) so that the Monte-Carlo return of the data is
the matching ground truth; no imagined updates.

python dreamerv3/sweep_offline.py --feats <npz> --out <json>
"""
import argparse
import itertools
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

f32 = jnp.float32


def dsilu(u):
  s = jax.nn.sigmoid(u)
  return s * (1 + u * (1 - s))


def mc_returns(rew, first, last, term, gamma):
  T, B = rew.shape
  G = np.zeros((T, B)); valid = np.zeros((T, B), bool)
  for b in range(B):
    g, ended = 0.0, False
    for t in range(T - 1, -1, -1):
      if t + 1 >= T or first[t + 1, b]:
        g = 0.0
        continue
      if last[t + 1, b]:
        g, ended = rew[t + 1, b], True
      else:
        g = rew[t + 1, b] + gamma * g
      G[t, b], valid[t, b] = g, ended
  return G, valid


def make_runner(n, d, A, gamma, lam, eps):

  def qv(Wg, Wo, dsyn, c, X):
    u = jnp.einsum('bnd,bad->ban', Wg[None] + dsyn, X)
    s = jax.nn.silu(u)
    return jnp.einsum('n,bn,ban->ba', Wo, c, s), u, s

  def run(hp, Wg, Wo, data):
    beta, eta, trace_norm, env_mean = hp
    B = data[0].shape[1]
    st = dict(
        Wg=Wg, Wo=Wo,
        dsyn=jnp.zeros((B, n, d), f32), e_d=jnp.zeros((B, n, d), f32),
        e_o=jnp.zeros((B, n), f32), gn=jnp.zeros((B,), f32),
        q_prev=jnp.zeros((B,), f32), valid=jnp.zeros((B,), bool))

    def step(st, inp):
      c, X, a, r, first, last, term = inp
      c, X = f32(c), f32(X)
      keep = (~first).astype(f32)
      dsyn = st['dsyn'] * keep[:, None, None]
      e_d = st['e_d'] * keep[:, None, None]
      e_o = st['e_o'] * keep[:, None]
      valid = st['valid'] & ~first
      Wg, Wo = st['Wg'], st['Wo']
      Q, _, _ = qv(Wg, Wo, dsyn, c, X)
      Qa = jnp.take_along_axis(Q, a[:, None], 1)[:, 0]
      delta = r + gamma * (1 - term) * Qa - st['q_prev']
      delta = jnp.where(valid, delta, 0.0)
      tn = (e_o ** 2).sum(-1) + (e_d ** 2).sum((1, 2))
      denom = jnp.where(trace_norm > 0, tn, st['gn']) + eps
      dsyn = dsyn + (beta * delta / denom)[:, None, None] * e_d
      dWo = (eta * delta / denom)[:, None] * e_o
      dWo = jnp.where(env_mean > 0, dWo.mean(0), dWo.sum(0))
      Wo = Wo + dWo
      # episode end: transfer
      Wg = Wg + jnp.einsum('b,bnd->nd', last.astype(f32), dsyn)
      # gradient at (s, a) with updated weights
      Q, u, s = qv(Wg, Wo, dsyn, c, X)
      take = lambda x: jnp.take_along_axis(
          x, a.reshape((-1,) + (1,) * (x.ndim - 1)), 1)[:, 0]
      ua, sa, xa = take(u, ), take(s), take(X)
      g_o = c * sa
      coef = Wo[None] * c * dsilu(ua)
      g_d = coef[:, :, None] * xa[:, None, :]
      gl = gamma * lam
      e_o = gl * e_o + g_o
      e_d = gl * e_d + g_d
      gn = (g_o ** 2).sum(-1) + (coef ** 2).sum(-1) * (xa ** 2).sum(-1)
      st = dict(Wg=Wg, Wo=Wo, dsyn=dsyn, e_d=e_d, e_o=e_o, gn=gn,
                q_prev=take(Q), valid=jnp.ones_like(valid))
      return st, jnp.abs(delta).sum() / jnp.maximum(valid.sum(), 1)

    st, dabs = jax.lax.scan(step, st, data)
    return st['Wg'], st['Wo'], dabs

  def evaluate(Wg, Wo, c, X, a):
    # slow weights only (dsyn = 0)
    u = jnp.einsum('nd,tbd->tbn', Wg, X)
    return jnp.einsum('n,tbn,tbn->tb', Wo, c, jax.nn.silu(u))

  return jax.jit(jax.vmap(run, in_axes=(0, 0, 0, None))), jax.jit(
      jax.vmap(evaluate, in_axes=(0, 0, None, None, None)))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--feats', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--epochs', type=int, default=3)
  ap.add_argument('--gamma', type=float, default=0.99)
  ap.add_argument('--lam', type=float, default=0.9)
  ap.add_argument('--gscale', type=float, default=1.0)
  args = ap.parse_args()
  D = np.load(args.feats)
  c, X = D['c'], D['X']                      # [T,B,n], [T,B,A,d]
  a = D['action'].astype(np.int32)
  rew = D['reward'].astype(np.float32)
  first, last, term = D['is_first'], D['is_last'], D['is_terminal']
  T, B, A, d = X.shape
  n = c.shape[-1]
  G, valid = mc_returns(rew, first, last, term, args.gamma)
  half = B // 2
  tr = slice(0, half); te = slice(half, B)
  print(f'T={T} B={B} A={A} d={d} successes={int((rew > 0).sum())}')

  run, evaluate = make_runner(n, d, A, args.gamma, args.lam, 1e-6)
  grid = list(itertools.product(
      [0.0, 0.03, 0.3],          # beta (fast)
      [0.002, 0.02, 0.2],        # eta (slow)
      [0.0, 1.0],                # trace_norm
      [0.0, 1.0]))               # env_mean
  hp = jnp.asarray(grid, f32)
  key = jax.random.PRNGKey(0)
  Wg0 = args.gscale / np.sqrt(d) * jax.random.normal(key, (n, d), f32)
  Wg = jnp.repeat(Wg0[None], len(grid), 0)
  Wo = jnp.zeros((len(grid), n), f32)
  data = dict(c=c[:, tr], X=X[:, tr, ], a=a[:, tr], r=rew[:, tr],
              first=first[:, tr], last=last[:, tr], term=term[:, tr])
  data = tuple(jnp.asarray(data[k]) for k in
               ('c', 'X', 'a', 'r', 'first', 'last', 'term'))
  # each epoch starts with fresh episodes
  data = (*data[:4], data[4].at[0].set(True), *data[5:])
  dabs_all = []
  results = []
  for ep in range(args.epochs):
    Wg, Wo, dabs = run(hp, Wg, Wo, data)
    dabs_all.append(np.asarray(dabs).mean(1))
    print('epoch', ep, 'done')
  # evaluate
  for split, sl in (('train', tr), ('test', te)):
    nb = sl.stop - sl.start
    Xa = X[:, sl][np.arange(T)[:, None], np.arange(nb)[None], a[:, sl]]
    Qs = np.asarray(evaluate(Wg, Wo, jnp.asarray(c[:, sl], f32),
                             jnp.asarray(Xa, f32), None))
    for i in range(len(grid)):
      if split == 'train':
        results.append(dict(zip(('beta', 'eta', 'trace_norm', 'env_mean'),
                                grid[i])))
      q = Qs[i][valid[:, sl]]
      g = G[:, sl][valid[:, sl]]
      ok = np.isfinite(q).all()
      results[i][f'{split}_r2'] = float(
          1 - ((g - q) ** 2).sum() / ((g - g.mean()) ** 2).sum()) if ok else None
      results[i][f'{split}_corr'] = float(np.corrcoef(q, g)[0, 1]) if (
          ok and q.std() > 0) else None
      results[i][f'{split}_Qmean'] = float(q.mean()) if ok else None
  Wgn = np.linalg.norm(np.asarray(Wg).reshape(len(grid), -1), axis=1)
  Won = np.linalg.norm(np.asarray(Wo), axis=1)
  for i in range(len(grid)):
    results[i]['Wg_norm'] = float(Wgn[i])
    results[i]['Wo_norm'] = float(Won[i])
    results[i]['delta_abs_by_epoch'] = [float(x[i]) for x in dabs_all]
  out = dict(G_mean_test=float(G[:, te][valid[:, te]].mean()),
             Wg0_norm=float(np.linalg.norm(np.asarray(Wg0))),
             results=results)
  pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
  for r in results:
    print({k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in r.items() if k != 'delta_abs_by_epoch'})


if __name__ == '__main__':
  main()
