"""Reference fit: MLP trained by backprop on the same dumped features.

Gives the held-out R^2 that a nonlinear readout of the world-model state
reaches on the Monte-Carlo returns, as a reference point for the ridge and
local-rule numbers. Same split as sweep_offline.py (first half of streams
train, second half test).

python dreamerv3/ref_mlp.py --feats <npz> --out <json>
"""
import argparse
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .sweep_offline import mc_returns


def fit(Xtr, ytr, Xte, yte, steps=4000, width=256, lr=1e-3, seed=0):
  mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
  Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
  k = jax.random.split(jax.random.PRNGKey(seed), 3)
  d = Xtr.shape[1]
  p = dict(
      W1=jax.random.normal(k[0], (d, width)) / np.sqrt(d), b1=jnp.zeros(width),
      W2=jax.random.normal(k[1], (width, width)) / np.sqrt(width),
      b2=jnp.zeros(width), W3=jnp.zeros((width, 1)), b3=jnp.zeros(1))
  f = lambda p, x: (jax.nn.silu(jax.nn.silu(x @ p['W1'] + p['b1'])
                                @ p['W2'] + p['b2']) @ p['W3'] + p['b3'])[:, 0]
  opt = optax.adamw(lr, weight_decay=1e-4)
  st = opt.init(p)
  Xtr, ytr = jnp.asarray(Xtr, jnp.float32), jnp.asarray(ytr, jnp.float32)

  @jax.jit
  def step(p, st, i):
    idx = jax.random.randint(jax.random.fold_in(k[2], i), (1024,), 0, len(Xtr))
    loss, g = jax.value_and_grad(
        lambda p: ((f(p, Xtr[idx]) - ytr[idx]) ** 2).mean())(p)
    u, st = opt.update(g, st, p)
    return optax.apply_updates(p, u), st, loss

  for i in range(steps):
    p, st, _ = step(p, st, i)
  r2 = lambda X, y: float(1 - ((np.asarray(f(p, jnp.asarray(X, jnp.float32)))
                                - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
  return r2(Xtr, np.asarray(ytr)), r2(Xte, yte)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--feats', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--gamma', type=float, default=0.99)
  args = ap.parse_args()
  D = np.load(args.feats)
  c, X = D['c'].astype(np.float32), D['X']
  a = D['action'].astype(np.int64)
  T, B, A, d = X.shape
  Xa = X[np.arange(T)[:, None], np.arange(B)[None], a].astype(np.float32)
  G, valid = mc_returns(D['reward'], D['is_first'], D['is_last'],
                        D['is_terminal'], args.gamma)
  half = B // 2
  tr = np.zeros((T, B), bool); tr[:, :half] = True
  res = {}
  for name, F in (('c', c), ('c_xa', np.concatenate([c, Xa], -1))):
    r_tr, r_te = fit(F[valid & tr], G[valid & tr], F[valid & ~tr],
                     G[valid & ~tr])
    res[f'mlp_{name}_train_r2'] = r_tr
    res[f'mlp_{name}_test_r2'] = r_te
    print(name, r_tr, r_te)
  pathlib.Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
  main()
