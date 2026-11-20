"""Linear fit of Monte-Carlo returns from Q-path features (diagnostic).

Loads a Q-agent checkpoint, replays stored trajectories through the world
model with the recorded actions, and fits ridge regressions of the discounted
return G_t (from recorded rewards) on
  K   = c * silu(W_g x_hat(a_t))   (Q-path features of the taken action)
  Xn  = x_hat(a_t)                 (normalized h_{t+1}(a_t))
  c   = h_t
Reports held-out R^2 (split by stream) and the fit of the agent's own Q.

python dreamerv3/diag_features.py --ckpt <ckpt dir> --replay <replay dir> \
    --steps 4000 --out <json>
"""
import argparse
import collections
import json
import pathlib
import sys

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import jax
import numpy as np
import ruamel.yaml as yaml

from . import main as mainmod


def load_streams(replay_dir, nstreams, steps):
  files = sorted(pathlib.Path(replay_dir).glob('*.npz'))
  by_uuid, succ_of = {}, {}
  for f in files:
    time, uuid, succ, length = f.stem.split('-')
    by_uuid[uuid] = f
    succ_of[uuid] = succ
  preds = set(succ_of.values())
  heads = [u for u in by_uuid if u not in preds]
  chains = []
  for h in heads:
    chain, u = [], h
    while u in by_uuid:
      chain.append(u)
      u = succ_of[u]
    chains.append(chain)
  chains.sort(key=len, reverse=True)
  keys = ('image', 'reward', 'is_first', 'is_last', 'is_terminal', 'action')
  out = []
  need = steps + 600
  for chain in chains[:nstreams]:
    parts, total = [], 0
    for u in reversed(chain):  # most recent data
      d = np.load(by_uuid[u])
      parts.insert(0, {k: d[k] for k in keys})
      total += len(d['reward'])
      if total >= need:
        break
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    start = int(np.argmax(data['is_first']))
    data = {k: v[start:start + steps] for k, v in data.items()}
    out.append(data)
  n = min(len(d['reward']) for d in out)
  return {k: np.stack([d[k][:n] for d in out], 1) for k in keys}  # [T, B]


def returns(rew, first, last, term, gamma):
  """G_t for the transition t -> t+1; valid only if the episode ends in the
  window. Truncated episodes bootstrap with 0."""
  T, B = rew.shape
  G = np.zeros((T, B))
  valid = np.zeros((T, B), bool)
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


def ridge_r2(Xtr, ytr, Xte, yte, lam=1e-2):
  mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
  A, B = (Xtr - mu) / sd, (Xte - mu) / sd
  ym = ytr.mean()
  w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (ytr - ym))
  pred = B @ w + ym
  return 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()


def main(argv):
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--replay', required=True)
  ap.add_argument('--steps', type=int, default=4000)
  ap.add_argument('--streams', type=int, default=16)
  ap.add_argument('--out', default='')
  ap.add_argument('--dump', default='')
  args, rest = ap.parse_known_args(argv)

  configs = yaml.YAML(typ='safe').load(
      elements.Path(folder / 'configs.yaml').read())
  config = elements.Config(configs['defaults']).update(configs['size1m'])
  config = elements.Flags(config).parse([
      '--task', 'navix_DoorKey-8x8-v0', '--agent_type', 'q',
      '--agent.dyn.rssm.unroll', 'True', '--logdir', '/tmp/claude-0/diag',
      '--agent.q.debug_outs', 'True', '--agent.q.beta', '0',
      '--agent.q.eta', '0', '--run.envs', str(args.streams), *rest])
  elements.Path(config.logdir).mkdir()
  agent = mainmod.make_agent(config)
  elements.checkpoint.load(args.ckpt, dict(agent=agent.load))

  data = load_streams(args.replay, args.streams, args.steps)
  T, B = data['reward'].shape
  print('streams', B, 'steps', T, 'successes',
        int((data['reward'] > 0).sum()))
  carry = agent.init_policy(B)
  feats = collections.defaultdict(list)
  feats_all = []
  for t in range(T):
    obs = {k: data[k][t] for k in
           ('image', 'reward', 'is_first', 'is_last', 'is_terminal')}
    with jax.transfer_guard('allow'):
      if t > 0:
        prev = data['action'][t - 1].astype(np.int32)
        carry = (*carry[:3], {'action': [jax.numpy.asarray(a) for a in prev]})
      carry, acts, outs = agent.policy(carry, obs)
    a = data['action'][t].astype(np.int64)
    idx = np.arange(B)
    feats['K'].append(outs['qdbg/K'][idx, a])
    feats['Xn'].append(outs['qdbg/Xn'][idx, a])
    feats['c'].append(outs['qdbg/c'])
    feats['Q'].append(outs['qdbg/Q'][idx, a])
    if args.dump:
      feats_all.append(outs['qdbg/Xn'])  # [B, A, d]
  feats = {k: np.stack(v) for k, v in feats.items()}  # [T, B, ...]
  if args.dump:
    np.savez(args.dump, c=feats['c'].astype(np.float16),
             X=np.stack(feats_all).astype(np.float16),
             **{k: data[k] for k in data if k != 'image'})
    print('dumped', args.dump)

  G, valid = returns(data['reward'], data['is_first'], data['is_last'],
                     data['is_terminal'], config.agent.q.gamma)
  half = B // 2
  tr = np.zeros((T, B), bool); tr[:, :half] = True
  res = {'streams': B, 'steps': T,
         'successes': int((data['reward'] > 0).sum()),
         'G_mean': float(G[valid].mean()), 'G_nonzero_frac': float((G[valid] > 0).mean())}
  for k in ('K', 'Xn', 'c'):
    X = feats[k].astype(np.float64)
    res[f'r2_{k}'] = float(ridge_r2(
        X[valid & tr], G[valid & tr], X[valid & ~tr], G[valid & ~tr]))
  Qa = feats['Q'][valid]
  res['corr_Q_G'] = float(np.corrcoef(Qa, G[valid])[0, 1])
  res['Q_mean'] = float(Qa.mean())
  # feature similarity across states (cosine of K between random pairs)
  K = feats['K'][valid]
  rng = np.random.default_rng(0)
  i, j = rng.integers(0, len(K), (2, 5000))
  cos = (K[i] * K[j]).sum(-1) / (np.linalg.norm(K[i], axis=-1) * np.linalg.norm(K[j], axis=-1) + 1e-12)
  res['K_cos_random_pairs_mean'] = float(cos.mean())
  print(json.dumps(res, indent=1))
  if args.out:
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
  main(sys.argv[1:])
