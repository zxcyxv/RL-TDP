"""Precision matrix over old-task Q-path inputs, for precision-weighted writes.

C = E[x x^T] over old-task states and all action candidates (x = normalized
h_{t+1}(a), what W_g reads). P = alpha (C + alpha I)^-1 shrinks writes along
directions the old task uses. For each alpha, reports on held-out pairs:
  protect   mean |x_A^T P x_B| / mean |x_A^T x_B|  (old read of a new write)
  keep_new  mean x_B^T P x_B / mean x_B^T x_B      (new read of a new write)
and saves P for the chosen alpha.

python dreamerv3/diag_precision.py --ckpt <agent ckpt> --replay_a <old> \
  --replay_b <new> --alphas 0.001,0.01,0.1,1 --save_alpha 0.01 --save <P.npy>
"""
import argparse
import json
import pathlib
import sys

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import numpy as np
import ruamel.yaml as yaml

from . import main as mainmod
from .diag_features import load_streams
from .diag_overlap import features


def main(argv):
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--replay_a', required=True)
  ap.add_argument('--replay_b', required=True)
  ap.add_argument('--steps', type=int, default=1500)
  ap.add_argument('--streams', type=int, default=16)
  ap.add_argument('--alphas', default='0.001,0.01,0.1,1')
  ap.add_argument('--save_alpha', type=float, default=0.0)
  ap.add_argument('--save', default='')
  ap.add_argument('--out', default='')
  args, rest = ap.parse_known_args(argv)

  configs = yaml.YAML(typ='safe').load(
      elements.Path(folder / 'configs.yaml').read())
  config = elements.Config(configs['defaults']).update(configs['size1m'])
  config = elements.Flags(config).parse([
      '--task', 'navix_DoorKey-8x8-v0', '--agent_type', 'q',
      '--agent.dyn.rssm.unroll', 'True', '--logdir', '/tmp/claude-0/diag_precision',
      '--agent.q.debug_outs', 'True', '--agent.q.beta', '0',
      '--agent.q.eta', '0', '--agent.q.frozen', 'True',
      '--run.envs', str(args.streams), *rest])
  elements.Path(config.logdir).mkdir()
  agent = mainmod.make_agent(config)
  elements.checkpoint.load(args.ckpt, dict(agent=agent.load))

  XA = features(agent, load_streams(args.replay_a, args.streams, args.steps))[1]
  XB = features(agent, load_streams(args.replay_b, args.streams, args.steps))[1]
  XA = XA.reshape(-1, XA.shape[-1]).astype(np.float64)   # all actions
  XB = XB.reshape(-1, XB.shape[-1]).astype(np.float64)
  d = XA.shape[1]
  rng = np.random.default_rng(0)
  perm = rng.permutation(len(XA))
  fit, held = XA[perm[: len(XA) // 2]], XA[perm[len(XA) // 2:]]
  C = fit.T @ fit / len(fit)
  ev = np.linalg.eigvalsh(C)[::-1]
  res = {'d': d, 'nA': len(XA), 'nB': len(XB),
         'C_trace': float(ev.sum()),
         'eig_top': [float(v) for v in ev[:8]],
         'eig_frac_top10': float(ev[:10].sum() / ev.sum()),
         'eig_frac_top50': float(ev[:50].sum() / ev.sum()),
         'participation_dim': float(ev.sum() ** 2 / (ev ** 2).sum())}
  i = rng.integers(0, len(held), 20000)
  j = rng.integers(0, len(XB), 20000)
  xa, xb = held[i], XB[j]
  base_ab = np.abs((xa * xb).sum(-1)).mean()
  base_bb = (xb * xb).sum(-1).mean()
  rows = []
  for a in [float(s) for s in args.alphas.split(',')]:
    P = a * np.linalg.inv(C + a * np.eye(d))
    Pxb = xb @ P
    rows.append({'alpha': a,
                 'protect': float(np.abs((xa * Pxb).sum(-1)).mean() / base_ab),
                 'keep_new': float((xb * Pxb).sum(-1).mean() / base_bb)})
  res['alphas'] = rows
  print(json.dumps(res, indent=1))
  if args.out:
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
  if args.save and args.save_alpha:
    P = args.save_alpha * np.linalg.inv(C + args.save_alpha * np.eye(d))
    np.save(args.save, P.astype(np.float32))
    print('saved', args.save)


if __name__ == '__main__':
  main(sys.argv[1:])
