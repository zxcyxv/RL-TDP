"""Task-boundary consolidation of the shared W_g (no replay).

Each task i leaves its input statistics C_i = mean of x x^T over all candidate
inputs of the steps it acted on (collected online, agent.q.collect_xx), and its
teacher W_i = the W_g at the end of that task. The shared W_g solves

  W C = B,   C = sum_i C_i,   B = sum_i W_i C_i

i.e. the least-squares fit of the unit inputs W_i x of every task on that
task's inputs, with equal weight per task. Directions outside the span of C
keep the newest teacher's values:

  W = W_new + (B - W_new C) C^+

The result is written into a copy of the newest teacher's checkpoint
(q/Wg and its target copy q/Wg_t; everything else unchanged).

python dreamerv3/merge_wg.py --ckpt A.ckpt --xx A.npz --ckpt B.ckpt --xx B.npz \
  --out merged.ckpt
(the last --ckpt is the newest teacher; its checkpoint is the base)
"""
import argparse
import json
import pathlib
import pickle
import shutil

import numpy as np


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', action='append', required=True)
  ap.add_argument('--xx', action='append', required=True)
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  assert len(args.ckpt) == len(args.xx) >= 2

  Ws, Cs, info = [], [], {'tasks': []}
  for ck, xx in zip(args.ckpt, args.xx):
    P = pickle.load(open(pathlib.Path(ck) / 'agent.pkl', 'rb'))['params']
    W = np.asarray(P['q/Wg'], np.float64)
    z = np.load(xx)
    C = z['S'] / z['n']
    Ws.append(W)
    Cs.append(C)
    info['tasks'].append({'ckpt': ck, 'xx': xx, 'n_inputs': float(z['n']),
                          'C_trace': float(np.trace(C))})
  C = sum(Cs)
  B = sum(W @ Ci for W, Ci in zip(Ws, Cs))
  Wn = Ws[-1]
  ev, V = np.linalg.eigh(C)
  d = C.shape[0]
  tol = np.finfo(np.float64).eps * d * ev.max()
  inv = np.where(ev > tol, 1.0 / np.where(ev > tol, ev, 1.0), 0.0)
  Cp = (V * inv) @ V.T
  Wm = Wn + (B - Wn @ C) @ Cp
  info['rank'] = int((ev > tol).sum())
  info['residual_max'] = float(np.abs(Wm @ C - B).max())
  info['dist_rel'] = {f'task{i}': float(np.linalg.norm(Wm - W) / np.linalg.norm(W))
                      for i, W in enumerate(Ws)}
  # unit-input fit error per task: mean ||(Wm - W_i) x||^2 / mean ||W_i x||^2
  info['unit_input_err_rel'] = {
      f'task{i}': float(np.trace((Wm - W) @ Ci @ (Wm - W).T) / np.trace(W @ Ci @ W.T))
      for i, (W, Ci) in enumerate(zip(Ws, Cs))}

  src, out = pathlib.Path(args.ckpt[-1]), pathlib.Path(args.out)
  shutil.copytree(src, out)
  agent = pickle.load(open(out / 'agent.pkl', 'rb'))
  dt = np.asarray(agent['params']['q/Wg']).dtype
  agent['params']['q/Wg'] = Wm.astype(dt)
  if 'q/Wg_t' in agent['params']:
    agent['params']['q/Wg_t'] = Wm.astype(dt)
  pickle.dump(agent, open(out / 'agent.pkl', 'wb'))
  (out / 'merge_info.json').write_text(json.dumps(info, indent=1))
  print(json.dumps(info, indent=1))


if __name__ == '__main__':
  main()
