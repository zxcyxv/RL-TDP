"""Where does the W_g change from new-task learning hit old-task Q?

Loads the agent before new-task learning, runs stored trajectories of the
old task (A) and new task (B) through the frozen world model to get
c = h_t and x(a) = normalized h_{t+1}(a) for every action. The world model
is frozen during new-task learning, so c and x are the same before and after.
With W_o fixed (freeze_wo run), the only change is W_g:

  Q(s, a) = sum_i W_o,i c_i silu(W_g,i . x(a))          (dsyn left out)

Reports, on old-task states:
  - exact change of Q and of the policy softmax(Q / tau), before -> after
  - first-order prediction  dQ ~ sum_i W_o,i c_i silu'(u_i) (dW_g,i . x)
  - per-unit spread of that change (how many units carry it)
and the two overlap factors of a single new-task write on an old-task read:
  unit overlap   sum_i W_o,i^2 c_A,i c_B,i silu'_A,i silu'_B,i (cosine-normed)
  input overlap  x_A . x_B (cosine)
compared with the same quantities within the old task.

python dreamerv3/diag_overlap.py --before <agent ckpt> --after <agent ckpt> \
  --replay_a <old-task replay> --replay_b <new-task replay> --out <json>
"""
import argparse
import json
import pathlib
import pickle
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
from .diag_features import load_streams


def features(agent, data):
  T, B = data['reward'].shape
  carry = agent.init_policy(B)
  cs, Xs = [], []
  for t in range(T):
    obs = {k: data[k][t] for k in
           ('image', 'reward', 'is_first', 'is_last', 'is_terminal')}
    with jax.transfer_guard('allow'):
      if t > 0:
        prev = data['action'][t - 1].astype(np.int32)
        carry = (*carry[:3], {'action': [jax.numpy.asarray(a) for a in prev]})
      carry, acts, outs = agent.policy(carry, obs)
    cs.append(np.asarray(outs['qdbg/c'], np.float32))
    Xs.append(np.asarray(outs['qdbg/Xn'], np.float32))
  return np.stack(cs), np.stack(Xs)  # [T, B, n], [T, B, A, d]


def silu(u):
  return u / (1 + np.exp(-u))


def dsilu(u):
  s = 1 / (1 + np.exp(-u))
  return s * (1 + u * (1 - s))


def qvals(Wo, Wg, c, X):
  u = np.einsum('nd,sad->san', Wg, X)
  return np.einsum('n,sn,san->sa', Wo, c, silu(u)), u


def softmax(z):
  z = z - z.max(-1, keepdims=True)
  e = np.exp(z)
  return e / e.sum(-1, keepdims=True)


def main(argv):
  ap = argparse.ArgumentParser()
  ap.add_argument('--before', required=True)
  ap.add_argument('--after', required=True)
  ap.add_argument('--replay_a', required=True)
  ap.add_argument('--replay_b', required=True)
  ap.add_argument('--steps', type=int, default=1500)
  ap.add_argument('--streams', type=int, default=16)
  ap.add_argument('--tail', type=int, default=10)
  ap.add_argument('--out', default='')
  args, rest = ap.parse_known_args(argv)

  configs = yaml.YAML(typ='safe').load(
      elements.Path(folder / 'configs.yaml').read())
  config = elements.Config(configs['defaults']).update(configs['size1m'])
  config = elements.Flags(config).parse([
      '--task', 'navix_DoorKey-8x8-v0', '--agent_type', 'q',
      '--agent.dyn.rssm.unroll', 'True', '--logdir', '/tmp/claude-0/diag_overlap',
      '--agent.q.debug_outs', 'True', '--agent.q.beta', '0',
      '--agent.q.eta', '0', '--agent.q.frozen', 'True',
      '--run.envs', str(args.streams), *rest])
  elements.Path(config.logdir).mkdir()
  agent = mainmod.make_agent(config)
  elements.checkpoint.load(args.before, dict(agent=agent.load))

  P0 = pickle.load(open(pathlib.Path(args.before) / 'agent.pkl', 'rb'))['params']
  P1 = pickle.load(open(pathlib.Path(args.after) / 'agent.pkl', 'rb'))['params']
  Wo0, Wg0 = np.asarray(P0['q/Wo'], np.float64), np.asarray(P0['q/Wg'], np.float64)
  Wo1, Wg1 = np.asarray(P1['q/Wo'], np.float64), np.asarray(P1['q/Wg'], np.float64)
  res = {'Wo_change_rel': float(np.linalg.norm(Wo1 - Wo0) / np.linalg.norm(Wo0)),
         'Wg_change_rel': float(np.linalg.norm(Wg1 - Wg0) / np.linalg.norm(Wg0))}

  out = {}
  for name, rdir in (('A', args.replay_a), ('B', args.replay_b)):
    data = load_streams(rdir, args.streams, args.steps)
    c, X = features(agent, data)
    T, B = c.shape[:2]
    # states near the end of successful episodes (old task: after the door)
    near = np.zeros((T, B), bool)
    for b in range(B):
      for t in np.where(data['reward'][:, b] > 0)[0]:
        near[max(0, t - args.tail):t, b] = True
    valid = np.ones((T, B), bool)
    out[name] = dict(c=c.reshape(T * B, -1), X=X.reshape(T * B, *X.shape[2:]),
                     a=data['action'].reshape(-1).astype(int),
                     near=near.reshape(-1), succ=int((data['reward'] > 0).sum()))
    res[f'{name}_states'] = int(T * B)
    res[f'{name}_successes'] = out[name]['succ']

  A, Bd = out['A'], out['B']
  Q0, u0 = qvals(Wo0, Wg0, A['c'], A['X'])
  Q1, _ = qvals(Wo1, Wg1, A['c'], A['X'])
  dQ = Q1 - Q0
  # first-order prediction from the W_g change
  dWgx = np.einsum('nd,sad->san', Wg1 - Wg0, A['X'])
  contrib = Wo0[None, None] * A['c'][:, None] * dsilu(u0) * dWgx  # [S, A, n]
  dQ_lin = contrib.sum(-1)
  # how many units carry the change: participation ratio of |contrib|
  cabs = np.abs(contrib).reshape(-1, contrib.shape[-1])
  pr = (cabs.sum(-1) ** 2) / ((cabs ** 2).sum(-1) + 1e-30)
  # policy change, tau = across-action std of Q (as tau_rel = 1)
  tau = Q0.std(-1).mean()
  p0, p1 = softmax(Q0 / tau), softmax(Q1 / tau)
  kl = (p0 * (np.log(p0 + 1e-12) - np.log(p1 + 1e-12))).sum(-1)
  adv0 = Q0 - Q0.mean(-1, keepdims=True)
  dadv = dQ - dQ.mean(-1, keepdims=True)
  for tag, m in (('all', np.ones(len(dQ), bool)), ('near_goal', A['near'])):
    res[f'oldtask_{tag}'] = {
        'n': int(m.sum()),
        'Q_before_mean': float(Q0[m].mean()),
        'dQ_mean': float(dQ[m].mean()),
        'dQ_abs_mean': float(np.abs(dQ[m]).mean()),
        'action_gap_before_abs_mean': float(np.abs(adv0[m]).mean()),
        'action_gap_change_abs_mean': float(np.abs(dadv[m]).mean()),
        'policy_KL_mean': float(kl[m].mean()),
        'argmax_changed_frac': float((Q0[m].argmax(-1) != Q1[m].argmax(-1)).mean()),
        'first_order_corr': float(np.corrcoef(dQ[m].ravel(), dQ_lin[m].ravel())[0, 1]),
        'units_participation_ratio_mean': float(pr.reshape(len(dQ), -1)[m].mean()),
    }

  # overlap factors between single reads/writes (taken actions)
  rng = np.random.default_rng(0)
  def pick(d, k):
    i = rng.integers(0, len(d['c']), k)
    xa = d['X'][i, d['a'][i]]
    c = d['c'][i]
    u = np.einsum('nd,sd->sn', Wg0, xa)
    g = Wo0[None] * c * dsilu(u)          # per-unit read/write weight
    return g, xa
  gA, xA = pick(A, 4000)
  gA2, xA2 = pick(A, 4000)
  gB, xB = pick(Bd, 4000)
  def cos(p, q):
    return (p * q).sum(-1) / (np.linalg.norm(p, axis=-1) * np.linalg.norm(q, axis=-1) + 1e-12)
  res['overlap'] = {
      'unit_cos_old_new': float(cos(gA, gB).mean()),
      'unit_cos_old_old': float(cos(gA, gA2).mean()),
      'input_cos_old_new': float(cos(xA, xB).mean()),
      'input_cos_old_old': float(cos(xA, xA2).mean()),
      'c_active_frac_abs_gt_0.1rms': float((np.abs(A['c']) > 0.1 * np.sqrt(
          (A['c'] ** 2).mean(-1, keepdims=True))).mean()),
  }
  print(json.dumps(res, indent=1))
  if args.out:
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
  main(sys.argv[1:])
