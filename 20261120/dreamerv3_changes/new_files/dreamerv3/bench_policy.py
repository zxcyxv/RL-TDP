"""Times steady-state policy calls (no env, no training).

python dreamerv3/bench_policy.py --configs size1m --task navix_DoorKey-8x8-v0 \
    --agent_type q --logdir /tmp/bench
"""
import pathlib
import sys
import time

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import numpy as np
import ruamel.yaml as yaml

from . import main as mainmod


def run(argv):
  configs = yaml.YAML(typ='safe').load(
      elements.Path(folder / 'configs.yaml').read())
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  elements.Path(config.logdir).mkdir()
  agent = mainmod.make_agent(config)
  B = config.run.envs
  rng = np.random.default_rng(0)
  carry = agent.init_policy(B)

  def obs(i):
    return {
        'image': rng.integers(0, 255, (B, 64, 64, 3), np.uint8),
        'reward': np.zeros(B, np.float32),
        'is_first': np.full(B, i == 0),
        'is_last': (np.arange(B) == i % B) & (i > 0),
        'is_terminal': np.zeros(B, bool),
    }

  for i in range(30):
    carry, acts, outs = agent.policy(carry, obs(i))
  N = 300
  t = time.perf_counter()
  for i in range(30, 30 + N):
    carry, acts, outs = agent.policy(carry, obs(i))
  dt = (time.perf_counter() - t) / N
  print(f'agent_type={config.agent_type} envs={B}: '
        f'{dt * 1e3:.2f} ms per policy call, {B / dt:.0f} env steps/s')

  # Train step.
  import jax
  BT, T = config.batch_size, config.batch_length + config.replay_context
  spaces = agent.spaces
  data = {k: np.zeros((BT, T, *v.shape), v.dtype) for k, v in spaces.items()}
  data['is_first'][:, 0] = True
  stream = iter(agent.stream(iter([dict(data) for _ in range(200)])))
  tcarry = agent.init_train(BT)
  for _ in range(10):
    tcarry, _, mets = agent.train(tcarry, next(stream))
  jax.block_until_ready(agent.params)
  t = time.perf_counter()
  for _ in range(40):
    tcarry, _, mets = agent.train(tcarry, next(stream))
  jax.block_until_ready(agent.params)
  dt = (time.perf_counter() - t) / 40
  print(f'agent_type={config.agent_type}: {dt * 1e3:.2f} ms per train call')

  import os
  tdir = os.environ.get('TRACE_DIR')
  if tdir:
    jax.profiler.start_trace(tdir, create_perfetto_trace=True)
    for _ in range(10):
      tcarry, _, mets = agent.train(tcarry, next(stream))
    jax.block_until_ready(agent.params)
    jax.profiler.stop_trace()


if __name__ == '__main__':
  run(sys.argv[1:])
