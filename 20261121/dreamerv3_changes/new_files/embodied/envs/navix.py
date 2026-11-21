import elements
import embodied
import numpy as np


class Navix(embodied.Env):
  """NAVIX (JAX MiniGrid) task, e.g. task='DoorKey-8x8-v0'.

  Observation: rgb_first_person (7x7 tiles of 8x8 px = 56x56x3), zero-padded
  to `size` so the DreamerV3 CNN sees a power-of-two resolution. The NAVIX
  step runs on the CPU so env worker processes do not touch the GPU.
  """

  def __init__(self, task, size=(64, 64), seed=None):
    import multiprocessing
    import jax
    if multiprocessing.parent_process() is not None:
      # Env worker (spawned by the driver): keep it off the GPU entirely, so
      # it creates no GPU client and preallocates no GPU memory. In the main
      # process the agent uses the GPU; there we only pin NAVIX to the CPU.
      jax.config.update('jax_platforms', 'cpu')
    self._jax = jax
    self._cpu = jax.devices('cpu')[0]
    with jax.default_device(self._cpu), jax.transfer_guard('allow'):
      import navix as nx
      self._env = nx.make(
          f'Navix-{task}',
          observation_fn=nx.observations.rgb_first_person)
      self._reset = jax.jit(self._env.reset)
      self._step = jax.jit(self._env.step)
      self._key = jax.random.PRNGKey(0 if seed is None else seed)
    self._size = tuple(size)
    self._nact = int(self._env.action_space.maximum) + 1
    self._timestep = None
    self._done = True

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, (*self._size, 3)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._nact),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    jax = self._jax
    with jax.default_device(self._cpu), jax.transfer_guard('allow'):
      if action['reset'] or self._done:
        self._key, key = jax.random.split(self._key)
        self._timestep = self._reset(key)
        self._done = False
        return self._obs(0.0, is_first=True)
      self._timestep = self._step(
          self._timestep, np.int32(action['action']))
      step_type = int(self._timestep.step_type)  # 0 transition, 1 trunc, 2 term
      reward = float(self._timestep.reward)
    self._done = step_type != 0
    return self._obs(
        reward,
        is_last=self._done, is_terminal=(step_type == 2))

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    with self._jax.transfer_guard('allow'):
      img = np.asarray(self._timestep.observation, np.uint8)
    H, W = self._size
    h, w = img.shape[:2]
    top, left = (H - h) // 2, (W - w) // 2
    image = np.zeros((H, W, 3), np.uint8)
    image[top:top + h, left:left + w] = img
    return dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
