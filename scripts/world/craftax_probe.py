"""Craftax-Classic (symbolic) 점검: 관측 형식, 지도 크기 변경, 속도, 무작위 정책의 재방문 빈도.

/venv/jax/bin/python scripts/world/craftax_probe.py [지도 한 변 ...]
"""
import sys
import time

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import OBS_DIM, BlockType
from craftax.craftax_classic.envs.craftax_state import StaticEnvParams
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv

B, T = 1024, 256
for size in [int(x) for x in sys.argv[1:]] or [64, 24, 16]:
    env = CraftaxClassicSymbolicEnv(StaticEnvParams(map_size=(size, size)))
    params = env.default_params

    def rollout(key):
        k0, key = jax.random.split(key)
        obs, state = env.reset(k0, params)

        def step(carry, k):
            state, _ = carry
            ka, ks = jax.random.split(k)
            a = jax.random.randint(ka, (), 0, 5)                       # 이동만 (NOOP + 네 방향)
            obs, state, r, done, _ = env.step(ks, state, a, params)
            return (state, obs), (state.player_position, done, r)

        _, out = jax.lax.scan(step, (state, obs), jax.random.split(key, T))
        return obs, out

    f = jax.jit(jax.vmap(rollout))
    obs, (pos, done, rew) = f(jax.random.split(jax.random.PRNGKey(0), B))
    t = time.time()
    obs, (pos, done, rew) = f(jax.random.split(jax.random.PRNGKey(1), B))
    pos.block_until_ready()
    dt = time.time() - t
    # 재방문: 같은 칸에 다시 서는 스텝의 비율, 그중 마지막 방문이 64스텝보다 전인 비율
    cell = pos[..., 0] * size + pos[..., 1]                                # [B, T]
    same = cell[:, :, None] == cell[:, None, :]                              # [B, t, s]
    earlier = jnp.tril(jnp.ones((T, T), bool), k=-1)[None]
    gap = jnp.where(same & earlier, jnp.arange(T)[None, :, None] - jnp.arange(T)[None, None, :], T + 1).min(-1)
    moved = jnp.concatenate([jnp.ones((B, 1), bool), cell[:, 1:] != cell[:, :-1]], 1)
    rev = (gap <= T) & moved
    print(f"지도 {size}x{size}: 관측 {obs.shape[1:]} (시야 {OBS_DIM}, 블록 {len(BlockType)}종), "
          f"{B * T / dt / 1e6:.2f}M 스텝/초, 에피소드 종료 {float(done.mean()) * T:.2f}회/256스텝, "
          f"이동 중 재방문 {float(rev.sum() / moved.sum()):.2f}, 그중 64스텝 넘은 것 {float((rev & (gap > 64)).sum() / rev.sum()):.2f}")
