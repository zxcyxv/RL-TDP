"""채집 격자 (JAX): 보상이 반복되고 배치가 에피소드마다 바뀌는 픽셀 과제. 설계: docs/world/model_design.md 7절, 9절.

- N×N 격자의 칸마다 무늬 타일 하나 (n_types종, 에피소드마다 무작위 배치). 격자 밖은 벽 타일.
- 관측: 에이전트 중심 V×V 타일을 그린 그림을 타일 단위 패치로 준다: [V*V, P*P*3]. Craftax와 같은 꼴 (타일 = 7×7 픽셀).
- 행동 4개 (이동). 보상 칸은 겉보기로는 다른 칸과 구별되지 않는다. 들어가면 +1을 받고 출발 칸으로 돌아간다.
- 출발 칸과 보상 칸은 에피소드마다 무작위. 그래서 "어디서 보상을 받는가"는 그 에피소드 안에서 알아내야 한다.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class EnvCfg(NamedTuple):
    N: int = 7
    V: int = 7
    n_types: int = 16
    P: int = 7
    T: int = 256
    A: int = 4


class State(NamedTuple):
    tiles: jnp.ndarray
    pos: jnp.ndarray
    start: jnp.ndarray
    goal: jnp.ndarray


MOVES = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])


def textures(cfg: EnvCfg, seed=0):
    """타일 무늬 [n_types + 1, P, P, 3]. 0번은 벽 (어두움). 학습과 평가에서 같은 무늬를 쓴다."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    base = jax.random.uniform(k1, (cfg.n_types + 1, 1, 1, 3))
    tex = 0.6 * base + 0.4 * jax.random.uniform(k2, (cfg.n_types + 1, cfg.P, cfg.P, 3))
    return tex.at[0].set(0.05)


def reset(key, cfg: EnvCfg):
    k1, k2, k3 = jax.random.split(key, 3)
    tiles = jax.random.randint(k1, (cfg.N, cfg.N), 1, cfg.n_types + 1)
    s = jax.random.randint(k2, (), 0, cfg.N * cfg.N)
    g = jax.random.randint(k3, (), 0, cfg.N * cfg.N - 1)
    g = jnp.where(g >= s, g + 1, g)
    start, goal = jnp.stack([s // cfg.N, s % cfg.N]), jnp.stack([g // cfg.N, g % cfg.N])
    return State(tiles, start, start, goal)


def patches(state: State, tex, cfg: EnvCfg):
    r = cfg.V // 2
    ids = jax.lax.dynamic_slice(jnp.pad(state.tiles, r), (state.pos[0], state.pos[1]), (cfg.V, cfg.V))
    return tex[ids].reshape(cfg.V * cfg.V, -1)


def step(state: State, a, cfg: EnvCfg):
    new = jnp.clip(state.pos + MOVES[a], 0, cfg.N - 1)
    r = jnp.all(new == state.goal)
    return state._replace(pos=jnp.where(r, state.start, new)), r.astype(jnp.float32)


def summarize(rew, dist):
    """rew [B, T] (0/1), dist [B] (출발→보상 최단 거리). 에피소드 안의 적응을 요약한다."""
    B, T = rew.shape
    n = rew.sum(1)
    t = jnp.arange(T)[None]
    first = jnp.where(rew > 0, t, T).min(1)                       # 첫 보상 스텝 (없으면 T)
    last = jnp.where(rew > 0, t, -1).max(1)
    multi = n >= 2
    lap = (last - first) / jnp.maximum(n - 1, 1)                  # 첫 보상 뒤, 보상 사이 평균 스텝
    return dict(rewards=n.mean(), found=(n > 0).mean(), first=jnp.where(n > 0, first + 1, 0).sum() / jnp.maximum((n > 0).sum(), 1),
                lap=jnp.where(multi, lap, 0).sum() / jnp.maximum(multi.sum(), 1),
                shortest=jnp.where(multi, dist, 0).sum() / jnp.maximum(multi.sum(), 1), multi=multi.mean())
