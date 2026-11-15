"""갈림길 과제: 한 번의 지연된 세계 결과 → 긴 대기 → 첫 재사용 (JAX). 설계: docs/world/plasticity_design.md 7.2절, 1단계 단순화는 docs/world/README.md 6절.

세계 하나의 흐름 (행동은 정해진 경로로 주어진다. 공통 경험 = 외생 경로의 진단):
  두 출입구 (왼쪽 LEFT / 오른쪽 RIGHT)를 무작위 순서로 한 번씩 지난다.
    갈림길 장면 (HUD 빈칸)에서 출입구 행동 → 복도 장면 d−1번 (FWD) → 끝에서 자원 장면 (가운데 칸 = 그 출입구의 자원 종류).
    복도는 두 출입구가 똑같다 (깊이마다 장면은 다르다). 그래서 자원을 출입구에 잇는 것은 d스텝 전의 (갈림길, 행동)뿐이다.
    첫 자원 장면 바로 다음이 둘째 출입구의 갈림길이다 (사건 뒤 대조: 대칭 창이면 둘째 출입구 키가 첫 자원과 묶인다).
  대기: 가려진 방 장면에서 D스텝 (STAY). 자원은 보이지 않는다.
  검사: 갈림길 장면, HUD 칸에 필요한 자원 종류. 첫 선택이 그 자원이 있는 출입구인지 잰다.
자원 종류 4개 중 2개를 세계마다 무작위로 두 출입구에 배정한다. 필요한 것은 그 둘 중 하나 (균등).
관측은 forage와 같은 꼴: 에이전트 중심 7×7 타일, 타일 하나 = 7×7×3 픽셀 패치 → [49, 147].
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import env as E

FWD, STAY, LEFT, RIGHT = 0, 1, 2, 3
A = 4
JUNC, CORR, ROOM, HUD_BLANK = 1, 2, 3, 9
RES0, N_RES = 5, 4                       # 자원 종류 = 타일 5, 6, 7, 8
HUD, CENTER = 0, 24                      # 패치 번호: HUD = 왼쪽 위 칸, 가운데 칸
ENV = E.EnvCfg()                         # V = 7, P = 7, n_types = 16 (타일 무늬는 forage와 같다)
TEX = E.textures(ENV)


def _backgrounds():
    """장면 종류마다 고정된 배경 (모든 세계가 공유. 느린 모델이 배울 수 있는 지형). 타일 10~16."""
    rng = np.random.default_rng(123)
    bg = {s: rng.integers(10, 17, size=(7, 7)) for s in ("junc", "room")}
    for i in range(16):                                              # 복도는 깊이마다 보이는 것이 다르다 (두 출입구는 같다)
        bg[f"corr{i}"] = rng.integers(10, 17, size=(7, 7))
    return bg


BG = {k: jnp.array(v) for k, v in _backgrounds().items()}


def scene(bg, center, hud):
    return bg.at[3, 3].set(center).at[0, 0].set(hud)


class World(NamedTuple):
    ids: jnp.ndarray       # [T, 7, 7] 매 스텝 관측의 타일 번호 (마지막 = 검사 장면)
    act: jnp.ndarray       # [T − 1] 주어진 행동
    need: jnp.ndarray      # 필요한 자원 종류 (타일 번호)
    answer: jnp.ndarray    # 정답 행동 (LEFT / RIGHT)


def make_world(key, d, D, gap=0):
    """d = 출입구 행동에서 자원이 보일 때까지의 스텝, D = 대기 스텝, gap = 두 노출 사이의 방 스텝 (진단용, 기본 0). 모두 정적."""
    k1, k2, k3 = jax.random.split(key, 3)
    types = jax.random.permutation(k1, N_RES)[:2] + RES0           # [왼쪽 자원, 오른쪽 자원]
    first = jax.random.randint(k2, (), 0, 2)                         # 0: 왼쪽 먼저
    which = jax.random.randint(k3, (), 0, 2)                         # 필요한 것이 왼쪽 자원인가 (0) 오른쪽인가 (1)
    J = scene(BG["junc"], JUNC, HUD_BLANK)
    R = scene(BG["room"], ROOM, HUD_BLANK)
    ids, act = [], []
    for i in range(2):
        door = jnp.where(i == first, 0, 1)                           # 이번에 지나는 출입구 (0 왼쪽, 1 오른쪽)
        ids.append(J); act.append(jnp.where(door == 0, LEFT, RIGHT))
        for j in range(d - 1):
            ids.append(scene(BG[f"corr{j}"], CORR, HUD_BLANK)); act.append(jnp.array(FWD))
        ids.append(scene(BG[f"corr{d - 1}"], types[door], HUD_BLANK)); act.append(jnp.array(FWD))
        if i == 0:
            for _ in range(gap):
                ids.append(R); act.append(jnp.array(STAY))
    for _ in range(D):
        ids.append(R); act.append(jnp.array(STAY))
    need = types[which]
    ids.append(scene(BG["junc"], JUNC, need))
    return World(jnp.stack(ids), jnp.stack(act), need, jnp.where(which == 0, LEFT, RIGHT))


def make_worlds(key, n, d, D, gap=0):
    return jax.vmap(lambda k: make_world(k, d, D, gap))(jax.random.split(key, n))


def patches(ids):
    """[..., 7, 7] 타일 번호 → [..., 49, 147] 픽셀 패치."""
    return TEX[ids].reshape(*ids.shape[:-2], 49, -1)
