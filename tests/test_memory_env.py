"""실험 2 단계 0 완료 기준: 소스 확인 항목 3개, 스크립트 정책 성공률 100%, 원-핫 인코딩."""
import inspect

import numpy as np
from minigrid.envs.memory import MemoryEnv as RawMemoryEnv

from fepoc.env.memory import (GROUP_SIZES, SUCCESS, MemoryEnv, ScriptedPolicy, render_obs,
                              to_onehot)


def test_source_checks():
    assert GROUP_SIZES == (11, 6, 3)
    env = MemoryEnv()
    assert env.max_steps == 845
    src = inspect.getsource(RawMemoryEnv.step)
    assert "self.success_pos" in src and "self.failure_pos" in src   # 위치 도달로 판정


def test_scripted_policy_always_succeeds():
    env = MemoryEnv()
    for seed in range(200):
        env.reset(seed)
        pol = ScriptedPolicy(env)
        pol.reset()
        done = False
        while not done:
            o, _, done = env.step(pol.act())
        assert o["outcome"] == SUCCESS, seed
        assert env.t <= 15


def test_onehot_is_valid_per_group():
    env = MemoryEnv()
    o = env.reset(0)
    x = to_onehot(o["grid"])
    assert x.shape == (20, 7, 7)
    start = 0
    for n in GROUP_SIZES:
        assert np.all(x[start:start + n].sum(0) == 1)
        start += n
    assert render_obs(o["grid"]).shape == (112, 112, 3)
