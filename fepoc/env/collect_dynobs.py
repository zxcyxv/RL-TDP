"""실험 2c 데이터 수집: MiniGrid-Dynamic-Obstacles-8x8-v0.

에피소드가 짧으므로(무작위 정책 평균 약 9스텝) 에피소드 수가 아니라 관측 수로 맞춘다.
- 학습용: 목표 지향 정책(ε = 0.2) 관측 약 10만 + 무작위 정책 관측 약 10만. 환경 시드 0부터
- 보류: 목표 지향 정책(ε = 0) 약 9천 + 무작위 정책 약 9천. 환경 시드 1,000,000부터
구간(segment) = 현재 시야에 보이는 공의 수: 0 / 1 / 2개 이상.

python -m fepoc.env.collect_dynobs --out data/dynobs
"""
import argparse
from collections import deque
from pathlib import Path

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX

from .memory import DIR_VEC, FORWARD, LEFT, N_ACTIONS, RIGHT, MemoryEnv

ENV_ID = "MiniGrid-Dynamic-Obstacles-8x8-v0"
BALL = OBJECT_TO_IDX["ball"]
SEGMENT_NAMES = np.array(["공 0개", "공 1개", "공 2개 이상"])
POLICY_RANDOM, POLICY_GOAL = 0, 1


def n_balls_visible(grid):
    return int((grid[..., 0] == BALL).sum())


class GoalPolicy:
    """특권 정보로 목표까지 BFS. 지금 공이 있는 칸은 막힌 것으로 본다. 매 스텝 다시 계획한다.
    경로가 없으면 앞이 비었을 때 전진, 아니면 회전."""

    def __init__(self, env, eps, rng):
        self.env, self.eps, self.rng = env, eps, rng

    def _free(self, x, y):
        c = self.env.u.grid.get(x, y)
        return c is None or c.type == "goal"

    def act(self):
        if self.eps > 0 and self.rng.random() < self.eps:
            return int(self.rng.integers(N_ACTIONS))
        (x0, y0), d0 = self.env.pose
        gx, gy = self.env.u.width - 2, self.env.u.height - 2
        prev = {(x0, y0, d0): None}
        q = deque([(x0, y0, d0)])
        while q:
            s = q.popleft()
            x, y, d = s
            if (x, y) == (gx, gy):
                acts = []
                while prev[s] is not None:
                    s, a = prev[s]
                    acts.append(a)
                return acts[-1] if acts else FORWARD
            dx, dy = DIR_VEC[d]
            for a, n in [(LEFT, (x, y, (d - 1) % 4)), (RIGHT, (x, y, (d + 1) % 4)), (FORWARD, (x + dx, y + dy, d))]:
                if n not in prev and self._free(n[0], n[1]):
                    prev[n] = (s, a)
                    q.append(n)
        dx, dy = DIR_VEC[d0]
        return FORWARD if self._free(x0 + dx, y0 + dy) else RIGHT


def run_episode(env, seed, policy, eps, rng):
    o = env.reset(seed)
    pol = GoalPolicy(env, eps, rng) if policy == POLICY_GOAL else None
    rec = dict(grid=[], dir=[], outcome=[], action=[], segment=[])
    done = False
    while True:
        rec["grid"].append(o["grid"]); rec["dir"].append(o["dir"]); rec["outcome"].append(o["outcome"])
        rec["segment"].append(min(n_balls_visible(o["grid"]), 2))
        if done:
            rec["action"].append(-1)
            break
        a = pol.act() if pol else int(rng.integers(N_ACTIONS))
        rec["action"].append(a)
        o, _, done = env.step(a)
    return rec


def collect(plan, seed0, rng):
    """plan: [(정책, ε, 목표 관측 수)]. 정책마다 목표 관측 수를 넘을 때까지 에피소드를 모은다."""
    env = MemoryEnv(ENV_ID)
    eps_list, meta, seed = [], [], seed0
    for policy, eps, target in plan:
        n = 0
        while n < target:
            r = run_episode(env, seed, policy, eps, rng)
            eps_list.append(r); meta.append((seed, policy, r["outcome"][-1]))
            n += len(r["action"]); seed += 1
    cat = {k: np.asarray(np.concatenate([np.asarray(r[k]) for r in eps_list]),
                         dtype=np.uint8 if k != "action" else np.int8) for k in eps_list[0]}
    lengths = np.array([len(r["action"]) for r in eps_list])
    cat["ep_start"] = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    cat["ep_len"] = lengths
    m = np.array(meta)
    cat["env_seed"], cat["policy"], cat["final_outcome"] = m[:, 0], m[:, 1], m[:, 2]
    cat["segment_names"] = SEGMENT_NAMES
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dynobs")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    for name, plan, seed0 in [("train", [(POLICY_GOAL, 0.2, 100_000), (POLICY_RANDOM, 0.0, 100_000)], 0),
                              ("holdout", [(POLICY_GOAL, 0.0, 9_000), (POLICY_RANDOM, 0.0, 9_000)], 1_000_000)]:
        d = collect(plan, seed0, rng)
        np.savez_compressed(out / f"{name}.npz", **d)
        for p, pn in [(POLICY_GOAL, "목표 지향"), (POLICY_RANDOM, "무작위")]:
            sel = d["policy"] == p
            L = d["ep_len"][sel] - 1
            print(f"{name} {pn}: 에피소드 {sel.sum()}개, 길이 평균 {L.mean():.1f} 최대 {L.max()}, "
                  f"성공 {np.mean(d['final_outcome'][sel] == 1):.2f} 실패 {np.mean(d['final_outcome'][sel] == 2):.2f}")
        seg = d["segment"]
        print(f"{name}: 관측 {len(seg)}, 구간 분포 " + ", ".join(f"{SEGMENT_NAMES[i]} {np.mean(seg == i):.2f}" for i in range(3)))


if __name__ == "__main__":
    main()
