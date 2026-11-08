"""MiniGrid-MemoryS13-v0 래퍼, 관측 인코딩, 스크립트 정책, 렌더러 (실험 2 단계 0).

관측 o_t = (grid, dir, outcome)
- grid: (7, 7, 3) uint8. MiniGrid 자기중심 시야의 칸별 (물체, 색, 상태) 인덱스. 축 순서는 MiniGrid 그대로 [x, y, 속성].
- dir: 0..3, outcome: 0 진행 / 1 성공 / 2 실패. 시간 초과(truncated)도 실패로 둔다.
모델 입력용 원-핫(20×7×7)은 to_onehot()이 만든다. 저장은 인덱스로 한다.
"""
from collections import deque

import gymnasium as gym
import minigrid  # noqa: F401  (환경 등록)
import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.grid import Grid

ENV_ID = "MiniGrid-MemoryS13-v0"
N_OBJ, N_COLOR, N_STATE = len(OBJECT_TO_IDX), len(COLOR_TO_IDX), len(STATE_TO_IDX)
GROUP_SIZES = (N_OBJ, N_COLOR, N_STATE)            # (11, 6, 3)
N_CH = sum(GROUP_SIZES)                            # 20
LEFT, RIGHT, FORWARD = 0, 1, 2
N_ACTIONS = 3
PROGRESS, SUCCESS, FAILURE = 0, 1, 2
DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]        # MiniGrid: 0 동, 1 남, 2 서, 3 북

assert GROUP_SIZES == (11, 6, 3), "격자 차원(980)을 다시 계산해야 한다"


def to_onehot(grid_idx: np.ndarray) -> np.ndarray:
    """(..., 7, 7, 3) 인덱스 → (..., 20, 7, 7) float32 원-핫."""
    parts = [np.eye(n, dtype=np.float32)[grid_idx[..., i]] for i, n in enumerate(GROUP_SIZES)]
    return np.moveaxis(np.concatenate(parts, -1), -1, -3)


def segment_of(pos) -> int:
    """에피소드 구간: 0 방 (x ≤ 3), 1 복도 (4 ≤ x ≤ 9), 2 갈림길 (x ≥ 10). S13 기준."""
    x = pos[0]
    return 0 if x <= 3 else (1 if x <= 9 else 2)


class MemoryEnv:
    """MiniGrid 래퍼. 기본은 MemoryS13. 같은 관측 형식의 다른 MiniGrid 환경도 env_id로 쓸 수 있다
    (예: MiniGrid-Dynamic-Obstacles-8x8-v0. 충돌 보상 −1은 실패로 처리된다)."""

    def __init__(self, env_id: str = ENV_ID, truncation_as_failure: bool = True):
        self.env = gym.make(env_id)
        self.truncation_as_failure = truncation_as_failure
        self.u = self.env.unwrapped
        self.max_steps = self.u.max_steps

    def _obs(self, raw, outcome):
        return dict(grid=raw["image"].astype(np.uint8), dir=int(raw["direction"]), outcome=outcome)

    def reset(self, seed: int):
        raw, _ = self.env.reset(seed=seed)
        self.t = 0
        return self._obs(raw, PROGRESS)

    def step(self, a: int):
        assert a in (LEFT, RIGHT, FORWARD)
        raw, reward, terminated, truncated, _ = self.env.step(a)
        self.t += 1
        if terminated:
            outcome = SUCCESS if reward > 0 else FAILURE
        elif truncated and self.truncation_as_failure:
            outcome = FAILURE
        else:
            outcome = PROGRESS          # 시간 초과를 절단으로 볼 때는 결과는 진행, 에피소드만 끝난다
        return self._obs(raw, outcome), reward, bool(terminated or truncated)

    # ---- 특권 정보 (스크립트 정책·분석용, 모델에는 주지 않는다) ----
    @property
    def pose(self):
        return (int(self.u.agent_pos[0]), int(self.u.agent_pos[1])), int(self.u.agent_dir)

    @property
    def cue_pos(self):
        return (1, self.u.height // 2 - 1)

    def cue_visible_from(self, pos, d) -> bool:
        """pos, d 자세에서 시작 방 물체가 (가림 없이) 보이는지."""
        old_pos, old_dir = self.u.agent_pos, self.u.agent_dir
        self.u.agent_pos, self.u.agent_dir = np.array(pos), d
        try:
            rel = self.u.relative_coords(*self.cue_pos)
            if rel is None:
                return False
            _, vis = self.u.gen_obs_grid()
            return bool(vis[rel[0], rel[1]])
        finally:
            self.u.agent_pos, self.u.agent_dir = old_pos, old_dir

    def passable(self, x, y) -> bool:
        c = self.u.grid.get(x, y)
        return c is None or c.can_overlap()


class ScriptedPolicy:
    """시작 방 물체가 보일 때까지 그쪽으로 가서 본 뒤, 같은 물체 옆(success_pos)으로 간다.
    매 스텝 현재 자세에서 BFS로 다시 계획하므로 무작위 행동이 섞여도 복구한다."""

    def __init__(self, env: MemoryEnv, eps: float = 0.0, rng=None):
        self.env, self.eps = env, eps
        self.rng = rng or np.random.default_rng()

    def reset(self):
        self.seen = False
        self._view_goals = None

    def _bfs(self, start, goal_fn):
        (sx, sy), sd = start
        prev = {(sx, sy, sd): None}
        q = deque([(sx, sy, sd)])
        while q:
            s = q.popleft()
            x, y, d = s
            if goal_fn(x, y, d):
                acts = []
                while prev[s] is not None:
                    s, a = prev[s]
                    acts.append(a)
                return acts[::-1]
            dx, dy = DIR_VEC[d]
            for a, n in [(LEFT, (x, y, (d - 1) % 4)), (RIGHT, (x, y, (d + 1) % 4)),
                         (FORWARD, (x + dx, y + dy, d))]:
                if n not in prev and self.env.passable(n[0], n[1]):
                    prev[n] = (s, a)
                    q.append(n)
        raise RuntimeError("경로 없음")

    def act(self) -> int:
        e = self.env
        pos, d = e.pose
        if not self.seen and e.cue_visible_from(pos, d):
            self.seen = True
        if self.eps > 0 and self.rng.random() < self.eps:
            return int(self.rng.integers(N_ACTIONS))
        if not self.seen:
            if self._view_goals is None:
                self._view_goals = {(x, y, dd) for x in range(1, e.u.width - 1) for y in range(1, e.u.height - 1)
                                    for dd in range(4) if e.passable(x, y) and e.cue_visible_from((x, y), dd)}
            goal = lambda x, y, dd: (x, y, dd) in self._view_goals
        else:
            sx, sy = e.u.success_pos
            goal = lambda x, y, dd: (x, y) == (sx, sy)
        return self._bfs((pos, d), goal)[0]


def render_obs(grid_idx: np.ndarray, tile_size: int = 16) -> np.ndarray:
    """인덱스 격자(7,7,3)를 MiniGrid 렌더러로 그린다. 에이전트는 시야 아래 가운데에서 위를 본다."""
    grid, vis = Grid.decode(grid_idx.astype(np.uint8))
    return grid.render(tile_size, agent_pos=(3, 6), agent_dir=3, highlight_mask=~vis)
