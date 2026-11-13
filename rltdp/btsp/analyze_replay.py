"""BTSP-DQN (rltdp/btsp/replay.py의 final.pt)의 장 해석. 특권 정보(배치, 에이전트 자세)를 쓴다.

출력층 W의 행 = 행동별 가치 뉴런. 장 = 자세마다의 Q(s, a) = W_a φ(s).
LavaGapS5의 배치(틈 위치) 전부에서, 빈 칸 × 방향마다 초기 상태에서 그 자세의 시야를 한 번 넣은 상태 (에피소드 첫 스텝과 같은 꼴)로
1) 참값과 비교: 가치 반복으로 구한 Q*(s, a) (성공 +1, 용암 −1, γ). 탐욕 행동이 최적인 비율, max Q와 V*의 상관·차이
2) 앞 칸 종류별 전진의 Q, 목표까지 최적 스텝 수별 max Q
3) 칸별 가치 지도 (네 방향의 max Q 평균) 대 V*
4) 실제 이력: 무작위 배치 100개에서 시작 자세부터 탐욕 정책(ε = 0)으로 굴린 결과
python -m rltdp.btsp.analyze_replay runs/btsp_replay_v2/delta_s0/final.pt
"""
import argparse

import numpy as np
import torch

from rltdp.btsp.replay import PerceptNet, features
from rltdp.common.grid import grid_onehot
from rltdp.common.minigrid_env import FAILURE, N_ACTIONS, PROGRESS, SUCCESS, MemoryEnv

ACT = ("왼", "오", "전")
VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))                                 # 동 남 서 북


def cell(u, x, y):
    c = u.grid.get(x, y)
    return "빈칸" if c is None else c.type


def optimal_q(u, gamma, iters=300):
    """자세 (x, y, d)마다 Q*(s, ·). 벽 앞 전진은 제자리, 용암 −1, 목표 +1에서 종료."""
    poses = [(x, y, d) for x in range(u.width) for y in range(u.height) for d in range(4)
             if cell(u, x, y) == "빈칸"]
    V = {p: 0.0 for p in poses}

    def nxt(p, a):
        x, y, d = p
        if a == 0:
            return 0.0, (x, y, (d - 1) % 4)
        if a == 1:
            return 0.0, (x, y, (d + 1) % 4)
        fx, fy = x + VEC[d][0], y + VEC[d][1]
        k = cell(u, fx, fy)
        if k == "goal":
            return 1.0, None
        if k == "lava":
            return -1.0, None
        return (0.0, p) if k == "wall" else (0.0, (fx, fy, d))

    for _ in range(iters):
        Q = {p: [r + (0.0 if s is None else gamma * V[s]) for r, s in (nxt(p, a) for a in range(N_ACTIONS))]
             for p in poses}
        V = {p: max(q) for p, q in Q.items()}
    return Q


@torch.no_grad()
def probe_q(net, W, u, x, y, d):
    u.agent_pos, u.agent_dir = np.array([x, y]), d
    g_oh = grid_onehot(torch.as_tensor(u.gen_obs()["image"]).long()[None])
    s, _ = net.step(net.init_state(1), g_oh, torch.tensor([d]), torch.tensor([PROGRESS]),
                    torch.tensor([N_ACTIONS]), net.tables(), None)
    return (features(net.block, s[1]) @ W.T)[0].numpy()


@torch.no_grad()
def rollout(net, W, env, seed):
    o = env.reset(seed)
    tab, s, a_prev, t = net.tables(), net.init_state(1), torch.tensor([N_ACTIONS]), 0
    while True:
        g_oh = grid_onehot(torch.as_tensor(o["grid"]).long()[None])
        s, _ = net.step(s, g_oh, torch.tensor([o["dir"]]), torch.tensor([o["outcome"]]), a_prev, tab, None)
        if o["outcome"] != PROGRESS or t >= env.max_steps:
            return o["outcome"], t
        a = int((features(net.block, s[1]) @ W.T).argmax())
        a_prev = torch.tensor([a])
        o, _, _ = env.step(a)
        t += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--env", default="MiniGrid-LavaGapS5-v0")
    ap.add_argument("--gamma", type=float, default=0.95)
    args = ap.parse_args()
    ck = torch.load(args.ckpt)
    net = PerceptNet(); net.load_state_dict(ck["net"]); net.eval()
    W = ck["W"]
    env = MemoryEnv(args.env, truncation_as_failure=False)
    u = env.u

    layouts, rows = {}, []
    for seed in range(10_000, 10_200):
        env.reset(seed)
        key = tuple(sorted((x, y) for x in range(u.width) for y in range(u.height) if cell(u, x, y) == "lava"))
        layouts.setdefault(key, seed)
    print(f"W ‖·‖ {W.norm():.2f}, 행별 " + ", ".join(f"{a} {n:.2f}" for a, n in zip(ACT, W.norm(dim=1).tolist())))
    for lava, seed in sorted(layouts.items(), key=lambda kv: kv[1]):
        env.reset(seed)
        Qs = optimal_q(u, args.gamma)
        print(f"\n배치 (시드 {seed}): 용암 {list(lava)}")
        vmap, vsmap = np.full((u.height, u.width), np.nan), np.full((u.height, u.width), np.nan)
        for (x, y, d), qs in Qs.items():
            q = probe_q(net, W, u, x, y, d)
            qs = np.array(qs)
            fx, fy = x + VEC[d][0], y + VEC[d][1]
            rows.append(dict(front=cell(u, fx, fy), q=q, qs=qs, vstar=qs.max(),
                             opt=bool(qs[q.argmax()] >= qs.max() - 1e-6), into_lava=cell(u, fx, fy) == "lava"))
            vmap[y, x] = np.nansum([vmap[y, x], q.max() / 4]); vsmap[y, x] = np.nansum([vsmap[y, x], qs.max() / 4])
        print("  칸별 가치 (네 방향 평균) — 모델 max Q | 참값 V*")
        for y in range(1, u.height - 1):
            line = lambda m: " ".join("  ·  " if np.isnan(m[y, x]) else f"{m[y, x]:+.2f}" for x in range(1, u.width - 1))
            print(f"   y={y}: {line(vmap)}   |   {line(vsmap)}")

    q = np.stack([r["q"] for r in rows]); qs = np.stack([r["qs"] for r in rows])
    vq, vs = q.max(1), qs.max(1)
    print(f"\n자세 {len(rows)}개 (배치 {len(layouts)}개 × 빈 칸 × 방향)")
    print(f"  탐욕 행동이 최적인 비율 {np.mean([r['opt'] for r in rows]):.2f}")
    print(f"  max Q 대 V*: 상관 {np.corrcoef(vq, vs)[0, 1]:.3f}, 평균 차이 (Q − V*) {np.mean(vq - vs):+.3f}, "
          f"평균 |차이| {np.mean(np.abs(vq - vs)):.3f}")
    print(f"  모든 (자세, 행동)에서 Q 대 Q*: 상관 {np.corrcoef(q.ravel(), qs.ravel())[0, 1]:.3f}")
    print("  앞 칸 종류별 전진의 Q (모델 / 참값), 탐욕이 전진인 비율")
    for kind in ("goal", "lava", "wall", "빈칸"):
        sel = [i for i, r in enumerate(rows) if r["front"] == kind]
        if sel:
            print(f"    {kind:5s} ({len(sel):2d}개): {q[sel, 2].mean():+.3f} / {qs[sel, 2].mean():+.3f}, "
                  f"{np.mean(q[sel].argmax(1) == 2):.2f}")
    print("  목표까지 최적 스텝 수별 max Q (모델 / 참값 γ^(k−1))")
    steps = np.rint(np.log(np.clip(vs, 1e-9, None)) / np.log(args.gamma)).astype(int) + 1
    for k in sorted(set(steps)):
        sel = steps == k
        print(f"    {k:2d}스텝 ({sel.sum():2d}개): {vq[sel].mean():+.3f} / {vs[sel].mean():+.3f}")

    res = [rollout(net, W, env, 20_000 + i) for i in range(100)]
    out = np.array([r[0] for r in res]); ln = np.array([r[1] for r in res])
    opt_len = []
    for i in range(100):
        env.reset(20_000 + i)
        Qs = optimal_q(u, args.gamma)
        p = (int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir))
        opt_len.append(int(round(np.log(max(Qs[p])) / np.log(args.gamma))) + 1)
    opt_len = np.array(opt_len); ok = out == SUCCESS
    print(f"\n실제 이력, 탐욕 정책 (무작위 배치 100개, 시작 자세부터): 성공 {ok.mean():.2f}, "
          f"실패 {(out == FAILURE).mean():.2f}, 시간 초과 {(out == PROGRESS).mean():.2f}; "
          f"성공 시 스텝 {ln[ok].mean():.2f} (최적 {opt_len[ok].mean():.2f})")


if __name__ == "__main__":
    main()
