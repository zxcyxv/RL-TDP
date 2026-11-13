"""플래토 게이팅 델타 규칙 W의 장 해석 (rltdp/btsp/delta.py의 final.pt). 특권 정보(에이전트 자세)를 쓴다.

LavaGap 방의 모든 빈 칸 × 방향에서, 그 자세의 시야 하나로 블록을 한 번 통과시킨 상태를 만들고
1) 장: W의 읽기 W·ê (토큰 평균)를 "성공 부호 − 실패 부호" 방향에 투영한 값. 부호 c_s, c_f는 같은 자세에서
   결과 토큰에 성공 / 실패를 넣었을 때의 결과 토큰 값이다 (사건 시점에 W에 쓰는 v*와 같은 양)
2) Q 영향: W가 있을 때와 없을 때(W = 0)의 Q값 차이 ΔQ (행동별)
python -m rltdp.btsp.analyze_delta runs/btsp_delta/plateau_s0/final.pt --seed 10000
"""
import argparse

import numpy as np
import torch

from rltdp.btsp.delta import TOK_OUT, Net, key_trace
from rltdp.common.grid import grid_onehot
from rltdp.common.minigrid_env import FAILURE, N_ACTIONS, PROGRESS, SUCCESS, MemoryEnv

ACT = ("왼쪽", "오른쪽", "전진")
DIR = ("동", "남", "서", "북")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--env", default="MiniGrid-LavaGapS5-v0")
    ap.add_argument("--seed", type=int, default=10000)
    args = ap.parse_args()
    ck = torch.load(args.ckpt)
    net = Net(); net.load_state_dict(ck["net"]); net.eval()
    W = ck["W"]
    env = MemoryEnv(args.env, truncation_as_failure=False)
    env.reset(args.seed)
    u = env.u
    cell = lambda x, y: (u.grid.get(x, y).type if u.grid.get(x, y) is not None else "빈칸")
    goal = next((x, y) for x in range(u.width) for y in range(u.height) if cell(x, y) == "goal")
    lava = [(x, y) for x in range(u.width) for y in range(u.height) if cell(x, y) == "lava"]
    print(f"목표 {goal}, 용암 {lava}, ‖W‖ {W.norm():.2f}")
    tab = net.tables()
    none = torch.tensor([N_ACTIONS])
    rows = []
    print("칸      방향  앞칸   거리 | 장(성공−실패) | Q(왼,오,전) W 있음        | ΔQ = Q_W − Q_0")
    for y in range(1, u.height - 1):
        for x in range(1, u.width - 1):
            if cell(x, y) in ("lava", "goal", "wall"):
                continue
            for d in range(4):
                u.agent_pos, u.agent_dir = np.array([x, y]), d
                g_oh = grid_onehot(torch.as_tensor(u.gen_obs()["image"]).long()[None])
                dt = torch.tensor([d])
                s0 = net.init_state(1)
                (H, ek, _), _ = net.step(s0, g_oh, dt, torch.tensor([PROGRESS]), none, tab, W)
                (H0, _, _), _ = net.step(s0, g_oh, dt, torch.tensor([PROGRESS]), none, tab, None)
                _, vs = net.step(s0, g_oh, dt, torch.tensor([SUCCESS]), none, tab, W)
                _, vf = net.step(s0, g_oh, dt, torch.tensor([FAILURE]), none, tab, W)
                code = (vs - vf)[0, :, TOK_OUT]                                   # [H, D]
                read = torch.einsum("hde,hte->htd", W, key_trace(net.block, ek)).mean(1)
                field = float((read * code).sum() / code.norm())
                qw, q0 = net.q(H)[0], net.q(H0)[0]
                fx, fy = u.front_pos
                front = cell(fx, fy)
                dist = abs(x - goal[0]) + abs(y - goal[1])
                rows.append(dict(front=front, dist=dist, field=field, qw=qw.numpy(), dq=(qw - q0).numpy()))
                print(f"{(x, y)}  {DIR[d]}   {front:5s}  {dist}  | {field:+.3f}       | "
                      + " ".join(f"{v:+.2f}" for v in qw.tolist()) + "  | "
                      + " ".join(f"{v:+.2f}" for v in (qw - q0).tolist()) + f"  → {ACT[int(qw.argmax())]}")
    print("\n앞 칸별 평균: 장, 전진의 Q (W 있음), 전진의 ΔQ")
    for kind in ("goal", "lava", "wall", "빈칸"):
        sel = [r for r in rows if r["front"] == kind]
        if sel:
            print(f"  {kind:5s} ({len(sel):2d}개): 장 {np.mean([r['field'] for r in sel]):+.3f}, "
                  f"Q전진 {np.mean([r['qw'][2] for r in sel]):+.3f}, ΔQ전진 {np.mean([r['dq'][2] for r in sel]):+.3f}")
    print("목표까지 거리별 평균: 장, max Q (W 있음), |ΔQ| 평균")
    for dist in sorted({r["dist"] for r in rows}):
        sel = [r for r in rows if r["dist"] == dist]
        print(f"  거리 {dist} ({len(sel):2d}개): 장 {np.mean([r['field'] for r in sel]):+.3f}, "
              f"maxQ {np.mean([r['qw'].max() for r in sel]):+.3f}, |ΔQ| {np.mean([np.abs(r['dq']).mean() for r in sel]):.3f}")


if __name__ == "__main__":
    main()
