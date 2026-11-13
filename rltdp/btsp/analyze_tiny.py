"""소규모 BTSP 에이전트의 장 해석 (rltdp/btsp/tiny.py의 final.pt).

LavaGap 방의 모든 빈 칸 × 방향 × 행동에서 목표 장 G와 위험 장 D의 평균 활동을 잰다. 특권 정보(에이전트 자세)를 쓴다.
python -m rltdp.btsp.analyze_tiny runs/btsp_tiny/btsp_s0/final.pt --seed 10000
"""
import argparse

import numpy as np
import torch

from rltdp.btsp.tiny import N_ACTIONS, POPS, Model
from rltdp.common.grid import grid_onehot
from rltdp.common.minigrid_env import MemoryEnv

ACT = ("왼쪽", "오른쪽", "전진")
DIR = ("동", "남", "서", "북")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--env", default="MiniGrid-LavaGapS5-v0")
    ap.add_argument("--seed", type=int, default=10000)
    args = ap.parse_args()
    ck = torch.load(args.ckpt)
    W = ck["W"]
    n_per = W.shape[0] // len(POPS)
    pop = {k: slice(i * n_per, (i + 1) * n_per) for i, k in enumerate(POPS)}
    env = MemoryEnv(args.env, truncation_as_failure=False)
    env.reset(args.seed)
    u = env.u
    gx, gy = next((x, y) for x in range(u.width) for y in range(u.height)
                  if u.grid.get(x, y) is not None and u.grid.get(x, y).type == "goal")
    lava = [(x, y) for x in range(u.width) for y in range(u.height)
            if u.grid.get(x, y) is not None and u.grid.get(x, y).type == "lava"]
    print(f"목표 {(gx, gy)}, 용암 {lava}, W 평균 " + ", ".join(f"{k} {W[pop[k]].mean():.3f}" for k in POPS))
    print("칸      방향  앞칸      | G(왼,오,전)          | D(왼,오,전)          | 점수 최대 행동")
    rows = []
    for y in range(1, u.height - 1):
        for x in range(1, u.width - 1):
            c = u.grid.get(x, y)
            if c is not None and c.type in ("lava", "goal", "wall"):
                continue
            for d in range(4):
                u.agent_pos, u.agent_dir = np.array([x, y]), d
                img = u.gen_obs()["image"]
                g_oh = grid_onehot(torch.as_tensor(img).long()[None])
                fx, fy = u.front_pos
                fc = u.grid.get(fx, fy)
                front = "빈칸" if fc is None else fc.type
                G, D = [], []
                for a in range(N_ACTIONS):
                    yv = Model.field(W, Model.presyn(g_oh, torch.tensor([a])))[0]
                    G.append(yv[pop["G"]].mean().item()); D.append(yv[pop["D"]].mean().item())
                sc = np.array(G) - np.array(D)
                rows.append(((x, y), d, front, G, D, int(sc.argmax())))
                print(f"{(x, y)}  {DIR[d]}   {front:7s} | " + " ".join(f"{v:.3f}" for v in G) + " | "
                      + " ".join(f"{v:.3f}" for v in D) + f" | {ACT[int(sc.argmax())]}")
    # 요약: 앞 칸 종류별 전진의 G, D
    for kind in ("goal", "lava", "wall", "빈칸"):
        sel = [r for r in rows if r[2] == kind]
        if sel:
            print(f"앞 칸 = {kind:5s} ({len(sel)}개 자세): 전진의 G 평균 {np.mean([r[3][2] for r in sel]):.3f}, "
                  f"D 평균 {np.mean([r[4][2] for r in sel]):.3f}, 점수 최대가 전진인 비율 "
                  f"{np.mean([r[5] == 2 for r in sel]):.2f}")


if __name__ == "__main__":
    main()
