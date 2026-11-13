"""BTSP로 구현한 DQN. Q-net = 지각 블록 + TDP 블록 θ (느린 가중치, 역전파) + 출력층 W (빠른 가중치, 델타 규칙). 재현 = 해마 재생.

    x_t = 지각 블록(주입(o_t, a_{t−1}))   같은 스텝 토큰 52개끼리만 softmax 어텐션 + FFN (스텝을 넘는 상태 없음)
    u_t = H_{t−1} + x_t                  → TDP 블록. 키가 처리된 현재 관측을 담는다
    φ_t = 모든 토큰·헤드의 키 트레이스 ê = normalize(Σ_r γ_r Re E_K,r)를 이어 붙인 벡터 (‖φ_t‖ = 1)
    Q(t, ·) = W φ_t                      W [행동 수, φ 차원]. 에피소드를 넘어 유지, TDP 상태는 에피소드마다 0에서 시작

목표망: θ⁻, W⁻는 재생 한 번마다 EMA로 따라간다 (θ⁻ ← (1−τ)θ⁻ + τθ, W⁻도 같음). φ⁻는 θ⁻로 계산한 특징.
목표 (Double Q): y_t = r_{t+1} + γ (1 − 종료_{t+1}) W⁻_{a*} φ⁻_{t+1},  a* = argmax_a W φ_{t+1}

행동하는 동안: TDP 상태만 진행하고 ε-greedy로 고른다. W와 θ는 바뀌지 않는다. 에피소드는 원시 관측 그대로 버퍼에 둔다.
에피소드가 끝나면 재생: 방금 에피소드 + 버퍼의 무작위 과거 에피소드 K개를 각각 처음부터 현재 θ (와 θ⁻)로 다시 돌린다.
--rule delta: 에피소드마다 역순으로 (방금 에피소드가 먼저)
    δ_t = y_t − W_{a_t} φ_t,   W_{a_t} ← W_{a_t} + β δ_t φ_tᵀ   (한 행동의 가치 뉴런에 플래토가 y_t를 실어 온다)
  같은 δ_t로 θ를 학습한다: ½ mean δ_t²을 φ를 통해 역전파 (쓰기 직전의 W 행과 y는 상수).
--rule adam (비교군): 같은 재생 묶음에서 W도 θ와 함께 Adam으로 학습한다. 순서는 의미가 없다.
보상: 성공 +1, 용암 −1, 시간 초과는 절단 (부트스트랩).

python -m rltdp.btsp.replay --rule delta --seed 0
"""
import argparse
import copy
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rltdp.btsp.delta import N_TOK, Net
from rltdp.common.grid import grid_onehot
from rltdp.common.minigrid_env import FAILURE, N_ACTIONS, PROGRESS, SUCCESS, MemoryEnv
from rltdp.common.tdp_block import rms_norm


class StepBlock(nn.Module):
    """같은 스텝의 토큰끼리만 softmax 어텐션 + FFN. TDP 블록과 같은 사후 RMSNorm."""

    def __init__(self, d, heads, inter):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, bias=False, batch_first=True)
        self.ffn_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.ffn_down = nn.Linear(inter, d, bias=False)

    def forward(self, x):
        h = rms_norm(x + self.attn(x, x, x, need_weights=False)[0])
        g, a = self.ffn_gate_up(h).chunk(2, -1)
        return rms_norm(h + self.ffn_down(F.silu(g) * a))


class PerceptNet(Net):
    """Net (delta.py)의 주입 앞에 지각 블록을 둔다. Q 머리는 없다 (Q는 W φ로만 읽는다)."""

    def __init__(self, d=32, heads=2, inter=96):
        super().__init__(d=d, heads=heads, inter=inter)
        self.q_head = None
        self.percept = StepBlock(d, heads, inter)

    def step(self, state, g_oh, d, o, a_prev, tab, W=None):
        H, ek, ev = state
        cells = g_oh.flatten(-2).transpose(-1, -2) @ self.cell_emb + self.pos_emb
        inj = torch.cat([cells, self.dir_emb[d][:, None], self.act_emb[a_prev][:, None], self.out_emb[o][:, None]], 1)
        h, ek, ev, v = self.block(H + self.percept(self.scale * inj), ek, ev, tab, W)
        return (h, ek, ev), v


def features(block, ek):
    """ek [B, H, T, 2, R, D] → φ [B, H·T·D], ‖φ‖ = 1."""
    gamma = block.bands()[2]                                                # [H, R]
    m = (gamma[None, :, None, :, None] * ek[:, :, :, 0]).sum(3)             # [B, H, T, D]
    e = m / (m.norm(dim=-1, keepdim=True) + 1e-8)
    return e.flatten(1) / math.sqrt(e.shape[1] * e.shape[2])


def reward(out):
    return torch.where(out == SUCCESS, 1.0, torch.where(out == FAILURE, -1.0, 0.0))


def stack(eps):
    """원시 에피소드 목록 → 덧댄 텐서 (grid [B, L, 7, 7, 3], dir, outcome, action [B, L]), 길이 n [B]."""
    B, L = len(eps), max(len(e["dir"]) for e in eps)
    grid = np.zeros((B, L) + eps[0]["grid"][0].shape, np.uint8)
    d, out, act = (np.zeros((B, L), np.int64) for _ in range(3))
    out[:] = PROGRESS
    n = np.array([len(e["dir"]) for e in eps])
    for i, e in enumerate(eps):
        grid[i, :n[i]], d[i, :n[i]] = np.stack(e["grid"]), e["dir"]
        out[i, :n[i]], act[i, :n[i]] = e["outcome"], e["action"]
    return torch.as_tensor(grid).long(), torch.as_tensor(d), torch.as_tensor(out), torch.as_tensor(act), n


def replay_features(net, grid, d, out, act):
    """에피소드를 처음부터 다시 돌려 φ [B, L, F]를 계산한다 (θ로 역전파 가능)."""
    B, L = d.shape
    tab, s = net.tables(), net.init_state(B)
    g_oh = grid_onehot(grid)
    none = torch.full((B,), N_ACTIONS, dtype=torch.long)
    phis = []
    for t in range(L):
        s, _ = net.step(s, g_oh[:, t], d[:, t], out[:, t], none if t == 0 else act[:, t - 1], tab, None)
        phis.append(features(net.block, s[1]))
    return torch.stack(phis, 1)


@torch.no_grad()
def delta_sweep(W, Wt, P, Pt, out, act, n, gamma, beta):
    """에피소드마다 역순으로 쓴다. 쓰기 직전의 W 행과 목표 y를 돌려준다 (θ의 손실용)."""
    ii, tt, rows, ys = [], [], [], []
    for i in range(P.shape[0]):
        for t in reversed(range(int(n[i]) - 1)):
            y = float(reward(out[i, t + 1]))
            if int(out[i, t + 1]) == PROGRESS:
                a_star = int((W @ P[i, t + 1]).argmax())
                y += gamma * float(Wt[a_star] @ Pt[i, t + 1])
            a = int(act[i, t])
            row = W[a].clone()
            W[a] += beta * (y - float(row @ P[i, t])) * P[i, t]
            ii.append(i); tt.append(t); rows.append(row); ys.append(y)
    return ii, tt, torch.stack(rows), torch.tensor(ys)


def adam_loss(W, Wt, P, Pt, out, act, n, gamma):
    B, L, _ = P.shape
    Q = P @ W.T                                                             # [B, L, A]
    with torch.no_grad():
        a_star = Q[:, 1:].argmax(-1, keepdim=True)
        q_next = (Pt[:, 1:] @ Wt.T).gather(-1, a_star).squeeze(-1)
        y = reward(out[:, 1:]) + gamma * (out[:, 1:] == PROGRESS).float() * q_next
    pred = Q[:, :-1].gather(-1, act[:, :-1, None]).squeeze(-1)
    valid = (torch.arange(L - 1)[None] < torch.as_tensor(n - 1)[:, None]).float()
    return 0.5 * (((y - pred) ** 2) * valid).sum() / valid.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniGrid-LavaGapS5-v0")
    ap.add_argument("--rule", choices=["delta", "adam"], default="delta")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.05, help="목표망 EMA 비율 (재생 한 번마다)")
    ap.add_argument("--replay_past", type=int, default=15, help="에피소드마다 재생할 무작위 과거 에피소드 수 K")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eps_steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/btsp_replay_v2")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = MemoryEnv(args.env, truncation_as_failure=False)
    net = PerceptNet()
    tgt = copy.deepcopy(net).requires_grad_(False)
    b = net.block
    W = torch.zeros(N_ACTIONS, b.H * N_TOK * b.D)
    params = list(net.parameters())
    if args.rule == "adam":
        W = nn.Parameter(W)
        params.append(W)
    Wt = W.detach().clone()
    opt = torch.optim.Adam(params, lr=args.lr)
    episodes, recent, log = [], deque(maxlen=50), []
    out_dir = Path(args.out) / f"{args.rule}_s{args.seed}"; out_dir.mkdir(parents=True, exist_ok=True)
    steps, ep_i, rounds, writes, t0, loss = 0, 0, 0, 0, time.time(), float("nan")

    while steps < args.steps:
        o = env.reset(10_000 * (args.seed + 1) + ep_i); ep_i += 1
        rec = dict(grid=[], dir=[], outcome=[], action=[])
        with torch.no_grad():
            tab, state = net.tables(), net.init_state(1)
            a_prev = torch.tensor([N_ACTIONS])
            while True:
                g_oh = grid_onehot(torch.as_tensor(o["grid"]).long()[None])
                state, _ = net.step(state, g_oh, torch.tensor([o["dir"]]), torch.tensor([o["outcome"]]), a_prev, tab, None)
                rec["grid"].append(o["grid"]); rec["dir"].append(o["dir"]); rec["outcome"].append(o["outcome"])
                if o["outcome"] != PROGRESS or len(rec["dir"]) > env.max_steps:
                    rec["action"].append(0)
                    break
                eps = max(0.05, 1 - steps / args.eps_steps)
                if rng.random() < eps:
                    a = int(rng.integers(N_ACTIONS))
                else:
                    a = int((features(b, state[1]) @ W.detach().T).argmax(-1).item())
                rec["action"].append(a)
                a_prev = torch.tensor([a])
                o, _, _ = env.step(a)
                steps += 1
        episodes.append(rec)
        recent.append((rec["outcome"][-1] == SUCCESS, rec["outcome"][-1] == FAILURE, len(rec["dir"]) - 1))

        # 재생: 방금 에피소드가 먼저, 그다음 무작위 과거 에피소드 K개
        past = rng.choice(len(episodes) - 1, size=min(args.replay_past, len(episodes) - 1), replace=False)
        grid, d, out, act, n = stack([rec] + [episodes[j] for j in past])
        P = replay_features(net, grid, d, out, act)
        with torch.no_grad():
            Pt = replay_features(tgt, grid, d, out, act)
        if args.rule == "delta":
            ii, tt, rows, ys = delta_sweep(W, Wt, P.detach(), Pt, out, act, n, args.gamma, args.beta)
            lo = 0.5 * ((ys - (rows * P[ii, tt]).sum(-1)) ** 2).mean()
            writes += len(ii)
        else:
            lo = adam_loss(W, Wt, P, Pt, out, act, n, args.gamma)
        opt.zero_grad(set_to_none=True); lo.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0); opt.step()
        with torch.no_grad():
            for pt, p in zip(tgt.parameters(), net.parameters()):
                pt.lerp_(p, args.tau)
            Wt.lerp_(W.detach(), args.tau)
        rounds += 1; loss = lo.item()

        if not log or steps // 1000 > log[-1]["steps"] // 1000:
            recd = dict(steps=steps, episodes=ep_i, rounds=rounds, writes=writes,
                        success=float(np.mean([x[0] for x in recent])), failure=float(np.mean([x[1] for x in recent])),
                        ep_len=float(np.mean([x[2] for x in recent])), W_norm=float(W.detach().norm()),
                        loss=round(loss, 4), sec=round(time.time() - t0, 1))
            log.append(recd)
            print(json.dumps(recd), flush=True)
            (out_dir / "log.json").write_text(json.dumps(log))
    torch.save(dict(net=net.state_dict(), W=W.detach()), out_dir / "final.pt")


if __name__ == "__main__":
    main()
