"""플래토 게이팅 델타 규칙 빠른 가중치를 둔 TDP 기억 + 순환 Q 학습 (CPU 확인용).

TDP 블록(rltdp/common/tdp_block.py)의 읽기에 빠른 가중치 W를 더한다: read = q̃ (Σ_τ v k̃ᵀ + M + W)ᵀ.
W는 기울기 없이, 게이트 p_t ∈ {0,1}가 켜질 때만 델타 규칙으로 쓴다 (헤드별):

    ê_τ = normalize( Σ_r γ_r · Re E_K,r(t−1, τ) )         직전 스텝까지의 키 흔적 (토큰 τ). 창은 대역 극점이 정한다
    v*  = v_결과(t)                                        이번 스텝 결과 토큰의 값 = 사건의 부호
    W  ← W + β · p_t · (1/T) Σ_τ (v* − W ê_τ) ê_τᵀ           0 < β < 2에서 유계 (정규화된 키의 LMS)

- 지금 맥락 방향 ê만 고치고, 직교하는 방향의 기록은 그대로 둔다. 목표보다 약하면 강화, 강하면 약화한다.
- 문턱, 시간 평균, 총량 정규화가 없다. 플래토는 이산 사건(성공·실패)이다.
- W는 에피소드를 넘어 이어 간다 (평생 기록).

게이트: off (W = 0) / always (매 스텝, 그때의 결과 토큰 값) / plateau (성공·실패 사건에서만).
상태 갱신: H_t = 블록(H_{t−1} + √d · 주입(o_t, a_{t−1})). 지금 관측을 포함한다 (이 확인에는 관측 예측이 없다).
정책·가치: H_t를 읽는 Q 머리. 재현 조각으로 1스텝 double DQN, 조각마다 흔적은 0에서 시작, W는 현재 값을 상수로.

python -m rltdp.btsp.delta --gate plateau --seed 0
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

from rltdp.common.grid import HW, N_CH, N_DIR, N_OUTCOME, grid_onehot, mlp
from rltdp.common.minigrid_env import FAILURE, N_ACTIONS, PROGRESS, SUCCESS, MemoryEnv
from rltdp.common.replay_buffer import Buffer
from rltdp.common.tdp_block import TDPBlock, rms_norm

N_CELL = HW * HW
TOK_DIR, TOK_ACT, TOK_OUT = N_CELL, N_CELL + 1, N_CELL + 2
N_TOK = N_CELL + 3


class FastTDPBlock(TDPBlock):
    """TDPBlock과 같고, 읽기에 빠른 가중치 W [H, Dv, Dk]를 더하며 값 v도 돌려준다.
    FFN 내부 폭은 inter로 직접 준다 (TDPBlock의 "256의 배수로 올림"은 d = 832용이라 d = 32에서는 3배로 커진다)."""

    def __init__(self, d, heads, bands, periods, inter):
        super().__init__(d, heads, bands, periods)
        self.ffn_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.ffn_down = nn.Linear(inter, d, bias=False)
        nn.init.zeros_(self.ffn_down.weight)

    def forward(self, u, ek, ev, tables, W=None):
        B, T, _ = u.shape
        heads = lambda x: x.reshape(B, T, self.H, self.D).transpose(1, 2)
        q, k, v = heads(self.q_proj(u)), heads(self.k_proj(u)), heads(self.v_proj(u))
        unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + self.eps)
        qr, kr = self.rope(unit(q), tables), self.rope(unit(k), tables)
        read_w = v.transpose(-1, -2) @ kr
        zr, zi, gamma, cr, ci = (x[None, :, None, :, None] for x in self.bands())
        ek_re, ek_im = ek.unbind(3)
        ev_re, ev_im = ev.unbind(3)
        ek_re, ek_im = zr * ek_re - zi * ek_im + gamma * kr.unsqueeze(3), zr * ek_im + zi * ek_re
        ev_re, ev_im = zr * ev_re - zi * ev_im + gamma * v.unsqueeze(3), zr * ev_im + zi * ev_re
        y_re, y_im = gamma * (cr * ev_re - ci * ev_im), gamma * (cr * ev_im + ci * ev_re)
        read_w = read_w + torch.einsum("bhtrd,bhtre->bhde", y_re, ek_re) + torch.einsum("bhtrd,bhtre->bhde", y_im, ek_im)
        if W is not None:
            read_w = read_w + W[None]
        read = torch.einsum("bhte,bhde->bhtd", qr, read_w).transpose(1, 2).reshape(B, T, self.d)
        h = rms_norm(u + self.out_proj(read))
        g, a = self.ffn_gate_up(h).chunk(2, -1)
        h = rms_norm(h + self.ffn_down(F.silu(g) * a))
        return h, torch.stack((ek_re, ek_im), 3), torch.stack((ev_re, ev_im), 3), v


class Net(nn.Module):
    def __init__(self, d=32, heads=2, bands=2, periods=(2.0, 8.0), inter=96):
        super().__init__()
        self.d, self.scale = d, math.sqrt(d)
        std = 1.0 / self.scale
        tn = lambda *s: nn.Parameter(nn.init.trunc_normal_(torch.empty(*s), std=std, a=-2 * std, b=2 * std))
        self.cell_emb, self.pos_emb = tn(N_CH, d), tn(N_CELL, d)
        self.dir_emb, self.act_emb, self.out_emb = tn(N_DIR, d), tn(N_ACTIONS + 1, d), tn(N_OUTCOME, d)
        self.register_buffer("init_hidden", nn.init.trunc_normal_(torch.empty(d), std=1.0, a=-2, b=2))
        self.block = FastTDPBlock(d, heads, bands, periods, inter)
        ar = torch.arange(N_CELL)
        self.register_buffer("pos_row", (ar // HW).float(), persistent=False)
        self.register_buffer("pos_col", (ar % HW).float(), persistent=False)
        self.q_head = mlp(4 * d, N_ACTIONS, hidden=128)

    def tables(self):
        th = self.block.rope_theta
        a = (th[..., 0, None] * self.pos_row + th[..., 1, None] * self.pos_col).transpose(-1, -2)
        a = torch.cat([a, a.new_zeros(a.shape[0], N_TOK - N_CELL, a.shape[-1])], 1)
        return a.cos(), a.sin()

    def init_state(self, N):
        b = self.block
        z = torch.zeros(N, b.H, N_TOK, 2, b.R, b.D)
        return self.init_hidden.expand(N, N_TOK, self.d), z, z.clone()

    def step(self, state, g_oh, d, o, a_prev, tab, W):
        """a_prev = N_ACTIONS이면 '직전 행동 없음'."""
        H, ek, ev = state
        cells = g_oh.flatten(-2).transpose(-1, -2) @ self.cell_emb + self.pos_emb
        inj = torch.cat([cells, self.dir_emb[d][:, None], self.act_emb[a_prev][:, None], self.out_emb[o][:, None]], 1)
        h, ek, ev, v = self.block(H + self.scale * inj, ek, ev, tab, W)
        return (h, ek, ev), v

    def q(self, H):
        return self.q_head(torch.cat([H[:, :N_CELL].mean(1), H[:, TOK_DIR], H[:, TOK_ACT], H[:, TOK_OUT]], -1))


def key_trace(block, ek):
    """ê_τ = normalize(Σ_r γ_r Re E_K,r) [H, T, D] (배치 1)."""
    gamma = block.bands()[2]                                       # [H, R]
    m = (gamma[:, None, :, None] * ek[0, :, :, 0]).sum(2)          # ek[0]: [H, T, 2, R, D] → Re: [H, T, R, D]
    return m / (m.norm(dim=-1, keepdim=True) + 1e-8)


def delta_write(W, e_hat, v_star, beta):
    """W ← W + β (1/T) Σ_τ (v* − W ê_τ) ê_τᵀ. W [H, Dv, Dk], e_hat [H, T, Dk], v_star [H, Dv]."""
    err = v_star[:, None, :] - torch.einsum("hde,hte->htd", W, e_hat)
    return W + beta * torch.einsum("htd,hte->hde", err, e_hat) / e_hat.shape[1]


def dqn_loss(net, tgt, batch, W, gamma):
    g_oh = grid_onehot(batch["grid"])
    B, L = batch["dir"].shape
    tab, ttab = net.tables(), tgt.tables()
    s, ts = net.init_state(B), tgt.init_state(B)
    none = torch.full((B,), N_ACTIONS, dtype=torch.long)
    Qs, TQs = [], []
    for t in range(L):
        a_prev = none if t == 0 else batch["action"][:, t - 1]
        s, _ = net.step(s, g_oh[:, t], batch["dir"][:, t], batch["outcome"][:, t], a_prev, tab, W)
        Qs.append(net.q(s[0]))
        with torch.no_grad():
            ts, _ = tgt.step(ts, g_oh[:, t], batch["dir"][:, t], batch["outcome"][:, t], a_prev, ttab, W)
            TQs.append(tgt.q(ts[0]))
    Q, TQ = torch.stack(Qs, 1), torch.stack(TQs, 1)                 # [B, L, A]
    out, mask = batch["outcome"], batch["mask"]
    r = torch.where(out == SUCCESS, 1.0, torch.where(out == FAILURE, -1.0, 0.0))
    done = (out != PROGRESS).float()
    q_sa = Q[:, :-1].gather(-1, batch["action"][:, :-1, None]).squeeze(-1)
    with torch.no_grad():
        a_star = Q[:, 1:].argmax(-1, keepdim=True)
        target = r[:, 1:] + gamma * (1 - done[:, 1:]) * TQ[:, 1:].gather(-1, a_star).squeeze(-1)
    valid = (mask[:, :-1] & mask[:, 1:] & (out[:, :-1] == PROGRESS)).float()
    return (F.smooth_l1_loss(q_sa, target, reduction="none") * valid).sum() / valid.sum().clamp_min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniGrid-LavaGapS5-v0")
    ap.add_argument("--gate", choices=["off", "always", "plateau"], default="plateau")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--train_every", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="병렬로 여러 실행을 돌릴 때는 1")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--target_every", type=int, default=250)
    ap.add_argument("--eps_steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/btsp_delta")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    env = MemoryEnv(args.env, truncation_as_failure=False)
    net = Net()
    tgt = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    b = net.block
    W = torch.zeros(b.H, b.D, b.D)
    buf = Buffer(args.steps + 2000, "cpu")
    recent, log = deque(maxlen=50), []
    out_dir = Path(args.out) / f"{args.gate}_s{args.seed}"; out_dir.mkdir(parents=True, exist_ok=True)
    steps, updates, ep_i, writes, t0, loss = 0, 0, 0, 0, time.time(), float("nan")

    while steps < args.steps:
        o = env.reset(10_000 * (args.seed + 1) + ep_i); ep_i += 1
        rec = dict(grid=[], dir=[], outcome=[], action=[], segment=[])
        tab = net.tables()
        with torch.no_grad():
            state = net.init_state(1)
            a_prev = torch.tensor([N_ACTIONS])
            while True:
                g_oh = grid_onehot(torch.as_tensor(o["grid"]).long()[None])
                ek_before = state[1]
                state, v = net.step(state, g_oh, torch.tensor([o["dir"]]), torch.tensor([o["outcome"]]), a_prev, tab,
                                     None if args.gate == "off" else W)
                done = o["outcome"] != PROGRESS
                gate = (args.gate == "always" and len(rec["dir"]) > 0) or (args.gate == "plateau" and done)
                if gate:
                    W = delta_write(W, key_trace(b, ek_before), v[0, :, TOK_OUT], args.beta)
                    writes += 1
                rec["grid"].append(o["grid"]); rec["dir"].append(o["dir"]); rec["outcome"].append(o["outcome"])
                rec["segment"].append(0)
                if done or len(rec["dir"]) > env.max_steps:
                    rec["action"].append(0)
                    break
                eps = max(0.05, 1 - steps / args.eps_steps)
                a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(net.q(state[0]).argmax(-1).item())
                rec["action"].append(a)
                a_prev = torch.tensor([a])
                o, _, _ = env.step(a)
                steps += 1
        L = len(rec["dir"]) - 1
        buf.add(rec)
        recent.append((rec["outcome"][-1] == SUCCESS, rec["outcome"][-1] == FAILURE, L))
        if steps > args.warmup:
            for _ in range(max(1, L // args.train_every)):
                lo = dqn_loss(net, tgt, buf.sample(args.batch, args.seq, gen), W, args.gamma)
                opt.zero_grad(set_to_none=True); lo.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
                updates += 1; loss = lo.item()
                if updates % args.target_every == 0:
                    tgt.load_state_dict(net.state_dict())
        if not log or steps // 1000 > log[-1]["steps"] // 1000:
            recd = dict(steps=steps, episodes=ep_i, updates=updates, writes=writes,
                        success=float(np.mean([x[0] for x in recent])), failure=float(np.mean([x[1] for x in recent])),
                        ep_len=float(np.mean([x[2] for x in recent])), W_norm=float(W.norm()), loss=round(loss, 4),
                        sec=round(time.time() - t0, 1))
            log.append(recd)
            print(json.dumps(recd), flush=True)
            (out_dir / "log.json").write_text(json.dumps(log))
    torch.save(dict(net=net.state_dict(), W=W), out_dir / "final.pt")


if __name__ == "__main__":
    main()
