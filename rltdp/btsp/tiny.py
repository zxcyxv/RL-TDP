"""소규모 M1(RSSM) + TDP 기억 + BTSP 시냅스 에이전트 (CPU 확인용). 설계: docs/btsp/m1_tdp_btsp_design.md

python -m rltdp.btsp.tiny --policy btsp --steps 15000
python -m rltdp.btsp.tiny --policy random --steps 15000         # 기준: 무작위 행동 (모델·장은 똑같이 학습)
python -m rltdp.btsp.tiny --policy value --steps 15000          # 비교: 가치 채널의 한 스텝 예측으로 행동

설계 문서에서 줄이거나 단순화한 것
- TDP 블록 d = 32, 헤드 2, 대역 2 (주기 2, 8). 토큰 53개 = 칸 49 + 방향·행동·결과·잠재
- 예측 = [격자 원-핫 980 (탈양자화, 대각 가우시안) ; 보상 ; 가치]. 방향은 예측하지 않는다
- 가치 채널은 의미 기울기 TD(0) (목표망 없음, 다음 시점의 가치를 detach)
- 시냅스전 입력은 학습된 TDP 특징이 아니라 희소한 결합 특징 x = 원-핫(행동) ⊗ (에이전트 주변 3×3 칸의 원-핫) (540차원,
  27개만 1). 학습 초기의 조밀한 특징이면 플래토 때 모든 시냅스가 똑같이 강화되어 장이 아무것도 구별하지 못한다.
  대가: 학습된 표상에 의한 일반화가 없다. 시점 t의 입력은 (o_t, a_t)이고, 시점 t의 플래토는 t−1까지의 입력에 공을 돌린다
- ET, IS는 "최댓값 유지 + 지수 감쇠" (ET ← max(ρ·ET, x), IS ← max(ρ·IS, p)): 최근 입력·플래토가 1에 가까운 값을 갖게 해
  Milstein et al. 2021의 문턱 (α₊ = 0.24, α₋ = 0.09)을 그대로 쓰기 위한 단순화. 플래토는 3 하위 스텝 동안 지속된다고 보고
  W를 3번 갱신한다 (그사이 IS·ET는 감쇠)
- BTSP 속도는 스텝 단위 k₊ = 0.5, k₋ = 0.07 (비율은 원문 2.27 : 0.33과 같게), W_max = 1, q₋ 기울기 200 (원문 약 2260)
- 플래토: 픽셀 놀람 S > 이동 97 백분위 → 새로움 N, δ > 0.3 → 목표 G, δ < −0.3 → 위험 D. 집단 안에서 지금 장 활동이
  가장 약한 단위 2개에 준다
- 행동 (btsp): 후보 행동마다 x(o_t, a)의 장 y(a), 점수 = 평균 y_G − λ_D·평균 y_D
- 사전 p(s_t | ·)에는 직전 입력 x(o_{t−1}, a_{t−1})의 장 y를 넣는다 (기울기 없음)
"""
import argparse
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
from rltdp.common.tdp_block import TDPBlock

N_CELL = HW * HW
TOK_DIR, TOK_ACT, TOK_OUT, TOK_LAT = N_CELL, N_CELL + 1, N_CELL + 2, N_CELL + 3
N_TOK = N_CELL + 4
N_PIX = N_CH * N_CELL
N_LOCAL = N_CH * 9
POPS = ("N", "G", "D")


def reward_of(outcome):
    return torch.where(outcome == SUCCESS, 1.0, torch.where(outcome == FAILURE, -1.0, 0.0))


def gauss_nll(x, mu, ls, smin):
    sig = smin + F.softplus(ls)
    return 0.5 * ((x - mu) / sig) ** 2 + torch.log(sig) + 0.5 * math.log(2 * math.pi)


class Model(nn.Module):
    def __init__(self, d=32, heads=2, bands=2, periods=(2.0, 8.0), s_dim=8, dx=4, n_per=16, e_dim=64):
        super().__init__()
        self.d, self.s_dim, self.n_per, self.scale = d, s_dim, n_per, math.sqrt(d)
        std = 1.0 / self.scale
        tn = lambda *s: nn.Parameter(nn.init.trunc_normal_(torch.empty(*s), std=std, a=-2 * std, b=2 * std))
        self.cell_emb, self.pos_emb = tn(N_CH, d), tn(N_CELL, d)
        self.dir_emb, self.act_emb, self.out_emb = tn(N_DIR + 1, d), tn(N_ACTIONS + 1, d), tn(N_OUTCOME + 1, d)
        self.lat = nn.Linear(s_dim, d)
        self.register_buffer("init_hidden", nn.init.trunc_normal_(torch.empty(d), std=1.0, a=-2, b=2))
        self.block = TDPBlock(d, heads, bands, periods)
        ar = torch.arange(N_CELL)
        self.register_buffer("pos_row", (ar // HW).float(), persistent=False)
        self.register_buffer("pos_col", (ar % HW).float(), persistent=False)
        self.n_pre = N_ACTIONS * N_LOCAL
        self.enc = mlp(N_PIX + N_DIR, e_dim, hidden=128)
        feat, ny = 4 * d, 3 * n_per
        self.prior = mlp(feat + ny, 2 * s_dim, hidden=128)
        self.post = mlp(feat + ny + e_dim, 2 * s_dim, hidden=128)
        self.dec = mlp(s_dim + feat, 2 * N_PIX + 4, hidden=128)      # 격자 (μ, log σ), 보상 (μ, log σ), 가치 (μ, log σ)

    # --- 기억 ---
    def tables(self):
        th = self.block.rope_theta
        a = (th[..., 0, None] * self.pos_row + th[..., 1, None] * self.pos_col).transpose(-1, -2)
        a = torch.cat([a, a.new_zeros(a.shape[0], N_TOK - N_CELL, a.shape[-1])], 1)
        return a.cos(), a.sin()

    def init_state(self, N):
        b = self.block
        z = torch.zeros(N, b.H, N_TOK, 2, b.R, b.D)
        return self.init_hidden.expand(N, N_TOK, self.d), z, z.clone()

    def inject(self, g_oh, d, o, a, s, has):
        hf = has.float()
        cells = (g_oh.flatten(-2).transpose(-1, -2) @ self.cell_emb) * hf[:, None, None] + self.pos_emb
        none = lambda i, n: torch.where(has, i, torch.full_like(i, n))
        return torch.cat([cells, self.dir_emb[none(d, N_DIR)][:, None], self.act_emb[none(a, N_ACTIONS)][:, None],
                          self.out_emb[none(o, N_OUTCOME)][:, None], (self.lat(s) * hf[:, None])[:, None]], 1)

    def step(self, state, inj, tab):
        H, ek, ev = state
        return self.block(H + self.scale * inj, ek, ev, tab)

    def feat(self, H):
        return torch.cat([H[:, :N_CELL].mean(1), H[:, TOK_DIR], H[:, TOK_ACT], H[:, TOK_LAT]], -1)

    @staticmethod
    def presyn(g_oh, a):
        """원-핫(행동) ⊗ 주변 3×3 칸의 원-핫 → [N, 3·180]. 칸 (x, y) ∈ [2, 4] × [4, 6]: 앞 칸 (3, 5), 에이전트 칸 (3, 6)."""
        loc = g_oh[:, :, 2:5, 4:7].flatten(1)
        x = torch.zeros(g_oh.shape[0], N_ACTIONS, loc.shape[1])
        x[torch.arange(g_oh.shape[0]), a] = loc
        return x.flatten(1)

    @staticmethod
    def field(W, x):
        """y_i = W_i·x / Σx: 틀과 지금 입력의 겹침 비율 (0 ~ W_max)."""
        return (x @ W.T) / x.sum(-1, keepdim=True).clamp_min(1e-6)

    # --- 머리 ---
    def gauss(self, raw):
        m, s = raw.chunk(2, -1)
        return m, F.softplus(s) + 1e-3

    def decode(self, s, f):
        o = self.dec(torch.cat([s, f], -1))
        return dict(pix_mu=o[:, :N_PIX], pix_ls=o[:, N_PIX:2 * N_PIX], r_mu=o[:, -4], r_ls=o[:, -3],
                    v_mu=o[:, -2], v_ls=o[:, -1])

    def obs_vec(self, g_oh, d):
        return torch.cat([g_oh.flatten(1), F.one_hot(d, N_DIR).float()], -1)

    # --- 학습 (재현 조각) ---
    def loss(self, batch, W, gamma, smin=0.05):
        g_oh = grid_onehot(batch["grid"])
        B, L = batch["dir"].shape
        tab = self.tables()
        state = self.init_state(B)
        s_prev = torch.zeros(B, self.s_dim)
        r = reward_of(batch["outcome"])
        done = (batch["outcome"] != PROGRESS).float()
        mask = batch["mask"].float()
        pix, rew, kls, vs, vls = [], [], [], [], []
        for t in range(L):
            if t == 0:
                z = torch.zeros(B, dtype=torch.long)
                inj = self.inject(torch.zeros_like(g_oh[:, 0]), z, z, z, s_prev, torch.zeros(B, dtype=torch.bool))
            else:
                inj = self.inject(g_oh[:, t - 1], batch["dir"][:, t - 1], batch["outcome"][:, t - 1],
                                  batch["action"][:, t - 1], s_prev, torch.ones(B, dtype=torch.bool))
            state = self.step(state, inj, tab)
            H = state[0]
            f = self.feat(H)
            if t == 0:
                y = torch.zeros(B, W.shape[0])
            else:
                y = self.field(W, self.presyn(g_oh[:, t - 1], batch["action"][:, t - 1]))
            pm, ps = self.gauss(self.prior(torch.cat([f, y], -1)))
            e = self.enc(self.obs_vec(g_oh[:, t], batch["dir"][:, t]))
            qm, qs = self.gauss(self.post(torch.cat([f, y, e], -1)))
            s = qm + qs * torch.randn_like(qm)
            kls.append((torch.log(ps / qs) + (qs ** 2 + (qm - pm) ** 2) / (2 * ps ** 2) - 0.5).sum(-1))
            dd = self.decode(s, f)
            xq = g_oh[:, t].flatten(1) + torch.rand(B, N_PIX)                    # 균등 탈양자화
            pix.append(gauss_nll(xq, dd["pix_mu"], dd["pix_ls"], smin).sum(-1))
            rew.append(gauss_nll(r[:, t], dd["r_mu"], dd["r_ls"], smin))
            vs.append(dd["v_mu"]); vls.append(dd["v_ls"])
            s_prev = s
        pix, rew, kl = torch.stack(pix, 1), torch.stack(rew, 1), torch.stack(kls, 1)
        v, vls = torch.stack(vs, 1), torch.stack(vls, 1)
        # 가치 목표: 종료 시점은 0, 그 밖에는 r_{t+1} + γ V_{t+1} (다음 시점이 조각 안에 있을 때만)
        nxt_ok = torch.cat([mask[:, 1:], torch.zeros(B, 1)], 1)
        tgt_next = torch.cat([r[:, 1:] + gamma * v[:, 1:].detach() * (1 - done[:, 1:]), torch.zeros(B, 1)], 1)
        tgt = torch.where(done.bool(), torch.zeros_like(v), tgt_next)
        vmask = mask * torch.where(done.bool(), torch.ones_like(v), nxt_ok)
        vloss = gauss_nll(tgt, v, vls, smin)
        F_t = pix + rew + kl
        n = mask.sum().clamp_min(1)
        total = (F_t * mask).sum() / n + (vloss * vmask).sum() / vmask.sum().clamp_min(1)
        return total, dict(F=((F_t * mask).sum() / n).item(), pix=((pix * mask).sum() / n).item(),
                           kl=((kl * mask).sum() / n).item(), v=((vloss * vmask).sum() / vmask.sum().clamp_min(1)).item())


class BTSP:
    """시냅스 W [단위 3·n_per, 시냅스전 수]. Milstein et al. 2021 식 1을 스텝 단위로 (모듈 설명 참고)."""

    def __init__(self, n_units, n_pre, k_plus=0.5, k_minus=0.07, w_max=1.0, a_plus=0.24, b_plus=30.0,
                 a_minus=0.09, b_minus=200.0, tau_et=3.0, tau_is=2.0):
        self.W = torch.zeros(n_units, n_pre)
        self.kp, self.km, self.wmax = k_plus, k_minus, w_max
        self.ap, self.bp, self.am, self.bm = a_plus, b_plus, a_minus, b_minus
        self.r_et, self.r_is = math.exp(-1 / tau_et), math.exp(-1 / tau_is)
        self.reset_traces(n_units, n_pre)

    def reset_traces(self, n_units=None, n_pre=None):
        self.ET = torch.zeros(n_pre or self.W.shape[1])
        self.IS = torch.zeros(n_units or self.W.shape[0])

    def q(self, om, a, b):
        lo = torch.sigmoid(torch.tensor(-b * a))
        return ((torch.sigmoid(b * (om - a)) - lo) / (1 - lo)).clamp_min(0)

    def write(self):
        if self.IS.max() < 1e-3:
            return
        om = self.IS[:, None] * self.ET[None, :]
        W = self.W
        self.W = (W + self.kp * (self.wmax - W) * self.q(om, self.ap, self.bp)
                  - self.km * W * self.q(om, self.am, self.bm)).clamp(0, self.wmax)

    def plateau(self, p, sub_steps=3):
        """시점 t의 플래토 p (0/1). ET는 t−1까지의 입력을 담고 있다. 플래토가 sub_steps 동안 지속되는 동안 W를
        갱신하고, 그사이 IS·ET는 감쇠한다. 플래토가 없으면 IS의 감쇠와 남은 IS에 의한 갱신만 한다."""
        self.IS = torch.maximum(self.r_is * self.IS, p)
        for i in range(sub_steps if p.any() else 1):
            self.write()
            if i < sub_steps - 1 and p.any():
                self.IS, self.ET = self.r_is * self.IS, self.r_et * self.ET

    def add_input(self, x):
        """시점 t의 시냅스전 입력 x(o_t, a_t)."""
        self.ET = torch.maximum(self.r_et * self.ET, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniGrid-LavaGapS5-v0")
    ap.add_argument("--policy", choices=["btsp", "random", "value"], default="btsp")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--train_every", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--beta", type=float, default=10.0, help="행동 softmax 정밀도")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--lam_d", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/btsp_tiny")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    env = MemoryEnv(args.env, truncation_as_failure=False)
    m = Model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    btsp = BTSP(3 * m.n_per, m.n_pre)
    pop = {k: slice(i * m.n_per, (i + 1) * m.n_per) for i, k in enumerate(POPS)}
    buf = Buffer(args.steps + 2000, "cpu")
    hist_S = deque(maxlen=2000)
    n_plateau = {k: 0 for k in POPS}
    recent, log = deque(maxlen=50), []
    name = f"{args.policy}_s{args.seed}"
    out = Path(args.out) / name; out.mkdir(parents=True, exist_ok=True)
    steps, updates, ep_i, t0, last = 0, 0, 0, time.time(), {}

    while steps < args.steps:
        o = env.reset(10_000 * (args.seed + 1) + ep_i); ep_i += 1
        rec = dict(grid=[], dir=[], outcome=[], action=[], segment=[])
        tab = m.tables()
        with torch.no_grad():
            state = m.init_state(1)
            prev, V_prev = None, None
            btsp.reset_traces()
            while True:
                g = torch.as_tensor(o["grid"]).long()[None]
                g_oh, d = grid_onehot(g), torch.tensor([o["dir"]])
                out_t = torch.tensor([o["outcome"]])
                # ① 기억 갱신
                if prev is None:
                    z = torch.zeros(1, dtype=torch.long)
                    inj = m.inject(torch.zeros_like(g_oh), z, z, z, torch.zeros(1, m.s_dim), torch.zeros(1, dtype=torch.bool))
                else:
                    inj = m.inject(*prev, torch.ones(1, dtype=torch.bool))
                state = m.step(state, inj, tab)
                H = state[0]
                f = m.feat(H)
                y = m.field(btsp.W, m.presyn(prev[0], prev[3])) if prev is not None else torch.zeros(1, 3 * m.n_per)
                # ② 관측 전 예측과 픽셀 놀람
                pm, _ = m.gauss(m.prior(torch.cat([f, y], -1)))
                dp = m.decode(pm, f)
                S = gauss_nll(g_oh.flatten(1) + 0.5, dp["pix_mu"], dp["pix_ls"], 0.05).mean().item()
                # ③ 사후, 가치, TD 오차
                e = m.enc(m.obs_vec(g_oh, d))
                qm, qs = m.gauss(m.post(torch.cat([f, y, e], -1)))
                s = qm + qs * torch.randn_like(qm)
                V = m.decode(s, f)["v_mu"].item()
                r = float(reward_of(out_t).item())
                done = o["outcome"] != PROGRESS
                delta = None if V_prev is None else r + args.gamma * V * (1 - done) - V_prev
                # ④ 플래토 (이산)
                p = torch.zeros(3 * m.n_per)
                gates = []
                if steps > args.warmup and len(hist_S) > 200 and S > np.percentile(hist_S, 97):
                    gates.append("N")
                if delta is not None and delta > 0.3:
                    gates.append("G")
                if delta is not None and delta < -0.3:
                    gates.append("D")
                for k in gates:
                    sl = pop[k]
                    idx = torch.argsort(y[0, sl])[:2] + sl.start
                    p[idx] = 1.0
                    n_plateau[k] += 1
                hist_S.append(S)
                # ⑤ BTSP 쓰기 (t−1까지의 입력에 공을 돌림)
                btsp.plateau(p)
                rec["grid"].append(o["grid"]); rec["dir"].append(o["dir"]); rec["outcome"].append(o["outcome"])
                rec["segment"].append(0)
                if done or len(rec["dir"]) > env.max_steps:
                    rec["action"].append(0)
                    break
                # ⑥ 행동
                if args.policy == "random":
                    a = int(rng.integers(N_ACTIONS))
                else:
                    sc = []
                    for ai in range(N_ACTIONS):
                        at = torch.tensor([ai])
                        y2 = m.field(btsp.W, m.presyn(g_oh, at))
                        if args.policy == "btsp":
                            sc.append(y2[0, pop["G"]].mean() - args.lam_d * y2[0, pop["D"]].mean())
                        else:                                # 가치 채널의 한 스텝 예측 (사전 평균)
                            st2 = m.step(state, m.inject(g_oh, d, out_t, at, s, torch.ones(1, dtype=torch.bool)), tab)
                            f2 = m.feat(st2[0])
                            pm2, _ = m.gauss(m.prior(torch.cat([f2, y2], -1)))
                            sc.append(m.decode(pm2, f2)["v_mu"][0])
                    pi = (1 - args.eps) * torch.softmax(args.beta * torch.stack(sc), 0) + args.eps / N_ACTIONS
                    a = int(torch.multinomial(pi, 1, generator=gen).item())
                btsp.add_input(m.presyn(g_oh, torch.tensor([a]))[0])
                rec["action"].append(a)
                prev = (g_oh, d, out_t, torch.tensor([a]), s)
                V_prev = V
                o, _, _ = env.step(a)
                steps += 1
        L = len(rec["dir"]) - 1
        buf.add(rec)
        recent.append((rec["outcome"][-1] == SUCCESS, rec["outcome"][-1] == FAILURE, L))
        # 재현 학습
        if steps > args.warmup:
            for _ in range(max(1, L // args.train_every)):
                batch = buf.sample(args.batch, args.seq, gen)
                loss, last = m.loss(batch, btsp.W, args.gamma)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0); opt.step()
                updates += 1
        if not log or steps // 1000 > log[-1]["steps"] // 1000:
            W = btsp.W
            recd = dict(steps=steps, episodes=ep_i, updates=updates,
                        success=float(np.mean([r[0] for r in recent])), failure=float(np.mean([r[1] for r in recent])),
                        ep_len=float(np.mean([r[2] for r in recent])), plateaus=dict(n_plateau),
                        W_mean={k: float(W[pop[k]].mean()) for k in POPS}, sec=round(time.time() - t0, 1),
                        **{k: round(v, 3) for k, v in last.items()})
            log.append(recd)
            print(json.dumps(recd, ensure_ascii=False), flush=True)
            (out / "log.json").write_text(json.dumps(log))
    torch.save(dict(model=m.state_dict(), W=btsp.W), out / "final.pt")


if __name__ == "__main__":
    main()
