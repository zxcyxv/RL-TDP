"""상상 속 actor-critic (DreamerV3 단순판). 보상 = Σ_y P(y)·ln C(y) + β·IG. 설계: docs/fep/exp2d_design.md"""
import torch
import torch.nn.functional as F
from torch import nn

from rltdp.common.minigrid_env import FAILURE, N_ACTIONS, PROGRESS, SUCCESS
from rltdp.fep.agent.imagine import predict_step
from rltdp.fep.wm.common import grid_onehot, mlp

LN_C = torch.tensor([0.0, 1.0, -1.0])       # 진행, 성공, 실패 순 (outcome 인덱스와 같음)
assert (PROGRESS, SUCCESS, FAILURE) == (0, 1, 2)


class ActorCritic(nn.Module):
    def __init__(self, feat_dim=512, horizon=15, gamma=0.985, lam=0.95, ent=3e-4, lr=3e-5, beta=0.0):
        super().__init__()
        self.actor = mlp(feat_dim, N_ACTIONS)
        self.critic = mlp(feat_dim, 1)
        self.H, self.gamma, self.lam, self.ent, self.beta = horizon, gamma, lam, ent, beta
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)
        self.register_buffer("ret_scale", torch.tensor(1.0))

    def policy(self, feat):
        return torch.distributions.Categorical(logits=self.actor(feat))

    def act(self, feat, greedy=False):
        d = self.policy(feat)
        return d.probs.argmax(-1) if greedy else d.sample()

    def imagine(self, wm, h, grid, direction, prev):
        """출발: 실제 상태 (h_t, o_t). prev = 직전 문맥 (pg, pd, pa, has)는 o_t를 예측할 때의 것이라 쓰지 않는다.
        반환: 특징 f_0..f_H, 행동 a_0..a_{H-1}, 보상 r_1..r_H, 생존 c_1..c_H, 진단값."""
        feats, acts, rews, conts, igs, prags = [], [], [], [], [], []
        x = grid_onehot(grid)
        e = wm.sk.embed(x, direction)
        for k in range(self.H):
            f = torch.cat([h, e], -1)
            a = self.act(f.detach())
            feats.append(f); acts.append(a)
            h = wm.sk.step(h, e, a)
            state = dict(h=h, pg=x, pd=direction, pa=a, has=torch.ones(h.shape[0], device=h.device))
            p = predict_step(wm, state, need_ig=bool(self.beta))
            prag = (p["out_logp"].exp() * LN_C.to(h.device)).sum(-1)
            rews.append(prag + self.beta * p["ig"] if self.beta else prag); igs.append(p["ig"]); prags.append(prag)
            conts.append((p["out"] == PROGRESS).float())
            x = grid_onehot(p["grid"]); direction = p["dir"]
            e = wm.sk.embed(x, direction)
        feats.append(torch.cat([h, e], -1))
        ig = torch.stack(igs)
        ig_f = ig[torch.isfinite(ig)]
        return (torch.stack(feats).detach(), torch.stack(acts), torch.stack(rews), torch.stack(conts),
                dict(ig=ig_f.mean().item() if len(ig_f) else float('nan'), ig_nonfinite=int((~torch.isfinite(ig)).sum()),
                     prag=torch.stack(prags).mean().item()))

    def update(self, wm, h, grid, direction):
        with torch.no_grad():
            feats, acts, rews, conts, diag = self.imagine(wm, h, grid, direction, None)
            v = self.critic(feats).squeeze(-1)                         # (H+1, N)
            # λ-리턴: R_k = r_{k+1} + γ c_{k+1} [(1−λ) V_{k+1} + λ R_{k+1}],  R_H = V_H
            R = [v[-1]]
            for k in reversed(range(self.H)):
                R.append(rews[k] + self.gamma * conts[k] * ((1 - self.lam) * v[k + 1] + self.lam * R[-1]))
            R = torch.stack(R[::-1][:-1])                              # (H, N), R_0..R_{H-1}
            # 상상이 끝난 뒤의 스텝은 가중치 0: w_k = Π_{j<k} c_j
            w = torch.cumprod(torch.cat([torch.ones_like(conts[:1]), conts[:-1]]), 0)
            lo, hi = torch.quantile(R.flatten(), 0.05), torch.quantile(R.flatten(), 0.95)
            self.ret_scale.mul_(0.99).add_(0.01 * (hi - lo))
            scale = torch.clamp(self.ret_scale, min=1.0)
        f = feats[:-1]
        val = self.critic(f).squeeze(-1)
        loss_c = (w * (val - R) ** 2).sum() / w.sum()
        pol = self.policy(f)
        adv = ((R - val.detach()) / scale)
        loss_a = -(w * (adv * pol.log_prob(acts) + self.ent * pol.entropy())).sum() / w.sum()
        self.opt_c.zero_grad(set_to_none=True); loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100); self.opt_c.step()
        self.opt_a.zero_grad(set_to_none=True); loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100); self.opt_a.step()
        diag.update(loss_c=loss_c.item(), ent=(w * pol.entropy()).sum().item() / w.sum().item(),
                    ret=(w * R).sum().item() / w.sum().item(), alive=w.mean().item())
        return diag


class ActorCriticKL(nn.Module):
    """정확한 KL 정책 손실 + 흡수 종료 (docs/fep/agent_design_v2.md 6·7절).

    버퍼의 실제 믿음 상태 b = (h_t, o_t)에서 세 행동을 모두 한 스텝 내다본다.
      Q(b, a) = Σ_y P(y | b, a)·u(y) + γ·P(진행 | b, a)·V̄(b′),   u(진행) = 0, u(종료 y) = ln C(y)/(1−γ)
    (Q는 −G 단위. 종료는 흡수 상태라 ln C(y)가 이후 모든 시점에 쌓인 기하 합)
      정책: min KL[π(·|b) ‖ softmax(Q(b,·)/α)],  α = 1/γ_π
      가치: V(b) ← E_{a∼π}[Q(b, a)]  (정교한 추론 형태, 엔트로피가 가치에 쌓이지 않음)
    V̄는 느린 목표 가치망 (EMA). 부트스트랩 안정화용 추정량."""

    def __init__(self, feat_dim=512, gamma=0.985, ent=1.0, lr=3e-4, tau=0.02, beta=0.0):
        super().__init__()
        import copy
        self.actor = mlp(feat_dim, N_ACTIONS)
        self.critic = mlp(feat_dim, 1)
        self.critic_t = copy.deepcopy(self.critic).requires_grad_(False)
        self.gamma, self.ent, self.tau, self.beta = gamma, ent, tau, beta
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        u = LN_C / (1 - gamma)
        u[PROGRESS] = 0.0
        self.register_buffer("u_term", u)

    def policy(self, feat):
        return torch.distributions.Categorical(logits=self.actor(feat))

    def act(self, feat, greedy=False):
        d = self.policy(feat)
        return d.probs.argmax(-1) if greedy else d.sample()

    @torch.no_grad()
    def q_values(self, wm, h, grid, direction):
        x = grid_onehot(grid)
        e = wm.sk.embed(x, direction)
        N = h.shape[0]
        Q, P_term = [], []
        for a_i in range(N_ACTIONS):
            a = torch.full((N,), a_i, device=h.device, dtype=torch.long)
            h2 = wm.sk.step(h, e, a)
            st = dict(h=h2, pg=x, pd=direction, pa=a, has=torch.ones(N, device=h.device))
            p = predict_step(wm, st, need_ig=False)
            py = p["out_logp"].exp()
            e2 = wm.sk.embed(grid_onehot(p["grid"]), p["dir"])
            v_next = self.critic_t(torch.cat([h2, e2], -1)).squeeze(-1)
            Q.append((py * self.u_term).sum(-1) + self.gamma * py[:, PROGRESS] * v_next)
            P_term.append(1 - py[:, PROGRESS])
        return torch.stack(Q, -1), torch.stack(P_term, -1), torch.cat([h, e], -1)

    def update(self, wm, h, grid, direction):
        Q, P_term, feat = self.q_values(wm, h, grid, direction)
        logits = self.actor(feat)
        logp = F.log_softmax(logits, -1)
        pi = logp.exp()
        target = F.log_softmax(Q / self.ent, -1)
        loss_a = (pi * (logp - target)).sum(-1).mean()                 # KL[π ‖ softmax(Q/α)]
        v_target = (pi.detach() * Q).sum(-1)
        loss_c = F.mse_loss(self.critic(feat).squeeze(-1), v_target)
        self.opt_a.zero_grad(set_to_none=True); loss_a.backward(); self.opt_a.step()
        self.opt_c.zero_grad(set_to_none=True); loss_c.backward(); self.opt_c.step()
        with torch.no_grad():
            for pt, p in zip(self.critic_t.parameters(), self.critic.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        return dict(loss_a=loss_a.item(), loss_c=loss_c.item(), ent=-(pi * logp).sum(-1).mean().item(),
                    ret=v_target.mean().item(), q_spread=(Q.max(-1).values - Q.min(-1).values).mean().item(),
                    p_term_fwd=P_term[:, 2].mean().item(), ig=float("nan"), prag=float("nan"), alive=float("nan"))
