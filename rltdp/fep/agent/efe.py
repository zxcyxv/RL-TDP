"""선호분포 C + 기대 자유에너지 G로 행동을 고르는 actor-critic. 설계: docs/fep/exp2f_design.md

ActorCriticKL(설계 2판 6·7절: 흡수 종료, KL 정책, 정교한 추론형 가치)과 정책·가치 학습은 같다. 바뀌는 것:
1. 선호를 정규화된 분포로 둔다: C(y) = softmax(c / λ_C), y ∈ {진행, 성공, 실패}. 격자·방향은 균등(상수)이다.
2. 진행 스텝마다 ln C(진행)을 더한다. ActorCriticKL은 ln C(진행) = 0을 가정하고 이 항을 뺐다.
   흡수 종료에서는 ln C 전체에 상수 k를 더하면 모든 Q가 k/(1−γ)만큼 같이 움직이므로, β = 0이고 c(진행) = 0이면
   ActorCriticKL과 같은 목적이다 (tests/fep/test_efe_agent.py).
3. 탐색 항: 공통 novelty 모듈의 정보 이득 β·I(θ; x_{t+1} | x_t, a)를 더한다 (rltdp/fep/agent/novelty.py).

Q는 −G 단위다. 한 스텝 내다본 뒤 실제 믿음 상태 b = (h_t, o_t)에서
  Q(b, a) = Σ_{y 종료} P(y | b, a)·ln C(y)/(1−γ) + P(진행 | b, a)·[ln C(진행) + γ·V̄(b′)] + β·I(θ; x_{t+1} | x_t, a)
종료는 흡수 상태라 ln C(y)가 이후 모든 시점에 쌓인 기하 합이 된다. 흡수 이후의 novelty는 0이다.
"""
import torch
import torch.nn.functional as F
from torch import nn

from rltdp.common.minigrid_env import N_ACTIONS, PROGRESS
from rltdp.fep.agent.ac import ActorCriticKL
from rltdp.fep.agent.imagine import predict_step
from rltdp.fep.agent.novelty import NoveltyEnsemble
from rltdp.fep.wm.common import grid_onehot


class Preference(nn.Module):
    """결과 위의 선호분포 C(y) = softmax(c / λ_C). c는 상대 로그 선호 (상수를 더해도 C는 같다),
    λ_C는 선호 정밀도의 역수 (ln C = r/λ_C + 상수 관례에서 r = c)."""

    def __init__(self, c=(0.0, 1.0, -1.0), lam_c=1.0):
        super().__init__()
        self.lam_c = float(lam_c)
        self.register_buffer("c", torch.tensor(c, dtype=torch.float32))
        self.register_buffer("ln_c", F.log_softmax(self.c / self.lam_c, -1))


def efe_q(py, ln_c, v_next, nov, gamma, beta):
    """Q(b, a) (−G 단위). py (N, 3) = P(y | b, a), ln_c (3,), v_next (N,) = V̄(b′), nov (N,) 또는 0."""
    is_prog = torch.arange(len(ln_c), device=ln_c.device) == PROGRESS
    u = torch.where(is_prog, ln_c, ln_c / (1 - gamma))
    return (py * u).sum(-1) + gamma * py[:, PROGRESS] * v_next + beta * nov


class ActorCriticEFE(ActorCriticKL):
    def __init__(self, feat_dim=512, gamma=0.985, ent=1.0, lr=3e-4, tau=0.02, beta=0.0,
                 pref=(0.0, 1.0, -1.0), lam_c=1.0, nov_k=5, nov_mode="joint", nov_samples=1):
        super().__init__(feat_dim=feat_dim, gamma=gamma, ent=ent, lr=lr, tau=tau, beta=beta)
        del self.u_term                                   # ActorCriticKL의 ln C(진행) = 0 가정은 쓰지 않는다
        self.pref = Preference(pref, lam_c)
        # actor·critic을 먼저 만든 뒤 novelty를 만든다: β = 0이면 같은 시드에서 ActorCriticKL과 초기값이 같다
        self.novelty = NoveltyEnsemble(K=nov_k, mode=nov_mode, n_samples=nov_samples) if beta else None
        self._diag = {}

    @torch.no_grad()
    def q_values(self, wm, h, grid, direction):
        x = grid_onehot(grid)
        e = wm.sk.embed(x, direction)
        N = h.shape[0]
        Q, P_term, novs, prags = [], [], [], []
        for a_i in range(N_ACTIONS):
            a = torch.full((N,), a_i, device=h.device, dtype=torch.long)
            h2 = wm.sk.step(h, e, a)
            st = dict(h=h2, pg=x, pd=direction, pa=a, has=torch.ones(N, device=h.device))
            p = predict_step(wm, st, need_ig=False)
            py = p["out_logp"].exp()
            e2 = wm.sk.embed(grid_onehot(p["grid"]), p["dir"])
            v_next = self.critic_t(torch.cat([h2, e2], -1)).squeeze(-1)
            nov = self.novelty.info_gain(x, a) if self.novelty is not None else torch.zeros(N, device=h.device)
            Q.append(efe_q(py, self.pref.ln_c, v_next, nov, self.gamma, self.beta))
            P_term.append(1 - py[:, PROGRESS])
            novs.append(nov); prags.append((py * self.pref.ln_c).sum(-1))
        novs = torch.stack(novs, -1)
        self._diag = dict(nov=novs.mean().item(), nov_fwd=novs[:, 2].mean().item(),
                          nov_max=novs.max().item(), prag=torch.stack(prags).mean().item())
        return torch.stack(Q, -1), torch.stack(P_term, -1), torch.cat([h, e], -1)

    def update(self, wm, h, grid, direction):
        diag = super().update(wm, h, grid, direction)
        diag.update(self._diag)
        return diag
