"""공통 novelty 모듈: 우도 파라미터에 대한 정보 이득 I(θ; x_{t+1} | x_t, a_t). 설계: docs/fep/exp2f_design.md

설계 2판 5.2절의 제안을 구현한다. 세 월드 모델에 똑같이 붙는 작은 앙상블이고, 입력은 실제 격자 원-핫 x_t와
행동 a_t뿐이다(월드 모델과 무관). 출력은 다음 격자의 칸별·속성 묶음별 범주형 로짓이다. 멤버는 초기화만 다르다.
"월드 모델 파라미터의 novelty"를 "공통 대리 모델 파라미터의 novelty"로 대체한 것이다 [근사(대체)].

정보 이득 (θ는 K개 멤버 위의 균등 분포, 1단계 5.4절):
  I(θ; x) = H[p̄(x)] − E_k H[p_k(x)] = E_k E_{x∼p_k} [ln p_k(x) − ln p̄(x)],   p̄ = (1/K) Σ_j p_j
- mi_joint (기본): 멤버 k마다 x를 뽑고 모든 멤버의 ln p_j(x)를 정확히 계산하는 몬테카를로. 불편 추정량이고
  기댓값은 [0, ln K] 안에 있다. 멤버가 칸별 범주형이라 멤버별 로그밀도가 정확하기 때문에 가능하다.
- mi_cellsum: Σ_칸 Σ_묶음 JSD. 닫힌 식이지만 결합 정보량의 **상한**이다.
  멤버가 칸별로 분해되므로 E_k H[p_k(x)] = Σ_칸 E_k H[p_k(x_칸)]이고, H[p̄(x)] ≤ Σ_칸 H[p̄(x_칸)] (부분 가법성).
  설계 2판 5.2절의 "칸 독립 가정 아래 Σ_칸 JSD로 정확히 계산된다"는 이 상한을 가리킨다.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from rltdp.common.minigrid_env import GROUP_SIZES, N_ACTIONS
from rltdp.fep.wm.common import HW, N_CH, grid_logprob, grid_onehot


def group_log_softmax(logits, group_sizes=GROUP_SIZES, dim=-3):
    """속성 묶음마다 log_softmax. 범주 축 = dim."""
    return torch.cat([F.log_softmax(g, dim) for g in torch.split(logits, list(group_sizes), dim)], dim)


def sample_onehot(logp, group_sizes=GROUP_SIZES, dim=-3, generator=None):
    """묶음마다 범주형 샘플을 원-핫으로 (검벨 최댓값)."""
    outs = []
    for g in torch.split(logp, list(group_sizes), dim):
        u = torch.rand(g.shape, device=g.device, generator=generator).clamp_min(1e-20)
        idx = (g - torch.log(-torch.log(u))).argmax(dim)
        outs.append(F.one_hot(idx, g.shape[dim]).movedim(-1, dim).to(g.dtype))
    return torch.cat(outs, dim)


def mi_joint(logits, group_sizes=GROUP_SIZES, n_samples=1, generator=None):
    """결합 정보량 I(θ; x)의 불편 추정. logits (N, K, C, H, W) → (N,).
    멤버 k에서 x^(k)를 뽑고 ln p_k(x^(k)) − ln p̄(x^(k))를 k에 대해 평균한다 (n_samples번 반복)."""
    lp = group_log_softmax(logits, group_sizes)
    K = lp.shape[1]
    total = 0.0
    for _ in range(n_samples):
        x = sample_onehot(lp, group_sizes, generator=generator)        # x[:, k] ∼ p_k
        lpj = torch.einsum("nkchw,njchw->nkj", x, lp)                 # ln p_j(x^(k))
        own = torch.diagonal(lpj, dim1=1, dim2=2)                       # ln p_k(x^(k))
        total = total + (own - torch.logsumexp(lpj, -1) + math.log(K)).mean(1)
    return total / n_samples


def mi_cellsum(logits, group_sizes=GROUP_SIZES):
    """Σ_칸 Σ_묶음 JSD (결합 정보량의 상한, 닫힌 식). logits (N, K, C, H, W) → (N,)."""
    lp = group_log_softmax(logits, group_sizes)
    K = lp.shape[1]
    lpbar = torch.logsumexp(lp, 1) - math.log(K)
    H_bar = -(lpbar.exp() * lpbar).flatten(1).sum(1)
    H_mem = -(lp.exp() * lp).flatten(2).sum(2).mean(1)
    return H_bar - H_mem


def _planes(x_oh, a):
    """[격자 원-핫 20, 행동 원-핫 3 (모든 칸 같은 값)] = 23채널."""
    a_oh = F.one_hot(a, N_ACTIONS).float()[:, :, None, None].expand(-1, -1, HW, HW)
    return torch.cat([x_oh, a_oh], 1)


class _Member(nn.Module):
    """멤버 하나: 국소 합성곱 + 전역 경로. 회전하면 칸이 시야 반대편으로 옮겨 가므로 3×3 합성곱만으로는
    수용 영역이 모자라서 전역 경로(평탄화 → 선형 → 칸 배치)를 둔다."""

    def __init__(self, hidden=32, glob=256):
        super().__init__()
        self.local = nn.Sequential(nn.Conv2d(N_CH + N_ACTIONS, hidden, 3, padding=1), nn.ELU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=1), nn.ELU())
        self.glob = nn.Sequential(nn.Flatten(), nn.Linear(hidden * HW * HW, glob), nn.ELU(),
                                  nn.Linear(glob, hidden * HW * HW), nn.ELU())
        self.out = nn.Sequential(nn.Conv2d(2 * hidden, 64, 3, padding=1), nn.ELU(),
                                 nn.Conv2d(64, N_CH, 3, padding=1))

    def forward(self, planes):
        loc = self.local(planes)
        return self.out(torch.cat([loc, self.glob(loc).view_as(loc)], 1))


class NoveltyEnsemble(nn.Module):
    def __init__(self, K=5, lr=3e-4, mode="joint", n_samples=1):
        super().__init__()
        assert mode in ("joint", "cellsum")
        self.members = nn.ModuleList(_Member() for _ in range(K))
        self.K, self.mode, self.n_samples = K, mode, n_samples
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def logits(self, x_oh, a):
        """x_oh (N, 20, 7, 7), a (N,) → (N, K, 20, 7, 7)."""
        planes = _planes(x_oh, a)
        return torch.stack([m(planes) for m in self.members], 1)

    def train_step(self, batch):
        """재현 버퍼 조각에서 (x_t, a_t) → x_{t+1} 쌍으로 학습한다. 멤버들은 같은 배치를 본다 (초기화만 다름).
        반환: 멤버 평균 음의 로그우도 (nats/관측)."""
        ok = batch["mask"][:, 1:]
        x = grid_onehot(batch["grid"][:, :-1])[ok]
        a = batch["action"][:, :-1][ok]
        tgt = batch["grid"][:, 1:][ok]
        lp = grid_logprob(self.logits(x, a), tgt[:, None].expand(-1, self.K, -1, -1, -1))   # (M, K)
        loss = -lp.mean(0).sum()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
        self.opt.step()
        return loss.item() / self.K

    @torch.no_grad()
    def info_gain(self, x_oh, a):
        """I(θ; x_{t+1} | x_t, a_t), (N,) nats."""
        lg = self.logits(x_oh, a)
        return mi_joint(lg, n_samples=self.n_samples) if self.mode == "joint" else mi_cellsum(lg)
