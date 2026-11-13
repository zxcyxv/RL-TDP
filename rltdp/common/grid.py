"""계열 공통: 자기중심 7×7 격자 관측(MiniGrid, 칸마다 물체·색·상태 원-핫 20채널)의 텐서 유틸과 작은 MLP."""
import torch
import torch.nn.functional as F
from torch import nn

from rltdp.common.minigrid_env import GROUP_SIZES

N_DIR, N_OUTCOME, N_CH, HW = 4, 3, sum(GROUP_SIZES), 7


def grid_onehot(grid_idx):
    """(..., 7, 7, 3) long → (..., 20, 7, 7) float."""
    parts = [F.one_hot(grid_idx[..., i], n) for i, n in enumerate(GROUP_SIZES)]
    return torch.cat(parts, -1).movedim(-1, -3).float()


def grid_logprob(logits, grid_idx):
    """칸마다 속성 묶음별 범주형 로그확률의 합. logits (..., 20, 7, 7), grid_idx (..., 7, 7, 3) → (...)."""
    total, s = 0.0, 0
    for i, n in enumerate(GROUP_SIZES):
        lp = F.log_softmax(logits[..., s:s + n, :, :], dim=-3)
        total = total + lp.gather(-3, grid_idx[..., i].unsqueeze(-3)).squeeze(-3).flatten(-2).sum(-1)
        s += n
    return total


def grid_from_logits(logits, sample: bool, gen=None):
    """범주형 로짓 → 인덱스 격자 (..., 7, 7, 3). sample=False면 최빈값."""
    outs, s = [], 0
    for n in GROUP_SIZES:
        lg = logits[..., s:s + n, :, :].movedim(-3, -1)
        if sample:
            g = -torch.log(-torch.log(torch.rand(lg.shape, device=lg.device, generator=gen).clamp_min(1e-20)))
            outs.append((lg + g).argmax(-1))
        else:
            outs.append(lg.argmax(-1))
        s += n
    return torch.stack(outs, -1)


def onehot_to_idx(x_oh):
    """(..., 20, 7, 7) 원-핫 → (..., 7, 7, 3) 인덱스."""
    outs, s = [], 0
    for n in GROUP_SIZES:
        outs.append(x_oh[..., s:s + n, :, :].argmax(-3)); s += n
    return torch.stack(outs, -1)


def cat_sample(logits, sample, gen=None):
    if not sample:
        return logits.argmax(-1)
    g = -torch.log(-torch.log(torch.rand(logits.shape, device=logits.device, generator=gen).clamp_min(1e-20)))
    return (logits + g).argmax(-1)


def mlp(d_in, d_out, hidden=256):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, d_out))
