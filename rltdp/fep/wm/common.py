"""두 모델이 공유하는 골격: 격자 임베더 + GRU 기억 + 범주형 머리 유틸.

h_t = GRU(h_{t-1}, [emb(o_{t-1}), onehot(a_{t-1})]),  h_0 = 0.  즉 h_t로 o_t를 예측한다.
"""
import torch
import torch.nn.functional as F
from torch import nn

from rltdp.common.grid import (HW, N_CH, N_DIR, N_OUTCOME, cat_sample, grid_from_logits,  # noqa: F401 (재수출)
                               grid_logprob, grid_onehot, mlp, onehot_to_idx)
from rltdp.common.minigrid_env import GROUP_SIZES, N_ACTIONS  # noqa: F401


class Skeleton(nn.Module):
    def __init__(self, emb_dim=256, h_dim=256):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(N_CH, 64, 3, padding=1), nn.ELU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ELU())
        self.lin = nn.Linear(64 * HW * HW + N_DIR, emb_dim)
        self.gru = nn.GRUCell(emb_dim + N_ACTIONS, h_dim)
        self.h_dim = h_dim

    def embed(self, grid_oh, direction):
        """grid_oh (..., 20, 7, 7), direction (...) → (..., 256)."""
        lead = grid_oh.shape[:-3]
        c = self.conv(grid_oh.reshape(-1, N_CH, HW, HW)).flatten(1)
        d = F.one_hot(direction.reshape(-1), N_DIR).float()
        return self.lin(torch.cat([c, d], -1)).view(*lead, -1)

    def step(self, h, e_prev, a_prev):
        return self.gru(torch.cat([e_prev, F.one_hot(a_prev, N_ACTIONS).float()], -1), h)

    def rollout(self, grid_oh, direction, action):
        """관측 시퀀스 (B, L, ...) → h (B, L, 256), e (B, L, 256). h[:, 0] = 0."""
        B, L = direction.shape
        e = self.embed(grid_oh, direction)
        h = torch.zeros(B, self.h_dim, device=e.device)
        hs = [h]
        for t in range(1, L):
            h = self.step(h, e[:, t - 1], action[:, t - 1])
            hs.append(h)
        return torch.stack(hs, 1), e


def prev_context(x_oh, direction, action):
    """직전 관측 문맥. 시점 t에 o_{t-1}(원-핫), 방향_{t-1}, a_{t-1}을 둔다. 조각의 첫 스텝은 직전이 없으므로
    0으로 두고 has = 0 (조각 시작에서 h = 0으로 두는 규칙과 맞춤). 입력 (B, L, ...) → 같은 앞 축."""
    pg = torch.zeros_like(x_oh); pg[:, 1:] = x_oh[:, :-1]
    pd = torch.zeros_like(direction); pd[:, 1:] = direction[:, :-1]
    pa = torch.zeros_like(action); pa[:, 1:] = action[:, :-1]
    has = torch.zeros(direction.shape, device=x_oh.device); has[:, 1:] = 1
    return dict(pg=pg, pd=pd, pa=pa, has=has)


def ctx_select(ctx, idx):
    """문맥의 앞 축에 같은 인덱싱(마스크 또는 슬라이스)을 적용한다."""
    return {k: v[idx] for k, v in ctx.items()}


def ctx_flat(ctx):
    return {k: v.reshape(-1, *v.shape[2:]) for k, v in ctx.items()}


def ctx_planes(ctx):
    """직전 문맥을 칸 위치 그대로의 채널로: [직전 원-핫 20, 직전 행동 3(모든 칸 같은 값), 직전 있음 1] = 24채널."""
    has = ctx["has"].float()
    a_oh = F.one_hot(ctx["pa"], N_ACTIONS).float() * has[:, None]
    return torch.cat([ctx["pg"] * has[:, None, None, None],
                      a_oh[:, :, None, None].expand(-1, -1, HW, HW),
                      has[:, None, None, None].expand(-1, 1, HW, HW)], 1)


def ctx_vec(ctx):
    """방향·결과 머리용: [직전 방향 원-핫 4, 직전 행동 원-핫 3, 직전 있음 1] = 8."""
    has = ctx["has"].float()
    return torch.cat([F.one_hot(ctx["pd"], N_DIR).float() * has[:, None],
                      F.one_hot(ctx["pa"], N_ACTIONS).float() * has[:, None], has[:, None]], -1)


CTX_PLANES, CTX_VEC = N_CH + N_ACTIONS + 1, N_DIR + N_ACTIONS + 1


class SkipDecoder(nn.Module):
    """디코더: 벡터 입력(h 또는 [s, h])에 더해 직전 시야를 칸 위치 그대로 합성곱 경로로 받는다.
    격자: [선형(vec) → 64×7×7] ⊕ [합성곱(직전 원-핫 20 + 직전 행동 3 + has 1) → 64] → ELU → conv 64 → ELU → conv 20
    방향·결과 머리: [vec, 직전 방향 원-핫, 직전 행동 원-핫, has] → MLP"""

    def __init__(self, vec_dim):
        super().__init__()
        self.vec = nn.Linear(vec_dim, 64 * HW * HW)
        self.sp = nn.Conv2d(N_CH + N_ACTIONS + 1, 64, 3, padding=1)
        self.conv = nn.Sequential(nn.ELU(), nn.Conv2d(128, 64, 3, padding=1), nn.ELU(),
                                  nn.Conv2d(64, N_CH, 3, padding=1))
        hin = vec_dim + N_DIR + N_ACTIONS + 1
        self.dir_head = mlp(hin, N_DIR)
        self.out_head = mlp(hin, N_OUTCOME)

    def forward(self, vec, ctx):
        has = ctx["has"].float()
        a_oh = F.one_hot(ctx["pa"], N_ACTIONS).float() * has[:, None]
        d_oh = F.one_hot(ctx["pd"], N_DIR).float() * has[:, None]
        planes = torch.cat([ctx["pg"] * has[:, None, None, None],
                            a_oh[:, :, None, None].expand(-1, -1, HW, HW),
                            has[:, None, None, None].expand(-1, 1, HW, HW)], 1)
        x = torch.cat([self.vec(vec).view(-1, 64, HW, HW), self.sp(planes)], 1)
        hv = torch.cat([vec, d_oh, a_oh, has[:, None]], -1)
        return self.conv(x), self.dir_head(hv), self.out_head(hv)
