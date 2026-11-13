"""argmax 들어 올림 q(v|x) (Hoogeboom et al. 2021의 임계값 방식). 학습 파라미터 없음 (D4).

기준: 균등 탈양자화 y = x + u. 차이는 범주마다 정수 칸 대신 "그 성분이 argmax인 영역"을 쓴다는 것뿐이다.
그래서 연속 공간 전체가 유효한 범주들의 영역으로 나뉘고, 범주 번호에 순서가 생기지 않는다.

묶음 하나(범주 K개, 정답 x)에 대해
  u ~ N(0, I_K),  T = u_x,  v_x = T,  v_i = T − softplus(T − u_i)  (i ≠ x)
  ln q(v|x) = ln N(u; 0, I) − Σ_{i≠x} ln σ(u_x − u_i)
격자 전체에서는 칸마다, 속성 묶음(물체 11, 색 6, 상태 3)마다 독립으로 적용한다.
"""
import math

import torch
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)


def _split(t, group_sizes, dim):
    return torch.split(t, list(group_sizes), dim=dim)


def lift_group(x_onehot: torch.Tensor, u: torch.Tensor, dim: int, tau: float = 1.0):
    """묶음 하나. x_onehot, u: 같은 shape, 범주 축 = dim. u ~ N(0, τ²I)로 뽑았다고 보고
    v, ln q(v|x)를 돌려준다 (범주 축으로 합친 값). 학습은 τ = 1 (D4). τ ≠ 1은 측정용 제안분포다."""
    T = (u * x_onehot).sum(dim, keepdim=True)
    v = torch.where(x_onehot.bool(), T, T - F.softplus(T - u))
    log_normal = (-0.5 * ((u / tau) ** 2 + LOG2PI) - math.log(tau)).sum(dim)
    log_jac = (F.logsigmoid(T - u) * (1 - x_onehot)).sum(dim)
    return v, log_normal - log_jac


def lift(x_onehot: torch.Tensor, group_sizes, dim: int = -3, generator=None, tau: float = 1.0):
    """x_onehot (..., C, H, W) → v 같은 shape, ln q(v|x) shape (...)."""
    u = tau * torch.randn(x_onehot.shape, generator=generator, device=x_onehot.device, dtype=x_onehot.dtype)
    vs, lq = [], 0.0
    for xg, ug in zip(_split(x_onehot, group_sizes, dim), _split(u, group_sizes, dim)):
        v, l = lift_group(xg, ug, dim, tau)
        vs.append(v)
        lq = lq + l
    return torch.cat(vs, dim), _sum_trailing(lq, dim % x_onehot.dim())


def _sum_trailing(t, keep_dims):
    """앞쪽 keep_dims개 축만 남기고 나머지(공간 축)를 더한다."""
    return t.reshape(*t.shape[:keep_dims], -1).sum(-1) if t.dim() > keep_dims else t


def to_category(v: torch.Tensor, group_sizes, dim: int = -3) -> torch.Tensor:
    """v → 묶음마다 argmax 원-핫. 언제나 유효한 원-핫이 나온다."""
    outs = [F.one_hot(g.argmax(dim), g.shape[dim]).movedim(-1, dim).to(v.dtype)
            for g in _split(v, group_sizes, dim)]
    return torch.cat(outs, dim)


def log_q_given(v: torch.Tensor, x_onehot: torch.Tensor, group_sizes, dim: int = -3):
    """임의의 v에서 ln q(v|x)를 닫힌 식으로 계산한다 (u를 거꾸로 푼다). v가 x의 영역 밖이면 −inf.
    u_x = T = v_x,  u_i = T − ln(e^{T − v_i} − 1)."""
    total = 0.0
    for vg, xg in zip(_split(v, group_sizes, dim), _split(x_onehot, group_sizes, dim)):
        T = (vg * xg).sum(dim, keepdim=True)
        gap = (T - vg).clamp_min(1e-30)
        inv_softplus = gap + torch.log(-torch.expm1(-gap))          # ln(e^{gap} − 1), 안정형
        u = torch.where(xg.bool(), T, T - inv_softplus)
        log_normal = (-0.5 * (u**2 + LOG2PI)).sum(dim)
        log_jac = (F.logsigmoid(T - u) * (1 - xg)).sum(dim)
        inside = ((T - vg) > 0) | xg.bool()
        lq = torch.where(inside.all(dim), log_normal - log_jac, torch.full_like(log_normal, -math.inf))
        total = total + lq
    return _sum_trailing(total, dim % x_onehot.dim())


def lift_gauss(x_onehot: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, group_sizes, dim: int = -3,
               generator=None):
    """학습되는 들어 올림 (Hoogeboom et al. 2021, Algorithm 3 + 학습되는 조건부 대각 가우시안 q(u|x)).
    u ~ N(mu, sigma²), 그다음 임계값 변환은 lift()와 같다. mu = 0, log_sigma = 0이면 고정 q와 정확히 같다.
    반환: v, ln q(v|x) (앞쪽 축만 남김)."""
    eps = torch.randn(x_onehot.shape, generator=generator, device=x_onehot.device, dtype=x_onehot.dtype)
    u = mu + log_sigma.exp() * eps
    log_normal_all = -0.5 * eps**2 - log_sigma - 0.5 * LOG2PI
    vs, lq = [], 0.0
    for xg, ug, lng in zip(_split(x_onehot, group_sizes, dim), _split(u, group_sizes, dim),
                           _split(log_normal_all, group_sizes, dim)):
        T = (ug * xg).sum(dim, keepdim=True)
        vs.append(torch.where(xg.bool(), T, T - F.softplus(T - ug)))
        lq = lq + lng.sum(dim) - (F.logsigmoid(T - ug) * (1 - xg)).sum(dim)
    return torch.cat(vs, dim), _sum_trailing(lq, dim % x_onehot.dim())


def log_q_given_gauss(v: torch.Tensor, x_onehot: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor,
                      group_sizes, dim: int = -3):
    """학습되는 q(v|x)에서 임의의 v의 ln q를 닫힌 식으로 계산 (u를 거꾸로 푼다). v가 x의 영역 밖이면 −inf."""
    total = 0.0
    for vg, xg, mg, lg in zip(_split(v, group_sizes, dim), _split(x_onehot, group_sizes, dim),
                              _split(mu, group_sizes, dim), _split(log_sigma, group_sizes, dim)):
        T = (vg * xg).sum(dim, keepdim=True)
        gap = (T - vg).clamp_min(1e-30)
        u = torch.where(xg.bool(), T, T - (gap + torch.log(-torch.expm1(-gap))))
        log_normal = (-0.5 * ((u - mg) / lg.exp()) ** 2 - lg - 0.5 * LOG2PI).sum(dim)
        log_jac = (F.logsigmoid(T - u) * (1 - xg)).sum(dim)
        inside = ((T - vg) > 0) | xg.bool()
        total = total + torch.where(inside.all(dim), log_normal - log_jac, torch.full_like(log_normal, -math.inf))
    return _sum_trailing(total, dim % x_onehot.dim())
