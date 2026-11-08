"""M0의 조건부 흐름 f(·; h). 2단계 문서 "M0" 표를 따른다.

기준: Glow. 차이는 두 가지다.
- 다중 스케일(squeeze)이 없다 (7×7은 홀수).
- 결합층 신경망이 기억 h를 조건 채널로 받는다. 조건 채널은 공유 선형층 Linear(h) → 16×7×7.

방향 규약: forward(v, h)는 밀도 방향 v → z이고 ln|det ∂z/∂v|를 함께 돌려준다.
inverse(z, h)는 생성 방향 z → v다. 층마다 inverse는 (값, 정방향 log-det)을 돌려준다.
ActNorm은 항등으로 초기화한다 (데이터 의존 초기화를 쓰지 않는다. D4와 같은 이유).
"""
import torch
import torch.nn.functional as F
from torch import nn


class ActNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.log_s = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, C, 1, 1))

    def forward(self, x, cond):
        H, W = x.shape[-2:]
        return x * self.log_s.exp() + self.b, self.log_s.sum() * H * W * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

    def inverse(self, z, cond):
        H, W = z.shape[-2:]
        return (z - self.b) * (-self.log_s).exp(), self.log_s.sum() * H * W * torch.ones(z.shape[0], device=z.device, dtype=z.dtype)


class InvConv1x1(nn.Module):
    """LU 분해로 매개화한 가역 1×1 합성곱."""

    def __init__(self, C):
        super().__init__()
        w = torch.linalg.qr(torch.randn(C, C))[0]
        P, L, U = torch.linalg.lu(w)
        s = torch.diagonal(U)
        self.register_buffer("P", P)
        self.register_buffer("sign_s", torch.sign(s))
        self.register_buffer("l_mask", torch.tril(torch.ones(C, C), -1))
        self.register_buffer("eye", torch.eye(C))
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(torch.triu(U, 1))
        self.log_s = nn.Parameter(torch.log(torch.abs(s)))

    def weight(self):
        L = self.L * self.l_mask + self.eye
        U = self.U * self.l_mask.T + torch.diag(self.sign_s * self.log_s.exp())
        return self.P @ L @ U

    def forward(self, x, cond):
        H, W = x.shape[-2:]
        w = self.weight().to(x.dtype)
        return torch.einsum("ij,bjhw->bihw", w, x), self.log_s.sum() * H * W * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

    def inverse(self, z, cond):
        # W = P L U 이므로 W⁻¹z = U⁻¹ L⁻¹ Pᵀ z. 일반 역행렬보다 정확한 삼각 풀이를 쓴다.
        B, C, H, W = z.shape
        L = (self.L * self.l_mask + self.eye).to(z.dtype)
        U = (self.U * self.l_mask.T + torch.diag(self.sign_s * self.log_s.exp())).to(z.dtype)
        y = (self.P.T.to(z.dtype) @ z.permute(1, 0, 2, 3).reshape(C, -1))
        y = torch.linalg.solve_triangular(L, y, upper=False, unitriangular=True)
        y = torch.linalg.solve_triangular(U, y, upper=True)
        x = y.reshape(C, B, H, W).permute(1, 0, 2, 3)
        return x, self.log_s.sum() * H * W * torch.ones(B, device=z.device, dtype=z.dtype)


class AffineCoupling(nn.Module):
    """mask = 'channel'(앞/뒤 채널 절반) 또는 'checker'(체커보드). parity로 어느 쪽을 바꿀지 뒤집는다.
    고정 쪽 x_a와 조건 채널로 (s, t)를 만들고, 바뀌는 쪽에 z_b = x_b · e^{s} + t, s = tanh(·)."""

    def __init__(self, C, H, W, mask, parity, hidden=64, cond_ch=16):
        super().__init__()
        self.mask_type, self.C = mask, C
        if mask == "channel":
            m = torch.zeros(1, C, 1, 1)
            half = C // 2
            if parity == 0:
                m[:, :half] = 1
            else:
                m[:, half:] = 1
        else:
            ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
            m = (((ii + jj) % 2) == parity).float().expand(1, C, H, W).clone()
        self.register_buffer("keep", m)                 # 1 = 고정, 0 = 변환
        self.inp = nn.Conv2d(C + cond_ch, hidden, 3, padding=1)
        self.res1 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.res2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, 2 * C, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def _st(self, x_keep, cond):
        h = F.elu(self.inp(torch.cat([x_keep, cond], 1)))
        h = h + self.res2(F.elu(self.res1(h)))
        s, t = self.out(F.elu(h)).chunk(2, 1)
        change = 1 - self.keep
        return torch.tanh(s) * change, t * change

    def forward(self, x, cond):
        s, t = self._st(x * self.keep, cond)
        return x * s.exp() + t, s.flatten(1).sum(1)

    def inverse(self, z, cond):
        s, t = self._st(z * self.keep, cond)
        return (z - t) * (-s).exp(), s.flatten(1).sum(1)


class CondFlow(nn.Module):
    """블록 = ActNorm → 1×1 conv → affine coupling. 마스크 순서: channel, checker, channel, checker, ..."""

    def __init__(self, C=20, H=7, W=7, h_dim=256, n_blocks=4, hidden=64, cond_ch=16, base_hidden=256,
                 extra_ch=0):
        super().__init__()
        self.shape = (C, H, W)
        self.cond_ch = cond_ch
        self.extra_ch = extra_ch                            # 칸 위치 그대로 붙이는 추가 조건 (M0-S의 직전 시야)
        self.cond = nn.Linear(h_dim, cond_ch * H * W)      # 모든 블록이 공유
        layers = []
        for b in range(n_blocks):
            mask = "channel" if b % 2 == 0 else "checker"
            parity = (b // 2) % 2
            if mask == "channel" and C < 2:
                mask = "checker"
            layers += [ActNorm(C), InvConv1x1(C), AffineCoupling(C, H, W, mask, parity, hidden, cond_ch + extra_ch)]
        self.layers = nn.ModuleList(layers)
        D = C * H * W
        self.base = nn.Sequential(nn.Linear(h_dim, base_hidden), nn.ELU(),
                                  nn.Linear(base_hidden, base_hidden), nn.ELU(),
                                  nn.Linear(base_hidden, 2 * D))
        nn.init.zeros_(self.base[-1].weight); nn.init.zeros_(self.base[-1].bias)
        if extra_ch:
            self.base_sp = nn.Conv2d(extra_ch, 2 * C, 3, padding=1)   # 0 초기화: 처음에는 원래 기저와 같다
            nn.init.zeros_(self.base_sp.weight); nn.init.zeros_(self.base_sp.bias)

    def _cond(self, h, extra=None):
        c = self.cond(h).view(-1, self.cond_ch, *self.shape[1:])
        return c if extra is None else torch.cat([c, extra], 1)

    def base_params(self, h, extra=None):
        mu, log_sigma = self.base(h).chunk(2, -1)
        mu, log_sigma = mu.view(-1, *self.shape), log_sigma.view(-1, *self.shape)
        if extra is not None:
            dmu, dls = self.base_sp(extra).chunk(2, 1)
            mu, log_sigma = mu + dmu, log_sigma + dls
        return mu, log_sigma

    def forward(self, v, h, extra=None):
        """v → z. 반환 z, ln|det ∂z/∂v|."""
        c = self._cond(h, extra)
        logdet = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
        for l in self.layers:
            v, ld = l(v, c)
            logdet = logdet + ld
        return v, logdet

    def inverse(self, z, h, extra=None):
        c = self._cond(h, extra)
        for l in reversed(self.layers):
            z, _ = l.inverse(z, c)
        return z

    def log_prob(self, v, h, extra=None):
        """ln p(v | h) = ln N(z; μ(h), σ(h)) + ln|det ∂z/∂v|."""
        z, logdet = self.forward(v, h, extra)
        mu, log_sigma = self.base_params(h, extra)
        lp = -0.5 * (((z - mu) / log_sigma.exp()) ** 2) - log_sigma - 0.9189385332046727
        return lp.flatten(1).sum(1) + logdet

    def sample(self, h, generator=None, extra=None):
        """v ~ p(v|h)와 ln p(v|h)를 함께 돌려준다 (흐름을 한 번만 통과)."""
        mu, log_sigma = self.base_params(h, extra)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        z = mu + log_sigma.exp() * eps
        lp_z = (-0.5 * eps**2 - log_sigma - 0.9189385332046727).flatten(1).sum(1)
        c = self._cond(h, extra)
        v, logdet_inv = z, torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for l in reversed(self.layers):
            v, ld = l.inverse(v, c)       # ld = 정방향(v → z) log-det
            logdet_inv = logdet_inv + ld
        return v, lp_z + logdet_inv
