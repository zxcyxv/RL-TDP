"""실험 1의 모델. 모든 모델은 선택지 원-핫 c를 조건으로 받는다.

- M0: 조건부 RealNVP. F = 정확한 −ln p(o|c).
- M1: 조건부 VAE (잠재 2차원, 사전 N(0, I)). F = −IWAE_k (k = 1이면 −ELBO).
  M1-a는 디코더 표준편차를 학습하고, M1-b는 0.1로 고정한다.
"""
import math

import torch
from torch import nn

LOG2PI = math.log(2 * math.pi)


def mlp(d_in, d_out, hidden, zero_last=False):
    net = nn.Sequential(nn.Linear(d_in, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
                        nn.Linear(hidden, d_out))
    if zero_last:
        nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    return net


def std_normal_logpdf(x):
    return -0.5 * (x**2 + LOG2PI).sum(-1)


def diag_normal_logpdf(x, mean, log_std):
    return (-0.5 * ((x - mean) / log_std.exp()) ** 2 - log_std - 0.5 * LOG2PI).sum(-1)


class RealNVP(nn.Module):
    """결합층마다 한 좌표를 다른 좌표와 c로 affine 변환한다. 스케일은 2·tanh(·/2)로 제한."""

    def __init__(self, n_layers=12, hidden=128, n_cond=2):
        super().__init__()
        self.nets = nn.ModuleList(mlp(1 + n_cond, 2, hidden, zero_last=True) for _ in range(n_layers))

    def _st(self, i, x_keep, c):
        s, t = self.nets[i](torch.cat([x_keep, c], -1)).chunk(2, -1)
        return 2 * torch.tanh(s / 2), t

    def to_latent(self, o, c):
        """o → z, ln|det ∂z/∂o|."""
        z, logdet = o, torch.zeros(o.shape[0], device=o.device)
        for i in range(len(self.nets)):
            k, m = i % 2, 1 - i % 2                    # k: 고정 좌표, m: 변환 좌표
            s, t = self._st(i, z[:, k:k + 1], c)
            zm = z[:, m:m + 1] * s.exp() + t
            z = torch.cat([z[:, :1], zm], -1) if m == 1 else torch.cat([zm, z[:, 1:]], -1)
            logdet = logdet + s.squeeze(-1)
        return z, logdet

    def F(self, o, c):
        z, logdet = self.to_latent(o, c)
        return -(std_normal_logpdf(z) + logdet)

    @torch.no_grad()
    def sample(self, c):
        x = torch.randn(c.shape[0], 2, device=c.device)
        for i in reversed(range(len(self.nets))):
            k, m = i % 2, 1 - i % 2
            s, t = self._st(i, x[:, k:k + 1], c)
            xm = (x[:, m:m + 1] - t) * (-s).exp()
            x = torch.cat([x[:, :1], xm], -1) if m == 1 else torch.cat([xm, x[:, 1:]], -1)
        return x


class CondVAE(nn.Module):
    def __init__(self, hidden=128, latent=2, n_cond=2, fixed_dec_std=None):
        super().__init__()
        self.latent = latent
        self.enc = mlp(2 + n_cond, 2 * latent, hidden)
        self.fixed_dec_std = fixed_dec_std
        self.dec = mlp(latent + n_cond, 2 if fixed_dec_std else 4, hidden)

    def decode(self, z, c):
        out = self.dec(torch.cat([z, c], -1))
        if self.fixed_dec_std:
            return out, torch.full_like(out, math.log(self.fixed_dec_std))
        mean, log_std = out.chunk(2, -1)
        return mean, log_std.clamp(min=-7.0)       # 수치 안전용. 실험 범위에서는 걸리지 않음

    def log_weights(self, o, c, k):
        """ln w_i = ln p(z_i) + ln p(o|z_i,c) − ln q(z_i|o,c), z_i ~ q. 반환 shape (k, N)."""
        q_mean, q_log_std = self.enc(torch.cat([o, c], -1)).chunk(2, -1)
        q_log_std = q_log_std.clamp(-10, 5)
        eps = torch.randn(k, *q_mean.shape, device=o.device)
        z = q_mean + q_log_std.exp() * eps
        c_k = c.expand(k, *c.shape)
        d_mean, d_log_std = self.decode(z, c_k)
        return (std_normal_logpdf(z) + diag_normal_logpdf(o, d_mean, d_log_std)
                - diag_normal_logpdf(z, q_mean, q_log_std))

    def F(self, o, c, k=1):
        """−IWAE_k. k = 1이면 −ELBO."""
        lw = self.log_weights(o, c, k)
        return -(torch.logsumexp(lw, 0) - math.log(k))

    @torch.no_grad()
    def sample(self, c):
        z = torch.randn(c.shape[0], self.latent, device=c.device)
        mean, log_std = self.decode(z, c)
        return mean + log_std.exp() * torch.randn_like(mean)
