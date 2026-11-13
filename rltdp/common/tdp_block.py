"""계열 공통: tdp-reasoner(github.com/zxcyxv/tdp-reasoner) Sudoku 설정 블록의 PyTorch 경로.

같은 단계 토큰의 선형 어텐션 Σ_τ v k̃ᵀ에, 대역별 키·값 흔적의 곱으로 만든 시냅스 M = Σ_r γ_r Re(c_r Σ_τ E_V conj(E_K)ᵀ)를
더해 읽는다 (쌍 기반 STDP의 흔적 꼴, 극점 z_r 학습). 블록 한 번 = 축 한 스텝 (재귀 단계 또는 환경 시간).
쓰는 곳: rltdp.fep.wm.tdp (실험 2g 월드 모델), rltdp.btsp.tiny / delta / replay.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def rms_norm(x, eps=1e-5):
    """tdp-reasoner(URM)의 RMSNorm: 스케일·편향 없음."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def trunc_normal_(t, std):
    with torch.no_grad():
        return nn.init.trunc_normal_(t, std=std, a=-2 * std, b=2 * std)


class TDPBlock(nn.Module):
    """tdp-reasoner Block + LT.bands_step(band_product, hebb)의 PyTorch 경로를 옮긴 것."""

    def __init__(self, d=256, heads=8, bands=4, periods=(2.0, 8.0, 32.0, 128.0), coef_std=0.1,
                 mlp_expansion=4.0, eps=1e-4, synapse=True):
        super().__init__()
        assert d % heads == 0 and (d // heads) % 2 == 0 and len(periods) == bands
        self.d, self.H, self.D, self.R, self.eps, self.synapse = d, heads, d // heads, bands, eps, synapse
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d, d, bias=False) for _ in range(4))
        inter = int(mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.ffn_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.ffn_down = nn.Linear(inter, d, bias=False)
        nn.init.zeros_(self.ffn_down.weight)
        self.rope_theta = nn.Parameter(torch.empty(heads, self.D // 2, 2).uniform_(-math.pi / 2, math.pi / 2))
        p = torch.tensor(periods, dtype=torch.float64)
        self.band_pole = nn.Parameter(torch.stack((-p.log(), 2 * math.pi / p))[:, None].repeat(1, heads, 1).float())
        self.band_c = nn.Parameter(torch.empty(2, heads, bands).normal_(0.0, coef_std))

    def bands(self):
        """대역별 [H, R]: Re z, Im z, γ = √(1 − |z|²), Re c, Im c."""
        nu, theta = self.band_pole
        mag = torch.exp(-torch.exp(nu))
        return mag * torch.cos(theta), mag * torch.sin(theta), torch.sqrt(-torch.expm1(-2 * torch.exp(nu))), *self.band_c

    @staticmethod
    def rope(x, tables):
        cos, sin = tables
        x0, x1 = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).reshape_as(x)

    def forward(self, u, ek, ev, tables):
        """u [B, T, d], ek/ev [B, H, T, 2, R, D] (Re, Im). 반환 H_t, 새 흔적."""
        B, T, _ = u.shape
        heads = lambda x: x.reshape(B, T, self.H, self.D).transpose(1, 2)
        q, k, v = heads(self.q_proj(u)), heads(self.k_proj(u)), heads(self.v_proj(u))
        unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + self.eps)
        qr, kr = self.rope(unit(q), tables), self.rope(unit(k), tables)
        read_w = v.transpose(-1, -2) @ kr                                  # 같은 단계: Σ_τ v k~ᵀ [B, H, Dv, Dk]
        zr, zi, gamma, cr, ci = (x[None, :, None, :, None] for x in self.bands())   # [1, H, 1, R, 1]
        ek_re, ek_im = ek.unbind(3)
        ev_re, ev_im = ev.unbind(3)
        ek_re, ek_im = zr * ek_re - zi * ek_im + gamma * kr.unsqueeze(3), zr * ek_im + zi * ek_re
        ev_re, ev_im = zr * ev_re - zi * ev_im + gamma * v.unsqueeze(3), zr * ev_im + zi * ev_re
        if self.synapse:                                                  # M = Σ_r Re(γ c E_V conj(E_K)ᵀ)
            y_re, y_im = gamma * (cr * ev_re - ci * ev_im), gamma * (cr * ev_im + ci * ev_re)
            read_w = read_w + torch.einsum("bhtrd,bhtre->bhde", y_re, ek_re) + torch.einsum("bhtrd,bhtre->bhde", y_im, ek_im)
        read = torch.einsum("bhte,bhde->bhtd", qr, read_w).transpose(1, 2).reshape(B, T, self.d)
        h = rms_norm(u + self.out_proj(read))
        g, a = self.ffn_gate_up(h).chunk(2, -1)
        return rms_norm(h + self.ffn_down(F.silu(g) * a)), torch.stack((ek_re, ek_im), 3), torch.stack((ev_re, ev_im), 3)
