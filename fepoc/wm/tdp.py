"""TDP 기억: GRU 대신 tdp-reasoner(github.com/zxcyxv/tdp-reasoner)의 블록을 환경 시간 축에 쓴다. 설계: docs/exp2g_design.md

tdp-reasoner는 블록 하나를 같은 입력에 재귀로 반복 적용하고, 재귀 단계를 가로지르는 빠른 가중치 기억 M을 둔다.
여기서는 재귀 축을 환경 시간 축으로 바꾼다: 블록 한 번 = 환경 한 스텝, 매 스텝 새 관측을 주입한다.
옮긴 것은 Sudoku 논문 설정(bands 4, band_product, hebb, learned 2D RoPE)의 블록이다 (sudoku/model.py bands_step).

토큰 52개: 칸 49개(자기중심 시야 7×7, τ = x·7 + y) + 방향·행동·결과 토큰. 시점 t의 상태 H_t [52, d]로 o_t를 예측한다.

    inj_t  = 주입(o_{t-1}, a_{t-1})              t = 0이면 "없음" 주입 (칸 내용 0, 특수 토큰은 "없음" 범주)
    U_t    = H_{t-1} + √d·inj_t                   H_{-1} = init_hidden (고정 난수 벡터, tdp-reasoner와 같음)
    q, k, v = 헤드별 투영(U_t);  q~, k~ = RoPE(q/(|q|+ε)), RoPE(k/(|k|+ε))   칸은 학습되는 2D RoPE, 특수 토큰은 각도 0
    E_K,r ← z_r E_K,r + γ_r k~,   E_V,r ← z_r E_V,r + γ_r v                   토큰별 흔적, 대역 r = 1..R
    M_t    = Σ_r γ_r Re(c_r Σ_τ E_V,r conj(E_K,r)ᵀ)                          재귀 축을 가로지르던 시냅스 = 시간 축 시냅스
    read   = q~ (Σ_τ v k~ᵀ + M_t)ᵀ
    H_t    = RMSNorm(H′ + W_D(silu(g)·a)),  H′ = RMSNorm(U_t + W_O read),  [g; a] = W_GU H′

z_r = exp(−e^ν + iθ) (학습), γ_r = √(1 − |z_r|²). synapse=False면 M = 0 (대조군: 같은 단계의 토큰 간 선형 어텐션과
H의 residual만 남는다). 흔적과 M은 과거 관측의 결정론적 함수라 기억이 결정론적이다 (정확한 우도의 조건 유지).
조각(학습 32스텝)마다 H, 흔적은 초기값에서 시작한다.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..env.memory import GROUP_SIZES, N_ACTIONS
from ..models.argmax_lift import lift_gauss, to_category
from ..models.flow import CondFlow
from .common import HW, N_CH, N_DIR, N_OUTCOME, cat_sample, grid_onehot, mlp, onehot_to_idx

N_CELL = HW * HW
TOK_DIR, TOK_ACT, TOK_OUT = N_CELL, N_CELL + 1, N_CELL + 2
N_TOK = N_CELL + 3


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


class TDPSkeleton(nn.Module):
    """GRU Skeleton의 자리. 상태 = (H [N, 52, d], E_K, E_V)."""

    def __init__(self, d=256, heads=8, bands=4, periods=(2.0, 8.0, 32.0, 128.0), synapse=True):
        super().__init__()
        self.d, self.scale = d, math.sqrt(d)
        std = 1.0 / self.scale
        self.cell_emb = nn.Parameter(trunc_normal_(torch.empty(N_CH, d), std))        # 원-핫 20 → d (속성 묶음 3개의 합)
        self.pos_emb = nn.Parameter(trunc_normal_(torch.empty(N_CELL, d), std))       # 자기중심 칸 위치
        self.dir_emb = nn.Parameter(trunc_normal_(torch.empty(N_DIR + 1, d), std))    # 마지막 행 = 없음
        self.act_emb = nn.Parameter(trunc_normal_(torch.empty(N_ACTIONS + 1, d), std))
        self.out_emb = nn.Parameter(trunc_normal_(torch.empty(N_OUTCOME + 1, d), std))
        self.register_buffer("init_hidden", trunc_normal_(torch.empty(d), 1.0))
        self.block = TDPBlock(d, heads, bands, periods, synapse=synapse)
        ar = torch.arange(N_CELL)
        self.register_buffer("pos_row", (ar // HW).float(), persistent=False)
        self.register_buffer("pos_col", (ar % HW).float(), persistent=False)

    def rope_tables(self):
        th = self.block.rope_theta                                                    # [H, P, 2]
        a = (th[..., 0, None] * self.pos_row + th[..., 1, None] * self.pos_col).transpose(-1, -2)   # [H, 49, P]
        a = torch.cat([a, a.new_zeros(a.shape[0], N_TOK - N_CELL, a.shape[-1])], 1)   # 특수 토큰 각도 0
        return a.cos(), a.sin()

    def init_state(self, N):
        b = self.block
        z = torch.zeros(N, b.H, N_TOK, 2, b.R, b.D, device=self.init_hidden.device)
        return self.init_hidden.expand(N, N_TOK, self.d), z, z.clone()

    def inject(self, grid_oh, direction, outcome, action, has):
        """직전 관측과 행동 → [N, 52, d]. has (N,) bool: False면 '없음' 주입 (조각의 첫 스텝)."""
        hf = has.float()
        cells = (grid_oh.flatten(-2).transpose(-1, -2) @ self.cell_emb) * hf[:, None, None] + self.pos_emb
        none = lambda idx, n: torch.where(has, idx, torch.full_like(idx, n))
        return torch.cat([cells, self.dir_emb[none(direction, N_DIR)][:, None],
                          self.act_emb[none(action, N_ACTIONS)][:, None],
                          self.out_emb[none(outcome, N_OUTCOME)][:, None]], 1)

    def step(self, state, grid_oh, direction, outcome, action, has, tables=None):
        H, ek, ev = state
        u = H + self.scale * self.inject(grid_oh, direction, outcome, action, has)
        return self.block(u, ek, ev, tables or self.rope_tables())

    def rollout(self, grid_oh, direction, outcome, action, keep_state=False):
        """관측 시퀀스 (B, L, ...) → H (B, L, 52, d). H[:, t]는 o_{<t}, a_{<t}만 본다.
        keep_state: 각 시점의 전체 상태 (H, E_K, E_V)를 (B, L, ...)로 함께 돌려준다 (k스텝 평가용)."""
        B, L = direction.shape
        tables = self.rope_tables()
        state = self.init_state(B)
        no = torch.zeros(B, dtype=torch.bool, device=direction.device)
        Hs, ks, vs = [], [], []
        for t in range(L):
            if t == 0:
                z = torch.zeros_like(direction[:, 0])
                state = self.step(state, torch.zeros_like(grid_oh[:, 0]), z, z, z, no, tables)
            else:
                state = self.step(state, grid_oh[:, t - 1], direction[:, t - 1], outcome[:, t - 1],
                                  action[:, t - 1], ~no, tables)
            Hs.append(state[0])
            if keep_state:
                ks.append(state[1]); vs.append(state[2])
        H = torch.stack(Hs, 1)
        return (H, (H, torch.stack(ks, 1), torch.stack(vs, 1))) if keep_state else (H, None)


class TokenCondFlow(CondFlow):
    """CondFlow와 같은 흐름. 조건과 기저 분포를 h 벡터 대신 칸 토큰 상태에서 칸별 선형으로 만든다.
    h 인자 자리에 pack(H) = [결합층 조건 16채널, 기저 μ 20채널, 기저 log σ 20채널] × 7 × 7을 넘긴다."""

    def __init__(self, d, n_blocks=4, cond_ch=16):
        super().__init__(h_dim=d, n_blocks=n_blocks, cond_ch=cond_ch)
        C = self.shape[0]
        del self.base
        self.cond = nn.Linear(d, cond_ch)
        self.base_tok = nn.Linear(d, 2 * C)
        nn.init.zeros_(self.base_tok.weight); nn.init.zeros_(self.base_tok.bias)     # 처음에는 기저 N(0, I)

    def pack(self, H):
        cells = H[:, :N_CELL]
        p = torch.cat([self.cond(cells), self.base_tok(cells)], -1)                   # [N, 49, 56]
        return p.transpose(1, 2).reshape(H.shape[0], -1, HW, HW)

    def _cond(self, pack, extra=None):
        return pack[:, :self.cond_ch]

    def base_params(self, pack, extra=None):
        return pack[:, self.cond_ch:].chunk(2, 1)


class CondFlowTDPWM(nn.Module):
    """M0-SQ-T: M0-SQ(흐름 + 학습되는 들어 올림 q(u|x))의 GRU 기억을 TDP 기억으로 바꾼 것. 직전 시야 연결은 없다
    (직전 시야는 U_t의 residual과 시냅스 기억으로 들어간다). 방향·결과 머리는 해당 토큰의 상태를 읽는다."""
    flow_model = True
    tokens = True
    eval_chunk_batch = 16                     # k스텝 평가에서 시점마다 흔적을 보관하므로 작게

    def __init__(self, n_blocks=4, d=256, heads=8, bands=4, periods=(2.0, 8.0, 32.0, 128.0), synapse=True):
        super().__init__()
        from .models import QNet
        self.name = "M0-SQ-T" if synapse else "M0-SQ-T0"
        self.sk = TDPSkeleton(d, heads, bands, periods, synapse)
        self.flow = TokenCondFlow(d, n_blocks)
        self.qnet = QNet()
        self.dir_head = mlp(d, N_DIR)
        self.out_head = mlp(d, N_OUTCOME)

    def _heads(self, H):
        return self.dir_head(H[:, TOK_DIR]), self.out_head(H[:, TOK_OUT])

    def _lift(self, x, gen=None):
        mu, ls = self.qnet(x)
        return lift_gauss(x, mu, ls, GROUP_SIZES, generator=gen)

    def step_terms(self, H, grid_idx, direction, outcome, gen=None):
        x = grid_onehot(grid_idx)
        v, lq = self._lift(x, gen)
        lp_v = self.flow.log_prob(v, self.flow.pack(H))
        ld, lo = self._heads(H)
        F_grid = -(lp_v - lq)
        return dict(F=F_grid + F.cross_entropy(ld, direction, reduction="none")
                    + F.cross_entropy(lo, outcome, reduction="none"), F_grid=F_grid)

    def loss(self, batch):
        from .models import masked_mean_seq_sum
        x = grid_onehot(batch["grid"])
        H, _ = self.sk.rollout(x, batch["dir"], batch["outcome"], batch["action"])
        B, L = batch["dir"].shape
        F_t = self.step_terms(H.reshape(B * L, N_TOK, -1), batch["grid"].reshape(B * L, 7, 7, 3),
                              batch["dir"].reshape(-1), batch["outcome"].reshape(-1))["F"].view(B, L)
        return masked_mean_seq_sum(F_t, batch["mask"].float()), F_t

    @torch.no_grad()
    def loglik_is(self, H, grid_idx, direction, outcome, S, chunk=250):
        """ln P̂(o_t | H_t)를 q(v|x)를 제안으로 하는 중요도 가중으로 추정 (CondFlowSkipLearnedQWM과 같음)."""
        x = grid_onehot(grid_idx)
        pack = self.flow.pack(H)
        lws = []
        for s0 in range(0, S, chunk):
            k = min(chunk, S - s0)
            rep = lambda t: t.unsqueeze(0).expand(k, *t.shape).reshape(-1, *t.shape[1:])
            v, lq = self._lift(rep(x))
            lws.append((self.flow.log_prob(v, rep(pack)) - lq).view(k, -1))
        ld, lo = self._heads(H)
        extra = -(F.cross_entropy(ld, direction, reduction="none") + F.cross_entropy(lo, outcome, reduction="none"))
        return torch.cat(lws, 0).T, extra

    def init_state(self, batch):
        return self.sk.init_state(batch)

    @torch.no_grad()
    def predict(self, state, sample=True, gen=None):
        H = state[0]
        pack = self.flow.pack(H)
        if sample:
            v, _ = self.flow.sample(pack, generator=gen)
        else:
            mu, _ = self.flow.base_params(pack)
            v = self.flow.inverse(mu, pack)
        ld, lo = self._heads(H)
        return onehot_to_idx(to_category(v, GROUP_SIZES)), cat_sample(ld, sample, gen), cat_sample(lo, sample, gen), None

    def imagine_step(self, state, a, sample=True, gen=None):
        """상태로 o_t를 만들고, 그것과 행동 a_t로 다음 상태를 만든다."""
        grid, d, o, _ = self.predict(state, sample, gen)
        has = torch.ones(a.shape[0], dtype=torch.bool, device=a.device)
        return self.sk.step(state, grid_onehot(grid), d, o, a, has), (grid, d, o), None

    @staticmethod
    def flat_state(st):
        """rollout(keep_state=True)의 (B, L, ...) 상태 → (B·L, ...)."""
        return tuple(x.reshape(-1, *x.shape[2:]) for x in st)
