"""TDP 기억 (fepoc/wm/tdp.py) 확인.

- 인과성: H_t는 o_{<t}, a_{<t}만 본다
- 시냅스 M = Σ_r γ_r Re(c_r Σ_τ E_V,r conj(E_K,r)ᵀ)가 복소수 텐서로 직접 계산한 값과 같다 (tdp-reasoner의 식)
- synapse=False는 c = 0인 synapse=True와 같다 (M만 빠진 대조군)
- 한 스텝씩 step으로 굴린 상태 = rollout의 상태 (상상 경로와 학습 경로가 같은 기억을 쓴다)
- 손실 역전파와 평가(evaluate)가 끝까지 돈다
"""
import numpy as np
import torch

from fepoc.wm.common import grid_onehot
from fepoc.wm.evaluate import evaluate
from fepoc.wm.models import build
from fepoc.wm.tdp import N_TOK, TDPBlock, TDPSkeleton


def _seq(B=3, L=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.stack([torch.randint(0, n, (B, L, 7, 7), generator=g) for n in (11, 6, 3)], -1)
    return dict(grid=grid, dir=torch.randint(0, 4, (B, L), generator=g), outcome=torch.randint(0, 3, (B, L), generator=g),
                action=torch.randint(0, 3, (B, L), generator=g), mask=torch.ones(B, L, dtype=torch.bool))


def _small_sk(synapse=True, seed=0):
    torch.manual_seed(seed)
    return TDPSkeleton(d=32, heads=4, bands=2, periods=(2.0, 16.0), synapse=synapse)


def test_causal():
    sk = _small_sk()
    b = _seq()
    H, _ = sk.rollout(grid_onehot(b["grid"]), b["dir"], b["outcome"], b["action"])
    b2 = {k: v.clone() for k, v in b.items()}
    b2["grid"][:, 5] = (b2["grid"][:, 5] + 1) % 3
    b2["action"][:, 5] = (b2["action"][:, 5] + 1) % 3
    H2, _ = sk.rollout(grid_onehot(b2["grid"]), b2["dir"], b2["outcome"], b2["action"])
    assert torch.equal(H[:, :6], H2[:, :6])                 # H_5는 o_5, a_5를 보지 않는다
    assert not torch.allclose(H[:, 6], H2[:, 6])


def test_synapse_matches_complex_formula():
    torch.manual_seed(0)
    blk = TDPBlock(d=32, heads=4, bands=3, periods=(2.0, 6.0, 20.0)).double()
    B, T, H, R, D = 2, 5, 4, 3, 8
    ek = torch.randn(B, H, T, 2, R, D, dtype=torch.float64)
    ev = torch.randn(B, H, T, 2, R, D, dtype=torch.float64)
    zr, zi, gamma, cr, ci = blk.bands()
    c = torch.complex(cr, ci)[None, :, None, :, None]
    EV, EK = torch.complex(ev[:, :, :, 0], ev[:, :, :, 1]), torch.complex(ek[:, :, :, 0], ek[:, :, :, 1])
    direct = (gamma[None, :, None, :, None] * c * EV)                                   # [B, H, T, R, D]
    M_direct = torch.einsum("bhtrd,bhtre->bhde", direct, EK.conj()).real
    y_re, y_im = (gamma[None, :, None, :, None] * (c.real * EV.real - c.imag * EV.imag),
                  gamma[None, :, None, :, None] * (c.real * EV.imag + c.imag * EV.real))
    M_code = torch.einsum("bhtrd,bhtre->bhde", y_re, EK.real) + torch.einsum("bhtrd,bhtre->bhde", y_im, EK.imag)
    assert torch.allclose(M_direct, M_code, atol=1e-12)
    # 극점: |z| = e^{-1/p}, 각도 2π/p, γ² + |z|² = 1
    assert torch.allclose(zr ** 2 + zi ** 2 + gamma ** 2, torch.ones_like(gamma), atol=1e-6)


def test_no_synapse_equals_zero_coefficients():
    a, b0 = _small_sk(True), _small_sk(False)
    b0.load_state_dict(a.state_dict())
    with torch.no_grad():
        a.block.band_c.zero_()
    b = _seq()
    x = grid_onehot(b["grid"])
    Ha, _ = a.rollout(x, b["dir"], b["outcome"], b["action"])
    Hb, _ = b0.rollout(x, b["dir"], b["outcome"], b["action"])
    assert torch.allclose(Ha, Hb, atol=1e-6)


def test_step_matches_rollout():
    sk = _small_sk()
    b = _seq(B=2, L=6)
    x = grid_onehot(b["grid"])
    H, (Hs, ks, vs) = sk.rollout(x, b["dir"], b["outcome"], b["action"], keep_state=True)
    assert H.shape == (2, 6, N_TOK, 32) and torch.equal(H, Hs)
    state = sk.init_state(2)
    z = torch.zeros(2, dtype=torch.long)
    state = sk.step(state, torch.zeros_like(x[:, 0]), z, z, z, torch.zeros(2, dtype=torch.bool))
    for t in range(1, 6):
        state = sk.step(state, x[:, t - 1], b["dir"][:, t - 1], b["outcome"][:, t - 1], b["action"][:, t - 1],
                        torch.ones(2, dtype=torch.bool))
        assert torch.allclose(state[0], H[:, t], atol=1e-6) and torch.allclose(state[1], ks[:, t], atol=1e-6)


class _Data:
    """evaluate가 쓰는 SeqData의 최소 대역 (에피소드 2개)."""
    segment_names = ("room", "corridor", "junction")

    def __init__(self):
        b = _seq(B=1, L=40, seed=1)
        self.grid, self.dir = b["grid"][0].to(torch.uint8), b["dir"][0]
        self.outcome, self.action = b["outcome"][0], b["action"][0]
        self.segment = torch.randint(0, 3, (40,))
        self.ep_start, self.ep_len = torch.tensor([0, 25]), torch.tensor([25, 15])

    def chunks(self, L):
        st, ln = [], []
        for s, n in zip(self.ep_start.tolist(), self.ep_len.tolist()):
            for o in range(0, n, L):
                st.append(s + o); ln.append(min(L, n - o))
        return torch.tensor(st), torch.tensor(ln)

    def batch_from(self, starts, lens, L):
        ar = torch.arange(L)
        idx = starts[:, None] + torch.minimum(ar[None], (lens - 1)[:, None])
        return dict(grid=self.grid[idx].long(), dir=self.dir[idx], outcome=self.outcome[idx], action=self.action[idx],
                    segment=self.segment[idx], mask=ar[None] < lens[:, None])


def test_loss_backward_and_evaluate():
    torch.manual_seed(0)
    for name in ("m0sqt", "m0sqt0"):
        m = build(name, n_blocks=2, d=32, heads=4, bands=2, periods=(2.0, 16.0))
        loss, F_t = m.loss(_seq(B=2, L=8))
        loss.backward()
        assert torch.isfinite(loss) and F_t.shape == (2, 8)
        assert m.sk.block.q_proj.weight.grad.abs().sum() > 0
        if name == "m0sqt":
            assert m.sk.block.band_pole.grad.abs().sum() > 0 and m.sk.block.band_c.grad.abs().sum() > 0
        else:                                         # 대조군: 시냅스가 없으니 극점·계수는 쓰이지 않는다
            assert m.sk.block.band_pole.grad is None and m.sk.block.band_c.grad is None
        ev = evaluate(m, _Data(), L=16, S_list=(4, 8), ks=(1, 2))
        assert np.isfinite(ev["F"]) and np.isfinite(ev["I_hat_8"]) and 0 <= ev["acc"]["k1_sample"] <= 1
