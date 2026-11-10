"""실험 2f 구성요소 확인 (docs/exp2f_design.md).

- 선호분포 C = softmax(c/λ_C)는 정규화되어 있고 c에 상수를 더해도 같다
- 흡수 종료에서 정규화된 C로 쓴 Q는 ActorCriticKL의 Q (ln C(진행) = 0 가정)와 상수 ln C(진행)/(1−γ)만큼만 다르다
- novelty 정보 이득: 결합 MC 추정 = 열거로 계산한 정확값, 0 ≤ I ≤ ln K, 칸별 JSD 합 ≥ I, 같은 멤버면 0
"""
import itertools
import math

import torch

from fepoc.agent.efe import ActorCriticEFE, Preference, efe_q
from fepoc.agent.novelty import NoveltyEnsemble, group_log_softmax, mi_cellsum, mi_joint
from fepoc.env.memory import PROGRESS


def test_preference_normalized_and_shift_invariant():
    p = Preference((0.0, 1.0, -1.0), lam_c=0.5)
    assert abs(p.ln_c.exp().sum().item() - 1) < 1e-6
    q = Preference((3.0, 4.0, 2.0), lam_c=0.5)
    assert torch.allclose(p.ln_c, q.ln_c, atol=1e-6)
    sharp, flat = Preference(lam_c=0.1).ln_c.exp(), Preference(lam_c=10.0).ln_c.exp()
    assert sharp.max() > flat.max()


def test_efe_q_equals_kl_actor_up_to_constant():
    g = torch.Generator().manual_seed(0)
    gamma, N = 0.985, 64
    c = torch.tensor([0.0, 1.0, -1.0])                       # c(진행) = 0: ActorCriticKL과 같은 조건
    py = torch.softmax(torch.randn(N, 3, generator=g), -1)
    v_old = 10 * torch.randn(N, generator=g)
    u_old = c / (1 - gamma); u_old[PROGRESS] = 0.0          # ActorCriticKL.u_term
    q_old = (py * u_old).sum(-1) + gamma * py[:, PROGRESS] * v_old
    ln_c = Preference(tuple(c.tolist())).ln_c
    shift = ln_c[PROGRESS] / (1 - gamma)                     # 모든 가치가 이만큼 같이 움직인다
    q_new = efe_q(py, ln_c, v_old + shift, 0.0, gamma, 0.0)
    assert torch.allclose(q_new, q_old + shift, atol=1e-3)


def _exact_mi(logits, group_sizes):
    """작은 격자에서 모든 x를 열거해 I(θ; x) = mean_k Σ_x p_k(x) ln(p_k(x)/p̄(x))를 계산한다. logits (K, C, H, W)."""
    lp = group_log_softmax(logits[None], group_sizes)[0].double()
    K, _, H, W = lp.shape
    cells = [(i, j) for i in range(H) for j in range(W)]
    offs = [sum(group_sizes[:g]) for g in range(len(group_sizes))]
    per_cell = list(itertools.product(*[range(n) for n in group_sizes]))
    lpx = []
    for x in itertools.product(per_cell, repeat=len(cells)):
        lpx.append(sum(lp[:, offs[g] + v[g], i, j] for (i, j), v in zip(cells, x) for g in range(len(group_sizes))))
    lpx = torch.stack(lpx, 1)                                 # (K, 전체 x 수)
    lpbar = torch.logsumexp(lpx, 0) - math.log(K)
    return (lpx.exp() * (lpx - lpbar)).sum(1).mean().item()


def test_mi_joint_matches_enumeration():
    gs = (2, 3)
    g = torch.Generator().manual_seed(1)
    for scale in (0.5, 2.0):
        logits = scale * torch.randn(4, 5, 1, 2, generator=g)       # K = 4, 칸 2개, 묶음 (2, 3)
        exact = _exact_mi(logits, gs)
        N = 200_000
        est = mi_joint(logits[None].expand(N, -1, -1, -1, -1), gs, generator=g)
        se = est.std().item() / math.sqrt(N)
        assert abs(est.mean().item() - exact) < 5 * se + 1e-4, (est.mean().item(), exact, se)
        assert 0 <= exact <= math.log(4)
        assert mi_cellsum(logits[None], gs).item() >= exact - 1e-6     # 칸별 JSD 합은 상한


def test_mi_zero_for_identical_members():
    logits = torch.randn(1, 20, 7, 7).expand(6, 3, -1, -1, -1).contiguous()   # N = 6, K = 3, 멤버 같음
    assert mi_joint(logits).abs().max().item() < 1e-4
    assert mi_cellsum(logits).abs().max().item() < 1e-4


def test_novelty_module_shapes_and_training():
    torch.manual_seed(0)
    nov = NoveltyEnsemble(K=3)
    B, L = 4, 6
    batch = dict(grid=torch.stack([torch.randint(0, n, (B, L, 7, 7)) for n in (11, 6, 3)], -1),
                 action=torch.randint(0, 3, (B, L)), mask=torch.ones(B, L, dtype=torch.bool))
    batch["mask"][0, 3:] = False
    losses = [nov.train_step(batch) for _ in range(30)]
    assert losses[-1] < losses[0]
    from fepoc.wm.common import grid_onehot
    ig = nov.info_gain(grid_onehot(batch["grid"][:, 0]), batch["action"][:, 0])
    assert ig.shape == (B,) and torch.isfinite(ig).all()


def test_efe_actor_init_matches_kl_actor_when_beta_zero():
    from fepoc.agent.ac import ActorCriticKL
    torch.manual_seed(0); a = ActorCriticKL()
    torch.manual_seed(0); b = ActorCriticEFE(beta=0.0)
    assert b.novelty is None
    for pa, pb in zip(a.actor.parameters(), b.actor.parameters()):
        assert torch.equal(pa, pb)
