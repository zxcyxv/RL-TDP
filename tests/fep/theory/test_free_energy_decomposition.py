"""1단계 2절 검증: V1 (F의 세 형태), V3 (에르고딕 3항 분해), V4 (θ 기울기 안의 KL 항).

기준 모델: 숨은 상태 s가 이산(K개), 관측 o가 이산(M개)인 일반적인 잠재변수 모델
p(s, o) = p(s) p(o|s). q(s)는 임의의 분포.
"""
import numpy as np
import pytest
import torch

rng = np.random.default_rng(0)


def rand_simplex(*shape):
    x = rng.gamma(1.0, size=shape)
    return x / x.sum(-1, keepdims=True)


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


@pytest.mark.parametrize("K", [2, 3, 5])
def test_v1_three_forms_of_F_agree(K):
    M = 4
    prior = rand_simplex(K)              # p(s)
    lik = rand_simplex(K, M)             # p(o|s), 행마다 o 위의 분포
    q = rand_simplex(K)                  # 임의의 근사 사후분포
    o = 2

    joint = prior * lik[:, o]            # p(s, o) for fixed o
    p_o = joint.sum()
    post = joint / p_o

    energy_minus_entropy = np.sum(q * -np.log(joint)) - (-np.sum(q * np.log(q)))
    divergence_plus_surprise = kl(q, post) - np.log(p_o)
    complexity_minus_accuracy = kl(q, prior) - np.sum(q * np.log(lik[:, o]))

    assert abs(energy_minus_entropy - divergence_plus_surprise) < 1e-9
    assert abs(energy_minus_entropy - complexity_minus_accuracy) < 1e-9


def _ergodic_setup(K=3, M=5):
    prior = rand_simplex(K)
    lik = rand_simplex(K, M)
    q = rand_simplex(M, K)               # 관측마다 다른 q(s|o) (인코더)
    p_theta_o = prior @ lik              # 모델의 관측 주변분포 p_θ(o)
    post = (prior[:, None] * lik).T / p_theta_o[:, None]   # p_θ(s|o), 행 = o
    F = np.array([kl(q[o], post[o]) - np.log(p_theta_o[o]) for o in range(M)])
    A = np.array([kl(q[o], post[o]) for o in range(M)])
    return p_theta_o, F, A


def test_v3_ergodic_three_term_identity_exact():
    p_theta_o, F, A = _ergodic_setup()
    p_star = rand_simplex(len(F))        # 세계의 정상분포 (모델과 다름)
    lhs = p_star @ F
    term_A = p_star @ A
    term_B = kl(p_star, p_theta_o)
    term_C = -np.sum(p_star * np.log(p_star))
    assert abs(lhs - (term_A + term_B + term_C)) < 1e-12
    # 시간 평균 놀람 = 교차 엔트로피 ≠ 엔트로피 (p* ≠ p_θ)
    assert term_B > 1e-3


def test_v3_ergodic_time_average_markov_chain():
    """에르고딕 마르코프 사슬로 관측을 만들고 F의 장기 시간 평균을 3항 합과 비교한다."""
    p_theta_o, F, A = _ergodic_setup()
    M = len(F)
    P = rand_simplex(M, M)               # 전이행렬, 모든 원소 > 0 → 에르고딕
    w, v = np.linalg.eig(P.T)
    p_star = np.real(v[:, np.argmin(abs(w - 1))])
    p_star /= p_star.sum()

    T = 1_000_000
    gen = np.random.default_rng(1)
    cdf = np.cumsum(P, axis=1)
    u = gen.random(T)
    o, counts = 0, np.zeros(M)
    for t in range(T):
        o = int(np.searchsorted(cdf[o], u[t]))
        counts[o] += 1
    time_avg_F = counts @ F / T

    three_terms = p_star @ A + kl(p_star, p_theta_o) - np.sum(p_star * np.log(p_star))
    assert abs(time_avg_F - three_terms) < 5e-3


def test_v4_theta_gradient_contains_kl_term():
    torch.manual_seed(0)
    K, M, o = 4, 3, 1
    prior_logits = torch.randn(K, dtype=torch.float64, requires_grad=True)
    lik_logits = torch.randn(K, M, dtype=torch.float64, requires_grad=True)
    theta = [prior_logits, lik_logits]

    def model_terms():
        log_prior = torch.log_softmax(prior_logits, 0)
        log_lik = torch.log_softmax(lik_logits, 1)[:, o]
        log_joint = log_prior + log_lik
        log_p_o = torch.logsumexp(log_joint, 0)
        return log_joint, log_p_o, log_joint - log_p_o

    def grad(f):
        g = torch.autograd.grad(f, theta)
        return torch.cat([x.flatten() for x in g])

    # 임의의 q: KL 항의 θ 기울기가 0이 아니다
    q = torch.softmax(torch.randn(K, dtype=torch.float64), 0)
    log_joint, log_p_o, log_post = model_terms()
    kl_term = torch.sum(q * (q.log() - log_post))
    g_kl = grad(kl_term)
    assert g_kl.abs().max() > 1e-3

    # ∇F = ∇KL − ∇ln p(o)
    log_joint, log_p_o, log_post = model_terms()
    F = torch.sum(q * (q.log() - log_joint))
    _, log_p_o2, _ = model_terms()
    assert torch.allclose(grad(F), g_kl - grad(log_p_o2), atol=1e-12)

    # q = 현재 사후분포(상수로 고정): KL 항의 θ 기울기가 수치적 0
    with torch.no_grad():
        q_star = model_terms()[2].exp()
    log_post = model_terms()[2]
    kl_star = torch.sum(q_star * (q_star.log() - log_post))
    assert grad(kl_star).abs().max() < 1e-12
