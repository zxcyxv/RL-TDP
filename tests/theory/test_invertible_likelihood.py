"""1단계 3절 검증: V6 (기억 있는 가역 모델의 정확한 우도), V7 (결정론 극한에서 ELBO 붕괴),
V9 (T자 미로의 기억 필요 판정).

기준 모델: V6는 RealNVP 같은 정규화 흐름, V7은 선형 가우시안 VAE(확률적 PCA).
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)


# ---------- V6 ----------
# 기억 h가 주어졌을 때의 조건부 흐름. h는 결정론적 기억에서 나온 값이라 고정된 벡터로 둔다.
# 밀도가 정규화되는지(적분 = 1)를 격자 수치적분으로 확인한다.

def flow_inv_1d(o, h):
    """단조 증가 변환 z = f⁻¹(o; h). 도함수 a + b·c·sech²(c·o) > 0 이 되도록 |b·c| < a."""
    a, b, c, d = 1.3 + 0.2 * np.tanh(h[0]), 0.8 * np.tanh(h[1]), 1.5, h[2]
    z = a * o + b * np.tanh(c * o) + d
    dz = a + b * c / np.cosh(c * o) ** 2
    return z, dz


def test_v6_1d_conditional_flow_density_integrates_to_one():
    for _ in range(3):
        h = rng.normal(size=3)
        mu, sigma = np.tanh(h[0]), np.exp(0.3 * h[1])        # 기저 p_z(·|h)
        o = np.linspace(-20, 20, 400_001)
        z, dz = flow_inv_1d(o, h)
        F = -norm.logpdf(z, mu, sigma) - np.log(np.abs(dz))  # 3.1절 식
        integral = np.trapezoid(np.exp(-F), o)
        assert abs(integral - 1) < 1e-6


def test_v6_2d_affine_coupling_density_integrates_to_one():
    h = rng.normal(size=4)
    W = rng.normal(size=(2, 2))

    def inv(o1, o2):
        # 결합층 역변환: z1 = o1, z2 = (o2 − t(o1,h))·exp(−s(o1,h))
        s = 0.7 * np.tanh(W[0, 0] * o1 + h[0])
        t = np.sin(W[1, 0] * o1 + h[1])
        z1, z2 = o1, (o2 - t) * np.exp(-s)
        logdet = -s
        return z1, z2, logdet

    g = np.linspace(-12, 12, 2001)
    o1, o2 = np.meshgrid(g, g, indexing="ij")
    z1, z2, logdet = inv(o1, o2)
    mu = np.tanh(h[2:])
    logp = norm.logpdf(z1, mu[0], 1.2) + norm.logpdf(z2, mu[1], 0.8) + logdet
    integral = np.trapezoid(np.trapezoid(np.exp(logp), g, axis=1), g)
    assert abs(integral - 1) < 1e-4


def test_v6_sequence_loglik_is_sum_of_F():
    """ln p(o_{1:T}) = −Σ F_t 를 이산 2-스텝 예시로 확인한다: 각 F_t가 정규화된 조건부이면
    곱이 결합 밀도가 된다. 1차원 연속 관측, h_t = tanh(h_{t-1} + z_t) 결정론적 기억."""
    o1_grid = np.linspace(-10, 10, 1201)
    o2_grid = np.linspace(-10, 10, 1201)
    h0 = np.array([0.3, -0.2, 0.1])

    z1, dz1 = flow_inv_1d(o1_grid, h0)
    F1 = -norm.logpdf(z1) - np.log(dz1)
    total = 0.0
    inner = []
    for i, o1 in enumerate(o1_grid):
        h1 = np.tanh(h0 + z1[i])                             # 기억 갱신 (결정론적)
        z2, dz2 = flow_inv_1d(o2_grid, h1)
        F2 = -norm.logpdf(z2, np.tanh(h1[0]), 1.0) - np.log(dz2)
        inner.append(np.trapezoid(np.exp(-F2), o2_grid))
    total = np.trapezoid(np.exp(-F1) * np.array(inner), o1_grid)
    assert abs(total - 1) < 1e-4


# ---------- V7 ----------
# 선형 가우시안 VAE: p(z) = N(0,1), p(o|z) = N(a·z, σ²). q = 정확한 사후분포라서 F = −ln p(o).
# σ → 0 이면 정확도 항은 +∞, 복잡도(KL) 항도 +∞로 발산하지만 F는 변수 변환 값으로 수렴한다.

def test_v7_elbo_terms_diverge_but_sum_converges():
    a, o = 1.7, 0.9
    exact_flow_F = -norm.logpdf(o / a) + np.log(a)           # σ = 0 일 때 3.1절 식
    accs, kls, Fs = [], [], []
    for sigma in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        post_var = 1 / (1 + a**2 / sigma**2)
        post_mean = post_var * a * o / sigma**2
        # E_q ln p(o|z)
        acc = norm.logpdf(o, a * post_mean, sigma) - a**2 * post_var / (2 * sigma**2)
        kl = 0.5 * (post_var + post_mean**2 - 1 - np.log(post_var))
        accs.append(acc); kls.append(kl); Fs.append(kl - acc)
        assert abs((kl - acc) - (-norm.logpdf(o, 0, np.sqrt(a**2 + sigma**2)))) < 1e-9
    assert np.all(np.diff(accs) > 2.0) and accs[-1] > 10     # +∞로 발산
    assert np.all(np.diff(kls) > 2.0)                         # 엔트로피 −∞ → KL +∞
    assert abs(Fs[-1] - exact_flow_F) < 1e-8                  # 합은 수렴


# ---------- V9 ----------
# T자 미로: 목표 쪽 η ∈ {L, R}이 반반. 시작에서 단서 o_1 = η를 보고, 갈림길 관측 o_t는 η와 무관하게 같다.

def test_v9_tmaze_memory_requirement_is_one_bit():
    # 결합분포 p(η, o_1, o_t). o_1은 η를 그대로 보여 주고, o_t는 항상 'junction'(0).
    p = np.zeros((2, 2, 1))
    p[0, 0, 0] = p[1, 1, 0] = 0.5

    def H(x):
        x = x[x > 0]
        return -np.sum(x * np.log2(x))

    # I(η; o_1 | o_t) = H(η|o_t) − H(η|o_1, o_t)
    p_eta_ot = p.sum(1)
    H_eta_given_ot = H(p_eta_ot.flatten()) - H(p_eta_ot.sum(0))
    H_eta_given_o1_ot = H(p.flatten()) - H(p.sum(0).flatten())
    assert abs((H_eta_given_ot - H_eta_given_o1_ot) - 1.0) < 1e-12
