"""1단계 4·5절 검증: V10, V11 (탈양자화 흐름 = VAE 특수형, 간극 상한),
V12, V13, V14 (기대 자유에너지 항등식), V15 (앙상블 상호정보량).
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

rng = np.random.default_rng(0)


def rand_simplex(*shape):
    x = rng.gamma(1.0, size=shape)
    return x / x.sum(-1, keepdims=True)


# ---------- V10, V11 ----------
# 기준 모델: 1차원 정수 관측 x, 연속 밀도 p_flow = N(μ, s²) 위에 균등 탈양자화 y = x + u.

def test_v10_v11_uniform_dequantization_gap():
    for mu, s in [(0.3, 0.4), (2.1, 1.5), (-1.0, 0.25)]:
        for x in range(int(mu) - 2, int(mu) + 3):
            P_x = norm.cdf(x + 1, mu, s) - norm.cdf(x, mu, s)
            F_uniform, _ = quad(lambda u: -norm.logpdf(x + u, mu, s), 0, 1)

            # VAE 형태 F = E_q[ln q(u) − ln p(x, u)]에서 정확도 항 ln 1[y∈B(x)] = 0이고
            # 균등 q의 ln q = 0이라 F = F_uniform 그대로다. 아래 분해로 같은 값임을 확인한다.
            # VAE 분해: F = KL[q(u|x) ‖ p(u|x)] − ln P(x), p(u|x) = p_flow(x+u)/P(x)
            kl, _ = quad(lambda u: -np.log(norm.pdf(x + u, mu, s) / P_x), 0, 1)
            assert abs(F_uniform - (kl - np.log(P_x))) < 1e-8

            # 간극 0 ≤ A_deq ≤ ln(sup/inf)  (가우시안이라 칸 안 극값은 끝점이나 μ)
            A = F_uniform + np.log(P_x)
            cand = [x, x + 1] + ([mu] if x <= mu <= x + 1 else [])
            logps = norm.logpdf(np.array(cand), mu, s)
            assert -1e-10 <= A <= logps.max() - logps.min() + 1e-10


# ---------- V12, V13, V14 ----------
# 기준 모델: 이산 능동적 추론 (Da Costa et al. 2020). 한 정책 π 아래
# Q(s) = Q(s|π), 관측 우도 P(o|s), 선호 C(o).

def _efe_terms(Q_s, P_o_s, C):
    Q_os = Q_s[:, None] * P_o_s                     # Q(s, o)
    Q_o = Q_os.sum(0)
    Q_s_o = Q_os / np.where(Q_o > 0, Q_o, 1)        # Q(s|o)
    m = Q_os > 0
    risk = np.sum(Q_o[Q_o > 0] * (np.log(Q_o[Q_o > 0]) - np.log(C[Q_o > 0])))
    ambiguity = -np.sum(Q_os[m] * np.log(P_o_s[m]))
    pragmatic = np.sum(Q_o * np.log(C))
    info_gain = np.sum(Q_os[m] * (np.log(Q_s_o[m]) - np.log(np.broadcast_to(Q_s[:, None], Q_os.shape)[m])))
    return dict(Q_os=Q_os, Q_o=Q_o, Q_s_o=Q_s_o, risk=risk, ambiguity=ambiguity,
                pragmatic=pragmatic, info_gain=info_gain, mask=m)


def test_v12_risk_plus_ambiguity():
    for _ in range(5):
        K, M = 4, 6
        t = _efe_terms(rand_simplex(K), rand_simplex(K, M), rand_simplex(M))
        assert abs((t["risk"] + t["ambiguity"]) - (-t["pragmatic"] - t["info_gain"])) < 1e-12


def test_v13_expected_future_F_equals_G_plus_info_gain():
    for _ in range(5):
        K, M = 4, 6
        Q_s, P_o_s = rand_simplex(K), rand_simplex(K, M)
        t = _efe_terms(Q_s, P_o_s, rand_simplex(M))
        P_os = rand_simplex(K * M).reshape(K, M)    # 임의의 생성 모델 P(o, s)
        Q_os, Q_s_o = t["Q_os"], t["Q_s_o"]
        future_F = np.sum(Q_os * (np.log(Q_s_o) - np.log(P_os)))
        G = np.sum(Q_os * (np.log(Q_s)[:, None] - np.log(P_os)))
        assert abs(future_F - (G + t["info_gain"])) < 1e-12
        assert t["info_gain"] >= 0


def test_v14_deterministic_decoder():
    K, M = 6, 4
    obs_of_state = np.array([0, 0, 1, 2, 3, 3])     # 결정론적(다대일) 디코더
    P_o_s = np.eye(M)[obs_of_state]                 # 원-핫 우도
    Q_s, C = rand_simplex(K), rand_simplex(M)
    t = _efe_terms(Q_s, P_o_s, C)
    Q_o = t["Q_o"]
    H_Qo = -np.sum(Q_o * np.log(Q_o))
    assert abs(t["ambiguity"]) < 1e-15
    G = t["risk"] + t["ambiguity"]
    assert abs(G - np.sum(Q_o * (np.log(Q_o) - np.log(C)))) < 1e-12
    assert abs(t["info_gain"] - H_Qo) < 1e-12
    assert abs(G - (-np.sum(Q_o * np.log(C)) - H_Qo)) < 1e-12


# ---------- V15 ----------
# 기준: 이산 결합분포 p(θ, o)에서 직접 계산한 상호정보량.

def test_v15_ensemble_mutual_information():
    E, M = 5, 7
    p_o_theta = rand_simplex(E, M)                  # 앙상블 멤버별 예측분포 (θ는 균등)
    H = lambda p: -np.sum(p * np.log(p), -1)
    formula = H(p_o_theta.mean(0)) - H(p_o_theta).mean()
    joint = p_o_theta / E
    direct = np.sum(joint * (np.log(joint) - np.log(joint.sum(1, keepdims=True))
                             - np.log(joint.sum(0, keepdims=True))))
    assert abs(formula - direct) < 1e-12 and formula > 0
    # 모든 멤버가 같은 분포면 (순수 우연적 잡음) 0
    same = np.tile(rand_simplex(M), (E, 1))
    assert abs(H(same.mean(0)) - H(same).mean()) < 1e-12
