"""R1 기억 모델 (한 층, 같은 기억 행렬 M: [d_v, d_k]). 설계: docs/memory/r1_predictions.md

gdn       Gated DeltaNet. 감쇠 α와 쓰기 세기 β를 토큰에서 예측.
kdn       Kalman Delta Network 대각 변형 (arXiv:2609.07816). 대각 불확실성 p를 전파, α·ω·r은 토큰에서 예측.
ours      델타 쓰기의 이득을 키별 칼만 이득으로 정한다. 키별 불확실성 u(k)와 잡음 σ²(k)를 벡터 두 개에 연상 저장하고,
          σ²는 오차 이력에서 추론, 변화는 σ²로 정규화한 로그 우도비로 감지한다. 게이트 신경망 없음.
ours_fix  ours에서 σ²(k)를 상수로 고정 (정밀도 추론 제거).
read_gate=True (gdn, kdn): 게이트 입력에 쓰기 직전의 읽기 Mk와 오차 v − Mk를 더한다.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Base(nn.Module):
    def __init__(self, n_keys, n_vals, dk=64, dv=64, hidden=128, read_gate=False):
        super().__init__()
        self.dk, self.dv, self.read_gate = dk, dv, read_gate
        self.ek = nn.Embedding(n_keys, dk)
        self.ev = nn.Embedding(n_vals, dv)
        self.out = nn.Linear(dv, n_vals)
        self.tok = nn.Sequential(nn.Linear(dk + dv * (3 if read_gate else 1), hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())

    def key(self, kid):
        return F.normalize(self.ek(kid), dim=-1)

    def feat(self, kid, v, m):
        x = [self.ek(kid), v] + ([m, v - m] if self.read_gate else [])
        return self.tok(torch.cat(x, -1))

    def init(self, B, device):
        return dict(M=torch.zeros(B, self.dv, self.dk, device=device))

    def read(self, st, kids):
        """kids [B, K] → 로짓 [B, K, n_vals]."""
        return self.out(torch.einsum("bvk,bnk->bnv", st["M"], self.key(kids)))


def outer(a, b):
    return a.unsqueeze(-1) * b.unsqueeze(1)


class GDN(Base):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gate = nn.Linear(self.tok[-2].out_features, 2)
        nn.init.constant_(self.gate.bias, 0.0)
        self.gate.bias.data[0] = 4.0                      # α ≈ 0.98에서 출발

    def write(self, st, kid, vid):
        M, k, v = st["M"], self.key(kid), self.ev(vid)
        m = (M @ k.unsqueeze(-1)).squeeze(-1)
        g = torch.sigmoid(self.gate(self.feat(kid, v, m)))
        a, b = g[:, :1, None], g[:, 1:, None]
        return dict(M=a * (M - b * outer(m, k)) + b * outer(v, k))


class KDN(Base):
    R_MIN = 1e-3

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        h = self.tok[-2].out_features
        self.alpha, self.omega, self.r = nn.Linear(h, self.dk), nn.Linear(h, self.dk), nn.Linear(h, 1)
        nn.init.constant_(self.alpha.bias, 4.0)
        nn.init.zeros_(self.omega.weight); nn.init.constant_(self.omega.bias, -4.0)
        nn.init.constant_(self.r.bias, -2.0)
        self.c0 = nn.Parameter(torch.tensor(0.5))        # 초기 불확실성 (softplus)
        self.mu = float(self.dk)                          # 정보 배율 (논문 실험값 μ = d_k)

    def init(self, B, device):
        st = super().init(B, device)
        st["p"] = F.softplus(self.c0).expand(B, self.dk)
        return st

    def write(self, st, kid, vid):
        M, p, k, v = st["M"], st["p"], self.key(kid), self.ev(vid)
        x = self.feat(kid, v, (M @ k.unsqueeze(-1)).squeeze(-1))
        al = torch.sigmoid(self.alpha(x))
        om = F.softplus(self.omega(x))
        r = self.R_MIN + F.softplus(self.r(x))
        Mh, ph = M * al.unsqueeze(1), al * al * p + om
        m = (Mh @ k.unsqueeze(-1)).squeeze(-1)
        kappa = ph * k / (r + (ph * k * k).sum(-1, keepdim=True))
        return dict(M=Mh + outer(v - m, kappa), p=1.0 / (1.0 / ph + self.mu * k * k / r))


class Ours(Base):
    SIG_MIN = 1e-3

    def __init__(self, *a, infer_noise=True, **kw):
        super().__init__(*a, **kw)
        del self.tok                                       # 게이트 신경망 없음
        self.infer_noise = infer_noise
        P = lambda x: nn.Parameter(torch.tensor(float(x)))
        self.log_tau2 = P(0.0)                             # 값의 사전 분산 (차원당)
        self.log_sig0 = P(math.log(0.1))                   # 잡음 분산의 사전 평균
        self.logit_h = P(math.log(0.05 / 0.95))            # 관측당 변화의 사전 확률
        self.log_nu = P(math.log(4.0))                     # 오차의 유효 자유도
        self.logit_eta = P(0.0)                            # 잡음 추정의 갱신율
        self.logit_rho = P(0.0)                            # 변화로 돌린 오차를 잡음 추정에서 빼는 정도

    def init(self, B, device):
        st = super().init(B, device)
        st["a"] = torch.zeros(B, self.dk, device=device)   # 1 − u(k)를 저장: 0이면 모르는 키
        st["s"] = torch.zeros(B, self.dk, device=device)   # σ²(k) − σ₀²를 저장
        return st

    def scalars(self):
        return dict(tau2=self.log_tau2.exp().item(), sig0=self.log_sig0.exp().item(),
                    h=torch.sigmoid(self.logit_h).item(), nu=self.log_nu.exp().item(),
                    eta=torch.sigmoid(self.logit_eta).item(), rho=torch.sigmoid(self.logit_rho).item())

    def write(self, st, kid, vid, info=None):
        M, a, s, k, v = st["M"], st["a"], st["s"], self.key(kid), self.ev(vid)
        tau2, sig0, nu = self.log_tau2.exp(), self.log_sig0.exp(), self.log_nu.exp()
        m = (M @ k.unsqueeze(-1)).squeeze(-1)
        eps = v - m
        e2 = eps.pow(2).mean(-1)
        u = (1.0 - (a * k).sum(-1)).clamp(1e-4, 1.0)
        sig2 = self.SIG_MIN + F.relu(sig0 + (s * k).sum(-1))
        # 변화의 증거: 오차가 "그대로" (분산 S0)보다 "새 값" (분산 S1)에서 더 그럴듯한가
        S0 = tau2 * u + sig2
        S1 = tau2 + m.pow(2).mean(-1) + sig2
        llr = 0.5 * nu * (e2 * (1 / S0 - 1 / S1) - torch.log(S1 / S0))
        c = torch.sigmoid(llr + self.logit_h)
        uh = u + c * (1 - u)                               # 예측: 변화 확률만큼 불확실성을 사전으로 되돌림
        beta = tau2 * uh / (tau2 * uh + sig2)              # 칼만 이득
        M = M + beta[:, None, None] * outer(eps, k)
        un = (1 - beta) * uh
        a = a + ((1 - un) - (a * k).sum(-1)).unsqueeze(-1) * k
        if self.infer_noise:                               # 정밀도 추론: 오차에서 기억 불확실성 몫을 뺀 나머지가 잡음
            z = (e2 - tau2 * u).clamp(min=0)
            w = torch.sigmoid(self.logit_eta) * (1 - torch.sigmoid(self.logit_rho) * c)
            s = s + (w * (z - sig2)).unsqueeze(-1) * k
        if info is not None:
            info.update(c=c, beta=beta, sig2=sig2, u=u)
        return dict(M=M, a=a, s=s)


class Simple(Base):
    """자유에너지의 경사 하나에서 나오는 가장 단순한 쓰기 (Bogacz 2017의 식 25, 63).
    M ← M + β ε kᵀ, β = τ / (τ + Σ(k)).  Σ(k) ← Σ(k) + η (ε² − Σ(k)): 그 키에서 평소 얼마나 틀리는가.
    불확실성 u와 변화 확률 c는 없다. 학습되는 스칼라는 τ, Σ₀, η 셋."""
    SIG_MIN = 1e-3

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        del self.tok
        P = lambda x: nn.Parameter(torch.tensor(float(x)))
        self.log_tau, self.log_sig0, self.logit_eta = P(0.0), P(math.log(0.1)), P(0.0)

    def init(self, B, device):
        st = super().init(B, device)
        st["s"] = torch.zeros(B, self.dk, device=device)
        return st

    def scalars(self):
        return dict(tau=self.log_tau.exp().item(), sig0=self.log_sig0.exp().item(), eta=torch.sigmoid(self.logit_eta).item())

    def write(self, st, kid, vid):
        M, s, k, v = st["M"], st["s"], self.key(kid), self.ev(vid)
        eps = v - (M @ k.unsqueeze(-1)).squeeze(-1)
        e2 = eps.pow(2).mean(-1)
        sig2 = self.SIG_MIN + F.relu(self.log_sig0.exp() + (s * k).sum(-1))
        tau = self.log_tau.exp()
        beta = tau / (tau + sig2)
        return dict(M=M + beta[:, None, None] * outer(eps, k),
                    s=s + (torch.sigmoid(self.logit_eta) * (e2 - sig2)).unsqueeze(-1) * k)


def build(name, cfg, read_gate=False, **kw):
    a = (cfg.n_keys, cfg.n_vals)
    if name == "gdn":
        return GDN(*a, read_gate=read_gate, **kw)
    if name == "kdn":
        return KDN(*a, read_gate=read_gate, **kw)
    if name in ("ours", "ours_fix"):
        return Ours(*a, infer_noise=name == "ours", **kw)
    if name == "simple":
        return Simple(*a, **kw)
    raise ValueError(name)
