"""R1 과제: 키마다 신뢰도가 다르고 값이 단서 없이 바뀌는 흐름형 연상 회상. 설계: docs/memory/r1_predictions.md

한 흐름마다 키 K개를 뽑는다. 매 스텝 그중 하나가 (키, 관측값)으로 들어온다.
  - 키별 오염률 η: frac_noisy 비율의 키는 eta_hi, 나머지는 0. 흐름마다 무작위 배정.
  - 키의 참값은 자기 관측 한 번당 확률 hazard로 바뀐다 (스텝당 hazard / K).
채점 대상: 마지막 변화 뒤 한 번 이상 관측된 키의 현재 참값.
"""
from dataclasses import dataclass

import torch

PRESETS = {
    "A": dict(eta_hi=0.0, hazard=0.0),
    "B": dict(eta_hi=0.4, hazard=0.0),
    "C": dict(eta_hi=0.4, hazard=1 / 16),
}


@dataclass
class StreamCfg:
    n_keys: int = 128      # 키 어휘
    n_vals: int = 64       # 값 어휘
    K: int = 8             # 흐름 하나의 키 수
    eta_hi: float = 0.0
    frac_noisy: float = 0.5
    hazard: float = 0.0


class Stream:
    def __init__(self, cfg: StreamCfg, B: int, device="cpu", seed=0):
        self.cfg, self.B, self.device = cfg, B, device
        g = self.g = torch.Generator(device=device).manual_seed(seed)
        K = cfg.K
        self.ar = torch.arange(B, device=device)
        self.kids = torch.rand(B, cfg.n_keys, generator=g, device=device).argsort(1)[:, :K]
        noisy = torch.rand(B, K, generator=g, device=device).argsort(1) < round(cfg.frac_noisy * K)
        self.noisy = noisy & (cfg.eta_hi > 0)
        self.eta = self.noisy.float() * cfg.eta_hi
        self.val = self._randint(cfg.n_vals, (B, K))
        self.since = torch.zeros(B, K, dtype=torch.long, device=device)   # 마지막 변화 뒤 관측 횟수

    def _randint(self, hi, shape, lo=0):
        return torch.randint(lo, hi, shape, generator=self.g, device=self.device)

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.device)

    def step(self):
        """참값 변화 → 키 하나 관측. 반환: 슬롯 j [B], 키 id [B], 관측값 [B]."""
        c = self.cfg
        if c.hazard > 0:
            ch = self._rand(self.B, c.K) < c.hazard / c.K
            new = (self.val + self._randint(c.n_vals, (self.B, c.K), lo=1)) % c.n_vals
            self.val = torch.where(ch, new, self.val)
            self.since = torch.where(ch, torch.zeros_like(self.since), self.since)
        j = self._randint(c.K, (self.B,))
        true = self.val[self.ar, j]
        corrupt = self._rand(self.B) < self.eta[self.ar, j]
        obs = torch.where(corrupt, self._randint(c.n_vals, (self.B,)), true)
        self.since[self.ar, j] += 1
        return j, self.kids[self.ar, j], obs


class BayesFilter:
    """상한: 키별 η와 hazard를 아는 정확한 필터 (값에 대한 범주형 믿음)."""

    def __init__(self, stream: Stream):
        c = stream.cfg
        self.s, self.V, self.hs = stream, c.n_vals, c.hazard / c.K
        self.b = torch.full((stream.B, c.K, c.n_vals), 1.0 / c.n_vals, device=stream.device)

    def write(self, j, obs):
        s = self.s
        self.b = (1 - self.hs) * self.b + self.hs * (1 - self.b) / (self.V - 1)
        eta = s.eta[s.ar, j].unsqueeze(-1)
        lik = eta / self.V + (1 - eta) * torch.nn.functional.one_hot(obs, self.V)
        bj = self.b[s.ar, j] * lik
        self.b[s.ar, j] = bj / bj.sum(-1, keepdim=True)

    def predict(self):
        return self.b.argmax(-1)
