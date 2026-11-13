"""실험 1의 두 선택지: 고리와 가우시안. 2단계 문서 "실험 1 > 과제" 표를 그대로 따른다."""
import math

import numpy as np
import torch
from scipy.integrate import quad
from scipy.stats import norm

RING_R_MEAN, RING_R_STD = 1.0, 0.1
GAUSS_STD = 0.388854          # 문서 값. 이 값의 엔트로피는 0.94877 (review-notes.md 2절 2번)
CHOICES = ("ring", "gauss")    # 선택지 인덱스 0 = 고리, 1 = 가우시안


def ring_entropy() -> float:
    """고리의 미분 엔트로피 H = H(r) + ln 2π + E[ln r] (극좌표 야코비안 r)."""
    h_r = 0.5 * math.log(2 * math.pi * math.e * RING_R_STD**2)
    e_ln_r, _ = quad(lambda r: norm.pdf(r, RING_R_MEAN, RING_R_STD) * math.log(r), 1e-9, 3)
    return h_r + math.log(2 * math.pi) + e_ln_r


def gauss_entropy() -> float:
    return math.log(2 * math.pi * math.e * GAUSS_STD**2)


ENTROPY = {"ring": ring_entropy(), "gauss": gauss_entropy()}


def sample(choice: int, n: int, gen: torch.Generator, device) -> torch.Tensor:
    if CHOICES[choice] == "ring":
        r = RING_R_MEAN + RING_R_STD * torch.randn(n, generator=gen, device=device)
        th = 2 * math.pi * torch.rand(n, generator=gen, device=device)
        return torch.stack([r * th.cos(), r * th.sin()], -1)
    return GAUSS_STD * torch.randn(n, 2, generator=gen, device=device)


def true_logpdf(choice: int, o: np.ndarray) -> np.ndarray:
    if CHOICES[choice] == "ring":
        r = np.linalg.norm(o, axis=-1)
        return norm.logpdf(r, RING_R_MEAN, RING_R_STD) - np.log(2 * np.pi * r)
    return norm.logpdf(o, 0, GAUSS_STD).sum(-1)
