"""진단: 저장된 불확실성 u(k)가 기억 M의 실제 오차와 맞는가. 흐름의 키 수 K를 늘려 포화를 본다.

python scripts/memory/diag_u.py runs/memory/A_ours.pt [K ...]
실제 오차 = mean((Mk − e(참값))²) / τ². u가 맞게 추적되면 둘이 같은 크기여야 한다.
"""
import sys

import torch

from rltdp.memory.eval import load
from rltdp.memory.stream import Stream, StreamCfg

path = sys.argv[1]
Ks = [int(x) for x in sys.argv[2:]] or [8, 16, 32, 64, 96]
model, cfg0, ck = load(path)
model.eval()
tau2 = model.log_tau2.exp()
print(f"{path} 조건 {ck['cond']}  스칼라 {({k: round(v, 3) for k, v in model.scalars().items()})}")
print("K | 정확도 | u 평균 | 실제 오차/τ² 평균 | u < 0.1인데 실제 > 0.5인 비율 | 키 겹침 |k·k'| 평균")
with torch.no_grad():
    for K in Ks:
        cfg = StreamCfg(**{**ck["cfg"], "K": K})
        s = Stream(cfg, 64, "cpu", seed=123)
        st = model.init(64, "cpu")
        T = 32 * K                                         # 키당 평균 32번 관측
        tot = dict(n=0.0, acc=0.0, u=0.0, err=0.0, bad=0.0)
        for t in range(T):
            j, kid, obs = s.step()
            st = model.write(st, kid, obs)
            if t < T // 2:
                continue
            q = model.key(s.kids)                           # [B, K, dk]
            mk = torch.einsum("bvk,bnk->bnv", st["M"], q)
            u = (1 - torch.einsum("bk,bnk->bn", st["a"], q)).clamp(1e-4, 1)
            err = (mk - model.ev(s.val)).pow(2).mean(-1) / tau2
            mask = (s.since > 0) & ~s.noisy
            hit = model.out(mk).argmax(-1) == s.val
            tot["n"] += mask.sum().item(); tot["acc"] += (hit & mask).sum().item()
            tot["u"] += (u * mask).sum().item(); tot["err"] += (err * mask).sum().item()
            tot["bad"] += ((u < 0.1) & (err > 0.5) & mask).sum().item()
        ov = torch.einsum("bnk,bmk->bnm", q, q).abs()
        ov = (ov.sum() - ov.diagonal(dim1=1, dim2=2).sum()) / (64 * K * (K - 1))
        n = tot["n"]
        print(f"{K:3d} | {tot['acc'] / n:.3f} | {tot['u'] / n:.3f} | {tot['err'] / n:.3f} | {tot['bad'] / n:.3f} | {ov:.3f}")
