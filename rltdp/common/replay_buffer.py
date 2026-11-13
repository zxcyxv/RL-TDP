"""계열 공통: 에피소드 단위 재현 버퍼 (MiniGrid 격자 관측). 쓰는 곳: rltdp.fep.agent.run, rltdp.btsp.tiny / delta."""
import numpy as np
import torch


class Buffer:
    """GPU에 미리 잡아 둔 재현 버퍼. 에피소드 단위로 넣고, 실험 2a와 같은 방식으로 조각을 뽑는다."""

    def __init__(self, capacity, device):
        self.dev, self.cap, self.n = device, capacity, 0
        self.grid = torch.zeros(capacity, 7, 7, 3, dtype=torch.uint8, device=device)
        z = lambda: torch.zeros(capacity, dtype=torch.long, device=device)
        self.dir, self.outcome, self.action, self.segment = z(), z(), z(), z()
        self.starts, self.lens = [], []
        self._ep = None

    def add(self, ep):
        L = len(ep["dir"])
        if self.n + L > self.cap:
            raise RuntimeError("버퍼 용량 초과")
        sl = slice(self.n, self.n + L)
        self.grid[sl] = torch.as_tensor(np.stack(ep["grid"]), device=self.dev)
        for k in ("dir", "outcome", "action", "segment"):
            getattr(self, k)[sl] = torch.as_tensor(np.asarray(ep[k]), device=self.dev)
        self.starts.append(self.n); self.lens.append(L); self.n += L
        self._ep = None

    def sample(self, B, L, gen):
        if self._ep is None:
            self._ep = (torch.tensor(self.starts, device=self.dev), torch.tensor(self.lens, device=self.dev))
        ep_start, ep_len = self._ep
        ep = torch.randint(len(ep_len), (B,), device=self.dev, generator=gen)
        n = ep_len[ep]
        off = (torch.rand(B, device=self.dev, generator=gen) * ((n - L).clamp_min(0) + 1)).long()
        lens = torch.minimum(n - off, torch.full_like(n, L))
        ar = torch.arange(L, device=self.dev)
        idx = (ep_start[ep] + off)[:, None] + torch.minimum(ar[None], (lens - 1)[:, None])
        return dict(grid=self.grid[idx].long(), dir=self.dir[idx], outcome=self.outcome[idx],
                    action=self.action[idx], segment=self.segment[idx], mask=ar[None] < lens[:, None])
