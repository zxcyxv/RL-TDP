"""실험 2a 시퀀스 데이터. 전부 GPU에 올려 두고 조각을 뽑는다.

조각 뽑기 (2단계 문서 "실험 2a > 학습"): 에피소드를 길이와 무관하게 균등하게 고르고, 그 안에서 시작점을
균등하게 고른다. 에피소드가 L보다 짧으면 처음부터 전체를 쓰고 나머지는 마스크한다.
행 t: 관측 o_t와 그 뒤의 행동 a_t (에피소드 마지막 관측은 a = -1).
"""
import numpy as np
import torch


class SeqData:
    def __init__(self, path, device):
        d = np.load(path)
        self.device = device
        t = lambda k, dt=torch.long: torch.as_tensor(d[k].astype(np.int64), device=device).to(dt)
        self.grid = t("grid", torch.uint8)                # (N, 7, 7, 3)
        self.dir, self.outcome, self.segment = t("dir"), t("outcome"), t("segment")
        self.action = t("action")
        self.ep_start, self.ep_len = t("ep_start"), t("ep_len")
        self.policy = t("policy")
        self.n_eps = len(self.ep_len)
        self.segment_names = tuple(str(x) for x in d["segment_names"]) if "segment_names" in d else \
            ("room", "corridor", "junction")

    def _gather(self, starts, lens, L):
        ar = torch.arange(L, device=self.device)
        valid = ar[None] < lens[:, None]
        idx = starts[:, None] + torch.minimum(ar[None], (lens - 1).clamp_min(0)[:, None])
        action = self.action[idx].clamp_min(0)            # 마지막(-1)과 패딩은 0으로, 마스크로 걸러진다
        return dict(grid=self.grid[idx].long(), dir=self.dir[idx], outcome=self.outcome[idx],
                    action=action, segment=self.segment[idx], mask=valid)

    def sample(self, B, L, gen):
        ep = torch.randint(self.n_eps, (B,), device=self.device, generator=gen)
        n = self.ep_len[ep]
        max_start = (n - L).clamp_min(0)
        off = (torch.rand(B, device=self.device, generator=gen) * (max_start + 1)).long()
        return self._gather(self.ep_start[ep] + off, torch.minimum(n - off, torch.full_like(n, L)), L)

    def chunks(self, L):
        """보류 평가용: 에피소드마다 L 길이의 겹치지 않는 조각으로 전부 나눈다 (조각 시작에서 h = 0)."""
        starts, lens = [], []
        for s, n in zip(self.ep_start.tolist(), self.ep_len.tolist()):
            for o in range(0, n, L):
                starts.append(s + o); lens.append(min(L, n - o))
        st = torch.tensor(starts, device=self.device)
        ln = torch.tensor(lens, device=self.device)
        return st, ln

    def batch_from(self, starts, lens, L):
        return self._gather(starts, lens, L)
