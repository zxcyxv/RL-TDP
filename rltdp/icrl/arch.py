"""B: 플래토로 관리하는 충분통계 기억을 신경망 구조로. 문맥 안 정책 평가를 사전학습한다 (CPU, torch).

과제군: 무작위 MRP (상태 n = 16, Dirichlet(0.3) 전이). 상태마다 숨은 특징 z(s) ~ N(0, I_4)이고 관측은 모든 과제가
공유하는 고정 비선형 사상 o(s) = tanh(U z(s)) (16차원). 보상 r(s) = z(s)ᵀθ, θ는 K = 400스텝마다 단서 없이 θ1 ↔ θ2.
사전학습 목표: 현재 국면의 참 가치 V (모든 상태의 관측을 질의로), 정규화 제곱 오차 Σ(V̂ − V)² / ΣV². 과제는 매 반복 새로 뽑는다.

학습되는 것은 인코더 φ = f_θ(o) ∈ R^5 (4차원 + 상수, 노름 1)뿐이다. 기억 규칙은 구조로 고정한다:
    전이마다 외적 O_t = φ_t (φ_t − γ φ_{t+1})ᵀ,  r_t φ_t  를 누적하고, 읽기 w = (A + εI)⁻¹ b,  V̂(s) = φ(s)ᵀ w
--model plateau  빠른 기억 (극점 20)과 긴 기억 (실행 길이 가중, 상한 200). 정밀도 가중 로그 우도비
                 L_t = (1 − 1/20) L_{t−1} + (δ_긴² − δ_빠름²) / (2σ²) 가 ln(1/h)를 넘으면 (플래토) 긴 기억 ← 빠른 기억,
                 실행 길이 20에서 다시 쌓음. 결정은 경사를 받지 않고 경사는 통계를 통해서만 흐른다. 읽기 = 긴 기억
--model fast / slow / window / cum   같은 구조에서 플래토 없이: 극점 20 / 극점 200 / 최근 200개 창 / 망각 없음
--model tf       비교군: 문맥 창 softmax Transformer (최근 200개 전이, 2층, 상태 질의 cross-attention)
--model gdn      비교군: Gated DeltaNet. 전이 토큰 [o_t, r_t, o_{t+1}]에서 k, v, 학습되는 게이트 α_t (감쇠), β_t (쓰기)
                 S_t = α_t S_{t−1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ,  v_t는 다음 관측의 기억 읽기를 입력으로 받는다 (TD 가능)
                 V̂(s) = u·S q(o(s)). 같은 목표로 끝까지 학습
python -m rltdp.icrl.arch --model plateau [--device cuda] [--iters 2000 --eval_every 250 --seed 0]
결과: <out>/<model>_s<seed>.json (학습 전·후 평가, 학습 곡선, 인자), .pt (가중치)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

GAMMA, N_STATE, M_OBS, K_LAT = 0.9, 16, 16, 4


def world(seed=1234):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(M_OBS, K_LAT, generator=g) * 1.5 / math.sqrt(K_LAT)


def sample_tasks(rng, U, B, T, K, device="cpu"):
    n = N_STATE
    z = rng.normal(size=(B, n, K_LAT))
    obs = np.tanh(z @ U.numpy().T)
    P = rng.dirichlet(np.full(n, 0.3), size=(B, n))
    theta = rng.normal(size=(B, 2, K_LAT)) / math.sqrt(K_LAT)
    r = np.einsum("bnk,bpk->bpn", z, theta)                                    # [B, 2, n]
    V = np.linalg.solve(np.eye(n)[None, None] - GAMMA * P[:, None], r[..., None])[..., 0]
    cum = np.cumsum(P, -1)
    s = np.empty((B, T + 1), np.int64)
    s[:, 0] = rng.integers(n, size=B)
    ar = np.arange(B)
    for t in range(T):
        s[:, t + 1] = np.minimum((cum[ar, s[:, t]] < rng.random(B)[:, None]).sum(1), n - 1)
    phase = (np.arange(T) // K) % 2
    R = r[ar[:, None], phase[None], s[:, :-1]]
    f = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    return dict(obs=f(obs), s=torch.as_tensor(s, device=device), R=f(R), V=f(V), phase=torch.as_tensor(phase), K=K)


class Enc(nn.Module):
    def __init__(self, d=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(M_OBS, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, d - 1))

    def forward(self, o):
        h = torch.cat([self.net(o), torch.ones_like(o[..., :1])], -1)
        return h / h.norm(dim=-1, keepdim=True)


def solve(A, b, ridge):
    return torch.linalg.solve(A + ridge * torch.eye(A.shape[-1], device=A.device), b.unsqueeze(-1)).squeeze(-1)


class MemModel(nn.Module):
    def __init__(self, variant, d=5, nf=20, ns=200, hazard=1e-3, ridge=1e-3):
        super().__init__()
        self.variant, self.d, self.nf, self.ns, self.ridge = variant, d, nf, ns, ridge
        self.thr = math.log(1 / hazard)
        self.enc = Enc(d)

    def forward(self, task, every=20):
        """체크포인트 (t, V̂ [B, n])의 목록과 플래토 횟수."""
        Phi = self.enc(task["obs"])                                            # [B, n, d]
        s, R = task["s"], task["R"]
        B, T = R.shape
        idx = lambda ss: ss.unsqueeze(-1).expand(-1, -1, self.d)
        X, Y = Phi.gather(1, idx(s[:, :-1])), Phi.gather(1, idx(s[:, 1:]))   # [B, T, d]
        O = X.unsqueeze(-1) * (X - GAMMA * Y).unsqueeze(-2)                    # [B, T, d, d]
        Bv = R.unsqueeze(-1) * X
        outs, events, v = [], 0, self.variant
        if v in ("window", "cum"):
            CO = torch.cat([O.new_zeros(B, 1, self.d, self.d), O.cumsum(1)], 1)
            CB = torch.cat([Bv.new_zeros(B, 1, self.d), Bv.cumsum(1)], 1)
            for t in range(every - 1, T, every):
                lo = max(0, t + 1 - self.ns) if v == "window" else 0
                A, b = (CO[:, t + 1] - CO[:, lo]) / (t + 1 - lo), (CB[:, t + 1] - CB[:, lo]) / (t + 1 - lo)
                outs.append((t, Phi @ solve(A, b, self.ridge).unsqueeze(-1)))
            return [(t, y.squeeze(-1)) for t, y in outs], events
        zero = lambda: (O.new_zeros(B, self.d, self.d), O.new_zeros(B, self.d))
        (Af, bf), (As, bs) = zero(), zero()
        n_run = O.new_zeros(B)
        L, sig2 = O.new_zeros(B), O.new_ones(B)
        wf, ws = O.new_zeros(B, self.d), O.new_zeros(B, self.d)
        zf, zs = 1 - 1 / self.nf, 1 - 1 / self.ns
        for t in range(T):
            o, bb = O[:, t], Bv[:, t]
            if v == "plateau":
                with torch.no_grad():
                    x, y, r = X[:, t], Y[:, t], R[:, t]
                    df = r + GAMMA * (y * wf).sum(-1) - (x * wf).sum(-1)
                    ds = r + GAMMA * (y * ws).sum(-1) - (x * ws).sum(-1)
                    sig2 = sig2 + (df ** 2 - sig2) / min(t + 1, self.ns)
                    L = (1 - 1 / self.nf) * L + (ds ** 2 - df ** 2) / (2 * sig2 + 1e-12)
            if v in ("fast", "plateau"):
                Af, bf = zf * Af + (1 - zf) * o, zf * bf + (1 - zf) * bb
            if v == "slow":
                As, bs = zs * As + (1 - zs) * o, zs * bs + (1 - zs) * bb
            if v == "plateau":
                n_run = torch.clamp(n_run + 1, max=self.ns)
                g = (1 / n_run)[:, None]
                As, bs = (1 - g[..., None]) * As + g[..., None] * o, (1 - g) * bs + g * bb
                ev = L > self.thr
                if ev.any():
                    As = torch.where(ev[:, None, None], Af, As); bs = torch.where(ev[:, None], bf, bs)
                    n_run = torch.where(ev, torch.full_like(n_run, self.nf), n_run)
                    L = torch.where(ev, torch.zeros_like(L), L)
                    events += int(ev.sum())
                with torch.no_grad():
                    wf, ws = solve(Af.detach(), bf.detach(), self.ridge), solve(As.detach(), bs.detach(), self.ridge)
            if (t + 1) % every == 0:
                A, b = (Af, bf) if v == "fast" else (As, bs)
                outs.append((t, (Phi @ solve(A, b, self.ridge).unsqueeze(-1)).squeeze(-1)))
        return outs, events


class GDN(nn.Module):
    def __init__(self, dk=16, dv=16, hidden=64):
        super().__init__()
        self.dk, self.dv = dk, dv
        self.tok = nn.Sequential(nn.Linear(2 * M_OBS + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.key = nn.Linear(hidden, dk)
        self.qry = nn.Sequential(nn.Linear(M_OBS, hidden), nn.GELU(), nn.Linear(hidden, dk))
        self.val = nn.Sequential(nn.Linear(hidden + dv, hidden), nn.GELU(), nn.Linear(hidden, dv))
        self.gate = nn.Linear(hidden, 2)
        self.out = nn.Linear(dv, 1)

    def forward(self, task, every=20):
        obs, s, R = task["obs"], task["s"], task["R"]
        B, T = R.shape
        ob = lambda ss: obs.gather(1, ss.unsqueeze(-1).expand(-1, -1, M_OBS))
        Ot, On = ob(s[:, :-1]), ob(s[:, 1:])
        H = self.tok(torch.cat([Ot, R.unsqueeze(-1), On], -1))                  # [B, T, h]
        Kt = F.normalize(self.key(H), dim=-1)
        Qn = F.normalize(self.qry(On), dim=-1)
        Qall = F.normalize(self.qry(obs), dim=-1)                                # [B, n, dk]
        G = torch.sigmoid(self.gate(H))
        S = H.new_zeros(B, self.dv, self.dk)
        outs = []
        for t in range(T):
            k, a, be = Kt[:, t], G[:, t, :1, None], G[:, t, 1:, None]
            read_next = (S @ Qn[:, t].unsqueeze(-1)).squeeze(-1)
            vv = self.val(torch.cat([H[:, t], read_next], -1))
            Sk = (S @ k.unsqueeze(-1))                                           # [B, dv, 1]
            S = a * (S - be * Sk @ k.unsqueeze(1)) + be * vv.unsqueeze(-1) @ k.unsqueeze(1)
            if (t + 1) % every == 0:
                outs.append((t, self.out(torch.einsum("bvk,bnk->bnv", S, Qall)).squeeze(-1)))
        return outs, 0


class TF(nn.Module):
    """비교군: 문맥 창 softmax Transformer. 질의 시점 t의 문맥 = 최근 C개 전이 토큰 [o, r, o'] (+ 나이 임베딩),
    문맥 안 전체 self-attention 2층 (pre-norm), 상태마다 관측 질의가 문맥을 cross-attention으로 읽어 V̂(s)를 낸다.
    문맥 밖 전이는 버린다 (창). 기억 = C개 토큰, 질의당 계산 ∝ C²."""

    def __init__(self, C=200, d=32, heads=2, layers=2, hidden=64):
        super().__init__()
        self.C = C
        self.emb = nn.Sequential(nn.Linear(2 * M_OBS + 1, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.age = nn.Parameter(torch.zeros(C, d))
        self.enc = nn.ModuleList(nn.TransformerEncoderLayer(d, heads, hidden, dropout=0.0, batch_first=True,
                                                            norm_first=True) for _ in range(layers))
        self.qry = nn.Sequential(nn.Linear(M_OBS, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.read = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, task, every=20, chunk=15):
        obs, s, R = task["obs"], task["s"], task["R"]
        B, T = R.shape
        ob = lambda ss: obs.gather(1, ss.unsqueeze(-1).expand(-1, -1, M_OBS))
        X = self.emb(torch.cat([ob(s[:, :-1]), R.unsqueeze(-1), ob(s[:, 1:])], -1))       # [B, T, d]
        X = torch.cat([X.new_zeros(B, self.C - 1, X.shape[-1]), X], 1)                    # 왼쪽 덧댐
        valid = torch.cat([torch.zeros(self.C - 1, dtype=torch.bool, device=X.device),
                           torch.ones(T, dtype=torch.bool, device=X.device)])
        Q = self.qry(obs)                                                                  # [B, n, d]
        ts = list(range(every - 1, T, every))
        outs = []
        for i in range(0, len(ts), chunk):
            tt = ts[i:i + chunk]
            seg = torch.stack([X[:, t:t + self.C] for t in tt], 1).flatten(0, 1)          # [B·G, C, d] (끝 = t)
            pad = ~torch.stack([valid[t:t + self.C] for t in tt]).repeat(B, 1)            # [B·G, C]
            h = seg + self.age.flip(0)                                                     # 마지막 토큰의 나이 0
            for layer in self.enc:
                h = layer(h, src_key_padding_mask=pad)
            q = Q.unsqueeze(1).expand(-1, len(tt), -1, -1).flatten(0, 1)                  # [B·G, n, d]
            r, _ = self.read(q, h, h, key_padding_mask=pad, need_weights=False)
            v = self.out(self.norm(q + r)).squeeze(-1).view(B, len(tt), -1)
            outs += [(t, v[:, j]) for j, t in enumerate(tt)]
        return outs, 0


def errors(outs, task):
    """체크포인트마다 과제별 정규화 오차 [C, B]."""
    V, ph = task["V"], task["phase"]
    es = []
    for t, Vh in outs:
        Vt = V[:, int(ph[t])]
        es.append(((Vh - Vt) ** 2).sum(-1) / (Vt ** 2).sum(-1))
    return torch.stack(es)


def evaluate(model, U, T, K, n_tasks=64, seed=777, device="cpu"):
    rng = np.random.default_rng(seed)
    task = sample_tasks(rng, U, n_tasks, T, K, device)
    with torch.no_grad():
        outs, events = model(task)
        E = errors(outs, task)                                                   # [C, B]
    ts = np.array([t for t, _ in outs])
    since = ts % K
    after = (ts >= K)
    sel = lambda m: torch.as_tensor(m, device=E.device)
    pick = lambda m: float(E[sel(m & after)].median(0).values.median()) if (m & after).any() else float("nan")
    per_task = E[sel(after)].mean(0)
    return dict(overall=float(per_task.median()), after_switch=pick(since < 100), steady=pick(since >= K - 200),
                plateaus_per_switch=events / n_tasks / max(1, T // K - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["plateau", "fast", "slow", "window", "cum", "gdn", "tf"], required=True)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--T", type=int, default=1200)
    ap.add_argument("--K", type=int, default=400)
    ap.add_argument("--ctx", type=int, default=200, help="tf의 문맥 길이 C")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4, help="CPU일 때만 의미")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval_every", type=int, default=0, help="0이면 끝에만. 아니면 이 반복마다 과제 32개로 평가")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/icrl_arch")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    U = world()
    dev = torch.device(args.device)
    model = (GDN() if args.model == "gdn" else TF(C=args.ctx) if args.model == "tf" else MemModel(args.model)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    pre = dict(T1200=evaluate(model, U, args.T, args.K, device=dev))
    print(json.dumps(dict(model=args.model, iter=0, eval=pre["T1200"])), flush=True)
    for it in range(1, args.iters + 1):
        task = sample_tasks(rng, U, args.batch, args.T, args.K, dev)
        outs, _ = model(task)
        loss = errors(outs, task).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if it % 50 == 0:
            rec = dict(model=args.model, iter=it, loss=round(float(loss.detach()), 4), sec=round(time.time() - t0, 1))
            if args.eval_every and it % args.eval_every == 0:
                rec["eval"] = evaluate(model, U, args.T, args.K, n_tasks=32, device=dev)
            log.append(rec); print(json.dumps(rec), flush=True)
    res = dict(model=args.model, before=pre, T1200=evaluate(model, U, args.T, args.K, device=dev),
               T3600=evaluate(model, U, 3 * args.T, args.K, device=dev), log=log, args=vars(args))
    print(json.dumps({k: res[k] for k in ("model", "T1200", "T3600")}), flush=True)
    name = f"{args.model}{f'_C{args.ctx}' if args.model == 'tf' and args.ctx != 200 else ''}_s{args.seed}"
    (out / f"{name}.json").write_text(json.dumps(res))
    torch.save(model.state_dict(), out / f"{name}.pt")


if __name__ == "__main__":
    main()
