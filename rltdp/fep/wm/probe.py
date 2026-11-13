"""중간 체크포인트 판정: M0의 손실 중 얼마가 들어 올림 간극이고 얼마가 모델 자체의 놀람인가.

고정된 보류 프레임 부분집합(기본 256개)에서
- F̂ = E_q[−ln p(v|h) + ln q(v|x)] (+ 방향·결과 항): 학습 목적함수를 샘플 평균으로 정확히
- ℑ̂_S = −ln[(1/S) Σ p/q]: S = 10², 10³, 10⁴, 10⁵. 참 놀람의 상한 추정이고 S가 커지면 내려간다
- 제안분포 τ = 1(학습과 같은 q), 2, 3 (측정 전용으로 노이즈 폭을 넓혀 범주 경계를 더 덮는다)
- 유효 샘플 비율
M1 체크포인트에도 같은 프레임에서 ℑ̂_S를 잰다 (비교용).

python -m rltdp.fep.wm.probe --ckpt runs/exp2a/m0_s0/ckpt/u040000.pt --model m0
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from rltdp.common.minigrid_env import GROUP_SIZES
from rltdp.fep.flows.argmax_lift import lift
from rltdp.fep.wm.common import grid_onehot
from rltdp.fep.wm.data import SeqData
from rltdp.fep.wm.models import build


@torch.no_grad()
def frames_subset(model, data, n, seed, L=32):
    starts, lens = data.chunks(L)
    hs, es, gs, ds, os_, segs = [], [], [], [], [], []
    for b0 in range(0, len(starts), 64):
        b = data.batch_from(starts[b0:b0 + 64], lens[b0:b0 + 64], L)
        h, e = model.sk.rollout(grid_onehot(b["grid"]), b["dir"], b["action"])
        m = b["mask"]
        hs.append(h[m]); es.append(e[m]); gs.append(b["grid"][m]); ds.append(b["dir"][m])
        os_.append(b["outcome"][m]); segs.append(b["segment"][m])
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = sum(len(x) for x in hs)
    idx = torch.randperm(N, generator=g)[:n].to(hs[0].device)
    cat = lambda xs: torch.cat(xs)[idx]
    return cat(hs), cat(es), cat(gs), cat(ds), cat(os_), cat(segs)


@torch.no_grad()
def stream_is(logw_fn, n_frames, S_max, S_marks, chunk, frame_block):
    """프레임마다 로그가중치를 흘려 받으며 S_marks 시점의 log-mean-exp, 평균 로그가중치, ESS를 기록한다."""
    res = {S: [] for S in S_marks}
    mean_lw, ess = [], []
    for f0 in range(0, n_frames, frame_block):
        fs = slice(f0, min(f0 + frame_block, n_frames))
        lse = None; s1 = 0.0; done = 0
        lse2 = None
        marks = sorted(res)
        while done < S_max:
            nxt = next(m for m in marks if m > done)
            k = min(chunk, nxt - done)                                   # 기록 시점에 정확히 멈추도록
            lw = logw_fn(fs, k)                                          # (frames, k)
            c_lse = torch.logsumexp(lw, 1)
            c_lse2 = torch.logsumexp(2 * lw, 1)
            lse = c_lse if lse is None else torch.logaddexp(lse, c_lse)
            lse2 = c_lse2 if lse2 is None else torch.logaddexp(lse2, c_lse2)
            s1 = s1 + lw.sum(1)
            done += k
            if done in res:
                res[done].append(lse - math.log(done))
        mean_lw.append(s1 / done)
        ess.append(torch.exp(2 * lse - lse2) / done)
    return {S: torch.cat(v) for S, v in res.items()}, torch.cat(mean_lw), torch.cat(ess)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", choices=["m0", "m1"], required=True)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S_max", type=int, default=100_000)
    ap.add_argument("--taus", default="1,2,3")
    ap.add_argument("--data", default="data/memory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda"
    model = build(args.model).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)); model.eval()
    data = SeqData(f"{args.data}/holdout.npz", dev)
    h, e, g, d, o, seg = frames_subset(model, data, args.frames, args.seed)
    S_marks = [S for S in (100, 1_000, 10_000, 100_000, 1_000_000) if S <= args.S_max]
    out = dict(ckpt=args.ckpt, model=args.model, frames=args.frames, S_marks=S_marks, results={})
    if args.model == "m0":
        x = grid_onehot(g)
        extra = -(torch.nn.functional.cross_entropy(model.dir_head(h), d, reduction="none")
                  + torch.nn.functional.cross_entropy(model.out_head(h), o, reduction="none"))
        for tau in [float(t) for t in args.taus.split(",")]:
            def logw(fs, k, tau=tau):
                xs, hs_ = x[fs], h[fs]
                n = xs.shape[0]
                xk = xs.unsqueeze(0).expand(k, *xs.shape).reshape(-1, *xs.shape[1:])
                hk = hs_.unsqueeze(0).expand(k, *hs_.shape).reshape(-1, hs_.shape[-1])
                v, lq = lift(xk, GROUP_SIZES, tau=tau)
                return (model.flow.log_prob(v, hk) - lq).view(k, n).T
            lls, mean_lw, ess = stream_is(logw, len(h), args.S_max, S_marks, chunk=500, frame_block=32)
            r = {f"I_hat_{S}": (-(lls[S] + extra)).mean().item() for S in S_marks}
            r["ess_frac_mean"] = ess.mean().item(); r["ess_frac_median"] = ess.median().item()
            if tau == 1.0:
                r["F_exact_mean"] = (-(mean_lw + extra)).mean().item()     # 학습 목적함수 (샘플 S_max개 평균)
            out["results"][f"tau{tau:g}"] = r
            print(f"τ={tau:g}", json.dumps(r), flush=True)
    else:
        def logw(fs, k):
            lw, _ = model.loglik_is(h[fs], e[fs], g[fs], d[fs], o[fs], k, chunk=k)
            return lw
        lls, mean_lw, ess = stream_is(logw, len(h), args.S_max, S_marks, chunk=500, frame_block=32)
        r = {f"I_hat_{S}": (-lls[S]).mean().item() for S in S_marks}
        r["ess_frac_mean"] = ess.mean().item()
        with torch.no_grad():
            r["F_mean"] = model.step_terms(h, e, g, d, o)["F"].mean().item()
        out["results"]["q"] = r
        print(json.dumps(r), flush=True)
    out["time_s"] = time.time() - t0
    path = Path(args.out) if args.out else Path(args.ckpt).with_suffix(".probe.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"저장: {path} ({out['time_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
