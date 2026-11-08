"""실험 1 학습과 측정. 모델 하나 × 시드 하나를 돌려 runs/exp1/<name>_s<seed>/에 결과를 저장한다.

예: python -m fepoc.exp1.run --model m1a --k 5 --seed 0
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import bandit
from .models import CondVAE, RealNVP

EVAL_N = 4000          # 선택지마다 새 샘플 수
EVAL_K_TRUE = 2000     # M1의 ℑ̂ 추정용 IWAE k


def build(args):
    if args.model == "m0":
        return RealNVP(n_layers=args.layers, hidden=args.hidden)
    fixed = 0.1 if args.model == "m1b" else None
    return CondVAE(hidden=args.hidden, fixed_dec_std=fixed)


def model_F(model, o, c, k):
    return model.F(o, c) if isinstance(model, RealNVP) else model.F(o, c, k)


@torch.no_grad()
def evaluate(model, args, device, gen):
    model.eval()
    res = {}
    for a, name in enumerate(bandit.CHOICES):
        o = bandit.sample(a, EVAL_N, gen, device)
        c = torch.nn.functional.one_hot(torch.full((EVAL_N,), a, device=device), 2).float()
        F = model_F(model, o, c, args.k)
        if isinstance(model, RealNVP):
            surprise = F
        else:  # ℑ̂ = −IWAE_2000, 메모리 때문에 나눠서 계산
            parts = [model.F(o[i:i + 250], c[i:i + 250], EVAL_K_TRUE) for i in range(0, EVAL_N, 250)]
            surprise = torch.cat(parts)
        F_bar, I_bar = F.mean().item(), surprise.mean().item()
        H = bandit.ENTROPY[name]
        res[name] = dict(F=F_bar, F_se=F.std().item() / math.sqrt(EVAL_N), I=I_bar,
                         A=F_bar - I_bar, B=I_bar - H, H=H)
    Fs = np.array([res[n]["F"] for n in bandit.CHOICES])
    p = np.exp(-Fs - np.logaddexp.reduce(-Fs))
    res["P_ring"] = float(p[0])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["m0", "m1a", "m1b"], required=True)
    ap.add_argument("--k", type=int, default=1, help="IWAE k (M1만)")
    ap.add_argument("--layers", type=int, default=12, help="M0 결합층 수")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    model = build(args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    half = args.batch // 2
    c = torch.nn.functional.one_hot(torch.arange(2, device=device).repeat_interleave(half), 2).float()
    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        o = torch.cat([bandit.sample(0, half, gen, device), bandit.sample(1, half, gen, device)])
        loss = model_F(model, o, c, args.k).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            log.append(dict(step=step, loss=loss.item()))
    train_time = time.time() - t0

    name = args.tag or (args.model + (f"_k{args.k}" if args.model != "m0" else f"_L{args.layers}")
                        + (f"_h{args.hidden}" if args.hidden != 128 else ""))
    out = Path(args.out) / f"{name}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    eval_gen = torch.Generator(device=device).manual_seed(10_000 + args.seed)
    res = evaluate(model, args, device, eval_gen)
    res.update(name=name, seed=args.seed, args=vars(args), train_time_s=train_time,
               n_params=sum(p.numel() for p in model.parameters()), train_log=log)
    with torch.no_grad():
        model.eval()
        cr = torch.nn.functional.one_hot(torch.zeros(2000, dtype=torch.long, device=device), 2).float()
        np.save(out / "ring_samples.npy", model.sample(cr).cpu().numpy())
    torch.save(model.state_dict(), out / "model.pt")
    (out / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ["name", "ring", "gauss", "P_ring", "n_params", "train_time_s"]}))


if __name__ == "__main__":
    main()
