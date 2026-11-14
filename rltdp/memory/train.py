"""R1 학습: 흐름을 처음부터 끝까지 펼쳐 역전파한다 (모든 모델 같은 흐름 분포, 같은 설정).

python -m rltdp.memory.train --model ours --cond C [--read_gate] [--iters 3000] [--device cuda]
"""
import argparse
import json
import math
import os
import time

import torch

from .eval import evaluate, rollout
from .models import build
from .stream import PRESETS, Stream, StreamCfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gdn", "kdn", "ours", "ours_fix", "simple"], required=True)
    ap.add_argument("--cond", choices=list(PRESETS), required=True)
    ap.add_argument("--read_gate", action="store_true")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--q_every", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/memory")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = StreamCfg(**PRESETS[args.cond])
    model = build(args.model, cfg, read_gate=args.read_gate).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * i / args.iters)))
    name = f"{args.cond}_{args.model}{'_read' if args.read_gate else ''}"
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, name + ".jsonl"), "w")
    t0 = time.time()
    for it in range(1, args.iters + 1):
        loss, _ = rollout(model, Stream(cfg, args.B, args.device, seed=args.seed * 1_000_003 + it), args.T, q_every=args.q_every)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % args.eval_every == 0 or it == args.iters:
            r = evaluate(model, cfg, args.T, device=args.device)
            rec = dict(it=it, loss=loss.item(), acc=r["model/all"], rel=r["model/rel"], noisy=r.get("model/noisy"),
                       bayes=r["bayes/all"], sec=round(time.time() - t0),
                       scalars=model.scalars() if hasattr(model, "scalars") else None)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(name, rec, flush=True)
    torch.save(dict(model=args.model, read_gate=args.read_gate, cond=args.cond, cfg=cfg.__dict__, state=model.state_dict()),
               os.path.join(args.out, name + ".pt"))


if __name__ == "__main__":
    main()
