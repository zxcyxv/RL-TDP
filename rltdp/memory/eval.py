"""R1 평가: 흐름을 굴리며 손실과 정확도를 모은다. 키 종류·변화 뒤 관측 횟수·흐름 구간별로 나눈다.

python -m rltdp.memory.eval --ckpt runs/memory/C_ours.pt [--T 256 8192] [--device cuda]
"""
import argparse
import json

import torch
import torch.nn.functional as F

from .models import build
from .stream import PRESETS, BayesFilter, Stream, StreamCfg

BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 9)]      # 마지막 변화 뒤 관측 횟수
BUCKET_NAMES = ["1", "2", "3-4", "5-8", "9+"]
SEGMENTS = 4


def rollout(model, stream, T, q_every=1, bayes=False, stats=False):
    """반환: 평균 손실 (model이 있을 때), 통계 dict (stats=True)."""
    dev, K = stream.device, stream.cfg.K
    st = model.init(stream.B, dev) if model is not None else None
    bf = BayesFilter(stream) if bayes else None
    loss, n_loss = 0.0, 0
    acc = {}

    def add(name, hit, mask):
        h, n = acc.get(name, (0.0, 0.0))
        acc[name] = (h + (hit & mask).sum().item(), n + mask.sum().item())

    for t in range(T):
        j, kid, obs = stream.step()
        if model is not None:
            st = model.write(st, kid, obs)
        if bf is not None:
            bf.write(j, obs)
        if (t + 1) % q_every:
            continue
        mask = stream.since > 0
        if model is not None:
            logits = model.read(st, stream.kids)
            ce = F.cross_entropy(logits.flatten(0, 1), stream.val.flatten(), reduction="none").view_as(mask)
            loss, n_loss = loss + (ce * mask).sum() / mask.sum().clamp(min=1), n_loss + 1
        if not stats:
            continue
        for tag, pred in (("model", logits.argmax(-1) if model is not None else None),
                          ("bayes", bf.predict() if bf is not None else None)):
            if pred is None:
                continue
            hit = pred == stream.val
            add(f"{tag}/all", hit, mask)
            add(f"{tag}/seg{t * SEGMENTS // T}", hit, mask)
            for kind, km in (("rel", ~stream.noisy), ("noisy", stream.noisy)):
                add(f"{tag}/{kind}", hit, mask & km)
                for (lo, hi), bn in zip(BUCKETS, BUCKET_NAMES):
                    add(f"{tag}/{kind}/{bn}", hit, mask & km & (stream.since >= lo) & (stream.since <= hi))
    out = {k: h / n for k, (h, n) in acc.items() if n > 0}
    return (loss / max(n_loss, 1) if model is not None else None), out


@torch.no_grad()
def evaluate(model, cfg, T, B=256, device="cpu", seed=10_000, bayes=True):
    if model is not None:
        model.eval()
    _, out = rollout(model, Stream(cfg, B, device, seed), T, bayes=bayes, stats=True)
    if model is not None:
        model.train()
    return out


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    cfg = StreamCfg(**ck["cfg"])
    model = build(ck["model"], cfg, read_gate=ck["read_gate"]).to(device)
    model.load_state_dict(ck["state"])
    return model, cfg, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cond", default=None, help="다른 조건의 흐름으로 평가 (기본: 학습 조건)")
    ap.add_argument("--T", type=int, nargs="+", default=[256, 8192])
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model, cfg, ck = load(args.ckpt, args.device)
    if args.cond:
        cfg = StreamCfg(**{**ck["cfg"], **PRESETS[args.cond]})
    res = dict(model=ck["model"], read_gate=ck["read_gate"], cond=args.cond or ck["cond"],
               scalars=model.scalars() if hasattr(model, "scalars") else None)
    for T in args.T:
        res[f"T{T}"] = evaluate(model, cfg, T, B=args.B if T <= 1024 else 64, device=args.device)
        r = res[f"T{T}"]
        print(f"T={T}: model {r['model/all']:.3f} (rel {r['model/rel']:.3f}, noisy {r.get('model/noisy', float('nan')):.3f})"
              f" | bayes {r['bayes/all']:.3f}")
    out = args.out or args.ckpt.replace(".pt", f"_eval{'_' + args.cond if args.cond else ''}.json")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
