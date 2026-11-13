"""실험 2a: 고정 데이터로 월드 모델 학습.

python -m rltdp.fep.wm.train --model m0 --seed 0                     # 본 실행 (갱신 50,000회)
python -m rltdp.fep.wm.train --model m0 --updates 300 --smoke        # 시간 측정용
공통 조건: Adam 3e-4, 기울기 노름 클리핑 100, 배치 32, 시퀀스 32, 손실 = 배치 평균 Σ_t F_t.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from rltdp.fep.wm.data import SeqData
from rltdp.fep.wm.evaluate import evaluate
from rltdp.fep.wm.models import build


def save_checkpoint(model, opt, gen, update, out, min_free_gb):
    """평가 시점마다 저장한다.
    - ckpt/u<update>.pt: 가중치만 (약 11MB). 중간 분석용으로 모두 보관한다
    - model.pt: 최신 가중치
    - last.pt: 재개용 전체 상태 (옵티마이저·난수 포함, 약 35MB). 최신 1개만 유지한다
    디스크 여유가 min_free_gb보다 적으면 스냅숏을 건너뛰고 경고한다 (model.pt, last.pt는 덮어쓰므로 유지)."""
    state = model.state_dict()
    torch.save(state, out / "model.pt")
    tmp = out / "last.pt.tmp"
    torch.save(dict(model=state, opt=opt.state_dict(), gen=gen.get_state(), update=update), tmp)
    tmp.replace(out / "last.pt")
    free_gb = shutil.disk_usage(out).free / 1e9
    if free_gb < min_free_gb:
        print(f"경고: 디스크 여유 {free_gb:.1f}GB < {min_free_gb}GB, 스냅숏 u{update} 생략", flush=True)
        return
    (out / "ckpt").mkdir(exist_ok=True)
    torch.save(state, out / "ckpt" / f"u{update:06d}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["m0", "m0f", "m1", "m0fs", "m1s", "m0s", "m0sq", "m0sqt", "m0sqt0"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=100.0)
    ap.add_argument("--eval_every", type=int, default=5_000)
    ap.add_argument("--flow_blocks", type=int, default=4)
    ap.add_argument("--data", default="data/memory")
    ap.add_argument("--out", default="runs/exp2a")
    ap.add_argument("--min_free_gb", type=float, default=2.0, help="디스크 여유가 이보다 적으면 스냅숏을 건너뛴다")
    ap.add_argument("--smoke", action="store_true", help="학습 시간만 재고 평가 1회 시간을 잰다")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_chunks", type=int, default=0, help="평가에 쓸 보류 조각 수 (0 = 전부). CPU에서 줄여 쓴다")
    ap.add_argument("--eval_S", type=int, nargs="+", default=[100, 1000], help="중요도 가중 샘플 수")
    args = ap.parse_args()

    dev = args.device
    sync = torch.cuda.synchronize if dev.startswith("cuda") else (lambda: None)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    train, hold = SeqData(f"{args.data}/train.npz", dev), SeqData(f"{args.data}/holdout.npz", dev)
    kw = dict(n_blocks=args.flow_blocks) if args.model in ("m0", "m0s", "m0sq", "m0sqt", "m0sqt0") else {}
    model = build(args.model, **kw).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    out = Path(args.out) / f"{args.model}_s{args.seed}{'_smoke' if args.smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{model.name} 파라미터 {n_params:,}", flush=True)

    ev_kw = dict(S_list=tuple(args.eval_S), max_chunks=args.eval_chunks or None)
    log, evals = [], []
    warm, t_start = 20, None
    for u in range(1, args.updates + 1):
        if u == warm + 1:
            sync(); t_start = time.time()
        loss, F_t = model.loss(train.sample(args.batch, args.seq, gen))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if u % 100 == 0:
            rec = dict(update=u, loss=loss.item(), grad_norm=gnorm.item())
            log.append(rec)
            if not torch.isfinite(loss):
                print("손실 발산", rec, flush=True); break
        if u % 1000 == 0:
            sync()
            print(f"update {u} loss {loss.item():.2f} gnorm {gnorm.item():.1f} "
                  f"{(time.time() - t_start) / (u - warm) * 1000:.1f} ms/update", flush=True)
        if not args.smoke and (u % args.eval_every == 0 or u == args.updates):
            ev = evaluate(model, hold, **ev_kw); ev["update"] = u
            evals.append(ev)
            print(json.dumps({k: ev[k] for k in ["update", "F", *[f"I_hat_{S}" for S in args.eval_S], "A_hat",
                                                  "ess_frac_mean", "acc", "eval_time_s"]}), flush=True)
            save_checkpoint(model, opt, gen, u, out, args.min_free_gb)
            (out / "evals.json").write_text(json.dumps(evals, indent=1))
    sync()
    ms = (time.time() - t_start) / (u - warm) * 1000
    res = dict(model=model.name, n_params=n_params, ms_per_update=ms, updates=u, args=vars(args),
               peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if dev.startswith("cuda") else float("nan"))
    if args.smoke:
        ev = evaluate(model, hold, **ev_kw)
        res["eval"] = ev
        print(json.dumps({k: ev[k] for k in ["F", *[f"I_hat_{S}" for S in args.eval_S], "A_hat", "ess_frac_mean", "acc",
                                             "eval_time_s", "n_frames"]}), flush=True)
    (out / "train_log.json").write_text(json.dumps(log))
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ["model", "n_params", "ms_per_update", "peak_mem_gb"]}), flush=True)


if __name__ == "__main__":
    main()
