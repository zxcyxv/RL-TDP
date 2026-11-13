"""실험 2a 보고서: 그림 2(월드 모델 지표 곡선), 그림 3(상상 프레임), 구간별 간극 표, 비용 표.

python -m rltdp.fep.wm.report [--runs runs/exp2a] [--seed 0]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rltdp.common.minigrid_env import render_obs
from rltdp.fep.wm.common import grid_onehot
from rltdp.fep.wm.data import SeqData
from rltdp.fep.wm.models import build

COLORS = {"M0": "#4f81bd", "M1": "#c0504d"}
SEG_KO = {"room": "방", "corridor": "복도", "junction": "갈림길"}


def setup_font():
    for fam in ["NanumGothic", "Noto Sans CJK KR"]:
        if any(fam == f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams["axes.unicode_minus"] = False


def fig2(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, (evals, _) in runs.items():
        u = [e["update"] for e in evals]
        c = COLORS[name]
        axes[0].plot(u, [e["I_hat_1000"] for e in evals], "o-", color=c, label=f"{name} ℑ̂ (S=1000)")
        axes[0].plot(u, [e["I_hat_100"] for e in evals], "--", color=c, alpha=0.6, label=f"{name} ℑ̂ (S=100)")
        axes[0].plot(u, [e["F"] for e in evals], ":", color=c, alpha=0.6, label=f"{name} F")
        for k, ls in [(1, "-"), (5, "--"), (10, ":")]:
            axes[1].plot(u, [e["acc"][f"k{k}_sample"] for e in evals], ls, color=c, label=f"{name} k={k}")
            axes[2].plot(u, [e["acc"][f"k{k}_mode"] for e in evals], ls, color=c, label=f"{name} k={k}")
    axes[0].set_ylabel("nats / 스텝"); axes[0].set_yscale("log"); axes[0].set_title("음의 이산 로그우도와 F (보류)")
    axes[1].set_title("k스텝 예측 정확도 (샘플)"); axes[2].set_title("k스텝 예측 정확도 (최빈값)")
    for a in axes:
        a.set_xlabel("갱신 횟수"); a.legend(fontsize=7)
    fig.suptitle("그림 2  월드 모델 지표 (보류 집합, 시드 0)")
    fig.tight_layout(); fig.savefig(out / "fig2_world_model_metrics.png", dpi=130); plt.close(fig)


@torch.no_grad()
def imagine_episode(model, data, ep, start, n_real=5, n_total=20):
    """start부터 h = 0으로 시작해 처음 n_real 스텝은 실제 관측, 이후는 실제 행동만 넣어 굴린다."""
    s0 = int(data.ep_start[ep]) + start
    n = min(n_total, int(data.ep_len[ep]) - start)
    idx = torch.arange(s0, s0 + n, device=data.device)
    grid, d, act = data.grid[idx].long(), data.dir[idx], data.action[idx].clamp_min(0)
    h = model.init_state(1)
    preds = []
    for t in range(n):
        g_hat, _, _, _ = model.predict(h, sample=True)
        preds.append(g_hat[0].cpu().numpy())
        if t < n_real:
            e = model.sk.embed(grid_onehot(grid[t:t + 1]), d[t:t + 1])
        else:
            e = model.sk.embed(grid_onehot(g_hat), d[t:t + 1])
        h = model.sk.step(h, e, act[t:t + 1])
    return grid.cpu().numpy(), np.stack(preds), data.segment[idx].cpu().numpy()


def fig3(models, data, out, n_real=5):
    # 스크립트 에피소드 하나(처음부터)와 무작위 에피소드 하나(갈림길이 처음 보이기 10스텝 전부터)
    picks = []
    scripted = (data.policy == 1).nonzero().flatten().tolist()
    picks.append(("스크립트", scripted[0], 0))
    for ep in (data.policy == 0).nonzero().flatten().tolist():
        seg = data.segment[data.ep_start[ep]:data.ep_start[ep] + data.ep_len[ep]]
        j = (seg == 2).nonzero()
        if len(j) and int(j[0]) >= 10:
            picks.append(("무작위", ep, int(j[0]) - 10)); break
    for label, ep, start in picks:
        rows = {}
        for name, m in models.items():
            real, pred, seg = imagine_episode(m, data, ep, start, n_real)
            rows["실제"], rows[name] = real, pred
        T = len(rows["실제"])
        names = ["실제"] + list(models)
        fig, axes = plt.subplots(len(names), T, figsize=(T * 1.1, len(names) * 1.25))
        for r, nm in enumerate(names):
            for t in range(T):
                ax = axes[r, t]
                ax.imshow(render_obs(rows[nm][t].astype(np.uint8), tile_size=12))
                ax.set_xticks([]); ax.set_yticks([])
                if t == 0:
                    ax.set_ylabel(nm, fontsize=9)
                if r == 0:
                    ax.set_title(f"t={start + t}" + ("\n(실제 입력)" if t < n_real else "") +
                                 ("\n갈림길" if seg[t] == 2 else ""), fontsize=7)
        fig.suptitle(f"그림 3  상상 프레임 ({label} 에피소드 #{ep}, 처음 {n_real}스텝만 실제 관측)")
        fig.tight_layout()
        fig.savefig(out / f"fig3_imagination_{'scripted' if label == '스크립트' else 'random'}.png", dpi=120)
        plt.close(fig)


def tables(runs, out):
    lines = ["## 구간별 간극 (마지막 평가)", "",
             "M0는 Â_deq(들어 올림의 국소 간극), M1은 Â(숨은 상태 추론 간극). 둘 다 F − ℑ̂_1000, nats/스텝.", "",
             "| 모델 | 전체 | 방 | 복도 | 갈림길 | 유효 샘플 비율 (평균 / 중앙값) | ℑ̂ S=100 → 1000 |",
             "|---|---|---|---|---|---|---|"]
    for name, (evals, _) in runs.items():
        e = evals[-1]
        seg = e["A_hat_by_segment"]
        lines.append(f"| {name} | {e['A_hat']:.3f} | {seg['room']:.3f} | {seg['corridor']:.3f} | {seg['junction']:.3f} | "
                     f"{e['ess_frac_mean']:.4f} / {e['ess_frac_median']:.4f} | {e['I_hat_100']:.3f} → {e['I_hat_1000']:.3f} |")
    e0 = next(iter(runs.values()))[0][-1]
    lines += ["", "구간별 프레임 수: " + ", ".join(f"{SEG_KO[k]} {v}" for k, v in e0["n_by_segment"].items()), "",
              "## 최종 지표", "", "| 모델 | F | ℑ̂ (S=1000) | 1스텝 정확도 (샘플/최빈값) | 5스텝 | 10스텝 |", "|---|---|---|---|---|---|"]
    for name, (evals, _) in runs.items():
        e, a = evals[-1], evals[-1]["acc"]
        lines.append(f"| {name} | {e['F']:.3f} | {e['I_hat_1000']:.3f} | {a['k1_sample']:.3f} / {a['k1_mode']:.3f} | "
                     f"{a['k5_sample']:.3f} / {a['k5_mode']:.3f} | {a['k10_sample']:.3f} / {a['k10_mode']:.3f} |")
    lines += ["", f"무작위 예측의 칸 일치율: {e0['acc_random_baseline']:.4f}", "",
              "## 비용", "", "| 모델 | 파라미터 | 갱신당 시간 (ms) | 평가 1회 (s) | GPU 메모리 최대 (GB) |", "|---|---|---|---|---|"]
    for name, (evals, summ) in runs.items():
        lines.append(f"| {name} | {summ['n_params']:,} | {summ['ms_per_update']:.1f} | "
                     f"{np.mean([e['eval_time_s'] for e in evals]):.0f} | {summ['peak_mem_gb']:.2f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/exp2a")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/memory")
    args = ap.parse_args()
    setup_font()
    out = Path(args.runs) / "report"; out.mkdir(parents=True, exist_ok=True)
    runs, models = {}, {}
    for key, name in [("m0", "M0"), ("m1", "M1")]:
        d = Path(args.runs) / f"{key}_s{args.seed}"
        if not (d / "summary.json").exists():
            continue
        runs[name] = (json.loads((d / "evals.json").read_text()), json.loads((d / "summary.json").read_text()))
        m = build(key).cuda(); m.load_state_dict(torch.load(d / "model.pt")); m.eval()
        models[name] = m
    fig2(runs, out)
    torch.manual_seed(0)
    fig3(models, SeqData(f"{args.data}/holdout.npz", "cuda"), out)
    md = "# 실험 2a 결과 (시드 %d)\n\n" % args.seed + tables(runs, out) + \
         "\n\n![그림 2](fig2_world_model_metrics.png)\n\n![그림 3 스크립트](fig3_imagination_scripted.png)\n\n" \
         "![그림 3 무작위](fig3_imagination_random.png)\n"
    (out / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
