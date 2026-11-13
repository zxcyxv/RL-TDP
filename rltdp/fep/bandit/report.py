"""runs/exp1/*/result.json을 모아 표 E1, 그림 E1-1, 그림 E1-2, 용량 비교 표를 만든다.

python -m rltdp.fep.bandit.report [--runs runs/exp1] [--out runs/exp1/report]
시드가 여러 개면 평균과 95% 신뢰구간(t분포)을, 하나면 단일 값을 쓴다.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from rltdp.fep.bandit import bandit


def load(runs):
    by_name = defaultdict(list)
    for f in sorted(Path(runs).glob("*/result.json")):
        r = json.loads(f.read_text())
        r["dir"] = f.parent
        by_name[r["name"]].append(r)
    return by_name


def agg(vals):
    vals = np.asarray(vals, float)
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    h = stats.t.ppf(0.975, len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals))
    return f"{vals.mean():.3f} ± {h:.3f}"


def table_e1(by_name, names):
    rows = ["| 모델 | 시드 수 | 선택지 | F̄ | H | B̂ | Â | P(고리) |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        rs = by_name.get(n)
        if not rs:
            continue
        for ch, label in [("ring", "고리"), ("gauss", "가우시안")]:
            g = lambda key: agg([r[ch][key] for r in rs])
            p = agg([r["P_ring"] for r in rs]) if ch == "ring" else ""
            rows.append(f"| {n} | {len(rs)} | {label} | {g('F')} | {rs[0][ch]['H']:.4f} | {g('B')} | {g('A')} | {p} |")
    return "\n".join(rows)


def table_capacity(by_name, names):
    rows = ["| 모델 | 파라미터 | 고리 F̄ | Â 고리 | P(고리) | 학습 시간(s) |", "|---|---|---|---|---|---|"]
    for n in names:
        rs = by_name.get(n)
        if not rs:
            continue
        rows.append(f"| {n} | {rs[0]['n_params']:,} | {agg([r['ring']['F'] for r in rs])} | "
                    f"{agg([r['ring']['A'] for r in rs])} | {agg([r['P_ring'] for r in rs])} | "
                    f"{np.mean([r['train_time_s'] for r in rs]):.0f} |")
    return "\n".join(rows)


def fig_scatter(by_name, out):
    panels = [("실제", None), ("M0", "m0_L12"), ("M1-a (k=1)", "m1a_k1"), ("M1-b (k=1)", "m1b_k1")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True, sharey=True)
    gen = torch.Generator().manual_seed(0)
    for ax, (title, n) in zip(axes, panels):
        if n is None:
            x = bandit.sample(0, 2000, gen, "cpu").numpy()
        elif n in by_name:
            x = np.load(by_name[n][0]["dir"] / "ring_samples.npy")
        else:
            continue
        ax.scatter(x[:, 0], x[:, 1], s=2, alpha=0.4, color="#3b6ea5")
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
    fig.suptitle("그림 E1-1  고리 선택지 샘플 (시드 0)")
    fig.tight_layout(); fig.savefig(out / "fig_e1_1_scatter.png", dpi=130); plt.close(fig)


def fig_dose_response(by_name, out):
    ks = [1, 5, 50]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for m, color, label in [("m1a", "#c0504d", "M1-a"), ("m1b", "#4f81bd", "M1-b")]:
        pts = [(k, by_name[f"{m}_k{k}"]) for k in ks if f"{m}_k{k}" in by_name]
        if not pts:
            continue
        kk = [k for k, _ in pts]
        dA = [np.mean([r["ring"]["A"] - r["gauss"]["A"] for r in rs]) for _, rs in pts]
        pr = [np.mean([r["P_ring"] for r in rs]) for _, rs in pts]
        a1.plot(kk, dA, "o-", color=color, label=label)
        a2.plot(kk, pr, "o-", color=color, label=label)
    if "m0_L12" in by_name:
        a2.axhline(np.mean([r["P_ring"] for r in by_name["m0_L12"]]), color="gray", ls="--", label="M0")
    a2.axhline(0.5, color="black", lw=0.6)
    for a, yl in [(a1, "Â(고리) - Â(가우시안)  [nats]"), (a2, "P(고리)")]:
        a.set_xscale("log"); a.set_xticks(ks, [str(k) for k in ks]); a.set_xlabel("IWAE k"); a.set_ylabel(yl)
        a.legend()
    fig.suptitle("그림 E1-2  IWAE k에 따른 간극 차이와 선택 편향")
    fig.tight_layout(); fig.savefig(out / "fig_e1_2_dose_response.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/exp1")
    ap.add_argument("--out", default="runs/exp1/report")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for fam in ["NanumGothic", "Noto Sans CJK KR", "UnDotum"]:
        if any(fam in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fam; break
    plt.rcParams["axes.unicode_minus"] = False

    by_name = load(args.runs)
    main_names = ["m0_L12"] + [f"{m}_k{k}" for m in ["m1a", "m1b"] for k in [1, 5, 50]]
    cap_names = ["m0_L2", "m0_L4", "m0_L12", "m1a_k1", "m1a_k1_h320", "m1b_k1", "m1b_k1_h320"]
    md = ["# 실험 1 결과", "", f"엔트로피: 고리 {bandit.ENTROPY['ring']:.5f}, 가우시안 {bandit.ENTROPY['gauss']:.5f} nats",
          "", "## 표 E1", "", table_e1(by_name, main_names), "", "## 용량 비교", "",
          table_capacity(by_name, cap_names), "",
          "![E1-1](fig_e1_1_scatter.png)", "", "![E1-2](fig_e1_2_dose_response.png)", ""]
    (out / "report.md").write_text("\n".join(md))
    fig_scatter(by_name, out)
    fig_dose_response(by_name, out)
    print("\n".join(md))


if __name__ == "__main__":
    main()
