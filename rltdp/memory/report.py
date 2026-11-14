"""R1 결과 표: runs/memory/*_eval.json을 조건별로 모은다. python -m rltdp.memory.report [--dir runs/memory]"""
import argparse
import glob
import json
import os

from .eval import BUCKET_NAMES

ORDER = ["gdn", "kdn", "gdn_read", "kdn_read", "ours_fix", "ours", "simple"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/memory")
    args = ap.parse_args()
    runs = {}
    for p in glob.glob(os.path.join(args.dir, "*_eval.json")):
        r = json.load(open(p))
        runs.setdefault(r["cond"], {})[r["model"] + ("_read" if r["read_gate"] else "")] = r
    f = lambda x: "  -  " if x is None else f"{x:.3f}"
    for cond in sorted(runs):
        rs = runs[cond]
        names = [n for n in ORDER if n in rs]
        bayes = next(iter(rs.values()))
        print(f"\n## 조건 {cond}")
        for T in ("T256", "T8192"):
            cols = ["all", "rel", "rel/1", "rel/2", "noisy"] + [f"noisy/{b}" for b in BUCKET_NAMES]
            print(f"\n{T} | " + " | ".join(cols))
            for n in names + ["bayes"]:
                src, tag = (bayes, "bayes") if n == "bayes" else (rs[n], "model")
                print(f"{n:9s} | " + " | ".join(f(src[T].get(f"{tag}/{c}")) for c in cols))
        print("\n길이 8,192의 구간별 정확도 (4등분)")
        for n in names:
            print(f"{n:9s} | " + " | ".join(f(rs[n]["T8192"].get(f"model/seg{i}")) for i in range(4)))
        for n in names:
            if rs[n].get("scalars"):
                print(n, {k: round(v, 4) for k, v in rs[n]["scalars"].items()})


if __name__ == "__main__":
    main()
