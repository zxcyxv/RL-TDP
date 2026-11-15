"""초기화 없는 흐름에서 망각 비교 (docs/world/README.md 7절). 결과를 JSON으로 저장한다.
/venv/jax/bin/python -I scripts/world/gate_stream.py docs/world/results/gate_20261115/stream.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from rltdp.world import gate_stream as GS

out, rows = sys.argv[1], []
win = [(0, 1), (1, 5), (5, 20), (20, 50), (50, 100), (100, 200)]
for surp in (False, True):
    for forget in ("none", "const", "evidence"):
        for seed in (0, 1, 2):
            acc = np.array(GS.run_stream(forget, seed, surp=surp))
            S = [round(float(2 * acc[a:b].mean() - 1), 3) for a, b in win]
            rows.append(dict(memory="ours_surp" if surp else "ours", forget=forget, seed=seed, windows=win, S=S, acc=acc.round(4).tolist()))
            print(rows[-1]["memory"], forget, seed, S, flush=True)
json.dump(rows, open(out, "w"))
