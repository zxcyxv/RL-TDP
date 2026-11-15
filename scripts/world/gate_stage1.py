"""갈림길 과제 1단계: 학습 없는 기억들의 S_mem (docs/world/README.md 6절). 결과를 JSON으로 저장한다.
/venv/jax/bin/python -I scripts/world/gate_stage1.py docs/world/results/gate_20261115/stage1.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rltdp.world import gate_memory as GM

out = sys.argv[1]
rows = []
for name in GM.MEMORIES:
    for d in (4, 8):
        for D in (32, 256, 2048):
            for gap in (0, 32):
                S = [GM.evaluate(name, s, d, D, gap=gap)[1] for s in (0, 1, 2)]
                rows.append(dict(memory=name, d=d, D=D, gap=gap, S=[round(x, 3) for x in S]))
                print(rows[-1], flush=True)
for d in (4, 8):                                                     # 기억 교환 개입 (ours)
    S = [GM.evaluate("ours", s, d, 256, swap=True)[1] for s in (0, 1, 2)]
    rows.append(dict(memory="ours_swapM", d=d, D=256, gap=0, S=[round(x, 3) for x in S]))
    print(rows[-1], flush=True)
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(rows, open(out, "w"), indent=1)
