"""학습된 인코더 키 대 난수 키 대 손 편향 (docs/world/README.md 8절).
/venv/jax/bin/python -I scripts/world/gate_learned.py docs/world/results/gate_20261115/learned.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rltdp.world import gate_learned as GL

out, rows = sys.argv[1], []
for kind in ("random", "surp", "learned"):
    for seed in (0, 1, 2):
        p, enc, kf = GL.train(seed, kind, log=lambda m: print(m, flush=True))
        for D in (32, 256, 2048):
            S, norms = GL.evaluate(p, enc, kf, kind, seed, 4, D)
            rows.append(dict(keys=kind, seed=seed, d=4, D=D, S=round(S, 3), key_norms={k: (round(v, 3) if v is not None else None) for k, v in norms.items()}))
            print(rows[-1], flush=True)
json.dump(rows, open(out, "w"), indent=1)
