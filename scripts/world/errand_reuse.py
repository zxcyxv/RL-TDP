"""지연 채집 과제: 재사용 F로 키 학습 (docs/world/README.md 12.5절).
/venv/jax/bin/python -I scripts/world/errand_reuse.py docs/world/results/errand_reuse_20261115.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import jax
from rltdp.world import errand_learned as EL, errand_reuse as RU

out, NW, D = sys.argv[1], 256, 256
sessions, stack = EL.stack_sessions(range(1000, 1000 + NW), D)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])
p, vmax = RU.train(log=lambda m: print(m, flush=True))
sc = np.array(jax.jit(jax.vmap(RU.probe_two(p, vmax, "learned")))(stack, plans, arrive))    # [NW, 2 (C 전, C 후), NSITE]
rng = np.random.default_rng(0)
res = {}
for j, when in enumerate(("before_C", "after_C")):
    s = sc[:, j]
    best = (s + 1e-9 * rng.random(s.shape)).argmax(1)
    res[when] = dict(b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)), other=float(np.mean((best != b1) & (best != b0))))
i = np.arange(NW)
d = sc[:, 1] - sc[:, 0]                                                       # C 쓰기로 바뀐 B 점수
others = np.array([np.delete(d[k], [b0[k], b1[k]]).mean() for k in range(NW)])
res["delta_B_score_by_C"] = dict(b0=float(d[i, b0].mean()), b1=float(d[i, b1].mean()), others=float(others.mean()))
print(res, flush=True)
json.dump(res, open(out, "w"), indent=1)
