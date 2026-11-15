"""지연 채집 과제: 학습된 월드 모델·키 위의 기억 내용 판정 (docs/world/README.md 12.1절).
/venv/jax/bin/python -I scripts/world/errand_learned.py docs/world/results/errand_learned_20261115.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import jax
from rltdp.world import errand_learned as EL

out, NW, D = sys.argv[1], 256, 256
sessions, stack = EL.stack_sessions(range(1000, 1000 + NW), D)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])
res = {}
trained = {}
for name, kind, use_wm, ev in (("random_keys_noWM", "random", False, False), ("random_keys_WM", "random", True, False), ("learned_keys_WM", "learned", True, False),
                               ("random_keys_noWM+evidence", "random", False, True), ("random_keys_WM+evidence", "random", True, True), ("learned_keys_WM+evidence", "learned", True, True)):
    if (kind, use_wm) not in trained:
        trained[(kind, use_wm)] = EL.train(kind, log=lambda m: print(m, flush=True)) if use_wm else (EL.init(jax.random.PRNGKey(0)), 1.0)
    p, vmax = trained[(kind, use_wm)]
    one = EL.probe(p, vmax, kind, use_wm, stack, evidence=ev)
    sc = np.array(jax.jit(jax.vmap(one))(stack, plans, arrive))
    best = sc.argmax(1)
    res[name] = dict(b1=float(np.mean(best == b1)), b0_stale=float(np.mean(best == b0)), other=float(np.mean((best != b1) & (best != b0))),
                     score_b1=float(sc[np.arange(NW), b1].mean()), score_b0=float(sc[np.arange(NW), b0].mean()))
    print(name, res[name], flush=True)
json.dump(res, open(out, "w"), indent=1)
