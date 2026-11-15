"""통합 시험 축소 1판: 핵심 선택 (docs/world/README.md 12절).
/venv/jax/bin/python -I scripts/world/errand.py docs/world/results/errand_20261115.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import jax
import jax.numpy as jnp
from rltdp.world import errand as ER

out, NW, D = sys.argv[1], 256, 256
sessions = [ER.make_session(1000 + i, D) for i in range(NW)]
L = max(len(s["acts"]) for s in sessions)
for s in sessions:                                     # 교사 구간 끝을 보급소 대기로 채워 길이를 맞춘다
    pad = L - len(s["acts"])
    s["acts"] = np.concatenate([s["acts"], np.full(pad, ER.WAIT)])
    s["req"] = np.concatenate([s["req"], np.full(pad, s["req"][-1])])
    s["rid"] = np.concatenate([s["rid"], np.full(pad, s["rid"][-1])])
stack = {k: jnp.array(np.stack([s[k] for s in sessions])) for k in ("tiles", "depot", "sites", "prod", "b0", "b1", "acts", "req", "rid", "change_t")}
print("teacher length", L, flush=True)
res = {}
for kind in ("random", "value", "fact"):
    f, r = jax.jit(jax.vmap(lambda sess, k: ER.run_session(kind, sess, k)))(stack, jax.random.split(jax.random.PRNGKey(0), NW))
    f, r = np.array(f), np.array(r)
    b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])
    res[kind] = dict(b1=float(np.mean(f == b1)), b0_stale=float(np.mean(f == b0)), other=float(np.mean((f >= 0) & (f != b1) & (f != b0))),
                     none=float(np.mean(f < 0)), teacher_reward=float(r.mean()))
    print(kind, res[kind], flush=True)
json.dump(res, open(out, "w"), indent=1)
