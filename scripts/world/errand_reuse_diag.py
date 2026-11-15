"""진단: 키 손실이 지도 예측과 경쟁해 상호작용 키가 작아지는가 (README 12.6).
argv: out.json  mask(all|slot)
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import jax, jax.numpy as jnp
from rltdp.world import errand_learned as EL, errand_reuse as RU, errand_prec as EP

out, which, NW, D = sys.argv[1], sys.argv[2], 256, 256
mask = None if which == "all" else jnp.zeros(EP.NP).at[EP.SLOT].set(1.0)
sessions, stack = EL.stack_sessions(range(1000, 1000 + NW), D)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])
p, vmax = RU.train(log=lambda m: print(m, flush=True), mask=mask)
s_, c_ = jax.jit(jax.vmap(lambda s: RU.key_norms(p, s)))(stack)
kn = np.array(s_.sum(0) / c_.sum(0))
sc = np.array(jax.jit(jax.vmap(RU.probe_two(p, vmax, "learned")))(stack, plans, arrive))
rng = np.random.default_rng(0)
res = dict(mask=which, key_norm={n: float(v) for n, v in zip(["E", "W", "S", "N", "interact", "wait"], kn)})
for j, when in enumerate(("before_C", "after_C")):
    s = sc[:, j]
    best = (s + 1e-9 * rng.random(s.shape)).argmax(1)
    res[when] = dict(b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)), other=float(np.mean((best != b1) & (best != b0))))
i = np.arange(NW)
d = sc[:, 1] - sc[:, 0]
others = np.array([np.delete(d[k], [b0[k], b1[k]]).mean() for k in range(NW)])
res["delta_B_score_by_C"] = dict(b0=float(d[i, b0].mean()), b1=float(d[i, b1].mean()), others=float(others.mean()))
print(res, flush=True)
json.dump(res, open(out, "w"), indent=1)
