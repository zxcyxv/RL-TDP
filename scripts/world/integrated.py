"""하나로 합친 에이전트 (docs/world/README.md 12절).
/venv/jax/bin/python -I scripts/world/integrated.py docs/world/results/integrated_20261115.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import jax
import jax.numpy as jnp
from rltdp.world import credit as C, integrated as I

out = sys.argv[1]
N, T, NE = 9, 512, 512
win = lambda r: [round(float(r[:, a:a + 64].sum(1).mean()), 2) for a in range(0, T, 64)]
conds = dict(new=jnp.full(T, 2), switch=jnp.concatenate([jnp.full(T // 2, 2), jnp.full(T // 2, 1)]))
p = I.train(0, log=lambda m: print(m, flush=True))
keys = I.make_keys(p, 0)
res = {}
for kind in ("value", "fact"):
    for cname, rt in conds.items():
        r = jax.jit(jax.vmap(lambda k: C.synapse_episode(kind, k, N, T, rt, keys=keys)))(jax.random.split(jax.random.PRNGKey(5), NE))
        res[f"{kind}_learnedkeys/{cname}"] = win(r)
        print(kind, "learned keys", cname, res[f"{kind}_learnedkeys/{cname}"], flush=True)
json.dump(res, open(out, "w"), indent=1)
