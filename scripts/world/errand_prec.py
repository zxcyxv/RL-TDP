"""지연 채집 과제: 칸별 정밀도 (docs/world/README.md 12.3절).
/venv/jax/bin/python -I scripts/world/errand_prec.py docs/world/results/errand_prec_20261115.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import jax
import jax.numpy as jnp
from rltdp.world import errand_learned as EL, errand_prec as EP

out, NW, D = sys.argv[1], 256, 256
sessions, stack = EL.stack_sessions(range(1000, 1000 + NW), D)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])
res = {}
for name, kind in (("random_keys_WM", "random"), ("learned_keys_WM", "learned")):
    p, vmax = EP.train(kind, log=lambda m: print(m, flush=True))
    for ev in (False,):
        sc = np.array(jax.jit(jax.vmap(EP.probe(p, vmax, kind, evidence=ev)))(stack, plans, arrive))
        mx = sc.max(1, keepdims=True)
        tied = (np.abs(sc - mx) < 1e-6).sum(1) > 1
        rng = np.random.default_rng(0)
        best = (sc + 1e-9 * rng.random(sc.shape)).argmax(1)                        # 동점은 무작위
        res[name] = dict(b1=float(np.mean(best == b1)), b0_stale=float(np.mean(best == b0)), other=float(np.mean((best != b1) & (best != b0))), ties=int(tied.sum()))
        print(name, res[name], flush=True)
    if kind == "learned":
        We, ybar = EL.emb_fixed(0)
        _, wm, keyf = EP.parts(p, We, ybar, kind)
        s0 = jax.tree_util.tree_map(lambda x: x[0], stack)
        O, _, _ = jax.jit(EL.observe_stream)(s0)
        kn = np.array(jax.vmap(lambda o, a: jnp.linalg.norm(keyf(o, a)))(O[:-1], s0["acts"]))
        acts = np.array(s0["acts"])
        res[name]["key_norm"] = dict(interact=float(kn[acts == 4].mean()), move=float(kn[acts < 4].mean()), wait=float(kn[acts == 5].mean()))
        lv = np.array(jax.vmap(lambda o, a: wm(o, a)[1])(O[:-1], s0["acts"]))
        res[name]["wm_var_by_patch_mean"] = np.exp(lv).mean(0).round(5).tolist()
json.dump(res, open(out.replace('.json', '_fixes.json'), "w"), indent=1)
