"""첫 보상 전 구간에서 TDP 정책 엔트로피를 스텝 구간별로 잰다 (균등 = ln 4 = 1.386). 느린 가중치가 기억이 빈 상태의 행동을 몰아가는지 보는 진단.
/venv/jax/bin/python -I scripts/world/diag_prefound_policy.py init runs/world/<실행>/tdp_params.npy ...
"""
import sys, jax, jax.numpy as jnp, jax.flatten_util
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", ".."))
from rltdp.world import env as E
from rltdp.world.agents import ModelCfg
from rltdp.world.tdp_agent import make_tdp
cfg = E.EnvCfg(); tex = E.textures(cfg); ag = make_tdp(cfg, ModelCfg(), tex)
p0 = ag["init"](jax.random.split(jax.random.PRNGKey(0))[1])
unravel = jax.flatten_util.ravel_pytree(p0)[1]
T, temp = cfg.T, 0.05
@jax.jit
def run(p):
    def ep(key):
        k1, k2 = jax.random.split(key)
        c = ag["carry0"](k1)
        def body(carry, k):
            c, found, seen = carry
            v = jax.vmap(lambda a: ag["forward"](p, c, a)[2])(jnp.arange(cfg.A))
            lp = jax.nn.log_softmax(v / temp)
            H = -(jnp.exp(lp) * lp).sum()
            c2, out = ag["step"](p, c, k)
            seen = seen.at[c2["st"].pos[0], c2["st"].pos[1]].set(True)
            return (c2, jnp.maximum(found, out["r"]), seen), (H, found)
        seen0 = jnp.zeros((cfg.N, cfg.N), bool).at[c["st"].pos[0], c["st"].pos[1]].set(True)
        (_, _, _), (H, found) = jax.lax.scan(body, (c, 0.0, seen0), jax.random.split(k2, T))
        return H, found
    H, found = jax.vmap(ep)(jax.random.split(jax.random.PRNGKey(10_000), 512))
    pre = found == 0
    bins = [(0, 1), (1, 8), (8, 32), (32, 128), (128, 256)]
    return [((H[:, a:b] * pre[:, a:b]).sum() / jnp.maximum(pre[:, a:b].sum(), 1), pre[:, a:b].mean()) for a, b in bins]
for f in sys.argv[1:]:
    p = p0 if f == "init" else unravel(jnp.load(f))
    res = run(p)
    print(f.split("/")[-1].ljust(24), "  ".join(f"t{a}-{b}: H={float(h):.3f} (pre {float(fr):.2f})" for (a, b), (h, fr) in zip([(0,1),(1,8),(8,32),(32,128),(128,256)], res)))
