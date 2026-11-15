"""스스로 움직이는 에이전트 + 목표 전환: GRU 대 GRU + 시냅스 (docs/world/README.md 10절).
/venv/jax/bin/python -I scripts/world/goal_forage.py docs/world/results/goal_forage_20261115.json [iters]
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import jax
import jax.numpy as jnp
import optax
from rltdp.world import goal_forage as GF

out = sys.argv[1]
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
B, L, seed = 64, 32, 0
rows = []
for kind in ("gru", "synapse"):
    ag = GF.make(kind, seed)
    params = ag["init"](jax.random.PRNGKey(seed))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    st = opt.init(params)

    def rollout(p, carry, key, n):
        def one(c, k):
            return jax.lax.scan(lambda cc, kk: ag["step"](p, cc, kk), c, jax.random.split(k, n))
        return jax.vmap(one)(carry, jax.random.split(key, carry["g"].shape[0]))

    @jax.jit
    def update(p, st, carry, key):
        def f(p):
            c2, outs = rollout(p, carry, key, L)
            return ag["loss"](p, outs, c2), (c2, outs)
        (l, (c2, outs)), g = jax.value_and_grad(f, has_aux=True)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, c2, l, outs["r"].sum(1).mean()

    @jax.jit
    def evaluate(p, key):
        k1, k2 = jax.random.split(key)
        carry = jax.vmap(ag["carry0"])(jax.random.split(k1, 1024))
        _, outs = rollout(p, carry, k2, GF.T)
        end, length, seen = jax.vmap(GF.segments)(outs["r"], outs["seen"])
        m_seen, m_new = end & seen, end & ~seen
        return dict(rewards=outs["r"].sum(1).mean(),
                    steps_seen=(length * m_seen).sum() / jnp.maximum(m_seen.sum(), 1), n_seen=m_seen.sum(),
                    steps_new=(length * m_new).sum() / jnp.maximum(m_new.sum(), 1), n_new=m_new.sum())

    reset = jax.jit(lambda k: jax.vmap(ag["carry0"])(jax.random.split(k, B)))
    key, t0 = jax.random.PRNGKey(100 + seed), time.time()
    for it in range(1, iters + 1):
        key, kr = jax.random.split(key)
        carry = reset(kr)
        for _ in range(GF.T // L):
            key, ku = jax.random.split(key)
            params, st, carry, l, rw = update(params, st, carry, ku)
        if it % 250 == 0 or it == iters:
            ev = {k: round(float(v), 3) for k, v in evaluate(params, jax.random.PRNGKey(77)).items()}
            rows.append(dict(agent=kind, it=it, sec=round(time.time() - t0), **ev))
            print(rows[-1], flush=True)
    json.dump(rows, open(out, "w"), indent=1)
