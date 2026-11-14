"""진단: 학습된 키들이 서로 얼마나 겹치는가, 그래서 키 흔적이 얼마나 길어지는가.

/venv/jax/bin/python scripts/world/key_overlap.py runs/world/synapse_params.npy
무작위로 걸으면서 (칸, 행동)의 키를 모아, 키끼리의 내적과 흔적 e의 길이를 잰다.
"""
import sys

import jax
import jax.flatten_util
import jax.numpy as jnp

from rltdp.world import env as E
from rltdp.world.agents import ModelCfg, encode, make

cfg, mc = E.EnvCfg(), ModelCfg(slow="slow" in "".join(sys.argv[1:]))
tex = E.textures(cfg)
ag = make("synapse", cfg, mc, tex)
p = ag["init"](jax.random.PRNGKey(0))
if len(sys.argv) > 1:
    p = jax.flatten_util.ravel_pytree(p)[1](jnp.load(sys.argv[1]))


def keys(o):
    k = jnp.einsum("akh,h->ak", p["Wk"], encode(p, o)) + p["bk"]
    return k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)


def episode(key):
    k0, key = jax.random.split(key)
    st = E.reset(k0, cfg)

    def step(c, kk):
        st, e = c
        a = jax.random.randint(kk, (), 0, cfg.A)
        K = keys(E.patches(st, tex, cfg))
        z = mc.gamma * mc.lam
        e = z * e + (1 - z * z) ** 0.5 * K[a]
        st2, r = E.step(st, a, cfg)
        return (st2, e), (K, K[a], e @ e, e @ K[a])

    _, out = jax.lax.scan(step, (st, jnp.zeros(mc.dk)), jax.random.split(key, 128))
    return out


K, k, ee, ek = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(1), 64))
same_state = jnp.einsum("btak,btck->btac", K, K)                    # 같은 칸, 다른 행동
off = (same_state.sum((-1, -2)) - cfg.A) / (cfg.A * (cfg.A - 1))
consec = (k[:, 1:] * k[:, :-1]).sum(-1)                             # 연달아 지나간 키
far = (k[:, 40:] * k[:, :-40]).sum(-1)                              # 40스텝 떨어진 키 (대체로 다른 칸)
kc = k - k.mean((0, 1))                                              # 모든 키에 공통인 성분을 뺀 뒤
kc = kc / jnp.linalg.norm(kc, axis=-1, keepdims=True)
far_c, consec_c = (kc[:, 40:] * kc[:, :-40]).sum(-1), (kc[:, 1:] * kc[:, :-1]).sum(-1)
print(f"{sys.argv[1] if len(sys.argv) > 1 else '학습 전'}")
print(f"  공통 성분을 뺀 뒤: 연달아 지나간 키 {float(consec_c.mean()):.3f} | 멀리 떨어진 키 {float(far_c.mean()):.3f} | 공통 성분의 크기 {float(jnp.linalg.norm(k.mean((0, 1)))):.3f}")
print(f"  키끼리의 내적: 같은 칸의 다른 행동 {float(off.mean()):.3f} | 연달아 지나간 키 {float(consec.mean()):.3f} | 멀리 떨어진 키 {float(far.mean()):.3f}")
print(f"  흔적: 길이의 제곱 e·e {float(ee[:, 20:].mean()):.2f} (키가 서로 직교하면 1.00) | 지금 키와의 내적 e·k {float(ek[:, 20:].mean()):.2f} (직교하면 {(1 - (mc.gamma * mc.lam) ** 2) ** 0.5:.2f})")
