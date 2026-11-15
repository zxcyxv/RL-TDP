# 초기 키를 c배 했을 때 재사용 F (도착 칸만 / 전 칸)가 어떻게 변하는가: 기억을 켜는 것이 이 F에서 이득인가
import sys; sys.path.insert(0, '/workspace/RL-TDP')
import jax, jax.numpy as jnp, numpy as np
from rltdp.world import errand_learned as EL, errand_reuse as RU, errand_prec as EP
We, ybar = EL.emb_fixed(0)
p, vmax = EP.train("learned", seed=0, D=256, iters_k=0, log=lambda m: None)
sessions, tr = EL.stack_sessions(range(50_000, 50_032), 256)
plans, _ = EL.plans_for(sessions)
slot = jnp.zeros(EP.NP).at[EP.SLOT].set(1.0)
for c in [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]:
    pp = dict(p, Kd=p["Kd"] * c)
    f = jax.jit(jax.vmap(lambda s, pl, m: RU.reuse_nll(pp, We, ybar, "learned", vmax, s, pl, m), in_axes=(0, 0, None)))
    print(c, "slot", float(f(tr, plans, slot).mean()), "all", float(f(tr, plans, jnp.ones(EP.NP)).mean()), flush=True)
