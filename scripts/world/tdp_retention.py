"""진단: TDP 시냅스에 쓴 것이 몇 스텝 뒤에 얼마나 남아 읽히는가 (흔적의 감쇠만으로 계산).

/venv/jax/bin/python scripts/world/tdp_retention.py [학습된 가중치.npy]
한 번 쓴 (값, 키) 쌍이 Δ스텝 뒤에 읽히는 배율 = Σ_r Re(c_r) γ_r³ |z_r|^(2Δ)  (헤드 평균). Δ=0의 값에 대한 비율도 낸다.
"""
import sys

import jax
import jax.flatten_util
import jax.numpy as jnp

from rltdp.world import env as E
from rltdp.world.tdp_agent import make_tdp

ag = make_tdp(E.EnvCfg(), None, E.textures(E.EnvCfg()))
p = ag["init"](jax.random.PRNGKey(0))
name = "학습 전"
if len(sys.argv) > 1:
    name = sys.argv[1]
    flat = jnp.load(sys.argv[1])
    try:
        p = jax.flatten_util.ravel_pytree(p)[1](flat)
    except Exception:                                     # 예전 가중치 (보상 토큰의 "아직 모름" 임베딩이 없던 때)
        p = {k: v for k, v in p.items() if k != "rmask"}
        p = jax.flatten_util.ravel_pytree(p)[1](flat)
m = p["mem"]
nu, th = m["pole"]
mag = jnp.exp(-jnp.exp(nu))                               # |z| [H, R]
g3 = jnp.sqrt(-jnp.expm1(-2 * jnp.exp(nu))) ** 3
print(name)
print("  대역별 |z| (헤드 평균):", [round(float(x), 3) for x in mag.mean(0)], "→ 반으로 줄 때까지", [round(float(jnp.log(0.5) / (2 * jnp.log(x))), 1) for x in mag.mean(0)], "스텝")
print("  대역별 Re c (헤드 평균):", [round(float(x), 2) for x in m["c"][0].mean(0)])
for d in (0, 1, 5, 10, 30, 60, 120):
    val = (m["c"][0] * g3 * mag ** (2 * d)).sum(-1).mean()
    base = (m["c"][0] * g3).sum(-1).mean()
    print(f"  {d:3d}스텝 뒤: 배율 {float(val):.4f} (쓴 직후의 {100 * float(val / base):.1f}%)")
