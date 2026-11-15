"""세션 안 신용할당으로 새 보상에 곧바로 적응하는가 (docs/world/README.md 11절).
/venv/jax/bin/python -I scripts/world/credit.py docs/world/results/credit_20261115.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import jax
import jax.numpy as jnp
import optax
from rltdp.world import credit as C

out = sys.argv[1]
N, T, NE = 9, 512, 512
res = {}
win = lambda r: [round(float(r[:, a:a + 64].sum(1).mean()), 2) for a in range(0, T, 64)]   # 64스텝 구간별 보상 (에피소드 평균)

conds = dict(new=jnp.full(T, 2), switch=jnp.concatenate([jnp.full(T // 2, 2), jnp.full(T // 2, 1)]), train=jnp.zeros(T, jnp.int32))
# 학습 없는 시냅스 에이전트
for kind in ("value", "fact"):
    for cname, rt in conds.items():
        r = jax.jit(jax.vmap(lambda k: C.synapse_episode(kind, k, N, T, rt)))(jax.random.split(jax.random.PRNGKey(5), NE))
        res[f"{kind}/{cname}"] = win(r)
        print(kind, cname, res[f"{kind}/{cname}"], flush=True)

# frozen: 학습 분포 (rtype 0, T = 256)에서 학습
p = C.frozen_init(jax.random.PRNGKey(0))
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))


def train(p, rt_train, iters, B=64, Tt=256, key=jax.random.PRNGKey(1)):
    st = opt.init(p)

    @jax.jit
    def upd(p, st, k):
        def f(p):
            outs = jax.vmap(lambda kk: C.frozen_rollout(p, kk, N, Tt, rt_train))(jax.random.split(k, B))
            return jax.vmap(lambda o: C.frozen_loss(p, o))(outs).mean(), outs["r"].sum(1).mean()
        (l, rw), g = jax.value_and_grad(f, has_aux=True)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, rw
    for it in range(iters):
        key, k = jax.random.split(key)
        p, st, rw = upd(p, st, k)
    return p, float(rw)


p, rw = train(p, jnp.zeros(256, jnp.int32), 1500)
print("frozen trained, train reward/256 steps", rw, flush=True)
evalf = jax.jit(lambda p, rt: jax.vmap(lambda k: C.frozen_rollout(p, k, N, T, rt)["r"])(jax.random.split(jax.random.PRNGKey(5), NE)))
for cname, rt in conds.items():
    res[f"frozen/{cname}"] = win(evalf(p, rt))
    print("frozen", cname, res[f"frozen/{cname}"], flush=True)

# frozen의 경사 적응: 새 보상 (rtype 2)으로 A2C를 이어서, 경사 갱신 횟수마다 보상
curve = []
pn = p
for block in range(8):
    pn, _ = train(pn, jnp.full(256, 2), 50, key=jax.random.PRNGKey(100 + block))
    r = evalf(pn, conds["new"])
    curve.append(dict(updates=(block + 1) * 50, env_steps=(block + 1) * 50 * 64 * 256, reward_per_512=round(float(r.sum(1).mean()), 2)))
    print("frozen finetune", curve[-1], flush=True)
res["frozen/finetune_new"] = curve
json.dump(res, open(out, "w"), indent=1)
