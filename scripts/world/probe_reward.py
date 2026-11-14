"""진단: 보상의 기억이 시냅스에 들어 있는가, 들어 있다면 모델이 그것을 꺼내 쓰고 있는가.

/venv/jax/bin/python scripts/world/probe_reward.py runs/world/tdp_params.npy

학습된 TDP 모델을 고정하고 굴리면서, 에피소드에서 첫 보상을 받은 뒤의 스텝만 모은다.
맞힐 것 (y): "지금 고른 행동이 바로 보상으로 이어지는가". 보상 칸은 겉으로 구별되지 않으므로 기억 없이는 맞힐 수 없다.
모델은 그대로 두고 작은 분류기만 따로 학습해, 무엇을 보면 y를 맞힐 수 있는지 잰다 (AUC: 0.5 = 못 맞힘, 1.0 = 완벽).
  out      보상 토큰의 최종 출력            → 모델이 이미 꺼내 놓은 것
  qread    모델의 질의로 시냅스를 읽은 값    → 모델의 지금 질의가 집어내는 것
  best_q   새 질의를 따로 학습해 시냅스를 읽음 → 어떤 질의로든 꺼낼 수 있는가 (시냅스에 정보가 있는가)
  shuffled best_q와 같되 다른 에피소드의 시냅스를 줌 → 대조 (0.5여야 한다)
"""
import sys

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax

from rltdp.world import env as E
from rltdp.world.tdp_agent import make_tdp

cfg = E.EnvCfg()
tex = E.textures(cfg)
ag = make_tdp(cfg, None, tex)
p = ag["init"](jax.random.PRNGKey(0))
p = jax.flatten_util.ravel_pytree(p)[1](jnp.load(sys.argv[1]))
B, TEMP = 256, 0.05


def episode(key):
    k0, key = jax.random.split(key)
    c = ag["carry0"](k0)

    def step(carry, kk):
        c, seen = carry
        o_hat, r_hat, v_hat = jax.vmap(lambda a: ag["forward"](p, c, a)[:3])(jnp.arange(cfg.A))
        a = jax.random.categorical(kk, v_hat / TEMP)
        f = ag["features"](p, c, a)
        st2, r = E.step(c["st"], a, cfg)
        _, _, _, ek, ev, _, ms = ag["forward"](p, c, a, r)
        rpe = r - r_hat[a]
        W = c["W"] + jnp.exp(p["log_kappa"]) * jnp.maximum(rpe, 0.0) * ms[0] - jnp.abs(rpe) * ms[1]
        new = dict(st=st2, o=E.patches(st2, tex, cfg), r=r, ek=ek, ev=ev, W=W)
        return (new, jnp.maximum(seen, r)), dict(f, y=r, seen=seen, r_hat=r_hat[a], v_hat=v_hat[a])

    _, out = jax.lax.scan(step, (c, jnp.zeros(())), jax.random.split(key, cfg.T))
    return out


roll = jax.jit(jax.vmap(episode))
data = {k: [] for k in ("M", "u", "qread", "out", "y", "ep", "r_hat", "v_hat")}
rng = np.random.default_rng(0)
for batch in range(8):
    o = jax.tree_util.tree_map(np.asarray, roll(jax.random.split(jax.random.PRNGKey(100 + batch), B)))
    ok = o["seen"] > 0                                               # 첫 보상을 받은 뒤의 스텝만
    pos = np.argwhere(ok & (o["y"] > 0))
    neg = np.argwhere(ok & (o["y"] == 0))
    neg = neg[rng.choice(len(neg), size=min(len(neg), 3 * len(pos)), replace=False)]
    idx = np.concatenate([pos, neg])
    for k in ("M", "u", "qread", "out", "y", "r_hat", "v_hat"):
        data[k].append(o[k][idx[:, 0], idx[:, 1]])
    data["ep"].append(idx[:, 0] + batch * B)
data = {k: np.concatenate(v) for k, v in data.items()}
n, y = len(data["y"]), data["y"]
test = data["ep"] % 4 == 0                                           # 에피소드 단위로 나눈다
print(f"표본 {n}개 (보상으로 이어짐 {int(y.sum())}개), 학습 {int((~test).sum())} / 평가 {int(test.sum())}")


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty(len(score)); rank[order] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = len(label) - n1
    return float((rank[label > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit(logit_fn, params, feats, steps=3000, lr=3e-3):
    opt = optax.adam(lr)
    st = opt.init(params)
    tr = {k: jnp.asarray(v[~test]) for k, v in feats.items()}
    yt = jnp.asarray(y[~test])

    @jax.jit
    def upd(params, st, key):
        i = jax.random.randint(key, (512,), 0, yt.shape[0])
        def loss(pp):
            z = logit_fn(pp, {k: v[i] for k, v in tr.items()})
            return optax.sigmoid_binary_cross_entropy(z, yt[i]).mean() + 1e-4 * sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(pp))
        g = jax.grad(loss)(params)
        u, st = opt.update(g, st)
        return optax.apply_updates(params, u), st

    key = jax.random.PRNGKey(0)
    for _ in range(steps):
        key, k = jax.random.split(key)
        params, st = upd(params, st, k)
    z = np.asarray(logit_fn(params, {k: jnp.asarray(v[test]) for k, v in feats.items()}))
    return auc(z, y[test])


lin = lambda pp, f: f["x"] @ pp["w"] + pp["b"]
norm = lambda x: (x - x[~test].mean(0)) / (x[~test].std(0) + 1e-6)
print(f"모델 자신의 다음 보상 예측 (r_hat):   AUC {auc(data['r_hat'][test], y[test]):.3f}")
print(f"모델 자신의 앞으로의 보상 예측 (v_hat): AUC {auc(data['v_hat'][test], y[test]):.3f}")
for name in ("out", "qread"):
    x = norm(data[name])
    print(f"{name:8s} 선형 분류기:               AUC {fit(lin, dict(w=jnp.zeros(x.shape[1]), b=jnp.zeros(())), dict(x=x)):.3f}")

Hh, D = data["M"].shape[1], data["M"].shape[2]
u_n, M_n = norm(data["u"]), data["M"] / (np.abs(data["M"][~test]).mean() + 1e-8)


def bil(pp, f):                                                      # 새 질의 q = A u 로 시냅스를 읽고 (M q), 선형으로 본다
    q = jnp.einsum("hed,nd->nhe", pp["A"], f["u"])
    return jnp.einsum("nhde,nhe,hd->n", f["M"], q, pp["b"]) + pp["c"]


init = lambda k: dict(A=jax.random.normal(k, (Hh, D, u_n.shape[1])) * 0.05, b=jax.random.normal(k, (Hh, D)) * 0.05, c=jnp.zeros(()))
print(f"best_q   새 질의를 학습:              AUC {fit(bil, init(jax.random.PRNGKey(1)), dict(u=u_n, M=M_n)):.3f}")
perm = rng.permutation(n)
print(f"shuffled 다른 에피소드의 시냅스:        AUC {fit(bil, init(jax.random.PRNGKey(1)), dict(u=u_n, M=M_n[perm])):.3f}")
