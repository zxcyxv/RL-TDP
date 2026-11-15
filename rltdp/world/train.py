"""채집 격자에서 학습·평가. 에피소드를 L스텝 조각으로 잘라 조각마다 한 번 갱신한다 (기억과 상태는 조각 사이에 이어 간다).

/venv/jax/bin/python -m rltdp.world.train --agent synapse [--iters 2000] [--no_trace | --no_precision | --no_future]
/venv/jax/bin/python -m rltdp.world.train --agent gru
/venv/jax/bin/python -m rltdp.world.train --agent random
평가는 항상 새 배치 (에피소드마다 타일·출발·보상 칸이 무작위)에서, 가중치를 고정하고 한다.
"""
import argparse
import json
import os
import time

import jax
import jax.flatten_util
import jax.numpy as jnp
import optax

from . import env as E
from .agents import ModelCfg
from .agents import make as _make


def make(name, cfg, mc, tex):
    if name == "tdp":
        from .tdp_agent import make_tdp
        return make_tdp(cfg, mc, tex)
    if name == "tdpc":
        from .tdp_credit import make_tdp_credit
        return make_tdp_credit(cfg, mc, tex)
    return _make(name, cfg, mc, tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["synapse", "tdp", "tdpc", "gru", "random", "oracle"], required=True)
    ap.add_argument("--iters", type=int, default=2000, help="에피소드 묶음 수 (묶음 하나 = 환경 B개 × T스텝)")
    ap.add_argument("--B", type=int, default=512)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_B", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--no_trace", action="store_true")
    ap.add_argument("--no_precision", action="store_true")
    ap.add_argument("--no_future", action="store_true")
    ap.add_argument("--no_lat", action="store_true")
    ap.add_argument("--slow", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--load", default=None, help="저장된 가중치로 평가만 한다 (학습 때와 다른 격자 크기·길이도 가능)")
    ap.add_argument("--eval_N", type=int, default=None)
    ap.add_argument("--eval_T", type=int, default=None)
    ap.add_argument("--out", default="runs/world")
    args = ap.parse_args()

    cfg = E.EnvCfg(N=args.N, T=args.T)
    mc = ModelCfg(trace=not args.no_trace, precision=not args.no_precision, future=not args.no_future, lat=not args.no_lat, slow=args.slow)
    tex = E.textures(cfg)
    ag = make(args.agent, cfg, mc, tex)
    name = args.tag or args.agent + "".join(s for f, s in ((args.no_trace, "_notrace"), (args.no_precision, "_noprec"), (args.no_future, "_nofuture"), (args.no_lat, "_nolat"), (args.slow, "_slow")) if f)
    os.makedirs(args.out, exist_ok=True)
    assert cfg.T % args.L == 0

    def rollout(params, carry, key, L):
        def one(c, k):
            return jax.lax.scan(lambda cc, kk: ag["step"](params, cc, kk), c, jax.random.split(k, L))
        return jax.vmap(one)(carry, jax.random.split(key, carry["st"].pos.shape[0]))

    @jax.jit
    def evaluate(params, key):
        k1, k2 = jax.random.split(key)
        carry = jax.vmap(ag["carry0"])(jax.random.split(k1, args.eval_B))
        dist = jnp.abs(carry["st"].start - carry["st"].goal).sum(-1)
        _, outs = rollout(params, carry, k2, cfg.T)
        return E.summarize(outs["r"], dist)

    key = jax.random.PRNGKey(args.seed)
    key, k0 = jax.random.split(key)
    params = ag["init"](k0)
    fmt = lambda m: {k: round(float(v), 3) for k, v in m.items()}
    if args.load or args.eval_N or args.eval_T:
        # 평가만: 격자 크기와 길이를 바꿔도 가중치의 모양은 같다 (시야가 같으므로)
        cfg = E.EnvCfg(N=args.eval_N or args.N, T=args.eval_T or args.T)
        ag = make(args.agent, cfg, mc, tex)
        if args.load:
            params = jax.flatten_util.ravel_pytree(params)[1](jnp.load(args.load))
        res = dict(agent=name, N=cfg.N, T=cfg.T, **fmt(evaluate(params, jax.random.PRNGKey(10_000))))
        print(json.dumps(res))
        return
    if ag["loss"] is None:
        res = fmt(evaluate(params, jax.random.PRNGKey(10_000)))
        print(name, res)
        json.dump(res, open(os.path.join(args.out, name + ".json"), "w"))
        return

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = opt.init(params)

    @jax.jit
    def update(params, opt_state, carry, key):
        def f(p):
            carry2, outs = rollout(p, carry, key, args.L)
            return ag["loss"](p, outs, carry2), (carry2, outs)
        (loss, (carry2, outs)), g = jax.value_and_grad(f, has_aux=True)(params)
        upd, opt_state = opt.update(g, opt_state)
        return optax.apply_updates(params, upd), opt_state, carry2, loss, {k: v.mean() for k, v in outs.items()}

    reset = jax.jit(lambda k: jax.vmap(ag["carry0"])(jax.random.split(k, args.B)))
    log = open(os.path.join(args.out, name + ".jsonl"), "w")
    t0, best = time.time(), -1.0
    for it in range(1, args.iters + 1):
        key, kr = jax.random.split(key)
        carry = reset(kr)
        for _ in range(cfg.T // args.L):
            key, ku = jax.random.split(key)
            params, opt_state, carry, loss, m = update(params, opt_state, carry, ku)
        if it % args.eval_every == 0 or it == args.iters:
            rec = dict(it=it, steps=it * args.B * cfg.T, sec=round(time.time() - t0), loss=round(float(loss), 4),
                       **{"tr_" + k: round(float(v), 4) for k, v in m.items() if k != "r"}, **fmt(evaluate(params, jax.random.PRNGKey(10_000))))
            for k in ("log_tau", "logit_bs", "logit_eta", "logit_beta"):
                if k in params:
                    rec[k] = round(float(params[k]), 3)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if rec["rewards"] > best:                                # 평가에서 가장 좋았던 시점의 가중치를 남긴다
                best = rec["rewards"]
                jnp.save(os.path.join(args.out, name + "_params_best.npy"), jax.flatten_util.ravel_pytree(params)[0])
            print(name, rec, flush=True)
    jnp.save(os.path.join(args.out, name + "_params.npy"), jax.flatten_util.ravel_pytree(params)[0])


if __name__ == "__main__":
    main()
