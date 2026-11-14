"""채집 격자의 에이전트 (JAX). 설계: docs/world/model_design.md

synapse  기억은 시냅스 M 하나. 매 스텝 정밀도 × 예측 오차 × 키 흔적으로 쓴다. 정책·가치 신경망이 없고 행동은 M을 읽어 고른다.
         느린 부품 (인코더, 디코더, 키 사영)은 같은 예측 오차를 BPTT로 줄여 배운다. M은 에피소드마다 비운다.
gru      비교군: 인코더 + GRU + 정책·가치 머리, A2C. 에피소드 안의 기억은 GRU 상태 벡터가 맡는다 (RL² 방식).
random   비교군: 무작위 행동.

각 에이전트는 init(key), carry0(key), step(params, carry, key) → (carry, out), loss(params, outs, carry) 를 낸다. out["r"]는 보상.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import env as E

sg = jax.lax.stop_gradient


class ModelCfg(NamedTuple):
    de: int = 32          # 패치 임베딩
    dh: int = 64          # 장면 벡터
    dk: int = 128         # 키
    hid: int = 256
    gamma: float = 0.9    # 보상 채널의 할인
    lam: float = 0.8      # 키 흔적
    temp: float = 0.05    # 행동 softmax 온도
    trace: bool = True    # 떼어 보기: 키 흔적
    precision: bool = True  # 떼어 보기: 정밀도 (끄면 쓰기 세기 상수)
    future: bool = True   # 떼어 보기: 앞으로의 보상 예측 (끄면 바로 다음 보상)
    lat: bool = True      # 장면 벡터끼리의 오차를 손실에 넣는가 (끄면 픽셀 복원만)
    slow: bool = False    # 느린 예측 경로: 다음 장면 = 신경망(장면, 행동) + 시냅스 읽기. 시냅스는 나머지만 맡는다


def dense(key, n_in, n_out, scale=1.0):
    return jax.random.normal(key, (n_in, n_out)) * scale / jnp.sqrt(n_in)


def enc_init(key, cfg, mc):
    ks = jax.random.split(key, 6)
    n_patch, d_patch = cfg.V * cfg.V, cfg.P * cfg.P * 3
    return dict(We=dense(ks[0], d_patch, mc.de), be=jnp.zeros(mc.de), pos=jax.random.normal(ks[1], (n_patch, mc.de)) * 0.1,
                W1=dense(ks[2], n_patch * mc.de, mc.hid), b1=jnp.zeros(mc.hid), W2=dense(ks[3], mc.hid, mc.dh), b2=jnp.zeros(mc.dh))


def encode(p, o):
    """패치 [n_patch, d_patch] → 장면 벡터 [dh] (크기 1로 맞춤)."""
    x = jax.nn.relu(o @ p["We"] + p["be"] + p["pos"])
    x = jax.nn.relu(x.reshape(-1) @ p["W1"] + p["b1"])
    h = x @ p["W2"] + p["b2"]
    return h * jax.lax.rsqrt(jnp.mean(h * h) + 1e-5)


def make_synapse(cfg: E.EnvCfg, mc: ModelCfg, tex):
    n_patch, d_patch, dh, dk = cfg.V * cfg.V, cfg.P * cfg.P * 3, mc.dh, mc.dk

    def init(key):
        ks = jax.random.split(key, 5)
        p = enc_init(ks[0], cfg, mc)
        p.update(D1=dense(ks[1], dh, mc.hid), d1=jnp.zeros(mc.hid), D2=dense(ks[2], mc.hid, n_patch * d_patch), d2=jnp.zeros(n_patch * d_patch),
                 Wk=jax.random.normal(ks[3], (cfg.A, dk, dh)) / jnp.sqrt(dh), bk=jnp.zeros((cfg.A, dk)),
                 log_tau=jnp.array(jnp.log(0.05)), logit_bs=jnp.array(0.0), logit_eta=jnp.array(-2.0))
        if mc.slow:
            k1, k2 = jax.random.split(ks[4])
            p.update(S1=dense(k1, dh + cfg.A, mc.hid), s1=jnp.zeros(mc.hid), S2=dense(k2, mc.hid, dh, 0.1), s2=jnp.zeros(dh))
        return p

    def decode(p, h):
        return (jax.nn.relu(h @ p["D1"] + p["d1"]) @ p["D2"] + p["d2"]).reshape(n_patch, d_patch)

    def keys(p, h):
        k = jnp.einsum("akh,h->ak", p["Wk"], h) + p["bk"]
        return k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

    def carry0(key):
        st = E.reset(key, cfg)
        # 시냅스 M의 출력: 다음 장면 [dh], 앞으로의 보상 [1], 그 예측의 평소 오차 [1]
        return dict(st=st, o=E.patches(st, tex, cfg), Ms=jnp.zeros((dh, dk)), Mr=jnp.zeros(dk), Mv=jnp.zeros(dk), e=jnp.zeros(dk))

    def step(p, c, key):
        h = encode(p, c["o"])
        K = keys(p, h)
        Q = K @ c["Mr"]                                                   # 읽기: 행동 후보마다 앞으로의 보상
        a = jax.random.categorical(key, sg(Q) / mc.temp)
        k, q = K[a], Q[a]
        pred = c["Ms"] @ k                                                # 읽기: 다음 장면
        if mc.slow:
            pred = pred + jax.nn.relu(jnp.concatenate([h, jax.nn.one_hot(a, cfg.A)]) @ p["S1"] + p["s1"]) @ p["S2"] + p["s2"]
        st2, r = E.step(c["st"], a, cfg)
        o2 = E.patches(st2, tex, cfg)
        h2 = encode(p, o2)
        Q2 = keys(p, h2) @ c["Mr"]
        boot = mc.gamma * (1 - r) * jnp.sum(jax.nn.softmax(Q2 / mc.temp) * Q2) if mc.future else 0.0
        delta = sg(r + boot - q)                                          # 보상 채널의 예측 오차. 느린 가중치의 학습에는 쓰지 않는다
        z = mc.gamma * mc.lam if (mc.trace and mc.future) else 0.0
        e = z * c["e"] + (1 - z * z) ** 0.5 * k                           # 키 흔적 (TDP처럼 크기 1로 맞춘다)
        var = jax.nn.relu(c["Mv"] @ k) + 1e-3                             # 이 키에서 보상 예측이 평소 틀리는 정도
        tau = jnp.exp(p["log_tau"])
        beta_r = tau / (tau + var) if mc.precision else jax.nn.sigmoid(p["log_tau"])
        err_s = sg(h2) - pred
        new = dict(st=st2, o=o2,
                   Ms=c["Ms"] + jax.nn.sigmoid(p["logit_bs"]) * jnp.outer(err_s, k),
                   Mr=c["Mr"] + beta_r * delta * e,
                   Mv=c["Mv"] + jax.nn.sigmoid(p["logit_eta"]) * (sg(delta * delta) - c["Mv"] @ k) * k,
                   e=e * (1 - r))
        rec = jnp.mean((decode(p, h) - c["o"]) ** 2)
        prd = jnp.mean((decode(p, pred) - o2) ** 2)
        lat = jnp.mean(err_s ** 2)
        return new, dict(r=r, loss=rec + prd + (lat if mc.lat else 0.0), rec=rec, prd=prd, lat=lat, td=delta ** 2, beta=beta_r)

    def loss(p, outs, carry):
        return outs["loss"].mean()

    return dict(init=init, carry0=carry0, step=step, loss=loss)


def make_gru(cfg: E.EnvCfg, mc: ModelCfg, tex, dg=128, gamma=0.95, lam=0.95, ent=0.01):
    dh, d_in = mc.dh, mc.dh + cfg.A + 1

    def init(key):
        ks = jax.random.split(key, 6)
        p = enc_init(ks[0], cfg, mc)
        p.update(Gi=dense(ks[1], d_in, 3 * dg), Gh=dense(ks[2], dg, 3 * dg), gb=jnp.zeros(3 * dg),
                 Pi=dense(ks[3], dg, cfg.A, 0.01), pb=jnp.zeros(cfg.A), Vw=dense(ks[4], dg, 1), vb=jnp.zeros(1))
        return p

    def cell(p, g, x):
        gi, gh = x @ p["Gi"] + p["gb"], g @ p["Gh"]
        (ir, iz, in_), (hr, hz, hn) = jnp.split(gi, 3), jnp.split(gh, 3)
        r, z = jax.nn.sigmoid(ir + hr), jax.nn.sigmoid(iz + hz)
        return (1 - z) * jnp.tanh(in_ + r * hn) + z * g

    def carry0(key):
        st = E.reset(key, cfg)
        return dict(st=st, o=E.patches(st, tex, cfg), g=jnp.zeros(dg), a=jnp.zeros(cfg.A), r=jnp.zeros(()))

    def forward(p, c):
        g = cell(p, c["g"], jnp.concatenate([encode(p, c["o"]), c["a"], c["r"][None]]))
        return g, g @ p["Pi"] + p["pb"], (g @ p["Vw"] + p["vb"])[0]

    def step(p, c, key):
        g, logits, v = forward(p, c)
        a = jax.random.categorical(key, sg(logits))
        logp = jax.nn.log_softmax(logits)
        st2, r = E.step(c["st"], a, cfg)
        new = dict(st=st2, o=E.patches(st2, tex, cfg), g=g, a=jax.nn.one_hot(a, cfg.A), r=r)
        return new, dict(r=r, logp=logp[a], v=v, H=-jnp.sum(jnp.exp(logp) * logp))

    def loss(p, outs, carry):
        """A2C + GAE. outs의 축은 [B, L]. 조각 끝의 가치는 마지막 carry에서 한 번 더 본다."""
        v_last = sg(jax.vmap(lambda c: forward(p, c)[2])(carry))
        v = sg(outs["v"])
        v_next = jnp.concatenate([v[:, 1:], v_last[:, None]], 1)
        d = outs["r"] + gamma * v_next - v

        def back(acc, x):
            acc = x + gamma * lam * acc
            return acc, acc

        _, adv = jax.lax.scan(back, jnp.zeros(d.shape[0]), d.T, reverse=True)
        adv = adv.T
        ret = adv + v
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-6)
        return (-(outs["logp"] * adv_n) + 0.5 * (outs["v"] - ret) ** 2 - ent * outs["H"]).mean()

    return dict(init=init, carry0=carry0, step=step, loss=loss)


def make_random(cfg: E.EnvCfg, mc: ModelCfg, tex):
    def carry0(key):
        return dict(st=E.reset(key, cfg))

    def step(p, c, key):
        st2, r = E.step(c["st"], jax.random.randint(key, (), 0, cfg.A), cfg)
        return dict(st=st2), dict(r=r)

    return dict(init=lambda key: dict(), carry0=carry0, step=step, loss=None)


def make_oracle(cfg: E.EnvCfg, mc: ModelCfg, tex):
    """상한: 첫 보상까지는 무작위로 걷고, 그 뒤로는 보상 칸까지 최단 경로로 간다."""
    def carry0(key):
        return dict(st=E.reset(key, cfg), found=jnp.zeros(()))

    def step(p, c, key):
        d = c["st"].goal - c["st"].pos
        go = jnp.where(d[0] != 0, jnp.where(d[0] > 0, 2, 3), jnp.where(d[1] > 0, 0, 1))
        a = jnp.where(c["found"] > 0, go, jax.random.randint(key, (), 0, cfg.A))
        st2, r = E.step(c["st"], a, cfg)
        return dict(st=st2, found=jnp.maximum(c["found"], r)), dict(r=r)

    return dict(init=lambda key: dict(), carry0=carry0, step=step, loss=None)


def make(name, cfg, mc, tex):
    return dict(synapse=make_synapse, gru=make_gru, random=make_random, oracle=make_oracle)[name](cfg, mc, tex)
