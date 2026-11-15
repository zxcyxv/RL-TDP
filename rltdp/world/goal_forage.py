"""스스로 움직이는 에이전트 + 세션 안 목표 전환 (docs/world/README.md 10절).

환경: N×N 격자 (N = 9), 시야 3×3 (V = 3, 타일 7×7×3 패치 → [9, 147]). 배경 무늬 (타일 1..12)와 자원 3종 (타일 13..15)의 위치는 세계마다 무작위.
  목표 자원 g는 별도 입력 (one-hot). 목표 자원 칸에 들어가면 보상 +1, 목표가 다른 종류로 바뀐다. 자원은 남는다. 에피소드 T = 256.
에이전트 (같은 A2C, GAE):
  gru      인코더 (패치 임베딩 학습) + GRU 128 + 정책·가치. 입력 [장면, 목표, 직전 행동, 직전 보상] (RL²).
  synapse  gru + 사실 기억 시냅스. 매 스텝 각 행동 a에 대해 "그 행동 뒤에 목표 자원이 나왔었나"를 읽어 GRU 입력에 더한다:
           feat_a = cos(M k(o_t, a), φ_g − ȳ). 시냅스 = 키 선행 흔적 (α = 2^(−1/8)) + 정밀도 가중 쓰기 (RLS, R = 0.01, P₀ = I),
           결과 y = 다음 관측 가운데 칸의 임베딩, ȳ = 타일 임베딩 평균. 키 k(o, a) = unit(Wk[a] flatten(emb(o))), 고정 난수, dk = 128.
           쓰기 세기 1 (이 환경은 배경이 세계마다 달라 모든 결과가 세계별이다). 시냅스에는 경사가 흐르지 않는다.
측정: 목표 전환마다 도착까지 스텝. 전환 시점에 그 목표 자원을 이번 에피소드에서 이미 본 적이 있는가로 나눈다.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import env as E

N, V, T, NRES = 9, 3, 256, 3
A = 4
CFG = E.EnvCfg(N=N, V=V, n_types=16, T=T)
TEX = E.textures(CFG)
RES0 = 13
MOVES = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
H_EMB, DK, DG = 8, 128, 128
ALPHA = 2.0 ** (-1 / 8)


class St(NamedTuple):
    tiles: jnp.ndarray
    pos: jnp.ndarray
    goal: jnp.ndarray
    res_pos: jnp.ndarray     # [NRES, 2]
    seen: jnp.ndarray        # [NRES] 이번 에피소드에서 시야에 들어온 적이 있나


def reset(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    tiles = jax.random.randint(k1, (N, N), 1, 13)
    cells = jax.random.permutation(k2, N * N)[: NRES + 1]
    res_pos = jnp.stack([cells[:NRES] // N, cells[:NRES] % N], -1)
    tiles = tiles.at[res_pos[:, 0], res_pos[:, 1]].set(RES0 + jnp.arange(NRES))
    pos = jnp.stack([cells[NRES] // N, cells[NRES] % N])
    st = St(tiles, pos, jax.random.randint(k3, (), 0, NRES), res_pos, jnp.zeros(NRES, bool))
    return st._replace(seen=visible(st))


def visible(st):
    return st.seen | jnp.all(jnp.abs(st.res_pos - st.pos) <= V // 2, -1)


def patches(st):
    r = V // 2
    ids = jax.lax.dynamic_slice(jnp.pad(st.tiles, r), (st.pos[0], st.pos[1]), (V, V))
    return TEX[ids].reshape(V * V, -1)


def step(st, a, key):
    new = jnp.clip(st.pos + MOVES[a], 0, N - 1)
    hit = jnp.all(new == st.res_pos[st.goal])
    g2 = jnp.where(hit, (st.goal + 1 + jax.random.randint(key, (), 0, NRES - 1)) % NRES, st.goal)
    st2 = st._replace(pos=new, goal=g2)
    return st2._replace(seen=visible(st2)), hit.astype(jnp.float32)


def dense(key, i, o, scale=1.0):
    return jax.random.normal(key, (i, o)) * scale / jnp.sqrt(i)


def make(kind, seed=0):
    kr = jax.random.split(jax.random.PRNGKey(900 + seed), 2)
    We_fix = jax.random.normal(kr[0], (147, H_EMB)) / jnp.sqrt(147.0)                 # 시냅스용 고정 임베딩
    Wk = jax.random.normal(kr[1], (A, DK, V * V * H_EMB)) / jnp.sqrt(V * V * H_EMB)
    tile_emb = TEX.reshape(TEX.shape[0], -1) @ We_fix
    ybar = tile_emb[1:].mean(0)
    phi = tile_emb[RES0:RES0 + NRES]                                                    # 목표 자원 무늬의 임베딩 (세계와 무관)
    n_feat = A if kind == "synapse" else 0
    d_in = 64 + NRES + A + 1 + n_feat

    def init(key):
        ks = jax.random.split(key, 7)
        return dict(We=dense(ks[0], 147, 16), W1=dense(ks[1], V * V * 16, 128), b1=jnp.zeros(128), W2=dense(ks[2], 128, 64), b2=jnp.zeros(64),
                    Gi=dense(ks[3], d_in, 3 * DG), Gh=dense(ks[4], DG, 3 * DG), gb=jnp.zeros(3 * DG),
                    Pi=dense(ks[5], DG, A, 0.01), pb=jnp.zeros(A), Vw=dense(ks[6], DG, 1), vb=jnp.zeros(1))

    def enc(p, o):
        x = jax.nn.relu(o @ p["We"]).reshape(-1)
        return jax.nn.relu(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]

    def keys(o):
        e = (o @ We_fix).reshape(-1)
        k = jnp.einsum("akh,h->ak", Wk, e)
        return k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)                 # [A, DK]

    def carry0(key):
        st = reset(key)
        return dict(st=st, g=jnp.zeros(DG), a=jnp.zeros(A), r=jnp.zeros(()),
                    M=jnp.zeros((H_EMB, DK)), P=jnp.eye(DK), e=jnp.zeros(DK))

    def forward(p, c):
        o = patches(c["st"])
        parts = [enc(p, o), jax.nn.one_hot(c["st"].goal, NRES), c["a"], c["r"][None]]
        if kind == "synapse":
            Kc = keys(o)
            yh = Kc @ c["M"].T                                                          # [A, H]
            gv = phi[c["st"].goal] - ybar
            feat = (yh @ gv) / (jnp.linalg.norm(yh, axis=-1) * jnp.linalg.norm(gv) + 1e-8)
            parts.append(jax.lax.stop_gradient(feat))
        x = jnp.concatenate(parts)
        gi, gh = x @ p["Gi"] + p["gb"], c["g"] @ p["Gh"]
        (ir, iz, in_), (hr, hz, hn) = jnp.split(gi, 3), jnp.split(gh, 3)
        rr, z = jax.nn.sigmoid(ir + hr), jax.nn.sigmoid(iz + hz)
        g = (1 - z) * jnp.tanh(in_ + rr * hn) + z * c["g"]
        return g, g @ p["Pi"] + p["pb"], (g @ p["Vw"] + p["vb"])[0], o

    def stepf(p, c, key):
        k1, k2 = jax.random.split(key)
        g, logits, v, o = forward(p, c)
        a = jax.random.categorical(k1, jax.lax.stop_gradient(logits))
        logp = jax.nn.log_softmax(logits)
        st2, r = step(c["st"], a, k2)
        new = dict(c, st=st2, g=g, a=jax.nn.one_hot(a, A), r=r)
        if kind == "synapse":                                                           # 쓰기: 이 (장면, 행동)의 흔적 → 다음 칸
            k = keys(o)[a]
            e = ALPHA * c["e"] + k
            y = (patches(st2) @ We_fix)[(V * V) // 2] - ybar
            Pz = c["P"] @ e
            S = 0.01 + e @ Pz
            M = c["M"] + jnp.outer(y - c["M"] @ e, Pz) / S
            P = c["P"] - jnp.outer(Pz, Pz) / S
            new.update(M=jax.lax.stop_gradient(M), P=jax.lax.stop_gradient(P), e=jax.lax.stop_gradient(e))
        H = -jnp.sum(jnp.exp(logp) * logp)
        info = dict(r=r, logp=logp[a], v=v, H=H, seen=c["st"].seen[c["st"].goal], goal=c["st"].goal)
        return new, info

    def loss(p, outs, carry, gamma=0.95, lam=0.95, ent=0.01):
        v_last = jax.lax.stop_gradient(jax.vmap(lambda c: forward(p, c)[2])(carry))
        v = jax.lax.stop_gradient(outs["v"])
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

    return dict(init=init, carry0=carry0, step=stepf, loss=loss)


def segments(r, seen):
    """에피소드 하나의 목표 구간 길이. r [T], seen [T] (그 스텝의 목표를 이미 본 적이 있나). 반환: 구간 끝 마스크, 길이, 시작 시점의 seen."""
    t = jnp.arange(r.shape[0])
    start = jnp.concatenate([jnp.array([0]), (t[:-1] + 1)])
    # 구간 시작 = 0 또는 직전 스텝에 보상
    is_start = jnp.concatenate([jnp.array([True]), r[:-1] > 0])
    seg_id = jnp.cumsum(is_start) - 1
    start_t = jax.ops.segment_max(jnp.where(is_start, t, -1), seg_id, num_segments=r.shape[0])
    start_seen = jax.ops.segment_max(jnp.where(is_start, seen.astype(jnp.int32), -1), seg_id, num_segments=r.shape[0])
    end = r > 0
    length = t - start_t[seg_id] + 1
    return end, length, start_seen[seg_id] > 0
