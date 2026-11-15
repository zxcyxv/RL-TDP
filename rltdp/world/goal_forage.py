"""스스로 움직이는 에이전트 + 세션 안 목표 전환 (docs/world/README.md 10절).

환경: N×N 격자, 시야 3×3 (V = 3, 타일 7×7×3 패치 → [9, 147]). 배경 무늬 (타일 1..12)와 자원 3종 (타일 13..15)의 위치는 세계마다 무작위.
  목표 자원 g는 별도 입력 (one-hot). 목표 자원 칸에 들어가면 보상 +1, 목표가 다른 종류로 바뀐다. 자원은 남는다. 에피소드 T.
에이전트 (같은 A2C, GAE, 같은 GRU 128):
  gru        인코더 (패치 임베딩 학습) + GRU + 정책·가치. 입력 [장면, 목표, 직전 행동, 직전 보상] (RL²).
  synapse    gru + 시냅스, 키 = 고정 난수 선형 사영, 쓰기 = 다음 칸 회귀 (RLS). 10절 첫 판 (실패).
  synapse_l  gru + 시냅스, 키 = 학습되는 SwiGLU([9칸 임베딩, 행동]) (칸 사이 곱셈 상호작용),
             쓰기 = 후속 특징식 TD(λ): δ = y_{t+1} + γ M k_{t+1} − M k_t,  M += β δ eᵀ / |e|²,  e ← γλ e + k_t  (γ = 0.9, λ = 0.8, β = 0.5)
             y = 칸 위치별 특징 (고정 임베딩 − 평균, 9 × 8), 읽기: v(a) = (M k(o, a))_가운데 · (φ_g − ȳ) / |φ_g − ȳ|  → GRU 입력 [4].
             정책 경사가 학습 조각 (32스텝) 안에서 시냅스 쓰기와 읽기를 거쳐 키 인코더로 흐른다 (조각 경계에서 M은 상수로 이어진다).
측정: 목표 전환마다 도착까지 스텝. 전환 시점에 그 목표 자원을 이번 에피소드에서 이미 본 적이 있는가 (시야 3×3 안에 들어온 적)로 나눈다.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import env as E

V, NRES, A = 3, 3, 4
TEX = E.textures(E.EnvCfg(N=9, V=V, n_types=16))
RES0 = 13
MOVES = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
H_EMB, DK, DG = 8, 128, 128
ALPHA = 2.0 ** (-1 / 8)
GAM, LAM, BETA = 0.9, 0.8, 0.5


class St(NamedTuple):
    tiles: jnp.ndarray
    pos: jnp.ndarray
    goal: jnp.ndarray
    res_pos: jnp.ndarray
    seen: jnp.ndarray


def make_env(N):
    def visible(st):
        return st.seen | jnp.all(jnp.abs(st.res_pos - st.pos) <= V // 2, -1)

    def reset(key):
        k1, k2, k3 = jax.random.split(key, 3)
        tiles = jax.random.randint(k1, (N, N), 1, 13)
        cells = jax.random.permutation(k2, N * N)[: NRES + 1]
        res_pos = jnp.stack([cells[:NRES] // N, cells[:NRES] % N], -1)
        tiles = tiles.at[res_pos[:, 0], res_pos[:, 1]].set(RES0 + jnp.arange(NRES))
        pos = jnp.stack([cells[NRES] // N, cells[NRES] % N])
        st = St(tiles, pos, jax.random.randint(k3, (), 0, NRES), res_pos, jnp.zeros(NRES, bool))
        return st._replace(seen=visible(st))

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
    return reset, patches, step


def dense(key, i, o, scale=1.0):
    return jax.random.normal(key, (i, o)) * scale / jnp.sqrt(i)


def make(kind, seed=0, N=9):
    reset, patches, step = make_env(N)
    kr = jax.random.split(jax.random.PRNGKey(900 + seed), 2)
    We_fix = jax.random.normal(kr[0], (147, H_EMB)) / jnp.sqrt(147.0)                 # 시냅스 결과용 고정 임베딩
    Wk = jax.random.normal(kr[1], (A, DK, V * V * H_EMB)) / jnp.sqrt(V * V * H_EMB)
    tile_emb = TEX.reshape(TEX.shape[0], -1) @ We_fix
    ybar = tile_emb[1:].mean(0)
    phi = tile_emb[RES0:RES0 + NRES]
    HY = V * V * H_EMB if kind == "synapse_l" else H_EMB
    n_feat = A if kind in ("synapse", "synapse_l") else 0
    d_in = 64 + NRES + A + 1 + n_feat
    SWI = 128

    def init(key):
        ks = jax.random.split(key, 10)
        p = dict(We=dense(ks[0], 147, 16), W1=dense(ks[1], V * V * 16, 128), b1=jnp.zeros(128), W2=dense(ks[2], 128, 64), b2=jnp.zeros(64),
                 Gi=dense(ks[3], d_in, 3 * DG), Gh=dense(ks[4], DG, 3 * DG), gb=jnp.zeros(3 * DG),
                 Pi=dense(ks[5], DG, A, 0.01), pb=jnp.zeros(A), Vw=dense(ks[6], DG, 1), vb=jnp.zeros(1))
        if kind == "synapse_l":
            p.update(Ku=dense(ks[7], V * V * H_EMB + A, 2 * SWI), Kd=dense(ks[8], SWI, DK))
        return p

    def enc(p, o):
        x = jax.nn.relu(o @ p["We"]).reshape(-1)
        return jax.nn.relu(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]

    def keys(p, o):
        e = (o @ We_fix).reshape(-1)
        if kind == "synapse_l":                                                          # SwiGLU([칸 임베딩, 행동]) → 키, 행동 4개
            x = jnp.concatenate([jnp.tile(e, (A, 1)), jnp.eye(A)], 1)
            g, u = jnp.split(x @ p["Ku"], 2, -1)
            return (jax.nn.silu(g) * u) @ p["Kd"]                                        # [A, DK]
        k = jnp.einsum("akh,h->ak", Wk, e)
        return k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)

    def carry0(key):
        st = reset(key)
        return dict(st=st, g=jnp.zeros(DG), a=jnp.zeros(A), r=jnp.zeros(()),
                    M=jnp.zeros((HY, DK)), P=jnp.eye(DK), e=jnp.zeros(DK), kp=jnp.zeros(DK), y=jnp.zeros(HY))

    def forward(p, c):
        o = patches(c["st"])
        parts = [enc(p, o), jax.nn.one_hot(c["st"].goal, NRES), c["a"], c["r"][None]]
        Kc = None
        if kind != "gru":
            Kc = keys(p, o)
            gv = phi[c["st"].goal] - ybar
            yh = Kc @ c["M"].T
            if kind == "synapse":
                feat = jax.lax.stop_gradient((yh @ gv) / (jnp.linalg.norm(yh, axis=-1) * jnp.linalg.norm(gv) + 1e-8))
            else:
                feat = yh.reshape(A, V * V, H_EMB)[:, (V * V) // 2] @ gv / jnp.linalg.norm(gv)
            parts.append(feat)
        x = jnp.concatenate(parts)
        gi, gh = x @ p["Gi"] + p["gb"], c["g"] @ p["Gh"]
        (ir, iz, in_), (hr, hz, hn) = jnp.split(gi, 3), jnp.split(gh, 3)
        rr, z = jax.nn.sigmoid(ir + hr), jax.nn.sigmoid(iz + hz)
        g = (1 - z) * jnp.tanh(in_ + rr * hn) + z * c["g"]
        return g, g @ p["Pi"] + p["pb"], (g @ p["Vw"] + p["vb"])[0], o, Kc

    def stepf(p, c, key):
        k1, k2 = jax.random.split(key)
        g, logits, v, o, Kc = forward(p, c)
        a = jax.random.categorical(k1, jax.lax.stop_gradient(logits))
        logp = jax.nn.log_softmax(logits)
        st2, r = step(c["st"], a, k2)
        new = dict(c, st=st2, g=g, a=jax.nn.one_hot(a, A), r=r)
        if kind == "synapse":
            k = Kc[a]
            e = ALPHA * c["e"] + k
            y = (patches(st2) @ We_fix)[(V * V) // 2] - ybar
            Pz = c["P"] @ e
            S = 0.01 + e @ Pz
            M = c["M"] + jnp.outer(y - c["M"] @ e, Pz) / S
            P = c["P"] - jnp.outer(Pz, Pz) / S
            new.update(M=jax.lax.stop_gradient(M), P=jax.lax.stop_gradient(P), e=jax.lax.stop_gradient(e))
        elif kind == "synapse_l":                                                       # 직전 전이 (kp → y)의 TD 오차, 지금 키로 부트스트랩
            k = Kc[a]
            M = c["M"]
            delta = c["y"] + GAM * M @ k - M @ c["kp"]
            M = M + BETA * jnp.outer(delta, c["e"]) / (c["e"] @ c["e"] + 1e-6)
            e = GAM * LAM * c["e"] + k
            y = ((patches(st2) @ We_fix) - ybar).reshape(-1)
            new.update(M=M, e=e, kp=k, y=y)
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
    """에피소드 하나의 목표 구간. 반환: 구간 끝 마스크, 길이, 구간 시작 때 그 목표를 이미 본 적이 있었나."""
    t = jnp.arange(r.shape[0])
    is_start = jnp.concatenate([jnp.array([True]), r[:-1] > 0])
    seg_id = jnp.cumsum(is_start) - 1
    start_t = jax.ops.segment_max(jnp.where(is_start, t, -1), seg_id, num_segments=r.shape[0])
    start_seen = jax.ops.segment_max(jnp.where(is_start, seen.astype(jnp.int32), -1), seg_id, num_segments=r.shape[0])
    end = r > 0
    length = t - start_t[seg_id] + 1
    return end, length, start_seen[seg_id] > 0
