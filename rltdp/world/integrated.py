"""하나로 합친 에이전트: 학습된 키 (느린, F) + 세션 안 시냅스 (사실 + 보상 신용할당) (docs/world/README.md 12절).

느린 학습 (보상과 무관): 키 k(o, a) = unit(SwiGLU([flatten(emb(o)), onehot(a)])), dk = 128.
  목적 = F의 정확도 항: 세션 안 시냅스가 다음 관측 특징 y (칸 위치별, 고정 임베딩 − 평균)를 예측한 오차.
  시냅스 쓰기는 흔적 (α = 2^(−1/8)) + 정밀도 가중 (RLS, R = 0.01, P₀ = I), 쓰기 세기 1
  (이 격자는 배경이 세계마다 무작위라 다음 칸이 전부 세계별이다: 학습된 월드 모델의 평균은 전체 평균, 놀라움은 어디서나 같다).
  무작위 탐색 자료, 에피소드 256스텝, 32스텝 조각 BPTT (조각 사이 시냅스는 상수로 이어짐).
빠른 시냅스 (세션 안, 경사 없음): credit.synapse_episode에 이 키를 넣는다 (value = TD 신용할당, fact = 후속 특징 + 보상 가중치).
"""
import jax
import jax.numpy as jnp
import optax

from . import credit as C
from . import goal_forage as GF

A, V, H_EMB, DK, SWI = GF.A, GF.V, GF.H_EMB, 128, 128


def init(key):
    k1, k2 = jax.random.split(key)
    return dict(Ku=jax.random.normal(k1, (V * V * H_EMB + A, 2 * SWI)) / jnp.sqrt(V * V * H_EMB + A),
                Kd=jax.random.normal(k2, (SWI, DK)) / jnp.sqrt(SWI))


def make_keys(p, seed=0):
    kr = jax.random.split(jax.random.PRNGKey(900 + seed), 2)
    We = jax.random.normal(kr[0], (147, H_EMB)) / jnp.sqrt(147.0)

    def keys(o):
        e = (o @ We).reshape(-1)
        x = jnp.concatenate([jnp.tile(e, (A, 1)), jnp.eye(A)], 1)
        g, u = jnp.split(x @ p["Ku"], 2, -1)
        k = (jax.nn.silu(g) * u) @ p["Kd"]
        return k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)
    return keys


def train(seed=0, N=9, T=256, L=32, B=64, iters=1500, lr=1e-3, log=print):
    reset, patches, step = C.make_env(N)
    _, feat = C.fixed(seed)
    HY = V * V * H_EMB
    p = init(jax.random.PRNGKey(seed))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    st = opt.init(p)

    def chunk(p, c, key):
        keys = make_keys(p, seed)

        def f(c, kk):
            o = patches(c["st"])
            a = jax.random.randint(kk, (), 0, A)
            k = keys(o)[a]
            st2, _ = step(c["st"], a, 0)
            y = feat(patches(st2))
            e = C.GF.ALPHA * c["e"] + k
            err = jnp.sum((y - c["M"] @ e) ** 2)
            Pz = c["P"] @ e
            S = 0.01 + e @ Pz
            M = c["M"] + jnp.outer(y - c["M"] @ e, Pz) / S
            P = c["P"] - jnp.outer(Pz, Pz) / S
            return dict(st=st2, M=M, P=P, e=e), err
        return jax.lax.scan(f, c, jax.random.split(key, L))

    @jax.jit
    def upd(p, st, carry, key):
        def lf(p):
            c2, err = jax.vmap(lambda c, k: chunk(p, c, k))(carry, jax.random.split(key, B))
            return err.mean(), c2
        (l, c2), g = jax.value_and_grad(lf, has_aux=True)(p)
        g, st = opt.update(g, st, p)
        return optax.apply_updates(p, g), st, jax.lax.stop_gradient(c2), l

    def c0(k):
        return dict(st=reset(k), M=jnp.zeros((HY, DK)), P=jnp.eye(DK), e=jnp.zeros(DK))
    key = jax.random.PRNGKey(2000 + seed)
    for it in range(iters):
        key, k = jax.random.split(key)
        carry = jax.vmap(c0)(jax.random.split(k, B))
        for _ in range(T // L):
            key, ku = jax.random.split(key)
            p, st, carry, l = upd(p, st, carry, ku)
        if it % 300 == 0 or it == iters - 1:
            log(f"  [keys seed {seed}] it {it} next-view prediction error {float(l):.3f}")
    return p
