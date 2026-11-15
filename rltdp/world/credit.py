"""세션 안 신용할당으로 새 보상에 곧바로 적응하는가 (docs/world/README.md 11절).

환경: goal_forage의 격자 (N = 9, 시야 3×3, 배경·자원 3종 위치는 세계마다 무작위). 목표 입력 없음.
  자원 3종 중 한 종류 (rtype)만 보상 +1 (밟을 때마다, 자원은 남는다). 무엇이 보상인지는 겪어서만 안다.
  학습 분포: rtype = 0 고정. 시험: rtype = 2 (학습 중 한 번도 보상이 아니었던 자원). 세션 중 전환: t = T/2에서 알리지 않고 rtype을 1로.
에이전트:
  frozen    DreamerV3식: 시야 → MLP 정책·가치, A2C로 학습 분포에서 학습한 뒤 고정. 새 보상에는 경사 갱신으로만 따라간다.
  value     시냅스 가치 캐시 (tdpc 규칙): Q(s, a) = w · k(s, a), 행동 = softmax(Q / 0.05),
            δ = r + γ Σ softmax(Q(s′)) Q(s′) − Q(s, a), e ← γλ e + k, w += β δ e.
  fact      시냅스 사실 기억 + 세션 안 보상 가중치: 후속 특징 Ψ (보상과 무관, "어디로 가면 무엇을 보나"),
            δψ = y_{t+1} + γ Ψ k_{t+1} − Ψ k_t, Ψ += β δψ eᵀ / |e|²; 보상 가중치 u (어떤 특징이 보상인가): u += β (r − u · y) y / |y|²;
            Q(s, a) = u · Ψ k(s, a), 행동 = softmax(Q / 0.05).
  키 k(s, a) = unit(Wk[a] flatten(emb(시야))), 고정 난수 (dk = 128). y = 칸 위치별 특징 (고정 임베딩 − 평균, 9 × 8).
  γ = 0.9, λ = 0.8, β = 0.5 (tdpc와 같음). value와 fact는 학습이 없다 (모든 적응이 세션 안 시냅스).
"""
import jax
import jax.numpy as jnp

from . import goal_forage as GF

A, V, H_EMB, DK = GF.A, GF.V, GF.H_EMB, 128
GAM, LAM, BETA, TEMP = 0.9, 0.8, 0.5, 0.05


def make_env(N):
    reset0, patches, _ = GF.make_env(N)

    def reset(key):
        return reset0(key)

    def step(st, a, rtype):
        new = jnp.clip(st.pos + GF.MOVES[a], 0, N - 1)
        r = jnp.all(new == st.res_pos[rtype]).astype(jnp.float32)
        st2 = st._replace(pos=new)
        return st2, r
    return reset, patches, step


def fixed(seed=0):
    kr = jax.random.split(jax.random.PRNGKey(900 + seed), 2)
    We = jax.random.normal(kr[0], (147, H_EMB)) / jnp.sqrt(147.0)
    Wk = jax.random.normal(kr[1], (A, DK, V * V * H_EMB)) / jnp.sqrt(V * V * H_EMB)
    ybar = (GF.TEX.reshape(GF.TEX.shape[0], -1) @ We)[1:].mean(0)

    def keys(o):
        k = jnp.einsum("akh,h->ak", Wk, (o @ We).reshape(-1))
        return k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)

    def feat(o):
        return ((o @ We) - ybar).reshape(-1)
    return keys, feat


def synapse_episode(kind, key, N, T, rtypes, seed=0):
    """학습 없는 시냅스 에이전트 한 에피소드. rtypes [T] 스텝별 보상 자원. 반환: 보상 [T]."""
    reset, patches, step = make_env(N)
    keys, feat = fixed(seed)
    HY = V * V * H_EMB
    k0, key = jax.random.split(key)
    st = reset(k0)

    def q_of(c, K):
        if kind == "value":
            return K @ c["w"]
        return (K @ c["Psi"].T) @ c["u"]

    def f(c, x):
        kk, rt = x
        o = patches(c["st"])
        K = keys(o)
        Q = q_of(c, K)
        a = jax.random.categorical(kk, Q / TEMP)
        k = K[a]
        st2, r = step(c["st"], a, rt)
        o2 = patches(st2)
        K2 = keys(o2)
        if kind == "value":
            Q2 = q_of(c, K2)
            v2 = jnp.sum(jax.nn.softmax(Q2 / TEMP) * Q2)
            e = GAM * LAM * c["e"] + k
            delta = r + GAM * v2 - Q[a]
            new = dict(c, st=st2, w=c["w"] + BETA * delta * e, e=e)
        else:
            y = feat(o2)
            u = c["u"] + BETA * (r - c["u"] @ y) * y / (y @ y + 1e-8)               # 이 특징을 보면 보상이 오는가
            # 후속 특징: 직전 전이 (kp → yp)의 TD 오차를 지금의 키로 부트스트랩
            Psi = c["Psi"]
            dpsi = c["yp"] + GAM * Psi @ k - Psi @ c["kp"]
            Psi = Psi + BETA * jnp.outer(dpsi, c["e"]) / (c["e"] @ c["e"] + 1e-8)
            e = GAM * LAM * c["e"] + k
            new = dict(c, st=st2, Psi=Psi, u=u, e=e, kp=k, yp=y)
        return new, r
    c0 = dict(st=st, e=jnp.zeros(DK))
    if kind == "value":
        c0.update(w=jnp.zeros(DK))
    else:
        c0.update(Psi=jnp.zeros((HY, DK)), u=jnp.zeros(HY), kp=jnp.zeros(DK), yp=jnp.zeros(HY))
    _, r = jax.lax.scan(f, c0, (jax.random.split(key, T), rtypes))
    return r


# ---------- frozen: 시야만 보는 MLP 정책 (기억 없음), A2C ----------

def frozen_init(key):
    ks = jax.random.split(key, 5)
    d = lambda k, i, o, s=1.0: jax.random.normal(k, (i, o)) * s / jnp.sqrt(i)
    return dict(We=d(ks[0], 147, 16), W1=d(ks[1], V * V * 16, 128), b1=jnp.zeros(128), Pi=d(ks[2], 128, A, 0.01), pb=jnp.zeros(A),
                Vw=d(ks[3], 128, 1), vb=jnp.zeros(1))


def frozen_forward(p, o):
    h = jax.nn.relu(jax.nn.relu(o @ p["We"]).reshape(-1) @ p["W1"] + p["b1"])
    return h @ p["Pi"] + p["pb"], (h @ p["Vw"] + p["vb"])[0]


def frozen_rollout(p, key, N, T, rtypes):
    reset, patches, step = make_env(N)
    k0, key = jax.random.split(key)

    def f(st, x):
        kk, rt = x
        o = patches(st)
        logits, v = frozen_forward(p, o)
        a = jax.random.categorical(kk, jax.lax.stop_gradient(logits))
        logp = jax.nn.log_softmax(logits)
        st2, r = step(st, a, rt)
        return st2, dict(r=r, logp=logp[a], v=v, H=-jnp.sum(jnp.exp(logp) * logp), o2=patches(st2))
    _, out = jax.lax.scan(f, reset(k0), (jax.random.split(key, T), rtypes))
    return out


def frozen_loss(p, out, gamma=0.95, lam=0.95, ent=0.01):
    v = jax.lax.stop_gradient(out["v"])
    v_last = jax.lax.stop_gradient(frozen_forward(p, out["o2"][-1])[1])
    v_next = jnp.concatenate([v[1:], v_last[None]])
    d = out["r"] + gamma * v_next - v

    def back(acc, x):
        acc = x + gamma * lam * acc
        return acc, acc
    _, adv = jax.lax.scan(back, jnp.zeros(()), d, reverse=True)
    ret = adv + v
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-6)
    return (-(out["logp"] * adv_n) + 0.5 * (out["v"] - ret) ** 2 - ent * out["H"]).mean()
