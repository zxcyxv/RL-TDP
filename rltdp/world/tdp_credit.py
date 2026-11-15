"""TDP 블록 + 공 배정 기억 (JAX). 설계 근거: docs/world/README.md 5절.

TDP 블록 (tdp_agent.make_tdp, 장기 W 없음)은 장면 표현과 작업 기억 (단기 STDP 흔적)만 맡는다. 보상 기억은 아래 세 장치가 맡는다.

1. 키 = 결정 시점에 있는 정보 (상황 × 행동), 결과는 값에만.
   행동 토큰을 "아직 안 고름"으로, 보상 토큰을 "아직 모름"으로 두고 블록을 한 번 돌아 장면 벡터 h(s)를 얻는다 (흔적은 쓰지 않는다).
   k(s, a) = unit(Wk[a] h(s) + bk[a]). 행동마다 사영이 달라 상황과 행동이 곱으로 묶인다.
2. 공은 결과보다 먼저 온 것에만, 짧은 창으로: 키만 담는 인과적 흔적 e ← γλ e + k(s_t, a_t). 보상 뒤 (출발 칸으로 돌아감) 끊는다.
3. 공 배정 창 (γλ)과 보존 (에피소드 동안 감쇠 없음)을 나눈다. 장기 기억 w에는 공을 받은 키만, 놀라움 δ (TD 오차)의 크기로 쓴다:
   δ = r + γ V(s') − Q(s, a),  w ← w + β δ e,  Q(s, a) = w · k(s, a),  V(s) = Σ_a softmax(Q/temp) Q.
   w는 에피소드마다 0에서 시작한다. 첫 보상 전에는 Q가 모두 0이라 행동은 균등하다.

학습 (BPTT, tdp_agent와 같은 손실): 다음 그림 + 즉시 보상 + 앞으로의 보상 (λ-수익). 앞으로의 보상의 예측값은 기억에서 읽은 Q다.
경사는 w의 쓰기를 거쳐 키 사영과 블록으로 흐른다 (δ와 목표는 경사를 끊는다). 느린 가중치에서 바로 나오는 가치 머리는 행동에 쓰지 않는다.

δ는 한 스텝 늦게 계산한다: 스텝 t의 장면을 읽을 때 V(s_t)가 나오므로, 그때 직전 전이 (s_{t−1}, a_{t−1})의 δ로 w를 고친 뒤 행동을 고른다.
"""
import jax
import jax.numpy as jnp

from . import env as E
from .agents import dense, sg
from .tdp_agent import make_tdp


def make_tdp_credit(cfg: E.EnvCfg, mc=None, tex=None, d=128, dk=128, gamma=0.9, lam=0.8, temp=0.05):
    base = make_tdp(cfg, mc, tex, d=d, gamma=gamma, lam=lam, temp=temp, gate=False)
    fwd, A = base["forward"], cfg.A
    unit = lambda x: x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)

    def init(key):
        k1, k2, k3 = jax.random.split(key, 3)
        p = base["init"](k1)
        p["act"] = jnp.concatenate([p["act"], jax.random.normal(k2, (1, d)) * 0.5])      # 마지막 행: 아직 안 고름
        p.update(Wk=jax.random.normal(k3, (A, dk, d)) / jnp.sqrt(d), bk=jnp.zeros((A, dk)), logit_beta=jnp.zeros(()))
        return p

    def carry0(key):
        c = base["carry0"](key)
        return dict(c, w=jnp.zeros(dk), e=jnp.zeros(dk), q=jnp.zeros(()))

    def step(p, c, key):
        h = fwd(p, c, A, hidden=True)[-1]                                        # 장면만 읽기 (쓰지 않는다)
        K = unit(jnp.einsum("akh,h->ak", p["Wk"], h) + p["bk"])                  # [A, dk]
        Q_old = K @ c["w"]
        V = jnp.sum(jax.nn.softmax(sg(Q_old) / temp) * Q_old)
        delta = sg(c["r"] + gamma * (1 - c["r"]) * V - c["q"])                   # 직전 전이의 TD 오차. 보상 뒤에는 부트스트랩하지 않는다
        w = c["w"] + jax.nn.sigmoid(p["logit_beta"]) * delta * c["e"]
        Q = K @ w
        a = jax.random.categorical(key, sg(Q) / temp)
        e = gamma * lam * (1 - c["r"]) * c["e"] + K[a]                           # 키 선행 흔적, 보상 뒤 끊김
        o_hat, r_hat = fwd(p, c, a)[:2]                                          # 표현 학습용 예측 (고른 행동)
        st2, r = E.step(c["st"], a, cfg)
        o2 = E.patches(st2, tex, cfg)
        _, _, _, ek, ev, nov, _ = fwd(p, c, a, r)                                # 작업 기억 쓰기: 그 장면, 그 행동, 그 보상
        new = dict(c, st=st2, o=o2, r=r, ek=ek, ev=ev, w=w, e=e, q=Q[a])
        return new, dict(r=r, v=Q[a], obs=jnp.mean((o_hat - o2) ** 2), rew=(r_hat - r) ** 2, nov=nov,
                         td=delta ** 2, wn=jnp.linalg.norm(w), beta=jax.nn.sigmoid(p["logit_beta"]))

    return dict(init=init, carry0=carry0, step=step, loss=base["loss"])
