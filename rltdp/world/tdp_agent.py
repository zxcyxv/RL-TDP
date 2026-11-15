"""TDP 블록을 뼈대로 한 에이전트 (JAX). rltdp/common/tdp_block.py (PyTorch)를 옮긴 것.

토큰 51개: 이미지 패치 49개 + 행동 토큰 + 보상 토큰. 토큰마다 같은 W_q, W_k, W_v로 질의·키·값을 만든다.
기억은 시냅스뿐이다: 토큰별 키 흔적 E_K와 값 흔적 E_V의 곱 M = Σ_r γ_r Re(c_r Σ_τ E_V conj(E_K)ᵀ).
흔적은 복소 EMA: E ← ρ exp(iθ) E + (1−ρ) 입력. 키와 값 입력은 유계다.
gate=True면 W ← Project(W + η(1[RPE>0] M − W G)), G는 PSD, trace ≤ 1, 0 ≤ η ≤ 1. 보상이 예상과 다를 때만, 그 순간의 흔적 내용이 W로 옮겨진다
W는 전체 감쇠 없이 head별 Frobenius norm ≤ D로 제한하며 에피소드마다 비운다.
스텝 사이에 넘어가는 것은 흔적과 W뿐이고, 블록의 출력은 넘기지 않는다.

한 스텝:
  1. 읽기: 지금 그림의 패치 49개, 후보 행동 a, "아직 모름" 보상 토큰을 넣고 행동 후보 4개에 대해 블록을 각각 돈다. 쓰지 않는다.
     각 후보에서 다음 그림 (패치 토큰), 다음 보상과 앞으로 받을 보상 (보상 토큰)을 예측한다.
  2. 앞으로 받을 보상이 큰 후보를 고르고 실제로 움직여 보상을 받는다.
  3. 쓰기: 같은 그림, 고른 행동, 방금 받은 보상을 토큰으로 넣고 블록을 한 번 더 돌아 흔적을 갱신한다.
     보상이 자기를 낳은 장면·행동과 같은 스텝에 쓰인다.
학습 (BPTT): 다음 그림, 다음 보상, 앞으로 받을 보상의 예측 오차. 정책을 직접 학습하는 신호는 없다.
"""
import math

import jax
import jax.numpy as jnp

from . import env as E
from .agents import dense, sg


def rms(x, eps=1e-5):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps)


def make_tdp(cfg: E.EnvCfg, mc=None, tex=None, d=128, heads=4, bands=4, periods=(2.0, 8.0, 32.0, 128.0), percept=1,
             gamma=0.9, lam=0.8, temp=0.05, delta=True, gate=True):
    H, D, R = heads, d // heads, bands
    n_patch, d_patch = cfg.V * cfg.V, cfg.P * cfg.P * 3
    T = n_patch + 2
    TOK_ACT, TOK_REW = n_patch, n_patch + 1
    inter = 2 * d

    def block_init(key):
        ks = jax.random.split(key, 8)
        p = jnp.array(periods)
        return dict(Wq=dense(ks[0], d, d), Wk=dense(ks[1], d, d), Wv=dense(ks[2], d, d), Wo=dense(ks[3], d, d),
                    Wo2=dense(ks[7], d, d),                                             # 오래 남는 시냅스에서 읽은 것의 전용 출력 사영
                    Wgu=dense(ks[4], d, 2 * inter), Wd=jnp.zeros((inter, d)),
                    rope=jax.random.uniform(ks[5], (H, D // 2, 2), minval=-math.pi / 2, maxval=math.pi / 2),
                    pole=jnp.tile(jnp.stack([-jnp.log(p), 2 * math.pi / p])[:, None], (1, H, 1)),      # [2, H, R]: ν, θ
                    c=jnp.stack([jnp.ones((H, R)), jnp.zeros((H, R))]) + jax.random.normal(ks[6], (2, H, R)) * 0.1)   # 쓴 것이 읽힐 때의 배율이 1 근처에서 출발

    def init(key):
        ks = jax.random.split(key, 9 + percept)
        return dict(percept=[block_init(k) for k in ks[9:]], mem=block_init(ks[8]),
                    Wp=dense(ks[0], d_patch, d), bp=jnp.zeros(d), act=jax.random.normal(ks[1], (cfg.A, d)) * 0.5,
                    wr=jax.random.normal(ks[2], (d,)) * 0.5, br=jax.random.normal(ks[3], (d,)) * 0.5,
                    rmask=jax.random.normal(ks[6], (d,)) * 0.5,                        # 보상 토큰: 아직 결과를 모를 때
                    log_kappa=jnp.zeros(()),                                           # 보상 예측 오차가 여는 문의 크기
                    Wout=dense(ks[4], d, d_patch), bout=jnp.zeros(d_patch), Whead=dense(ks[5], d, 2, 0.1), bhead=jnp.zeros(2))

    h0 = jax.random.normal(jax.random.PRNGKey(7), (T, d)) * 0.5                       # 고정 난수 (토큰의 자리 표시)
    xy = jnp.stack(jnp.meshgrid(jnp.arange(cfg.V), jnp.arange(cfg.V), indexing="ij"), -1).reshape(n_patch, 2).astype(jnp.float32)

    def rope(x, cos, sin):
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return jnp.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1).reshape(x.shape)

    def trace_coeff(p):
        nu, th = p["pole"]
        # Keep finite forgetting times; no global decay is applied to long memory.
        rate = jnp.clip(jnp.exp(jnp.clip(nu, -12.0, 4.0)), 1 / 512, 1.0)
        rho = jnp.exp(-rate)
        b = lambda x: x[:, None, :, None]
        return (b(rho * jnp.cos(th)), b(rho * jnp.sin(th)),
                b(-jnp.expm1(-rate)), b(jax.nn.softmax(p["c"][0], axis=-1)))

    def moments(kr, ki, vr, vi, weights):
        # A common denominator keeps write targets and the PSD erase operator consistent.
        cross = (jnp.einsum("htrd,htre->hde", weights * vr, kr)
                 + jnp.einsum("htrd,htre->hde", weights * vi, ki))
        gram = (jnp.einsum("htrd,htre->hde", weights * kr, kr)
                + jnp.einsum("htrd,htre->hde", weights * ki, ki))
        denom = jnp.maximum(jnp.trace(gram, axis1=-2, axis2=-1), 1.0)[:, None, None]
        return cross / denom, gram / denom

    def block(p, u, ek=None, ev=None, W=None):
        """u [T, d], ek / ev [H, T, 2, R, D] (실수부, 허수부). 반환: 블록 출력, 새 흔적, 새로 쓴 비율, 흔적으로 만든 시냅스.
        ek가 None이면 지각 블록: 같은 스텝의 토큰끼리만 섞고 시냅스는 없다.
        W [H, D, D]가 있으면 오래 남는 시냅스로 같이 읽는다."""
        heads_ = lambda x: x.reshape(T, H, D).transpose(1, 0, 2)
        u = rms(u)  # pre-normalize every projection, including the first perceptual block
        q, k, v = heads_(u @ p["Wq"]), heads_(u @ p["Wk"]), jnp.tanh(heads_(u @ p["Wv"]))
        unit = lambda x: x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-4)
        ang = jnp.einsum("hjc,tc->htj", p["rope"], xy)                                 # 패치는 2차원 위치로 회전, 특수 토큰은 0
        ang = jnp.concatenate([ang, jnp.zeros((H, 2, D // 2))], 1)
        cos, sin = jnp.cos(ang), jnp.sin(ang)
        qr, kr = rope(unit(q), cos, sin), rope(unit(k), cos, sin)
        read_w = jnp.einsum("htd,hte->hde", v, kr) / T                                 # 같은 스텝: Σ_τ v k̃ᵀ
        nov, m_short, erase, long = jnp.zeros(()), jnp.zeros((H, D, D)), jnp.zeros((H, D, D)), 0.0
        if ek is not None:
            zr, zi, gain, weights = trace_coeff(p)
            ek_re, ek_im, ev_re, ev_im = ek[:, :, 0], ek[:, :, 1], ev[:, :, 0], ev[:, :, 1]
            ek_re, ek_im = zr * ek_re - zi * ek_im, zr * ek_im + zi * ek_re
            ev_re, ev_im = zr * ev_re - zi * ev_im, zr * ev_im + zi * ev_re
            old, _ = moments(ek_re, ek_im, ev_re, ev_im, weights)
            pred = jnp.einsum("hde,hte->htd", old, kr)
            # Bounded innovations: rotation/decay plus (1-rho) input is contractive.
            w = jnp.tanh(v - pred) if delta else v
            nov = jnp.mean((v - pred) ** 2) / (jnp.mean(v * v) + 1e-8)
            ek_re = ek_re + gain * kr[:, :, None]
            ev_re = ev_re + gain * w[:, :, None]
            m_short, gram = moments(ek_re, ek_im, ev_re, ev_im, weights)
            read_w = read_w + m_short
            if W is not None:
                long = jnp.einsum("hte,hde->htd", qr, W).transpose(1, 0, 2).reshape(T, d) @ p["Wo2"]
                # gram is PSD with trace <= 1; I - eta*gram is non-expansive.
                erase = jnp.einsum("hde,hef->hdf", W, gram)
            ek, ev = jnp.stack([ek_re, ek_im], 2), jnp.stack([ev_re, ev_im], 2)
        read = jnp.einsum("hte,hde->htd", qr, read_w).transpose(1, 0, 2).reshape(T, d)
        h = rms(u + read @ p["Wo"] + long)
        gg, a = jnp.split(h @ p["Wgu"], 2, -1)
        return rms(h + (jax.nn.silu(gg) * a) @ p["Wd"]), ek, ev, nov, (m_short, erase)

    def carry0(key):
        st = E.reset(key, cfg)
        z = jnp.zeros((H, T, 2, R, D))
        return dict(st=st, o=E.patches(st, tex, cfg), r=jnp.zeros(()), ek=z, ev=z, W=jnp.zeros((H, D, D)))

    def forward(p, c, a, r=None, hidden=False):
        """장면 c["o"], 행동 a, 보상 토큰으로 블록을 한 번 돈다. 반환: 예측과 그때의 새 흔적.
        r=None이면 읽기 (행동을 고르기 전, 보상 토큰은 "아직 모름"), r이 주어지면 쓰기 (그 행동의 결과로 받은 보상).
        a가 행동 수와 같으면 p["act"]의 마지막 행 ("아직 안 고름")을 쓴다 (tdp_credit). hidden=True면 보상 토큰의 최종 출력을 덧붙인다."""
        rew = p["rmask"] if r is None else r * p["wr"] + p["br"]
        tok = jnp.concatenate([c["o"] @ p["Wp"] + p["bp"], p["act"][a][None], rew[None]])
        u = h0 + math.sqrt(d) * tok
        for bp in p["percept"]:                                                        # 지각 블록: 토큰끼리 섞는다 (기억 없음)
            u = block(bp, u)[0]
        u, ek, ev, nov, ms = block(p["mem"], u, c["ek"], c["ev"], c["W"] if gate else None)   # 기억 블록: 시냅스를 읽고 쓴다
        head = u[TOK_REW] @ p["Whead"] + p["bhead"]
        out = (u[:n_patch] @ p["Wout"] + p["bout"], head[0], head[1], ek, ev, nov, ms)
        return out + (u[TOK_REW],) if hidden else out

    def features(p, c, a):
        """진단용. 쓰기 직전 시냅스 M [H, D, D], 기억 블록에 들어가는 보상 토큰 벡터, 모델의 질의로 M을 읽은 값, 보상 토큰의 최종 출력."""
        tok = jnp.concatenate([c["o"] @ p["Wp"] + p["bp"], p["act"][a][None], p["rmask"][None]])
        u = h0 + math.sqrt(d) * tok
        for bp in p["percept"]:
            u = block(bp, u)[0]
        m = p["mem"]
        zr, zi, _, weights = trace_coeff(m)
        ek_re, ek_im, ev_re, ev_im = c["ek"][:, :, 0], c["ek"][:, :, 1], c["ev"][:, :, 0], c["ev"][:, :, 1]
        ek_re, ek_im = zr * ek_re - zi * ek_im, zr * ek_im + zi * ek_re
        ev_re, ev_im = zr * ev_re - zi * ev_im, zr * ev_im + zi * ev_re
        M, _ = moments(ek_re, ek_im, ev_re, ev_im, weights)
        u = rms(u)
        q = (u[TOK_REW] @ m["Wq"]).reshape(H, D)
        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-4)                    # 특수 토큰은 회전 없음
        M = M + c["W"] if gate else M
        out = block(m, u, c["ek"], c["ev"], c["W"] if gate else None)[0]
        return dict(M=M, u=u[TOK_REW], qread=jnp.einsum("he,hde->hd", q, M).reshape(-1), out=out[TOK_REW])

    def step(p, c, key):
        o_hat, r_hat, v_hat = jax.vmap(lambda a: forward(p, c, a)[:3])(jnp.arange(cfg.A))           # 읽기: 후보 넷 모두 (쓰지 않는다)
        a = jax.random.categorical(key, sg(v_hat) / temp)
        st2, r = E.step(c["st"], a, cfg)
        o2 = E.patches(st2, tex, cfg)
        _, _, _, ek, ev, nov, ms = forward(p, c, a, r)                                            # 쓰기: 그 장면, 그 행동, 그 결과 보상을 함께
        rpe = r - sg(r_hat[a])                                                                    # 보상 예측 오차 (예상보다 좋으면 +)
        # 오래 남는 시냅스의 델타 규칙. 예상보다 좋으면 지금의 흔적 내용을 옮겨 쓰고, 오차의 크기만큼 그 키들에 있던 옛 내용을 지운다.
        # 예상했는데 안 오면 지우기만 한다.
        ms, erase = ms
        eta = -jnp.expm1(-jax.nn.softplus(p["log_kappa"]) * jnp.abs(rpe))
        W_raw = c["W"] + eta * (jnp.where(rpe > 0, ms, 0.0) - erase)
        # Projected delta rule: radius D permits D orthogonal unit keys with
        # values of norm sqrt(D). Projection guarantees bounded state across training.
        wn = jnp.sqrt(jnp.sum(W_raw * W_raw, axis=(-2, -1), keepdims=True) + 1e-12)
        W = W_raw / jnp.maximum(wn / D, 1.0) if gate else jnp.zeros_like(W_raw)
        new = dict(st=st2, o=o2, r=r, ek=ek, ev=ev, W=W)
        return new, dict(r=r, v=v_hat[a], obs=jnp.mean((o_hat[a] - o2) ** 2), rew=(r_hat[a] - r) ** 2, nov=nov,
                         Wn=jnp.sqrt(jnp.mean(W * W)), Mn=jnp.sqrt(jnp.mean(ms * ms)),
                         eta=eta, Wclip=jnp.mean(wn > D), Vn=jnp.sqrt(jnp.mean(ev * ev)),
                         kappa=jax.nn.softplus(p["log_kappa"]))

    def loss(p, outs, carry):
        """outs의 축 [B, L]. 앞으로 받을 보상의 정답은 조각 안에서 뒤에서부터 계산한 λ-수익 (정답 쪽은 경사를 끊는다)."""
        v = sg(outs["v"])
        v_next = jnp.concatenate([v[:, 1:], v[:, -1:]], 1)

        def back(acc, x):
            r, vn = x
            acc = r + gamma * ((1 - lam) * vn + lam * acc)
            return acc, acc

        _, ret = jax.lax.scan(back, v[:, -1], (outs["r"].T, v_next.T), reverse=True)
        val = (outs["v"] - ret.T) ** 2
        return (outs["obs"] + outs["rew"] + val).mean()                                   # 그림 (픽셀 평균) : 보상 : 앞으로의 보상 = 1 : 1 : 1

    return dict(init=init, carry0=carry0, step=step, loss=loss, forward=forward, features=features)
