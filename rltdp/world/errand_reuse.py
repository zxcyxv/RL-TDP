"""지연 채집 과제: 키 학습의 F를 '기억을 다시 읽을 때'까지 넓힌다 (architecture_proposal.md 4절, README 12.5절).

결합 (월드 모델, 학습된 키, 흔적, 칸별 정밀도 가중 쓰기, 판정)은 errand_prec 그대로. 바꾸는 것은 키 학습의 목적과 경사 연결 하나:
  학습용 세계마다 교사 경험으로 시냅스에 쓴다 (C 도착 뒤로는 쓰지 않음, 판정과 같은 조건).
  그 기억을 고정한 채 채집처 5곳을 판정과 같은 형식 (보급소 → 채집처, 상호작용, 돌아오며 대기)으로 다시 질의하고,
  질의 경로의 모든 관측을 칸별 정밀도 가중 음의 로그우도로 예측한 값이 손실 (S_p = R + s_p² eᵀ P_p e).
  경사는 처음 쓴 순간부터 다시 읽는 순간까지 끊지 않는다 (스텝마다 재계산으로 메모리 절약).
  b0·b1 역할, 정답표, 작업 시작·도착 신호는 주지 않는다. 학습 세계와 시험 세계는 분리.
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax

from . import errand as ER
from . import errand_learned as EL
from . import errand_prec as EP

H, NP = EP.H, EP.NP


def last_arrival(O):
    blank = ER.TEX[ER.BLANK].reshape(-1)
    arrived = ~jnp.all(jnp.isclose(O[1:, EP.SLOT], blank), -1)
    return jnp.max(jnp.where(arrived, jnp.arange(arrived.shape[0]), -1))


def write_scan_ckpt(p, We, ybar, kind, vmax, O, acts, stop_t):
    """errand_prec.write_scan과 같은 쓰기, 스텝마다 재계산 (경사를 세션 전체로)."""
    feat, wm, key = EP.parts(p, We, ybar, kind)
    T = acts.shape[0]

    @jax.checkpoint
    def f(c, x):
        M, P, e = c
        o, a, o2, t = x
        e = EP.ALPHA * e + key(o, a)
        mu, lv = wm(o, a)
        s = jnp.sqrt(jnp.exp(lv) / vmax)
        r = feat(o2) - mu

        def per_patch(Mp, Pp, sp, rp):
            z = sp * e
            err = rp - Mp @ z
            Pz = Pp @ z
            S = EP.R_OBS + z @ Pz
            return Mp + jnp.outer(err, Pz) / S, Pp - jnp.outer(Pz, Pz) / S
        M2, P2 = jax.vmap(per_patch)(M, P, s, r)
        keep = t > stop_t
        return (jnp.where(keep, M, M2), jnp.where(keep, P, P2), e), None
    c0 = EP.c_init()[:3]
    (M, P, e), _ = jax.lax.scan(f, c0, (O[:-1], acts, O[1:], jnp.arange(T)))
    return M, P, e


def reuse_nll(p, We, ybar, kind, vmax, sess, plans):
    """세션 하나: 쓰고, 고정한 기억으로 채집처마다 다시 질의한 경로의 관측 예측 NLL (경로 스텝 평균)."""
    feat, wm, key = EP.parts(p, We, ybar, kind)
    O, s_end, _ = EL.observe_stream(sess)
    M, P, e = write_scan_ckpt(p, We, ybar, kind, vmax, O, sess["acts"], last_arrival(O))

    def query(plan):
        def f(c, a):
            s, o, e = c
            e = EP.ALPHA * e + key(o, a)
            mu, lv = wm(o, a)
            sv = jnp.sqrt(jnp.exp(lv) / vmax)
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            y = feat(o2) - mu

            def per_patch(Mp, Pp, sp, yp):
                z = sp * e
                S = EP.R_OBS + z @ Pp @ z
                err = yp - Mp @ z
                return 0.5 * (err @ err / S + H * jnp.log(S))
            return (s2, o2, e), jax.vmap(per_patch)(M, P, sv, y).sum()
        _, nll = jax.lax.scan(f, (s_end, O[-1], e), plan)
        return nll.mean()
    return jax.vmap(query)(plans).mean()


def train(seed=0, D=256, n_train=256, iters_k=300, B=8, log=print):
    We, ybar = EL.emb_fixed(seed)
    p, vmax = EP.train("learned", seed=seed, D=D, iters_k=0, log=log)                # 월드 모델만 (칸별 NLL)
    sessions, tr = EL.stack_sessions(range(50_000, 50_000 + n_train), D)
    plans, _ = EL.plans_for(sessions)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    q = {k: p[k] for k in ("Ku", "Kd")}
    st = opt.init(q)

    @jax.jit
    def upd(q, st, idx):
        def lf(q):
            pp = dict(p, **q)
            sb = jax.tree_util.tree_map(lambda x: x[idx], tr)
            return jax.vmap(lambda s, pl: reuse_nll(pp, We, ybar, "learned", vmax, s, pl))(sb, plans[idx]).mean()
        l, g = jax.value_and_grad(lf)(q)
        g, st = opt.update(g, st, q)
        return optax.apply_updates(q, g), st, l
    key = jax.random.PRNGKey(11)
    for it in range(iters_k):
        key, k = jax.random.split(key)
        q, st, l = upd(q, st, jax.random.randint(k, (B,), 0, n_train))
        if it % 50 == 0 or it == iters_k - 1:
            log(f"  [keys, reuse F] it {it} nll {float(l):.2f}")
    p.update(q)
    return p, vmax


def probe_two(p, vmax, kind, seed=0):
    """C를 쓰기 직전과 직후의 기억으로 같은 판정 점수 (채집처 5곳). 반환 [2, NSITE]."""
    We, ybar = EL.emb_fixed(seed)
    feat, wm, key = EP.parts(p, We, ybar, kind)
    phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
    gB = phi[1] - phi.mean(0)

    def one(sess, plans, arrive_idx):
        O, s_end, _ = EL.observe_stream(sess)
        t_c = last_arrival(O)
        out = []
        for stop in (t_c - 1, t_c):
            M, P, e = write_scan_ckpt(p, We, ybar, kind, vmax, O, sess["acts"], stop)

            def sim(plan, ai):
                def f(c, a):
                    s, o, e = c
                    e = EP.ALPHA * e + key(o, a)
                    _, lv = wm(o, a)
                    sv = jnp.sqrt(jnp.exp(lv) / vmax)
                    yh = M[EP.SLOT] @ (sv[EP.SLOT] * e)
                    s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
                    return (s2, o2, e), yh
                _, YH = jax.lax.scan(f, (s_end, O[-1], e), plan)
                yh = YH[ai]
                return (yh @ gB) / (jnp.linalg.norm(yh) * jnp.linalg.norm(gB) + 1e-8)
            out.append(jax.vmap(sim)(plans, arrive_idx))
        return jnp.stack(out)
    return one
