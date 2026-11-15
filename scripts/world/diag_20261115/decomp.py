# b0, b1 자리에서 읽은 도착 예측을 A, B, C 산출물 특징으로 분해 (C 쓰기 전/후). 난수 키, 칸별 정밀도 시냅스 (12.4).
import sys; sys.path.insert(0, '/workspace/RL-TDP')
import jax, jax.numpy as jnp, numpy as np
from rltdp.world import errand as ER, errand_learned as EL, errand_prec as EP, errand_reuse as RU
We, ybar = EL.emb_fixed(0)
p, vmax = EP.train("random", seed=0, D=256, iters_k=0, log=lambda m: None)
feat, wm, key = EP.parts(p, We, ybar, "random")
phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
NW = 256
sessions, stack = EL.stack_sessions(range(1000, 1000 + NW), 256)
plans, arrive = EL.plans_for(sessions)
def one(sess, plans, ai_all):
    O, s_end, _ = EL.observe_stream(sess)
    t_c = RU.last_arrival(O)
    out = []
    for stop in (t_c - 1, t_c):
        M, P, e = RU.write_scan_ckpt(p, We, ybar, "random", vmax, O, sess["acts"], stop)
        def sim(plan, ai):
            def f(c, a):
                s, o, e = c
                e = EP.ALPHA * e + key(o, a)
                mu, lv = wm(o, a)
                sv = jnp.sqrt(jnp.exp(lv) / vmax)
                s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
                return (s2, o2, e), (M[EP.SLOT] @ (sv[EP.SLOT] * e), mu[EP.SLOT])
            _, (YH, MU) = jax.lax.scan(f, (s_end, O[-1], e), plan)
            return YH[ai], MU[ai]
        out.append(jax.vmap(sim)(plans, ai_all))
    return out
res = jax.jit(jax.vmap(one))(stack, plans, arrive)
b0, b1 = np.array(stack["b0"]), np.array(stack["b1"])
i = np.arange(NW)
blank = (np.array(ER.TEX[ER.BLANK]).reshape(-1) @ np.array(We)) - np.array(ybar)
X = np.vstack([np.array(phi), blank]).T             # [8, 4] A B C blank
for j, when in enumerate(["before C", "after C"]):
    YH, MU = np.array(res[j][0]), np.array(res[j][1])
    for name, idx in (("b0", b0), ("b1", b1)):
        y = YH[i, idx]                                  # 시냅스 몫
        coef = np.linalg.lstsq(X, y.T, rcond=None)[0].T
        resid = np.linalg.norm(y - coef @ X.T, axis=1) / np.linalg.norm(y, axis=1)
        print(when, name, "coef A/B/C/blank", coef.mean(0).round(2), "resid %.2f" % resid.mean(), "|syn| %.3f |wm| %.3f" % (np.linalg.norm(YH[i, idx], axis=1).mean(), np.linalg.norm(MU[i, idx], axis=1).mean()))
