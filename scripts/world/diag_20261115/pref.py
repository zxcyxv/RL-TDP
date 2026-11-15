# errand: 세션 안 선호 학습 (델타 규칙). argv[1] = 0 비공개 / 1 공개 의뢰. 난수 키, 칸별 정밀도 시냅스 (12.4), 키 학습 없음.
import sys; sys.path.insert(0, '/workspace/RL-TDP')
import jax, jax.numpy as jnp, numpy as np
from rltdp.world import errand as ER
ER.PUBLIC = bool(int(sys.argv[1])); EVID = bool(int(sys.argv[2]))
from rltdp.world import errand_learned as EL, errand_prec as EP, errand_reuse as RU
We, ybar = EL.emb_fixed(0)
p, vmax = EP.train("random", seed=0, D=256, iters_k=0, log=lambda m: None)
feat, wm, key = EP.parts(p, We, ybar, "random")
phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
gB = phi[1] - phi.mean(0)
CARD = 9
NW = 256
sessions, stack = EL.stack_sessions(range(1000, 1000 + NW), 256)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])

def delta(Z, Rw):
    def f(u, x):
        z, r = x
        return u + 0.5 * (r - u @ z) * z / (z @ z + 1e-8), None
    return jax.lax.scan(f, jnp.zeros(Z.shape[1]), (Z, Rw))[0]

def one(sess, plans, ai_all):
    O, s_end, Rw = EL.observe_stream(sess)
    t_c = RU.last_arrival(O)
    (M, P, e, _, _), _ = EP.write_scan(p, We, ybar, "random", vmax, O, sess["acts"], EP.c_init(), evidence=EVID, stop_t=t_c)
    F = jax.vmap(feat)(O[1:])
    ya, yc = F[:, EP.SLOT], F[:, CARD]
    u1 = delta(ya, Rw)                                                  # 도착 산출물만
    u2 = delta(jax.vmap(lambda a, c: jnp.outer(a, c).reshape(-1))(ya, yc), Rw)   # 도착 × 카드
    def sim(plan, ai):
        def f(c, a):
            s, o, e = c
            e = EP.ALPHA * e + key(o, a)
            mu, lv = wm(o, a)
            sv = jnp.sqrt(jnp.exp(lv) / vmax)
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (s2, o2, e), (M[EP.SLOT] @ (sv[EP.SLOT] * e), mu[EP.SLOT], feat(o2)[CARD])
        _, (YH, MU, YC) = jax.lax.scan(f, (s_end, O[-1], e), plan)
        yh = YH[ai] + MU[ai]
        return jnp.stack([(YH[ai] @ gB) / (jnp.linalg.norm(YH[ai]) * jnp.linalg.norm(gB) + 1e-8),
                          yh @ u1, jnp.outer(yh, YC[ai]).reshape(-1) @ u2])
    cardB = feat(ER.TEX[jnp.full(12, ER.ITEM0 + 1 if ER.PUBLIC else ER.CARD_BACK)].reshape(12, -1))[CARD]
    pref2 = jax.vmap(lambda x: jnp.outer(x, cardB).reshape(-1) @ u2)(phi)
    return jax.vmap(sim)(plans, ai_all), jnp.stack([u1 @ phi.T, pref2])
sc, up = jax.jit(jax.vmap(one))(stack, plans, arrive)
sc, up = np.array(sc), np.array(up)
rng = np.random.default_rng(0)
print("PUBLIC", ER.PUBLIC, "EVIDENCE", EVID)
for j, name in enumerate(["hand gB", "u(arrival)", "u(arrival x card)"]):
    s = sc[:, :, j]; best = (s + 1e-9 * rng.random(s.shape)).argmax(1)
    print(" %-18s b1 %.3f b0 %.3f other %.3f" % (name, np.mean(best == b1), np.mean(best == b0), np.mean((best != b1) & (best != b0))))
print(" preference of A,B,C (arrival / arrival x card-B):", up.mean(0).round(2).tolist())
