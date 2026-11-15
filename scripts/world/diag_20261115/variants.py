# 진단: (s1) 도착 칸 척도 s = 1, (cat) 주소 = [흔적 e; 지금 키 k]. 난수 키, 칸별 정밀도 시냅스, C 도착 뒤 쓰기 없음 (12.4). 손 선호 판정 + 쓰기 직후 남는 몫.
import sys; sys.path.insert(0, '/workspace/RL-TDP')
import jax, jax.numpy as jnp, numpy as np
from rltdp.world import errand as ER, errand_learned as EL, errand_prec as EP, errand_reuse as RU
S1, CAT = bool(int(sys.argv[1])), bool(int(sys.argv[2])); SALL = len(sys.argv) > 3 and sys.argv[3] == '1'
ER.PUBLIC = len(sys.argv) > 4 and sys.argv[4] == '1'
We, ybar = EL.emb_fixed(0)
p, vmax = EP.train("random", seed=0, D=256, iters_k=0, log=lambda m: None)
feat, wm, key = EP.parts(p, We, ybar, "random")
phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
gB = phi[1] - phi.mean(0)
DZ = 2 * EP.DK if CAT else EP.DK
NW = 256
sessions, stack = EL.stack_sessions(range(1000, 1000 + NW), 256)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])

def scale(o, a):
    _, lv = wm(o, a)
    s = jnp.sqrt(jnp.exp(lv) / vmax)
    if SALL:
        return jnp.ones_like(s)
    return s.at[EP.SLOT].set(1.0) if S1 else s

def addr(e, k):
    return jnp.concatenate([e, k]) if CAT else e

def write(O, acts, stop_t):
    @jax.checkpoint
    def f(c, x):
        M, P, e = c
        o, a, o2, t = x
        k = key(o, a)
        e = EP.ALPHA * e + k
        mu, _ = wm(o, a)
        s = scale(o, a)
        r = feat(o2) - mu
        def pp(Mp, Pp, sp, rp):
            z = sp * addr(e, k)
            err = rp - Mp @ z; Pz = Pp @ z; S = EP.R_OBS + z @ Pz
            return Mp + jnp.outer(err, Pz) / S, Pp - jnp.outer(Pz, Pz) / S, (Mp + jnp.outer(err, Pz) / S) @ z @ rp / (rp @ rp + 1e-12)
        M2, P2, got = jax.vmap(pp)(M, P, s, r)
        keep = t > stop_t
        return (jnp.where(keep, M, M2), jnp.where(keep, P, P2), e), got[EP.SLOT]
    c0 = (jnp.zeros((EP.NP, EP.H, DZ)), jnp.tile(jnp.eye(DZ), (EP.NP, 1, 1)), jnp.zeros(EP.DK))
    return jax.lax.scan(f, c0, (O[:-1], acts, O[1:], jnp.arange(acts.shape[0])))

def one(sess, plans, ai_all):
    O, s_end, _ = EL.observe_stream(sess)
    t_c = RU.last_arrival(O)
    Y = jax.vmap(feat)(O[1:])[:, EP.SLOT]
    blank = feat(ER.TEX[jnp.full(12, ER.BLANK)].reshape(12, -1))[EP.SLOT]
    arrived = ~jnp.all(jnp.isclose(Y, blank, atol=1e-5), -1)
    (M, P, e), got = write(O, sess["acts"], t_c)
    F = jax.vmap(feat)(O[1:])
    Zu = jax.vmap(lambda a, c: jnp.outer(a, c).reshape(-1))(F[:, EP.SLOT], F[:, 9])
    _, Rw = EL.observe_stream(sess)[1:]
    def fu(u, x):
        z, r = x
        return u + 0.5 * (r - u @ z) * z / (z @ z + 1e-8), None
    u, _ = jax.lax.scan(fu, jnp.zeros(Zu.shape[1]), (Zu, Rw))
    got_mean = jnp.sum(jnp.where(arrived, got, 0)) / jnp.maximum(arrived.sum(), 1)       # 도착 스텝에서 쓰기 직후 자기 주소로 읽은 몫
    def sim(plan, ai):
        def f(c, a):
            s, o, e = c
            k = key(o, a)
            e = EP.ALPHA * e + k
            sv = scale(o, a)
            yh = M[EP.SLOT] @ (sv[EP.SLOT] * addr(e, k))
            mu, _ = wm(o, a)
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (s2, o2, e), (yh, mu[EP.SLOT], feat(o2)[9])
        _, (YH, MU, YC) = jax.lax.scan(f, (s_end, O[-1], e), plan)
        yh = YH[ai]
        pu = jnp.outer(yh + MU[ai], YC[ai]).reshape(-1) @ u
        return jnp.stack([(yh @ gB) / (jnp.linalg.norm(yh) * jnp.linalg.norm(gB) + 1e-8), pu]), jnp.linalg.norm(yh)
    sc, nrm = jax.vmap(sim)(plans, ai_all)
    return sc, nrm, got_mean
sc, nrm, got = jax.jit(jax.vmap(one))(stack, plans, arrive)
sc = np.array(sc)
for j, nm in enumerate(["hand B", "learned u"]):
    x = sc[:, :, j]
    best = (x + 1e-9 * np.random.default_rng(0).random(x.shape)).argmax(1)
    print("s_slot=1 %s, addr=[e;k] %s, s_all=1 %s, public %s | %-9s b1 %.3f b0 %.3f other %.3f | written %.2f | |read| %.3f" % (
        S1, CAT, SALL, ER.PUBLIC, nm, np.mean(best == b1), np.mean(best == b0), np.mean((best != b1) & (best != b0)), float(np.mean(got)), float(np.mean(nrm))), flush=True)
