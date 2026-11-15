# 도착 칸: B가 도착한 스텝의 주소 z_t로, 그 뒤 k스텝 쓰기를 더 한 기억에서 다시 읽으면 B 몫이 얼마 남나. 이웃 스텝 흔적 코사인.
import sys; sys.path.insert(0, '/workspace/RL-TDP')
import jax, jax.numpy as jnp, numpy as np
from rltdp.world import errand as ER, errand_learned as EL, errand_prec as EP
We, ybar = EL.emb_fixed(0)
p, vmax = EP.train("random", seed=0, D=256, iters_k=0, log=lambda m: None)
feat, wm, key = EP.parts(p, We, ybar, "random")
phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
NW = 32
sessions, stack = EL.stack_sessions(range(1000, 1000 + NW), 256)
K = 12
def one(sess):
    O, _, Rw = EL.observe_stream(sess)
    T = sess["acts"].shape[0]
    def f(c, x):
        M, P, e = c
        o, a, o2 = x
        e = EP.ALPHA * e + key(o, a)
        mu, lv = wm(o, a)
        s = jnp.sqrt(jnp.exp(lv) / vmax)
        z = s[EP.SLOT] * e
        r = feat(o2)[EP.SLOT] - mu[EP.SLOT]
        Mp, Pp = M, P
        err = r - Mp @ z; Pz = Pp @ z; S = EP.R_OBS + z @ Pz
        M2, P2 = Mp + jnp.outer(err, Pz) / S, Pp - jnp.outer(Pz, Pz) / S
        return (M2, P2, e), (M2, z, e, r)
    _, (Ms, Zs, Es, Rs) = jax.lax.scan(f, (jnp.zeros((EP.H, EP.DK)), jnp.eye(EP.DK), jnp.zeros(EP.DK)), (O[:-1], sess["acts"], O[1:]))
    blank = feat(ER.TEX[jnp.full(12, ER.BLANK)].reshape(12, -1))[EP.SLOT]
    isB = jnp.all(jnp.isclose(feat(O[1:].reshape(-1, 12, O.shape[-1])[:, :, :] if False else O[1:])[:, EP.SLOT] if False else jax.vmap(feat)(O[1:])[:, EP.SLOT], phi[1], atol=1e-4), -1)
    tB = jnp.argmax(isB)                                           # 첫 B 도착 (관찰 국면)
    reads = jnp.stack([Ms[tB + k] @ Zs[tB] for k in range(K)])     # [K, 8]
    target = Rs[tB]
    frac = reads @ target / (target @ target)
    cosn = (Es[tB] @ Es[tB + 1]) / (jnp.linalg.norm(Es[tB]) * jnp.linalg.norm(Es[tB + 1]))
    return frac, cosn
frac, cosn = jax.jit(jax.vmap(one))(stack)
print("fraction of B arrival error recalled at its own address after k more writes (k=0..11):")
print(np.array(frac).mean(0).round(2))
print("cos(trace_t, trace_t+1) at arrival: %.3f" % float(np.array(cosn).mean()))
