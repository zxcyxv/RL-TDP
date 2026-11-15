"""갈림길 과제: 같은 기억에서 목표만 바꿔 묻기 A → B → A (docs/world/README.md 9절, insession_design.md 10.2절).

1. 두 문을 한 번씩 노출 (원래 과제와 같음, 대기 없음).
2. 둘 중 자원 A를 목표로 (HUD에 A), A가 있는 문으로 8번 실행, A를 볼 때마다 보상 +1.
3. 검사 위치 (갈림길)로 돌아온다.
4. 그 기억 상태에서 쓰기 없이 순수 읽기로 목표 g = A, B, A를 차례로 묻는다 (B = 같은 세계에서 본 다른 자원).
기억:
  fact   gate_learned의 learned_wm (학습된 키 + 학습된 월드 모델의 정밀도 가중 쓰기). 읽기: cos(ŷ(a), φ_g − μ̄).
  value  정규화된 난수 키 (1단계 random) 위의 TD(λ) 가치 캐시 (SARSA, γ = 0.9, λ = 0.8, β = 0.5, 보상 뒤 흔적·부트스트랩을 끊음 = tdpc와 같은 규칙).
         학습된 키를 쓰면 (정규화해도) 스텝끼리 방향이 닮아 흔적이 커지고 TD 걸음이 진동해 부호가 뒤집힌다 (README 9절). 새 목표에 못 맞추는 것은 규칙의 성질이라 키와 무관하다.
         읽기: argmax_a Q(갈림길(HUD = g), a). 목표는 HUD를 통해서만 들어간다.
"""
import jax
import jax.numpy as jnp

from . import gate as G
from . import gate_learned as GL
from . import gate_memory as GM

N_RUNS = 8
unit = lambda k: k / (jnp.linalg.norm(k) + 1e-8)


def goal_world(key, d):
    w = G.make_world(key, d, 0)
    types = jnp.stack([w.ids[d, 3, 3], w.ids[2 * d + 1, 3, 3]])          # 노출 순서대로 본 두 자원
    first_left = w.act[0] == G.LEFT
    pick = jax.random.randint(jax.random.fold_in(key, 3), (), 0, 2)        # A = 노출 순서상 몇 번째 자원인가
    A, B = types[pick], types[1 - pick]
    doorA_left = (pick == 0) == first_left
    actA = jnp.where(doorA_left, G.LEFT, G.RIGHT)
    run_ids = [G.scene(G.BG["junc"], G.JUNC, A)] + [G.scene(G.BG[f"corr{j}"], G.CORR, G.HUD_BLANK) for j in range(d - 1)] \
        + [G.scene(G.BG[f"corr{d - 1}"], A, G.HUD_BLANK)]
    run_act = [actA] + [jnp.array(G.FWD)] * d                                # 자원 장면에서 FWD → 다음 실행의 갈림길
    ids = [*w.ids[:-1]] + run_ids * N_RUNS + [G.scene(G.BG["junc"], G.JUNC, G.HUD_BLANK)]
    act = [*w.act[:-1], jnp.array(G.FWD)] + run_act * N_RUNS
    act = act[: len(ids) - 1]
    rew = jnp.zeros(len(ids) - 1)
    for i in range(N_RUNS):                                                  # A가 보이는 전이에 보상
        t = len(w.ids) - 1 + i * (d + 1) + d - 1
        rew = rew.at[t].set(1.0)
    return jnp.stack(ids), jnp.stack(act), rew, A, B, actA


def value_cache(p, kf, ids, act, rew, gamma=0.9, lam=0.8, beta=0.5):
    K = jax.vmap(lambda o, a: unit(kf(p, o, a)))(ids[:-1], act)
    Q = lambda w, k: w @ k

    def f(c, x):
        w, e = c
        k, k2, r = x
        e = gamma * lam * e + k
        delta = r + gamma * (1 - r) * Q(w, k2) - Q(w, k)
        w = w + beta * delta * e
        e = e * (1 - r)
        return (w, e), None
    (w, _), _ = jax.lax.scan(f, (jnp.zeros(K.shape[1]), jnp.zeros(K.shape[1])), (K[:-1], K[1:], rew[:-1]))
    return w


def evaluate(p, enc, kf, seed, d=4, n=1024):
    kr = GL.keys_fn("random", enc)                                           # 가치 캐시용 키
    keys = jax.random.split(jax.random.PRNGKey(40_000 + seed), n)
    corr = [G.scene(G.BG[f"corr{j}"], G.CORR, G.HUD_BLANK) for j in range(d - 1)]
    mu_end, lv_end = GL.wm_apply(p["wm"], enc, corr[-1], G.FWD)
    s_star = jnp.sqrt(jnp.exp(lv_end) / p["vmax"])

    def one(key):
        ids, act, rew, A, B, actA = goal_world(key, d)
        mu, s = GL.wm_stats(p["wm"], p["vmax"], enc, ids, act)
        (M, P, e), _ = GL.run(p, kf, enc, ids, act, mu, s, jnp.ones(act.shape[0]))
        w = value_cache(p, kr, ids, act, rew)
        out = []
        for g in (A, B, A):
            J = G.scene(G.BG["junc"], G.JUNC, g)
            def walk(a):
                ee = GM.ALPHA * e + kf(p, J, a)
                for c in corr:
                    ee = GM.ALPHA * ee + kf(p, c, G.FWD)
                return s_star * ee
            Z = jnp.stack([walk(G.LEFT), walk(G.RIGHT)])
            a_fact, _ = GM.choose(M, Z, GM.emb(enc, J)[G.HUD], mu_end)
            qv = jnp.stack([w @ unit(kr(p, J, G.LEFT)), w @ unit(kr(p, J, G.RIGHT))])
            a_val = jnp.where(qv[0] >= qv[1], G.LEFT, G.RIGHT)
            right = jnp.where(g == A, actA, jnp.where(actA == G.LEFT, G.RIGHT, G.LEFT))
            out.append(jnp.stack([a_fact == right, a_val == right]))
        return jnp.stack(out)                                               # [3 질의, 2 기억]
    ok = jax.lax.map(one, keys, batch_size=64).mean(0)
    S = 2 * ok - 1
    return dict(fact=[round(float(x), 3) for x in S[:, 0]], value=[round(float(x), 3) for x in S[:, 1]])
