"""지연 채집과 바뀌는 의뢰: 통합 시험의 축소 1판 (docs/world/integrated_experiment_design.md, README 12절).

환경 (JAX): N×N 격자 (N = 11, 바깥은 벽), 바닥 무늬 12종 (세계마다 무작위), 보급소 1곳, 채집처 5곳 (겉모습 같음).
  산출물 {A, A, B, B, C}를 세계마다 섞어 배정 (A, B, C = 산출물 그림 3종).
  관측: 자기 중심 3×3 세계 + HUD 3칸 (의뢰 카드, 도착한 산출물, 완료 도장) = 12패치 × 147.
  행동 6개: 동·서·남·북·상호작용·대기.
  채집처에서 상호작용 (작업 없을 때) → d스텝 뒤 관측에 산출물이 HUD에 도착 (출발지 표시 없음).
  보급소에서 상호작용 → 다음 의뢰 (이 판에서는 교사가 의뢰 종류를 정한다: 비공개 = 카드 뒷면).
  의뢰 산출물이 도착하면 +1, 완료 도장. 의뢰당 한 번.
한 세션 (교사 경로, 이후 자율):
  관찰 (의뢰 없이 5곳 1회씩) → A 의뢰 ×4 (같은 A 채집처) → B 의뢰: b0에서 첫 +1 → 새 B 의뢰 →
  보존 (보급소 주변 D스텝) 중 b0의 산출물을 B → C로 → b0 재방문 (C 도착, 보상 0) → [자율] B 의뢰 계속:
  첫 상호작용한 채집처가 b1 (관찰 때 B를 봤지만 보상받은 적 없는 곳)인가. 64스텝 안.
"""
import numpy as np
import jax
import jax.numpy as jnp

from . import env as E

N, D_DELAY, NSITE = 11, 8, 5
A_MOVES = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])       # 동 서 남 북
INTERACT, WAIT, NA = 4, 5, 6
FLOOR0, SITE, DEPOT, ITEM0, CARD_BACK, BLANK, STAMP = 1, 13, 15, 16, 20, 21, 22
TEX = E.textures(E.EnvCfg(N=N, V=3, n_types=24))


def gen_world(rng):
    """numpy: 지도, 보급소, 채집처, 산출물 배정, 역할 (a0, b0, b1, c0)."""
    while True:
        tiles = rng.integers(FLOOR0, FLOOR0 + 12, size=(N, N))
        tiles[0, :] = tiles[-1, :] = tiles[:, 0] = tiles[:, -1] = 0
        cells = [(i, j) for i in range(1, N - 1) for j in range(1, N - 1)]
        idx = rng.permutation(len(cells))
        inner = [k for k in idx if 2 <= cells[k][0] <= N - 3 and 2 <= cells[k][1] <= N - 3]
        depot = cells[inner[0]]
        idx = [k for k in idx if k != inner[0]]
        sites = []
        for k in idx:
            c = cells[k]
            if all(max(abs(c[0] - s[0]), abs(c[1] - s[1])) >= 3 for s in sites) and abs(c[0] - depot[0]) + abs(c[1] - depot[1]) >= 4:
                sites.append(c)
            if len(sites) == NSITE:
                break
        if len(sites) == NSITE:
            break
    prod = rng.permutation([0, 0, 1, 1, 2])                    # 0 = A, 1 = B, 2 = C
    a_sites = [i for i in range(NSITE) if prod[i] == 0]
    b_sites = [i for i in range(NSITE) if prod[i] == 1]
    roles = dict(a0=a_sites[0], b0=b_sites[0], b1=b_sites[1])
    for s in sites:
        tiles[s] = SITE
    tiles[depot] = DEPOT
    return tiles, np.array(depot), np.array(sites), prod, roles


def path(src, dst):
    """열린 격자의 최단 이동 (먼저 세로, 다음 가로)."""
    acts, p = [], np.array(src)
    while p[0] != dst[0]:
        a = 2 if dst[0] > p[0] else 3
        acts.append(a); p = p + A_MOVES[a]
    while p[1] != dst[1]:
        a = 0 if dst[1] > p[1] else 1
        acts.append(a); p = p + A_MOVES[a]
    return acts


def teacher(rng, depot, sites, roles, D):
    """교사 행동열과 의뢰·변경 시각표. 반환: acts, req_type[t] (−1 없음 / 0 A / 1 B), req_id[t] (보급소 상호작용마다 증가), change_t."""
    acts, req, rid = [], [], []
    cur = [-1, 0]

    def go(dst):
        nonlocal pos
        for a in path(pos, dst):
            acts.append(a); req.append(cur[0]); rid.append(cur[1])
        pos = np.array(dst)

    def interact(new_request=False):
        if new_request:
            cur[1] += 1
        acts.append(INTERACT); req.append(cur[0]); rid.append(cur[1])

    def errand(site):                                          # 채집처로 가서 작업 시작 → 보급소로 돌아오며 d스텝 기다림
        go(sites[site]); interact()
        back = path(pos, depot)
        t_start = len(acts)
        go(depot)
        while len(acts) - t_start < D_DELAY + 1:
            acts.append(WAIT); req.append(cur[0]); rid.append(cur[1])
    pos = np.array(depot)
    for s in rng.permutation(NSITE):                           # 관찰
        errand(int(s))
    cur[0] = 0
    for _ in range(4):                                         # A 의뢰 ×4
        interact(new_request=True)                             # 보급소에서 의뢰 받기
        errand(roles["a0"])
    cur[0] = 1
    interact(new_request=True); errand(roles["b0"])            # 새 보상: b0에서 첫 B
    interact(new_request=True)                                 # 새 B 의뢰
    change_t = len(acts) + D // 2
    for i in range(D):                                         # 보존: 보급소 주변 맴돌기 (보급소는 벽에서 2칸 이상 안쪽이라 막히지 않음)
        acts.append([0, 1, 2, 3][i % 4] if i % 8 < 4 else WAIT); req.append(cur[0]); rid.append(cur[1])
    errand(roles["b0"])                                        # 변경 증거: b0 → C, 보상 0 (같은 B 의뢰라 도장 없음)
    return np.array(acts), np.array(req), np.array(rid), change_t


def make_session(seed, D):
    rng = np.random.default_rng(seed)
    tiles, depot, sites, prod, roles = gen_world(rng)
    acts, req, rid, change_t = teacher(rng, depot, sites, roles, D)
    return dict(tiles=tiles, depot=depot, sites=sites, prod=prod, b0=roles["b0"], b1=roles["b1"], a0=roles["a0"],
                acts=acts, req=req, rid=rid, change_t=change_t)


# ---------- JAX 환경 ----------

def obs(tiles, pos, card, arrived, stamp):
    ids = jax.lax.dynamic_slice(jnp.pad(tiles, 1), (pos[0], pos[1]), (3, 3)).reshape(-1)
    hud = jnp.stack([card, arrived, stamp])
    return TEX[jnp.concatenate([ids, hud])].reshape(12, -1)


def env_step(s, a, req_now, rid_now, change_now, b0):
    """s: dict(tiles, pos, sites, prod, pending_item, due, t, stamp, req). 반환: s2, 관측 입력용 HUD, 보상."""
    move = jnp.array(A_MOVES)[jnp.minimum(a, 3)]
    new = s["pos"] + jnp.where(a < 4, move, 0)
    blocked = s["tiles"][new[0], new[1]] == 0
    pos = jnp.where(blocked, s["pos"], new)
    prod = jnp.where(change_now, s["prod"].at[b0].set(2), s["prod"])
    at_site = jnp.all(s["sites"] == pos, -1)
    start = (a == INTERACT) & at_site.any() & (s["pending"] < 0)
    item = jnp.where(start, prod[jnp.argmax(at_site)], s["pending"])
    due = jnp.where(start, s["t"] + D_DELAY, s["due"])
    t = s["t"] + 1
    arrive = (item >= 0) & (t == due)
    arrived_item = jnp.where(arrive, item, -1)
    new_req = req_now
    stamp = jnp.where(rid_now != s["rid"], 0, s["stamp"])                       # 새 의뢰면 도장 지움
    r = (arrive & (arrived_item == new_req) & (stamp == 0)).astype(jnp.float32)
    stamp = jnp.where(r > 0, 1, stamp)
    s2 = dict(s, pos=pos, prod=prod, pending=jnp.where(arrive, -1, item), due=due, t=t, stamp=stamp, req=new_req, rid=rid_now)
    card = jnp.where(new_req >= 0, CARD_BACK, BLANK)
    hud_item = jnp.where(arrived_item >= 0, ITEM0 + arrived_item, BLANK)
    hud_stamp = jnp.where(stamp > 0, STAMP, BLANK)
    return s2, obs(s["tiles"], pos, card, hud_item, hud_stamp), r, jnp.where(start, jnp.argmax(at_site), -1)


# ---------- 학습 없는 시냅스 에이전트 ----------

H_EMB, DK = 8, 128
GAM, LAM, BETA, TEMP = 0.9, 0.8, 0.5, 0.05


def fixed(seed=0):
    kr = jax.random.split(jax.random.PRNGKey(900 + seed), 2)
    We = jax.random.normal(kr[0], (147, H_EMB)) / jnp.sqrt(147.0)
    Wk = jax.random.normal(kr[1], (NA, DK, 12 * H_EMB)) / jnp.sqrt(12.0 * H_EMB)
    ybar = (TEX.reshape(TEX.shape[0], -1) @ We).mean(0)

    def keys(o):
        k = jnp.einsum("akh,h->ak", Wk, (o @ We).reshape(-1))
        return k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)

    def feat(o):
        return ((o @ We) - ybar).reshape(-1)
    return keys, feat


def run_session(kind, sess, key, auto_steps=64, seed=0):
    """kind: value (TD 신용할당) / fact (후속 특징 + 보상 가중치) / random. 반환: 자율 구간에서 처음 작업을 시작한 채집처 (−1 없음), 교사 구간 보상 합."""
    keys, feat = fixed(seed)
    HY = 12 * H_EMB
    s = dict(tiles=jnp.array(sess["tiles"]), pos=jnp.array(sess["depot"]), sites=jnp.array(sess["sites"]), prod=jnp.array(sess["prod"]),
             pending=jnp.array(-1), due=jnp.array(-1), t=jnp.array(0), stamp=jnp.array(0), req=jnp.array(-1), rid=jnp.array(0))
    o0 = obs(s["tiles"], s["pos"], BLANK, BLANK, BLANK)
    b0 = sess["b0"]
    Tt = sess["acts"].shape[0]
    acts = jnp.array(sess["acts"])
    req = jnp.array(sess["req"]); rid = jnp.array(sess["rid"])
    change = jnp.arange(Tt) == sess["change_t"]

    def q_of(m, K):
        if kind == "value":
            return K @ m["w"]
        return (K @ m["Psi"].T) @ m["u"]

    def update(m, k, o2, r, Q_next_fn):
        if kind == "value":
            Qn = Q_next_fn(m)
            v2 = jnp.sum(jax.nn.softmax(Qn / TEMP) * Qn)
            delta = r + GAM * v2 - k @ m["w"]
            e = GAM * LAM * m["e"] + k
            return dict(m, w=m["w"] + BETA * delta * e / (e @ e + 1e-8), e=e)      # 흔적 크기로 정규화 (HUD 때문에 키가 서로 닮아 정규화 없이는 발산)
        y = feat(o2)
        u = m["u"] + BETA * (r - m["u"] @ y) * y / (y @ y + 1e-8)
        Psi = m["Psi"]
        dpsi = m["yp"] + GAM * Psi @ k - Psi @ m["kp"]
        Psi = Psi + BETA * jnp.outer(dpsi, m["e"]) / (m["e"] @ m["e"] + 1e-8)
        e = GAM * LAM * m["e"] + k
        return dict(m, Psi=Psi, u=u, e=e, kp=k, yp=y)

    m0 = dict(e=jnp.zeros(DK))
    if kind == "value":
        m0.update(w=jnp.zeros(DK))
    else:
        m0.update(Psi=jnp.zeros((HY, DK)), u=jnp.zeros(HY), kp=jnp.zeros(DK), yp=jnp.zeros(HY))

    def teach(c, x):
        s, o, m = c
        a, rq, ri, ch = x
        k = keys(o)[a]
        s2, o2, r, _ = env_step(s, a, rq, ri, ch, b0)
        m = update(m, k, o2, r, lambda mm: q_of(mm, keys(o2)))
        return (s2, o2, m), r
    (s, o, m), rs = jax.lax.scan(teach, (s, o0, m0), (acts, req, rid, change))

    def auto(c, kk):
        s, o, m, first = c
        K = keys(o)
        Q = q_of(m, K)
        a = jax.random.categorical(kk, Q / TEMP) if kind != "random" else jax.random.randint(kk, (), 0, NA)
        s2, o2, r, started = env_step(s, a, jnp.array(1), rid[-1], jnp.array(False), b0)
        m = update(m, K[a], o2, r, lambda mm: q_of(mm, keys(o2))) if kind != "random" else m
        first = jnp.where((first < 0) & (started >= 0), started, first)
        return (s2, o2, m, first), None
    (_, _, _, first), _ = jax.lax.scan(auto, (s, o, m, jnp.array(-1)), jax.random.split(key, auto_steps))
    return first, rs.sum()
