"""1단계: errand에서 실제로 b1 고르기 (docs/world/spec_memory_wm_agent.md 3절, 4절). 2026-11-19.

환경과 교사 구간은 rltdp/world/errand.py 그대로 (관찰 5곳 → A 의뢰 ×4 → b0에서 첫 B +1 → 보존 D스텝 중 b0: B → C → b0 재방문 C, 보상 0).
자율: 후보 경로 5개 (관찰 구간의 교사 방문 5번, 보급소 → 채집처 → 상호작용 → 귀환·대기) 중 하나를 고르면
공통 실행기가 그 경로의 교사 행동열을 실제 환경에서 실행한다 (64스텝 예산). 첫 상호작용 채집처와 실제 보상을 잰다.

느린 부분 (학습 세계에서 학습, 평가 때 고정):
  인코더 h^o = GRU(장면 9칸, 도착 산출물 ξ, 이전 행동), h^r = GRU(관측 12칸, 이전 행동, 보상). 결과 경로와 키는 h^o, 보상 경로는 h^r.
  ψ_θ(h^o, 경로) ≈ Y (할인 결과), u_θ(h^r) (r ≈ uᵀξ), v_θ(h^r, 경로) ≈ G (가치 캐시 대조용 할인 보상).
  경로 기술 = 보급소 출발부터 작업 시작 직전까지의 장면 관측·행동 (산출물·보상 없음). 키 k(경로) = MLP(경로 기술).
  보상 문맥 b_θ(h^r). 칼만 매개변수 (p0, R, Q)는 기억마다. 키·b·칼만 매개변수는 쓰기 직전 혁신의 음의 로그우도로 맞춘다.
빠른 기억 (세션 안, 역전파 없음, 칼만 쓰기, P⁻ = P + Q·(경과 스텝)·I, Joseph 꼴 갱신):
  결과 M_Ψ: 작업이 끝날 때 d = k(경로), y = Y − ψ_θ(h^o_시작, 경로), Y = Σ_j γ^j ξ_{시작+j+1}.
  보상 Δ_u: 산출물이 도착한 스텝마다 d = b_θ(h^r_t) ⊗ ξ_{t+1}, y = r_{t+1} − u_θ(h^r_t)ᵀ ξ_{t+1}.
  가치 캐시 (대조): d = k(경로), y = G − v_θ, G = Σ_j γ^j r_{시작+j+1}.
행동: Q(경로) = ûᵀ Ψ̂(경로), Ψ̂ = ψ_θ + M_Ψ k, û = u_θ + b_θᵀ Δ_u. 최대 (동점은 무작위).
조건: full / sf_off (M_Ψ 안 씀) / q0 (Q = 0) / value (가치 캐시) / forced_b1 (양성 대조) / random.
판정 (실행 전 고정): 주 지표 = 64스텝 안 첫 상호작용이 b1인 비율. full − sf_off 짝 차이의 95% 신뢰구간 (세계 재표집)이 0보다 크고,
b1을 고른 세계에서 실제 +1이 관측되면 통과.
"""
import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

from . import errand as ER
from . import errand_learned as EL

jax.config.update("jax_default_matmul_precision", "highest")

GAMMA = 0.9
DK, HS, DX, DB = 128, 64, EL.H, 16
SLOT = EL.SLOT
NE = 11                      # 교사 구간의 작업 수 (관찰 5 + A 4 + B 1 + 재방문 1)
AUTO = 64
R_FLOOR = 1e-6
DR = 9 * DX + ER.NA          # 경로 기술의 한 스텝 (장면 9칸 + 행동)
DRIN = DR + 9 * DX + 1       # 경로 기술 = [평균, 채집처에서의 장면, 길이]

We, _ = EL.emb_fixed(0)
E_BLANK = ER.TEX[ER.BLANK].reshape(-1) @ We
XI_ITEM = jnp.stack([ER.TEX[ER.ITEM0 + i].reshape(-1) @ We - E_BLANK for i in range(3)])   # A, B, C


# ---------- 자료 ----------

def rollout(sess):
    """교사 행동열을 환경에서 실행. 반환: 임베딩 관측 [L+1, 12, DX], 보상 [L], 작업 시작 채집처 [L], 끝 상태."""
    s, o0 = EL.env0(sess)
    change = jnp.arange(sess["acts"].shape[0]) == sess["change_t"]

    def f(s, x):
        a, rq, ri, ch = x
        s2, o2, r, st = ER.env_step(s, a, rq, ri, ch, sess["b0"])
        return s2, (o2 @ We, r, st)
    sT, (Em, Rw, St) = jax.lax.scan(f, s, (sess["acts"], sess["req"], sess["rid"], change))
    return jnp.concatenate([(o0 @ We)[None], Em]), Rw, St, sT


def build(seeds, D):
    sessions, stack = EL.stack_sessions(seeds, D)
    Em, Rw, St, sT = jax.vmap(rollout)(stack)
    Em, Rw, St = np.asarray(Em), np.asarray(Rw), np.asarray(St)
    S, L = Rw.shape
    Xi = Em[:, :, SLOT] - np.asarray(E_BLANK)                         # 도착 산출물 특징
    Xi = np.where(np.abs(Xi).max(-1, keepdims=True) > 1e-3, Xi, 0.0)  # 빈 칸은 정확히 0 (행렬곱 반올림 차이 제거)
    acts = np.asarray(stack["acts"])
    e_start, e_ts, e_end, e_site = (np.zeros((S, NE), np.int32) for _ in range(4))
    Y = np.zeros((S, NE, DX), np.float32)
    G = np.zeros((S, NE), np.float32)
    arr_t = np.zeros((S, NE), np.int32)
    route_acts = np.full((S, ER.NSITE, AUTO), ER.WAIT, np.int32)
    rin = np.zeros((S, ER.NSITE, DRIN), np.float32)
    for n, s in enumerate(sessions):
        ts_list = np.where(St[n] >= 0)[0]
        assert len(ts_list) == NE, len(ts_list)
        for e, t in enumerate(ts_list):
            i = int(St[n, t])
            go, back = ER.path(s["depot"], s["sites"][i]), ER.path(s["sites"][i], s["depot"])
            st, en = t - len(go), t + 1 + max(len(back), ER.D_DELAY + 1)
            assert list(acts[n, st:t]) == go and acts[n, t] == ER.INTERACT
            e_start[n, e], e_ts[n, e], e_end[n, e], e_site[n, e] = st, t, en, i
            gam = GAMMA ** np.arange(en - st)
            Y[n, e] = (gam[:, None] * Xi[n, st + 1:en + 1]).sum(0)
            G[n, e] = (gam * Rw[n, st:en]).sum()
            if e < ER.NSITE:                                           # 관찰 구간의 방문 = 후보 경로
                route_acts[n, i, :en - st] = acts[n, st:en]
                scene = Em[n, st:t + 1, :9].reshape(t + 1 - st, -1)
                oh = np.zeros((t + 1 - st, ER.NA), np.float32)
                oh[np.arange(t - st), acts[n, st:t]] = 1.0
                rin[n, i] = np.concatenate([np.concatenate([scene, oh], 1).mean(0), scene[-1], [len(go) / 16.0]])
        assert sorted(e_site[n, :ER.NSITE]) == list(range(ER.NSITE))
        arr = np.where(np.abs(Xi[n, 1:]).sum(-1) > 0)[0]                # 스텝 t: ξ_{t+1} ≠ 0
        assert len(arr) == NE, len(arr)
        arr_t[n] = arr
    oh_prev = np.zeros((S, L + 1, ER.NA), np.float32)
    oh_prev[np.arange(S)[:, None], np.arange(1, L + 1)[None], acts] = 1.0
    r_in = np.concatenate([np.zeros((S, 1)), Rw], 1)[..., None]
    xo = np.concatenate([Em[:, :, :9].reshape(S, L + 1, -1), Xi, oh_prev], -1).astype(np.float32)
    xr = np.concatenate([Em.reshape(S, L + 1, -1), oh_prev, r_in], -1).astype(np.float32)
    nidx = np.arange(S)[:, None]
    d = dict(xo=xo, xr=xr, e_start=e_start, e_end=e_end, e_site=e_site, Y=Y, G=G, arr_t=arr_t,
             xi_arr=Xi[nidx, arr_t + 1], r_arr=Rw[nidx, arr_t], rin=rin, rin_e=rin[nidx, e_site], route_acts=route_acts,
             b0=np.asarray(stack["b0"]), b1=np.asarray(stack["b1"]), rid_last=np.asarray(stack["rid"])[:, -1], L=L)
    d = {k: jnp.asarray(v) for k, v in d.items() if isinstance(v, np.ndarray)}
    d["L"] = jnp.asarray(L)
    return d, sT


# ---------- 망 ----------

def mlp_init(k, sizes, last_scale=1.0):
    ks = jax.random.split(k, len(sizes) - 1)
    out = []
    for j, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        s = last_scale if j == len(sizes) - 2 else 1.0
        out.append((jax.random.normal(ks[j], (a, b)) * s / jnp.sqrt(a), jnp.zeros(b)))
    return out


def mlp(p, x):
    for W, b in p[:-1]:
        x = jax.nn.silu(x @ W + b)
    W, b = p[-1]
    return x @ W + b


def gru_init(k, din, dh):
    k1, k2 = jax.random.split(k)
    return dict(Wx=jax.random.normal(k1, (din, 3 * dh)) / jnp.sqrt(din), Wh=jax.random.normal(k2, (dh, 3 * dh)) / jnp.sqrt(dh),
                b=jnp.zeros(3 * dh))


def gru_run(p, X):
    def step(h, x):
        zx, rx, nx = jnp.split(x @ p["Wx"] + p["b"], 3)
        zh, rh, nh = jnp.split(h @ p["Wh"], 3)
        z, r = jax.nn.sigmoid(zx + zh), jax.nn.sigmoid(rx + rh)
        h = (1 - z) * jnp.tanh(nx + r * nh) + z * h
        return h, h
    return jax.lax.scan(step, jnp.zeros(p["Wh"].shape[0]), X)[1]


def init_slow(key):
    ks = jax.random.split(key, 5)
    return dict(enc_o=gru_init(ks[0], 9 * DX + DX + ER.NA, HS), enc_r=gru_init(ks[1], 12 * DX + ER.NA + 1, HS),
                psi=mlp_init(ks[2], [HS + DRIN, 128, DX], 0.1), u=mlp_init(ks[3], [HS, 64, DX], 0.1),
                v=mlp_init(ks[4], [HS + DRIN, 128, 1], 0.1))


def init_fast(key):
    k1, k2 = jax.random.split(key)
    kal = lambda p0, R, Q: jnp.log(jnp.array([p0, R, Q]))
    return dict(key=mlp_init(k1, [DRIN, 256, DK]), b=mlp_init(k2, [HS, 64, DB]),
                kal=dict(sf=kal(1.0, 1e-2, 1e-4), rw=kal(1.0, 1e-1, 1e-4), val=kal(1.0, 1e-1, 1e-4)))


def encode(p, d, idx=None):
    xo, xr = (d["xo"], d["xr"]) if idx is None else (d["xo"][idx], d["xr"][idx])
    return jax.vmap(gru_run, (None, 0))(p["enc_o"], xo), jax.vmap(gru_run, (None, 0))(p["enc_r"], xr)


def take(H, t):
    return jnp.take_along_axis(H, t[..., None], axis=1)


def slow_loss(p, d, idx):
    Ho, Hr = encode(p, d, idx)
    es, at = d["e_start"][idx], d["arr_t"][idx]
    rin_e = d["rin_e"][idx]
    psi = mlp(p["psi"], jnp.concatenate([take(Ho, es), rin_e], -1))
    l_psi = jnp.mean(jnp.sum((d["Y"][idx] - psi) ** 2, -1))
    u = mlp(p["u"], take(Hr, at))
    l_u = jnp.mean((d["r_arr"][idx] - jnp.sum(u * d["xi_arr"][idx], -1)) ** 2)
    v = mlp(p["v"], jnp.concatenate([take(Hr, es), rin_e], -1))[..., 0]
    l_v = jnp.mean((d["G"][idx] - v) ** 2)
    return l_psi + l_u + l_v, jnp.stack([l_psi, l_u, l_v])


def slow_outputs(p, d):
    """느린 망을 고정한 뒤 빠른 쓰기에 필요한 값. 반환: ψ_e, uᵀξ (도착), v_e, h^r (도착), h^o_T, h^r_T."""
    Ho, Hr = encode(p, d)
    es, at, rin_e = d["e_start"], d["arr_t"], d["rin_e"]
    psi = mlp(p["psi"], jnp.concatenate([take(Ho, es), rin_e], -1))
    hr_arr = take(Hr, at)
    uxi = jnp.sum(mlp(p["u"], hr_arr) * d["xi_arr"], -1)
    v = mlp(p["v"], jnp.concatenate([take(Hr, es), rin_e], -1))[..., 0]
    return dict(psi=psi, uxi=uxi, v=v, hr_arr=hr_arr, ho_T=Ho[:, -1], hr_T=Hr[:, -1])


# ---------- 칼만 쓰기 ----------

def kalman(ds, ys, times, kal, q_on=1.0):
    """ds [n, din], ys [n, dout], times [n]. 반환: M [dout, din], P, 혁신 NLL 합, S 최솟값."""
    p0, R, Q = jnp.exp(kal)
    R = R + R_FLOOR
    Q = Q * q_on
    din, dout = ds.shape[-1], ys.shape[-1]
    I = jnp.eye(din)

    def step(c, x):
        M, P, tp = c
        dd, y, t = x
        Pm = P + Q * (t - tp) * I
        Pd = Pm @ dd
        S = R + dd @ Pd
        g = Pd / S
        nu = y - M @ dd
        nll = 0.5 * (jnp.sum(nu ** 2) / S + dout * jnp.log(S))
        A = I - jnp.outer(g, dd)
        return (M + jnp.outer(nu, g), A @ Pm @ A.T + R * jnp.outer(g, g), t), (nll, S)
    (M, P, _), (nll, S) = jax.lax.scan(step, (jnp.zeros((dout, din)), p0 * I, jnp.zeros(())), (ds, ys, times.astype(jnp.float32)))
    return M, P, nll.sum(), S.min()


def fast_inputs(pf, d, so):
    k_e = mlp(pf["key"], d["rin_e"])                                    # [S, NE, DK]
    b = mlp(pf["b"], so["hr_arr"])                                      # [S, NE, DB]
    du = (b[..., :, None] * d["xi_arr"][..., None, :]).reshape(b.shape[0], NE, DB * DX)
    return dict(sf=(k_e, d["Y"] - so["psi"], d["e_end"]),
                rw=(du, (d["r_arr"] - so["uxi"])[..., None], d["arr_t"] + 1),
                val=(jax.lax.stop_gradient(k_e), (d["G"] - so["v"])[..., None], d["e_end"]))


def fast_loss(pf, d, so):
    fi = fast_inputs(pf, d, so)
    tot, parts = 0.0, []
    for m in ("sf", "rw", "val"):
        nll = jax.vmap(lambda a, b_, c: kalman(a, b_, c, pf["kal"][m])[2])(*fi[m])
        parts.append(nll.mean())
        tot = tot + nll.mean()
    return tot, jnp.stack(parts)


# ---------- 학습 ----------

def train(dtr, dva, iters1, iters2, B, log, seed=1):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    p = init_slow(k1)
    opt = optax.adam(1e-3)
    st = opt.init(p)

    @jax.jit
    def step1(p, st, idx, dd):
        (l, parts), g = jax.value_and_grad(slow_loss, has_aux=True)(p, dd, idx)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, parts

    va1 = jax.jit(lambda p, dd: slow_loss(p, dd, jnp.arange(dd["Y"].shape[0])))
    S = dtr["Y"].shape[0]
    best, best_p, curve1 = np.inf, p, []
    for it in range(iters1 + 1):
        if it % 100 == 0:
            lv, pv = va1(p, dva)
            curve1.append([it, float(lv)] + [float(x) for x in pv])
            if float(lv) < best:
                best, best_p = float(lv), p
            log(f"slow {it} val {float(lv):.5f} (psi {float(pv[0]):.5f} u {float(pv[1]):.5f} v {float(pv[2]):.5f})")
        if it == iters1:
            break
        k3, kb = jax.random.split(k3)
        p, st, _ = step1(p, st, jax.random.choice(kb, S, (B,), replace=False), dtr)
    p = best_p
    so_tr, so_va = jax.jit(slow_outputs)(p, dtr), jax.jit(slow_outputs)(p, dva)

    pf = init_fast(k2)
    lab = jax.tree_util.tree_map(lambda _: "net", pf)
    lab["kal"] = jax.tree_util.tree_map(lambda _: "kal", pf["kal"])
    opt2 = optax.multi_transform({"net": optax.adam(1e-3), "kal": optax.adam(0.05)}, lab)
    st2 = opt2.init(pf)

    @jax.jit
    def step2(pf, st2, dd, so):
        (l, parts), g = jax.value_and_grad(fast_loss, has_aux=True)(pf, dd, so)
        u, st2 = opt2.update(g, st2, pf)
        return optax.apply_updates(pf, u), st2, parts

    va2 = jax.jit(fast_loss)
    best, best_pf, curve2 = np.inf, pf, []
    for it in range(iters2 + 1):
        if it % 50 == 0:
            lv, pv = va2(pf, dva, so_va)
            curve2.append([it, float(lv)] + [float(x) for x in pv])
            if np.isfinite(float(lv)) and float(lv) < best:
                best, best_pf = float(lv), pf
            log(f"fast {it} val nll {float(lv):.4f} (sf {float(pv[0]):.4f} rw {float(pv[1]):.4f} val {float(pv[2]):.4f})")
        if it == iters2:
            break
        pf, st2, _ = step2(pf, st2, dtr, so_tr)
    return p, best_pf, dict(slow=curve1, fast=curve2)


# ---------- 평가 ----------

def evaluate(p, pf, d, sT, seed=1):
    so = jax.jit(slow_outputs)(p, d)
    fi = fast_inputs(pf, d, so)
    W = d["Y"].shape[0]
    run = {}
    for m, q_on in (("sf", 1.0), ("rw", 1.0), ("val", 1.0), ("sf_q0", 0.0), ("rw_q0", 0.0)):
        base = m.split("_")[0]
        M, P, nll, Smin = jax.vmap(lambda a, b_, c: kalman(a, b_, c, pf["kal"][base], q_on))(*fi[base])
        run[m] = dict(M=M, P=P, Smin=Smin)
    k_c = mlp(pf["key"], d["rin"])                                       # [W, 5, DK]
    hoT = jnp.broadcast_to(so["ho_T"][:, None], (W, ER.NSITE, HS))
    hrT = jnp.broadcast_to(so["hr_T"][:, None], (W, ER.NSITE, HS))
    psi_c = mlp(p["psi"], jnp.concatenate([hoT, d["rin"]], -1))         # [W, 5, DX]
    u_T = mlp(p["u"], so["hr_T"])                                       # [W, DX]
    b_T = mlp(pf["b"], so["hr_T"])                                      # [W, DB]
    v_c = mlp(p["v"], jnp.concatenate([hrT, d["rin"]], -1))[..., 0]

    def pref(sf, rw):
        Psi = psi_c + jnp.einsum("wxk,wck->wcx", run[sf]["M"], k_c) if sf else psi_c
        uh = u_T + jnp.einsum("wi,wij->wj", b_T, run[rw]["M"][:, 0].reshape(W, DB, DX))
        return jnp.einsum("wcx,wx->wc", Psi, uh), Psi, uh

    Q_full, Psi_full, u_full = pref("sf", "rw")
    Q_off, _, _ = pref(None, "rw")
    Q_q0, _, _ = pref("sf_q0", "rw_q0")
    Q_val = v_c + jnp.einsum("wk,wck->wc", run["val"]["M"][:, 0], k_c)
    key = jax.random.PRNGKey(seed)
    ktie, krand = jax.random.split(key)

    def choose(Q, k):
        tie = jax.random.uniform(k, Q.shape) * 1e-9 * (jnp.abs(Q).max(-1, keepdims=True) + 1e-12)
        return jnp.argmax(Q + tie, -1)
    choices = dict(full=choose(Q_full, ktie), sf_off=choose(Q_off, ktie), q0=choose(Q_q0, ktie), value=choose(Q_val, ktie),
                   forced_b1=d["b1"], random=jax.random.randint(krand, (W,), 0, ER.NSITE))

    def execute(s, acts, rid_last, b0):
        def f(c, a):
            s, first, rs = c
            s2, _, r, st = ER.env_step(s, a, jnp.array(1), rid_last, jnp.array(False), b0)
            return (s2, jnp.where((first < 0) & (st >= 0), st, first), rs + r), None
        return jax.lax.scan(f, (s, jnp.array(-1), jnp.array(0.0)), acts)[0][1:]

    exe = jax.jit(jax.vmap(execute))
    out = {}
    b0, b1 = np.asarray(d["b0"]), np.asarray(d["b1"])
    for name, c in choices.items():
        acts = jnp.take_along_axis(d["route_acts"], c[:, None, None], axis=1)[:, 0]
        first, rs = exe(sT, acts, d["rid_last"], d["b0"])
        first, rs = np.asarray(first), np.asarray(rs)
        out[name] = dict(first=first, reward=rs, chose=np.asarray(c))
    # 진단: 판정 시점의 결과 믿음과 보상 믿음 (full)
    coef = jnp.linalg.pinv(XI_ITEM.T)                                   # Ψ ≈ Σ_i c_i ξ_i 의 최소제곱 계수 (A, B, C 특징은 직교가 아님)
    proj = jnp.einsum("ix,wcx->wci", coef, Psi_full)                    # [W, 5, 3] (A, B, C 무게)
    wi = jnp.arange(W)
    diag = dict(psi_b0=np.asarray(proj[wi, d["b0"]]), psi_b1=np.asarray(proj[wi, d["b1"]]),
                u_items=np.asarray(u_full @ XI_ITEM.T), Q_full=np.asarray(Q_full))
    num = {}
    for m in run:
        P = run[m]["P"]
        num[m] = dict(S_min=float(jnp.min(run[m]["Smin"])), P_min_eig=float(jnp.min(jax.vmap(jnp.linalg.eigvalsh)(P))),
                      nonfinite=bool(~jnp.all(jnp.isfinite(run[m]["M"])) | ~jnp.all(jnp.isfinite(P))))
    return out, diag, num, b0, b1


def summarize(out, b0, b1, n_boot=10000, seed=1):
    rng = np.random.default_rng(seed)
    W = len(b0)
    res = {}
    for name, o in out.items():
        f = o["first"]
        is_b1 = f == b1
        res[name] = dict(b1=float(is_b1.mean()), b0=float((f == b0).mean()), other=float(((f >= 0) & (f != b0) & (f != b1)).mean()),
                         none=float((f < 0).mean()), reward=float(o["reward"].mean()),
                         reward_when_b1=float(o["reward"][is_b1].mean()) if is_b1.any() else None)
    idx = rng.integers(0, W, (n_boot, W))
    pairs = {}
    for a, b in (("full", "sf_off"), ("full", "value"), ("full", "q0"), ("full", "random")):
        dlt = (out[a]["first"] == b1).astype(float) - (out[b]["first"] == b1).astype(float)
        bs = dlt[idx].mean(1)
        pairs[f"{a}-{b}"] = dict(mean=float(dlt.mean()), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])
    return res, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1024)
    ap.add_argument("--n_val", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=256)
    ap.add_argument("--iters1", type=int, default=2000)
    ap.add_argument("--iters2", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    log(f"devices {jax.devices()}")
    dtr, _ = build(range(50000, 50000 + a.n_train), a.D)
    dva, _ = build(range(51024, 51024 + a.n_val), a.D)
    dev, sT = build(range(1000, 1000 + a.n_eval), a.D)
    log(f"data train {dtr['Y'].shape[0]} val {dva['Y'].shape[0]} eval {dev['Y'].shape[0]} L {dtr['L']}/{dva['L']}/{dev['L']}")
    p, pf, curves = train(dtr, dva, a.iters1, a.iters2, a.batch, log)
    kal = {m: [float(x) for x in jnp.exp(v)] for m, v in pf["kal"].items()}
    log(f"kalman (p0, R, Q): {kal}")
    out, diag, num, b0, b1 = evaluate(p, pf, dev, sT)
    res, pairs = summarize(out, b0, b1)
    wi = np.arange(len(b0))
    chose_full = out["full"]["chose"]
    dsum = dict(psi_b0_ABC=diag["psi_b0"].mean(0).tolist(), psi_b1_ABC=diag["psi_b1"].mean(0).tolist(),
                u_ABC=diag["u_items"].mean(0).tolist(),
                frac_psi_b1_B_gt_b0_B=float((diag["psi_b1"][:, 1] > diag["psi_b0"][:, 1]).mean()),
                frac_u_B_max=float((diag["u_items"].argmax(-1) == 1).mean()),
                frac_Q_b1_max=float((diag["Q_full"].argmax(-1) == b1).mean()),
                Q_b1_minus_b0=float((diag["Q_full"][wi, b1] - diag["Q_full"][wi, b0]).mean()),
                chose_full=dict(b1=int((chose_full == b1).sum()), b0=int((chose_full == b0).sum()),
                                other=int(((chose_full != b1) & (chose_full != b0)).sum())))
    summary = dict(config=vars(a) | dict(gamma=GAMMA, DK=DK, HS=HS, DX=DX, DB=DB, seed=1), kalman=kal, results=res, paired=pairs,
                   diag=dsum, numerics=num, curves=curves, seconds=time.time() - t0)
    print("RESULT_JSON " + json.dumps(summary), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
