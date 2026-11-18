# 칸별 키판 (2026-11-17): 명세 1에서 키만 칸마다 다르게. 시냅스 (M_p, P_p, σ_p, R_p, q_p)는 명세 1처럼 칸마다 따로.
#   u_p = w + MLP([w; onehot p]) (마지막 층 0 → 초기에는 모든 칸 u_p = w, 명세 1 초기와 같음), k̂_p = W_k[a] u_p,
#   ℓ_p = 선형([w; onehot a; onehot p]), 흔적 e_p = α e_p + k_p (칸별, α 공유), z_p = σ_p [k_p; e_p].
# 이하 명세 1 설명 (키 경로 외에는 그대로):
# 명세 1 (docs/world/synapse_layer_spec_basic.md): 12.13 구성에서 키 경로만 학습.
#   u = w + MLP(w) (마지막 층 0), k̂ = W_k[a] u (초기 = 12.13의 Wr[a]), k = exp(ℓ) k̂/|k̂| (ℓ = 선형([w; onehot a]), 초기 0),
#   e = α e + k (α = sigmoid(θ_α), 초기 2^(-1/8)), x = [k; e], z_p = σ_p x. 쓰기는 eb_scale.py write() (QN=1) 그대로.
#   목적: 학습 세계 교사 세션의 순차 예측 NLL (경험적 베이즈). 키 경로 lr 1e-3, 칼만 매개변수 lr 0.05, 배치 8.
# 환경변수: ITERS (최대 반복, 기본 2000; 0 = 학습 전 판정만), PATIENCE (기본 500), OUT (결과 JSON 경로)
import sys, os, json, time; sys.path.insert(0, '/workspace/RL-TDP')
import os as _os
import jax
FREEZE_KEYS = _os.environ.get('FREEZE_KEYS', '0') == '1'   # 1: 키 경로 고정 (12.13처럼 칼만 매개변수만 경험적 베이즈)
DIAGP = _os.environ.get('DIAGP', '0') == '1'      # 대각 공분산 (KDN식)
SQRTKF = _os.environ.get('SQRTKF', '0') == '1'   # Codex ①: 칼만 공분산을 제곱근형 (P = S Sᵀ)으로. 식은 그대로
KF64 = _os.environ.get('KF64', '0') == '1'      # Codex ①: 칼만 상태 (M, P)와 그 갱신만 float64. 식·목적·매개변수는 그대로
import jax.numpy as jnp, numpy as np, optax

from rltdp.world import errand as ER, errand_learned as EL, errand_prec as EP, errand_reuse as RU

ER.PUBLIC = False
We, ybar = EL.emb_fixed(0)
# 월드 모델 고정: WM_LOAD=경로면 저장된 μ⁰를 쓰고, 없으면 학습한 뒤 WM_SAVE=경로에 저장 (실행마다 다시 학습하면 μ⁰가 달라짐)
_WL, _WS = os.environ.get('WM_LOAD'), os.environ.get('WM_SAVE')
if _WL:
    _z = np.load(_WL); p = {k: jnp.array(_z[k]) for k in _z.files}; vmax = None
else:
    p, vmax = EP.train("random", seed=0, D=256, iters_k=0, log=lambda m: None)
    if _WS:
        np.savez(_WS, **{k: np.array(v) for k, v in p.items()})
feat, wm, _ = EP.parts(p, We, ybar, "random")
# Codex ①: 월드 모델·임베딩은 float32로 만든 뒤에야 x64를 켠다 (같은 μ⁰). 이후 매개변수는 명시적으로 float32, 칼만 상태만 float64.
if KF64:
    jax.config.update("jax_enable_x64", True)
KDT = jnp.float64 if KF64 else jnp.float32
KD, NPOLE = EP.DK, 2
DZ = NPOLE * KD
DW = 9 * EP.H
# 관측 잡음 바닥: 관측을 8비트 영상으로 보면 픽셀당 양자화 분산 (1/255)²/12, 고정 임베딩 We를 거친 특징 차원당 분산 = 평균 ‖We 열‖² × 그 값.
# OBS_NOISE=0이면 끔 (이전 판과 같음).
_ON = os.environ.get('OBS_NOISE', '1')
if _ON == 'deq':
    # 역양자화 바닥: 관측은 타일 25개 중 하나 (고정 임베딩). 가장 가까운 두 타일 간격 d_min 폭의 균등 잡음, 차원당 분산 d_min²/12.
    _Et = (np.array(ER.TEX).reshape(ER.TEX.shape[0], -1) @ np.array(We)) - np.array(ybar)
    _Dt = np.sqrt(((_Et[:, None] - _Et[None]) ** 2).sum(-1)) + np.eye(len(_Et)) * 1e9
    R0 = float(_Dt.min() ** 2 / 12.0)
else:
    R0 = float(np.mean(np.sum(np.array(We) ** 2, 0))) / 255.0 ** 2 / 12.0 if _ON == '1' else 0.0
print("R0 (observation noise floor)", R0, flush=True)
phi = (ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1) @ We) - ybar
gB = phi[1] - phi.mean(0)
LOG1213 = '/workspace/RL-TDP/docs/world/results/errand_diag_20261115b/eb_process_noise.log'
p13 = json.loads(open(LOG1213).read().strip().splitlines()[-1])

# ---- 매개변수: 키 경로 (kp)와 칼만 (kf)
kr = jax.random.split(jax.random.PRNGKey(11), 2)
kp = dict(W1=jax.random.normal(kr[0], (DW + EP.NP, 128)) / np.sqrt(DW + EP.NP), b1=jnp.zeros(128),
          W2=jnp.zeros((128, DW)), b2=jnp.zeros(DW),
          Wk=jnp.array(p["Wr"]),                                   # [NA, 128, 72], 12.13의 난수 키
          lw=jnp.zeros(DW + ER.NA + EP.NP), lb=jnp.zeros(()),
          ta=jnp.array(np.log(2.0 ** (-1 / 8) / (1 - 2.0 ** (-1 / 8)))))
kf = dict(ls=jnp.log(jnp.array(p13["sigma2"])), lr=jnp.log(jnp.array(p13["R"])), lq=jnp.log(jnp.array(p13["q"])))
if os.environ.get('KF_INIT'):                                     # 칼만 매개변수 초기값을 다른 실행 결과 (…_params.npz)에서
    _zk = np.load(os.environ['KF_INIT']); kf = {k: jnp.array(_zk[f"kf/{k}"]) for k in kf}
th = dict(kp=kp, kf=kf)
th = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), th)


POS = jnp.eye(EP.NP, dtype=jnp.float32)


def key_fn(kp, o, a):
    """칸 12개의 키 [12, KD]와 ℓ [12]."""
    w = feat(o)[:9].reshape(-1)
    wp = jnp.concatenate([jnp.tile(w, (EP.NP, 1)), POS], -1)                  # [12, DW+12]
    u = w + jax.nn.silu(wp @ kp["W1"] + kp["b1"]) @ kp["W2"] + kp["b2"]        # [12, DW]
    kh = u @ kp["Wk"][a].T                                                    # [12, KD]
    li = jnp.concatenate([jnp.tile(jnp.concatenate([w, jax.nn.one_hot(a, ER.NA)]), (EP.NP, 1)), POS], -1)
    ell = li @ kp["lw"] + kp["lb"]
    return jnp.exp(ell)[:, None] * kh / (jnp.linalg.norm(kh, axis=-1, keepdims=True) + 1e-8), ell


def alpha(kp):
    return jax.nn.sigmoid(kp["ta"])


def step_e(kp, e, o, a):
    k, ell = key_fn(kp, o, a)
    e = jnp.stack([k, alpha(kp) * e[:, 1] + k], 1)              # [12, 2, KD], 칸별 극 {0, 반감기 학습}: [지금 키; 흔적]
    return e.astype(jnp.float32), ell


def write(th, O, acts, stop_t):
    kp, kf = th["kp"], th["kf"]
    R, Q, sg = jnp.exp(kf["lr"]), jnp.exp(kf["lq"]), jnp.exp(0.5 * kf["ls"])

    @jax.checkpoint
    def f(c, x):
        M, P, e = c
        o, a, o2, t = x
        e, ell = step_e(kp, e, o, a)
        mu, _ = wm(o, a)
        r = feat(o2) - mu

        def pp_sqrt(Mp, Sp, sp, rp, Rp, Qp, ep):
            # Codex ①: 같은 칼만 (P⁻ = P + qI, 같은 이득·사후), P = S Sᵀ의 제곱근형으로만 계산.
            # 과정 잡음: S ← chol(S Sᵀ + qI). 쓰기: Potter (scripts/world/diag_20261115/eb_scale.py mult_step와 같은 꼴).
            z = sp * ep.reshape(-1)
            C = Sp @ Sp.T + Qp * jnp.eye(DZ)
            Sp = jnp.linalg.cholesky(0.5 * (C + C.T))
            fv = Sp.T @ z; Sv = Rp + R0 + fv @ fv
            err = rp - Mp @ z
            al = 1.0 / Sv; ga = 1.0 / (1.0 + jnp.sqrt(al * (Rp + R0)))
            K = al * (Sp @ fv)
            M2 = Mp + jnp.outer(err, K)
            got = (M2 @ z) @ rp / (rp @ rp + 1e-12)
            return M2, Sp - ga * jnp.outer(K, fv), 0.5 * (err @ err / Sv + EP.H * jnp.log(Sv)), got

        def pp_diag(Mp, pd, sp, rp, Rp, Qp, ep):
            # 대각 공분산 (KDN식 평균장 투영): P = diag(p). 예측 분산 ρ + Σ p z², 이득 p∘z / S,
            # 분산 갱신 p ← p / (1 + p z² / ρ) (정밀도에 z²/ρ를 더함, 빼기 없음 → 늘 양수). ρ = R + R₀.
            pd = pd + Qp
            z = sp * ep.reshape(-1)
            rho = Rp + R0
            pz = pd * z; S = rho + z @ pz
            err = rp - Mp @ z
            M2 = Mp + jnp.outer(err, pz) / S
            got = (M2 @ z) @ rp / (rp @ rp + 1e-12)
            return M2, pd / (1.0 + pd * z * z / rho), 0.5 * (err @ err / S + EP.H * jnp.log(S)), got

        def pp(Mp, Pp, sp, rp, Rp, Qp, ep):
            sp, rp, Rp, Qp, ep = (x.astype(KDT) for x in (sp, rp, Rp, Qp, ep))
            Pp = Pp + Qp * jnp.eye(DZ, dtype=KDT)
            z = sp * ep.reshape(-1)
            err = rp - Mp @ z; Pz = Pp @ z; S = Rp + R0 + z @ Pz             # R0: 관측 잡음 바닥 (8비트 양자화)
            M2 = Mp + jnp.outer(err, Pz) / S
            got = (M2 @ z) @ rp / (rp @ rp + 1e-12)
            return M2, Pp - jnp.outer(Pz, Pz) / S, 0.5 * (err @ err / S + EP.H * jnp.log(S)), got
        M2, P2, nll, got = jax.vmap(pp_diag if DIAGP else (pp_sqrt if SQRTKF else pp))(M, P, sg, r, R, Q, e)
        keep = t > stop_t
        return (jnp.where(keep, M, M2), jnp.where(keep, P, P2), e), (jnp.where(keep, 0.0, nll.sum()), got[EP.SLOT], ell[EP.SLOT], e[EP.SLOT].reshape(-1))
    P0 = jnp.ones((EP.NP, DZ), KDT) if DIAGP else jnp.tile(jnp.eye(DZ, dtype=KDT), (EP.NP, 1, 1))
    c0 = (jnp.zeros((EP.NP, EP.H, DZ), KDT), P0, jnp.zeros((EP.NP, NPOLE, KD), jnp.float32))
    if SQRTKF:                                                        # 제곱근형은 행렬곱 정밀도 highest (12.15·12.17 교훈)
        with jax.default_matmul_precision("highest"):
            return jax.lax.scan(f, c0, (O[:-1], acts, O[1:], jnp.arange(acts.shape[0])))
    return jax.lax.scan(f, c0, (O[:-1], acts, O[1:], jnp.arange(acts.shape[0])))


def sess_nll(th, sess):
    O, _, _ = EL.observe_stream(sess)
    _, (nll, _, _, _) = write(th, O, sess["acts"], 10 ** 9)
    return nll.sum()


# ---- 자료
ER.EXTRA_REVISIT = os.environ.get('EXTRA_REVISIT', '0') == '1'   # Codex ②: 학습·검증 세계의 교사 끝에 안정 사실 재방문 추가
_, tr = EL.stack_sessions(range(50_000, 51_024), 256)
_, va = EL.stack_sessions(range(51_024, 51_152), 256)
ER.EXTRA_REVISIT = False                                          # 판정 세계는 그대로
print("lengths: train", tr["acts"].shape, "val", va["acts"].shape, flush=True)


@jax.jit
def val_nll(th):
    def chunk(c, idx):
        sb = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, idx, 16), va)
        return c + jax.vmap(lambda s: sess_nll(th, s))(sb).sum(), None
    tot, _ = jax.lax.scan(chunk, 0.0, jnp.arange(0, 128, 16))
    return tot / 128


# ---- 판정 (eb_scale.py와 같음, 키·흔적만 학습된 식)
sessions, stack = EL.stack_sessions(range(1000, 1256), 256)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])


def judge_one(th, sess, plans, ai_all, zt=False):
    kp, kf = th["kp"], th["kf"]
    O, s_end, _ = EL.observe_stream(sess)
    t_c = RU.last_arrival(O)
    Y = jax.vmap(feat)(O[1:])[:, EP.SLOT]
    blank = feat(ER.TEX[jnp.full(12, ER.BLANK)].reshape(12, -1))[EP.SLOT]
    arrived = ~jnp.all(jnp.isclose(Y, blank, atol=1e-5), -1)
    (M, P, e), (_, got, ell, X) = write(th, O, sess["acts"], t_c)
    got_mean = jnp.sum(jnp.where(arrived, got, 0)) / jnp.maximum(arrived.sum(), 1)
    T = sess["acts"].shape[0]
    live = jnp.arange(T) <= t_c
    a_ = sess["acts"]
    def mean_where(v, m):
        return jnp.sum(jnp.where(m, v, 0)) / jnp.maximum(m.sum(), 1)
    ksz = jnp.exp(ell)
    k_int = mean_where(ksz, live & (a_ == ER.INTERACT))
    k_mov = mean_where(ksz, live & (a_ < 4))
    k_wait = mean_where(ksz, live & (a_ == ER.WAIT))
    Xn = X / (jnp.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)
    En = X[:, KD:] / (jnp.linalg.norm(X[:, KD:], axis=-1, keepdims=True) + 1e-12)
    l2 = live[1:] & live[:-1]
    cos_x = mean_where(jnp.sum(Xn[1:] * Xn[:-1], -1), l2)
    cos_e = mean_where(jnp.sum(En[1:] * En[:-1], -1), l2)
    sgs = jnp.exp(0.5 * kf["ls"][EP.SLOT])

    def sim(plan, ai):
        def f(c, a):
            s, o, e = c
            e, _ = step_e(kp, e, o, a)
            yh = M[EP.SLOT] @ (sgs * e[EP.SLOT].reshape(-1))
            s2, o2, _, _ = ER.env_step(s, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (s2, o2, e), yh
        _, YH = jax.lax.scan(f, (s_end, O[-1], jnp.zeros_like(e) if zt else e), plan)   # zt: probe를 흔적 0에서 시작
        yh = YH[ai]
        return (yh @ gB) / (jnp.linalg.norm(yh) * jnp.linalg.norm(gB) + 1e-8), jnp.linalg.norm(yh)
    sc, nrm = jax.vmap(sim)(plans, ai_all)
    return sc, nrm, got_mean, k_int, k_mov, k_wait, cos_x, cos_e


import functools
@functools.partial(jax.jit, static_argnames=('zt',))
def judge_chunk(th, sb, pl, ar, zt=False):
    return jax.vmap(lambda s, a, b: judge_one(th, s, a, b, zt))(sb, pl, ar)


JCH = int(os.environ.get('JCHUNK', 32))


def judge(th, zt=False):
    outs = []
    for i in range(0, 256, JCH):
        sl = lambda x: x[i:i + JCH]
        outs.append(judge_chunk(th, jax.tree_util.tree_map(sl, stack), plans[i:i + JCH], arrive[i:i + JCH], zt))
    sc, nrm, got, ki, km, kw, cx, ce = [np.concatenate([np.array(o[j]) for o in outs]) for j in range(8)]
    best = (sc + 1e-9 * np.random.default_rng(0).random(sc.shape)).argmax(1)
    return dict(b1=float(np.mean(best == b1)), b0=float(np.mean(best == b0)),
                other=float(np.mean((best != b1) & (best != b0))),
                written=float(np.mean(got)), read_norm=float(np.mean(nrm)),
                ksize_interact=float(np.mean(ki)), ksize_move=float(np.mean(km)), ksize_wait=float(np.mean(kw)),
                cos_addr_neighbor=float(np.mean(cx)), cos_trace_neighbor=float(np.mean(ce)),
                per_world_b1=(best == b1).astype(int).tolist())


def params_summary(th):
    a = float(alpha(th["kp"]))
    kf = th["kf"]
    return dict(alpha=a, half_life=float(-np.log(2) / np.log(a)),
                sigma2=np.exp(np.array(kf["ls"])).tolist(), R=np.exp(np.array(kf["lr"])).tolist(),
                q=np.exp(np.array(kf["lq"])).tolist())


DIAG_PR = os.environ.get('DIAG_PR')   # NaN 원인 진단: 체크포인트로 학습 세계 순전파, 칸별 zᵀPz·S·P 최소 고유값
if DIAG_PR:
    z_ = np.load(DIAG_PR); thD = {a: {b: jnp.array(z_[f"{a}/{b}"]) for b in th[a]} for a in th}
    def diag_one(sess):
        kp, kf = thD["kp"], thD["kf"]
        R, Q, sg = jnp.exp(kf["lr"]), jnp.exp(kf["lq"]), jnp.exp(0.5 * kf["ls"])
        O, _, _ = EL.observe_stream(sess)
        def f(c, x):
            M, P, e = c
            o, a, o2 = x
            e, _ = step_e(kp, e, o, a)
            mu, _ = wm(o, a)
            r = feat(o2) - mu
            def pp(Mp, Pp, sp, rp, Rp, Qp, ep):
                Pp = Pp + Qp * jnp.eye(DZ)
                z = sp * ep.reshape(-1)
                err = rp - Mp @ z; Pz = Pp @ z; zPz = z @ Pz; S = Rp + R0 + zPz
                return Mp + jnp.outer(err, Pz) / S, Pp - jnp.outer(Pz, Pz) / S, zPz, S, err @ err / S
            M2, P2, zPz, S, e2 = jax.vmap(pp)(M, P, sg, r, R, Q, e)
            return (M2, P2, e), (zPz, S, e2)
        c0 = (jnp.zeros((EP.NP, EP.H, DZ)), jnp.tile(jnp.eye(DZ), (EP.NP, 1, 1)), jnp.zeros((EP.NP, NPOLE, KD)))
        (M, P, _), (zPz, S, e2) = jax.lax.scan(f, c0, (O[:-1], sess["acts"], O[1:]))
        eigmin = jax.vmap(lambda A: jnp.linalg.eigvalsh(0.5 * (A + A.T))[0])(P)       # 세션 끝 칸별 P 최소 고유값
        return zPz.min(0), (zPz < 0).mean(0), S.min(0), e2.max(0), eigmin
    dj = jax.jit(jax.vmap(diag_one))
    sb = jax.tree_util.tree_map(lambda x: x[:64], tr)
    outs = [dj(jax.tree_util.tree_map(lambda x: x[i:i + 16], sb)) for i in range(0, 64, 16)]
    zmin, zneg, smin, e2max, eig = [np.concatenate([np.array(o[j]) for o in outs]) for j in range(5)]
    R_ = np.exp(np.array(thD["kf"]["lr"]))
    rep = dict(ckpt=DIAG_PR, R=R_.tolist(),
               zPz_min=zmin.min(0).tolist(), frac_steps_zPz_negative=zneg.mean(0).tolist(),
               S_min=smin.min(0).tolist(), S_min_over_R=(smin.min(0) / R_).tolist(),
               max_err2_over_S=e2max.max(0).tolist(), P_eigmin_end=eig.min(0).tolist(),
               sessions_with_negative_zPz=int((zneg.max(1) > 0).sum()), any_nonfinite=bool(~np.all(np.isfinite(smin))))
    for k, v in rep.items():
        print(k, np.array(v).round(8).tolist() if isinstance(v, list) else v, flush=True)
    json.dump(rep, open(os.environ['OUT'], 'w'), indent=1)
    sys.exit(0)


ZTJ = os.environ.get('ZT_JUDGE')      # 흔적 0 판정: 쉼표로 체크포인트 (start = 12.13 시작 상태). 이어받은 흔적 판정과 나란히.
if ZTJ:
    out = {}
    for ck in ZTJ.split(','):
        t_ = th if ck == 'start' else (lambda z: {a: {b: jnp.array(z[f"{a}/{b}"]) for b in th[a]} for a in th})(np.load(ck))
        for zt in (False, True):
            j = judge(t_, zt)
            name = f"{ck.split('/')[-1]}|{'zero_trace' if zt else 'carried_trace'}"
            out[name] = j
            print(f"[{name}] b1 {j['b1']:.4f} b0 {j['b0']:.4f} other {j['other']:.4f} read_norm {j['read_norm']:.4f}", flush=True)
    json.dump(out, open(os.environ['OUT'], 'w'))
    sys.exit(0)


DECOMP = os.environ.get('DECOMP')     # 읽기 분해: 쉼표로 체크포인트 (start = 12.13 시작 상태)


def decomp_one(th, sess, plans, ai_all):
    """도착 칸 읽기 ŷ_i = M_T z_i = Σ_t w_t r_t 를 과거 쓰기 스텝별 가중치로 분해.
    M_{t+1} = M_t A_t + r_t g_tᵀ, A_t = I − z_t g_tᵀ  →  w_t = g_tᵀ u_{t+1}, u_t = A_t u_{t+1}, u_T = z_i."""
    kp, kf = th["kp"], th["kf"]
    S_ = EP.SLOT
    R, Q, sg = jnp.exp(kf["lr"][S_]), jnp.exp(kf["lq"][S_]), jnp.exp(0.5 * kf["ls"][S_])
    O, s_end, _ = EL.observe_stream(sess)
    t_c = RU.last_arrival(O)
    T = sess["acts"].shape[0]
    # 스텝별 시작 채집처 → 도착 스텝의 채집처
    s0, _ = EL.env0(sess)
    change = jnp.arange(T) == sess["change_t"]
    def fenv(c, x):
        st, cur = c
        a, rq, ri, ch = x
        st2, o2, _, site = ER.env_step(st, a, rq, ri, ch, sess["b0"])
        cur = jnp.where(site >= 0, site, cur)
        return (st2, cur), cur
    _, cur_site = jax.lax.scan(fenv, (s0, jnp.array(-1)), (sess["acts"], sess["req"], sess["rid"], change))
    blank = ER.TEX[ER.BLANK].reshape(-1)
    arrived = ~jnp.all(jnp.isclose(O[1:, S_], blank), -1)                       # 쓰기 스텝 t의 목표 o_{t+1}에 도착
    items = ER.TEX[ER.ITEM0:ER.ITEM0 + 3].reshape(3, -1)
    item = jnp.argmin(jnp.sum((O[1:, S_][:, None] - items[None]) ** 2, -1), -1)
    # 도착 칸 쓰기 순서열 (z_t, g_t, r_t)
    def fw(c, x):
        Mp, Pp, e = c
        o, a, o2, t = x
        e, _ = step_e(kp, e, o, a)
        mu, _ = wm(o, a)
        r = (feat(o2) - mu)[S_]
        Pp = Pp + Q * jnp.eye(DZ)
        z = sg * e[S_].reshape(-1)
        err = r - Mp @ z; Pz = Pp @ z; Sv = R + R0 + z @ Pz
        g = Pz / Sv
        keep = t > t_c
        Mn = jnp.where(keep, Mp, Mp + jnp.outer(err, g)); Pn = jnp.where(keep, Pp, Pp - jnp.outer(Pz, Pz) / Sv)
        return (Mn, Pn, e), (z, jnp.where(keep, 0.0, 1.0) * g, r)
    c0 = (jnp.zeros((EP.H, DZ)), jnp.eye(DZ), jnp.zeros((EP.NP, NPOLE, KD)))
    (MT, _, e_end), (Z, G, Rr) = jax.lax.scan(fw, c0, (O[:-1], sess["acts"], O[1:], jnp.arange(T)))
    def probe_z(plan, ai):
        def f(c, a):
            st, o, e = c
            e, _ = step_e(kp, e, o, a)
            st2, o2, _, _ = ER.env_step(st, a, jnp.array(1), sess["rid"][-1], jnp.array(False), sess["b0"])
            return (st2, o2, e), sg * e[S_].reshape(-1)
        _, Zs = jax.lax.scan(f, (s_end, O[-1], e_end), plan)
        return Zs[ai]
    Zq = jax.vmap(probe_z)(plans, ai_all)                                          # [NSITE, DZ]
    def weights(zq):
        def fb(u, x):
            z, g = x
            w = g @ u
            return u - z * w, w
        _, W = jax.lax.scan(fb, zq, (Z, G), reverse=True)
        return W
    W = jax.vmap(weights)(Zq)                                                      # [NSITE, T]
    yh_direct = Zq @ MT.T                                                          # [NSITE, 8]
    yh_sum = W @ Rr
    gBn = gB / jnp.linalg.norm(gB)
    b0, b1 = sess["b0"], sess["b1"]
    cat = jnp.stack([arrived & (cur_site == b1),
                     arrived & (cur_site == b0) & (item == 1),
                     arrived & (cur_site == b0) & (item == 2),
                     arrived & (cur_site != b0) & (cur_site != b1),
                     ~arrived], 0)                                               # [5, T]
    wsum = jnp.einsum('st,ct->sc', W, cat)                                         # [NSITE, 5] 가중치 합
    bproj = jnp.einsum('st,ct,t->sc', W, cat, Rr @ gBn)                           # [NSITE, 5] B 방향 기여
    ncnt = cat.sum(1)
    return wsum, bproj, ncnt, jnp.max(jnp.abs(yh_direct - yh_sum)), yh_direct


if DECOMP:
    CATS = ["b1_arrivals(B)", "b0_arrivals_B", "b0_arrivals_C", "other_site_arrivals", "non_arrival_steps"]
    dj = jax.jit(jax.vmap(lambda s, a, b, t_: decomp_one(t_, s, a, b), in_axes=(0, 0, 0, None)))
    out = {}
    for ck in DECOMP.split(','):
        if ck == 'start':
            t_ = th
        else:
            z = np.load(ck); t_ = {a: {b: jnp.array(z[f"{a}/{b}"]) for b in th[a]} for a in th}
        res_ = [dj(jax.tree_util.tree_map(lambda x: x[i:i + JCH], stack), plans[i:i + JCH], arrive[i:i + JCH], t_) for i in range(0, 256, JCH)]
        wsum, bproj, ncnt, err, yh = [np.concatenate([np.array(r[j]) for r in res_]) for j in range(5)]
        sc = (yh @ np.array(gB)) / (np.linalg.norm(yh, axis=-1) * np.linalg.norm(np.array(gB)) + 1e-8)
        best = (sc + 1e-9 * np.random.default_rng(0).random(sc.shape)).argmax(1)
        ok = best == b1
        wi = np.arange(256)
        rep = dict(b1=float(ok.mean()), max_abs_recon_err=float(err.max()), counts_mean=dict(zip(CATS, ncnt.mean(0).round(2).tolist())))
        for grp, m in [("all", np.ones(256, bool)), ("correct", ok), ("wrong", ~ok)]:
            for site, idx in [("probe_b1", b1), ("probe_b0", b0)]:
                rep[f"{grp}/{site}/weight"] = dict(zip(CATS, wsum[wi, idx][m].mean(0).round(4).tolist()))
                rep[f"{grp}/{site}/Bproj"] = dict(zip(CATS, bproj[wi, idx][m].mean(0).round(5).tolist()))
            rep[f"{grp}/cosB_b1"] = float(sc[wi, b1][m].mean()); rep[f"{grp}/cosB_b0"] = float(sc[wi, b0][m].mean())
        out[ck] = rep
        print(f"== {ck}", json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    json.dump(out, open(os.environ.get('OUT'), 'w'), ensure_ascii=False, indent=1)
    sys.exit(0)

LOAD = os.environ.get('LOAD')        # 체크포인트 판정만: 그대로 / 도착 칸 q만 12.13 값 / 도착 칸 q·σ² 12.13 값
if LOAD:
    z = np.load(LOAD)
    thL = {a: {b: jnp.array(z[f"{a}/{b}"]) for b in th[a]} for a in th}
    S = EP.SLOT
    v_q = dict(thL, kf=dict(thL["kf"], lq=thL["kf"]["lq"].at[S].set(np.log(p13["q"][S]))))
    v_qs = dict(v_q, kf=dict(v_q["kf"], ls=v_q["kf"]["ls"].at[S].set(np.log(p13["sigma2"][S]))))
    out = {}
    for name, t_ in [("loaded", thL), ("slot_q_1213", v_q), ("slot_q_sigma2_1213", v_qs)]:
        j = judge(t_)
        out[name] = dict(j, slot_q=float(np.exp(t_["kf"]["lq"][S])), slot_sigma2=float(np.exp(t_["kf"]["ls"][S])))
        print(f"[{name}] b1 {j['b1']:.4f} b0 {j['b0']:.4f} other {j['other']:.4f}  slot q {out[name]['slot_q']:.3g} sigma2 {out[name]['slot_sigma2']:.3g}", flush=True)
    json.dump(out, open(os.environ.get('OUT', LOAD.replace('.npz', '_qdiag.json')), 'w'))
    sys.exit(0)

t0 = time.time()
v0 = float(val_nll(th))
j0 = judge(th)
_START = ("KF_INIT=" + os.environ["KF_INIT"].split("/")[-1]) if os.environ.get('KF_INIT') else "12.13 (무작위 키 + 12.13 경험적 베이즈 σ²·R·q)"
print(f"[start = {_START}{' + WM_LOAD' if _WL else ''}] val_nll {v0:.1f}  b1 {j0['b1']:.4f} b0 {j0['b0']:.4f} other {j0['other']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
res = dict(spec="cellkey_separate_synapse", R0=R0, diag_p=DIAGP, obs_noise=_ON, freeze_keys=FREEZE_KEYS, kf64=KF64, sqrt_kf=SQRTKF, extra_revisit=os.environ.get("EXTRA_REVISIT", "0") == "1", before=dict(val_nll=v0, **j0, **params_summary(th)))
OUT = os.environ.get('OUT', '/workspace/RL-TDP/docs/world/results/synapse_layer_20261117/cellkey.json')

ITERS = int(os.environ.get('ITERS', 2000))
PAT = int(os.environ.get('PATIENCE', 500))
JUDGE_EVERY = int(os.environ.get('JUDGE_EVERY', 500))   # 이 반복마다 판정 (b1/b0/다른 곳)
opt = optax.multi_transform({"kp": optax.adam(0.0) if FREEZE_KEYS else optax.adam(1e-3), "kf": optax.adam(0.05)}, {"kp": "kp", "kf": "kf"})
st = opt.init(th)


GACC = int(os.environ.get('GRAD_ACC', 1))   # 메모리용: 배치 8을 GACC개로 나눠 기울기를 평균 (같은 배치의 같은 기울기)


@jax.jit
def upd(th, st, idx):
    def lf(th, ii):
        sb = jax.tree_util.tree_map(lambda x: x[ii], tr)
        return jax.vmap(lambda s: sess_nll(th, s))(sb).mean()
    if GACC == 1:                                                    # 원래 경로 (scan으로 감싸지 않음)
        l, g = jax.value_and_grad(lf)(th, idx)
    else:
        parts = idx.reshape(GACC, -1)
        def acc(c, ii):
            l_, g_ = jax.value_and_grad(lf)(th, ii)
            return jax.tree_util.tree_map(lambda a, b: a + b / GACC, c, (l_, g_)), None
        z0 = (jnp.zeros(()), jax.tree_util.tree_map(jnp.zeros_like, th))
        (l, g), _ = jax.lax.scan(acc, z0, parts)
    g, st = opt.update(g, st, th)
    return optax.apply_updates(th, g), st, l


curve = [dict(it=0, val_nll=v0)]
best_v, best_th, best_it = v0, th, 0
rk = jax.random.PRNGKey(5)
for it in range(1, ITERS + 1):
    rk, k = jax.random.split(rk)
    th, st, l = upd(th, st, jax.random.randint(k, (8,), 0, 1024))
    if it <= 5 or it == 10:
        print(f"  [time] it {it} loss {float(l):.1f} ({time.time()-t0:.1f}s)", flush=True)
    if not np.isfinite(float(l)):
        print("nan at", it, flush=True); break
    if it % 100 == 0:
        v = float(val_nll(th))
        curve.append(dict(it=it, val_nll=v, train_nll=float(l)))
        if v < best_v:
            best_v, best_th, best_it = v, th, it
            np.savez(OUT.replace('.json', '_best_params.npz'), **{f"{a}/{b}": np.array(vv) for a in best_th for b, vv in best_th[a].items()})   # 최저 검증 NLL 체크포인트 (중간 저장)
        print(f"it {it} train {float(l):.1f} val {v:.1f} best {best_v:.1f}@{best_it} half_life {params_summary(th)['half_life']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if it % JUDGE_EVERY == 0:
            jj = judge(th)
            curve[-1].update(b1=jj["b1"], b0=jj["b0"], other=jj["other"])
            print(f"  [judge it {it}] b1 {jj['b1']:.4f} b0 {jj['b0']:.4f} other {jj['other']:.4f}", flush=True)
            if os.environ.get('SAVE_JUDGE_CKPT') == '1':                   # 2026-11-18: 판정 시점 체크포인트 (하락·회복 분해용)
                np.savez(OUT.replace('.json', f'_it{it}_params.npz'), **{f"{a}/{b}": np.array(vv) for a in th for b, vv in th[a].items()})
        if it - best_it >= PAT:
            print("early stop", flush=True); break

j1 = judge(best_th)
res.update(after=dict(val_nll=best_v, best_it=best_it, **j1, **params_summary(best_th)), curve=curve,
           note="train 50000-51023, val 51024-51151, judge 1000-1255; padding with WAIT as stack_sessions")
print(f"[after] val_nll {best_v:.1f}@{best_it}  b1 {j1['b1']:.4f} b0 {j1['b0']:.4f} other {j1['other']:.4f}", flush=True)
print(json.dumps({k: v for k, v in res["after"].items() if k != "per_world_b1"}), flush=True)
json.dump(res, open(OUT, 'w'))
np.savez(OUT.replace('.json', '_params.npz'), **{f"{a}/{b}": np.array(v) for a in best_th for b, v in best_th[a].items()})
