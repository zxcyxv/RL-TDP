"""1단계 진단 (docs/world/results/stage1_20261119). 2026-11-19.

본 실행 (stage1_route.py, 커밋 8ebe9c2)은 학습된 매개변수를 저장하지 않았다. 같은 설정·시드로 다시 맞추고 저장한 뒤 (있으면 불러옴),
판정 시점의 값을 분해한다. 학습 설정은 바꾸지 않는다.
  1. 선택 내역을 채집처 역할별로 (a0 = 교사의 A 채집처, a1 = 다른 A, b0, b1, c).
  2. 경로별 Ψ̂ = ψ_θ (느린 사전) + M_Ψ k (빠른 읽기)의 산출물 무게 (A, B, C 최소제곱)와 범위 밖 잔차.
  3. û = u_θ (느린) + b_θᵀ Δ_u (빠른)의 산출물별 예측 보상.
  4. Q = ûᵀ Ψ̂를 느린 몫과 빠른 몫으로. 거리 (경로 길이)와 할인.
  5. 빠른 읽기의 회수: b1·b0 경로에서 M_Ψ k가 그 경로에 쓴 목표를 얼마나 되돌려 주는가. 세계 안 경로 키의 코사인.
  6. 진단용 대체 선택 (조건이 아님): û를 느린 몫만 / 빠른 몫만으로, 또는 선호를 B 특징 방향으로 둔 경우의 b1 비율.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp

from . import errand as ER
from . import stage1_route as S1

ROLES = ["a0", "a1", "b0", "b1", "c"]


def role_table(seeds, D):
    """판정 세계마다 채집처 i의 역할 번호 (ROLES 순서)."""
    out = []
    for s in seeds:
        se = ER.make_session(s, D)
        r = np.zeros(ER.NSITE, np.int32)
        for i in range(ER.NSITE):
            if i == se["a0"]:
                r[i] = 0
            elif se["prod"][i] == 0:
                r[i] = 1
            elif i == se["b0"]:
                r[i] = 2
            elif i == se["b1"]:
                r[i] = 3
            else:
                r[i] = 4
        out.append(r)
    return np.stack(out)


def to_np(t):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="/workspace/runs/stage1_params.pkl")
    ap.add_argument("--out", default="")
    ap.add_argument("--n_train", type=int, default=1024)
    ap.add_argument("--n_val", type=int, default=128)
    ap.add_argument("--n_eval", type=int, default=256)
    ap.add_argument("--iters1", type=int, default=2000)
    ap.add_argument("--iters2", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    D = 256
    dtr, _ = S1.build(range(50000, 50000 + a.n_train), D)
    dva, _ = S1.build(range(51024, 51024 + a.n_val), D)
    dev, sT = S1.build(range(1000, 1000 + a.n_eval), D)
    if os.path.exists(a.params):
        p, pf = pickle.load(open(a.params, "rb"))
        log(f"loaded {a.params}")
    else:
        p, pf, _ = S1.train(dtr, dva, a.iters1, a.iters2, 32 if a.n_train >= 32 else 8, lambda m: None)
        os.makedirs(os.path.dirname(a.params), exist_ok=True)
        pickle.dump((to_np(p), to_np(pf)), open(a.params, "wb"))
        log(f"trained and saved {a.params}")
    p = jax.tree_util.tree_map(jnp.asarray, p)
    pf = jax.tree_util.tree_map(jnp.asarray, pf)

    # 본 실행과 같은 판정 (재현 확인)
    out, diag, num, b0, b1 = S1.evaluate(p, pf, dev, sT)
    res, pairs = S1.summarize(out, b0, b1)
    log("repro b1 " + " ".join(f"{k} {v['b1']:.3f}" for k, v in res.items()))

    W = len(b0)
    role = role_table(range(1000, 1000 + a.n_eval), D)                              # [W, 5]
    wi = np.arange(W)

    def by_role(choice):
        r = role[wi, choice]
        return {ROLES[j]: int((r == j).sum()) for j in range(5)}
    choice_roles = {k: by_role(v["chose"]) for k, v in out.items()}

    so = jax.jit(S1.slow_outputs)(p, dev)
    fi = S1.fast_inputs(pf, dev, so)
    M_sf = jax.vmap(lambda x, y, t: S1.kalman(x, y, t, pf["kal"]["sf"])[0])(*fi["sf"])      # [W, DX, DK]
    M_rw = jax.vmap(lambda x, y, t: S1.kalman(x, y, t, pf["kal"]["rw"])[0])(*fi["rw"])      # [W, 1, DB*DX]
    k_c = S1.mlp(pf["key"], dev["rin"])                                                     # [W, 5, DK]
    hoT = jnp.broadcast_to(so["ho_T"][:, None], (W, ER.NSITE, S1.HS))
    psi_slow = S1.mlp(p["psi"], jnp.concatenate([hoT, dev["rin"]], -1))                     # [W, 5, DX]
    psi_fast = jnp.einsum("wxk,wck->wcx", M_sf, k_c)
    u_slow = S1.mlp(p["u"], so["hr_T"])                                                     # [W, DX]
    b_T = S1.mlp(pf["b"], so["hr_T"])
    u_fast = jnp.einsum("wi,wij->wj", b_T, M_rw[:, 0].reshape(W, S1.DB, S1.DX))
    uh = u_slow + u_fast
    XI = S1.XI_ITEM
    coef = jnp.linalg.pinv(XI.T)

    def w_items(Psi):
        w = jnp.einsum("ix,wcx->wci", coef, Psi)
        resid = jnp.linalg.norm(Psi - jnp.einsum("wci,ix->wcx", w, XI), axis=-1)
        return np.asarray(w), np.asarray(resid)
    ws, rs_ = w_items(psi_slow)
    wf, rf_ = w_items(psi_fast)
    wt, rt_ = w_items(psi_slow + psi_fast)
    Qs = np.asarray(jnp.einsum("wcx,wx->wc", psi_slow, uh))
    Qf = np.asarray(jnp.einsum("wcx,wx->wc", psi_fast, uh))
    lgo = np.asarray(dev["rin"][..., -1]) * 16.0
    disc = S1.GAMMA ** (lgo + ER.D_DELAY - 1)

    def role_mean(x):                                                    # x [W, 5, ...] → 역할별 평균
        return {ROLES[j]: np.asarray(x)[role == j].mean(0).round(4).tolist() for j in range(5)}

    # 빠른 읽기의 회수: 마지막으로 그 경로에 쓴 목표 대비 읽기
    es = np.asarray(dev["e_site"])
    tgt = np.asarray(fi["sf"][1])                                        # [W, NE, DX] (Y − ψ, 쓴 목표)
    rec = {}
    for name, site_arr in (("b1", np.asarray(b1)), ("b0", np.asarray(b0))):
        last = np.array([np.where(es[w] == site_arr[w])[0][-1] for w in range(W)])
        y = tgt[wi, last]
        r_ = np.asarray(psi_fast)[wi, site_arr]
        cos = (y * r_).sum(-1) / (np.linalg.norm(y, axis=-1) * np.linalg.norm(r_, axis=-1) + 1e-12)
        rec[name] = dict(cos_mean=float(cos.mean()), ratio_norm=float((np.linalg.norm(r_, axis=-1) / (np.linalg.norm(y, axis=-1) + 1e-12)).mean()))
    kn = np.asarray(k_c / jnp.linalg.norm(k_c, axis=-1, keepdims=True))
    cosk = np.einsum("wck,wdk->wcd", kn, kn)
    off = cosk[:, ~np.eye(ER.NSITE, dtype=bool)]
    key_cos = dict(mean=float(off.mean()), max_mean=float(off.max(-1).mean()), b0_b1=float(cosk[wi, b0, b1].mean()))

    # 진단용 대체 선택 (조건이 아님)
    xiB = XI[1] / jnp.linalg.norm(XI[1])

    def rate(Q):
        return float((np.asarray(Q).argmax(-1) == np.asarray(b1)).mean())
    Psi = psi_slow + psi_fast
    alt = dict(u_slow_only=rate(jnp.einsum("wcx,wx->wc", Psi, u_slow)), u_fast_only=rate(jnp.einsum("wcx,wx->wc", Psi, u_fast)),
               prefB_full=rate(Psi @ xiB), prefB_fast_only=rate(psi_fast @ xiB), prefB_slow_only=rate(psi_slow @ xiB))
    # 거리: full이 A 채집처를 골랐을 때 그곳이 b1보다 가까웠는가
    ch = out["full"]["chose"]
    chA = np.isin(role[wi, ch], [0, 1])
    nearer = float((lgo[wi, ch][chA] < lgo[wi, b1][chA]).mean()) if chA.any() else None

    rep = dict(
        repro=dict(results=res, paired=pairs),
        choice_by_role=choice_roles,
        u_items=dict(slow=np.asarray(u_slow @ XI.T).mean(0).round(4).tolist(), fast=np.asarray(u_fast @ XI.T).mean(0).round(4).tolist()),
        psi_weights_ABC=dict(slow=role_mean(ws), fast=role_mean(wf), total=role_mean(wt)),
        psi_resid_outside_items=dict(slow=role_mean(rs_), fast=role_mean(rf_)),
        Q_parts=dict(slow=role_mean(Qs), fast=role_mean(Qf)),
        len_go=role_mean(lgo), discount=role_mean(disc),
        full_chose_A=dict(frac=float(chA.mean()), A_nearer_than_b1=nearer),
        fast_recall=rec, key_cos=key_cos, alt_choice_b1=alt,
        kalman={m: np.exp(np.asarray(v)).tolist() for m, v in pf["kal"].items()},
    )
    for k in ("choice_by_role", "u_items", "full_chose_A", "fast_recall", "key_cos", "alt_choice_b1"):
        log(f"{k} {json.dumps(rep[k])}"[:480])
    for k in ("slow", "fast", "total"):
        log(f"psiw {k} {json.dumps(rep['psi_weights_ABC'][k])}"[:480])
    for k in ("slow", "fast"):
        log(f"Q {k} {json.dumps(rep['Q_parts'][k])}"[:480])
    log(f"len_go {json.dumps(rep['len_go'])}"[:480])
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
