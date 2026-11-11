"""실험 2a 보류 집합 평가 (그림 2 지표와 구간별 간극).

- F: 학습과 같은 단일 샘플 F_t (nats/스텝)
- ℑ̂_S: 중요도 가중 이산 로그우도의 음수, S = 100, 1000. 유효 샘플 비율(ESS/S)도 기록한다
- Â = F − ℑ̂_1000 (M0에서는 Â_deq), 구간(방/복도/갈림길)별 평균
- k스텝 예측 정확도 (k = 1, 5, 10): 참 관측으로 만든 h_t0에서 실제 행동만 넣고 굴린 뒤
  예측 격자와 실제 격자의 칸 단위 일치율 (칸의 세 속성이 모두 같아야 일치). 샘플/최빈값 두 방식.
  비교 기준 "copy": 마지막 실제 관측을 그대로 복사한 예측 (t0 ≥ 1에서만)
- 샘플 기준 정확도는 간극과 무관하다: M0에서 v ~ p(v|h)의 argmax는 이산 분포 P(x|h)의 정확한 샘플이다
평가 조각은 32스텝, 조각 시작에서 h = 0 (변경 이력 2026-11-08).
"""
import math
import time

import numpy as np
import torch

from .common import ctx_flat, ctx_select, grid_onehot, prev_context

SEGMENTS = ("room", "corridor", "junction")


@torch.no_grad()
def evaluate(model, data, L=32, S_list=(100, 1000), ks=(1, 5, 10), chunk_batch=64, frame_block=64,
             max_chunks=None):
    model.eval()
    t0 = time.time()
    starts, lens = data.chunks(L)
    if max_chunks and max_chunks < len(starts):    # 고르게 뽑는다 (보류 집합은 무작위 정책 에피소드가 앞에 몰려 있다)
        sel = torch.linspace(0, len(starts) - 1, max_chunks, device=starts.device).round().long().unique()
        starts, lens = starts[sel], lens[sel]
    S_max = max(S_list)
    rec = {k: [] for k in ["F", "seg", "lw_extra", "ess"] + [f"ll{S}" for S in S_list]}
    acc = {(k, m): [] for k in ks for m in ("sample", "mode", "copy")}
    is_m0 = model.name == "M0"
    exact = getattr(model, "exact", False)          # M0F: 우도가 정확하므로 중요도 가중이 필요 없다
    uses_ctx = getattr(model, "uses_ctx", False)    # -S 모델: 디코더가 직전 시야 문맥을 받는다
    tokens = getattr(model, "tokens", False)        # TDP 기억 (fepoc/wm/tdp.py): 상태가 토큰 [52, d] + 흔적
    chunk_batch = getattr(model, "eval_chunk_batch", chunk_batch)
    for b0 in range(0, len(starts), chunk_batch):
        batch = data.batch_from(starts[b0:b0 + chunk_batch], lens[b0:b0 + chunk_batch], L)
        x = grid_onehot(batch["grid"])
        if tokens:
            h, full = model.sk.rollout(x, batch["dir"], batch["outcome"], batch["action"], keep_state=True)
            e = h                                    # 쓰지 않는 자리 (M1 계열만 e를 쓴다)
        else:
            h, e = model.sk.rollout(x, batch["dir"], batch["action"])
        m = batch["mask"]
        hv, ev = h[m], e[m]
        gv, dv, ov = batch["grid"][m], batch["dir"][m], batch["outcome"][m]
        ctx = prev_context(x, batch["dir"], batch["action"]) if uses_ctx else None
        cv = ctx_select(ctx, m) if uses_ctx else None
        flow_ctx = uses_ctx and getattr(model, "flow_model", False)     # M0-S
        if tokens:
            F_t = model.step_terms(hv, gv, dv, ov)["F"]
        elif (uses_ctx and exact) or flow_ctx:
            F_t = model.step_terms(hv, cv, gv, dv, ov)["F"]
        elif uses_ctx:
            F_t = model.step_terms(hv, ev, cv, gv, dv, ov)["F"]
        elif is_m0 or exact:
            F_t = model.step_terms(hv, gv, dv, ov)["F"]
        else:
            F_t = model.step_terms(hv, ev, gv, dv, ov)["F"]
        rec["F"].append(F_t); rec["seg"].append(batch["segment"][m])
        if exact:
            for S in S_list:
                rec[f"ll{S}"].append(-F_t)
            rec["ess"].append(torch.ones_like(F_t))
        for f0 in range(0, 0 if exact else len(hv), frame_block):
            sl = slice(f0, f0 + frame_block)
            if tokens:
                lw, extra = model.loglik_is(hv[sl], gv[sl], dv[sl], ov[sl], S_max)
            elif is_m0:
                lw, extra = model.loglik_is(hv[sl], gv[sl], dv[sl], ov[sl], S_max)
            elif flow_ctx:
                lw, extra = model.loglik_is(hv[sl], ctx_select(cv, sl), gv[sl], dv[sl], ov[sl], S_max)
            elif uses_ctx:
                lw, extra = model.loglik_is(hv[sl], ev[sl], ctx_select(cv, sl), gv[sl], dv[sl], ov[sl], S_max)
            else:
                lw, extra = model.loglik_is(hv[sl], ev[sl], gv[sl], dv[sl], ov[sl], S_max)
            for S in S_list:
                rec[f"ll{S}"].append(torch.logsumexp(lw[:, :S], 1) - math.log(S) + extra)
            w = torch.softmax(lw, 1)
            rec["ess"].append(1.0 / (w**2).sum(1) / S_max)

        # k스텝 예측: 모든 유효 시작점 t0에서 동시에 굴린다
        B = h.shape[0]
        ar = torch.arange(L, device=h.device)
        n_valid = batch["mask"].sum(1, keepdim=True)
        # 비교 기준: 마지막 실제 관측 o_{t0-1}을 그대로 복사해 o_{t0+k-1}의 예측으로 쓴다 (t0 ≥ 1)
        prev = batch["grid"].roll(1, dims=1)
        for k in ks:
            tt = (ar[None] + k - 1).expand(B, L)
            ok = (tt < n_valid) & m & (ar[None] >= 1)
            tgt = batch["grid"].gather(1, tt.clamp(max=L - 1)[..., None, None, None].expand(-1, -1, 7, 7, 3))
            acc[(k, "copy")].append((prev == tgt).all(-1).float().mean((-1, -2))[ok])
        for mode in ("sample", "mode"):
            if tokens:
                hc = model.flat_state(full)
            else:
                hc = dict(h=h.reshape(B * L, -1), **ctx_flat(ctx)) if uses_ctx else h.reshape(B * L, -1)
            for j in range(max(ks)):
                tgt_t = (ar[None] + j).expand(B, L)                     # 예측 대상 시점 t0 + j
                ok = (tgt_t < batch["mask"].sum(1, keepdim=True)) & m
                tt = tgt_t.clamp(max=L - 1)
                tgt = batch["grid"].gather(1, tt[..., None, None, None].expand(-1, -1, 7, 7, 3))
                act = batch["action"].gather(1, tt)
                hc, (g, _, _), _ = model.imagine_step(hc, act.reshape(-1), sample=(mode == "sample"))
                if (j + 1) in ks:
                    match = (g.view(B, L, 7, 7, 3) == tgt).all(-1).float().mean((-1, -2))
                    acc[(j + 1, mode)].append(match[ok])
    cat = {k: torch.cat(v) for k, v in rec.items() if v}
    out = dict(n_frames=int(len(cat["F"])), F=cat["F"].mean().item())
    for S in S_list:
        out[f"I_hat_{S}"] = (-cat[f"ll{S}"]).mean().item()
    I_best = -cat[f"ll{S_max}"]
    gap = cat["F"] - I_best
    out["A_hat"] = gap.mean().item()
    names = getattr(data, "segment_names", SEGMENTS)
    out["A_hat_by_segment"] = {name: gap[cat["seg"] == i].mean().item() for i, name in enumerate(names)}
    out["F_by_segment"] = {name: cat["F"][cat["seg"] == i].mean().item() for i, name in enumerate(names)}
    out["I_hat_by_segment"] = {name: I_best[cat["seg"] == i].mean().item() for i, name in enumerate(names)}
    out["n_by_segment"] = {name: int((cat["seg"] == i).sum()) for i, name in enumerate(names)}
    out["ess_frac_mean"] = cat["ess"].mean().item()
    out["ess_frac_median"] = cat["ess"].median().item()
    out["acc"] = {f"k{k}_{m}": torch.cat(v).mean().item() for (k, m), v in acc.items()}
    out["acc_random_baseline"] = 1.0 / (11 * 6 * 3)
    out["eval_time_s"] = time.time() - t0
    model.train()
    return out
