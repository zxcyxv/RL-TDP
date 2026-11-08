"""실험 2의 두 월드 모델. 기억(Skeleton)은 같고, o_t의 밀도를 만드는 방식만 다르다.

공통 인터페이스 (2단계 문서): init_state, observe, imagine, info_gain, loss.
- M0 (CondFlowWM): 격자는 argmax 들어 올림 + h 조건부 흐름, 방향·결과는 h에서 범주형 머리.
- M1 (RSSMWM): 잠재 s ~ N(32), 디코더는 칸별 독립 범주형. KL은 해석식을 쓴다.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..env.memory import GROUP_SIZES
from ..models.argmax_lift import lift, lift_gauss, to_category
from ..models.flow import CondFlow
from .common import (CTX_PLANES, CTX_VEC, N_DIR, N_OUTCOME, Skeleton, SkipDecoder, cat_sample, ctx_flat,
                     ctx_planes, ctx_vec, grid_from_logits, grid_logprob, grid_onehot, mlp, onehot_to_idx,
                     prev_context)


def masked_mean_seq_sum(F_t, mask):
    """유효 스텝 전체에 대한 F_t의 평균 (배치·시간 평균). 문서의 Σ_t F_t를 시간 평균으로 바꿨다
    (변경 이력 2026-11-08: 클리핑 100이 시간 평균 손실 기준이라고 해석). Adam에서는 클리핑만 달라진다."""
    return (F_t * mask).sum() / mask.sum()


class CondFlowWM(nn.Module):
    name = "M0"

    def __init__(self, n_blocks=4):
        super().__init__()
        self.sk = Skeleton()
        self.flow = CondFlow(n_blocks=n_blocks)
        self.dir_head = mlp(256, N_DIR)
        self.out_head = mlp(256, N_OUTCOME)

    # --- 스텝별 항 ---
    def step_terms(self, h, grid_idx, direction, outcome, gen=None):
        """F_t와 구성 항. h (N, 256), 나머지 (N, ...)."""
        x = grid_onehot(grid_idx)
        v, lq = lift(x, GROUP_SIZES, generator=gen)
        lp_v = self.flow.log_prob(v, h)
        lp_dir = -F.cross_entropy(self.dir_head(h), direction, reduction="none")
        lp_out = -F.cross_entropy(self.out_head(h), outcome, reduction="none")
        F_grid = -(lp_v - lq)
        return dict(F=F_grid - lp_dir - lp_out, F_grid=F_grid, lp_dir=lp_dir, lp_out=lp_out)

    def loss(self, batch):
        x = grid_onehot(batch["grid"])
        h, _ = self.sk.rollout(x, batch["dir"], batch["action"])
        B, L = batch["dir"].shape
        terms = self.step_terms(h.reshape(B * L, -1), batch["grid"].reshape(B * L, 7, 7, 3),
                                batch["dir"].reshape(-1), batch["outcome"].reshape(-1))
        F_t = terms["F"].view(B, L)
        return masked_mean_seq_sum(F_t, batch["mask"].float()), F_t

    @torch.no_grad()
    def loglik_is(self, h, grid_idx, direction, outcome, S, chunk=250):
        """ln P̂(o_t | h_t)를 q(v|x)를 제안으로 하는 중요도 가중으로 추정. 반환 (N, S) 로그가중치 + 방향·결과 항."""
        x = grid_onehot(grid_idx)
        lws = []
        for s0 in range(0, S, chunk):
            k = min(chunk, S - s0)
            xk = x.unsqueeze(0).expand(k, *x.shape).reshape(-1, *x.shape[1:])
            hk = h.unsqueeze(0).expand(k, *h.shape).reshape(-1, h.shape[-1])
            v, lq = lift(xk, GROUP_SIZES)
            lws.append((self.flow.log_prob(v, hk) - lq).view(k, -1))
        lp_dir = -F.cross_entropy(self.dir_head(h), direction, reduction="none")
        lp_out = -F.cross_entropy(self.out_head(h), outcome, reduction="none")
        return torch.cat(lws, 0).T, lp_dir + lp_out

    # --- 인터페이스 ---
    def init_state(self, batch):
        return torch.zeros(batch, self.sk.h_dim, device=next(self.parameters()).device)

    @torch.no_grad()
    def predict(self, h, sample=True, gen=None):
        """h에서 다음 관측 하나를 만든다. sample=False면 기저 평균 z = μ(h)와 머리의 최빈값 (근사 최빈값)."""
        if sample:
            v, logp = self.flow.sample(h, generator=gen)
        else:
            mu, _ = self.flow.base_params(h)
            v, logp = self.flow.inverse(mu, h), None
        grid = onehot_to_idx(to_category(v, GROUP_SIZES))
        d = cat_sample(self.dir_head(h), sample, gen)
        o = cat_sample(self.out_head(h), sample, gen)
        return grid, d, o, logp

    def observe(self, h_prev, a_prev, grid_idx, direction, outcome, e_prev=None):
        raise NotImplementedError("2b에서 구현")

    def imagine_step(self, h, a, sample=True, gen=None):
        """h_t에서 ô_t를 만들고, 그것과 행동 a_t로 h_{t+1}을 만든다."""
        grid, d, o, logp = self.predict(h, sample, gen)
        e = self.sk.embed(grid_onehot(grid), d)
        return self.sk.step(h, e, a), (grid, d, o), logp

    def info_gain(self, *a, **k):
        raise NotImplementedError("2b에서 구현")


class RSSMWM(nn.Module):
    name = "M1"

    def __init__(self, s_dim=32):
        super().__init__()
        self.sk = Skeleton()
        self.s_dim = s_dim
        self.prior = mlp(256, 2 * s_dim)
        self.post = mlp(256 + 256, 2 * s_dim)
        self.dec_in = nn.Linear(s_dim + 256, 64 * 7 * 7)
        self.dec_conv = nn.Sequential(nn.ELU(), nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
                                      nn.Conv2d(64, sum(GROUP_SIZES), 3, padding=1))
        self.dir_head = mlp(s_dim + 256, N_DIR)
        self.out_head = mlp(s_dim + 256, N_OUTCOME)

    @staticmethod
    def _gauss(raw):
        mean, std_raw = raw.chunk(2, -1)
        return mean, F.softplus(std_raw) + 1e-4

    def decode(self, s, h):
        sh = torch.cat([s, h], -1)
        logits = self.dec_conv(self.dec_in(sh).view(-1, 64, 7, 7))
        return logits, self.dir_head(sh), self.out_head(sh)

    def obs_logprob(self, s, h, grid_idx, direction, outcome):
        logits, ld, lo = self.decode(s, h)
        return (grid_logprob(logits, grid_idx) - F.cross_entropy(ld, direction, reduction="none")
                - F.cross_entropy(lo, outcome, reduction="none"))

    def step_terms(self, h, e, grid_idx, direction, outcome, gen=None):
        pm, ps = self._gauss(self.prior(h))
        qm, qs = self._gauss(self.post(torch.cat([h, e], -1)))
        s = qm + qs * torch.randn(qm.shape, device=qm.device, generator=gen)
        rec = self.obs_logprob(s, h, grid_idx, direction, outcome)
        kl = (torch.log(ps / qs) + (qs**2 + (qm - pm) ** 2) / (2 * ps**2) - 0.5).sum(-1)
        return dict(F=kl - rec, kl=kl, rec=rec)

    def loss(self, batch):
        x = grid_onehot(batch["grid"])
        h, e = self.sk.rollout(x, batch["dir"], batch["action"])
        B, L = batch["dir"].shape
        terms = self.step_terms(h.reshape(B * L, -1), e.reshape(B * L, -1), batch["grid"].reshape(B * L, 7, 7, 3),
                                batch["dir"].reshape(-1), batch["outcome"].reshape(-1))
        F_t = terms["F"].view(B, L)
        return masked_mean_seq_sum(F_t, batch["mask"].float()), F_t

    @torch.no_grad()
    def loglik_is(self, h, e, grid_idx, direction, outcome, S, chunk=250):
        """ln P̂(o_t | h_t)를 q(s|h,e)를 제안으로 하는 중요도 가중으로 추정. h가 s에 의존하지 않으므로
        입자 필터가 스텝별 중요도 가중과 같다. 반환 (N, S) 로그가중치."""
        pm, ps = self._gauss(self.prior(h))
        qm, qs = self._gauss(self.post(torch.cat([h, e], -1)))
        lws = []
        for s0 in range(0, S, chunk):
            k = min(chunk, S - s0)
            eps = torch.randn(k, *qm.shape, device=qm.device)
            s = qm + qs * eps
            log_q = (-0.5 * eps**2 - torch.log(qs) - 0.5 * math.log(2 * math.pi)).sum(-1)
            log_p = (-0.5 * ((s - pm) / ps) ** 2 - torch.log(ps) - 0.5 * math.log(2 * math.pi)).sum(-1)
            rep = lambda t: t.unsqueeze(0).expand(k, *t.shape).reshape(-1, *t.shape[1:])
            rec = self.obs_logprob(s.reshape(-1, self.s_dim), rep(h), rep(grid_idx), rep(direction), rep(outcome))
            lws.append(rec.view(k, -1) + log_p - log_q)
        return torch.cat(lws, 0).T, torch.zeros(h.shape[0], device=h.device)

    def init_state(self, batch):
        return torch.zeros(batch, self.sk.h_dim, device=next(self.parameters()).device)

    @torch.no_grad()
    def predict(self, h, sample=True, gen=None):
        pm, ps = self._gauss(self.prior(h))
        s = pm + ps * torch.randn(pm.shape, device=pm.device, generator=gen) if sample else pm
        logits, ld, lo = self.decode(s, h)
        return grid_from_logits(logits, sample, gen), cat_sample(ld, sample, gen), cat_sample(lo, sample, gen), None

    def imagine_step(self, h, a, sample=True, gen=None):
        grid, d, o, _ = self.predict(h, sample, gen)
        e = self.sk.embed(grid_onehot(grid), d)
        return self.sk.step(h, e, a), (grid, d, o), None

    def observe(self, *a, **k):
        raise NotImplementedError("2b에서 구현")

    def info_gain(self, *a, **k):
        raise NotImplementedError("2b에서 구현")


class FactorizedWM(nn.Module):
    """M0F: 칸별 독립 범주형. M1에서 잠재 s만 뺀 모델이다 (디코더 구조는 M1과 같다).
    P(o_t | h_t) = Π_칸 Π_속성 P(x | h_t) · P(방향 | h_t) · P(결과 | h_t)가 정확히 계산되므로
    F_t = −ln P(o_t | h_t) = ℑ_t, 간극은 0이다. 칸 사이 상관은 h를 통해서만 표현된다."""
    name = "M0F"
    exact = True

    def __init__(self):
        super().__init__()
        self.sk = Skeleton()
        self.dec_in = nn.Linear(256, 64 * 7 * 7)
        self.dec_conv = nn.Sequential(nn.ELU(), nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
                                      nn.Conv2d(64, sum(GROUP_SIZES), 3, padding=1))
        self.dir_head = mlp(256, N_DIR)
        self.out_head = mlp(256, N_OUTCOME)

    def decode(self, h):
        return self.dec_conv(self.dec_in(h).view(-1, 64, 7, 7)), self.dir_head(h), self.out_head(h)

    def step_terms(self, h, grid_idx, direction, outcome, gen=None):
        logits, ld, lo = self.decode(h)
        lp = (grid_logprob(logits, grid_idx) - F.cross_entropy(ld, direction, reduction="none")
              - F.cross_entropy(lo, outcome, reduction="none"))
        return dict(F=-lp)

    def loss(self, batch):
        x = grid_onehot(batch["grid"])
        h, _ = self.sk.rollout(x, batch["dir"], batch["action"])
        B, L = batch["dir"].shape
        F_t = self.step_terms(h.reshape(B * L, -1), batch["grid"].reshape(B * L, 7, 7, 3),
                              batch["dir"].reshape(-1), batch["outcome"].reshape(-1))["F"].view(B, L)
        return masked_mean_seq_sum(F_t, batch["mask"].float()), F_t

    def init_state(self, batch):
        return torch.zeros(batch, self.sk.h_dim, device=next(self.parameters()).device)

    @torch.no_grad()
    def predict(self, h, sample=True, gen=None):
        logits, ld, lo = self.decode(h)
        return grid_from_logits(logits, sample, gen), cat_sample(ld, sample, gen), cat_sample(lo, sample, gen), None

    def imagine_step(self, h, a, sample=True, gen=None):
        grid, d, o, _ = self.predict(h, sample, gen)
        e = self.sk.embed(grid_onehot(grid), d)
        return self.sk.step(h, e, a), (grid, d, o), None

    def observe(self, *a, **k):
        raise NotImplementedError("2b에서 구현")

    def info_gain(self, *a, **k):
        raise NotImplementedError("2b에서 구현")


def _imagine_next(model, state, grid, d, a):
    """상상한 관측으로 다음 상태를 만든다 (기억 갱신 + 직전 문맥 교체)."""
    x = grid_onehot(grid)
    h = model.sk.step(state["h"], model.sk.embed(x, d), a)
    return dict(h=h, pg=x, pd=d, pa=a, has=torch.ones_like(state["has"]))


class FactorizedSkipWM(nn.Module):
    """M0F-S: M0F + 직전 시야 연결(SkipDecoder). 관측된 과거만 조건으로 넣으므로 우도는 여전히 정확하고 간극은 0이다."""
    name = "M0F-S"
    exact = True
    uses_ctx = True

    def __init__(self):
        super().__init__()
        self.sk = Skeleton()
        self.dec = SkipDecoder(256)

    def step_terms(self, h, ctx, grid_idx, direction, outcome, gen=None):
        logits, ld, lo = self.dec(h, ctx)
        lp = (grid_logprob(logits, grid_idx) - F.cross_entropy(ld, direction, reduction="none")
              - F.cross_entropy(lo, outcome, reduction="none"))
        return dict(F=-lp)

    def loss(self, batch):
        x = grid_onehot(batch["grid"])
        h, _ = self.sk.rollout(x, batch["dir"], batch["action"])
        ctx = ctx_flat(prev_context(x, batch["dir"], batch["action"]))
        B, L = batch["dir"].shape
        F_t = self.step_terms(h.reshape(B * L, -1), ctx, batch["grid"].reshape(B * L, 7, 7, 3),
                              batch["dir"].reshape(-1), batch["outcome"].reshape(-1))["F"].view(B, L)
        return masked_mean_seq_sum(F_t, batch["mask"].float()), F_t

    @torch.no_grad()
    def predict(self, state, sample=True, gen=None):
        logits, ld, lo = self.dec(state["h"], state)
        return grid_from_logits(logits, sample, gen), cat_sample(ld, sample, gen), cat_sample(lo, sample, gen), None

    def imagine_step(self, state, a, sample=True, gen=None):
        grid, d, o, _ = self.predict(state, sample, gen)
        return _imagine_next(self, state, grid, d, a), (grid, d, o), None


class RSSMSkipWM(RSSMWM):
    """M1-S: M1 + 같은 직전 시야 연결. 사후분포는 M1과 같이 공유 임베더 출력을 쓴다 (M1의 고유한 경로는 유지)."""
    name = "M1-S"
    uses_ctx = True

    def __init__(self, s_dim=32):
        super().__init__(s_dim)
        del self.dec_in, self.dec_conv, self.dir_head, self.out_head
        self.dec = SkipDecoder(s_dim + 256)

    def obs_logprob(self, s, h, ctx, grid_idx, direction, outcome):
        logits, ld, lo = self.dec(torch.cat([s, h], -1), ctx)
        return (grid_logprob(logits, grid_idx) - F.cross_entropy(ld, direction, reduction="none")
                - F.cross_entropy(lo, outcome, reduction="none"))

    def step_terms(self, h, e, ctx, grid_idx, direction, outcome, gen=None):
        pm, ps = self._gauss(self.prior(h))
        qm, qs = self._gauss(self.post(torch.cat([h, e], -1)))
        s = qm + qs * torch.randn(qm.shape, device=qm.device, generator=gen)
        rec = self.obs_logprob(s, h, ctx, grid_idx, direction, outcome)
        kl = (torch.log(ps / qs) + (qs**2 + (qm - pm) ** 2) / (2 * ps**2) - 0.5).sum(-1)
        return dict(F=kl - rec, kl=kl, rec=rec)

    def loss(self, batch):
        x = grid_onehot(batch["grid"])
        h, e = self.sk.rollout(x, batch["dir"], batch["action"])
        ctx = ctx_flat(prev_context(x, batch["dir"], batch["action"]))
        B, L = batch["dir"].shape
        F_t = self.step_terms(h.reshape(B * L, -1), e.reshape(B * L, -1), ctx, batch["grid"].reshape(B * L, 7, 7, 3),
                              batch["dir"].reshape(-1), batch["outcome"].reshape(-1))["F"].view(B, L)
        return masked_mean_seq_sum(F_t, batch["mask"].float()), F_t

    @torch.no_grad()
    def loglik_is(self, h, e, ctx, grid_idx, direction, outcome, S, chunk=250):
        pm, ps = self._gauss(self.prior(h))
        qm, qs = self._gauss(self.post(torch.cat([h, e], -1)))
        lws = []
        for s0 in range(0, S, chunk):
            k = min(chunk, S - s0)
            eps = torch.randn(k, *qm.shape, device=qm.device)
            s = qm + qs * eps
            log_q = (-0.5 * eps**2 - torch.log(qs) - 0.5 * math.log(2 * math.pi)).sum(-1)
            log_p = (-0.5 * ((s - pm) / ps) ** 2 - torch.log(ps) - 0.5 * math.log(2 * math.pi)).sum(-1)
            rep = lambda t: t.unsqueeze(0).expand(k, *t.shape).reshape(-1, *t.shape[1:])
            rctx = {kk: rep(v) for kk, v in ctx.items()}
            rec = self.obs_logprob(s.reshape(-1, self.s_dim), rep(h), rctx, rep(grid_idx), rep(direction), rep(outcome))
            lws.append(rec.view(k, -1) + log_p - log_q)
        return torch.cat(lws, 0).T, torch.zeros(h.shape[0], device=h.device)

    @torch.no_grad()
    def predict(self, state, sample=True, gen=None):
        h = state["h"]
        pm, ps = self._gauss(self.prior(h))
        s = pm + ps * torch.randn(pm.shape, device=pm.device, generator=gen) if sample else pm
        logits, ld, lo = self.dec(torch.cat([s, h], -1), state)
        return grid_from_logits(logits, sample, gen), cat_sample(ld, sample, gen), cat_sample(lo, sample, gen), None

    def imagine_step(self, state, a, sample=True, gen=None):
        grid, d, o, _ = self.predict(state, sample, gen)
        return _imagine_next(self, state, grid, d, a), (grid, d, o), None


class CondFlowSkipWM(nn.Module):
    """M0-S: 흐름 M0 + 직전 시야 연결. 결합층 조건과 기저분포가 직전 시야(칸 위치 그대로)를 받고,
    방향·결과 머리가 직전 방향·행동을 받는다. 들어 올림 q(v|x)는 원안 그대로(고정)."""
    name = "M0-S"
    flow_model = True
    uses_ctx = True

    def __init__(self, n_blocks=4):
        super().__init__()
        self.sk = Skeleton()
        self.flow = CondFlow(n_blocks=n_blocks, extra_ch=CTX_PLANES)
        self.dir_head = mlp(256 + CTX_VEC, N_DIR)
        self.out_head = mlp(256 + CTX_VEC, N_OUTCOME)

    def _heads(self, h, ctx):
        hv = torch.cat([h, ctx_vec(ctx)], -1)
        return self.dir_head(hv), self.out_head(hv)

    def step_terms(self, h, ctx, grid_idx, direction, outcome, gen=None):
        x = grid_onehot(grid_idx)
        v, lq = lift(x, GROUP_SIZES, generator=gen)
        lp_v = self.flow.log_prob(v, h, ctx_planes(ctx))
        ld, lo = self._heads(h, ctx)
        lp_dir = -F.cross_entropy(ld, direction, reduction="none")
        lp_out = -F.cross_entropy(lo, outcome, reduction="none")
        F_grid = -(lp_v - lq)
        return dict(F=F_grid - lp_dir - lp_out, F_grid=F_grid)

    def loss(self, batch):
        x = grid_onehot(batch["grid"])
        h, _ = self.sk.rollout(x, batch["dir"], batch["action"])
        ctx = ctx_flat(prev_context(x, batch["dir"], batch["action"]))
        B, L = batch["dir"].shape
        F_t = self.step_terms(h.reshape(B * L, -1), ctx, batch["grid"].reshape(B * L, 7, 7, 3),
                              batch["dir"].reshape(-1), batch["outcome"].reshape(-1))["F"].view(B, L)
        return masked_mean_seq_sum(F_t, batch["mask"].float()), F_t

    @torch.no_grad()
    def loglik_is(self, h, ctx, grid_idx, direction, outcome, S, chunk=250):
        x = grid_onehot(grid_idx)
        planes = ctx_planes(ctx)
        lws = []
        for s0 in range(0, S, chunk):
            k = min(chunk, S - s0)
            rep = lambda t: t.unsqueeze(0).expand(k, *t.shape).reshape(-1, *t.shape[1:])
            v, lq = lift(rep(x), GROUP_SIZES)
            lws.append((self.flow.log_prob(v, rep(h), rep(planes)) - lq).view(k, -1))
        ld, lo = self._heads(h, ctx)
        extra = -(F.cross_entropy(ld, direction, reduction="none") + F.cross_entropy(lo, outcome, reduction="none"))
        return torch.cat(lws, 0).T, extra

    @torch.no_grad()
    def predict(self, state, sample=True, gen=None):
        h, planes = state["h"], ctx_planes(state)
        if sample:
            v, _ = self.flow.sample(h, generator=gen, extra=planes)
        else:
            mu, _ = self.flow.base_params(h, planes)
            v = self.flow.inverse(mu, h, planes)
        ld, lo = self._heads(h, state)
        return onehot_to_idx(to_category(v, GROUP_SIZES)), cat_sample(ld, sample, gen), cat_sample(lo, sample, gen), None

    def imagine_step(self, state, a, sample=True, gen=None):
        grid, d, o, _ = self.predict(state, sample, gen)
        return _imagine_next(self, state, grid, d, a), (grid, d, o), None


class QNet(nn.Module):
    """학습되는 들어 올림 분포 q(u|x)의 파라미터. 입력은 실제 관측 격자 원-핫뿐이다 (논문과 같이 x에만 조건화).
    출력 층을 0으로 초기화해서 학습 시작 시점에는 고정 q (N(0, I))와 같다."""

    def __init__(self, C=20, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(C, hidden, 3, padding=1), nn.ELU(),
                                 nn.Conv2d(hidden, hidden, 3, padding=1), nn.ELU(),
                                 nn.Conv2d(hidden, 2 * C, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_oh):
        mu, log_sigma = self.net(x_oh).chunk(2, 1)
        return mu, log_sigma.clamp(-7, 3)


class CondFlowSkipLearnedQWM(CondFlowSkipWM):
    """M0-SQ: M0-S에서 고정 q를 학습되는 조건부 대각 가우시안 q(u|x)로 바꾼 것 (Argmax Flows 논문의 방식).
    q는 범주 영역 안의 위치만 다루며 숨은 원인을 추론하지 않는다."""
    name = "M0-SQ"

    def __init__(self, n_blocks=4):
        super().__init__(n_blocks)
        self.qnet = QNet()

    def _lift(self, x, gen=None):
        mu, ls = self.qnet(x)
        return lift_gauss(x, mu, ls, GROUP_SIZES, generator=gen)

    def step_terms(self, h, ctx, grid_idx, direction, outcome, gen=None):
        x = grid_onehot(grid_idx)
        v, lq = self._lift(x, gen)
        lp_v = self.flow.log_prob(v, h, ctx_planes(ctx))
        ld, lo = self._heads(h, ctx)
        lp_dir = -F.cross_entropy(ld, direction, reduction="none")
        lp_out = -F.cross_entropy(lo, outcome, reduction="none")
        F_grid = -(lp_v - lq)
        return dict(F=F_grid - lp_dir - lp_out, F_grid=F_grid)

    @torch.no_grad()
    def loglik_is(self, h, ctx, grid_idx, direction, outcome, S, chunk=250):
        x = grid_onehot(grid_idx)
        planes = ctx_planes(ctx)
        lws = []
        for s0 in range(0, S, chunk):
            k = min(chunk, S - s0)
            rep = lambda t: t.unsqueeze(0).expand(k, *t.shape).reshape(-1, *t.shape[1:])
            v, lq = self._lift(rep(x))
            lws.append((self.flow.log_prob(v, rep(h), rep(planes)) - lq).view(k, -1))
        ld, lo = self._heads(h, ctx)
        extra = -(F.cross_entropy(ld, direction, reduction="none") + F.cross_entropy(lo, outcome, reduction="none"))
        return torch.cat(lws, 0).T, extra


def build(name, **kw):
    return {"m0": CondFlowWM, "m0f": FactorizedWM, "m1": RSSMWM,
            "m0fs": FactorizedSkipWM, "m1s": RSSMSkipWM, "m0s": CondFlowSkipWM,
            "m0sq": CondFlowSkipLearnedQWM}[name](**kw)
