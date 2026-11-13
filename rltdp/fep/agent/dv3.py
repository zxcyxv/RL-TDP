"""DreamerV3 actor-critic을 공식 레포(danijar/dreamerv3, 2026-10-04 커밋 e01491f)의 agent.py / configs.yaml 기본값에
맞춰 PyTorch로 옮긴 것. 월드 모델은 우리 것(M0F-S / M0-SQ / M1-S)을 쓴다.

레포와 같은 것
- 상상: 리플레이 배치의 모든 위치에서 출발, 15스텝 (imag_last 0, imag_length 15)
- contdisc: 계속 확률 = P(종료 아님)·(1 − 1/horizon), horizon 333. 리턴 할인 disc = 1
- λ-리턴 λ = 0.95 (lambda_return 그대로), slowtar False (목표는 현재 가치망)
- 가중치 weight = cumprod(con)
- 리턴 정규화: 백분위 5~95 범위, EMA 0.01, 하한 1 (retnorm perc). advnorm·valnorm 없음
- 정책 손실 −(logπ·adv + 3e-4·H), 범주형 정책에 unimix 1%, 출력층 0.01 스케일 초기화
- 가치: symexp twohot 255칸, 출력층 0 초기화, slowreg 1.0 (느린 가치망 EMA 0.02, 매 갱신)
- 리플레이 가치 손실 repval (가중 0.3): 실제 시퀀스의 λ-리턴, 각 위치의 부트스트랩은 상상 리턴
- 최적화: lr 4e-5, AGC 0.3, eps 1e-20, 워밍업 1000

레포와 다른 것 (우리 월드 모델 구조 때문)
- 보상·계속 확률은 별도 머리 대신 월드 모델의 결과(진행/성공/실패) 예측에서 계산: r = P(성공)·1 + P(실패)·(−1)
- 정책·가치 입력은 [h_t, emb(o_t)] (세 월드 모델 공통)
- 망 크기: 3층 × 256, SiLU, RMSNorm (레포 기본 1024보다 작음)
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from rltdp.common.minigrid_env import FAILURE, N_ACTIONS, PROGRESS, SUCCESS
from rltdp.fep.agent.imagine import predict_step
from rltdp.fep.wm.common import grid_onehot

R_OUT = torch.tensor([0.0, 1.0, -1.0])           # 진행, 성공, 실패의 보상


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * torch.expm1(x.abs())


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def mlp(d_in, d_out, layers=3, units=256, outscale=1.0):
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, units), RMSNorm(units), nn.SiLU()]
        d = units
    out = nn.Linear(d, d_out)
    with torch.no_grad():
        if outscale == 0:
            out.weight.zero_(); out.bias.zero_()
        else:
            out.weight.mul_(outscale); out.bias.zero_()
    return nn.Sequential(*mods, out)


class TwoHot(nn.Module):
    """symexp twohot 회귀 머리 (255칸, symlog 공간에서 −20 ~ 20 균등)."""

    def __init__(self, d_in, bins=255):
        super().__init__()
        self.net = mlp(d_in, bins, outscale=0.0)
        self.register_buffer("bins", torch.linspace(-20, 20, bins))

    def logits(self, x):
        return self.net(x)

    def pred(self, x):
        return symexp((F.softmax(self.logits(x), -1) * self.bins).sum(-1))

    def loss(self, x, target):
        y = symlog(target).clamp(self.bins[0], self.bins[-1])
        idx = torch.searchsorted(self.bins, y.contiguous(), right=True).clamp(1, len(self.bins) - 1)
        lo, hi = self.bins[idx - 1], self.bins[idx]
        w_hi = ((y - lo) / (hi - lo)).clamp(0, 1)
        tgt = torch.zeros(*y.shape, len(self.bins), device=y.device)
        tgt.scatter_(-1, (idx - 1)[..., None], (1 - w_hi)[..., None])
        tgt.scatter_add_(-1, idx[..., None], w_hi[..., None])
        return -(tgt * F.log_softmax(self.logits(x), -1)).sum(-1)


class PercNorm:
    """retnorm: 백분위 5~95 범위의 EMA, 하한 1 (레포 Normalize impl=perc, debias False)."""

    def __init__(self, rate=0.01, limit=1.0):
        self.rate, self.limit, self.lo, self.hi = rate, limit, 0.0, 0.0

    def __call__(self, x, update=True):
        if update:
            q = torch.quantile(x.detach().flatten().float(), torch.tensor([0.05, 0.95], device=x.device))
            self.lo = (1 - self.rate) * self.lo + self.rate * q[0].item()
            self.hi = (1 - self.rate) * self.hi + self.rate * q[1].item()
        return 0.0, max(self.limit, self.hi - self.lo)


class AGCAdam:
    """Adam + 적응형 기울기 클리핑 (AGC 0.3) + 선형 워밍업 1000. 레포 opt 기본값."""

    def __init__(self, params, lr=4e-5, agc=0.3, eps=1e-20, warmup=1000):
        self.params = [p for p in params]
        self.opt = torch.optim.Adam(self.params, lr=lr, eps=eps)
        self.agc, self.lr, self.warmup, self.t = agc, lr, warmup, 0

    def step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                pn = p.norm().clamp_min(1e-3); gn = p.grad.norm()
                if gn > self.agc * pn:
                    p.grad.mul_(self.agc * pn / gn)
        self.t += 1
        for g in self.opt.param_groups:
            g["lr"] = self.lr * min(1.0, self.t / self.warmup)
        self.opt.step()


def lambda_return(last, term, rew, val, boot, disc, lam):
    """레포의 lambda_return 그대로. 입력 (N, T+1)."""
    rets = [boot[:, -1]]
    live = (1 - term)[:, 1:] * disc
    cont = (1 - last)[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return torch.stack(list(reversed(rets))[:-1], 1)


class DreamerV3AC(nn.Module):
    def __init__(self, feat_dim=512, horizon=333, imag_length=15, lam=0.95, actent=3e-4, slowreg=1.0,
                 slow_rate=0.02, repval_scale=0.3, unimix=0.01):
        super().__init__()
        self.pol = mlp(feat_dim, N_ACTIONS, outscale=0.01)
        self.val = TwoHot(feat_dim)
        self.slowval = copy.deepcopy(self.val).requires_grad_(False)
        self.H, self.lam, self.actent, self.slowreg = imag_length, lam, actent, slowreg
        self.slow_rate, self.repval_scale, self.unimix = slow_rate, repval_scale, unimix
        self.cdisc = 1 - 1 / horizon
        self.retnorm = PercNorm()
        self.opt_pol = AGCAdam(self.pol.parameters())
        self.opt_val = AGCAdam(self.val.parameters())

    def policy(self, feat):
        probs = F.softmax(self.pol(feat), -1)
        probs = (1 - self.unimix) * probs + self.unimix / N_ACTIONS
        return torch.distributions.Categorical(probs=probs)

    def act(self, feat, greedy=False):
        d = self.policy(feat)
        return d.probs.argmax(-1) if greedy else d.sample()

    @torch.no_grad()
    def imagine(self, wm, h, grid, direction, outcome):
        """출발 (h_t, o_t) N개. 반환 feat (N, H+1, D), act (N, H+1), rew (N, H+1), con (N, H+1)."""
        N, dev = h.shape[0], h.device
        r_out = R_OUT.to(dev)
        x = grid_onehot(grid); d = direction
        e = wm.sk.embed(x, d)
        feats, acts = [torch.cat([h, e], -1)], []
        rews = [r_out[outcome]]
        cons = [(outcome == PROGRESS).float() * self.cdisc]
        for _ in range(self.H):
            a = self.act(feats[-1]); acts.append(a)
            h = wm.sk.step(h, e, a)
            st = dict(h=h, pg=x, pd=d, pa=a, has=torch.ones(N, device=dev))
            p = predict_step(wm, st, need_ig=False)
            py = p["out_logp"].exp()
            rews.append((py * r_out).sum(-1))
            cons.append(py[:, PROGRESS] * self.cdisc)
            x = grid_onehot(p["grid"]); d = p["dir"]
            e = wm.sk.embed(x, d)
            feats.append(torch.cat([h, e], -1))
        acts.append(self.act(feats[-1]))
        st = lambda xs: torch.stack(xs, 1)
        return st(feats), st(acts), st(rews), st(cons)

    def update(self, wm, h_all, e_all, batch):
        """h_all, e_all: (B, T, D) 리플레이 배치의 기억과 실제 관측 임베딩 (기울기 없음)."""
        B, T = batch["dir"].shape
        valid = batch["mask"]
        flat = lambda t: t.reshape(B * T, *t.shape[2:])
        feat, act, rew, con = self.imagine(wm, flat(h_all), flat(batch["grid"]), flat(batch["dir"]),
                                           flat(batch["outcome"]))
        w = torch.cumprod(con, 1)                                  # 레포: cumprod(disc·con)/disc, disc = 1
        w = w * flat(valid.float())[:, None]
        with torch.no_grad():
            val = self.val.pred(feat)                              # slowtar False: 목표는 현재 가치망
            ret = lambda_return(torch.zeros_like(con), 1 - con, rew, val, val, 1.0, self.lam)   # (N, H)
            _, rscale = self.retnorm(ret[w[:, :-1] > 0] if (w[:, :-1] > 0).any() else ret)
            adv = (ret - val[:, :-1]) / rscale
        pol = self.policy(feat[:, :-1])
        logpi = pol.log_prob(act[:, :-1])
        ent = pol.entropy()
        loss_pol = (w[:, :-1] * -(logpi * adv + self.actent * ent)).sum() / w[:, :-1].sum().clamp_min(1)
        vf = feat[:, :-1]
        with torch.no_grad():
            slow_tgt = self.slowval.pred(vf)
        loss_val = (w[:, :-1] * (self.val.loss(vf, ret) + self.slowreg * self.val.loss(vf, slow_tgt))).sum() \
            / w[:, :-1].sum().clamp_min(1)

        # 리플레이 가치 손실 (repval): 실제 시퀀스 위의 λ-리턴, 부트스트랩 = 각 위치의 상상 리턴
        out = batch["outcome"]
        r_out = R_OUT.to(out.device)
        rrew = r_out[out]
        rterm = (out != PROGRESS).float()
        n_valid = valid.sum(1)
        ar = torch.arange(T, device=out.device)
        rlast = ((ar[None] == (n_valid - 1)[:, None]) | ~valid).float()   # 조각의 끝(에피소드 끝 또는 조각 경계)
        boot = ret[:, 0].reshape(B, T)
        rfeat = torch.cat([h_all, e_all], -1)
        with torch.no_grad():
            rval = self.val.pred(rfeat)
            rret = lambda_return(rlast, rterm, rrew, rval, boot, self.cdisc, self.lam)    # (B, T−1)
            rslow = self.slowval.pred(rfeat[:, :-1])
        rw = (1 - rlast[:, :-1]) * valid[:, :-1].float()
        loss_rep = (rw * (self.val.loss(rfeat[:, :-1], rret) + self.slowreg * self.val.loss(rfeat[:, :-1], rslow))).sum() \
            / rw.sum().clamp_min(1)

        self.opt_pol.step(loss_pol)
        self.opt_val.step(loss_val + self.repval_scale * loss_rep)
        with torch.no_grad():
            for ps, p in zip(self.slowval.parameters(), self.val.parameters()):
                ps.mul_(1 - self.slow_rate).add_(self.slow_rate * p)
        return dict(loss_pol=loss_pol.item(), loss_val=loss_val.item(), loss_rep=loss_rep.item(),
                    ent=(w[:, :-1] * ent).sum().item() / w[:, :-1].sum().clamp_min(1).item(),
                    ret=ret[:, 0].mean().item(), rscale=rscale, adv_mag=adv.abs().mean().item(),
                    con=con[:, 1:].mean().item(), ig=float("nan"), prag=float("nan"), alive=float("nan"))
