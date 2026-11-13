"""상상 한 스텝: 월드 모델별로 다음 관측 샘플, 결과 분포, 정보 이득(IG)을 낸다. 설계: docs/fep/exp2d_design.md

state = dict(h, pg, pd, pa, has)  — 기억과 직전 관측 문맥 (모델의 예측 시점 상태)
반환 dict(grid, dir, out, out_logp, ig)
"""
import torch
import torch.nn.functional as F

from rltdp.common.minigrid_env import GROUP_SIZES
from rltdp.fep.flows.argmax_lift import log_q_given_gauss, to_category
from rltdp.fep.wm.common import ctx_planes, grid_from_logits, grid_logprob, grid_onehot, onehot_to_idx
from rltdp.fep.wm.models import CondFlowSkipLearnedQWM, FactorizedSkipWM, RSSMSkipWM


def _cat(logits):
    return torch.distributions.Categorical(logits=logits).sample()


@torch.no_grad()
def predict_step(model, state, need_ig=True):
    h = state["h"]
    if isinstance(model, FactorizedSkipWM):
        logits, ld, lo = model.dec(h, state)
        grid = grid_from_logits(logits, sample=True)
        ig = -grid_logprob(logits, grid) if need_ig else torch.zeros(h.shape[0], device=h.device)
    elif isinstance(model, CondFlowSkipLearnedQWM):
        v, lp = model.flow.sample(h, extra=ctx_planes(state))
        x_oh = to_category(v, GROUP_SIZES)
        if need_ig:
            mu, ls = model.qnet(x_oh)
            ig = -lp + log_q_given_gauss(v, x_oh, mu, ls, GROUP_SIZES)   # −ln p(v|h) + ln q(v|x̂)
        else:
            ig = torch.zeros(h.shape[0], device=h.device)
        grid = onehot_to_idx(x_oh)
        ld, lo = model._heads(h, state)
    elif isinstance(model, RSSMSkipWM):
        pm, ps = model._gauss(model.prior(h))
        s = pm + ps * torch.randn_like(pm)
        logits, ld, lo = model.dec(torch.cat([s, h], -1), state)
        grid = grid_from_logits(logits, sample=True)
        d_tmp = _cat(ld)
        if need_ig:
            e_hat = model.sk.embed(grid_onehot(grid), d_tmp)
            qm, qs = model._gauss(model.post(torch.cat([h, e_hat], -1)))
            ig = (torch.log(ps / qs) + (qs**2 + (qm - pm) ** 2) / (2 * ps**2) - 0.5).sum(-1)   # KL[q(s|h,ê) ‖ p(s|h)]
        else:
            ig = torch.zeros(h.shape[0], device=h.device)
        return dict(grid=grid, dir=d_tmp, out=_cat(lo), out_logp=F.log_softmax(lo, -1), ig=ig)
    else:
        raise TypeError(type(model))
    return dict(grid=grid, dir=_cat(ld), out=_cat(lo), out_logp=F.log_softmax(lo, -1), ig=ig)
