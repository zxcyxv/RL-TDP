import pytest
import torch

from rltdp.memory.eval import evaluate, rollout
from rltdp.memory.models import build
from rltdp.memory.stream import PRESETS, BayesFilter, Stream, StreamCfg


def test_stream_is_reproducible_and_scores_only_observed_keys():
    cfg = StreamCfg(**PRESETS["C"])
    a, b = Stream(cfg, 16, seed=3), Stream(cfg, 16, seed=3)
    assert (a.since == 0).all()
    for _ in range(50):
        ja, ka, oa = a.step()
        jb, kb, ob = b.step()
        assert (ja == jb).all() and (ka == kb).all() and (oa == ob).all()
    assert (a.noisy.sum(1) == 4).all()
    assert (a.kids.sort(1).values.diff(dim=1) > 0).all()          # 흐름 안에서 키가 겹치지 않는다


def test_bayes_filter_is_exact_without_noise_and_bounded_with_it():
    assert evaluate(None, StreamCfg(**PRESETS["A"]), 64, B=64)["bayes/all"] == 1.0
    r = evaluate(None, StreamCfg(**PRESETS["C"]), 256, B=128)
    assert r["bayes/rel"] == 1.0 and 0.7 < r["bayes/noisy"] < 0.95


@pytest.mark.parametrize("name,read", [("gdn", False), ("kdn", False), ("gdn", True), ("kdn", True), ("ours", False), ("ours_fix", False)])
def test_models_write_read_and_backprop(name, read):
    cfg = StreamCfg(**PRESETS["C"])
    m = build(name, cfg, read_gate=read)
    loss, _ = rollout(m, Stream(cfg, 8, seed=0), 40, q_every=2)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_ours_new_key_is_written_fully_and_reread_exactly():
    cfg = StreamCfg()
    m = build("ours", cfg)
    st = m.init(1, "cpu")
    kid, vid = torch.tensor([5]), torch.tensor([7])
    st = m.write(st, kid, vid)
    got = (st["M"] @ m.key(kid).unsqueeze(-1)).squeeze(-1)
    beta = 1.0 / (1.0 + (0.1 + m.SIG_MIN))                        # τ² = 1, u = 1, σ² = σ₀² + 하한
    assert torch.allclose(got, beta * m.ev(vid), atol=1e-5)
    info = {}
    m.write(st, kid, vid, info=info)                               # 같은 값이 다시 오면 불확실성이 줄어 있다
    assert info["u"].item() < 0.2
