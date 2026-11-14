"""R1 점검: 조건별 베이즈 상한과 모델별 반복당 시간. python scripts/memory/smoke.py"""
import time

import torch

from rltdp.memory.eval import evaluate, rollout
from rltdp.memory.models import build
from rltdp.memory.stream import PRESETS, Stream, StreamCfg

for c in "ABC":
    cfg = StreamCfg(**PRESETS[c])
    r = evaluate(None, cfg, 256, B=256)
    print(c, {k: round(v, 3) for k, v in r.items() if k.count("/") == 1 or "noisy/" in k or "rel/1" in k})
cfg = StreamCfg(**PRESETS["C"])
for dev in ("cpu", "cuda"):
    for name in ("gdn", "kdn", "ours", "ours_fix"):
        m = build(name, cfg).to(dev)
        t = time.time()
        for i in range(2):
            loss, _ = rollout(m, Stream(cfg, 128, dev, i), 256, q_every=2)
            loss.backward()
        print(dev, name, round((time.time() - t) / 2, 2), "s/iter", round(loss.item(), 3), sum(p.numel() for p in m.parameters()))
