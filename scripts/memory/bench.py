"""R1 속도 점검: 배치 크기별 반복당 시간 (GPU). python scripts/memory/bench.py"""
import time

import torch

from rltdp.memory.eval import rollout
from rltdp.memory.models import build
from rltdp.memory.stream import PRESETS, Stream, StreamCfg

cfg = StreamCfg(**PRESETS["C"])
m = build("ours", cfg).to("cuda")
for B in (128, 1024, 4096):
    for i in range(3):
        if i == 1:
            torch.cuda.synchronize(); t = time.time()
        loss, _ = rollout(m, Stream(cfg, B, "cuda", i), 256, q_every=2)
        loss.backward()
    torch.cuda.synchronize()
    print(f"B={B}: {(time.time() - t) / 2:.2f} s/iter, {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
