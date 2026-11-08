#!/bin/bash
# 실험 2a 본 실행 (시드 0). M0 → M1 순서.
source /venv/main/bin/activate >/dev/null
set -e
S=${SEED:-0}
for m in m0 m1; do python -m fepoc.wm.train --model $m --seed $S 2>&1 | grep -v Warning; done
