#!/bin/bash
# 실험 2c 본 실행 (시드 0): M0-SQ → M1-S → M0F-S 순서
source /venv/main/bin/activate >/dev/null
set -e
for m in m0sq m1s m0fs; do
  python -m fepoc.wm.train --model $m --data data/dynobs --out runs/exp2c --seed ${SEED:-0} 2>&1 | grep --line-buffered -v Warning > runs/exp2c_${m}_s0.log
done
