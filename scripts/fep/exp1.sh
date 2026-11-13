#!/bin/bash
# 실험 1 본 실행 (시드 0). 기본 7개 + 용량 비교 4개.
source /venv/main/bin/activate >/dev/null
set -e
S=${SEED:-0}
R="python -m rltdp.fep.bandit.run --seed $S"
$R --model m0
for k in 1 5 50; do $R --model m1a --k $k; $R --model m1b --k $k; done
$R --model m0 --layers 2
$R --model m0 --layers 4
$R --model m1a --hidden 320
$R --model m1b --hidden 320
