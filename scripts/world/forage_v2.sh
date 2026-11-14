#!/bin/bash
# 채집 격자 2차: 보상 채널의 오차를 느린 가중치 학습에서 빼고, 키 흔적을 TDP처럼 크기 1로 맞춘 뒤.
# 흔적 켬 / 끔을 같이 학습하고, 끝나면 키 겹침과 큰 격자 평가를 잇는다. 결과: runs/world/v2_summary.txt
set -u
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/venv/jax/bin/python; OUT=runs/world
B=256 ITERS=${ITERS:-600} RUNS="synapse: synapse:--no_trace" ./scripts/world/forage.sh
{
  for f in synapse synapse_notrace; do PYTHONPATH=. $PY scripts/world/key_overlap.py $OUT/${f}_params.npy; done
  for n in 7 11 15; do
    $PY -m rltdp.world.train --agent synapse --load $OUT/synapse_params.npy --eval_N $n --eval_T 512
    $PY -m rltdp.world.train --agent synapse --no_trace --load $OUT/synapse_notrace_params.npy --eval_N $n --eval_T 512
  done
} 2>&1 | grep -v -i warn > $OUT/v2_summary.txt
echo finished
