#!/bin/bash
# 채집 격자 3차: 키가 서로 겹치게 되는 원인을 가린다 (모두 흔적 켬, 600반복).
#   nolat       장면 벡터끼리의 오차를 손실에서 뺀다 (픽셀 복원만)
#   slow        느린 예측 경로를 둔다 (시냅스는 나머지만)
#   nolat_slow  둘 다
# 기준은 2차의 synapse. 결과: runs/world/v3_summary.txt
set -u
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/venv/jax/bin/python; OUT=runs/world
run() { $PY -m rltdp.world.train --agent synapse "$@" --iters 600 --B 256 --out $OUT > $OUT/v3_$1${2:-}.log 2>&1 || echo "FAILED $*"; }
run --no_lat & run --slow &
wait
{
  for f in synapse_nolat synapse_slow; do PYTHONPATH=. $PY scripts/world/key_overlap.py $OUT/${f}_params.npy; done
  for n in 7 11; do
    $PY -m rltdp.world.train --agent synapse --no_lat --load $OUT/synapse_nolat_params.npy --eval_N $n --eval_T 512
    $PY -m rltdp.world.train --agent synapse --slow --load $OUT/synapse_slow_params.npy --eval_N $n --eval_T 512
  done
} 2>&1 | grep -v -i warn > $OUT/v3b_summary.txt
echo finished
