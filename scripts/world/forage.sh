#!/bin/bash
# 채집 격자: 한 번에 4개씩 같이 돌린다 (시드 1개). 설계: docs/world/model_design.md
#   ./scripts/world/forage.sh                         gru, synapse, synapse --no_trace, synapse --no_future
#   RUNS="synapse:--no_precision" ./scripts/world/forage.sh
set -u
PY=${PY:-/venv/jax/bin/python}; OUT=${OUT:-runs/world}; mkdir -p $OUT
export XLA_PYTHON_CLIENT_PREALLOCATE=false          # JAX가 GPU 메모리를 통째로 잡지 않게 (여러 개를 같이 올린다)
for spec in ${RUNS:-gru: synapse: synapse:--no_trace synapse:--no_future}; do
  agent=${spec%%:*}; flag=${spec#*:}
  tag=$agent$(echo "$flag" | sed 's/--no_/_no/; s/precision/prec/')
  $PY -m rltdp.world.train --agent $agent $flag --iters ${ITERS:-1500} --B ${B:-512} --out $OUT > $OUT/$tag.log 2>&1 || echo "FAILED $tag" &
done
wait
echo done
