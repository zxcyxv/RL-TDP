#!/bin/bash
# R1: 조건마다 모델 4개를 한 묶음으로 같이 돌린다 (시드 1개). 설계와 사전 예측: docs/memory/r1_predictions.md
#   ./scripts/memory/r1.sh                      조건 A → B → C, 모델 gdn kdn ours ours_fix
#   CONDS=C MODELS="gdn kdn" READ=1 ./scripts/memory/r1.sh     2단계: 게이트가 읽기를 보는 변형
set -u
PY=${PY:-python}; OUT=${OUT:-runs/memory}; mkdir -p $OUT
for c in ${CONDS:-A B C}; do
  iters=${ITERS:-1500}; [ "$c" = "A" ] && iters=${ITERS_A:-600}
  for m in ${MODELS:-gdn kdn ours ours_fix}; do
    tag=${c}_${m}; flag=""
    [ "${READ:-0}" = "1" ] && { tag=${tag}_read; flag="--read_gate"; }
    ( $PY -m rltdp.memory.train --model $m --cond $c $flag --iters $iters --B ${BATCH:-512} --out $OUT > $OUT/$tag.log 2>&1 \
      && $PY -m rltdp.memory.eval --ckpt $OUT/$tag.pt >> $OUT/$tag.log 2>&1 || echo "FAILED $tag" ) &
  done
  wait
  echo "done $c"
done
