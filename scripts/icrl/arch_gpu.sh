#!/bin/bash
# 계열 3 (icrl) B: 플래토 관리 기억 대 Gated DeltaNet 대 문맥 창 Transformer, 긴 학습. 인수인계: docs/HANDOVER.md
# 기본: plateau → gdn → tf, 각 2,000반복, 250반복마다 평가 (과제 32개), 시드 0, GPU.
#   MODELS="plateau gdn tf fast slow window cum" SEEDS="0 1 2" ITERS=2000 BATCH=16 DEVICE=cuda ./scripts/icrl/arch_gpu.sh
#   CTX=50 MODELS=tf OUT=runs/icrl_arch_gpu_C50 ./scripts/icrl/arch_gpu.sh    (tf 문맥 길이만 바꿔서)
#   PARALLEL=1 이면 모델·시드마다 따로 띄운다 (한 GPU에 같이 올림. 각 실행은 작은 커널 위주라 GPU를 거의 못 채운다)
set -u
OUT=${OUT:-runs/icrl_arch_gpu}; mkdir -p $OUT
for s in ${SEEDS:-0}; do
  for m in ${MODELS:-plateau gdn tf}; do
    cmd="python -m rltdp.icrl.arch --model $m --seed $s --iters ${ITERS:-2000} --batch ${BATCH:-16} --eval_every 250 --device ${DEVICE:-cuda} --ctx ${CTX:-200} --out $OUT"
    if [ "${PARALLEL:-0}" = "1" ]; then
      $cmd > $OUT/${m}_s${s}.log 2>&1 &
    else
      $cmd > $OUT/${m}_s${s}.log 2>&1 || echo "FAILED $m s$s"
    fi
  done
done
wait
