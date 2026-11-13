#!/bin/bash
# 실행 중인 학습(구버전 train.py)이 덮어쓰는 model.pt를 평가 시점마다 ckpt/u<update>.pt로 보관한다.
D=$1; mkdir -p $D/ckpt; last=""
while true; do
  if [ -f $D/evals.json ]; then
    U=$(python3 -c "import json;print(json.load(open('$D/evals.json'))[-1]['update'])" 2>/dev/null)
    if [ -n "$U" ] && [ "$U" != "$last" ]; then sleep 5; cp $D/model.pt $D/ckpt/u$(printf %06d $U).pt; last=$U; echo "saved $U"; fi
    [ "$U" = "50000" ] && exit 0
  fi
  [ -f $D/summary.json ] && exit 0
  sleep 20
done
