#!/bin/bash
# 조건 하나의 학습을 지켜본다: 가장 느린 모델 (ours)이 평가를 찍을 때마다 네 모델의 최신 줄을 한 줄로 요약한다.
#   ./scripts/memory/watch.sh C        (평가 4개가 모두 끝나거나 오류가 나면 종료)
c=${1:-C}; d=${OUT:-runs/memory}; last=0
while true; do
  if grep -qE "Traceback|FAILED|Error" $d/${c}_*.log 2>/dev/null; then echo "오류: $(grep -lE 'Traceback|FAILED|Error' $d/${c}_*.log | tr '\n' ' ')"; exit 1; fi
  n=$(wc -l < $d/${c}_ours.jsonl 2>/dev/null || echo 0)
  if [ "$n" -gt "$last" ]; then
    last=$n
    line="조건 $c"
    for m in gdn kdn ours_fix ours; do
      line="$line | $m $(tail -n 1 $d/${c}_$m.jsonl | sed -E 's/.*"it": ([0-9]+).*"acc": ([0-9.]{5}).*"rel": ([0-9.]{5}).*"noisy": ([0-9.]{5}).*/it\1 전체\2 깨끗\3 오염\4/')"
    done
    echo "$line"
  fi
  if [ "$(grep -l 'T=8192' $d/${c}_*.log 2>/dev/null | wc -l)" -ge 4 ]; then echo "조건 $c 평가 완료"; exit 0; fi
  sleep 10
done
