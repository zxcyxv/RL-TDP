#!/bin/bash
# synapse / gru / tdp를 같은 설정으로 동시에 돌리고, 평가 기록을 docs/world/results/로 복사해 주기적으로 커밋한다.
# 실행 출력(runs/)은 git에 안 올라가므로 결과는 반드시 docs/ 쪽으로 옮긴다.
cd /workspace/RL-TDP
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=runs/world/rerun_20261114; RES=docs/world/results/rerun_20261114
mkdir -p $OUT $RES
ARGS="--iters 960 --B 40 --L 32 --eval_every 80 --eval_B 1024 --seed 0 --out $OUT"
pids=()
for a in synapse gru tdp; do
  /venv/jax/bin/python -u -m rltdp.world.train --agent $a $ARGS > $OUT/$a.log 2>&1 &
  pids+=($!)
done
sync_commit() {
  cp $OUT/*.jsonl $RES/ 2>/dev/null
  git add $RES && git -c user.name=Claude -c user.email=noreply@anthropic.com commit -q -m "world rerun 20261114: evaluation records ($1)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Jm2MnvRx4y4BKGc15zR8mZ" && echo "committed ($1)"
  git push -q origin claude/r1-memory 2>&1 | tail -1 || true
}
while kill -0 ${pids[@]} 2>/dev/null; do sleep 300; sync_commit "in progress"; done
wait; sync_commit "final"; tail -n 3 $OUT/*.log
