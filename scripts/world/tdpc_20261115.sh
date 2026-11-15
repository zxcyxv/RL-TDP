#!/bin/bash
# tdpc 시드 0, 1, 2를 4절과 같은 설정으로 동시에 돌리고, 평가 기록을 docs/world/results/tdpc_20261115/로 복사해 5분마다 커밋한다.
cd /workspace/RL-TDP
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=runs/world/tdpc_20261115; RES=docs/world/results/tdpc_20261115
mkdir -p $OUT $RES
pids=()
for s in 0 1 2; do
  /venv/jax/bin/python -u -m rltdp.world.train --agent tdpc --tag tdpc_s$s --seed $s --iters 960 --B 40 --L 32 --eval_every 80 --eval_B 1024 --out $OUT > $OUT/tdpc_s$s.log 2>&1 &
  pids+=($!)
done
sync_commit() {
  cp $OUT/*.jsonl $RES/ 2>/dev/null
  git add $RES && git -c user.name=Claude -c user.email=noreply@anthropic.com commit -q -m "world tdpc 20261115: evaluation records ($1)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Jm2MnvRx4y4BKGc15zR8mZ" && echo "committed ($1)"
  git push -q origin claude/r1-memory 2>&1 | tail -1 || true
}
while kill -0 ${pids[@]} 2>/dev/null; do sleep 300; sync_commit "in progress"; done
wait; sync_commit "final"
