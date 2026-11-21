#!/bin/bash
# New-task phase on Unlock-v0, starting from the 0412 run at 1.8M steps.
# One run at a time. train_ratio 0 = no train steps: world model frozen,
# no imagination updates. Q path still updates online from real transitions
# (W_o, dsyn, dsyn -> W_g at episode end) unless agent.q.frozen True.
set -u
cd /workspace/ext/dreamerv3
source /venv/main/bin/activate >/dev/null
CK=/workspace/ext/logs/20261121_0412_q_dk8_target_tracenorm_eta02/ckpt_steps/01800000
COMMON="--configs size1m --agent.dyn.rssm.unroll True --run.train_ratio 0 \
  --replay.size 1e5 --run.log_every 120 --run.save_steps 50000 \
  --agent_type q --agent.q.tau_rel 1.0 --agent.q.trace_norm True --agent.q.eta 0.2"
run() {
  tag=$1; shift
  L=/workspace/ext/logs/20261121_$(date +%H%M)_${tag}
  echo "start $(basename $L) $(date +%T)"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 python dreamerv3/main.py --logdir $L $COMMON "$@" > $L.out 2>&1
  echo "end $(basename $L) exit $? $(date +%T)"
}
# 1. original task, frozen
run eval_dk8_before --task navix_DoorKey-8x8-v0 --run.steps 100000 --agent.q.frozen True --run.from_checkpoint $CK
# 2. new task, frozen (performance before learning it)
run eval_unlock_before --task navix_Unlock-v0 --run.steps 50000 --agent.q.frozen True --run.from_checkpoint $CK
# 3. new task, learning (Q path only)
run learn_unlock --task navix_Unlock-v0 --run.steps 200000 --run.from_checkpoint $CK
L3=$(ls -d /workspace/ext/logs/20261121_*_learn_unlock | tail -1)
# 4. original task again, frozen
run eval_dk8_after --task navix_DoorKey-8x8-v0 --run.steps 100000 --agent.q.frozen True --run.from_checkpoint $L3/ckpt_steps/00200000
