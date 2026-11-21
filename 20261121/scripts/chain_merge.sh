#!/bin/bash
# Task-boundary consolidation of W_g (Codex proposal, 중간보고 6-7절).
# Start: 0412 at 1.8M. World model frozen (train_ratio 0), no imagination.
# 1. DoorKey frozen eval, collecting C_A (teacher W_A = loaded W_g)
# 2. Unlock learning 500k, W_o frozen, no P (as 0540), collecting C_B
# 3. teacher W_B: frozen evals on Unlock and DoorKey
# 4. merge: W_g C = B, C = C_A + C_B, B = W_A C_A + W_B C_B
# 5. merged: frozen evals on DoorKey and Unlock
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
  LAST=$L
}
run merge_eval_dk8_teacherA --task navix_DoorKey-8x8-v0 --run.steps 100000 --agent.q.frozen True --agent.q.collect_xx True --run.from_checkpoint $CK
LA=$LAST
run merge_learn_unlock --task navix_Unlock-v0 --run.steps 500000 --agent.q.freeze_wo True --agent.q.collect_xx True --run.from_checkpoint $CK
LB=$LAST
run merge_eval_unlock_teacherB --task navix_Unlock-v0 --run.steps 50000 --agent.q.frozen True --run.from_checkpoint $LB/ckpt_steps/00500000
run merge_eval_dk8_teacherB --task navix_DoorKey-8x8-v0 --run.steps 100000 --agent.q.frozen True --run.from_checkpoint $LB/ckpt_steps/00500000
M=/workspace/ext/logs/20261121_merged_AB_ckpt
echo "merge $(date +%T)"
python dreamerv3/merge_wg.py --ckpt $CK --xx $LA/xx_stats.npz --ckpt $LB/ckpt_steps/00500000 --xx $LB/xx_stats.npz --out $M > $M.merge.out 2>&1
echo "merge exit $? $(date +%T)"
run merge_eval_dk8_merged --task navix_DoorKey-8x8-v0 --run.steps 100000 --agent.q.frozen True --run.from_checkpoint $M
run merge_eval_unlock_merged --task navix_Unlock-v0 --run.steps 50000 --agent.q.frozen True --run.from_checkpoint $M
