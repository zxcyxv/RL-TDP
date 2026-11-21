#!/bin/bash
# Same as chain_unlock (learn Unlock 500k from the 0412 run at 1.8M, then
# measure DoorKey frozen), but W_o is fixed during Unlock learning
# (agent.q.freeze_wo True): only dsyn and the dsyn -> W_g transfer learn.
set -u
cd /workspace/ext/dreamerv3
source /venv/main/bin/activate >/dev/null
CK=/workspace/ext/logs/20261121_0412_q_dk8_target_tracenorm_eta02/ckpt_steps/01800000
COMMON="--configs size1m --agent.dyn.rssm.unroll True --run.train_ratio 0 \
  --replay.size 1e5 --run.log_every 120 --run.save_steps 50000 \
  --agent_type q --agent.q.tau_rel 1.0 --agent.q.trace_norm True --agent.q.eta 0.2"
L3=/workspace/ext/logs/20261121_$(date +%H%M)_learn_unlock_freezewo
echo "start $(basename $L3) $(date +%T)"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 python dreamerv3/main.py --logdir $L3 $COMMON \
  --task navix_Unlock-v0 --run.steps 500000 --agent.q.freeze_wo True \
  --run.from_checkpoint $CK > $L3.out 2>&1
echo "end $(basename $L3) exit $? $(date +%T)"
L4=/workspace/ext/logs/20261121_$(date +%H%M)_eval_dk8_after_freezewo
echo "start $(basename $L4) $(date +%T)"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 python dreamerv3/main.py --logdir $L4 $COMMON \
  --task navix_DoorKey-8x8-v0 --run.steps 100000 --agent.q.frozen True \
  --run.from_checkpoint $L3/ckpt_steps/00500000 > $L4.out 2>&1
echo "end $(basename $L4) exit $? $(date +%T)"
