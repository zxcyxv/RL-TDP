#!/bin/bash
# Paired comparison of decision parts on a trained world model.
# World model (enc, dyn, dec, rew, con) from the 12:53 baseline at 600k steps,
# decision part fresh. One run at a time (RAM).
set -u
cd /workspace/ext/dreamerv3
source /venv/main/bin/activate >/dev/null
CK=/workspace/ext/logs/20261120_1253_dreamer_dk8/ckpt_steps/00600000
COMMON="--configs size1m --task navix_DoorKey-8x8-v0 --agent.dyn.rssm.unroll True \
  --run.steps 200000 --run.train_ratio 64 --replay.size 1e6 --run.log_every 120 \
  --run.save_steps 50000 --run.from_checkpoint $CK \
  --run.from_checkpoint_regex ^(enc|dyn|dec|rew|con)/"
run() {
  tag=$1; shift
  L=/workspace/ext/logs/20261120_$(date +%H%M)_${tag}
  echo "start $(basename $L) $(date +%T)"
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 python dreamerv3/main.py --logdir $L $COMMON "$@" > $L.out 2>&1
  echo "end $(basename $L) exit $? $(date +%T)"
}
run dreamer_dk8_wm600k
run q_dk8_wm600k_default --agent_type q
run q_dk8_wm600k_tn_eta02_beta03 --agent_type q --agent.q.trace_norm True --agent.q.eta 0.2 --agent.q.beta 0.3
run q_dk8_wm600k_tn_eta02_beta003 --agent_type q --agent.q.trace_norm True --agent.q.eta 0.2 --agent.q.beta 0.03
run q_dk8_wm600k_imag10 --agent_type q --agent.q.imag_eta 0.5 --agent.q.imag_beta 0.5
