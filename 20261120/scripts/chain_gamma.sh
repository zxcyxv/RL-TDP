#!/bin/bash
# After the paired-WM chain finishes: gamma 0.997 vs the existing gamma 0.99
# default run, same trained world model (12:53 baseline, 600k).
set -u
cd /workspace/ext/dreamerv3
source /venv/main/bin/activate >/dev/null
while pgrep -f "[c]hain_wm.sh" >/dev/null || pgrep -f "dreamerv3/[m]ain.py" >/dev/null; do sleep 15; done
CK=/workspace/ext/logs/20261120_1253_dreamer_dk8/ckpt_steps/00600000
L=/workspace/ext/logs/20261120_$(date +%H%M)_q_dk8_wm600k_gamma0997
echo "start $(basename $L) $(date +%T)"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 python dreamerv3/main.py --logdir $L \
  --configs size1m --task navix_DoorKey-8x8-v0 --agent.dyn.rssm.unroll True \
  --run.steps 200000 --run.train_ratio 64 --replay.size 1e6 --run.log_every 120 \
  --run.save_steps 50000 --run.from_checkpoint $CK \
  --run.from_checkpoint_regex '^(enc|dyn|dec|rew|con)/' \
  --agent_type q --agent.q.gamma 0.997 > $L.out 2>&1
echo "end $(basename $L) exit $? $(date +%T)"
