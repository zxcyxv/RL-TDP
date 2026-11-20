#!/bin/bash
# Print each new progress.jsonl line of a run as it appears; exit when the run ends.
L=$1
seen=0
while true; do
  if [ -f $L/progress.jsonl ]; then
    n=$(wc -l < $L/progress.jsonl)
    if [ $n -gt $seen ]; then
      tail -n $((n - seen)) $L/progress.jsonl | python3 -c "
import sys, json
for l in sys.stdin:
  r = json.loads(l)
  if not r['episodes']: continue
  g = lambda k: r.get('train/qimag/' + k, float('nan'))
  print(f\"PROGRESS {r['step']}: {r['successes']}/{r['episodes']} len {r['mean_length']:.1f} \"
        f\"Wo {g('Wo_norm'):.3f} pimax {g('pi_maxprob'):.3f} rhatmax {g('rhat_max'):.3f} \"
        f\"Qimag {g('Q'):.4f} imag_dWo_proj {g('dWo_proj'):.2e}\", flush=True)
"
      seen=$n
    fi
  fi
  pgrep -f "dreamerv3/[m]ain.py" >/dev/null || { tail -1 $L/qstats.jsonl 2>/dev/null; echo DONE; exit 0; }
  sleep 10
done
