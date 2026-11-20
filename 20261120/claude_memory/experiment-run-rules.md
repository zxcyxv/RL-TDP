---
name: experiment-run-rules
description: "RL-TDP run rules (set 2026-11-20): announce/ask before launching runs; long jobs in background; one run at a time (RAM); step checkpoints + progress logs; 50만-step budget"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 57750bd7-0bbb-4b92-a326-37810ef7c12e
  modified: 2026-11-20T14:16:48.668Z
---

Rules the user set on 2026-11-20 for running experiments:
- Don't launch experiments on your own without telling the user, even under an autonomy goal. The user objected to an unannounced 5-run chain ("돌리라고 한 기억이 없는데") and stopped everything ("싹 다 꺼"). Propose the run, and launch it when the user says so (e.g. "0.997 돌려").
- Run long jobs in the background. Never wait on them in the foreground.
- Run one job at a time. A run with `replay.size 1e6` (~15GB RAM at 1M steps) plus any other job overflowed the 30GB RAM twice (runs were Killed).
- Envs serial (`run.debug True`). 16 parallel NAVIX workers take ~17.6GB RAM.
- Save checkpoints every 100k steps (`--run.save_steps`, copies in `ckpt_steps/`) and write `progress.jsonl` (success counts and train-metric means per window). Report at each window when asked.
- Default budget is 500k steps per run.

**Why:** Unannounced or concurrent runs wasted time, caused OOM kills, and made the user lose track of what was running.

**How to apply:** Before each new run, say in one line what it changes and why, then launch it only when the user gives the go-ahead. Related: [[run-naming]], [[dont-trust-own-docs]].
