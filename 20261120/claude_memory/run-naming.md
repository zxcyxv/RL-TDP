---
name: run-naming
description: "RL-TDP: don't label runs v1/v2/v3; name and refer to runs by date_starttime_agent_task (e.g. 20261120_1253_dreamer_dk8)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 57750bd7-0bbb-4b92-a326-37810ef7c12e
  modified: 2026-11-20T13:03:18.805Z
---

Don't call experiment runs v1, v2, v3 (in log dir names or when talking about them). On 2026-11-20 the user objected after "기준선 v2" made them think the architecture had changed. It was only a rerun.

**Why:** Version labels suggest that something in the model changed, and they hide what actually differs.

**How to apply:**
- Name log dirs `<YYYYMMDD>_<HHMM start>_<agent>_<task>`, e.g. `20261120_1253_dreamer_dk8`.
- When talking about a run, use its start time and say what differs ("12:53에 다시 띄운 기준선, 설정 동일").
- Related: [[dont-trust-own-docs]].
