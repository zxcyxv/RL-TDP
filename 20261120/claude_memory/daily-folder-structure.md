---
name: daily-folder-structure
description: "RL-TDP project management rule (set 2026-11-20): shared utils folder(s) + one folder per day; records and handover are cut per day"
metadata:
  node_type: memory
  type: project
  originSessionId: 57750bd7-0bbb-4b92-a326-37810ef7c12e
  modified: 2026-11-20T05:13:33.416Z
---

From 2026-11-20 the user manages RL-TDP as: shared utility folder(s) (not necessarily just one), plus a new folder for each day. Each day's records and handover live in that day's folder and are cut off at the day boundary.

**Why:** Keeps each day's notes self-contained so Claude-written content doesn't accumulate and compound across sessions (see [[dont-trust-own-docs]]).

**How to apply:** Put the day's work, logs and handover in that day's folder; only reusable code goes in the utility folder(s). Exact folder names and placement of existing files: follow the user's instructions, don't decide unilaterally.
