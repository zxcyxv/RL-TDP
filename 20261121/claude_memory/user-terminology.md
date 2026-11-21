---
name: user-terminology
description: "How to read the user's RL-TDP terms and summaries — by project intent, not literal textbook meaning; don't nitpick paraphrases"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 57750bd7-0bbb-4b92-a326-37810ef7c12e
  modified: 2026-11-20T06:32:42.628Z
---

Read the user's wording through the project's principles (docs/guidelines.md), not literal textbook definitions.
- "BPTT 없이" means the memory learns without backprop at all (principle 1: fast synapse, local rule) — NOT "truncated BPTT that still backprops within a step".
- When the user restates an explanation in their own words (e.g. "장면 예측 손실" for reconstruction loss on real frames), assume they mean the sensible reading. Don't "correct" wording that is right under that reading.


**Why:** On 2026-11-20 I re-read the user's correct summaries literally and "corrected" them, and answered "BPTT 없이" as truncated BPTT — both confused the user and wasted turns.

**How to apply:** Correct only a real error in substance under the intended reading; if unsure what they meant, ask briefly instead of correcting. Related: [[dont-trust-own-docs]].
