---
name: explain-concretely
description: "RL-TDP: explain mechanisms with a concrete example and standard terms in a few lines; never coin vague phrases (e.g. '넓게 올린다') or build long speculative chains"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 57750bd7-0bbb-4b92-a326-37810ef7c12e
  modified: 2026-11-20T13:58:53.235Z
---

When the user asks why something happens, answer in a few lines: the standard term, one concrete example (e.g. grid cells A/B/C), and what was measured vs. inferred. Don't coin loose phrases ("넓게 올린다"), and don't stack speculative explanations. If a claim contradicts a measurement, drop it. Don't keep defending it.

**Why:** On 2026-11-20 a coined phrase plus a speculative chain ("generalization raises Q everywhere, then zero-reward transitions erase it") took many turns to unpack. It contradicted my own measurement and missed the obvious point that value *should* rise near the goal. The user was very frustrated.

**How to apply:**
- Lead with the concrete picture.
- Label inference as inference.
- Before explaining a failure mechanism, check that it matches the logged numbers.
- Related: [[user-terminology]], [[dont-trust-own-docs]].
