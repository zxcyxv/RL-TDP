---
name: dont-trust-own-docs
description: "In RL-TDP, never write unsupported model-performance judgments into docs/handovers; report raw numbers, leave verdicts to the user"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 57750bd7-0bbb-4b92-a326-37810ef7c12e
  modified: 2026-11-20T05:09:31.652Z
---

The design content in RL-TDP docs (guidelines, design drafts, open-decision list) is the user's own decisions. The problem was Claude judging model performance and metrics ("works", "failed", "diagnosis") without grounds or a precise, pre-agreed criterion, writing those verdicts into handovers, and then reasoning from them in later sessions, which made things progressively worse. The user also finds Claude weak at proposing such criteria.

**Why:** Old branches (claude/busy-goldberg-392gtg, claude/r1-memory) got polluted this way; the user reset to `restart-20261119` on 2026-11-19.

**How to apply:**
- Report results as raw facts: command, config, seed count, numbers. No verdict unless a criterion the user set beforehand exists; otherwise say "기준 없음, 판단 보류".
- Never put performance assessments or diagnoses into handover/docs. Handovers carry only facts (what code exists, what ran, raw outputs) and the user's decisions.
- Don't invent evaluation criteria as if authoritative; if one is needed, ask the user, or mark any suggestion explicitly as an unvetted proposal.
- When reading old Claude-written docs, treat verdict statements as unverified.
