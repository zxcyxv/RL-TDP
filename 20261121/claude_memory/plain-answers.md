---
name: plain-answers
description: "RL-TDP: answer only what was asked, in the question's own frame, in plain words; name each formula/symbol in words before using it; don't assume the user knows the RL notation"
metadata:
  node_type: memory
  type: feedback
  originSessionId: e7dffcb8-5d26-4c3b-82d4-a1c13982d7f5
  modified: 2026-11-21T01:39:01.981Z
---

On 2026-11-21 the user got very frustrated with my explanations of the Q-path update:
- I wrote "δ = r + γ·V(s′) − Q(s,a)" assuming they knew it was the TD error. They didn't: "난 몰랐어 저 수식이 TD오차인지 ... 니가 읽기 싫게 답변을 한거고."
- They asked a question framed by backprop (all layers updated at once with exact gradients, so why is W_o moving a problem?). I answered by repeating my earlier explanation instead of in their frame, so they had to ask again.
- I answered more than was asked ("내가 물어본 것만 대답하는게 어때").
- I listed formulas without the key fact first: the dsyn update is one-layer backprop through W_o, which conflicts with principle 1. Then I pointed at the design draft as if that excused it.

**Why:** Dense symbols with undefined terms are unreadable to them. Unasked-for extras and answers outside their frame force repeat questions.

**How to apply:**
- Lead with the direct answer to the exact question, in words.
- Before any formula, say in words what it is ("TD 오차, 즉 예측과 목표의 차이").
- Read the question in the frame the user set (e.g. backprop) and answer within it.
- Stop there. No extra options or side topics unless asked.
- Own errors plainly. Never shift blame to docs or to the user.
- Related: [[explain-concretely]], [[user-terminology]].
