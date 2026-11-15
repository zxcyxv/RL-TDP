"""지연 채집 과제: 학습된 월드 모델·키 위의 기억 내용 판정 (docs/world/README.md 12.1절).
/venv/jax/bin/python -I scripts/world/errand_learned.py docs/world/results/errand_learned_20261115.json
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
import jax
from rltdp.world import errand_learned as EL

out, NW, D = sys.argv[1], 256, 256
sessions, stack = EL.stack_sessions(range(1000, 1000 + NW), D)
plans, arrive = EL.plans_for(sessions)
b1, b0 = np.array(stack["b1"]), np.array(stack["b0"])
res = {}
trained = {}
for name, kind, use_wm, ev in (("learned_keys_WM", "learned", True, False), ("learned_keys_WM+evidence_eb", "learned", True, True)):
    if (kind, use_wm) not in trained:
        trained[(kind, use_wm)] = EL.train(kind, log=lambda m: print(m, flush=True)) if use_wm else (EL.init(jax.random.PRNGKey(0)), 1.0)
    p, vmax = trained[(kind, use_wm)]
    vbar = EL.content_var(p, vmax, kind, use_wm) if ev else None
    if ev:
        print(name, "v̄ =", round(vbar, 4), flush=True)
    one = EL.probe(p, vmax, kind, use_wm, stack, evidence=ev, vbar=vbar)
    sc_all = np.array(jax.jit(jax.vmap(one))(stack, plans, arrive))          # [NW, NSITE, 2]: (B 칸 점수, 예측 보상)
    res[name] = {}
    for j, what in enumerate(("slot_B",)):
        sc = sc_all[..., j]
        best = sc.argmax(1)
        res[name][what] = dict(b1=float(np.mean(best == b1)), b0_stale=float(np.mean(best == b0)), other=float(np.mean((best != b1) & (best != b0))),
                               score_b1=float(sc[np.arange(NW), b1].mean()), score_b0=float(sc[np.arange(NW), b0].mean()))
    print(name, res[name], flush=True)
    if kind == "learned" and not ev:                                  # 국면별 키 크기 (교사 세션 하나)
        We, ybar = EL.emb_fixed(0)
        _, _, keyf = EL.parts(p, We, ybar, kind)
        s0 = jax.tree_util.tree_map(lambda x: x[0], stack)
        O, _, _ = jax.jit(EL.observe_stream)(s0)
        kn = np.array(jax.vmap(lambda o, a: jax.numpy.linalg.norm(keyf(o, a)))(O[:-1], s0["acts"]))
        acts = np.array(s0["acts"])
        print("  키 크기: 상호작용", kn[acts == 4].mean().round(3), "| 이동", kn[acts < 4].mean().round(3), "| 대기", kn[acts == 5].mean().round(3), flush=True)
json.dump(res, open(out.replace('.json', '_heads.json'), "w"), indent=1)
