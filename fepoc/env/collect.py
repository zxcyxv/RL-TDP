"""실험 2a 데이터 수집 (단계 0).

- 학습용: 환경 시드 0..499 스크립트(ε = 0.2) + 500..999 무작위 = 1,000 에피소드
- 보류: 환경 시드 1_000_000.. 무작위 50 + 스크립트(ε = 0) 50 = 100 에피소드

python -m fepoc.env.collect --out data/memory
에피소드는 이어 붙여 저장하고 ep_start로 경계를 표시한다. 스텝 t의 관측 o_t와 그 뒤 행동 a_t를
같은 행에 둔다(마지막 관측의 행동은 -1).
"""
import argparse
from pathlib import Path

import numpy as np

from .memory import N_ACTIONS, MemoryEnv, ScriptedPolicy, segment_of

POLICY_RANDOM, POLICY_SCRIPTED = 0, 1


def run_episode(env, seed, policy, eps, rng):
    o = env.reset(seed)
    pol = ScriptedPolicy(env, eps=eps, rng=rng) if policy == POLICY_SCRIPTED else None
    if pol:
        pol.reset()
    rec = dict(grid=[], dir=[], outcome=[], action=[], segment=[])
    done = False
    while True:
        rec["grid"].append(o["grid"]); rec["dir"].append(o["dir"]); rec["outcome"].append(o["outcome"])
        rec["segment"].append(segment_of(env.pose[0]))
        if done:
            rec["action"].append(-1)
            break
        a = pol.act() if pol else int(rng.integers(N_ACTIONS))
        rec["action"].append(a)
        o, _, done = env.step(a)
    return rec


def collect(specs, rng):
    env = MemoryEnv()
    eps_list, meta = [], []
    for seed, policy, eps in specs:
        r = run_episode(env, seed, policy, eps, rng)
        eps_list.append(r)
        meta.append((seed, policy, len(r["action"]) - 1, r["outcome"][-1]))
    cat = {k: np.asarray(np.concatenate([np.asarray(r[k]) for r in eps_list]),
                         dtype=np.uint8 if k != "action" else np.int8) for k in eps_list[0]}
    lengths = np.array([len(r["action"]) for r in eps_list])
    cat["ep_start"] = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    cat["ep_len"] = lengths                       # 관측 개수 = 행동 수 + 1
    m = np.array(meta)
    cat["env_seed"], cat["policy"], cat["final_outcome"] = m[:, 0], m[:, 1], m[:, 3]
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/memory")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)

    train = [(s, POLICY_SCRIPTED, 0.2) for s in range(500)] + [(s, POLICY_RANDOM, 0.0) for s in range(500, 1000)]
    hold = [(1_000_000 + s, POLICY_RANDOM, 0.0) for s in range(50)] + \
           [(1_000_050 + s, POLICY_SCRIPTED, 0.0) for s in range(50)]
    for name, specs in [("train", train), ("holdout", hold)]:
        d = collect(specs, rng)
        np.savez_compressed(out / f"{name}.npz", **d)
        for p, pn in [(POLICY_SCRIPTED, "스크립트"), (POLICY_RANDOM, "무작위")]:
            sel = d["policy"] == p
            L = d["ep_len"][sel] - 1
            sr = np.mean(d["final_outcome"][sel] == 1)
            print(f"{name} {pn}: {sel.sum()}개, 길이 평균 {L.mean():.1f} 최대 {L.max()}, 성공률 {sr:.2f}")
        print(f"{name}: 총 관측 {len(d['grid'])}")


if __name__ == "__main__":
    main()
