"""실험 2d: 월드 모델 + 상상 속 actor-critic, 온라인 학습. 설계: docs/fep/exp2d_design.md

python -m rltdp.fep.agent.run --model m1s --beta 0 --env_steps 50000 --seed 0
python -m rltdp.fep.agent.run --model m1s --beta 0 --env_steps 3000 --tag smoke     # 시간 측정
python -m rltdp.fep.agent.run --model m0sq --actor efe --beta 1 --env_steps 15000    # 실험 2f (docs/fep/exp2f_design.md)
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from rltdp.fep.collect_dynobs import ENV_ID, n_balls_visible
from minigrid.core.constants import OBJECT_TO_IDX

from rltdp.common.minigrid_env import FAILURE, N_ACTIONS, PROGRESS, SUCCESS, MemoryEnv

FREE = (OBJECT_TO_IDX["empty"], OBJECT_TO_IDX["goal"])


def front_blocked(grid):
    """자기중심 시야에서 바로 앞 칸 (3, 5)가 벽·공 등으로 막혔는가. 충돌 판정과 정확히 같다."""
    return int(grid[3, 5, 0]) not in FREE
from rltdp.common.replay_buffer import Buffer  # noqa: F401 (재수출)
from rltdp.fep.agent.ac import ActorCritic, ActorCriticKL
from rltdp.fep.agent.dv3 import DreamerV3AC
from rltdp.fep.agent.efe import ActorCriticEFE
from rltdp.fep.wm.common import grid_onehot
from rltdp.fep.wm.models import build


@torch.no_grad()
def run_episode(env, seed, wm, ac, dev, random_policy=False, rng=None, ent_log=None):
    o = env.reset(seed)
    h = torch.zeros(1, 256, device=dev)
    ep = dict(grid=[], dir=[], outcome=[], action=[], segment=[])
    done = False
    while True:
        ep["grid"].append(o["grid"]); ep["dir"].append(o["dir"]); ep["outcome"].append(o["outcome"])
        ep["segment"].append(min(n_balls_visible(o["grid"]), 2))
        if done:
            ep["action"].append(0)
            ep["truncated"] = o["outcome"] == PROGRESS
            return ep
        g = torch.as_tensor(o["grid"], device=dev).long()[None]
        d = torch.tensor([o["dir"]], device=dev)
        e = wm.sk.embed(grid_onehot(g), d)
        if random_policy:
            a = int(rng.integers(N_ACTIONS))
        else:
            pol = ac.policy(torch.cat([h, e], -1))
            a = int(pol.sample().item())
            if ent_log is not None:
                ent_log[front_blocked(o["grid"])].append(pol.entropy().item())
        h = wm.sk.step(h, e, torch.tensor([a], device=dev))
        ep["action"].append(a)
        o, _, done = env.step(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["m0fs", "m0sq", "m1s"], required=True)
    ap.add_argument("--beta", type=float, default=0.0, help="IG 계수 (efe: novelty 계수, 0이면 novelty 모듈 없음)")
    ap.add_argument("--ent", type=float, default=3e-4, help="정책 엔트로피 계수 = 1/γ_π (nats)")
    ap.add_argument("--actor", choices=["reinforce", "kl", "dv3", "efe"], default="kl")
    ap.add_argument("--pref", default="0,1,-1", help="efe: 결과(진행, 성공, 실패)의 상대 로그 선호 c. C = softmax(c/λ_C)")
    ap.add_argument("--lam_c", type=float, default=1.0, help="efe: 선호 정밀도의 역수 λ_C")
    ap.add_argument("--nov_k", type=int, default=5, help="efe: novelty 앙상블 크기 K")
    ap.add_argument("--nov_mode", choices=["joint", "cellsum"], default="joint",
                    help="efe: joint = 결합 정보량 불편 추정 / cellsum = 칸별 JSD 합 (상한)")
    ap.add_argument("--nov_samples", type=int, default=1, help="efe: joint 추정의 멤버당 샘플 수")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--env_steps", type=int, default=50_000)
    ap.add_argument("--prefill_eps", type=int, default=100)
    ap.add_argument("--train_every", type=int, default=2)
    ap.add_argument("--starts", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/exp2d")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    env = MemoryEnv(ENV_ID, truncation_as_failure=False)
    wm = build(args.model).to(dev)
    opt = torch.optim.Adam(wm.parameters(), lr=3e-4)
    if args.actor == "dv3":
        ac = DreamerV3AC().to(dev)
    elif args.actor == "kl":
        ac = ActorCriticKL(ent=args.ent).to(dev)
    elif args.actor == "efe":
        ac = ActorCriticEFE(ent=args.ent, beta=args.beta, pref=tuple(float(v) for v in args.pref.split(",")),
                            lam_c=args.lam_c, nov_k=args.nov_k, nov_mode=args.nov_mode,
                            nov_samples=args.nov_samples).to(dev)
    else:
        ac = ActorCritic(beta=args.beta, ent=args.ent).to(dev)
    buf = Buffer(args.env_steps + 50_000, dev)
    name = f"{args.model}_{args.actor}_b{args.beta:g}_e{args.ent:g}_s{args.seed}"
    if args.actor == "efe":
        name += f"_lc{args.lam_c:g}" + (f"_{args.nov_mode}" if args.beta else "")
    name += f"_{args.tag}" if args.tag else ""
    out = Path(args.out) / name; out.mkdir(parents=True, exist_ok=True)

    env_seed = 10_000_000 * (args.seed + 1)
    for _ in range(args.prefill_eps):
        buf.add(run_episode(env, env_seed, wm, ac, dev, random_policy=True, rng=rng)); env_seed += 1
    print(f"{name}: 버퍼 시작 {buf.n} 스텝", flush=True)

    log, recent = [], deque(maxlen=50)
    ent_log = {0: deque(maxlen=2000), 1: deque(maxlen=2000)}     # 앞이 빈 상태 / 막힌 상태
    steps, updates, t0, t_upd = 0, 0, time.time(), 0.0
    while steps < args.env_steps:
        ep = run_episode(env, env_seed, wm, ac, dev, ent_log=ent_log); env_seed += 1
        trunc = ep.pop("truncated")
        buf.add(ep)
        L = len(ep["dir"]) - 1
        steps += L
        recent.append((ep["outcome"][-1] == SUCCESS, L, ep["outcome"][-1] == FAILURE, trunc))
        # 환경 train_every 스텝마다 갱신 1회 (에피소드가 끝날 때 몰아서)
        for _ in range(max(1, L // args.train_every)):
            ts = time.time()
            batch = buf.sample(32, 32, gen)
            loss, _ = wm.loss(batch)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 100); opt.step()
            with torch.no_grad():
                h, e_all = wm.sk.rollout(grid_onehot(batch["grid"]), batch["dir"], batch["action"])
            nov_loss = ac.novelty.train_step(batch) if getattr(ac, "novelty", None) is not None else None
            if args.actor == "dv3":
                diag = ac.update(wm, h, e_all, batch)
                updates += 1; t_upd += time.time() - ts
                continue
            ok = batch["mask"] & (batch["outcome"] == PROGRESS)
            idx = ok.nonzero()
            idx = idx[torch.randperm(len(idx), device=dev, generator=gen)[:args.starts]]
            b, t = idx[:, 0], idx[:, 1]
            diag = ac.update(wm, h[b, t], batch["grid"][b, t], batch["dir"][b, t])
            if nov_loss is not None:
                diag["nov_loss"] = nov_loss
            updates += 1; t_upd += time.time() - ts
        if len(log) == 0 or steps // 1000 > log[-1]["steps"] // 1000:
            sr = float(np.mean([r[0] for r in recent])); ln = float(np.mean([r[1] for r in recent]))
            rec = dict(steps=steps, updates=updates, episodes=env_seed - 10_000_000 * (args.seed + 1),
                       success=sr, collision=float(np.mean([r[2] for r in recent])),
                       truncated=float(np.mean([r[3] for r in recent])), ep_len=ln,
                       ent_free=float(np.mean(ent_log[0])) if ent_log[0] else float("nan"),
                       ent_blocked=float(np.mean(ent_log[1])) if ent_log[1] else float("nan"),
                       wm_loss=loss.item(), **diag,
                       sec=time.time() - t0, ms_per_update=1000 * t_upd / max(1, updates))
            log.append(rec)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            (out / "log.json").write_text(json.dumps(log))
    torch.save(dict(wm=wm.state_dict(), ac=ac.state_dict()), out / "final.pt")
    (out / "log.json").write_text(json.dumps(log))
    print(f"완료: {steps} 스텝, 갱신 {updates}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
