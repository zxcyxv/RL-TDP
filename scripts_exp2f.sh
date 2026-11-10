#!/bin/bash
# 실험 2f: 선호분포 C + 기대 자유에너지 G (ActorCriticKL + 공통 novelty). 설계: docs/exp2f_design.md
# 기본: M0-SQ, β = 1(주) 다음 β = 0(대조), 환경 15,000스텝, 시드 0
source /venv/main/bin/activate >/dev/null
mkdir -p runs
for m in ${MODELS:-m0sq}; do
  for b in ${BETAS:-1 0}; do
    python -m fepoc.agent.run --model $m --actor efe --beta $b --ent ${ENT:-3e-4} --env_steps ${STEPS:-15000} \
      --seed ${SEED:-0} --out runs/exp2f 2>&1 | grep --line-buffered -v Warning > runs/exp2f_${m}_b${b}.log
  done
done
