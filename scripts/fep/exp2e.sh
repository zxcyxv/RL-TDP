#!/bin/bash
# 실험 2e: 같은 DreamerV3 에이전트, 월드 모델만 교체. 환경 15,000스텝 (M1-S는 앞 실행 기록 사용), 시드 0
source /venv/main/bin/activate >/dev/null
for m in m0sq m0fs; do
  python -m rltdp.fep.agent.run --model $m --actor dv3 --env_steps 15000 --out runs/exp2e 2>&1 | grep --line-buffered -v Warning > runs/exp2e_${m}.log
done
