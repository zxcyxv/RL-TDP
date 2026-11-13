#!/bin/bash
# 계열 3 (icrl) 토이 실험 1~4 (numpy, CPU 수십 초씩). 결과 정리: docs/icrl/icrl_memory_gate.md
set -e
mkdir -p runs
python -m rltdp.icrl.toy memory       | tee runs/icrl_toy_memory.txt          # 실험 1: 문맥 창 대 순환 기억
python -m rltdp.icrl.toy nonstat      | tee runs/icrl_toy_nonstat.txt         # 실험 2: 보상 전환, 망각
python -m rltdp.icrl.toy gate         | tee runs/icrl_toy_gate.txt            # 실험 3: 플래토 게이트 (공간 축)
python -m rltdp.icrl.toy nonstat_gate | tee runs/icrl_toy_nonstat_gate.txt    # 실험 4: 플래토 (시간 축)
