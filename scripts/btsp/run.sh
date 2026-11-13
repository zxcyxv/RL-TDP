#!/bin/bash
# 계열 2 (btsp): LavaGapS5 위의 TDP + BTSP 에이전트 (CPU, 시드 하나). 정리: docs/btsp/results.md
# 이 계열은 정리됨. 재현용으로만 둔다.
set -e
mkdir -p runs
# (1) Milstein 문턱 규칙 소형 에이전트
python -m rltdp.btsp.tiny --seed 0 | tee runs/btsp_tiny.log
# (2) 플래토 게이팅 델타 규칙 W + 순환 DQN, 게이트 끔(= DQN) 대조, 장 해석
python -m rltdp.btsp.delta --gate plateau --seed 0 | tee runs/btsp_delta_plateau.log
python -m rltdp.btsp.delta --gate off --seed 0 | tee runs/btsp_delta_off.log
python -m rltdp.btsp.analyze_delta runs/btsp_delta/plateau_s0/final.pt --seed 10000 | tee runs/btsp_delta_plateau_fields.txt
# (3) BTSP로 구현한 DQN (지각 블록 + EMA 목표망 + Double Q), Adam 대조, 장 해석
python -m rltdp.btsp.replay --rule delta --seed 0 | tee runs/btsp_replay2_delta.log
python -m rltdp.btsp.replay --rule adam --seed 0 | tee runs/btsp_replay2_adam.log
python -m rltdp.btsp.analyze_replay runs/btsp_replay_v2/delta_s0/final.pt | tee runs/btsp_replay2_fields.txt
