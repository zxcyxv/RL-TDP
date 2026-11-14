#!/bin/bash
# Codex에 모델 설계 연구를 시킨다. 프롬프트: docs/world/codex_prompt_model_design.md, 산출물: docs/world/model_design.md
#   ./scripts/world/run_codex_design.sh       (모델 gpt-6-astra, effort ultra)
# 이 인스턴스는 권한 없는 컨테이너라 Codex의 샌드박스 (bwrap)가 뜨지 않는다.
# 사용자 승인 (2026-11-14)에 따라 샌드박스와 승인을 끄고 돌린다. Codex가 저장소를 직접 읽고 문서를 쓴다.
set -u
mkdir -p runs/codex
codex exec -m "${MODEL:-gpt-6-astra}" -c model_reasoning_effort="${EFFORT:-ultra}" \
  --dangerously-bypass-approvals-and-sandbox \
  -o runs/codex/model_design_last_message.md - < docs/world/codex_prompt_model_design.md > runs/codex/model_design.log 2>&1
echo "codex exit $?"
