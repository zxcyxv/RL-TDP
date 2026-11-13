# 계열 1 (fep): 정확한 우도 월드 모델 기반 자유에너지 에이전트 PoC — 정리됨

원래 인수인계서와 사양: [`handover/`](handover/README.md) (1단계 이론, 2단계 모델·과제 사양, 원본 PDF). 실험별 경과와 수치는 `../progress.md`.

| 문서 | 내용 |
|---|---|
| `handover/` | 출발점: F = 정확한 −ln p가 되는 가역 구조, M0 (조건부 흐름) 대 M1 (RSSM-VAE), 실험 1·2 사양 |
| `agent_design_draft.md`, `agent_design_v2.md` | 에이전트 설계 초안과 2판 |
| `exp2c_predictions.md`, `exp2c_results.md` | 실험 2c: Dynamic-Obstacles 8×8 월드 모델 비교 (사전 예측 → 결과) |
| `exp2d_design.md` | 실험 2d: 월드 모델 + 상상 속 actor-critic 온라인 학습 |
| `exp2e_predictions.md`, `exp2e_results.md` | 실험 2e: 같은 DreamerV3식 에이전트, 월드 모델만 교체 |
| `exp2f_design.md` | 실험 2f: 선호분포 C + 기대 자유에너지 G (ActorCriticKL + 공통 novelty) |
| `memory_energy_theory.md` | F를 연상기억 에너지로 보는 이론 (방안 B), TDP 선형 기억 = 고전 홉필드 근사 |
| `aif_integration.md` | 월드 모델과 정책을 능동적 추론 그림으로 통합하는 안 |

코드: `rltdp/fep/` (`bandit/`, `flows/`, `wm/`, `agent/`, `collect*.py`), 실행: `scripts/fep/`, 테스트: `tests/fep/`.
