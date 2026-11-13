# RL-TDP

TDP 기억 (tdp-reasoner의 쌍 기반 STDP 흔적 꼴 빠른 가중치)과 시냅스 가소성으로 강화학습을 시도한 계열들.

| 계열 | 코드 | 상태 |
|---|---|---|
| fep: 정확한 우도 월드 모델 기반 자유에너지 에이전트 PoC | `rltdp/fep/` | 정리됨 |
| btsp: TDP + BTSP 에이전트 (BTSP로 구현한 DQN까지) | `rltdp/btsp/` | 정리됨 |
| icrl: 문맥 안 TD의 기억 — 문맥 창 대 플래토로 관리하는 고정 크기 기억 | `rltdp/icrl/` | 진행 중 |

공통 모듈은 `rltdp/common/`. 문서 색인: [`docs/README.md`](docs/README.md). 현재 상태와 이어 할 일: [`docs/HANDOVER.md`](docs/HANDOVER.md).

```bash
pip install -e .
python -m pytest -q
./scripts/icrl/arch_gpu.sh     # 지금 GPU에서 돌릴 것
```
