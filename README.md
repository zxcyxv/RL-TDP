# RL-TDP

마인크래프트에서 실시간으로 적응하는 에이전트를 목표로 한다. 기억이 새롭고, 믿을 만하고, 관련 있는 입력만 쓰고, 증거가 있을 때만 버리고,
필요한 것만 읽어서 같은 크기로 훨씬 먼 과거까지 쓸모 있게 만드는 것이 핵심이다. 틀은 자유에너지원리와 능동적 추론, 비교군은 DreamerV3.

- 목표와 단계: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- 현재 상태와 할 일: [`docs/HANDOVER.md`](docs/HANDOVER.md)
- 문서 색인: [`docs/README.md`](docs/README.md)

| 코드 | 내용 |
|---|---|
| `rltdp/common/` | 공통 모듈 (MiniGrid 래퍼, 격자 유틸, TDP 블록, 재현 버퍼) |
| `rltdp/fep/` | 월드 모델과 능동적 추론 에이전트 구현 |
| `rltdp/btsp/` | TDP 흔적 위의 가소성 규칙 에이전트 |
| `rltdp/icrl/` | 고정 크기 기억의 토이 실험과 사전학습 비교 |

```bash
pip install -e .
python -m pytest -q
```
