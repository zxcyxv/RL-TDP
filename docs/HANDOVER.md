# 인수인계서

## 0. 한눈에

- **목표**: 마인크래프트에서 실시간으로 적응하는 에이전트. 실질 도달점은 DreamerV3의 마인크래프트 다이아몬드 설정에서 같은 조건으로 DreamerV3를 넘는 것이다.
- **핵심 가설**: 기억이 새롭고, 믿을 만하고, 관련 있는 입력만 쓰고 (쓰기), 증거가 있을 때만 버리고 (망각), 필요한 것만 읽으면 (읽기), 같은 크기의 기억이 사건 단위로 훨씬 먼 과거까지 쓸모 있게 된다.
- **틀**: 자유에너지원리와 능동적 추론 (쓰기 = 정밀도 가중 믿음 갱신, 기억 행렬에 대한 칼만 필터). 엔지니어링 뼈대와 비교군은 DreamerV3.
- **단계**: R1 쓰기 → R2 유지·망각 → R3 읽기 → R4 월드 모델 결합 (Crafter / Craftax) → R5 마인크래프트. 상세는 [`ROADMAP.md`](ROADMAP.md).
- **지금 할 일**: R1 — 방해가 섞인 흐름형 연상 회상 과제를 만들고, 쓰기 규칙을 비교군과 비교해 입증한다 (4절).

## 1. 설치와 확인

```bash
pip install -e .                 # torch ≥ 2.4, gymnasium 1.4.0, minigrid 3.2.0, numpy, scipy, matplotlib
python -m pytest -q              # 39개 통과 (CPU 약 1분)
```

`data/`, `runs/`는 git에 없다. R1~R3은 합성 과제라 데이터가 필요 없다.

## 2. 저장소 구조

| 경로 | 내용 |
|---|---|
| `rltdp/common/` | 공통: MiniGrid 래퍼 (`minigrid_env.py`), 격자 유틸·MLP (`grid.py`), tdp-reasoner 블록 (`tdp_block.py`), 재현 버퍼 (`replay_buffer.py`) |
| `rltdp/fep/` | 월드 모델과 능동적 추론 에이전트 구현: 정확한 우도 월드 모델 (조건부 흐름) 대 RSSM-VAE, DreamerV3식 상상 속 actor-critic, 선호분포·기대 자유에너지 actor, novelty (`bandit/`, `flows/`, `wm/`, `agent/`) |
| `rltdp/btsp/` | TDP 흔적 위의 가소성 규칙으로 학습하는 에이전트 (LavaGap) |
| `rltdp/icrl/` | 고정 크기 기억의 토이 실험 (`toy.py`: 충분통계, 망각, 플래토 게이트)과 사전학습 비교 (`arch.py`: 정밀도 가중 플래토 기억, Gated DeltaNet, 문맥 창 Transformer) |
| `tests/` | 단위 테스트, 이론 검증 |
| `scripts/` | 실행 스크립트 |
| `docs/` | [`ROADMAP.md`](ROADMAP.md) (목표와 단계), 이 문서, 주제별 문서 (`fep/`, `btsp/`, `icrl/`), 날짜순 기록 (`progress.md`) |

R1 설계에 직접 쓰이는 근거는 `docs/icrl/icrl_memory_gate.md`에 있다 (ROADMAP 5절 요약).

## 3. 결정해 둔 것

- 비교군은 DreamerV3. 공식 코드 (JAX) 그대로를 기준으로, 기억 모듈만 RSSM 결정론 경로에 끼운다.
- 쓰기 규칙의 출발 형태는 칼만 쓰기 + 관련성 흔적. 자유 파라미터는 실행 전에 고정한다.
- 주장에는 사전 예측 문서, 시드 여러 개, 가장 단순한 비교군과 가장 강한 비교군 (정답 게이트 상한 포함)을 붙인다.
- 실패하면 문턱이나 손실을 덧붙이지 않고 요구의 형식화로 돌아간다.

## 4. 지금 할 일: R1

### 4.1 구현

새 패키지 `rltdp/memory/`에 둔다 (제안):

| 파일 | 내용 |
|---|---|
| `stream.py` | 과제 생성기. 손잡이: 중요 비율 ρ, 방해 구성 (잡음 / 반복 / 무관한 새것), 종류별 잡음 수준·중요도, 결과 (보상 토큰) 지연, 질의 지연 분포 (학습 길이의 수십 배까지), 흐름 길이 |
| `models.py` | 같은 상태 크기의 기억들: Hebb 덧셈, 델타 규칙, Gated DeltaNet (학습 게이트), 칼만 쓰기 (+ 관련성 흔적), 정답 게이트 |
| `train.py` | 느린 가중치 (키·값 사상, 종류 정밀도·관련성 예측기) 사전학습. Gated DeltaNet은 짧은 조각으로 학습 |
| `eval.py` | 지연별 회상 정확도, 회상 거리, ρ·방해 구성·기억 크기·흐름 길이 훑기, 제거 실험 |

참고할 기존 구현: Gated DeltaNet과 델타 쓰기 (`rltdp/icrl/arch.py`), 정밀도 가중 로그 우도비와 사건 뒤 흔적 (`rltdp/icrl/toy.py`).
외부 참고: MQAR 과제 정의 (Zoology), memax의 Gated DeltaNet (`memax/equinox/semigroups/gdn.py`, 결합 스캔).

### 4.2 순서

1. 선행 연구 확인: 연관 기억의 칼만 / RLS 해석 (ROADMAP 2절).
2. 사전 예측 문서 `docs/memory/r1_predictions.md`를 실행 전에 쓰고 고정한다 (ROADMAP 4절의 예측 표가 출발점).
3. **R1-a**: 관련성 방해 없이 잡음·반복만. 칼만 쓰기 대 Hebb·델타·Gated DeltaNet·정답 게이트.
4. **R1-b**: 무관한 새것을 더하고 관련성 흔적을 켠다. 회상 거리 ∝ 1/ρ와 정답 게이트 근접을 본다.
5. 제거 실험 (정밀도·새로움·관련성), 길이 불변성 (학습 길이의 수십 배).

### 4.3 성공 기준

- 잡음·반복 방해를 늘려도 칼만 쓰기의 회상 거리가 유지되고, 델타·Hebb은 예측대로 무너진다 (R1-a).
- 무관한 새것 속에서 회상 거리가 1/ρ에 비례하고 정답 게이트 상한에 가깝다 (R1-b).
- 각 인자를 빼면 해당 방해에서 무너진다.
- 학습 길이의 수십 배에서 곡선이 변하지 않고, Gated DeltaNet은 학습 길이 밖에서 회상 거리가 준다.

연산: 합성 과제라 CPU로 시작할 수 있다. 훑기 (ρ × 방해 × 기억 크기 × 시드)는 GPU에서 병렬로.

## 5. 그다음 (R2~R5)

- **R2 유지·망각**: R1 과제에 "일부 연합만 단서 없이 값이 바뀜"을 더한다. 키 방향별 변동성이 바뀐 것만 고치고 나머지를 보존하는지, 정답 초기화 상한과 비교.
- **R3 읽기**: 저장 항목이 많을 때의 읽기 간섭.
- **R4 월드 모델**: DreamerV3 코드의 RSSM 결정론 경로에 기억을 넣어 Crafter / Craftax-classic 100만 스텝. 비교 수치: Craftax-classic에서 DreamerV3 53.2%, 개선된 TWM 약 67~70%, 숙련된 사람 65%.
- **R5 마인크래프트**: DreamerV3 다이아몬드 설정 (`dreamerv3/configs.yaml`의 `minecraft` = `minecraft_diamond`, 기본 run 설정: 200M급 모델, bf16, 배치 16 × 64, train_ratio 32, 환경 16개, 1억 스텝).
  - 원 논문: A100 하나로 9일. RTX 5090 하나면 7~12일로 추정 (처음 1~2시간의 초당 갱신 수로 다시 잴 것).
  - 시스템 RAM: 기본 재현 버퍼 500만 스텝의 이미지만 약 61GB. CPU: MineRL 인스턴스 16개에 코어 약 16개, 인스턴스당 2~3GB. MineRL은 Java 8, Malmo 빌드, 화면 없는 서버에서는 가상 디스플레이가 필요할 수 있다.
