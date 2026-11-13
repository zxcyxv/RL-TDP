# 인수인계서 (2026-11-13): GPU에서 이어서 할 일

## 0. 한눈에

- 저장소는 시도 계열 셋으로 나뉜다: **fep** (자유에너지 PoC, 정리됨), **btsp** (TDP + BTSP 에이전트, 정리됨), **icrl** (문맥 안 TD의 기억, 진행 중).
- 지금 할 일: icrl의 **B 비교를 GPU에서 길게** — 플래토 관리 기억 (plateau) 대 Gated DeltaNet (gdn) 대 문맥 창 Transformer (tf), 각 2,000반복.
- 패키지 이름이 `fepoc` → `rltdp`로 바뀌었다. 예전 설치가 있으면 `pip install -e .`을 다시 한다. 명령은 모두 `python -m rltdp.<계열>.<모듈>`.

## 1. 설치와 확인

```bash
pip install -e .                 # torch ≥ 2.4 (CUDA), gymnasium 1.4.0, minigrid 3.2.0, numpy, scipy, matplotlib
python -m pytest -q              # 39개 통과 (CPU 약 1분)
```

- `data/`, `runs/`는 git에 없다. icrl과 btsp는 데이터가 필요 없다 (과제를 그 자리에서 만들거나 환경을 직접 돌린다).
- fep 실험의 데이터는 다시 모은다: `python -m rltdp.fep.collect` (MemoryS13 → `data/memory`), `python -m rltdp.fep.collect_dynobs` (`data/dynobs`).
- `scripts/fep/*.sh`는 예전 GPU 머신의 `/venv/main` 활성화 줄을 그대로 두었다.

## 2. 저장소 구조

| 경로 | 내용 |
|---|---|
| `rltdp/common/` | 계열 공통: `minigrid_env.py` (MiniGrid 래퍼, 상수), `grid.py` (격자 원-핫 유틸, MLP), `tdp_block.py` (tdp-reasoner 블록: 같은 단계 선형 어텐션 + 대역 흔적 곱 시냅스), `replay_buffer.py` |
| `rltdp/fep/` | 계열 1: `bandit/` (실험 1), `flows/` (정규화 흐름, argmax 들어 올림), `wm/` (월드 모델 M0·M1 계열, 2g의 TDP 월드 모델 `wm/tdp.py`), `agent/` (DreamerV3식 AC, KL·EFE actor, novelty), `collect*.py` (데이터 수집) |
| `rltdp/btsp/` | 계열 2: `tiny.py`, `delta.py`, `replay.py` (BTSP로 구현한 DQN), `analyze_*.py` (장 해석) |
| `rltdp/icrl/` | 계열 3: `toy.py` (실험 1~4, numpy), `arch.py` (B: 사전학습 비교, GPU 지원) |
| `tests/common`, `tests/fep` | 단위 테스트, 이론 검증 (`tests/fep/theory`) |
| `scripts/<계열>/` | 실행 스크립트 |
| `docs/<계열>/` | 계열별 설계·결과 문서. 날짜순 전체 기록은 `docs/progress.md` |

## 3. 계열별 상태

### fep (정리됨) — `docs/fep/README.md`
정확한 우도 월드 모델 M0 (조건부 흐름) 대 RSSM-VAE M1. 실험 1 (밴딧), 2a~2f (MiniGrid Memory·Dynamic-Obstacles, DreamerV3식 에이전트,
선호분포 + EFE), 2g (GRU → TDP 기억, 구현만). PoC 마무리 후 M1 + TDP + BTSP 설계 (`docs/btsp/m1_tdp_btsp_design.md`)로 넘어갔다.

### btsp (정리됨) — `docs/btsp/results.md`
LavaGap에서 BTSP 쓰기를 TDP 흔적 위에 올린 세 시도. 마지막 형태 (BTSP로 구현한 DQN v2)는 최단 경로를 배웠지만 기존 DQN 변형
(NEC, LS-DQN, EBU)을 넘는 기여가 없어 정리했다.

### icrl (진행 중) — `docs/icrl/icrl_memory_gate.md`
질문: 문맥 안 강화학습 (Wang et al. ICLR 2025)에서 문맥 창 대신 고정 크기 지속 기억을 쓰면 무엇을 얻고 잃는가, 그리고 이산 플래토가 그 기억을
어떻게 관리해야 하는가.

- 토이 1: 선형 어텐션 ICRL의 계산은 LSTD 충분통계 (A, b)에만 의존한다 → 실수 30개 순환 기억이 창 11,000개의 바닥을 넘어 TD 고정점에 도달.
- 토이 2: 보상이 바뀌면 망각이 필요하고, 극점 하나로는 적응과 정확도를 동시에 못 얻는다.
- 토이 3: 플래토 생 게이트는 해롭다 (특이·발산·편향). 할인 속도로 감쇠하는 앞쪽 흔적 (emphatic follow-on)으로 걸러야 이득.
- 토이 4: 정밀도 가중 증거 (로그 우도비 > 사전 승산비)로 터지고 긴 기억을 덮어쓰는 플래토가 모든 고정 시간척도와 온라인 TD를 이긴다.
- B (신경망, CPU 600반복, 시드 하나): 같은 구조의 대조군을 모두 이기고, Gated DeltaNet과는 전체 오차가 같다 (0.390 대 0.392).
  gdn은 전환 직후가 빠르고 plateau는 정상 상태가 정확하다. gdn은 600반복에서도 손실이 내려가던 중이었다.

## 4. GPU에서 할 일

### 4.1 B 긴 학습 (지금 할 것)

```bash
./scripts/icrl/arch_gpu.sh                                   # plateau → gdn → tf, 2,000반복, 시드 0, runs/icrl_arch_gpu/
SEEDS="0 1 2" PARALLEL=1 ./scripts/icrl/arch_gpu.sh          # 시드 셋, 동시에
MODELS="plateau gdn tf fast slow window cum" ./scripts/icrl/arch_gpu.sh    # 대조군까지
```

- 결과: `runs/icrl_arch_gpu/<model>_s<seed>.json` — `before`, `T1200`, `T3600` (각 `overall`, `after_switch`, `steady`, `plateaus_per_switch`), `log` (250반복마다 `eval`), `args`.
- 속도 (CPU 4스레드 기준): plateau 0.53 s/반복, gdn 1.7 s/반복, window·cum 0.05 s/반복, tf는 측정 전.
  모든 모델이 1,200스텝을 작은 연산으로 순차 진행하므로 GPU에서는 커널 실행 지연이 지배한다. 반복당 시간이 CPU보다 빠르지 않을 수 있다.
  GPU를 채우려면 `BATCH`를 키운다 (예: 64~128). 그때는 모든 모델을 같은 배치로 돌려 비교하고, CPU 결과 (배치 16)와는 따로 본다.
- CPU에서 멈춘 긴 학습의 일부: plateau 250반복 평가 (과제 32개) 전체 0.41, 전환 직후 0.43, 정상 상태 0.16, 전환당 플래토 0.64.

요약 보기:

```bash
python - <<'EOF'
import json, glob
for f in sorted(glob.glob("runs/icrl_arch_gpu/*.json")):
    r = json.load(open(f)); a, b = r["T1200"], r["T3600"]
    print(f"{r['model']:8s} s{r['args']['seed']}  1200: {a['overall']:.3f} {a['after_switch']:.3f} {a['steady']:.3f}  "
          f"3600: {b['overall']:.3f} {b['after_switch']:.3f} {b['steady']:.3f}")
EOF
```

### 4.2 판정

- **gdn (2,000반복)이 plateau를 전체 오차에서 분명히 앞서면**: "원리로 정한 이산 플래토 ≥ 학습된 연속 게이트"는 이 규모에서 깨진다.
  남는 주장은 파라미터 효율 (5,508 대 15,875)과 정상 상태 정확도뿐이다. 계열 3의 방향을 다시 검토한다.
- **비기거나 plateau가 앞서면**: 시드를 3개 이상으로 늘려 확인한다.
- **tf**: 정확도에서 둘 다 이기면 주장은 비용 축으로 바뀐다 (기억: 문맥 C개 토큰 대 d² + d, 스텝당 계산: C²에 비례 대 상수, 길이: 창 밖 버림 대 제한 없음).
  그 경우 tf를 작은 문맥으로도 돌려 같은 비용에서 비교한다: `CTX=50 MODELS=tf OUT=runs/icrl_arch_gpu_C50 ./scripts/icrl/arch_gpu.sh`.

## 5. 알려진 한계와 주의

- 지표: 정규화 제곱 오차 Σ(V̂ − V)² / ΣV² (1.0 = 전부 0으로 예측). 첫 전환 이후만. 전환 직후 = 전환 후 100스텝, 정상 상태 = 다음 전환 전 200스텝.
  과제별 체크포인트 평균의 중앙값 (전체), 체크포인트 중앙값의 과제 중앙값 (구간).
- 정상 상태 바닥 (약 0.2)은 상태별 5차원 특징 위 선형 TD의 근사 한계다 (참 V는 전이를 통해 다른 상태의 특징이 섞인 값). plateau의 정상 상태 우위가 이 바닥에서 나온다.
- plateau의 고정값 (nf 20, ns 200, h 10⁻³, ridge 10⁻³)은 실험 전에 정했고 튜닝하지 않았다. 덮어쓰기 결정은 경사를 받지 않는다.
- gdn은 값 입력에 "다음 관측의 기억 읽기"를 넣어 TD 꼴 계산이 가능하게 했다 (보통의 Gated DeltaNet보다 유리한 쪽).
- tf: 문맥 C = 200 (window 대조군과 같은 정보량), 2층, d = 32, 위치 = 나이 임베딩, 상태 질의 cross-attention.
- 모든 결과가 시드 하나다.

## 6. B 다음 후보

| | 내용 |
|---|---|
| 표현 표류 | 인코더가 학습 중일 때 순환 요약 (옛 좌표) 대 문맥 재인코딩. 실제 강화학습에서 순환 기억이 잃는 두 번째 경로 |
| 제어 | 작은 격자에서 결정 지점 플래토 (토이 3의 제어판), 보상 위치 전환 (토이 4의 제어판) |
| 이론 | 토이 3에서 중간 길이 흔적 (0 < β < γ)이 해로운 이유, 같은 증거로 읽기를 바꾸면 지고 쓰기를 바꾸면 이기는 이유 |
| TDP와의 관계 | TDP의 학습된 고정 시간 창 대 증거 의존 덮어쓰기 (과거 추론을 현재에 얼마나 반영할 것인가에 대한 두 답). 재귀 추론에는 외부 증거가 없어 바로 옮기기 어렵다 |
