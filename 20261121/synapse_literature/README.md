# 시냅스 가소성으로 파국적 망각을 막는 방법: 문헌 조사 (2026-11-21)

요청: "시냅스 가소성이 파국적 망각에 대해 어떤 해법을 제시할 수 있는지 문헌 조사" (사용자, 2026-11-21).

**읽는 법**
- 논문 내용은 초록·본문·공식 페이지에서 확인한 만큼만 적었습니다. 확인 범위가 좁은 것은 따로 표시했습니다.
- "이 프로젝트와의 대응"은 문헌의 식과 `20261121/실험기록.md`의 식·숫자를 나란히 놓은 것입니다.
  - 대수로 같은 식인지 보인 곳은 **유도**라고 표시했습니다.
  - 이 프로젝트 모델의 성능 판정은 하지 않았습니다(기준 없음).

용어:
- **파국적 망각:** 새 과제를 배우는 가중치 변화가 옛 과제가 쓰던 가중치를 덮어써서 옛 과제 성능이 떨어지는 것입니다(간섭).
- **가소성:** 새로 배우는 능력입니다.
- **안정성:** 옛 것을 지키는 능력입니다.

---

## 1. 일반론: 시냅스가 왜 잊는가 (이론 한계)

- **크기에 한계가 있는 시냅스는 많이 기억하지 못합니다** ([Fusi & Abbott 2007](https://www.nature.com/articles/nn1859)).
  - 고전 연상기억 모형의 큰 용량은 시냅스 세기에 상한이 없다는 가정에서 나옵니다.
  - 현실적인 상한을 두면 용량이 크게 줄어듭니다. 잘 바뀌는 시냅스의 기억 용량은 시냅스 수 N의 로그로만 늘어납니다.
  - 좋은 기억에는 상태가 여럿인 시냅스가 필요하다고 결론 내립니다.
- **시냅스를 내부 상태가 여럿인 동역학계로 보면 한계가 늘어납니다** ([Lahiri & Ganguli 2013](https://proceedings.nips.cc/paper_files/paper/2013/file/7f24d240521d99071c93af3917215ef7-Paper.pdf)).
  - 디지털(이진) 시냅스의 용량은 log N에 비례합니다.
  - 내부 상태가 여럿인 시냅스 모형 전반에 대해 기억 성능의 상한(memory frontier)을 유도했습니다.
- **여러 시간 척도로 연결된 시냅스 변수** ([Benna & Fusi 2016](https://ar5iv.arxiv.org/html/1507.07580), Nature Neuroscience 19:1697).
  - 빠른 변수가 느린 변수로 차례로 새어 들어가는 사슬로, 많은 기억을 오래 보존합니다.
  - 망각 곡선이 지수가 아니라 거듭제곱 꼴이 됩니다.
  - 표현 지연, 메타가소성, 간격 효과를 설명합니다.

**이 절의 한계가 다루는 설정 (사용자 지적으로 고침, 2026-11-21)**
- 위 세 논문은 같은 시냅스 집단에 기억이 계속 흘러 들어오고, 새 기억이 옛 기억을 덮어쓰는 설정을 다룹니다. 그 안에서 기억 수명과 용량의 한계를 말합니다.
- 이 프로젝트의 설계(지침의 최종 목표 2–3)는 그 설정이 아닙니다.
  - 빠른 시냅스(dsyn)는 과제 하나만 배우고, 느린 가중치로 옮긴 뒤 비웁니다. 그래서 오래 지킬 필요가 없습니다.
  - 과제 N개를 지켜야 하는 것은 느린 가중치(W_g, W_o)입니다. 이 가중치는 크기 상한이 없는 실수입니다.
- 따라서 이 설계에서 걸리는 것은 위의 "유계 시냅스의 기억 수명" 한계가 아닙니다. 과제를 차례로 증류할 때 **느린 가중치 안에서 과제끼리 생기는 간섭**과 **느린 망이 과제 N개를 함께 표현할 용량**입니다.
- 이 문제를 직접 다루는 것은 2절(정밀도 가중), 3절(사영·게이팅), 6절(경계 병합)입니다. 사영의 용량 한계도 3.1절의 NCL에서 다룹니다.
- 4절(다중 시간 척도)과 5절(선택적 공고화)은 빠른 것에서 느린 것으로 옮기는 단계(무엇을, 어떻게 옮기나)에 해당합니다.

---

## 2. 중요한 방향을 지키기: 정밀도(불확실성) 가중

**핵심:** 옛 과제에 중요한(확실한) 가중치는 덜 바꾸고, 불확실한 가중치는 많이 바꿉니다.

| 문헌 | 내용 | 국소성 / 과제 경계 |
|---|---|---|
| [Kirkpatrick et al. 2017](https://www.pnas.org/doi/10.1073/pnas.1611835114) (EWC, PNAS) | 옛 과제의 피셔 정보(가중치가 손실에 주는 민감도)로 가중한 이차 벌점으로 가중치를 옛 값 근처에 묶음 | 역전파, 과제 경계 필요 |
| [Zenke, Poole & Ganguli 2017](https://proceedings.mlr.press/v70/zenke17a.html) (SI, ICML) | 학습 경로에서 각 시냅스가 손실 감소에 기여한 양을 시냅스마다 온라인으로 누적해 중요도로 씀 | 시냅스별 누적은 국소, 학습 자체는 역전파 |
| [Aitchison et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7617048) (Nature Neuroscience 24:565) | 시냅스가 가중치의 평균과 분산(오차 막대)을 갖고, 불확실할수록 학습률이 크다는 가설. 시냅스 후 전위 크기의 큰 변동을 불확실성의 표현으로 설명 | 국소 (시냅스별 평균·분산) |
| [Bonnet et al. 2025](https://arxiv.org/abs/2504.13569) (MESU, Nature Communications 16:9614) | 매개변수마다 불확실성으로 학습을 비례 조절하는 베이즈 갱신. 사후를 잘라 내는 방식으로 오래된 정보를 서서히 잊음. **과제 경계와 재현이 필요 없음.** 순열 MNIST 200과제, CIFAR-100 증분 | 매개변수별, 경계 불필요 |

**이 프로젝트와의 대응**
- 이전 브랜치의 칼만 시냅스(P, R, Q)가 Aitchison et al.의 "시냅스별 평균·분산" 가설과 같은 계열입니다. 그 시냅스는 가중치 행렬 전체에 공분산을 둔 판이었습니다.
- 11-21 미정 항목의 "정밀도를 온라인으로 갱신하는 형태"와 "오차 쪽 정밀도(R) 없음"이 MESU·Aitchison이 다루는 부분입니다.

---

## 3. 옛 입력이 쓰는 방향과 직교하게만 쓰기 (사영), 그리고 맥락 게이팅

### 3.1 사영

**핵심:** 층의 가중치 변화를 옛 과제 입력이 차지한 부분공간과 직교하는 방향으로만 허용합니다. 그러면 옛 입력에 대한 그 층의 출력이 거의 바뀌지 않습니다.

| 문헌 | 내용 |
|---|---|
| [Zeng et al. 2019](https://arxiv.org/abs/1810.01256) (OWM, Nature Machine Intelligence) | 사영 P = I − A(AᵀA + αI)⁻¹Aᵀ (A의 열 = 이전에 학습한 입력, α = 작은 상수). 갱신 ΔW = κ·P·ΔW^BP를 층마다 입력 쪽에 적용합니다. P는 RLS의 상관 역행렬 (Σ x xᵀ + αI)⁻¹와 같은 꼴로 재귀 갱신하며, 이전 입력을 저장하지 않습니다. 기본판은 과제가 끝날 때 P를 갱신합니다(배치마다 갱신하는 판도 보고됐다고 하나 본문으로 확인하지 못함). 맥락이 감각 표현을 조절하는 별도 모듈(CDP)과 함께 씁니다 |
| [Saha, Garg & Roy 2021](https://arxiv.org/abs/2103.09762) (GPM, ICLR) | 과제가 끝날 때 각 층의 활성 행렬을 SVD해서 중요한 기저를 저장하고, 새 과제의 기울기를 그 기저와 직교하게 사영합니다 |
| [Kao et al. 2021](https://arxiv.org/abs/2106.08085) (NCL, NeurIPS) | 베이즈 가중치 규제(정밀도 벌점)와 기울기 사영을 하나로 합칩니다. 사영을 피셔 정보 기준의 신뢰 영역으로 봅니다. 사영 방법은 과제가 쌓이면 용량이 바닥날 수 있다고 지적합니다. 순환망에서도 시험했습니다 |

**이 프로젝트와의 대응 (유도)**
- 11-21의 정밀도 행렬은 P = α(C + αI)⁻¹(C = DoorKey 입력 x의 E[x xᵀ])이고, dsyn 쓰기에 x 대신 P·x를 씁니다.
- 우드버리 항등식으로 I − A(AᵀA + αI)⁻¹Aᵀ = α(AAᵀ + αI)⁻¹입니다.
- 따라서 C를 입력의 합 AAᵀ로 두면 11-21의 P는 **OWM의 사영과 같은 식**입니다. 11-21의 C는 평균이라, 합으로 쓸 때와는 α의 척도만 n배 다릅니다.
- 또 dsyn 갱신의 방향 δ·(P x)ᵀ는 (δ xᵀ)·P이므로(P는 대칭), OWM의 "입력 쪽에 P를 곱하기"와 같습니다.
- 차이:
  - OWM은 P를 과제마다 재귀 갱신합니다. 11-21은 DoorKey 자료로 한 번 계산해서 고정했고, α는 손으로 정했습니다.
- 11-21에 잰 맞교환 숫자도 있습니다. α = 1에서 보호 0.052, 새 과제 유지 0.170입니다. NCL이 말하는 "사영의 용량 한계"와 같은 축의 양입니다.

### 3.2 맥락 게이팅과 가지(수상돌기)별 분리

**핵심:** 과제나 맥락마다 서로 덜 겹치는 단위나 가지가 학습하게 해서, 한 과제의 쓰기가 다른 과제의 시냅스에 닿지 않게 합니다.

| 문헌 | 내용 |
|---|---|
| [Masse, Grant & Freedman 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6217392) (PNAS 115:E10467) | 과제마다 희소하고 대부분 겹치지 않는 단위 집합만 켭니다(맥락 의존 게이팅, 논문 그림은 80%를 끔). **게이팅만으로는 100과제에서 부분적으로만 효과가 있었고**, SI·EWC 같은 시냅스 안정화와 합치면 100과제를 거의 잊지 않고 배웠습니다 |
| [Veness et al. 2021](https://cdn.aaai.org/ojs/17202/17202-13-20696-1-2-20210518.pdf) (Gated Linear Networks, AAAI) | 역전파 없이 뉴런마다 국소 손실로 학습합니다. 맥락(부가 정보)의 초평면 게이팅이 뉴런마다 다른 가중치 벡터를 고르고, 고른 것만 갱신합니다. 표준 연속학습 시험에서 EWC와 비슷한 망각 저항을 보고했습니다 |
| [Sezener et al. 2021 / Kostadinov et al.](https://dimvladkost.github.io/publication/sezener2021/sezener2021.pdf) (Dendritic Gated Networks, bioRxiv) | GLN의 생물학적 판입니다. 억제 사이뉴런이 수상돌기 가지를 입력에 따라 켜고 끄며, 국소 학습 규칙을 씁니다. 저자 보고로는 데이터 효율이 높고 망각에 강합니다(독립 재현은 찾지 못함) |
| [Iyer et al. 2022](https://www.doi.org/10.3389/fnbot.2022.846219) (Active dendrites, Frontiers in Neurorobotics) | 맥락 신호가 수상돌기 분절을 통해 단위를 조절하고, k-승자 독식 희소성과 합쳐 다과제 강화학습과 연속학습에서 간섭을 줄입니다 |
| [Cichon & Gan 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4476301) (Nature 520:180, 생물 실험) | 생쥐 운동피질에서 서로 다른 운동 과제는 서로 다른 첨단 가지에 Ca²⁺ 스파이크를 냈고, 그 가지의 가시가 오래 강화됐습니다. SST 억제 사이뉴런을 끄면 다른 과제가 같은 가지에서 스파이크를 내고, 한 과제로 강화된 가시가 다른 과제로 약화됐습니다 |

**이 프로젝트와의 대응 (사실)**
- 11-21 Q 경로의 맥락은 곱셈 게이트 c = h_t(GRU 상태, 512차원)입니다.
- 11-21 측정값:
  - c의 활성 유닛 비율(|c| > RMS의 10%)은 90%입니다.
  - 유닛 쪽 코사인은 DoorKey–Unlock 0.126, DoorKey 안 0.151입니다.
- Masse et al.의 게이팅은 과제마다 대부분의 단위를 끄는 희소 게이트입니다. GLN의 게이팅은 맥락에 따라 서로 다른 가중치 벡터를 고르는 선택입니다. 둘 다 곱셈으로 크기를 바꾸는 조밀한 게이트와는 구조가 다릅니다.

---

## 4. 시냅스 안의 여러 시간 척도와 메타가소성

**핵심:** 시냅스 하나에 빠른 변수와 느린 변수를 둡니다. 느린(숨은) 변수가 그 시냅스가 얼마나 쉽게 바뀌는지를 정합니다.

| 문헌 | 내용 |
|---|---|
| [Benna & Fusi 2016](https://ar5iv.arxiv.org/html/1507.07580) | 1절. 빠른 변수에서 느린 변수로 이어지는 사슬 |
| [Kaplanis, Shanahan & Clopath 2018](https://proceedings.mlr.press/v80/kaplanis18a.html) (ICML) | Benna–Fusi 시냅스를 표 기반·심층 강화학습에 넣어, 두 과제 사이와 한 과제 안의 망각을 줄였습니다. 재현 버퍼 의존도 줄일 수 있었다고 보고합니다. 복잡한 Atari에서는 효과가 없었다는 제3자 발표 자료가 있으나 논문으로는 확인하지 못했습니다 |
| [Laborieux et al. 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8100137/) (Nature Communications 12) | 이진화 신경망의 숨은 실수 가중치를 메타가소 변수로 읽습니다. 숨은 값이 클수록 부호가 바뀌기 어렵게 합니다. 과거 자료를 저장하지 않고 과제별 손실도 없이, 순열 MNIST에서 주류 방법에 가까운 성능을 냈습니다 |

**이 프로젝트와의 대응:** dsyn(빠름) → W_g(느림) 두 변수는 이 갈래의 두 시간 척도와 같은 구도입니다. 11-21의 전이는 에피소드 끝에 W_g ← W_g + dsyn으로 한 번에 옮기는 꼴이고, Benna–Fusi처럼 연속적으로 새어 들어가는 꼴은 아닙니다.

---

## 5. 고른 것만 굳히기 (선택적 공고화)

| 문헌 | 내용 |
|---|---|
| [Frey & Morris 1997](https://www.nature.com/articles/385533a0) (Nature 385:533) | 시냅스 표지와 포획: 약한 자극은 시냅스에 임시 표지만 남기고, 같은 시기에 강한 사건이 단백질 합성을 일으키면 표지된 시냅스만 오래 가는 변화로 굳습니다 |
| [Clopath et al. 2008](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000248) (TagTriC, PLoS Comput Biol) | 표지–촉발–공고화의 수학 모형. 초기 LTP/LTD(빠름)와 후기(느림)를 다른 변수로 둡니다 |
| [Lindsey & Litwin-Kumar 2024](https://elifesciences.org/articles/90793) (eLife 12) | 회상 게이트 공고화: 단기 기억에 이미 저장된 지식과 맞는 갱신만 장기 기억으로 굳힙니다. 잡음이 많은 환경에서 기억의 신호 대 잡음 비를 크게 높이고, 간격 효과와 과제별 공고화 속도를 재현합니다 |

**이 프로젝트와의 대응:** 11-21의 dsyn → W_g 전이는 조건 없이 모두 옮깁니다. 이 갈래는 옮길지를 일관성(회상)이나 강한 사건(보상·두드러짐)으로 고르는 규칙입니다.

---

## 6. 과제 경계에서 합치기 (가중치 병합)

| 문헌 | 내용 |
|---|---|
| [Jin et al. 2023](https://arxiv.org/abs/2212.09849) (RegMean, ICLR) | 선형층마다 각 모델의 입력 그람 행렬로 가중한 닫힌 꼴로 합칩니다. 합친 모델과 각 모델의 출력 사이 ℓ2 거리를 각 학습 자료 위에서 최소화하는 선형 모형의 최적해이며, 재학습이 필요 없습니다 |
| [Matena & Raffel 2022](https://arxiv.org/abs/2111.09832) (Fisher merging, NeurIPS) | 매개변수별 피셔 정보로 가중한 평균으로 합칩니다(대각 근사) |

**이 프로젝트와의 대응 (유도)**
- y = W x 꼴의 층에서 과제 i의 입력 그람 G_i = Σ x xᵀ로 Σ_i Σ_x |W x − W_i x|²를 최소화하면 W(Σ G_i) = Σ W_i G_i가 나옵니다.
- 이것은 11-21 `merge_wg.py`의 **W C = B(C = Σ C_i, B = Σ W_i C_i)와 같은 식**입니다.
- 차이: 11-21은 C_i = S_i / n_i(과제별 평균)를 써서 과제마다 같은 무게를 줍니다. RegMean은 합을 씁니다.
- 2절 정밀도와의 관계: 입력 그람은 선형 회귀 가중치의 정밀도(역공분산)입니다. 그래서 이 병합은 과제별 선형 회귀 사후를 정밀도로 합친 것과 같은 꼴입니다.

---

## 7. 자유에너지 원리와 국소 가소성, 여러 가소성의 조합

| 문헌 | 내용 |
|---|---|
| [Isomura, Shimazaki & Friston 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8760273/) (Communications Biology 5:55) | 신경 활동과 가소성이 같은 비용을 최소화하는 율 부호 망은, 헤브 가소성이 지연된 조절을 받으면 부분 관측 마르코프 결정 모형 아래의 변분 자유에너지 최소화(능동적 추론)와 같다고 보입니다. 지연 조절된 헤브 가소성과 문턱 적응만으로 베이즈 최적 추론과 제어에 이른다고 주장합니다 |
| [Zenke, Agnes & Gerstner 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4411307) (Nature Communications 6:6922) | 헤브 가소성, 빠른 이종시냅스 가소성(시냅스 후 뉴런 상태에만 의존), 느린 항상성, 공고화를 함께 두어야 스파이킹 망이 기억을 안정적으로 형성하고 회상합니다. 하나라도 막으면 안정적인 기억 망으로 작동하지 않았습니다 |

**이 프로젝트와의 대응:** 11-21 미정 항목에 "dsyn 갱신식에 W_o가 들어가 한 층짜리 역전파와 같은 식(원리 1과 다름)"이 있습니다. 이 갈래는 역전파 없는 국소 규칙을 자유에너지 최소화로 유도한 예입니다. GLN·DGN(3.2절)도 국소 손실로 학습하는 예입니다.

---

## 8. 반대편: 가소성 상실

- [Dohare et al. 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11338828/) (Nature 632:768)
  - 표준 심층학습은 연속학습에서 가소성을 점점 잃어, 결국 얕은 망보다 나을 것이 없어집니다(ImageNet, 강화학습).
  - 잘 쓰이지 않는 단위의 일부를 계속 다시 초기화하는 연속 역전파로 가소성을 유지했습니다.
  - 저자들은 경사 하강만으로는 지속 학습에 부족하고, 무작위적인 비경사 요소가 필요하다고 결론 내립니다.
- 관련: 지침의 최종 목표 2("새 과제 학습 → 증류를 반복")는 망각뿐 아니라 반복 뒤의 가소성 유지도 요구합니다.

---

## 9. 갈래별 요약

| 갈래 | 대표 문헌 | 역전파 필요 | 과제 경계 필요 | 이 프로젝트에 있는 대응 부품 (11-21) |
|---|---|---|---|---|
| 정밀도 가중 | EWC, SI, Aitchison 2021, MESU 2025 | EWC·SI 예 / 베이즈 시냅스는 국소 | EWC 예 / MESU 아니오 | 이전 브랜치 칼만 시냅스. 11-21은 오차 쪽 정밀도 없음 |
| 사영 | OWM 2019, GPM 2021, NCL 2021 | 기본은 역전파 위에서 | OWM 기본판 예 | P = α(C + αI)⁻¹ (= OWM 사영, 유도). 고정, α 손으로 |
| 맥락 게이팅 | Masse 2018, GLN 2021, DGN, Iyer 2022 | GLN·DGN 아니오 | 게이팅 신호로 대체 | 곱셈 게이트 c = h_t (활성 90%) |
| 다중 시간 척도·메타가소성 | Benna–Fusi 2016, Kaplanis 2018, Laborieux 2021 | 아니오(시냅스 동역학) | 아니오 | dsyn → W_g (에피소드 끝 한 번에) |
| 선택적 공고화 | Frey & Morris 1997, Clopath 2008, Lindsey 2024 | 아니오 | 아니오 | 전이는 조건 없음 |
| 경계 병합 | RegMean 2023, Fisher merging 2022 | 아니오(닫힌 꼴) | 예 | W C = B (= RegMean, 유도) |
| 국소 규칙의 유도 | Isomura & Friston 2022, Zenke 2015 | 아니오 | — | dsyn 갱신에 W_o (한 층 역전파) |
| 가소성 유지 | Dohare 2024 | — | — | 반복 학습–증류에서 해당 |

## 10. 문헌에서 나온 질문 (이 프로젝트에 대한 판정 아님)

- 사영의 용량: 과제가 쌓이면 직교 여유가 줄어듭니다(NCL). 정밀도 벌점과 사영을 합치면 그 맞교환이 어떻게 달라지는지가 NCL의 주제입니다.
- 게이팅만으로 충분한지: Masse et al.은 게이팅 단독으로는 부분 효과였고 안정화와 합쳐야 했다고 보고합니다.
- 과제 경계 없이 할 수 있는지: OWM 기본판, GPM, RegMean은 경계를 씁니다. MESU는 경계 없이 불확실성으로만 합니다.
- 무엇을 굳힐지: 표지·포획과 회상 게이트는 모두 옮길 것을 고르는 규칙입니다.

## 참고문헌

- Fusi S, Abbott LF (2007). Limits on the memory storage capacity of bounded synapses. *Nature Neuroscience*. [링크](https://www.nature.com/articles/nn1859)
- Lahiri S, Ganguli S (2013). A memory frontier for complex synapses. *NIPS 26*, 1034–1042. [PDF](https://proceedings.nips.cc/paper_files/paper/2013/file/7f24d240521d99071c93af3917215ef7-Paper.pdf)
- Benna MK, Fusi S (2016). Computational principles of synaptic memory consolidation. *Nature Neuroscience* 19:1697–1706. [arXiv 1507.07580](https://ar5iv.arxiv.org/html/1507.07580)
- Kirkpatrick J et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS* 114:3521–3526. [링크](https://www.pnas.org/doi/10.1073/pnas.1611835114)
- Zenke F, Poole B, Ganguli S (2017). Continual learning through synaptic intelligence. *ICML*. [링크](https://proceedings.mlr.press/v70/zenke17a.html)
- Aitchison L et al. (2021). Synaptic plasticity as Bayesian inference. *Nature Neuroscience* 24:565–571. [PMC7617048](https://pmc.ncbi.nlm.nih.gov/articles/PMC7617048)
- Bonnet D et al. (2025). Bayesian continual learning and forgetting in neural networks. *Nature Communications* 16:9614. [arXiv 2504.13569](https://arxiv.org/abs/2504.13569)
- Zeng G, Chen Y, Cui B, Yu S (2019). Continual learning of context-dependent processing in neural networks. *Nature Machine Intelligence*. [arXiv 1810.01256](https://arxiv.org/abs/1810.01256)
- Saha G, Garg I, Roy K (2021). Gradient projection memory for continual learning. *ICLR*. [arXiv 2103.09762](https://arxiv.org/abs/2103.09762)
- Kao T-C, Jensen K, van de Ven G, Bernacchia A, Hennequin G (2021). Natural continual learning. *NeurIPS*. [arXiv 2106.08085](https://arxiv.org/abs/2106.08085)
- Masse NY, Grant GD, Freedman DJ (2018). Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization. *PNAS* 115:E10467–E10475. [PMC6217392](https://pmc.ncbi.nlm.nih.gov/articles/PMC6217392)
- Veness J et al. (2021). Gated linear networks. *AAAI*. [PDF](https://cdn.aaai.org/ojs/17202/17202-13-20696-1-2-20210518.pdf)
- Sezener E, Grabska-Barwińska A, Kostadinov D et al. (2021). A rapid and efficient learning rule for biological neural circuits (Dendritic gated networks). *bioRxiv*. [PDF](https://dimvladkost.github.io/publication/sezener2021/sezener2021.pdf)
- Iyer A et al. (2022). Avoiding catastrophe: active dendrites enable multi-task learning in dynamic environments. *Frontiers in Neurorobotics* 16:846219. [DOI](https://www.doi.org/10.3389/fnbot.2022.846219)
- Cichon J, Gan W-B (2015). Branch-specific dendritic Ca²⁺ spikes cause persistent synaptic plasticity. *Nature* 520:180–185. [PMC4476301](https://pmc.ncbi.nlm.nih.gov/articles/PMC4476301)
- Kaplanis C, Shanahan M, Clopath C (2018). Continual reinforcement learning with complex synapses. *ICML*, PMLR 80:2497–2506. [링크](https://proceedings.mlr.press/v80/kaplanis18a.html)
- Laborieux A, Ernoult M, Hirtzlin T, Querlioz D (2021). Synaptic metaplasticity in binarized neural networks. *Nature Communications* 12. [PMC8100137](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8100137/)
- Frey U, Morris RGM (1997). Synaptic tagging and long-term potentiation. *Nature* 385:533–536. [링크](https://www.nature.com/articles/385533a0)
- Clopath C, Ziegler L, Vasilaki E, Büsing L, Gerstner W (2008). Tag-trigger-consolidation: a model of early and late long-term-potentiation and depression. *PLoS Computational Biology* 4:e1000248. [링크](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000248)
- Lindsey JW, Litwin-Kumar A (2024). Selective consolidation of learning and memory via recall-gated plasticity. *eLife* 12. [링크](https://elifesciences.org/articles/90793)
- Jin X, Ren X, Preotiuc-Pietro D, Cheng P (2023). Dataless knowledge fusion by merging weights of language models. *ICLR*. [arXiv 2212.09849](https://arxiv.org/abs/2212.09849)
- Matena M, Raffel C (2022). Merging models with Fisher-weighted averaging. *NeurIPS*. [arXiv 2111.09832](https://arxiv.org/abs/2111.09832)
- Isomura T, Shimazaki H, Friston KJ (2022). Canonical neural networks perform active inference. *Communications Biology* 5:55. [PMC8760273](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8760273/)
- Zenke F, Agnes EJ, Gerstner W (2015). Diverse synaptic plasticity mechanisms orchestrated to form and retrieve memories in spiking neural networks. *Nature Communications* 6:6922. [PMC4411307](https://pmc.ncbi.nlm.nih.gov/articles/PMC4411307)
- Dohare S et al. (2024). Loss of plasticity in deep continual learning. *Nature* 632:768–774. [PMC11338828](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11338828/)
