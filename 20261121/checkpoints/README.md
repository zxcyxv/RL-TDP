# 체크포인트 (agent.pkl + step.pkl)

| 폴더 | 내용 |
|---|---|
| `0412_dk8_1800000` | 1단계 `0412` (DoorKey-8x8, \|e\|² 정규화, η 0.2) 180만 스텝. 새 과제 단계의 출발점 |
| `1012_unlock_teacher_500000` | 위에서 Unlock 50만 학습 (W_o 고정, P 없음). Unlock 교사 |
| `merged_AB` | 두 교사를 `merge_wg.py`로 통합 (q/Wg, q/Wg_t만 교체). `merge_info.json` 포함 |

쓰기: `--run.from_checkpoint <폴더>` (`/workspace/ext/dreamerv3`에서, `scripts/chain_merge.sh`의 COMMON 설정과 함께).
