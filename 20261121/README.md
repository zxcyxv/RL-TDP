# 2026-11-21

- `실험기록.md`: 오늘 바꾼 코드, 돌린 실행과 숫자, 사용자 결정, 미정 항목
- `setup_instance.sh`: 새 인스턴스에서 작업 환경을 만든다 (어제와 같은 방식, 이 폴더의 패치를 적용)
- `dreamerv3_changes/`: 공식 DreamerV3 (`danijar/dreamerv3` e01491f)에 대한 패치와 새 파일. 공식 저장소에 적용한 결과가 오늘 작업 트리와 파일 단위로 같음을 확인했다
- `env/`: 어제와 같은 패키지 목록
- `logs/<실행>/`: 실행별 `config.yaml`, `metrics.jsonl`, `scores.jsonl`, `progress.jsonl`, `qstats.jsonl`와 표준 출력 `<실행>.out`(재개는 `.resume.out`). 체크포인트·리플레이는 넣지 않았다
- `logs/*.json`, `logs/*.npy`: 진단 결과와 정밀도 행렬 P (α 1)
- `logs/*.log`: 실행 묶음 스크립트의 시작·종료 기록
- `scripts/`: 새 과제 단계 실행 묶음
- `claude_memory/`: Claude Code 메모리 사본
- `synapse_literature/README.md`: 시냅스 가소성으로 파국적 망각을 막는 방법 문헌 조사 (저녁, Claude 클라우드 세션)

## 새 인스턴스에서

```bash
git clone https://github.com/zxcyxv/RL-TDP.git /workspace/RL-TDP
cd /workspace/RL-TDP && git checkout restart-20261119
bash 20261121/setup_instance.sh
```

## 실행 명령 (`/workspace/ext/dreamerv3`에서, `/venv/main` 활성화)

1단계 (오늘 마지막 구조, `0412`):

```bash
python dreamerv3/main.py --logdir /workspace/ext/logs/<날짜>_<시각>_q_dk8 \
  --configs size1m --task navix_DoorKey-8x8-v0 --agent.dyn.rssm.unroll True \
  --run.steps 1000000 --run.train_ratio 64 --replay.size 1e6 --run.log_every 120 \
  --run.save_steps 100000 --agent_type q --agent.q.tau_rel 1.0 \
  --agent.q.trace_norm True --agent.q.eta 0.2
```

새 과제 단계는 `scripts/chain_unlock*.sh`. 시작점인 `0412`의 180만 체크포인트는 저장소에 없으므로 1단계를 다시 돌려 만들어야 한다. 프로세스 메모리가 계속 늘어 30GB RAM에서 약 190만 스텝 근처에 바닥난다.
