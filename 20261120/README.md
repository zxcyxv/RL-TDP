# 2026-11-20

- `실험기록.md`: 오늘 돌린 실행, 숫자, 사용자 결정, 미정 항목, 다음 할 일
- `setup_instance.sh`: 새 인스턴스에서 같은 작업 환경을 만든다
- `dreamerv3_changes/`: 공식 DreamerV3 (`danijar/dreamerv3` e01491f)에 대한 패치와 새 파일. `setup_instance.sh`가 적용한다
- `env/requirements_pinned.txt`: 설치하는 패키지 버전. `env/pip_freeze_full.txt`: 오늘 인스턴스의 전체 목록 (참고용)
- `logs/<실행>/`: 실행별 `config.yaml`, `metrics.jsonl`, `scores.jsonl`, `progress.jsonl`, `qstats.jsonl`와 표준 출력 `<실행>.out`. 체크포인트·리플레이·영상은 넣지 않았다
- `logs/*.json`: 회귀 진단·오프라인 스윕 결과 (근거로 쓰지 않기로 함, 실험기록 참고)
- `scripts/`: 진행 확인(`status.py`, `watch_progress.sh`)과 실행 묶음 스크립트
- `claude_memory/`: Claude Code 메모리 사본. `setup_instance.sh`가 `~/.claude/projects/-workspace-RL-TDP/memory/`로 복사한다

## 새 인스턴스에서

```bash
git clone https://github.com/zxcyxv/RL-TDP.git /workspace/RL-TDP
cd /workspace/RL-TDP && git checkout restart-20261119
bash 20261120/setup_instance.sh
```

확인한 것: 공식 저장소를 새로 클론해 패치·새 파일을 적용한 결과가 오늘 작업 트리와 파일 단위로 같다. 패키지 목록은 `uv pip install --dry-run`으로 풀린다. GPU에서 실제 설치·실행은 새 인스턴스에서 확인해야 한다.

## 실행 명령 (`/workspace/ext/dreamerv3`에서, `/venv/main` 활성화)

한 번에 하나만 돌린다 (리플레이 1e6이면 RAM 약 15GB). 길면 백그라운드로.

기준선 (공식 DreamerV3):

```bash
python dreamerv3/main.py --logdir /workspace/ext/logs/<날짜>_<시각>_dreamer_dk8 \
  --configs size1m --task navix_DoorKey-8x8-v0 --agent.dyn.rssm.unroll True \
  --run.steps 500000 --run.train_ratio 64 --replay.size 1e6 --run.log_every 120 \
  --run.save_steps 100000
```

Q 경로 에이전트: 위에 `--agent_type q`와 `--agent.q.*` 옵션을 더한다 (옵션 목록은 `dreamerv3/configs.yaml`의 `agent.q`).

학습된 월드모델에 의사결정 부분만 새로 붙이기 (오늘 `wm600k` 실행들):

```bash
  --run.from_checkpoint <기준선 logdir>/ckpt_steps/00600000 \
  --run.from_checkpoint_regex '^(enc|dyn|dec|rew|con)/'
```

오늘 쓴 기준선 60만 체크포인트는 저장소에 없다. 새 인스턴스에서는 기준선을 60만 스텝까지 다시 돌려 만들어야 한다 (`--run.steps 600000`). 같은 설정이라도 실행마다 학습이 올라오는 시점이 다를 수 있다 (오늘 두 기준선이 약 10만 스텝 차이).
