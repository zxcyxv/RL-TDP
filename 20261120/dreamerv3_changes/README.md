# DreamerV3 변경분 (2026-11-20)

공식 저장소 `danijar/dreamerv3` 커밋 `e01491f`에 대한 변경. `/workspace/ext/dreamerv3`에서 작업했다.

- `official_e01491f.patch`: 공식 파일 수정분 (`git apply`로 적용). jax 0.12.1 API 호출 수정, NAVIX 등록, 설정 추가(`agent_type`, `agent.q`, `agent.dyn.rssm.unroll`, `run.save_steps`, `run.from_checkpoint_regex`), 학습 루프의 단계별 체크포인트·`progress.jsonl`.
- `new_files/`: 새 파일. 같은 경로로 복사한다.
  - `embodied/envs/navix.py`: NAVIX 환경
  - `dreamerv3/qagent.py`, `dreamerv3/qwrapper.py`: Q 경로 에이전트
  - `dreamerv3/bench_policy.py`: 정책·학습 스텝 시간 측정
  - `dreamerv3/diag_features.py`, `dreamerv3/sweep_offline.py`, `dreamerv3/ref_mlp.py`: 진단 (근거로 쓰지 않기로 한 것 포함, 실험기록 참고)

복원: `git clone https://github.com/danijar/dreamerv3 && cd dreamerv3 && git checkout e01491f && git apply official_e01491f.patch && cp -r new_files/* .`
