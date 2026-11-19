# id: 4
# vast.ai 인스턴스에서 한 번 실행할 명령. id를 바꿔 푸시하면 runner.sh가 실행한다.
cd /workspace/ctl
mkdir -p /workspace/runs
PYTHONPATH=/workspace/ctl /venv/main/bin/python -m rltdp.world.stage1_route --out /workspace/runs/stage1_full.json 2>&1
