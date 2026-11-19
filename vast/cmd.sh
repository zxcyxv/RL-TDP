# id: 1
# vast.ai 인스턴스에서 한 번 실행할 명령. id를 바꿔 푸시하면 runner.sh가 실행한다.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
free -g | head -2
nproc
df -h /workspace | tail -1
which python; python -V
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
git -C /workspace/ctl log --oneline -1
