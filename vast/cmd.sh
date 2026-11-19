# id: 2
# vast.ai 인스턴스에서 한 번 실행할 명령. id를 바꿔 푸시하면 runner.sh가 실행한다.
ls -d /venv/*/bin/python 2>/dev/null; command -v python3
for p in /venv/main/bin/python python3; do [ -x "$(command -v $p)" ] && $p -c "import sys,torch; print(sys.executable, sys.version.split()[0], 'torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())" 2>&1 | tail -1; done
