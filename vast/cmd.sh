# id: 3
# vast.ai 인스턴스에서 한 번 실행할 명령. id를 바꿔 푸시하면 runner.sh가 실행한다.
cd /workspace
/venv/main/bin/pip install -q -U "jax[cuda13]" optax 2>&1 | tail -3
/venv/main/bin/python -c "import jax, optax; print('jax', jax.__version__, 'optax', optax.__version__, jax.devices())"
