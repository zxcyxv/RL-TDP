# 실행 환경

## 인스턴스 (2026-11-19 기준)

- vast.ai 인스턴스 `61412376`. 이미지 `vastai/pytorch` (CUDA 13).
- GPU RTX 5090 32GB, CPU 16코어, 램 30GB, 디스크 16GB (`/` 오버레이).
- `/workspace`가 호스트 볼륨인지는 `vast-capabilities | jq '.instance.workspace_is_volume'`로 확인한다. 볼륨이 아니면 인스턴스를 폐기할 때 함께 사라지므로, 남길 것은 git으로 올린다.

## Python

- `/venv/main/bin/python` (3.12). 시스템 `python3`에는 jax가 없다.
- 설치:

  ```bash
  /venv/main/bin/pip install -U "jax[cuda13]" optax
  ```

  확인된 버전: jax 0.12.1, optax 0.2.9. GPU 확인:

  ```bash
  /venv/main/bin/python -c "import jax; print(jax.devices())"
  ```

- 실행할 때는 저장소 루트에서 `PYTHONPATH=.`을 둔다. 여러 실행이 GPU 메모리를 나눠 쓰려면 `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## 환경 동작 확인

```bash
PYTHONPATH=. /venv/main/bin/python -c "from rltdp.world import errand as ER; s = ER.make_session(1000, 256); print(len(s['acts']))"
```

지금은 교사 일정으로 세션을 만드는 것까지만 된다 (`errand_env_spec.md`).

## DreamerV3 기준 구현

정하지 않았다. 정할 것:

1. 공식 구현 (github.com/danijar/dreamerv3, JAX)의 어느 리비전을 고정할지.
2. errand를 그 구현의 환경 인터페이스에 어떻게 붙일지.
3. errand 규모에 맞는 모델 크기와 환경 스텝 예산.
