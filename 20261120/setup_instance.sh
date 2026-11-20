#!/bin/bash
# Rebuild the 2026-11-20 working environment on a fresh Vast.ai PyTorch
# instance (RTX 5090 / Blackwell, driver with CUDA 13 support).
#
#   git clone https://github.com/zxcyxv/RL-TDP.git /workspace/RL-TDP
#   cd /workspace/RL-TDP && git checkout restart-20261119
#   bash 20261120/setup_instance.sh
#
# Result: /workspace/ext/dreamerv3 = danijar/dreamerv3 e01491f + our patch and
# new files; Python deps installed into /venv/main; Claude memory restored.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXT=/workspace/ext
mkdir -p $EXT/logs

# 1. DreamerV3 at the pinned commit, plus our changes.
if [ ! -d $EXT/dreamerv3 ]; then
  git clone https://github.com/danijar/dreamerv3 $EXT/dreamerv3
fi
cd $EXT/dreamerv3
git checkout -q e01491f
git apply "$HERE/dreamerv3_changes/official_e01491f.patch"
cp -r "$HERE/dreamerv3_changes/new_files/"* .

# 2. Python packages (into the image's /venv/main, Python 3.12).
#    Official requirements pin jax[cuda12]==0.4.33, which does not run on
#    Blackwell; we use jax 0.12.1 with CUDA 13 instead (see 실험기록.md).
source /venv/main/bin/activate
grep -v -E "ale_py|autorom|^jax|nvidia-cuda-nvcc|^numpy" requirements.txt \
  > /tmp/dv3req.txt
UV_LINK_MODE=copy uv pip install --no-cache -r "$HERE/env/requirements_pinned.txt"
UV_LINK_MODE=copy uv pip install --no-cache -r /tmp/dv3req.txt

# 3. Claude Code memory for this project.
MEM=/root/.claude/projects/-workspace-RL-TDP/memory
mkdir -p $MEM
cp -n "$HERE/claude_memory/"*.md $MEM/ || true

# 4. Check.
python - <<'EOF'
import jax, navix, numpy
print('jax', jax.__version__, 'navix', navix.__version__ if hasattr(navix, '__version__') else '?',
      'numpy', numpy.__version__, jax.devices())
EOF
echo "setup done"
