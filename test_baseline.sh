#!/usr/bin/env bash
# Run the unchanged starter configuration and keep all results in one run folder.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PYTHON=.venv/bin/python
    else
        PYTHON=python3
    fi
fi
RUN_DIR="${1:-runs/baseline_$(date +%Y%m%d_%H%M%S)}"
mkdir -p -- "$RUN_DIR"
if [[ -e "$RUN_DIR/train.log" || -e "$RUN_DIR/best.pt" ]]; then
    echo "Run directory already contains results: $RUN_DIR" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1

"$PYTHON" - <<'PY' | tee "$RUN_DIR/environment.log"
import torch
from model import build_model
from utils import read_json
config = read_json('config.json')
print('Config:', config)
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('Trainable parameters:', sum(p.numel() for p in build_model(config).parameters() if p.requires_grad))
PY
"$PYTHON" baseline.py train --config config.json --run "$RUN_DIR" 2>&1 | tee "$RUN_DIR/train.log"
"$PYTHON" baseline.py export --run "$RUN_DIR" --output "$RUN_DIR/model.onnx" 2>&1 | tee "$RUN_DIR/export.log"
"$PYTHON" check_submission.py "$RUN_DIR/model.onnx" 2>&1 | tee "$RUN_DIR/check.log"
"$PYTHON" baseline.py evaluate "$RUN_DIR/model.onnx" 2>&1 | tee "$RUN_DIR/evaluate.log"
echo "Baseline results saved to: $RUN_DIR"
