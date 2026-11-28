#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_ROOT"
if [[ ! -x .venv-ubuntu/bin/python ]]; then
    echo "Linux environment missing: .venv-ubuntu/bin/python" >&2
    exit 1
fi
export PACKET_CUT_MODELS_DIR="$APP_ROOT/models"
export WORKFLOW_DEVICE=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export YOLO_CONFIG_DIR="$APP_ROOT/.runtime/ultralytics"
export PACKET_OCR_DET_MODEL="${PACKET_OCR_DET_MODEL:-PP-OCRv5_mobile_det}"
export PACKET_OCR_REC_MODEL="${PACKET_OCR_REC_MODEL:-en_PP-OCRv5_mobile_rec}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
mkdir -p "$YOLO_CONFIG_DIR"
exec .venv-ubuntu/bin/python automated_app/app.py --cpu "$@"
