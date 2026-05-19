set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python -m sliceworld.train.finetune_ctrg --config "$ROOT/configs/finetune_ctrate_qwen3_1p7b.yaml"
