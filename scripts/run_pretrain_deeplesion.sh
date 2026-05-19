set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python -m sliceworld.train.pretrain_world_model --config "$ROOT/configs/pretrain_deeplesion.yaml"
