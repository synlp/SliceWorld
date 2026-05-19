set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python -m sliceworld.eval.evaluate_ctrg --config "$ROOT/configs/eval_ctrate.yaml"
