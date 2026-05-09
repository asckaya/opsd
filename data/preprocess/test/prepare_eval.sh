#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=${HF_TOKEN:-}  # set via environment or huggingface-cli login
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

SCRIPTS=(
    aime2025
    amc
    mmlupro
    math500
)

for name in "${SCRIPTS[@]}"; do
    echo "==> $name"
    python "$SCRIPT_DIR/${name}.py"
done

echo "Done. Output files:"
ls -lh data/json/
