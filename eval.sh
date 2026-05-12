#!/bin/bash
#
# Standalone OPSD eval driver — launches a sglang server, runs eval_sglang.py
# against the configured datasets, writes a timestamped JSON to ./eval_results/.
#
# Defaults align with paper Table 8 (eval-time sampling):
#   max_new_tokens=38912 (override via dataset-level max_response_len in YAML)
#   top_p=0.95, top_k=-1 (slime default), temperature=1.0
#
# Common overrides via env var:
#   MODEL=/path/to/ckpt   bash eval.sh
#   TP=2 N_SAMPLES=8      bash eval.sh
#   EVAL_CONFIG=data/preprocess/test/eval_config.yaml  bash eval.sh   # quick training-time eval
#   PORT=30001            bash eval.sh
#   APPLY_CHAT_KW='{"enable_thinking": false}'  bash eval.sh          # TM-off student per paper
#
# Outputs:
#   eval_results/<model_basename>__<timestamp>.json

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL="${MODEL:-/path/to/Qwen3-1.7B}"
EVAL_CONFIG="${EVAL_CONFIG:-data/preprocess/test/eval.yaml}"
TP="${TP:-1}"
N_SAMPLES="${N_SAMPLES:-16}"          # fallback when YAML doesn't set n_samples_per_eval_prompt
MAX_LEN="${MAX_LEN:-38912}"           # paper Table 8 MaxNewTokens
TEMPERATURE="${TEMPERATURE:-1.0}"     # paper Table 8 Temp
TOP_P="${TOP_P:-0.95}"                # paper Table 8 TopP
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CONCURRENCY="${CONCURRENCY:-128}"
APPLY_CHAT_KW="${APPLY_CHAT_KW:-}"    # JSON, e.g. '{"enable_thinking": false}'
NO_SERVER="${NO_SERVER:-0}"           # set to 1 to reuse an already-running sglang on $PORT

# ── Output path ───────────────────────────────────────────────────────────────
mkdir -p eval_results
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="$(basename "${MODEL}")"
OUTPUT="${OUTPUT:-eval_results/${MODEL_TAG}__${TIMESTAMP}.json}"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ ! -d "${MODEL}" ]]; then
    echo "ERROR: MODEL path does not exist: ${MODEL}" >&2
    exit 1
fi
if [[ ! -f "${EVAL_CONFIG}" ]]; then
    echo "ERROR: EVAL_CONFIG yaml not found: ${EVAL_CONFIG}" >&2
    exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}."

# ── Banner ────────────────────────────────────────────────────────────────────
cat <<EOF
╔══════════════════════════════════════════════════════════════════════════════╗
║ OPSD eval                                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ MODEL        : ${MODEL}
║ EVAL_CONFIG  : ${EVAL_CONFIG}
║ TP / N_SAMPL : ${TP} / ${N_SAMPLES}
║ MAX_LEN      : ${MAX_LEN}
║ TEMP / TOP_P : ${TEMPERATURE} / ${TOP_P}
║ PORT / MEM   : ${PORT} / ${MEM_FRACTION}
║ OUTPUT       : ${OUTPUT}
╚══════════════════════════════════════════════════════════════════════════════╝
EOF

# ── Run ───────────────────────────────────────────────────────────────────────
ARGS=(
    --model "${MODEL}"
    --tp "${TP}"
    --eval-config "${EVAL_CONFIG}"
    --n-samples "${N_SAMPLES}"
    --max-len "${MAX_LEN}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --port "${PORT}"
    --mem-fraction "${MEM_FRACTION}"
    --concurrency "${CONCURRENCY}"
    --output "${OUTPUT}"
)

if [[ -n "${APPLY_CHAT_KW}" ]]; then
    ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_KW}")
fi
if [[ "${NO_SERVER}" == "1" ]]; then
    ARGS+=(--no-server)
fi

python3 eval_sglang.py "${ARGS[@]}"
