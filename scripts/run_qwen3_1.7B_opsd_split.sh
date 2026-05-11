#!/bin/bash

# Diverse Self-Privileged OPSD Training Script for Qwen3-1.7B — SPLIT MODE
# Variant of run_qwen3_1.7B_opsd.sh that disaggregates actor and rollout onto
# separate GPU sets (4 actor + 4 rollout, total 8 GPUs).
#
# When this beats the colocate version:
#   • Rollout (sglang generation) and training step can overlap rather than
#     run sequentially.
#   • SGLang owns its 4 GPUs at high memory fraction (no contention with
#     training), so it can use larger KV cache and run faster per request.
#
# When colocate wins:
#   • Training step is the dominant cost and benefits from more DP workers
#     (8 GPUs for actor vs 4). OPSD makes training-step heavy due to the
#     teacher forwards, so this is the case to benchmark against.

set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# --- Configuration ---
HF_CHECKPOINT=${HF_CHECKPOINT:-"/path/to/Qwen3-1.7B"}
PROMPT_DATA=${PROMPT_DATA:-"data/opsd_math_30k.jsonl"}
SAVE_DIR=${SAVE_DIR:-"./checkpoints/qwen3-1.7b-opsd-split"}
LOAD_DIR=${LOAD_DIR:-"${HF_CHECKPOINT}"}
EVAL_CONFIG_PATH=${EVAL_CONFIG_PATH:-"data/preprocess/test/eval_config.yaml"}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}

# Environment setup
export PYTHONPATH=$PYTHONPATH${PYTHONPATH:+:}.
source "scripts/models/qwen3-1.7B.sh"

# --- Arguments ---

CKPT_ARGS=(
   --hf-checkpoint ${HF_CHECKPOINT}
   --load ${LOAD_DIR}
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rollout-batch-size 16
   --rollout-max-response-len 2048
   --rollout-temperature 1.0

   --global-batch-size 128
   --n-samples-per-prompt 1  # Base n_samples, OPSD rollout will override this to opsd_k
)

# OPSD Plugin Arguments
OPSD_ARGS=(
   --rollout-function-path slime_plugins.opsd.generate_rollout
   --custom-megatron-init-path slime_plugins.opsd.init_hook
   --custom-megatron-before-train-step-hook-path slime_plugins.opsd.before_train_step_hook
   --loss-type custom_loss
   --custom-loss-function-path slime_plugins.opsd.loss_function

   --opsd-k 16
   --opsd-n 4
   --opsd-kb 8
   --opsd-alpha 1.0
   --opsd-kl-weight 1.0
   --opsd-entropy-weight 0.5
   --opsd-diversity-weight 0.5
   --opsd-temperature 1.0
   --opsd-weight-top-k 512
   --opsd-diversity-metric unigram_jsd  # see run_qwen3_1.7B_opsd.sh for rationale
   --opsd-diversity-top-k 128
   --no-opsd-freeze-teacher   # path-A in-loss swap is currently inert; pin
                              # to the sync-with-student teacher until path B
                              # (precompute in before_train_step_hook) lands.
)

# Performance & Parallelism — actor side
# With 4 actor GPUs instead of 8, each GPU now sees ~2x more samples per step.
# Keep micro-batch-size 1 and let dynamic batching pack to --max-tokens-per-gpu.
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --attention-backend flash
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

RM_ARGS=(
   --rm-type math
)

TB_ARGS=(
    --use-tensorboard
    --tb-project-name opsd
    --tb-experiment-name qwen3-1.7b-opsd-split
)

WANDB_ARGS=(
    # --use-wandb
    # --wandb-project opsd
    # --wandb-group qwen3-1.7b-opsd-split
    # --wandb-key ${WANDB_KEY}
)

EVAL_ARGS=(
    --eval-interval ${EVAL_INTERVAL}
    --eval-config ${EVAL_CONFIG_PATH}
)

# SGLang owns its 4 dedicated GPUs — push memory fraction up to use them well.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.85
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# --- Execution ---

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray stop --force || true
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8088

# Submit Job — note: NO --colocate, and the actor/rollout GPU splits are 4/4.
ray job submit --address="http://127.0.0.1:8088" \
   --runtime-env-json='{"env_vars": {"CUDA_DEVICE_MAX_CONNECTIONS": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}}' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPSD_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${RM_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${TB_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    --num-rollout 1000

# Cleanup
# ray stop --force
