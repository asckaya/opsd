#!/bin/bash

# Diverse Self-Privileged OPSD Training Script for Qwen3-1.7B
# This script uses the OPSD plugin to generate 16 samples and select up to 8 diverse traces for distillation.

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
# Path to the Qwen3-1.7B checkpoint and data
HF_CHECKPOINT=${HF_CHECKPOINT:-"/path/to/Qwen3-1.7B"}
PROMPT_DATA=${PROMPT_DATA:-"data/opsd_math_30k.jsonl"}
SAVE_DIR=${SAVE_DIR:-"./checkpoints/qwen3-1.7b-opsd"}
LOAD_DIR=${LOAD_DIR:-"${HF_CHECKPOINT}"}
EVAL_CONFIG_PATH=${EVAL_CONFIG_PATH:-"data/preprocess/test/eval_config.yaml"}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}

# Environment setup
export PYTHONPATH=$PYTHONPATH${PYTHONPATH:+:}.
# If you haven't prepared the dataset yet, run data/preprocess/prepare_opsd_dataset.py first.
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
   # Function paths (pointing to the __init__.py entry points)
   --rollout-function-path slime_plugins.opsd.generate_rollout
   --custom-megatron-init-path slime_plugins.opsd.init_hook
   --custom-megatron-before-train-step-hook-path slime_plugins.opsd.before_train_step_hook
   --loss-type custom_loss
   --custom-loss-function-path slime_plugins.opsd.loss_function
   
   # OPSD Hyperparameters
   --opsd-k 16                # Sample 16 trajectories (K)
   --opsd-n 8                 # Select at most 8 diverse traces (N)
   --opsd-kb 16               # Pre-filter top 16 by quality (K_b)
   --opsd-alpha 1.0           # Distillation loss weight (alpha)
   --opsd-kl-weight 1.0       # Mixture weight KL coeff (beta)
   --opsd-entropy-weight 0.5  # Mixture weight Entropy coeff (gamma)
   --opsd-diversity-weight 0.5 # Mixture weight Diversity coeff (rho)
   --opsd-temperature 1.0     # Temperature for mixture weights
   --opsd-weight-top-k 512    # Efficiency: truncate vocab for weights
   --opsd-diversity-top-k 128 # Top-K vocab truncation for token-level JSD
)

# Performance & Parallelism
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
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

# Optimizer
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
    --tb-experiment-name qwen3-1.7b-opsd
)

WANDB_ARGS=(
    # --use-wandb
    # --wandb-project opsd
    # --wandb-group qwen3-1.7b-opsd
    # --wandb-key ${WANDB_KEY}
)

EVAL_ARGS=(
    --eval-interval ${EVAL_INTERVAL}
    --eval-config ${EVAL_CONFIG_PATH}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.35
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# --- Execution ---

# Start Ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray stop --force || true
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8088

# Submit Job
ray job submit --address="http://127.0.0.1:8088" \
   --runtime-env-json='{"env_vars": {"CUDA_DEVICE_MAX_CONNECTIONS": "1"}}' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --rollout-num-gpus 8 \
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
