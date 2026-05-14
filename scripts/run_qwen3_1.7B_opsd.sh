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
HF_CHECKPOINT=${HF_CHECKPOINT:-"/nfs/ofs-llm-ssd/models/opensource/Qwen3-1.7B"}
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
   --rollout-batch-size 32
   # Paper Table 6 (OPSD column): MaxCompletionLength=1024, SamplingTemperature=1.1.
   # Table 8 evaluation params add top_p=0.95, top_k=-1 (slime default); the OPSD
   # reference scripts also pass top_k=20 to vLLM. We use top_k=20 here to match
   # the reference; flip to -1 if you want the eval-time setting.
   --rollout-max-response-len 16384
   --rollout-temperature 1.1
   --rollout-top-p 0.95
   --rollout-top-k 20

   # Paper Table 6: EffectiveBatchSize = 32.
   --global-batch-size 32
   --n-samples-per-prompt 1  # Base n_samples; OPSD rollout overrides this to opsd_k+1 (1 student + K candidates)
)

# OPSD Plugin Arguments
OPSD_ARGS=(
   # ── Function paths ─────────────────────────────────────────────────
   --rollout-function-path slime_plugins.opsd.generate_rollout
   --custom-megatron-init-path slime_plugins.opsd.init_hook
   --custom-megatron-before-train-step-hook-path slime_plugins.opsd.before_train_step_hook
   --loss-type custom_loss
   --custom-loss-function-path slime_plugins.opsd.loss_function

   # ── Pool / selection sizes (ALGO §1.3 recommends K=8-32, N=2-4, K_b=8-16) ─
   --opsd-k 16                          # K (default 8). Total rollouts/prompt = K+1 = 17 (1 student + K candidates)
   --opsd-n 4                           # N selected diverse traces (default 2)
   --opsd-kb 8                          # TopK_b pre-filter on quality score (default off ⇒ keep all K)
   --opsd-fallback-to-gt                # default; flip to --no-opsd-fallback-to-gt to drop prompts with no correct candidate

   # ── Loss composition ───────────────────────────────────────────────
   --opsd-alpha 1.0                     # scale on L_distill; ALGO §1.1 step 8 default
   --opsd-rkl-weight 0.0                # off; >0 adds α_RKL·KL(p_θ‖q_mix), paper recommends α_RKL ≪ 1 (e.g. 0.01)
   # --opsd-mix-with-policy-loss        # opt-in ablation: layer GRPO PG-loss on top of L_distill
   --opsd-jsd-token-clip 0.05           # per-position KL clamp post sum-over-vocab; matches paper's --jsd_token_clip 0.05
   # --opsd-pointwise-kl-clip 0.05      # opt-in: per-(pos,vocab) one-sided clip. CAN drive per-token KL negative — prefer the line above

   # ── Mixture weights (ALGO §1.1 step 6) ─────────────────────────────
   --opsd-kl-weight 1.0                 # β (KL coefficient)
   --opsd-entropy-weight 0.5            # γ (entropy coefficient)
   --opsd-diversity-weight 0.5          # ρ (diversity coefficient)
   --opsd-temperature 1.0               # 1.0 ⇒ raw distributions; ALGO has no temperature term
   --opsd-weight-top-k 512              # top-K vocab truncation for the mixture-weight softmax

   # ── Quality scoring (ALGO §1.1 step 3) ─────────────────────────────
   --opsd-quality-len-weight 0.1        # η_l (length penalty)
   --opsd-quality-format-weight 0.2     # η_f (format penalty)
   --opsd-quality-conf-weight 0.5       # η_c (Conf reward)
   --opsd-quality-conf-norm rank        # rank → [0,1]; puts η_c on the same axis as Len/Format. Alts: zscore | minmax | raw

   # ── Diversity selection (ALGO §1.1 step 4) ─────────────────────────
   --opsd-diversity-metric token_jsd    # k-center greedy distance. Alt: unigram_jsd → selects BEFORE q-forward, saves (K_b - N) q-forwards/sample
   --opsd-diversity-top-k 128           # top-K vocab truncation for token-JSD

   # ── Teacher snapshot ───────────────────────────────────────────────
   --opsd-freeze-teacher                # default; flip to --no-opsd-freeze-teacher for legacy "teacher = current student" mode

   # ── Memory knob (lower → safer; raise on small V/T to reduce launch overhead) ─
   --opsd-kl-chunk 256                  # token-axis chunk for the vocab-parallel KL/RKL forward+backward
)

# Performance & Parallelism
PERF_ARGS=(
   --tensor-model-parallel-size 1
   # sequence-parallel only helps when TP>1; with TP=1 it's pure overhead.
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
   # Paper Table 6 (OPSD column): LearningRate = 5e-6.
   --lr 5e-6
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
   --sglang-mem-fraction-static 0.3
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
    --num-rollout 400
# Cleanup
# ray stop --force
