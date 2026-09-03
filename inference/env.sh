#!/bin/bash
# ============================================
# Centralized configuration for PodBench vLLM inference.
# Edit this file to change the model, paths, and generation hyperparameters.
# run_inference.sh sources it, so this is the single source of truth.
# ============================================

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Model path (local directory or HuggingFace model ID)
MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"

# Display name, used for output directory naming
MODEL_NAME="${MODEL_NAME:-your-model-name}"

# Benchmark query file
INPUT_DATA_FILE="${INPUT_DATA_FILE:-${ROOT_DIR}/benchmark_query/podbench_800.json}"

# Output and log directories
OUTPUT_BASE_DIR="${SCRIPT_DIR}/llm_infer_results"
LOG_DIR="${SCRIPT_DIR}/infer_logs"

# ---------- Parallelism ----------
# Values here are defaults; an inline environment variable takes precedence,
# e.g. TP_SIZE=2 bash inference/run_inference.sh

# GPUs per model replica. Keep at 1 unless the model does not fit on one GPU.
TP_SIZE="${TP_SIZE:-1}"

# Model replicas. Defaults to (visible GPUs / TP_SIZE) when left empty.
DP_SIZE="${DP_SIZE:-}"

# Shard MoE experts across replicas instead of replicating them.
ENABLE_EP="${ENABLE_EP:-false}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# Optional context-length cap; leave empty to use the model config's value.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

# ---------- Generation Hyperparameters ----------
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12000}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-40}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.005}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-1}"

# 0 = instruct mode, 1 = thinking mode
IS_THINKING_MODE="${IS_THINKING_MODE:-0}"

# ---------- Derived (do not edit) ----------
GENERATION_CONFIG_STR="max_new_token_${MAX_NEW_TOKENS}_temperature_${TEMPERATURE}_top_p_${TOP_P}_top_k_${TOP_K}_return_num_${NUM_RETURN_SEQUENCES}"
