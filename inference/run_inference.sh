#!/bin/bash
# Launch PodBench inference with vLLM data parallelism.
#
# Parallelism is derived from the visible GPUs: the model is replicated dp times,
# each replica spanning tp GPUs. Override TP_SIZE / DP_SIZE in env.sh or inline.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash inference/run_inference.sh
#   TP_SIZE=2 bash inference/run_inference.sh          # model needs 2 GPUs
#   ENABLE_EP=true TP_SIZE=2 bash inference/run_inference.sh   # MoE models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# ---------- GPU discovery ----------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
    N_GPU=${#GPU_LIST[@]}
else
    N_GPU="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    [[ -z "${N_GPU}" || "${N_GPU}" == "0" ]] && N_GPU=1
fi

# ---------- Parallelism ----------
# One replica per GPU by default. Models too large for a single GPU need TP_SIZE>1,
# which reduces the replica count accordingly.
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-$(( N_GPU / TP_SIZE ))}"
ENABLE_EP="${ENABLE_EP:-false}"

if (( DP_SIZE < 1 )); then
    echo "ERROR: tp=${TP_SIZE} exceeds the ${N_GPU} available GPUs" >&2
    exit 1
fi

NEED_GPU=$(( TP_SIZE * DP_SIZE ))
if (( NEED_GPU > N_GPU )); then
    echo "ERROR: tp(${TP_SIZE}) x dp(${DP_SIZE}) = ${NEED_GPU} > available GPUs (${N_GPU})" >&2
    exit 1
fi

if [[ ! -f "${INPUT_DATA_FILE}" ]]; then
    echo "ERROR: benchmark file not found: ${INPUT_DATA_FILE}" >&2
    exit 1
fi

OUTPUT_FILE="${OUTPUT_BASE_DIR}/${MODEL_NAME}/${GENERATION_CONFIG_STR}/generations.jsonl"

echo "=============================================="
echo "PodBench vLLM Inference"
echo "  model      : ${MODEL_NAME} (${MODEL_PATH})"
echo "  input      : ${INPUT_DATA_FILE}"
echo "  output     : ${OUTPUT_FILE}"
echo "  gpus       : ${N_GPU}  (tp=${TP_SIZE}, dp=${DP_SIZE}, ep=${ENABLE_EP})"
echo "  generation : ${GENERATION_CONFIG_STR}"
echo "  thinking   : ${IS_THINKING_MODE}"
echo "=============================================="

mkdir -p "${LOG_DIR}"

EXTRA_ARGS=()
[[ "${ENABLE_EP}" == "true" ]] && EXTRA_ARGS+=(--enable-expert-parallel)
[[ -n "${MAX_MODEL_LEN:-}" ]] && EXTRA_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")

python3 "${SCRIPT_DIR}/run_inference.py" \
    --model-path "${MODEL_PATH}" \
    --input-data-file "${INPUT_DATA_FILE}" \
    --output-data-file "${OUTPUT_FILE}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --data-parallel-size "${DP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --enable-thinking "${IS_THINKING_MODE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --repetition-penalty "${REPETITION_PENALTY}" \
    --num-return-sequences "${NUM_RETURN_SEQUENCES}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Done. Generations: ${OUTPUT_FILE}"
