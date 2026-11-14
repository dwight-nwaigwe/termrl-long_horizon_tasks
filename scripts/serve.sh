#!/usr/bin/env bash
# Serve a model with vLLM under the name "policy" on port 8000.
#   bash scripts/serve.sh [MODEL]          (MODEL = HF id or local path)
# vLLM runs in its own throwaway environment via uvx, so its torch/CUDA pins
# never conflict with the training environment.
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
[ -f /etc/profile.d/termrl.sh ] && source /etc/profile.d/termrl.sh

exec uvx --from "vllm==${VLLM_VERSION:-0.19.1}" vllm serve "$MODEL" \
  --served-model-name policy \
  --port "${PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --gpu-memory-utilization "${GPU_MEM:-0.85}" \
  --tensor-parallel-size "${TP:-1}" \
  "${@:2}"
