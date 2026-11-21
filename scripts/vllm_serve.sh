#!/usr/bin/env bash
# Optional local model-serving example (default TP_SIZE: 4).
# Select GPUs externally; override TP_SIZE to match your hardware.
# Example: CUDA_VISIBLE_DEVICES=0 TP_SIZE=1 bash scripts/vllm_serve.sh

vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --dtype auto \
  --api-key token-abc123 \
  --port 10005 \
  --seed 42 \
  --tensor-parallel-size "${TP_SIZE:-4}" \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --swap-space 4 \
  --disable-log-requests
