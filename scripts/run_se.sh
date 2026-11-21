#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Connect to an available model API; this script does not deploy a model.
# Choose configs/se_api.yaml for an existing OpenAI-compatible API.
CONFIG="${CONFIG:-configs/se.yaml}"
DATASET="${DATASET:-lsrtransform}"

# Sub-benchmarks (select one with DATASET):
# lsrtransform   : LSR-Transform (default)
# phys_osc       : Physics Oscillator
# matsci         : Materials Science
# bio_pop_growth : Biological Population Growth
# chem_react     : Chemical Reaction Kinetics

ARGS=(--dataset "$DATASET" --searcher_config "$CONFIG")
# Optional override for the local vLLM client; API mode uses OPENAI_API_URL.
if [[ -n "${VLLM_PORT:-}" ]]; then
  ARGS+=(--local_llm_port "$VLLM_PORT")
fi

python eval.py "${ARGS[@]}"
