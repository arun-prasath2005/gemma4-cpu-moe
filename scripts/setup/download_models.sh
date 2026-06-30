#!/usr/bin/env bash
# Download the GGUFs the recipes need, into ./models.
# Base model + MTP drafter are both official Google releases; we host the exact Q4 GGUFs we
# benchmarked so you get a clean one-click repro and the precise numbers.
set -euo pipefail
DEST="${1:-$(pwd)/models}"
mkdir -p "$DEST"

# The HF repo holding the benchmarked GGUFs. Override with HF_REPO=... if you mirror it.
HF_REPO="${HF_REPO:-arunprasath/gemma4-cpu-moe-gguf}"

command -v huggingface-cli >/dev/null 2>&1 || pip install -U "huggingface_hub[cli]"
huggingface-cli download "$HF_REPO" \
  gemma-4-26B_q4_0-it.gguf mtp-q4.gguf \
  --local-dir "$DEST"

echo "models in $DEST"
# Alternative (no mirror): pull google/gemma-4-26B-A4B-it and google/gemma-4-26B-A4B-it-assistant
# and convert+quantize them yourself with the fork's convert_hf_to_gguf.py + llama-quantize Q4_0.
