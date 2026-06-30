#!/usr/bin/env bash
# The ~124 tok/s batched-throughput recipe (docs/03). Read the S_TG column (aggregate gen tok/s).
set -euo pipefail
ENG="${ENGINES:-$(pwd)/engines}"
MOD="${MODELS:-$(pwd)/models}"
BB="$ENG/atomic-llama/build/bin/llama-batched-bench"
THREADS="${THREADS:-24}"    # all physical cores. batched is compute-bound, so the E-cores help here.
BATCHES="${BATCHES:-1,2,4,8,16,32}"

"$BB" -m "$MOD/gemma-4-26B_q4_0-it.gguf" \
  --override-kv gemma4.expert_used_count=int:3 \
  -c 8192 -npp 64 -ntg 128 -npl "$BATCHES" -t "$THREADS"
