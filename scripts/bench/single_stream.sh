#!/usr/bin/env bash
# The ~40 tok/s lossless single-stream recipe (docs/02): top-3 experts + MTP speculative decoding.
set -euo pipefail
ENG="${ENGINES:-$(pwd)/engines}"
MOD="${MODELS:-$(pwd)/models}"
CLI="$ENG/atomic-llama/build/bin/llama-cli"
THREADS="${THREADS:-8}"     # P-cores. single-stream is bandwidth-bound, so more threads don't help.
NPRED="${NPRED:-200}"
PROMPT="${PROMPT:-Explain how a CPU cache hierarchy reduces average memory latency, step by step.}"

PF="$(mktemp)"
printf '<|turn>user\n%s<turn|>\n<|turn>model\n<|channel>thought\n<channel|>' "$PROMPT" > "$PF"
"$CLI" -m "$MOD/gemma-4-26B_q4_0-it.gguf" \
  --override-kv gemma4.expert_used_count=int:3 \
  --model-draft "$MOD/mtp-q4.gguf" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-threads "$THREADS" \
  -t "$THREADS" -f "$PF" -n "$NPRED" --temp 0 --seed 1 -st --simple-io --no-display-prompt
rm -f "$PF"
