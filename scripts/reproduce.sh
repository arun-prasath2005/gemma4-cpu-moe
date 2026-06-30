#!/usr/bin/env bash
# One command: build the engines, fetch the models, run both recipes.
# (Linux/macOS. Windows users: the same steps as .ps1 are coming; for now run the four scripts by hand.)
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/setup/build_engines.sh
bash scripts/setup/download_models.sh

echo; echo "================ single-stream (latency) ================"
bash scripts/bench/single_stream.sh
echo; echo "================ batched (throughput) ==================="
bash scripts/bench/batched.sh
echo; echo "done. compare your numbers against results/i9-13900k.md, and PR your row to results/community.csv"
