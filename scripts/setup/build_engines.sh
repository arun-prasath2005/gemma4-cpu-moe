#!/usr/bin/env bash
# Build the two public llama.cpp forks this project uses, for AVX2. See docs/05-methodology.md.
#   atomic-llama  (AtomicBot-ai)  -> MTP speculative decoding + llama-batched-bench
#   ik_llama      (ikawrakow)     -> IQK low-bit quant kernels + runtime repack
# Pinned to the commits the numbers were measured on. Needs: git, cmake, a C++ compiler.
set -euo pipefail
ROOT="${1:-$(pwd)/engines}"
JOBS="$(nproc 2>/dev/null || echo 4)"
# AVX512 is OFF because the reference i9-13900K doesn't have it. If your CPU does, flip it ON.
CMFLAGS="-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF \
         -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF"
mkdir -p "$ROOT"; cd "$ROOT"

echo ">> atomic-llama (spec-decode + batched bench)"
[ -d atomic-llama ] || git clone https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant atomic-llama
( cd atomic-llama && git checkout --quiet d86eb0b
  cmake -B build $CMFLAGS
  cmake --build build --config Release -j "$JOBS" --target llama-cli llama-batched-bench )

echo ">> ik_llama (quant kernels + repack)"
[ -d ik_llama ] || git clone https://github.com/ikawrakow/ik_llama.cpp ik_llama
( cd ik_llama && git checkout --quiet f96eadd
  cmake -B build $CMFLAGS
  cmake --build build --config Release -j "$JOBS" --target llama-cli llama-quantize )

echo "done. binaries under $ROOT/{atomic-llama,ik_llama}/build/bin"
