# methodology: how every number was measured

The setup, so you can trust the numbers or go poke holes in them.

## the machine

One desktop. Intel i9-13900K (8 performance cores plus 16 efficiency cores), 64 GB DDR5-4800, no GPU. AVX2, no
AVX-512 (the 13900K has it fused off). Nothing overclocked, the RAM runs at its stock 4800.

## the engines

Two forks of llama.cpp, both built for AVX2:

- one with the **IQK low-bit quant kernels** and runtime weight repacking, used for the byte-budget and
  head-quant work (it has the 2.4-bit formats and a fast CPU path for them)
- one with the **MTP speculative-decode path** (`--spec-type draft-mtp`) and the batched benchmark, used for
  the latency and throughput recipes

The exact repos and commits are pinned in `scripts/setup/build_engines`. You don't need both for every recipe;
the script notes which is which.

## what "lossless" means, and how it's checked

Two things are going on, and only one of them is bit-exact.

Speculative decoding is exactly lossless: the big model verifies every proposed token, so the output is
identical, token for token, to what plain Q4_0 would produce. Nothing to check, it's structural.

Top-3 routing is not bit-identical to top-8, it's an approximation. I checked it the only way that's meaningful
for an instruction model: generate on real prompts (a word problem, some code, a factual question) at top-3 and
top-8, compare. Same answers. The raw-perplexity number is misleading here, which is its own story in the
[dead-ends](04-dead-ends.md).

So "~40 tok/s lossless" means: identical to Q4_0 output, with top-3 as a separately-validated,
quality-equivalent approximation layered on top.

## the bandwidth number

The "RAM does 64.5 GB/s" figure is from a compiled STREAM-triad benchmark
(`scripts/bench/stream_bandwidth.c`), not a spec sheet. It measures what the machine actually sustains, which
is 84% of the 76.8 GB/s the DDR5-4800 spec implies. Decode reaches about 78% of that 64.5, with repacking on.
That's how I know ~40 single-stream is near the silicon's ceiling and not just a lazy kernel.

## a word on reproducibility

Tokens/sec drifts with machine state, turbo, temperature, whatever else is running. The careful single-stream
number on this box is 37.7; an idle, cool machine turbos to ~41. The batched numbers are steadier because
they're compute-bound. If your machine lands somewhere different, that's expected. The part that should
reproduce is the *shape*: threads flat for latency and threads helping for throughput, the byte-budget
proportions, the head compressing for free. If the shape doesn't hold, that's a real bug, open an issue.
