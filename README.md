# How fast can a 26B MoE run on a desktop with no GPU?

*An honest, reproducible map of what limits **Gemma-4-26B-A4B** on a consumer CPU — the recipe, the
roofline, and the dead-ends, so you don't have to rediscover them.*

> **Reference machine:** Intel i9-13900K · 64 GB DDR5-4800 · **no GPU.**
> **Model:** Gemma-4-26B-A4B — 26B total, ~3.8B active, 128 experts (top-8), 262K vocab.

---

## TL;DR

Two honest numbers on the same chip — they answer two different questions:

| question | answer | bound by |
|---|---|---|
| **one request, as fast as possible?** (latency) | **~40 tok/s**, lossless | memory bandwidth |
| **most total work across many requests?** (throughput) | **~124 tok/s** | CPU compute |

And one finding that surprised me enough to reorganize the whole project:

> **For this model, you quantize the *LM head*, not the experts.**
> The 262K-vocab head is **32%** of the bytes read per token; the experts are only **16%**.

Nothing here is a new algorithm or a speed record. The value is the *map* — every lever measured, including
the ones that didn't work, and the one number I got excited about that turned out to be noise.

---

## What I was after

A simple question: a modern **26B mixture-of-experts** model, on a **normal desktop with no GPU**, decoding
**one stream** as fast as possible — and kept **lossless** (the big model's output never changes; speedups
come from verification, not approximation). That regime — large MoE, single-stream, no GPU — is the one
almost nobody writes down numbers for. So I went and measured it, end to end.

## Where it started

Plain `Q4_0`, one stream: **~25 tok/s**, and flat whether I used 8 threads or 24. That flatness was the first
real clue — the cores weren't the limit, the **memory bus** was. Everything after follows from that one fact:

> **decode speed = memory bandwidth ÷ bytes read per token.**

## The path — what worked, what didn't

**1. Speculative decoding — the real win.** Google ships an official MTP "companion" drafter alongside
Gemma-4. It proposes a few tokens; the big model verifies them in one pass. Lossless, and it took us from
**~25 → ~40 tok/s**. This is the single biggest lever, and it's the deployed default.

**2. Top-3 routing — basically free.** Run 3 of the 8 selected experts instead of 8. The scary part was a
1.6× jump in raw-text perplexity — but that turned out to be an artifact of measuring an instruction model on
raw Wikipedia. On *actual* tasks (math, facts, code), top-3 and top-8 give equivalent answers. Verified, not
assumed.

**3. "Where are the bytes?" — the surprise.** I finally *counted* the bytes read per token, straight from the
model file. The standard MoE wisdom is "quantize the experts." But here the experts are only **16%** of
per-token bytes — the always-on attention + dense layers are **52%**, and the **262K-vocab LM head is 32%**.
That head, it turns out, compresses to **2.4-bit with no measurable quality loss.** The experts were never the
right target.

**4. Aggressive quantization — an honest dead-end (for serving).** Shrinking the model *does* speed up plain
decode (+18%). But stacked with speculative decoding it gave **0%** — a 32%-smaller model ran the same ~41
tok/s. Quantization and spec-decode fight over the *same* resource (memory bandwidth), and spec already wins
it. I briefly measured a 43.6 and got excited; re-running showed it was noise around ~41. That correction is in
the docs on purpose.

**5. The bandwidth wall — real, and I made sure.** A direct memory benchmark says the RAM can stream
**64.5 GB/s**; decode already uses **~78%** of that (weight-repacking, on by default, does most of the work).
The gap that's left is intrinsic to how MoE reads memory, not a lazy engine — pinning, threads, and more
quantization all moved it by ~nothing. So **~40 tok/s single-stream is near the hardware ceiling** for this RAM.

**6. The reframe — batching.** Single-stream is bandwidth-walled. But the moment you serve *several* requests
at once, each weight is read **once** and reused across all of them — the bottleneck flips from memory to
compute, and the 24 cores that sat idle the whole time finally earn their keep: **~124 tok/s aggregate.** This
is the vLLM-style throughput win, and it works fine on a CPU.

## Where it landed

| regime | ceiling on the reference i9 | bound by | how you'd push it |
|---|---|---|---|
| single-stream **latency** | **~40 tok/s** lossless | memory bandwidth | faster RAM / more channels |
| aggregate **throughput** | **~124 tok/s** | CPU compute | more cores / wider SIMD |

So "100 tok/s on a no-GPU desktop?" — **yes, as throughput, today.** As single-stream latency, no: that's a
bandwidth wall, and clearing it needs faster RAM or a platform with more memory channels (a workstation does it
easily). Two ceilings, 3× apart, on the same chip — and now you know exactly which physical resource each one
hits.

## What's honestly novel — and what isn't

I want to be straight about this, because overclaiming would undercut the point.

**Not novel:** the techniques. Speculative decoding, expert routing, quantization, batching — all known. None
of the raw numbers is a record.

**Worth sharing:**
- the **byte budget** that says *head, not experts* for this model — measured, counterintuitive, and not
  written down anywhere I could find;
- a **complete, honest map** of a regime (large MoE, single-stream, no-GPU CPU) that's usually left blank;
- the **dead-ends and the one correction**, because most writeups quietly drop those, and they're exactly what
  saves the next person a weekend.

## Reproduce it

Everything runs on **public models** (the base model and the MTP drafter are both official Google releases).

```bash
# 1. get the models (or grab our prebuilt GGUFs — see scripts/setup)
scripts/setup/download_models.sh
# 2. build the engines (or use the prebuilt binaries in Releases)
scripts/setup/build_engines.sh
# 3. measure your own machine
scripts/bench/single_stream.sh     # the ~40 tok/s latency recipe
scripts/bench/batched.sh           # the ~124 tok/s throughput recipe
```

Step-by-step recipes: [`docs/02-recipe-latency.md`](docs/) and [`docs/03-recipe-throughput.md`](docs/).
How every number was measured: [`docs/05-methodology.md`](docs/).

## Bring your own hardware

The most interesting open question — *does faster RAM clear the single-stream wall?* — I couldn't test on one
machine. Run the benchmark and **PR your row** to [`results/community.csv`](results/): your CPU, RAM speed, and
your two numbers. Especially curious about DDR5-6400+ and anything with more than two memory channels.

## Credits & license

- **Model:** Gemma-4-26B-A4B and its MTP drafter — © Google, used under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- **Engines:** built on [llama.cpp](https://github.com/ggml-org/llama.cpp) and two forks (one for the IQK quant
  kernels, one for MTP spec-decode); see [`docs/05-methodology.md`](docs/).
- **This repo's** scripts and docs are MIT-licensed.

*This is a measurement project. If a number here doesn't reproduce on your machine, that's a bug worth an
issue — open one.*
