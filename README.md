# Running Gemma-4 26B at 124 tokens/sec on a CPU, no GPU

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22762963.svg)](https://doi.org/10.5281/zenodo.22762963)

> The full write-up of this build, with the byte budget and the dead ends: [apeg.dev/writing/running-gemma4-26b-on-a-cpu](https://apeg.dev/writing/running-gemma4-26b-on-a-cpu/). What self-hosting an LLM actually requires, beyond the hardware: [apeg.dev/writing/self-hosted-llm-without-a-gpu](https://apeg.dev/writing/self-hosted-llm-without-a-gpu/). By [Arun Prasath E G](https://apeg.dev/about/).

I wanted to see how fast you can run a 26B mixture-of-experts model on a regular desktop, with no graphics
card at all. Just the CPU: an i9-13900K, 64GB of ordinary DDR5, the kind of box a lot of people already have
sitting around.

Short version: about 40 tokens/sec for a single stream (and lossless, which I'll get to), or about 124
tokens/sec if you're serving a handful of requests at once. For a 26B model with no GPU I think that's a
little wild, and most of what follows is me working out why it's possible and where exactly the walls are.

One thing surprised me enough that I'll just say it up front. Everyone's instinct with a mixture-of-experts
model is to quantize the experts, because that's where all the parameters are. But I sat down and actually
counted the bytes you read per token, and the experts are only 16% of it. The output head, the big matrix that
projects to the 262K-token vocabulary, is 32%, twice as much. For this model the head is the thing to
compress, not the experts. I did not expect that.

## the model, and what "lossless" means

Gemma-4-26B-A4B: 26B parameters total, but only ~3.8B active per token, because it's a mixture of experts
(128 of them, 8 used at a time).

When I say lossless I mean it literally. I never change what the big model outputs. Every trick here either
skips work the model was going to throw away anyway, or guesses ahead and has the model check the guess. So
the tokens coming out are exactly the tokens plain Q4_0 would have produced, just faster. No "quality is
basically the same" handwaving, which matters if you're going to trust the numbers.

## where it starts, and the only equation you need

Plain Q4_0, one token at a time: about 25 tokens/sec. The interesting part wasn't the number, it was that it
didn't budge when I changed the thread count. 8 threads, 24 threads, identical. That's the tell: if more cores
don't help, you're not compute-bound, you're waiting on memory.

And that's basically the whole story in one line:

    tokens/sec = memory bandwidth / bytes read per token

To make one token you stream almost the entire active model out of RAM, once. So your speed is just how fast
RAM hands you bytes, divided by how many bytes you have to pull. Everything I tried after this is a fight over
one of those two numbers: make RAM faster, or read fewer bytes.

## the path

**Speculative decoding** is the big unlock, and the nice part is Google already did the hard work. Gemma-4
ships with a little official "MTP" drafter that guesses the next few tokens; the big model then verifies all of
them in a single forward pass instead of grinding them out one by one. When the guesses are good you get
several tokens for the price of one pass. This is what takes you from ~25 to ~40 tok/s, and it stays lossless
because the big model still signs off on every token. Biggest lever in the whole project, and you basically
get it for free.

**Running 3 of the 8 experts.** The router picks 8 experts per token; I just run 3 of them. This is where I
almost talked myself out of a good thing. Perplexity on raw Wikipedia jumped 1.6x and that looked alarming, but
it's an artifact: an instruction-tuned, "thinking" model scores terribly on raw text no matter what, and that
bad regime exaggerates small differences. When I actually read the outputs on real prompts, a word problem,
some code, a factual question, top-3 and top-8 give the same answers. So it's free speed, but only because I
went and looked instead of believing the scary number. There's a lesson in there.

**Then I stopped guessing about the bytes and counted them.** This is the part I keep coming back to. Per
token: the always-on stuff (attention plus the dense layers) is 52%, the experts are 16%, and the output head
is 32%. The head is enormous because the vocabulary is 262K tokens, so it's a giant matrix you read in full on
every single token. And it turns out you can crush it down to about 2.4 bits per weight with no measurable
change to the outputs. The standard "quantize the experts" move would have been optimizing the smallest slice.

**Quantization, and the trap.** Fewer bytes should mean more speed, and it does: shrink the model and plain
decode gets ~18% faster. But the moment I stacked it on top of speculative decoding, it bought nothing. A model
32% smaller on disk ran at the exact same ~41 tok/s. Quantization and spec decoding are both attacking the same
thing (memory bandwidth), and spec already cashed it in, so they don't multiply. I measured a 43.6 once and got
briefly excited, then re-ran it and it was just noise around 41. I'm leaving that whole detour in the docs on
purpose, because "the obvious stack didn't work" is exactly what people tend not to write down.

**The wall, and checking that it's actually the wall.** I wanted to know whether ~40 is a real hardware limit
or just me leaving performance on the floor, so I measured the raw memory bandwidth with a little benchmark.
The RAM can do about 64.5 GB/s, and decode is already using ~78% of it. I tried the usual things to recover the
rest (pinning threads to specific cores, different thread counts, heavier quant) and none of it moved. The gap
that's left is just how a mixture-of-experts reads memory, scattered rather than in one clean sweep. So ~40
single-stream isn't laziness, it's close to what this silicon can do with this RAM.

**The way around the wall.** Single-stream is stuck on bandwidth. But look at that equation again, it's per
token, for one stream. The moment you serve a few requests at once, you read each weight once and reuse it
across all of them. Now you're not bandwidth-bound, you're compute-bound, and all 24 cores that were sitting
idle the whole time suddenly have work to do. Throughput climbs to ~124 tokens/sec. This is the trick vLLM made
famous on GPUs (continuous batching), and it works fine on a CPU. It just took flipping the goal from "lowest
latency" to "most total work."

## so, can you hit 100 tokens/sec on a CPU?

Yes, but you have to be precise about which 100. As throughput across a few streams you're already past it
(124). As single-stream latency, no: that one's pinned at ~40 by the memory bus, and the only ways up are
faster RAM or a machine with more memory channels (a workstation does it easily, more channels means more
bandwidth). Two ceilings, about 3x apart, on the same chip, bound by different things: one by how fast your RAM
is, the other by how many cores you have. Worth knowing which one you're hitting before you go optimizing.

## what's actually here

The headline is the speed, but the thing I'd really point you to is the byte budget. Working out that for a
262K-vocab model the output head is where the bytes are, not the experts, and that you can crush it to 2.4 bits
for free, changes how you'd quantize a model like this in the first place. Around that is a full set of real
numbers for a setup people mostly don't bother to measure (a big MoE, single stream, no GPU), with the
dead-ends and the one correction left in, because that's what makes a recipe you can actually trust and build
on.

If a number here doesn't reproduce on your machine, that's a bug, open an issue.

## try it

Everything runs on public models (the base model and the drafter are both official Google releases).

    scripts/setup/download_models.sh    # or grab the prebuilt GGUFs, see scripts/setup
    scripts/setup/build_engines.sh      # clones + builds both forks, ~15 min
    scripts/bench/single_stream.sh      # the ~40 tok/s lossless recipe
    scripts/bench/batched.sh            # the ~124 tok/s throughput recipe

Step-by-step is in docs/. And if you run it, I'd love a row in results/community.csv with your CPU and RAM
speed. I'm especially curious whether faster RAM (DDR5-6400+, or more than two channels) breaks the
single-stream wall, since I could only test the one machine.

## credits

Gemma-4 and its drafter are © Google, under the Gemma Terms. Built on llama.cpp and a couple of forks, one for
the quant kernels and one for the MTP spec decoding; details in docs/. The scripts and writeup here are MIT.

## cite

Archived on Zenodo, so the numbers have a fixed record: [10.5281/zenodo.22762963](https://doi.org/10.5281/zenodo.22762963)
(this release) and [10.5281/zenodo.22762962](https://doi.org/10.5281/zenodo.22762962) (always the latest release).

    Arun Prasath E G. Running Gemma-4 26B at 124 tokens/sec on a CPU, no GPU: recipe, byte budget, roofline.
    Version 1.0, September 2026. https://doi.org/10.5281/zenodo.22762963

GitHub also reads the CITATION.cff in this repo, so the "Cite this repository" button on the right gives the same thing.
