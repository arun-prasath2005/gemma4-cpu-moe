# the byte budget: where the bytes actually go

The whole project runs on one equation. A single token costs you `bytes read / bandwidth`, and the bandwidth
is fixed by your RAM. So if you want to go faster without buying hardware, you read fewer bytes per token.
Which means the very first thing you should do, before optimizing anything, is find out where the bytes
actually are. I didn't do this first. I should have.

## counting it

Here's the thing people skip: "bytes per token" is not the size of the model on disk. For a mixture-of-experts
model those are wildly different. The model is 13.4 GB on disk, but you don't read all of it to make one token.
You read all of the attention and dense layers (those run every token), the full output head (you need a score
for every word in the vocabulary), and only the handful of experts that actually fire (3 of 128, in our setup).
Everything else just sits in RAM doing nothing that step.

So I added up exactly what gets touched per token, straight from the model file. For Gemma-4-26B-A4B at top-3:

    always-on (attention + dense layers)     972 MB   52%
    output head (262K-vocab projection)      606 MB   32%
    active experts (3 of 128)                301 MB   16%
    -----------------------------------------------------
    total per token                         1878 MB

Quick sanity check: 1878 MB × 24.75 tokens/sec ≈ 46.5 GB/s, which is right about what this memory bus actually
delivers. The numbers hold together.

## the surprise

Look at that table again. On disk, the experts basically *are* the model. They're something like 90% of the
parameters, and the entire instinct around MoE quantization is "go squeeze the experts." But per token they're
the smallest slice, 16%, because only 3 of the 128 ever fire. They're huge but cheap.

The output head is the opposite. It's only 600 MB, but you read all of it on every single token, because to
pick the next token you need a score for all 262K of them. Gemma-4's vocabulary is enormous, and that head is a
[2816 × 262144] matrix you sweep end to end every step. Small on disk, expensive per token.

So for this model the byte you most want to cut lives in the head, not the experts. That's the opposite of the
usual advice, and it's the single most useful thing I learned here.

## and the head compresses beautifully

The head in the base model sits at about 6.5 bits per weight. I dropped it to 2.4 bits (IQ2_K) and went looking
for the damage. I couldn't find any. Greedy decoding on a word problem, some code, and a factual question all
came out identical in substance to the full-precision head. That's 600 MB → 225 MB on the single most-read
tensor in the whole model.

2.4 bits is about the floor, though. I tried 1.75 (IQ1_M) and it managed to be worse on both axes at once: it
got *slower* (the cores spend more time unpacking the tighter format than they save reading fewer bytes) and it
started producing garbage, little repetition loops in the middle of the arithmetic. So IQ2_K is the sweet spot,
not a step on the way to something smaller.

One honest caveat, and it's a good one. Cutting the head speeds up *plain* decode, but it does almost nothing
once speculative decoding is in the mix, and the reason why is genuinely interesting ([the dead-ends](04-dead-ends.md)
has that story). Read the byte budget as a map of the model, not a to-do list of guaranteed speedups.

## reproduce it

The counting is a small script that reads the GGUF and tallies bytes by tensor type and role:

    python scripts/analysis/byte_budget.py path/to/gemma-4-26B_q4_0-it.gguf

It prints the table above for any GGUF you point it at, plus the per-type breakdown if you want to check the
head's bit-width for yourself.
