# the dead-ends (so you don't re-walk them)

The stuff that didn't work usually gets deleted from writeups, which is a shame, because it's the most reusable
part. Here's what I tried that didn't pan out, and why.

## quantization doesn't stack with speculative decoding

This is the big one, and it fooled me for a bit. The byte budget says cut bytes, quantization cuts bytes, so
quantization should be faster. And on plain decode it is: a heavily-quantized model (most of it down to ~2-3
bits) runs about 18% faster than Q4_0.

Then you turn on speculative decoding and the gain vanishes. A model 32% smaller on disk ran at the *exact*
same ~41 tok/s as Q4_0 once spec was in the mix. I even measured a 43.6 once and got briefly excited, then
re-ran it and it was just noise around 41.

The reason is that the two are fighting over the same resource. Both speed things up by moving fewer bytes over
the memory bus per token, and speculative decoding already does that (it checks several tokens per weight-read).
By the time spec has had its way there aren't many bytes left for quantization to save. They overlap, they
don't multiply. So if you're already running spec, don't expect a smaller model to help, at least not on a
bandwidth-bound CPU.

## you can't push the experts below 4 bits anyway (and it wouldn't matter)

I wanted to quantize the experts hard. Turns out you can't, easily. The expert "down" projections have a width
(704) that isn't a multiple of the 256-wide block the good low-bit formats need, so they quietly fall back to 4
bits no matter what you ask for. I went looking for a format that fits, including the fancy trellis quants, and
they all fall back the same way.

But it doesn't matter, and that's the point. From the byte budget the experts are only 16% of per-token bytes,
and the stuck part is a fraction of that. Forcing it down would buy maybe 5%, nowhere near worth a custom
kernel. It's the flip side of "go after the head": the experts aren't just the wrong target, they're also the
hard one.

## the top-3 perplexity scare

When I dropped from 8 experts to 3, perplexity on raw Wikipedia jumped 1.6x. That looks like you broke the
model, and I almost backed it out. But raw-text perplexity is meaningless for an instruction-tuned, "thinking"
model. It scores terribly on plain text whether you touch the experts or not, and that bad regime blows small
differences way out of proportion. On actual prompts, top-3 and top-8 give the same answers. The lesson: when
one aggregate metric panics you, go read actual outputs before you act on it.

## the head below 2.4 bits

Covered in the [byte budget](01-byte-budget.md), but it belongs here too. The 1.75-bit head (IQ1_M) is both
slower (the cores spend longer unpacking the tighter format than they save on bytes) and broken (little
repetition loops in the arithmetic). 2.4 bits (IQ2_K) is the floor, not a waypoint to something smaller.
