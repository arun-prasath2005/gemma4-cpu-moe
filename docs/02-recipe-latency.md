# recipe: ~40 tokens/sec, single stream, lossless

This is the latency recipe: one request, as fast as possible, with output identical to plain Q4_0. Two
ingredients, both lossless, and they stack.

## what you need

- the base model, Gemma-4-26B-A4B at Q4_0 (about 13 GB)
- the MTP drafter, Google's official companion model, quantized to Q4 (about 300 MB)
- the spec-decode engine (the build script sets this up; see [methodology](05-methodology.md))

Both models are public Google releases. `scripts/setup/download_models.sh` grabs them.

## the two levers

**Run 3 experts instead of 8.** The router picks 8; you tell it to run 3. One flag:

    --override-kv gemma4.expert_used_count=int:3

This is the "free speed" from the byte budget. It isn't bit-identical to top-8, but it gives the same answers
on real tasks (I checked, after a perplexity number nearly talked me out of it; that story is in the
[dead-ends](04-dead-ends.md)).

**Speculative decoding with the MTP drafter.** The little model proposes a few tokens, the big one verifies
them all in a single pass:

    --model-draft mtp-q4.gguf --spec-type draft-mtp --spec-draft-n-max 3

`n-max 3` is the sweet spot for this drafter. This part is exactly lossless: the big model still emits every
token it would have, just in fewer passes.

## the whole command

    llama-cli -m gemma-4-26B_q4_0-it.gguf \
      --override-kv gemma4.expert_used_count=int:3 \
      --model-draft mtp-q4.gguf --spec-type draft-mtp --spec-draft-n-max 3 \
      -t 8 -f your_prompt.txt -n 200 --temp 0 -st --simple-io --no-display-prompt

Or just run `scripts/bench/single_stream.sh`, which wraps this and prints the tokens/sec.

## what to expect

On the reference i9 (8 P-cores, DDR5-4800): plain Q4_0 with neither lever is about 25 tok/s. Add both and
you're in the high 30s to low 40s, depending on how cool and idle the machine is (the careful number is 37.7;
an idle box turbos to ~41).

One note on threads: use 8, not 24. Single-stream decode is bandwidth-bound, so the 16 E-cores add nothing
here and pinning to the 8 P-cores is as good as it gets. Batched is a completely different story; that's the
[throughput recipe](03-recipe-throughput.md).
