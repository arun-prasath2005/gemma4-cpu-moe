# recipe: ~124 tokens/sec, batched throughput

This is the throughput recipe: instead of one request as fast as possible, you run several at once and measure
the total. Different question, very different answer, and it's where the no-GPU machine gets to show off.

## why batching changes everything

Single-stream decode is bandwidth-bound: you read each weight out of RAM once per token. But if you have, say,
16 requests in flight, you read each weight once and use it for all 16 of them. The bytes-per-token drop by
16x, and the bottleneck slides off the memory bus and onto the cores. Now the 16 E-cores that did nothing for
single-stream have real work, and they pull their weight.

It's the same trick vLLM made famous (continuous batching). It just usually isn't pointed at a CPU.

## measuring it

The engine ships a batched benchmark. Sweep the number of parallel streams and watch the generation throughput
(the `S_TG` column):

    llama-batched-bench -m gemma-4-26B_q4_0-it.gguf \
      --override-kv gemma4.expert_used_count=int:3 \
      -c 8192 -npp 64 -ntg 128 -npl 1,2,4,8,16,32 -t 24

Or run `scripts/bench/batched.sh`.

## what to expect

Aggregate generation tokens/sec on the reference i9, sweeping batch size at 8 threads:

    batch  1     28
    batch  4     69
    batch  8     81
    batch 16     93

and then the interesting part: bump that batch-16 run from 8 threads to all 24 cores and it jumps from 93 to
114, and batch 32 gets you to 124. That contrast is the whole story in one experiment. For single-stream, more
threads did nothing (you were waiting on memory). Here, more threads help, because batching made it
compute-bound and the cores finally matter.

## the catch

This is aggregate, not per-stream. At batch 32 each individual request is getting about 4 tok/s; at batch 16,
about 7. So it's the right number if you're serving a handful of users who each tolerate a slower stream, and
the wrong number if you have one user who wants their answer now. Pick the batch size for your situation. And
note this runs without speculative decoding, the two don't combine cleanly (spec is a latency trick, batching
is a throughput trick).
