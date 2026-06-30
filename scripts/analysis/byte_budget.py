#!/usr/bin/env python3
"""Per-token active-byte budget for a GGUF model. See docs/01-byte-budget.md.

    python byte_budget.py path/to/model.gguf [--experts-used N]

Needs the gguf reader:  pip install gguf
"""
import sys, os, argparse
from collections import defaultdict
try:
    from gguf import GGUFReader
except ImportError:
    sys.exit("need the gguf reader:  pip install gguf")

# bits per weight, by ggml type
BPW = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q4_0": 4.5, "Q4_1": 5.0, "Q5_0": 5.5, "Q5_1": 6.0, "Q8_0": 8.5,
    "Q2_K": 2.625, "Q3_K": 3.4375, "Q4_K": 4.5, "Q5_K": 5.5, "Q6_K": 6.5625,
    "IQ1_S": 1.56, "IQ1_M": 1.75, "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5,
    "IQ2_K": 2.4375, "IQ3_K": 3.4375, "IQ3_S": 3.44, "IQ4_XS": 4.25, "IQ4_NL": 4.5, "IQ4_K": 4.5,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--experts-used", type=int, default=3,
                    help="experts active per token at serve time (default 3)")
    ap.add_argument("--experts-total", type=int, default=128)
    a = ap.parse_args()

    r = GGUFReader(a.gguf)
    always = head = expert = embd = 0.0
    per_type = defaultdict(float)
    for t in r.tensors:
        n = 1
        for d in t.shape:
            n *= int(d)
        ttype = t.tensor_type.name
        b = n * BPW.get(ttype, 4.5) / 8.0
        per_type[ttype] += b
        nm = t.name
        if "_exps" in nm:
            expert += b
        elif nm == "output.weight":
            head += b
        elif nm == "token_embd.weight":
            embd += b
        else:
            always += b
    if head == 0.0:           # tied embedding: token_embd IS the head, read in full every token
        head = embd

    active_exp = expert * (a.experts_used / a.experts_total)
    total = always + head + active_exp
    pct = lambda x: 100 * x / total
    print(f"\n{os.path.basename(a.gguf)}   (top-{a.experts_used} of {a.experts_total} experts)")
    print(f"  always-on (attention + dense)   {always/1e6:8.1f} MB   {pct(always):3.0f}%")
    print(f"  LM head (full-vocab projection) {head/1e6:8.1f} MB   {pct(head):3.0f}%")
    print(f"  active experts                  {active_exp/1e6:8.1f} MB   {pct(active_exp):3.0f}%")
    print(f"  ------------------------------------------------------")
    print(f"  total active bytes / token      {total/1e6:8.1f} MB")
    print(f"\n  bytes by type (whole file):")
    for k, v in sorted(per_type.items(), key=lambda x: -x[1]):
        print(f"    {k:8s} {v/1e9:6.2f} GB")

if __name__ == "__main__":
    main()
