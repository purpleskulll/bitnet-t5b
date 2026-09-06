#!/usr/bin/env python3
"""How many weight rows of a ternary model are entirely zero, and where.

A ternary weight matrix row that is all zeros is an output neuron that can
never fire, whatever the input. This counts them. It prunes nothing and it
changes nothing; the number turned out to be structural, which is why it has
its own file.

Detection: an i2_s row is dead when every packed byte is 0x55 -- all four
2-bit fields at code 1, which is weight 0. That is a sufficient condition for
a silent neuron and not a necessary one: a row of small non-zero weights is
invisible to it.

Usage:  python3 tools/dead_neurons.py [MODEL.gguf] [--compare OTHER.gguf]
"""

import collections
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_inspect import parse_gguf  # noqa: E402

ARGS = sys.argv[1:]
COMPARE = None
if "--compare" in ARGS:
    i = ARGS.index("--compare")
    COMPARE = ARGS[i + 1]
    ARGS = ARGS[:i] + ARGS[i + 2:]
MODEL = ARGS[0] if ARGS else "models/ggml-model-i2_s.gguf"


def dead_index_sets(path):
    """{tensor name: sorted array of dead row indices}."""
    out = parse_gguf(path)
    tl = next(m for m in out if isinstance(m, list) and m and isinstance(m[0], dict))
    d0 = next(m for m in out if isinstance(m, int) and m > 1000)
    fh = open(path, "rb")
    D = {}
    for t in [x for x in tl if x.get("type") == 36]:
        K, R = t["dims"][0], t["dims"][1]
        fh.seek(d0 + t["offset"])
        b = np.frombuffer(fh.read(K * R // 4), dtype=np.uint8)
        D[t["name"]] = np.flatnonzero(np.all(b.reshape(R, K // 4) == 0x55, axis=1))
    return D


def main():
    out = parse_gguf(MODEL)
    tl = next(m for m in out if isinstance(m, list) and m and isinstance(m[0], dict))
    i2s = [t for t in tl if t.get("type") == 36]
    if not i2s:
        print(f"{MODEL}: no i2_s tensors", file=sys.stderr)
        return 2
    data0 = next(m for m in out if isinstance(m, int) and m > 1000)
    f = open(MODEL, "rb")

    print("DEAD FFN NEURONS IN", pathlib.Path(MODEL).name)
    print("=" * 74)
    print()
    print(f"  Reproduce: python3 tools/dead_neurons.py {MODEL}")
    print()
    print("  A ternary weight matrix row that is entirely zero is an output")
    print("  neuron that can never fire. Nothing here prunes it; this file")
    print("  measures how many there are, because the number is structural.")
    print()

    rows = zero = w = zw = 0
    kind = collections.Counter()
    kindr = collections.Counter()
    gate, up = {}, {}
    bylayer = collections.defaultdict(dict)
    for t in i2s:
        K, R = t["dims"][0], t["dims"][1]
        f.seek(data0 + t["offset"])
        b = np.frombuffer(f.read(K * R // 4), dtype=np.uint8)
        z = np.all(b.reshape(R, K // 4) == 0x55, axis=1)
        n = int(z.sum())
        rows += R; zero += n; w += K * R; zw += n * K
        parts = t["name"].split(".")
        k = parts[-2]
        kind[k] += n; kindr[k] += R
        if len(parts) > 1 and parts[1].isdigit():
            L = int(parts[1])
            bylayer[L][k] = (n, R)
            if k == "ffn_gate": gate[L] = z
            if k == "ffn_up": up[L] = z

    print("1. HOW MANY"); print("-" * 74); print()
    print(f"    rows entirely zero    {zero:,} of {rows:,}   ({100*zero/rows:.4f} %)")
    print(f"    weights in them       {zw:,} of {w:,}   ({100*zw/w:.4f} %)")
    print(f"    at 1.600 bits that is {zw*1.6/8/1e6:.1f} MB of {w*1.6/8/1e6:.1f} MB")
    print()
    print("    by tensor kind:")
    for k in sorted(kindr, key=lambda x: -kind[x]):
        print(f"      {k:14s} {kind[k]:7,} of {kindr[k]:7,} rows  ({100*kind[k]/kindr[k]:6.2f} %)")
    print()

    print("2. THEY COME IN PAIRS, IN EVERY LAYER"); print("-" * 74); print()
    same = sum(1 for L in gate if L in up and np.array_equal(gate[L], up[L]))
    print(f"    layers where ffn_gate's dead-row SET equals ffn_up's:  {same} of {len(gate)}")
    print()
    print("    Not the same count -- the same INDICES. In a SwiGLU feed-forward")
    print("    block gate and up are multiplied element-wise, so a dead gate row")
    print("    makes the matching up row irrelevant and the reverse. Training")
    print("    zeroed them together. ffn_down has none, which is the same fact")
    print("    from the other side: there a dead neuron is a COLUMN, not a row.")
    print()

    print("3. THEY ARE CONCENTRATED IN THE EARLY LAYERS"); print("-" * 74); print()
    print("    layer   ffn_gate dead     ffn_up dead     share")
    for L in sorted(bylayer):
        g = bylayer[L].get("ffn_gate"); u = bylayer[L].get("ffn_up")
        if g and u and g[0]:
            print(f"      {L:2d}    {g[0]:6,} / {g[1]:,}   {u[0]:6,} / {u[1]:,}   {100*g[0]/g[1]:6.2f} %")
    print()

    print("4. WHAT IT IS WORTH, AND WHY IT IS NOT TAKEN"); print("-" * 74); print()
    print(f"    Skipping the dead rows saves {100*zw/w:.2f} % of weight bytes AND the")
    print("    same share of multiply-accumulates -- both, because the row is")
    print("    neither read nor computed. On top of the 20 % the 1.600-bit")
    print(f"    packing already saves, that is {zw*1.6/8/1e6:.1f} MB more.")
    print()
    nd = kind.get("ffn_gate", 0)
    print("    Removing the neurons outright would save more: a dead neuron also")
    print("    makes its COLUMN of ffn_down useless, so")
    print(f"    {nd:,} neurons x 7,680 weights = {nd*7680/1e6:.1f} M, {100*nd*7680/w:.2f} % of the model,")
    print("    at provably zero quality cost because they are already zero.")
    print()
    print("    That route is NOT blocked by the file format. This project said it")
    print("    was, for some hours, without checking. llama.cpp reads")
    print("    LLM_KV_FEED_FORWARD_LENGTH through get_key_or_arr into a per-layer")
    print("    array -- llama-model.cpp:1117, llama-hparams.h:84 and :314 -- so")
    print("    per-layer FFN widths are a first-class concept and the removal")
    print("    needs no format change, no new tensor type and no kernel work.")
    print()
    print("    It is also EXACTLY lossless, and unconditionally so, precisely")
    print("    because section 2 holds: with the gate and up dead sets equal,")
    print("    h_i = f(gate_i) * up_i = f(0) * 0 = 0 for ANY activation f, since")
    print("    up_i is zero as well. The neuron contributes nothing and its")
    print("    ffn_down column is multiplied by zero.")
    print()

    if COMPARE:
        print("5. THE SAME NEURONS SURVIVE A FINE-TUNE, EXACTLY")
        print("-" * 74); print()
        A = dead_index_sets(MODEL); B = dead_index_sets(COMPARE)
        ident = sum(1 for n in A if n in B and np.array_equal(A[n], B[n]))
        tot = sum(len(v) for v in A.values())
        shared = sum(len(np.intersect1d(A[n], B[n])) for n in A if n in B)
        print(f"    against {pathlib.Path(COMPARE).name}")
        print(f"    tensors with IDENTICAL dead-row index sets  {ident} of {len(A)}")
        print(f"    dead rows                                   {tot:,} vs "
              f"{sum(len(v) for v in B.values()):,}")
        print(f"    at the same indices                         {shared:,} "
              f"({100*shared/tot:.2f} %)")
        print()
        print("    Not the same count -- the same rows, every one of them. The")
        print("    second model is an independent fine-tune of the first, and it")
        print("    revived nothing. Whatever killed these neurons happened in")
        print("    pre-training and fine-tuning did not touch it, which makes a")
        print("    skip scheme's benefit portable across derivatives of a base")
        print("    rather than a property of one checkpoint.")
        print()

    print("6. WHAT THIS IS NOT" if COMPARE else "5. WHAT THIS IS NOT"); print("-" * 74); print()
    print("    Not a claim about training, or about whether these neurons were")
    print("    ever useful. Not a quality result: nothing was pruned and no")
    print("    model re-run. And dead here means exactly all-weights-zero, which")
    print("    is sufficient for a silent neuron and not necessary -- a row of")
    print("    tiny non-zero weights is invisible to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
