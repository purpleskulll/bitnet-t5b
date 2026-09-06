# bitnet-t5b — ternary weights at 1.600 bits, decoded arithmetically

Supporting code and evidence for *Sub-Two-Bit Ternary Weight Storage with
Table-Free Arithmetic Decoding for CPU Inference* (`paper/`).

**The mechanism in one sentence.** A lossless radix-3 re-encoding of weights that
are already ternary — five values as the base-3 digits of one byte, 1.600 bits
instead of `i2_s`'s 2.000 — together with an AVX2 kernel that recovers all four
quotients from the packed byte with one independent `vpmulhuw` each and contracts
on the digits with `vpmaddubsw`, so no weight is ever materialised. Not pruning,
not weight sharing, not quantisation: the weights are unchanged bit for bit.

## What is and is not novel here

The base-3 five-per-byte packing is **not** new. `llama.cpp` has carried `TQ1_0`
since August 2024 (PR #8151), annotated in `ggml-common.h` as *"5 elements per
byte (3^5 = 243 < 256)"*, table-free, with an AVX2 dot product. `TQ1_0` lands at
1.6875 bits only because each 256-element block carries an f16 scale and a
16-element two-bit remainder. An earlier draft of the paper claimed the packing
as a contribution; a reviewer corrected it, and the claim is now confined to
what is actually different:

- a **per-tensor scale** in place of the per-block f16 and the remainder, which
  removes the block overhead (1.600 against 1.6875) and is drop-in for the
  `i2_s` scale model of `bitnet.cpp`;
- an **interleave identical to `i2_s`'s**, so a digit plane still addresses 32
  consecutive activations and the access pattern is unchanged;
- **four independent magic multiplies** as the decode rather than a chain,
  which is possible because a five-trit byte is bounded by 242 and therefore sits
  deep inside the exactness range of the multipliers;
- a **column-blocked** matrix–matrix kernel;
- an explicit, **measurable** condition for when any denser format pays;
- an end-to-end evaluation against the format shipped with the checkpoint.

## Result

| | `i2_s` (baseline) | t5b | |
|---|---:|---:|---|
| bits/weight | 2.000 | **1.600** | −20% |
| model file | 1.10 GiB | **1.00 GiB** | −102.2 MB |
| `pp512` @ 4 threads | 119.65 ± 2.27 t/s | **135.45 ± 1.21** | **1.132×** |
| `tg128` @ 4 threads | 22.56 ± 1.26 t/s | **25.78 ± 0.87** | **1.143×** |
| WikiText-2 perplexity | 96.8243 ± 3.55673 | 96.8243 ± 3.55673 | identical |

Faster in 5 of 5 `llama-bench` invocations on both measures, load order
alternated. **Measured on one AVX2 part without VNNI** — see the limitations,
which lead the paper's §9 rather than closing it.

## Layout

| | |
|---|---|
| `paper/` | the paper: Markdown, LaTeX source, compiled PDF |
| `src/` | the format, its three kernels, the ggml-facing glue, the test suite |
| `benchmarks/` | port throughput, thread scaling, one token's weight traffic, per-kernel rate, and the upstream tiled shape used as a control |
| `tools/` | GGUF converters and analysis: to t5b, architecture retag, depth synthesis, dead-neuron census |
| `results/` | every evidence file the paper cites, each stating its own reproduction command |

## Reproducing

The kernels and their tests are self-contained:

```bash
gcc -O3 -mavx2 -mfma -march=native -std=c11 src/test_t5b.c src/ternary_t5b.c -o test_t5b && ./test_t5b
```

The benchmarks need only a compiler; `benchmarks/bench_ports.c` establishes which
execution port binds, and `benchmarks/bench_threads.c` the memory-versus-
arithmetic ratio that the paper's profitability condition turns on.

Model-level results need a `llama.cpp` build carrying the type, a converted
GGUF, and the model itself; the integration is applied as a patch to an
unmodified checkout and is **not** included here.

## Honest limits

1. **No VNNI on the measurement host.** `VPDPBUSD` makes the contraction cheaper
   and leaves the decode unchanged, moving the profitability condition against a
   packed format. An independent system built for VNNI (*Litespark*) stores
   ternary weights at eight bits for that reason.
2. **One model size.** No public ternary checkpoint above 2 B exists.
3. **A shared, noisy host.** Run-to-run spread is comparable to the effect,
   which is why the headline is a five-invocation replication.
4. **The type identifier (43) is not reserved upstream**, so a stock
   `llama.cpp` cannot read a t5b file.

## Licence

Apache License 2.0 — see `LICENSE`. Attribution for the MIT-licensed `llama.cpp`
and `ggml` material this work builds on is in `NOTICE`. No model weights are
distributed.
