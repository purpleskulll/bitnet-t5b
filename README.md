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

## Quick start

From a fresh clone, with a compiler and Python 3:

```bash
make test          # builds and runs the suite: 177,734 checks, 0 failures
make bench         # builds the benchmarks; says LOUDLY what it skips and why
```

`make test` is the gate and exits non-zero on any failure. Nothing above needs a
model, a network, or a `llama.cpp` checkout.

To build and run one test binary by hand, which is what the paper's §4 refers to:

```bash
gcc -O3 -mavx2 -mfma -march=native -std=c11 \
    src/test_t5b.c src/ternary_t5b.c -o test_t5b && ./test_t5b
```

Three of the four benchmarks compare against upstream's own `i2_s` kernel, which
this repository does **not** carry — it is MIT-licensed upstream code and is
extracted from a pinned commit rather than redistributed here:

```bash
tools/fetch_i2s_reference.sh                    # clone the pinned commit
tools/fetch_i2s_reference.sh --from <checkout>  # or use one you already have
make bench                                      # now builds all four
```

The extraction is digest-checked against the exact revision every measurement in
the paper was made on, and **refuses** rather than silently emitting different
arithmetic under the same name.

## Layout

| | |
|---|---|
| `paper/` | the paper: Markdown, LaTeX source, compiled PDF |
| `src/` | the format, its kernels, the ggml-facing glue, the test suite |
| `benchmarks/` | port throughput, thread scaling, one token's weight traffic, per-kernel rate, the VNNI comparison, the instrumentation's own cost, and the upstream tiled shape used as a control |
| `tools/` | GGUF converters and analysis: to t5b, architecture retag, depth synthesis, dead-neuron census, tensor structure, the `i2_s` reference fetcher |
| `integration/` | the `llama.cpp` change, as a unified diff **and** as the script that applies it |
| `results/` | every evidence file the paper cites, each stating its own reproduction command |
| `Makefile` | builds and runs all of the above |
| `Dockerfile.t5b` | the incremental image; needs a base image you build yourself |

`build/` is produced by `make` and is not checked in. `src/ggml_i2s_ternary.{c,h}`
are **generated** by `tools/fetch_i2s_reference.sh` and are likewise not checked
in — see the licence note below.

## What is reproducible from this repository alone

Everything in this list needs only a clone, a C compiler with AVX2, and Python 3.

| claim | how |
|---|---|
| the packing is lossless and the AVX2 kernel matches the scalar one | `make test` — 97,529 checks for t5b, 80,205 for t10, 0 failures |
| 1.600 bits/weight, and what padding costs at real tensor widths | printed by `make test` |
| the accumulator provably wraps and the answer is still right | part of `make test` |
| which execution port binds | `make bench && taskset -c 5 ./build/bench_ports` |
| the ggml-facing glue still matches the kernel signatures | `make` builds `build/ggml_t5b_glue.o` |
| the `llama.cpp` change, in full, reviewable | `integration/t5b-integration.patch` |
| the GGUF container reader | `python3 tools/gguf_io.py` — self-checks with no model |
| the tensor-structure estimator's own statistics | `python3 tools/tensor_structure.py --self-test` |

With `tools/fetch_i2s_reference.sh` run first (needs the pinned upstream tree,
by clone or by path — no model, no build of `llama.cpp`):

| claim | how |
|---|---|
| t5b against upstream `i2_s`, one token's weight traffic | `./build/bench_token 6 3` |
| arithmetic surplus against core count | `./build/bench_threads 0.5` |
| per-kernel rate, instruction and spill counts | `./build/bench_alu` |
| the VNNI kernels compute the right answer | `./build/bench_vnni_emu` — runs anywhere, no VNNI needed |
| what the in-situ cycle counter costs | `./build/bench_probe_cost` — and why the counter is per-thread |
| VNNI against AVX2, timed | `./build/bench_vnni` — needs `avx512vnni` or `avx_vnni`; exits 3 otherwise |

## What is **not** reproducible from this repository alone, and why

Being direct about this is the point of the section. The paper says every figure
is recomputed by a script in this repository; for the figures below, the script
is here and its **inputs** are not.

1. **Anything with a number of tokens per second, or a perplexity, in it.**
   That is the whole of the results table above, `results/inference_t5b.txt` and
   `results/perplexity_t5b.txt`. They need three things this repository does not
   and will not contain:
   - **the checkpoint.** `microsoft/bitnet-b1.58-2B-4T-gguf`, ~1.1 GiB, under its
     own licence. Download it yourself; no model weights are distributed here.
   - **a patched, built `llama.cpp`.** `integration/` has the complete change and
     names the commit. Building it is on you: `Dockerfile.t5b` is *incremental*
     and starts from a base image with an already-configured BitNet tree that is
     not published anywhere.
   - **the measuring host.** AMD Ryzen 5 3600, six cores, AVX2 without VNNI,
     under real background load. Your numbers will differ. The paper's honest
     limits below say why that matters more than usual here.

2. **The tensor-level censuses.** `results/tensor_structure.txt`,
   `results/weight_density.txt`, `results/real_weights_check.txt` and
   `results/dead_neurons.txt` are recomputed by `tools/tensor_structure.py`,
   `tools/gguf_to_t5b.py --verify` and `tools/dead_neurons.py` — all present,
   all runnable — but each reads the checkpoint. Get the model and they run.

3. **`src/ggml_i2s_ternary.{c,h}`.** Generated, not stored. See below.

4. **`results/phase_status.txt`**, which the paper's evidence table names, is not
   in this repository. It belongs to a different phase of a larger private
   project and is not part of this work.

5. **The absolute perplexity figure of 96.82** is high for a 2 B model and is
   reported as an identity check between the two formats, not as a quality
   claim. It is the same number for both arms; that identity is the finding.

## Where the paper's evidence table points, and where the file actually is

The paper was written against a private tree with a different layout. The paths
in its evidence table map onto this repository like this:

| the paper says | in this repository |
|---|---|
| `src_modifications/bench/bench_ports.c` | `benchmarks/bench_ports.c` |
| `src_modifications/bench/bench_token.c` | `benchmarks/bench_token.c` |
| `src_modifications/ternary_t5b.{c,h}` | `src/ternary_t5b.{c,h}` |
| `src_modifications/tests/` | `src/test_t5b.c`, `src/test_t10.c` |
| `src_modifications/ggml_i2s_ternary.c` | generated by `tools/fetch_i2s_reference.sh` |
| `scripts/apply_integration.py` | `integration/apply_integration.py` |
| `Dockerfile.t5b` | `Dockerfile.t5b` (incremental — read its header first) |
| `results/phase_status.txt` | not in this repository; see above |

Every other path the paper names — `results/*.txt`, `tools/*.py` — is where it
says it is.

## Honest limits

1. **No VNNI on the measurement host.** `VPDPBUSD` makes the contraction cheaper
   and leaves the decode unchanged, moving the profitability condition against a
   packed format. An independent system built for VNNI (*Litespark*) stores
   ternary weights at eight bits for that reason.

   Two scripts address this, and neither is a substitute for the other:

   - `./microarch_model.sh` needs no second machine. It runs `llvm-mca` over the
     inner loops `gcc` actually emits and reports `S` for `znver2/3/4/5`,
     `icelake-server` and `sapphirerapids`. It **calibrates first** — its Zen 2
     prediction is 2.50 against the 2.428 measured here, and it exits non-zero
     if that ever drifts past 8% — and it finds `S` between 2.10 and 2.50, so
     the ratio is not an artefact of one narrow part. It also bounds VNNI's cost
     at 0–14% while reporting, in its own output, the three reasons that half is
     *not* a result: hand-written loops that have never been executed, a metric
     that failed calibration, and a negative control that did not fire.
     Requires `llvm-mca`; the header says how to fetch it without root.
   - `./second_datapoint.sh user@host` runs everything on a second machine,
     including **step 5, which is the VNNI measurement itself**. That step is
     new and it exists because the earlier claim was wrong: this script used to
     say a run on a VNNI part would settle the question, and a reviewer showed
     with `objdump` that its binaries contained zero `vpdpbusd`. No compiler
     turns `vpmaddubsw`+`vpaddw` into that instruction. `benchmarks/bench_vnni.c`
     writes it, verifies at run time that its own binary contains it, checks
     every row against exact arithmetic before printing a rate, and exits 3
     rather than run on a CPU without the feature.
     **If you have a machine reporting `avx512vnni` or `avx_vnni`, this is the
     single most useful thing you can contribute to this work.** No model, no
     Docker, no privileges.

2. **One model size.** No public ternary checkpoint above 2 B exists.
3. **A shared, noisy host.** Run-to-run spread is comparable to the effect,
   which is why the headline is a five-invocation replication.
4. **The type identifier (43) is not reserved upstream**, so a stock
   `llama.cpp` cannot read a t5b file.

## Licence

Apache License 2.0 — see `LICENSE`. Attribution for the MIT-licensed `llama.cpp`
and `ggml` material this work builds on is in `NOTICE`. No upstream source file
is redistributed here: the one piece of upstream code the benchmarks need is
fetched from a named commit by `tools/fetch_i2s_reference.sh`, on your machine,
from your checkout. No model weights are distributed.
