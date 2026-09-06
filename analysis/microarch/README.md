# analysis/microarch — what happens on the parts this project cannot run on

`scripts/microarch_model.sh` produces `results/microarch_model.txt`. This
directory holds the one input to that script which is **not** derived from the
source: the two hand-written VNNI loops.

## Why they are hand-written

The AVX2 loops in the report are extracted from `gcc -O3 -mavx2 -mfma` output at
run time, so the analysis tracks the kernel source. The VNNI loops cannot be:
no compiler here targets VNNI usefully, and more importantly they represent a
*claim about what a VNNI implementation would look like*. A claim has to be
reviewable, so it is written down rather than generated.

**They have never been executed.** No machine in this project has
`avx512vnni` or `avx_vnni`. What follows is the argument that the substitution
is semantically sound; it is an argument, not a test.

## The substitution

Both kernels end their contraction with the same pair:

```
vpmaddubsw  mem, %ymmC, %ymmT     # T = 16-bit sums of 2 byte products
vpaddw      %ymmT, %ymmA, %ymmA   # A += T
```

`VPDPBUSD` performs exactly this fused, at a different reduction width:

```
vpdpbusd    mem, %ymmC, %ymmA     # A += 32-bit sums of 4 byte products
```

Four points make the replacement safe here, and each is a property of *these*
kernels rather than a general fact about the instruction:

1. **Operand signedness matches.** `VPDPBUSD` multiplies the first source as
   **u**nsigned and the second as **s**igned, which is the same convention as
   `VPMADDUBSW`. In both kernels the register operand holds weight codes or
   digits (`i2_s`: `c = w+1 ∈ {0,1,2}`; `t5b`: `d_k ∈ {0,1,2}`, both unsigned)
   and the memory operand holds `int8` activations. The AT&T operand order is
   therefore identical and the substitution is textual.

2. **The reduction width does not change the answer.** `VPMADDUBSW` sums pairs
   of byte products into `int16` lanes; `VPDPBUSD` sums quadruples into `int32`
   lanes. Both kernels finish with a horizontal sum over the whole accumulator,
   and addition is associative over the integers, so the scalar result is the
   same. Only the intermediate grouping differs.

3. **It removes an overflow constraint rather than adding one.** `VPMADDUBSW`
   saturates at the `int16` boundary. `ternary_t5b.c` argues that saturation
   cannot engage (two products of at most 2·128 sum to 512), and
   `ternary_t5b_dot_avx2` carries an accumulator fold every Φ = 12 blocks to
   respect `2560·Φ ≤ 32767`. Accumulating into `int32` makes both concerns
   vacuous: **the fold disappears entirely.** That is a saving the per-iteration
   instruction counts in the report do **not** credit, so the VNNI figures for
   `t5b` are, in that one respect, pessimistic.

4. **The accumulation tree collapses.** With `VPMADDUBSW` each contraction
   produces a temporary that must be summed. With `VPDPBUSD` accumulating in
   place, the tree is gone: `i2_s` loses 4 `vpaddw`, `t5b` loses 5 `vpaddw` and
   a `vmovdqa`.

## What is NOT modelled, and why the resource bound is used

`VPDPBUSD` accumulates **in place**, so each accumulator carries a loop-borne
dependency of the instruction's full latency — 10 cycles on `znver4`, 12 on
`icelake-server`, 13 on `sapphirerapids`. A loop with one accumulator per
contraction and no unrolling is bound by that latency and nothing else. On
`sapphirerapids` the effect is stark enough to be worth recording: the
dependency-bound figure for `loop_i2s_vnni.s` is **7.01 cycles against 5.01 for
the AVX2 version it replaces** — VNNI apparently making the kernel slower, which
is an artefact of these files carrying four accumulators, not a property of the
instruction.

A competent implementation unrolls until the recurrence stops binding. That
limit is exactly `llvm-mca`'s **Block RThroughput**, which is why the VNNI
comparison in the report is read off the resource bound and the AVX2 comparison
is read off total cycles.

This is not free: `scripts/microarch_model.sh` reports that the resource bound
**fails** to reproduce the measured `S` on `znver2` (1.96 against 2.43), because
the real Zen 2 loop *is* dependency-bound. The two metrics answer two different
questions and the script prints both rather than choosing silently.

## The control that does not work

`llvm-mca` 23.1.0 assembles and times `VPDPBUSD` for `-mcpu=znver2`, a part
without VNNI, and keeps doing so under
`-mattr=-avx512vnni,-avxvnni`, inventing latency 11 and throughput 1.00. So the
tool cannot distinguish a legal VNNI sequence from an illegal one, and offers no
independent check that these files would even assemble for the intended target.
The control is kept in the script and reported as failing.

## Why the two repositories print slightly different numbers

This analysis exists in both `bitnet-baremetal` (private) and `bitnet-t5b`
(public), and they report $S$ on `znver3`/`znver4`/`znver5` as **2.449** and
**2.441** respectively. That is not a discrepancy to reconcile; it is worth
writing down because it looks like one.

The baseline kernel differs between the trees. The private copy of
`ggml_i2s_ternary.c` wraps the upstream routine in instrumentation — call
counters, an environment-variable gate, an `atexit` report — because the
end-to-end work needed to prove the new path was actually taken. The public copy
is the verbatim extraction produced by `tools/fetch_i2s_reference.sh`, with none
of that.

The instrumentation is **not in the loop**; it is per call. Diffing the two
extracted loops confirms it: both are twenty instructions with the identical
opcode multiset — one `vmovdqu`, three `vpsrlw`, four `vpand`, four
`vpmaddubsw`, four `vpaddw`, one `vmovdqa`, and the scalar increment and
compare. What differs is register allocation and scheduling order, because gcc
sees a different translation unit around them, and `llvm-mca`'s scheduler
responds to the order. The effect is 7532 cycles against 7524 over 2000
iterations: **0.1%**, carried into $S$ as 0.3%. Both sit far inside the 8%
calibration gate, and neither changes a conclusion.

## Files

| file | what it is |
|---|---|
| `loop_i2s_vnni.s` | `bitnet_vec_dot_i2_i8_s_reference` inner loop, 4 × `vpmaddubsw`+`vpaddw` → 4 × `vpdpbusd`, 17 → 12 vector ops |
| `loop_t5b_vnni.s` | `ternary_t5b_dot_avx2` inner loop, 5 × `vpmaddubsw`+`vpaddw` → 5 × `vpdpbusd`, accumulation tree and `vmovdqa` removed, 42 → 36 vector ops |

Accumulators use `%ymm16`–`%ymm20`, which exist only under AVX-512VL. That is
satisfied on every target the VNNI table covers (`znver4`, `znver5`,
`icelake-server`, `sapphirerapids`) but **not** on VEX-only AVX-VNNI parts such
as Alder Lake, which have sixteen vector registers. On those the loops would
need reworking and the counts above would change.
