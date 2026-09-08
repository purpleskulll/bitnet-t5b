# Corrections made during preparation

Nine claims in earlier drafts of the paper were wrong and were corrected before
submission. They live here rather than in the paper so that the paper states
what holds and this states how it got there — the change a reviewer asked for
after counting seven "an earlier draft claimed" passages scattered through
twenty pages.

Each entry names what was claimed, what replaced it, and how it was found.
Where the finding is now automated, the check is named; those run as steps of
`scripts/run_all.sh` and are shipped in the public repository.

---

**1. The base-3 packing was claimed as novel.**
It is `TQ1_0`'s construction in `llama.cpp`, and it is published with a
correctness theorem — $2^p > 3^k$, with $p=8$, $k=5$ recommended — as
Spectra 1.1's `TQ1` (arXiv:2506.23025). Withdrawn in §1.1; §7.7 measures against
it and §3.2 states the relationship to their Theorem 1.
*Found by a reviewer, twice: first the pull request, then the paper.*

**2. "The removal needs no format change and no kernel work."**
False. `src/models/bitnet.cpp` allocates its four feed-forward tensors with the
global `n_ff`, so a file with per-layer widths fails at load; four lines change
that. Corrected in §8.

**3. "Unconditionally exact on the product" (dead-neuron removal).**
False. The sub-layer RMS norm divides by the root mean over all 6,912 entries,
so deleting $d$ of them rescales the output by $\sqrt{(K-d)/K}$ — 0.751 in
layer 1, not a rounding matter. Pinning the divisor restores it to 2.5 ulp of
float32, not to zero. Corrected in §8.

**4. Matmuls were given as 30% of a token.**
That figure came from the standalone replay, which drives the reference kernel;
the in-situ counter of §7.4 measures 43.6%. The replay is a lower bound and was
read as an estimate. Corrected in §7.5.

**5. Upstream's `int16` overflow was reported as "200 of 200" rows.**
Over all thirty tensors at $K = 6912$ it is 11,998 of 12,000: with 0.78% of
headroom the outcome turns on each row's own code mean, so the statement is
statistical, not absolute.
*Found by extending the dot-product check from seven tensors to all 210.*

**6. "Four runs out of four."**
Those four were taken in the quietest window available; four further runs at
ordinary load include one the denser format loses. The headline became the
five-invocation replication of §7.4.

**7. The `i2_s` matmul time was derived by difference.**
Across two `llama-bench` runs on a host at load 2 to 10, which puts every
fluctuation between them into that one number. Both arms are now probed at the
same dispatch site. The stated reason for not instrumenting the control arm —
that it must stay unperturbed — did not survive measurement:
`benchmarks/bench_probe_cost.c` puts the probe at 0.18% of a matmul call.
*Since run: nine interleaved rounds give a median ratio of 1.257 against the
1.268 the difference method derived, and the two matmul times agree to 2.3% and
1.1%. The objection was right and the number it doubted survived it.*

**8. `second_datapoint.sh` claimed a run on VNNI hardware would settle §9.1.**
It would not have. No compiler emits `VPDPBUSD` from the `vpmaddubsw`+`vpaddw`
idiom, so its binaries contained none and it measured AVX2 on whatever machine
it was given. `benchmarks/bench_vnni.c` writes the instruction and verifies at
run time that its own binary contains it; §9.2 is the resulting measurement.
*Found by a reviewer with `objdump`.*

**9. "Every digit is in 0..2 for every one of the 256 possible byte values —
verified exhaustively."**
False, and the same file's header said so correctly eleven lines apart: the
thirteen bytes $243\ldots255$ all decode to $d_4=3$, because
$\lfloor x/81\rfloor = 3$ there. The claim mattered because the digit range is
what the accumulator bound rests on — $2\cdot128\cdot2=512$ per plane — so
$d\le2$ is a consequence of the $x\le242$ precondition and not something the
decode enforces. A block of foreign bytes bounds a lane at
$2\cdot128\cdot11=2816$ and twelve of them overflow `int16`. The property
`vpsubb` actually needs is weaker and *is* unconditional: no digit subtraction
borrows, for any of the 256 values. Corrected in `src/ternary_t5b.{c,h}` and
§4.3; the header's foreign-byte figure of 2794 was recomputed as 2816, having
been taken at $|a|\le127$ while the derivation eighteen lines above it takes
128.
*Found by re-deriving the decode independently while checking an unrelated
proposal — not by any check in this repository, and no check here would have
caught it: the kernel is correct, and it was the justification that was wrong.*

---

Three of these were found by a reviewer and the rest by checks written
afterwards. Those checks are in the repository and run as gate steps:

| check | holds |
|---|---|
| `tools/tex_to_md.py --check` | `paper/arxiv.md` is generated from the `.tex`, not maintained beside it |
| `tools/check_paper_parity.py` | both renderings carry the same numeric claims |
| `tools/check_paper_paths.py` | every path the paper names resolves in the repository a reader clones |
| `tools/check_patch_chain.py` | later integration hunks anchor on what earlier ones emit |
| `tools/check_patch_parity.py` | the shipped patch describes the same change as the script |
| `tools/fix_patch_hunks.py --check` | the patch's `@@` counts match its content |
| `tools/check_buildable.py` | every shipped source is named by a build rule |
| `tools/check_decode_claims.py` | the decode's stated arithmetic, re-derived over all 256 bytes |
| `mutation_test.sh` | twenty-one injected kernel defects, eighteen killed, three proved equivalent |

The count is nine. It was called seven until this list was written out — items 2
and 3 are two distinct false statements about the same paragraph, made at
different times, and treating them as one was itself a small piece of
under-reporting — and nine only after item 9, which none of the checks above
could have found at the time. Every check that existed then tested a claim
against another artefact: the `.md` against the `.tex`, a path against the tree,
a hunk against its anchor. Item 9 was a claim about arithmetic that no artefact
in the repository disagreed with — the file's own header contradicted it, eleven
lines away, and no comparison in this list reads comments — so it took
re-deriving the arithmetic to see it. `check_decode_claims.py` is that
re-derivation, added with item 9 and listed above; it is the only entry in the
table that evaluates a property rather than comparing two texts.

Its first negative control did not fire, which is worth recording because the
reason is not carelessness. The control perturbed the $/81$ multiplier 811 to
810 on the belief that the mutation suite had proved those equivalent only on
$x\le242$; sweeping the constants shows 810, 811 and 812 are all exact over the
whole 256-byte domain, so the mutant is equivalent everywhere and no sweep of
the domain can catch it. The controls that replaced it were chosen by sweeping
rather than by reasoning about them, and one of them independently reproduces
the paper's own claim that 813 first fails at $x=242$.
