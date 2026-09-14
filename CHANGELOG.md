# Corrections made during preparation

Twenty-nine claims in earlier drafts of the paper were wrong and were corrected before
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
the in-situ counter measures 49.5% for the same arm. The replay is a lower bound
and was read as an estimate. Corrected in §7.5.

*Re-stated 2026-09-13: the correction originally read "43.6%", which is the
**t5b** arm's share against the **i2_s** arm's replay — two arms in one
sentence. Matched, the replay predicts t5b's matmul time to 1.2% and misses
i2_s's by 1.606×, so the error is the baseline's and the honest pair is 49.5%
against 30.8%.*

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

**10. The independent-magic-multiply decode was claimed as ours.**
It is not, and it is anticipated at exactly $A=3$, $k=5$, $p=8$. The NTRU
reference implementation has recovered all five trits of a packed byte since
2019 with four independent multiplies read straight from that byte — constants
171, 57, 19, 203 with shifts 9, 9, 9, 14 — at dependency depth one and without
a lookup table (`ref-common/pack3.c`, `poly_S3_frombytes`). We verified the
four identities ourselves over all 256 byte values before conceding the point.
What §1.1 now claims is only what follows the extraction: NTRU materialises
every coefficient into an array and defers the mod-3 reduction to a separate
sweep, where this kernel contracts on the digit planes so that no weight is
ever materialised.

Two further attributions were added at the same time and for the same reason.
The packing predates ternary language models by fifteen years — Breyer and Korf
store heuristic residues modulo three at 1.6 bits for pattern databases (AAAI
2010) — and was published a year before `TQ1_0` in statistical genomics as
*miraculix*'s `5codes`, whose stated motivation is the one taken here. And the
profitability condition is Equation 1 of Zukowski et al., ICDE 2006, rearranged;
what is new is that both of its terms are measured, one level down the memory
hierarchy from where that paper measured them.

*Found by a wave of agents sent to open every external source an earlier
adversarial pass had leaned on without reading. Twelve sources; none
fabricated; three required a claim here to be weakened. The same pass also
overturned one of its own predecessors' verdicts: the ISCA 2026 entropy-coding
paper does not close the entropy direction, because its format space bottoms
out at INT4 and the strings `ternary`, `BitNet` and `1.58` appear zero times in
its source.*

**11. "Contraction on the digit planes, so no weight is ever materialised" was
claimed as what distinguishes this kernel.**
It is not, and the counter-example is the kernel this paper positions itself
against, in the codebase it integrates into. `TQ1_0`'s NEON path
(`arch/arm/quants.c`) derives all four scalings 3, 9, 27, 81 of the loaded byte
with one `vmulq_u8` each — every one from the *original* vector, not from its
predecessor — extracts the digits table-free by multiplying by three and keeping
the two bits above eight, and passes them straight to `vdotq_s32`. Depth one,
table-free, no materialisation, shipped since PR #8151. Its AVX2 path is the
same without materialisation but at depth *two*, because AVX2 has no 8-bit
multiply and the scalings are chained as shift-and-add — except in its
four-element tail, which reaches depth one via `_mm256_mullo_epi16`.

So mechanism (3) is prior art on both instruction sets and mechanism (2) is
prior art in substance. What survives is one instantiation: the depth-one form
in the AVX2 *main* path through the high half of a 16-bit multiply, which
upstream itself names as an open question in the file — `// TODO: can
_mm256_mulhi_epu16 be faster even if 16-bits?`, `arch/x86/quants.c:1406`. §1.1
now claims that and nothing more.

*Found by a domain sweep that was looking for something else entirely: it
returned zero surviving application leads out of twenty-four, and its most
consequential output was this correction to our own prior-art inventory. Every
sentence above was verified against the vendored kernel at `390c307` here,
not taken from the agent that raised it. That is four repetitions of the same
failure — the packing, the recommendation, the extraction arithmetic, and now
the fused contraction — and the pattern is worth naming: each time, the thing
believed novel was already in a codebase this work had read part of.*

**12. The per-tensor scale model was claimed as ours, and it is in the pull
request we already cite.**
`block_q1_3` lived on ggml-org/llama.cpp#8151 from 2024-06-19 (`bd807499f7`) to
2024-07-30 (`77b8f84ae7`, which replaced it with `TQ1_0`). Read here from
`ggml-common.h` at `dd3e62a703`:

```c
#define QK1_3 64
typedef struct {
    uint8_t q[(QK1_3 - 4*QK1_3/64)/5]; // 5 elements per byte (3^5 = 243 < 256)
    uint8_t qs[QK1_3/64]; // 4 elements per byte
} block_q1_3;
```

No scale field. The scale was a per-layer `ggml` tensor of extent one applied
after the matmul — `Qcur = ggml_mul(ctx0, Qcur, model.layers[il].wq_scale)` in
`build_bitnet`, verified in `src/llama.cpp` at the same commit. That is exactly
what §1.1 called "the scale model".

The arithmetic was redone here rather than taken: of the 20,828,160 bytes t5b
saves against `TQ1_0`, `Q1_3` already delivered 78.2 %. The residual is
4,546,560 bytes — 1.03 % of the `TQ1_0` ternary payload, 0.38 % of the file —
and it is a block-length choice, since 64 = 12·5+4 needs a remainder byte
(1.6250 bpw) and 160 = 32·5 does not (1.6000). What is genuinely unanticipated
is only that the scale rides in `i2_s`'s existing 32-byte tail, so no graph node
and no loader change is needed where `Q1_3` required both.

*Found by an adversary sent specifically to look for a fifth anticipation after
four had fallen in a day. It found one. Every element was re-verified here
against the GitHub API — the commit list of #8151, the struct at `dd3e62a703`,
and the `ggml_mul` call site — before the concession was written.*

**13. "§4 answers it with a measurement", and "the depth-one extraction costs
more multiply-port operations than the serial form."**
Neither held. The first was a pointer to nothing: §4 contains no
mulhi-versus-shift-and-add comparison, and the string `0.983` appeared zero
times in the paper while the measurement sat in `pr/llamacpp-tq1_0-mulhi/`,
uncited. The second is contradicted by the kernel's own header eleven hundred
lines away — "Same instruction count as a divide-by-three chain, one eighth the
dependency depth" — and it is the header that is right: a serial chain
`q_{j+1} = mulhi(q_j, 21846)` over the two 16-bit views issues two multiplies at
each of four levels, the same eight the independent form issues. The sentence
was true only against a *fractional* encoding's serial form, which uses no
multiply-port instruction at all, so it compared this format's decode against
another format's alternative.

§1.1 now reports the measurement, names the arm it was taken on, and explains
why its negative result does not transfer: `TQ1_0` stores ceil(256v/243) and can
therefore extract digits in byte lanes with no multiply at all, while t5b stores
the integer and has no byte-lane route to lose to. It also states the weaker
consequence plainly — depth one costs nothing here, and relieves a constraint
that §7.2 measures as not binding, the loop running at 67 % of its multiply-port
ceiling.

*Found by an agent sent to check whether the one surviving novelty claim pointed
at a measurement that contradicts it. It found something better: the two
measurements do not conflict, they are about different objects, and the real
defects were a dangling forward reference and a cost claim the source file had
already refuted.*

**14. The interleave was called "i2_s's own layout extended from four digits to
five".**
The extension was `TQ1_0`'s, in the same file the paper quotes its open TODO
from: `quantize_row_tq1_0_ref` packs `x[m + n*32]` under the comment *"5
elements per byte, along 32 bytes"*, and the AVX2 dot contracts five digit
planes of one 32-byte load against activations at +0, +32, +64, +96, +128 — the
same stride, the same instruction set, five digits. §3.1 now says the interleave
is adopted at both widths rather than extended at one, and states what does
differ, because it is smaller than it looks and load-bearing: the digit order is
reversed, and `TQ1_0` stores ceil(256v/243) where this format stores the base-3
value v, which is exactly why the exactness argument of §4 applies here and not
there.

**15. The six-thread traffic row was quoted without noticing it exceeds the
model's own ceiling.**
(4) caps the gain at min(F, Σ/S) = 1.248 at six threads; the row is 1.355, which
is 8.6 % above it — and 8.9 % above the tighter ceiling the achieved byte ratio
gives, so substituting the honest F makes it worse rather than better. The
excess cannot be a density effect at all: reproducing 1.355× from bytes would
take 1.476 bits per weight, below this checkpoint's own measured entropy of
1.5603, so no lossless re-encoding reaches it.

The cause is that the two figures are not the same quantity — one is a quotient
of roofline times, the other of two measurements — and at six threads i2_s's
draw stands 13.0 % above its own transport floor against 3.7 % for its t5b
counterpart, the widest divergence in the eight paired observations. On medians
the model gives 1.097 against a measured 1.120, and best-against-best 1.110.
§7.3 now says the condition predicts the crossing, not the size of the win.

*Both found by agents sent to check unresolved findings from a cold read, and in
both cases the agent corrected the reviewer as well: the proposed cause of the
overshoot — that i2_s regresses from four threads to six — does not survive the
eight observations, which show it flat, exactly as the model predicts.*

**16. The headline throughput ratio was one invocation of five.**
The table printed 1.132 and 1.143; `results/inference_t5b.txt:215` shows those
are invocation 5, and line 217 gives the median over the five as 1.119 and
1.133. The per-invocation ratios were listed underneath all along, so nothing
was hidden — but quoting the fifth pair in the table, the abstract and the
conclusion, with a spread running to 1.253, is a choice rather than a
measurement. Both papers now quote the median and say which figure is which.

**17. "Σ values ranging from 1.35 to 3.08."**
No Σ measurement supports 3.08. Every Σ the paper prints runs 1.35, 1.63, 1.78,
1.86, 3.03, and a grep over both trees finds 3.08 in no results file as a
surplus. Corrected to 3.03.

*Both found by a sweep over the forty minor findings of five cold reads —
26 held, 8 were already fixed, 4 did not. These two are the ones a referee
recomputes.*

**18. Two different accumulator bounds for the same baseline kernel, and the
public bug report inherited one of them.**
§4.2 gave the reference kernel's worst case as 128 × 508 = 65,024, taking the
code range as 0..2; §7.5 derived 3048 per block from the range 0..3 and called
the kernel's 32-block fold "three times its own bound", which only works on the
second basis. The kernel settles it: `_mm256_set1_epi8(0x03)` at `quants.c:1400`
masks to 0..3, so its own bound is 128 × 762 = 97,536 and 65,024 is what the
encoder can actually produce. Both are now given and labelled, as are the two
$|a|$ conventions — 128 for t5b, covering $a=-128$, and 127 for the reference
kernel, which is what its own comments use.

This one had already left the building: `microsoft/BitNet#629`, filed an hour
earlier, quoted the 508 figure while `#628` derived 762 from the mask. Two
issues from the same reporter quoting two operand ranges for one representation
is not a detail. Corrected in a comment on #629 rather than by editing the issue
body, so the record shows what was filed.

**19. "18.01 predicted against 19.7 measured per 160-weight block."**
No file records 19.7. The paper's own constants — 33.74 GMAC/s at 3.91 GHz —
give 160 × 3.91 / 33.74 = 18.54 cycles, against which the model is optimistic by
2.9 %, not the stated 6–11 %. The 19.7 would need a 4.15 GHz clock. The derived
figure and the discrepancy are now both reported rather than one being chosen,
and §7.7's dependent claim is re-derived: 2.45 vector operations per cycle
against 2.32, both above the two per cycle this part should sustain, which is
itself a reason to treat the absolutes as indicative.

**20. Three things the evidence files disclosed and the paper did not.**
The adverse round of the cheaper-decode measurement read 0.6086 — an excursion
eight times the control spread asserted in the same sentence — and was caused by
a second benchmark on the same pinned core; `results/shufdig_measurement.txt`
says so in full and the paper said nothing. The concurrency series was read as
"the advantage grows" from four unreplicated points, one of which
`results/inference_t5b.txt` records as an outlier. And the cheaper decode's build
option was never stated to be off by default, while every figure elsewhere
assumes it is. All three now say in the paper what the file behind them already
said.

**21. The token budget was presented as a cross-check, and it is an identity.**
The paragraph written to fix correction #4 claimed that the decomposition
"predicts 1.117 end to end" and that "the same run's wall clock gives 1.117 —
agreement to 0.02 %". Only the packed arm carries a counter;
`results/insitu_kernel_rate.txt` says so in its own header — i2_s "has no
counter and is obtained by difference". With the non-matmul time defined from
one arm and the baseline's matmul time defined as its wall clock minus that, the
prediction collapses to

    T_A / (T_A − (M_A − M_B))  =  T_A / T_B

which is the measured ratio, for any inputs. Computed here the two agree to
0.00e+00, not 0.02 % — the 0.02 % was rounding. It could not have disagreed.

This is the failure this project has a name for: a check that cannot fail. It
was introduced today, in the commit that fixed a different defect in the same
paragraph, and it survived a rebuild and seven green gates because no gate
reads algebra.

Both papers now call the table a budget, keep it for what it does show — the
untouched 21.6 ms is the larger half of a token, which bounds what a format
acting on the smaller half can do — and report the matmul ratio from the
measurement that is real: nine interleaved rounds, both arms probed at one
dispatch site, median 1.257, baseline slower in 9 of 9.

*Found by a referee sent to read the four-page submission cold. It also caught
that the same paragraph's "independent instrument" was being compared against
the derived number rather than the measured one.*

**22. "$S = 2.43$ lies inside that interval" — it does not.**
The short paper gave Sigma at six threads as 1.65 to 2.38 under load and then
said S = 2.43 lies inside it. 2.43 is *above* 2.38. The true statement is
sharper and worse for the paper: S sits above every loaded measurement and below
the idle 3.03, so at six threads the condition holds **only while the host is
quiet**. Both papers now say that; the long one's wording ("the interval it
spans", which included the idle value) was defensible but ambiguous and is now
explicit.

*Found by a fact-checker sent to take every claim in the four-page submission
back to its evidence. It confirmed the rest of the foundation at primary
sources: all five concession citations, the magic multipliers over all 256
bytes, $\Phi=12$, the 2816 foreign-byte bound, the no-borrow property, the
4.7 %/1.9 % arithmetic and the 78.2 % split — and, in the vendored kernel, the
claim I trusted least: `TQ1_0` really has no GEMM equivalent (`assert(nrc==1)`
in both arch paths, no `.gemm` trait, no sgemm case, absent from `repack.cpp`).*

**23. Five things the cut re-introduced or overstated.**
Cutting 27 pages to four dropped the qualifications with the prose. The
submission claimed the crossing is "observed between four and six threads" while
reporting a four-thread win two pages later — the resolution, that the
dispatched baseline is itself 1.6x slower than the reference the ratio was taken
against, had been cut. It credited 4,546,560 bytes jointly to the block length
and to the scale-in-tail, though the latter saves no bytes at all. It said the
depth-one decode is one "upstream has only on NEON", which upstream's own
four-element AVX2 tail falsifies. It never said the cheaper decode is off by
default. And its availability section promised that "each figure above is
recomputed by a named script" while naming none. All five corrected.

**24. "No file records that 19.7" — two do, and one of them ships.**
The microarchitecture section calibrated its static model by observing that the
measured 19.7 cycles per block appears nowhere, so the figure implied by the
paper's own two constants, 18.54, should be used instead. The 18.54 is right and
the negative existence claim was false: `src/ternary_t5b.c` states 19.7 twice in
the header comment a reader of the public tree opens first, and
`results/shufdig_measurement.txt` backs an issue rate out of it. Corrected to
say 19.7 is recorded but not derived, and the two issue rates that follow from
the two cycle counts — 2.18 against 2.32, 2.30 against 2.45 — are now named
together instead of appearing in different sections as one quantity with two
values.
*Found by reading the evidence files against each other rather than against the
paper. No check compared a shipped comment to a claim, and none does now either.*

**25. Four figures the paper's own text contradicted.**
The abstract called the arithmetic ratio "stable across Zen 3/4/5, Ice Lake and
Sapphire Rapids" where §8 declines exactly that — the three Zen targets return
identical counts, so the table holds four predictions and not six, and what is
supported is one-sided. A subsection heading and the conclusion quoted 14 %,
which is the fifth of five `llama-bench` invocations, two pages after the paper
says in its own words that quoting a single one "is a choice rather than a
measurement"; the medians are 11.9 % and 13.3 %. The availability section
promised "all four benchmarks" build from a clone where the repository ships
eight and three of them need a fetched reference. And a mutation paragraph said
"six aimed at the GEMM path" then "one of the six", where the file lists seven
and the sentence that follows discusses a decode mutant. All four corrected.

**26. The four-thread verdict was stated three incompatible ways.**
§5 said the condition is satisfied only at six threads and the code "should lose
at four"; §7 showed it holds at four in situ; §7.6 said it "failed at one to
four threads and still fails"; the conclusion said it "holds from four cores
upward". Only the conclusion had absorbed the resolution — that the $S=2.43$ of
the replay is measured against the *reference* kernel while the path
`llama.cpp` dispatches is about 1.6x slower, which is what changes the sign of
that one cell. Every site now names which $S$ it argues from. No number moved;
the paper had been saying two true things in a way that read as one false one.
*Found by reading the whole document in sequence after thirty edits that had each
been checked alone.*

**27. "In a SwiGLU block" — this checkpoint is not one, as the paper says three
pages later.**
The dead-neuron section explained the gate/up index coincidence through a SwiGLU
block while §9 quotes the technical report saying the model uses squared ReLU
"instead of the commonly used SwiGLU activation". No number moves — the evidence
file already proves the result for any activation $f$ — but the paper
contradicted itself, and the same wrong label was in
`results/dead_neurons.txt`. Both corrected to a gated block with $f$ named.

**28. The third digit of the microarchitecture table is not a property of the
target.**
Not a wrong claim but a missing limitation, and it surfaced because the two
copies of `results/microarch_model.txt` disagreed: the public tree printed 3.77
and $S=2.441$ for the three Zen targets where this one printed 3.76 and 2.449.
Both reproduce today on the same `llvm-mca`. The cause is the translation unit,
not the target — the two `i2_s` inner loops are twenty instructions each with an
identical opcode multiset and identical memory offsets, differing only in
register allocation and order, which the model prices at 0.01 cycles on
znver3/4/5 and at exactly zero on the other three. Those three zeros are its
control. The effect is 0.33 % on $S$, two orders below the 2.10–2.50 spread the
table is cited for, and it is now stated beside the table.
`results/microarch_translation_unit.txt` isolates it.

**29. A gate that could not fail, shipped for a day.**
`tools/check_short_parity.py` returned 0 with the message "not present --
skipping" when the short paper was absent — and the short paper was not in the
public tree at all, so in the repository a reader clones, the gate had never
once run. It now returns 1, `scripts/sync_public.sh` ships the pair together,
and the page counts both documents quoted are read from the build logs rather
than asserted, because that number had already drifted twice in one day.
*This is the failure this list's own preamble describes. It was introduced by the
commit that added the short paper and found by auditing a cold clone.*

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
| `tools/check_short_parity.py` | the six-page submission carries no figure the long paper lacks |
| `mutation_test.sh` | twenty-one injected kernel defects, eighteen killed, three proved equivalent |

The count is twenty-three. It was called seven until this list was written out — items 2
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
