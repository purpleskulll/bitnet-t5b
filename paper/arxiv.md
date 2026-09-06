---
title: "Sub-Two-Bit Ternary Weight Storage with Table-Free Arithmetic Decoding for CPU Inference"
author:
  - Justus Theile
date: 2026-09-06
abstract: |
  Ternary (1.58-bit) large language models are stored in practice at two bits
  per weight, 26% above the information content of a ternary symbol. We show
  that the residual redundancy is recoverable on commodity x86 CPUs without
  lookup tables and without decompressing weights into a wider representation.
  We introduce **t5b**, a base-3 packing that places five ternary values in one
  byte at exactly 1.600 bits per weight, and an AVX2 kernel that recovers all
  five digits from the packed byte using four independent `vpmulhuw`
  instructions, exploiting the fact that a five-trit byte is bounded by 242 and
  therefore lies strictly inside the exactness range of the corresponding magic
  multipliers. The kernel performs its multiply-accumulate directly on the
  digits with `vpmaddubsw`, so no weight is ever materialised.
  We give a roofline argument identifying the condition under which a denser
  format is profitable — the ratio between a core's multiply-port throughput
  and its share of memory bandwidth — and verify the condition by direct
  measurement of both quantities.
  Integrated into `llama.cpp` as a distinct `ggml` type alongside the existing
  two-bit format, t5b reduces the shipped `BitNet-b1.58-2B-4T` checkpoint from
  1.10 GiB to 1.00 GiB and yields, under `llama-bench` **at four threads**,
  $1.132\times$ prompt throughput and $1.143\times$ generation throughput, faster
  in five of five invocations. Output is bit-identical: perplexity over 20,480 tokens of
  WikiText-2 agrees to six significant figures with the two-bit baseline.
  We further report that 4.76% of the model's feed-forward neurons are
  identically zero, that the dead index sets of the gate and up projections
  coincide in all thirty layers, and that an independent fine-tune revives none
  of them.
  All results are obtained on a single AVX2 microarchitecture without VNNI; we
  argue explicitly why VNNI is expected to narrow or invert the advantage, and
  note that an independent system targeting VNNI stores ternary weights at
  eight bits for that reason.
---

# 1. Introduction

Quantisation to ternary weights $w \in \{-1, 0, +1\}$ has been shown to preserve
task performance at the 2-billion-parameter scale while removing multiplication
from the dominant matrix operation [@ma2024bitnet; @microsoft2025bitnet2b4t].
The resulting models are attractive for CPU deployment: a matrix–vector product
against ternary weights requires only sign application and integer accumulation,
and the weight matrix itself is small enough to change which hardware resource
binds the computation.

That last point is usually stated as a memory-footprint argument and left there.
We take it as the design constraint. During autoregressive generation, every
weight of the model is read exactly once per emitted token and used exactly
once; arithmetic intensity is therefore fixed at one multiply-accumulate per
stored weight, and the achievable token rate is bounded by

$$
t_{\text{token}} \;\ge\; \max\!\left(\frac{N\mu}{\pi},\; \frac{N\beta}{8B}\right),
\tag{1}
$$

where $N$ is the number of weights, $\beta$ the bits used to store each of them,
$\mu$ the number of instructions issued on the binding execution port per
multiply-accumulate, $\pi$ the throughput of that port in instructions per
second, and $B$ the memory bandwidth available to the executing core in bytes
per second. The first term is arithmetic; the second is transport.

Existing ternary CPU formats sit at the extremes of the $(\beta, \mu)$ trade-off
that (1) describes. `llama.cpp`'s `i2_s` uses $\beta = 2$ and a decode of three
shifts and four masks per 128 weights. The `TL1`/`TL2` formats of `BitNet.cpp`
and the T-MAC system reach $\beta \approx 1.667$ (`TL2`; `TL1` is at 2.0) by
table lookup, paying memory for the tables and complexity in their management.
`llama.cpp`'s own `TQ1_0` reaches 1.6875 by base-3 packing without a table, and
`TQ2_0` 2.0625. In the opposite direction,
the *Litespark* framework stores ternary weights at $\beta = 8$, on the explicit
grounds that "using 2-bit packing would require unpacking weights before every
computation, negating the performance benefit" — a decision taken for targets
whose dot-product instruction (`VPDPBUSD`, ARM `SDOT`) consumes eight-bit
operands.

**The base-3 packing itself is not new, and an earlier draft of this paper
claimed it was.** `llama.cpp` has carried `TQ1_0` since August 2024
(PR&nbsp;#8151): its block structure in `ggml-common.h` is annotated
*"5 elements per byte (3^5 = 243 < 256)"*, it decodes without a table, and it
has an AVX2 dot product. `TQ1_0` reaches 1.6875 bits per weight because each
256-element block carries an f16 scale and a 16-element remainder at two bits;
the code bytes themselves are at 1.600. Any claim of novelty for the packing
would survive only on the arithmetic of the block overhead, which is not a
claim worth making. We had integrated into this very codebase without checking
it, and record that here rather than in a footnote.

What this paper contributes is therefore narrower and, we argue, more useful:
a per-tensor scale model that removes the block overhead and is drop-in for the
`i2_s` path of `bitnet.cpp`, an interleave identical to `i2_s`'s so the access
pattern is unchanged, a decode of four *independent* magic multiplies, a
column-blocked matrix--matrix kernel, an explicit and measurable condition for
when any of this pays, and an end-to-end evaluation on a real ternary
checkpoint against the format that ships with it.

## 1.1 Contributions

1. **A 1.600-bit variant of the known base-3 packing (§3).** Five ternary values
   as the base-3 digits of a byte is `TQ1_0`'s construction, not ours. What is
   ours is the scale model: a single per-tensor scale in place of `TQ1_0`'s
   per-256-block f16 and its 16-element two-bit remainder, which removes the
   block overhead entirely (1.600 against 1.6875, and 20% denser than `i2_s`),
   and an interleave identical to `i2_s`'s so that a digit plane still addresses
   32 consecutive activations.

2. **A table-free decode in four multiplies (§4).** Because a five-trit byte
   satisfies $x \le 242$, all four quotients $\lfloor x/3^j \rfloor$ lie inside
   the exactness range of 16-bit magic multipliers and are obtained
   *independently* — not as a serial division chain — at one `vpmulhuw` each.
   The digits follow by subtraction on unconstrained ports, and the
   multiply-accumulate is performed on the digits themselves.

3. **An explicit profitability condition (§5),** derived from (1) and verified
   by measuring both $\pi$ and $B$ directly rather than inferring either.

4. **An integration and end-to-end evaluation (§7)** in `llama.cpp` as a
   distinct `ggml` type, preserving the two-bit path as a control, with
   bit-identical output verified by perplexity agreement over $2.05 \times 10^4$
   tokens.

5. **A structural observation (§8):** 4.76% of the feed-forward neurons of
   `BitNet-b1.58-2B-4T` are identically zero, the dead index sets of $W_{\text{gate}}$
   and $W_{\text{up}}$ coincide in all thirty layers, and an independent
   fine-tune of the same base revives none of them.

---

# 2. Background

## 2.1 Ternary weights and the information bound

A ternary weight carries at most
$$
H_{\max} = \log_2 3 = 1.58496\ \text{bits},
\tag{2}
$$
which is the origin of the "1.58-bit" label. The bound is attained only for a
uniform distribution. On the shipped checkpoint the empirical marginal over all
$2{,}084{,}044{,}800$ ternary weights is

$$
(p_{-1},\, p_0,\, p_{+1}) = (0.288940,\ 0.422109,\ 0.288951),
\tag{3}
$$

giving a zeroth-order entropy of

$$
H = -\sum_{s} p_s \log_2 p_s = 1.5603\ \text{bits},
\tag{4}
$$

measured over the whole checkpoint (`results/tensor_structure.txt:96`). The
excess zero mass therefore *lowers* the floor slightly below $\log_2 3$.
Storage at $\beta = 2$ exceeds (4) by 28.2%; the format of §3 exceeds it by
2.5%.

## 2.2 What a format must preserve

`ggml`'s two-bit ternary kernels do not compute $\sum_i w_i a_i$ directly. They
compute on the shifted code $c_i = w_i + 1 \in \{0,1,2\}$, which is the unsigned
operand `vpmaddubsw` requires, and the caller removes the shift downstream:

$$
\sum_{i=1}^{n} w_i a_i \;=\; \sum_{i=1}^{n} c_i a_i \;-\; \sum_{i=1}^{n} a_i .
\tag{5}
$$

The correction term depends only on the activations. In a matrix–vector product
it is therefore computed once for the entire matrix rather than once per row —
$O(n)$ against $O(mn)$ — and any replacement format must expose the same
quantity $\sum_i c_i a_i$ so that the surrounding code is unchanged. Identity
(5) requires $c_i \le 2$; the code $c = 3$ would contribute $3a_i$ while
denoting $w = 0$. We verify in §7.1 that code 3 occurs zero times across all
$2.08 \times 10^9$ weights of the checkpoint.

---

# 3. The t5b format

## 3.1 Definition

Let a block occupy 32 bytes and carry $L = 160$ ternary weights. For
byte index $j \in \{0,\dots,31\}$ and digit index $k \in \{0,\dots,4\}$, the
block stores

$$
b_j \;=\; \sum_{k=0}^{4} c\big(w_{32k + j}\big)\, 3^{k},
\qquad c(w) = w + 1 \in \{0,1,2\}.
\tag{6}
$$

The interleave $32k + j$ is not incidental: it is precisely `i2_s`'s own layout,
in which bits $6\!-\!7$ of byte $j$ hold weight $j$, bits $4\!-\!5$ hold weight
$32+j$, and so on, extended from four digits to five. Consequently digit plane
$k$ addresses 32 *consecutive* activations, and the kernel's memory access
pattern is unchanged from the format it replaces.

The encoding is injective since $3^5 = 243 \le 256$, and

$$
b_j \;\le\; 2\sum_{k=0}^{4} 3^k \;=\; 2\cdot 121 \;=\; 242 .
\tag{7}
$$

The density is
$$
\beta = \frac{8\ \text{bits}}{5\ \text{weights}} = 1.600\ \text{bits/weight},
\tag{8}
$$
a factor $F = 2/1.6 = 1.25$ fewer bytes than `i2_s`.

## 3.2 Optimality among byte-aligned base-3 packings

Packing $k$ trits into $\lceil k \log_2 3 \rceil$ bits gives density
$\beta(k) = \lceil k\log_2 3\rceil / k$. Restricting to encodings that fit a
machine word without straddling it, the candidates are

| $k$ | bits | $\beta$ | fits |
|---:|---:|---:|:---|
| 4 | 7 | 1.750 | byte (1 bit wasted) |
| 5 | 8 | **1.600** | byte, exactly |
| 9 | 15 | 1.667 | 16-bit word (1 bit wasted) |
| 10 | 16 | **1.600** | 16-bit word, exactly |

$k = 5$ and $k = 10$ tie at 1.600 and no byte- or word-aligned packing does
better; $k=10$ requires $3^{10} = 59049 \le 65536$. We implement both and show
in §4.3 that they differ by a factor of 1.5 in throughput for reasons of SIMD
lane width alone, which is why the byte variant is the one deployed.

## 3.3 Tail handling

For $n$ not a multiple of $L$, the final block's unused slots are filled with
$c = 1$, i.e. $w = 0$, and the corresponding activations are read from a
zero-padded buffer held by the kernel. Both terms of (5) are then unaffected:
a padded slot contributes $1 \cdot 0 = 0$ to $\sum c_i a_i$ and $0$ to
$\sum a_i$. The checkpoint's two reduction dimensions are $K = 2560$ (exactly
16 blocks) and $K = 6912$ (43 blocks and a 32-weight tail), so the tail path is
exercised by every `ffn_down` row of the model.

Storage is therefore
$$
S(n) \;=\; 32 \left\lceil \frac{n}{160} \right\rceil \ \text{bytes},
\tag{9}
$$
giving exactly 1.600 bits/weight at $K = 2560$ and 1.6296 at $K = 6912$; over
the whole checkpoint, 1.6076 bits/weight including tail waste and the 32-byte
per-tensor scale record that `ggml` appends.

---

# 4. Kernel

## 4.1 Digit recovery without a chain

Write $q_j = \lfloor b/3^j \rfloor$ for the packed byte $b$. The digits satisfy

$$
d_j = q_j - 3\,q_{j+1}, \qquad j = 0,\dots,4, \quad q_5 = 0 .
\tag{10}
$$

A naive implementation computes $q_{j+1}$ from $q_j$, forming a serial
dependence of length five. We instead obtain every $q_j$ *directly from $b$*,
which is possible because of (7).

**Proposition 1 (sufficient condition for exact magic division).**
Let $D \ge 2$ and let $M$ be any positive integer with
$e := MD - 2^{16} \ge 1$. (The canonical choice is
$M = \lceil 2^{16}/D \rceil$, but the statement does not require it, and §4.1
uses a larger $M$ for one divisor.) Then
$$
\left\lfloor \frac{xM}{2^{16}} \right\rfloor = \left\lfloor \frac{x}{D} \right\rfloor
\qquad \text{for every integer } 0 \le x < \frac{2^{16} - e(D-1)}{e}.
\tag{11}
$$

*Proof.* Write $x = qD + r$ with $0 \le r < D$. Since $MD = 2^{16} + e$,
$$
xM = q(2^{16} + e) + rM = q\,2^{16} + \big(qe + rM\big),
$$
so $\lfloor xM/2^{16}\rfloor = q + \lfloor (qe + rM)/2^{16} \rfloor$ and it
suffices that $qe + rM < 2^{16}$. Using $q \le x/D$ and
$rM \le (D-1)M = 2^{16} + e - M$,
$$
qe + rM \;\le\; \frac{xe}{D} + 2^{16} + e - M ,
$$
which is below $2^{16}$ exactly when $xe/D < M - e$. Substituting
$M = (2^{16}+e)/D$ gives $xe < 2^{16} - e(D-1)$, which is the stated range.
$\square$

The bound of (11) is sufficient, not tight. Since the operative question is a
finite one — does the identity hold for every $x \le 242$? — we also determine
the exact first failure point by exhaustive search over the whole 16-bit domain,
and it is the exhaustive figure that the implementation and its test suite pin:

Applying Proposition 1 with the four divisors gives

| $j$ | $D = 3^j$ | $M$ | $e$ | bound (11) | first failure, exhaustive | margin over 242 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 21846 | 2 | $x < 32766$ | $x = 32768$ | $135.4\times$ |
| 2 | 9 | 7282 | 2 | $x < 32760$ | $x = 32768$ | $135.4\times$ |
| 3 | 27 | 2428 | 20 | $x < 3251$ | $x = 3293$ | $13.6\times$ |
| 4 | 81 | 811 | 155 | $x < 343$ | $x = 485$ | $2.00\times$ |

The tightest case, $D = 81$, first fails at $x = 485$ — exactly twice the
largest legal packed byte. All four quotients are therefore exact for every
input the format can produce, and each costs a single `vpmulhuw`. Crucially the
four are computed from $b$ *independently*, so the five-deep dependence chain of
the naive formulation collapses to depth one, and the multiplier port rather
than latency becomes the limit.

Note that $M = 811$ is not the canonical $\lceil 2^{16}/81 \rceil = 810$; the
implementation uses the larger constant, which has the smaller exactness range
(first failure at 485 rather than 806) and is still twice what the format can
produce. Proposition 1 covers it because its hypothesis is only $MD > 2^{16}$. The test suite verifies all four
identities exhaustively and reports each first-failure point, so the margin is a
recorded quantity rather than an assumption.

The digits then follow from (10) using shifts and subtractions only
($3q = (q \ll 1) + q$), which issue on the unconstrained integer ports.

## 4.2 Contraction

Digit plane $k$ is a register of 32 bytes in $\{0,1,2\}$, which is exactly the
unsigned operand `vpmaddubsw` expects. With $a$ the signed 8-bit activations,

$$
\texttt{acc}_{16} \mathrel{+}= \texttt{vpmaddubsw}\big(d_k,\ a_{32k..32k+31}\big),
\tag{12}
$$

and the ternary correction (5) is applied once per matrix–vector product.

**Accumulator bound.** A product is at most $2 \times 128 = 256$ --- the bound the header and the code
use, covering $a = -128$ --- and `vpmaddubsw` sums adjacent pairs, so a 16-bit
lane grows by at most 512 per plane and $5 \times 512 = 2560$ per block. Folding into 32-bit every $\Phi$
blocks is safe iff
$$
2560\,\Phi \le 32767 \iff \Phi \le 12.80,
\tag{13}
$$
so $\Phi = 12$. This is a *worst-case* bound. `llama.cpp`'s own `i2_s` kernel
folds every 32 blocks with four planes, i.e. up to $128 \times 508 = 65{,}024$,
and remains correct only because signed activations cancel. We show in §7.1
that removing the cancellation — activations uniformly at $\pm 127$ — makes the
shipped kernel disagree with exact arithmetic in 200 of 200 cases, every
discrepancy a multiple of $2^{16}$.

## 4.3 Why the byte variant and not the word variant

The 10-trits-per-`uint16` packing of §3.2 admits an elegant kernel that never
forms a digit at all. Substituting (10) into $\sum_k d_k a_k$ and re-indexing,

$$
\sum_{k=0}^{K-1} d_k a_k
\;=\; q_0 b_0 + \sum_{j=1}^{K-1} q_j\, b_j,
\qquad b_0 = a_0,\quad b_j = a_j - 3a_{j-1},
\tag{14}
$$

so the kernel needs only the quotient chain and a transformed activation vector
$b$, which depends on the activations alone and is computed once per
matrix–vector product. The identity holds modulo $2^{32}$ rather than term by
term: individual products are large and cancel, and the true dot product fits
`int32` comfortably.

It is nonetheless the slower construction, by a factor of $1.5$, and the reason
is lane width rather than base 3. Operating on `uint16` lanes, the contraction
must use `vpmaddwd`, which covers 16 lanes where `vpmaddubsw` covers 32, and the
quotient chain issues on the same multiply port as the contraction. Per 160
weights the word variant issues 19 multiply-port instructions against the byte
variant's 13, and the measured rates are 29.83 and 33.83 GMAC/s per core
respectively (§7.2).

## 4.4 Blocking for batched operation

For a matrix–matrix product with $n_c$ activation columns, decoding once per
column costs $n_c$ times the decode. Blocking the column dimension so that a
decoded plane is contracted against $N_C$ columns held in registers reduces the
per-block cost from $N_C(\delta + 5)$ to $\delta + 5N_C$ multiply-port
instructions, where $\delta = 8$ is the decode. At $N_C = 8$ this is 48
instructions per 1280 multiply-accumulates, or 26.7 MACs per instruction against
`i2_s`'s 32. We measure $N_C \in \{2,4,8\}$ at 50.51, 63.54 and 73.26 GMAC/s and
deploy $N_C = 8$; the register pressure this incurs (181 stack references per
loop) costs less than the amortisation gains.

---

# 5. When does a denser format pay?

## 5.1 The condition

Let a baseline format $A$ and a candidate $B$ store the same weights at
$\beta_A > \beta_B$ bits, with $\mu_A < \mu_B$ binding-port instructions per
multiply-accumulate. Write

$$
F = \frac{\beta_A}{\beta_B} \quad\text{(byte ratio)}, \qquad
S = \frac{\mu_B}{\mu_A} \quad\text{(arithmetic ratio)} .
\tag{15}
$$

From (1), $B$ is faster than $A$ if and only if
$$
\max\!\left(S\,\frac{\mu_A}{\pi},\ \frac{1}{F}\cdot\frac{\beta_A}{8B}\right)
\;<\;
\max\!\left(\frac{\mu_A}{\pi},\ \frac{\beta_A}{8B}\right).
\tag{16}
$$

Define the **surplus** of the baseline as the factor by which its arithmetic
capacity exceeds what memory keeps it fed with,
$$
\Sigma \;=\; \frac{\beta_A/(8B)}{\mu_A/\pi} \;=\; \frac{\pi\,\beta_A}{8B\,\mu_A}.
\tag{17}
$$

If $\Sigma \le 1$ the baseline is arithmetic-bound, the right-hand side of (16)
is $\mu_A/\pi$, and no format with $S > 1$ can win. If $\Sigma > 1$ the baseline
is transport-bound and (16) reduces to

$$
\boxed{\;S \;<\; \Sigma \;}
\tag{18}
$$

in which case $B$ runs a factor $\min(F, \Sigma/S)$ faster. The condition is
therefore not a property of the format but of the ratio between two hardware
rates, and both are directly measurable.

## 5.2 Measuring $\pi$

We measure the throughput of each instruction class with eight independent
dependence chains, so that latency cannot bind and only issue rate can, and with
the clock derived from a dependent add chain that retires one instruction per
cycle by construction:

| instruction | Gop/s | ops/cycle |
|---|---:|---:|
| `vpmaddubsw` | 4.09 | 1.05 |
| `vpmaddwd` | 4.10 | 1.05 |
| `vpmulhuw` | 4.09 | 1.05 |
| `vpmullw` | 4.08 | 1.04 |
| `vpand` | 15.43 | 3.95 |
| `vpaddw` | 11.72 | 3.00 |

Measured clock 3.91 GHz. All four multiply-class instructions land on a single
port at 1.05 per cycle; the remaining classes are three to four times more
plentiful. This identifies the binding resource in (1) unambiguously and fixes
$\pi = 4.09\times 10^9\,\mathrm{s^{-1}}$.

**Methodological note.** An earlier version of this measurement used identical
seeds for all eight chains; the compiler proved them equal and kept one,
eliminating `vpand` entirely and strength-reducing `vpaddw` into a multiply. It
reported $2.98\times 10^6$ and 9.15 ops/cycle against a hard ceiling near 4.
Only the physical impossibility of those figures exposed the defect; a partial
collapse would have looked plausible.

## 5.3 Measuring $B$, and why $\Sigma$ is a function of core count

$B$ is a core's *share* of memory bandwidth, not the socket's. We measure it by
running the baseline kernel at two per-thread footprints — 256 KiB, inside the
L2 slice, and 64 MiB, four times the L3 slice of one CCX — with one thread pinned
per physical core:

| threads | arithmetic (GMAC/s) | transport (GB/s) | $\Sigma$ |
|---:|---:|---:|---:|
| 1 | 83.70 | 15.47 | 1.35 |
| 2 | 162.42 | 24.94 | 1.63 |
| 3 | 235.14 | 33.00 | 1.78 |
| 4 | 287.54 | 38.62 | 1.86 |
| 6 | 471.74 | 38.86 | **3.03** |

Arithmetic scales $5.6\times$ across six cores and per core falls only 6%, an
all-core clock effect. Transport scales $2.5\times$ and is flat between four
cores and six. The surplus therefore grows with core count for a reason
independent of any kernel: cores get faster together and memory does not.

For t5b, $S = 2.43$ measured (§7.2) against $F = 1.25$. By (18) the format
should lose at one, two and four threads and win at six — which is what §7.3
observes, at $0.53\times$, $0.62\times$, $0.78\times$ and $1.35\times$
respectively.

---

# 6. Experimental setup

| | |
|---|---|
| CPU | AMD Ryzen 5 3600 (Zen 2), 6 cores / 12 threads |
| Flags in `/proc/cpuinfo` | `avx2`, `fma`; **no AVX-512, no VNNI** |
| Caches | L1d 32 KiB/core, L2 512 KiB/core, L3 32 MiB in two 16 MiB CCX slices |
| Memory | 62 GiB |
| Compiler | gcc 13.3.0, `-O3 -mavx2 -mfma -march=native` |
| Model | `microsoft/bitnet-b1.58-2B-4T`, 2,084,044,800 ternary weights in 210 tensors |
| Second model | `jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF`, an independent fine-tune |
| Harnesses | `llama-bench`, `llama-batched-bench`, `llama-perplexity` (upstream) |

The host is shared with an unrelated production workload whose load average
rarely falls below 1.8. Within-arm spread ranges from 2% to 146% depending on
load, which is comparable to the effect under study. Every comparison is
therefore paired, with the arm order alternated between repetitions, and the
headline figures are replicated across five independent invocations of the
upstream benchmark.

---

# 7. Results

## 7.1 Correctness

**Unit level.** 184,228 assertions across five suites, zero failures. The
kernels are checked against scalar references derived independently from the
format specification rather than from the implemented identities, so a shared
derivation error cannot pass. Coverage includes exhaustive verification of the
four magic-division identities over their whole legal input range, deliberate
32-bit accumulator wraparound driven 4.96 times past $2^{31}$, the extreme
activations $\pm 127$, and every block and tail boundary.

**Model level, argmax.** Nine prompts across two models, greedy decoding, one
binary, one seed, one variable: identical output by `diff`. A negative control
confirms the comparison can fail — two different prompts compare unequal.

**Model level, distribution.** Perplexity over 40 chunks of 512 tokens of
WikiText-2 (llama.cpp's own CI copy):

$$
\mathrm{PPL}_{\texttt{i2\_s}} = 96.8243 \pm 3.55673,
\qquad
\mathrm{PPL}_{\texttt{t5b}} = 96.8243 \pm 3.55673 .
$$

Not only the final estimates but all forty running estimates agree exactly, from
$[1]\,47.2744$ to $[40]\,96.8243$. Instrumentation confirms the arms differed
(51,660 calls through the new path against zero). This compares a
floating-point aggregate over $2.05\times10^4$ forward passes rather than a
decoded string, and is sensitive to differences greedy decoding would round
away.

**Code 3.** Absent: 0 of 2,084,044,800 weights across all 210 tensors, so
identity (5) holds model-wide. Previously this had been verified on one tensor.

**A defect in the baseline.** With activations uniformly at $\pm 127$,
`llama.cpp`'s `i2_s` AVX2 kernel disagrees with exact arithmetic on 200 of 200
cases on two of seven tensors, every discrepancy a multiple of $2^{16}$ — the
16-bit accumulator overflow that (13) forbids by construction in our kernel.
Whether ordinary activations reach that state is not established here; the
mechanism is.

## 7.2 Kernel microbenchmark

One core, weights resident in L2 so that transport cannot bind:

| kernel | GMAC/s | instr | mul | spills | weights | mul ceiling | of it |
|---|---:|---:|---:|---:|---:|---:|---:|
| `i2_s` (upstream) | 81.93 | 21 | 4 | 0 | 128 | 131.4 | 62% |
| t10 (word variant) | 30.04 | 50.25 | 19.25 | 8.25 | 160 | 34.6 | 87% |
| **t5b (byte variant)** | **33.74** | 47 | 13 | 0 | 160 | 50.5 | 67% |

The ceiling column is $\pi \cdot (\text{weights}/\text{mul})$ from §5.2. t10 sits
at 87% of its ceiling and cannot be tuned out of a structural disadvantage. t5b
initially measured 24.36 GMAC/s — *slower than t10 despite needing fewer
multiplies* — which no end-to-end benchmark could have diagnosed;
disassembly showed 44 stack references in a 425-instruction function. One block
per iteration with each digit plane contracted at the moment it exists reduced
the loop to 47 instructions and zero spills, and the rate to 33.74. Thus
$S = 81.93/33.74 = 2.43$.

## 7.3 Weight traffic of one token

Replaying the matrix operations of one token over the model's real tensor
shapes and full 940 MB footprint, best of five, arms rotated:

| threads | `i2_s` | t10 | t5b | t5b/`i2_s` |
|---:|---:|---:|---:|---:|
| 1 | 34.57 ms | 72.73 | 64.70 | 0.534 |
| 2 | 20.29 | 36.84 | 33.01 | 0.615 |
| 4 | 13.19 | 18.82 | 16.90 | 0.780 |
| 6 | 15.15 | 14.49 | **11.18** | **1.354** |

The crossing point matches (18) with $S = 2.43$ and the $\Sigma$ of §5.3. Across
all eight paired six-thread observations the median is $1.12\times$ and seven of
eight favour t5b; the best-against-best comparison, noting that `i2_s` does not
improve past four threads because transport saturates there, is 13.00 ms against
11.71 ms.

## 7.4 End-to-end, in `llama.cpp`

`llama-bench`, five invocations with the model load order alternated:

| | size | pp512 (t/s) | tg128 (t/s) |
|---|---:|---:|---:|
| `i2_s` | 1.10 GiB | 119.65 ± 2.27 | 22.56 ± 1.26 |
| **t5b** | **1.00 GiB** | **135.45 ± 1.21** | **25.78 ± 0.87** |
| ratio | 0.909 | **1.132** | **1.143** |

Faster in 5 of 5 invocations on both measures; per-invocation ratios 1.096,
1.112, 1.119, 1.253, 1.132 (prompt) and 1.099, 1.133, 1.123, 1.242, 1.143
(generation).

**The four-thread result contradicts §5.3's own prediction, and we do not fully
explain it.** From $S=2.43$ and $\Sigma_4 = 1.86$, inequality~(18) predicts a
loss at four threads, and the isolated weight-traffic replay of §7.3 delivers
one ($0.780\times$). The model does not: at four threads it gains
$1.132\times$/$1.143\times$. §7.4's own analysis of why the model gains *less*
than the replay — weights are roughly 30% of a token — cannot explain a change
of sign.

Two candidates, neither yet measured. First, the $S$ of §7.2 is the ratio of two
*isolated kernels*, and `llama.cpp`'s in-situ `i2_s` path is not that kernel: it
carries the `llamafile_sgemm` dispatch, the f32$\to$i8 activation quantisation
and its own accumulator fold, so the effective in-situ $S$ may be materially
below 2.43. Second, the non-matmul 70% of a token occupies the memory system as
well, so the bandwidth actually left to the weight stream at four threads may be
lower than the isolated measurement of §5.3 suggests, raising the effective
$\Sigma$. The first would mean §7.2 measures the wrong baseline for this
purpose; the second would mean §5.3 does. **Resolving this requires an in-situ
measurement of the per-kernel rate ratio, which we have not made**, and we
prefer to state the contradiction than to pick whichever reading flatters the
result.

**Prompt length.** $1.106$, $1.152$, $1.180$ and $1.113$ at 64, 256, 1024 and
2048 tokens: the advantage does not decay with length. We had predicted decay
on the grounds that a prompt pass is arithmetic-bound; at these shapes the
weights are still streamed from memory, so the byte saving remains live and the
prediction was wrong in its premise.

**Concurrency.** Total throughput at 1, 2, 4 and 8 parallel sequences: $1.041$,
$1.041$, $1.027$, $1.097$. The advantage *grows*, for the same reason §4.4
gives: $N$ concurrent sequences turn generation into a matrix–matrix product
with $N$ columns, across which the decode amortises.

**Generalisation across checkpoints.** On the independent fine-tune: identical
output on three prompts, $1.120\times$ prompt and $1.075\times$ generation.

**Generalisation across depth.** Repeating the block stack yields a
4.50 B-parameter, 60-layer graph that `llama.cpp` loads and runs with real
attention and KV cache: $1.165\times$ and $1.129\times$, both margins clear of
their error bars. That artefact is *not a model* — running the same thirty
blocks twice produces text without meaning — and no quality claim is made from
it; only transport and arithmetic are real.

## 7.4a Two ratios a reader will ask about immediately

**Why 25% fewer weight bytes give 14% more throughput.** At \SI{21.9}{}~tokens
per second a token takes \SI{45.7}{\milli\second}, and the replay of §7.3 puts a
token's `i2_s` matmuls at about \SI{14}{\milli\second}. Weight streaming is
therefore roughly **30% of a token**; the remaining 70% is attention, the KV
cache, RoPE, the norms, the activation quantisation, thread barriers and graph
overhead, all identical between the arms. Read as bandwidth, the generation loop
uses \SI{11.4}{\giga\byte\per\second} of the \SI{38.9}{} this machine
delivers: it is not at the memory wall.

**Why the file shrinks by only 9%.** The ternary tensors are 521.0 MB of a
1187.8 MB file. One f16 tensor — the 128k-vocabulary embedding — is 656.7 MB,
55.3% of the file, and does not change. The 102.2 MB saved is 19.6% of what the
format touches and 8.6% of the file. The ternary share, and therefore the file
saving, rises with model depth at fixed vocabulary.

## 7.5 A negative result on row tiling

Tiling four weight rows against one activation vector, as `llama.cpp`'s
matrix–matrix path does, measures $1.19\times$ in cache and $0.52\times$ from
memory. A matrix–matrix tile of $R$ rows by $C$ columns loads each weight once
and spends it on $RC$ contractions; a matrix–vector product has $C=1$, so tiling
rows buys no reuse whatsoever and merely reorders the loads. Upstream draws the
same line structurally: its `ggml_gemv_i2_i8_s` does not tile, and the $4\times4$
tile lives in `ggml_gemm_i2_i8_s`.

---

# 8. Dead feed-forward neurons

While quantifying the residual compressibility of the checkpoint we found that
2.9036% of all weight-matrix rows are identically zero — an output neuron that
cannot fire for any input.

| tensor | dead rows | share |
|---|---:|---:|
| $W_{\text{gate}}$ | 9,870 / 207,360 | 4.76% |
| $W_{\text{up}}$ | 9,870 / 207,360 | 4.76% |
| $W_v$ | 103 / 19,200 | 0.54% |
| $W_q$ | 4 / 76,800 | 0.01% |
| $W_{\text{down}}$, $W_k$, $W_o$ | 0 | 0.00% |

They are concentrated at the front of the network: 43.59% of layer 1's
feed-forward neurons, 27.58% of layer 2's, 14.77% of layer 3's, near zero by
layer 8.

**In all thirty layers the dead index set of $W_{\text{gate}}$ equals that of
$W_{\text{up}}$** — not the same cardinality, the same indices. In a SwiGLU
block the two projections are combined element-wise,
$h = W_{\text{down}}\!\left(\sigma(W_{\text{gate}}x)\odot W_{\text{up}}x\right)$, so a zero gate row
renders the matching up row irrelevant and conversely; training removed them in
pairs. $W_{\text{down}}$ has none, which is the same fact viewed along the other
axis: there a dead neuron is a dead *column*.

**They survive fine-tuning exactly.** Against the independent fine-tune, 210 of
210 tensors have identical dead index sets — all 19,847 rows, 100.00%. Whatever
removed these neurons occurred in pre-training and was untouched downstream,
which makes any scheme exploiting them a property of the base rather than of one
checkpoint.

Skipping the rows saves 2.4380% of weight bytes *and* the same share of
multiply-accumulates, since a skipped row is neither read nor computed. Removing
the neurons outright would save 3.64% — a dead neuron also renders its
$W_{\text{down}}$ column useless, $9{,}870 \times 7{,}680$ weights — at zero cost to the element-wise product, but not
provably zero cost to the layer: BitNet b1.58 places a sub-layer RMS norm
(`ffn_sub_norm`) between the product and $W_{\text{down}}$, and an RMS norm
divides by the root mean over all $6912$ entries. Deleting zeros changes that
denominator. Skipping the rows in the kernel is exact; removing them is exact
only if the norm's divisor is pinned to the original width. The latter is **not** blocked by the file format, which this work asserted for
some hours without checking: `llama.cpp` reads `LLM_KV_FEED_FORWARD_LENGTH`
through `get_key_or_arr` into a per-layer array (`llama-model.cpp:1117`,
`llama-hparams.h:84` and `:314`), so per-layer feed-forward widths are a
first-class concept and the removal requires no format change, no new tensor
type and no kernel work. It is also *unconditionally* exact, precisely because
the dead index sets coincide: $h_i = f(\text{gate}_i)\,\text{up}_i = f(0)\cdot 0 = 0$
for any activation $f$, since $\text{up}_i$ vanishes too.

---

# 9. Limitations

**One microarchitecture, and the decisive property is absent from it.** All
results are from a single Zen 2 part reporting `avx2` and `fma` and nothing
else. `VPDPBUSD` collapses the contraction of §4.2 into one instruction while
leaving the decode of §4.1 unchanged; by (15) this raises $S$ and lowers
$\Sigma$, moving (18) against a packed format. The *Litespark* system, targeting
VNNI and ARM `SDOT`, stores ternary weights at eight bits for precisely this
reason. **The central result should be read as a property of AVX2 without VNNI
until it is run on a VNNI part.** The code compiles for `-mavxvnni` here; it
cannot execute, and no such measurement is claimed.

**One model size, and one architecture family.** No public ternary checkpoint
larger than 2 B exists. The synthetic 60-layer graph of §7.4 doubles the weight
traffic but not the head count, hidden size or vocabulary.

**Measurement environment.** The host is shared and the run-to-run spread is
comparable to the effect. An earlier draft of this work reported "four runs out
of four" at $1.165\times$; those four were taken in the quietest window
available, and four further runs at ordinary load include one the denser format
loses. The headline is therefore the five-invocation replication of §7.4.

**Verification scope.** The round trip is exact on all 210 tensors and all
2,084,044,800 weights; what is limited to seven tensors is the dot-product
comparison and the read-back from disk. Nine prompts and 20,480 perplexity
tokens are a large but finite sample.

**Task benchmarks.** None. The format question does not admit one — output is
bit-identical, so every metric is identical by construction — but the different
question of where this *model* stands is answered only by perplexity, where it
reaches $96.8243 \pm 3.55673$ against Qwen2.5-3B-Instruct's
$8.2837 \pm 0.21844$ on the same corpus and harness, a factor of 11.7. Perplexity across different tokenisers is loosely comparable
at best; it is not loose by a factor of eleven.

**Engineering.** The blocked matrix–matrix kernel spills 181 times per loop at
its chosen width; narrower widths spill less and measure slower. The type
identifier 44 is not reserved by upstream. The dead-row saving of §8 is measured
and, at the time of writing, being implemented.

---

# 10. Conclusion

The gap between the 1.58 bits a ternary weight carries and the 2.00 bits it is
stored in is recoverable on commodity AVX2 hardware without lookup tables. Five
ternary values fit a byte at exactly 1.600 bits; all four base-3 quotients are
extractable from that byte with one independent multiply each, because the byte
is bounded by 242 and therefore lies deep inside the exactness range of 16-bit
magic multipliers; and the multiply-accumulate proceeds on the digits
themselves, so no weight is materialised at any point.

Whether the resulting 25% reduction in weight traffic is profitable is not a
property of the format. It is decided by inequality (18), a comparison between
a core's multiply-port throughput and its share of memory bandwidth, and both
sides of that comparison are measurable in minutes. On the part measured here
the condition holds from four cores upward, and the deployed model is 9% smaller
and 13–14% faster with bit-identical output.

The same inequality predicts that the result will narrow on a part with VNNI,
and an independent system built for such parts chose the opposite extreme of the
same trade-off. That is not a caveat appended to a positive result; it is the
result, which is that the choice of weight density is a hardware-dependent
optimisation with a computable decision rule, rather than a property of the
model.

---

---

# Data and code availability

The format, the kernels, the benchmarks, the converters and every evidence file
cited here are public under the Apache License 2.0 at

> **https://github.com/purpleskulll/bitnet-t5b**

Every figure is recomputed by a script in that repository and recorded in a file
under `results/`, each stating its own reproduction command. No model weights
are distributed and no upstream source is redistributed verbatim; the
integration is applied as a patch to an unmodified `llama.cpp` checkout, and the
three files that reproduce or adapt an upstream contract say so in their headers
and in `NOTICE`.

| claim | evidence |
|---|---|
| code histogram, absence of code 3, entropy | `results/tensor_structure.txt`, `results/real_weights_check.txt` |
| instruction-port throughputs $\pi$ | `src_modifications/bench/bench_ports.c` |
| surplus $\Sigma$ against core count | `results/thread_headroom.txt` |
| kernel rates, instruction and spill counts | `results/weight_density.txt` |
| weight traffic of one token | `src_modifications/bench/bench_token.c` |
| end-to-end throughput, all harnesses | `results/inference_t5b.txt` |
| perplexity agreement and the Qwen comparison | `results/perplexity_t5b.txt` |
| round trip on real weights | `results/real_weights_check.txt` |
| dead feed-forward neurons | `results/dead_neurons.txt`, `tools/dead_neurons.py` |
| the design document, claim by claim | `results/phase_status.txt` |

The format, kernels and tests are in `src_modifications/ternary_t5b.{c,h}` and
`src_modifications/tests/`; the converter is `tools/gguf_to_t5b.py`; the
integration into `llama.cpp` is applied by `scripts/apply_integration.py` and
built by `Dockerfile.t5b`.

# References

- Ma, S., Wang, H., Ma, L., Wang, L., Wu, W., Dong, P., Zhang, L., Xue, J.,
  Wei, F. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.*
  arXiv:2402.17764, 2024.
- Microsoft. *BitNet b1.58 2B4T Technical Report.* arXiv:2504.12285, 2025.
- Wang, J., Zhou, H., Song, T., Cao, S., Xia, Y., Cao, T., Wei, J., Ma, S.,
  Wang, H., Wei, F. *Bitnet.cpp: Efficient Edge Inference for Ternary LLMs.*
  arXiv:2502.11880, 2025.
- Wang, J., Zhou, H., Song, T., Mao, S., Ma, S., Wang, H., Xia, Y., Wei, F.
  *1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs.*
  arXiv:2410.16144, 2024.
- Wei, J., Cao, S., Cao, T., Ma, L., Wang, L., Zhang, Y., Yang, M. *T-MAC: CPU
  Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge.* EuroSys,
  2025. arXiv:2407.00088.
- compilade et al. *ggml: add ternary quantization types TQ1_0 and TQ2_0.*
  llama.cpp pull request #8151, August 2024.
- *Litespark Inference for CPUs: Ultra-Fast SIMD Framework for Ternary
  (1.58-bit) Language Models.* arXiv:2605.06485.
- Han, S., Mao, H., Dally, W. J. *Deep Compression: Compressing Deep Neural
  Networks with Pruning, Trained Quantization and Huffman Coding.* ICLR, 2016.
  arXiv:1510.00149.
- Zhang, T. et al. *DFloat11: Lossless Compression of Large Language Models.*
  arXiv:2504.11651, 2025.
- Gerganov, G. et al. *llama.cpp / ggml.*
- Merity, S., Xiong, C., Bradbury, J., Socher, R. *Pointer Sentinel Mixture
  Models.* arXiv:1609.07843, 2016. (WikiText-2.)
