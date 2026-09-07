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
  instructions per 16-bit view — eight per 32-byte block — exploiting the fact that a five-trit byte is bounded by 242 and
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
  Against `TQ1_0`, `llama.cpp`'s own base-3 ternary type and the nearest prior
  art, the margin is narrower and we report it as such: 1.60755 against 1.68750
  achieved bits per weight, 4.7% of the payload, plus a prompt-processing
  advantage that comes from having a batched kernel rather than from the packing.
  We further report that 4.76% of the model's feed-forward neurons are
  identically zero, that the dead index sets of the gate and up projections
  coincide in all thirty layers, and that an independent fine-tune revives none
  of them.
  Our own results are obtained on a single AVX2 microarchitecture without VNNI.
  Two independent runs bound that limitation: a static pipeline model, calibrated
  against the host to 3.0%, finds the arithmetic ratio stable across Zen 3/4/5,
  Ice Lake and Sapphire Rapids, and a reviewer's Intel Xeon reproduces it at 2.11
  against our 2.43. On real VNNI hardware the same reviewer measures the ratio
  rising by 8.8% -- the direction predicted, from the predicted mechanism, at a
  magnitude smaller than the eight-bit storage choice of a system built for those
  targets would suggest, though on a shared host whose spread exceeds the
  effect.
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
`TQ2_0` 2.0625; Spectra 1.1 publishes that family with a correctness theorem and
CPU kernels as `TQ1` and `TQ2`, recommending exactly the $p=8$, $k=5$ this work
uses. Every format named so far is lossless. In the lossy direction, *Sherry*
reaches 1.25 bits by imposing 3:4 fine-grained sparsity on the ternary weights
and packing four of them into five bits, which restores power-of-two alignment
at the cost of a training-time constraint and of exactness — a different
trade from the one studied here, where the weights are unchanged bit for bit. In
the opposite direction,
the *Litespark* framework stores ternary weights at $\beta = 8$, on the explicit
grounds that "using 2-bit packing would require unpacking weights before every
computation, negating the performance benefit" — a decision taken for targets
whose dot-product instruction (`VPDPBUSD`, ARM `SDOT`) consumes eight-bit
operands.

**The base-3 packing itself is not new.** `llama.cpp` has carried `TQ1_0` since August 2024
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
   as the base-3 digits of a byte is not ours: it is `TQ1_0`'s construction in
   `llama.cpp`, and it is published with a correctness theorem by Vaidhya et al.
   as Spectra 1.1's `TQ1` [Spectra], whose $p=8$, $k=5$ recommendation and 1.6-bit
   figure are the same as ours. What is ours is the scale model: a single
   per-tensor scale in place of `TQ1_0`'s per-256-block f16 and its 16-element
   two-bit remainder, which removes the block overhead entirely (1.600 against
   1.6875, and 20% denser than `i2_s`), and an interleave identical to `i2_s`'s
   so that a digit plane still addresses 32 consecutive activations.

2. **A decode of depth one rather than depth five (§4).** That a
   multiplication-based decode avoids division and modulo is also known:
   Spectra 1.1 exploits $3^5 \approx 2^8$ to extract the trits *iteratively*,
   $b_{i+1} = (3b_i) \wedge \texttt{0xFF}$ for $i = 0 \dots 4$, which is a
   strictly serial chain of five dependent multiplies. Ours is the observation
   that a five-trit byte satisfies $x \le 242$, so all four quotients
   $\lfloor x/3^j \rfloor$ lie inside the exactness range of 16-bit magic
   multipliers and can be taken **independently, straight from the original
   byte**, at one `vpmulhuw` each. The dependency depth falls from five to one;
   the digits follow by subtraction on unconstrained ports, and the
   multiply–accumulate is performed on the digits themselves. This costs more
   multiply-port operations than the serial form and buys latency; §5 gives the
   condition under which that is the right trade, and we note that no head-to-head
   measurement against Spectra's kernel was made — their figures are from a
   Mac M4 and an EPYC part, and ours from one Zen 2 host.

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

**Relation to Spectra 1.1's Theorem 1.** Vaidhya et al. [Spectra] state the
*correctness* condition for this family: packing $k$ trits into a $p$-bit
integer is lossless if and only if $2^p > 3^k$, and they likewise recommend
$p=8$, $k=5$ for an effective 1.6 bits. The table above is the *optimality*
counterpart — which admissible $(p,k)$ minimises $p/k$ subject to $p$ being a
whole machine word — and every entry in it satisfies their condition by
construction. We claim no priority over the condition, the recommendation or the
resulting bit rate: all three are theirs, and $k=5$ is `TQ1_0`'s construction in
`llama.cpp` before either. What §3.2 adds is the alignment argument that makes
$k=10$ the only other candidate at the same density, which §4.3 then rules out
on lane width.

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
variant's 13, and the measured rates are 30.04 and 33.74 GMAC/s per core
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

**Unit level.** 184,228 assertions across five suites, zero failures; of these
the 177,734 covering the format and both kernel variants are in the public
repository and run from a clone with a compiler alone (`make test`). The other
three suites cover parts of the wider project not published here. The
kernels are checked against scalar references derived independently from the
format specification rather than from the implemented identities, so a shared
derivation error cannot pass. Coverage includes exhaustive verification of the
four magic-division identities over their whole legal input range, deliberate
32-bit accumulator wraparound driven 4.96 times past $2^{31}$, the extreme
activations $\pm 127$, and every block and tail boundary.

**How much that assertion count is worth.** An assertion count measures effort,
not power, so the suite's power is measured directly by mutation
(`mutation_test.sh`): fourteen defects are injected one at a time into
the kernel and its header — each magic multiplier perturbed, the digit
multiplication changed from $3y$ to $5y$, the odd-byte view shifted by seven
instead of eight, the low-byte mask narrowed, two digit planes given each
other's activations, the radix changed, the digit count reduced, and the
accumulator fold raised from 12 to 13 and to upstream's 32. **Twelve are killed.**

The remaining two survive, and checking rather than filing them is what makes
the result meaningful: both are *exactly equivalent* on every reachable input,
proved exhaustively rather than argued. Replacing the magic multiplier 811 by
810 leaves $\lfloor x/81 \rfloor$ unchanged on all 256 byte values; replacing the
byte-wise digit subtraction by a word-wise one changes nothing on any of the
$65{,}536$ word lanes, because no low byte ever borrows — which is the invariant
§4.1 asserts, here confirmed independently of the text that asserts it. The
script carries both proofs and fails if either mutant is ever *killed*, since
that would mean the kernel had lost the property the proof rests on.

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
`llama.cpp`'s `i2_s` AVX2 kernel disagrees with exact arithmetic on **11,998 of
12,000** rows at $K = 6912$, every discrepancy a multiple of $2^{16}$ — the
16-bit accumulator overflow that (13) forbids by construction in our kernel. The
prediction that overflow requires $n_b = K/128 \ge 32$, i.e. $K \ge 4096$, is
confirmed on both sides: **0 of 108,000** rows at $K = 2560$ wrap, under the same
activations.

The two exceptions are the bound speaking rather than noise, and they matter for
how the claim is phrased. The margin is $256 \times 127 = 32{,}512$ against
$32{,}767$, or $0.78\%$; whether a given row overflows therefore turns on its own
code mean, and a row slightly below average clears the ceiling. At 12,000 rows
that tail is visible, so the correct statement is statistical rather than
absolute: direction and magnitude hold exactly, "must" does not.

Whether ordinary activations reach that state is not established here; the
mechanism is. On random `int8` the baseline is exact on all 42,000 rows tried,
which is why the two kernels agree wherever it matters.

## 7.2 Kernel microbenchmark

One core, weights resident in L2 so that transport cannot bind:

| kernel | GMAC/s | instr | mul | spills | weights | mul ceiling | of it |
|---|---:|---:|---:|---:|---:|---:|---:|
| `i2_s` (upstream) | 81.93 | 21 | 4 | 0 | 128 | 130.9 | 63% |
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
shapes and full weight footprint of both arms together, 940 MB, of which the 521 MB of `i2_s` codes is what a single arm streams, best of five, arms rotated:

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

**The four-thread result contradicted §5.3's prediction; the contradiction is
now resolved, and the prediction was right.** From $S=2.43$ and $\Sigma_4=1.86$,
(18) predicts a loss at four threads and the isolated replay of §7.3 delivers
one ($0.780\times$); the model gains $1.132\times$. Instrumenting the glue with
a cycle counter — `rdtsc` at the invariant TSC rate of 3.599978 GHz, not the
core clock, with `llama-bench`'s untimed warmup removed by differencing $r=2$
against $r=10$ — gives the matmul time directly for one arm, and by difference
for the other, since everything else a token does is identical:

| kernel | in situ | replay (§7.3) | ratio |
|---|---:|---:|---:|
| t5b | 16.70 ms/token | 16.90 ms | **0.988** |
| `i2_s` | 21.18 ms/token | 13.19 ms | **1.606** |

The replay predicts *our* kernel to 1.2% and mispredicts `llama.cpp`'s `i2_s`
path by 61%. The effective in-situ ratio is therefore
$S_{\text{eff}} = 2.43/1.606 = 1.51$ against $\Sigma_4 = 1.86$, so
$S < \Sigma$ **holds** at four threads and (18) predicts the observed gain
rather than contradicting it.

What does not survive is the use of `bitnet_vec_dot_i2_i8_s_reference` as a
stand-in for `llama.cpp`'s real `i2_s` performance: it is $1.6\times$ faster than
what the model runs, so every ratio computed against it — the 2.43 included —
flatters the baseline. No end-to-end number changes; §7.4 and §7.6 measure whole
models and never used the reference kernel. Matmuls are **43.6%** of generation
wall time, not the 30% §7.4a estimates from the replay. Why the in-situ `i2_s`
path costs $1.6\times$ its reference kernel is not established here; the
dispatch, the accumulator fold and the per-column post-processing are all
candidates, and isolating them would mean instrumenting the control arm.

**The 1.606 is a weaker number than the table makes it look, and the integration
now carries the fix.** It is not measured: it is derived from two `llama-bench`
runs, taken separately on a host whose load ranged from 2 to 10 (4.905 s against
5.479 s per repetition), so every fluctuation *between* those runs lands entirely
in the `i2_s` matmul figure. The integration therefore probes **both** arms at
the same dispatch site in the same expression, so the ratio is a quotient of two measured quantities and
whatever the dispatch itself costs is common to both and divides out.

The probe's cost is measured rather than assumed
(`benchmarks/bench_probe_cost.c`). An `rdtsc` pair alone costs 19 ns and is flat
in thread count. The same pair with an atomic add to **one shared counter** —
which is what the original instrumentation used — costs 18.7 ns at one thread
and 87.6 ns at six, a factor of 4.7, because every `ggml` worker issues a
read-modify-write to the same cache line and they serialise on the coherence
protocol. Padding the counter per thread returns it to ~19 ns, flat. Against the
94.8 µs a single matmul call takes, even the worst case is 0.09% per probe and
0.18% for the entry/exit pair, so no version of this endangers the measurement;
but the shared counter's cost *grew along the thread axis §7.4 compares on*, and
a systematic error that tracks the independent variable is worth removing at any
size. The counters are per-thread.

**What this does not yet produce is a number.** Running it requires rebuilding
the patched `llama.cpp` image, which needs a Docker daemon this host does not
grant. The instrumentation is in `integration/t5b-integration.patch` and in
`apply_integration.py`, and the three-way consistency of those with each other
is gated (`tools/check_patch_chain.py`, `check_patch_parity.py`,
`fix_patch_hunks.py`); the measurement it enables is not claimed here, and the
1.606 above stands as the derived quantity it is.

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
per second a token takes \SI{45.7}{\milli\second}, of which the weight matmuls
are **43.6%** — \SI{19.9}{\milli\second}. That figure is measured in situ by the
cycle counter of §7.4, not inferred: the remaining 56.4% is attention, the KV
cache, RoPE, the norms, the activation quantisation, thread barriers and graph
overhead, all identical between the arms, so a 25% cut in weight bytes acts on
under half the token and 14% is the expected order.

The standalone replay of §7.3 puts the same matmuls at about
\SI{14}{\milli\second}, or 30%. That gap is the replay's, not the model's: it
drives
`bitnet_vec_dot_i2_i8_s_reference`, which §7.4 measures at $1.6\times$ the speed
of the path `llama.cpp` actually dispatches. The replay is a lower bound on what
the matmuls cost and was read as an estimate of it. Read as bandwidth, the
generation loop uses \SI{11.4}{\giga\byte\per\second} of the \SI{38.9}{} this
machine delivers: it is not at the memory wall either way.

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

### 7.6 Against `TQ1_0`, the nearest prior art

§1 records that the base-3 packing is `TQ1_0`'s, and that Spectra 1.1 publishes
the same $p=8$, $k=5$ construction with a correctness theorem. This section
measures against `TQ1_0`, which is the form of it that ships in the codebase
integrated into; no run against Spectra's own kernels was made. The comparison file is produced by re-encoding the same ternary values into
`TQ1_0`'s block layout, with the per-tensor `i2_s` scale written into every
block's f16 field, so the two files carry identical weights.

| | file | ternary payload | achieved bits/weight |
|---|---:|---:|---:|
| `i2_s` | 1,187,801,280 | 521,011,200 | 2.00000 |
| `TQ1_0` | 1,106,386,560 | 439,603,200 | 1.68750 |
| **t5b** | **1,085,565,120** | **418,775,040** | **1.60755** |

`TQ1_0`'s 1.6875 is $54$ bytes per 256 weights: 48 for the five-per-byte codes,
4 for a 16-element remainder at two bits, and 2 for the f16 block scale. t5b's
1.60755 is $32$ bytes per 160 weights plus the tail waste at $K=6912$ and one
32-byte record per tensor.

**On size the honest margin is small.** 4.7% of the ternary payload and 1.9% of
the file. Four fifths of the saving this work reports against `i2_s` was already
present in the vendored tree under an upstream type number; the remaining fifth
is what the per-tensor scale model contributes.

**On speed the margin is large and mostly not about the packing.** Three
rotated `llama-bench` invocations, all three files in each:

| | `pp512` (t/s) | `tg128` (t/s) |
|---|---|---|
| `i2_s` | 122.25 ± 1.52, 77.06 ± 4.49, 118.12 ± 3.26 | 23.61, 15.27, 23.56 |
| `TQ1_0` | 48.27 ± 0.73, 43.95 ± 5.26, 49.01 ± 0.29 | 21.68, 16.07, 22.27 |
| **t5b** | 130.65 ± 4.35, 137.38 ± 0.52, 128.17 ± 8.07 | 22.95, 26.43, 26.43 |

`TQ1_0` loses prompt processing by a factor of about three, and the reason is
structural rather than arithmetic: `i2_s` has a `llamafile_sgemm` case upstream,
t5b has one because we wrote it, and `TQ1_0` has none — so it processes a prompt
one activation column at a time. **A `TQ1_0` sgemm case, which nobody has
written, would be expected to erase that margin.** At `tg128`, where no format
has a batched path to exploit, all three land within about 10% and the ordering
does not survive this host's noise.

**What survives.** Against `TQ1_0` on this machine, t5b is worth 4.7% of the
ternary bytes, plus the fact of being wired into the batched path. The claim
that a sub-two-bit ternary packing is itself novel does not survive and is
withdrawn in §1. The narrower claims — the scale model that removes the block
overhead, the `i2_s`-compatible interleave, the independent-quotient decode, the
column-blocked kernel and the profitability condition — do.

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

Skipping the rows in the kernel saves 2.4380% of weight bytes and the same share
of multiply–accumulates, and is unconditionally exact. Removing the neurons
outright was also attempted, and the outcome is a negative result worth stating
because three prior claims about it — two by us — were wrong.

It is expressible: `llama.cpp` reads `LLM_KV_FEED_FORWARD_LENGTH` through
`get_key_or_arr` into a per-layer array, so the container and the
hyper-parameters carry per-layer widths. It is *not* free of loader changes:
`src/models/bitnet.cpp` allocates its four feed-forward tensors with the global
`n_ff`, so a file whose layers differ in width fails at load. Four lines change
that, and with a scalar `feed_forward_length` the broadcast makes them a no-op
for every existing checkpoint.

With those four lines the pruned model loads — 2.35 B parameters against 2.41 B,
1.08 GiB against 1.10, 60,948,480 ternary weights and 15,268,736 bytes removed.
Every surviving weight is bit-identical and every attention tensor is
byte-identical; the feed-forward block's int8 input is unchanged value for value
across all 1,642,496 of them. But it is **not** exact at the token level, and
the reason is the sub-layer RMS norm BitNet places between the element-wise
product and $W_{\text{down}}$. That norm divides by the root mean over all 6,912
entries, so deleting $d$ of them rescales the output by

$$
\sqrt{\frac{K-d}{K}},
$$

which in layer 1, where $d/K = 43.59\%$, is $0.751$ and not a rounding matter.
Pinning the divisor to the original width restores it to $2.5$ ulp of float32 —
not to zero. On three prompts the output is identical on one and diverges on two
— late, inside repetitive passages where successive candidates are near-ties and
the last bits decide. And it buys nothing measurable:
`pp512` $119.24 \pm 3.11 \to 109.90 \pm 9.48$, `tg128` $23.49 \pm 0.51 \to
23.53 \pm 0.72$.

The honest summary is that 2.9% of the ternary payload can be removed, that
doing so costs four lines and changes the output, and that it makes the model no
faster. We do not ship it.

---

# 9. Limitations

**The port analysis is Zen 2 specific — but $S$ is not.** §5.2 measures one
multiply-class port at 1.05 instructions per cycle against 3.0–4.0 for the cheap
classes. A reviewer ran the same benchmark on an Intel part with AVX-512 and
obtained 1.60 against 3.09: two 256-bit multiplier ports rather than one. The
ratio that inequality~(18) turns on is therefore a property of the
microarchitecture at a finer grain than the instruction set, and §5.3's surplus
table would have to be re-measured on any other part before (18) could be
applied to it.

The worry this raises is sharper than the port count alone, and it is worth
stating in its strongest form. Zen 2 cracks every 256-bit integer vector
operation into two 128-bit micro-operations issued over four floating-point
pipes, retiring roughly two vector operations per cycle; Zen 3 and later, and
Ice Lake and later, execute 256 bits natively. The packed kernel spends
$42/160 = 0.2625$ vector operations per weight against the baseline's
$17/128 = 0.1328$ — it is the kernel that issues *more* of them. A machine that
doubles the vector-op ceiling therefore relieves the packed format
preferentially, $S$ could collapse, and the entire motivation for a roofline
condition would be an artefact of one narrow part.

It does not collapse. §9.1 measures this against a calibrated static model: $S$
lands between 2.10 and 2.45 on Zen 3/4/5, Ice Lake and Sapphire Rapids, against
2.50 on Zen 2. Both kernels get roughly a third faster and the *ratio* stays
put.

**One microarchitecture, and the decisive property is absent from it.** All
results are from a single Zen 2 part reporting `avx2` and `fma` and nothing
else. `VPDPBUSD` collapses the contraction of §4.2 into one instruction while
leaving the decode of §4.1 unchanged; by (15) this raises $S$ and lowers
$\Sigma$, moving (18) against a packed format. The *Litespark* system, targeting
VNNI and ARM `SDOT`, stores ternary weights at eight bits for precisely this
reason. §9.1 *bounds* the size of that move at 0–14 % from a static model; §9.2
*measures* it at 8.8 % on hardware we do not own. **The central result is still
ours only for AVX2 without VNNI** — every end-to-end number in §7 comes from this
one host, and no VNNI machine has run the model. What §9.2 settles is the
kernel-level ratio, not the deployed one.

## 9.1 How far the model can substitute for the hardware

The two objections above differ in one respect that matters: the first concerns
code that exists and can be analysed, the second concerns code that does not.
We separate them.

`llvm-mca`, LLVM's static pipeline simulator, is run over the inner loops that
`gcc -O3 -mavx2 -mfma` actually emits for `ternary_t5b_dot_avx2` and
`bitnet_vec_dot_i2_i8_s_reference`, against the vendor-derived scheduler models
for six targets. Re-deriving the loop shapes reproduces the counts of §7.2
exactly for the baseline — 21 instructions, 4 multiply-class, 128 weights — and
to within one instruction for the packed kernel (46 against 47; the multiply
count of 13 and the 160 weights are exact), the difference being a compiler
version.

**The model is calibrated before it is used.** Its Zen 2 prediction is
$S = 2.501$ against the $S = 81.93/33.74 = 2.428$ that §7.2 measured on this
host: an error of $3.0\%$. `microarch_model.sh` exits non-zero if that
ever drifts past $8\%$. The absolute cycle counts are uniformly optimistic by
$6$–$11\%$ — 18.01 predicted against 19.7 measured per 160-weight block — which
is why only the ratio is carried forward.

| target | packed, cyc/block | baseline, cyc/block | $S$ |
|---|---:|---:|---:|
| `znver2` (this host) | 18.01 | 5.76 | 2.501 |
| `znver3` | 11.51 | 3.76 | 2.449 |
| `znver4` | 11.51 | 3.76 | 2.449 |
| `znver5` | 11.51 | 3.76 | 2.449 |
| `icelake-server` | 14.02 | 5.34 | 2.098 |
| `sapphirerapids` | 14.52 | 5.01 | 2.317 |

**These are modelled numbers, and the licence for printing them is that the
model was tested twice.** Once in-sample, against the host it was calibrated on
(3.0%); once *out of sample*, against a machine it had never seen — the reviewer
run reported below measured $S = 2.11$ on an Intel part, where this table
predicts 2.098 and 2.317 for the two Intel targets. A model that reproduces its
calibration set proves nothing; one that predicts a case withheld from it has
earned a table. Every figure here is nonetheless labelled modelled, and the
sections it feeds say so at each use.

Two readings of this table would overstate it. `znver3`, `znver4` and `znver5`
return *identical* cycle counts, so they are one prediction and not three:
LLVM's models for those three agree on this instruction mix, and the table
contains four distinct predictions, not six. And every newer target sits at or
below the Zen 2 value rather than scattered around it — the ratio improves by up
to 16% on a wider machine and never degrades — so the claim supported here is
one-sided (*$S$ does not run away*) and not a claim that $S$ is constant.

For the VNNI question the same apparatus is far weaker, and we set out why
rather than reporting the number alone. No compiler here targets VNNI usefully,
so the two VNNI loops are written by hand: in both kernels the contraction ends
in a `vpmaddubsw` followed by an accumulating `vpaddw`, and `VPDPBUSD` performs
exactly that pair fused, with the same unsigned-times-signed operand
convention. The substitution is argued sound — the reduction width changes from
`int16` pairs to `int32` quadruples, which leaves the horizontal sum invariant,
and it makes the accumulator fold of §4.2 unnecessary — but it is an argument,
not a test. **These loops have never been executed.**

Three further caveats bound what the resulting numbers mean.

1. `VPDPBUSD` accumulates in place with latency 10–13, so a loop carrying one
   accumulator per contraction is bound by that latency until it is unrolled.
   On `sapphirerapids` this makes the hand-written VNNI baseline *slower* than
   the AVX2 one it replaces — 7.01 cycles against 5.01 — which is a property of
   the accumulator count, not of VNNI. The VNNI comparison is therefore read at
   the resource bound, where an adequately unrolled implementation lands.
2. That metric fails the calibration the dependency-bound metric passed: on Zen
   2 it gives $S = 1.96$ against the measured 2.428, an error of $-19\%$,
   because the real loop is dependency-bound. Both are reported.
3. The negative control did not fire. `llvm-mca` 23.1.0 assembles and times
   `VPDPBUSD` for `-mcpu=znver2`, a part without the instruction, and continues
   to do so under `-mattr=-avx512vnni,-avxvnni`, assigning it latency 11 and
   throughput 1.00. The tool therefore cannot distinguish a legal VNNI sequence
   from an illegal one and offers no check on this subsection beyond timing.
   The control is kept in place and reported failing.

Subject to all three, VNNI moves $S$ in the predicted direction — against the
packed format, because the baseline spends 4 of 17 operations in the contraction
that `VPDPBUSD` collapses where the packed kernel spends 5 of 42 — and moves it
by $0$ to $14\%$: $S$ reaches 2.08 on `znver4` and 2.51 on `icelake-server` and
`sapphirerapids`. Modest, not decisive.

**None of this is a measurement.** It also cannot become one by recompiling: no
compiler contracts `vpmaddubsw` followed by an accumulating `vpaddw` into
`VPDPBUSD` as an idiom, so an AVX2 kernel built with `-march=native` on a VNNI
part contains **zero `vpdpbusd`** and measures AVX2 on that part. The
instruction has to be written.
`benchmarks/bench_vnni.c` now writes the instruction: `i2_s` and
t5b in both forms, the VNNI pair contracting into `int32` accumulators, which
for t5b also removes the fold of §4.2 entirely. It verifies with `objdump` at
run time that its own binary contains the instruction, checks every row against
exact `int64` arithmetic before printing a rate, and exits 3 rather than run on
a CPU without the feature. A second build models `VPDPBUSD` in AVX2 so the
algorithm can be checked where the instruction cannot execute; it passes on all
100 rows for all four kernels and deliberately prints no timings.

### 9.2 VNNI, measured

A reviewer ran that benchmark on an Intel Xeon reporting `avx512_vnni`: eight
invocations, the ISA check passing on each, all four kernels exact against
`int64` arithmetic on every row. **This is the paper's largest limitation moving
from modelled to measured**, and it moves only part of the way, so both parts are
stated.

| | AVX2 | VNNI | |
|---|---:|---:|---|
| median $S$ (`i2_s`/t5b) | 2.04 | **2.22** | $+8.8\%$ |
| `i2_s` gain from VNNI | — | — | $1.24\times$ |
| t5b gain from VNNI | — | — | $1.15\times$ |
| runs in which $S$ rose | — | 6 of 8 | |

**The direction is the predicted one and the mechanism is the predicted one.**
§9.1 argued that `VPDPBUSD` must help the baseline more than the packed format,
because the baseline spends 4 of its 17 vector operations in the contraction the
instruction collapses while the packed kernel spends 5 of 42. The measurement
agrees: $1.24\times$ against $1.15\times$, and $S$ rises accordingly.

**The size is modest and the spread is not smaller than the effect.** $S$ rose in
six runs of eight, which is a majority and not a clean separation; the host is a
shared two-vCPU sandbox, and the reviewer's own caveat — that run-to-run
variation there exceeds the effect being measured — is adopted rather than
argued away. An $8.8\%$ shift in $S$ against $\Sigma$ values that range from 1.35
to 3.08 does not by itself decide (18) either way at any thread count.

**What the static model got right, and where it was pessimistic.** §9.1 bounded
VNNI's cost at 0–14% and this measurement lands at 8.8%, inside that band. The
model predicted $S(\text{vnni}) = 2.08$ on `znver4` and 2.51 on the two Intel
targets; the measured 2.22 sits between them. Given that the model's negative
control failed and its VNNI loops had never been executed, agreement to this
degree is more than was claimed for it.

**What remains open is a clean machine, not the question.** Two vCPUs, no
pinning, shared tenancy. A dedicated part would turn six-of-eight into a
separation or refute it, and nothing here forecloses either. The claim this
subsection supports is narrow: on real VNNI hardware the packed format's
arithmetic disadvantage grows by roughly a tenth, in the direction and for the
reason §9.1 predicted, which is a smaller penalty than the eight-bit storage
choice of a system built for those targets would suggest.

### A second microarchitecture, measured

A reviewer ran the benchmarks on an Intel Xeon (AVX-512 capable, 2 vCPU cloud
sandbox), and the result is a genuine second data point for everything except
VNNI. Their caveats are adopted here rather than paraphrased: with two vCPUs and
no pinning, only the one- and two-thread rows carry information, and the
four-thread `bench_token` row and every `bench_threads` row from $T=3$ are
oversubscribed and mean nothing.

| | this host (Zen 2) | reviewer (Intel Xeon) |
|---|---:|---:|
| multiply class, ops/cycle | 1.05 | 1.32–1.44 |
| cheap class, ops/cycle | 3.0–4.0 | 2.97–3.04 |
| `i2_s` reference, GMAC/s/core | 81.93 | 49.86 |
| t5b, GMAC/s/core | 33.74 | 23.61 |
| **$S$** | **2.43** | **2.11** |
| $\Sigma$ at 1 thread | 1.35 | 1.65 |
| $\Sigma$ at 2 threads | 1.63 | 2.23 |

Two readings, and the second is the load-bearing one. First, $S$ does not
collapse on a wider machine: 2.11 against 2.43, two runs within 1% of each other.
Second, **this is an out-of-sample test of §9.1's model, and it passes.** That
model predicted $S = 2.098$ for `icelake-server` and 2.317 for
`sapphirerapids` — bracketing the measured 2.11 — from a calibration performed
only against this Zen 2 host. A static pipeline model tuned on one
microarchitecture predicted a second one it had never seen, to within 1%.

At two threads the reviewer's machine gives $S = 2.11 < \Sigma = 2.23$, so (18)
predicts a marginal win there and none at one thread. That is a prediction, not
a result: no end-to-end run was made on that host, and the elevated $\Sigma$
comes from the VM's low per-core DRAM bandwidth (\SI{8.1}{\giga\byte\per\second}
single-threaded) rather than from the microarchitecture.

**One model size, and one architecture family.** No public ternary checkpoint
larger than 2 B exists. The synthetic 60-layer graph of §7.4 doubles the weight
traffic but not the head count, hidden size or vocabulary.

**Measurement environment.** The host is shared and the run-to-run spread is
comparable to the effect: across eight invocations at ordinary load, one has the
denser format losing. The headline is therefore the five-invocation replication
of §7.4 rather than any single run.

**Verification scope.** The round trip is exact on all 210 tensors and all
2,084,044,800 weights, and so — since this revision — is the dot-product
comparison: 126,000 row-and-activation cases across every `i2_s` tensor, on which
the packed kernel matches exact `int64` arithmetic without exception. What
remains limited is the read-back from disk, which covers the 210 plus the
embedding rather than all 332 tensors; the converter's own reread covers the
dims, type, offset and alignment of all 332. Nine prompts and 20,480 perplexity
tokens are a large but finite sample.

Two things this does **not** establish, and they are the ones a reader should
hold against it. Nothing here runs the converted model: that the weights survive
repacking is not evidence that a loader reading them emits the same tokens, and
it cannot be until a loader knows type 43. And exhaustiveness over tensors is not
exhaustiveness over inputs — 200 rows per tensor under three activation patterns
is a large sample of a space that is not finite.

**Task benchmarks.** None, and for the format question none is possible: the
output is bit-identical, so every metric is identical by construction. Where
this *model* stands against others is a different question and this paper does
not address it. We ran a cross-model perplexity comparison and do not report it
as a result: the checkpoint's GGUF requires `tokenizer.ggml.pre` and the EOS id
to be supplied through `--override-kv` because its own metadata is wrong, so an
absolute perplexity from it is at least as likely to reflect a tokenisation
artefact as the model. It does not affect the identity result of §7.1, where
both arms are affected equally.

**Engineering.** The blocked matrix–matrix kernel spills 181 times per loop at
its chosen width; narrower widths spill less and measure slower. The type
identifier used (43) is not reserved by upstream, so a stock `llama.cpp` cannot
read a t5b file by construction rather than by accident. Neither the row skip
nor the neuron removal of §8 is implemented.

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
and §9.2 measures that narrowing at 8.8% — the direction and mechanism the
condition names, at a magnitude the eight-bit storage choice of a system built
for such parts would not have led one to expect. That is not a
caveat appended to a positive result; it is the result, which is that the choice
of weight density is a hardware-dependent optimisation with a computable
decision rule, rather than a property of the model. What §9.1 does settle is
that the rule's input is not parochial: across four distinct scheduler models
spanning two vendors and three generations, $S$ ranges from 2.10 to 2.50 and
every newer target sits at or *below* the Zen 2 value on which it was derived,
by at most 16%. The ratio never worsens on a wider machine, so the decision rule
is worth applying rather than an accident of the one part that produced it.

---

---

# Data and code availability

The format, the kernels, the benchmarks, the converters and every evidence file
cited here are public under the Apache License 2.0 at

> **https://github.com/purpleskulll/bitnet-t5b**

No model weights are distributed and no upstream source is redistributed
verbatim; the integration is applied as a patch to a checkout the user supplies,
and the files that reproduce or adapt an upstream contract say so in their own
headers and in `NOTICE`.

| claim | file |
|---|---|
| code histogram, code 3 absence, entropy | `results/tensor_structure.txt`, `results/real_weights_check.txt` |
| instruction-port throughput $\pi$ | `benchmarks/bench_ports.c` |
| surplus $\Sigma$ against core count | `results/thread_headroom.txt`, `benchmarks/bench_threads.c` |
| kernel rates, instruction and spill counts | `results/weight_density.txt`, `benchmarks/bench_alu.c` |
| weight traffic of one token | `benchmarks/bench_token.c` |
| end-to-end throughput | `results/inference_t5b.txt` |
| perplexity agreement between the formats | `results/perplexity_t5b.txt` |
| round trip on real weights | `results/real_weights_check.txt` |
| dead feed-forward neurons | `results/dead_neurons.txt`, `tools/dead_neurons.py` |

The format and its kernels are `src/ternary_t5b.{c,h}`, the `ggml`-facing
contract `src/ggml_t5b_glue.{c,h}`, the suite `src/test_t5b.c`; the word-lane
variant of §4.3 is `src/ternary_t10.{c,h}`; the converter is
`tools/gguf_to_t5b.py`.

**What the public repository does not contain.** The measurements that need a
model — §7.4's end-to-end table, the perplexity agreement — additionally require
a `llama.cpp` build carrying the type, a converted GGUF and the checkpoint
itself. The integration is supplied as a patch against a named upstream commit
rather than as copied sources, and the `i2_s` reference kernel used as the
baseline is fetched from that commit by a script rather than redistributed.
Everything else — the packing, the kernels, the suite, all four benchmarks, the
converters and every evidence file — builds and runs from a clone with a
compiler and Python alone.

# Appendix A: Corrections made during preparation

Seven claims in earlier drafts of this paper were wrong and were corrected
before submission. They are collected here rather than annotated through the
body, so that the body states what holds and this states how it got there. Each
was found by a named mechanism, and where that mechanism is now automated the
gate step is given.

1. **The base-3 packing was claimed as novel.** It is `TQ1_0`'s construction in
   `llama.cpp` and is published with a correctness theorem as Spectra 1.1's
   `TQ1`. Withdrawn in §1.1; §7.6 measures against it. Found by a reviewer,
   twice — first the pull request, then the paper.

2. **"The removal needs no format change and no kernel work" and
   "unconditionally exact on the product"** (dead-neuron removal). Both false:
   `src/models/bitnet.cpp` allocates with the global `n_ff`, four lines must
   change, and the sub-layer RMS norm rescales the output by
   $\sqrt{(K-d)/K} = 0.751$ in layer 1. Corrected in §8.

3. **Matmuls were given as 30% of a token** from the standalone replay, while
   the in-situ counter of §7.4 measures 43.6%. The replay is a lower bound and
   was read as an estimate.

4. **Upstream's `int16` overflow was reported as "200 of 200"** rows at
   $K = 6912$. Over all thirty such tensors it is 11,998 of 12,000: with 0.78%
   of headroom the outcome turns on each row's own code mean. Found by extending
   the dot-product check from seven tensors to all 210.

5. **"Four runs out of four"** at $1.165\times$ were four runs from the quietest
   window available; four further runs at ordinary load include one the denser
   format loses. The headline became the five-invocation replication.

6. **The `i2_s` matmul time was derived by difference** across two `llama-bench`
   runs on a host at load 2 to 10, which puts every fluctuation between them into
   that one number. Both arms are now probed at the same dispatch site. The
   stated reason for not instrumenting the control arm — that it must stay
   unperturbed — did not survive measurement: `benchmarks/bench_probe_cost.c`
   puts the probe at 0.18% of a matmul call.

7. **`second_datapoint.sh` claimed a run on a VNNI part would settle the VNNI
   question.** It would not have: no compiler emits `VPDPBUSD` from the AVX2
   idiom, so its binaries contained none. Found by a reviewer with `objdump`.
   `benchmarks/bench_vnni.c` writes the instruction and verifies at run time that
   its own binary contains it; §9.2 is the resulting measurement.

Three of the seven were found by a reviewer and four by checks written after the
fact. Those checks are in the repository and run as gate steps: the two
renderings of this paper are compared claim by claim, every path it names is
resolved against the public tree, and the integration's three representations
are checked against each other.


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
- compilade. *ggml-quants: ternary packing for TriLMs and BitNet b1.58.*
  llama.cpp pull request #8151, August 2024.
- Vaidhya, T., Kaushal, A., Jain, V., Couture-Harpin, F., Shishodia, P.,
  Behbahani, M., Nevmyvaka, Y., Rish, I. *Spectra 1.1: Scaling Laws and
  Efficient Inference for Ternary Language Models.* arXiv:2506.23025, 2025.
- Huang, H., Wu, D., Hu, Q., Yu, G., Yang, J., Zhu, J., Liu, X., Wu, D.
  *Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained
  Sparsification.* arXiv:2601.07892, 2026.
- *Litespark Inference on Consumer CPUs: Custom SIMD Kernels for Ternary Neural
  Networks.* arXiv:2605.06485.
- Han, S., Mao, H., Dally, W. J. *Deep Compression: Compressing Deep Neural
  Networks with Pruning, Trained Quantization and Huffman Coding.* ICLR, 2016.
  arXiv:1510.00149.
- Zhang, T. et al. *70% Size, 100% Accuracy: Lossless LLM Compression for
  Efficient GPU Inference via Dynamic-Length Float.* arXiv:2504.11651, 2025.
- Gerganov, G. et al. *llama.cpp / ggml.*
- Merity, S., Xiong, C., Bradbury, J., Socher, R. *Pointer Sentinel Mixture
  Models.* arXiv:1609.07843, 2016. (WikiText-2.)
