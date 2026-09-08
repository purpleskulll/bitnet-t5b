/*
 * ternary_t5b.h -- 1.6-bit ternary weight packing, second attempt: five trits
 *                  per BYTE, contracted in BYTE lanes.
 *
 * Why a second 1.6-bit packing
 * ---------------------------
 * ternary_t10.h holds the first one and the measurement that refuted it: t10
 * is 2.0x to 2.9x slower than the 2-bit i2_s kernel everywhere, and its
 * throughput is FLAT from L2 to DRAM, which says it never reaches the memory
 * wall a cheaper format would help it climb. What binds it is the multiply
 * port. On Zen 2 vpmulhuw, vpmaddwd and vpmaddubsw all issue there, one 256-bit
 * op per cycle, so the figure that decides these kernels is MACs per
 * multiplier-port instruction:
 *
 *     upstream i2_s   4 vpmaddubsw per 128 weights    = 32.0 MACs/mul-op
 *     t10            19 (9 vpmulhuw + 10 vpmaddwd)
 *                        per 160 weights              =  8.4 MACs/mul-op
 *
 * t10 pays twice over. It spends nine multiplier ops dividing before it
 * multiplies anything useful, and it contracts with vpmaddwd, which covers 16
 * lanes where vpmaddubsw covers 32. This layout attacks both terms:
 *
 *     t5b             8 vpmulhuw + 5 vpmaddubsw
 *                        per 160 weights              = 12.3 MACs/mul-op
 *
 * a 1.47x improvement on the resource that was measured to be binding, and
 * the compiler delivers it. Counted in the emitted object rather than assumed:
 *
 *     objdump -d build/obj/ternary_t5b.o \
 *         | grep -cE 'vpmulhuw|vpmaddwd|vpmaddubsw|vpmullw'
 *
 * the steady-state inner loop is 67 instructions for one block, of which
 * exactly 8 vpmulhuw + 5 vpmaddubsw = 13 are multiplier-port ops. The 12-block
 * fold adds one vpmaddwd, giving 13.08 per 160 weights, against t10's measured
 * 19.25 (its loop does four blocks per iteration: 77 multiplier ops per 640
 * weights, the 77th being its bias-correction madd).
 *
 * THIS IS A CLAIM ABOUT ONE RESOURCE AND NOT ABOUT WALL-CLOCK TIME, and the
 * same disassembly says why. Per 160 weights:
 *
 *                       multiplier-port     other vector ALU     loads
 *     t10                     19.25                21            8.75
 *     t5b                     13.08                41            9
 *
 * t5b buys back 6.2 multiplier ops with 20 extra shift/sub/or/add ops. Those
 * issue on the wider ports, so each is individually cheaper -- but at roughly
 * three vector-integer ops per cycle, 41 of them is about 13.7 cycles against
 * 13 cycles of multiplier work. The two resources are now BALANCED, which
 * means the multiplier port may no longer be the thing that binds. So the
 * honest statement of what this design does is: it removes the roof t10 was
 * measured to be sitting on, and puts a different one at approximately the
 * same height. Whether that is a net win is a timing question, and this file
 * does not answer it -- it establishes correctness and the instruction count,
 * and leaves the wall clock to the benchmark.
 *
 * Layout ("t5b")
 * -------------
 * code = w + 1, so code in {0,1,2} for w in {-1,0,+1}.
 *
 * Five trits carry 5 * log2(3) = 7.92 bits and 3^5 = 243 <= 256, so five of
 * them fit in ONE BYTE: 1.6 bits per weight, the same density as t10 and 20 %
 * below i2_s's 2.0. A BLOCK is 32 bytes carrying 160 weights. For byte j
 * (0..31) and digit k (0..4):
 *
 *     byte_j = sum_{k=0}^{4} code(w[32*k + j]) * 3^k            (0 .. 242)
 *
 * so digit plane k lines up with the 32 CONSECUTIVE weights at block offset
 * 32k .. 32k+31, and block m covers weights 160m .. 160m+159. That stride-32
 * interleave is llama.cpp's own i2_s interleave (ggml_i2s_ternary.h: bits 6-7
 * of byte j hold weight 0+j, bits 4-5 hold weight 32+j, ...) extended from
 * four power-of-two fields to five base-3 digits. One difference worth
 * naming: i2_s puts the EARLIEST weight in the most significant field, this
 * puts it in the LEAST significant digit. Nothing depends on the order -- it
 * is the stride that matters, because it is what makes each digit plane a
 * plain contiguous 32-byte slice of the activation array.
 *
 * There is no endianness question here, unlike t10: a block is 32 independent
 * bytes, not 16 multi-byte lanes.
 *
 * Decoding, per 32-byte block
 * ---------------------------
 * 1. Two int16 views of the same register, no permute and no shuffle:
 *        even = x & 0x00FF          (lane L = byte 2L)
 *        odd  = x >> 8              (lane L = byte 2L+1)
 * 2. All four quotients come straight from the ORIGINAL value, one
 *    _mm256_mulhi_epu16 each, with no chain between them -- four independent
 *    5-cycle multiplies per view instead of a four-deep dependency, which is
 *    where the instruction-level parallelism comes from:
 *        x1 = mulhi(x, 21846)   floor(x/3)
 *        x2 = mulhi(x,  7282)   floor(x/9)
 *        x3 = mulhi(x,  2428)   floor(x/27)
 *        x4 = mulhi(x,   811)   floor(x/81)
 *    Each is exact only up to a bound, and the bounds are not the same one.
 *    Measured exhaustively over the whole uint16 range (test section (a)):
 *
 *        21846 / 3    exact 0..32767,  first wrong at x = 32768   135x margin
 *         7282 / 9    exact 0..32767,  first wrong at x = 32768   135x margin
 *         2428 / 27   exact 0..3292,   first wrong at x = 3293     13.6x
 *          811 / 81   exact 0..484,    first wrong at x = 485       2.0x
 *
 *    A packed byte never exceeds 242, so all four hold with room -- but the
 *    room is 2.0x, not "obviously fine", which is why it is on record. (810
 *    would divide correctly to 889 and give 3.7x; 811 is what this file uses
 *    and what the test pins.) The margin is the reason the <= 242 precondition
 *    below is a precondition and not a stylistic note.
 * 3. Digits, with no further multiplies -- 3*y is (y<<1)+y:
 *        d4 = x4;  d3 = x3 - 3*x4;  d2 = x2 - 3*x3;  d1 = x1 - 3*x2;
 *        d0 = x - 3*x1
 * 4. Recombine each plane's two views into one 32-byte register:
 *        plane_k = even_k | (odd_k << 8)
 * 5. Contract exactly as upstream i2_s does, digit unsigned, activation signed:
 *        acc16 += _mm256_maddubs_epi16(plane_k, a + 32k)
 *
 * NO ACTIVATION TRANSFORM. This kernel reads the raw int8 activations. t10
 * could not: its telescoping identity needs a prepared b vector, a separate
 * 4n-byte pass per GEMV plus a scratch buffer the caller has to size and
 * zero-pad correctly. That whole apparatus is gone here, and it is a real
 * advantage of contracting on digits rather than on quotients -- one fewer
 * buffer, one fewer precondition, and one fewer pass over the activations.
 *
 * The int16 accumulator, and the fold
 * -----------------------------------
 * This is the one place where a wrong answer could hide silently, so the
 * bound is worst-case and not typical-case.
 *
 * vpmaddubsw multiplies unsigned by signed and adds ADJACENT PAIRS, so one
 * int16 lane receives two products per plane. A product is at most
 * 2 * 128 = 256 in absolute value (code <= 2; |a| <= 128, taking the full int8
 * range rather than the [-127,127] convention, so the bound survives a -128),
 * hence at most 512 per plane and 5 * 512 = 2560 per block. Therefore:
 *
 *        12 blocks * 2560 = 30720 <= 32767      TERNARY_T5B_FOLD = 12
 *        13 blocks * 2560 = 33280 >  32767      would overflow
 *
 * The kernel folds the int16 accumulator into int32 with
 * _mm256_madd_epi16(acc16, ones) after every 12 blocks, and never lets more
 * than 12 accumulate. It deliberately does NOT copy upstream's 32-block fold:
 * upstream's codes are centred and its partial sums cancel, and this one's do
 * not -- every product of an all-+1 row against an all-+127 activation vector
 * has the same sign. The saturation in vpmaddubsw never engages either, for
 * the same reason: 512 is nowhere near the int16 clamp.
 *
 * Note that 2560 is the bound for bytes this packer produced, and that the
 * `code <= 2` in it is a consequence of the <= 242 precondition rather than a
 * property of a byte. Exactly 13 byte values decode outside {0,1,2}: 243..255
 * all give d4 = 3, because floor(x/81) = 3 there. The per-plane maxima over the
 * whole 256-byte domain are (2,2,2,2,3), so a block of foreign bytes bounds a
 * lane at 2 * 128 * (2+2+2+2+3) = 2816, and 12 * 2816 = 33792 > 32767. The
 * fold would overflow. This is one more consequence of the <= 242 precondition
 * rather than a separate hazard.
 *
 * (An earlier revision put that figure at 2794, which is the same count taken
 * at |a| <= 127 while the derivation above takes 128. The conclusion is
 * unchanged -- 12 * 2794 = 33528 also exceeds 32767 -- but the two numbers were
 * computed under different conventions eighteen lines apart.)
 *
 * Correction
 * ----------
 * The contraction yields sum(code*a). The caller wants sum(w*a), which is
 * sum(code*a) - sum(a); `sum_a` is that per-GEMV constant, exactly as
 * ternary_dot_a8_avx2_p2b_code and ternary_t10_dot_avx2 take it.
 *
 * PRECONDITIONS. As in ternary_avx2.h and ternary_t10.h, each of these is a
 * place where two paths of this file disagree, or where both agree on
 * something that is not the dot product, so no amount of differential testing
 * on well-formed inputs finds them:
 *
 *   weights in {-1, 0, +1} for ternary_t5b_pack. A value of 2 encodes as
 *     code 3, which CARRIES: it does not merely corrupt its own weight, it
 *     changes the weight 32 slots later, in the next digit plane. Measured:
 *     w[0] = 2 with every other weight 0 packs byte 0 as 123 instead of 121
 *     and unpacks as w[0] = -1 AND w[32] = +1, with w[64] untouched. Use
 *     ternary_t5b_pack_checked() to reject the input instead; it is the same
 *     function with the range test, and it returns -1.
 *
 *   byte values <= 242, i.e. buffers this packer produced. A foreign byte is
 *     not merely mis-decoded, it is decoded DIFFERENTLY BY THE TWO PATHS: for
 *     0xFF the scalar reference computes digit 4 as (255/81) % 3 = 0, while
 *     the kernel's x4 = mulhi(255, 811) = 3 is used as-is, with no mod, so the
 *     kernel runs 3*a[128+j] high. Both answers are wrong; they are wrong by a
 *     predictable difference, which is why the bound is written down and
 *     pinned by a test rather than assumed away.
 *
 *   activations: the kernel reads a[0 .. n-1] and NOTHING ELSE. n need not be
 *     a multiple of 160; a partial final block is copied into a zeroed 160-byte
 *     stack buffer, so a caller may allocate exactly n bytes. (ternary_t10.h
 *     documents the other discipline -- the caller supplies a padded buffer --
 *     because its prep_b pass had to write one anyway. With no transform to
 *     piggyback on, a bounds requirement that only bites on non-multiples of
 *     160 is a footgun for no gain, so it is handled internally. K = 6912 in
 *     the real model is 43 whole blocks plus 32 weights, so this path is live,
 *     not theoretical.) Padded weight slots carry code 1, whose contribution
 *     is 1 * 0 = 0 against the zeroed activation.
 *
 *   `sum_a` must be the sum over the FIRST n activations only, matching the n
 *     passed to the kernel. It is the caller's one remaining obligation.
 *
 *   int8 activations in [-127, 127], as everywhere else in this project. This
 *     path calls no _mm256_sign_epi8, so unlike the kernels in ternary_avx2.h
 *     a -128 does not negate to itself and nothing here actually breaks -- the
 *     accumulator bound above is stated at |a| <= 128 precisely so that it
 *     covers the case. The range is stated to keep one convention across the
 *     project.
 *
 * All arithmetic is exact integer. No rounding, no float, no FMA.
 */

#ifndef TERNARY_T5B_H
#define TERNARY_T5B_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Geometry: 32 bytes x 5 base-3 digits = 160 weights in 32 bytes. */
#define TERNARY_T5B_BLOCK    160
#define TERNARY_T5B_BYTES     32
#define TERNARY_T5B_DIGITS     5

/* Blocks accumulated in the int16 accumulator before folding to int32. The
 * worst case is 2560 per block, so 12 * 2560 = 30720 fits and 13 would not;
 * see the derivation above. Public so a test can size a case to sit exactly
 * on the boundary instead of hoping a random one lands there. */
#define TERNARY_T5B_FOLD      12

/* Packed bytes for `n` weights, rounded up to a whole block. */
size_t ternary_t5b_size(size_t n);

/* Weight slots those bytes cover: blocks * 160, which is >= n. The slots from
 * n upward hold code 1 (weight 0). Nothing in the API requires the caller to
 * know this; it exists so the tests can check the padding. */
size_t ternary_t5b_slots(size_t n);

/* Pack `n` weights into `dst`, which must hold ternary_t5b_size(n) bytes.
 * PRECONDITION: w[i] in {-1,0,+1}; out of range CARRIES into the neighbouring
 * digit plane. Use the _checked form to reject instead. */
void ternary_t5b_pack(const int8_t *w, uint8_t *dst, size_t n);

/* Same, but returns -1 and packs nothing if any weight is out of range.
 * Fails loud rather than clamping, as ternary_pack2b does. */
int  ternary_t5b_pack_checked(const int8_t *w, uint8_t *dst, size_t n);

/* Decode one weight, by digit extraction. Ground truth for the round trip. */
int8_t ternary_t5b_unpack_at(const uint8_t *wp, size_t i);

/* Scalar reference dot product. Derived from the ENCODING definition -- plain
 * / and % to recover each digit -- and not from the quotient identities the
 * kernel uses, so the test compares two genuinely different derivations rather
 * than one derivation against itself. Reads a[0..n-1] only, and needs no
 * sum_a because it works on the weights directly. */
int32_t ternary_t5b_dot_scalar(const uint8_t *wp, const int8_t *a, size_t n);

/* AVX2 kernel. Returns sum over i < n of w[i]*a[i].
 *
 * Takes the RAW activations -- there is no prepared vector and no scratch
 * buffer, which is the interface-level payoff of contracting on digits rather
 * than on quotients. Compare ternary_t10_dot_avx2, which needs a b vector of
 * ternary_t10_slots(n) int16 built by a separate pass. */
int32_t ternary_t5b_dot_avx2(const uint8_t *wp, const int8_t *a,
                             size_t n, int32_t sum_a);

/* GEMV, optionally row-tiled: `rows` consecutive rows of ternary_t5b_size(n)
 * bytes each against ONE activation vector. Writes y[r] = the same value
 * ternary_t5b_dot_avx2 returns for row r. `sum_a` is the sum over the first n
 * activations, as for the per-row kernel, and is one constant shared by every
 * row rather than one the caller re-passes per call.
 *
 * THE TILING IS OFF BY DEFAULT, AND THE MEASUREMENT THAT TURNED IT OFF
 * --------------------------------------------------------------------
 * TERNARY_T5B_TILE defaults to 1: this entry point does NOT interleave rows. It
 * is still worth having, for reasons that have nothing to do with tiling -- see
 * the last section. The row-tiled widths are compiled in and reachable with
 * -DTERNARY_T5B_TILE=2..4, and they are off because tiling a GEMV was measured
 * to be a LOSS at the only footprint a real model has.
 *
 * L2-resident, 100 rows of n = 2560, all five in ONE process with the reps
 * interleaved so a clock drift cannot favour whichever ran first (taskset -c 5,
 * best of five, median of three passes):
 *
 *     tile   insn/block   VECTOR ops/block   stack mov   GMAC/s/core
 *      per-row kernel                                       33.30    <- control
 *      1        47.00          43.00             0          34.43     1.034x
 *      2        45.00          43.00             0          34.80     1.045x
 *      3        44.67          43.33             1          35.43     1.064x
 *      4        44.25          43.25             2          34.70     1.042x
 *
 * DRAM-resident, the SAME kernels, 256 MiB of weights at n = 2560 (a single core
 * on this part sees 16 MiB of L3, so this is cold), median of seven passes:
 *
 *     tile      GB/s     GMAC/s      relative
 *      per-row  6.14      30.71       1.000x   <- control
 *      1        6.34      31.68       1.032x
 *      2        4.76      23.79       0.775x
 *      3        4.30      21.52       0.701x
 *      4        4.41      22.04       0.717x
 *
 * The shape that is fastest in cache is the slowest in DRAM, and the crossover
 * is not marginal: width 3 goes from the best of the five to 30 % BELOW the
 * kernel it was meant to beat. Tiling T rows means T interleaved weight streams
 * at a 512-byte stride instead of one linear stream, and this part delivers
 * 4.3 GB/s to that pattern against 6.1 to a linear one. The per-row kernel needs
 * 6.1 to keep its arithmetic fed, so the reordering turns an arithmetic-bound
 * loop into a bandwidth-bound one. Not a threshold effect, and not a fluke of
 * one size -- width 3 measures 0.691x at 128 MiB, 0.701x at 256 and 0.701x at
 * 512, and at the real model's K = 6912 the three widths land at 0.676x, 0.752x
 * and 0.775x.
 *
 * WHY A GEMV CANNOT WIN THIS TRADE, which is the part worth keeping
 * ----------------------------------------------------------------
 * Upstream draws the same line and draws it structurally. In
 * ggml/src/ggml-cpu/ggml-cpu-i2s.c, ggml_gemv_i2_i8_s -- the path a single
 * generated token takes -- calls ggml_vec_dot_i2_i8_s row by row with no row
 * interleaving at all, while the 4x4 tile lives in ggml_gemm_i2_i8_s, the
 * many-token path. That is not a tuning accident. A GEMM tile of 4 rows by 4
 * activation columns loads each weight byte once and spends it on FOUR
 * contractions: real reuse, which pays for any amount of access-pattern damage.
 * A GEMV has ONE activation column, so tiling rows buys NO reuse whatsoever --
 * every weight byte is still loaded once and used once. It only reorders the
 * loads. There is nothing on the credit side of that trade except instruction
 * scheduling, and the numbers above are what the debit side costs.
 *
 * The cache-resident benchmark cannot see any of this, which is the methodological
 * point: bench_alu.c is L2-resident by design, and on its evidence alone the
 * right answer here is width 3 and a 6 % win.
 *
 * WHY THE ENTRY POINT SURVIVES ANYWAY
 * -----------------------------------
 * At width 1 this is a single linear weight stream -- byte for byte the access
 * pattern of a loop over ternary_t5b_dot_avx2 -- and it is faster than that loop
 * at BOTH footprints: 1.034x in L2 and 1.032x in DRAM. Nothing about tiling is
 * involved. The gain is that the five loop-invariant decode constants and the
 * zero-padded tail block are hoisted out of the per-row call: the per-row kernel
 * rebuilds that 160-byte buffer with a memset and a memcpy for EVERY row, and at
 * K = 6912 -- 43 whole blocks plus 32 weights -- the real model takes that path
 * on every row of every GEMV.
 *
 * WHAT THE L2 TABLE STILL SAYS ABOUT THE LOOP
 * -------------------------------------------
 * The VECTOR op count per block does not move: 43 at every width, because tiling
 * changes how many independent decodes are in flight and not what a decode
 * costs. What shrinks is the SCALAR overhead -- two pointer increments and the
 * loop compare -- amortised over more rows, 47 instructions per block down to
 * 44.25. In cache the speed tracks that instruction count and not the added
 * parallelism: 47/44.67 predicts 1.052 and width 3 delivers 1.064. The loop is
 * issue-bound, not latency-bound; it already runs at ~2.2 vector ops per cycle,
 * the Zen 2 roof for 256-bit integer ops, so there is no idle slot for a second
 * row's independent work to occupy. That agrees with the per-row kernel's own
 * header from the other direction -- "the kernel is bound by the TOTAL vector op
 * count, and the only thing that moves it is removing ops" -- and row tiling
 * removes none.
 *
 * The register cliff that was expected never arrived, for the record: GCC places
 * three rows with one stack reference per block loop and four rows with two, so
 * the 44-spill collapse an earlier version of this file hit (24.36 GMAC/s) is
 * not what caps the width. Bandwidth is.
 *
 * The activation sharing that motivates upstream's GEMM tile is worth nothing
 * here for a third reason: t5b_contract folds the activation load into
 * vpmaddubsw as a memory operand, so there is no load instruction to amortise --
 * hoisting the five per block into registers would cost five of the sixteen to
 * save instructions the front end was not short of.
 *
 * Rows not a multiple of the tile are handled by SMALLER TILES, not by falling
 * back to the per-row kernel: a remainder of two runs the width-2 body and a
 * remainder of one the width-1 body, which is the per-row loop with the same
 * scalar fold. rows = 0 writes nothing.
 *
 * The tail block, when n is not a multiple of 160, is copied into a zeroed
 * 160-byte buffer ONCE PER CALL rather than once per row -- the per-row kernel
 * has to redo that memset and memcpy for every row, and at K = 6912 (43 whole
 * blocks plus 32 weights) the real model takes this path on every GEMV. */
void ternary_t5b_gemv_avx2(const uint8_t *W, const int8_t *a, int32_t *y,
                           size_t rows, size_t n, int32_t sum_a);

/* GEMM: `rows` weight rows against `cols` activation columns.
 *
 *     W    packed weights, row r at W + r*ternary_t5b_size(n)
 *     B    activations, column c at B + c*ldb, n bytes read and no more
 *     C    output, C[c*ldc + r] = sum over i < n of code(w[r][i]) * B[c][i]
 *
 * NOTE THE CONVENTION, which differs from the other two entry points in this
 * file: this one returns sum(CODE*a), not sum(w*a), and takes no sum_a to
 * convert it. That is what ggml's post-processing wants -- it subtracts the
 * activation row sum itself, at ggml-cpu.c:1244 and sgemm.cpp:1582 -- and
 * ggml_t5b_glue.h section 2 has the derivation. A caller that wants sum(w*a)
 * subtracts sum(B[c][0..n-1]) from C[c*ldc + r] for itself; the per-column
 * constant is not something a GEMM should be handed once per call and then
 * applied rows*cols times.
 *
 * WHY THIS IS NOT A LOOP OVER THE GEMV, which is what it replaces
 * ---------------------------------------------------------------
 * The GEMV decodes a weight block and spends it on ONE activation column. Run
 * once per column, that re-runs the whole decode -- eight vpmulhuw, the digit
 * subtractions, the plane recombination -- for every column, which for i2_s
 * would be three shifts and four masks and for t5b is the entire cost of the
 * kernel. This decodes each block once and contracts it against
 * TERNARY_T5B_GEMM_NC columns held in registers, turning
 *
 *     NC * (decode + 5 contractions)     into     decode + NC * 5 contractions
 *
 * per 32-byte block. The source has the op counts, the register budget that
 * caps NC at 8, and the measured table.
 *
 * This is the PROMPT-PROCESSING path and it is the only place the format was
 * losing badly: 41.3-65.7 t/s against i2_s's 91.9-114.7 in the real model
 * (results/inference_t5b.txt section 3), where generation -- one column, nothing
 * to amortise -- was already 1.057x at four threads.
 *
 * Rows are NOT interleaved and deliberately so; the GEMV section above has the
 * DRAM measurement that rules row tiling out, and blocking columns is the one
 * axis on which the reuse is real and the weight stream stays linear.
 *
 * n need not be a multiple of 160; each column's partial final block is copied
 * into a zeroed 160-byte buffer, once per strip rather than once per row.
 * rows = 0 or cols = 0 writes nothing; n = 0 still writes all rows*cols entries
 * of C, as zeros, because a caller that sized C from rows and cols is entitled
 * to find it filled. */
void ternary_t5b_gemm_avx2(const uint8_t *W, const int8_t *B, size_t ldb,
                           int32_t *C, size_t ldc,
                           size_t rows, size_t cols, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_T5B_H */
