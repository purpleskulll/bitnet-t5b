/*
 * ternary_t10.h -- 1.6-bit ternary weight packing, and an AVX2 kernel that
 *                  computes on it WITHOUT ever materialising a weight.
 *
 * Why this exists
 * ---------------
 * At the real model's working set the ternary kernels are memory-bandwidth
 * bound, not ALU bound: results/bench_code_kernel.txt has upstream's i2_s
 * kernel and this project's fastest kernel both at 14.8 GB/s and 59.2-59.3
 * GMAC/s at 256 MiB, indistinguishable. Once two kernels saturate the same
 * bus, the only remaining lever is reading fewer weight BYTES.
 *
 * Ten ternary values carry 10 * log2(3) = 15.85 bits, and 3^10 = 59049 <=
 * 65536, so ten of them fit in one uint16: 1.6 bits per weight against the
 * 2.0 bits of llama.cpp's i2_s. 20 % fewer bytes for the same weights.
 *
 * MEASURED AND REFUTED, 2026-09-05. The packing works -- 1.6000 bits/weight
 * at n = 2560, 1.6250 at n = 4096 where one block is part-padded, against
 * i2_s's 2.0000 -- and it does not buy speed. On this machine, n = 4096,
 * single thread, minimum of 5 reps, against ternary_dot_a8_avx2_p2b_code on
 * identical weights in the same process:
 *
 *     working set            t10 (1.6 bit)      p2b CODE (2 bit)
 *     64 rows,     L2         29.31 GMAC/s        84.51 GMAC/s
 *     1024 rows,   L3         29.23               84.87
 *     8192 rows,   DRAM       28.49               81.37
 *     65536 rows,  DRAM       27.18               53.90
 *
 * Two things in that table. First, t10 is 2.0x to 2.9x SLOWER everywhere,
 * so 18.75 % fewer bytes did not convert into any time at all. Second, and
 * this is the diagnosis rather than the symptom, t10's number is FLAT from
 * L2 to DRAM while p2b's falls by 36 % -- p2b runs into the memory wall and
 * t10 never reaches it. t10 moves 5.5-5.9 GB/s of weight bytes where the
 * bus delivers ~13-15, so bandwidth was never its constraint and a cheaper
 * format cannot help it.
 *
 * What binds it is the multiply port. Decoding costs 9 vpmulhuw (the
 * quotient chain) plus 10 vpmaddwd (one per digit plane) per 160 weights =
 * 0.119 multiply-pipe ops per weight, where the maddubs path on 2-bit codes
 * needs 4 vpmaddubsw per 128 weights = 0.033, a factor of 3.6. Measured
 * against the clock: 35.3 GMAC/s at the best unroll is 18.6 cycles per
 * block at 4.1 GHz for 19 multiply-class ops -- 0.98 of them per cycle,
 * which is exactly the one-256-bit-integer-multiply-per-cycle roof of this
 * Zen 2 core. Unrolling cannot move it: U = 1 through 8 spans 28.0 to 35.3
 * GMAC/s with no knee. It is a roof, not a tuning problem.
 *
 * So the entry in the ledger is: below 2 bits per weight is REACHABLE and,
 * by this route, not PROFITABLE on AVX2. A format that costs a base-3
 * division per digit trades the whole bandwidth saving for arithmetic and
 * then some. Anything that beats i2_s from here has to decode with the
 * shift-and-mask class of instruction, which is to say it has to keep the
 * digit boundaries on power-of-two bit positions -- which is the property
 * 1.6-bit packing gives up by construction. Kept, built and tested, because
 * a refuted approach with numbers attached is worth more than an untried
 * one: it closes the "just pack it tighter" question with a measurement
 * instead of leaving it open as a plausible next idea.
 *
 * Layout ("t10")
 * -------------
 * code = w + 1, so code in {0,1,2} for w in {-1,0,+1}.
 *
 * A BLOCK is 16 uint16 lanes = 32 bytes and carries 160 weights. For lane
 * j (0..15) and digit k (0..9):
 *
 *     lane_j = sum_{k=0}^{9} code(w[16*k + j]) * 3^k          (0 .. 59048)
 *
 * so digit plane k lines up with the 16 CONSECUTIVE weights at block offset
 * 16k .. 16k+15. Block m covers weights 160m .. 160m+159. Lanes are stored
 * little-endian, explicitly, by both the packer and the scalar decoder, so
 * the scalar path does not silently depend on the host being x86.
 *
 * The stride-16 interleave is the same idea as the p2b layout's stride-32
 * one (ternary_avx2.h): it makes each digit plane a single aligned 16-lane
 * vector of the activation array, with no shuffle and no per-lane shift.
 *
 * Computing without decoding
 * --------------------------
 * Let x_j = floor(x / 3^j), so x_0 = x, x_10 = 0, and code_k = x_k - 3*x_{k+1}.
 * Substituting and re-indexing telescopes:
 *
 *     sum_{k=0}^{9} code_k * a_k  =  x_0*b_0 + sum_{j=1}^{9} x_j*b_j
 *     where  b_0 = a_0  and  b_j = a_j - 3*a_{j-1}   for j >= 1
 *
 * The kernel therefore never computes a digit. It needs the QUOTIENT CHAIN
 * x_0 .. x_9 -- nine successive divisions by three, each one
 * _mm256_srli_epi16(_mm256_mulhi_epu16(x, 43691), 1), exact for the whole
 * uint16 range and checked as such over all 65536 inputs by the test -- and
 * a transformed activation vector b, which depends only on the activations
 * and is therefore computed ONCE per GEMV rather than once per row.
 *
 * The identity yields sum(code*a). The caller wants sum(w*a), which is
 * sum(code*a) - sum(a); `sum_a` is that per-GEMV constant, exactly as
 * ternary_dot_a8_avx2_p2b_code takes it.
 *
 * SIGNED INTERMEDIATES AND DELIBERATE WRAPAROUND. _mm256_madd_epi16 is
 * signed and x_0 reaches 59048, which is not an int16. The kernel biases it,
 * x_0' = x_0 XOR 0x8000 (that is, x_0 - 32768 read as two's complement), and
 * adds back 32768 * sum(b_0) at the end; x_j for j >= 1 is at most 19682 and
 * needs no bias. Those intermediates are large and they CANCEL: the
 * algorithm is exact modulo 2^32, and the true dot product is bounded by
 * 127*n so it fits int32 comfortably, but the running sums do not and they
 * WRAP. That is correct, not a bug. _mm256_add_epi32 wraps by definition;
 * every scalar accumulation of these intermediates is done in uint32_t,
 * where wraparound is defined, and only the final value is cast to int32_t,
 * because signed overflow in C is undefined behaviour and -fwrapv is not in
 * this project's CFLAGS. A test constructs a case that provably wraps -- the
 * untruncated int64 accumulator exceeds 2^31 -- and pins the right answer.
 *
 * PRECONDITIONS. As in ternary_avx2.h, each of these is a place where two
 * paths of this file disagree, or where both agree on something that is not
 * the dot product, so no amount of differential testing finds them:
 *
 *   weights in {-1, 0, +1} for ternary_t10_pack. A value of 2 encodes as
 *     code 3, which CARRIES: it does not merely corrupt its own weight, it
 *     changes the weight 16 slots later, in the next digit plane. Measured:
 *     w[0] = 2 with every other weight 0 unpacks as w[0] = -1 AND w[16] =
 *     +1. Use ternary_t10_pack_checked() to reject the input instead; it is
 *     the same function with the range test, and it returns -1.
 *
 *   lane values <= 59048, i.e. buffers this packer produced. The telescoping
 *     drops the term -3 * x_10 * a_9 because x_10 = 0, which holds for every
 *     packed lane but not for an arbitrary uint16. A foreign lane of 0xFFFF
 *     has x_10 = 1, and then the AVX2 kernel and the scalar decoder return
 *     DIFFERENT answers for the same bytes -- both of them wrong, in
 *     different ways. Pinned by a test rather than assumed.
 *
 *   `b` comes from ternary_t10_prep_b() for the SAME n, and holds
 *     ternary_t10_slots(n) int16, not n. The kernel reads whole blocks, so
 *     for n = 4096 it reads 4160 int16: a buffer sized to n is 128 bytes
 *     short and the kernel runs off the end of it.
 *
 *   activations beyond n contribute NOTHING. The tail block is padded with
 *     code 1 (w = 0), and a padded slot's contribution to the identity is
 *     1 * a_pad - a_pad = 0 only because a_pad is zero. prep_b enforces this
 *     itself: it never reads a[i] for i >= n and writes zero-derived b there,
 *     so a caller cannot get it wrong by allocating exactly n activations.
 *     What it protects against is real -- a test writes a nonzero b into the
 *     pad region by hand and the answer moves by exactly that amount -- and
 *     `sum_a` must likewise be the sum over the first n activations only.
 *
 *   int8 activations in [-127, 127], as everywhere else in this project.
 *     Unlike the kernels in ternary_avx2.h this path never calls
 *     _mm256_sign_epi8, so -128 does not negate to itself here; the bound
 *     that matters is instead that |b_j| = |a_j - 3*a_{j-1}| <= 4*127 = 508
 *     stays an int16, which -128 would also satisfy. The range is stated to
 *     keep one convention across the project, not because this file breaks.
 *
 * All arithmetic is exact integer. No rounding, no float, no FMA.
 */

#ifndef TERNARY_T10_H
#define TERNARY_T10_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Geometry: 16 uint16 lanes x 10 base-3 digits = 160 weights in 32 bytes. */
#define TERNARY_T10_BLOCK    160
#define TERNARY_T10_BYTES     32
#define TERNARY_T10_LANES     16
#define TERNARY_T10_DIGITS    10

/* Packed bytes for `n` weights, rounded up to a whole block. */
size_t ternary_t10_size(size_t n);

/* Number of int16 slots ternary_t10_prep_b() writes and the kernel reads:
 * blocks * 160, which is >= n. See the third precondition above. */
size_t ternary_t10_slots(size_t n);

/* Pack `n` weights into `dst`, which must hold ternary_t10_size(n) bytes.
 * PRECONDITION: w[i] in {-1,0,+1}; out of range CARRIES into the neighbouring
 * digit plane. Use the _checked form to reject instead. */
void ternary_t10_pack(const int8_t *w, uint8_t *dst, size_t n);

/* Same, but returns -1 and packs nothing if any weight is out of range.
 * Fails loud rather than clamping, as ternary_pack2b does. */
int  ternary_t10_pack_checked(const int8_t *w, uint8_t *dst, size_t n);

/* Decode one weight, by digit extraction. Ground truth for the round trip. */
int8_t ternary_t10_unpack_at(const uint8_t *wp, size_t i);

/* Scalar reference dot product. Derived from the ENCODING definition -- it
 * divides and takes remainders to recover each digit -- and not from the
 * telescoping identity, so that the test compares two genuinely different
 * derivations rather than one derivation against itself. Reads a[0..n-1]
 * only, and needs no sum_a because it works on the weights directly. */
int32_t ternary_t10_dot_scalar(const uint8_t *wp, const int8_t *a, size_t n);

/* The b transform, once per GEMV: b[i] = a[i] for the first 16 slots of each
 * block, b[i] = a[i] - 3*a[i-16] elsewhere. Writes ternary_t10_slots(n)
 * int16; reads a[0..n-1] and treats everything past n as zero. */
void ternary_t10_prep_b(const int8_t *a, int16_t *b, size_t n);

/* AVX2 kernel. Returns sum over i < n of w[i]*a[i].
 *
 * SIGNATURE NOTE: the brief's shape for this was (wp, a, n, sum_a), taking
 * the raw activations. It takes the PREPARED b instead, deliberately. The b
 * transform is the same 4n byte pass over the activations for every row of a
 * GEMV, and recomputing it per row would cost more than the whole packing
 * saves -- it is exactly the "once per GEMV rather than per row" discipline
 * that sum_a already follows. Passing b also removes the scratch buffer the
 * other shape would have needed and makes the precondition on the pad region
 * checkable by the caller. sum_a stays a separate argument rather than being
 * folded into b so that it keeps the same meaning it has in ternary_avx2.h.
 *
 * The 32768 * sum(b_0) bias correction is NOT a parameter: the kernel
 * accumulates sum(b_0) in a vector register as it goes. That is ~5 % more
 * ALU in a loop whose cost is elsewhere, and it keeps a term that overflows
 * int32 for n > ~8k out of the public interface entirely. */
int32_t ternary_t10_dot_avx2(const uint8_t *wp, const int16_t *b,
                             size_t n, int32_t sum_a);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_T10_H */
