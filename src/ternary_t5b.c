/*
 * ternary_t5b.c -- see ternary_t5b.h for the layout contract, the exactness
 * bounds of the four magic constants, the fold derivation, and the five
 * preconditions.
 *
 * Build: gcc -O3 -mavx2 -mfma -march=native -c ternary_t5b.c
 */

#include "ternary_t5b.h"

#include <immintrin.h>
#include <string.h>

/* 3^0 .. 3^4. The bound a packed byte never reaches is 3^5 = 243, which is
 * also why five digits fit in a byte at all. */
static const uint32_t T5B_POW3[TERNARY_T5B_DIGITS] = { 1u, 3u, 9u, 27u, 81u };

/* ====================================================================== */
/* Geometry                                                               */
/* ====================================================================== */

static inline size_t t5b_blocks(size_t n)
{
    return (n + TERNARY_T5B_BLOCK - 1) / TERNARY_T5B_BLOCK;
}

size_t ternary_t5b_size(size_t n)
{
    return t5b_blocks(n) * TERNARY_T5B_BYTES;
}

size_t ternary_t5b_slots(size_t n)
{
    return t5b_blocks(n) * TERNARY_T5B_BLOCK;
}

/* ====================================================================== */
/* Packing                                                                */
/* ====================================================================== */

void ternary_t5b_pack(const int8_t *w, uint8_t *dst, size_t n)
{
    const size_t blocks = t5b_blocks(n);

    for (size_t m = 0; m < blocks; ++m) {
        for (size_t j = 0; j < TERNARY_T5B_BYTES; ++j) {
            uint32_t byte = 0;

            for (size_t k = 0; k < TERNARY_T5B_DIGITS; ++k) {
                const size_t i = m * TERNARY_T5B_BLOCK
                               + k * TERNARY_T5B_BYTES + j;
                /* Slots past n are padded with code 1, which is weight 0.
                 * They contribute code*a - a = 0 only because the kernel
                 * makes the matching activation zero; see the header. */
                const uint32_t code = (i < n) ? (uint32_t)(w[i] + 1) : 1u;
                byte += code * T5B_POW3[k];
            }

            /* Maximum 2 * (1+3+9+27+81) = 242, so this never truncates for a
             * well-formed input -- and for a malformed one it carries into
             * the next digit rather than truncating, which is exactly the
             * first precondition. */
            dst[m * TERNARY_T5B_BYTES + j] = (uint8_t)byte;
        }
    }
}

int ternary_t5b_pack_checked(const int8_t *w, uint8_t *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        if (w[i] < -1 || w[i] > 1) {
            return -1;                   /* fail loud, never silently clamp */
        }
    }
    ternary_t5b_pack(w, dst, n);
    return 0;
}

int8_t ternary_t5b_unpack_at(const uint8_t *wp, size_t i)
{
    const size_t m = i / TERNARY_T5B_BLOCK;
    const size_t p = i % TERNARY_T5B_BLOCK;
    const size_t k = p / TERNARY_T5B_BYTES;
    const size_t j = p % TERNARY_T5B_BYTES;

    const uint32_t byte = wp[m * TERNARY_T5B_BYTES + j];
    const int32_t  code = (int32_t)((byte / T5B_POW3[k]) % 3u);
    return (int8_t)(code - 1);
}

/* ====================================================================== */
/* Scalar reference                                                       */
/* ====================================================================== */

/* Written from the ENCODING, by division and remainder, and deliberately not
 * from the quotient identities the kernel uses: the point of the differential
 * test is that two independent derivations agree, and a reference sharing the
 * kernel's algebra would only prove the compiler works. In particular this
 * takes `% 3` where the kernel takes the raw mulhi result, which is what makes
 * the two disagree on a foreign byte -- a difference the header documents and
 * a test pins. */
int32_t ternary_t5b_dot_scalar(const uint8_t *wp, const int8_t *a, size_t n)
{
    const size_t blocks = t5b_blocks(n);
    uint32_t s = 0;                      /* unsigned: wraps, defined */

    for (size_t m = 0; m < blocks; ++m) {
        for (size_t j = 0; j < TERNARY_T5B_BYTES; ++j) {
            const uint32_t byte = wp[m * TERNARY_T5B_BYTES + j];

            for (size_t k = 0; k < TERNARY_T5B_DIGITS; ++k) {
                const size_t i = m * TERNARY_T5B_BLOCK
                               + k * TERNARY_T5B_BYTES + j;
                if (i >= n) {
                    continue;            /* padding: never read past a[n-1] */
                }
                const int32_t code = (int32_t)((byte / T5B_POW3[k]) % 3u);
                s += (uint32_t)((code - 1) * (int32_t)a[i]);
            }
        }
    }
    return (int32_t)s;
}

/* ====================================================================== */
/* AVX2 kernel                                                            */
/* ====================================================================== */

static inline int32_t t5b_hsum_epi32(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* 3*y in BYTE lanes, written with int16 ops. Never _mm256_mullo_epi16: that
 * would issue on the multiplier port, and spending multiplier ops is the one
 * thing this layout exists to avoid. vpsllw and vpaddw issue elsewhere.
 *
 * AVX2 has no byte shift, so this uses the int16 pair -- which computes the
 * byte-wise triple only if neither byte carries into the next. It never does.
 * A quotient byte is mulhi(b, magic) for b <= 255, and the largest any of the
 * four reaches is x1 = mulhi(255, 21846) = 85 (exhaustively over all 256 byte
 * values, not just the <= 242 a packer emits):
 *
 *        y  <= 85     the shift needs 2*85 = 170 <= 255
 *        3*y <= 255   the add   needs           255 <= 255
 *
 * so the low byte's result never reaches bit 8 and the high byte is untouched.
 * The second bound is TIGHT -- exactly 255, with zero margin -- which is why it
 * is measured and written down rather than waved at. It holds for a foreign
 * byte too, so this function does not change what the kernel returns for one. */
static inline __m256i t5b_triple(__m256i y)
{
    return _mm256_add_epi16(_mm256_slli_epi16(y, 1), y);
}

/* One digit plane against its 32 activations, accumulated. Split out so each
 * plane can be contracted the moment it exists: a plane is dead as soon as its
 * vpmaddubsw has read it, and forming all five before contracting any would
 * keep five of them live at once for no reason.
 *
 * Digit unsigned, activation signed -- the operand order matters. The
 * saturating add inside vpmaddubsw cannot engage: two products of at most
 * 2*128 sum to 512, far from the int16 clamp. */
static inline __m256i t5b_contract(__m256i acc, __m256i plane, const int8_t *a)
{
    const __m256i act = _mm256_loadu_si256((const __m256i *)a);
    return _mm256_add_epi16(acc, _mm256_maddubs_epi16(plane, act));
}

/* One 32-byte block: 160 weights against 160 activations, five vpmaddubsw
 * accumulated into an int16 register. Eight vpmulhuw and five vpmaddubsw --
 * thirteen multiplier-port ops for 160 MACs. Everything else here is shift,
 * logic and add.
 *
 * WHAT ACTUALLY BINDS THIS LOOP, measured rather than assumed. The multiplier
 * port is not it: thirteen ops at the 1.05 per cycle this host issues is a
 * 12.4-cycle floor per block, and the block takes 19.7. Counting the emitted
 * fold-loop body instead (objdump over the steady-state loop, not the whole
 * function) gives 43 vector ops, and 43 / 19.7 = 2.19 vector ops per cycle.
 * That IS the ceiling: this is a Zen 2 part, every 256-bit integer vector op
 * splits into two 128-bit uops over four FP pipes, so ~2 per cycle is the
 * hardware limit and the small excess is the activation load folded into
 * vpmaddubsw as a memory operand. The kernel is bound by the TOTAL vector op
 * count, and the only thing that moves it is removing ops.
 *
 * That is why this file is written the way it is, and the alternatives are on
 * record because each one was built and timed (GMAC/s/core, taskset -c 5):
 *
 *     int16 digit chain, recombine per plane     63 vec ops    24.7
 *     byte-packed quotients (this file)          43 vec ops    33.7
 *     ...as above, two blocks per iteration      43 vec ops    33.3
 *     ...as above, low digit contracted first    43 vec ops    33.6
 *     ...as above, 3*y as y+y+y not (y<<1)+y     43 vec ops    33.3
 *
 * Every variant that left the op count alone left the speed alone; the one
 * that removed 20 moved it 37 %. Register pressure was the mechanism -- the
 * old form kept x, two views, eight quotients and five planes live against
 * sixteen ymm registers and spilled, 44 stack-touching vector moves in the
 * function against 9 now -- but the spills mattered because they were extra
 * ops, not because of the memory traffic.
 *
 * `a` must be readable for 160 bytes; the caller arranges that for the tail. */
static inline __m256i t5b_block(__m256i acc, const uint8_t *wp, const int8_t *a)
{
    /* Loop-invariant; the compiler hoists these out of the caller's loop. */
    const __m256i lo8 = _mm256_set1_epi16(0x00FF);
    const __m256i m1  = _mm256_set1_epi16(21846);   /* floor(x/3),  exact <= 32767 */
    const __m256i m2  = _mm256_set1_epi16(7282);    /* floor(x/9),  exact <= 32767 */
    const __m256i m3  = _mm256_set1_epi16(2428);    /* floor(x/27), exact <= 3292  */
    const __m256i m4  = _mm256_set1_epi16(811);     /* floor(x/81), exact <= 484   */

    const __m256i x = _mm256_loadu_si256((const __m256i *)wp);

    /* Two int16 views of the same 32 bytes. No permute, no shuffle: the even
     * bytes are already in the low half of each int16 lane and the odd bytes
     * are already in the high half. */
    const __m256i xe = _mm256_and_si256(x, lo8);
    const __m256i xo = _mm256_srli_epi16(x, 8);

    /* Four INDEPENDENT quotients, each straight from the original value rather
     * than from the previous quotient. Same instruction count as a
     * divide-by-three chain, one eighth the dependency depth.
     *
     * The two views are re-interleaved into BYTE lanes here, at the quotient,
     * rather than later at each of the five digit planes. The header derives
     * the digits in int16 lanes and recombines afterwards; recombining first
     * is the same arithmetic, because x_k -> (x_k^even | x_k^odd << 8) is a
     * relabelling that commutes with a subtraction no byte can carry out of
     * (proved for t5b_triple above, and for the subtraction below). Doing it
     * in this order pays the two-op interleave FOUR times, once per quotient,
     * instead of FIVE times, once per plane -- and, far more to the point, it
     * kills e_k and o_k one instruction after they are born instead of holding
     * eight of them live at once. Sixteen ymm registers do not hold x, xe, xo,
     * eight quotients, five planes and an accumulator; they hold this. */
    const __m256i q1 = _mm256_or_si256(
                           _mm256_mulhi_epu16(xe, m1),
                           _mm256_slli_epi16(_mm256_mulhi_epu16(xo, m1), 8));
    const __m256i q2 = _mm256_or_si256(
                           _mm256_mulhi_epu16(xe, m2),
                           _mm256_slli_epi16(_mm256_mulhi_epu16(xo, m2), 8));
    const __m256i q3 = _mm256_or_si256(
                           _mm256_mulhi_epu16(xe, m3),
                           _mm256_slli_epi16(_mm256_mulhi_epu16(xo, m3), 8));
    const __m256i q4 = _mm256_or_si256(
                           _mm256_mulhi_epu16(xe, m4),
                           _mm256_slli_epi16(_mm256_mulhi_epu16(xo, m4), 8));

    /* d_k = x_k - 3*x_{k+1}, with x_5 = 0 so d4 = x4, now on bytes. vpsubb and
     * not vpsubw: no digit subtraction ever borrows, for any of the 256
     * possible byte values -- verified exhaustively, including the foreign
     * bytes above 242 the header's second precondition covers -- so no lane
     * carries out of its byte and the byte-width subtract makes that
     * structural rather than incidental.
     *
     * NOT "every digit is in 0..2 for all 256 bytes", which an earlier revision
     * of this comment claimed and which is false: 243..255 give d4 = 3, since
     * floor(x/81) = 3 there. The header says so correctly and this line
     * contradicted it. The distinction is load-bearing rather than pedantic,
     * because the digit range is exactly what the accumulator bound rests on --
     * 2 * 128 * 2 = 512 per plane -- so `d <= 2` is a consequence of the <= 242
     * precondition, not something the decode guarantees on its own. Borrow-
     * freedom IS unconditional over the whole domain, and it is all vpsubb
     * needs.
     *
     * Emitted high digit first: q4 is read by planes 4 and 3 and is dead after
     * the second, q3 by planes 3 and 2, and so on, so each quotient dies one
     * step after its last use and at most four are ever live. Each plane is
     * contracted the instant it exists and never survives to the next. */
    acc = t5b_contract(acc, q4, a + 4 * TERNARY_T5B_BYTES);
    acc = t5b_contract(acc, _mm256_sub_epi8(q3, t5b_triple(q4)),
                       a + 3 * TERNARY_T5B_BYTES);
    acc = t5b_contract(acc, _mm256_sub_epi8(q2, t5b_triple(q3)),
                       a + 2 * TERNARY_T5B_BYTES);
    acc = t5b_contract(acc, _mm256_sub_epi8(q1, t5b_triple(q2)),
                       a + 1 * TERNARY_T5B_BYTES);
    acc = t5b_contract(acc, _mm256_sub_epi8(x,  t5b_triple(q1)),
                       a + 0 * TERNARY_T5B_BYTES);
    return acc;
}

int32_t ternary_t5b_dot_avx2(const uint8_t *wp, const int8_t *a,
                             size_t n, int32_t sum_a)
{
    const __m256i ones = _mm256_set1_epi16(1);
    const size_t  full = n / TERNARY_T5B_BLOCK;   /* blocks read straight from `a` */
    const size_t  rem  = n % TERNARY_T5B_BLOCK;   /* weights in the partial block   */

    __m256i acc32 = _mm256_setzero_si256();
    size_t  m     = 0;

    /* Whole groups of TERNARY_T5B_FOLD blocks, folded to int32 at the end of
     * each. The bound is worst-case, not typical: 2560 per block (see the
     * header), 12 * 2560 = 30720 <= 32767, and 13 would not fit. Upstream's
     * 32-block fold relies on cancellation this kernel does not have. */
    for (; m + TERNARY_T5B_FOLD <= full; m += TERNARY_T5B_FOLD) {
        /* One block per iteration. Two blocks into two independent
         * accumulators was measured and is slightly SLOWER (33.30 against
         * 33.73 GMAC/s/core), which says the loop is not waiting on the
         * accumulator chain and that a wider body only costs front-end. Note
         * that `#pragma GCC unroll` cannot express either choice here: at 1, 2
         * or absent the compiler emits a byte-identical object (checked with
         * md5sum over the disassembly), so the factor lives in the source. */
        __m256i acc16 = _mm256_setzero_si256();
        for (size_t t = 0; t < TERNARY_T5B_FOLD; ++t) {
            acc16 = t5b_block(acc16,
                              wp + (m + t) * TERNARY_T5B_BYTES,
                              a  + (m + t) * TERNARY_T5B_BLOCK);
        }
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(acc16, ones));
    }

    /* The remainder: at most FOLD-1 = 11 whole blocks plus at most one partial
     * block, so at most 12 -- the same bound, one fold. */
    {
        __m256i acc16 = _mm256_setzero_si256();

        for (; m < full; ++m) {
            acc16 = t5b_block(acc16, wp + m * TERNARY_T5B_BYTES,
                                     a  + m * TERNARY_T5B_BLOCK);
        }

        if (rem != 0) {
            /* The kernel reads whole 160-activation blocks, and the caller's
             * buffer ends at n. Copy the live part into a ZEROED block instead
             * of requiring the caller to over-allocate and zero-pad: the padded
             * weight slots hold code 1, and 1 * 0 = 0, so the padding
             * contributes nothing. One <=160-byte copy per GEMV row, against
             * 160 MACs per block -- and it removes an out-of-bounds read that
             * would otherwise fire on every n that is not a multiple of 160,
             * which the real model's K = 6912 is not. */
            int8_t pad[TERNARY_T5B_BLOCK];
            memset(pad, 0, sizeof pad);
            memcpy(pad, a + full * TERNARY_T5B_BLOCK, rem);
            acc16 = t5b_block(acc16, wp + full * TERNARY_T5B_BYTES, pad);
        }

        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(acc16, ones));
    }

    /* acc32 holds sum(code*a) over the whole packed region, which is
     * sum(code*a) over i < n because the padded slots contributed zero. It is
     * bounded by 2*128*n, so it is exact in int32 for any n below 8.4 million;
     * uint32_t is used for the final step only because signed overflow is
     * undefined in C and -fwrapv is not in this project's CFLAGS. */
    const uint32_t s = (uint32_t)t5b_hsum_epi32(acc32);

    /* sum(code*a) - sum(a) = sum(w*a). */
    return (int32_t)(s - (uint32_t)sum_a);
}

/* ====================================================================== */
/* Row-tiled GEMV                                                         */
/* ====================================================================== */

/* Rows interleaved against one activation vector per pass. ONE by default,
 * i.e. no interleaving: widths 2 to 4 win in L2 and lose 23 % to 30 % once the
 * weights are DRAM-resident, which is the only footprint a real model has. The
 * header holds both tables and the reason a GEMV cannot win that trade. The
 * widths stay compiled in and reachable with -DTERNARY_T5B_TILE=2..4 -- they are
 * the right shape for a many-token GEMM, where the reuse that pays for the
 * access pattern actually exists, and for a weight matrix small enough to stay
 * in cache. */
#ifndef TERNARY_T5B_TILE
#define TERNARY_T5B_TILE 1
#endif

/* The widest tile any instantiation below asks for, and the size of the two
 * small local arrays. Kept separate from TERNARY_T5B_TILE so that overriding
 * the tile does not silently resize the arrays out from under the width-2 and
 * width-1 remainder bodies. */
#define T5B_TILE_MAX 4

#if TERNARY_T5B_TILE < 1 || TERNARY_T5B_TILE > T5B_TILE_MAX
#error "TERNARY_T5B_TILE must be between 1 and 4"
#endif

/* One tile of `rm` rows. `rm` is a literal at every call site and this is
 * always_inline, so the row loops below are straight-line code and the rm
 * accumulators are registers rather than a stack array -- checked in the
 * disassembly, not assumed. At the default rm = 1 the twelve-block loop is 47
 * instructions with 13 multiplier-port ops and no stack reference, the same
 * body ternary_t5b_dot_avx2 emits. At rm = 3 it is 134 instructions with 39 and
 * ONE stack reference, against 3 * 47 = 141 for three passes of the width-1
 * loop: fewer instructions for the same 43 vector ops per block, which is why
 * it wins in cache and why that win does not survive contact with DRAM.
 *
 * `tail` is the caller's zero-padded copy of the final partial block, built
 * once per GEMV and shared by every row of every tile.
 *
 * `rowbytes` is ternary_t5b_size(n), the stride from one row to the next. */
static inline __attribute__((always_inline)) void
t5b_tile(const uint8_t *W, size_t rowbytes, const int8_t *a, const int8_t *tail,
         int32_t *y, size_t n, int32_t sum_a, const size_t rm)
{
    const __m256i ones = _mm256_set1_epi16(1);
    const size_t  full = n / TERNARY_T5B_BLOCK;
    const size_t  rem  = n % TERNARY_T5B_BLOCK;

    /* The int32 accumulator lives in a GENERAL-PURPOSE register here, where the
     * per-row kernel keeps it in a ymm. That kernel can afford to: it has one.
     * rm of them would cost rm of the sixteen vector registers this decode
     * already fills, and a register is worth more inside the block loop than it
     * is holding a value touched once per twelve blocks. Folding to a scalar
     * costs one extra hsum per row per fold -- five instructions against the
     * fold's 12 * 47 -- and returns the identical sum, because the reassociation
     * is over wrapping unsigned addition, which is associative. */
    uint32_t part[T5B_TILE_MAX];
    for (size_t r = 0; r < rm; ++r) {
        part[r] = 0;
    }

    size_t m = 0;

    /* Whole groups of TERNARY_T5B_FOLD blocks. The int16 bound is per row and
     * unchanged by tiling: each row still accumulates at most 12 * 2560 = 30720
     * before its own fold, because the rows share activations, not accumulators. */
    for (; m + TERNARY_T5B_FOLD <= full; m += TERNARY_T5B_FOLD) {
        __m256i acc16[T5B_TILE_MAX];
        for (size_t r = 0; r < rm; ++r) {
            acc16[r] = _mm256_setzero_si256();
        }

        for (size_t t = 0; t < TERNARY_T5B_FOLD; ++t) {
            /* One block index, rm rows. The five activation addresses are
             * common to all rm of them; they stay memory operands of
             * vpmaddubsw rather than being hoisted into registers, which would
             * cost five of the sixteen to save instructions the front end is
             * not short of. */
            const int8_t  *ab = a + (m + t) * TERNARY_T5B_BLOCK;
            const uint8_t *wb = W + (m + t) * TERNARY_T5B_BYTES;
            for (size_t r = 0; r < rm; ++r) {
                acc16[r] = t5b_block(acc16[r], wb + r * rowbytes, ab);
            }
        }

        for (size_t r = 0; r < rm; ++r) {
            part[r] += (uint32_t)t5b_hsum_epi32(
                           _mm256_madd_epi16(acc16[r], ones));
        }
    }

    /* The remainder: at most FOLD-1 = 11 whole blocks plus at most one partial
     * block, so at most 12 -- the same bound, one fold. */
    {
        __m256i acc16[T5B_TILE_MAX];
        for (size_t r = 0; r < rm; ++r) {
            acc16[r] = _mm256_setzero_si256();
        }

        for (; m < full; ++m) {
            const int8_t  *ab = a + m * TERNARY_T5B_BLOCK;
            const uint8_t *wb = W + m * TERNARY_T5B_BYTES;
            for (size_t r = 0; r < rm; ++r) {
                acc16[r] = t5b_block(acc16[r], wb + r * rowbytes, ab);
            }
        }

        if (rem != 0) {
            /* `tail` is already zero-padded. The per-row kernel rebuilds this
             * buffer for every row; here every row of every tile reads the one
             * the caller built. */
            const uint8_t *wb = W + full * TERNARY_T5B_BYTES;
            for (size_t r = 0; r < rm; ++r) {
                acc16[r] = t5b_block(acc16[r], wb + r * rowbytes, tail);
            }
        }

        for (size_t r = 0; r < rm; ++r) {
            part[r] += (uint32_t)t5b_hsum_epi32(
                           _mm256_madd_epi16(acc16[r], ones));
        }
    }

    for (size_t r = 0; r < rm; ++r) {
        y[r] = (int32_t)(part[r] - (uint32_t)sum_a);
    }
}

void ternary_t5b_gemv_avx2(const uint8_t *W, const int8_t *a, int32_t *y,
                           size_t rows, size_t n, int32_t sum_a)
{
    const size_t rowbytes = ternary_t5b_size(n);
    const size_t rem      = n % TERNARY_T5B_BLOCK;

    /* Built ONCE for the whole call. The per-row kernel pays this memset and
     * memcpy for EVERY row, and at the real model's K = 6912 it takes that path
     * on every row of every GEMV. This is the larger half of why the entry point
     * is worth having at width 1, where no rows are interleaved at all. Always
     * zeroed, even when rem is 0, so that no path can hand t5b_block an
     * uninitialised buffer -- 160 bytes per GEMV against rows * n MACs. */
    int8_t tail[TERNARY_T5B_BLOCK];
    memset(tail, 0, sizeof tail);
    if (rem != 0) {
        memcpy(tail, a + (n - rem), rem);
    }

    size_t r = 0;

    for (; r + TERNARY_T5B_TILE <= rows; r += TERNARY_T5B_TILE) {
        t5b_tile(W + r * rowbytes, rowbytes, a, tail, y + r, n, sum_a,
                 TERNARY_T5B_TILE);
    }

    /* Fewer than TERNARY_T5B_TILE rows left, so none at the default width and at
     * most three at width 4. Smaller tiles rather than a call back into
     * ternary_t5b_dot_avx2: the width-1 body IS that kernel's loop, and routing
     * the remainder through the same code keeps one tail-block discipline in the
     * file instead of two. Compiled out entirely at width 1, where the loop
     * above has already taken every row. */
#if TERNARY_T5B_TILE > 2
    for (; r + 2 <= rows; r += 2) {
        t5b_tile(W + r * rowbytes, rowbytes, a, tail, y + r, n, sum_a, 2);
    }
#endif
#if TERNARY_T5B_TILE > 1
    if (r < rows) {
        t5b_tile(W + r * rowbytes, rowbytes, a, tail, y + r, n, sum_a, 1);
    }
#endif
    (void)r;
}

/* ====================================================================== */
/* Column-blocked GEMM                                                    */
/* ====================================================================== */

/* Activation columns contracted against one decode. The width is a compile-time
 * constant because it decides the register allocation, and overriding it is how
 * the widths in the table below were measured against each other: this file
 * compiled three times with -DTERNARY_T5B_GEMM_NC=2, 4, 8 and every public
 * symbol renamed, linked into ONE process with the reps interleaved so that a
 * clock change could not favour whichever ran first. */
#ifndef TERNARY_T5B_GEMM_NC
#define TERNARY_T5B_GEMM_NC 8
#endif

/* The widest strip any instantiation below asks for, and the size of the local
 * arrays. Kept separate from TERNARY_T5B_GEMM_NC for the reason T5B_TILE_MAX is
 * kept separate from TERNARY_T5B_TILE. */
#define T5B_GEMM_NC_MAX 8

#if TERNARY_T5B_GEMM_NC < 1 || TERNARY_T5B_GEMM_NC > T5B_GEMM_NC_MAX
#error "TERNARY_T5B_GEMM_NC must be between 1 and 8"
#endif

/* Weight rows kept resident across the column-strip loop: 32 rows is
 * 32*1408 = 44 KB at K = 6912, inside one core's L2 alongside the activation
 * strip. Matches the chunk ggml_t5b_glue.c already hands this kernel. */
#define T5B_GEMM_ROWS 32

/* Activation columns contracted against ONE decode. This is the only place in
 * the file where a decoded weight is used more than once, and it is the reason
 * the GEMM is not just a loop over the GEMV.
 *
 * WHAT THE GEMV LOOP COSTS, which is what this exists to remove. Per 32-byte
 * block the decode is 33 vector ops and each activation column adds 10 -- five
 * vpmaddubsw and the five vpaddw that accumulate them. Calling the GEMV once
 * per column pays the whole 43 for every column; contracting NC columns against
 * one decode pays 33 + 10*NC for all of them. That is the design, and the
 * EMITTED code is close to it but not equal to it, because the accumulators
 * start spilling. Counted in the object, steady-state block loop only
 * (loopcount over ternary_t5b_gemm_avx2, the loop with 5*NC vpmaddubsw):
 *
 *     NC   insn  vector ops  mul-port  stack ymm moves   design  MACs/vec op
 *      1     47      43         13            0            43       3.72
 *      2     59      55         18            1            53       5.82
 *      4     88      83         28           11            73       7.71
 *      8    155     148         48           36           113       8.65
 *
 * (NC = 1 is ternary_t5b_gemv_avx2 itself, the control, and its 43 is the
 * number this file's other header derives.)
 *
 * MEASURED, single thread, taskset -c 5, best of five, DRAM-resident weights
 * (a 192 MiB pool against 16 MiB of L3 per CCX, every pass reading a copy the
 * previous pass did not touch), GMAC/s, load average 3.7 -- higher than one
 * would choose, so the arms were interleaved inside one process and the run
 * was repeated three times, agreeing to within 1.5 %:
 *
 *     shape                    GEMV loop    NC=2    NC=4    NC=8   NC8/loop
 *     K=2560 rows=2560 c=16      34.13      56.81   67.60   80.03    2.34x
 *     K=2560 rows=2560 c=256     33.88      56.58   67.88   79.58    2.35x
 *     K=6912 rows=6912 c=16      33.94      57.24   68.44   80.44    2.37x
 *     K=6912 rows=6912 c=256     33.55      56.34   66.84   77.48    2.31x
 *     any shape,       c=1       32.4-33.0   ~34     ~34    34.6-34.9 1.06x
 *
 * 2.31x to 2.37x, and it tracks the TOTAL vector op count and not the
 * multiplier port: 8.65/3.72 = 2.32x predicted from the table above against
 * 2.31-2.37x delivered. The multiplier-port model -- the figure the FORMAT was
 * designed around -- predicts 2.17x (12.3 MACs per multiplier op at NC=1
 * against 26.7 at NC=8, i2_s being 32) and is the less accurate of the two.
 * Both are on record because they were the two candidate explanations.
 *
 * Nothing here is bandwidth-bound the way the row-tiled GEMV was: one linear
 * weight stream is preserved, because the COLUMNS are blocked and the rows are
 * not. That is also why the rate barely moves between c=16 and c=256, and why
 * the control sits at its L2 rate (33.5) rather than its DRAM rate (30.7) --
 * the row chunk is reused across the column loop in both arms.
 *
 * THE REGISTER BUDGET, which is what caps NC, and the two attempts to lift it.
 * Sixteen ymm. Live across a block: NC accumulators, x, xe, xo, two quotients
 * and one plane -- NC + 6 -- plus the magic constants. At NC = 8 that is 14
 * before a single constant, and GCC spills the accumulators to the stack: 36
 * stack-touching vector moves per block, the same failure mode as the 44-spill
 * collapse this file hit before (24.36 GMAC/s against 33.74). TWO restructurings
 * were built and timed to remove them, and NEITHER paid:
 *
 *   a divide-by-3 CHAIN (q_{k+1} = floor(q_k/3)) needs two magic constants
 *     instead of five, freeing three registers -- but it re-splits each quotient
 *     into byte halves, and the extra ops cost more than the spills it removed:
 *     152 vector ops per block at NC=8 against 148, spills only 36 -> 33.
 *
 *   a TWO-PASS group, decoding twelve blocks into a 1920-byte L1 buffer with no
 *     accumulator live, then contracting from it with nc + 1 registers: zero
 *     spills in the decode pass, and 78.4 GMAC/s against 80.0. No better in the
 *     middle and clearly WORSE at one column (30.6 against 34.7), because the
 *     plane buffer's stores and reloads replace the spills one for one.
 *
 * The conclusion those two support is negative and worth stating as such:
 * removing the spilled accumulator traffic buys nothing measurable, and the
 * 8-wide strip that spills 36 times per block beats every narrower one that
 * does not. Whether that is because the spills are free or because every way of
 * removing them costs what they cost is NOT settled by these two runs.
 *
 * QUOTIENTS ARE COMPUTED LAZILY HERE, high digit first, which t5b_block does
 * not need to do. It contracts each plane the instant it exists and so never
 * holds more than four quotients; this one holds NC accumulators as well, so it
 * defers each vpmulhuw pair to the plane that needs it and keeps at most TWO
 * quotients live. Same instructions, same order of contractions, four fewer
 * registers occupied at the peak. */
static inline __m256i t5b_quotient(__m256i xe, __m256i xo, __m256i magic)
{
    return _mm256_or_si256(
               _mm256_mulhi_epu16(xe, magic),
               _mm256_slli_epi16(_mm256_mulhi_epu16(xo, magic), 8));
}

/* One digit plane against NC activation columns. `nc` is a literal at every
 * call site and this is always_inline, so the loop unrolls to straight-line
 * code.
 *
 * acc[] is NOT necessarily registers, and that is measured rather than hoped
 * for: at NC = 2 GCC keeps it in ymm, and at NC = 4 and 8 it puts part of it on
 * the stack and reads it back with a memory-operand vpaddw (11 and 36
 * stack-touching moves per block). Four separate attempts to prevent that --
 * sizing the array to nc, replacing it with a struct of eight named members,
 * and the two restructurings the header records -- moved the spill count by at
 * most three and the wall clock not at all. What that establishes is negative
 * and worth stating as such: the spills are not what caps this loop, or every
 * way of removing them costs exactly what they cost. It does not establish
 * which, and no measurement here separates the two.
 *
 * The five activation addresses per column stay memory operands of vpmaddubsw,
 * as in the per-row kernel: hoisting them would cost registers this function
 * has none of to spare. */
static inline __attribute__((always_inline)) void
t5b_spread(__m256i *acc, __m256i plane, const int8_t *a, size_t astride,
           size_t k, const size_t nc)
{
    for (size_t c = 0; c < nc; ++c) {
        acc[c] = t5b_contract(acc[c], plane,
                              a + c * astride + k * TERNARY_T5B_BYTES);
    }
}

/* One 32-byte weight block, decoded ONCE, contracted against nc columns.
 * `a` is the block's activations for column 0 and must be readable for 160
 * bytes at each of the nc column offsets; the caller arranges that for the
 * tail exactly as ternary_t5b_gemv_avx2 does. */
static inline __attribute__((always_inline)) void
t5b_block_nc(__m256i *acc, const uint8_t *wp, const int8_t *a, size_t astride,
             const size_t nc)
{
    const __m256i lo8 = _mm256_set1_epi16(0x00FF);

    const __m256i x  = _mm256_loadu_si256((const __m256i *)wp);
    const __m256i xe = _mm256_and_si256(x, lo8);
    const __m256i xo = _mm256_srli_epi16(x, 8);

    /* Same four magic constants, same exactness bounds, same byte-lane
     * subtraction as t5b_block -- see its comment and the header. The only
     * change is the ORDER in which the quotients are materialised. */
    const __m256i q4 = t5b_quotient(xe, xo, _mm256_set1_epi16(811));
    t5b_spread(acc, q4, a, astride, 4, nc);

    const __m256i q3 = t5b_quotient(xe, xo, _mm256_set1_epi16(2428));
    t5b_spread(acc, _mm256_sub_epi8(q3, t5b_triple(q4)), a, astride, 3, nc);

    const __m256i q2 = t5b_quotient(xe, xo, _mm256_set1_epi16(7282));
    t5b_spread(acc, _mm256_sub_epi8(q2, t5b_triple(q3)), a, astride, 2, nc);

    const __m256i q1 = t5b_quotient(xe, xo, _mm256_set1_epi16(21846));
    t5b_spread(acc, _mm256_sub_epi8(q1, t5b_triple(q2)), a, astride, 1, nc);

    t5b_spread(acc, _mm256_sub_epi8(x, t5b_triple(q1)), a, astride, 0, nc);
}

/* One weight row against nc activation columns.
 *
 * The int16 bound is per COLUMN and is the same one the per-row kernel obeys:
 * columns share the decoded planes, not the accumulators, so each still takes
 * at most 12 * 2560 = 30720 before its own fold. Blocking columns cannot
 * overflow an accumulator that blocking rows could not.
 *
 * The int32 partial lives in a general-purpose register per column, for the
 * reason t5b_tile gives: at nc = 8 a ymm accumulator per column on top of the
 * nc int16 ones would not fit, and the value is touched once per twelve blocks.
 * The reassociation is over wrapping unsigned addition and returns the identical
 * sum. */
static inline __attribute__((always_inline)) void
t5b_gemm_row(const uint8_t *w, const int8_t *a, size_t astride,
             const int8_t *tail, int32_t *out, size_t ldc,
             size_t n, const size_t nc)
{
    const __m256i ones = _mm256_set1_epi16(1);
    const size_t  full = n / TERNARY_T5B_BLOCK;
    const size_t  rem  = n % TERNARY_T5B_BLOCK;

    uint32_t part[T5B_GEMM_NC_MAX];
    for (size_t c = 0; c < nc; ++c) {
        part[c] = 0;
    }

    size_t m = 0;

    for (; m + TERNARY_T5B_FOLD <= full; m += TERNARY_T5B_FOLD) {
        __m256i acc[T5B_GEMM_NC_MAX];
        for (size_t c = 0; c < nc; ++c) {
            acc[c] = _mm256_setzero_si256();
        }

        for (size_t t = 0; t < TERNARY_T5B_FOLD; ++t) {
            t5b_block_nc(acc, w + (m + t) * TERNARY_T5B_BYTES,
                         a + (m + t) * TERNARY_T5B_BLOCK, astride, nc);
        }

        for (size_t c = 0; c < nc; ++c) {
            part[c] += (uint32_t)t5b_hsum_epi32(
                           _mm256_madd_epi16(acc[c], ones));
        }
    }

    /* At most FOLD-1 whole blocks plus at most one partial block, so at most
     * 12 -- the same bound, one fold. */
    {
        __m256i acc[T5B_GEMM_NC_MAX];
        for (size_t c = 0; c < nc; ++c) {
            acc[c] = _mm256_setzero_si256();
        }

        for (; m < full; ++m) {
            t5b_block_nc(acc, w + m * TERNARY_T5B_BYTES,
                         a + m * TERNARY_T5B_BLOCK, astride, nc);
        }

        if (rem != 0) {
            /* The zero-padded tails the caller built, one per column, laid out
             * back to back at a 160-byte stride -- which is why this call, and
             * only this one, passes TERNARY_T5B_BLOCK as the column stride
             * instead of ldb. */
            t5b_block_nc(acc, w + full * TERNARY_T5B_BYTES, tail,
                         TERNARY_T5B_BLOCK, nc);
        }

        for (size_t c = 0; c < nc; ++c) {
            part[c] += (uint32_t)t5b_hsum_epi32(
                           _mm256_madd_epi16(acc[c], ones));
        }
    }

    for (size_t c = 0; c < nc; ++c) {
        out[c * ldc] = (int32_t)part[c];
    }
}

/* `rows` weight rows against a strip of nc columns. The tail buffers are built
 * ONCE for the strip and read by every row in it, the same amortisation the
 * GEMV makes across rows. */
static inline __attribute__((always_inline)) void
t5b_gemm_strip(const uint8_t *W, size_t rowbytes, size_t rows,
               const int8_t *B, size_t ldb, int8_t *tail,
               int32_t *C, size_t ldc, size_t n, const size_t nc)
{
    const size_t rem = n % TERNARY_T5B_BLOCK;

    if (rem != 0) {
        for (size_t c = 0; c < nc; ++c) {
            int8_t *t = tail + c * TERNARY_T5B_BLOCK;
            memset(t, 0, TERNARY_T5B_BLOCK);
            memcpy(t, B + c * ldb + (n - rem), rem);
        }
    }

    for (size_t r = 0; r < rows; ++r) {
        t5b_gemm_row(W + r * rowbytes, B, ldb, tail, C + r, ldc, n, nc);
    }
}

void ternary_t5b_gemm_avx2(const uint8_t *W, const int8_t *B, size_t ldb,
                           int32_t *C, size_t ldc,
                           size_t rows, size_t cols, size_t n)
{
    const size_t rowbytes = ternary_t5b_size(n);

    /* nc * 160 bytes, built per strip. Sized for the widest instantiation
     * below rather than for TERNARY_T5B_GEMM_NC, so that overriding the width
     * cannot resize the buffer out from under the remainder bodies. */
    int8_t tail[T5B_GEMM_NC_MAX * TERNARY_T5B_BLOCK];

    /* ROWS OUTSIDE, COLUMNS INSIDE, and the nesting is the point.
     *
     * A block of T5B_GEMM_ROWS weight rows is 44 KB at the real model's
     * K = 6912; a strip of 8 activation columns is 54 KB. Both sit in one
     * core's 512 KB L2, so the weights are streamed from DRAM ONCE for the
     * whole row block and every column strip contracts against them out of
     * cache. The other nesting -- columns outside -- re-reads the entire weight
     * matrix once per strip, which at cols = 256 and NC = 8 is thirty-two
     * passes over 11.9 MB.
     *
     * Note what is NOT done here: the rows are not interleaved. The row-tiled
     * GEMV measured 0.70x to 0.78x in DRAM because T interleaved weight streams
     * at a 512-byte stride get 4.3 GB/s where one linear stream gets 6.1 (see
     * the header). Blocking COLUMNS costs nothing on that axis -- the weight
     * stream stays linear and the reuse comes from the activation side, which
     * is in L2 either way. That asymmetry is why the GEMM gets a block and the
     * GEMV correctly does not. */
    for (size_t r0 = 0; r0 < rows; r0 += T5B_GEMM_ROWS) {
        size_t rn = rows - r0;
        if (rn > T5B_GEMM_ROWS) {
            rn = T5B_GEMM_ROWS;
        }

        const uint8_t *Wr = W + r0 * rowbytes;
        size_t c = 0;

        for (; c + TERNARY_T5B_GEMM_NC <= cols; c += TERNARY_T5B_GEMM_NC) {
            t5b_gemm_strip(Wr, rowbytes, rn, B + c * ldb, ldb, tail,
                           C + c * ldc + r0, ldc, n, TERNARY_T5B_GEMM_NC);
        }

        /* Columns left over. Narrower strips rather than a fall-back to the
         * GEMV: the width-1 body IS a decode-once-contract-once loop, so the
         * remainder keeps one tail-block discipline in the file instead of two,
         * and a prompt whose token count is not a multiple of the width still
         * gets the blocked path for all but its last few columns. */
#if TERNARY_T5B_GEMM_NC > 4
        for (; c + 4 <= cols; c += 4) {
            t5b_gemm_strip(Wr, rowbytes, rn, B + c * ldb, ldb, tail,
                           C + c * ldc + r0, ldc, n, 4);
        }
#endif
#if TERNARY_T5B_GEMM_NC > 2
        for (; c + 2 <= cols; c += 2) {
            t5b_gemm_strip(Wr, rowbytes, rn, B + c * ldb, ldb, tail,
                           C + c * ldc + r0, ldc, n, 2);
        }
#endif
#if TERNARY_T5B_GEMM_NC > 1
        for (; c < cols; ++c) {
            t5b_gemm_strip(Wr, rowbytes, rn, B + c * ldb, ldb, tail,
                           C + c * ldc + r0, ldc, n, 1);
        }
#endif
        (void)c;
    }
}
