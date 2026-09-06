/*
 * ternary_t10.c -- see ternary_t10.h for the layout contract, the telescoping
 * identity, and the five preconditions.
 *
 * Build: gcc -O3 -mavx2 -mfma -march=native -c ternary_t10.c
 */

#include "ternary_t10.h"

#include <immintrin.h>

/* 3^0 .. 3^10. The last entry, 59049, is the bound a packed lane never
 * reaches and the reason x_10 is zero. */
static const uint32_t T10_POW3[TERNARY_T10_DIGITS + 1] = {
    1u, 3u, 9u, 27u, 81u, 243u, 729u, 2187u, 6561u, 19683u, 59049u
};

/* ====================================================================== */
/* Geometry                                                               */
/* ====================================================================== */

static inline size_t t10_blocks(size_t n)
{
    return (n + TERNARY_T10_BLOCK - 1) / TERNARY_T10_BLOCK;
}

size_t ternary_t10_size(size_t n)
{
    return t10_blocks(n) * TERNARY_T10_BYTES;
}

size_t ternary_t10_slots(size_t n)
{
    return t10_blocks(n) * TERNARY_T10_BLOCK;
}

/* ====================================================================== */
/* Packing                                                                */
/* ====================================================================== */

void ternary_t10_pack(const int8_t *w, uint8_t *dst, size_t n)
{
    const size_t blocks = t10_blocks(n);

    for (size_t m = 0; m < blocks; ++m) {
        uint32_t lane[TERNARY_T10_LANES] = {0};

        for (size_t k = 0; k < TERNARY_T10_DIGITS; ++k) {
            for (size_t j = 0; j < TERNARY_T10_LANES; ++j) {
                const size_t i = m * TERNARY_T10_BLOCK
                               + k * TERNARY_T10_LANES + j;
                /* Slots past n are padded with code 1, which is weight 0.
                 * They contribute code*a - a = 0 only because prep_b makes
                 * the matching activation zero; see the header. */
                const uint32_t code = (i < n) ? (uint32_t)(w[i] + 1) : 1u;
                lane[j] += code * T10_POW3[k];
            }
        }

        /* Little-endian, written out rather than assumed, so the scalar
         * decoder below is the same function on any host. */
        for (size_t j = 0; j < TERNARY_T10_LANES; ++j) {
            dst[m * TERNARY_T10_BYTES + 2 * j    ] = (uint8_t)(lane[j] & 0xFFu);
            dst[m * TERNARY_T10_BYTES + 2 * j + 1] = (uint8_t)((lane[j] >> 8) & 0xFFu);
        }
    }
}

int ternary_t10_pack_checked(const int8_t *w, uint8_t *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        if (w[i] < -1 || w[i] > 1) {
            return -1;                   /* fail loud, never silently clamp */
        }
    }
    ternary_t10_pack(w, dst, n);
    return 0;
}

int8_t ternary_t10_unpack_at(const uint8_t *wp, size_t i)
{
    const size_t m = i / TERNARY_T10_BLOCK;
    const size_t p = i % TERNARY_T10_BLOCK;
    const size_t k = p / TERNARY_T10_LANES;
    const size_t j = p % TERNARY_T10_LANES;
    const size_t o = m * TERNARY_T10_BYTES + 2 * j;

    const uint32_t lane = (uint32_t)wp[o] | ((uint32_t)wp[o + 1] << 8);
    const int32_t  code = (int32_t)((lane / T10_POW3[k]) % 3u);
    return (int8_t)(code - 1);
}

/* ====================================================================== */
/* Scalar reference                                                       */
/* ====================================================================== */

/* Written from the ENCODING, by division and remainder, and deliberately not
 * from the telescoping identity: the point of the differential test is that
 * two independent derivations agree, and a reference derived from the same
 * algebra as the kernel would only prove the compiler works. */
int32_t ternary_t10_dot_scalar(const uint8_t *wp, const int8_t *a, size_t n)
{
    const size_t blocks = t10_blocks(n);
    uint32_t s = 0;                      /* unsigned: wraps, defined */

    for (size_t m = 0; m < blocks; ++m) {
        for (size_t j = 0; j < TERNARY_T10_LANES; ++j) {
            const size_t o = m * TERNARY_T10_BYTES + 2 * j;
            const uint32_t lane = (uint32_t)wp[o] | ((uint32_t)wp[o + 1] << 8);

            for (size_t k = 0; k < TERNARY_T10_DIGITS; ++k) {
                const size_t i = m * TERNARY_T10_BLOCK
                               + k * TERNARY_T10_LANES + j;
                if (i >= n) {
                    continue;            /* padding: never read past a[n-1] */
                }
                const int32_t code = (int32_t)((lane / T10_POW3[k]) % 3u);
                s += (uint32_t)((code - 1) * (int32_t)a[i]);
            }
        }
    }
    return (int32_t)s;
}

/* ====================================================================== */
/* The activation transform, once per GEMV                                */
/* ====================================================================== */

void ternary_t10_prep_b(const int8_t *a, int16_t *b, size_t n)
{
    const size_t slots = ternary_t10_slots(n);

    for (size_t i = 0; i < slots; ++i) {
        const int32_t a_i = (i < n) ? (int32_t)a[i] : 0;

        if (i % TERNARY_T10_BLOCK < TERNARY_T10_LANES) {
            b[i] = (int16_t)a_i;                        /* digit plane 0 */
        } else {
            const size_t prev = i - TERNARY_T10_LANES;  /* same lane, k-1 */
            const int32_t a_p = (prev < n) ? (int32_t)a[prev] : 0;
            /* |a - 3a'| <= 4*127 = 508: an int16 with room to spare. */
            b[i] = (int16_t)(a_i - 3 * a_p);
        }
    }
}

/* ====================================================================== */
/* AVX2 kernel                                                            */
/* ====================================================================== */

static inline int32_t t10_hsum_epi32(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* floor(x/3) for every uint16, exactly. 43691 = ceil(2^17/3), so mulhi then
 * >>1 computes floor(x*43691 / 2^17) = floor(x/3 + x/393216), and the error
 * term is under 1/6 for x <= 65535 -- never enough to cross an integer
 * boundary. The test checks all 65536 inputs rather than trusting that. */
static inline __m256i t10_div3(__m256i x, __m256i m3)
{
    return _mm256_srli_epi16(_mm256_mulhi_epu16(x, m3), 1);
}

int32_t ternary_t10_dot_avx2(const uint8_t *wp, const int16_t *b,
                             size_t n, int32_t sum_a)
{
    const __m256i m3   = _mm256_set1_epi16((short)0xAAAB);   /* 43691 */
    const __m256i sgn  = _mm256_set1_epi16((short)0x8000);   /* the x_0 bias */
    const __m256i ones = _mm256_set1_epi16(1);
    const size_t  blocks = t10_blocks(n);

    /* Four accumulators for four independent quotient chains. Every add here
     * is _mm256_add_epi32, which wraps by definition; the sums DO exceed
     * int32 and the wraparound is what makes the identity come out right. */
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    __m256i accb = _mm256_setzero_si256();   /* sum of digit plane 0 of b */

    size_t m = 0;

    /* Four blocks in flight. The reason is NOT latency, which is what this
     * comment claimed until it was measured. Dividing by three is a 6-cycle
     * dependency (vpmulhuw 5 + vpsrlw 1) and there are nine per block, so a
     * single chain looks like 54 idle cycles per 32 weight bytes -- but the
     * chains of successive blocks are independent and the out-of-order
     * engine already overlaps them. Sweeping the unroll depth over the same
     * data (n=4096, 1024 rows, L3-resident, Ryzen 5 3600) gives 28.4, 28.0,
     * 32.8, 32.8, 35.3, 32.7 GMAC/s for U = 1, 2, 3, 4, 6, 8: flat within a
     * quarter, with no knee. Four is a point in that flat region and nothing
     * more. The kernel is pinned by multiply THROUGHPUT instead -- see the
     * measured note at the top of ternary_t10.h. */
    for (; m + 4 <= blocks; m += 4) {
        __m256i x0 = _mm256_loadu_si256((const __m256i *)(wp + (m + 0) * TERNARY_T10_BYTES));
        __m256i x1 = _mm256_loadu_si256((const __m256i *)(wp + (m + 1) * TERNARY_T10_BYTES));
        __m256i x2 = _mm256_loadu_si256((const __m256i *)(wp + (m + 2) * TERNARY_T10_BYTES));
        __m256i x3 = _mm256_loadu_si256((const __m256i *)(wp + (m + 3) * TERNARY_T10_BYTES));

        const int16_t *q0 = b + (m + 0) * TERNARY_T10_BLOCK;
        const int16_t *q1 = b + (m + 1) * TERNARY_T10_BLOCK;
        const int16_t *q2 = b + (m + 2) * TERNARY_T10_BLOCK;
        const int16_t *q3 = b + (m + 3) * TERNARY_T10_BLOCK;

        const __m256i p0 = _mm256_loadu_si256((const __m256i *)q0);
        const __m256i p1 = _mm256_loadu_si256((const __m256i *)q1);
        const __m256i p2 = _mm256_loadu_si256((const __m256i *)q2);
        const __m256i p3 = _mm256_loadu_si256((const __m256i *)q3);

        /* Digit plane 0: x_0 reaches 59048, so bias it into int16 range. */
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_xor_si256(x0, sgn), p0));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_xor_si256(x1, sgn), p1));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_xor_si256(x2, sgn), p2));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_xor_si256(x3, sgn), p3));

        /* ...and carry the 32768*sum(b_0) that the bias owes back. Summing
         * the four planes first costs three int16 adds instead of three
         * madds; |b_0| <= 127 so four of them stay well inside int16. */
        accb = _mm256_add_epi32(accb,
                   _mm256_madd_epi16(_mm256_add_epi16(_mm256_add_epi16(p0, p1),
                                                      _mm256_add_epi16(p2, p3)),
                                     ones));

        /* Digit planes 1..9: x_j <= 19682 is already a non-negative int16. */
        for (int k = 1; k < TERNARY_T10_DIGITS; ++k) {
            const size_t off = (size_t)k * TERNARY_T10_LANES;
            x0 = t10_div3(x0, m3);
            x1 = t10_div3(x1, m3);
            x2 = t10_div3(x2, m3);
            x3 = t10_div3(x3, m3);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(x0, _mm256_loadu_si256((const __m256i *)(q0 + off))));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(x1, _mm256_loadu_si256((const __m256i *)(q1 + off))));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(x2, _mm256_loadu_si256((const __m256i *)(q2 + off))));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(x3, _mm256_loadu_si256((const __m256i *)(q3 + off))));
        }
    }

    /* Leftover blocks: the same body, one chain. */
    for (; m < blocks; ++m) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(wp + m * TERNARY_T10_BYTES));
        const int16_t *q = b + m * TERNARY_T10_BLOCK;
        const __m256i p = _mm256_loadu_si256((const __m256i *)q);

        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_xor_si256(x, sgn), p));
        accb = _mm256_add_epi32(accb, _mm256_madd_epi16(p, ones));

        for (int k = 1; k < TERNARY_T10_DIGITS; ++k) {
            x = t10_div3(x, m3);
            acc0 = _mm256_add_epi32(acc0,
                       _mm256_madd_epi16(x, _mm256_loadu_si256(
                           (const __m256i *)(q + (size_t)k * TERNARY_T10_LANES))));
        }
    }

    const __m256i acc = _mm256_add_epi32(_mm256_add_epi32(acc0, acc1),
                                         _mm256_add_epi32(acc2, acc3));

    /* Everything from here is modulo 2^32 ON PURPOSE. The pieces are far
     * larger than the answer and cancel; uint32_t is used because that is
     * where wraparound is defined, and signed overflow is not. */
    uint32_t s = (uint32_t)t10_hsum_epi32(acc)
               + 32768u * (uint32_t)t10_hsum_epi32(accb);

    /* sum(code*a) - sum(a) = sum(w*a). The true value fits int32, so this
     * final narrowing is exact even though everything feeding it wrapped. */
    return (int32_t)(s - (uint32_t)sum_a);
}
