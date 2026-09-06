/*
 * test_t5b.c -- correctness gate for the byte-lane 1.6-bit ternary packing.
 *
 * The AVX2 kernel is checked against a scalar reference derived from the
 * ENCODING (divide, take the remainder, get a digit), while the kernel is
 * derived from four QUOTIENT IDENTITIES it never takes a remainder of. Two
 * different derivations, so agreement means something.
 *
 * The three places a wrong answer could hide silently, each pinned rather
 * than reasoned about: the exactness range of the four magic constants, the
 * int16 accumulator between folds, and the tail block's activations.
 *
 * Exit 0 = all green, 1 = at least one mismatch. Fails loud and prints the
 * first ten differing cases with their inputs.
 */

#include "../ternary_t5b.h"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks   = 0;

static void check_eq(const char *what, size_t n, int32_t got, int32_t want)
{
    ++g_checks;
    if (got != want) {
        ++g_failures;
        if (g_failures <= 10) {
            fprintf(stderr, "FAIL %-40s n=%-9zu got=%-12d want=%d\n",
                    what, n, got, want);
        }
    }
}

static void check_true(const char *what, int cond)
{
    ++g_checks;
    if (!cond) {
        ++g_failures;
        if (g_failures <= 10) {
            fprintf(stderr, "FAIL %s\n", what);
        }
    }
}

/* Deterministic PRNG so a failure is always reproducible. */
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rng_next(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static void fill_ternary(int8_t *w, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        w[i] = (int8_t)((int)(rng_next() % 3) - 1);          /* -1, 0, +1 */
    }
}

static void fill_act8(int8_t *a, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        a[i] = (int8_t)((int)(rng_next() % 255) - 127);      /* -127..127 */
    }
}

static int32_t sum_i8(const int8_t *a, size_t n)
{
    int32_t s = 0;
    for (size_t i = 0; i < n; ++i) {
        s += a[i];
    }
    return s;
}

static int32_t dot_truth(const int8_t *w, const int8_t *a, size_t n)
{
    int32_t s = 0;
    for (size_t i = 0; i < n; ++i) {
        s += (int32_t)w[i] * (int32_t)a[i];
    }
    return s;
}

/* ---------------------------------------------------------------------- */
/* (a) The load-bearing step: four multiply-high identities                */
/* ---------------------------------------------------------------------- */

/* Runs the REAL instruction over every uint16, not scalar arithmetic that
 * resembles it. Two separate jobs:
 *
 *   - assert agreement with integer division over 0..242, the only range a
 *     packed byte can occupy. That is what correctness rests on.
 *   - find and print where each identity FIRST fails. The four constants have
 *     four different ranges, the narrowest by a factor of 68, and a design
 *     that treats them as interchangeable is one edit away from being wrong.
 *     The margin belongs on the record, not in a reader's head.
 */
static unsigned probe_mulhi(const char *name, unsigned magic, unsigned d)
{
    const __m256i mv = _mm256_set1_epi16((short)magic);
    unsigned first_bad = 0;
    int found = 0;

    for (uint32_t base = 0; base < 65536u; base += 16) {
        uint16_t in[16], out[16];
        for (int t = 0; t < 16; ++t) {
            in[t] = (uint16_t)(base + (uint32_t)t);
        }
        const __m256i v = _mm256_loadu_si256((const __m256i *)in);
        _mm256_storeu_si256((__m256i *)out, _mm256_mulhi_epu16(v, mv));

        for (int t = 0; t < 16; ++t) {
            const uint32_t x    = base + (uint32_t)t;
            const uint32_t want = x / d;
            const int      bad  = (out[t] != (uint16_t)want);

            if (x <= 242u) {                    /* the range that must hold */
                ++g_checks;
                if (bad) {
                    ++g_failures;
                    if (g_failures <= 10) {
                        fprintf(stderr, "FAIL mulhi %s x=%u got=%u want=%u\n",
                                name, x, (unsigned)out[t], want);
                    }
                }
            }
            if (bad && !found) {
                first_bad = x;
                found     = 1;
            }
        }
    }

    if (found) {
        printf("      mulhi(x,%-5u) = %-8s exact 0..%-6u first wrong at x=%-6u"
               " (%.1fx margin over 242)\n",
               magic, name, first_bad - 1, first_bad,
               (double)(first_bad - 1) / 242.0);
    } else {
        printf("      mulhi(x,%-5u) = %-8s exact over the whole uint16 range\n",
               magic, name);
    }
    check_true("identity is exact past the 242 a packed byte can reach",
               !found || first_bad > 242u);
    return first_bad;
}

static void test_mulhi_identities(void)
{
    const unsigned b3  = probe_mulhi("x/3",  21846, 3);
    const unsigned b9  = probe_mulhi("x/9",   7282, 9);
    const unsigned b27 = probe_mulhi("x/27",  2428, 27);
    const unsigned b81 = probe_mulhi("x/81",   811, 81);

    /* The narrowest of the four is the one that decides the packing's
     * headroom; naming it here means a future edit that widens the byte range
     * fails this assertion instead of silently returning wrong digits. */
    check_true("x/81 is the binding constraint of the four",
               b81 < b3 && b81 < b9 && b81 < b27);
    check_true("...and even it clears 2x the maximum packed byte",
               b81 > 2u * 242u);
}

/* ---------------------------------------------------------------------- */
/* (b) pack -> scalar unpack round trip                                    */
/* ---------------------------------------------------------------------- */

static void test_pack_roundtrip(void)
{
    const size_t sizes[] = {0, 1, 31, 32, 33, 159, 160, 161, 319, 320,
                            1000, 1920, 2560, 4096, 6912};

    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
        const size_t n     = sizes[s];
        const size_t slots = ternary_t5b_slots(n);
        int8_t  *w  = calloc(n + 1, 1);
        uint8_t *wp = calloc(ternary_t5b_size(n) + 1, 1);

        fill_ternary(w, n);
        check_eq("pack_checked accepts in-range weights", n,
                 ternary_t5b_pack_checked(w, wp, n), 0);

        for (size_t i = 0; i < n; ++i) {
            ++g_checks;
            const int8_t back = ternary_t5b_unpack_at(wp, i);
            if (back != w[i]) {
                ++g_failures;
                if (g_failures <= 10) {
                    fprintf(stderr, "FAIL roundtrip n=%zu i=%zu got=%d want=%d\n",
                            n, i, (int)back, (int)w[i]);
                }
            }
        }
        /* The padded tail must decode as weight 0, i.e. code 1. */
        for (size_t i = n; i < slots; ++i) {
            ++g_checks;
            if (ternary_t5b_unpack_at(wp, i) != 0) {
                ++g_failures;
                if (g_failures <= 10) {
                    fprintf(stderr, "FAIL pad slot %zu of n=%zu is not 0\n", i, n);
                }
            }
        }
        free(w);
        free(wp);
    }
}

/* The round trip above only proves pack and unpack agree WITH EACH OTHER --
 * they would still agree if both moved to a different layout together, and
 * the packed bytes are a format other code will have to read. So pin the
 * layout itself against the specification, recomputed here from
 * byte_j = sum_k code(w[160m + 32k + j]) * 3^k with a separate power table. */
static void test_layout_matches_spec(void)
{
    static const uint32_t pow3[5] = {1, 3, 9, 27, 81};
    const size_t n     = 500;                   /* 4 blocks, tail padded */
    const size_t slots = ternary_t5b_slots(n);
    int8_t  *w  = calloc(slots, 1);
    uint8_t *wp = calloc(ternary_t5b_size(n), 1);

    fill_ternary(w, n);
    ternary_t5b_pack(w, wp, n);

    check_true("a block is 160 weights in 32 bytes",
               TERNARY_T5B_BLOCK == 160 && TERNARY_T5B_BYTES == 32);
    check_true("which is 1.6 bits per weight",
               TERNARY_T5B_BYTES * 8 * 10 == TERNARY_T5B_BLOCK * 16);

    for (size_t m = 0; m < slots / TERNARY_T5B_BLOCK; ++m) {
        for (size_t j = 0; j < TERNARY_T5B_BYTES; ++j) {
            uint32_t want = 0;
            for (size_t k = 0; k < TERNARY_T5B_DIGITS; ++k) {
                const size_t i = m * 160 + 32 * k + j;
                want += (uint32_t)((i < n ? w[i] : 0) + 1) * pow3[k];
            }
            const uint32_t got = wp[m * 32 + j];
            check_eq("packed byte equals the specified base-3 sum",
                     m * 32 + j, (int32_t)got, (int32_t)want);
            check_true("a packed byte never reaches 3^5 = 243", want <= 242u);
        }
    }
    free(w);
    free(wp);
}

/* ---------------------------------------------------------------------- */
/* (c) AVX2 against the scalar reference, randomised                       */
/* ---------------------------------------------------------------------- */

/* Every case allocates one extra block of activations past n and fills it
 * with a nonzero pattern. If the kernel read any of it, the padded slots --
 * whose code is 1 -- would add it to the answer, so this makes the "reads
 * a[0..n-1] and nothing else" precondition a property of EVERY case below
 * rather than of one dedicated test. */
static void run_case(const char *what, const int8_t *w, const int8_t *a, size_t n)
{
    uint8_t *wp   = calloc(ternary_t5b_size(n) + 1, 1);
    int8_t  *apad = malloc(n + TERNARY_T5B_BLOCK + 1);

    memcpy(apad, a, n);
    memset(apad + n, 99, TERNARY_T5B_BLOCK + 1);     /* junk past n */

    ternary_t5b_pack(w, wp, n);

    const int32_t truth  = dot_truth(w, a, n);
    const int32_t scalar = ternary_t5b_dot_scalar(wp, apad, n);
    const int32_t vec    = ternary_t5b_dot_avx2(wp, apad, n, sum_i8(a, n));

    check_eq(what, n, scalar, truth);
    check_eq(what, n, vec, scalar);

    free(wp);
    free(apad);
}

static void test_random_triples(void)
{
    /* Forced first: n = 0, n below one block, n straddling every block and
     * fold boundary, and n = 6912 -- the real model's K, which is 43 whole
     * blocks plus 32 weights, so its tail path is live and not theoretical.
     * The rest are random. */
    const size_t forced[] = {0, 1, 2, 31, 32, 33, 159, 160, 161, 319, 320,
                             321, 1759, 1760, 1919, 1920, 1921, 2079, 2080,
                             2081, 3839, 3840, 3841, 4096, 6912, 6913, 13824};
    for (size_t k = 0; k < sizeof forced / sizeof forced[0]; ++k) {
        const size_t n = forced[k];
        int8_t *w = calloc(n + 1, 1), *a = calloc(n + 1, 1);
        fill_ternary(w, n);
        fill_act8(a, n);
        run_case("avx2 == scalar (forced size)", w, a, n);
        free(w);
        free(a);
    }

    /* 256 random triples; the modulus is deliberately not a multiple of 160,
     * so most of them exercise a partial tail block. */
    for (int t = 0; t < 256; ++t) {
        const size_t n = (size_t)(rng_next() % 5000u);
        int8_t *w = calloc(n + 1, 1), *a = calloc(n + 1, 1);
        fill_ternary(w, n);
        fill_act8(a, n);
        run_case("avx2 == scalar (random)", w, a, n);
        free(w);
        free(a);
    }
}

/* ---------------------------------------------------------------------- */
/* (d) The extremes, where the int16 accumulator peaks                     */
/* ---------------------------------------------------------------------- */

static void test_extremes(void)
{
    const size_t sizes[] = {160, 1920, 4096, 6912};
    /* -128 is outside the project's stated [-127,127] convention and is
     * included anyway: this path calls no _mm256_sign_epi8, so nothing here
     * breaks on it, and the accumulator bound in the header is stated at
     * |a| <= 128 precisely so that it covers this row. */
    const int acts[] = {-128, -127, 127};

    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
        const size_t n = sizes[s];
        int8_t *w = malloc(n), *a = malloc(n);

        for (size_t v = 0; v < sizeof acts / sizeof acts[0]; ++v) {
            /* Every weight +1, then every weight -1: |w| = 1 throughout, so
             * every product is at its maximum magnitude and every int16 lane
             * grows by the full 2560 per block with no cancellation at all. */
            for (int sign = 1; sign >= -1; sign -= 2) {
                for (size_t i = 0; i < n; ++i) {
                    w[i] = (int8_t)sign;
                    a[i] = (int8_t)acts[v];
                }
                run_case("all |w|=1 at the int8 activation extreme", w, a, n);
            }
            /* Alternating signs: same magnitudes, opposite arrangement, so a
             * bug that happens to cancel in the uniform case does not here. */
            for (size_t i = 0; i < n; ++i) {
                w[i] = (int8_t)((i & 1) ? 1 : -1);
                a[i] = (int8_t)((i & 1) ? acts[v] : -acts[v]);
            }
            run_case("alternating weights at the same extreme", w, a, n);
        }
        free(w);
        free(a);
    }
}

/* ---------------------------------------------------------------------- */
/* (e) The int16 accumulator, exactly at the fold boundary                 */
/* ---------------------------------------------------------------------- */

/* An int64 model of what the kernel's int16 accumulator holds between folds:
 * the same terms, in the same lane grouping (vpmaddubsw folds byte lanes 2L
 * and 2L+1 into int16 lane L), with no truncation and no saturation. Its job
 * is to say how close the real accumulator came to 32767, so "it still gives
 * the right answer" is a claim about the bound and not about a case that
 * happened to stay small.
 *
 * Returns the largest |lane| over all folds. `a` must already be the
 * zero-padded activation array of ternary_t5b_slots(n) bytes, which is what
 * the kernel effectively sees. */
static int64_t model_acc16_peak(const uint8_t *wp, const int8_t *apad, size_t n)
{
    static const uint32_t pow3[5] = {1, 3, 9, 27, 81};
    const size_t blocks = ternary_t5b_slots(n) / TERNARY_T5B_BLOCK;
    const size_t full   = n / TERNARY_T5B_BLOCK;
    int64_t peak = 0;
    size_t  m    = 0;

    while (m < blocks) {
        /* The kernel folds after TERNARY_T5B_FOLD whole blocks; the final
         * group is whatever is left, which is at most FOLD-1 whole blocks
         * plus the partial one. */
        size_t group = (m + TERNARY_T5B_FOLD <= full) ? TERNARY_T5B_FOLD
                                                      : (blocks - m);
        int64_t lane[16] = {0};

        for (size_t t = 0; t < group; ++t) {
            const uint8_t *wb = wp    + (m + t) * TERNARY_T5B_BYTES;
            const int8_t  *ab = apad  + (m + t) * TERNARY_T5B_BLOCK;
            for (size_t L = 0; L < 16; ++L) {
                for (size_t k = 0; k < TERNARY_T5B_DIGITS; ++k) {
                    for (size_t h = 0; h < 2; ++h) {
                        const size_t j    = 2 * L + h;
                        const int64_t code = (wb[j] / pow3[k]) % 3u;
                        lane[L] += code * (int64_t)ab[32 * k + j];
                    }
                }
            }
        }
        for (size_t L = 0; L < 16; ++L) {
            const int64_t v = lane[L] < 0 ? -lane[L] : lane[L];
            if (v > peak) {
                peak = v;
            }
        }
        m += group;
    }
    return peak;
}

static void test_fold_boundary(void)
{
    /* All weights +1 (code 2) and all activations at the int8 extreme: every
     * one of the 5 planes contributes 2 * (2*|a|) to every int16 lane, with
     * no cancellation anywhere, which is the worst case the header's bound is
     * derived from. n = FOLD*160 puts exactly FOLD blocks in one accumulator,
     * i.e. exactly on the boundary. */
    const size_t nfold = (size_t)TERNARY_T5B_FOLD * TERNARY_T5B_BLOCK;
    const struct { size_t n; int a; const char *label; } cases[] = {
        { nfold,                  127,  "exactly FOLD blocks, a=+127"  },
        { nfold,                 -128,  "exactly FOLD blocks, a=-128"  },
        { nfold - 160,           -128,  "FOLD-1 blocks, a=-128"        },
        { nfold + 160,           -128,  "FOLD+1 blocks, a=-128"        },
        { nfold - 1,             -128,  "FOLD blocks, last one partial"},
        { nfold * 10,            -128,  "10 full folds, a=-128"        },
        { 6912,                  -128,  "the real model's K, a=-128"   },
    };

    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; ++c) {
        const size_t n     = cases[c].n;
        const size_t slots = ternary_t5b_slots(n);
        int8_t  *w    = malloc(n);
        int8_t  *a    = malloc(n);
        int8_t  *apad = calloc(slots, 1);
        uint8_t *wp   = malloc(ternary_t5b_size(n));

        for (size_t i = 0; i < n; ++i) {
            w[i] = 1;
            a[i] = (int8_t)cases[c].a;
        }
        memcpy(apad, a, n);                     /* zero past n, as the kernel sees it */
        ternary_t5b_pack(w, wp, n);

        const int64_t peak  = model_acc16_peak(wp, apad, n);
        const int32_t truth = dot_truth(w, a, n);
        const int32_t vec   = ternary_t5b_dot_avx2(wp, a, n, sum_i8(a, n));

        printf("      %-32s n=%-6zu peak int16 lane %6lld  (%.1f %% of 32767)"
               "  answer %d\n",
               cases[c].label, n, (long long)peak,
               100.0 * (double)peak / 32767.0, vec);

        check_true("the modelled accumulator stays inside int16", peak <= 32767);
        check_eq("...and the kernel agrees with the truth at that peak",
                 n, vec, truth);
        check_eq("...as does the scalar reference", n,
                 ternary_t5b_dot_scalar(wp, apad, n), truth);

        free(w);
        free(a);
        free(apad);
        free(wp);
    }

    /* The bound is TIGHT, not a round number picked for comfort: one more
     * block of the same worst case would not fit. Asserted from the same
     * arithmetic the header states, so an edit to TERNARY_T5B_FOLD that
     * overflows fails here rather than in production. */
    {
        const long worst_per_block = 5L * 2L * (2L * 128L);         /* 2560 */
        check_eq("worst case per block is 2560", 0,
                 (int32_t)worst_per_block, 2560);
        check_true("FOLD blocks fit in int16",
                   TERNARY_T5B_FOLD * worst_per_block <= 32767);
        check_true("FOLD+1 blocks would not",
                   (TERNARY_T5B_FOLD + 1) * worst_per_block > 32767);
    }
}

/* ---------------------------------------------------------------------- */
/* The documented preconditions, pinned rather than assumed                */
/* ---------------------------------------------------------------------- */

static void test_documented_preconditions(void)
{
    /* 1. A weight of 2 encodes as code 3, which CARRIES into the next digit
     *    plane: it corrupts the weight 32 slots away, not just its own. */
    {
        const size_t n = 160;
        int8_t  w[160] = {0};
        uint8_t wp[TERNARY_T5B_BYTES];
        w[0] = 2;
        ternary_t5b_pack(w, wp, n);
        check_eq("w[0]=2 packs byte 0 as 123, not the all-zero 121", 0,
                 (int32_t)wp[0], 123);
        check_eq("w[0]=2 decodes as -1", 0, ternary_t5b_unpack_at(wp, 0), -1);
        check_eq("...and CARRIES into w[32], which was 0", 32,
                 ternary_t5b_unpack_at(wp, 32), 1);
        check_eq("...while w[64] is untouched", 64,
                 ternary_t5b_unpack_at(wp, 64), 0);
        check_eq("pack_checked rejects it instead", n,
                 ternary_t5b_pack_checked(w, wp, n), -1);
    }

    /* 2. A byte above 242 is not something this packer can produce, and for
     *    it the two paths disagree BY A PREDICTABLE AMOUNT: the scalar
     *    reference takes digit 4 as (255/81) % 3 = 0, while the kernel uses
     *    the raw quotient mulhi(255,811) = 3 with no mod, so the kernel runs
     *    3*a[128] high. Both answers are wrong; they are wrong differently,
     *    which is the point of writing the bound down. */
    {
        const size_t n = 160;
        uint8_t wp[TERNARY_T5B_BYTES];
        int8_t  a[160] = {0};

        memset(wp, 121, sizeof wp);          /* every byte the all-zeros code */
        wp[0] = 0xFF;                        /* ...except one no packer writes */
        a[4 * TERNARY_T5B_BYTES] = 100;      /* digit plane 4 of byte 0 */

        check_eq("foreign byte: the scalar decoder reads digit 4 as code 0", n,
                 ternary_t5b_dot_scalar(wp, a, n), -100);
        check_eq("...the kernel is 3*a = 300 higher, by the un-modded quotient",
                 n, ternary_t5b_dot_avx2(wp, a, n, sum_i8(a, n)), 200);
    }

    /* 3. The kernel reads a[0..n-1] and nothing else. run_case() already puts
     *    junk past n in every one of the ~290 cases above; this states the
     *    property directly, at a size whose tail block is 128 slots of
     *    padding -- the largest possible over-read had the kernel taken one. */
    {
        const size_t n = 6912;                        /* 43 blocks + 32 weights */
        const size_t slots = ternary_t5b_slots(n);    /* 7040 */
        int8_t  *w  = malloc(n);
        int8_t  *a  = malloc(slots);
        uint8_t *wp = malloc(ternary_t5b_size(n));

        check_eq("K=6912 leaves a partial tail block", 0,
                 (int32_t)(slots - n), 128);

        fill_ternary(w, n);
        fill_act8(a, slots);                          /* the tail is NOT zero */
        ternary_t5b_pack(w, wp, n);

        int32_t tail = 0;
        for (size_t i = n; i < slots; ++i) {
            tail += a[i];
        }
        check_true("the junk past n actually sums to something", tail != 0);

        check_eq("a nonzero region past n does not move the answer", n,
                 ternary_t5b_dot_avx2(wp, a, n, sum_i8(a, n)),
                 dot_truth(w, a, n));

        free(w);
        free(a);
        free(wp);
    }

    /* 4. sum_a is the caller's one remaining obligation, and it is the sum
     *    over the FIRST n activations. Getting it wrong shifts the answer by
     *    exactly the difference, with no other symptom -- so pin the failure
     *    mode rather than only the success. */
    {
        const size_t n = 1000;
        int8_t  *w  = malloc(n), *a = malloc(n);
        uint8_t *wp = malloc(ternary_t5b_size(n));

        fill_ternary(w, n);
        fill_act8(a, n);
        ternary_t5b_pack(w, wp, n);

        const int32_t good = sum_i8(a, n);
        check_eq("the right sum_a gives the right answer", n,
                 ternary_t5b_dot_avx2(wp, a, n, good), dot_truth(w, a, n));
        check_eq("...and a sum_a off by 7 is off by exactly 7", n,
                 ternary_t5b_dot_avx2(wp, a, n, good + 7),
                 dot_truth(w, a, n) - 7);

        free(w);
        free(a);
        free(wp);
    }
}

/* ---------------------------------------------------------------------- */
/* (f) the row-tiled GEMV                                                 */
/* ---------------------------------------------------------------------- */

/* Everything the tiled entry point can get wrong that the per-row kernel
 * cannot, checked on every case rather than in one dedicated test:
 *
 *   the row stride -- every row is compared INDIVIDUALLY against the scalar
 *     reference for that row's own bytes, so reading row r's weights at row
 *     r-1's offset shows up as a wrong number and not as a wrong total;
 *
 *   the tile remainder -- rows is swept over 0..9, which covers every residue
 *     of every tile width the file can be compiled with (1, 2, 3, 4) and both
 *     of the smaller bodies the remainder ladder falls through;
 *
 *   the SHARED tail block -- built once per call and read by every row of every
 *     tile, so a case with n not a multiple of 160 and rows past the tile width
 *     is what would catch it being clobbered between tiles. The forced sweep
 *     below is mostly such cases;
 *
 *   writing past the last row -- a sentinel sits at y[rows];
 *
 *   reading past a[n-1] -- the activations are followed by junk, exactly as in
 *     run_case, so the "reads a[0..n-1] and nothing else" precondition binds
 *     here too. The tail buffer is the one place this kernel could overrun.
 *
 * The comparison is against ternary_t5b_dot_scalar (the encoding-derived
 * reference), against dot_truth (the weights themselves), and against
 * ternary_t5b_dot_avx2 (the per-row kernel the tiling is meant to be
 * indistinguishable from). Three, because agreeing with the per-row kernel
 * alone would only prove the two share a bug. */
#define GEMV_SENTINEL 0x5A5AA5A5

static void run_gemv_case(const char *what, size_t rows, size_t n)
{
    const size_t rb = ternary_t5b_size(n);

    int8_t  *w    = calloc(rows * n + 1, 1);
    int8_t  *a    = calloc(n + 1, 1);
    uint8_t *W    = calloc(rows * rb + 1, 1);
    int8_t  *apad = malloc(n + TERNARY_T5B_BLOCK + 1);
    int32_t *y    = malloc((rows + 1) * sizeof(int32_t));

    fill_ternary(w, rows * n);
    fill_act8(a, n);

    memcpy(apad, a, n);
    memset(apad + n, 99, TERNARY_T5B_BLOCK + 1);     /* junk past n */

    for (size_t r = 0; r < rows; ++r) {
        ternary_t5b_pack(w + r * n, W + r * rb, n);
    }

    const int32_t sa = sum_i8(a, n);

    y[rows] = GEMV_SENTINEL;
    ternary_t5b_gemv_avx2(W, apad, y, rows, n, sa);

    for (size_t r = 0; r < rows; ++r) {
        check_eq(what, n, y[r], ternary_t5b_dot_scalar(W + r * rb, apad, n));
        check_eq(what, n, y[r], dot_truth(w + r * n, a, n));
        check_eq(what, n, y[r], ternary_t5b_dot_avx2(W + r * rb, apad, n, sa));
    }
    check_true("gemv wrote nothing past the last row",
               y[rows] == GEMV_SENTINEL);

    free(w);
    free(a);
    free(W);
    free(apad);
    free(y);
}

static void test_gemv_tiled(void)
{
    /* Forced first. rows 0..9 against sizes that put the tail block in every
     * state it has: none (n a multiple of 160), one partial block and nothing
     * else (n < 160), a partial block after several whole ones, a partial block
     * exactly at the fold boundary, and the real model's K = 6912, which is 43
     * whole blocks plus 32 weights. n = 0 is in the list because a GEMV of no
     * weights still has to write rows entries and write them correctly. */
    const size_t ns[] = {0, 1, 32, 159, 160, 161, 1919, 1920, 1921, 2560, 6912};

    for (size_t k = 0; k < sizeof ns / sizeof ns[0]; ++k) {
        for (size_t rows = 0; rows <= 9; ++rows) {
            run_gemv_case("gemv == scalar (forced)", rows, ns[k]);
        }
    }

    /* 128 random (rows, n) pairs, on top of the 110 forced ones. The modulus is
     * deliberately not a multiple of 160, so most exercise a partial tail. */
    for (int t = 0; t < 128; ++t) {
        const size_t rows = (size_t)(rng_next() % 9u) + 1u;
        const size_t n    = (size_t)(rng_next() % 5000u);
        run_gemv_case("gemv == scalar (random)", rows, n);
    }
}

/* ---------------------------------------------------------------------- */
/* (g) the column-blocked GEMM                                            */
/* ---------------------------------------------------------------------- */

/* The GEMM decodes each weight block ONCE and contracts it against a strip of
 * activation columns, which is a different set of ways to be wrong than the
 * GEMV has. Each of them is what one part of this case is built to catch:
 *
 *   a plane contracted against the wrong column, or a column's accumulator
 *     picking up a neighbour's product -- every (row, column) pair is compared
 *     individually, so a crossed pair is a wrong number in a known cell rather
 *     than a wrong total. The activations differ per column by construction;
 *
 *   the strip remainder -- cols is swept over 1, 2, 3, 5, 7, 16 and 33, which
 *     covers every residue of every width the file can be compiled with
 *     (1, 2, 4, 8) and every rung of the 4-2-1 ladder the leftover columns fall
 *     through. A width-8 kernel that is right and a width-1 body that is not
 *     produces a suite that passes on cols = 16 and fails on cols = 7;
 *
 *   the ROW BLOCK -- the kernel processes 32 rows at a time so a block of
 *     weights stays in L2 across the column strips, and rows is therefore swept
 *     over 31, 32, 33 and 65 as well as small values. A row-block boundary that
 *     resets a column pointer would be invisible below 32 rows;
 *
 *   the SHARED tails -- one zero-padded 160-byte buffer per column, built once
 *     per strip and read by all 32 rows in it. A case with n not a multiple of
 *     160 and rows past the row block is what catches one being clobbered
 *     between rows or, worse, read at the wrong column's offset -- which is why
 *     the tail is where the per-column stride changes from ldb to 160;
 *
 *   ldb and ldc -- both are swept with and without padding. With padding the
 *     columns of B are NOT contiguous and the gaps hold junk, so a kernel that
 *     assumed a packed activation matrix reads the junk; the cells of C outside
 *     the written rectangle hold a sentinel that is checked afterwards;
 *
 *   reading past a[n-1] -- every byte of B that is not one of the cols*n live
 *     activations is junk, including the 160 bytes after the last column, so
 *     the "reads a[0..n-1] and nothing else" precondition binds per column.
 *
 * Three references per pair, for the reason run_gemv_case gives: the
 * encoding-derived scalar, the raw weights, and the GEMV this replaces.
 * Agreeing with the GEMV alone would only prove the two share a decode. Note
 * the +sum(a): the GEMM returns sum(code*a) and the two scalar references
 * return sum(w*a), which differ by exactly the column's activation sum -- the
 * convention ggml_t5b_glue.h section 2 derives, checked here rather than
 * asserted there. */
#define GEMM_SENTINEL 0x3C3CC3C3

static void run_gemm_case(const char *what, size_t rows, size_t cols, size_t n,
                          int pad)
{
    const size_t rb  = ternary_t5b_size(n);
    const size_t ldb = n + (pad ? 37u : 0u);
    const size_t ldc = rows + (pad ? 5u : 0u);

    int8_t  *w = calloc(rows * n + 1, 1);
    uint8_t *W = calloc(rows * rb + 1, 1);
    /* Junk everywhere first, live activations written over it afterwards, so
     * both the inter-column gaps and the 160 bytes past the last column are
     * poison rather than zero. */
    int8_t  *B = malloc(cols * ldb + TERNARY_T5B_BLOCK + 1);
    int32_t *C = malloc((cols * ldc + 1) * sizeof(int32_t));
    int32_t *y = malloc(sizeof(int32_t));
    int32_t *sa = malloc((cols + 1) * sizeof(int32_t));

    memset(B, 99, cols * ldb + TERNARY_T5B_BLOCK + 1);
    fill_ternary(w, rows * n);
    for (size_t c = 0; c < cols; ++c) {
        fill_act8(B + c * ldb, n);
        sa[c] = sum_i8(B + c * ldb, n);
    }
    for (size_t r = 0; r < rows; ++r) {
        ternary_t5b_pack(w + r * n, W + r * rb, n);
    }

    for (size_t i = 0; i < cols * ldc + 1; ++i) {
        C[i] = GEMM_SENTINEL;
    }

    ternary_t5b_gemm_avx2(W, B, ldb, C, ldc, rows, cols, n);

    for (size_t c = 0; c < cols; ++c) {
        const int8_t *a = B + c * ldb;
        for (size_t r = 0; r < rows; ++r) {
            const int32_t got = C[c * ldc + r];
            check_eq(what, n, got,
                     ternary_t5b_dot_scalar(W + r * rb, a, n) + sa[c]);
            check_eq(what, n, got, dot_truth(w + r * n, a, n) + sa[c]);
            /* sum_a = 0: the GEMV's raw sum(code*a), the same convention. */
            ternary_t5b_gemv_avx2(W + r * rb, a, y, 1, n, 0);
            check_eq(what, n, got, y[0]);
        }
    }

    /* Every cell the kernel had no business touching. At pad = 1 that is five
     * int32 between one column of C and the next, which is where an off-by-one
     * in the ldc arithmetic lands. */
    {
        int intact = 1;
        for (size_t c = 0; c < cols; ++c) {
            for (size_t i = rows; i < ldc; ++i) {
                intact &= (C[c * ldc + i] == GEMM_SENTINEL);
            }
        }
        intact &= (C[cols * ldc] == GEMM_SENTINEL);
        check_true("gemm wrote only inside the rows x cols rectangle", intact);
    }

    free(w);
    free(W);
    free(B);
    free(C);
    free(y);
    free(sa);
}

/* The discriminating case for the sum(code*a) convention, and the one number in
 * this file that no plausible bug can imitate. A row of all-ZERO weights packs
 * every byte as 1*(1+3+9+27+81) = 121, so sum(code*a) must be EXACTLY sum(a)
 * for that row against every column -- after which ggml's own "- act_sums"
 * yields 0.0f. A missing correction gives 0, a doubled one gives 2*sum(a).
 * Neither can hide behind a plausible-looking number, which is the property
 * ggml_t5b_glue.h asks for and does not itself test. */
static void test_gemm_zero_row_is_sum_a(void)
{
    const size_t ns[] = {160, 2560, 6912, 6911};
    const size_t cols = 7;

    for (size_t k = 0; k < sizeof ns / sizeof ns[0]; ++k) {
        const size_t n    = ns[k];
        const size_t rb   = ternary_t5b_size(n);
        const size_t rows = 3;

        int8_t  *w = calloc(n, 1);                       /* every weight zero */
        uint8_t *W = calloc(rows * rb, 1);
        int8_t  *B = malloc(cols * n);
        int32_t *C = malloc(cols * rows * sizeof(int32_t));

        for (size_t r = 0; r < rows; ++r) {
            ternary_t5b_pack(w, W + r * rb, n);
        }
        for (size_t c = 0; c < cols; ++c) {
            fill_act8(B + c * n, n);
        }

        ternary_t5b_gemm_avx2(W, B, n, C, rows, rows, cols, n);

        for (size_t c = 0; c < cols; ++c) {
            const int32_t want = sum_i8(B + c * n, n);
            for (size_t r = 0; r < rows; ++r) {
                check_eq("all-zero weights -> exactly sum(a)", n,
                         C[c * rows + r], want);
            }
        }

        free(w);
        free(W);
        free(B);
        free(C);
    }
}

static void test_gemm_blocked(void)
{
    /* The (rows, cols) grid. cols covers 1, 2, 3, 5, 7, 16 and 33 -- every
     * residue class the 8-4-2-1 remainder ladder can land in -- and rows covers
     * both sides of the 32-row block boundary. */
    static const size_t rc[][2] = {
        {1, 1}, {1, 2}, {1, 3}, {1, 5}, {1, 7}, {1, 16},
        {2, 3}, {3, 1}, {3, 7}, {5, 3}, {7, 5}, {8, 16},
        {31, 1}, {31, 3}, {32, 2}, {33, 5}, {33, 16}, {65, 7},
    };
    /* n covers: nothing at all, one weight, less than a block, one short of a
     * block, exactly a block, one past it, the fold boundary from both sides,
     * and the real model's K = 6912 (43 whole blocks plus 32 weights). */
    static const size_t ns[] = {0, 1, 32, 159, 160, 161, 1919, 1920, 2560, 6912};

    const size_t nrc = sizeof rc / sizeof rc[0];
    const size_t nn  = sizeof ns / sizeof ns[0];

    for (size_t i = 0; i < nrc; ++i) {
        for (size_t k = 0; k < nn; ++k) {
            run_gemm_case("gemm == scalar (forced)", rc[i][0], rc[i][1],
                          ns[k], (int)((i + k) & 1u));
        }
    }

    /* rows = 0 and cols = 0 write nothing; run them so that a kernel which
     * writes anyway trips the sentinel rather than nobody noticing. */
    for (size_t k = 0; k < nn; ++k) {
        run_gemm_case("gemm, no rows", 0, 5, ns[k], 1);
        run_gemm_case("gemm, no columns", 5, 0, ns[k], 1);
    }

    /* 48 random shapes on top of the 180 forced ones. The moduli are chosen so
     * that most cases have a partial tail block and a column count that is not
     * a multiple of the strip width. */
    for (int t = 0; t < 48; ++t) {
        const size_t rows = (size_t)(rng_next() % 40u) + 1u;
        const size_t cols = (size_t)(rng_next() % 19u) + 1u;
        const size_t n    = (size_t)(rng_next() % 3000u);
        run_gemm_case("gemm == scalar (random)", rows, cols, n,
                      (int)(rng_next() & 1u));
    }
}

/* ---------------------------------------------------------------------- */

static void report_density(void)
{
    const size_t ns[] = {2560, 4096, 6912};
    for (size_t k = 0; k < sizeof ns / sizeof ns[0]; ++k) {
        const size_t n     = ns[k];
        const size_t bytes = ternary_t5b_size(n);
        printf("  n=%-5zu -> %zu bytes = %.4f bits/weight "
               "(i2_s would be %zu bytes at 2.0000; %.2f %% saved)\n",
               n, bytes, (double)(bytes * 8) / (double)n, (n * 2) / 8,
               100.0 * (1.0 - (double)bytes / (double)((n * 2) / 8)));
    }
}

/* Per-section counts, printed rather than asserted: a suite that says only
 * "all green" cannot show which part of it did the work, and a section that
 * silently stopped covering anything would still print PASS. */
static int g_mark = 0;
static void section(const char *name)
{
    if (name) {
        printf("  %-46s %6d checks\n", name, g_checks - g_mark);
    }
    g_mark = g_checks;
}

int main(void)
{
    printf("== 1.6-bit ternary packing, byte lanes (t5b) correctness ==\n");
    section(NULL);
    test_mulhi_identities();
    section("(a) four mulhi identities, all 65536 uint16");
    test_pack_roundtrip();       section("(b) pack -> unpack round trip + padding");
    test_layout_matches_spec();  section("    packed layout equals the spec");
    test_random_triples();       section("(c) avx2 == scalar, 283 (n,w,a) triples");
    test_extremes();             section("(d) a = -128/-127/+127, all |w| = 1");
    printf("-- the int16 accumulator between folds --\n");
    test_fold_boundary();        section("(e) exactly at the fold boundary");
    test_documented_preconditions();
    section("    documented preconditions, pinned");
    test_gemv_tiled();
    section("(f) row-tiled gemv == scalar, every row");
    test_gemm_blocked();
    section("(g) column-blocked gemm == scalar, every cell");
    test_gemm_zero_row_is_sum_a();
    section("    all-zero weights return exactly sum(a)");
    printf("-- packed density --\n");
    report_density();

    printf("%d checks, %d failures\n", g_checks, g_failures);
    if (g_failures) {
        printf("RESULT: FAIL\n");
        return 1;
    }
    printf("RESULT: PASS\n");
    return 0;
}
