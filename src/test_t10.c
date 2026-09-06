/*
 * test_t10.c -- correctness gate for the 1.6-bit ternary packing.
 *
 * The AVX2 kernel is checked against a scalar reference derived from the
 * ENCODING (divide, take the remainder, get a digit), while the kernel is
 * derived from the TELESCOPING IDENTITY (never compute a digit at all). Two
 * different derivations, so agreement means something.
 *
 * Exit 0 = all green, 1 = at least one mismatch. Fails loud and prints the
 * first ten differing cases with their inputs.
 */

#include "ternary_t10.h"

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
/* (a) The load-bearing step: floor(x/3) by multiply-high, for every uint16 */
/* ---------------------------------------------------------------------- */

static void test_div3_exhaustive(void)
{
    const __m256i m3 = _mm256_set1_epi16((short)0xAAAB);     /* 43691 */
    int reported = 0;

    for (uint32_t base = 0; base < 65536u; base += 16) {
        uint16_t in[16], out[16];
        for (int t = 0; t < 16; ++t) {
            in[t] = (uint16_t)(base + (uint32_t)t);
        }
        const __m256i v = _mm256_loadu_si256((const __m256i *)in);
        const __m256i r = _mm256_srli_epi16(_mm256_mulhi_epu16(v, m3), 1);
        _mm256_storeu_si256((__m256i *)out, r);

        for (int t = 0; t < 16; ++t) {
            const uint32_t x = base + (uint32_t)t;
            ++g_checks;
            if (out[t] != (uint16_t)(x / 3u)) {
                ++g_failures;
                if (++reported <= 5) {
                    fprintf(stderr, "FAIL div3 x=%u got=%u want=%u\n",
                            x, (unsigned)out[t], (unsigned)(x / 3u));
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------- */
/* (b) pack -> scalar unpack round trip                                    */
/* ---------------------------------------------------------------------- */

static void test_pack_roundtrip(void)
{
    const size_t sizes[] = {0, 1, 15, 16, 17, 159, 160, 161, 319, 320,
                            1000, 2560, 4096, 4097};

    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
        const size_t n     = sizes[s];
        const size_t slots = ternary_t10_slots(n);
        int8_t  *w  = calloc(n + 1, 1);
        uint8_t *wp = calloc(ternary_t10_size(n) + 1, 1);

        fill_ternary(w, n);
        check_eq("pack_checked accepts in-range weights", n,
                 ternary_t10_pack_checked(w, wp, n), 0);

        for (size_t i = 0; i < n; ++i) {
            ++g_checks;
            const int8_t back = ternary_t10_unpack_at(wp, i);
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
            if (ternary_t10_unpack_at(wp, i) != 0) {
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
 * the packed bytes are a format that other code will have to read. So pin
 * the layout itself against the specification, recomputed here from
 * lane_j = sum_k code(w[160m + 16k + j]) * 3^k, little-endian, with a
 * separate power table. */
static void test_layout_matches_spec(void)
{
    static const uint32_t pow3[10] =
        {1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683};
    const size_t n = 500;                       /* 4 blocks, tail padded */
    const size_t slots = ternary_t10_slots(n);
    int8_t  *w  = calloc(slots, 1);
    uint8_t *wp = calloc(ternary_t10_size(n), 1);

    fill_ternary(w, n);
    ternary_t10_pack(w, wp, n);

    check_true("a block is 160 weights in 32 bytes",
               TERNARY_T10_BLOCK == 160 && TERNARY_T10_BYTES == 32);

    for (size_t m = 0; m < slots / TERNARY_T10_BLOCK; ++m) {
        for (size_t j = 0; j < TERNARY_T10_LANES; ++j) {
            uint32_t want = 0;
            for (size_t k = 0; k < TERNARY_T10_DIGITS; ++k) {
                const size_t i = m * 160 + 16 * k + j;
                want += (uint32_t)((i < n ? w[i] : 0) + 1) * pow3[k];
            }
            const size_t o = m * 32 + 2 * j;
            const uint32_t got = (uint32_t)wp[o] | ((uint32_t)wp[o + 1] << 8);
            check_eq("packed lane equals the specified base-3 sum",
                     m * 16 + j, (int32_t)got, (int32_t)want);
            check_true("a packed lane never reaches 3^10", want <= 59048u);
        }
    }
    free(w);
    free(wp);
}

/* ---------------------------------------------------------------------- */
/* (c) AVX2 against the scalar reference, randomised                       */
/* ---------------------------------------------------------------------- */

static void run_case(const char *what, const int8_t *w, const int8_t *a, size_t n)
{
    const size_t slots = ternary_t10_slots(n);
    uint8_t *wp = calloc(ternary_t10_size(n) + 1, 1);
    int16_t *b  = calloc(slots + 1, sizeof(int16_t));

    ternary_t10_pack(w, wp, n);
    ternary_t10_prep_b(a, b, n);

    const int32_t truth  = dot_truth(w, a, n);
    const int32_t scalar = ternary_t10_dot_scalar(wp, a, n);
    const int32_t vec    = ternary_t10_dot_avx2(wp, b, n, sum_i8(a, n));

    check_eq(what, n, scalar, truth);
    check_eq(what, n, vec, scalar);

    free(wp);
    free(b);
}

static void test_random_triples(void)
{
    /* Forced first: n = 0, n below one block, n straddling every block and
     * unroll boundary. The rest are random. */
    const size_t forced[] = {0, 1, 2, 15, 16, 17, 31, 159, 160, 161, 319,
                             320, 321, 480, 639, 640, 641, 800, 1600, 4096,
                             2560, 4097};
    for (size_t k = 0; k < sizeof forced / sizeof forced[0]; ++k) {
        const size_t n = forced[k];
        int8_t *w = calloc(n + 1, 1), *a = calloc(n + 1, 1);
        fill_ternary(w, n);
        fill_act8(a, n);
        run_case("avx2 == scalar (forced size)", w, a, n);
        free(w);
        free(a);
    }

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
/* (d) The extremes, where the intermediate magnitudes peak                */
/* ---------------------------------------------------------------------- */

static void test_extremes(void)
{
    const size_t sizes[] = {160, 1600, 4096};
    const int8_t acts[]  = {-127, 127};

    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
        const size_t n = sizes[s];
        int8_t *w = malloc(n), *a = malloc(n);

        for (size_t v = 0; v < sizeof acts / sizeof acts[0]; ++v) {
            /* Every weight +1, then every weight -1: |w| = 1 throughout, so
             * every lane sits at its maximum (59048) or minimum (0). */
            for (int sign = 1; sign >= -1; sign -= 2) {
                for (size_t i = 0; i < n; ++i) {
                    w[i] = (int8_t)sign;
                    a[i] = acts[v];
                }
                run_case("all |w|=1 at the int8 activation extreme", w, a, n);
            }
            /* Alternating +1/-1 maximises |b_j| = |a - 3a'| at 508. */
            for (size_t i = 0; i < n; ++i) {
                w[i] = (int8_t)((i & 1) ? 1 : -1);
                a[i] = (int8_t)((i & 1) ? acts[v] : -acts[v]);
            }
            run_case("alternating weights, |b| at its 508 maximum", w, a, n);
        }
        free(w);
        free(a);
    }
}

/* ---------------------------------------------------------------------- */
/* (e) A case where the int32 accumulators provably wrap                   */
/* ---------------------------------------------------------------------- */

/* An int64 model of what the kernel's accumulator lanes hold: the same
 * terms, in the same lane grouping (_mm256_madd_epi16 folds int16 lanes 2L
 * and 2L+1 into int32 lane L), with no truncation anywhere. Its only job is
 * to establish that the real accumulators exceed int32 -- so "it still gives
 * the right answer" is a claim about wraparound and not about a case that
 * happened to stay small. */
static int64_t model_lane_totals(const uint8_t *wp, const int16_t *b,
                                 size_t n, int64_t lane[8])
{
    const size_t blocks = ternary_t10_slots(n) / TERNARY_T10_BLOCK;
    int64_t total = 0;
    for (int L = 0; L < 8; ++L) {
        lane[L] = 0;
    }

    for (size_t m = 0; m < blocks; ++m) {
        for (size_t j = 0; j < TERNARY_T10_LANES; ++j) {
            const size_t o = m * TERNARY_T10_BYTES + 2 * j;
            uint32_t x = (uint32_t)wp[o] | ((uint32_t)wp[o + 1] << 8);
            const int16_t *q = b + m * TERNARY_T10_BLOCK + j;

            /* Digit plane 0 carries the -32768 bias the kernel applies. */
            int64_t t = ((int64_t)x - 32768) * (int64_t)q[0];
            for (int k = 1; k < TERNARY_T10_DIGITS; ++k) {
                x /= 3u;
                t += (int64_t)x * (int64_t)q[k * TERNARY_T10_LANES];
            }
            lane[j / 2] += t;
            total       += t;
        }
    }
    return total;
}

static void test_accumulator_wraps(void)
{
    /* All weights +1 and all activations +127: every lane is 59048, every
     * b_0 is 127 and every other b_j is 127 - 3*127 = -254, so all 1280
     * blocks are identical and the four unrolled accumulators receive
     * exactly a quarter of the work each. */
    const size_t sizes[] = {8192, 204800};

    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; ++s) {
        const size_t n     = sizes[s];
        const size_t slots = ternary_t10_slots(n);
        int8_t  *w  = malloc(n);
        int8_t  *a  = malloc(n);
        uint8_t *wp = malloc(ternary_t10_size(n));
        int16_t *b  = malloc(slots * sizeof(int16_t));

        for (size_t i = 0; i < n; ++i) {
            w[i] = 1;
            a[i] = 127;
        }
        ternary_t10_pack(w, wp, n);
        ternary_t10_prep_b(a, b, n);

        int64_t lane[8];
        const int64_t total = model_lane_totals(wp, b, n, lane);
        int64_t worst = 0;
        for (int L = 0; L < 8; ++L) {
            const int64_t m = lane[L] < 0 ? -lane[L] : lane[L];
            if (m > worst) {
                worst = m;
            }
        }

        const int32_t truth = dot_truth(w, a, n);
        const int32_t vec   = ternary_t10_dot_avx2(wp, b, n, sum_i8(a, n));

        printf("  n=%-7zu untruncated accumulator total %+14lld, "
               "worst lane %+14lld (= %.2f x 2^31 before the 4-way split), "
               "answer %d\n",
               n, (long long)total, (long long)lane[0],
               (double)worst / 2147483648.0, vec);

        check_eq("wrapping case still gives the right answer", n, vec, truth);
        check_eq("...and the scalar reference agrees", n,
                 ternary_t10_dot_scalar(wp, a, n), truth);

        if (n == 204800) {
            /* Each of the four accumulators gets blocks/4 = 320 identical
             * blocks, so its lane holds exactly worst/4. Requiring that to
             * exceed 2^31 proves the registers themselves wrap, not merely
             * the final horizontal sum. */
            check_true("a single int32 accumulator lane exceeds 2^31",
                       worst / 4 > 2147483648LL);
        } else {
            /* Smaller case: the lanes survive, the horizontal sum does not. */
            const int64_t habs = total < 0 ? -total : total;
            check_true("the horizontal sum exceeds 2^31", habs > 2147483648LL);
        }

        free(w);
        free(a);
        free(wp);
        free(b);
    }
}

/* ---------------------------------------------------------------------- */
/* The documented preconditions, pinned rather than assumed                */
/* ---------------------------------------------------------------------- */

static void test_documented_preconditions(void)
{
    /* 1. A weight of 2 encodes as code 3, which CARRIES into the next digit
     *    plane: it corrupts the weight 16 slots away, not just its own. */
    {
        const size_t n = 160;
        int8_t  w[160] = {0};
        uint8_t wp[TERNARY_T10_BYTES];
        w[0] = 2;
        ternary_t10_pack(w, wp, n);
        check_eq("w[0]=2 decodes as -1", 0, ternary_t10_unpack_at(wp, 0), -1);
        check_eq("...and CARRIES into w[16], which was 0", 16,
                 ternary_t10_unpack_at(wp, 16), 1);
        check_eq("...while w[32] is untouched", 32,
                 ternary_t10_unpack_at(wp, 32), 0);
        check_eq("pack_checked rejects it instead", n,
                 ternary_t10_pack_checked(w, wp, n), -1);
    }

    /* 2. A lane above 59048 is not something this packer can produce, and
     *    for it the two paths disagree BY A PREDICTABLE AMOUNT: the
     *    telescoping drops -3*x_10*a_9, so with x = 0xFFFF (x_10 = 1) the
     *    kernel runs 3*a_9 high. Both answers are wrong; they are wrong
     *    differently, which is the point of writing the bound down. */
    {
        const size_t n = 160;
        uint8_t wp[TERNARY_T10_BYTES];
        int8_t  a[160] = {0};
        int16_t b[160];

        /* Every lane the all-zeros encoding (code 1 ten times = 29524)... */
        for (size_t j = 0; j < TERNARY_T10_LANES; ++j) {
            wp[2 * j]     = (uint8_t)(29524u & 0xFFu);
            wp[2 * j + 1] = (uint8_t)((29524u >> 8) & 0xFFu);
        }
        /* ...except lane 0, which no packer would ever write. */
        wp[0] = 0xFF;
        wp[1] = 0xFF;

        a[9 * TERNARY_T10_LANES] = 100;          /* a_9 of lane 0 */
        ternary_t10_prep_b(a, b, n);

        check_eq("foreign lane: the scalar decoder reads digit 9 as code 0", n,
                 ternary_t10_dot_scalar(wp, a, n), -100);
        check_eq("...the kernel is 3*a_9 = 300 higher, by the dropped term", n,
                 ternary_t10_dot_avx2(wp, b, n, sum_i8(a, n)), 200);
    }

    /* 3. Activations past n must be zero. prep_b enforces it, so the only
     *    way to violate it is to prepare b for a LONGER activation vector
     *    than the n the kernel is called with -- and then the padded slots,
     *    whose code is 1, add their activations to the answer. */
    {
        const size_t n     = 100;
        const size_t slots = ternary_t10_slots(n);       /* 160 */
        int8_t  w[160], a[160];
        uint8_t wp[TERNARY_T10_BYTES];
        int16_t b[160];

        fill_ternary(w, n);
        fill_act8(a, slots);                             /* tail NOT zero */
        for (size_t i = n; i < slots; ++i) {
            w[i] = 0;
        }
        ternary_t10_pack(w, wp, n);

        int32_t tail = 0;
        for (size_t i = n; i < slots; ++i) {
            tail += a[i];
        }

        ternary_t10_prep_b(a, b, slots);                 /* the wrong b */
        check_eq("a nonzero pad region shifts the answer by exactly its sum", n,
                 ternary_t10_dot_avx2(wp, b, n, sum_i8(a, n)),
                 dot_truth(w, a, n) + tail);

        ternary_t10_prep_b(a, b, n);                     /* the right b */
        check_eq("...and prep_b for the same n zeroes it, restoring the answer",
                 n, ternary_t10_dot_avx2(wp, b, n, sum_i8(a, n)),
                 dot_truth(w, a, n));
    }
}

/* ---------------------------------------------------------------------- */

static void report_density(void)
{
    const size_t ns[] = {2560, 4096};
    for (size_t k = 0; k < sizeof ns / sizeof ns[0]; ++k) {
        const size_t n     = ns[k];
        const size_t bytes = ternary_t10_size(n);
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
    printf("== 1.6-bit ternary packing (t10) correctness ==\n");
    section(NULL);
    test_div3_exhaustive();      section("(a) floor(x/3) for all 65536 uint16");
    test_pack_roundtrip();       section("(b) pack -> unpack round trip + padding");
    test_layout_matches_spec();  section("    packed layout equals the spec");
    test_random_triples();       section("(c) avx2 == scalar, 278 (n,w,a) triples");
    test_extremes();             section("(d) a = +-127, all |w| = 1, |b| = 508");
    printf("-- deliberate int32 wraparound --\n");
    test_accumulator_wraps();    section("(e) accumulators that provably wrap");
    test_documented_preconditions();
    section("    documented preconditions, pinned");
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
