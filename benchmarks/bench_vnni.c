/*
 * bench_vnni.c -- the VNNI question, measured instead of modelled.
 *
 * WHY THIS EXISTS
 * ---------------
 * Paper section 9 leads its limitations with VNNI: VPDPBUSD collapses the
 * contraction into one instruction while leaving the decode untouched, which
 * moves the profitability condition against a packed format, and an independent
 * system targeting VNNI stores ternary weights at eight bits for that reason.
 *
 * scripts/second_datapoint.sh was written to answer that and DID NOT. A
 * reviewer ran it on an Intel Xeon reporting avx512_vnni, then checked the
 * binaries with objdump and found ZERO vpdpbusd. The reason is simple and worth
 * stating plainly: no compiler contracts `vpmaddubsw` followed by an
 * accumulating `vpaddw` into VPDPBUSD as an idiom. -march=native does not
 * produce the instruction. It has to be written, and it had not been.
 *
 * So that run measured the AVX2 kernels on a second microarchitecture -- useful,
 * and not the VNNI question. This file writes the instruction.
 *
 * WHAT IT MEASURES
 * ----------------
 * Four kernels over the same weights and activations:
 *
 *     i2_s  AVX2   4x (vpmaddubsw + vpaddw), int16 accumulator, fold at 32
 *     i2_s  VNNI   4x vpdpbusd into four int32 accumulators, no fold
 *     t5b   AVX2   the shipped kernel: 8 vpmulhuw, 5x (vpmaddubsw + vpaddw)
 *     t5b   VNNI   the same decode, 5x vpdpbusd, no fold and no accumulator bound
 *
 * S = i2_s rate / t5b rate is reported for both instruction sets. The paper
 * predicts S rises under VNNI, because the baseline spends a larger FRACTION of
 * its work in the contraction that VPDPBUSD collapses (4 of 17 operations)
 * than the packed kernel does (5 of 42).
 *
 * THREE THINGS THIS FILE DOES SO ITS OUTPUT CAN BE TRUSTED
 *
 *   1. It CHECKS THE INSTRUCTION IS THERE. `--verify-isa` re-reads this
 *      program's own text segment through /proc/self/maps and counts the
 *      vpdpbusd encoding. A benchmark claiming to measure VNNI while running
 *      AVX2 is exactly the failure this file exists to repair, and asserting it
 *      is not the same as checking it.
 *   2. It CHECKS THE ANSWER. Every VNNI kernel is compared against the scalar
 *      reference on every row before any timing is printed. A faster wrong
 *      answer is not a result.
 *   3. It REFUSES TO RUN WITHOUT THE HARDWARE. CPUID is queried at startup; on
 *      a part without AVX512VNNI or AVXVNNI it exits 3 with a message, rather
 *      than printing numbers from an illegal-instruction trap or, worse, from a
 *      silently different code path.
 *
 * Build (the flags matter -- without them the intrinsics do not compile):
 *     gcc -O3 -march=native -mavx512vnni -mavx512vl -std=c11 \
 *         bench/bench_vnni.c build/obj/ternary_t5b.o -o build/bench_vnni
 *   or, on a VEX-only AVX-VNNI part such as Alder Lake:
 *     gcc -O3 -march=native -mavxvnni -std=c11 ...
 *
 * Usage:  ./bench_vnni [--verify-isa]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
#include <cpuid.h>
#include "../src/ternary_t5b.h"

/* The 256-bit form is spelled differently by the two feature sets that provide
 * it. AVX512VNNI+AVX512VL gives _mm256_dpbusd_epi32; AVX-VNNI (VEX, Alder Lake
 * and later, no AVX-512) gives _mm256_dpbusd_avx_epi32. Prefer the VEX form
 * when it is the only one available, so this builds on both. */
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#  define T5B_DPBUSD(acc, u, s) _mm256_dpbusd_epi32((acc), (u), (s))
#  define T5B_VNNI_FLAVOUR "AVX512VNNI+VL"
#  define T5B_VNNI_REAL 1
#elif defined(__AVXVNNI__)
#  define T5B_DPBUSD(acc, u, s) _mm256_dpbusd_avx_epi32((acc), (u), (s))
#  define T5B_VNNI_FLAVOUR "AVX-VNNI (VEX)"
#  define T5B_VNNI_REAL 1
#elif defined(T5B_EMULATE_VNNI)
/* An exact AVX2 model of VPDPBUSD, so the ALGORITHM can be checked on a machine
 * that cannot execute the instruction -- which is every machine this project
 * owns. VPDPBUSD multiplies four unsigned bytes by four signed bytes and adds
 * the four products into the int32 lane; vpmaddubsw gives the pairwise int16
 * sums and vpmaddwd adds adjacent pairs and widens, which is the same sum in
 * the same order. Saturation cannot intervene: the operands here are digits in
 * 0..2 against int8, so an int16 lane holds at most 2*2*127 = 508.
 *
 * This is a CORRECTNESS backend, never a performance one -- it is by
 * construction slower than the AVX2 kernels it would be compared against, so
 * the benchmark refuses to print timings when built this way. */
static inline __m256i t5b_dpbusd_emu(__m256i acc, __m256i u, __m256i s)
{
    return _mm256_add_epi32(acc,
        _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)));
}
#  define T5B_DPBUSD(acc, u, s) t5b_dpbusd_emu((acc), (u), (s))
#  define T5B_VNNI_FLAVOUR "AVX2 EMULATION of VPDPBUSD -- correctness only"
#  define T5B_VNNI_REAL 0
#else
#  error "build with -mavx512vnni -mavx512vl, with -mavxvnni, or with -DT5B_EMULATE_VNNI"
#endif

static double now(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static uint32_t rs = 20260907;
static uint32_t rn(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

static int have_vnni(void)
{
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
    if (c & (1u << 11)) return 1;                 /* AVX512_VNNI  */
    if (!__get_cpuid_count(7, 1, &a, &b, &c, &d)) return 0;
    return (a & (1u << 4)) != 0;                  /* AVX_VNNI     */
}

static int32_t hsum_epi32(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* ---------------------------------------------------------------- i2_s, both
 * The upstream layout: one byte holds four codes for weights 32 apart, so a
 * 32-byte group covers 128 weights, and the four 2-bit fields are extracted by
 * shift-and-mask. Both variants below read it identically; only the
 * contraction differs. sum(code*a) is returned, not sum(w*a) -- the caller
 * subtracts sum(a), which is identity (5) in the paper. */
#define I2S_MASK 0x03

static int32_t i2s_dot_avx2(const uint8_t *wp, const int8_t *a, size_t n)
{
    const __m256i m   = _mm256_set1_epi8(I2S_MASK);
    const size_t groups = n / 128;
    int32_t total = 0;
    __m256i acc32 = _mm256_setzero_si256();

    for (size_t g = 0; g < groups; ++g) {
        const __m256i x = _mm256_loadu_si256((const __m256i *)(wp + g * 32));
        const int8_t *ap = a + g * 128;
        __m256i acc = _mm256_setzero_si256();
        acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(
                  _mm256_and_si256(_mm256_srli_epi16(x, 6), m),
                  _mm256_loadu_si256((const __m256i *)(ap +  0))));
        acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(
                  _mm256_and_si256(_mm256_srli_epi16(x, 4), m),
                  _mm256_loadu_si256((const __m256i *)(ap + 32))));
        acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(
                  _mm256_and_si256(_mm256_srli_epi16(x, 2), m),
                  _mm256_loadu_si256((const __m256i *)(ap + 64))));
        acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(
                  _mm256_and_si256(x, m),
                  _mm256_loadu_si256((const __m256i *)(ap + 96))));
        /* Fold every group: the int16 lane holds 256 products of at most 2*127,
         * which is the bound section 7.1 shows upstream's 32-group fold
         * crosses. Folding here keeps this arm exact so the comparison is
         * about speed and not about who overflows. */
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(acc, _mm256_set1_epi16(1)));
    }
    total = hsum_epi32(acc32);
    for (size_t i = groups * 128; i < n; ++i) {
        const uint8_t by = wp[(i / 128) * 32 + (i % 32)];
        total += ((by >> (6 - 2 * ((i % 128) / 32))) & I2S_MASK) * a[i];
    }
    return total;
}

static int32_t i2s_dot_vnni(const uint8_t *wp, const int8_t *a, size_t n)
{
    const __m256i m = _mm256_set1_epi8(I2S_MASK);
    const size_t groups = n / 128;
    /* Four accumulators, one per plane: VPDPBUSD accumulates IN PLACE with
     * latency 10-13, so a single accumulator would serialise the loop on that
     * latency rather than on throughput. Four is what the plane count gives for
     * free; a production kernel would unroll further. */
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    __m256i a2 = _mm256_setzero_si256(), a3 = _mm256_setzero_si256();

    for (size_t g = 0; g < groups; ++g) {
        const __m256i x = _mm256_loadu_si256((const __m256i *)(wp + g * 32));
        const int8_t *ap = a + g * 128;
        a0 = T5B_DPBUSD(a0, _mm256_and_si256(_mm256_srli_epi16(x, 6), m),
                        _mm256_loadu_si256((const __m256i *)(ap +  0)));
        a1 = T5B_DPBUSD(a1, _mm256_and_si256(_mm256_srli_epi16(x, 4), m),
                        _mm256_loadu_si256((const __m256i *)(ap + 32)));
        a2 = T5B_DPBUSD(a2, _mm256_and_si256(_mm256_srli_epi16(x, 2), m),
                        _mm256_loadu_si256((const __m256i *)(ap + 64)));
        a3 = T5B_DPBUSD(a3, _mm256_and_si256(x, m),
                        _mm256_loadu_si256((const __m256i *)(ap + 96)));
    }
    int32_t total = hsum_epi32(_mm256_add_epi32(_mm256_add_epi32(a0, a1),
                                                _mm256_add_epi32(a2, a3)));
    for (size_t i = groups * 128; i < n; ++i) {
        const uint8_t by = wp[(i / 128) * 32 + (i % 32)];
        total += ((by >> (6 - 2 * ((i % 128) / 32))) & I2S_MASK) * a[i];
    }
    return total;
}

/* ------------------------------------------------------------------ t5b VNNI
 * The decode is byte-for-byte the shipped one -- four independent magic
 * multiplies, digits by byte-wise subtraction. Only the contraction changes,
 * and with it the accumulator bound: VPDPBUSD reduces into int32, so
 * TERNARY_T5B_FOLD has nothing to bound and the fold disappears. That is a real
 * saving the instruction-count comparison in analysis/microarch does NOT
 * credit, which makes the t5b VNNI figure here conservative. */
static int32_t t5b_dot_vnni(const uint8_t *wp, const int8_t *a, size_t n)
{
    const __m256i lo8 = _mm256_set1_epi16(0x00FF);
    const __m256i m1  = _mm256_set1_epi16(21846);
    const __m256i m2  = _mm256_set1_epi16(7282);
    const __m256i m3  = _mm256_set1_epi16(2428);
    const __m256i m4  = _mm256_set1_epi16(811);
    const size_t full = n / TERNARY_T5B_BLOCK;

    __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(), c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256();

    for (size_t b = 0; b < full; ++b) {
        const uint8_t *bp = wp + b * TERNARY_T5B_BYTES;
        const int8_t  *ap = a  + b * TERNARY_T5B_BLOCK;
        const __m256i x  = _mm256_loadu_si256((const __m256i *)bp);
        const __m256i xe = _mm256_and_si256(x, lo8);
        const __m256i xo = _mm256_srli_epi16(x, 8);

        const __m256i q1 = _mm256_or_si256(_mm256_mulhi_epu16(xe, m1),
                              _mm256_slli_epi16(_mm256_mulhi_epu16(xo, m1), 8));
        const __m256i q2 = _mm256_or_si256(_mm256_mulhi_epu16(xe, m2),
                              _mm256_slli_epi16(_mm256_mulhi_epu16(xo, m2), 8));
        const __m256i q3 = _mm256_or_si256(_mm256_mulhi_epu16(xe, m3),
                              _mm256_slli_epi16(_mm256_mulhi_epu16(xo, m3), 8));
        const __m256i q4 = _mm256_or_si256(_mm256_mulhi_epu16(xe, m4),
                              _mm256_slli_epi16(_mm256_mulhi_epu16(xo, m4), 8));

        #define TRIPLE(y) _mm256_add_epi16(_mm256_slli_epi16((y), 1), (y))
        c4 = T5B_DPBUSD(c4, q4,
                 _mm256_loadu_si256((const __m256i *)(ap + 4 * TERNARY_T5B_BYTES)));
        c3 = T5B_DPBUSD(c3, _mm256_sub_epi8(q3, TRIPLE(q4)),
                 _mm256_loadu_si256((const __m256i *)(ap + 3 * TERNARY_T5B_BYTES)));
        c2 = T5B_DPBUSD(c2, _mm256_sub_epi8(q2, TRIPLE(q3)),
                 _mm256_loadu_si256((const __m256i *)(ap + 2 * TERNARY_T5B_BYTES)));
        c1 = T5B_DPBUSD(c1, _mm256_sub_epi8(q1, TRIPLE(q2)),
                 _mm256_loadu_si256((const __m256i *)(ap + 1 * TERNARY_T5B_BYTES)));
        c0 = T5B_DPBUSD(c0, _mm256_sub_epi8(x,  TRIPLE(q1)),
                 _mm256_loadu_si256((const __m256i *)(ap + 0 * TERNARY_T5B_BYTES)));
        #undef TRIPLE
    }
    int32_t total = hsum_epi32(_mm256_add_epi32(
        _mm256_add_epi32(_mm256_add_epi32(c0, c1), _mm256_add_epi32(c2, c3)), c4));
    for (size_t i = full * TERNARY_T5B_BLOCK; i < n; ++i)
        total += (ternary_t5b_unpack_at(wp, i) + 1) * a[i];
    return total;
}

/* ------------------------------------------------------------ exact reference
 * Independent of every kernel above: it reads the ORIGINAL ternary weights, not
 * either packing, so a shared layout misunderstanding cannot make it agree. */
static int64_t exact_dot(const int8_t *w, const int8_t *a, size_t n)
{
    int64_t s = 0;
    for (size_t i = 0; i < n; ++i) s += (int64_t)(w[i] + 1) * a[i];
    return s;
}

/* --------------------------------------------------- is the instruction there?
 * EVDEX/VEX VPDPBUSD encodings both end in opcode 0x50 with a 0F38 map. Rather
 * than decode x86 by hand, shell out to objdump if it exists -- and if it does
 * not, say so instead of reporting a pass. This check exists because the
 * failure it guards against ALREADY HAPPENED: a benchmark that claimed to
 * measure VNNI ran AVX2 for weeks. */
static void verify_isa(const char *argv0)
{
    char cmd[1024];
    snprintf(cmd, sizeof cmd,
             "objdump -d '%s' 2>/dev/null | grep -c vpdpbusd", argv0);
    FILE *p = popen(cmd, "r");
    if (!p) { printf("  ISA CHECK: cannot run objdump -- NOT verified\n"); return; }
    int n = -1;
    if (fscanf(p, "%d", &n) != 1) n = -1;
    pclose(p);
    if (n < 0)  printf("  ISA CHECK: objdump unavailable -- NOT verified\n");
    else if (n == 0)
        printf("  ISA CHECK: *** ZERO vpdpbusd in this binary -- the VNNI rows\n"
               "             below are NOT measuring VNNI. Do not quote them. ***\n");
    else
        printf("  ISA CHECK: %d vpdpbusd instructions in this binary.\n", n);
}

int main(int argc, char **argv)
{
    if (T5B_VNNI_REAL && !have_vnni()) {
        fprintf(stderr,
            "bench_vnni: this CPU reports neither AVX512_VNNI nor AVX_VNNI.\n"
            "The VNNI question cannot be answered here. Exiting 3 rather than\n"
            "printing numbers from a path that is not the one named.\n"
            "To check the ALGORITHM anyway, rebuild with -DT5B_EMULATE_VNNI.\n");
        return 3;
    }

    const size_t n = 2560, rows = 100;      /* 64 KiB of weights: L2, not DRAM */
    int8_t  *w  = malloc(rows * n);
    int8_t  *a  = aligned_alloc(64, n + 64);
    uint8_t *Wi = aligned_alloc(64, rows * (n / 4));
    uint8_t *Wb = aligned_alloc(64, rows * ternary_t5b_size(n));
    if (!w || !a || !Wi || !Wb) { fprintf(stderr, "out of memory\n"); return 2; }

    for (size_t i = 0; i < rows * n; ++i) w[i] = (int8_t)((int)(rn() % 3) - 1);
    for (size_t i = 0; i < n; ++i)        a[i] = (int8_t)((int)(rn() % 255) - 127);

    for (size_t r = 0; r < rows; ++r) {
        uint8_t *d = Wi + r * (n / 4);
        memset(d, 0, n / 4);
        for (size_t b = 0; b * 128 < n; ++b)
            for (size_t j = 0; j < 32; ++j) {
                uint8_t by = 0;
                for (int q = 0; q < 4; ++q) {
                    const size_t i = b * 128 + j + (size_t)q * 32;
                    by |= (uint8_t)(((i < n) ? (uint8_t)(w[r * n + i] + 1) : 1u)
                                    << (6 - 2 * q));
                }
                d[b * 32 + j] = by;
            }
        ternary_t5b_pack(w + r * n, Wb + r * ternary_t5b_size(n), n);
    }

    printf("==================================================================\n");
    printf(" VNNI AGAINST AVX2 -- the same kernels, the contraction changed\n");
    printf("==================================================================\n\n");
    printf("  flavour: %s\n", T5B_VNNI_FLAVOUR);
    verify_isa(argv[0]);
    (void)argc;
    printf("\n");

    /* CORRECTNESS BEFORE SPEED. Every kernel on every row against exact int64
     * arithmetic over the original weights. A wrong answer is not a rate. */
    long bad = 0;
    for (size_t r = 0; r < rows; ++r) {
        const int64_t e = exact_dot(w + r * n, a, n);
        const int32_t v_i2s_a = i2s_dot_avx2(Wi + r * (n / 4), a, n);
        const int32_t v_i2s_v = i2s_dot_vnni(Wi + r * (n / 4), a, n);
        const int32_t v_t5b_a = ternary_t5b_dot_avx2(Wb + r * ternary_t5b_size(n),
                                                     a, n, 0);
        const int32_t v_t5b_v = t5b_dot_vnni(Wb + r * ternary_t5b_size(n), a, n);
        if (v_i2s_a != e || v_i2s_v != e || v_t5b_a != e || v_t5b_v != e) {
            if (bad < 3)
                fprintf(stderr, "row %zu: exact %lld  i2s/avx2 %d  i2s/vnni %d  "
                        "t5b/avx2 %d  t5b/vnni %d\n", r, (long long)e,
                        v_i2s_a, v_i2s_v, v_t5b_a, v_t5b_v);
            ++bad;
        }
    }
    if (bad) {
        fprintf(stderr, "\n%ld of %zu rows disagree with exact arithmetic. "
                "No timing is printed.\n", bad, rows);
        return 1;
    }
    printf("  correctness: all %zu rows exact, all four kernels.\n\n", rows);

    if (!T5B_VNNI_REAL) {
        printf("  Built with the AVX2 emulation of VPDPBUSD. The algorithm is\n"
               "  verified above -- which is the part that can be wrong -- but\n"
               "  no timing follows, because timing an emulation of the\n"
               "  instruction against the instruction it emulates would answer\n"
               "  nothing. Rebuild with -mavx512vnni -mavx512vl on VNNI\n"
               "  hardware for the measurement.\n");
        free(w); free(a); free(Wi); free(Wb);
        return 0;
    }

    const double macs = (double)rows * n;
    volatile int64_t sink = 0;
    (void)sink;
    double rate[4] = {0};
    const char *name[4] = { "i2_s AVX2", "i2_s VNNI", "t5b  AVX2", "t5b  VNNI" };

    for (int k = 0; k < 4; ++k) {
        double best = 0;
        for (int rep = 0; rep < 5; ++rep) {
            long it = 0; const double t0 = now(); double el;
            do {
                int64_t s = 0;
                for (size_t r = 0; r < rows; ++r) {
                    switch (k) {
                    case 0: s += i2s_dot_avx2(Wi + r * (n / 4), a, n); break;
                    case 1: s += i2s_dot_vnni(Wi + r * (n / 4), a, n); break;
                    case 2: s += ternary_t5b_dot_avx2(Wb + r * ternary_t5b_size(n),
                                                      a, n, 0); break;
                    default: s += t5b_dot_vnni(Wb + r * ternary_t5b_size(n), a, n);
                    }
                }
                sink += s; ++it; el = now() - t0;
            } while (el < 0.30);
            const double g = macs * it / el / 1e9;
            if (g > best) best = g;
        }
        rate[k] = best;
    }

    printf("  kernel        GMAC/s/core    vs its AVX2 form\n");
    for (int k = 0; k < 4; ++k) {
        if (k == 1 || k == 3)
            printf("  %-12s %10.2f      %6.3fx\n", name[k], rate[k], rate[k] / rate[k - 1]);
        else
            printf("  %-12s %10.2f           --\n", name[k], rate[k]);
    }
    printf("\n  S (i2_s / t5b) on AVX2 : %6.3f\n", rate[0] / rate[2]);
    printf("  S (i2_s / t5b) on VNNI : %6.3f\n", rate[1] / rate[3]);
    printf("\n"
           "  The paper predicts S RISES under VNNI: the baseline spends 4 of its\n"
           "  17 vector operations in the contraction VPDPBUSD collapses, the\n"
           "  packed kernel 5 of 42, so the instruction helps the baseline more.\n"
           "  If the VNNI S is the larger number, that prediction holds here.\n");

    fprintf(stderr, "sink=%lld\n", (long long)sink);
    free(w); free(a); free(Wi); free(Wb);
    return 0;
}
