/*
 * bench_gemm.c -- does the SHUFDIG decode help the GEMM path, or not?
 *
 * WHY THIS EXISTS. bench_alu measures the GEMV dot with memory taken out, and
 * there the shuffle decode is a clear win. The refuting agent reported the
 * opposite sign for the wide path: ternary_t5b_gemm_avx2 at NC=8 measured
 * 0.9965x, five trials of five below unity, i.e. prompt processing gains
 * NOTHING. That claim decides whether the improvement may be quoted next to
 * pp512 at all, and no benchmark in the repository drives the GEMM in
 * isolation, so it had never been checked here.
 *
 * The GEMM decodes each 32-byte block ONCE and spreads it across `cols`
 * activation columns, so the decode is amortised over the strip width. If the
 * decode is a smaller share of the work, a cheaper decode must help less -- and
 * the three constant tables it needs occupy registers that the wide strip wants
 * for accumulators. Those two effects run opposite ways and only measurement
 * separates them.
 *
 * Shapes are the model's: n = 6912 is bitnet-b1.58-2B-4T's hidden dimension,
 * and cols sweeps 1 (which is GEMV) to 64 across the widths llama.cpp actually
 * dispatches for prompt batches.
 *
 * Estimator is the MINIMUM over reps, as bench_alu uses: contention can only
 * slow a sample, never speed one, so the minimum is the least contaminated
 * statistic available on a shared box.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "../src/ternary_t5b.h"

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static uint64_t rs = 0x243F6A8885A308D3ull;
static uint32_t rng(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }

int main(int argc, char **argv)
{
    const size_t n    = 6912;                 /* the model's hidden dimension */
    const size_t rows = 512;
    const double min_sec = (argc > 1) ? atof(argv[1]) : 0.25;
    const size_t colset[] = {1, 4, 8, 16, 32, 64};
    const size_t ncol = sizeof(colset) / sizeof(colset[0]);
    const size_t rowbytes = ternary_t5b_size(n);

    int8_t  *w  = aligned_alloc(64, rows * n);
    uint8_t *W  = aligned_alloc(64, (rows * rowbytes + 63) & ~63ull);
    int8_t  *B  = aligned_alloc(64, 64 * n);
    int32_t *C  = aligned_alloc(64, rows * 64 * sizeof(int32_t));
    if (!w || !W || !B || !C) { fprintf(stderr, "alloc\n"); return 2; }

    for (size_t i = 0; i < rows * n; ++i) w[i] = (int8_t)((int)(rng() % 3) - 1);
    for (size_t r = 0; r < rows; ++r) ternary_t5b_pack(w + r * n, W + r * rowbytes, n);
    for (size_t i = 0; i < 64 * n; ++i) B[i] = (int8_t)((int)(rng() % 255) - 127);

    for (size_t s = 0; s < ncol; ++s) {
        const size_t cols = colset[s];
        double best = 0;
        for (int rep = 0; rep < 5; ++rep) {
            size_t iter = 0;
            const double t0 = now();
            double el;
            do {
                ternary_t5b_gemm_avx2(W, B, n, C, cols, rows, cols, n);
                ++iter;
                el = now() - t0;
            } while (el < min_sec);
            const double g = (double)iter * rows * cols * n / el * 1e-9;
            if (g > best) best = g;
        }
        printf("  gemm cols=%-3zu  %8.2f GMAC/s\n", cols, best);
    }
    /* Keep C alive so the whole call cannot be optimised away. */
    printf("sink=%d\n", C[0] + C[rows * 1 - 1]);
    return 0;
}
