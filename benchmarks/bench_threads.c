/*
 * bench_threads.c -- how much ALU headroom is there at the real model's
 * working set, once the model is running on more than one core?
 *
 * WHY THIS EXISTS
 * ---------------
 * Every throughput figure in this project is single-threaded, and the paper
 * says so. That was defensible while the question was "which kernel does more
 * arithmetic per second". It stops being defensible the moment the question
 * becomes "would reading FEWER WEIGHT BYTES make the model faster", because
 * the answer to that one depends entirely on whether the memory system or the
 * ALU is the binding constraint -- and llama.cpp runs the model on every core.
 *
 * results/bench_code_kernel.txt already shows the crossover single-threaded:
 *
 *     working set     upstream i2_s     GB/s     what binds
 *     4 MiB              83.00          20.75    the ALU
 *     256 MiB            59.33          14.83    DRAM
 *
 * 14.83 GB/s is a single core's DRAM read limit on this part, not the socket's.
 * Add cores and the per-core share falls while each core's ALU stays exactly as
 * fast, so the surplus grows. THAT is the quantity this measures, because it is
 * the budget any denser weight format has to fit inside: a format that reads
 * 20% fewer bytes but costs 80% more arithmetic wins if and only if the surplus
 * is above 1.8x.
 *
 * WHAT IT MEASURES
 *   For each thread count T, the SAME per-thread working set is run twice:
 *     cache arm   256 KiB of weights per thread -- fits L2, no DRAM traffic,
 *                 so the rate is the core's pure arithmetic rate
 *     dram  arm    64 MiB of weights per thread -- far past the 16 MiB CCX L3
 *                 slice, so the rate is what the memory system delivers
 *   surplus = cache rate / dram rate, per core. Above 1.0 means arithmetic is
 *   being wasted waiting for memory, and that waste is a budget.
 *
 * Every thread owns a private weight slice and private activation and output
 * buffers, so the arms differ in footprint and nothing else -- no false
 * sharing, no shared read stream that the prefetchers could turn into one.
 *
 * The kernel is upstream's own AVX2 i2_s branch, verbatim, from
 * ggml_i2s_ternary.c. Measuring the CONTRIBUTED kernel here would answer a
 * question about this project; measuring upstream's answers the question about
 * llama.cpp, which is the one that decides whether a new format is worth
 * building.
 *
 * Build:  gcc -O3 -mavx2 -mfma -march=native -std=c11 -pthread \
 *             bench/bench_threads.c build/obj/ggml_i2s_ternary.o -o build/bench_threads
 * Usage:  ./build/bench_threads [seconds_per_case]
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "../src/ggml_i2s_ternary.h"

#define MAX_THREADS 12
#define REPS 3

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint32_t rng_state = 0x2545F491u;
static uint32_t rng_next(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

typedef struct {
    const uint8_t *W;      /* this thread's private packed weight slice */
    const int8_t  *a;      /* private activations */
    float         *y;      /* private outputs */
    size_t         rows;
    size_t         n;
    long           iters;
    int            cpu;        /* logical CPU to pin to, -1 for none */
    pthread_barrier_t *start;
} job;

/* One logical CPU per PHYSICAL core, alternating between the two CCXs.
 * /sys/devices/system/cpu/cpuN/topology/core_id on this host reads
 * 0 1 2 4 5 6 0 1 2 4 5 6 for cpu0..cpu11, so cpu0-5 are the six distinct
 * cores and cpu6-11 are their SMT siblings; core_id 0,1,2 sit on one CCX and
 * 4,5,6 on the other. Without pinning the scheduler is free to put two threads
 * on one physical core, which depresses the cache arm and understates the
 * surplus, and free to keep a 3-thread run inside a single CCX, which
 * understates bandwidth. The order below spends cores alternately from the two
 * CCXs so every thread count is measured on a balanced placement. */
static const int CPU_ORDER[6] = {0, 3, 1, 4, 2, 5};

static void *worker(void *vp)
{
    job *j = (job *)vp;
    if (j->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(j->cpu, &set);
        /* Reported, not asserted: a failure here changes what was measured but
         * does not invalidate it, and a silent failure would be worse than
         * either. */
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
            fprintf(stderr, "warning: could not pin thread to cpu %d\n", j->cpu);
    }
    pthread_barrier_wait(j->start);
    for (long it = 0; it < j->iters; ++it) {
        bitnet_vec_dot_i2_i8_s_reference((int)j->n, j->y, 1, j->W, j->n,
                                         j->a, 0, (int)j->rows);
    }
    return NULL;
}

/* Pack `rows` rows of n ternary weights into the i2_s block layout: 32 bytes
 * per 128 weights, bits 6-7 of byte k holding weight 0+k of the block. Written
 * out here rather than shared with bench_ternary.c so this file builds against
 * nothing but the kernel it measures. */
static void pack_i2s(const int8_t *w, uint8_t *dst, size_t rows, size_t n)
{
    const size_t rb = n / 4;
    memset(dst, 0, rows * rb);
    for (size_t r = 0; r < rows; ++r) {
        const int8_t *wr = w + r * n;
        uint8_t *d = dst + r * rb;
        for (size_t blk = 0; blk * 128 < n; ++blk) {
            for (size_t k = 0; k < 32; ++k) {
                uint8_t byte = 0;
                for (int q = 0; q < 4; ++q) {
                    const size_t idx = blk * 128 + k + (size_t)q * 32;
                    const uint8_t code = (idx < n) ? (uint8_t)(wr[idx] + 1) : 1u;
                    byte |= (uint8_t)(code << (6 - 2 * q));
                }
                d[blk * 32 + k] = byte;
            }
        }
    }
}

/* One arm: T threads, each over `rows` private rows. Returns MACs/s aggregate,
 * taking the BEST of REPS -- contention can only slow a sample, never speed
 * one, so the minimum time is the least contaminated estimate. */
static double run_arm(int T, size_t rows, size_t n, uint8_t *bigW,
                      int8_t *bigA, float *bigY, double min_sec)
{
    const size_t rb = n / 4;
    double best = 0.0;

    /* Calibrate the iteration count on one thread so every arm runs for
     * roughly the same wall time regardless of how fast it is. */
    long iters = 1;
    for (;;) {
        job j = {bigW, bigA, bigY, rows, n, iters, -1, NULL};
        pthread_barrier_t b;
        pthread_barrier_init(&b, NULL, 1);
        j.start = &b;
        const double t0 = now_sec();
        worker(&j);
        const double el = now_sec() - t0;
        pthread_barrier_destroy(&b);
        if (el > min_sec / 4.0 || iters > (1L << 24)) {
            iters = (long)((double)iters * (min_sec / (el > 1e-9 ? el : 1e-9)));
            if (iters < 1) iters = 1;
            break;
        }
        iters *= 4;
    }

    for (int rep = 0; rep < REPS; ++rep) {
        pthread_t th[MAX_THREADS];
        job jobs[MAX_THREADS];
        pthread_barrier_t start;
        pthread_barrier_init(&start, NULL, (unsigned)T + 1);

        for (int t = 0; t < T; ++t) {
            jobs[t].W     = bigW + (size_t)t * rows * rb;
            jobs[t].a     = bigA + (size_t)t * n;
            jobs[t].y     = bigY + (size_t)t * rows;
            jobs[t].rows  = rows;
            jobs[t].n     = n;
            jobs[t].iters = iters;
            jobs[t].cpu   = CPU_ORDER[t % 6];
            jobs[t].start = &start;
            if (pthread_create(&th[t], NULL, worker, &jobs[t]) != 0) {
                fprintf(stderr, "pthread_create failed\n");
                exit(2);
            }
        }
        pthread_barrier_wait(&start);       /* all threads are now spinning */
        const double t0 = now_sec();
        for (int t = 0; t < T; ++t) pthread_join(th[t], NULL);
        const double el = now_sec() - t0;
        pthread_barrier_destroy(&start);

        const double macs = (double)T * (double)rows * (double)n * (double)iters;
        const double rate = macs / el;
        if (rate > best) best = rate;
    }
    return best;
}

int main(int argc, char **argv)
{
    const size_t n = 4096;
    const double min_sec = (argc > 1) ? atof(argv[1]) : 0.50;
    const size_t rb = n / 4;                       /* 1024 packed bytes/row */

    /* Per-thread footprints: 256 KiB fits the 512 KiB L2 slice of one core,
     * 64 MiB is four times the entire 16 MiB L3 slice of one CCX. */
    const size_t rows_cache = (256u * 1024u) / rb;      /*   256 rows */
    const size_t rows_dram  = (64u * 1024u * 1024u) / rb; /* 65536 rows */

    const int Ts[] = {1, 2, 3, 4, 6};
    const int nT = (int)(sizeof(Ts) / sizeof(Ts[0]));
    const int Tmax = Ts[nT - 1];

    printf("bench_threads: upstream i2_s AVX2 kernel, n=%zu, best of %d\n",
           n, REPS);
    printf("per-thread footprint: cache arm %zu rows = %.0f KiB, "
           "dram arm %zu rows = %.0f MiB\n",
           rows_cache, (double)(rows_cache * rb) / 1024.0,
           rows_dram, (double)(rows_dram * rb) / (1024.0 * 1024.0));
    printf("host: 6 physical cores, L2 512 KiB/core, L3 16 MiB per CCX\n\n");

    /* One allocation per arm, sliced per thread. */
    int8_t  *w  = malloc((size_t)Tmax * rows_dram * n);
    uint8_t *Wd = aligned_alloc(64, (size_t)Tmax * rows_dram * rb);
    uint8_t *Wc = aligned_alloc(64, (size_t)Tmax * rows_cache * rb);
    int8_t  *A  = aligned_alloc(64, (size_t)Tmax * n);
    float   *Yd = aligned_alloc(64, (size_t)Tmax * rows_dram * sizeof(float));
    if (!w || !Wd || !Wc || !A || !Yd) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }

    for (size_t i = 0; i < (size_t)Tmax * rows_dram * n; ++i)
        w[i] = (int8_t)((int)(rng_next() % 3) - 1);
    for (size_t i = 0; i < (size_t)Tmax * n; ++i)
        A[i] = (int8_t)((int)(rng_next() % 255) - 127);

    pack_i2s(w, Wd, (size_t)Tmax * rows_dram, n);
    pack_i2s(w, Wc, (size_t)Tmax * rows_cache, n);

    printf("  T   arm     aggregate GMAC/s   aggregate GB/s   per-core GMAC/s   "
           "surplus\n");
    printf("  --  ------  ----------------   --------------   ---------------   "
           "-------\n");

    double cache_per_core[16] = {0}, dram_per_core[16] = {0};

    for (int i = 0; i < nT; ++i) {
        const int T = Ts[i];
        const double rc = run_arm(T, rows_cache, n, Wc, A, Yd, min_sec);
        const double rd = run_arm(T, rows_dram,  n, Wd, A, Yd, min_sec);
        cache_per_core[i] = rc / T;
        dram_per_core[i]  = rd / T;

        printf("  %2d  cache   %16.2f   %14.2f   %15.2f   %7s\n",
               T, rc / 1e9, rc * 0.25 / 1e9, rc / T / 1e9, "-");
        printf("  %2d  dram    %16.2f   %14.2f   %15.2f   %6.2fx\n",
               T, rd / 1e9, rd * 0.25 / 1e9, rd / T / 1e9,
               cache_per_core[i] / dram_per_core[i]);
    }

    printf("\nWHAT THE SURPLUS COLUMN MEANS\n");
    printf("  It is the factor by which a core's arithmetic capacity exceeds\n");
    printf("  what the memory system keeps it fed with, at the real model's\n");
    printf("  working set. A weight format that reads 1/F of the bytes and\n");
    printf("  costs S times the arithmetic per weight is a net win exactly\n");
    printf("  when S < surplus, and then runs F times faster.\n\n");
    printf("  1.6 bits/weight vs i2_s 2.0 gives F = 1.25, and the base-3\n");
    printf("  quotient-chain kernel costs about S = 1.6 (0.25 vector ops per\n");
    printf("  MAC against upstream's 0.156). Read the 6-thread row.\n");

    fprintf(stderr, "sink=%f\n", (double)Yd[0]);
    free(w); free(Wd); free(Wc); free(A); free(Yd);
    return 0;
}
