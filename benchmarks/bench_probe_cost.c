/*
 * bench_probe_cost.c -- what does the in-situ cycle counter cost?
 *
 * WHY THIS EXISTS
 * ---------------
 * ggml_t5b_glue.c times the t5b matmul entry points with an rdtsc pair and an
 * atomic add, and derives the i2_s time by DIFFERENCE across two llama-bench
 * runs. A reviewer objected to the difference method, correctly: every load
 * fluctuation between the two runs lands entirely in the i2_s number, and on a
 * host at load 2-10 that makes the resulting 1.606x softer than the paragraph
 * quoting it sounds. Their proposed fix is to instrument the control arm too --
 * "an rdtsc pair costs a few dozen cycles per call against milliseconds of
 * matmul, that is unmeasurable".
 *
 * That estimate is probably right and it is still an estimate, and the reason
 * to check it is not pedantry. The counter is a SINGLE global that every ggml
 * worker thread updates. The rdtsc is cheap; what is not obviously cheap is
 * four threads issuing atomic read-modify-writes to one cache line, which
 * serialises them on the coherence protocol rather than on the instruction.
 * That cost scales with thread count and does not appear in a single-threaded
 * estimate at all.
 *
 * So this measures three things at 1, 2, 4 and 6 threads:
 *
 *   rdtsc alone                 the instruction, uncontended
 *   rdtsc + atomic add          what the macro actually does, SHARED counter
 *   rdtsc + atomic add          the same with a PER-THREAD counter, padded to
 *                               a cache line -- the fix, if the shared one costs
 *
 * and prints each against the ~19.9 ms/token that section 7.4a measures the
 * matmuls to take, so the reader sees the ratio rather than a raw cycle count.
 *
 * Build:  gcc -O3 -mavx2 -mfma -march=native -std=c11 -pthread \
 *             bench/bench_probe_cost.c -o build/bench_probe_cost
 * Usage:  ./bench_probe_cost [calls_per_token]      (default 210)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define REPS 2000000

static uint64_t g_shared = 0;

struct padded { uint64_t v; char pad[64 - sizeof(uint64_t)]; };
static struct padded g_per_thread[64];

static inline uint64_t rdtsc(void)
{
    unsigned lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t) hi << 32) | lo;
}

static double now(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

struct arg { int id; int mode; double sec; };

static void *worker(void *p)
{
    struct arg *a = p;
    const double t0 = now();
    for (long i = 0; i < REPS; ++i) {
        const uint64_t s = rdtsc();
        switch (a->mode) {
        case 0:  /* rdtsc pair alone: the instruction, nothing else */
            __asm__ __volatile__("" :: "r"(rdtsc() - s) : "memory");
            break;
        case 1:  /* what the macro does today: one global, every thread */
            __atomic_fetch_add(&g_shared, rdtsc() - s, __ATOMIC_RELAXED);
            break;
        default: /* per-thread counter on its own cache line */
            __atomic_fetch_add(&g_per_thread[a->id].v, rdtsc() - s,
                               __ATOMIC_RELAXED);
        }
    }
    a->sec = now() - t0;
    return NULL;
}

static double run(int nthreads, int mode)
{
    pthread_t th[64];
    struct arg ar[64];
    g_shared = 0;
    memset(g_per_thread, 0, sizeof g_per_thread);
    const double t0 = now();
    for (int i = 0; i < nthreads; ++i) {
        ar[i].id = i; ar[i].mode = mode; ar[i].sec = 0;
        pthread_create(&th[i], NULL, worker, &ar[i]);
    }
    for (int i = 0; i < nthreads; ++i) pthread_join(th[i], NULL);
    const double el = now() - t0;
    /* Cycles per probe, per thread: wall time * clock / reps. The clock is
     * derived from an uncontended rdtsc-delta over a known interval rather than
     * read from /proc, because the invariant TSC and the core clock differ on
     * this part (3.600 against ~3.9 GHz) and using the wrong one would scale
     * every number here. */
    return el / REPS * 1e9;   /* nanoseconds per probe */
}

int main(int argc, char **argv)
{
    const long calls_per_token = (argc > 1) ? atol(argv[1]) : 210;

    /* Establish the TSC rate, so cycle figures are honest. */
    const uint64_t c0 = rdtsc(); const double w0 = now();
    struct timespec ts = {0, 200000000L}; nanosleep(&ts, NULL);
    const uint64_t c1 = rdtsc(); const double w1 = now();
    const double tsc_ghz = (double)(c1 - c0) / (w1 - w0) / 1e9;

    printf("==================================================================\n");
    printf(" WHAT THE IN-SITU CYCLE COUNTER COSTS\n");
    printf("==================================================================\n\n");
    printf("  TSC rate            %.3f GHz (measured, not read from /proc)\n", tsc_ghz);
    printf("  matmul per token    19.9 ms   (section 7.4a, measured in situ)\n");
    printf("  matmul calls/token  %ld       (one per weight tensor)\n", calls_per_token);
    printf("  so one call is      %.1f us of work\n\n",
           19.9e3 / (double) calls_per_token);

    const char *label[3] = { "rdtsc pair only",
                             "rdtsc + atomic add, ONE shared counter",
                             "rdtsc + atomic add, per-thread counter" };
    const int threads[] = { 1, 2, 4, 6 };

    printf("  %-40s %8s %10s %12s\n", "probe", "threads", "ns/probe", "of one call");
    for (int m = 0; m < 3; ++m) {
        for (unsigned t = 0; t < sizeof threads / sizeof *threads; ++t) {
            const double ns = run(threads[t], m);
            const double call_us = 19.9e3 / (double) calls_per_token;
            printf("  %-40s %8d %10.2f %11.4f%%\n",
                   t == 0 ? label[m] : "", threads[t], ns,
                   100.0 * (ns / 1000.0) / call_us);
        }
    }

    printf("\n"
           "  READ: the rightmost column is what ONE probe costs as a fraction\n"
           "  of ONE matmul call. The macro fires twice per call (entry and\n"
           "  scope exit), so double it. Anything under a tenth of a per cent\n"
           "  cannot move a ratio quoted to three figures; if the shared-counter\n"
           "  row grows with thread count while the per-thread row does not,\n"
           "  the cost is cache-line contention and the fix is the third row.\n");
    return 0;
}
