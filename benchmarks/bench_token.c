/*
 * bench_token.c -- what one generated token costs in weight traffic, at the
 * model's real shapes and its real total footprint, in both weight formats.
 *
 * WHY THIS EXISTS
 * ---------------
 * results/thread_headroom.txt establishes a budget: at the thread counts this
 * project actually runs the model with, a core has more arithmetic capacity
 * than the memory system keeps it fed with, so a denser weight format can pay
 * for its own decode. A budget is not a result. This spends it.
 *
 * The shapes are the model's, read out of models/ggml-model-i2_s.gguf with
 * tools/gguf_inspect.py and hard-coded here so the benchmark needs no GGUF
 * parser:
 *
 *     210 i2_s tensors, 2,084,044,800 ternary weights
 *        60 x  K=2560, 6912 rows      (ffn_gate, ffn_up)
 *        60 x  K=2560,  640 rows      (attn_k, attn_v)
 *        60 x  K=2560, 2560 rows      (attn_q, attn_output)
 *        30 x  K=6912, 2560 rows      (ffn_down)
 *
 * ALL 210 ARE ALLOCATED, and that is the whole point of the file.
 *
 * A FIRST VERSION OF THIS BENCHMARK WAS WRONG, IN A WAY WORTH RECORDING
 * --------------------------------------------------------------------
 * It allocated one tensor per distinct SHAPE -- four buffers instead of 210 --
 * and timed each repeatedly, on the reasoning that the streaming behaviour
 * would be the same and 152 MB is cheaper than 940. It is not the same. The
 * largest of those buffers is 4.4 MB and the L3 slice is 16 MB, so every
 * repetition after the first read the weights out of cache. It reported
 * 53-60 GB/s, which no single socket on this host can deliver from DRAM, and
 * that impossible number is the only reason the error was caught. The property
 * that matters here is not the shape of a matmul, it is that 521 MB of weights
 * are COLD, and the only way to have that property is to have 521 MB.
 *
 * WHAT IT IS AND IS NOT
 *   IS:  the weight-streaming cost of one token's worth of matmuls, over the
 *        model's full weight footprint, threaded the way llama.cpp threads
 *        them (rows split across workers), with the activation-side work each
 *        format really needs -- upstream's none, and t10's b transform,
 *        charged once per tensor as a GEMV would.
 *   NOT: a token. There is no attention, no KV cache, no norm, no sampling and
 *        no runtime. Those cost time too, and they are the same for both arms,
 *        so a ratio here is an upper bound on the ratio a real token would
 *        show, never a claim about tokens per second.
 *
 * The weights are random ternary rather than the model's own. Every byte is
 * still a distinct byte streamed from DRAM, which is the quantity being
 * measured; the VALUES would matter only if either kernel were data-dependent,
 * and neither is -- both do the same work on every input.
 *
 * Build:  gcc -O3 -mavx2 -mfma -march=native -std=c11 -pthread \
 *             bench/bench_token.c build/obj/ggml_i2s_ternary.o \
 *             build/obj/ternary_t10.o -o build/bench_token
 * Usage:  ./build/bench_token [threads] [reps] [scale]
 *
 * `scale` multiplies how many tensors of each shape are allocated, emulating a
 * DEEPER model with the same layer shapes -- scale 2 is twice the layers and
 * twice the weight bytes per token. It exists to answer whether the advantage
 * of a denser format grows with model size, which cannot be settled by argument:
 * generation reads every weight once per token, so a bigger model is more
 * memory-bound, but the arithmetic-bound kernel also has more work to do and
 * the two move together. Measure it.
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "../ggml_i2s_ternary.h"
#include "../ternary_t10.h"
#include "../ternary_t5b.h"
#include "i2s_tiled.h"

#define MAX_THREADS 12

static const int CPU_ORDER[6] = {0, 3, 1, 4, 2, 5};

typedef struct { size_t k, rows, count; const char *what; } shape;

/* dims=[ne0, ne1] with ne0 the row length, so ne0 is K and ne1 the row count. */
static const shape SHAPES[] = {
    {2560, 6912, 60, "ffn_gate / ffn_up"},
    {2560,  640, 60, "attn_k / attn_v"},
    {2560, 2560, 60, "attn_q / attn_output"},
    {6912, 2560, 30, "ffn_down"},
};
#define NSHAPES ((int)(sizeof(SHAPES) / sizeof(SHAPES[0])))

typedef struct {
    size_t   k, rows;
    uint8_t *Wi;      /* i2_s, k/4 bytes per row */
    uint8_t *Wt;      /* t10,  ternary_t10_size(k) bytes per row */
    uint8_t *Wb;      /* t5b,  ternary_t5b_size(k) bytes per row */
    const int8_t  *a;
    const int16_t *b;
    int32_t  sum_a;
    float   *yf;
    int32_t *yi;
} tensor;

static tensor TENS[2048];
static int    NTENS = 0;

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint32_t rs = 0x9E3779B9u;
static uint32_t rng_next(void)
{
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs;
}

static void pack_row_i2s(const int8_t *w, uint8_t *d, size_t n)
{
    const size_t rb = n / 4;
    memset(d, 0, rb);
    for (size_t blk = 0; blk * 128 < n; ++blk) {
        for (size_t j = 0; j < 32; ++j) {
            uint8_t byte = 0;
            for (int q = 0; q < 4; ++q) {
                const size_t idx = blk * 128 + j + (size_t)q * 32;
                const uint8_t code = (idx < n) ? (uint8_t)(w[idx] + 1) : 1u;
                byte |= (uint8_t)(code << (6 - 2 * q));
            }
            d[blk * 32 + j] = byte;
        }
    }
}

typedef struct {
    int    arm;        /* 0 = i2_s vec_dot, 1 = t10, 2 = t5b, 3 = i2_s tiled4 */
    int    id, nthreads;
    int    cpu;
    pthread_barrier_t *start;
} job;

/* Each worker walks EVERY tensor and does its own row slice of each. No
 * barrier between tensors: this is a traffic replay, the matmuls do not feed
 * one another, and a barrier per tensor would put 210 synchronisations inside
 * the measurement. Threads are created once per timed pass, not once per
 * tensor -- an earlier version created 840 of them per token and was measuring
 * pthread_create. */
static void *worker(void *vp)
{
    job *j = (job *)vp;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(j->cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
        fprintf(stderr, "warning: could not pin to cpu %d\n", j->cpu);
    pthread_barrier_wait(j->start);

    for (int ti = 0; ti < NTENS; ++ti) {
        const tensor *t = &TENS[ti];
        const size_t per = (t->rows + (size_t)j->nthreads - 1) / (size_t)j->nthreads;
        const size_t r0  = (size_t)j->id * per;
        if (r0 >= t->rows) continue;
        const size_t nr  = (r0 + per <= t->rows) ? per : t->rows - r0;

        if (j->arm == 0) {
            bitnet_vec_dot_i2_i8_s_reference((int)t->k, t->yf + r0, 1,
                                             t->Wi + r0 * (t->k / 4),
                                             t->k, t->a, 0, (int)nr);
        } else if (j->arm == 1) {
            const size_t rb = ternary_t10_size(t->k);
            for (size_t r = 0; r < nr; ++r) {
                t->yi[r0 + r] = ternary_t10_dot_avx2(t->Wt + (r0 + r) * rb,
                                                     t->b, t->k, t->sum_a);
            }
        } else if (j->arm == 3) {
            /* The kernel llama.cpp actually runs: tinyBLAS_I2S_AVX tiles four
             * weight rows against one activation column, so the four
             * activation loads per block are paid once instead of four times.
             * Arm 0 is the vec_dot branch, which GGML_LLAMAFILE=ON leaves dead. */
            i2s_tiled_gemv(t->Wi + r0 * (t->k / 4), t->a, t->yi + r0, nr, t->k);
        } else {
            /* t5b takes the RAW activations: five trits per byte means the
             * digits come out as bytes and vpmaddubsw eats them directly, so
             * there is no b transform to prepare.
             *
             * Called through the GEMV entry point rather than a loop over the
             * per-row kernel. That is not tiling -- the recommended width is 1,
             * because a GEMV has one activation column and interleaving rows
             * buys no reuse at all, only a worse access pattern. What it does
             * hoist out of the per-row call is the five decode constants and,
             * more importantly, the zeroed 160-byte tail buffer: the per-row
             * kernel rebuilds that with a memset and a memcpy for EVERY row,
             * and at K = 6912 -- 43 whole blocks plus 32 weights -- the real
             * model takes that path on every row of every GEMV. Worth 1.03x. */
            ternary_t5b_gemv_avx2(t->Wb + r0 * ternary_t5b_size(t->k),
                                  t->a, t->yi + r0, nr, t->k, t->sum_a);
        }
    }
    return NULL;
}

/* One full token pass over all 210 tensors. Returns seconds. */
static double one_token(int arm, int T)
{
    pthread_t th[MAX_THREADS];
    job jobs[MAX_THREADS];
    pthread_barrier_t start;
    pthread_barrier_init(&start, NULL, (unsigned)T + 1);

    for (int i = 0; i < T; ++i) {
        jobs[i].arm = arm; jobs[i].id = i; jobs[i].nthreads = T;
        jobs[i].cpu = CPU_ORDER[i % 6]; jobs[i].start = &start;
        if (pthread_create(&th[i], NULL, worker, &jobs[i]) != 0) {
            fprintf(stderr, "pthread_create failed\n"); exit(2);
        }
    }
    pthread_barrier_wait(&start);
    const double t0 = now_sec();
    for (int i = 0; i < T; ++i) pthread_join(th[i], NULL);
    const double el = now_sec() - t0;
    pthread_barrier_destroy(&start);
    return el;
}

int main(int argc, char **argv)
{
    const int T     = (argc > 1) ? atoi(argv[1]) : 4;
    const int reps  = (argc > 2) ? atoi(argv[2]) : 3;
    const int scale = (argc > 3) ? atoi(argv[3]) : 1;
    if (scale < 1 || scale > 8) { fprintf(stderr, "scale 1..8\n"); return 2; }
    if (T < 1 || T > MAX_THREADS) { fprintf(stderr, "threads 1..%d\n", MAX_THREADS); return 2; }

    printf("bench_token: one token's weight traffic, the model's full footprint"
           "%s\n", scale > 1 ? " x scale" : "");
    printf("threads=%d (published runs use -t 4), %d token passes per arm, "
           "arms interleaved\n\n", T, reps);

    /* Activations are shared across tensors of the same K -- they are read-only
     * and tiny, and sharing them keeps the weight arrays the only thing large
     * enough to matter. Two K values occur. */
    int8_t  *a2560 = aligned_alloc(64, 2560);
    int8_t  *a6912 = aligned_alloc(64, 6912);
    int16_t *b2560 = aligned_alloc(64, ternary_t10_slots(2560) * sizeof(int16_t));
    int16_t *b6912 = aligned_alloc(64, ternary_t10_slots(6912) * sizeof(int16_t));
    if (!a2560 || !a6912 || !b2560 || !b6912) { fprintf(stderr, "alloc\n"); return 2; }
    for (size_t i = 0; i < 2560; ++i) a2560[i] = (int8_t)((int)(rng_next() % 255) - 127);
    for (size_t i = 0; i < 6912; ++i) a6912[i] = (int8_t)((int)(rng_next() % 255) - 127);
    ternary_t10_prep_b(a2560, b2560, 2560);
    ternary_t10_prep_b(a6912, b6912, 6912);
    int32_t s2560 = 0, s6912 = 0;
    for (size_t i = 0; i < 2560; ++i) s2560 += a2560[i];
    for (size_t i = 0; i < 6912; ++i) s6912 += a6912[i];

    size_t tot_i = 0, tot_t = 0, tot_b = 0, tot_w = 0;
    int8_t *row = aligned_alloc(64, 6912 + 64);
    if (!row) { fprintf(stderr, "alloc\n"); return 2; }

    printf("allocating and packing 210 tensors ...\n");
    const double pk0 = now_sec();
    for (int s = 0; s < NSHAPES; ++s) {
        for (size_t c = 0; c < SHAPES[s].count * (size_t)scale; ++c) {
            const size_t k = SHAPES[s].k, rows = SHAPES[s].rows;
            const size_t rb_i = k / 4, rb_t = ternary_t10_size(k);
            const size_t rb_b = ternary_t5b_size(k);
            if (NTENS >= (int)(sizeof(TENS)/sizeof(TENS[0]))) {
                fprintf(stderr, "tensor table full at %d\n", NTENS); return 2;
            }
            tensor *t = &TENS[NTENS++];
            t->k = k; t->rows = rows;
            t->Wi = aligned_alloc(64, ((rows * rb_i) + 63) & ~63ull);
            t->Wt = aligned_alloc(64, ((rows * rb_t) + 63) & ~63ull);
            t->Wb = aligned_alloc(64, ((rows * rb_b) + 63) & ~63ull);
            t->yf = aligned_alloc(64, rows * sizeof(float));
            t->yi = aligned_alloc(64, rows * sizeof(int32_t));
            if (!t->Wi || !t->Wt || !t->Wb || !t->yf || !t->yi) {
                fprintf(stderr, "allocation failed at tensor %d\n", NTENS); return 2;
            }
            t->a     = (k == 2560) ? a2560 : a6912;
            t->b     = (k == 2560) ? b2560 : b6912;
            t->sum_a = (k == 2560) ? s2560 : s6912;

            for (size_t r = 0; r < rows; ++r) {
                for (size_t i = 0; i < k; ++i)
                    row[i] = (int8_t)((int)(rng_next() % 3) - 1);
                pack_row_i2s(row, t->Wi + r * rb_i, k);
                ternary_t10_pack(row, t->Wt + r * rb_t, k);
                ternary_t5b_pack(row, t->Wb + r * rb_b, k);
            }
            tot_i += rows * rb_i; tot_t += rows * rb_t;
            tot_b += rows * rb_b; tot_w += rows * k;
        }
    }
    printf("  %d tensors, %zu weights, i2_s %.1f MB, t10 %.1f MB, t5b %.1f MB"
           "  (%.0f s)\n\n", NTENS, tot_w, (double)tot_i / 1e6,
           (double)tot_t / 1e6, (double)tot_b / 1e6, now_sec() - pk0);

    /* Agreement on one row of every distinct shape before either arm is timed.
     * A throughput comparison between a right answer and a wrong one is not a
     * comparison, and K=6912 is 43 whole blocks plus a 32-weight tail, which
     * is exactly where a packing bug would hide. */
    for (int s = 0, seen = 0, ti = 0; s < NSHAPES; ti += (int)SHAPES[s].count, ++s) {
        const tensor *t = &TENS[ti];
        const size_t rb_t = ternary_t10_size(t->k);
        bitnet_vec_dot_i2_i8_s_reference((int)t->k, t->yf, 1, t->Wi, t->k,
                                         t->a, 0, 4);
        for (int r = 0; r < 4; ++r) {
            const int32_t want = (int32_t)t->yf[r] - t->sum_a;
            const int32_t got  = ternary_t10_dot_avx2(t->Wt + (size_t)r * rb_t,
                                                      t->b, t->k, t->sum_a);
            const int32_t gob  = ternary_t5b_dot_avx2(
                                     t->Wb + (size_t)r * ternary_t5b_size(t->k),
                                     t->a, t->k, t->sum_a);
            if (want != got || want != gob) {
                fprintf(stderr, "FATAL: arms disagree, K=%zu row %d: "
                        "i2_s %d, t10 %d, t5b %d\n", t->k, r, want, got, gob);
                return 3;
            }
            ++seen;
        }
        if (s == NSHAPES - 1) printf("cross-check: %d rows agree across all "
                                     "%d shapes\n\n", seen, NSHAPES);
    }

    double best_i = 1e30, best_t = 1e30, best_b = 1e30, best_q = 1e30;
    printf("  pass   vecdot     t10       t5b     tiled4\n");
    for (int r = 0; r < reps; ++r) {
        /* Interleaved AND rotated. Interleaving alone is not enough: with a
         * fixed order the arm measured last benefits from any clock ramp that
         * happens during the pass, and on this host that is not hypothetical.
         * A run with three passes in fixed order once reported t10 at 12.97 ms
         * against i2_s at 13.57 -- a 1.047x "win" -- with the per-pass times
         * falling 15.18, 13.46, 12.97 as the machine came up to speed. Five
         * passes on a settled machine put t10 at 15.89. Rotating the order
         * gives every arm each position an equal number of times. */
        double e[4];
        for (int s = 0; s < 4; ++s) {
            const int arm = (r + s) % 4;
            e[arm] = one_token(arm, T);
        }
        if (e[0] < best_i) best_i = e[0];
        if (e[1] < best_t) best_t = e[1];
        if (e[2] < best_b) best_b = e[2];
        if (e[3] < best_q) best_q = e[3];
        printf("  %4d %9.2f %9.2f %9.2f %9.2f  (order %d%d%d%d)\n", r + 1,
               e[0] * 1e3, e[1] * 1e3, e[2] * 1e3, e[3] * 1e3,
               r % 4, (r + 1) % 4, (r + 2) % 4, (r + 3) % 4);
    }

    printf("\n  ONE TOKEN, all %d i2_s tensors, %d threads, best of %d\n",
           NTENS, T, reps);
    printf("      i2_s  2.000 bits/weight   %8.1f MB   %8.2f ms   %6.2f GB/s\n",
           (double)tot_i / 1e6, best_i * 1e3, (double)tot_i / best_i / 1e9);
    printf("      t10   %.4f bits/weight   %8.1f MB   %8.2f ms   %6.2f GB/s\n",
           8.0 * (double)tot_t / (double)tot_w,
           (double)tot_t / 1e6, best_t * 1e3, (double)tot_t / best_t / 1e9);
    printf("      t5b   %.4f bits/weight   %8.1f MB   %8.2f ms   %6.2f GB/s\n",
           8.0 * (double)tot_b / (double)tot_w,
           (double)tot_b / 1e6, best_b * 1e3, (double)tot_b / best_b / 1e9);

    printf("      i2_s TILED (the live kernel shape) %8.1f MB   %8.2f ms   %6.2f GB/s\n",
           (double)tot_i / 1e6, best_q * 1e3, (double)tot_i / best_q / 1e9);
    printf("      t10 vs vecdot %.3fx   t5b vs vecdot %.3fx\n",
           best_i / best_t, best_i / best_b);
    printf("      t10 vs TILED  %.3fx   t5b vs TILED  %.3fx      <- the honest one\n",
           best_q / best_t, best_q / best_b);
    printf("\n  The byte ratio is fixed by the formats. The time ratio is what\n");
    printf("  the decode costs against what the bytes buy: at the byte ratio\n");
    printf("  the decode is free, below it the decode eats part of the saving,\n");
    printf("  and at 1.000 the two formats are worth the same.\n");
    return 0;
}
