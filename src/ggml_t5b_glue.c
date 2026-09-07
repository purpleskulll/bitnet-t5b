/*
 * ggml_t5b_glue.c -- ggml entry points for GGML_TYPE_T5B. See the header for
 * the three decisions this file owns (row stride, sum(a) convention,
 * counters) and why each of them is silent when wrong.
 *
 * Everything here is a thin adapter over ternary_t5b.{c,h}. No decoding is
 * re-derived: the packing rule, the magic-division bounds, the 12-block int16
 * fold and the zero-padded tail block all live in the tested kernel, and
 * re-implementing any of them here would create a second derivation that
 * could drift from the one the 24,168 assertions cover.
 */

#include "ggml_t5b_glue.h"
#include "ternary_t5b.h"

#include <stdio.h>
#include <stdlib.h>

uint64_t bitnet_t5b_calls       = 0;
uint64_t bitnet_t5b_sgemm_calls = 0;

/* Cycles spent INSIDE the matmul entry points, summed across worker threads.
 *
 * WHY THIS EXISTS
 * ---------------
 * Section 5 of the paper predicts, from S = 2.43 and the surplus table, that
 * this format loses at four threads; the standalone weight-traffic replay
 * agrees and gives 0.780x; the real model gains 1.132x. Something between the
 * isolated kernels and the model changes the sign, and the paper says so
 * without being able to say what.
 *
 * The two candidates are distinguishable by one number: how long the matmuls
 * actually take in situ. If the in-situ ratio is far below the isolated 2.43,
 * llama.cpp's real i2_s path is slower than the reference kernel section 7.2
 * measures and S was the wrong quantity. If it is near 2.43, then the surplus
 * is the wrong quantity instead and the memory left to the weight stream in a
 * real run is smaller than the isolated measurement suggests.
 *
 * THIS FILE USED TO BE THE ONLY INSTRUMENTED ONE, and the i2_s time was
 * obtained by DIFFERENCE instead:
 *
 *     T_other       = T_total(t5b) - T_matmul(t5b)     [both measured here]
 *     T_matmul(i2s) = T_total(i2s) - T_other           [T_total measured]
 *
 * The stated reason was that the control arm must stay unperturbed. A reviewer
 * rejected that and was right: the two T_total figures come from two separate
 * llama-bench runs on a host at load 2 to 10 (4.905 s against 5.479 s per rep),
 * so every load fluctuation BETWEEN the runs lands entirely in the i2_s matmul
 * number. The resulting 1.606x is softer than the paragraph quoting it sounds,
 * and it cannot separate the three candidate causes -- dispatch, accumulator
 * fold, per-column post-processing -- because it produces one number for all
 * three together. The integration now carries a SYMMETRIC pair of counters
 * (bitnet_sgemm_cycles_add, wired into the sgemm dispatch site by
 * scripts/apply_integration.py) so the ratio is a division of two measured
 * quantities rather than of one measured and one inferred.
 *
 * THE COUNTER IS PER-THREAD, and that is not premature tidying. It was one
 * shared global, and bench_probe_cost.c measures what that costs: 18.7 ns per
 * probe at one thread rising to 87.6 ns at six, a factor of 4.7, because every
 * ggml worker issues an atomic read-modify-write to the same cache line and
 * they serialise on the coherence protocol rather than on the instruction. The
 * padded per-thread form measures flat at ~19 ns from one thread to six.
 *
 * Neither figure endangers the measurement -- against the 94.8 us a single
 * matmul call takes, the worst case is 0.09 % per probe and 0.18 % for the
 * entry/exit pair -- but the shared counter's cost GREW WITH THREAD COUNT,
 * which is exactly the axis section 7.4 compares along, and a systematic error
 * that tracks the independent variable is the kind worth removing even when it
 * is small. */
#define T5B_MAX_THREADS 64

static inline uint64_t t5b_rdtsc(void)
{
    unsigned lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t) hi << 32) | lo;
}


struct t5b_counter { uint64_t v; char pad[64 - sizeof(uint64_t)]; };
static struct t5b_counter g_t5b_cycles[T5B_MAX_THREADS];

/* Slot per thread, assigned once on first use. __thread and not a hash of
 * pthread_self(): ggml reuses its worker pool, so the assignment survives, and
 * an id collision would silently merge two threads' counters back onto one
 * line and reintroduce exactly the contention this removes. */
static _Thread_local int t5b_slot = -1;
static int g_t5b_next_slot = 0;

static inline int t5b_my_slot(void)
{
    if (t5b_slot < 0) {
        const int s = __atomic_fetch_add(&g_t5b_next_slot, 1, __ATOMIC_RELAXED);
        /* More threads than slots is not a reason to lose counts: fall back to
         * slot 0, which is contended but correct. Silently dropping the sample
         * would understate the matmul time, which is the direction that
         * flatters this work. */
        t5b_slot = (s < T5B_MAX_THREADS) ? s : 0;
    }
    return t5b_slot;
}

uint64_t bitnet_t5b_cycles_total(void)
{
    uint64_t s = 0;
    for (int i = 0; i < T5B_MAX_THREADS; ++i)
        s += __atomic_load_n(&g_t5b_cycles[i].v, __ATOMIC_RELAXED);
    return s;
}

/* The symmetric pair, timed at the SGEMM DISPATCH SITE rather than inside the
 * kernels -- [0] = i2_s, [1] = t5b. Both arms are probed at the same point in
 * the same expression, so whatever the dispatch itself costs is common to both
 * and divides out of the ratio. That is the property the difference method
 * lacked and the reason this is not simply "the i2_s counter": measuring one
 * arm inside its kernel and the other at its call site would have swapped one
 * asymmetry for another.
 *
 * These are separate from g_t5b_cycles above, which stays because it covers the
 * vec_dot entry points that have no dispatch-site equivalent. Nothing is
 * double-counted: they are different counters answering different questions. */
static struct t5b_counter g_sgemm_cycles[2][T5B_MAX_THREADS];

uint64_t bitnet_probe_tsc(void) { return t5b_rdtsc(); }

void bitnet_sgemm_cycles_add(int is_t5b, uint64_t t0)
{
    __atomic_fetch_add(&g_sgemm_cycles[is_t5b ? 1 : 0][t5b_my_slot()].v,
                       t5b_rdtsc() - t0, __ATOMIC_RELAXED);
}

uint64_t bitnet_sgemm_cycles_total(int is_t5b)
{
    uint64_t s = 0;
    for (int i = 0; i < T5B_MAX_THREADS; ++i)
        s += __atomic_load_n(&g_sgemm_cycles[is_t5b ? 1 : 0][i].v,
                             __ATOMIC_RELAXED);
    return s;
}

static inline void t5b_acc(const uint64_t *t0)
{
    __atomic_fetch_add(&g_t5b_cycles[t5b_my_slot()].v, t5b_rdtsc() - *t0,
                       __ATOMIC_RELAXED);
}

/* GCC's cleanup attribute, not an accumulate before each return: these entry
 * points have early exits (an empty row range, a zero-sized tile), and an
 * accumulate written at the end of the body would silently miss them and
 * undercount. The attribute fires on every scope exit. rdtsc is unguarded
 * because this translation unit is AVX2-only by construction. */
#define T5B_TIME() \
    const uint64_t t5b_t0 __attribute__((cleanup(t5b_acc))) = t5b_rdtsc()

/* 0 = not yet registered. Registration must happen exactly once even though
 * every ggml worker thread reaches these entry points; a plain lazy flag lets
 * several through and prints the counter line once per registration, which
 * reads like a defect in the evidence rather than in the harness. Same
 * compare-exchange as ggml_i2s_ternary.c:36-60, for the same reason. */
static int g_t5b_registered = 0;

void bitnet_t5b_report(void)
{
    fprintf(stderr,
            "[bitnet-t5b] type=43 bits/weight=1.600 calls=%llu sgemm=%llu "
            "matmul_cycles=%llu sgemm_cycles_i2s=%llu sgemm_cycles_t5b=%llu\n",
            (unsigned long long) __atomic_load_n(&bitnet_t5b_calls,
                                                 __ATOMIC_RELAXED),
            (unsigned long long) __atomic_load_n(&bitnet_t5b_sgemm_calls,
                                                 __ATOMIC_RELAXED),
            (unsigned long long) bitnet_t5b_cycles_total(),
            (unsigned long long) bitnet_sgemm_cycles_total(0),
            (unsigned long long) bitnet_sgemm_cycles_total(1));
}

static void t5b_register_report(void)
{
    int expected = 0;
    if (__atomic_compare_exchange_n(&g_t5b_registered, &expected, 1,
                                    0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        atexit(bitnet_t5b_report);
    }
}

static void t5b_tick(uint64_t *counter)
{
    t5b_register_report();
    __atomic_fetch_add(counter, 1, __ATOMIC_RELAXED);
}

/* ---------------------------------------------------------------- geometry */

int64_t ggml_t5b_row_bytes(int64_t ne00)
{
    return (int64_t) ternary_t5b_size((size_t) ne00);
}

int64_t ggml_t5b_nbytes(int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03)
{
    /* + 32 for the scale footer, which t5b carries over from i2_s unchanged.
     * Must agree byte for byte with ggml_nbytes()'s T5B arm and with
     * tools/gguf_to_t5b.py:145, or the file is rejected at load. */
    return ggml_t5b_row_bytes(ne00) * ne01 * ne02 * ne03 + 32;
}

/* The stride a caller believes against the stride the kernel will actually
 * use. These are the same number computed in two places -- ggml-cpu.c derives
 * it from ne00, the kernel from the n it is handed -- and risk 4.1 is exactly
 * the case where they disagree and nothing crashes. Checking costs one
 * comparison per matmul call and converts the quietest failure in this
 * integration into an abort with the two numbers in it. */
static void t5b_check_stride(int64_t k, int64_t a_row_bytes, const char *where)
{
    const int64_t want = ggml_t5b_row_bytes(k);
    if (a_row_bytes != want) {
        fprintf(stderr,
                "[bitnet-t5b] FATAL: %s was given a row stride of %lld bytes "
                "for k=%lld, but a t5b row is %lld bytes. Reading at the wrong "
                "stride does not fault -- the tensors are mmap'd contiguously "
                "-- it returns fluent, wrong numbers. Refusing.\n",
                where, (long long) a_row_bytes, (long long) k,
                (long long) want);
        abort();
    }
}

/* ----------------------------------------------------------------- vec_dot */

/* Rows per gemv call on the contiguous fast path. The caller's block is 16
 * (ggml-cpu.c:1207 blck_0); 32 leaves headroom without putting a large buffer
 * on a ggml worker stack. */
#define T5B_VEC_DOT_CHUNK 32

void ggml_vec_dot_t5b_i8_s(int n, float *s, size_t bs,
                           const void *vx, size_t bx,
                           const void *vy, size_t by, int nrc)
{
    const uint8_t *w = (const uint8_t *) vx;
    const int8_t  *a = (const int8_t *)  vy;
    const size_t rowbytes = ternary_t5b_size((size_t) n);

    t5b_tick(&bitnet_t5b_calls);
    T5B_TIME();

    if (nrc <= 0) {
        return;
    }

    /* Fast path: consecutive rows at the packed stride, all against the same
     * activation column, results written contiguously. That is precisely what
     * ggml-cpu.c:1240 asks for, and routing it through the GEMV entry point
     * builds the zero-padded tail block ONCE for the whole call instead of
     * once per row -- at K = 6912 (43 whole blocks plus 32 weights) every row
     * of every matmul takes that path, so it is not a rare case.
     *
     * sum_a = 0: return sum(code*a), the i2_s contract. See the header. */
    if (by == 0 && bs == 1 && bx == rowbytes && nrc > 1) {
        int32_t y[T5B_VEC_DOT_CHUNK];
        int r = 0;
        while (r < nrc) {
            int c = nrc - r;
            if (c > T5B_VEC_DOT_CHUNK) {
                c = T5B_VEC_DOT_CHUNK;
            }
            ternary_t5b_gemv_avx2(w + (size_t) r * rowbytes, a, y,
                                  (size_t) c, (size_t) n, 0);
            for (int i = 0; i < c; ++i) {
                s[(size_t) (r + i)] = (float) y[i];
            }
            r += c;
        }
        return;
    }

    /* General path. Honours bx and by as byte strides, which covers the
     * single-row call at ggml-cpu.c:1249 (bs = bx = by = 0, nrc = 1) and any
     * future caller that pairs rows of A with rows of B. */
    for (int r = 0; r < nrc; ++r) {
        s[(size_t) r * bs] =
            (float) ternary_t5b_dot_avx2(w + (size_t) r * bx,
                                         a + (size_t) r * by,
                                         (size_t) n, 0);
    }
}

/* -------------------------------------------------------------- gemv / gemm */

void ggml_gemv_t5b_i8_s(int n, float *s, size_t bs,
                        const void *vx, const void *vy, int nr, int nc)
{
    /* nr is the activation-column count and is 1 on this path by construction
     * (ggml-cpu.c:1538). Anything else would mean the caller wanted the GEMM. */
    (void) bs;
    (void) nr;

    t5b_tick(&bitnet_t5b_calls);
    T5B_TIME();

    if (nc <= 0) {
        return;
    }

    {
        const size_t rowbytes = ternary_t5b_size((size_t) n);
        int32_t y[T5B_VEC_DOT_CHUNK];
        int r = 0;
        while (r < nc) {
            int c = nc - r;
            if (c > T5B_VEC_DOT_CHUNK) {
                c = T5B_VEC_DOT_CHUNK;
            }
            ternary_t5b_gemv_avx2((const uint8_t *) vx + (size_t) r * rowbytes,
                                  (const int8_t *) vy, y,
                                  (size_t) c, (size_t) n, 0);
            for (int i = 0; i < c; ++i) {
                s[(size_t) (r + i)] = (float) y[i];
            }
            r += c;
        }
    }
}

/* --------------------------------------------------------- the GEMM tile */

/* THE DEFECT THIS REPLACES, and it was in this file rather than in the kernel.
 *
 * Both GEMM entry points used to loop ternary_t5b_gemv_avx2 once per activation
 * column over a chunk of rows. The row-chunk-outside nesting was right and is
 * kept below -- it is what holds the weight chunk in L2 across the columns --
 * but the body re-ran the ENTIRE t5b decode for every column: eight vpmulhuw
 * per 32-byte block, the four digit subtractions, the plane recombination, all
 * of it, once per column over the same weights. For i2_s that would be nearly
 * free, because its decode is three shifts and four masks. For t5b the decode
 * IS the kernel, and multiplying it by the column count is exactly backwards.
 *
 * ternary_t5b_gemm_avx2 decodes each block once and contracts it against a
 * strip of columns held in registers, so per 32-byte block
 *
 *     NC * (decode + 5 contractions)     becomes     decode + NC * 5
 *
 * and the measured effect on this host is 2.31x to 2.37x at K = 2560 and 6912
 * for 16 to 256 columns, and 1.06x at ONE column (see the table in
 * ternary_t5b.c). Never a loss, so there is no column count below which this
 * should fall back to the GEMV. Generation is untouched either way: it does not
 * reach these two functions at all -- ggml_gemv_t5b_i8_s and
 * ggml_vec_dot_t5b_i8_s still take the GEMV path they were measured on.
 *
 * WHY THE STAGING BUFFER. The kernel returns int32 and ggml wants float, and
 * the two post-processings below differ (the sgemm path applies the scale and
 * the activation-sum subtraction, this one does not). Rather than teach the
 * kernel about either, the results land in a small column-major int32 tile and
 * are converted while they are still in L1. 32 x 32 int32 is 4 KB on a ggml
 * worker stack, which has megabytes; the alternative -- one buffer sized
 * rows*cols -- would be an allocation on the hot path, and writing int32 into
 * the caller's float array and converting in place would alias two types
 * through the same pointer. */
#define T5B_GEMM_TILE_ROWS 32
#define T5B_GEMM_TILE_COLS 32

void ggml_gemm_t5b_i8_s(int n, float *s, size_t bs,
                        const void *vx, const void *vy, int nr, int nc)
{
    const size_t rowbytes = ternary_t5b_size((size_t) n);

    t5b_tick(&bitnet_t5b_calls);
    T5B_TIME();

    if (nc <= 0 || nr <= 0) {
        return;
    }

    /* Row-chunk OUTSIDE, column tile INSIDE. The obvious nesting (column
     * outside) re-reads the whole weight matrix once per column and turns an
     * arithmetic-bound loop into a bandwidth-bound one; this way a chunk of
     * rows is streamed from DRAM once and every column contracts against it
     * out of L2.
     *
     * 32 rows at K = 6912 is 32*1408 = 44 KB, comfortably L2-resident on the
     * parts this project targets. */
    for (int r0 = 0; r0 < nc; r0 += T5B_GEMM_TILE_ROWS) {
        int rows = nc - r0;
        if (rows > T5B_GEMM_TILE_ROWS) {
            rows = T5B_GEMM_TILE_ROWS;
        }

        for (int c0 = 0; c0 < nr; c0 += T5B_GEMM_TILE_COLS) {
            int cols = nr - c0;
            if (cols > T5B_GEMM_TILE_COLS) {
                cols = T5B_GEMM_TILE_COLS;
            }

            int32_t tile[T5B_GEMM_TILE_ROWS * T5B_GEMM_TILE_COLS];

            /* Activation columns are n bytes apart (ggml-cpu-i2s.c:38-43).
             * sum_a is not a parameter here: the kernel returns sum(code*a),
             * the convention section 2 of the header derives. */
            ternary_t5b_gemm_avx2((const uint8_t *) vx + (size_t) r0 * rowbytes,
                                  (const int8_t *) vy + (size_t) c0 * n,
                                  (size_t) n,
                                  tile, (size_t) rows,
                                  (size_t) rows, (size_t) cols, (size_t) n);

            for (int c = 0; c < cols; ++c) {
                for (int i = 0; i < rows; ++i) {
                    s[(size_t) (c0 + c) * bs + (size_t) (r0 + i)] =
                        (float) tile[(size_t) c * rows + i];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------- sgemm */

void bitnet_t5b_sgemm(int64_t m, int64_t n, int64_t k,
                      const uint8_t *A, int64_t a_row_bytes,
                      const int8_t *B, int64_t ldb,
                      float *C, int64_t ldc,
                      int ith, int nth,
                      const float *act_scales, const int32_t *act_sums,
                      float weight_scale)
{
    t5b_check_stride(k, a_row_bytes, "bitnet_t5b_sgemm");

    /* Rows split across threads; each thread writes only its own rows of C,
     * so no barrier -- the contract llamafile_sgemm_i2s already has. */
    int64_t m0 = ( (int64_t) ith      * m) / nth;
    int64_t m1 = (((int64_t) ith + 1) * m) / nth;
    if (m0 >= m1) {
        return;
    }

    t5b_tick(&bitnet_t5b_sgemm_calls);
    T5B_TIME();

    /* THIS is the path the real model takes, and the reason the rewiring above
     * would have been worth nothing on its own. The counter line from the first
     * run of the 1.6-bit model reads
     *
     *     [bitnet-t5b] type=43 bits/weight=1.600 calls=0 sgemm=14280
     *
     * (results/inference_t5b.txt section 1): zero calls through
     * ggml_gemm_t5b_i8_s, every one of them through here. A blocked GEMM wired
     * only into the entry point ggml never reaches would have measured faster
     * in a benchmark and changed nothing in llama.cpp. */
    for (int64_t r0 = m0; r0 < m1; r0 += T5B_GEMM_TILE_ROWS) {
        int64_t rows = m1 - r0;
        if (rows > T5B_GEMM_TILE_ROWS) {
            rows = T5B_GEMM_TILE_ROWS;
        }

        for (int64_t j0 = 0; j0 < n; j0 += T5B_GEMM_TILE_COLS) {
            int64_t cols = n - j0;
            if (cols > T5B_GEMM_TILE_COLS) {
                cols = T5B_GEMM_TILE_COLS;
            }

            int32_t tile[T5B_GEMM_TILE_ROWS * T5B_GEMM_TILE_COLS];

            ternary_t5b_gemm_avx2(A + r0 * a_row_bytes, B + j0 * ldb,
                                  (size_t) ldb, tile, (size_t) rows,
                                  (size_t) rows, (size_t) cols, (size_t) k);

            /* The post-processing, applied HERE and nowhere else on this path:
             * the caller returns immediately after us. The tile holds
             * sum(code*a), so the -act_sums is the single application of the
             * offset, exactly as sgemm.cpp:1582 does it for i2_s. */
            for (int64_t c = 0; c < cols; ++c) {
                const int64_t  j = j0 + c;
                const int32_t *y = tile + c * rows;

                if (act_scales) {
                    const float post = weight_scale / act_scales[j];
                    const float asum = (float) act_sums[j];
                    for (int64_t i = 0; i < rows; ++i) {
                        C[ldc * j + r0 + i] = ((float) y[i] - asum) * post;
                    }
                } else {
                    for (int64_t i = 0; i < rows; ++i) {
                        C[ldc * j + r0 + i] = (float) y[i];
                    }
                }
            }
        }
    }
}
