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

/* 0 = not yet registered. Registration must happen exactly once even though
 * every ggml worker thread reaches these entry points; a plain lazy flag lets
 * several through and prints the counter line once per registration, which
 * reads like a defect in the evidence rather than in the harness. Same
 * compare-exchange as ggml_i2s_ternary.c:36-60, for the same reason. */
static int g_t5b_registered = 0;

void bitnet_t5b_report(void)
{
    fprintf(stderr,
            "[bitnet-t5b] type=43 bits/weight=1.600 calls=%llu sgemm=%llu\n",
            (unsigned long long) __atomic_load_n(&bitnet_t5b_calls,
                                                 __ATOMIC_RELAXED),
            (unsigned long long) __atomic_load_n(&bitnet_t5b_sgemm_calls,
                                                 __ATOMIC_RELAXED));
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
