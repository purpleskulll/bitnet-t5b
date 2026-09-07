/*
 * ggml_t5b_glue.h -- the ggml-facing entry points for GGML_TYPE_T5B (= 43),
 *                    ternary weights at 1.600 bits each.
 *
 * ternary_t5b.{c,h} hold the packing and the kernels and are NOT touched by
 * this integration: they are tested (24,168 assertions) and they know nothing
 * about ggml. This file is the adapter. It owns exactly three things --
 * the row stride, the sum(a) convention, and the call counters -- and every
 * one of them is a place where being wrong is silent.
 *
 *
 * 1. THE ROW STRIDE, AND WHY IT IS A FUNCTION AND NOT AN EXPRESSION
 * -----------------------------------------------------------------
 * i2_s packs 4 weights per byte, so its row stride is ne00/4 and that
 * expression is written out at roughly fourteen sites across ggml-cpu.c,
 * ggml-cpu-i2s.c, sgemm.cpp and repack.cpp. t5b packs 160 weights per 32
 * bytes, so its stride is ceil(ne00/160)*32 -- and for the real model's
 * K = 6912 those differ by 320 bytes per row (1408 against 1728).
 *
 * If ONE site keeps a stale formula, row r is read at the wrong offset in a
 * buffer that is mmap'd contiguously with the next tensor: no segfault, no
 * NaN, fluent and wrong text. So there is exactly one definition of the
 * stride in this project's C -- ggml_t5b_row_bytes() -- and every patched
 * site calls it. A grep for "/ 4" in a patched hunk is then a bug report.
 *
 * The rounding is PER ROW, not per tensor. ceil(K*M/160)*32 would give
 * 3,538,944 bytes for a (6912, 2560) tensor where each row needs 1408, i.e.
 * 3,604,480. tools/gguf_to_t5b.py:145 rounds the same way (t5b_size(ne0)*ne1
 * + 32); the two formulas must agree or gguf.cpp's running-offset check
 * rejects the file at load, which is the loud failure we want.
 *
 *
 * 2. THE sum(a) CONVENTION -- THE ONE THAT CANNOT CRASH
 * ------------------------------------------------------
 * These functions return sum(code * a), NOT sum(w * a).
 *
 * That is deliberate and it is the i2_s contract, not the kernel's own.
 * The two conventions differ by a constant:
 *
 *     sum(code*a) = sum(w*a) + sum(a)        [code = w + 1, codes 0..2]
 *
 * ggml's post-processing subtracts the activation row sum downstream and
 * always has -- ggml-cpu.c:1244, :1528, :1547 and sgemm.cpp:1582 all read
 * "(dot - act_sums[col]) * scale". Returning sum(w*a) here would make that
 * subtraction fire a second time; returning sum(code*a) and leaving every
 * one of those lines untouched makes the double-subtraction unreachable
 * rather than merely tested for.
 *
 * The kernel is told which one to produce through its sum_a argument:
 * ternary_t5b.c:469 computes y[r] = part[r] - sum_a where part[r] is the raw
 * sum(code*a). So THIS FILE ALWAYS PASSES sum_a = 0. Nothing else in the
 * integration needs to know the convention exists.
 *
 * Why the padding cannot disturb it: a row of K weights occupies
 * ceil(K/160)*160 slots, and the slots past K carry code 1. The kernel zeroes
 * the activation tail internally (ternary_t5b.c:485-489), so those slots
 * contribute 1*0 = 0 to the code sum. sum(code*a) is therefore taken over
 * exactly the first K terms, which is the same range act_sums covers.
 *
 * The discriminating test, if this is ever doubted: a row of all-ZERO weights
 * packs every byte as 1*(1+3+9+27+81) = 121, and these functions must then
 * return exactly sum(a) -- after which ggml's "- act_sums" yields exactly
 * 0.0f for any activations. A missing correction gives +sum(a), a doubled one
 * -sum(a). Neither can hide behind a plausible-looking number.
 *
 *
 * 3. THE COUNTERS
 * ---------------
 * Same discipline as ggml_i2s_ternary.h:67,81 and for the same reason: a
 * build that silently did not carry the kernel is indistinguishable from one
 * that did until something counts. Unlike i2_s there is no A/B switch here --
 * a t5b tensor has no other kernel it could reach, and a binary without the
 * type cannot load the file at all (gguf.cpp rejects type 43) -- so these
 * counters answer "which path ran", not "did the switch work".
 */

#ifndef GGML_T5B_GLUE_H
#define GGML_T5B_GLUE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bytes per packed weight row, = ceil(ne00/160)*32. The single definition;
 * see section 1. Mirrors ternary_t5b_size(), which is the same rounding
 * expressed over a size_t, and is checked against it by the test suite. */
int64_t ggml_t5b_row_bytes(int64_t ne00);

/* Bytes of packed weight data for a whole tensor INCLUDING the 32-byte scale
 * footer -- what ggml_nbytes() must return for a t5b tensor. Exposed so the
 * patched call sites can assert their stride against the allocation instead
 * of trusting it (risk 4.1: the check that fires before a byte is read). */
int64_t ggml_t5b_nbytes(int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03);

/* vec_dot, the ggml type-traits entry point. Returns sum(code*a); see (2).
 *
 * IMPORTANT: bx is in BYTES, unlike the i2_s vec_dot, which is handed nb01
 * (an element count) and divides by 4 internally. Pushing the division into
 * the callee is what forced the "/ 4" to be repeated at every call site in
 * the first place; the patched sites pass ggml_t5b_row_bytes(ne00) instead.
 *
 * by is the activation stride between rows and is 0 on every path this type
 * reaches (nrows = 1 in the traits, so ggml never pairs rows of A with rows
 * of B). It is honoured anyway rather than ignored, so a future caller that
 * passes a non-zero one gets the right answer instead of a quiet wrong one. */
void ggml_vec_dot_t5b_i8_s(int n, float *s, size_t bs,
                           const void *vx, size_t bx,
                           const void *vy, size_t by, int nrc);

/* GEMV / GEMM, signature-compatible with ggml_gemv_i2_i8_s and
 * ggml_gemm_i2_i8_s (ggml-cpu-i2s.h:10-11) so the dispatch at ggml-cpu.c:1518
 * and :1538 needs no reshaping. Layout, as documented at ggml-cpu-i2s.c:38-43:
 *
 *   n   inner dimension, in elements
 *   nc  number of weight rows   (A side), each ggml_t5b_row_bytes(n) bytes
 *   nr  number of activation columns (B side), each n bytes
 *   s   output, s[col*bs + row]
 *
 * Both return sum(code*a) per element; see (2). */
void ggml_gemv_t5b_i8_s(int n, float *s, size_t bs,
                        const void *vx, const void *vy, int nr, int nc);
void ggml_gemm_t5b_i8_s(int n, float *s, size_t bs,
                        const void *vx, const void *vy, int nr, int nc);

/* The tinyBLAS-equivalent entry, called from sgemm.cpp's llamafile_sgemm_t5b.
 * Declared in plain C with no ggml types so this file compiles standalone.
 *
 * Unlike the other two this one DOES apply the post-processing, because its
 * caller (ggml-cpu.c:1458's branch) returns immediately afterwards and there
 * is no other place left to do it:
 *
 *     C[ldc*j + i] = (sum(code*a) - act_sums[j]) * (weight_scale/act_scales[j])
 *
 * act_scales may be NULL, in which case the raw sum(code*a) is stored -- the
 * convention llamafile_sgemm's own type cases use.
 *
 * Threading: rows are split across [ith, nth) and each thread writes only its
 * own rows of C, so no barrier is needed -- the same contract
 * llamafile_sgemm_i2s has. */
void bitnet_t5b_sgemm(int64_t m, int64_t n, int64_t k,
                      const uint8_t *A, int64_t a_row_bytes,
                      const int8_t *B, int64_t ldb,
                      float *C, int64_t ldc,
                      int ith, int nth,
                      const float *act_scales, const int32_t *act_sums,
                      float weight_scale);

/* Evidence of execution, not a quantity anything depends on; incremented from
 * ggml worker threads with a relaxed atomic. Printed unconditionally at exit
 * by bitnet_t5b_report(), registered exactly once on first kernel entry. */
extern uint64_t bitnet_t5b_calls;       /* vec_dot / gemv / gemm path */
extern uint64_t bitnet_t5b_sgemm_calls;

/* Summed over the per-thread slots; see the .c for why they are per-thread. */
uint64_t bitnet_t5b_cycles_total(void);

/* The symmetric SGEMM probe. Both matmul arms are timed at the SAME dispatch
 * site -- index 0 for i2_s, 1 for t5b -- so the dispatch overhead is common to
 * both and divides out of the ratio. This replaces deriving the i2_s time by
 * difference across two llama-bench runs, which put every load fluctuation
 * between those runs into the i2_s number. Overhead measured by
 * bench/bench_probe_cost.c: ~19 ns per probe, flat in thread count, against the
 * 94.8 us a single matmul call takes. */
uint64_t bitnet_probe_tsc(void);
void     bitnet_sgemm_cycles_add(int is_t5b, uint64_t t0);
uint64_t bitnet_sgemm_cycles_total(int is_t5b);

/* Phase split inside llama.cpp's i2_s kernel: 0 = contraction, 1 = per-column
 * post-processing. Dispatch is what remains of the dispatch-site total. */
void     bitnet_i2s_phase_add(int phase, uint64_t cycles);
uint64_t bitnet_i2s_phase_total(int phase); /* the llamafile path         */
void bitnet_t5b_report(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_T5B_GLUE_H */
