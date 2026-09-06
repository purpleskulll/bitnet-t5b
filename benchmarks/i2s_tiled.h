/*
 * i2s_tiled.h -- llama.cpp's LIVE i2_s kernel shape, so the comparison is
 * against what actually runs.
 *
 * WHY THIS EXISTS
 * ---------------
 * Every i2_s figure this project has published comes from
 * bitnet_vec_dot_i2_i8_s_reference, which is ggml's AVX2 vec_dot branch copied
 * verbatim. With GGML_LLAMAFILE=ON -- the default, and what the Dockerfile
 * builds -- that function never executes. llamafile_sgemm_i2s intercepts every
 * i2_s matmul and hands it to tinyBLAS_I2S_AVX, whose mnpack dispatch tiles up
 * to 4 rows by 4 columns. During generation there is one token, so it runs
 * gemm<RM,1> and amortises each activation load across RM weight rows.
 *
 * That is not a small difference for a kernel whose limit is instruction issue.
 * The vec_dot branch spends 21 instructions per 128-weight block: one weight
 * load, four activation loads, three shifts, four masks, four vpmaddubsw, and
 * the accumulation. Four rows at a time pays the four activation loads once
 * instead of four times.
 *
 * WHAT IS COPIED AND WHAT IS NOT
 * The arithmetic is upstream's, from
 * ggml/src/ggml-cpu/llamafile/sgemm.cpp :: tinyBLAS_I2S_AVX::gemm -- the same
 * >>6 / >>4 / >>2 / &3 unpack against mask_2bit, the same four vpmaddubsw per
 * block, the same int16 accumulation folded with one16. What is NOT copied is
 * upstream's register allocation: it holds a_vals[RM][4], sixteen unpacked
 * planes at RM=4, plus four activation registers, which exceeds the sixteen ymm
 * this machine has. Here the activations are hoisted out of the row loop and
 * each row is unpacked inside it, so four activation registers, four planes and
 * four accumulators are live at once. That is twelve, and it fits. The measured
 * work per weight is identical; only the spilling differs, and spilling is not
 * a property of the algorithm anyone is trying to compare.
 *
 * Upstream also has a VNNI path (_mm256_dpbusd_epi32) which Zen 2 does not
 * have and which is therefore not reachable on this host either way.
 *
 * CONTRACT
 * Same as bitnet_vec_dot_i2_i8_s_reference: returns sum(code*a) with the RAW
 * code in {0,1,2}, NOT the ternary weight. n must be a multiple of 128.
 */

#ifndef I2S_TILED_H
#define I2S_TILED_H

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

static inline int32_t i2s_tiled_hsum(__m256i v)
{
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* `rows` rows of `n` weights in the i2_s block layout, one activation column.
 * Writes sum(code*a) per row into y. Rows are taken four at a time; a remainder
 * of one to three rows runs through the same body with a smaller tile. */
static void i2s_tiled_gemv(const uint8_t *W, const int8_t *a, int32_t *y,
                           size_t rows, size_t n)
{
    const __m256i mask2 = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);
    const size_t  rb    = n / 4;          /* packed bytes per row */
    const size_t  nb    = n / 128;        /* 128-weight blocks per row */

    for (size_t r0 = 0; r0 < rows; r0 += 4) {
        const size_t rm = (r0 + 4 <= rows) ? 4 : rows - r0;

        __m256i acc32[4];
        for (size_t r = 0; r < rm; ++r) acc32[r] = _mm256_setzero_si256();

        /* The int16 accumulator is folded every 32 blocks, which is upstream's
         * own fold factor: four products of at most 2*127 summed pairwise is
         * 508 per block, and upstream relies on sign cancellation to stay
         * inside int16 across 32 of them. Copied deliberately -- changing it
         * here would make this a different kernel from the one being used as
         * the baseline. */
        for (size_t g = 0; g < nb; g += 32) {
            const size_t gend = (g + 32 < nb) ? g + 32 : nb;
            __m256i acc16[4];
            for (size_t r = 0; r < rm; ++r) acc16[r] = _mm256_setzero_si256();

            for (size_t b = g; b < gend; ++b) {
                /* Loaded ONCE for all rm rows. This is the whole point. */
                const int8_t *ab = a + b * 128;
                const __m256i a0 = _mm256_loadu_si256((const __m256i *)(ab));
                const __m256i a1 = _mm256_loadu_si256((const __m256i *)(ab + 32));
                const __m256i a2 = _mm256_loadu_si256((const __m256i *)(ab + 64));
                const __m256i a3 = _mm256_loadu_si256((const __m256i *)(ab + 96));

                for (size_t r = 0; r < rm; ++r) {
                    const __m256i p = _mm256_loadu_si256(
                        (const __m256i *)(W + (r0 + r) * rb + b * 32));
                    const __m256i c0 = _mm256_and_si256(_mm256_srli_epi16(p, 6), mask2);
                    const __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(p, 4), mask2);
                    const __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(p, 2), mask2);
                    const __m256i c3 = _mm256_and_si256(p, mask2);

                    acc16[r] = _mm256_add_epi16(acc16[r],
                        _mm256_add_epi16(_mm256_maddubs_epi16(c0, a0),
                                         _mm256_maddubs_epi16(c1, a1)));
                    acc16[r] = _mm256_add_epi16(acc16[r],
                        _mm256_add_epi16(_mm256_maddubs_epi16(c2, a2),
                                         _mm256_maddubs_epi16(c3, a3)));
                }
            }
            for (size_t r = 0; r < rm; ++r)
                acc32[r] = _mm256_add_epi32(acc32[r],
                               _mm256_madd_epi16(acc16[r], one16));
        }
        for (size_t r = 0; r < rm; ++r)
            y[r0 + r] = i2s_tiled_hsum(acc32[r]);
    }
}

#endif /* I2S_TILED_H */
