#!/usr/bin/env python3
"""
apply_integration.py -- wire GGML_TYPE_T5B into a BitNet / llama.cpp checkout.

WHAT THIS DOES. Adds a new ggml tensor type (43) carrying ternary weights at
1.600 bits each, its CPU kernels, and its dispatch entries. It does NOT touch
the i2_s path: i2_s is the control arm of every A/B this project reports, and a
result measured against a moved baseline is not a result.

WHY A SCRIPT AND NOT ONLY A DIFF. integration/t5b-integration.patch is the
same change as a unified diff and is the thing to READ. The script exists
because the change is not only hunks: four source files have to be copied into
the tree as well, and because it is idempotent and anchor-based, it survives
upstream moving a few lines in a way a context diff does not. Both are shipped.
The patch applies cleanly to the pinned commit; the script applies to that
commit and to trees near it, and refuses loudly on anything else.

WHERE IT INSERTS, and why each is necessary rather than sufficient:

  ggml/include/ggml.h            the type enum: GGML_TYPE_T5B = 43,
                                 GGML_TYPE_COUNT 43 -> 44
  ggml/src/ggml.c                the type-traits table (blck_size, type_size,
                                 to_float, from_float, vec_dot) and ggml_nbytes
  ggml/src/ggml-cpu/ggml-cpu.c   the CPU traits entry and the mul_mat dispatch
  ggml/src/ggml-cpu/llamafile/   sgemm: with GGML_LLAMAFILE=ON, which is the
    sgemm.{h,cpp}                DEFAULT, ggml_compute_forward_mul_mat returns
                                 through llamafile_sgemm before the vec_dot
                                 fallback is ever consulted. A tree patched
                                 only in ggml-cpu.c builds, runs, produces
                                 correct output, and never executes the kernel
                                 once. Measured -- the call counter stayed at
                                 zero for a whole inference run.
  ggml/src/ggml-cpu/             the source list, so the two new .c files are
    CMakeLists.txt               actually compiled

Idempotent: every step checks for its own marker and skips, and every anchor is
required to be present EXACTLY once before anything is written. A half-applied
patch to ggml-cpu.c is far worse than a refusal -- it leaves a tree that either
does not compile or, worse, computes one stride the old way and one the new
way. Exit is non-zero and loud on any missing anchor, because a silently
unpatched tree builds and behaves normally, which is the worst outcome
available.

  Pinned upstream (what this was written and verified against):
    microsoft/BitNet     0b341e582afbf9e1011f24744b554c96a3477eb5
    3rdparty/llama.cpp   390c307752ab78fd8189f359d6954c9ba1be74af
                         from https://github.com/isHuangXin/llama.cpp.git

Usage:  python3 integration/apply_integration.py <path-to-BitNet-checkout>
        python3 integration/apply_integration.py <checkout> --sources <dir>

--sources defaults to ../src relative to this file, i.e. this repository's own
src/ directory, which holds ternary_t5b.{c,h} and ggml_t5b_glue.{c,h}.
"""

import argparse
import shutil
import sys
from pathlib import Path

MARK = "/* === bitnet-baremetal integration === */"


class PatchError(RuntimeError):
    pass


def _read(p: Path) -> str:
    if not p.is_file():
        raise PatchError(f"missing file: {p}")
    return p.read_text(encoding="utf-8")


def _write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")
    print(f"    patched {p}")


def _require_once(haystack: str, needle: str, what: str) -> None:
    n = haystack.count(needle)
    if n != 1:
        raise PatchError(f"{what}: expected exactly 1 occurrence of anchor, found {n}")


# ------------------------------------------------------------------- t5b ---
#
# GGML_TYPE_T5B = 43: ternary weights at 1.600 bits each, five trits per byte,
# 160 weights per 32-byte block. Read src/ggml_t5b_glue.h before
# changing anything here -- it names the three decisions that are SILENT when
# wrong (row stride, the sum(a) convention, the counters).
#
# WHY A NEW TYPE AND NOT A FLAG ON i2_s. Three mechanical reasons, none of them
# stylistic:
#   (a) ggml_nbytes(const struct ggml_tensor *) takes only a tensor and has no
#       route to a GGUF metadata key. A flag would have to be a process-wide
#       mutable global read by a pure function, which breaks two models in one
#       process and makes a type's meaning depend on load order.
#   (b) The A/B methodology depends on i2_s still working. Three i2_s GGUFs and
#       scripts/run_inference_ab.sh compare arms on ONE binary; overloading the
#       type deletes the control arm.
#   (c) Version skew fails LOUD instead of silently. An old binary meeting a
#       t5b file is rejected by gguf.cpp ("invalid ggml type 43"); with a
#       metadata key it would read wrong-but-valid bytes and speak fluently.
#
# 43 is free: ggml.h lists every value and the highest assigned is
# GGML_TYPE_TL2 = 42, with GGML_TYPE_COUNT = 43. Slots 36/37/38 only LOOK free
# in the traits table -- ggml.c declares them "REMOVED" and then later
# designated initialisers re-assign them to I2_S/I8_S/TL1, and in C the later
# entry wins. tools/gguf_to_t5b.py:87 writes this same 43 into the file.
#
# blck_size MUST stay 1, as it is for I2_S/TL1/TL2. Setting it to 160 has two
# hard stops -- gguf.cpp rejects the file because 6912 % 160 = 32, and
# ggml_row_size asserts -- plus one silent one, a wrong nb[1]. With
# blck_size = 1 and type_size = 1 the nb[] of a t5b tensor is byte-identical to
# an i2_s one, which is what keeps this change containable.

T5B_SOURCES = ("ternary_t5b.c", "ternary_t5b.h",
               "ggml_t5b_glue.c", "ggml_t5b_glue.h")

# Anchored on an UPSTREAM line, so this applies to a pristine checkout.
# (In the private tree it anchored on a line an earlier phase had added,
# which is not present in a tree only this patch has touched.)
T5B_CMAKE_ANCHOR = "        ggml-cpu/ggml-cpu-i2s.c\n"
T5B_CMAKE_REPLACEMENT = ("        ggml-cpu/ggml-cpu-i2s.c\n"
                         "        ggml-cpu/ternary_t5b.c\n"
                         "        ggml-cpu/ggml_t5b_glue.c\n")

# --- ggml/include/ggml.h ----------------------------------------------------

T5B_GGML_H_PATCHES = [
    (
        "type enum",
        """        GGML_TYPE_TL2     = 42,
        GGML_TYPE_COUNT   = 43,""",

        """        GGML_TYPE_TL2     = 42,
        GGML_TYPE_T5B     = 43, // five trits per byte: 160 weights in 32 bytes
        GGML_TYPE_COUNT   = 44,""",
    ),
]

# --- ggml/src/ggml.c --------------------------------------------------------

T5B_GGML_C_PATCHES = [
    (
        "type traits",
        """    [GGML_TYPE_TL2] = {
        .type_name                = "tl2",
        .blck_size                = 1,
        .type_size                = sizeof(int8_t),
        .is_quantized             = false,
    },
};""",

        # TL2's own entry is reproduced verbatim, is_quantized = false and all:
        # this hunk ADDS a neighbour, it does not revise one. T5B mirrors I2_S
        # (ggml.c: is_quantized = true) rather than TL2, and leaves .to_float
        # NULL -- there is no dequantize_row_t5b, which is what makes the
        # get_rows path abort loudly instead of returning a wrong embedding.
        """    [GGML_TYPE_TL2] = {
        .type_name                = "tl2",
        .blck_size                = 1,
        .type_size                = sizeof(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_T5B] = {
        .type_name                = "t5b",
        .blck_size                = 1,
        .type_size                = sizeof(int8_t),
        .is_quantized             = true,
    },
};""",
    ),
    (
        "ggml_nbytes",
        """        } else if (tensor->type == GGML_TYPE_TL2) {
            nbytes = (tensor->ne[0] - 256) * tensor->ne[1] / 3 * 5 / 8 + 256 * tensor->ne[1] / 2 * 4 / 8;
            if (nbytes % 32 != 0) nbytes = 32 - nbytes % 32 + nbytes;
            nbytes += 32;
        }""",

        """        } else if (tensor->type == GGML_TYPE_TL2) {
            nbytes = (tensor->ne[0] - 256) * tensor->ne[1] / 3 * 5 / 8 + 256 * tensor->ne[1] / 2 * 4 / 8;
            if (nbytes % 32 != 0) nbytes = 32 - nbytes % 32 + nbytes;
            nbytes += 32;
        } else if (tensor->type == GGML_TYPE_T5B) {
            // Five trits per byte: 160 weights per 32-byte block, rounded up
            // PER ROW, then ne[1]*ne[2]*ne[3] rows, then the 32-byte scale
            // footer that t5b carries over from i2_s unchanged.
            //
            // The per-ROW round-up is not cosmetic. For K = 6912 a whole-tensor
            // round-up gives 2560*6912/160*32 = 3,538,944 bytes, but each row
            // needs ceil(6912/160)*32 = 1408 and 2560 of them need 3,604,480 --
            // 65,536 short, per tensor. llama-model-loader bounds-checks against
            // the FILE size, not the per-tensor size, so a short answer here
            // passes and every tensor after it is read at the wrong offset.
            //
            // tools/gguf_to_t5b.py:145 computes the identical formula. If the
            // two ever disagree, gguf.cpp's running-offset check rejects the
            // file at load with "has offset N, expected M" -- loud, which is
            // why the writer and the reader must share one rule.
            const int64_t rowb = ((tensor->ne[0] + 159) / 160) * 32;
            nbytes = (size_t) rowb * tensor->ne[1] * tensor->ne[2] * tensor->ne[3] + 32;
        }""",
    ),
]

# --- ggml/src/ggml-cpu/ggml-cpu.c -------------------------------------------
#
# Every t5b stride below comes from ggml_t5b_row_bytes(). The i2_s expressions
# are left untouched: the i2_s path is the control arm of every A/B this
# project runs, so t5b is added ALONGSIDE it, never as an edit to it.

T5B_CPU_PATCHES = [
    (
        "header include",
        '#include "ggml-cpu-i2s.h"\n',
        '#include "ggml-cpu-i2s.h"\n#include "ggml_t5b_glue.h"\n',
    ),
    (
        "cpu type traits",
        """    [GGML_TYPE_I2_S] = {
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_i2_i8_s,
        .vec_dot_type             = GGML_TYPE_I8_S,
        .nrows                    = 1,
    },
};""",

        """    [GGML_TYPE_I2_S] = {
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_i2_i8_s,
        .vec_dot_type             = GGML_TYPE_I8_S,
        .nrows                    = 1,
    },
    [GGML_TYPE_T5B] = {
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_t5b_i8_s,
        .vec_dot_type             = GGML_TYPE_I8_S,
        .nrows                    = 1,
    },
};""",
    ),
    (
        "one_chunk entry",
        """    // I2_S: weights are 2-bit packed (4 elements per byte), need special addressing and post-processing
    if (src0->type == GGML_TYPE_I2_S) {
        const float * scale      = (const float *)((const uint8_t *)src0->data + (ne00 * ne01 / 4));""",

        """    // I2_S: weights are 2-bit packed (4 elements per byte), need special addressing and post-processing
    // T5B: five trits per byte, 160 weights per 32-byte block. The activation
    // side, the scale footer and the post-processing are identical to i2_s --
    // only the weight row stride differs (ceil(ne00/160)*32 against ne00/4),
    // and with it the offset of the scale that follows the packed rows.
    if (src0->type == GGML_TYPE_I2_S || src0->type == GGML_TYPE_T5B) {
        const bool    is_t5b   = src0->type == GGML_TYPE_T5B;
        const int64_t t5b_rowb = is_t5b ? ggml_t5b_row_bytes(ne00) : 0;
        // The cheapest catch for the quietest failure in this integration: a
        // stride that disagrees with the allocation fails HERE, before a single
        // weight byte is read. Reading at a stale ne00/4 stride does not fault
        // -- the tensors are mmap'd contiguously, so row 6911 lands 884 KB into
        // the NEXT tensor and the model produces fluent, wrong text.
        GGML_ASSERT(!is_t5b || ggml_t5b_nbytes(ne00, ne01, ne02, ne03) == (int64_t) ggml_nbytes(src0));
        const float * scale      = is_t5b
            ? (const float *)((const uint8_t *)src0->data + (size_t) (t5b_rowb * ne01))
            : (const float *)((const uint8_t *)src0->data + (ne00 * ne01 / 4));""",
    ),
    (
        "one_chunk 16-row vec_dot",
        """                        vec_dot(ne00, &tmp[0], 1,
                            src0_row + iir0 * nb01 / 4, nb01,
                            src1_col_de, 0, 16);""",

        """                        // The 5th argument is the row stride handed INTO vec_dot:
                        // BYTES for t5b (ggml_t5b_glue.h), an element count for
                        // i2_s, whose callee does the /4 itself.
                        vec_dot(ne00, &tmp[0], 1,
                            is_t5b ? src0_row + iir0 * t5b_rowb : src0_row + iir0 * nb01 / 4,
                            is_t5b ? (size_t) t5b_rowb : nb01,
                            src1_col_de, 0, 16);""",
    ),
    (
        "one_chunk 1-row vec_dot",
        """                            vec_dot(ne00, &tmp[ir0 - iir0], 0,
                                src0_row + ir0 * nb01 / 4, 0,
                                src1_col_de, 0, 1);""",

        """                            vec_dot(ne00, &tmp[ir0 - iir0], 0,
                                is_t5b ? src0_row + ir0 * t5b_rowb : src0_row + ir0 * nb01 / 4, 0,
                                src1_col_de, 0, 1);""",
    ),
    (
        "activation scale/sum buffers",
        """        // I2_S needs extra space for activation scales and sums
        float * act_scales = NULL;
        int32_t * act_sums = NULL;
        if (src0->type == GGML_TYPE_I2_S) {""",

        """        // I2_S needs extra space for activation scales and sums
        // T5B needs exactly the same, and gets it for free: the work-buffer
        // size at GGML_OP_MUL_MAT keys off vec_dot_type == GGML_TYPE_I8_S,
        // not off src0->type, and t5b's vec_dot_type is I8_S. Nothing there
        // changes -- verified by reading it, not assumed.
        float * act_scales = NULL;
        int32_t * act_sums = NULL;
        if (src0->type == GGML_TYPE_I2_S || src0->type == GGML_TYPE_T5B) {""",
    ),
    (
        "activation quantisation",
        """                if (src0->type == GGML_TYPE_I2_S) {
                    // I2_S: distribute rows across threads (each row needs full-row amax/sum)""",

        """                if (src0->type == GGML_TYPE_I2_S || src0->type == GGML_TYPE_T5B) {
                    // I2_S: distribute rows across threads (each row needs full-row amax/sum)
                    // T5B MUST be admitted here and it is not cosmetic: I8_S has
                    // NO entry in type_traits_cpu, so from_float in the else
                    // branch below is NULL. Leaving t5b out of this test does not
                    // merely skip the amax/sum -- it calls through a null pointer.""",
    ),
    (
        "llamafile dispatch entry",
        """        // I2_S: use fused sgemm with inline post-processing (no barrier needed)
        if (src0->type == GGML_TYPE_I2_S) {
            const float * scale = (const float *)((const uint8_t *)src0->data + (ne00 * ne01 / 4));""",

        """        // I2_S: use fused sgemm with inline post-processing (no barrier needed)
        // T5B takes the same route through a PARALLEL entry point,
        // llamafile_sgemm_t5b, rather than a branch inside tinyBLAS_I2S_AVX.
        // The i2_s kernel is the control arm of every A/B here; it stays
        // byte-identical, and a bug in the t5b decode cannot reach it.
        if (src0->type == GGML_TYPE_I2_S || src0->type == GGML_TYPE_T5B) {
            const bool    sg_is_t5b   = src0->type == GGML_TYPE_T5B;
            const int64_t sg_t5b_rowb = sg_is_t5b ? ggml_t5b_row_bytes(ne00) : 0;
            GGML_ASSERT(!sg_is_t5b || ggml_t5b_nbytes(ne00, ne01, ne02, ne03) == (int64_t) ggml_nbytes(src0));
            const float * scale = sg_is_t5b
                ? (const float *)((const uint8_t *)src0->data + (size_t) (sg_t5b_rowb * ne01))
                : (const float *)((const uint8_t *)src0->data + (ne00 * ne01 / 4));""",
    ),
    (
        "llamafile dispatch call",
        """                    if (!llamafile_sgemm_i2s(params,
                                             ne01, ne11, ne00,
                                             (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                             nb01/ggml_type_size(src0->type),
                                             (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                             row_size/ggml_type_size(vec_dot_type),
                                             (char *)dst->data + i12*nb2 + i13*nb3,
                                             nb1/ggml_type_size(dst->type),
                                             i2s_act_scales, i2s_act_sums, *scale))""",

        """                    // t5b is handed the REAL byte stride as lda, and
                    // bitnet_t5b_sgemm re-derives it from k and aborts if the
                    // two disagree. i2_s keeps passing nb01/type_size, which
                    // tinyBLAS_I2S_AVX ignores in favour of k/4 -- unchanged.
                    if (!(sg_is_t5b
                          ? llamafile_sgemm_t5b(params,
                                             ne01, ne11, ne00,
                                             (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                             sg_t5b_rowb,
                                             (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                             row_size/ggml_type_size(vec_dot_type),
                                             (char *)dst->data + i12*nb2 + i13*nb3,
                                             nb1/ggml_type_size(dst->type),
                                             i2s_act_scales, i2s_act_sums, *scale)
                          : llamafile_sgemm_i2s(params,
                                             ne01, ne11, ne00,
                                             (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                             nb01/ggml_type_size(src0->type),
                                             (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                             row_size/ggml_type_size(vec_dot_type),
                                             (char *)dst->data + i12*nb2 + i13*nb3,
                                             nb1/ggml_type_size(dst->type),
                                             i2s_act_scales, i2s_act_sums, *scale)))""",
    ),
    (
        "gemv/gemm fast path entry",
        """    // I2_S GEMV/GEMM fast path: use optimized SIMD kernels for 2D weight matrices
    if (src0->type == GGML_TYPE_I2_S && ggml_n_dims(src0) == 2) {""",

        """    // I2_S GEMV/GEMM fast path: use optimized SIMD kernels for 2D weight matrices
    // Unreachable when GGML_USE_LLAMAFILE is on, because the branch above
    // returns. It is the arm the project builds with GGML_LLAMAFILE=OFF to
    // prove WHICH path ran, so it is kept correct rather than left to rot.
    if ((src0->type == GGML_TYPE_I2_S || src0->type == GGML_TYPE_T5B) && ggml_n_dims(src0) == 2) {""",
    ),
    (
        "gemv/gemm fast path scale",
        """        const float * scale      = (const float *)((const uint8_t *)src0->data + (ne00 * ne01 / 4));
        const float * act_scales = (const float *)((const char *)src1_wdata + (ne11 * ne10));""",

        """        const bool    fp_is_t5b   = src0->type == GGML_TYPE_T5B;
        const int64_t fp_t5b_rowb = fp_is_t5b ? ggml_t5b_row_bytes(ne00) : 0;
        GGML_ASSERT(!fp_is_t5b || ggml_t5b_nbytes(ne00, ne01, ne02, ne03) == (int64_t) ggml_nbytes(src0));
        const float * scale      = fp_is_t5b
            ? (const float *)((const uint8_t *)src0->data + (size_t) (fp_t5b_rowb * ne01))
            : (const float *)((const uint8_t *)src0->data + (ne00 * ne01 / 4));
        const float * act_scales = (const float *)((const char *)src1_wdata + (ne11 * ne10));""",
    ),
    (
        "gemm call",
        """                ggml_gemm_i2_i8_s(ne00, tmp, n_rows,
                    (const char *)src0->data + src0_start * nb01 / 4,
                    (const char *)src1_wdata + src1_col_stride * col_start,
                    4, n_rows);""",

        """                if (fp_is_t5b) {
                    ggml_gemm_t5b_i8_s(ne00, tmp, n_rows,
                        (const char *)src0->data + src0_start * fp_t5b_rowb,
                        (const char *)src1_wdata + src1_col_stride * col_start,
                        4, n_rows);
                } else {
                    ggml_gemm_i2_i8_s(ne00, tmp, n_rows,
                        (const char *)src0->data + src0_start * nb01 / 4,
                        (const char *)src1_wdata + src1_col_stride * col_start,
                        4, n_rows);
                }""",
    ),
    (
        "gemv call",
        """                ggml_gemv_i2_i8_s(ne00, tmp, ne01,
                    (const char *)src0->data + src0_start * nb01 / 4,
                    (const char *)src1_wdata + src1_col_stride * iter,
                    1, n_rows);""",

        """                if (fp_is_t5b) {
                    ggml_gemv_t5b_i8_s(ne00, tmp, ne01,
                        (const char *)src0->data + src0_start * fp_t5b_rowb,
                        (const char *)src1_wdata + src1_col_stride * iter,
                        1, n_rows);
                } else {
                    ggml_gemv_i2_i8_s(ne00, tmp, ne01,
                        (const char *)src0->data + src0_start * nb01 / 4,
                        (const char *)src1_wdata + src1_col_stride * iter,
                        1, n_rows);
                }""",
    ),
]

# --- ggml/src/ggml-cpu/llamafile/sgemm.{h,cpp} ------------------------------
#
# A SIBLING entry point, not a parameter on llamafile_sgemm_i2s. Changing that
# signature would touch the i2_s call site's ABI for no gain, and sgemm.h
# deliberately includes only <stdint.h>/<stdbool.h> -- an `enum ggml_type`
# parameter would drag ggml.h into it.

T5B_SGEMM_H_PATCHES = [
    (
        "declaration",
        """bool llamafile_sgemm_i2s(const struct ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                         const void *A, int64_t lda, const void *B, int64_t ldb, void *C, int64_t ldc,
                         const float *act_scales, const int32_t *act_sums, float weight_scale);""",

        """bool llamafile_sgemm_i2s(const struct ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                         const void *A, int64_t lda, const void *B, int64_t ldb, void *C, int64_t ldc,
                         const float *act_scales, const int32_t *act_sums, float weight_scale);

/* === bitnet-baremetal integration ===
 * Same shape as llamafile_sgemm_i2s, for GGML_TYPE_T5B. `lda` is the packed
 * row stride in BYTES -- ceil(k/160)*32 -- not an element count. */
bool llamafile_sgemm_t5b(const struct ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                         const void *A, int64_t lda, const void *B, int64_t ldb, void *C, int64_t ldc,
                         const float *act_scales, const int32_t *act_sums, float weight_scale);""",
    ),
]

T5B_SGEMM_CPP_PATCHES = [
    (
        "header include",
        '#include "sgemm.h"\n',
        '#include "sgemm.h"\n'
        '/* === bitnet-baremetal integration === */\n'
        '#include "ggml_t5b_glue.h"\n',
    ),
    (
        "t5b entry point",
        """bool llamafile_sgemm_i2s(const struct ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                         const void *A, int64_t lda, const void *B, int64_t ldb, void *C, int64_t ldc,
                         const float *act_scales, const int32_t *act_sums, float weight_scale) {""",

        """/* === bitnet-baremetal integration ===
 * The t5b path. Deliberately NOT a branch inside tinyBLAS_I2S_AVX: that class
 * is the control arm of every A/B this project runs, and its 128-weights-per-
 * 32-bytes decode has nothing in common with t5b's 160. Everything below the
 * type dispatch lives in ggml_t5b_glue.c, which adapts the already-tested
 * ternary_t5b kernels rather than re-deriving the decode here.
 *
 * The convention, because it is the one that cannot crash: bitnet_t5b_sgemm
 * hands the kernel sum_a = 0, so it computes sum(code*a), and then applies
 * "- act_sums[j]" exactly once itself -- the same arithmetic tinyBLAS does at
 * the bottom of its gemm(). See ggml_t5b_glue.h section 2. */
bool llamafile_sgemm_t5b(const struct ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                         const void *A, int64_t lda, const void *B, int64_t ldb, void *C, int64_t ldc,
                         const float *act_scales, const int32_t *act_sums, float weight_scale) {
#if defined(__AVX2__)
    bitnet_t5b_sgemm(m, n, k, (const uint8_t *)A, lda, (const int8_t *)B, ldb,
                     (float *)C, ldc, params->ith, params->nth,
                     act_scales, act_sums, weight_scale);
    return true;
#else
    (void)params; (void)m; (void)n; (void)k;
    (void)A; (void)lda; (void)B; (void)ldb; (void)C; (void)ldc;
    (void)act_scales; (void)act_sums; (void)weight_scale;
    return false;
#endif
}

bool llamafile_sgemm_i2s(const struct ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                         const void *A, int64_t lda, const void *B, int64_t ldb, void *C, int64_t ldc,
                         const float *act_scales, const int32_t *act_sums, float weight_scale) {""",
    ),
]


def apply_t5b(cpu_dir: Path, src: Path) -> None:
    """GGML_TYPE_T5B = 43 -> the ggml CPU backend. See the block comment above.

    Every step is independently idempotent. A step that returned out of the
    whole function on finding its own work already done would make every
    later step unreachable on a re-run -- which is how half-patched trees
    get built and then behave normally.
    """
    print("t5b (1.600 bits/weight, type 43) -> ggml")

    ggml_dir = cpu_dir.parent.parent          # ggml/src/ggml-cpu -> ggml
    if not (ggml_dir / "include/ggml.h").is_file():
        raise PatchError(f"not a llama.cpp tree: {ggml_dir}/include/ggml.h missing")

    _t5b_sources(cpu_dir, src)
    _t5b_cmake(cpu_dir)
    _t5b_hunks(ggml_dir / "include/ggml.h", "GGML_TYPE_T5B",
               T5B_GGML_H_PATCHES, "ggml.h")
    _t5b_hunks(ggml_dir / "src/ggml.c", "GGML_TYPE_T5B",
               T5B_GGML_C_PATCHES, "ggml.c")
    _t5b_hunks(cpu_dir / "ggml-cpu.c", "ggml_t5b_glue.h",
               T5B_CPU_PATCHES, "ggml-cpu.c")
    _t5b_hunks(cpu_dir / "llamafile/sgemm.h", "llamafile_sgemm_t5b",
               T5B_SGEMM_H_PATCHES, "sgemm.h")
    _t5b_hunks(cpu_dir / "llamafile/sgemm.cpp", "llamafile_sgemm_t5b",
               T5B_SGEMM_CPP_PATCHES, "sgemm.cpp")


def _t5b_sources(cpu_dir: Path, src: Path) -> None:
    # Unconditional, so a re-run picks up an edited kernel rather than silently
    # building the stale copy already in the tree. ternary_t5b.{c,h} are copied
    # VERBATIM -- they are the tested artefact and the glue is what adapts.
    for name in T5B_SOURCES:
        s = src / name
        if not s.is_file():
            raise PatchError(f"missing t5b source: {s}")
        shutil.copy2(s, cpu_dir / name)
        print(f"    copied {name}")


def _t5b_cmake(cpu_dir: Path) -> None:
    # Idempotent on the presence of the t5b entry specifically, not on any
    # marker a different patch might have written: a CMakeLists that another
    # change has already touched must still get the t5b sources.
    cml = cpu_dir / "CMakeLists.txt"
    text = _read(cml)
    if "ggml-cpu/ggml_t5b_glue.c" in text:
        print("    CMakeLists already lists the t5b sources (idempotent)")
        return
    _require_once(text, T5B_CMAKE_ANCHOR, "CMakeLists t5b source list")
    _write(cml, text.replace(T5B_CMAKE_ANCHOR, T5B_CMAKE_REPLACEMENT))


def _t5b_hunks(path: Path, mark: str, hunks, what: str) -> None:
    """Apply a hunk list, all-or-nothing.

    Every anchor is checked for being present EXACTLY once before anything is
    written. Half a patch to ggml-cpu.c is far worse than a refusal: it would
    leave a tree that either does not compile or, worse, computes one stride
    the old way and one the new way.
    """
    text = _read(path)
    if mark in text:
        print(f"    {what} already carries t5b (idempotent)")
        return

    for label, anchor, _ in hunks:
        _require_once(text, anchor, f"{what} {label}")

    for _, anchor, replacement in hunks:
        text = text.replace(anchor, replacement, 1)

    _write(path, text)
    print(f"    applied {len(hunks)} hunks to {what}")


# ------------------------------------------------------------------- main ---

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Wire GGML_TYPE_T5B into a BitNet / llama.cpp checkout.")
    ap.add_argument("tree", type=Path,
                    help="BitNet checkout to patch (the one containing "
                         "3rdparty/llama.cpp), or a llama.cpp checkout")
    ap.add_argument("--sources", type=Path, default=None,
                    help="directory holding ternary_t5b.{c,h} and "
                         "ggml_t5b_glue.{c,h} (default: this repo's src/)")
    args = ap.parse_args()

    src = args.sources or (Path(__file__).resolve().parent.parent / "src")
    if not src.is_dir():
        print(f"error: sources directory {src} not found", file=sys.stderr)
        return 2

    # Accept either a BitNet checkout or a bare llama.cpp checkout. Guessing
    # between them silently would patch the wrong tree on a mislaid argument;
    # this looks for the directory that must exist and says which it found.
    candidates = [args.tree / "3rdparty/llama.cpp/ggml/src/ggml-cpu",
                  args.tree / "ggml/src/ggml-cpu"]
    cpu_dir = next((c for c in candidates if c.is_dir()), None)
    if cpu_dir is None:
        print(f"error: neither {candidates[0]} nor {candidates[1]} exists.",
              file=sys.stderr)
        print("       Pass a BitNet checkout or a llama.cpp checkout.",
              file=sys.stderr)
        return 2
    print(f"tree:    {cpu_dir}")
    print(f"sources: {src}")

    try:
        apply_t5b(cpu_dir, src)
    except PatchError as e:
        print(f"\nPATCH FAILED: {e}", file=sys.stderr)
        print("The tree is NOT integrated. Fix the anchor before building --",
              file=sys.stderr)
        print("a silently unpatched tree builds and runs normally, which is the",
              file=sys.stderr)
        print("worst possible outcome.", file=sys.stderr)
        return 1

    print("\nintegration applied. Rebuild with:")
    print("    cmake --build <tree>/build --config Release --target llama-cli -j 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
