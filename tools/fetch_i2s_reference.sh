#!/usr/bin/env bash
#
# fetch_i2s_reference.sh -- rebuild src/ggml_i2s_ternary.{c,h} from upstream.
#
# WHY THIS IS A SCRIPT AND NOT A FILE IN THE REPOSITORY
# -----------------------------------------------------
# The i2_s baseline this project measures against is upstream's own AVX2 dot
# product. Every speed ratio in the paper is against that code, so a
# re-derivation of it would not be the baseline -- it would be a second
# implementation, and any difference would be unattributable.
#
# The honest way to have exactly upstream's arithmetic without redistributing
# an MIT-licensed source file inside an Apache-2.0 tree is to extract it, at a
# NAMED commit, from a checkout the user provides or the script fetches. That
# is what this does. The extraction is anchored and digest-checked: if upstream
# has moved, the script refuses rather than silently emitting different
# arithmetic under the same name.
#
# WHAT IT PRODUCES
#   src/ggml_i2s_ternary.h   declares bitnet_vec_dot_i2_i8_s_reference
#   src/ggml_i2s_ternary.c   the two extracted functions plus that wrapper
#
# WHAT NEEDS IT
#   benchmarks/bench_alu.c, bench_threads.c, bench_token.c   (the i2_s arm)
#   tools/gguf_to_t5b.py --verify   (builds it into libt5bcheck.so)
# bench_ports.c does not: it measures instruction ports and includes nothing.
#
# THE CONTRACT OF WHAT IS EXTRACTED
# ---------------------------------
# A block is 32 bytes carrying 128 weights, paired with 128 int8 activations.
# For lane j of the 32-byte load:
#
#     bits 6-7  ->  activation   0 + j
#     bits 4-5  ->  activation  32 + j
#     bits 2-3  ->  activation  64 + j
#     bits 0-1  ->  activation  96 + j
#
# The function returns sum(code_i * y_i) with the RAW code in {0,1,2,3} -- not
# the ternary weight. The -1 offset is applied downstream by the caller. This
# matters: a caller that expects sum(w*y) and gets sum(code*y) is wrong by
# sum(y), which is a plausible-looking number rather than an obvious fault.
#
# Note the scalar #else branch in the same upstream function assumes a
# DIFFERENT layout (four consecutive activations sharing one byte). The two
# branches are not equivalent. Only the AVX2 one runs on AVX2 hardware, so the
# discrepancy is invisible in practice -- but this script extracts the whole
# function, both branches, exactly as upstream has it, and does not attempt to
# reconcile them. Reporting upstream's code as upstream has it is the point.
#
# USAGE
#   tools/fetch_i2s_reference.sh                  # clone the pinned commit
#   tools/fetch_i2s_reference.sh --from <path>    # use an existing checkout
#   tools/fetch_i2s_reference.sh --check          # exit 0 iff already present
#
# --from takes either a llama.cpp checkout or a BitNet checkout (in which case
# 3rdparty/llama.cpp is used). It is checked for the pinned commit and WARNS,
# loudly, if the checkout is at a different revision -- the digest check below
# is the real gate, but a revision mismatch is worth naming before it fires.

set -euo pipefail

# ---------------------------------------------------------------- the pins ---
#
# These are read off the tree this project's Docker images were built from, not
# guessed. microsoft/BitNet does not tag releases and its Dockerfile clones the
# default branch with --depth 1, so there is no upstream tag to name; the
# commit below is the one that clone resolved to and every measurement in the
# paper was made on.
#
#   microsoft/BitNet          0b341e582afbf9e1011f24744b554c96a3477eb5
#                             2026-07-27, "Add VibeASR.cpp release to News (#596)"
#   3rdparty/llama.cpp        390c307752ab78fd8189f359d6954c9ba1be74af
#                             2026-07-15, "Fix build-cann workflow by adding
#                             placeholder job"
#
# BitNet's .gitmodules points 3rdparty/llama.cpp at isHuangXin/llama.cpp,
# branch release-bitnet-embedding-0.6b-270m -- a fork, not llama.cpp master.
# The i2_s type does not exist upstream in ggml-org/llama.cpp, so naming
# ggml-org here would be wrong.
LLAMA_REPO="${LLAMA_REPO:-https://github.com/isHuangXin/llama.cpp.git}"
LLAMA_COMMIT="${LLAMA_COMMIT:-390c307752ab78fd8189f359d6954c9ba1be74af}"
BITNET_COMMIT="0b341e582afbf9e1011f24744b554c96a3477eb5"

# SHA-256 of the extracted region, verbatim, at LLAMA_COMMIT. Recomputed and
# compared on every run. A mismatch means the checkout is not what this script
# was written against; it stops rather than emitting arithmetic the paper never
# measured.
EXPECT_SHA256="869a1bf8f2deecf9b7c2017392ac4997b74aef2bb132abe0d539a3d289e58ed2"

REL_SOURCE="ggml/src/ggml-cpu/quants.c"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_C="$ROOT/src/ggml_i2s_ternary.c"
OUT_H="$ROOT/src/ggml_i2s_ternary.h"

FROM=""
MODE="fetch"
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="${2:-}"; shift 2 ;;
        --from=*) FROM="${1#--from=}"; shift ;;
        --check) MODE="check"; shift ;;
        -h|--help) sed -n '2,60p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ "$MODE" = check ]; then
    if [ -f "$OUT_C" ] && [ -f "$OUT_H" ]; then
        echo "i2_s reference present: $OUT_C"
        exit 0
    fi
    echo "i2_s reference NOT present (run tools/fetch_i2s_reference.sh)" >&2
    exit 1
fi

# ------------------------------------------------------- locate the source ---

WORKDIR=""
cleanup() { [ -n "$WORKDIR" ] && rm -rf -- "$WORKDIR" || true; }
trap cleanup EXIT

if [ -n "$FROM" ]; then
    if [ -f "$FROM/3rdparty/llama.cpp/$REL_SOURCE" ]; then
        TREE="$FROM/3rdparty/llama.cpp"
    elif [ -f "$FROM/$REL_SOURCE" ]; then
        TREE="$FROM"
    else
        echo "FATAL: $FROM contains neither $REL_SOURCE nor 3rdparty/llama.cpp/$REL_SOURCE" >&2
        echo "       Pass a llama.cpp checkout or a BitNet checkout." >&2
        exit 1
    fi
    have="$(git -C "$TREE" rev-parse HEAD 2>/dev/null || echo unknown)"
    if [ "$have" != "$LLAMA_COMMIT" ]; then
        echo "WARNING: $TREE is at $have," >&2
        echo "         not the pinned $LLAMA_COMMIT." >&2
        echo "         Continuing -- the digest check below is the real gate." >&2
    fi
else
    command -v git >/dev/null || { echo "FATAL: git not found and --from not given" >&2; exit 1; }
    WORKDIR="$(mktemp -d)"
    TREE="$WORKDIR/llama.cpp"
    echo "fetching $LLAMA_COMMIT from $LLAMA_REPO ..."
    echo "  (BitNet ${BITNET_COMMIT} pins this submodule commit)"
    git init --quiet "$TREE"
    git -C "$TREE" remote add origin "$LLAMA_REPO"
    if ! git -C "$TREE" fetch --quiet --depth 1 origin "$LLAMA_COMMIT"; then
        echo "FATAL: could not fetch $LLAMA_COMMIT." >&2
        echo "       Some servers refuse fetch-by-sha. Clone the tree yourself and use" >&2
        echo "         git clone --recursive https://github.com/microsoft/BitNet.git" >&2
        echo "         git -C BitNet checkout $BITNET_COMMIT" >&2
        echo "         git -C BitNet submodule update --init --recursive" >&2
        echo "         tools/fetch_i2s_reference.sh --from BitNet" >&2
        exit 1
    fi
    git -C "$TREE" checkout --quiet FETCH_HEAD
fi

SRC="$TREE/$REL_SOURCE"
[ -f "$SRC" ] || { echo "FATAL: $SRC missing" >&2; exit 1; }

# ------------------------------------------------------------- extract it ---
#
# Anchored on the two function signatures and closed by brace matching, not by
# line numbers: line numbers in a moving upstream are the one thing guaranteed
# to be wrong. Braces inside string and character literals are skipped; there
# are none in this function today, and a brace counter that trusts that is a
# counter that breaks silently the first time there is.

EXTRACT="$(python3 - "$SRC" <<'PYEOF'
import re, sys

path = sys.argv[1]
text = open(path, encoding="utf-8").read()

WANTED = [
    r"^static void ggml_vec_dot_i2_i8_s_1x1\(",
    r"^void ggml_vec_dot_i2_i8_s\(",
]

def span(src, start):
    """From the '{' at or after start, return the index just past its match."""
    i = src.index("{", start)
    depth, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '"' or c == "'":
            q, i = c, i + 1
            while i < n and src[i] != q:
                i += 2 if src[i] == "\\" else 1
            i += 1
            continue
        if src.startswith("/*", i):
            i = src.index("*/", i) + 2
            continue
        if src.startswith("//", i):
            i = src.find("\n", i)
            if i < 0:
                break
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise SystemExit(f"FATAL: unbalanced braces from offset {start} in {path}")

out = []
for pat in WANTED:
    m = list(re.finditer(pat, text, re.M))
    if len(m) != 1:
        raise SystemExit(
            f"FATAL: {pat!r} matched {len(m)} times in {path}, expected exactly 1.\n"
            "       Upstream has moved; this script must be updated deliberately\n"
            "       rather than made to guess.")
    a = m[0].start()
    out.append(text[a:span(text, a)])

sys.stdout.write("\n\n".join(out) + "\n")
PYEOF
)"

DIGEST="$(printf '%s' "$EXTRACT" | sha256sum | cut -d' ' -f1)"
echo "extracted $(printf '%s' "$EXTRACT" | wc -l) lines, sha256 $DIGEST"

if [ "$EXPECT_SHA256" != "__PLACEHOLDER__" ] && [ "$DIGEST" != "$EXPECT_SHA256" ]; then
    echo "FATAL: extracted text does not match the pinned digest." >&2
    echo "       expected $EXPECT_SHA256" >&2
    echo "       got      $DIGEST" >&2
    echo "       The checkout is not the revision the paper measured. Refusing to" >&2
    echo "       emit arithmetic under a name that promises upstream's." >&2
    exit 1
fi

# --------------------------------------------------------------- emit them ---

mkdir -p "$ROOT/src"

cat > "$OUT_H" <<EOF
/*
 * ggml_i2s_ternary.h -- GENERATED by tools/fetch_i2s_reference.sh. Do not edit.
 *
 * Declares the i2_s baseline: upstream's own AVX2 dot product, extracted
 * verbatim from
 *     $REL_SOURCE
 * at llama.cpp commit $LLAMA_COMMIT
 * (the 3rdparty/llama.cpp submodule of microsoft/BitNet at
 *  $BITNET_COMMIT).
 *
 * Upstream is MIT-licensed, Copyright (c) 2023-2024 The ggml authors. This
 * file is generated on the user's machine from the user's own checkout; no
 * upstream source is redistributed in this repository. See NOTICE.
 *
 * THE CONTRACT. Returns sum(code_i * y_i) with the RAW code in {0,1,2,3}, not
 * the ternary weight. The -1 offset is applied downstream by the caller.
 */

#ifndef GGML_I2S_TERNARY_H
#define GGML_I2S_TERNARY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Upstream's AVX2 i2_s dot product. The baseline every ratio in the paper is
 * measured against.
 *
 *   n    number of weights (a multiple of 128)
 *   s    output, one float per row
 *   bs   output row stride -- unused, kept for signature compatibility
 *   vx   packed 2-bit weights, n/4 bytes per row
 *   bx   weight row stride in WEIGHTS (the code divides by 4 internally)
 *   vy   int8 activations, n of them, shared by every row
 *   by   unused, kept for signature compatibility
 *   nrc  number of rows
 */
void bitnet_vec_dot_i2_i8_s_reference(int n, float *s, size_t bs,
                                      const void *vx, size_t bx,
                                      const void *vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif

#endif /* GGML_I2S_TERNARY_H */
EOF

{
cat <<EOF
/*
 * ggml_i2s_ternary.c -- GENERATED by tools/fetch_i2s_reference.sh. Do not edit.
 *
 * Extracted verbatim from $REL_SOURCE
 * at llama.cpp commit $LLAMA_COMMIT.
 * sha256 of the extracted region: $DIGEST
 *
 * Upstream is MIT-licensed, Copyright (c) 2023-2024 The ggml authors.
 *
 * Three things are added around the extracted text and nothing inside it is
 * changed:
 *
 *   1. QK_I2_S and hsum_i32_8, which upstream gets from headers this file
 *      does not include. Both are reproduced with upstream's own definitions.
 *   2. \`static\` on ggml_vec_dot_i2_i8_s, so that linking this object next to
 *      a real ggml cannot collide with the symbol of the same name. The body
 *      is untouched.
 *   3. bitnet_vec_dot_i2_i8_s_reference, a one-line wrapper, which is the only
 *      symbol this translation unit exports.
 */

#include "ggml_i2s_ternary.h"

#include <immintrin.h>

/* quants.c:1294 -- 128 on AVX2. */
#define QK_I2_S 128

/* ggml-cpu/simd-mappings.h. Upstream's horizontal sum of eight int32 lanes. */
static inline int32_t hsum_i32_8(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* bs and by are unused in upstream's AVX2 branch and this build is -Wextra.
 * Silencing the warning around the extracted text is preferable to touching
 * the text. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

/* ====================== BEGIN VERBATIM UPSTREAM ====================== */

EOF

# The only edit inside the extracted region: file-local linkage on the
# non-static wrapper. Applied to the signature line only, and anchored, so it
# cannot silently hit anything else.
printf '%s\n' "$EXTRACT" \
  | sed 's|^void ggml_vec_dot_i2_i8_s(|static void ggml_vec_dot_i2_i8_s(|'

cat <<'EOF'

/* ======================= END VERBATIM UPSTREAM ======================= */

#pragma GCC diagnostic pop

void bitnet_vec_dot_i2_i8_s_reference(int n, float *s, size_t bs,
                                      const void *vx, size_t bx,
                                      const void *vy, size_t by, int nrc)
{
    ggml_vec_dot_i2_i8_s(n, s, bs, vx, bx, vy, by, nrc);
}
EOF
} > "$OUT_C"

echo "wrote $OUT_H"
echo "wrote $OUT_C"
echo
echo "the i2_s baseline is now buildable:"
echo "    make bench"
echo "    python3 tools/gguf_to_t5b.py MODEL.gguf --verify"
