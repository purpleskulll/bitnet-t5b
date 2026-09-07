#!/usr/bin/env bash
#
# build_insitu.sh -- build a patched llama.cpp WITHOUT a container, and check
# that the shipped patch is the thing that builds.
#
# WHY THIS EXISTS
# ---------------
# Two claims went unverified for as long as this project had no way to build
# llama.cpp:
#
#   - that integration/t5b-integration.patch APPLIES to the pinned commit. The
#     header said "verified to apply cleanly"; nothing here had checked it since
#     the hunks were last edited.
#   - that what it applies COMPILES. It did not. Adding the symmetric SGEMM
#     probe put three statements into the body of two brace-less `for` loops,
#     so i12 and i13 fell out of scope and ggml-cpu.c did not build. Every
#     consistency check passed on that patch -- chain, parity, hunk headers --
#     because all three compare text, and none of them is a compiler.
#
# The obstacle was believed to be Docker, which this host does not grant. It was
# not: llama.cpp builds with cmake and gcc, no daemon and no privileges. cmake
# is not installed system-wide either, so fetch it into a user directory:
#
#   mkdir -p ~/.local/cmake && curl -sL \
#     "$(curl -s https://api.github.com/repos/Kitware/CMake/releases/latest \
#        | grep -o '"browser_download_url": "[^"]*linux-x86_64.tar.gz"' \
#        | head -1 | cut -d'"' -f4)" \
#     | tar -xzf - -C ~/.local/cmake --strip-components=1
#
# Usage:
#   scripts/build_insitu.sh              apply with the script, build
#   scripts/build_insitu.sh --patch      apply the shipped DIFF instead, build
#
# The second form is the one that matters for a reader: it proves the artefact
# in the public repository is the artefact that works.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/.verify/rec"
MODE="${1:-script}"
OUT="$ROOT/.verify/$([ "$MODE" = --patch ] && echo patchtest || echo insitu)"

CMAKE="$(command -v cmake || true)"
[ -z "$CMAKE" ] && [ -x "$HOME/.local/cmake/bin/cmake" ] && CMAKE="$HOME/.local/cmake/bin/cmake"
if [ -z "$CMAKE" ]; then
    echo "cmake not found -- see this script's header for a user-directory install." >&2
    exit 4
fi
[ -d "$SRC/3rdparty/llama.cpp" ] || {
    echo "no llama.cpp tree at $SRC -- run scripts/prefetch_sources.sh." >&2
    exit 4
}

echo "=================================================================="
echo " BUILD THE INTEGRATION, NATIVELY"
echo "=================================================================="
echo "  source   $SRC  ($(git -C "$SRC/3rdparty/llama.cpp" log --oneline -1 | cut -c1-40))"
echo "  mode     $([ "$MODE" = --patch ] && echo 'the shipped unified diff' || echo 'apply_integration.py')"
echo "  cmake    $("$CMAKE" --version | head -1)"
echo

rm -rf "$OUT"; cp -a "$SRC" "$OUT"

if [ "$MODE" = --patch ]; then
    PATCH="$ROOT/../bitnet-t5b/integration/t5b-integration.patch"
    [ -f "$PATCH" ] || { echo "shipped patch not found at $PATCH" >&2; exit 4; }
    ( cd "$OUT/3rdparty/llama.cpp"
      git apply --check "$PATCH" || {
          echo "  the shipped patch does NOT apply to the pinned commit." >&2; exit 1; }
      git apply "$PATCH"
      echo "  patch applies cleanly." )
    # The four added files are the repository's own sources, copied in rather
    # than carried by the diff -- see the patch header.
    cp "$ROOT/../bitnet-t5b/src/ternary_t5b.c" "$ROOT/../bitnet-t5b/src/ternary_t5b.h" \
       "$ROOT/../bitnet-t5b/src/ggml_t5b_glue.c" "$ROOT/../bitnet-t5b/src/ggml_t5b_glue.h" \
       "$OUT/3rdparty/llama.cpp/ggml/src/ggml-cpu/"
else
    python3 "$ROOT/scripts/apply_integration.py" "$OUT" --phase 3 > /dev/null
    echo "  integration applied by script."
fi

LOG="$(mktemp)"; trap 'rm -f "$LOG"' EXIT
"$CMAKE" -S "$OUT/3rdparty/llama.cpp" -B "$OUT/build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_LLAMAFILE=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    > "$LOG" 2>&1 || { echo "  cmake configure FAILED:"; tail -20 "$LOG"; exit 1; }
"$CMAKE" --build "$OUT/build" -j"$(nproc)" --target llama-bench >> "$LOG" 2>&1 || {
    echo "  BUILD FAILED:"; grep -iE "error:" "$LOG" | head -20; exit 1; }

echo "  built:   $OUT/build/bin/llama-bench"

# The probe must be in what was built. A green build says the file compiled; it
# does not say the instrumentation survived, and this project has already
# shipped one benchmark that measured a path it was not on.
N="$(objdump -T "$OUT/build/bin/libggml-cpu.so" 2>/dev/null \
     | grep -c bitnet_sgemm_cycles_add || true)"
if [ "${N:-0}" -eq 0 ]; then
    echo "  but libggml-cpu.so does not export bitnet_sgemm_cycles_add." >&2
    exit 1
fi
echo "  probe:   bitnet_sgemm_cycles_add present in libggml-cpu.so"
echo
echo "PASS -- the integration applies and compiles, and carries the probe."
