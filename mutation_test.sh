#!/usr/bin/env bash
#
# mutation_test.sh -- does the suite actually catch a broken kernel?
#
# WHY THIS EXISTS
# ---------------
# test_t5b reports 80,205 checks and test_t5s another 97,529, and the paper
# quotes those totals. A check count is a measure of effort, not of power: the
# three most expensive mistakes in this project were all GREEN checks answering
# the wrong question, and the standing rule that came out of them is to make a
# check fail on purpose once before believing it.
#
# This does that systematically. It injects one defect at a time into
# ternary_t5b.c or its header, rebuilds, runs the suite, and requires the suite
# to FAIL. A mutation the suite still passes is a hole, and this script exits
# non-zero and names it.
#
# TWO CLASSES OF MUTATION, and they are not equally interesting.
#
#   VECTOR-ONLY -- a magic multiplier, a shift, a lane width, a plane's
#   activation offset. These break the AVX2 decode while leaving the scalar
#   reference intact, so any comparison between the two must catch them. If one
#   of these survives, the suite is not comparing what it claims to.
#
#   SHARED -- T5B_POW3, TERNARY_T5B_DIGITS, the fold bound. These change the
#   packer and the scalar unpacker TOGETHER, so a pack-then-unpack round trip
#   agrees with itself and proves nothing. Catching these requires an
#   independent anchor: the density arithmetic, the byte bound, or the i2_s
#   codes. They are the ones worth watching.
#
# Usage:  scripts/mutation_test.sh [outfile]
# Exit:   0 every mutation killed; 1 at least one survived; 2 setup failure.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Runs UNCHANGED in both trees. The private one keeps the kernel in
# src_modifications/ with its suite under tests/; the public one has both in
# src/. Locating beats assuming: this file is copied between the two, and a copy
# that needs editing on arrival is a copy that drifts.
if   [ -d "$ROOT/src_modifications" ]; then SRC="$ROOT/src_modifications"; TESTDIR="tests"
elif [ -d "$ROOT/src" ];              then SRC="$ROOT/src";               TESTDIR="."
elif [ -d "$(dirname "${BASH_SOURCE[0]}")/src" ]; then
     ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; SRC="$ROOT/src"; TESTDIR="."
else
     echo "mutation_test.sh: no kernel directory found (tried src_modifications/, src/)" >&2
     exit 2
fi
OUT="${1:-}"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

CFLAGS="-O3 -mavx2 -mfma -march=native -Wall -std=c11"

# label | file | sed program | class
#
# Class `equivalent` marks a mutation that is NOT a defect on any reachable
# input, proved exhaustively rather than argued. Those are expected to SURVIVE,
# and this script fails if one is ever killed -- because that would mean the
# kernel no longer has the property the proof rests on. Both entries below were
# found by this script reporting them as holes; checking the claim instead of
# filing it is what turned them into checks.
MUTATIONS=(
  "magic m1 (floor x/3) off by one|ternary_t5b.c|s/21846/21845/|vector"
  "magic m2 (floor x/9) off by one|ternary_t5b.c|s/7282/7281/|vector"
  "magic m3 (floor x/27) off by one|ternary_t5b.c|s/2428/2427/|vector"
  "magic m4 (floor x/81): 811 -> 810|ternary_t5b.c|s/811/810/|equivalent"
  "t5b_triple computes 5y not 3y|ternary_t5b.c|s/_mm256_slli_epi16(y, 1)/_mm256_slli_epi16(y, 2)/|vector"
  "odd-byte view shifted 7 not 8|ternary_t5b.c|s/_mm256_srli_epi16(x, 8)/_mm256_srli_epi16(x, 7)/|vector"
  "low-byte mask 0x007F not 0x00FF|ternary_t5b.c|s/set1_epi16(0x00FF)/set1_epi16(0x007F)/|vector"
  "digit subtract in words not bytes|ternary_t5b.c|s/_mm256_sub_epi8/_mm256_sub_epi16/g|equivalent"
  "planes 4 and 3 read swapped activations|ternary_t5b.c|s/a + 4 \* TERNARY_T5B_BYTES/a + 3 * TERNARY_T5B_BYTES/|vector"
  "packer digit weights 1,3,9,27,81 -> 1,3,9,27,80|ternary_t5b.c|s/27u, 81u/27u, 80u/|shared"
  "packer digit weights base 4 not base 3|ternary_t5b.c|s/{ 1u, 3u, 9u, 27u, 81u }/{ 1u, 4u, 16u, 64u, 256u }/|shared"
  "four digits per byte instead of five|ternary_t5b.h|s/define TERNARY_T5B_DIGITS     5/define TERNARY_T5B_DIGITS     4/|shared"
  "accumulator fold at 13 blocks, past the int16 bound|ternary_t5b.h|s/define TERNARY_T5B_FOLD      12/define TERNARY_T5B_FOLD      13/|shared"
  "accumulator fold at 32 blocks, upstream's value|ternary_t5b.h|s/define TERNARY_T5B_FOLD      12/define TERNARY_T5B_FOLD      32/|shared"
)

# The two equivalence proofs, run here rather than asserted in a comment. Each
# is exhaustive over the whole input domain, not a sample.
prove_equivalences() {
    python3 - <<'PY'
M = (21846, 7282, 2428, 811)
def mulhi16(x, m): return ((x * m) >> 16) & 0xFFFF
def triple(y):     return ((y << 1) + y) & 0xFFFF
def sub8(a, b):
    return (((a & 0xFF) - (b & 0xFF)) & 0xFF) | ((((a >> 8) - (b >> 8)) & 0xFF) << 8)
def sub16(a, b):   return (a - b) & 0xFFFF

bad = [x for x in range(256) if mulhi16(x, 810) != mulhi16(x, 811)]
print(f"  m4 811 vs 810: differ on {len(bad)} of 256 byte values"
      f"{'  <-- NOT equivalent' if bad else '  (and both equal floor(x/81))'}")
assert not [x for x in range(256) if mulhi16(x, 811) != x // 81]

d = b = 0
for x in range(65536):
    xe, xo = x & 0x00FF, x >> 8
    q = [(mulhi16(xe, m) | (mulhi16(xo, m) << 8)) & 0xFFFF for m in M]
    for a, y in ((q[2], q[3]), (q[1], q[2]), (q[0], q[1]), (x, q[0])):
        t = triple(y)
        if sub8(a, t) != sub16(a, t): d += 1
        if (a & 0xFF) < (t & 0xFF):   b += 1
print(f"  vpsubb vs vpsubw over all 65,536 word lanes: differ on {d}, "
      f"low byte borrows on {b}"
      f"{'  <-- NOT equivalent' if d else ''}")
PY
}

emit() { if [ -n "$OUT" ]; then tee -a "$OUT"; else cat; fi; }

{
echo "=================================================================="
echo " MUTATION TEST -- can the suite tell a broken kernel from a good one?"
echo "=================================================================="
echo
echo "Reproduce:  scripts/mutation_test.sh"
echo "Compiler:   $(${CC:-gcc} --version | head -1)"
echo "Flags:      $CFLAGS"
echo

# The control that has to come first: the UNMUTATED tree must pass. Without it
# a build error would read as "every mutation killed" and this whole script
# would be the kind of check it exists to catch.
cp -r "$SRC" "$W/base"
if ! (cd "$W/base" && ${CC:-gcc} $CFLAGS $TESTDIR/test_t5b.c ternary_t5b.c -o t5b_test \
        > "$W/base.build" 2>&1) ; then
    echo "CONTROL FAILED: the unmutated tree does not build."; sed 's/^/    /' "$W/base.build"
    exit 2
fi
BASE_OUT="$("$W/base/t5b_test" 2>&1 || true)"
if ! grep -q "RESULT: PASS" <<< "$BASE_OUT"; then
    echo "CONTROL FAILED: the unmutated suite does not pass."; sed 's/^/    /' <<< "$BASE_OUT" | tail -5
    exit 2
fi
# The TOTAL, which is the last such line. Taking the first reported 978 -- the
# count of section (a) alone -- and made this script look like it was running a
# hundredth of the suite it was actually running.
echo "control: unmutated suite PASSES ($(grep -oE '[0-9]+ checks, [0-9]+ failures' <<< "$BASE_OUT" | tail -1))"
echo
echo "equivalence proofs, exhaustive over the whole input domain:"
prove_equivalences
echo

printf "%-56s %-11s %s\n" "mutation" "class" "verdict"
printf "%-56s %-11s %s\n" "--------" "-----" "-------"

survived=0; killed=0; unbuildable=0; wrong_equiv=0
for m in "${MUTATIONS[@]}"; do
    IFS='|' read -r label file prog class <<< "$m"
    rm -rf "$W/m"; cp -r "$SRC" "$W/m"
    sed -i "$prog" "$W/m/$file"
    if cmp -s "$W/m/$file" "$SRC/$file"; then
        printf "%-56s %-11s %s\n" "$label" "$class" "NOT APPLIED -- pattern did not match"
        survived=$((survived+1)); continue
    fi
    if ! (cd "$W/m" && ${CC:-gcc} $CFLAGS $TESTDIR/test_t5b.c ternary_t5b.c -o t5b_test \
            > "$W/m.build" 2>&1); then
        # A mutation that will not compile is not evidence about the suite. It is
        # reported as its own outcome rather than counted as a kill, which would
        # flatter the result.
        printf "%-56s %-11s %s\n" "$label" "$class" "does not compile -- no evidence"
        unbuildable=$((unbuildable+1)); continue
    fi
    # `|| true` on BOTH the run and the count: a mutant that segfaults or divides
    # by zero prints no failure count, and an earlier version of this script died
    # at that point under set -e, silently truncating the table after ten rows.
    res="$("$W/m/t5b_test" 2>&1 || true)"; rc=$?
    if grep -q "RESULT: PASS" <<< "$res"; then
        if [ "$class" = equivalent ]; then
            printf "%-56s %-11s %s\n" "$label" "$class" "survived, as PROVED above"
        else
            printf "%-56s %-11s %s\n" "$label" "$class" "*** SURVIVED ***"
            survived=$((survived+1))
        fi
    else
        n=$(grep -oE "[0-9]+ failures" <<< "$res" | tail -1 || true)
        [ -z "$n" ] && n="crashed, no summary"
        if [ "$class" = equivalent ]; then
            printf "%-56s %-11s %s\n" "$label" "$class" "*** KILLED, but proved equivalent ***"
            wrong_equiv=$((wrong_equiv+1))
        else
            printf "%-56s %-11s %s\n" "$label" "$class" "killed ($n)"
            killed=$((killed+1))
        fi
    fi
done

echo
echo "  killed $killed   survived $survived   uncompilable $unbuildable"
echo
if [ "$wrong_equiv" -gt 0 ]; then
    echo "  A mutation proved EQUIVALENT was killed. The proof above and the"
    echo "  kernel no longer agree; one of them changed. Read both before"
    echo "  touching anything else."
    exit 1
fi
if [ "$survived" -gt 0 ]; then
    echo "  A SURVIVING MUTATION IS A HOLE IN THE SUITE, not a curiosity: the"
    echo "  kernel was broken in a way the whole suite could not see. Before"
    echo "  writing a test, check whether it is a defect AT ALL on reachable"
    echo "  input -- the first run of this script reported two holes and both"
    echo "  turned out to be exactly equivalent."
    exit 1
fi
echo "  Every injected defect was caught, and every mutation that survived was"
echo "  proved beforehand to be no defect. The check count is doing work."
echo "=================================================================="
} | emit
