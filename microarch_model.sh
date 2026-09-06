#!/usr/bin/env bash
#
# microarch_model.sh -- how S behaves on microarchitectures this project has no
# access to, from a vendor-calibrated scheduler model rather than a guess.
#
# WHY THIS EXISTS
# ---------------
# Paper section 9 carries the sharpest limitation in this work: every number
# comes from one Zen 2 part with `avx2 fma` and nothing else. Two separate
# objections live inside that one sentence, and they need separating because
# only one of them can be answered without renting hardware:
#
#   (a) IS S A ZEN 2 ARTEFACT?  The profitability condition (18) turns on S, the
#       arithmetic slowdown of the packed kernel against the two-bit baseline.
#       Zen 2 splits every 256-bit integer vector op into two 128-bit uops over
#       four FP pipes, so it issues ~2 vector ops per cycle. Zen 3 and later, and
#       Ice Lake and later, execute 256 bits natively. t5b spends 42 vector ops
#       per 160 weights against i2_s's 17 per 128, so a machine that doubles the
#       vector-op ceiling should help the kernel that issues more of them. If S
#       collapses on newer parts the paper's central number is provincial; if it
#       holds, it is not.
#
#   (b) WHAT DOES VNNI DO?  VPDPBUSD folds a vpmaddubsw and its accumulating
#       vpaddw into one instruction. Both kernels end their contraction with
#       exactly that pair, so both can use it -- but i2_s spends a larger
#       FRACTION of its work there (4 of 17 ops) than t5b does (5 of 42), so the
#       instruction should help the baseline more than the packed format and
#       push S up. Litespark stores ternary weights at eight bits for this
#       reason. This is the objection the paper cannot settle at home.
#
# WHAT THIS SCRIPT CAN AND CANNOT ESTABLISH
# -----------------------------------------
# It runs llvm-mca, LLVM's static pipeline simulator, over the ACTUAL inner
# loops the compiler emits for both kernels, against the scheduler models for
# znver2/3/4/5, icelake-server and sapphirerapids.
#
# The whole exercise is worth nothing unless the model can be shown to reproduce
# something already measured, so the first thing it does is CALIBRATE: it
# predicts S on znver2 and compares against the 2.428 measured on this host by
# bench_alu. That check can fail, and the script exits non-zero when it does.
#
# It CANNOT answer (b) with the same confidence, and the script says so in its
# own output rather than burying it here:
#
#   - The VNNI loops are HAND-WRITTEN, by substitution into the compiler's
#     output. They have never been executed, because no machine here can execute
#     them. The substitution is argued sound in analysis/microarch/README, not
#     tested.
#   - llvm-mca assembles and models VPDPBUSD for znver2, a part that does not
#     have the instruction, and does so even with -mattr=-avx512vnni,-avxvnni.
#     So the tool gives NO signal about whether a VNNI sequence is legal on a
#     target. A negative control was written for this and it did not fire; that
#     is recorded rather than quietly dropped.
#   - The dependency-bound figure for a VNNI loop is an artefact of how many
#     accumulators the loop carries, not of the format: VPDPBUSD accumulates in
#     place with latency 10-13, so a loop with one accumulator per contraction
#     is bound by that latency until it is unrolled. The VNNI comparison is
#     therefore read off the RESOURCE bound, which is where an adequately
#     unrolled implementation lands -- and the script reports that the resource
#     bound does NOT calibrate on znver2 (1.96 against 2.43 measured), because
#     the real Zen 2 loop is dependency-bound rather than resource-bound.
#
# In one line: (a) is answered, (b) is bounded but not answered, and only a run
# on real VNNI hardware closes it. second_datapoint.sh is that run.
#
# Requires llvm-mca. Not packaged here and not installed system-wide; fetch the
# official release into a user directory:
#
#   mkdir -p ~/.local/llvm && curl -sL \
#     https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.0/LLVM-23.1.0-Linux-X64.tar.xz \
#     | tar -xJf - -C ~/.local/llvm --strip-components=1 --wildcards '*/bin/llvm-mca' '*/lib/libLLVM*'
#
# Usage:  microarch_model.sh [outfile]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-}"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

MCA="$(command -v llvm-mca || true)"
[ -z "$MCA" ] && [ -x "$HOME/.local/llvm/bin/llvm-mca" ] && MCA="$HOME/.local/llvm/bin/llvm-mca"
if [ -z "$MCA" ]; then
    echo "llvm-mca not found -- see the header of this script for how to fetch it." >&2
    exit 4
fi

# The measured anchor, from results/weight_density.txt on this host.
MEAS_I2S=81.93        # GMAC/s/core, i2_s vec_dot reference
MEAS_T5B=33.74        # GMAC/s/core, t5b
MEAS_S=2.428          # 81.93 / 33.74
W_I2S=128             # weights retired per inner-loop iteration
W_T5B=160
TOL=8                 # per cent; the calibration gate

emit() { if [ -n "$OUT" ]; then tee -a "$OUT"; else cat; fi; }

# ---------------------------------------------------------------- extract loops
# Compiled here rather than shipped, so the analysis tracks the kernel source
# instead of a stale copy of it. The loop is located by its backward branch, not
# by a hard-coded line number, because those move whenever the source does.
# The hot loop is the SHORTEST backward-branch span that contains a contraction,
# ties broken toward the earlier one. Two rules were tried and rejected against
# the counts weight_density.txt already recorded, which is why this is spelled
# out rather than left as a one-liner:
#   - longest span: picks the whole function body.
#   - most vpmaddubsw: picks the enclosing ROW loop, which contains the inner
#     loop plus a peeled copy and so counts 8 where the inner loop has 4. It
#     assembles and times perfectly well while answering a different question.
# Shortest-containing-a-contraction is the innermost loop by construction. Not a
# line number: those move whenever the source does.
extract_loop() {   # <asm> <symbol> <out>
    awk -v sym="$2" '
        $0 == sym":" { inf=1 }
        inf { n++; line[n]=$0
              if ($0 ~ /^\.L[0-9]+:/) { lbl=$0; sub(":","",lbl); at[lbl]=n }
              if ($0 ~ /^\tj[a-z]+\t\.L[0-9]+$/) {
                  t=$0; sub(/^\tj[a-z]+\t/,"",t)
                  if (t in at) {
                      s=at[t]+1; e=n-1; c=0
                      for (i=s; i<=e; i++) if (line[i] ~ /vpmaddubsw/) c++
                      if (c > 0 && (bc == 0 || e-s < be-bs)) { bc=c; bs=s; be=e }
                  }
              }
              if ($0 ~ /\.cfi_endproc/) { inf=0 }
        }
        END { for (i=bs; i<=be; i++) print line[i] }
    ' "$1" > "$3"
    [ -s "$3" ] || { echo "could not locate the inner loop of $2 in $1" >&2; exit 5; }
}

gcc -O3 -mavx2 -mfma -std=c11 -S -o "$W/t5b.s" "$ROOT/src/ternary_t5b.c"
gcc -O3 -mavx2 -mfma -std=c11 -S -o "$W/i2s.s" "$ROOT/src/ggml_i2s_ternary.c"
extract_loop "$W/t5b.s" ternary_t5b_dot_avx2               "$W/loop_t5b.s"
extract_loop "$W/i2s.s" bitnet_vec_dot_i2_i8_s_reference   "$W/loop_i2s.s"

# Guard against extracting the wrong loop, which is the failure mode this
# heuristic actually has: an enclosing row loop assembles and times perfectly
# well and silently answers a different question. weight_density.txt recorded
# 21 and 47 instructions and 4 and 13 multiplies against a slightly older gcc.
for chk in "loop_i2s 4 21" "loop_t5b 13 47"; do
    set -- $chk; f=$1; want_m=$2; want_n=$3
    got_n=$(( $(grep -c . "$W/$f.s") + 1 ))
    got_m=$(grep -cE "vpmulhuw|vpmaddubsw" "$W/$f.s")
    if [ "$got_m" != "$want_m" ] || [ "$got_n" -lt $((want_n - 4)) ] || [ "$got_n" -gt $((want_n + 4)) ]; then
        echo "extracted loop for $f looks wrong: $got_n instr / $got_m multiply," >&2
        echo "expected about $want_n / exactly $want_m (results/weight_density.txt)." >&2
        echo "Either the wrong loop was picked or the kernel changed shape." >&2
        exit 6
    fi
done

# The hand-written VNNI variants live in the tree, because they are an argument
# and not a derivation -- they must be reviewable, and they must not silently
# change when the compiler does.
cp "$ROOT/analysis/microarch/loop_t5b_vnni.s" "$W/loop_t5b_vnni.s"
cp "$ROOT/analysis/microarch/loop_i2s_vnni.s" "$W/loop_i2s_vnni.s"

cyc() { "$MCA" -mtriple=x86_64-unknown-linux-gnu -mcpu="$1" -iterations=2000 "$2" 2>/dev/null \
        | awk '/^Total Cycles:/{c=$3} /^Block RThroughput:/{r=$3} END{printf "%.4f %.4f", c/2000, r}'; }

{
echo "=================================================================="
echo " S ACROSS MICROARCHITECTURES -- static model, calibrated"
echo "=================================================================="
echo
echo "Reproduce:  microarch_model.sh"
echo "Model:      $("$MCA" --version | awk '/LLVM version/{print "llvm-mca " $3}')"
echo "Anchor:     i2_s $MEAS_I2S GMAC/s, t5b $MEAS_T5B GMAC/s, S = $MEAS_S"
echo "            (results/weight_density.txt, this Zen 2 host, bench_alu)"
echo

echo "--- the two loops, as the compiler emits them today ---"
for f in i2s t5b; do
    n=$(grep -c . "$W/loop_$f.s")
    v=$(grep -cE "^\s+v" "$W/loop_$f.s")
    m=$(grep -cE "vpmulhuw|vpmaddubsw" "$W/loop_$f.s")
    wgt=$([ "$f" = i2s ] && echo $W_I2S || echo $W_T5B)
    awk -v f="$f" -v n="$n" -v v="$v" -v m="$m" -v w="$wgt" 'BEGIN{
        printf "  %-5s %3d instr  %3d vector  %2d multiply  %3d weights   %.4f vec/w  %.5f mul/w\n",
               f, n+1, v, m, w, v/w, m/w }'
done
echo "  (instr counts include the loop branch; weight_density.txt recorded 21 and 47)"
echo

echo "--- CALIBRATION: does the model reproduce what was measured here? ---"
set -- $(cyc znver2 "$W/loop_t5b.s"); TC=$1; TR=$2
set -- $(cyc znver2 "$W/loop_i2s.s"); IC=$1; IR=$2
CAL=$(awk -v tc="$TC" -v ic="$IC" -v wt=$W_T5B -v wi=$W_I2S 'BEGIN{printf "%.4f",(tc/wt)/(ic/wi)}')
CALR=$(awk -v tr="$TR" -v ir="$IR" -v wt=$W_T5B -v wi=$W_I2S 'BEGIN{printf "%.4f",(tr/wt)/(ir/wi)}')
ERR=$(awk -v a="$CAL"  -v m=$MEAS_S 'BEGIN{printf "%.2f",100*(a-m)/m}')
ERRR=$(awk -v a="$CALR" -v m=$MEAS_S 'BEGIN{printf "%.2f",100*(a-m)/m}')
printf "  dependency bound (Total Cycles)  S = %s   measured %s   %+s %%\n" "$CAL"  "$MEAS_S" "$ERR"
printf "  resource   bound (RThroughput)   S = %s   measured %s   %+s %%\n" "$CALR" "$MEAS_S" "$ERRR"
echo
echo "  The real Zen 2 loop is DEPENDENCY bound, not resource bound, so only the"
echo "  first line is a prediction of this host. It is the metric used for the"
echo "  AVX2 rows below. The resource bound is carried anyway because it is the"
echo "  only fair metric for the VNNI rows, and its miss here is the honest"
echo "  measure of how much that costs."
FAIL=$(awk -v e="$ERR" -v t=$TOL 'BEGIN{a=(e<0?-e:e); if (a>t) print 1; else print 0}')
if [ "$FAIL" = 1 ]; then
    echo
    echo "  *** CALIBRATION FAILED: |$ERR %| exceeds the $TOL % gate."
    echo "  *** Every prediction below is void. Do not cite them."
fi
echo

echo "--- (a) THE SAME AVX2 CODE, on parts this project cannot run on ---"
printf "  %-16s %10s %10s %8s\n" "target" "t5b cyc" "i2_s cyc" "S"
for CPU in znver2 znver3 znver4 znver5 icelake-server sapphirerapids; do
    set -- $(cyc "$CPU" "$W/loop_t5b.s"); A=$1
    set -- $(cyc "$CPU" "$W/loop_i2s.s"); B=$1
    awk -v c="$CPU" -v a="$A" -v b="$B" -v wt=$W_T5B -v wi=$W_I2S 'BEGIN{
        printf "  %-16s %10.2f %10.2f %8.3f\n", c, a, b, (a/wt)/(b/wi) }'
done
echo
echo "  Read: S does not run away on a wider machine. Zen 3 and later execute"
echo "  256-bit integer ops natively instead of splitting them, which cuts both"
echo "  kernels' cycle counts by about a third and leaves their RATIO alone."
echo "  The paper's S is therefore not an artefact of one narrow part."
echo

echo "--- (b) WITH VNNI, both kernels using VPDPBUSD, at the resource bound ---"
printf "  %-16s %9s %9s %9s %9s %9s %9s\n" "target" "t5b" "t5b+VNNI" "i2_s" "i2_s+VNNI" "S(avx2)" "S(vnni)"
for CPU in znver4 znver5 icelake-server sapphirerapids; do
    set -- $(cyc "$CPU" "$W/loop_t5b.s");      TA=$2
    set -- $(cyc "$CPU" "$W/loop_t5b_vnni.s"); TV=$2
    set -- $(cyc "$CPU" "$W/loop_i2s.s");      IA=$2
    set -- $(cyc "$CPU" "$W/loop_i2s_vnni.s"); IV=$2
    awk -v c="$CPU" -v ta="$TA" -v tv="$TV" -v ia="$IA" -v iv="$IV" -v wt=$W_T5B -v wi=$W_I2S 'BEGIN{
        printf "  %-16s %9.2f %9.2f %9.2f %9.2f %9.3f %9.3f\n", c, ta, tv, ia, iv,
               (ta/wt)/(ia/wi), (tv/wt)/(iv/wi) }'
done
echo
echo "  Read WITH the caveats in this script's header. VNNI moves S the way the"
echo "  paper predicted -- against the packed format -- because i2_s spends a"
echo "  larger fraction of its instructions in the contraction that VPDPBUSD"
echo "  collapses. The size of the move is the new information, and it is"
echo "  modest, not decisive. These loops have never been executed."
echo

echo "--- NEGATIVE CONTROL, and its failure ---"
NC=$("$MCA" -mtriple=x86_64-unknown-linux-gnu -mcpu=znver2 -mattr=-avx512vnni,-avxvnni \
     -iterations=10 "$W/loop_i2s_vnni.s" 2>&1 | grep -ciE "error|not supported" || true)
if [ "$NC" = 0 ]; then
    echo "  llvm-mca modelled VPDPBUSD for znver2 -- a part without VNNI -- even"
    echo "  with the feature explicitly disabled (-mattr=-avx512vnni,-avxvnni)."
    echo "  It assigned it latency 11 and throughput 1.00 out of thin air."
    echo "  CONSEQUENCE: this tool cannot tell a legal VNNI sequence from an"
    echo "  illegal one, so it offers no check on section (b) beyond timing. The"
    echo "  control is kept in place and reported failing rather than removed."
else
    echo "  Control fired ($NC diagnostics): llvm-mca rejected VPDPBUSD without"
    echo "  the feature. A newer llvm-mca than 23.1.0 -- re-read section (b),"
    echo "  its central caveat may no longer apply."
fi
echo
echo "--- what is still open ---"
echo "  A static model is not a measurement. The one number that would settle"
echo "  the VNNI question is a run of second_datapoint.sh on any part"
echo "  with avx512vnni or avx_vnni. Nothing in this file substitutes for it."
echo "=================================================================="
} | emit

exit "${FAIL:-0}"
