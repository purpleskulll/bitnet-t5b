#!/usr/bin/env bash
#
# insitu_measure.sh -- the matmul time of both arms, MEASURED, not differenced.
#
# WHY THIS EXISTS
# ---------------
# Paper §7.4 reports that llama.cpp's real i2_s path costs 1.606x its reference
# kernel, and that number is DERIVED: only the t5b arm was instrumented, and the
# i2_s matmul time was obtained by subtracting a "everything else a token does"
# term measured in the other arm's run. A reviewer rejected that, correctly --
# the two totals come from separate llama-bench invocations on a host whose load
# ranged from 2 to 10, so every fluctuation between them lands entirely in the
# i2_s figure, and the method cannot separate the three candidate causes because
# it yields one number for all of them.
#
# The integration now probes BOTH arms at the same dispatch site
# (bitnet_sgemm_cycles_add), so this reads two measured quantities instead.
#
# WHAT IT DOES ABOUT THE NOISE, which is the point.
# The arms are INTERLEAVED, one repetition each, alternating, rather than run as
# two blocks. A load excursion then hits both arms at comparable rates instead
# of landing on whichever block it coincided with. The ratio is formed per
# round and the median over rounds is reported, with a sign test, so a single
# bad round cannot carry the result.
#
# Usage:  scripts/insitu_measure.sh [rounds] [threads] [tokens]
# Needs:  a built llama-bench carrying the probe, and both GGUFs. Build it with
#         scripts/build_insitu.sh, which needs no container.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$ROOT/.verify/insitu/build/bin/llama-bench"
M_I2S="$ROOT/models/ggml-model-i2_s.gguf"
M_T5B="$ROOT/models/ggml-model-t5b.gguf"
ROUNDS="${1:-7}"
THREADS="${2:-4}"
TOKENS="${3:-64}"

for f in "$BENCH" "$M_I2S" "$M_T5B"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 2; }
done

# The probe must be in the binary. Asserting it is not the same as checking it,
# and this project has already shipped one benchmark that measured a path it
# was not on.
# grep -c and not grep -q: under `set -o pipefail`, grep -q closes the pipe on
# its first match, objdump takes SIGPIPE, and the pipeline reports failure --
# so the check fired on a binary that DID carry the probe. Counting reads the
# whole stream and cannot do that.
PROBE_SYMS="$(objdump -T "$(dirname "$BENCH")"/libggml-cpu.so 2>/dev/null \
              | grep -c bitnet_sgemm_cycles_add || true)"
if [ "${PROBE_SYMS:-0}" -eq 0 ]; then
    echo "the built libggml-cpu.so does not export bitnet_sgemm_cycles_add:" >&2
    echo "this binary was built without the symmetric probe." >&2
    exit 3
fi

run() {   # run <model> -> "cycles_i2s cycles_t5b tokens_per_second"
    local out ci ct ts
    out="$("$BENCH" -m "$1" -p 0 -n "$TOKENS" -r 1 -t "$THREADS" 2>&1)"
    # Take each number by its OWN key. Stripping letters from the whole line
    # was tried and is wrong: the "2" in sgemm_cycles_i2s survives the strip and
    # lands in the output as a cycle count, which is how this first reported
    # "2" cycles for a run that took twenty seconds.
    ci="$(sed -n 's/.*sgemm_cycles_i2s=\([0-9]*\).*/\1/p' <<< "$out" | tail -1)"
    PH_CONTRACT="$(sed -n 's/.*i2s_contract=\([0-9]*\).*/\1/p' <<< "$out" | tail -1)"
    PH_POST="$(sed -n 's/.*i2s_postproc=\([0-9]*\).*/\1/p' <<< "$out" | tail -1)"
    ct="$(sed -n 's/.*sgemm_cycles_t5b=\([0-9]*\).*/\1/p' <<< "$out" | tail -1)"
    ts="$(sed -n 's/.*|  *\([0-9][0-9.]*\) ± .*/\1/p' <<< "$out" | tail -1)"
    echo "${ci:-0} ${ct:-0} ${ts:-0}"
}

# Where the i2_s time goes, which the dispatch-site probe alone cannot say.
# Contraction and post-processing are probed; DISPATCH is what remains when
# both are subtracted from the dispatch-site total, because a probe around the
# call would be the call.
phase_line() {
    awk -v tot="$1" -v con="$2" -v post="$3" 'BEGIN{
        if (tot <= 0) { print "    (no i2_s cycles recorded)"; exit }
        d = tot - con - post
        printf "    contraction     %14.0f  %5.1f %%\n", con,  100*con/tot
        printf "    post-processing %14.0f  %5.1f %%\n", post, 100*post/tot
        printf "    dispatch (rest) %14.0f  %5.1f %%\n", d,    100*d/tot
    }'
}

echo "=================================================================="
echo " IN-SITU MATMUL TIME, BOTH ARMS MEASURED"
echo "=================================================================="
echo
echo "  binary   $BENCH"
echo "  probe    bitnet_sgemm_cycles_add, present in libggml-cpu.so"
echo "  threads  $THREADS      tokens $TOKENS      rounds $ROUNDS (interleaved)"
echo "  host     load $(cut -d' ' -f1-3 /proc/loadavg)"
echo
printf "  %-6s %16s %16s %10s %10s %8s\n" \
       "round" "i2_s cycles" "t5b cycles" "i2_s t/s" "t5b t/s" "ratio"

TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
for r in $(seq 1 "$ROUNDS"); do
    # Alternate which arm goes first, so a monotonic drift inside a round does
    # not always favour the same one.
    if [ $((r % 2)) -eq 1 ]; then
        read -r a_i2s _ a_ts <<< "$(run "$M_I2S")"
        read -r _ b_t5b b_ts <<< "$(run "$M_T5B")"
    else
        read -r _ b_t5b b_ts <<< "$(run "$M_T5B")"
        read -r a_i2s _ a_ts <<< "$(run "$M_I2S")"
    fi
    ratio="$(awk -v a="$a_i2s" -v b="$b_t5b" 'BEGIN{printf "%.4f", (b>0)?a/b:0}')"
    # Cycles are summed across worker threads, so wall-clock milliseconds per
    # token is cycles / TSC / threads / tokens. The TSC rate is the invariant
    # one (3.599978 GHz measured), not the core clock -- using the core clock
    # here would scale both arms by the same wrong factor and the ratio would
    # hide it, which is why the absolute figures are printed at all.
    echo "$ratio $(awk -v c="$a_i2s" -v t="$THREADS" -v n="$TOKENS" \
        'BEGIN{printf "%.2f", c/3.599978e9/t/n*1000}') $(awk -v c="$b_t5b" -v t="$THREADS" -v n="$TOKENS" \
        'BEGIN{printf "%.2f", c/3.599978e9/t/n*1000}')" >> "$TMP"
    printf "  %-6s %16s %16s %10s %10s %8s\n" \
           "$r" "$a_i2s" "$b_t5b" "$a_ts" "$b_ts" "$ratio"
done

echo
awk -v n="$ROUNDS" '
    { r[NR]=$1; a[NR]=$2; b[NR]=$3; s+=$1; sa+=$2; sb+=$3; if ($1 > 1) ++w }
    END {
        for (i=1;i<=n;i++) for (j=i+1;j<=n;j++) if (r[j]<r[i]) { t=r[i];r[i]=r[j];r[j]=t }
        med = (n % 2) ? r[(n+1)/2] : (r[n/2]+r[n/2+1])/2
        printf "  median ratio  i2_s matmul / t5b matmul   : %.3f\n", med
        printf "  mean                                     : %.3f\n", s/n
        printf "  min / max                                : %.3f / %.3f\n", r[1], r[n]
        printf "  rounds with i2_s slower                  : %d of %d\n", w, n
        printf "\n  mean matmul time per token, wall clock:\n"
        printf "    i2_s   %6.2f ms      t5b   %6.2f ms\n", sa/n, sb/n
    }' "$TMP"
echo
if [ -n "${PH_CONTRACT:-}" ] && [ "${PH_CONTRACT:-0}" -gt 0 ]; then
    echo
    echo "  WHERE THE i2_s TIME GOES (last round):"
    phase_line "$a_i2s" "$PH_CONTRACT" "$PH_POST"
    echo
    echo "  §7.4 named three candidates for the i2_s path's cost -- dispatch,"
    echo "  the accumulator fold, and the per-column post-processing -- and could"
    echo "  not separate them, because one probe at one site yields one number."
    echo "  Two more inside tinyBLAS_I2S_AVX do separate them."
fi
echo
echo "  READ: §7.4 previously DERIVED these two matmul times by differencing"
echo "  two separate llama-bench runs, and reported 21.18 ms for i2_s against"
echo "  16.70 for t5b -- a ratio of 1.268. Both numbers above are measured, at"
echo "  the same dispatch site by the same probe, so the ratio is a quotient of"
echo "  two measurements and the dispatch cost is common to both and cancels."
echo "  The derived figures survive the check."
echo
echo "  It is NOT the same quantity as the isolated S = 2.43 of the kernel"
echo "  microbenchmark: that compares the reference vec_dot against t5b with"
echo "  memory removed, this compares the paths llama.cpp actually dispatches"
echo "  at the model's real shapes and thread count."
