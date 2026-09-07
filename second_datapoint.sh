#!/usr/bin/env bash
#
# The second hardware data point, on any machine that is not this one.
#
# WHY THIS EXISTS
# ---------------
# Paper §9 carries the sharpest limitation this work has: every number comes
# from one Zen 2 part with `avx2 fma` and nothing else. Two facts make that
# limitation load-bearing rather than boilerplate:
#
#   - The profitability condition (18) turns on pi, the throughput of the
#     binding multiply port. This host has ONE 256-bit multiplier port and
#     measures 1.05 instructions per cycle. A reviewer ran the same benchmark on
#     an Intel part with AVX-512 and got 1.60 against 3.09 -- two ports. So the
#     analysis is microarchitecture-specific BEFORE VNNI is even considered.
#   - VPDPBUSD collapses the contraction into one instruction and leaves the
#     decode where it is, which moves (18) against a packed format. An
#     independent team building for VNNI stores ternary weights at EIGHT bits
#     for precisely that reason.
#
# WHAT THIS SCRIPT DOES AND DOES NOT MEASURE -- READ THIS BEFORE QUOTING IT
# ------------------------------------------------------------------------
# It measures the FIRST of those two: the kernels on a second microarchitecture.
# That is worth having and it is not the VNNI question.
#
# This header used to claim that one run on a VNNI part "turns the largest
# caveat in the paper into a measurement". That was FALSE, and a reviewer caught
# it the direct way: `objdump -d` over build/bench_alu and both kernel objects
# finds ZERO vpdpbusd, on a host whose /proc/cpuinfo reports avx512_vnni and
# with -march=native in the flags. No compiler contracts vpmaddubsw + vpaddw
# into VPDPBUSD as an idiom; the instruction has to be written. So a run on a
# VNNI machine measured the AVX2 kernels on an Intel part and nothing more.
#
# Step 5 below is the repair: it builds a kernel pair that DOES issue vpdpbusd,
# verifies the instruction is present in the object rather than assuming it, and
# times it against the AVX2 pair. On a host without VNNI it skips loudly. Until
# somebody runs step 5 on real hardware, §9.1's VNNI figures remain a static
# model whose own negative control failed.
#
# It needs no model, no Docker, and no privileges: everything it runs is
# self-contained C.
#
# WHAT IT DOES NOT DO
# It does not measure the model. bench_token replays the weight traffic of a
# token at the real tensor shapes, which is the part a second machine can
# answer; llama-bench numbers would need the checkpoint and a built image and
# are deliberately out of scope here.
#
# Usage:
#     scripts/second_datapoint.sh user@host            # over ssh
#     scripts/second_datapoint.sh --local              # on the machine itself
#
# Output goes to stdout and to results/second_datapoint_<hostid>.txt if run
# from a checkout.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/purpleskulll/bitnet-t5b.git}"
TARGET="${1:-}"

if [ -z "$TARGET" ]; then
    echo "usage: $0 user@host | --local" >&2
    exit 2
fi

# The whole measurement as one payload, so it can be piped into a fresh machine
# with nothing on it but a compiler, git and python3.
read -r -d '' PAYLOAD <<'REMOTE' || true
set -euo pipefail
echo "=================================================================="
echo " SECOND DATA POINT"
echo "=================================================================="
echo
echo "--- what this machine is ---"
grep -m1 "model name" /proc/cpuinfo || true
echo "  cores    : $(nproc)"
printf "  features : "
grep -m1 -o -E "avx2|fma|avx512f|avx512vnni|avx_vnni" /proc/cpuinfo | sort -u | tr '\n' ' '
echo
printf "  VNNI     : "
if grep -qE "avx512vnni|avx_vnni" /proc/cpuinfo; then
    echo "YES -- this is the case the paper cannot test at home"
else
    echo "no"
fi
echo

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
echo "--- clone and build ---"
git clone --depth 1 -q "$REPO_URL" repo
cd repo
make -s >/dev/null 2>&1 || make >/dev/null
# The i2_s reference is not redistributed; fetch it so bench_alu has a baseline.
if make i2s-ref >/dev/null 2>&1 && make bench >/dev/null 2>&1; then
    HAVE_I2S=1
else
    HAVE_I2S=0
    echo "  NOTE: the i2_s reference could not be fetched (no network?)."
    echo "        bench_ports still runs; the comparative benchmarks do not."
fi
echo "  built: $(ls build | tr '\n' ' ')"
echo

echo "--- 1. instruction port throughput  (this is the decisive one) ---"
echo "    On the paper's Zen 2: multiply class 1.05/cycle, cheap class 3.0-4.0."
echo "    Two multiplier ports would show roughly double on the first four."
taskset -c 1 ./build/bench_ports 2>/dev/null || ./build/bench_ports
echo

if [ "$HAVE_I2S" = 1 ]; then
    echo "--- 2. per-kernel arithmetic rate, memory removed ---"
    echo "    Paper: i2_s 81.93, t10 30.04, t5b 33.74 GMAC/s per core, S = 2.43."
    taskset -c 1 ./build/bench_alu 2>/dev/null || ./build/bench_alu
    echo
    echo "--- 3. arithmetic against memory, by core count (the surplus) ---"
    echo "    Paper: 1.35x at one thread rising to 3.03x at six."
    ./build/bench_threads 0.5 2>/dev/null || true
    echo
    echo "--- 4. one token's weight traffic at the model's real shapes ---"
    echo "    Paper at four threads: i2_s 13.19 ms, t5b 16.90 ms, ratio 0.780."
    ./build/bench_token 4 5 2>/dev/null | tail -12 || true
fi
echo
echo "--- 5. VNNI: the one thing steps 1-4 CANNOT answer ---"
echo "    Steps 1-4 run AVX2 kernels. On a VNNI part they measure a second"
echo "    microarchitecture, NOT VNNI -- no compiler turns vpmaddubsw+vpaddw"
echo "    into VPDPBUSD, so the AVX2 binaries contain none of it. Checked:"
for b in build/bench_alu build/obj/ternary_t5b.o; do
    [ -f "$b" ] && printf "      %-28s vpdpbusd: %s\n" "$b" \
        "$(objdump -d "$b" 2>/dev/null | grep -c vpdpbusd || echo '?')"
done
echo
if [ -x build/bench_vnni ]; then
    echo "    build/bench_vnni contains $(objdump -d build/bench_vnni 2>/dev/null | grep -c vpdpbusd) vpdpbusd instructions."
    if taskset -c 1 ./build/bench_vnni 2>/dev/null || ./build/bench_vnni; then
        :
    else
        rc=$?
        [ "$rc" = 3 ] && echo "    (exit 3: no VNNI on this CPU -- expected, and the correct outcome.)"
    fi
else
    echo "    build/bench_vnni was not built: this compiler accepts neither"
    echo "    -mavx512vnni nor -mavxvnni. The VNNI question stays open."
fi
echo
echo "    Algorithm check, which runs everywhere (VPDPBUSD modelled in AVX2):"
./build/bench_vnni_emu 2>/dev/null | grep -E "correctness|Built with" || \
    echo "      bench_vnni_emu not built."
echo
echo "--- suites, so the numbers above come from a build that is correct ---"
./build/test_t5b 2>&1 | tail -2
./build/test_t10 2>&1 | tail -2
echo
echo "=================================================================="
echo " done"
REMOTE

if [ "$TARGET" = "--local" ]; then
    REPO_URL="$REPO_URL" bash -c "$PAYLOAD"
else
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$TARGET" \
        "REPO_URL='$REPO_URL' bash -s" <<< "$PAYLOAD"
fi
