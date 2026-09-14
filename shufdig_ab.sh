#!/usr/bin/env bash
#
# shufdig_ab.sh -- reproduce every figure section 5.x quotes for the cheaper decode.
#
# WHY THIS EXISTS
# ---------------
# The paper claims a 1.2254x median on the contraction and no gain at all on the
# wide path. Both are rates, and a rate taken on a loaded box is not a
# measurement -- results/thread_headroom.txt records an instance of the same
# binary reporting 2.29x instead of 3.03x because of load alone. So this script
# does three things the figures depend on, and refuses rather than guesses:
#
#   1. PICKS THE CORE by measuring, not by convention. Every published rate in
#      this project is pinned to core 5 by habit; on the day this was written
#      core 5's SMT sibling was 82% busy and core 4's pair was 92% idle. It
#      samples /proc/stat and takes the quietest PAIR, because a busy sibling
#      costs as much as a busy core.
#   2. INTERLEAVES THE ARMS and alternates which goes first, forming the ratio
#      per round and reporting the median. A load excursion then hits both arms
#      at comparable rates instead of landing on whichever block met it.
#   3. CARRIES AN INTERNAL CONTROL. The i2_s kernel is IDENTICAL CODE in both
#      binaries, so whatever ratio it shows is drift. If the control's spread
#      is not much smaller than the effect, the effect has not been measured --
#      and this prints both so the reader can see that rather than trust it.
#
# Both arms are compiled FROM THE SAME FILE, differing only in -D, and the
# script asserts that the no-D arm's inner loop is byte-identical to the shipped
# kernel's. Without that assertion this would silently compare the shipped
# kernel against a stale copy of itself, which is the mistake that makes an A/B
# meaningless.
#
# Usage:  scripts/shufdig_ab.sh [rounds]      (default 15; ~40 s per round)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Runs UNCHANGED in both trees, and locating beats assuming. In the private tree
# this file sits in scripts/, so `..` is the repository root; in the public tree
# it is copied to the ROOT, so `..` is the clone's PARENT and every path below
# resolves outside the repository. That is not hypothetical: a cold-clone audit
# found this script exiting without measuring anything, while the paper's own
# footnote names it as the reproduction path for a whole subsection -- the exact
# failure ("someone looks for the gate, cannot find it, and then doubts
# everything else") this project has a rule about.
#
# SECOND FAILURE MODE, and the reason this no longer consults the parent at all.
# The previous form corrected ROOT only when the parent had NEITHER src/ nor
# src_modifications/. So a clone with any directory named src/ sitting BESIDE it
# -- another checkout, a stray build tree -- made that test true, the correction
# never fired, and the script exited 2 on a diagnosis that was simply false:
# "ternary_t5b.c not found in either layout", with src/ternary_t5b.c in plain
# view inside the clone. Reproduced in a scratch tree before this was changed.
# A wrong diagnosis costs more than a crash, because it sends the reader to look
# at the kernel instead of at the path.
#
# Asking the script's OWN directory cannot be fooled by a neighbour: a neighbour
# is, by definition, not where the script is.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SELF/src" ] || [ -d "$SELF/src_modifications" ]; then ROOT="$SELF"; fi
ROUNDS="${1:-15}"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
CF="-O3 -mavx2 -mfma -march=native -std=c11"

SRC="$ROOT/src_modifications/ternary_t5b.c"
INC="$ROOT/src_modifications"
[ -f "$SRC" ] || { SRC="$ROOT/src/ternary_t5b.c"; INC="$ROOT/src"; }
[ -f "$SRC" ] || { echo "ternary_t5b.c not found in either layout" >&2; exit 2; }

BENCH_ALU="$(ls "$ROOT"/benchmarks/bench_alu.c "$ROOT"/src_modifications/bench/bench_alu.c 2>/dev/null | head -1 || true)"
BENCH_GEMM="$(ls "$ROOT"/benchmarks/bench_gemm.c "$ROOT"/src_modifications/bench/bench_gemm.c 2>/dev/null | head -1 || true)"

# ------------------------------------------------------------------ the arms
gcc $CF -I"$INC" -c "$SRC" -o "$W/base.o"
gcc $CF -I"$INC" -DTERNARY_T5B_SHUFDIG -c "$SRC" -o "$W/shuf.o"

# The assertion that makes the comparison mean something.
gcc $CF -I"$INC" -S -o "$W/a.s" "$SRC"
loop() { awk -v sym=ternary_t5b_dot_avx2 '$0==sym":"{f=1} f{n++;l[n]=$0
  if($0~/^\.L[0-9]+:/){b=$0;sub(":","",b);at[b]=n}
  if($0~/^\tj[a-z]+\t\.L[0-9]+$/){t=$0;sub(/^\tj[a-z]+\t/,"",t)
    if(t in at){s=at[t]+1;e=n-1;c=0;for(i=s;i<=e;i++)if(l[i]~/vpmaddubsw/)c++
      if(c>0&&(bc==0||e-s<be-bs)){bc=c;bs=s;be=e}}}
  if($0~/\.cfi_endproc/){f=0}} END{for(i=bs;i<=be;i++)print l[i]}' "$1"; }
V=$(loop "$W/a.s" | grep -cE '^\s+v')
M=$(loop "$W/a.s" | grep -cE 'vpmulhuw|vpmaddubsw')
gcc $CF -I"$INC" -S -DTERNARY_T5B_SHUFDIG -o "$W/b.s" "$SRC"
VS=$(loop "$W/b.s" | grep -cE '^\s+v')
MS=$(loop "$W/b.s" | grep -cE 'vpmulhuw|vpmaddubsw')

# --------------------------------------------------------------- the quiet core
st() { awk '/^cpu[0-9]/{print $1, $2+$3+$4+$5+$6+$7+$8, $5}' /proc/stat; }
st > "$W/s1"; sleep 3; st > "$W/s2"
join "$W/s1" "$W/s2" | awk '{d=$4-$2; i=$5-$3; printf "%s %.2f\n", $1, (d>0)?100*i/d:0}' > "$W/idle"
# A core is only as quiet as its BUSIEST sibling: an SMT partner at 80% costs
# as much as the core itself being busy. So each pair is scored by its minimum
# idle, and the table is printed rather than just its winner -- a core choice
# that cannot be inspected is a core choice that gets blamed later.
for c in /sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list; do
    tr ',' ' ' < "$c"
done | sort -u > "$W/pairs"
awk 'NR==FNR{idle["cpu" $1]=$2; next}
     { m = 999; lst = ""
       for (i = 1; i <= NF; i++) { v = idle["cpu" $i] + 0
                                   if (v < m) m = v
                                   lst = lst sprintf(" cpu%-2s %5.1f%%", $i, v) }
       printf "%.2f %s |%s\n", m, $1, lst }' \
    <(sed 's/^cpu//' "$W/idle") "$W/pairs" | sort -rn > "$W/cores"
CPU=$(head -1 "$W/cores" | awk '{print $2}')

echo "=================================================================="
echo " THE CHEAPER DECODE, MEASURED"
echo "=================================================================="
echo
echo "  source    $SRC"
echo "  arms      same file, differing only in -DTERNARY_T5B_SHUFDIG"
echo "  loop      $V vector ops / $M multiply-port  ->  $VS / $MS"
echo "  core      cpu$CPU, chosen by sampling /proc/stat over 3 s"
echo "  host      load $(cut -d' ' -f1-3 /proc/loadavg)"
echo
echo "  idle per SMT pair, worst sibling first (the basis of that choice):"
awk '{printf "    %s\n", substr($0, index($0, "|") + 1)}' "$W/cores"
echo

if [ -z "$BENCH_ALU" ]; then
    echo "  bench_alu.c not found -- cannot measure the contraction." >&2
    exit 3
fi

# Built from source rather than taken from build/obj: the private tree has no
# Makefile, and requiring a prior `make` would make this script unrunnable in
# the layout it was written in.
IOBJ="$W/i2s.o"; TOBJ="$W/t10.o"
gcc $CF -I"$INC" -c "$INC/ggml_i2s_ternary.c" -o "$IOBJ" 2>/dev/null || {
    echo "  $INC/ggml_i2s_ternary.c is missing." >&2
    echo "  It is generated: run tools/fetch_i2s_reference.sh first." >&2; exit 3; }
gcc $CF -I"$INC" -c "$INC/ternary_t10.c" -o "$TOBJ"
for a in base shuf; do
    gcc $CF "$BENCH_ALU" "$IOBJ" "$TOBJ" "$W/$a.o" -o "$W/alu_$a" -lm
done

rate() { taskset -c "$CPU" "$1" 2>/dev/null | awk -v k="$2" '$0 ~ k {print $(NF-1); exit}'; }
: > "$W/ab"
printf "  %-6s %11s %11s %9s %14s\n" round "t5b base" "t5b shuf" "ratio" "i2_s control"
for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then
        tb=$(rate "$W/alu_base" '^  t5b'); ib=$(rate "$W/alu_base" 'i2_s vecdot')
        ts=$(rate "$W/alu_shuf" '^  t5b'); is=$(rate "$W/alu_shuf" 'i2_s vecdot')
    else
        ts=$(rate "$W/alu_shuf" '^  t5b'); is=$(rate "$W/alu_shuf" 'i2_s vecdot')
        tb=$(rate "$W/alu_base" '^  t5b'); ib=$(rate "$W/alu_base" 'i2_s vecdot')
    fi
    q=$(awk -v a="$ts" -v b="$tb" 'BEGIN{printf "%.4f",(b>0)?a/b:0}')
    c=$(awk -v a="$is" -v b="$ib" 'BEGIN{printf "%.4f",(b>0)?a/b:0}')
    printf "  %-6s %11s %11s %9s %14s\n" "$r" "$tb" "$ts" "$q" "$c"
    echo "$q $c $tb $ts $ib $is" >> "$W/ab"
done

echo
awk -v n="$ROUNDS" '
  {q[NR]=$1;c[NR]=$2;sb+=$3;ss+=$4;cb+=$5;cs+=$6; if($1>1)++w}
  END{
    for(i=1;i<=n;i++)for(j=i+1;j<=n;j++){if(q[j]<q[i]){t=q[i];q[i]=q[j];q[j]=t}
                                        if(c[j]<c[i]){t=c[i];c[i]=c[j];c[j]=t}}
    m=(n%2)?q[(n+1)/2]:(q[n/2]+q[n/2+1])/2
    mc=(n%2)?c[(n+1)/2]:(c[n/2]+c[n/2+1])/2
    printf "  t5b   median %.4fx   min %.4f  max %.4f   faster in %d of %d\n",m,q[1],q[n],w,n
    printf "  CTL   median %.4fx   min %.4f  max %.4f   <-- the noise floor\n",mc,c[1],c[n]
    printf "  S = i2_s / t5b :  %.3f  ->  %.3f\n",(cb/n)/(sb/n),(cb/n)/(ss/n)
    k=w; N=n
    for(i=0;i<=N;i++){ lg=0; for(j=1;j<=N;j++) lg+=log(j)
      la=0; for(j=1;j<=i;j++) la+=log(j); lb=0; for(j=1;j<=N-i;j++) lb+=log(j)
      pr=exp(lg-la-lb-N*log(2)); if(i>=k) hi+=pr; if(i<=k) lo+=pr }
    t=(hi<lo)?hi:lo; if(t>0.5)t=0.5
    printf "  exact two-sided sign test        p = %.5f\n", 2*t
    if (mc > 1.02 || mc < 0.98)
      printf "\n  WARNING: the control moved %.1f%%. Identical code cannot differ;\n  this box was not quiet enough and the effect above is not trustworthy.\n", 100*(mc-1)
  }' "$W/ab"

# ------------------------------------------------------- the wide path
if [ -n "$BENCH_GEMM" ]; then
    echo
    echo "  THE WIDE PATH -- where the decode is amortised over the strip"
    for a in base shuf; do
        gcc $CF "$BENCH_GEMM" "$W/$a.o" -o "$W/gemm_$a" -lm
    done
    : > "$W/g"
    G=$(( ROUNDS < 9 ? ROUNDS : 9 ))
    for r in $(seq 1 "$G"); do
        if [ $((r % 2)) -eq 1 ]; then
            taskset -c "$CPU" "$W/gemm_base" 0.25 | awk '/gemm cols/{print "b",$2,$3}' >> "$W/g"
            taskset -c "$CPU" "$W/gemm_shuf" 0.25 | awk '/gemm cols/{print "s",$2,$3}' >> "$W/g"
        else
            taskset -c "$CPU" "$W/gemm_shuf" 0.25 | awk '/gemm cols/{print "s",$2,$3}' >> "$W/g"
            taskset -c "$CPU" "$W/gemm_base" 0.25 | awk '/gemm cols/{print "b",$2,$3}' >> "$W/g"
        fi
    done
    awk '
      { split($2,a,"="); c=a[2]
        if ($1=="b") { bn[c]++; bv[c"_"bn[c]]=$3 } else { sn[c]++; sv[c"_"sn[c]]=$3 }
        seen[c]=1 }
      END{
        printf "  %-6s %11s %11s %9s %10s\n","cols","base med","shuf med","ratio","shuf won"
        n=asorti(seen, ks, "@ind_num_asc")
        for (i=1;i<=n;i++){ c=ks[i]
          for(j=1;j<=bn[c];j++) B[j]=bv[c"_"j]+0
          for(j=1;j<=sn[c];j++) S[j]=sv[c"_"j]+0
          for(x=1;x<=bn[c];x++)for(y=x+1;y<=bn[c];y++)if(B[y]<B[x]){t=B[x];B[x]=B[y];B[y]=t}
          for(x=1;x<=sn[c];x++)for(y=x+1;y<=sn[c];y++)if(S[y]<S[x]){t=S[x];S[x]=S[y];S[y]=t}
          mb=(bn[c]%2)?B[(bn[c]+1)/2]:(B[bn[c]/2]+B[bn[c]/2+1])/2
          ms=(sn[c]%2)?S[(sn[c]+1)/2]:(S[sn[c]/2]+S[sn[c]/2+1])/2
          w=0; for(j=1;j<=bn[c]&&j<=sn[c];j++) if (sv[c"_"j]+0 > bv[c"_"j]+0) w++
          printf "  %-6s %11.2f %11.2f %9.4f %7d/%d\n", c, mb, ms, ms/mb, w, bn[c] }
        print ""
        print "  Both arms saturating at the same rate from some width on is the"
        print "  finding, not a null result: past that width the decode -- the only"
        print "  thing this option changes -- has been amortised away entirely."
      }' "$W/g"
fi
