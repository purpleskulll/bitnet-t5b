/*
 * bench_ports.c -- which execution port binds these kernels, measured directly.
 *
 * WHY THIS EXISTS
 * ---------------
 * results/weight_density.txt explains why a 1.6-bit weight format loses to
 * llama.cpp's 2-bit one by pointing at a single execution port: vpmaddubsw,
 * vpmaddwd, vpmulhuw and vpmullw are claimed to share it, so what decides a
 * kernel's rate is MACs per multiplier-port instruction rather than
 * instructions per weight. That explanation was first derived by dividing a
 * measured throughput by an assumed instruction count -- which is fitting, not
 * measuring, and would have "confirmed" any story. This measures the port.
 *
 * THE FIRST VERSION OF THIS FILE MEASURED NOTHING, LOUDLY
 * ------------------------------------------------------
 * Its eight accumulators started from the same value and followed the same
 * recurrence, so the compiler proved them equal and kept one. vpand was
 * eliminated outright (a & k with a == k is loop-invariant) and vpaddw was
 * strength-reduced into a multiply. They reported 2,982,616 and 9.15 ops per
 * cycle, against a hard ceiling of about 4. Numbers that impossible are the
 * only reason the bug was caught; had the collapse been partial the run would
 * have looked plausible and been wrong.
 *
 * So: distinct seeds per chain, and an empty asm barrier on every accumulator
 * every iteration, which makes each chain observably different and stops both
 * the collapse and the strength reduction. Eight independent chains so that
 * instruction LATENCY cannot bind and only issue rate can. The clock is
 * measured rather than assumed, from a dependent add chain that retires one
 * per cycle by definition.
 *
 * Build:  gcc -O3 -mavx2 -march=native -std=c11 bench/bench_ports.c -o build/bench_ports
 * Usage:  taskset -c 5 ./build/bench_ports
 */
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static int64_t sink=0;
static double ghz=0.0;

#define KEEP(v) __asm__ __volatile__("" : "+x"(v))
#define RUN(NAME,OP) do{                                                      \
    __m256i a0=_mm256_set1_epi16(0x0103),a1=_mm256_set1_epi16(0x0205),        \
            a2=_mm256_set1_epi16(0x0307),a3=_mm256_set1_epi16(0x0409),        \
            a4=_mm256_set1_epi16(0x050B),a5=_mm256_set1_epi16(0x060D),        \
            a6=_mm256_set1_epi16(0x070F),a7=_mm256_set1_epi16(0x0811);        \
    const long N=20000000L; double t0=now();                                  \
    for(long i=0;i<N;++i){                                                    \
      a0=OP(a0,k);a1=OP(a1,k);a2=OP(a2,k);a3=OP(a3,k);                        \
      a4=OP(a4,k);a5=OP(a5,k);a6=OP(a6,k);a7=OP(a7,k);                        \
      KEEP(a0);KEEP(a1);KEEP(a2);KEEP(a3);KEEP(a4);KEEP(a5);KEEP(a6);KEEP(a7);\
    }                                                                         \
    double el=now()-t0;                                                       \
    __m256i s=_mm256_add_epi32(_mm256_add_epi32(a0,a1),_mm256_add_epi32(a2,a3)); \
    s=_mm256_add_epi32(s,_mm256_add_epi32(_mm256_add_epi32(a4,a5),            \
                                          _mm256_add_epi32(a6,a7)));          \
    sink+=_mm256_extract_epi32(s,0);                                          \
    printf("  %-14s %8.2f Gop/s   %5.2f ops/cycle\n",                         \
           NAME, 8.0*N/el/1e9, 8.0*N/el/(ghz*1e9));                           \
  }while(0)

int main(void){
    const __m256i k=_mm256_set1_epi16(0x0101);
    /* Measure the clock this loop actually runs at: a dependent add chain
     * retires one per cycle, so its rate IS the frequency. */
    { __m256i a=_mm256_set1_epi16(1); const long N=400000000L;
      double t0=now(); for(long i=0;i<N;++i){a=_mm256_add_epi32(a,k);KEEP(a);}
      ghz=N/(now()-t0)/1e9; sink+=_mm256_extract_epi32(a,0);
      printf("  measured clock  %.2f GHz  (dependent vpaddd chain, 1/cycle)\n\n",ghz); }
    RUN("vpmaddubsw", _mm256_maddubs_epi16);
    RUN("vpmaddwd",   _mm256_madd_epi16);
    RUN("vpmulhuw",   _mm256_mulhi_epu16);
    RUN("vpmullw",    _mm256_mullo_epi16);
    RUN("vpand",      _mm256_and_si256);
    RUN("vpaddw",     _mm256_add_epi16);
    fprintf(stderr,"sink=%lld\n",(long long)sink);
    return 0;
}
