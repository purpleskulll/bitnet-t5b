/*
 * bench_alu.c -- each kernel's arithmetic rate with memory taken out of it.
 *
 * WHY THIS EXISTS
 * ---------------
 * bench_token.c measures what a token costs, which is what matters, and which
 * is exactly why it cannot say WHY one format beats another: at the model's
 * footprint the answer mixes bytes moved with arithmetic done. This runs the
 * same kernels over 100 rows -- 64 KiB of weights, inside the 512 KiB L2 slice
 * of one core -- so memory cannot bind and what is left is the kernel's own
 * issue rate. The ratio it prints is the S in "a format that reads 1/F of the
 * bytes and costs S times the arithmetic wins when S < surplus"; the surplus is
 * in results/thread_headroom.txt.
 *
 * It is also how the register spills in ternary_t5b were caught. That kernel
 * has FEWER multiply-port instructions per weight than ternary_t10 and was
 * slower anyway -- 24.36 against 29.67 GMAC/s -- which is not something the
 * token benchmark could have shown, and which pointed straight at the front end
 * rather than the algorithm. After the spills came out it reads 33.74.
 *
 * Single-threaded and pinned on purpose: taskset -c 5 keeps it off cpu0, and
 * one thread means no bandwidth sharing and no all-core clock penalty.
 *
 * Build:  gcc -O3 -mavx2 -mfma -march=native -std=c11 bench/bench_alu.c \
 *             build/obj/ggml_i2s_ternary.o build/obj/ternary_t10.o \
 *             build/obj/ternary_t5b.o -o build/bench_alu
 * Usage:  taskset -c 5 ./build/bench_alu
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "../src/ggml_i2s_ternary.h"
#include "../src/ternary_t10.h"
#include "../src/ternary_t5b.h"
#include "i2s_tiled.h"
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static uint32_t rs=1234567; static uint32_t rn(void){rs^=rs<<13;rs^=rs>>17;rs^=rs<<5;return rs;}
static void pack_i2s(const int8_t*w,uint8_t*d,size_t n){size_t rb=n/4;memset(d,0,rb);
  for(size_t b=0;b*128<n;++b)for(size_t j=0;j<32;++j){uint8_t by=0;
    for(int q=0;q<4;++q){size_t i=b*128+j+(size_t)q*32;uint8_t c=(i<n)?(uint8_t)(w[i]+1):1u;by|=(uint8_t)(c<<(6-2*q));}
    d[b*32+j]=by;}}
int main(void){
  const size_t n=2560, rows=100;           /* 100 rows x 640 B = 64 KiB, L2 */
  int8_t *w=malloc(rows*n); int8_t *a=aligned_alloc(64,n+64);
  uint8_t *Wi=aligned_alloc(64,rows*(n/4)), *Wt=aligned_alloc(64,rows*ternary_t10_size(n)),
          *Wb=aligned_alloc(64,rows*ternary_t5b_size(n));
  int16_t *b=aligned_alloc(64,ternary_t10_slots(n)*2);
  float *yf=aligned_alloc(64,rows*4); int32_t *yi=aligned_alloc(64,rows*4);
  for(size_t i=0;i<rows*n;++i) w[i]=(int8_t)((int)(rn()%3)-1);
  for(size_t i=0;i<n;++i) a[i]=(int8_t)((int)(rn()%255)-127);
  int32_t sa=0; for(size_t i=0;i<n;++i) sa+=a[i];
  ternary_t10_prep_b(a,b,n);
  for(size_t r=0;r<rows;++r){pack_i2s(w+r*n,Wi+r*(n/4),n);
    ternary_t10_pack(w+r*n,Wt+r*ternary_t10_size(n),n);
    ternary_t5b_pack(w+r*n,Wb+r*ternary_t5b_size(n),n);}
  /* The tiled kernel must agree with the reference it is standing in for,
   * or the row it prints is a rate for the wrong answer. Both return
   * sum(code*a), so they are directly comparable. */
  { int32_t chk[8]; i2s_tiled_gemv(Wi,a,chk,8,n);
    bitnet_vec_dot_i2_i8_s_reference((int)n,yf,1,Wi,n,a,0,8);
    for(int r=0;r<8;++r) if((int32_t)yf[r]!=chk[r]){
      fprintf(stderr,"FATAL: tiled disagrees with vec_dot at row %d: %d vs %d\n",
              r,(int32_t)yf[r],chk[r]); return 3; } }
  const double macs=(double)rows*n; long it; double t0,el; volatile int64_t sink=0;
  printf("  kernel      GMAC/s/core   relative to i2_s\n");
  double base=0;
  for(int k=0;k<4;++k){
    double best=0;
    for(int rep=0;rep<5;++rep){
      it=0;t0=now();
      do{ if(k==0) bitnet_vec_dot_i2_i8_s_reference((int)n,yf,1,Wi,n,a,0,(int)rows);
          else if(k==1) i2s_tiled_gemv(Wi,a,yi,rows,n);
          else if(k==2){for(size_t r=0;r<rows;++r) yi[r]=ternary_t10_dot_avx2(Wt+r*ternary_t10_size(n),b,n,sa);}
          else {for(size_t r=0;r<rows;++r) yi[r]=ternary_t5b_dot_avx2(Wb+r*ternary_t5b_size(n),a,n,sa);}
          ++it; el=now()-t0; }while(el<0.30);
      double rate=macs*it/el/1e9; if(rate>best)best=rate;
    }
    sink+=yi[0]+(int64_t)yf[0]; if(k==0)base=best;
    printf("  %-10s %9.2f     %6.3fx\n",
           k==0?"i2_s vecdot":k==1?"i2_s tiled4":k==2?"t10":"t5b", best, best/base);
  }
  fprintf(stderr,"sink=%lld\n",(long long)sink); return 0;}
