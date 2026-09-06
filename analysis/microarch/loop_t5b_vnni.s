	vmovdqu	(%rcx), %ymm1
	addq	$32, %rcx
	vpsrlw	$8, %ymm1, %ymm2
	vpand	%ymm8, %ymm1, %ymm0
	vpmulhuw	%ymm7, %ymm2, %ymm12
	vpmulhuw	%ymm7, %ymm0, %ymm3
	vpmulhuw	%ymm6, %ymm2, %ymm13
	vpmulhuw	%ymm5, %ymm2, %ymm14
	vpmulhuw	%ymm4, %ymm2, %ymm2
	vpsllw	$8, %ymm12, %ymm12
	vpsllw	$8, %ymm13, %ymm13
	vpsllw	$8, %ymm14, %ymm14
	vpor	%ymm12, %ymm3, %ymm3
	vpmulhuw	%ymm6, %ymm0, %ymm12
	vpsllw	$8, %ymm2, %ymm2
	vpor	%ymm13, %ymm12, %ymm12
	vpmulhuw	%ymm5, %ymm0, %ymm13
	vpmulhuw	%ymm4, %ymm0, %ymm0
	vpor	%ymm14, %ymm13, %ymm13
	vpor	%ymm2, %ymm0, %ymm0
	vpdpbusd	128(%rax), %ymm0, %ymm16
	vpsllw	$1, %ymm0, %ymm14
	vpaddw	%ymm0, %ymm14, %ymm0
	vpsllw	$1, %ymm13, %ymm14
	vpsubb	%ymm0, %ymm13, %ymm0
	vpaddw	%ymm13, %ymm14, %ymm13
	vpsllw	$1, %ymm12, %ymm14
	vpsubb	%ymm13, %ymm12, %ymm13
	vpaddw	%ymm12, %ymm14, %ymm12
	vpsllw	$1, %ymm3, %ymm14
	vpdpbusd	96(%rax), %ymm0, %ymm17
	vpdpbusd	64(%rax), %ymm13, %ymm18
	vpsubb	%ymm12, %ymm3, %ymm12
	vpaddw	%ymm3, %ymm14, %ymm3
	vpdpbusd	32(%rax), %ymm12, %ymm19
	vpsubb	%ymm3, %ymm1, %ymm1
	vpdpbusd	(%rax), %ymm1, %ymm20
	addq	$160, %rax
	cmpq	%rax, %rsi
