	vmovdqu	(%r15), %ymm9
	addq	$32, %r15
	vpsrlw	$2, %ymm9, %ymm5
	vpsrlw	$4, %ymm9, %ymm0
	vpand	%ymm2, %ymm9, %ymm10
	vpsrlw	$6, %ymm9, %ymm7
	vpand	%ymm2, %ymm0, %ymm0
	vpand	%ymm2, %ymm7, %ymm7
	vpand	%ymm2, %ymm5, %ymm5
	vpdpbusd	96(%rax), %ymm10, %ymm16
	vpdpbusd	32(%rax), %ymm0, %ymm17
	vpdpbusd	(%rax), %ymm7, %ymm18
	vpdpbusd	64(%rax), %ymm5, %ymm19
	subq	$-128, %rax
	cmpq	%rcx, %r15
