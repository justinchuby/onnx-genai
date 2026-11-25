
/tmp/nxrt-block-context-study/baseline.binary:	file format mach-o arm64

Disassembly of section __TEXT,__text:

000000010033fe70 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block>:
10033fe70:     	stp	x29, x30, [sp, #-0x10]!
10033fe74:     	mov	x29, sp
10033fe78:     	cbz	x1, 0x10033ffa4 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x134>
10033fe7c:     	mov	x8, x0
10033fe80:     	ldrb	w9, [x0]
10033fe84:     	mov	w10, #0x7fc00000        ; =2143289344
10033fe88:     	fmov	s0, w10
10033fe8c:     	mov	w10, #0x400000          ; =4194304
10033fe90:     	fmov	s1, w10
10033fe94:     	cmp	w9, #0xff
10033fe98:     	lsl	w10, w9, #23
10033fe9c:     	fmov	s2, w10
10033fea0:     	fcsel	s0, s0, s2, eq
10033fea4:     	cmp	w9, #0x0
10033fea8:     	fcsel	s0, s1, s0, eq
10033feac:     	add	x9, x2, #0x40
10033feb0:     	mov	w0, #0x10               ; =16
10033feb4:     	cmp	x3, #0x10
10033feb8:     	csel	x10, x3, x0, lo
10033febc:     	sub	x10, x10, x3
10033fec0:     	neg	x11, x3
10033fec4:     	mov	w12, #0x1               ; =1
10033fec8:     	sub	x12, x12, x1
10033fecc:     	add	x8, x8, #0x1
10033fed0:     	adrp	x13, 0x1008d3000 <_anon.3ea6753b9c9dbe2e759668c2822d19e2.431+0x62a2>
10033fed4:     	add	x13, x13, #0x8b0
10033fed8:     	add	x14, x12, x0
10033fedc:     	cmp	x14, #0x10
10033fee0:     	b.eq	0x10033ff80 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x110>
10033fee4:     	add	x14, x8, x0
10033fee8:     	ldurb	w14, [x14, #-0x10]
10033feec:     	tst	w14, #0x7
10033fef0:     	b.eq	0x10033ff10 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0xa0>
10033fef4:     	and	x15, x14, #0xf
10033fef8:     	ldr	s1, [x13, x15, lsl #2]
10033fefc:     	fmul	s1, s0, s1
10033ff00:     	add	x15, x11, x0
10033ff04:     	cmp	x15, #0x10
10033ff08:     	b.ne	0x10033ff20 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0xb0>
10033ff0c:     	b	0x10033ff90 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x120>
10033ff10:     	movi.2d	v1, #0000000000000000
10033ff14:     	add	x15, x11, x0
10033ff18:     	cmp	x15, #0x10
10033ff1c:     	b.eq	0x10033ff90 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x120>
10033ff20:     	stur	s1, [x9, #-0x40]
10033ff24:     	tst	w14, #0x70
10033ff28:     	b.eq	0x10033ff48 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0xd8>
10033ff2c:     	lsr	x14, x14, #4
10033ff30:     	ldr	s1, [x13, x14, lsl #2]
10033ff34:     	fmul	s1, s0, s1
10033ff38:     	add	x14, x10, x0
10033ff3c:     	cmp	x14, #0x10
10033ff40:     	b.ne	0x10033ff58 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0xe8>
10033ff44:     	b	0x10033ff70 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x100>
10033ff48:     	movi.2d	v1, #0000000000000000
10033ff4c:     	add	x14, x10, x0
10033ff50:     	cmp	x14, #0x10
10033ff54:     	b.eq	0x10033ff70 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x100>
10033ff58:     	str	s1, [x9], #0x4
10033ff5c:     	add	x0, x0, #0x1
10033ff60:     	cmp	x0, #0x20
10033ff64:     	b.ne	0x10033fed8 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x68>
10033ff68:     	ldp	x29, x30, [sp], #0x10
10033ff6c:     	ret
10033ff70:     	adrp	x2, 0x100a99000 <_anon.3ea6753b9c9dbe2e759668c2822d19e2.247+0xa0>
10033ff74:     	add	x2, x2, #0xeb8
10033ff78:     	mov	x1, x3
10033ff7c:     	bl	0x100854328 <__RNvNtCslWxY2MhVcag_4core9panicking18panic_bounds_check>
10033ff80:     	sub	x0, x0, #0xf
10033ff84:     	adrp	x2, 0x100a99000 <_anon.3ea6753b9c9dbe2e759668c2822d19e2.247+0xa0>
10033ff88:     	add	x2, x2, #0xe88
10033ff8c:     	bl	0x100854328 <__RNvNtCslWxY2MhVcag_4core9panicking18panic_bounds_check>
10033ff90:     	adrp	x2, 0x100a99000 <_anon.3ea6753b9c9dbe2e759668c2822d19e2.247+0xa0>
10033ff94:     	add	x2, x2, #0xea0
10033ff98:     	mov	x0, x3
10033ff9c:     	mov	x1, x3
10033ffa0:     	bl	0x100854328 <__RNvNtCslWxY2MhVcag_4core9panicking18panic_bounds_check>
10033ffa4:     	adrp	x2, 0x100a99000 <_anon.3ea6753b9c9dbe2e759668c2822d19e2.247+0xa0>
10033ffa8:     	add	x2, x2, #0xe70
10033ffac:     	mov	x0, #0x0                ; =0
10033ffb0:     	bl	0x100854328 <__RNvNtCslWxY2MhVcag_4core9panicking18panic_bounds_check>
