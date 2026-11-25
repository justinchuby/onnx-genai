
/tmp/nxrt-block-context-study/candidate.binary:	file format mach-o arm64

Disassembly of section __TEXT,__text:

000000010040d9c0 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block>:
10040d9c0:     	stp	x29, x30, [sp, #-0x10]!
10040d9c4:     	mov	x29, sp
10040d9c8:     	cbz	x1, 0x10040daf4 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x134>
10040d9cc:     	mov	x8, x0
10040d9d0:     	ldrb	w9, [x0]
10040d9d4:     	mov	w10, #0x7fc00000        ; =2143289344
10040d9d8:     	fmov	s0, w10
10040d9dc:     	mov	w10, #0x400000          ; =4194304
10040d9e0:     	fmov	s1, w10
10040d9e4:     	cmp	w9, #0xff
10040d9e8:     	lsl	w10, w9, #23
10040d9ec:     	fmov	s2, w10
10040d9f0:     	fcsel	s0, s0, s2, eq
10040d9f4:     	cmp	w9, #0x0
10040d9f8:     	fcsel	s0, s1, s0, eq
10040d9fc:     	add	x9, x2, #0x40
10040da00:     	mov	w0, #0x10               ; =16
10040da04:     	cmp	x3, #0x10
10040da08:     	csel	x10, x3, x0, lo
10040da0c:     	sub	x10, x10, x3
10040da10:     	neg	x11, x3
10040da14:     	mov	w12, #0x1               ; =1
10040da18:     	sub	x12, x12, x1
10040da1c:     	add	x8, x8, #0x1
10040da20:     	adrp	x13, 0x1008db000 <_anon.1dc4c1243e6292eeac6ae69d27196f4c.445+0x5e04>
10040da24:     	add	x13, x13, #0xd50
10040da28:     	add	x14, x12, x0
10040da2c:     	cmp	x14, #0x10
10040da30:     	b.eq	0x10040dad0 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x110>
10040da34:     	add	x14, x8, x0
10040da38:     	ldurb	w14, [x14, #-0x10]
10040da3c:     	tst	w14, #0x7
10040da40:     	b.eq	0x10040da60 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0xa0>
10040da44:     	and	x15, x14, #0xf
10040da48:     	ldr	s1, [x13, x15, lsl #2]
10040da4c:     	fmul	s1, s0, s1
10040da50:     	add	x15, x11, x0
10040da54:     	cmp	x15, #0x10
10040da58:     	b.ne	0x10040da70 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0xb0>
10040da5c:     	b	0x10040dae0 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x120>
10040da60:     	movi.2d	v1, #0000000000000000
10040da64:     	add	x15, x11, x0
10040da68:     	cmp	x15, #0x10
10040da6c:     	b.eq	0x10040dae0 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x120>
10040da70:     	stur	s1, [x9, #-0x40]
10040da74:     	tst	w14, #0x70
10040da78:     	b.eq	0x10040da98 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0xd8>
10040da7c:     	lsr	x14, x14, #4
10040da80:     	ldr	s1, [x13, x14, lsl #2]
10040da84:     	fmul	s1, s0, s1
10040da88:     	add	x14, x10, x0
10040da8c:     	cmp	x14, #0x10
10040da90:     	b.ne	0x10040daa8 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0xe8>
10040da94:     	b	0x10040dac0 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x100>
10040da98:     	movi.2d	v1, #0000000000000000
10040da9c:     	add	x14, x10, x0
10040daa0:     	cmp	x14, #0x10
10040daa4:     	b.eq	0x10040dac0 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x100>
10040daa8:     	str	s1, [x9], #0x4
10040daac:     	add	x0, x0, #0x1
10040dab0:     	cmp	x0, #0x20
10040dab4:     	b.ne	0x10040da28 <__RNvNtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels22block_quantized_matmul18decode_mxfp4_block+0x68>
10040dab8:     	ldp	x29, x30, [sp], #0x10
10040dabc:     	ret
10040dac0:     	adrp	x2, 0x100aa5000 <_anon.1dc4c1243e6292eeac6ae69d27196f4c.446+0x130>
10040dac4:     	add	x2, x2, #0x698
10040dac8:     	mov	x1, x3
10040dacc:     	bl	0x100853e3c <__RNvNtCslWxY2MhVcag_4core9panicking18panic_bounds_check>
10040dad0:     	sub	x0, x0, #0xf
10040dad4:     	adrp	x2, 0x100aa5000 <_anon.1dc4c1243e6292eeac6ae69d27196f4c.446+0x130>
10040dad8:     	add	x2, x2, #0x668
10040dadc:     	bl	0x100853e3c <__RNvNtCslWxY2MhVcag_4core9panicking18panic_bounds_check>
10040dae0:     	adrp	x2, 0x100aa5000 <_anon.1dc4c1243e6292eeac6ae69d27196f4c.446+0x130>
10040dae4:     	add	x2, x2, #0x680
10040dae8:     	mov	x0, x3
10040daec:     	mov	x1, x3
10040daf0:     	bl	0x100853e3c <__RNvNtCslWxY2MhVcag_4core9panicking18panic_bounds_check>
10040daf4:     	adrp	x2, 0x100aa5000 <_anon.1dc4c1243e6292eeac6ae69d27196f4c.446+0x130>
10040daf8:     	add	x2, x2, #0x650
10040dafc:     	mov	x0, #0x0                ; =0
10040db00:     	bl	0x100853e3c <__RNvNtCslWxY2MhVcag_4core9panicking18panic_bounds_check>
