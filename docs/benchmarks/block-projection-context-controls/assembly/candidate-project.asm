
/tmp/nxrt-block-context-study/candidate.binary:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001001ed3f4 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_>:
1001ed3f4:     	sub	sp, sp, #0x190
1001ed3f8:     	stp	x28, x27, [sp, #0x130]
1001ed3fc:     	stp	x26, x25, [sp, #0x140]
1001ed400:     	stp	x24, x23, [sp, #0x150]
1001ed404:     	stp	x22, x21, [sp, #0x160]
1001ed408:     	stp	x20, x19, [sp, #0x170]
1001ed40c:     	stp	x29, x30, [sp, #0x180]
1001ed410:     	add	x29, sp, #0x180
1001ed414:     	mov	x21, x7
1001ed418:     	mov	x20, x6
1001ed41c:     	str	x3, [sp, #0x8]
1001ed420:     	mov	x24, x1
1001ed424:     	mov	x19, x8
1001ed428:     	mov	x27, #0x0               ; =0
1001ed42c:     	ldr	x25, [x29, #0x10]
1001ed430:     	stp	x4, x5, [sp, #0x10]
1001ed434:     	stp	x6, x7, [sp, #0x20]
1001ed438:     	add	x26, sp, #0x88
1001ed43c:     	str	x25, [sp, #0x30]
1001ed440:     	stur	x5, [x29, #-0x88]
1001ed444:     	stur	x6, [x29, #-0x78]
1001ed448:     	mov	x23, #0x0               ; =0
1001ed44c:     	mov	x28, #0x0               ; =0
1001ed450:     	cbz	x4, 0x1001ed498 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0xa4>
1001ed454:     	cbz	x20, 0x1001ed498 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0xa4>
1001ed458:     	udiv	x8, x20, x0
1001ed45c:     	msub	x9, x8, x0, x20
1001ed460:     	cmp	x9, #0x0
1001ed464:     	cinc	x27, x8, ne
1001ed468:     	cbz	x27, 0x1001ed82c <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x438>
1001ed46c:     	cmp	x27, x24
1001ed470:     	csel	x8, x27, x24, lo
1001ed474:     	umulh	x9, x8, x5
1001ed478:     	cmp	xzr, x9
1001ed47c:     	b.ne	0x1001ed774 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x380>
1001ed480:     	mul	x28, x8, x5
1001ed484:     	udiv	x8, x20, x27
1001ed488:     	msub	x9, x8, x27, x20
1001ed48c:     	cmp	x9, #0x0
1001ed490:     	cinc	x8, x8, ne
1001ed494:     	mul	x23, x8, x28
1001ed498:     	mov	x22, x2
1001ed49c:     	stur	x27, [x29, #-0x88]
1001ed4a0:     	add	x8, sp, #0x88
1001ed4a4:     	mov	x1, x24
1001ed4a8:     	mov	x2, x4
1001ed4ac:     	mov	x3, x5
1001ed4b0:     	mov	x4, x20
1001ed4b4:     	bl	0x100207394 <__RNvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB4_21DecodedProjectionPlan6layout>
1001ed4b8:     	ldr	x8, [sp, #0x88]
1001ed4bc:     	cmn	x8, #0x1
1001ed4c0:     	b.eq	0x1001ed4f4 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x100>
1001ed4c4:     	ldur	q0, [x26, #0x8]
1001ed4c8:     	stur	q0, [x19, #0x8]
1001ed4cc:     	ldur	q0, [x26, #0x18]
1001ed4d0:     	stur	q0, [x19, #0x18]
1001ed4d4:     	ldur	q0, [x26, #0x38]
1001ed4d8:     	stur	q0, [x19, #0x38]
1001ed4dc:     	ldr	x9, [sp, #0xd0]
1001ed4e0:     	ldur	q0, [x26, #0x28]
1001ed4e4:     	str	x8, [x19]
1001ed4e8:     	stur	q0, [x19, #0x28]
1001ed4ec:     	str	x9, [x19, #0x48]
1001ed4f0:     	b	0x1001ed594 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x1a0>
1001ed4f4:     	ldr	x10, [sp, #0xb8]
1001ed4f8:     	cmp	x24, #0x4
1001ed4fc:     	b.ne	0x1001ed548 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x154>
1001ed500:     	ldp	x8, x9, [sp, #0x10]
1001ed504:     	umulh	x11, x8, x9
1001ed508:     	cmp	xzr, x11
1001ed50c:     	b.ne	0x1001ed548 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x154>
1001ed510:     	mul	x9, x8, x9
1001ed514:     	ldr	x11, [sp, #0x8]
1001ed518:     	cmp	x9, x11
1001ed51c:     	b.ne	0x1001ed548 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x154>
1001ed520:     	umulh	x9, x8, x20
1001ed524:     	cmp	xzr, x9
1001ed528:     	b.ne	0x1001ed548 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x154>
1001ed52c:     	ldr	x26, [x29, #0x20]
1001ed530:     	mul	x9, x8, x20
1001ed534:     	cmp	x9, x26
1001ed538:     	b.ne	0x1001ed548 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x154>
1001ed53c:     	cbz	x21, 0x1001ed5b4 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x1c0>
1001ed540:     	cmp	x25, x20
1001ed544:     	b.eq	0x1001ed5b4 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x1c0>
1001ed548:     	str	x10, [sp, #0x38]
1001ed54c:     	adrp	x8, 0x10080b000 <__RNvXs_NtNtCslWxY2MhVcag_4core3str5errorNtB4_9Utf8ErrorNtNtB8_3fmt7Display3fmt+0x38>
1001ed550:     	add	x8, x8, #0x1b4
1001ed554:     	add	x9, sp, #0x10
1001ed558:     	stp	x9, x8, [sp, #0x88]
1001ed55c:     	add	x9, sp, #0x18
1001ed560:     	stp	x9, x8, [sp, #0x98]
1001ed564:     	add	x9, sp, #0x20
1001ed568:     	stp	x9, x8, [sp, #0xa8]
1001ed56c:     	add	x9, sp, #0x38
1001ed570:     	stp	x9, x8, [sp, #0xb8]
1001ed574:     	adrp	x0, 0x100925000 <_anon.06fd55396524d10657ac4bc31c527ccd.0+0x7fe>
1001ed578:     	add	x0, x0, #0xae1
1001ed57c:     	add	x8, x19, #0x8
1001ed580:     	add	x1, sp, #0x88
1001ed584:     	bl	0x1007fd874 <__RNvNvNtCs6KVRSXc8uZF_5alloc3fmt6format12format_inner>
1001ed588:     	mov	x8, #0x5                ; =5
1001ed58c:     	movk	x8, #0x8000, lsl #48
1001ed590:     	str	x8, [x19]
1001ed594:     	ldp	x29, x30, [sp, #0x180]
1001ed598:     	ldp	x20, x19, [sp, #0x170]
1001ed59c:     	ldp	x22, x21, [sp, #0x160]
1001ed5a0:     	ldp	x24, x23, [sp, #0x150]
1001ed5a4:     	ldp	x26, x25, [sp, #0x140]
1001ed5a8:     	ldp	x28, x27, [sp, #0x130]
1001ed5ac:     	add	sp, sp, #0x190
1001ed5b0:     	ret
1001ed5b4:     	ldr	x9, [x29, #0x40]
1001ed5b8:     	ldp	x24, x21, [x29, #0x28]
1001ed5bc:     	ldr	x25, [x29, #0x18]
1001ed5c0:     	cmp	x8, #0x1
1001ed5c4:     	b.ls	0x1001ed5fc <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x208>
1001ed5c8:     	cmp	x20, #0x1
1001ed5cc:     	csel	x11, x26, xzr, hi
1001ed5d0:     	cmp	x21, x11
1001ed5d4:     	b.ne	0x1001ed548 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x154>
1001ed5d8:     	cmp	x9, x23
1001ed5dc:     	b.ne	0x1001ed548 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x154>
1001ed5e0:     	cbz	x26, 0x1001ed76c <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x378>
1001ed5e4:     	cmp	x20, #0x1
1001ed5e8:     	b.ls	0x1001ed60c <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x218>
1001ed5ec:     	mov	w23, #0x1               ; =1
1001ed5f0:     	mov	x10, x24
1001ed5f4:     	mov	x11, x21
1001ed5f8:     	b	0x1001ed618 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x224>
1001ed5fc:     	cbnz	x21, 0x1001ed548 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x154>
1001ed600:     	cmp	x9, x23
1001ed604:     	b.ne	0x1001ed548 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x154>
1001ed608:     	cbz	x26, 0x1001ed76c <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x378>
1001ed60c:     	mov	w23, #0x0               ; =0
1001ed610:     	mov	x10, x25
1001ed614:     	mov	x11, x26
1001ed618:     	mul	x8, x8, x27
1001ed61c:     	cbz	x28, 0x1001ed684 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x290>
1001ed620:     	cbz	x8, 0x1001ed7d4 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x3e0>
1001ed624:     	ldr	x12, [x29, #0x38]
1001ed628:     	stp	x10, x11, [sp, #0x88]
1001ed62c:     	stp	x8, x12, [sp, #0x98]
1001ed630:     	stp	x9, x28, [sp, #0xa8]
1001ed634:     	ldr	x8, [sp, #0x8]
1001ed638:     	stp	x22, x8, [sp, #0xb8]
1001ed63c:     	add	x8, sp, #0x10
1001ed640:     	add	x9, sp, #0x18
1001ed644:     	stp	x8, x9, [sp, #0xc8]
1001ed648:     	add	x8, sp, #0x28
1001ed64c:     	sub	x9, x29, #0x88
1001ed650:     	stp	x8, x9, [sp, #0xd8]
1001ed654:     	add	x8, x29, #0x48
1001ed658:     	sub	x9, x29, #0x59
1001ed65c:     	stp	x8, x9, [sp, #0xe8]
1001ed660:     	sturb	wzr, [x29, #-0x79]
1001ed664:     	stp	x9, x9, [x29, #-0x78]
1001ed668:     	sub	x8, x29, #0x79
1001ed66c:     	stur	x8, [x29, #-0x68]
1001ed670:     	add	x8, sp, #0x38
1001ed674:     	add	x0, sp, #0x88
1001ed678:     	sub	x1, x29, #0x78
1001ed67c:     	bl	0x1003ab604 <__RINvXs0_NtNtCs3xVcNbiH0lw_5rayon4iter3mapINtB6_3MapINtNtB8_9enumerate9EnumerateINtNtB8_3zip3ZipINtNtNtBa_5slice6chunks9ChunksMutfEB1v_EENCINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB2i_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB2k_22block_quantized_matmulNtB4a_20PackedProjectionRows7decoder0NCINvB4a_28linear_packed_prepared_tiledKB3Z_E0INvB4a_19project_packed_tileKB3Z_EEs0_0ENtB8_16ParallelIterator15drive_unindexedINtNtB8_10try_reduce17TryReduceConsumerINvNvB6D_12try_for_each2okINtNtCslWxY2MhVcag_4core6result6ResultuNtNtCscgqFgCZgHVw_19onnx_runtime_ep_api5error7EpErrorEENvYuNtNtB8p_7default7Default7defaultEEB2m_>
1001ed680:     	b	0x1001ed6dc <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x2e8>
1001ed684:     	cbz	x8, 0x1001ed7ec <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x3f8>
1001ed688:     	stp	x10, x11, [sp, #0x88]
1001ed68c:     	stp	x8, x22, [sp, #0x98]
1001ed690:     	add	x8, sp, #0x10
1001ed694:     	ldr	x9, [sp, #0x8]
1001ed698:     	stp	x9, x8, [sp, #0xa8]
1001ed69c:     	add	x8, sp, #0x18
1001ed6a0:     	add	x9, sp, #0x28
1001ed6a4:     	stp	x8, x9, [sp, #0xb8]
1001ed6a8:     	sub	x8, x29, #0x88
1001ed6ac:     	add	x9, x29, #0x48
1001ed6b0:     	stp	x8, x9, [sp, #0xc8]
1001ed6b4:     	sub	x8, x29, #0x59
1001ed6b8:     	str	x8, [sp, #0xd8]
1001ed6bc:     	sturb	wzr, [x29, #-0x79]
1001ed6c0:     	stp	x8, x8, [x29, #-0x78]
1001ed6c4:     	sub	x8, x29, #0x79
1001ed6c8:     	stur	x8, [x29, #-0x68]
1001ed6cc:     	add	x8, sp, #0x38
1001ed6d0:     	add	x0, sp, #0x88
1001ed6d4:     	sub	x1, x29, #0x78
1001ed6d8:     	bl	0x1003ab740 <__RINvXs0_NtNtCs3xVcNbiH0lw_5rayon4iter3mapINtB6_3MapINtNtB8_9enumerate9EnumerateINtNtNtBa_5slice6chunks9ChunksMutfEENCINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB1X_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB1Z_22block_quantized_matmulNtB3P_20PackedProjectionRows7decoder0NCINvB3P_28linear_packed_prepared_tiledKB3E_E0INvB3P_19project_packed_tileKB3E_EEs_0ENtB8_16ParallelIterator15drive_unindexedINtNtB8_10try_reduce17TryReduceConsumerINvNvB6h_12try_for_each2okINtNtCslWxY2MhVcag_4core6result6ResultuNtNtCscgqFgCZgHVw_19onnx_runtime_ep_api5error7EpErrorEENvYuNtNtB83_7default7Default7defaultEEB21_>
1001ed6dc:     	ldr	x8, [sp, #0x38]
1001ed6e0:     	cmn	x8, #0x1
1001ed6e4:     	b.eq	0x1001ed70c <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x318>
1001ed6e8:     	ldur	q0, [sp, #0x58]
1001ed6ec:     	ldur	q1, [sp, #0x68]
1001ed6f0:     	stp	q0, q1, [x19, #0x20]
1001ed6f4:     	ldur	q0, [sp, #0x78]
1001ed6f8:     	str	q0, [x19, #0x40]
1001ed6fc:     	ldur	q0, [sp, #0x48]
1001ed700:     	ldur	q1, [sp, #0x38]
1001ed704:     	stp	q1, q0, [x19]
1001ed708:     	b	0x1001ed594 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x1a0>
1001ed70c:     	tbz	w23, #0x0, 0x1001ed76c <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x378>
1001ed710:     	cbz	x20, 0x1001ed804 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x410>
1001ed714:     	udiv	x8, x26, x20
1001ed718:     	mul	x8, x8, x20
1001ed71c:     	cmp	x20, x8
1001ed720:     	b.hi	0x1001ed76c <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x378>
1001ed724:     	mov	x11, #0x0               ; =0
1001ed728:     	ldr	x9, [sp, #0x10]
1001ed72c:     	lsl	x10, x20, #2
1001ed730:     	mov	x12, #0x0               ; =0
1001ed734:     	mov	x0, x11
1001ed738:     	sub	x8, x8, x20
1001ed73c:     	add	x11, x11, #0x1
1001ed740:     	cmp	x0, x21
1001ed744:     	b.hs	0x1001ed81c <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x428>
1001ed748:     	ldr	s0, [x24, x0, lsl #2]
1001ed74c:     	str	s0, [x25, x12]
1001ed750:     	add	x0, x0, x9
1001ed754:     	add	x12, x12, #0x4
1001ed758:     	cmp	x10, x12
1001ed75c:     	b.ne	0x1001ed740 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x34c>
1001ed760:     	add	x25, x25, x10
1001ed764:     	cmp	x20, x8
1001ed768:     	b.ls	0x1001ed730 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x33c>
1001ed76c:     	mov	x8, #-0x1               ; =-1
1001ed770:     	b	0x1001ed590 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x19c>
1001ed774:     	adrp	x8, 0x10080b000 <__RNvXs_NtNtCslWxY2MhVcag_4core3str5errorNtB4_9Utf8ErrorNtNtB8_3fmt7Display3fmt+0x38>
1001ed778:     	add	x8, x8, #0x1b4
1001ed77c:     	sub	x9, x29, #0x88
1001ed780:     	stp	x9, x8, [sp, #0x38]
1001ed784:     	sub	x9, x29, #0x78
1001ed788:     	stp	x9, x8, [sp, #0x48]
1001ed78c:     	add	x8, sp, #0x88
1001ed790:     	adrp	x0, 0x100926000 <_anon.06b6bc08fc3403f3a3355b160600f94f.83+0x24a>
1001ed794:     	add	x0, x0, #0x3e9
1001ed798:     	add	x8, x8, #0x8
1001ed79c:     	add	x1, sp, #0x38
1001ed7a0:     	bl	0x1007fd874 <__RNvNvNtCs6KVRSXc8uZF_5alloc3fmt6format12format_inner>
1001ed7a4:     	ldr	x8, [sp, #0xa0]
1001ed7a8:     	ldp	q0, q1, [x26, #0x20]
1001ed7ac:     	stp	q0, q1, [x19, #0x20]
1001ed7b0:     	ldr	q0, [x26, #0x40]
1001ed7b4:     	str	q0, [x19, #0x40]
1001ed7b8:     	ldur	q0, [x26, #0x8]
1001ed7bc:     	mov	x9, #0x5                ; =5
1001ed7c0:     	movk	x9, #0x8000, lsl #48
1001ed7c4:     	str	x9, [x19]
1001ed7c8:     	stur	q0, [x19, #0x8]
1001ed7cc:     	str	x8, [x19, #0x18]
1001ed7d0:     	b	0x1001ed594 <__RINvMs_NtNtCs42teQl6N5qM_19onnx_runtime_ep_cpu7kernels18decoded_projectionNtB5_21DecodedProjectionPlan7projectKj4_NCNvMs7_NtB7_22block_quantized_matmulNtB1W_20PackedProjectionRows7decoder0NCINvB1W_28linear_packed_prepared_tiledKB1L_E0INvB1W_19project_packed_tileKB1L_EEB9_+0x1a0>
1001ed7d4:     	adrp	x0, 0x1008de000 <_anon.1dc4c1243e6292eeac6ae69d27196f4c.803+0x22a>
1001ed7d8:     	add	x0, x0, #0x9f
1001ed7dc:     	adrp	x2, 0x100a8e000 <_anon.a08b98e357d9c30c0ee8e1a8378508b3.47+0x258>
1001ed7e0:     	add	x2, x2, #0x0
1001ed7e4:     	mov	w1, #0x37               ; =55
1001ed7e8:     	bl	0x100853fac <__RNvNtCslWxY2MhVcag_4core9panicking9panic_fmt>
1001ed7ec:     	adrp	x0, 0x1008de000 <_anon.1dc4c1243e6292eeac6ae69d27196f4c.803+0x22a>
1001ed7f0:     	add	x0, x0, #0x9f
1001ed7f4:     	adrp	x2, 0x100a8d000 <_anon.d2af6fe06533fb757835afa17425b9a1.1+0x110>
1001ed7f8:     	add	x2, x2, #0xfe8
1001ed7fc:     	mov	w1, #0x37               ; =55
1001ed800:     	bl	0x100853fac <__RNvNtCslWxY2MhVcag_4core9panicking9panic_fmt>
1001ed804:     	adrp	x0, 0x1008c6000 <_anon.a08b98e357d9c30c0ee8e1a8378508b3.46+0x25b>
1001ed808:     	add	x0, x0, #0x12e
1001ed80c:     	adrp	x2, 0x100a8e000 <_anon.a08b98e357d9c30c0ee8e1a8378508b3.47+0x258>
1001ed810:     	add	x2, x2, #0x18
1001ed814:     	mov	w1, #0x37               ; =55
1001ed818:     	bl	0x100853fac <__RNvNtCslWxY2MhVcag_4core9panicking9panic_fmt>
1001ed81c:     	adrp	x2, 0x100a8e000 <_anon.a08b98e357d9c30c0ee8e1a8378508b3.47+0x258>
1001ed820:     	add	x2, x2, #0x30
1001ed824:     	mov	x1, x21
1001ed828:     	bl	0x100853e3c <__RNvNtCslWxY2MhVcag_4core9panicking18panic_bounds_check>
1001ed82c:     	adrp	x0, 0x100a8e000 <_anon.a08b98e357d9c30c0ee8e1a8378508b3.47+0x258>
1001ed830:     	add	x0, x0, #0xac0
1001ed834:     	bl	0x100854088 <__RNvNtNtCslWxY2MhVcag_4core9panicking11panic_const23panic_const_div_by_zero>
