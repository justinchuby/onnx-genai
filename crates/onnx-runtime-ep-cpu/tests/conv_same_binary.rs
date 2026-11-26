//! Independent Conv factories in one linked-reference process. No ORT CPU EP.
#![cfg(feature = "ops-cnn")]

#[cfg(not(feature = "mlas"))]
#[test]
fn default_registry_keeps_its_native_conv() {
    use onnx_runtime_ir::{Node, NodeId};
    let registry = onnx_runtime_ep_cpu::kernels::build_cpu_registry();
    let kernel = registry
        .lookup("Conv", "", 18)
        .unwrap()
        .create(
            &Node::new(NodeId(0), "Conv", vec![], vec![]),
            &[vec![1, 2, 5, 5], vec![3, 2, 1, 1]],
        )
        .unwrap();
    assert!(!onnx_runtime_ep_cpu::backend_ab::mlas_available());
    assert!(
        kernel
            .as_ref()
            .as_any()
            .is::<onnx_runtime_ep_cpu::kernels::conv::ConvKernel>()
    );
}

#[cfg(feature = "mlas")]
mod linked {
    use onnx_runtime_ep_api::{
        DeviceId, DevicePtr, DevicePtrMut, KernelFactory, TensorMut, TensorView,
    };
    use onnx_runtime_ep_cpu::{backend_ab, dispatch_ledger};
    use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, compute_contiguous_strides};

    struct Case {
        x: [usize; 4],
        w: [usize; 4],
        group: usize,
        strides: [usize; 2],
        dilation: [usize; 2],
        pads: [usize; 4],
        rank3: bool,
    }
    impl Case {
        fn output(&self) -> [usize; 4] {
            let spatial = [0, 1].map(|i| {
                let padded = self.x[i + 2] + self.pads[i] + self.pads[i + 2];
                let extent = (self.w[i + 2] - 1) * self.dilation[i] + 1;
                padded
                    .checked_sub(extent)
                    .map_or(0, |size| size / self.strides[i] + 1)
            });
            [self.x[0], self.w[0], spatial[0], spatial[1]]
        }
        fn shape(&self, shape: [usize; 4]) -> Vec<usize> {
            if self.rank3 {
                assert_eq!(shape[2], 1);
                vec![shape[0], shape[1], shape[3]]
            } else {
                shape.to_vec()
            }
        }
        fn node(&self, relu: bool) -> Node {
            let mut node = Node::new(NodeId(0), "Conv", vec![], vec![]);
            node.attributes
                .insert("group".into(), Attribute::Int(self.group as i64));
            for (name, values) in [("strides", self.strides), ("dilations", self.dilation)] {
                node.attributes.insert(
                    name.into(),
                    Attribute::Ints(if self.rank3 {
                        vec![values[1] as i64]
                    } else {
                        values.map(|x| x as i64).to_vec()
                    }),
                );
            }
            node.attributes.insert(
                "pads".into(),
                Attribute::Ints(if self.rank3 {
                    vec![self.pads[1] as i64, self.pads[3] as i64]
                } else {
                    self.pads.map(|x| x as i64).to_vec()
                }),
            );
            if relu {
                node.attributes
                    .insert("activation".into(), Attribute::String(b"Relu".to_vec()));
            }
            node
        }
    }

    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 32) as u32 as f64 / u32::MAX as f64 - 0.5) as f32
            })
            .collect()
    }

    fn oracle(case: &Case, x: &[f32], w: &[f32], bias: Option<&[f32]>, relu: bool) -> Vec<f64> {
        let [batch, ic, ih, iw] = case.x;
        let [oc, per_group, kh, kw] = case.w;
        let [_, _, oh, ow] = case.output();
        let mut output = vec![0.; batch * oc * oh * ow];
        for b in 0..batch {
            for c in 0..oc {
                for y in 0..oh {
                    for z in 0..ow {
                        let group = c / (oc / case.group);
                        let mut sum = bias.map_or(0., |bias| bias[c] as f64);
                        for i in 0..per_group {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let yy = (y * case.strides[0] + ky * case.dilation[0])
                                        .checked_sub(case.pads[0]);
                                    let zz = (z * case.strides[1] + kx * case.dilation[1])
                                        .checked_sub(case.pads[1]);
                                    if let (Some(yy), Some(zz)) = (yy, zz)
                                        && yy < ih
                                        && zz < iw
                                    {
                                        let xx =
                                            ((b * ic + group * per_group + i) * ih + yy) * iw + zz;
                                        let ww = ((c * per_group + i) * kh + ky) * kw + kx;
                                        sum += x[xx] as f64 * w[ww] as f64;
                                    }
                                }
                            }
                        }
                        output[((b * oc + c) * oh + y) * ow + z] =
                            if relu { sum.max(0.) } else { sum };
                    }
                }
            }
        }
        output
    }

    fn compare(
        factory: &dyn KernelFactory,
        case: &Case,
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        relu: bool,
        native: bool,
    ) -> std::any::TypeId {
        let xs = case.shape(case.x);
        let ws = case.shape(case.w);
        let ys = case.shape(case.output());
        let mut kernel = factory
            .create(&case.node(relu), &[xs.clone(), ws.clone()])
            .unwrap();
        let kind = kernel.as_ref().as_any().type_id();
        kernel.set_constant_inputs(if bias.is_some() {
            &[false, true, true]
        } else {
            &[false, true]
        });
        let expected = oracle(case, x, w, bias, relu);
        let (sx, sw, sy) = (
            compute_contiguous_strides(&xs)
                .expect("fixture shape has representable contiguous strides"),
            compute_contiguous_strides(&ws)
                .expect("fixture shape has representable contiguous strides"),
            compute_contiguous_strides(&ys)
                .expect("fixture shape has representable contiguous strides"),
        );
        let bs = [case.w[0]];
        let mut inputs = vec![
            TensorView::new(
                DevicePtr(x.as_ptr().cast()),
                DataType::Float32,
                &xs,
                &sx,
                DeviceId::cpu(),
            ),
            TensorView::new(
                DevicePtr(w.as_ptr().cast()),
                DataType::Float32,
                &ws,
                &sw,
                DeviceId::cpu(),
            ),
        ];
        if let Some(bias) = bias {
            inputs.push(TensorView::new(
                DevicePtr(bias.as_ptr().cast()),
                DataType::Float32,
                &bs,
                &[1],
                DeviceId::cpu(),
            ));
        }
        let mut output = vec![f32::NAN; expected.len()];
        for _ in 0..2 {
            output.fill(f32::NAN);
            kernel
                .execute(
                    &inputs,
                    &mut [TensorMut::new(
                        DevicePtrMut(output.as_mut_ptr().cast()),
                        DataType::Float32,
                        &ys,
                        &sy,
                        DeviceId::cpu(),
                    )],
                )
                .unwrap();
            assert_eq!(output.len(), expected.len());
            for (i, (&got, &want)) in output.iter().zip(&expected).enumerate() {
                assert!(
                    got.is_finite() && (got as f64 - want).abs() <= 2e-5 * (1. + want.abs()),
                    "native={native} X={xs:?} W={ws:?} bias={} relu={relu} element={i} got={got} expected={want}",
                    bias.is_some()
                );
            }
        }
        kind
    }

    #[test]
    fn linked_conv_factories_are_independent_and_match_f64_on_real_execute() {
        assert!(backend_ab::mlas_available());
        let pair = backend_ab::convolution_factories();
        let cases = [
            Case {
                x: [2, 4, 7, 9],
                w: [6, 2, 2, 3],
                group: 2,
                strides: [1, 1],
                dilation: [2, 2],
                pads: [0; 4],
                rank3: false,
            },
            Case {
                x: [1, 2, 5, 7],
                w: [3, 2, 3, 3],
                group: 1,
                strides: [1, 1],
                dilation: [1, 1],
                pads: [1; 4],
                rank3: false,
            },
            Case {
                x: [2, 2, 5, 7],
                w: [3, 2, 2, 3],
                group: 1,
                strides: [2, 1],
                dilation: [1, 2],
                pads: [1, 0, 0, 2],
                rank3: false,
            },
            Case {
                x: [1, 4, 5, 7],
                w: [6, 2, 1, 1],
                group: 2,
                strides: [1, 1],
                dilation: [1, 1],
                pads: [0; 4],
                rank3: false,
            },
            Case {
                x: [1, 2, 5, 7],
                w: [3, 2, 1, 1],
                group: 1,
                strides: [1, 1],
                dilation: [1, 1],
                pads: [0; 4],
                rank3: false,
            },
            Case {
                x: [2, 3, 5, 7],
                w: [3, 1, 3, 3],
                group: 3,
                strides: [1, 1],
                dilation: [1, 1],
                pads: [1; 4],
                rank3: false,
            },
            Case {
                x: [1, 3, 5, 7],
                w: [6, 1, 3, 3],
                group: 3,
                strides: [2, 2],
                dilation: [1, 1],
                pads: [1; 4],
                rank3: false,
            },
            Case {
                x: [1, 2, 1, 9],
                w: [3, 2, 1, 3],
                group: 1,
                strides: [1, 2],
                dilation: [1, 1],
                pads: [0, 1, 0, 1],
                rank3: true,
            },
            Case {
                x: [2, 4, 1, 9],
                w: [6, 2, 1, 3],
                group: 2,
                strides: [1, 1],
                dilation: [1, 2],
                pads: [0, 0, 0, 1],
                rank3: true,
            },
        ];
        let registry = onnx_runtime_ep_cpu::kernels::build_cpu_registry();
        let witness = &cases[0];
        let registered = registry
            .lookup("Conv", "", 18)
            .unwrap()
            .create(
                &witness.node(false),
                &[witness.x.to_vec(), witness.w.to_vec()],
            )
            .unwrap();
        assert!(
            registered
                .as_ref()
                .as_any()
                .is::<onnx_runtime_ep_cpu::kernels::conv::ConvKernel>()
        );
        let reference_kind = registered.as_ref().as_any().type_id();
        dispatch_ledger::enable();
        let mut comparisons = 0;
        for (index, case) in cases.iter().enumerate() {
            let x = values(case.x.iter().product(), 71 + index as u64);
            let w = values(case.w.iter().product(), 101 + index as u64);
            let b = values(case.w[0], 37);
            for bias in [None, Some(b.as_slice())] {
                for relu in [false, true] {
                    dispatch_ledger::reset();
                    let native_kind = compare(
                        pair.native_generic_gemm.as_ref(),
                        case,
                        &x,
                        &w,
                        bias,
                        relu,
                        true,
                    );
                    assert_ne!(
                        native_kind, reference_kind,
                        "native and reference must be independent concrete kernels"
                    );
                    if index == 0 {
                        let seen = dispatch_ledger::snapshot();
                        assert_eq!(
                            seen.len(),
                            8,
                            "two real executes must each dispatch batch*group native GEMMs"
                        );
                        assert!(
                            seen.iter()
                                .all(|o| o.family == dispatch_ledger::KernelFamily::MatMulF32
                                    && o.backend == dispatch_ledger::Backend::Native
                                    && o.shape == (3, 25, 12)),
                            "native Conv must not route GEMM to MLAS: {seen:?}"
                        );
                        assert_eq!(dispatch_ledger::dropped(), 0);
                    }
                    let kind = compare(pair.mlas.as_ref(), case, &x, &w, bias, relu, false);
                    assert_eq!(kind, reference_kind);
                    comparisons += 1;
                }
            }
        }
        dispatch_ledger::disable();
        assert_eq!(comparisons, 36);
        // The empty-batch native boundary is exercised separately: it should
        // not require MLAS to support zero-batch convolution preparation.
        let empty = Case {
            x: [0, 4, 5, 7],
            w: [6, 2, 1, 1],
            group: 2,
            strides: [1, 1],
            dilation: [1, 1],
            pads: [0; 4],
            rank3: false,
        };
        compare(
            pair.native_generic_gemm.as_ref(),
            &empty,
            &[],
            &values(12, 99),
            None,
            false,
            true,
        );
        println!(
            "compared {comparisons} native/MLAS cases with two executions per factory; native empty batch also executed"
        );
    }
}
