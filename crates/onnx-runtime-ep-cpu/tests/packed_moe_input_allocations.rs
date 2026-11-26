//! Attribute floating payload copies to the actual packed MoE launch.
#[path = "support/allocation_observer.rs"]
mod allocation_observer;
#[path = "../benches/common/mod.rs"]
mod common;
use allocation_observer::watched;
use common::{FloatDType, PreparedKernel, Tensor};
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cpu::kernels::block_quantized_moe::{
    BLOCK_QUANT_MOE_CACHED_DENSE_TEST_HITS, BLOCK_QUANT_MOE_DENSE_EXPANSIONS,
};
use onnx_runtime_ep_cpu::{CpuExecutionProvider, CpuMemoryResources, CpuWeightCacheLimits};
use onnx_runtime_ir::{Attribute, Node, NodeId, ValueId, compute_contiguous_strides};
use std::hint::black_box;
use std::sync::atomic::Ordering;
static COUNTER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn packed_moe_inputs_borrow_or_convert_inside_admitted_regions() {
    let _guard = COUNTER_TEST_LOCK.lock().unwrap();
    const R: usize = 3;
    const H: usize = 96;
    const I: usize = 64;
    const E: usize = 19;
    let sizes = [[R * H * 4, R * E * 4], [E * I * 4, E * H * 4]];
    for pair in sizes {
        let (_, control) = watched(pair, || {
            black_box(vec![0u8; black_box(pair[0])]);
            black_box(vec![0u8; black_box(pair[1])]);
        });
        assert_eq!(control, [1, 1]);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let shapes = [vec![R,H],vec![R,E],vec![E,I,H.div_ceil(32),34],vec![E,I],vec![E,H,I.div_ceil(32),34],vec![E,H],vec![E,I,H.div_ceil(32),34],vec![E,I],vec![R,E]];
        for strided in [false, true] {
            let owners = shapes.iter().enumerate().map(|(slot, shape)| {
                let len = shape.iter().product::<usize>();
                if matches!(slot, 2 | 4 | 6) {
                    let mut bytes = vec![0; len];
                    for block in bytes.chunks_exact_mut(34) {
                        block[..2].copy_from_slice(&half::f16::ONE.to_le_bytes());
                    }
                    Tensor::u8(shape, &bytes)
                } else {
                    let mut values = vec![if slot == 0 {127.} else if slot == 5 {2.} else {0.}; len];
                    if matches!(slot,1|8) {
                        for row in values.chunks_exact_mut(E) { row[0] = 1.; }
                    }
                    if strided {
                        let values = values.into_iter().flat_map(|v| [v, f32::NAN]).collect::<Vec<_>>();
                        Tensor::floats(FloatDType::F32, &[len*2], &values)
                    } else { Tensor::floats(FloatDType::F32, shape, &values) }
                }
            }).collect::<Vec<_>>();
            let strides = shapes.iter().map(|shape| compute_contiguous_strides(shape).unwrap().into_iter().map(|s| s*2).collect::<Vec<_>>()).collect::<Vec<_>>();
            let inputs = owners.iter().enumerate().map(|(slot,tensor)| {
                let mut view = tensor.view();
                if strided && !matches!(slot,2|4|6) { view.shape=&shapes[slot]; view.strides=&strides[slot]; }
                view
            }).collect::<Vec<_>>();
            for (arithmetic, streamed) in [("float32",true),("float32",false),("q8_0",true)] {
                let mut node = Node::new(NodeId(0), "BlockQuantizedMoE",(0..9).map(|i|Some(ValueId(i))).collect(),vec![ValueId(9)]);
                node.domain=onnx_runtime_ir::RUNTIME_DOMAIN.into();
                for (name,value) in [
                    ("fc1_format",Attribute::String(b"q8_0".to_vec())),
                    ("fc2_format",Attribute::String(b"q8_0".to_vec())),
                    ("fc3_format",Attribute::String(b"q8_0".to_vec())),
                    ("activation_type",Attribute::String(b"silu".to_vec())),
                    ("k",Attribute::Int(1)),
                    ("normalize_routing_weights",Attribute::Int(1)),
                    ("swiglu_fusion",Attribute::Int(0)),
                    ("activation_arithmetic",Attribute::String(arithmetic.as_bytes().to_vec())),
                    ("block_layout_version",Attribute::Int(if arithmetic=="q8_0" {2}else{1})),
                ] { node.attributes.insert(name.into(),value); }
                let cpu = CpuExecutionProvider::with_resources(CpuMemoryResources::new(Default::default(), CpuWeightCacheLimits { dense_bytes: if streamed {0} else {1 << 20}, expert_bytes:0 }).unwrap());
                let before = (BLOCK_QUANT_MOE_DENSE_EXPANSIONS.load(Ordering::Relaxed), BLOCK_QUANT_MOE_CACHED_DENSE_TEST_HITS.load(Ordering::Relaxed));
                let mut kernel=cpu.get_kernel(&node,&[],1).unwrap();
                kernel.set_constant_inputs(&[false,false,true,false,true,false,true,false,false]);
                let mut y=Tensor::zeros(FloatDType::F32,&[R,H]);
                let mut prepared=PreparedKernel::new(&cpu,kernel,&inputs,&[y.view_mut()]);
                for pair in sizes {
                    for _ in 0..3 {
                        let (result,counts)=watched(pair,||prepared.execute(&inputs,&mut [y.view_mut()]));
                        result.unwrap();
                        assert_eq!(counts,[0;2],"{arithmetic} streamed={streamed} strided={strided} payload sizes={pair:?}");
                        assert!(y.f32s().iter().all(|&v|v==2.));
                    }
                }
                let expanded = BLOCK_QUANT_MOE_DENSE_EXPANSIONS.load(Ordering::Relaxed);
                assert_eq!(expanded-before.0, if arithmetic=="float32" && !streamed {3} else {0});
                if arithmetic=="float32" && !streamed { assert!(BLOCK_QUANT_MOE_CACHED_DENSE_TEST_HITS.load(Ordering::Relaxed)>before.1); }
            }
        }
    });
}

#[test]
fn packed_moe_strided_experts_use_admitted_lane_without_payload_allocations() {
    let _guard = COUNTER_TEST_LOCK.lock().unwrap();
    const R: usize = 3;
    const H: usize = 96;
    const I: usize = 64;
    const E: usize = 7;
    let sizes = [I * (H / 32) * 18, H * (I / 32) * 34];
    let (_, control) = watched(sizes, || {
        black_box(vec![0u8; black_box(sizes[0])]);
        black_box(vec![0u8; black_box(sizes[1])]);
    });
    assert_eq!(control, [1, 1]);
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            let shapes = [
                vec![R, H],
                vec![R, E],
                vec![E, I, H / 32, 18],
                vec![E, I],
                vec![E, H, I / 32, 34],
                vec![E, H],
                vec![E, I, H / 32, 18],
            ];
            for strided in [false, true] {
                let owners = shapes
                    .iter()
                    .enumerate()
                    .map(|(slot, shape)| {
                        let len = shape.iter().product::<usize>();
                        if matches!(slot, 2 | 4 | 6) {
                            let width = if slot == 4 { 34 } else { 18 };
                            let mut bytes = vec![0; len];
                            for block in bytes.chunks_exact_mut(width) {
                                block[..2]
                                    .copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
                                // Nonzero weights and activations exercise every FC.
                                block[2..].fill(if slot == 4 { 1 } else { 0x99 });
                            }
                            if strided {
                                Tensor::u8(
                                    &[len * 2],
                                    &bytes
                                        .into_iter()
                                        .flat_map(|b| [b, 0xa5])
                                        .collect::<Vec<_>>(),
                                )
                            } else {
                                Tensor::u8(shape, &bytes)
                            }
                        } else {
                            let mut values = vec![if slot == 0 { 0.25 } else { 0.0 }; len];
                            if slot == 1 {
                                for row in values.chunks_exact_mut(E) {
                                    row[0] = 1.0;
                                }
                            }
                            Tensor::floats(FloatDType::F32, shape, &values)
                        }
                    })
                    .collect::<Vec<_>>();
                let strides = shapes
                    .iter()
                    .map(|s| {
                        compute_contiguous_strides(s)
                            .unwrap()
                            .into_iter()
                            .map(|n| n * 2)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let inputs = owners
                    .iter()
                    .enumerate()
                    .map(|(slot, owner)| {
                        let mut view = owner.view();
                        if strided && matches!(slot, 2 | 4 | 6) {
                            view.shape = &shapes[slot];
                            view.strides = &strides[slot];
                        }
                        view
                    })
                    .collect::<Vec<_>>();
                for (arithmetic, dense) in [("float32", false), ("float32", true), ("q8_0", false)]
                {
                    let mut node = Node::new(
                        NodeId(0),
                        "BlockQuantizedMoE",
                        (0..7).map(|i| Some(ValueId(i))).collect(),
                        vec![ValueId(7)],
                    );
                    node.domain = onnx_runtime_ir::RUNTIME_DOMAIN.into();
                    for (name, value) in [
                        ("fc1_format", Attribute::String(b"gguf:q4_0".to_vec())),
                        ("fc2_format", Attribute::String(b"q8_0".to_vec())),
                        ("fc3_format", Attribute::String(b"gguf:q4_0".to_vec())),
                        ("activation_type", Attribute::String(b"silu".to_vec())),
                        ("k", Attribute::Int(1)),
                        ("normalize_routing_weights", Attribute::Int(1)),
                        ("swiglu_fusion", Attribute::Int(0)),
                        (
                            "activation_arithmetic",
                            Attribute::String(arithmetic.as_bytes().to_vec()),
                        ),
                        (
                            "block_layout_version",
                            Attribute::Int(if arithmetic == "q8_0" { 2 } else { 1 }),
                        ),
                    ] {
                        node.attributes.insert(name.into(), value);
                    }
                    let cpu = CpuExecutionProvider::with_resources(
                        CpuMemoryResources::new(
                            Default::default(),
                            CpuWeightCacheLimits {
                                dense_bytes: if dense { 1 << 20 } else { 0 },
                                expert_bytes: 0,
                            },
                        )
                        .unwrap(),
                    );
                    let mut kernel = cpu.get_kernel(&node, &[], 1).unwrap();
                    kernel.set_constant_inputs(&[false, false, true, false, true, false, true]);
                    let mut y = Tensor::zeros(FloatDType::F32, &[R, H]);
                    let mut prepared = PreparedKernel::new(&cpu, kernel, &inputs, &[y.view_mut()]);
                    let before = BLOCK_QUANT_MOE_DENSE_EXPANSIONS.load(Ordering::Relaxed);
                    for launch in 0..3 {
                        let (result, counts) =
                            watched(sizes, || prepared.execute(&inputs, &mut [y.view_mut()]));
                        result.unwrap();
                        assert_eq!(
                            counts,
                            [0, 0],
                            "{arithmetic} dense={dense} strided={strided} launch={launch}"
                        );
                        // FC1=FC3=3; FC2=sum of 64 identical gated entries / 8.
                        let expected = 8.0 * 9.0 / (1.0 + (-3.0f32).exp());
                        assert!(
                            y.f32s().iter().all(|&v| (v - expected).abs() < 0.002),
                            "{arithmetic} dense={dense} strided={strided}: {:?}",
                            &y.f32s()[..4]
                        );
                    }
                    assert_eq!(
                        BLOCK_QUANT_MOE_DENSE_EXPANSIONS.load(Ordering::Relaxed) - before,
                        if dense { 3 } else { 0 }
                    );
                }
            }
        });
}
