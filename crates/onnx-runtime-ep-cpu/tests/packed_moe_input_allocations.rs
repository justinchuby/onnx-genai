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

#[test]
fn packed_moe_borrows_contiguous_payloads_and_materializes_strided_sources() {
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
                prepared.execute(&inputs,&mut [y.view_mut()]).unwrap();
                assert!(y.f32s().iter().all(|&v|v==2.));
                let expanded = BLOCK_QUANT_MOE_DENSE_EXPANSIONS.load(Ordering::Relaxed);
                assert_eq!(expanded-before.0, if arithmetic=="float32" && !streamed {3} else {0});
                for (pair, expected) in sizes.into_iter().zip([[1,2],[2,1]]) {
                    for _ in 0..3 {
                        let (result,counts)=watched(pair,||prepared.execute(&inputs,&mut [y.view_mut()]));
                        result.unwrap();
                        assert_eq!(counts,if strided {expected}else{[0;2]},"{arithmetic} streamed={streamed} strided={strided} payload sizes={pair:?}");
                        assert!(y.f32s().iter().all(|&v|v==2.));
                    }
                }
                if arithmetic=="float32" && !streamed { assert!(BLOCK_QUANT_MOE_CACHED_DENSE_TEST_HITS.load(Ordering::Relaxed)>before.1); }
            }
        }
    });
}
