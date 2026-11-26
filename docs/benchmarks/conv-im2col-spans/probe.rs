//! Whole-process Conv study. Parent wait4 owns CPU/RSS accounting.
use onnx_runtime_ep_api::{
    DeviceId, DevicePtr, DevicePtrMut, Kernel, KernelFactory, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::{backend::CpuBackend, dispatch_ledger};
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, compute_contiguous_strides};
use std::{hint::black_box, time::Instant};

struct Case {
    x: [usize; 4],
    w: [usize; 4],
    group: usize,
    dilation: usize,
    stride: usize,
    pad: usize,
    rank3: bool,
}
impl Case {
    fn shape(&self, s: [usize; 4]) -> Vec<usize> {
        if self.rank3 {
            vec![s[0], s[1], s[3]]
        } else {
            s.to_vec()
        }
    }
    fn output(&self) -> [usize; 4] {
        let h = if self.rank3 {
            1
        } else {
            (self.x[2] + 2 * self.pad - ((self.w[2] - 1) * self.dilation + 1)) / self.stride + 1
        };
        let w =
            (self.x[3] + 2 * self.pad - ((self.w[3] - 1) * self.dilation + 1)) / self.stride + 1;
        [self.x[0], self.w[0], h, w]
    }
    fn node(&self) -> Node {
        let mut n = Node::new(NodeId(0), "Conv", vec![], vec![]);
        n.attributes
            .insert("group".into(), Attribute::Int(self.group as i64));
        for (name, v) in [("strides", self.stride), ("dilations", self.dilation)] {
            n.attributes.insert(
                name.into(),
                Attribute::Ints(vec![v as i64; if self.rank3 { 1 } else { 2 }]),
            );
        }
        n.attributes.insert(
            "pads".into(),
            Attribute::Ints(vec![self.pad as i64; if self.rank3 { 2 } else { 4 }]),
        );
        n
    }
}
fn cases() -> [Case; 6] {
    [
        Case {
            x: [1, 8, 16, 16],
            w: [16, 8, 3, 3],
            group: 1,
            dilation: 1,
            stride: 1,
            pad: 1,
            rank3: false,
        },
        Case {
            x: [1, 8, 16, 16],
            w: [16, 8, 3, 3],
            group: 1,
            dilation: 2,
            stride: 1,
            pad: 2,
            rank3: false,
        },
        Case {
            x: [2, 16, 12, 12],
            w: [24, 8, 3, 3],
            group: 2,
            dilation: 1,
            stride: 1,
            pad: 1,
            rank3: false,
        },
        Case {
            x: [1, 16, 16, 16],
            w: [16, 1, 3, 3],
            group: 16,
            dilation: 1,
            stride: 1,
            pad: 1,
            rank3: false,
        },
        Case {
            x: [1, 256, 4, 4],
            w: [128, 256, 1, 1],
            group: 1,
            dilation: 1,
            stride: 1,
            pad: 0,
            rank3: false,
        },
        Case {
            x: [1, 8, 1, 128],
            w: [16, 8, 1, 5],
            group: 1,
            dilation: 1,
            stride: 2,
            pad: 2,
            rank3: true,
        },
    ]
}
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 32) as u32 as f64 / u32::MAX as f64) - 0.5) as f32 * 0.5
        })
        .collect()
}
fn oracle(c: &Case, x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f64> {
    let [batches, ic, ih, iw] = c.x;
    let [oc, ig, kh, kw] = c.w;
    let [_, _, oh, ow] = c.output();
    let mut out = vec![0.; batches * oc * oh * ow];
    for b in 0..batches {
        for o in 0..oc {
            for y in 0..oh {
                for z in 0..ow {
                    let g = o / (oc / c.group);
                    let mut v = bias[o] as f64;
                    for i in 0..ig {
                        for yy in 0..kh {
                            for zz in 0..kw {
                                let sy = if c.rank3 {
                                    0
                                } else {
                                    y * c.stride + yy * c.dilation
                                };
                                let sx = z * c.stride + zz * c.dilation;
                                let py = if c.rank3 {
                                    Some(0)
                                } else {
                                    sy.checked_sub(c.pad)
                                };
                                let px = sx.checked_sub(c.pad);
                                if let (Some(py), Some(px)) = (py, px) {
                                    if py < ih && px < iw {
                                        let a = ((b * ic + g * ig + i) * ih + py) * iw + px;
                                        let q = ((o * ig + i) * kh + yy) * kw + zz;
                                        v += x[a] as f64 * w[q] as f64;
                                    }
                                }
                            }
                        }
                    }
                    out[((b * oc + o) * oh + y) * ow + z] = v;
                }
            }
        }
    }
    out
}
fn validate(output: &[f32], reference: &[f64]) -> f64 {
    assert_eq!(output.len(), reference.len());
    output
        .iter()
        .zip(reference)
        .map(|(&a, &b)| {
            let d = (a as f64 - b).abs();
            assert!(
                a.is_finite() && d <= 2e-5 * (1. + b.abs()),
                "Conv f64 mismatch {a} vs {b}"
            );
            d
        })
        .fold(0., f64::max)
}
#[inline(never)]
fn execute(
    kernel: &dyn Kernel,
    inputs: &[TensorView<'_>],
    shape: &[usize],
    strides: &[i64],
    output: &mut [f32],
) {
    black_box(kernel)
        .execute(
            black_box(inputs),
            &mut [TensorMut::new(
                DevicePtrMut(output.as_mut_ptr().cast()),
                DataType::Float32,
                shape,
                strides,
                DeviceId::cpu(),
            )],
        )
        .expect("Conv execute must succeed on the validated fixture");
    black_box(output);
}
fn create(factory: &dyn KernelFactory, node: &Node, shapes: &[Vec<usize>]) -> Box<dyn Kernel> {
    let mut k = black_box(factory)
        .create(black_box(node), black_box(shapes))
        .expect("Conv factory must prepare the fixture geometry");
    k.set_constant_inputs(&[false, true, true]);
    k
}
fn fpcr() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        let v: u64;
        unsafe {
            std::arch::asm!("mrs {v}, fpcr",v=out(reg)v,options(nomem,nostack,preserves_flags))
        };
        v
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        0
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    assert_eq!(
        args.len(),
        5,
        "usage: after|before case0..5 reuse|new calls output.f32"
    );
    let arm = match args[0].as_str() {
        "after" => 0,
        "before" => 1,
        other => panic!("invalid arm {other}; use after or before"),
    };
    let case_id = args[1]
        .parse::<usize>()
        .expect("case must be an integer in 0..6");
    let is_new = match args[2].as_str() {
        "reuse" => false,
        "new" => true,
        other => panic!("invalid phase {other}; use reuse or new"),
    };
    let calls = args[3]
        .parse::<usize>()
        .expect("calls must be a nonnegative integer; zero denotes setup");
    let list = cases();
    let c = list.get(case_id).expect("case must be in 0..6");
    assert_eq!(
        rayon::current_num_threads(),
        1,
        "study requires RAYON_NUM_THREADS=1"
    );
    let factories: Vec<Box<dyn KernelFactory>> = vec![
        Box::new(onnx_runtime_ep_cpu::kernels::conv::ConvFactory),
        Box::new(onnx_runtime_ep_cpu::kernels::im2col_before::ConvFactory),
    ];
    let xs = c.shape(c.x);
    let ws = c.shape(c.w);
    let ys = c.shape(c.output());
    let shapes = vec![xs.clone(), ws.clone(), vec![c.w[0]]];
    let node = c.node();
    let (sx, sw, sy) = (
        compute_contiguous_strides(&xs)?,
        compute_contiguous_strides(&ws)?,
        compute_contiguous_strides(&ys)?,
    );
    let bs = [c.w[0]];
    let x = values(c.x.iter().product(), 71);
    let w = values(c.w.iter().product(), 101);
    let bias = values(c.w[0], 37);
    let reference = oracle(c, &x, &w, &bias);
    assert!(
        !reference.is_empty() && reference.iter().any(|x| *x != 0.),
        "fixture must demand nonzero output"
    );
    let inputs = [
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
        TensorView::new(
            DevicePtr(bias.as_ptr().cast()),
            DataType::Float32,
            &bs,
            &[1],
            DeviceId::cpu(),
        ),
    ];
    let mut output = vec![f32::NAN; reference.len()];
    let mut kernels = Vec::new();
    let mut create_ns = Vec::new();
    let mut first_ns = Vec::new();
    let mut second_ns = Vec::new();
    let mut native_gemm = Vec::new();
    let mut mlas_gemm = Vec::new();
    let mut first_bits = None;
    for factory in &factories {
        let t = Instant::now();
        let k = create(factory.as_ref(), &node, &shapes);
        create_ns.push(t.elapsed().as_nanos());
        dispatch_ledger::disable();
        let t = Instant::now();
        execute(k.as_ref(), &inputs, &ys, &sy, &mut output);
        first_ns.push(t.elapsed().as_nanos());
        let bits = output.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        if let Some(previous) = &first_bits { assert_eq!(&bits, previous, "before/after complete outputs must match bits"); } else { first_bits = Some(bits); }
        validate(&output, &reference);
        let t = Instant::now();
        execute(k.as_ref(), &inputs, &ys, &sy, &mut output);
        second_ns.push(t.elapsed().as_nanos());
        validate(&output, &reference);
        // Reachability recording is separate from the uninstrumented first /
        // second wall diagnostics and is disabled before the work loop.
        dispatch_ledger::reset();
        dispatch_ledger::enable();
        execute(k.as_ref(), &inputs, &ys, &sy, &mut output);
        execute(k.as_ref(), &inputs, &ys, &sy, &mut output);
        validate(&output, &reference);
        let seen = dispatch_ledger::snapshot();
        native_gemm.push(
            seen.iter()
                .filter(|o| {
                    o.family == dispatch_ledger::KernelFamily::MatMulF32
                        && o.backend == dispatch_ledger::Backend::Native
                })
                .count(),
        );
        mlas_gemm.push(
            seen.iter()
                .filter(|o| {
                    o.family == dispatch_ledger::KernelFamily::MatMulF32
                        && o.backend == dispatch_ledger::Backend::Mlas
                })
                .count(),
        );
        assert_eq!(dispatch_ledger::dropped(), 0);
        dispatch_ledger::disable();
        kernels.push(k);
    }
    if factories.len() == 2 {
        assert_ne!(
            kernels[0].as_ref().as_any().type_id(),
            kernels[1].as_ref().as_any().type_id(),
            "linked comparison cannot alias kernels"
        );
        assert_eq!(mlas_gemm[0], 0, "native arithmetic must stay independent");
        if [1, 2, 4].contains(&case_id) {
            assert!(
                native_gemm[0] > 0,
                "forced grouped/dilated/pointwise GEMM witness must execute"
            );
        }
    }
    for _ in 0..8 {
        execute(kernels[arm].as_ref(), &inputs, &ys, &sy, &mut output);
    }
    validate(&output, &reference);
    let begin_fpcr = fpcr();
    if calls > 0 {
        output.fill(f32::NAN);
        for _ in 0..calls {
            if is_new {
                let k = create(factories[arm].as_ref(), &node, &shapes);
                execute(k.as_ref(), &inputs, &ys, &sy, &mut output);
            } else {
                execute(kernels[arm].as_ref(), &inputs, &ys, &sy, &mut output);
            }
        }
    }
    let max_error = validate(&output, &reference);
    assert_eq!(
        begin_fpcr,
        fpcr(),
        "Conv work must preserve floating control state"
    );
    let mut digest = 0xcbf29ce484222325u64;
    let mut bytes = Vec::with_capacity(output.len() * 4);
    for v in &output {
        let b = v.to_le_bytes();
        bytes.extend_from_slice(&b);
        for x in b {
            digest = (digest ^ x as u64).wrapping_mul(0x100000001b3);
        }
    }
    std::fs::write(&args[4], bytes).expect("write complete output bytes");
    let actual_backend = format!("{:?}", CpuBackend::auto_detect());
    println!(
        "{{\"arm\":\"{}\",\"case\":{case_id},\"phase\":\"{}\",\"calls\":{calls},\"linked\":{},\"auto_backend\":\"{actual_backend}\",\"shape_x\":{xs:?},\"shape_w\":{ws:?},\"shape_y\":{ys:?},\"create_wall_ns\":{create_ns:?},\"first_wall_ns\":{first_ns:?},\"second_wall_ns\":{second_ns:?},\"native_gemm_witnesses\":{native_gemm:?},\"mlas_gemm_witnesses\":{mlas_gemm:?},\"max_abs_f64_error\":{max_error},\"digest\":\"{digest:016x}\",\"fpcr\":{begin_fpcr}}}",
        args[0],
        args[2],
        false
    );
    Ok(())
}
