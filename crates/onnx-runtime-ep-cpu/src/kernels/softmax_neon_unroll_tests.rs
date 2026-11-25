//! Same-binary control for the ARM64 loop-unrolling candidate.

use super::*;

// Match the public native entry's Result ABI, complete-row validation and
// runtime ISA seam. Only the row body's compile-time unroll parameter differs.
#[inline(never)]
fn original_rows<const OUTLINE: bool>(
    src: &[f32],
    dst: &mut [f32],
    columns: usize,
) -> onnx_runtime_ep_api::Result<()> {
    if src.len() != dst.len() {
        return Err(onnx_runtime_ep_api::EpError::KernelFailed(format!(
            "Softmax rows: source has {} elements, destination has {}, columns={columns}; supply equal-length buffers containing complete rows",
            src.len(),
            dst.len(),
        )));
    }
    let Some(width) = super::super::checked_row_width(src.len(), columns)? else {
        return Ok(());
    };
    for (input, output) in src
        .chunks_exact(width.get())
        .zip(dst.chunks_exact_mut(width.get()))
    {
        let d = input.len();
        if d >= super::super::MIN_NEON_SOFTMAX_ROW
            && std::arch::is_aarch64_feature_detected!("neon")
        {
            // SAFETY: runtime detection; complete, disjoint rows.
            if OUTLINE {
                // SAFETY: this branch forwards the same proven row contract.
                unsafe { original_row_outlined(input.as_ptr(), output.as_mut_ptr(), d) };
            } else {
                // SAFETY: the same complete-row and runtime ISA proof.
                unsafe { softmax_row_impl::<true, false>(input.as_ptr(), output.as_mut_ptr(), d) };
            }
        } else {
            // SAFETY: complete, disjoint scalar rows.
            unsafe { super::super::softmax_row_scalar(input.as_ptr(), output.as_mut_ptr(), d) };
        }
    }
    Ok(())
}

fn original(src: &[f32], dst: &mut [f32], columns: usize) {
    original_rows::<false>(src, dst, columns).unwrap();
}

#[target_feature(enable = "neon")]
#[inline(never)]
unsafe fn original_row_outlined(src: *const f32, dst: *mut f32, d: usize) {
    // SAFETY: caller supplies NEON and a complete identical/disjoint row.
    unsafe { softmax_row_impl::<true, false>(src, dst, d) };
}

#[cfg(feature = "mlas")]
fn original_outlined(src: &[f32], dst: &mut [f32], columns: usize) {
    original_rows::<true>(src, dst, columns).unwrap();
}

#[test]
fn unrolled_rows_preserve_original_bits_and_inplace_tail_bounds() {
    super::tests::standard_fpcr();
    for d in (1..=137).chain([257, 1024, 4099]) {
        for offset in 0..4 {
            for kind in 0..5 {
                let mut input: Vec<f32> = (0..d)
                    .map(|i| ((i * 7919 % 100003) as f32 - 50001.) / 313.)
                    .collect();
                match kind {
                    1 => input[d / 2] = f32::NAN,
                    2 => input.fill(f32::NEG_INFINITY),
                    3 => input[d / 2] = f32::INFINITY,
                    4 => {
                        input.fill(-0.);
                        input[d / 2] = 0.;
                    }
                    _ => {}
                }
                let mut expected = vec![0.; d];
                original(&input, &mut expected, d);
                let mut output = vec![12345.; offset + d + 1];
                super::super::softmax_rows_serial_out_native(
                    &input,
                    &mut output[offset..offset + d],
                    d,
                )
                .unwrap();
                assert!(output[..offset].iter().all(|&x| x == 12345.));
                assert_eq!(output[offset + d], 12345.);
                assert_eq!(
                    output[offset..offset + d]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "out-of-place d={d}, offset={offset}, kind={kind}"
                );
                output[offset..offset + d].copy_from_slice(&input);
                super::super::softmax_rows_serial_native(&mut output[offset..offset + d], d)
                    .unwrap();
                assert_eq!(
                    output[offset..offset + d]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "in-place d={d}, offset={offset}, kind={kind}"
                );
                assert!(output[..offset].iter().all(|&x| x == 12345.));
                assert_eq!(output[offset + d], 12345.);
            }
        }
    }
}

#[cfg(feature = "mlas")]
#[test]
#[ignore = "whole-process CPU diagnostic; use the parent wait4 runner under the host flock"]
fn softmax_unroll_whole_process() {
    use std::hint::black_box;
    let argument = |name: &str| -> usize {
        std::env::var(name)
            .unwrap_or_else(|error| panic!("set {name} for the whole-process runner: {error}"))
            .parse()
            .unwrap_or_else(|error| panic!("{name} must be an unsigned integer: {error}"))
    };
    let width = argument("NXRT_UNROLL_WIDTH");
    let case = argument("NXRT_UNROLL_CASE");
    let arm = argument("NXRT_UNROLL_ARM");
    let calls = argument("NXRT_UNROLL_CALLS");
    assert!([1, 4, 8].contains(&width) && case < 8 && arm < 5 && calls <= 1_000_000_000);
    let shapes = [
        (1, 4),
        (1, 7),
        (1, 8),
        (1, 16),
        (1, 257),
        (32, 1024),
        (128, 4099),
        (16, 65539),
    ];
    let (n, d) = shapes[case];
    crate::task_runtime::set_task_thread_budget(Some(width)).unwrap();
    assert_eq!(crate::task_runtime::width(), width);
    assert!(rayon::current_thread_index().is_none() && !crate::task_runtime::in_task());
    let fpcr = super::tests::standard_fpcr();
    let source: Vec<f32> = (0..n * d)
        .map(|i| -16. + (i * 7919 % 100003) as f32 * (16. / 100002.))
        .collect();
    let mut output = vec![0.; source.len()];
    let kernel: fn(&[f32], &mut [f32], usize) = match arm {
        0 => original,
        1 | 2 => |src, dst, d| super::super::softmax_rows_serial_out_native(src, dst, d).unwrap(),
        3 => |src, dst, d| mlas_sys::compute_softmax(src, dst, src.len() / d, d),
        4 => original_outlined,
        _ => unreachable!(),
    };
    let run = |out: &mut [f32]| {
        super::super::softmax_rows_with_kernel(
            black_box(source.as_slice()),
            black_box(out),
            n,
            d,
            kernel,
        )
    };
    for _ in 0..3 {
        run(&mut output);
    }
    let before = crate::task_runtime::testing::counters();
    run(&mut output);
    let after = crate::task_runtime::testing::counters();
    let workers =
        (after.tasks - before.tasks) - (after.tasks_by_dispatcher - before.tasks_by_dispatcher);
    let parallel = width > 1 && super::super::fan_out_is_worthwhile(n, d);
    assert_eq!(workers > 0, parallel, "actual worker participation differs");
    let mut worst = 0_f64;
    for (src, out) in source.chunks(d).zip(output.chunks(d)) {
        let maximum = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = src
            .iter()
            .map(|&x| (f64::from(x) - f64::from(maximum)).exp())
            .sum();
        for (&x, &y) in src.iter().zip(out) {
            let error = (f64::from(y) - (f64::from(x) - f64::from(maximum)).exp() / sum).abs();
            assert!(y.is_finite() && error <= 1e-6);
            worst = worst.max(error);
        }
    }
    let digest = |values: &[f32]| {
        values.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, x| {
            (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3)
        })
    };
    let hash = digest(&output);
    let before = SOFTMAX_NEON_TEST_HITS.load(std::sync::atomic::Ordering::Relaxed);
    // Parent wait4 records CPU for the whole process, including the fixed
    // setup/validation above. Calls=0 is an explicit setup-only control, and
    // doubled-work processes test whether that fixed cost obscures the result.
    for _ in 0..calls {
        run(&mut output);
    }
    let hits = SOFTMAX_NEON_TEST_HITS.load(std::sync::atomic::Ordering::Relaxed) - before;
    assert_eq!(
        hits,
        if arm != 3 && d >= super::super::MIN_NEON_SOFTMAX_ROW {
            n * calls
        } else {
            0
        }
    );
    assert_eq!(
        digest(&output),
        hash,
        "complete output changed during repeated work"
    );
    assert_eq!(super::tests::standard_fpcr(), fpcr);
    println!(
        "UNROLL_PROCESS,{width},{case},{n},{d},{arm},{calls},{hash},{worst},{workers},{hits},{fpcr}"
    );
}
