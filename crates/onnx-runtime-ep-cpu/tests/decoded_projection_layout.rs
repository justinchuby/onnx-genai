//! Exercise the private layout/resource seam without expanding the public EP API.
#[path = "../src/kernels/decoded_projection.rs"]
mod decoded_projection;

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn tiled_decoder_reuses_shard_state_and_preserves_partial_columns_and_bias() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for n in [5, 13] {
        for k in [0, 5] {
            let m = 3;
            let input: Vec<f32> = (0..m * k).map(|i| i as f32 / 8. - 1.).collect();
            let bias: Vec<f32> = (0..n).map(|i| -(i as f32) / 4.).collect();
            let factories = AtomicUsize::new(0);
            let decoded = Mutex::new(Vec::new());
            let projected = Mutex::new(Vec::new());
            let mut actual = vec![f32::NAN; m * n];
            pool.install(|| {
                decoded_projection::linear_decoded_grouped_into::<4, _>(
                    "tile fixture",
                    &input,
                    m,
                    k,
                    n,
                    Some(&bias),
                    &mut actual,
                    || {
                        factories.fetch_add(1, Ordering::Relaxed);
                        let decoded = &decoded;
                        move |feature, row: &mut [f32]| {
                            assert_eq!(row.len(), k);
                            decoded.lock().unwrap().push(feature);
                            for (p, value) in row.iter_mut().enumerate() {
                                *value = (feature * 7 + p) as f32 / 16.;
                            }
                            Ok(())
                        }
                    },
                    |a, weights, output, rows, width, features| {
                        assert_eq!(weights.len(), features * width);
                        assert_eq!(output.len(), features * rows);
                        assert!((1..=4).contains(&features));
                        projected.lock().unwrap().push(features);
                        for feature in 0..features {
                            for row in 0..rows {
                                output[feature * rows + row] = (0..width).fold(0., |sum, p| {
                                    sum + a[row * width + p] * weights[feature * width + p]
                                });
                            }
                        }
                        Ok(())
                    },
                    decoded_projection::rayon_decoded_columns,
                )
                .unwrap()
            });
            let mut calls = decoded.into_inner().unwrap();
            calls.sort_unstable();
            assert_eq!(
                calls,
                (0..n).collect::<Vec<_>>(),
                "decode every feature exactly once"
            );
            assert_eq!(
                factories.load(Ordering::Relaxed),
                2,
                "one decoder state per actual shard, reused across tiles"
            );
            let mut counts = projected.into_inner().unwrap();
            counts.sort_unstable();
            assert_eq!(counts, if n == 5 { vec![2, 3] } else { vec![2, 3, 4, 4] });
            let expected: Vec<f32> = (0..m)
                .flat_map(|row| {
                    let input = &input;
                    let bias = &bias;
                    (0..n).map(move |feature| {
                        (0..k).fold(0., |sum, p| {
                            sum + input[row * k + p] * (feature * 7 + p) as f32 / 16.
                        }) + bias[feature]
                    })
                })
                .collect();
            assert_eq!(
                actual, expected,
                "M={m} K={k} N={n}: preserve row-major output and global bias indices"
            );
        }
    }
}

#[test]
fn incomplete_decoder_shards_fail_before_creating_state_or_projecting() {
    let factories = AtomicUsize::new(0);
    let mut output = [37.0; 6];
    let result = decoded_projection::linear_decoded_grouped_into::<4, _>(
        "malformed tile fixture",
        &[1.; 6],
        2,
        3,
        3,
        None,
        &mut output,
        || {
            factories.fetch_add(1, Ordering::Relaxed);
            |_, _: &mut [f32]| panic!("malformed shards must not decode")
        },
        |_, _, _, _, _, _| panic!("malformed shards must not project"),
        |columns, _, compute| {
            let incomplete = columns.len() - 1;
            compute(0, &mut columns[..incomplete])
        },
    );
    let message = result.unwrap_err().to_string();
    assert!(
        message.contains("malformed tile fixture")
            && message.contains("complete output columns")
            && message.contains("supply"),
        "{message}"
    );
    assert_eq!(factories.load(Ordering::Relaxed), 0);
    assert_eq!(output, [37.0; 6]);
}

#[test]
fn fixed_plan_bounds_shards_in_a_larger_pool_and_reuses_borrowed_scratch() {
    use decoded_projection::DecodedProjectionPlan;
    use std::num::NonZeroUsize as Nz;
    let plan = DecodedProjectionPlan::new(Nz::new(2).unwrap(), Nz::new(4).unwrap());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build()
        .unwrap();
    for m in [0, 1, 3] {
        for n in [1usize, 5, 13] {
            for k in [0, 5] {
                let a: Vec<f32> = (0..m * k).map(|i| i as f32 / 8. - 1.).collect();
                let bias: Vec<f32> = (0..n).map(|i| i as f32 / 4.).collect();
                let per_shard = n.div_ceil(2);
                let shards = n.div_ceil(per_shard);
                let columns = if m > 1 && n > 1 { m * n } else { 0 };
                let weights = if m == 0 {
                    0
                } else {
                    shards * 4.min(per_shard) * k
                };
                let layout = plan.layout(m, k, n).unwrap();
                assert_eq!(layout.bytes(), (columns + weights) * 4);
                let factories = AtomicUsize::new(0);
                let mut col = vec![f32::NAN; columns];
                let mut scratch = vec![f32::NAN; weights];
                for _ in 0..3 {
                    factories.store(0, Ordering::Relaxed);
                    let mut output = vec![f32::NAN; m * n];
                    pool.install(|| {
                        plan.project::<4, _>(
                            &a,
                            m,
                            k,
                            n,
                            Some(&bias),
                            &mut output,
                            &mut col,
                            &mut scratch,
                            || {
                                factories.fetch_add(1, Ordering::Relaxed);
                                |feature, row: &mut [f32]| {
                                    for (p, value) in row.iter_mut().enumerate() {
                                        *value = (feature * 7 + p) as f32 / 16.;
                                    }
                                    Ok(())
                                }
                            },
                            |a, weights, output, rows, width, features| {
                                for feature in 0..features {
                                    for row in 0..rows {
                                        output[feature * rows + row] =
                                            (0..width).fold(0., |sum, p| {
                                                sum + a[row * width + p]
                                                    * weights[feature * width + p]
                                            });
                                    }
                                }
                                Ok(())
                            },
                        )
                    })
                    .unwrap();
                    assert_eq!(
                        factories.load(Ordering::Relaxed),
                        if m == 0 { 0 } else { shards }
                    );
                    let expected: Vec<f32> = (0..m)
                        .flat_map(|row| {
                            let a = &a;
                            let bias = &bias;
                            (0..n).map(move |feature| {
                                (0..k).fold(0., |sum, p| {
                                    sum + a[row * k + p] * (feature * 7 + p) as f32 / 16.
                                }) + bias[feature]
                            })
                        })
                        .collect();
                    assert_eq!(output, expected);
                    col.fill(f32::NAN);
                    scratch.fill(f32::NAN);
                }
            }
        }
    }
    // Pool selection happens once; execution never enlarges this plan.
    let captured = pool.install(|| DecodedProjectionPlan::for_current_pool(Nz::new(4).unwrap()));
    assert!(captured.layout(3, 5, 13).unwrap().bytes() > 0);
}

#[test]
fn fixed_plan_rejects_overflow_and_mismatched_storage_before_decoding() {
    use decoded_projection::DecodedProjectionPlan;
    use std::num::NonZeroUsize as Nz;
    let plan = DecodedProjectionPlan::new(Nz::new(2).unwrap(), Nz::new(4).unwrap());
    assert!(
        plan.layout(usize::MAX, 5, 13)
            .unwrap_err()
            .to_string()
            .contains("reduce batch or projection")
    );
    assert!(
        plan.layout(1, usize::MAX, 13)
            .unwrap_err()
            .to_string()
            .contains("reduce projection")
    );
    let error = plan
        .project::<4, _>(
            &[0.; 15],
            3,
            5,
            13,
            None,
            &mut [0.; 39],
            &mut [0.; 39],
            &mut [],
            || -> fn(usize, &mut [f32]) -> onnx_runtime_ep_api::Result<()> {
                panic!("invalid storage must fail before decoder creation")
            },
            |_, _, _, _, _, _| panic!("invalid storage must fail before compute"),
        )
        .unwrap_err();
    assert!(error.to_string().contains("bind the matching plan"));
}

#[test]
fn parallel_gather_initializes_complete_rows_and_recovers_after_refusal() {
    use decoded_projection::ProjectionScratch;
    use onnx_runtime_ep_api::{DevicePtrMut, WorkspaceView};
    use std::{mem::MaybeUninit, num::NonZeroUsize};
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for chunk_rows in [1, 2, 8] {
        pool.install(|| {
            let mut backing = [MaybeUninit::<f32>::uninit(); 8];
            let bytes = std::mem::size_of_val(&backing);
            // SAFETY: private Host storage is exclusively lent within this call.
            let mut scratch = unsafe {
                ProjectionScratch::bind(
                    "parallel gather fixture",
                    bytes,
                    Some(WorkspaceView::new(
                        DevicePtrMut(backing.as_mut_ptr().cast()),
                        bytes,
                    )),
                )
            }
            .unwrap();
            let chunk = NonZeroUsize::new(chunk_rows).unwrap();
            let input = [0., 1., 2., 3., 4., 5., 6., 7.];
            assert!(
                scratch
                    .gather_rows_parallel(&input, 2, &[0, 1, 2, 3, 0], chunk)
                    .unwrap_err()
                    .to_string()
                    .contains("workspace has 8")
            );
            assert!(
                scratch
                    .gather_rows_parallel(&input, 2, &[0, 8, 1], chunk)
                    .unwrap_err()
                    .to_string()
                    .contains("exceeds 8 input values")
            );
            for _ in 0..3 {
                assert_eq!(
                    scratch
                        .gather_rows_parallel(&input, 2, &[3, 1, 3], chunk)
                        .unwrap(),
                    [6., 7., 2., 3., 6., 7.]
                );
            }
            assert!(
                scratch
                    .gather_rows_parallel(&[], 0, &[9], chunk)
                    .unwrap()
                    .is_empty()
            );
            assert!(
                scratch
                    .gather_rows_parallel(&input, 2, &[], chunk)
                    .unwrap()
                    .is_empty()
            );
        });
    }
}

#[test]
fn zeroed_prefix_keeps_whole_bank_initialization_separate() {
    use decoded_projection::ProjectionScratch;
    use onnx_runtime_ep_api::{DevicePtrMut, WorkspaceView};
    use std::mem::MaybeUninit;
    let mut backing = [MaybeUninit::<f32>::uninit(); 8];
    let bytes = std::mem::size_of_val(&backing);
    // SAFETY: the private Host storage remains exclusively lent to this test.
    let mut scratch = unsafe {
        ProjectionScratch::bind(
            "prefix fixture",
            bytes,
            Some(WorkspaceView::new(
                DevicePtrMut(backing.as_mut_ptr().cast()),
                bytes,
            )),
        )
    }
    .unwrap();
    assert!(scratch.zeroed_prefix(0).unwrap().is_empty());
    let prefix = scratch.zeroed_prefix(2).unwrap();
    assert_eq!(prefix, [0., 0.]);
    prefix[0] = 7.;
    assert!(
        scratch
            .zeroed_prefix(9)
            .unwrap_err()
            .to_string()
            .contains("workspace has 8")
    );
    // A partial write cannot certify untouched MaybeUninit tail storage.
    assert_eq!(scratch.initialized(), [0.; 8]);
    scratch.initialized().fill(42.);
    assert_eq!(scratch.zeroed_prefix(3).unwrap(), [0.; 3]);
    assert_eq!(scratch.initialized(), [0., 0., 0., 42., 42., 42., 42., 42.]);
}

#[test]
fn caller_destination_preserves_rows_bias_zero_axes_and_guards() {
    for (m, k, n) in [
        (0, 3, 5),
        (3, 3, 0),
        (1, 3, 5),
        (3, 3, 1),
        (3, 0, 5),
        (3, 3, 5),
    ] {
        for offset in [0, 1, 3] {
            let input: Vec<_> = (0..m * k).map(|i| i as f32 * 0.125 - 1.).collect();
            let bias: Vec<_> = (0..n).map(|i| i as f32 * 0.25 - 0.5).collect();
            let expected: Vec<_> = (0..m)
                .flat_map(|row| {
                    let input = &input;
                    let bias = &bias;
                    (0..n).map(move |feature| {
                        (0..k).fold(0., |sum, p| {
                            sum + input[row * k + p] * (feature + p + 1) as f32
                        }) + bias[feature]
                    })
                })
                .collect();
            let mut storage = vec![29.0f32; offset + m * n + 7];
            storage[offset..offset + m * n].fill(f32::NAN);
            let pointer = storage.as_ptr().wrapping_add(offset) as usize;
            let calls = AtomicUsize::new(0);
            decoded_projection::linear_decoded_grouped_into::<4, _>(
                "caller destination fixture",
                &input,
                m,
                k,
                n,
                Some(&bias),
                &mut storage[offset..offset + m * n],
                || {
                    |feature, weights: &mut [f32]| {
                        for (p, w) in weights.iter_mut().enumerate() {
                            *w = (feature + p + 1) as f32;
                        }
                        Ok(())
                    }
                },
                |input, weights, output, rows, width, features| {
                    for feature in 0..features {
                        for row in 0..rows {
                            output[feature * rows + row] = (0..width).fold(0., |sum, p| {
                                sum + input[row * width + p] * weights[feature * width + p]
                            });
                        }
                    }
                    Ok(())
                },
                |columns, rows, compute| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    if m == 1 || n == 1 {
                        assert_eq!(
                            columns.as_ptr() as usize,
                            pointer,
                            "unit axes must consume caller storage directly"
                        );
                    } else {
                        assert_ne!(
                            columns.as_ptr() as usize,
                            pointer,
                            "multirow columns need separate transpose staging"
                        );
                    }
                    decoded_projection::rayon_decoded_columns(columns, rows, compute)
                },
            )
            .unwrap();
            assert_eq!(&storage[offset..offset + m * n], expected);
            assert!(
                storage[..offset]
                    .iter()
                    .chain(&storage[offset + m * n..])
                    .all(|&x| x == 29.)
            );
            assert_eq!(calls.load(Ordering::Relaxed), usize::from(m != 0 && n != 0));
        }
    }
}

#[test]
fn invalid_caller_geometry_refuses_before_dispatch_or_destination_writes() {
    for (input, m, k, n, bias, output_len) in [
        (&[1.; 5][..], 2, 3, 3, None, 6),
        (&[1.; 6][..], 2, 3, 3, Some(&[1.; 2][..]), 6),
        (&[1.; 6][..], 2, 3, 3, None, 5),
        (&[1.; 6][..], 2, 3, 3, None, 7),
        (&[][..], usize::MAX, 0, 2, None, 0),
    ] {
        let mut output = vec![31.0; output_len + 2];
        let error = decoded_projection::linear_decoded_grouped_into::<1, _>(
            "rejected caller destination",
            input,
            m,
            k,
            n,
            bias,
            &mut output[..output_len],
            || |_, _: &mut [f32]| panic!("invalid geometry must never decode"),
            |_, _, _, _, _, _| panic!("invalid geometry must never project"),
            |_, _, _| panic!("invalid geometry must never dispatch"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("rejected caller destination"));
        assert!(error.to_string().contains("supply"), "{error}");
        assert!(output.iter().all(|&x| x == 31.));
    }
}

#[test]
fn late_decoder_failure_respects_transpose_publication_and_staging_guards() {
    for rows in [1, 2] {
        let input = vec![1.; rows];
        let mut output = vec![41.; rows * 3 + 4];
        let error = decoded_projection::linear_decoded_grouped_into::<1, _>(
            "late failure fixture",
            &input,
            rows,
            1,
            3,
            None,
            &mut output[..rows * 3],
            || {
                |feature, weights: &mut [f32]| {
                    if feature == 1 {
                        return Err(onnx_runtime_ep_api::EpError::KernelFailed(
                            "decoder channel 1 refused".into(),
                        ));
                    }
                    weights.fill(2.);
                    Ok(())
                }
            },
            |_, _, output, _, _, _| {
                output.fill(2.);
                Ok(())
            },
            |columns, _, compute| compute(0, columns),
        )
        .unwrap_err();
        assert!(error.to_string().contains("decoder channel 1 refused"));
        if rows == 1 {
            assert_eq!(
                output[0], 2.,
                "work before refusal may modify caller staging"
            );
        } else {
            assert!(
                output[..rows * 3].iter().all(|&x| x == 41.),
                "failed column staging must not transpose"
            );
        }
        assert!(output[rows * 3..].iter().all(|&x| x == 41.));
    }
}
