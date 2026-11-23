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
            let actual = pool.install(|| {
                decoded_projection::linear_decoded_grouped::<4, _>(
                    "tile fixture",
                    &input,
                    m,
                    k,
                    n,
                    Some(&bias),
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
    let result = decoded_projection::linear_decoded_grouped::<4, _>(
        "malformed tile fixture",
        &[1.; 6],
        2,
        3,
        3,
        None,
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
