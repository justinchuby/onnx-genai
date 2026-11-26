//! Test-only loop-order experiment. Production continues to use project_q8.
use super::*;
#[cfg(target_os = "macos")]
use std::hint::black_box;
#[cfg(target_os = "macos")]
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
fn block_first(
    format: BlockFormat,
    packed: &[u8],
    input: &[f32],
    rows: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    scratch: &mut ProjectionScratch<'_>,
    output: &mut [f32],
) -> Result<()> {
    let format = DotFormat::from_format(format)?;
    let blocks = k.div_ceil(QK);
    let stride = blocks
        .checked_mul(format.bytes())
        .ok_or_else(|| error("q8_0 packed weight row byte count overflows; reduce K"))?;
    if rows.checked_mul(k) != Some(input.len())
        || n.checked_mul(stride) != Some(packed.len())
        || rows.checked_mul(n) != Some(output.len())
        || bias.is_some_and(|b| b.len() != n)
    {
        return Err(error(format!(
            "q8_0 projection storage does not match rows={rows}, K={k}, N={n}; supply exact activation, packed weight, output and bias lengths"
        )));
    }
    let (column_count, activation_bytes) = q8_geometry(rows, k, n)?;
    let (column_major, activation_storage) =
        scratch.split_initialized_bytes(column_count, activation_bytes)?;
    if output.is_empty() {
        return Ok(());
    }
    let activations = prepare_activations(input, rows, k, activation_storage)?;
    let transpose = rows > 1 && n > 1;
    let columns = if transpose {
        &mut *column_major
    } else {
        &mut *output
    };
    rayon_decoded_columns(columns, rows, &|first, columns| {
        for (local, values) in columns.chunks_exact_mut(rows).enumerate() {
            let feature = first + local;
            values.fill(0.);
            for block in 0..blocks {
                let weight = &packed[feature * stride + block * format.bytes()
                    ..feature * stride + (block + 1) * format.bytes()];
                let scale = half::f16::from_le_bytes([weight[0], weight[1]]).to_f32();
                if !scale.is_finite() {
                    return Err(error(format!(
                        "q8_0 projection weight output {feature} block {block} has non-finite scale; select activation_arithmetic='float32' with block_layout_version=1"
                    )));
                }
                for (row, value) in values.iter_mut().enumerate() {
                    let activation = &activations[(row * blocks + block) * ACTIVATION_BYTES
                        ..(row * blocks + block + 1) * ACTIVATION_BYTES];
                    let activation_scale =
                        half::f16::from_le_bytes([activation[0], activation[1]]).to_f32();
                    *value += format.dot(weight, activation) as f32 * (scale * activation_scale);
                }
            }
            for value in values {
                *value += bias.map_or(0., |bias| bias[feature]);
            }
        }
        Ok(())
    })?;
    if transpose {
        for (row, target) in output.chunks_exact_mut(n).enumerate() {
            for (feature, value) in target.iter_mut().enumerate() {
                *value = column_major[feature * rows + row];
            }
        }
    }
    onnx_runtime_ep_api::record_kernel_variant!(
        "packed_projection.q8_0",
        "grouped Q8 activation integer dots: rows={rows}, K={k}, N={n}"
    );
    Ok(())
}

struct Fixture {
    format: BlockFormat,
    rows: usize,
    k: usize,
    n: usize,
    x: Vec<f32>,
    packed: Vec<u8>,
    bias: Vec<f32>,
    oracle: Vec<f32>,
}
impl Fixture {
    fn new(q4: bool, rows: usize, k: usize, n: usize) -> Self {
        let blocks = k.div_ceil(32);
        let width = if q4 { 18 } else { 34 };
        let format = if q4 {
            BlockFormat::Onnx(onnx_runtime_ir::dequantize_packed::Format::Q4_0)
        } else {
            BlockFormat::Q8_0
        };
        let mut x = vec![0.; rows * k];
        for row in 0..rows {
            for i in 0..k {
                let block = i / 32;
                let code = if (row + block + 1).is_multiple_of(5) {
                    0
                } else if i.is_multiple_of(32) {
                    127
                } else {
                    ((row * 17 + block * 11 + (i % 32) * 13) % 255) as i32 - 127
                };
                x[row * k + i] = code as f32 * 2f32.powi(((row + block) % 3) as i32 - 2);
            }
        }
        let code = |feature: usize, block: usize, lane: usize| {
            ((feature * 7 + block * 11 + lane * 3) % if q4 { 16 } else { 9 }) as i32
                - if q4 { 8 } else { 4 }
        };
        let scale = |feature: usize, block: usize| [0., 0.125, 0.25, -0.5][(feature + block) % 4];
        let mut packed = vec![0; n * blocks * width];
        for feature in 0..n {
            for block in 0..blocks {
                let target = &mut packed[(feature * blocks + block) * width..][..width];
                target[..2]
                    .copy_from_slice(&half::f16::from_f32(scale(feature, block)).to_le_bytes());
                if q4 {
                    for lane in 0..16 {
                        target[2 + lane] = ((code(feature, block, lane) + 8) as u8)
                            | (((code(feature, block, lane + 16) + 8) as u8) << 4);
                    }
                } else {
                    for lane in 0..32 {
                        target[2 + lane] = code(feature, block, lane) as i8 as u8;
                    }
                }
            }
        }
        let bias = (0..n)
            .map(|f| (f % 7) as f32 / 16. - 0.1875)
            .collect::<Vec<_>>();
        // Independent element-wise oracle: no activation converter, serialized
        // block decoder or production integer dot. These dyadic fixtures have
        // exact activation reconstruction and exact accumulated f32 results.
        let oracle = (0..rows)
            .flat_map(|row| (0..n).map(move |feature| (row, feature)))
            .map(|(row, feature)| {
                let value = (0..k)
                    .map(|i| {
                        f64::from(x[row * k + i])
                            * f64::from(scale(feature, i / 32))
                            * f64::from(code(feature, i / 32, i % 32))
                    })
                    .sum::<f64>();
                (value + f64::from(bias[feature])) as f32
            })
            .collect();
        Self {
            format,
            rows,
            k,
            n,
            x,
            packed,
            bias,
            oracle,
        }
    }
    fn storage(&self) -> Vec<f32> {
        vec![
            f32::NAN;
            PackedProjectionPlan::Q8_0
                .layout(self.rows, self.k, self.n)
                .unwrap()
                .bytes()
                / 4
        ]
    }
    fn check(&self, actual: &[f32]) -> u64 {
        assert_eq!(actual.len(), self.oracle.len());
        let mut digest = 0xcbf29ce484222325u64;
        for (index, (&a, &b)) in actual.iter().zip(&self.oracle).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "oracle mismatch at {index}: {a} vs {b}"
            );
            digest = (digest ^ u64::from(a.to_bits())).wrapping_mul(0x100000001b3);
        }
        digest
    }
}
type Projection = fn(
    BlockFormat,
    &[u8],
    &[f32],
    usize,
    usize,
    usize,
    Option<&[f32]>,
    &mut ProjectionScratch<'_>,
    &mut [f32],
) -> Result<()>;

#[test]
fn loop_orders_preserve_oracle_bits_dirty_reuse_and_scale_errors() {
    for q4 in [false, true] {
        for rows in [0, 1, 3, 8, 32] {
            for k in [0, 1, 31, 32, 33, 1003, 1024] {
                for n in [0, 1, 7] {
                    let mut f = Fixture::new(q4, rows, k, n);
                    for mode in 0..3 {
                        let bias = mode != 0;
                        for (feature, value) in f.bias.iter_mut().enumerate() {
                            *value = if mode == 1 {
                                -0.
                            } else {
                                (feature % 7) as f32 / 16. - 0.1875
                            };
                        }
                        let expected = (0..rows)
                            .flat_map(|row| (0..n).map(move |feature| (row, feature)))
                            .map(|(row, feature)| {
                                // Remove the original dyadic bias from the oracle exactly.
                                let original = (feature % 7) as f32 / 16. - 0.1875;
                                f.oracle[row * n + feature] - original
                                    + if bias { f.bias[feature] } else { 0. }
                            })
                            .collect::<Vec<_>>();
                        for apply in [project_q8 as Projection, block_first] {
                            let mut storage = f.storage();
                            let mut scratch = ProjectionScratch::test_binding(&mut storage);
                            let mut output = vec![f32::NAN; rows * n];
                            for _ in 0..2 {
                                output.fill(f32::NAN);
                                apply(
                                    f.format,
                                    &f.packed,
                                    &f.x,
                                    rows,
                                    k,
                                    n,
                                    if bias { Some(&f.bias) } else { None },
                                    &mut scratch,
                                    &mut output,
                                )
                                .unwrap();
                                assert_eq!(
                                    output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                    "q4={q4} rows={rows} K={k} N={n} bias={bias}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    for q4 in [false, true] {
        let mut f = Fixture::new(q4, 3, 33, 7);
        let width = if q4 { 18 } else { 34 };
        let last = f.packed.len() - width;
        f.packed[last..last + 2].copy_from_slice(&half::f16::INFINITY.to_le_bytes());
        for apply in [project_q8 as Projection, block_first] {
            let mut storage = f.storage();
            let mut output = vec![37.; f.rows * f.n];
            let error = apply(
                f.format,
                &f.packed,
                &f.x,
                f.rows,
                f.k,
                f.n,
                Some(&f.bias),
                &mut ProjectionScratch::test_binding(&mut storage),
                &mut output,
            )
            .unwrap_err();
            assert!(
                error.to_string().contains("weight output 6 block 1"),
                "{error}"
            );
        }
    }
}
#[cfg(target_os = "macos")]
fn cpu_seconds() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: valid writable rusage storage; success initializes it.
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
        0
    );
    let usage = unsafe { usage.assume_init() };
    usage.ru_utime.tv_sec as f64
        + usage.ru_utime.tv_usec as f64 / 1e6
        + usage.ru_stime.tv_sec as f64
        + usage.ru_stime.tv_usec as f64 / 1e6
}
#[cfg(target_os = "macos")]
fn run(
    f: &Fixture,
    apply: Projection,
    calls: usize,
    work: usize,
    storage: &mut [f32],
    output: &mut [f32],
) -> (f64, f64) {
    let before = cpu_seconds();
    let start = Instant::now();
    // Synchronous window: fresh checked scratch binding and the complete Q8
    // projection (activation conversion, dots and transpose), without allocation,
    // fixture/oracle work, routing, final tensor publication or quota admission.
    for _ in 0..calls {
        if work == 0 {
            let mut scratch = ProjectionScratch::test_binding(black_box(&mut *storage));
            black_box(&mut scratch);
        } else {
            for _ in 0..work {
                let mut scratch = ProjectionScratch::test_binding(black_box(&mut *storage));
                apply(
                    f.format,
                    black_box(&f.packed),
                    black_box(&f.x),
                    f.rows,
                    f.k,
                    f.n,
                    Some(&f.bias),
                    &mut scratch,
                    black_box(&mut *output),
                )
                .unwrap();
            }
        }
        black_box(&*output);
    }
    let wall = start.elapsed().as_secs_f64();
    (wall, cpu_seconds() - before)
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "bounded research process; run only through the locked study driver"]
fn paired_loop_order_process() {
    let get = |name: &str| std::env::var(name).unwrap();
    let rows = get("NXRT_LOOP_ROWS").parse::<usize>().unwrap();
    let k = get("NXRT_LOOP_K").parse::<usize>().unwrap();
    let format = get("NXRT_LOOP_FORMAT");
    assert!(
        matches!(format.as_str(), "q4" | "q8"),
        "study format must be q4 or q8; use the frozen study driver"
    );
    let q4 = format == "q4";
    let calls = get("NXRT_LOOP_CALLS").parse::<usize>().unwrap();
    let reverse = get("NXRT_LOOP_REVERSE") == "1";
    assert!(
        [1, 3, 8, 32].contains(&rows) && [32, 1024].contains(&k) && calls <= 1_000_000,
        "study requires frozen R/K/call bounds; use the study driver"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        assert_eq!(rayon::current_num_threads(), 1, "study must execute in its established one-worker pool");
        let f=Fixture::new(q4,rows,k,256);
        let mut storage=f.storage();
        let mut output=vec![f32::NAN;rows*256];
        for apply in [project_q8 as Projection,block_first] {
            run(&f,black_box(apply),1,1,&mut storage,&mut output);
            f.check(&output);
        }
        if calls==0 {
            let mut amount=1;
            loop {
                let (wall,_)=run(&f,black_box(project_q8 as Projection),amount,1,&mut storage,&mut output);
                if wall>=0.05 {
                    let calls=((amount as f64*0.2/wall).ceil() as usize).max(1);
                    println!("NXRT_LOOP_CALIBRATION {{\"calls\":{calls},\"pilot_calls\":{amount},\"pilot_wall\":{wall},\"digest\":\"{:016x}\"}}",f.check(&output));
                    break;
                }
                amount*=2;
            }
            return;
        }
        let a=("row",project_q8 as Projection,1);
        let b=("block",block_first as Projection,1);
        let paired=if reverse {[b,a,a,b]}else{[a,b,b,a]};
        let controls = [
            ("aa1", project_q8 as Projection, 1),
            ("aa2", project_q8 as Projection, 1),
            ("baa1", block_first as Projection, 1),
            ("baa2", block_first as Projection, 1),
            ("setup", project_q8 as Projection, 0),
            ("double", project_q8 as Projection, 2),
            ("bdouble", block_first as Projection, 2),
        ];
        for (label,apply,work) in paired.into_iter().chain(controls) {
            let (wall,cpu)=run(&f,black_box(apply),calls,work,&mut storage,&mut output);
            let digest=f.check(&output);
            println!("NXRT_LOOP_WINDOW {{\"label\":\"{label}\",\"calls\":{calls},\"projections\":{},\"wall_s\":{wall},\"cpu_s\":{cpu},\"digest\":\"{digest:016x}\"}}",calls*work);
        }
    });
}
