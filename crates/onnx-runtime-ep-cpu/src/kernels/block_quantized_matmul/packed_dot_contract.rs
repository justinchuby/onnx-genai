//! Test-only feasibility model: grouped Q8 weight/activation integer dots.
//! This is not a production backend or a ggml numeric-parity claim. In
//! particular, quantizing activations is a distinct arithmetic contract.
use super::{BlockFormat, linear_packed_grouped};
use half::f16;

const WIDTH: usize = 32;
const BYTES: usize = 34;

struct ActivationBlock {
    scale: f16,
    quants: [i8; WIDTH],
}
#[derive(Debug, Default, PartialEq, Eq)]
struct Work {
    activation_conversions: usize,
    integer_blocks: usize,
}

fn quantize_block(input: &[f32], row: usize, block: usize) -> Result<ActivationBlock, String> {
    if input.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "Q8 experiment activation row {row} block {block} contains non-finite values; use the existing f32 projection"
        ));
    }
    let maximum = input.iter().fold(0f32, |a, v| a.max(v.abs()));
    let scale = maximum / 127.;
    let stored_scale = f16::from_f32(scale);
    if !stored_scale.is_finite() || (maximum != 0. && stored_scale == f16::ZERO) {
        return Err(format!(
            "Q8 experiment activation row {row} block {block} scale {scale} is outside nonzero finite f16 storage; use the existing f32 projection"
        ));
    }
    let mut quants = [0; WIDTH];
    if maximum != 0. {
        for (quant, value) in quants.iter_mut().zip(input) {
            // The prototype's policy is explicit: nearest, ties away from zero,
            // with the original f32 scale for quantization and f16 stored scale.
            *quant = (value / scale).round().clamp(-127., 127.) as i8;
        }
    }
    Ok(ActivationBlock {
        scale: stored_scale,
        quants,
    })
}

/// Caller-owned projection lanes would replace these test-owned Vecs before
/// production adoption. No resource, scheduler, cache or fallback is modeled.
fn project(
    input: &[f32],
    weights: &[u8],
    rows: usize,
    k: usize,
    outputs: usize,
) -> Result<(Vec<f32>, Work), String> {
    if k == 0 {
        return Err("Q8 experiment requires positive K; use the existing f32 projection".into());
    }
    let blocks = k.div_ceil(WIDTH);
    let stride = blocks.checked_mul(BYTES);
    if rows.checked_mul(k) != Some(input.len())
        || stride.and_then(|stride| outputs.checked_mul(stride)) != Some(weights.len())
    {
        return Err(format!(
            "Q8 experiment geometry mismatch: input={} weights={} for rows={rows} K={k} outputs={outputs}; supply exact row-major inputs and padded Q8_0 weight blocks",
            input.len(),
            weights.len()
        ));
    }
    let count = rows
        .checked_mul(outputs)
        .ok_or("Q8 experiment output count overflow; reduce rows or outputs")?;
    let mut result = vec![0.; count];
    let mut work = Work::default();
    if rows == 0 || outputs == 0 {
        return Ok((result, work));
    }
    let mut activations = Vec::new();
    // Crucially outside the output-channel loop: each grouped input block is
    // quantized once, then reused by every channel of this projection.
    for row in 0..rows {
        for block in 0..blocks {
            let start = row * k + block * WIDTH;
            let end = (start + WIDTH).min((row + 1) * k);
            activations.push(quantize_block(&input[start..end], row, block)?);
            work.activation_conversions += 1;
        }
    }
    for row in 0..rows {
        for output in 0..outputs {
            let mut sum = 0.;
            for block in 0..blocks {
                let offset = (output * blocks + block) * BYTES;
                let weight = &weights[offset..offset + BYTES];
                let weight_scale = f16::from_le_bytes([weight[0], weight[1]]).to_f32();
                let activation = &activations[row * blocks + block];
                // Even -128 weights and -127 activations fit i32 over 32 lanes.
                let integer: i32 = weight[2..]
                    .iter()
                    .zip(activation.quants)
                    .map(|(&weight, activation)| i32::from(weight as i8) * i32::from(activation))
                    .sum();
                sum += integer as f32 * (weight_scale * activation.scale.to_f32());
                work.integer_blocks += 1;
            }
            result[row * outputs + output] = sum;
        }
    }
    Ok((result, work))
}

fn weights(k: usize, outputs: usize) -> Vec<u8> {
    let blocks = k.div_ceil(WIDTH);
    let mut bytes = vec![0; outputs * blocks * BYTES];
    for (block, bytes) in bytes.chunks_exact_mut(BYTES).enumerate() {
        bytes[..2].copy_from_slice(&f16::from_f32(0.125 * (1 + block % 4) as f32).to_le_bytes());
        for (lane, quant) in bytes[2..].iter_mut().enumerate() {
            *quant = ((block * 17 + lane * 31) % 256) as u8;
        }
    }
    bytes
}

#[test]
fn activation_policy_has_literal_codes_and_zero_padding() {
    let block = quantize_block(&[127., -127., 63.5, -63.5, 0.], 0, 0).unwrap();
    assert_eq!(block.scale, f16::ONE);
    assert_eq!(&block.quants[..5], &[127, -127, 64, -64, 0]);
    assert!(block.quants[5..].iter().all(|&q| q == 0));
    let zero = quantize_block(&[0.; WIDTH], 0, 0).unwrap();
    assert_eq!(zero.scale, f16::ZERO);
    assert_eq!(zero.quants, [0; WIDTH]);
}

#[test]
fn grouped_integer_dots_match_an_independent_f64_oracle_and_quantize_once() {
    for (rows, k, outputs) in [(1, 32, 1), (3, 65, 7), (16, 96, 9)] {
        let input: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 17 % 97) as f32 - 48.) / 29.)
            .collect();
        let weights = weights(k, outputs);
        let (actual, work) = project(&input, &weights, rows, k, outputs).unwrap();
        let blocks = k.div_ceil(WIDTH);
        assert_eq!(work.activation_conversions, rows * blocks);
        assert_eq!(work.integer_blocks, rows * outputs * blocks);
        let (_, one_channel) = project(&input, &weights[..blocks * BYTES], rows, k, 1).unwrap();
        assert_eq!(
            one_channel.activation_conversions,
            work.activation_conversions
        );
        for row in 0..rows {
            for output in 0..outputs {
                let mut expected = 0f64;
                let mut original = 0f64;
                let mut quantization_bound = 0f64;
                let mut magnitude = 0f64;
                for index in 0..k {
                    let block = index / WIDTH;
                    let lane = index % WIDTH;
                    let offset = (output * blocks + block) * BYTES;
                    let scale = f16::from_le_bytes([weights[offset], weights[offset + 1]]).to_f64();
                    let w = scale * f64::from(weights[offset + 2 + lane] as i8);
                    let slice = &input[row * k + block * WIDTH
                        ..(row * k + (block + 1) * WIDTH).min((row + 1) * k)];
                    // Independent, element-wise f64 oracle: no integer block
                    // dot, no prototype quantizer or decoder in expected math.
                    let maximum = slice.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    let step = maximum / 127.;
                    let code = if step == 0. {
                        0.
                    } else {
                        (input[row * k + index] / step).round().clamp(-127., 127.)
                    };
                    let reconstructed = f64::from(f16::from_f32(step).to_f32()) * f64::from(code);
                    expected += w * reconstructed;
                    magnitude += (w * reconstructed).abs();
                    original += w * f64::from(input[row * k + index]);
                    quantization_bound +=
                        w.abs() * (f64::from(input[row * k + index]) - reconstructed).abs();
                }
                let value = f64::from(actual[row * outputs + output]);
                let rounding_bound =
                    8. * f64::from(f32::EPSILON) * (blocks + 1) as f64 * magnitude + 1e-7;
                assert!(
                    (value - expected).abs() <= rounding_bound,
                    "{rows}x{k}x{outputs} [{row},{output}]: {value} != {expected}"
                );
                assert!((value - original).abs() <= quantization_bound + rounding_bound);
            }
        }
    }
}

#[test]
fn activation_quantization_can_flip_the_winner_of_the_current_f32_projection() {
    let mut input = [0.; WIDTH];
    input[0] = 1.;
    input[1] = 1. / 1024.;
    let mut packed = [0u8; 2 * BYTES];
    packed[..2].copy_from_slice(&f16::ONE.to_le_bytes());
    packed[3] = 1; // output 0 selects the small activation.
    packed[BYTES..BYTES + 2].copy_from_slice(&f16::from_f32(1. / 2048.).to_le_bytes());
    packed[BYTES + 2] = 1; // output 1 selects the large activation at a smaller scale.
    let original =
        linear_packed_grouped(BlockFormat::Q8_0, &packed, &input, 1, WIDTH, 2, None).unwrap();
    let (quantized, _) = project(&input, &packed, 1, WIDTH, 2).unwrap();
    assert_eq!(original, [1. / 1024., 1. / 2048.]);
    assert!(original[0] > original[1]);
    assert_eq!(quantized[0], 0.);
    assert!(quantized[1] > quantized[0]);
}

#[test]
fn unsupported_activation_ranges_report_the_row_and_block_without_changing_default_math() {
    for value in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN_POSITIVE,
    ] {
        let message = project(&[value], &weights(1, 1), 1, 1, 1).unwrap_err();
        assert!(
            message.contains("row 0 block 0") && message.contains("existing f32 projection"),
            "{message}"
        );
    }
    assert!(
        project(&[], &[], 1, 0, 2)
            .unwrap_err()
            .contains("positive K")
    );
    let message = project(&[1.], &[], 2, 1, 1).unwrap_err();
    assert!(message.contains("geometry mismatch"), "{message}");
    assert_eq!(
        project(&[], &weights(33, 2), 0, 33, 2).unwrap(),
        (vec![], Work::default())
    );
    assert_eq!(
        project(&[f32::NAN], &[], 1, 1, 0).unwrap(),
        (vec![], Work::default())
    );
}
