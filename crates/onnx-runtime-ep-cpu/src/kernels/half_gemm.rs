//! Portable blocked `f16`/`bf16` GEMM with `f32` accumulation.
//!
//! Operands stay in their 16-bit storage format until they are packed into
//! cache-sized `f32` panels. The shared register-tiled kernel then accumulates
//! in `f32`; callers perform one final narrowing step to the requested output
//! dtype. This is the portable correctness path on every CPU, including
//! AVX2-only x86-64 and aarch64 hosts.

use rayon::prelude::*;

const MR: usize = 4;
const NR: usize = 4;
const KC: usize = 128;
const NC: usize = 64;
const MAX_MC: usize = 64;

/// The 16-bit floating-point storage format used by a GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HalfFormat {
    F16,
    Bf16,
}

/// Logical matrix strides, measured in 16-bit elements.
///
/// A contiguous row-major matrix uses `(columns, 1)`. Swapping the two strides
/// provides a transposed logical view without materializing another buffer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatrixLayout {
    pub(crate) row_stride: usize,
    pub(crate) column_stride: usize,
}

impl MatrixLayout {
    pub(crate) const fn row_major(columns: usize) -> Self {
        Self {
            row_stride: columns,
            column_stride: 1,
        }
    }

    pub(crate) const fn transposed(stored_columns: usize) -> Self {
        Self {
            row_stride: 1,
            column_stride: stored_columns,
        }
    }
}

trait HalfElement: Send + Sync {
    fn to_f32(bits: u16) -> f32;
}

struct F16;

impl HalfElement for F16 {
    #[inline]
    fn to_f32(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }
}

struct Bf16;

impl HalfElement for Bf16 {
    #[inline]
    fn to_f32(bits: u16) -> f32 {
        half::bf16::from_bits(bits).to_f32()
    }
}

/// Compute `c[m,n] = a[m,k] @ b[k,n]` with `f32` accumulation.
///
/// `a` and `b` contain raw `f16` or `bf16` bit patterns selected by `format`.
/// Their logical layouts may be row-major or transposed. `c` is overwritten.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    format: HalfFormat,
    a: &[u16],
    a_layout: MatrixLayout,
    b: &[u16],
    b_layout: MatrixLayout,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(c.len(), m * n);
    match format {
        HalfFormat::F16 => gemm_impl::<F16>(a, a_layout, b, b_layout, c, m, k, n),
        HalfFormat::Bf16 => gemm_impl::<Bf16>(a, a_layout, b, b_layout, c, m, k, n),
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_impl<T: HalfElement>(
    a: &[u16],
    a_layout: MatrixLayout,
    b: &[u16],
    b_layout: MatrixLayout,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    c.fill(0.0);

    let threads = rayon::current_num_threads();
    let mc = if threads <= 1 {
        MAX_MC.min(m)
    } else {
        let rows = m.div_ceil(threads.saturating_mul(2)).clamp(1, MAX_MC);
        if rows == 1 {
            1
        } else {
            rows.div_ceil(MR).saturating_mul(MR).min(MAX_MC)
        }
    };

    c.par_chunks_mut(mc * n)
        .enumerate()
        .for_each(|(block_index, c_block)| {
            let first_row = block_index * mc;
            let rows = c_block.len() / n;
            gemm_block::<T>(a, a_layout, b, b_layout, c_block, first_row, rows, k, n);
        });
}

#[allow(clippy::too_many_arguments)]
fn gemm_block<T: HalfElement>(
    a: &[u16],
    a_layout: MatrixLayout,
    b: &[u16],
    b_layout: MatrixLayout,
    c: &mut [f32],
    first_row: usize,
    rows: usize,
    k: usize,
    n: usize,
) {
    let mut a_panel = Vec::with_capacity(rows * KC);
    let mut b_panel = Vec::with_capacity(KC * NC);

    for depth_start in (0..k).step_by(KC) {
        let panel_depth = KC.min(k - depth_start);
        pack_a::<T>(
            a,
            a_layout,
            first_row,
            rows,
            depth_start,
            panel_depth,
            &mut a_panel,
        );

        for column_start in (0..n).step_by(NC) {
            let panel_columns = NC.min(n - column_start);
            pack_b::<T>(
                b,
                b_layout,
                depth_start,
                panel_depth,
                column_start,
                panel_columns,
                &mut b_panel,
            );

            for row_start in (0..rows).step_by(MR) {
                let tile_rows = MR.min(rows - row_start);
                for panel_column_start in (0..panel_columns).step_by(NR) {
                    let tile_columns = NR.min(panel_columns - panel_column_start);
                    micro_kernel(
                        &a_panel,
                        &b_panel,
                        c,
                        n,
                        row_start,
                        panel_column_start,
                        column_start,
                        panel_depth,
                        panel_columns,
                        tile_rows,
                        tile_columns,
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_a<T: HalfElement>(
    source: &[u16],
    layout: MatrixLayout,
    first_row: usize,
    rows: usize,
    depth_start: usize,
    panel_depth: usize,
    packed: &mut Vec<f32>,
) {
    packed.clear();
    packed.resize(rows * panel_depth, 0.0);
    for row in 0..rows {
        for depth in 0..panel_depth {
            let source_index = (first_row + row) * layout.row_stride
                + (depth_start + depth) * layout.column_stride;
            packed[row * panel_depth + depth] = T::to_f32(source[source_index]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_b<T: HalfElement>(
    source: &[u16],
    layout: MatrixLayout,
    depth_start: usize,
    panel_depth: usize,
    column_start: usize,
    panel_columns: usize,
    packed: &mut Vec<f32>,
) {
    packed.clear();
    packed.resize(panel_depth * panel_columns, 0.0);
    for depth in 0..panel_depth {
        for column in 0..panel_columns {
            let source_index = (depth_start + depth) * layout.row_stride
                + (column_start + column) * layout.column_stride;
            packed[depth * panel_columns + column] = T::to_f32(source[source_index]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn micro_kernel(
    a_panel: &[f32],
    b_panel: &[f32],
    c: &mut [f32],
    c_columns: usize,
    row_start: usize,
    panel_column_start: usize,
    column_start: usize,
    panel_depth: usize,
    panel_columns: usize,
    tile_rows: usize,
    tile_columns: usize,
) {
    let mut accumulators = [[0.0f32; NR]; MR];
    for depth in 0..panel_depth {
        let b_row = &b_panel[depth * panel_columns + panel_column_start
            ..depth * panel_columns + panel_column_start + tile_columns];
        for (tile_row, accumulator_row) in accumulators.iter_mut().enumerate().take(tile_rows) {
            let a_value = a_panel[tile_row.saturating_add(row_start) * panel_depth + depth];
            for (tile_column, accumulator) in
                accumulator_row.iter_mut().enumerate().take(tile_columns)
            {
                *accumulator += a_value * b_row[tile_column];
            }
        }
    }

    for (tile_row, accumulator_row) in accumulators.iter().enumerate().take(tile_rows) {
        let output_start = (row_start + tile_row) * c_columns + column_start + panel_column_start;
        let output_row = &mut c[output_start..output_start + tile_columns];
        for (output, accumulator) in output_row.iter_mut().zip(accumulator_row) {
            *output += accumulator;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut output = vec![0.0; m * n];
        for row in 0..m {
            for column in 0..n {
                for depth in 0..k {
                    output[row * n + column] += a[row * k + depth] * b[depth * n + column];
                }
            }
        }
        output
    }

    #[test]
    fn portable_half_gemm_matches_widened_f32_reference_and_is_deterministic() {
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 127, 65),
            (3, 5, 7),
            (17, 130, 11),
            (5, 257, 70),
            (33, 9, 2),
        ];
        let mut state = 0xC001_D00D_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.5
        };

        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            for &(m, k, n) in SHAPES {
                let a_source: Vec<f32> = (0..m * k).map(|_| next()).collect();
                let b_source: Vec<f32> = (0..k * n).map(|_| next()).collect();
                let (a_bits, a_wide): (Vec<_>, Vec<_>) = a_source
                    .iter()
                    .map(|&value| match format {
                        HalfFormat::F16 => {
                            let value = half::f16::from_f32(value);
                            (value.to_bits(), value.to_f32())
                        }
                        HalfFormat::Bf16 => {
                            let value = half::bf16::from_f32(value);
                            (value.to_bits(), value.to_f32())
                        }
                    })
                    .unzip();
                let (b_bits, b_wide): (Vec<_>, Vec<_>) = b_source
                    .iter()
                    .map(|&value| match format {
                        HalfFormat::F16 => {
                            let value = half::f16::from_f32(value);
                            (value.to_bits(), value.to_f32())
                        }
                        HalfFormat::Bf16 => {
                            let value = half::bf16::from_f32(value);
                            (value.to_bits(), value.to_f32())
                        }
                    })
                    .unzip();
                let expected = reference(&a_wide, &b_wide, m, k, n);
                let mut first = vec![0.0; m * n];
                let mut second = vec![0.0; m * n];
                gemm(
                    format,
                    &a_bits,
                    MatrixLayout::row_major(k),
                    &b_bits,
                    MatrixLayout::row_major(n),
                    &mut first,
                    m,
                    k,
                    n,
                );
                gemm(
                    format,
                    &a_bits,
                    MatrixLayout::row_major(k),
                    &b_bits,
                    MatrixLayout::row_major(n),
                    &mut second,
                    m,
                    k,
                    n,
                );

                assert_eq!(
                    first, second,
                    "{format:?} {m}x{k}x{n} was not deterministic"
                );
                let max_error = first
                    .iter()
                    .zip(expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_error <= 2e-5,
                    "{format:?} {m}x{k}x{n}: max f32 accumulation error {max_error}"
                );
            }
        }
    }

    #[test]
    fn portable_half_gemm_supports_transposed_layouts() {
        let logical_a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let logical_b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let stored_a = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        let stored_b = [7.0f32, 9.0, 11.0, 8.0, 10.0, 12.0];
        let expected = reference(&logical_a, &logical_b, 2, 3, 2);

        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            let convert = |value: f32| match format {
                HalfFormat::F16 => half::f16::from_f32(value).to_bits(),
                HalfFormat::Bf16 => half::bf16::from_f32(value).to_bits(),
            };
            let a: Vec<u16> = stored_a.iter().copied().map(convert).collect();
            let b: Vec<u16> = stored_b.iter().copied().map(convert).collect();
            let mut actual = vec![0.0; 4];
            gemm(
                format,
                &a,
                MatrixLayout::transposed(2),
                &b,
                MatrixLayout::transposed(3),
                &mut actual,
                2,
                3,
                2,
            );
            assert_eq!(actual, expected);
        }
    }
}
