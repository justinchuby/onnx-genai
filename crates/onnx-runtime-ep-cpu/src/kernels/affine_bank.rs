//! Borrowed affine expert-bank storage. No decoded bank or parameter payload.

use crate::dtype::NumericElem;
use onnx_runtime_ep_api::{EpError, Result, TensorMetadata, TensorView};
use onnx_runtime_ir::DataType;
use std::marker::PhantomData;

fn error(detail: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("Affine expert bank: {detail}"))
}

struct Storage<'a, T> {
    origin: *const T,
    strides: [isize; 3],
    width: usize,
    columns: usize,
    rows: usize,
    dense: bool,
    source: PhantomData<&'a T>,
}
impl<'a, T: NumericElem> Storage<'a, T> {
    fn new(name: &str, view: &'a TensorView<'_>) -> Result<Self> {
        let count = crate::strided::validate_host_storage::<T>("Affine expert bank", name, view)?;
        let &[experts, width, columns] = view.shape else {
            return Err(error(format!(
                "{name} shape {:?} must have rank three [experts, rows, columns]",
                view.shape
            )));
        };
        let rows = experts.checked_mul(width).ok_or_else(|| {
            error(format!(
                "{name} row count overflows; reduce experts or projection width"
            ))
        })?;
        // The span proof bounds every active stride to isize; unused strides
        // on singleton/empty axes need not be representable.
        let strides = std::array::from_fn(|axis| {
            if count == 0 || view.shape[axis] == 1 {
                0
            } else {
                view.strides[axis] as isize
            }
        });
        Ok(Self {
            origin: view.data_ptr::<T>(),
            strides,
            width,
            columns,
            rows,
            dense: view.is_contiguous(),
            source: PhantomData,
        })
    }
    fn row(&self, row: usize) -> Row<'_, T> {
        debug_assert!(row < self.rows);
        // SAFETY: caller bounds row once per projection; construction proves all
        // element/byte offsets fit. Owner supplies backing bounds and lifetime.
        let origin = unsafe {
            if self.dense {
                self.origin.add(row * self.columns)
            } else {
                self.origin.offset(
                    (row / self.width) as isize * self.strides[0]
                        + (row % self.width) as isize * self.strides[1],
                )
            }
        };
        Row {
            origin,
            step: if self.dense { 1 } else { self.strides[2] },
            columns: self.columns,
            source: PhantomData,
        }
    }
}
struct Row<'a, T> {
    origin: *const T,
    step: isize,
    columns: usize,
    source: PhantomData<&'a T>,
}
impl<T: Copy> Row<'_, T> {
    #[inline]
    fn get<const DENSE: bool>(&self, column: usize) -> T {
        debug_assert!(column < self.columns);
        // SAFETY: column is bounded by the validated bank format; the row's
        // complete stride range was checked before any reads. Packed origins
        // may be unaligned. No typed shared reference to tensor data is formed.
        unsafe {
            if DENSE {
                self.origin.add(column).read_unaligned()
            } else {
                self.origin
                    .offset(column as isize * self.step)
                    .read_unaligned()
            }
        }
    }
}

fn invalid_scale(value: f32) -> bool {
    value <= 0.0 || !value.is_finite()
}
impl Storage<'_, f32> {
    fn first_invalid_scale(&self) -> Option<(usize, usize, f32)> {
        if self.columns == 0 {
            return None;
        }
        if self.dense && self.origin.is_aligned() {
            // SAFETY: validated contiguous aligned Host f32 storage with a
            // checked complete extent. This temporary borrow ends before any
            // output publication; no reference is retained in the bank.
            let values =
                unsafe { std::slice::from_raw_parts(self.origin, self.rows * self.columns) };
            values
                .iter()
                .copied()
                .position(invalid_scale)
                .map(|i| (i / self.columns, i % self.columns, values[i]))
        } else {
            for row in 0..self.rows {
                let values = self.row(row);
                for column in 0..self.columns {
                    let value = values.get::<false>(column);
                    if invalid_scale(value) {
                        return Some((row, column, value));
                    }
                }
            }
            None
        }
    }
}

pub(super) fn validate_metadata(
    codes: &TensorMetadata<'_>,
    scales: &TensorMetadata<'_>,
    zeros: Option<&TensorMetadata<'_>>,
    [experts, width, depth]: [usize; 3],
    block: usize,
    zero: u8,
) -> Result<()> {
    if block == 0 || !depth.is_multiple_of(block) || zero > 15 {
        return Err(error(format!(
            "block {block} must divide depth {depth} and zero point {zero} must fit UINT4; supply a valid affine format"
        )));
    }
    let blocks = depth / block;
    if !codes.present
        || !scales.present
        || codes.dtype != DataType::Uint8
        || codes.shape != [experts, width, depth.div_ceil(2)]
        || scales.dtype != DataType::Float32
        || scales.shape != [experts, width, blocks]
        || zeros.is_some_and(|z| {
            z.dtype != DataType::Uint8 || z.shape != [experts, width, blocks.div_ceil(2)]
        })
    {
        return Err(error(format!(
            "codes {:?} {:?}, scales {:?} {:?}, zeros {:?} do not match [experts, width, depth]=[{experts},{width},{depth}], block={block}; bind matching packed bytes and FLOAT scales",
            codes.dtype,
            codes.shape,
            scales.dtype,
            scales.shape,
            zeros.map(|z| (z.dtype, z.shape))
        )));
    }
    let rows = experts
        .checked_mul(width)
        .ok_or_else(|| error("expert row count overflows; reduce bank dimensions"))?;
    for (name, columns, width) in [
        ("codes", depth.div_ceil(2), 1usize),
        ("scales", depth / block, 4),
        ("zero points", (depth / block).div_ceil(2), 1),
    ]
    .into_iter()
    .take(if zeros.is_some() { 3 } else { 2 })
    {
        let count = rows
            .checked_mul(columns)
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| error(format!("{name} size overflows; reduce bank dimensions")))?;
        std::alloc::Layout::from_size_align(count, width).map_err(|_| {
            error(format!(
                "{name} exceeds addressable capacity; reduce bank dimensions"
            ))
        })?;
    }
    Ok(())
}

pub(super) struct PreparedAffineBank<'a>(AffineBank<'a>);
impl<'a> PreparedAffineBank<'a> {
    pub(super) fn validate(self) -> Result<AffineBank<'a>> {
        if let Some((row, block, value)) = self.0.scales.first_invalid_scale() {
            return Err(error(format!(
                "scales row {row}, block {block} is {value}; DequantizeLinear scales must be positive and finite; bind a valid scale bank"
            )));
        }
        Ok(self.0)
    }
}

pub(super) struct AffineBank<'a> {
    codes: Storage<'a, u8>,
    scales: Storage<'a, f32>,
    zeros: Option<Storage<'a, u8>>,
    depth: usize,
    block: usize,
    zero: u8,
}
impl<'a> AffineBank<'a> {
    /// Validate metadata and Host address proofs without reading scale values.
    pub(super) fn prepare(
        codes: &'a TensorView<'_>,
        scales: &'a TensorView<'_>,
        zeros: Option<&'a TensorView<'_>>,
        dimensions: [usize; 3],
        block: usize,
        zero: u8,
    ) -> Result<PreparedAffineBank<'a>> {
        let metadata =
            |v: &'a TensorView<'_>| TensorMetadata::new(v.dtype, v.shape, !v.is_absent());
        let c = metadata(codes);
        let s = metadata(scales);
        let z = zeros.map(metadata);
        validate_metadata(&c, &s, z.as_ref(), dimensions, block, zero)?;
        let depth = dimensions[2];
        // Validate every source before reading any scale (or packed byte).
        let codes = Storage::new("codes", codes)?;
        let scales = Storage::<f32>::new("scales", scales)?;
        let zeros = zeros.map(|z| Storage::new("zero points", z)).transpose()?;
        Ok(PreparedAffineBank(Self {
            codes,
            scales,
            zeros,
            depth,
            block,
            zero,
        }))
    }
    pub(super) fn dot(&self, row: usize, input: &[f32]) -> f32 {
        assert!(
            row < self.codes.rows,
            "affine projection row {row} exceeds {} validated rows; check caller route geometry",
            self.codes.rows
        );
        assert_eq!(
            input.len(),
            self.depth,
            "affine projection input must match validated depth"
        );
        if self.depth == 0 {
            return 0.;
        }
        // Resolve expert/row addressing once, outside the per-weight loop.
        let codes = self.codes.row(row);
        let scales = self.scales.row(row);
        let zeros = self.zeros.as_ref().map(|z| z.row(row));
        if self.codes.dense && self.scales.dense && self.zeros.as_ref().is_none_or(|z| z.dense) {
            self.dot_rows::<true>(codes, scales, zeros, input)
        } else {
            self.dot_rows::<false>(codes, scales, zeros, input)
        }
    }
    // Two static address modes share one arithmetic body. The common dense
    // route adds no dynamic stride multiplication or branch per weight.
    fn dot_rows<const DENSE: bool>(
        &self,
        codes: Row<'_, u8>,
        scales: Row<'_, f32>,
        zeros: Option<Row<'_, u8>>,
        input: &[f32],
    ) -> f32 {
        let mut sum = 0.0;
        // Metadata proves that block is positive and divides depth. Scale and
        // zero point are constant within a block; resolve them once rather than
        // dividing/reading parameters for every weight. Preserve each column's
        // rounded weight, multiplication and accumulation in its original order.
        for (p, values) in input.chunks_exact(self.block).enumerate() {
            let scale = scales.get::<DENSE>(p);
            let zero = zeros
                .as_ref()
                .map_or(self.zero, |z| (z.get::<DENSE>(p / 2) >> ((p % 2) * 4)) & 15);
            let start = p * self.block;
            for (offset, &x) in values.iter().enumerate() {
                let col = start + offset;
                let q = (codes.get::<DENSE>(col / 2) >> ((col % 2) * 4)) & 15;
                let weight = f32::from(i16::from(q) - i16::from(zero)) * scale;
                sum += x * weight;
            }
        }
        sum
    }
}

#[cfg(test)]
mod tests;
