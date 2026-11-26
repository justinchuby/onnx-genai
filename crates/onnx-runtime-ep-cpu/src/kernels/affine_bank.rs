//! Borrowed affine expert-bank storage. No decoded bank or parameter payload.

use crate::dtype::NumericElem;
use onnx_runtime_ep_api::{EpError, Result, TensorView};
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
        view.validate().map_err(|e| {
            error(format!(
                "invalid {name}: {e}; bind valid rank-three Host storage"
            ))
        })?;
        if !view.device.is_host_accessible() || view.dtype != T::DTYPE {
            return Err(error(format!(
                "{name} must be Host-accessible {:?}, got {:?} on {:?}; explicitly transfer and bind the matching storage type",
                T::DTYPE,
                view.dtype,
                view.device
            )));
        }
        let &[experts, width, columns] = view.shape else {
            return Err(error(format!(
                "{name} shape {:?} must have rank three [experts, rows, columns]",
                view.shape
            )));
        };
        let invalid = || {
            error(format!(
                "{name} shape {:?}, strides {:?}, byte_offset {} has an unaddressable span; bind valid Host backing with representable strides and offsets",
                view.shape, view.strides, view.byte_offset
            ))
        };
        let rows = experts.checked_mul(width).ok_or_else(invalid)?;
        let count = rows.checked_mul(columns).ok_or_else(invalid)?;
        std::alloc::Layout::array::<T>(count).map_err(|_| invalid())?;
        let origin = (view.data.0 as usize)
            .checked_add(view.byte_offset)
            .ok_or_else(invalid)?;
        let mut strides = [0isize; 3];
        if count != 0 {
            let (mut low, mut high) = (0i128, 0i128);
            for (axis, (&dim, &stride)) in view.shape.iter().zip(view.strides).enumerate() {
                let extent = ((dim - 1) as i128)
                    .checked_mul(i128::from(stride))
                    .ok_or_else(invalid)?;
                if extent < 0 {
                    low = low.checked_add(extent).ok_or_else(invalid)?;
                } else {
                    high = high.checked_add(extent).ok_or_else(invalid)?;
                }
                strides[axis] = if dim == 1 {
                    0
                } else {
                    isize::try_from(stride).map_err(|_| invalid())?
                };
            }
            let bytes = high
                .checked_sub(low)
                .and_then(|n| n.checked_add(1))
                .and_then(|n| n.checked_mul(size_of::<T>() as i128))
                .ok_or_else(invalid)?;
            if bytes > isize::MAX as i128 {
                return Err(invalid());
            }
            let offset = |value: i128| {
                value
                    .checked_mul(size_of::<T>() as i128)
                    .and_then(|v| isize::try_from(v).ok())
                    .ok_or_else(invalid)
            };
            origin
                .checked_add_signed(offset(low)?)
                .ok_or_else(invalid)?;
            origin
                .checked_add_signed(offset(high.checked_add(1).ok_or_else(invalid)?)?)
                .ok_or_else(invalid)?;
        }
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

pub(super) struct AffineBank<'a> {
    codes: Storage<'a, u8>,
    scales: Storage<'a, f32>,
    zeros: Option<Storage<'a, u8>>,
    depth: usize,
    block: usize,
    zero: u8,
}
impl<'a> AffineBank<'a> {
    pub(super) fn new(
        codes: &'a TensorView<'_>,
        scales: &'a TensorView<'_>,
        zeros: Option<&'a TensorView<'_>>,
        [experts, width, depth]: [usize; 3],
        block: usize,
        zero: u8,
    ) -> Result<Self> {
        if block == 0 || !depth.is_multiple_of(block) || zero > 15 {
            return Err(error(format!(
                "block {block} must divide depth {depth} and zero point {zero} must fit UINT4; supply a valid affine format"
            )));
        }
        let blocks = depth / block;
        if codes.dtype != DataType::Uint8
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
        // Validate every source before reading any scale (or packed byte).
        let codes = Storage::new("codes", codes)?;
        let scales = Storage::<f32>::new("scales", scales)?;
        let zeros = zeros.map(|z| Storage::new("zero points", z)).transpose()?;
        if let Some((row, block, value)) = scales.first_invalid_scale() {
            return Err(error(format!(
                "scales row {row}, block {block} is {value}; DequantizeLinear scales must be positive and finite; bind a valid scale bank"
            )));
        }
        Ok(Self {
            codes,
            scales,
            zeros,
            depth,
            block,
            zero,
        })
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
        input.iter().enumerate().fold(0.0, |sum, (col, &x)| {
            let p = col / self.block;
            let q = (codes.get::<DENSE>(col / 2) >> ((col % 2) * 4)) & 15;
            let z = zeros
                .as_ref()
                .map_or(self.zero, |z| (z.get::<DENSE>(p / 2) >> ((p % 2) * 4)) & 15);
            let weight = f32::from(i16::from(q) - i16::from(z)) * scales.get::<DENSE>(p);
            sum + x * weight
        })
    }
}

#[cfg(test)]
mod tests;
