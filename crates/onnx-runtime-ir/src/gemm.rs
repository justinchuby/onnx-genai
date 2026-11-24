//! Shape contract shared by symbolic inference and native Gemm kernels.

pub const ND_GEMM_OPSET: u64 = 29;

#[derive(Clone, Copy, Debug)]
pub struct GemmAxes {
    /// A's contraction axis. The other trailing axis, if present, is M.
    pub a_k: usize,
    pub b_k: usize,
    pub trans_a: bool,
}

impl GemmAxes {
    pub fn new(
        a_rank: usize,
        b_rank: usize,
        trans_a: bool,
        trans_b: bool,
        nd: bool,
    ) -> Result<Self, String> {
        if a_rank == 0 || (!nd && a_rank != 2) {
            let required = if nd {
                "rank at least 1"
            } else {
                "rank 2 before opset 29"
            };
            return Err(format!(
                "Gemm input A must have {required}, got rank {a_rank}; supply a compatible A shape"
            ));
        }
        if b_rank != 2 {
            return Err(format!(
                "Gemm input B must have rank 2, got rank {b_rank}; supply a matrix for B"
            ));
        }
        let trans_a = trans_a && a_rank > 1;
        Ok(Self {
            a_k: a_rank - 1 - usize::from(trans_a),
            b_k: usize::from(trans_b),
            trans_a,
        })
    }

    pub fn output_dims<'a, T>(self, a: &'a [T], b: &'a [T]) -> impl Iterator<Item = &'a T> {
        a[..a.len().saturating_sub(2)]
            .iter()
            .chain((a.len() >= 2).then(|| &a[a.len() - 2 + usize::from(self.trans_a)]))
            .chain(std::iter::once(&b[1 - self.b_k]))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GemmGeometry {
    pub axes: GemmAxes,
    pub batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub rows: usize,
    pub elements: usize,
}

impl GemmGeometry {
    pub fn new(
        a: &[usize],
        b: &[usize],
        trans_a: bool,
        trans_b: bool,
        nd: bool,
    ) -> Result<Self, String> {
        let axes = GemmAxes::new(a.len(), b.len(), trans_a, trans_b, nd)?;
        let (k, kb) = (a[axes.a_k], b[axes.b_k]);
        if k != kb {
            return Err(format!(
                "Gemm inner dimensions disagree ({k} vs {kb}): A {a:?} transA={trans_a}, B {b:?} transB={trans_b}; make the contraction dimensions equal"
            ));
        }
        let m = if a.len() == 1 {
            1
        } else {
            a[a.len() - 2 + usize::from(axes.trans_a)]
        };
        let n = b[1 - axes.b_k];
        let prefix = &a[..a.len().saturating_sub(2)];
        let batch = checked_elements(prefix).ok_or_else(|| capacity(a, b))?;
        let output_capacity = || {
            format!(
                "Gemm output for A {a:?} and B {b:?} exceeds addressable element capacity; reduce the output dimensions"
            )
        };
        let rows = batch.checked_mul(m).ok_or_else(output_capacity)?;
        let elements = rows.checked_mul(n).ok_or_else(output_capacity)?;
        rows.checked_mul(k)
            .and_then(|_| k.checked_mul(n))
            .ok_or_else(|| capacity(a, b))?;
        Ok(Self {
            axes,
            batch,
            m,
            k,
            n,
            rows,
            elements,
        })
    }

    /// Swapping a singleton axis does not change dense element order.
    pub fn transposes_a_storage(self) -> bool {
        self.axes.trans_a && self.m > 1 && self.k > 1
    }

    pub fn validate_output(self, a: &[usize], b: &[usize], y: &[usize]) -> Result<(), String> {
        if self.axes.output_dims(a, b).copied().eq(y.iter().copied()) {
            return Ok(());
        }
        let expected: Vec<_> = self.axes.output_dims(a, b).copied().collect();
        Err(format!(
            "Gemm output Y has shape {y:?}, expected {expected:?}; correct the output metadata"
        ))
    }
}

pub fn checked_elements(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        return Some(0);
    }
    shape.iter().try_fold(1usize, |n, &dim| n.checked_mul(dim))
}

pub fn validate_bias(c: &[usize], y: &[usize]) -> Result<(), String> {
    if bias_broadcasts(c, y, |&d| Some(d)) {
        Ok(())
    } else {
        Err(format!(
            "Gemm bias C {c:?} does not broadcast to output {y:?}; use a right-aligned shape with equal or singleton dimensions"
        ))
    }
}

pub fn bias_broadcasts<T, D: PartialEq + From<u8>>(
    c: &[T],
    y: &[T],
    dimension: impl Fn(&T) -> Option<D>,
) -> bool {
    c.len() <= y.len() && c.iter().rev().zip(y.iter().rev()).all(|(c, y)| {
        !matches!((dimension(c), dimension(y)), (Some(c), Some(y)) if c != D::from(1) && c != y)
    })
}

fn capacity(a: &[usize], b: &[usize]) -> String {
    format!(
        "Gemm inputs A {a:?} and B {b:?} exceed addressable element capacity; reduce the matrix dimensions"
    )
}
