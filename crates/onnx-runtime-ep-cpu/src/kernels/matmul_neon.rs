//! Register-only NT tiles: SIMD lanes are independent output columns, not K
//! partial sums. Four consecutive K values are loaded at once and consumed in
//! their original order, with separate multiplication and addition rounding.

use std::arch::aarch64::*;

#[cfg(test)]
thread_local! {
    static BLOCK_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
pub(super) fn block_calls() -> usize {
    BLOCK_CALLS.get()
}

/// C[rows,N] = A[rows,K] @ B[N,K]^T, overwriting every output.
///
/// # Safety
/// NEON must be available. The caller supplies exact contiguous matrix extents
/// (`rows*K`, `N*K`, `rows*N`); input storage must not alias mutable output.
#[target_feature(enable = "neon")]
#[inline(never)]
pub(super) unsafe fn block(a: &[f32], bt: &[f32], c: &mut [f32], rows: usize, k: usize, n: usize) {
    #[cfg(test)]
    BLOCK_CALLS.set(BLOCK_CALLS.get() + 1);
    debug_assert_eq!(rows.checked_mul(k), Some(a.len()));
    debug_assert_eq!(n.checked_mul(k), Some(bt.len()));
    debug_assert_eq!(rows.checked_mul(n), Some(c.len()));
    let full_rows = rows / 4 * 4;
    let full_cols = n / 4 * 4;
    let full_k = k / 4 * 4;
    // SAFETY: whole tiles stay inside the exact matrix extents; vector loads
    // require only f32 alignment. Accumulators live in registers and stores
    // address disjoint output tiles. Remainders use the portable implementation.
    unsafe {
        for i in (0..full_rows).step_by(4) {
            let a0 = a.as_ptr().add(i * k);
            let a1 = a0.add(k);
            let a2 = a1.add(k);
            let a3 = a2.add(k);
            for j in (0..full_cols).step_by(4) {
                let b0 = bt.as_ptr().add(j * k);
                let b1 = b0.add(k);
                let b2 = b1.add(k);
                let b3 = b2.add(k);
                let mut c0 = vdupq_n_f32(0.0);
                let mut c1 = c0;
                let mut c2 = c0;
                let mut c3 = c0;
                for p in (0..full_k).step_by(4) {
                    let av0 = vld1q_f32(a0.add(p));
                    let av1 = vld1q_f32(a1.add(p));
                    let av2 = vld1q_f32(a2.add(p));
                    let av3 = vld1q_f32(a3.add(p));
                    let bv0 = vld1q_f32(b0.add(p));
                    let bv1 = vld1q_f32(b1.add(p));
                    let bv2 = vld1q_f32(b2.add(p));
                    let bv3 = vld1q_f32(b3.add(p));
                    // Transpose a 4x4 B register tile. Each resulting vector
                    // holds four output-column operands for one original K.
                    let t0 = vtrn1q_f32(bv0, bv1);
                    let t1 = vtrn2q_f32(bv0, bv1);
                    let t2 = vtrn1q_f32(bv2, bv3);
                    let t3 = vtrn2q_f32(bv2, bv3);
                    let bp0 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
                    let bp1 = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
                    let bp2 = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
                    let bp3 = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));
                    macro_rules! step {
                        ($lane:literal, $b:ident) => {
                            c0 = vaddq_f32(c0, vmulq_laneq_f32::<$lane>($b, av0));
                            c1 = vaddq_f32(c1, vmulq_laneq_f32::<$lane>($b, av1));
                            c2 = vaddq_f32(c2, vmulq_laneq_f32::<$lane>($b, av2));
                            c3 = vaddq_f32(c3, vmulq_laneq_f32::<$lane>($b, av3));
                        };
                    }
                    step!(0, bp0);
                    step!(1, bp1);
                    step!(2, bp2);
                    step!(3, bp3);
                }
                for p in full_k..k {
                    let b = [*b0.add(p), *b1.add(p), *b2.add(p), *b3.add(p)];
                    let bv = vld1q_f32(b.as_ptr());
                    c0 = vaddq_f32(c0, vmulq_n_f32(bv, *a0.add(p)));
                    c1 = vaddq_f32(c1, vmulq_n_f32(bv, *a1.add(p)));
                    c2 = vaddq_f32(c2, vmulq_n_f32(bv, *a2.add(p)));
                    c3 = vaddq_f32(c3, vmulq_n_f32(bv, *a3.add(p)));
                }
                let out = c.as_mut_ptr().add(i * n + j);
                vst1q_f32(out, c0);
                vst1q_f32(out.add(n), c1);
                vst1q_f32(out.add(2 * n), c2);
                vst1q_f32(out.add(3 * n), c3);
            }
        }
    }
    // Keep the portable four-row reuse for the column remainder too: calling
    // a separate one-row kernel would reload each tail weight four times.
    if full_cols < n {
        for i in (0..full_rows).step_by(4) {
            let ar = std::array::from_fn::<_, 4, _>(|row| &a[(i + row) * k..(i + row + 1) * k]);
            for j in full_cols..n {
                let br = &bt[j * k..(j + 1) * k];
                let mut acc = [0.0f32; 4];
                for p in 0..k {
                    let weight = br[p];
                    for row in 0..4 {
                        acc[row] += ar[row][p] * weight;
                    }
                }
                for row in 0..4 {
                    c[(i + row) * n + j] = acc[row];
                }
            }
        }
    }
    if full_rows < rows {
        super::gemm_bt_block_portable(
            &a[full_rows * k..],
            bt,
            &mut c[full_rows * n..],
            rows - full_rows,
            k,
            n,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neon_nt_tiles_preserve_nonfinite_classes_and_signed_zero() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let values = [
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(1),
            -f32::from_bits(1),
            1.0,
        ];
        let a: Vec<_> = (0..20).map(|i| values[i % values.len()]).collect();
        let b: Vec<_> = (0..20)
            .map(|i| values[(i * 3 + 1) % values.len()])
            .collect();
        let mut expected = [f32::NAN; 16];
        super::super::gemm_bt_block_portable(&a, &b, &mut expected, 4, 5, 4);
        let mut guarded = [29.0f32; 18];
        // SAFETY: runtime NEON check, exact extents, separate guarded output.
        unsafe { block(&a, &b, &mut guarded[1..17], 4, 5, 4) };
        assert_eq!((guarded[0], guarded[17]), (29.0, 29.0));
        for (actual, expected) in guarded[1..17].iter().zip(expected) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn neon_nt_tiles_preserve_scalar_chains_for_full_and_partial_tiles() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let mut state = 0x5123_9abcu32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 16) as i16 as f32 / 17003.
        };
        for (m, n) in [
            (0, 5),
            (5, 0),
            (1, 1),
            (3, 3),
            (4, 4),
            (5, 7),
            (8, 9),
            (13, 5),
            (16, 64),
        ] {
            for k in [0, 1, 3, 4, 5, 15, 16, 17, 33, 65, 257] {
                let a: Vec<_> = (0..m * k).map(|_| next()).collect();
                let b: Vec<_> = (0..n * k).map(|_| next()).collect();
                let mut expected = vec![f32::NAN; m * n];
                super::super::gemm_bt_block_portable(&a, &b, &mut expected, m, k, n);
                let mut actual = vec![f32::NAN; m * n];
                // SAFETY: runtime NEON check and exact, separate matrix storage.
                unsafe { block(&a, &b, &mut actual, m, k, n) };
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "M={m} K={k} N={n}"
                );
            }
        }
    }

    #[test]
    fn neon_nt_tiles_neither_fuse_multiply_add_nor_reassociate_k() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let eps = f32::EPSILON;
        for (a, b, want) in [
            ([-1., 1. + eps, 0., 0.], [1., 1. - eps, 0., 0.], 0.),
            ([1e20, 1., -1e20, 1.], [1.; 4], 1.),
        ] {
            let a = a.repeat(4);
            let b = b.repeat(4);
            let mut c = [f32::NAN; 16];
            // SAFETY: runtime NEON check and exact, separate matrix storage.
            unsafe { block(&a, &b, &mut c, 4, 4, 4) };
            assert!(c.iter().all(|x| x.to_bits() == f32::to_bits(want)), "{c:?}");
        }
    }
}
