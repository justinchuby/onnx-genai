//! Register-only NT tiles: SIMD lanes are independent output columns, not K
//! partial sums. Four consecutive K values are loaded at once and consumed in
//! their original order, with separate multiplication and addition rounding.

use std::arch::aarch64::*;

#[cfg(test)]
thread_local! {
    static BLOCK_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static EIGHT_ROW_TILES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
pub(super) fn block_calls() -> usize {
    BLOCK_CALLS.get()
}

#[cfg(test)]
pub(super) fn eight_row_tiles() -> usize {
    EIGHT_ROW_TILES.get()
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
    let wide_rows = rows / 8 * 8;
    let full_rows = rows / 4 * 4;
    let full_cols = n / 4 * 4;
    let full_k = k / 4 * 4;
    macro_rules! load_b_columns {
        ($b0:ident, $b1:ident, $b2:ident, $b3:ident, $p:ident) => {{
            let bv0 = vld1q_f32($b0.add($p));
            let bv1 = vld1q_f32($b1.add($p));
            let bv2 = vld1q_f32($b2.add($p));
            let bv3 = vld1q_f32($b3.add($p));
            let t0 = vtrn1q_f32(bv0, bv1);
            let t1 = vtrn2q_f32(bv0, bv1);
            let t2 = vtrn1q_f32(bv2, bv3);
            let t3 = vtrn2q_f32(bv2, bv3);
            [
                vcombine_f32(vget_low_f32(t0), vget_low_f32(t2)),
                vcombine_f32(vget_low_f32(t1), vget_low_f32(t3)),
                vcombine_f32(vget_high_f32(t0), vget_high_f32(t2)),
                vcombine_f32(vget_high_f32(t1), vget_high_f32(t3)),
            ]
        }};
    }
    // SAFETY: whole tiles stay inside the exact matrix extents; vector loads
    // require only f32 alignment. Accumulators live in registers and stores
    // address disjoint output tiles. Remainders use the portable implementation.
    unsafe {
        // Reuse each B register transpose across eight independent output
        // rows before handling the remaining four-row tile below.
        for i in (0..wide_rows).step_by(8) {
            let a0 = a.as_ptr().add(i * k);
            let a1 = a0.add(k);
            let a2 = a1.add(k);
            let a3 = a2.add(k);
            let a4 = a3.add(k);
            let a5 = a4.add(k);
            let a6 = a5.add(k);
            let a7 = a6.add(k);
            for j in (0..full_cols).step_by(4) {
                #[cfg(test)]
                EIGHT_ROW_TILES.set(EIGHT_ROW_TILES.get() + 1);
                let b0 = bt.as_ptr().add(j * k);
                let b1 = b0.add(k);
                let b2 = b1.add(k);
                let b3 = b2.add(k);
                let mut c0 = vdupq_n_f32(0.0);
                let mut c1 = c0;
                let mut c2 = c0;
                let mut c3 = c0;
                let mut c4 = c0;
                let mut c5 = c0;
                let mut c6 = c0;
                let mut c7 = c0;
                for p in (0..full_k).step_by(4) {
                    let av0 = vld1q_f32(a0.add(p));
                    let av1 = vld1q_f32(a1.add(p));
                    let av2 = vld1q_f32(a2.add(p));
                    let av3 = vld1q_f32(a3.add(p));
                    let av4 = vld1q_f32(a4.add(p));
                    let av5 = vld1q_f32(a5.add(p));
                    let av6 = vld1q_f32(a6.add(p));
                    let av7 = vld1q_f32(a7.add(p));
                    let [bp0, bp1, bp2, bp3] = load_b_columns!(b0, b1, b2, b3, p);
                    macro_rules! step {
                        ($lane:literal, $b:ident) => {
                            c0 = vaddq_f32(c0, vmulq_laneq_f32::<$lane>($b, av0));
                            c1 = vaddq_f32(c1, vmulq_laneq_f32::<$lane>($b, av1));
                            c2 = vaddq_f32(c2, vmulq_laneq_f32::<$lane>($b, av2));
                            c3 = vaddq_f32(c3, vmulq_laneq_f32::<$lane>($b, av3));
                            c4 = vaddq_f32(c4, vmulq_laneq_f32::<$lane>($b, av4));
                            c5 = vaddq_f32(c5, vmulq_laneq_f32::<$lane>($b, av5));
                            c6 = vaddq_f32(c6, vmulq_laneq_f32::<$lane>($b, av6));
                            c7 = vaddq_f32(c7, vmulq_laneq_f32::<$lane>($b, av7));
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
                    c4 = vaddq_f32(c4, vmulq_n_f32(bv, *a4.add(p)));
                    c5 = vaddq_f32(c5, vmulq_n_f32(bv, *a5.add(p)));
                    c6 = vaddq_f32(c6, vmulq_n_f32(bv, *a6.add(p)));
                    c7 = vaddq_f32(c7, vmulq_n_f32(bv, *a7.add(p)));
                }
                let out = c.as_mut_ptr().add(i * n + j);
                vst1q_f32(out, c0);
                vst1q_f32(out.add(n), c1);
                vst1q_f32(out.add(2 * n), c2);
                vst1q_f32(out.add(3 * n), c3);
                vst1q_f32(out.add(4 * n), c4);
                vst1q_f32(out.add(5 * n), c5);
                vst1q_f32(out.add(6 * n), c6);
                vst1q_f32(out.add(7 * n), c7);
            }
        }
        for i in (wide_rows..full_rows).step_by(4) {
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
                    let [bp0, bp1, bp2, bp3] = load_b_columns!(b0, b1, b2, b3, p);
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
        let b: Vec<_> = (0..20)
            .map(|i| values[(i * 3 + 1) % values.len()])
            .collect();
        for rows in [4, 8, 12] {
            let a: Vec<_> = (0..rows * 5).map(|i| values[i % values.len()]).collect();
            let mut expected = vec![f32::NAN; rows * 4];
            super::super::gemm_bt_block_portable(&a, &b, &mut expected, rows, 5, 4);
            let mut guarded = vec![29.0f32; rows * 4 + 2];
            // SAFETY: runtime NEON check, exact extents, separate guarded output.
            unsafe { block(&a, &b, &mut guarded[1..1 + rows * 4], rows, 5, 4) };
            assert_eq!((guarded[0], guarded[rows * 4 + 1]), (29.0, 29.0));
            for (actual, expected) in guarded[1..1 + rows * 4].iter().zip(expected) {
                if expected.is_nan() {
                    assert!(actual.is_nan());
                } else {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
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
            (7, 4),
            (8, 9),
            (9, 4),
            (12, 4),
            (13, 5),
            (15, 7),
            (16, 64),
            (17, 13),
            (24, 9),
            (64, 16),
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
            for rows in [4, 8, 12] {
                let a = a.repeat(rows);
                let b = b.repeat(4);
                let mut c = vec![f32::NAN; rows * 4];
                // SAFETY: runtime NEON check and exact, separate matrix storage.
                unsafe { block(&a, &b, &mut c, rows, 4, 4) };
                assert!(
                    c.iter().all(|x| x.to_bits() == f32::to_bits(want)),
                    "rows={rows}: {c:?}"
                );
            }
        }
    }
}
