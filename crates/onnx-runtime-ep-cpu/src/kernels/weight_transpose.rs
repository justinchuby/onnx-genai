//! Shared tiled transpose algorithm. Cache ownership, identity, admission and
//! reclamation belong to the node's WeightCache in matmul.rs.

use onnx_runtime_ep_api::{EpError, Result};
use rayon::prelude::*;

/// Allocate and transpose a complete matrix for call-owned computation.
pub fn transpose_row_major<T: Copy + Default + Send + Sync>(
    src: &[T],
    k: usize,
    n: usize,
) -> Vec<T> {
    assert!(
        k.checked_mul(n) == Some(src.len()),
        "transpose_row_major: source length {} does not match [{k}, {n}]",
        src.len()
    );
    let mut out = vec![T::default(); src.len()];
    transpose_row_major_into(src, k, n, &mut out)
        .expect("source geometry checked and matching output allocated");
    out
}

/// Fill already admitted storage; no second matrix-sized allocation is made.
pub(crate) fn transpose_row_major_into<T: Copy + Default + Send + Sync>(
    src: &[T],
    k: usize,
    n: usize,
    out: &mut [T],
) -> Result<()> {
    if k.checked_mul(n) != Some(src.len()) || out.len() != src.len() {
        return Err(EpError::KernelFailed(format!(
            "weight transpose [{k}, {n}] has {} source and {} output elements; provide matching complete matrix storage",
            src.len(),
            out.len()
        )));
    }
    transpose_map_into(k, n, out, |i, j| src[i * n + j])
}

/// Share tiled traversal with conversion: each output element is produced
/// directly from its original source coordinate, without a dense staging copy.
pub(crate) fn transpose_map_into<T: Copy + Send + Sync>(
    k: usize,
    n: usize,
    out: &mut [T],
    read: impl Fn(usize, usize) -> T + Send + Sync,
) -> Result<()> {
    // SAFETY: MaybeUninit<T> has T's layout. This exclusive initialized slice
    // may be written as MaybeUninit because Copy elements require no drop;
    // the complete fill below reestablishes initialized storage before return.
    let uninit = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len()) };
    transpose_map_uninit(k, n, uninit, read).map(|_| ())
}

/// Initialize every destination element exactly once, including fresh workspace.
pub(crate) fn transpose_map_uninit<T: Copy + Send + Sync>(
    k: usize,
    n: usize,
    out: &mut [std::mem::MaybeUninit<T>],
    read: impl Fn(usize, usize) -> T + Send + Sync,
) -> Result<&mut [T]> {
    if k.checked_mul(n) != Some(out.len()) {
        return Err(EpError::KernelFailed(format!(
            "weight transpose [{k}, {n}] cannot fill {} output elements; provide the complete matrix destination",
            out.len()
        )));
    }
    if out.is_empty() {
        return Ok(&mut []);
    }
    let threads = rayon::current_num_threads();
    let rows_per_thread = n.div_ceil(threads).max(1);
    out.par_chunks_mut(rows_per_thread * k)
        .enumerate()
        .for_each(|(t, chunk)| {
            let j0 = t * rows_per_thread;
            let j_end = (j0 + rows_per_thread).min(n);
            let chunk_n = j_end - j0;
            const TILE: usize = 64;
            for i0 in (0..k).step_by(TILE) {
                let ie = (i0 + TILE).min(k);
                for jj in 0..chunk_n {
                    let j = j0 + jj;
                    for i in i0..ie {
                        chunk[jj * k + i].write(read(i, j));
                    }
                }
            }
        });
    // SAFETY: checked geometry and the disjoint parallel output rows ensure
    // every element was initialized above; no source reader can skip a write.
    Ok(unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initializes_fresh_workspace_without_a_zero_fill() {
        for (k, n) in [(1, 1), (3, 5), (65, 63), (129, 7)] {
            let mut output = vec![std::mem::MaybeUninit::<f32>::uninit(); k * n];
            let values =
                transpose_map_uninit(k, n, &mut output, |i, j| (i * n + j) as f32).unwrap();
            for j in 0..n {
                for i in 0..k {
                    assert_eq!(values[j * k + i], (i * n + j) as f32);
                }
            }
        }
    }

    #[test]
    fn admitted_transpose_output_is_checked_before_any_write() {
        let source = [1u16, 2, 3, 4, 5, 6];
        let mut wrong = [99; 5];
        let error = transpose_row_major_into(&source, 2, 3, &mut wrong)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("[2, 3]") && error.contains("6 source") && error.contains("5 output"),
            "{error}"
        );
        assert_eq!(wrong, [99; 5]);
        let mut output = [99; 6];
        assert!(transpose_row_major_into(&source[..5], 2, 3, &mut output).is_err());
        assert_eq!(output, [99; 6]);
        assert!(transpose_row_major_into(&source, usize::MAX, 2, &mut output).is_err());
        assert_eq!(output, [99; 6]);
        transpose_row_major_into(&source, 2, 3, &mut output).unwrap();
        assert_eq!(output, [1, 4, 2, 5, 3, 6]);
        transpose_row_major_into::<u16>(&[], 0, 7, &mut []).unwrap();
        let reads = std::sync::atomic::AtomicUsize::new(0);
        let mut short = [99u16; 5];
        assert!(
            transpose_map_into(2, 3, &mut short, |_, _| {
                reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                0
            })
            .is_err()
        );
        assert_eq!(reads.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(short, [99; 5]);
    }

    fn reference_transpose(src: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n * k];
        for j in 0..n {
            for i in 0..k {
                out[j * k + i] = src[i * n + j];
            }
        }
        out
    }

    fn fill(buf: &mut [f32], seed: f32) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = seed + i as f32 * 0.5;
        }
    }

    #[test]
    fn tiled_transpose_matches_reference() {
        for (k, n) in [(1usize, 1usize), (3, 5), (64, 64), (65, 63), (129, 7)] {
            let mut buf = vec![0.0f32; k * n];
            fill(&mut buf, 0.125);
            assert_eq!(
                transpose_row_major(&buf, k, n),
                reference_transpose(&buf, k, n),
                "[{k}, {n}] transpose mismatch"
            );
        }
    }
}
