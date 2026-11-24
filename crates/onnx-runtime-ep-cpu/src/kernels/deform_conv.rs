//! DeformConv with bounded spatial panels and native GEMM.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::deform_conv::{Attributes, Geometry, validate_types};
use onnx_runtime_ir::{DataType, Node};

use crate::backend::CpuBackend;
use crate::dtype::{to_dense, to_dense_f32_widen, write_dense, write_dense_f32_narrow};

const OP: &str = "DeformConv";

pub struct Factory(pub u64);

struct DeformConv {
    attrs: Attributes,
    version: u64,
}

fn fail(detail: String) -> EpError {
    EpError::KernelFailed(format!("{OP}: {detail}"))
}

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(DeformConv {
            attrs: Attributes::parse(node).map_err(fail)?,
            version: self.0,
        }))
    }
}

impl Kernel for DeformConv {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        super::check_arity(OP, inputs, outputs, 3, 5, 1)?;
        let slots = std::array::from_fn(|i| inputs.get(i).filter(|v| !v.is_absent()));
        validate_types(
            self.version,
            slots.map(|s| s.map(|t| t.dtype)),
            Some(outputs[0].dtype),
        )
        .map_err(fail)?;
        let g = self
            .attrs
            .resolve(slots.map(|s| s.map(|t| t.shape)))
            .map_err(fail)?;
        if g.output != outputs[0].shape {
            return Err(fail(format!(
                "Y shape {:?}, expected {:?}",
                outputs[0].shape, g.output
            )));
        }
        if g.output.contains(&0) {
            return Ok(());
        }
        let [Some(x), Some(w), Some(offset), b, mask] = slots else {
            unreachable!("geometry checks required inputs");
        };
        if x.dtype == DataType::Float64 {
            let x = to_dense::<f64>(x)?;
            let w = to_dense::<f64>(w)?;
            let offset = to_dense::<f64>(offset)?;
            let b = b.map(to_dense::<f64>).transpose()?;
            let mask = mask.map(to_dense::<f64>).transpose()?;
            let y = run(&g, &x, &w, &offset, b.as_deref(), mask.as_deref())?;
            write_dense(&mut outputs[0], &y)
        } else {
            let load = |v| to_dense_f32_widen(OP, v);
            let x = load(x)?;
            let w = load(w)?;
            let offset = load(offset)?;
            let b = b.map(load).transpose()?;
            let mask = mask.map(load).transpose()?;
            let y = run(&g, &x, &w, &offset, b.as_deref(), mask.as_deref())?;
            write_dense_f32_narrow(OP, &mut outputs[0], &y)
        }
    }
}

trait Number:
    Copy + Default + Into<f64> + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self>
{
    fn from_f64(x: f64) -> Self;
    fn multiply(
        w: &[Self],
        panel: &[Self],
        y: &mut [Self],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()>;
}

impl Number for f32 {
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn multiply(
        w: &[Self],
        panel: &[Self],
        y: &mut [Self],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        if m > 1 {
            let backend = CpuBackend::auto_detect();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if backend == CpuBackend::SimdX86 {
                super::matmul::x86_sgemm::sgemm_simd_panel(w, panel, y, m, k, n);
                return Ok(());
            }
            return super::matmul::gemm_with_backend(backend, w, panel, y, m, k, n);
        }
        direct_dot(w, panel, y, k, n);
        Ok(())
    }
}

impl Number for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    fn multiply(
        w: &[Self],
        panel: &[Self],
        y: &mut [Self],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        for row in 0..m {
            direct_dot(&w[row * k..][..k], panel, &mut y[row * n..][..n], k, n);
        }
        Ok(())
    }
}

fn direct_dot<T: Number>(weights: &[T], panel: &[T], y: &mut [T], k: usize, n: usize) {
    y.fill(T::default());
    for (j, &w) in weights[..k].iter().enumerate() {
        for (sum, &x) in y.iter_mut().zip(&panel[j * n..][..n]) {
            *sum = *sum + w * x;
        }
    }
}

fn run<T: Number>(
    g: &Geometry,
    x: &[T],
    w: &[T],
    offsets: &[T],
    bias: Option<&[T]>,
    mask: Option<&[T]>,
) -> Result<Vec<T>> {
    let m = g.output_per_group;
    let k = g.input_per_group * g.kernel_spatial;
    let output_channels = g.output[1];
    let spatial = g.output_spatial;
    let mut y = vec![T::default(); g.output.iter().product()];
    if k == 0 {
        if let Some(bias) = bias {
            for (i, plane) in y.chunks_exact_mut(spatial).enumerate() {
                plane.fill(bias[i % output_channels]);
            }
        }
        return Ok(y);
    }
    // Keep the sampled panel and GEMM output close to L1 on typical channel counts.
    let bytes_per_column = k.saturating_add(m).saturating_mul(std::mem::size_of::<T>());
    let tile = (32768 / bytes_per_column).clamp(1, 128).min(spatial);
    let mut panel = vec![T::default(); k * tile];
    let mut tmp = vec![T::default(); m * tile];
    let mut neighbors = vec![Neighbors::default(); g.axes.len()];
    // At a split conv-group boundary the next group starts in the same offset
    // group. Retain all taps for that one offset group if scratch stays bounded.
    let reuse_stencils = g.axes.len() == 2
        && output_channels > m
        && g.input_per_group % g.input_per_offset_group != 0
        && g.kernel_spatial.saturating_mul(tile)
            <= 65536 / std::mem::size_of::<Option<Bilinear<T>>>();
    let mut bilinear = if g.axes.len() == 2 {
        vec![
            None;
            if reuse_stencils {
                g.kernel_spatial * tile
            } else {
                tile
            }
        ]
    } else {
        Vec::new()
    };
    let groups = output_channels / m;
    let tiles = spatial.div_ceil(tile);
    let (outer, inner) = if reuse_stencils {
        (tiles, groups)
    } else {
        (groups, tiles)
    };
    for batch in 0..g.output[0] {
        for i in 0..outer {
            let mut cached_og = None;
            for j in 0..inner {
                let (group, tile_index) = if reuse_stencils { (j, i) } else { (i, j) };
                let start = tile_index * tile;
                let n = tile.min(spatial - start);
                let first = group * g.input_per_group;
                let last = first + g.input_per_group;
                let mut c = first;
                while c < last {
                    let og = c / g.input_per_offset_group;
                    let end = last.min((og + 1) * g.input_per_offset_group);
                    for j in 0..g.kernel_spatial {
                        let mask_plane =
                            (batch * g.mask_channels + og * g.kernel_spatial + j) * spatial;
                        if let [h, w] = g.axes.as_slice() {
                            let kh = j / w.kernel * h.dilation;
                            let kw = j % w.kernel * w.dilation;
                            let offset_plane = mask_plane * 2;
                            let slot = if reuse_stencils { j * tile } else { 0 };
                            let stencils = &mut bilinear[slot..][..n];
                            if cached_og != Some(og) {
                                for (p, entry) in stencils.iter_mut().enumerate() {
                                    let point = start + p;
                                    let y = ((point / w.output) * h.stride + kh) as f64
                                        - h.pad as f64
                                        + offsets[offset_plane + point].into();
                                    let x = ((point % w.output) * w.stride + kw) as f64
                                        - w.pad as f64
                                        + offsets[offset_plane + spatial + point].into();
                                    *entry = axis_neighbors(y, h.input, w.input)
                                        .zip(axis_neighbors(x, w.input, 1))
                                        .map(|(r, c)| Bilinear::new(r, c));
                                }
                            }
                            let modulation = mask.map(|v| &v[mask_plane + start..][..n]);
                            for ic in c..end {
                                let plane = (batch * g.input_channels + ic) * g.input_spatial;
                                let source = &x[plane..][..g.input_spatial];
                                let row =
                                    &mut panel[((ic - first) * g.kernel_spatial + j) * n..][..n];
                                for (p, (dst, stencil)) in
                                    row.iter_mut().zip(&*stencils).enumerate()
                                {
                                    let value = stencil.map_or(T::default(), |s| s.sample(source));
                                    *dst = modulation.map_or(value, |m| value * m[p]);
                                }
                            }
                            continue;
                        }
                        for p in 0..n {
                            let point = start + p;
                            let offset = (batch * g.offset_channels
                                + (og * g.kernel_spatial + j) * g.axes.len())
                                * spatial
                                + point;
                            let valid = stencil(g, offsets, offset, j, point, &mut neighbors);
                            let modulation =
                                mask.map_or(T::from_f64(1.0), |v| v[mask_plane + point]);
                            for ic in c..end {
                                let value = if valid {
                                    let plane = (batch * g.input_channels + ic) * g.input_spatial;
                                    sample(&x[plane..][..g.input_spatial], &neighbors, 0)
                                } else {
                                    T::default()
                                };
                                panel[((ic - first) * g.kernel_spatial + j) * n + p] =
                                    value * modulation;
                            }
                        }
                    }
                    cached_og = reuse_stencils.then_some(og);
                    c = end;
                }
                T::multiply(
                    &w[group * m * k..][..m * k],
                    &panel[..k * n],
                    &mut tmp[..m * n],
                    m,
                    k,
                    n,
                )?;
                for oc in 0..m {
                    let channel = group * m + oc;
                    let b = bias.map_or(T::default(), |v| v[channel]);
                    let base = (batch * output_channels + channel) * spatial + start;
                    for (dst, &src) in y[base..][..n].iter_mut().zip(&tmp[oc * n..][..n]) {
                        *dst = src + b;
                    }
                }
            }
        }
    }
    Ok(y)
}

#[derive(Clone, Copy, Default)]
struct Neighbors {
    lo: Option<usize>,
    hi: Option<usize>,
    frac: f64,
}

#[derive(Clone, Copy)]
struct Bilinear<T> {
    indices: [usize; 4],
    row: [T; 2],
    col: [T; 2],
}

impl<T: Number> Bilinear<T> {
    #[inline]
    fn new(row: Neighbors, col: Neighbors) -> Self {
        let index = |r: Option<usize>, c| r.zip(c).map_or(usize::MAX, |(r, c)| r + c);
        Self {
            indices: [
                index(row.lo, col.lo),
                index(row.lo, col.hi),
                index(row.hi, col.lo),
                index(row.hi, col.hi),
            ],
            row: [T::from_f64(1.0 - row.frac), T::from_f64(row.frac)],
            col: [T::from_f64(1.0 - col.frac), T::from_f64(col.frac)],
        }
    }

    #[inline]
    fn sample(self, x: &[T]) -> T {
        let [a, b, c, d] = self.indices.map(|i| x.get(i).copied().unwrap_or_default());
        (a * self.col[0] + b * self.col[1]) * self.row[0]
            + (c * self.col[0] + d * self.col[1]) * self.row[1]
    }
}

#[inline]
fn axis_neighbors(position: f64, size: usize, pitch: usize) -> Option<Neighbors> {
    if size == 0 || !(position > -1.0 && position < size as f64) {
        return None;
    }
    if position < 0.0 {
        return Some(Neighbors {
            lo: None,
            hi: Some(0),
            frac: position + 1.0,
        });
    }
    let low = position as usize;
    let lo = Some(low * pitch);
    let frac = position - low as f64;
    let hi = (frac != 0.0 && low + 1 < size).then(|| (low + 1) * pitch);
    Some(Neighbors { lo, hi, frac })
}

fn stencil<T: Number>(
    g: &Geometry,
    offsets: &[T],
    offset: usize,
    mut kernel: usize,
    mut point: usize,
    neighbors: &mut [Neighbors],
) -> bool {
    let mut pitch = 1;
    for (d, a) in g.axes.iter().enumerate().rev() {
        let center = ((point % a.output) * a.stride + (kernel % a.kernel) * a.dilation) as f64
            - a.pad as f64;
        point /= a.output;
        kernel /= a.kernel;
        let position = center + offsets[offset + d * g.output_spatial].into();
        let Some(axis) = axis_neighbors(position, a.input, pitch) else {
            return false;
        };
        neighbors[d] = axis;
        pitch *= a.input;
    }
    true
}

#[inline]
fn blend<T: Number>(lo: T, hi: T, frac: f64) -> T {
    lo * T::from_f64(1.0 - frac) + hi * T::from_f64(frac)
}

#[inline]
fn sample<T: Number>(x: &[T], axes: &[Neighbors], base: usize) -> T {
    match axes {
        [] => x[base],
        [a] => blend(
            a.lo.map_or(T::default(), |i| x[base + i]),
            a.hi.map_or(T::default(), |i| x[base + i]),
            a.frac,
        ),
        [a, rest @ ..] => blend(
            a.lo.map_or(T::default(), |i| sample(x, rest, base + i)),
            a.hi.map_or(T::default(), |i| sample(x, rest, base + i)),
            a.frac,
        ),
    }
}
