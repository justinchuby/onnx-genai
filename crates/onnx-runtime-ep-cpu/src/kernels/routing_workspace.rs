//! One admitted arena for routing scratch and stable expert grouping.
use super::{decoded_projection::ProjectionScratch, moe::RoutingBuffers};
use onnx_runtime_ep_api::EpError;
fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("MoE routing workspace: {}", message.into()))
}
use onnx_runtime_ep_api::{DevicePtrMut, Result, WorkspaceLayout, WorkspaceView};
use std::{alloc::Layout, mem::MaybeUninit, num::NonZeroUsize};

#[derive(Clone, Copy)]
struct Decision {
    expert: usize,
    weight: f32,
}

#[derive(Clone, Copy)]
struct Geometry {
    rows: usize,
    experts: usize,
    fanout: NonZeroUsize,
    routes: usize,
    router_elements: usize,
    has_weights: bool,
}
impl Geometry {
    fn new(rows: usize, experts: usize, k: usize, has_weights: bool) -> Result<Self> {
        let invalid = || {
            error(format!(
                "routing geometry rows={rows}, experts={experts}, k={k} exceeds addressable capacity or has no selected expert; reduce batch/expert dimensions and use positive k"
            ))
        };
        let fanout = NonZeroUsize::new(k.min(experts)).ok_or_else(invalid)?;
        let routes = rows.checked_mul(fanout.get()).ok_or_else(invalid)?;
        let router_elements = rows.checked_mul(experts).ok_or_else(invalid)?;
        experts.checked_add(1).ok_or_else(invalid)?;
        Ok(Self {
            rows,
            experts,
            fanout,
            routes,
            router_elements,
            has_weights,
        })
    }
    fn partition(self) -> Result<WorkspaceLayout<7>> {
        let layout = |n, kind| {
            Layout::array::<usize>(n).map_err(|_| error(format!("routing {kind} array of {n} entries exceeds addressable capacity; reduce batch/expert dimensions")))
        };
        WorkspaceLayout::new([
            Layout::array::<Decision>(self.routes).map_err(|_| {
                error("routing decisions exceed addressable capacity; reduce batch or k")
            })?,
            layout(self.routes, "bucket indices")?,
            layout(self.experts, "cursors")?,
            layout(self.experts + 1, "prefixes")?,
            layout(self.experts, "TopK indices")?,
            Layout::array::<f32>(if self.has_weights { 0 } else { self.experts }).map_err(
                |_| error("routing softmax storage exceeds addressable capacity; reduce experts"),
            )?,
            Layout::array::<(usize, f32)>(self.fanout.get()).map_err(|_| {
                error("routing selected weights exceed addressable capacity; reduce k")
            })?,
        ])
    }
}

pub(super) struct RoutingArena<'a> {
    geometry: Geometry,
    decisions: &'a mut [MaybeUninit<Decision>],
    order: &'a mut [MaybeUninit<usize>],
    cursors: &'a mut [usize],
    starts: &'a mut [usize],
    indices: &'a mut [usize],
    exponentials: &'a mut [f32],
    selected: &'a mut [(usize, f32)],
}

// SAFETY: the caller proves an exact typed layout, alignment, exclusive Host
// lifetime and disjointness for this raw region. No initialized T is exposed.
unsafe fn uninit_region<'a, T>(region: WorkspaceView) -> &'a mut [MaybeUninit<T>] {
    unsafe {
        std::slice::from_raw_parts_mut(
            region.ptr().0.cast(),
            region.bytes() / std::mem::size_of::<T>(),
        )
    }
}
fn initialize<T: Copy>(storage: &mut [MaybeUninit<T>], value: T) -> &mut [T] {
    storage.fill(MaybeUninit::new(value));
    // SAFETY: every element was just initialized; padding is never inspected.
    unsafe { assume_init(storage) }
}
// SAFETY: every logical element must have been written, and the exclusive
// storage borrow covers all returned references. No padding/tail is exposed.
unsafe fn assume_init<T>(storage: &mut [MaybeUninit<T>]) -> &mut [T] {
    unsafe { std::slice::from_raw_parts_mut(storage.as_mut_ptr().cast(), storage.len()) }
}

impl<'a> RoutingArena<'a> {
    pub(super) fn layout(
        rows: usize,
        experts: usize,
        k: usize,
        has_weights: bool,
    ) -> Result<Layout> {
        let partition = Geometry::new(rows, experts, k, has_weights)?.partition()?;
        Layout::from_size_align(partition.bytes(), partition.alignment()).map_err(|_| {
            error("routing composite exceeds addressable capacity; reduce batch/expert dimensions")
        })
    }
    pub(super) fn bind(
        rows: usize,
        experts: usize,
        k: usize,
        has_weights: bool,
        lane: &'a mut ProjectionScratch<'_>,
    ) -> Result<Self> {
        let geometry = Geometry::new(rows, experts, k, has_weights)?;
        let partition = geometry.partition()?;
        // This revokes f32 initialization proofs before arbitrary typed writes.
        let bytes = lane.byte_storage(partition.bytes())?;
        let view = WorkspaceView::new(DevicePtrMut(bytes.as_mut_ptr().cast()), bytes.len());
        let [
            decisions,
            order,
            cursors,
            starts,
            indices,
            exponentials,
            selected,
        ] = partition.bind(Some(view))?;
        // SAFETY: complete binding checks all typed layouts, maximum alignment,
        // capacity and disjoint offsets. The parent exclusively lends Host
        // backing, with lifetime tied to lane. Only metadata scratch is cleared;
        // route-sized decisions/order remain uninitialized until build fills them.
        Ok(unsafe {
            Self {
                geometry,
                decisions: uninit_region(decisions),
                order: uninit_region(order),
                cursors: initialize(uninit_region(cursors), 0),
                starts: initialize(uninit_region(starts), 0),
                indices: initialize(uninit_region(indices), 0),
                exponentials: initialize(uninit_region(exponentials), 0.),
                selected: initialize(uninit_region(selected), (0, 0.)),
            }
        })
    }
    pub(super) fn build(
        self,
        logits: &[f32],
        weights: Option<&[f32]>,
        normalize: bool,
    ) -> Result<ExpertRoutes<'a>> {
        let g = self.geometry;
        if logits.len() != g.router_elements
            || weights.is_some() != g.has_weights
            || weights.is_some_and(|w| w.len() != g.router_elements)
        {
            return Err(error(format!(
                "routing arena expects {} logits and matching aggregation presence={}, got {} logits and aggregation length {:?}; refine/admit the current input geometry before routing",
                g.router_elements,
                g.has_weights,
                logits.len(),
                weights.map(<[f32]>::len)
            )));
        }
        let mut written = 0;
        for row in 0..g.rows {
            let range = row * g.experts..(row + 1) * g.experts;
            let route = RoutingBuffers::new(
                &mut *self.indices,
                &mut *self.exponentials,
                &mut *self.selected,
            )
            .weights(
                &logits[range.clone()],
                weights.map(|w| &w[range]),
                g.fanout.get(),
                normalize,
            )?;
            // Preserve score-order decision indices as contribution slots.
            // Stable bucketing changes access order, never slot identity.
            for &(expert, weight) in route.iter() {
                self.decisions[written].write(Decision { expert, weight });
                self.cursors[expert] += 1;
                written += 1;
            }
        }
        if written != g.routes {
            return Err(error(format!(
                "routing initialized {written} decisions, expected {}; prepare the current routing geometry",
                g.routes
            )));
        }
        // SAFETY: each exact-prefix entry was written once above; failed builds
        // never publish a plan. The arena cannot be reused while this plan lives.
        let decisions = unsafe { assume_init(self.decisions) };
        let mut active = 0;
        for expert in 0..g.experts {
            if self.cursors[expert] != 0 {
                // TopK indices are dead after every row has been routed.
                self.indices[active] = expert;
                active += 1;
            }
            self.starts[expert + 1] = self.starts[expert]
                .checked_add(self.cursors[expert])
                .ok_or_else(|| error("routing prefix overflow; reduce batch or k"))?;
            self.cursors[expert] = self.starts[expert];
        }
        if self.starts[g.experts] != g.routes {
            return Err(error(
                "routing counts do not cover the prepared routes; rebuild the routing plan",
            ));
        }
        for (index, decision) in decisions.iter().enumerate() {
            let target = self.cursors[decision.expert];
            self.order[target].write(index);
            self.cursors[decision.expert] += 1;
        }
        // SAFETY: counts partition [0,routes). Every decision advances exactly
        // one cursor, writing every slot of every bucket once, in row order.
        let order = unsafe { assume_init(self.order) };
        // Bucket cursors are now dead. Publish counts from immutable prefixes,
        // not from the cursors that advanced while writing bucket indices.
        for (ordinal, &expert) in self.indices[..active].iter().enumerate() {
            self.cursors[ordinal] = self.starts[expert + 1] - self.starts[expert];
        }
        let counts = ExpertGroupCounts {
            experts: &self.indices[..active],
            counts: &self.cursors[..active],
            assignments: g.routes,
        };
        Ok(ExpertRoutes {
            decisions,
            order,
            starts: self.starts,
            fanout: g.fanout,
            counts,
        })
    }
}

/// Distinct ascending expert membership and assignment counts. Private fields
/// make production construction exclusive to a completed arena build. Zero-count
/// membership is supported by the checked test constructor for metrics fixtures.
pub(crate) struct ExpertGroupCounts<'a> {
    experts: &'a [usize],
    counts: &'a [usize],
    assignments: usize,
}
impl ExpertGroupCounts<'_> {
    pub(crate) fn len(&self) -> usize {
        self.experts.len()
    }
    pub(crate) fn assignments(&self) -> usize {
        self.assignments
    }
    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = (usize, usize)> + Clone {
        self.experts
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
    }
    pub(crate) fn expert_ids(&self) -> impl ExactSizeIterator<Item = usize> + Clone {
        self.experts.iter().copied()
    }
    #[cfg(test)]
    pub(crate) fn checked<'a>(
        experts: &'a [usize],
        counts: &'a [usize],
    ) -> Result<ExpertGroupCounts<'a>> {
        if experts.len() != counts.len() || experts.windows(2).any(|p| p[0] >= p[1]) {
            return Err(error(
                "expert count membership must be distinct, ascending and match count lengths; supply one count per expert",
            ));
        }
        let assignments = counts
            .iter()
            .try_fold(0usize, |n, &c| n.checked_add(c))
            .ok_or_else(|| error("expert assignment count overflow"))?;
        Ok(ExpertGroupCounts {
            experts,
            counts,
            assignments,
        })
    }
}

pub(super) struct ExpertRoutes<'a> {
    decisions: &'a [Decision],
    order: &'a [usize],
    starts: &'a [usize],
    fanout: NonZeroUsize,
    counts: ExpertGroupCounts<'a>,
}
impl ExpertRoutes<'_> {
    pub(super) fn empty(k: usize) -> Result<Self> {
        let fanout = NonZeroUsize::new(k).ok_or_else(|| error("routing k must be positive"))?;
        Ok(Self {
            decisions: &[],
            order: &[],
            starts: &[0],
            fanout,
            counts: ExpertGroupCounts {
                experts: &[],
                counts: &[],
                assignments: 0,
            },
        })
    }
    pub(super) fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
    pub(super) fn route_count(&self) -> usize {
        self.order.len()
    }
    pub(super) fn counts(&self) -> &ExpertGroupCounts<'_> {
        &self.counts
    }
    pub(super) fn row_selections(
        &self,
        row: usize,
    ) -> Result<impl ExactSizeIterator<Item = (usize, f32)> + Clone> {
        let start = row
            .checked_mul(self.fanout.get())
            .ok_or_else(|| error("routing row offset overflow"))?;
        let end = start
            .checked_add(self.fanout.get())
            .ok_or_else(|| error("routing row end overflow"))?;
        let decisions = self.decisions.get(start..end).ok_or_else(|| {
            error(format!(
                "routing row {row} is outside the prepared decision plan; use a valid input row"
            ))
        })?;
        Ok(decisions.iter().map(|d| (d.expert, d.weight)))
    }
    pub(super) fn groups(&self) -> impl ExactSizeIterator<Item = (usize, ExpertTasks<'_>)> + Clone {
        self.counts.expert_ids().map(move |expert| {
            (
                expert,
                ExpertTasks {
                    decisions: self.decisions,
                    order: &self.order[self.starts[expert]..self.starts[expert + 1]],
                    fanout: self.fanout,
                },
            )
        })
    }
}
#[derive(Clone, Copy)]
pub(super) struct RoutedTask {
    pub(super) row: usize,
    pub(super) slot: usize,
    pub(super) weight: f32,
}
#[derive(Clone, Copy)]
pub(super) struct ExpertTasks<'a> {
    decisions: &'a [Decision],
    order: &'a [usize],
    fanout: NonZeroUsize,
}
impl ExpertTasks<'_> {
    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = RoutedTask> + Clone + '_ {
        self.order.iter().map(move |&slot| RoutedTask {
            row: slot / self.fanout.get(),
            slot,
            weight: self.decisions[slot].weight,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::moe::reference_routing_weights;
    use std::collections::BTreeMap;

    #[test]
    fn admitted_groups_preserve_independent_route_bits_and_expert_row_order() {
        for (rows, experts, k) in [
            (1, 1, 1),
            (3, 2, 1),
            (7, 257, 1),
            (31, 13, 3),
            (5, 7, 7),
            (3, 1024, 2),
        ] {
            for normalize in [false, true] {
                for aggregation in [false, true] {
                    for special in [false, true] {
                        let logits = (0..rows * experts)
                            .map(|i| {
                                if special {
                                    [
                                        0.,
                                        -0.,
                                        f32::INFINITY,
                                        f32::NEG_INFINITY,
                                        f32::from_bits(0x7fc00032),
                                        1.,
                                        1.,
                                    ][i % 7]
                                } else {
                                    ((i * 17) % 23) as f32 / 8. - 1.
                                }
                            })
                            .collect::<Vec<_>>();
                        let weights = (0..rows * experts)
                            .map(|i| {
                                if special {
                                    [0., -0., 1., -1., f32::INFINITY, f32::from_bits(0xffc00051)]
                                        [i % 6]
                                } else {
                                    (i % 5) as f32 / 4.
                                }
                            })
                            .collect::<Vec<_>>();
                        let weights = aggregation.then_some(weights.as_slice());
                        let mut expected = BTreeMap::<usize, Vec<(usize, usize, u32)>>::new();
                        for row in 0..rows {
                            let range = row * experts..(row + 1) * experts;
                            let routes = reference_routing_weights(
                                &logits[range.clone()],
                                weights.map(|w| &w[range]),
                                k,
                                normalize,
                            );
                            for (rank, (expert, weight)) in routes.into_iter().enumerate() {
                                expected.entry(expert).or_default().push((
                                    row,
                                    row * k.min(experts) + rank,
                                    weight.to_bits(),
                                ));
                            }
                        }
                        let layout = RoutingArena::layout(rows, experts, k, aggregation).unwrap();
                        let mut backing = vec![u128::MAX; layout.size().div_ceil(16)];
                        for poison in [u128::MAX, 0xcccccccccccccccccccccccccccccccc] {
                            backing.fill(poison);
                            // SAFETY: this exclusive live aligned Host owner covers
                            // the exact composite; no references escape this loop.
                            let [mut lane] = unsafe {
                                ProjectionScratch::bind_layout(
                                    "routing test",
                                    WorkspaceLayout::new([layout]).unwrap(),
                                    Some(WorkspaceView::new(
                                        DevicePtrMut(backing.as_mut_ptr().cast()),
                                        layout.size(),
                                    )),
                                )
                            }
                            .unwrap();
                            let plan = RoutingArena::bind(rows, experts, k, aggregation, &mut lane)
                                .unwrap()
                                .build(&logits, weights, normalize)
                                .unwrap();
                            assert!(!plan.is_empty());
                            assert_eq!(plan.route_count(), rows * k.min(experts));
                            assert_eq!(plan.groups().len(), expected.len());
                            assert_eq!(plan.counts().len(), expected.len());
                            assert_eq!(plan.counts().assignments(), rows * k.min(experts));
                            assert_eq!(
                                plan.groups().clone().map(|(id, _)| id).collect::<Vec<_>>(),
                                expected.keys().copied().collect::<Vec<_>>()
                            );
                            for row in 0..rows {
                                let range = row * experts..(row + 1) * experts;
                                let old = reference_routing_weights(
                                    &logits[range.clone()],
                                    weights.map(|w| &w[range]),
                                    k,
                                    normalize,
                                );
                                let actual = plan
                                    .row_selections(row)
                                    .unwrap()
                                    .map(|(id, w)| (id, w.to_bits()))
                                    .collect::<Vec<_>>();
                                assert_eq!(
                                    actual,
                                    old.into_iter()
                                        .map(|(id, w)| (id, w.to_bits()))
                                        .collect::<Vec<_>>()
                                );
                            }
                            let actual = plan
                                .groups()
                                .map(|(expert, tasks)| {
                                    (
                                        expert,
                                        tasks
                                            .iter()
                                            .map(|task| {
                                                (task.row, task.slot, task.weight.to_bits())
                                            })
                                            .collect::<Vec<_>>(),
                                    )
                                })
                                .collect::<BTreeMap<_, _>>();
                            assert_eq!(
                                actual, expected,
                                "rows={rows} experts={experts} k={k} normalize={normalize} aggregation={aggregation} special={special}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn expert_count_views_reject_ambiguous_membership_and_preserve_zero_counts() {
        assert!(ExpertGroupCounts::checked(&[3, 3], &[1, 1]).is_err());
        assert!(ExpertGroupCounts::checked(&[9, 3], &[1, 1]).is_err());
        assert!(ExpertGroupCounts::checked(&[3], &[1, 2]).is_err());
        assert!(ExpertGroupCounts::checked(&[3, 9], &[usize::MAX, 1]).is_err());
        let counts = ExpertGroupCounts::checked(&[3, 9], &[3, 0]).unwrap();
        assert_eq!(counts.len(), 2);
        assert_eq!(counts.assignments(), 3);
        assert_eq!(counts.iter().collect::<Vec<_>>(), [(3, 3), (9, 0)]);
        let empty = ExpertRoutes::empty(3).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.groups().len(), 0);
        assert!(empty.row_selections(0).is_err());
    }

    #[test]
    fn typed_layout_is_bounded_and_short_binding_is_transactional() {
        let layout = RoutingArena::layout(3, 2, 1, false).unwrap();
        if std::mem::size_of::<usize>() == 8 {
            assert_eq!(layout.size(), 152);
            assert_eq!(layout.align(), 8);
            assert_eq!(RoutingArena::layout(3, 2, 1, true).unwrap().size(), 144);
        }
        for (rows, experts, k) in [
            (usize::MAX, 2, 1),
            (1, usize::MAX, 1),
            (usize::MAX / 2, 3, 3),
            (1, 0, 1),
            (1, 2, 0),
        ] {
            assert!(RoutingArena::layout(rows, experts, k, false).is_err());
        }
        let mut backing = vec![91u128; layout.size().div_ceil(16)];
        // SAFETY: u128 ownership provides alignment and valid initialized f32
        // representations for this short prefix. Binding must not touch it.
        let short = unsafe {
            std::slice::from_raw_parts_mut(
                backing.as_mut_ptr().cast::<f32>(),
                layout.size() / 4 - 1,
            )
        };
        let mut lane = ProjectionScratch::test_binding(short);
        assert!(RoutingArena::bind(3, 2, 1, false, &mut lane).is_err());
        assert!(backing.iter().all(|&v| v == 91));
    }
}
