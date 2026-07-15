//! Scheduling policy implementations.

use crate::Priority;

const CLASS_COUNT: usize = 3;

/// Deficit-weighted round-robin across priority classes.
///
/// Only backlogged classes accrue credit. A class that becomes idle has its
/// credit reset, so it cannot hoard service and return with an unbounded burst.
/// Weights must be non-zero to preserve the anti-starvation guarantee.
#[derive(Debug, Clone)]
pub struct FairSharePolicy {
    weights: [u32; CLASS_COUNT],
    deficits: [i128; CLASS_COUNT],
    tie_cursor: usize,
}

impl Default for FairSharePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl FairSharePolicy {
    /// Create an equal-share policy.
    pub fn new() -> Self {
        Self::with_weights(1, 1, 1)
    }

    /// Create a policy with weights for `(Low, Normal, High)`.
    ///
    /// # Panics
    ///
    /// Panics if any weight is zero, because a zero-weight class could starve.
    pub fn with_weights(low: u32, normal: u32, high: u32) -> Self {
        assert!(
            low > 0 && normal > 0 && high > 0,
            "fair-share weights must be non-zero"
        );
        Self {
            weights: [low, normal, high],
            deficits: [0; CLASS_COUNT],
            tie_cursor: 0,
        }
    }

    /// Change one class's weight without carrying credit from the old weights.
    ///
    /// # Panics
    ///
    /// Panics if `weight` is zero.
    pub fn set_weight(&mut self, priority: Priority, weight: u32) {
        assert!(weight > 0, "fair-share weights must be non-zero");
        let index = class_index(priority);
        self.weights[index] = weight;
        self.deficits = [0; CLASS_COUNT];
    }

    /// Select the next class from the currently backlogged classes.
    pub fn select<I>(&mut self, active_priorities: I) -> Option<Priority>
    where
        I: IntoIterator<Item = Priority>,
    {
        let mut active = [false; CLASS_COUNT];
        for priority in active_priorities {
            active[class_index(priority)] = true;
        }

        if !active.iter().any(|is_active| *is_active) {
            self.deficits = [0; CLASS_COUNT];
            return None;
        }

        let mut total_weight = 0_i128;
        for index in 0..CLASS_COUNT {
            if active[index] {
                let weight = i128::from(self.weights[index]);
                self.deficits[index] = self.deficits[index].saturating_add(weight);
                total_weight = total_weight.saturating_add(weight);
            } else {
                // Idle classes neither accumulate nor retain burst credit.
                self.deficits[index] = 0;
            }
        }

        let mut selected = None;
        for offset in 0..CLASS_COUNT {
            let index = (self.tie_cursor + offset) % CLASS_COUNT;
            if !active[index] {
                continue;
            }
            selected = match selected {
                Some(current) if self.deficits[current] >= self.deficits[index] => Some(current),
                _ => Some(index),
            };
        }

        let selected = selected.expect("at least one class is active");
        self.deficits[selected] = self.deficits[selected].saturating_sub(total_weight);
        self.tie_cursor = (selected + 1) % CLASS_COUNT;
        Some(priority_from_index(selected))
    }
}

fn class_index(priority: Priority) -> usize {
    priority as usize
}

fn priority_from_index(index: usize) -> Priority {
    match index {
        0 => Priority::Low,
        1 => Priority::Normal,
        2 => Priority::High,
        _ => unreachable!("priority class index is bounded"),
    }
}

// TODO: Implement preemption policies (recompute vs swap)
// TODO: SLA-aware deadline scheduling

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatch_counts(
        policy: &mut FairSharePolicy,
        active: &[Priority],
        steps: usize,
    ) -> [usize; CLASS_COUNT] {
        let mut counts = [0; CLASS_COUNT];
        for _ in 0..steps {
            let selected = policy.select(active.iter().copied()).unwrap();
            counts[class_index(selected)] += 1;
        }
        counts
    }

    #[test]
    fn equal_weights_split_saturating_demand_evenly() {
        let mut policy = FairSharePolicy::new();
        let counts = dispatch_counts(&mut policy, &[Priority::Low, Priority::High], 1_000);

        assert_eq!(counts[class_index(Priority::Low)], 500);
        assert_eq!(counts[class_index(Priority::High)], 500);
    }

    #[test]
    fn three_to_one_weights_split_saturating_demand_proportionally() {
        let mut policy = FairSharePolicy::with_weights(1, 1, 3);
        let counts = dispatch_counts(&mut policy, &[Priority::Low, Priority::High], 400);

        assert_eq!(counts[class_index(Priority::Low)], 100);
        assert_eq!(counts[class_index(Priority::High)], 300);
    }

    #[test]
    fn low_weight_class_has_a_bounded_wait() {
        let mut policy = FairSharePolicy::with_weights(1, 1, 3);
        let mut high_streak = 0;

        for _ in 0..400 {
            match policy.select([Priority::Low, Priority::High]).unwrap() {
                Priority::Low => high_streak = 0,
                Priority::High => {
                    high_streak += 1;
                    assert!(
                        high_streak <= 3,
                        "weight-1 class waited beyond the 3:1 service bound"
                    );
                }
                Priority::Normal => unreachable!(),
            }
        }
    }

    #[test]
    fn idle_class_cannot_hoard_credit_for_a_return_burst() {
        let mut policy = FairSharePolicy::with_weights(1, 1, 3);

        dispatch_counts(&mut policy, &[Priority::Low, Priority::High], 40);
        for _ in 0..10_000 {
            assert_eq!(policy.select([Priority::High]), Some(Priority::High));
        }

        let counts = dispatch_counts(&mut policy, &[Priority::Low, Priority::High], 40);
        assert_eq!(counts[class_index(Priority::Low)], 10);
        assert_eq!(counts[class_index(Priority::High)], 30);
    }

    #[test]
    fn reweighting_resets_all_classes_to_a_fair_baseline() {
        let mut policy = FairSharePolicy::with_weights(1, 1, 10);

        dispatch_counts(&mut policy, &[Priority::Low, Priority::High], 104);
        policy.set_weight(Priority::High, 1);

        let mut previous = None;
        let mut counts = [0; CLASS_COUNT];
        for _ in 0..20 {
            let selected = policy.select([Priority::Low, Priority::High]).unwrap();
            assert_ne!(
                previous,
                Some(selected),
                "reweighting should not leave either class with stale burst credit"
            );
            counts[class_index(selected)] += 1;
            previous = Some(selected);
        }

        assert_eq!(counts[class_index(Priority::Low)], 10);
        assert_eq!(counts[class_index(Priority::High)], 10);
    }

    #[test]
    fn maximum_weights_do_not_overflow_deficit_updates() {
        let mut policy = FairSharePolicy::with_weights(u32::MAX, u32::MAX, u32::MAX);
        let counts = dispatch_counts(
            &mut policy,
            &[Priority::Low, Priority::Normal, Priority::High],
            3_000,
        );

        assert_eq!(counts, [1_000, 1_000, 1_000]);
    }
}
