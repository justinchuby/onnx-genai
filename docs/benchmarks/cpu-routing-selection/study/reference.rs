
#[cfg(test)]
thread_local! { static ROUTE_ORDER_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
impl RoutingScratch {
    fn fullsort_weights(
        &mut self,
        logits: &[f32],
        aggregation_weights: Option<&[f32]>,
        k: usize,
        normalize: bool,
    ) -> &mut [(usize, f32)] {
        self.indices.clear();
        self.indices.extend(0..logits.len());
        self.indices.sort_unstable_by(|&a, &b| {
            ROUTE_ORDER_COMPARISONS.with(|count| count.set(count.get() + 1));
            logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b))
        });
        self.indices.truncate(k);
        self.selected.clear();
        if let Some(weights) = aggregation_weights {
            let denominator = if normalize {
                self.indices.iter().map(|&i| weights[i]).sum()
            } else {
                1.0
            };
            self.selected.extend(self.indices.iter().map(|&i| {
                let weight = if denominator == 0.0 {
                    0.0
                } else {
                    weights[i] / denominator
                };
                (i, weight)
            }));
        } else {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            self.exponentials.clear();
            self.exponentials
                .extend(logits.iter().map(|&v| (v - max).exp()));
            let all_sum: f32 = self.exponentials.iter().sum();
            let denominator = if normalize {
                self.indices.iter().map(|&i| self.exponentials[i]).sum()
            } else {
                all_sum
            };
            self.selected.extend(
                self.indices
                    .iter()
                    .map(|&i| (i, self.exponentials[i] / denominator)),
            );
        }
        &mut self.selected
    }
}
