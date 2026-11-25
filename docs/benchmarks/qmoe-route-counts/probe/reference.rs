
#[cfg(test)]
impl WeightOffloadMetrics {
    fn reference_record_routes(&self, layer: u32, token_counts: &BTreeMap<usize, usize>) {
        let active = token_counts.values().copied().sum::<usize>();
        self.layer_executions.fetch_add(1, Ordering::Relaxed);
        self.active_experts
            .fetch_add(active as u64, Ordering::Relaxed);
        self.unique_experts_per_batch
            .fetch_add(token_counts.len() as u64, Ordering::Relaxed);
        self.routed_tokens
            .fetch_add(active as u64, Ordering::Relaxed);
        let mut totals = self
            .tokens_per_expert
            .lock()
            .expect("weight-offload metrics lock poisoned");
        for (&expert, &tokens) in token_counts {
            let total = totals.entry(expert).or_default();
            *total = total.saturating_add(tokens as u64);
        }

        drop(totals);

        let mut layers = self
            .per_layer
            .lock()
            .expect("weight-offload layer metrics lock poisoned");
        let layer_stats = layers.entry(layer).or_default();
        layer_stats.executions = layer_stats.executions.saturating_add(1);
        layer_stats.active_experts = layer_stats.active_experts.saturating_add(active as u64);
        layer_stats.unique_experts = layer_stats
            .unique_experts
            .saturating_add(token_counts.len() as u64);
        for (&expert, &tokens) in token_counts {
            let total = layer_stats.tokens_per_expert.entry(expert).or_default();
            *total = total.saturating_add(tokens as u64);
        }
    }

}
