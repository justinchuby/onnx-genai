use onnx_genai_kv::{KvCacheOps, KvDType, LayerKv, PageTensorConfig, PagedKvCache};

fn config() -> PageTensorConfig {
    PageTensorConfig {
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 2,
        page_size: 2,
        dtype: KvDType::F32,
    }
}

fn layers(base: f32) -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..2)
        .map(|layer| {
            let key = (0..4)
                .map(|i| base + layer as f32 * 100.0 + i as f32)
                .collect();
            let value = (0..4)
                .map(|i| base + layer as f32 * 100.0 + 50.0 + i as f32)
                .collect();
            (key, value)
        })
        .collect()
}

fn borrowed_layers(data: &[(Vec<f32>, Vec<f32>)]) -> Vec<LayerKv<'_>> {
    data.iter()
        .map(|(key, value)| LayerKv { key, value })
        .collect()
}

fn expected_layer(tokens: &[&[(Vec<f32>, Vec<f32>)]], layer_idx: usize, is_key: bool) -> Vec<f32> {
    let mut expected = Vec::new();
    for head in 0..2 {
        for token in tokens {
            let tensor = if is_key {
                &token[layer_idx].0
            } else {
                &token[layer_idx].1
            };
            expected.extend_from_slice(&tensor[head * 2..head * 2 + 2]);
        }
    }
    expected
}

#[test]
fn forked_tensor_sequences_share_then_diverge_with_copy_on_write() {
    let mut cache = PagedKvCache::new_with_tensor_config(config(), 8);
    let parent = cache.create_sequence();

    let t0 = layers(0.0);
    let t1_original = layers(10.0);
    let t2_original = layers(20.0);
    cache
        .append_token_kv(parent, &borrowed_layers(&t0))
        .unwrap();
    cache
        .append_token_kv(parent, &borrowed_layers(&t1_original))
        .unwrap();
    cache
        .append_token_kv(parent, &borrowed_layers(&t2_original))
        .unwrap();

    let shared_pages = cache.page_table.get_sequence(parent).unwrap().to_vec();
    let child = cache.fork(parent, 3).unwrap();
    assert_eq!(
        cache.page_table.get_sequence(child).unwrap(),
        shared_pages.as_slice()
    );
    for page_id in &shared_pages {
        assert_eq!(cache.page_table.pages[page_id].ref_count, 2);
    }

    let parent_t1 = layers(110.0);
    let parent_t2 = layers(120.0);
    let child_t3 = layers(230.0);
    cache
        .write_token_kv(parent, 1, &borrowed_layers(&parent_t1))
        .unwrap();
    cache
        .write_token_kv(parent, 2, &borrowed_layers(&parent_t2))
        .unwrap();
    cache
        .append_token_kv(child, &borrowed_layers(&child_t3))
        .unwrap();

    assert_eq!(cache.len(parent).unwrap(), 3);
    assert_eq!(cache.len(child).unwrap(), 4);
    assert_ne!(
        cache.page_table.get_sequence(parent).unwrap()[0],
        cache.page_table.get_sequence(child).unwrap()[0]
    );
    assert_ne!(
        cache.page_table.get_sequence(parent).unwrap()[1],
        cache.page_table.get_sequence(child).unwrap()[1]
    );

    let parent_materialized = cache.materialize_sequence(parent).unwrap();
    let child_materialized = cache.materialize_sequence(child).unwrap();
    let parent_tokens = [&t0[..], &parent_t1[..], &parent_t2[..]];
    let child_tokens = [&t0[..], &t1_original[..], &t2_original[..], &child_t3[..]];

    assert_eq!(parent_materialized.sequence_len, 3);
    assert_eq!(child_materialized.sequence_len, 4);
    for layer_idx in 0..2 {
        assert_eq!(
            parent_materialized.layers[layer_idx].key,
            expected_layer(&parent_tokens, layer_idx, true)
        );
        assert_eq!(
            parent_materialized.layers[layer_idx].value,
            expected_layer(&parent_tokens, layer_idx, false)
        );
        assert_eq!(
            child_materialized.layers[layer_idx].key,
            expected_layer(&child_tokens, layer_idx, true)
        );
        assert_eq!(
            child_materialized.layers[layer_idx].value,
            expected_layer(&child_tokens, layer_idx, false)
        );
    }
}
