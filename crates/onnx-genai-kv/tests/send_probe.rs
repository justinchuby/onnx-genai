//! Probe: does PageTable/PagedKvCache remain Send + Sync?
fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn page_table_is_send_and_sync() {
    assert_send::<onnx_genai_kv::PageTable>();
    assert_sync::<onnx_genai_kv::PageTable>();
}

#[test]
fn paged_kv_cache_is_send_and_sync() {
    assert_send::<onnx_genai_kv::PagedKvCache>();
    assert_sync::<onnx_genai_kv::PagedKvCache>();
}
