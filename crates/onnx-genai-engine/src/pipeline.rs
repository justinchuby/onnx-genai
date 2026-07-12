//! Multi-model pipeline orchestrator.

// TODO: Implement pipeline execution based on metadata.pipeline spec
// - Dataflow wiring between models
// - Phase gating (prompt_only, always, final_only)
// - Cross-model tensor passing (with device placement awareness)
//
// See DESIGN.md §3.7 for detailed design.
