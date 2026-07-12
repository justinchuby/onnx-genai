//! Speculative decoding engine.

// TODO: Implement speculative decoding loop
// - Draft K tokens using producer (draft model / ngram / self-speculative)
// - Verify in one target forward pass
// - Accept/reject using configured acceptance rule
// - Rewind KV cache for rejected tokens
//
// See DESIGN.md §3.5 for detailed design.
