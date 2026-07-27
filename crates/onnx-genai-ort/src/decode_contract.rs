//! Backend-neutral classification of decoder key/value (KV) tensor port names.
//!
//! Decoder-with-past graphs expose their KV cache through paired ports: a
//! `past` input that seeds the cache and a `present` output that carries the
//! grown cache back out. Every decode backend (the engine ORT adapter, the
//! native custom-runtime adapter, and the low-level ORT [`crate::decode`]
//! session) must classify and pair these ports from their tensor names using
//! the same conventions. This module is the single, dependency-neutral home for
//! that pure-string contract so the backends cannot drift apart.
//!
//! The logic is intentionally free of any ORT or engine types: it operates only
//! on `&str` names so both crates can reach it without a new dependency edge.

/// Which prefix conventions the KV-name normalizer recognizes.
///
/// The dotted conventions (`past_key_values.`, `present_key_values.`, `past.`,
/// `present.`) cover the common decoder-with-past exports. Encoder-decoder
/// exports (for example Whisper `past_key_self_%d` / `present_key_self_%d`)
/// additionally use bare `past_` / `present_` prefixes; the engine ORT adapter
/// opts into those with [`KvNamingConvention::DottedAndGeneric`] so its
/// self-attention KV ports pair correctly while cross-attention ports keep a
/// distinct suffix and are never matched as a growing self-KV layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvNamingConvention {
    /// Only the dotted export conventions.
    Dotted,
    /// The dotted conventions plus the generic `past_` / `present_` prefixes.
    DottedAndGeneric,
}

/// Dotted suffix prefixes, ordered most-specific first so that
/// `past_key_values.` is stripped before the shorter `past.` can apply.
const DOTTED_SUFFIX_PREFIXES: [&str; 4] = [
    "past_key_values.",
    "present_key_values.",
    "past.",
    "present.",
];

/// Generic encoder-decoder prefixes, checked last so the dotted conventions keep
/// their more specific pairing.
const GENERIC_SUFFIX_PREFIXES: [&str; 2] = ["past_", "present_"];

/// Dotted `past` prefixes used by the prefix-style classifier.
const DOTTED_PAST_PREFIXES: [&str; 2] = ["past_key_values.", "past."];

/// Dotted `present` prefixes used by the prefix-style classifier.
const DOTTED_PRESENT_PREFIXES: [&str; 2] = ["present_key_values.", "present."];

fn generic_prefixes(convention: KvNamingConvention) -> &'static [&'static str] {
    match convention {
        KvNamingConvention::Dotted => &[],
        KvNamingConvention::DottedAndGeneric => &GENERIC_SUFFIX_PREFIXES,
    }
}

/// Normalize a KV port name to its layer suffix by stripping the recognized
/// `past`/`present` prefix. Returns `None` when the name does not carry a known
/// prefix. Matching a present output to its past input reduces to both names
/// producing the same suffix.
pub fn kv_suffix(name: &str, convention: KvNamingConvention) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    for prefix in DOTTED_SUFFIX_PREFIXES
        .iter()
        .chain(generic_prefixes(convention).iter())
    {
        if let Some(suffix) = lower.strip_prefix(prefix) {
            return Some(suffix.to_string());
        }
    }
    None
}

/// Find the past input whose layer suffix matches `present_name`.
pub fn matching_past_input<'a>(
    present_name: &str,
    inputs: &'a [String],
    convention: KvNamingConvention,
) -> Option<&'a String> {
    let present_suffix = kv_suffix(present_name, convention)?;
    inputs
        .iter()
        .find(|input| kv_suffix(input, convention).as_deref() == Some(present_suffix.as_str()))
}

/// "Contains" style classification: `true` when the name looks like a past KV
/// input (contains `past` and either `key` or `value`, case-insensitively).
pub fn name_contains_past_key_value(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("past") && (lower.contains("key") || lower.contains("value"))
}

/// "Contains" style classification: `true` when the name looks like a present
/// KV output (contains `present` and either `key` or `value`).
pub fn name_contains_present_key_value(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("present") && (lower.contains("key") || lower.contains("value"))
}

/// "Prefix" style classification: `true` when the name begins with a recognized
/// `past` prefix.
pub fn has_past_prefix(name: &str, convention: KvNamingConvention) -> bool {
    let lower = name.to_ascii_lowercase();
    DOTTED_PAST_PREFIXES
        .iter()
        .chain(generic_prefixes(convention).iter().take(1))
        .any(|prefix| lower.starts_with(prefix))
}

/// "Prefix" style classification: `true` when the name begins with a recognized
/// `present` prefix.
pub fn has_present_prefix(name: &str, convention: KvNamingConvention) -> bool {
    let lower = name.to_ascii_lowercase();
    DOTTED_PRESENT_PREFIXES
        .iter()
        .chain(generic_prefixes(convention).iter().skip(1))
        .any(|prefix| lower.starts_with(prefix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotted_suffix_pairs_past_and_present() {
        assert_eq!(
            kv_suffix("present_key_values.3.key", KvNamingConvention::Dotted),
            Some("3.key".to_string())
        );
        assert_eq!(
            kv_suffix("past_key_values.3.key", KvNamingConvention::Dotted),
            Some("3.key".to_string())
        );
        assert_eq!(
            kv_suffix("present.2.value", KvNamingConvention::Dotted),
            Some("2.value".to_string())
        );
    }

    #[test]
    fn generic_prefixes_only_apply_when_opted_in() {
        assert_eq!(
            kv_suffix("past_key_self_0", KvNamingConvention::Dotted),
            None
        );
        assert_eq!(
            kv_suffix("past_key_self_0", KvNamingConvention::DottedAndGeneric),
            Some("key_self_0".to_string())
        );
        assert_eq!(
            kv_suffix("present_key_self_0", KvNamingConvention::DottedAndGeneric),
            Some("key_self_0".to_string())
        );
    }

    #[test]
    fn matching_past_input_pairs_by_suffix() {
        let inputs = vec![
            "past_key_values.0.key".to_string(),
            "past_key_values.0.value".to_string(),
        ];
        assert_eq!(
            matching_past_input(
                "present_key_values.0.value",
                &inputs,
                KvNamingConvention::Dotted
            ),
            Some(&inputs[1])
        );
    }

    #[test]
    fn encoder_decoder_self_and_cross_stay_distinct() {
        let inputs = vec![
            "past_key_self_0".to_string(),
            "past_key_cross_0".to_string(),
        ];
        assert_eq!(
            matching_past_input(
                "present_key_self_0",
                &inputs,
                KvNamingConvention::DottedAndGeneric
            ),
            Some(&inputs[0])
        );
        // A cross-attention present has no self-KV past to grow.
        assert_eq!(
            matching_past_input(
                "present_key_cross_0",
                &inputs,
                KvNamingConvention::DottedAndGeneric
            ),
            Some(&inputs[1])
        );
    }

    #[test]
    fn contains_classifiers_match_expected_names() {
        assert!(name_contains_past_key_value(
            "decoder.past_key_values.0.key"
        ));
        assert!(name_contains_present_key_value("present.0.value"));
        assert!(!name_contains_present_key_value("logits"));
        assert!(!name_contains_past_key_value("input_ids"));
    }

    #[test]
    fn prefix_classifiers_are_dotted_only_by_default() {
        assert!(has_past_prefix(
            "past_key_values.0.key",
            KvNamingConvention::Dotted
        ));
        assert!(has_present_prefix(
            "present.0.value",
            KvNamingConvention::Dotted
        ));
        // Generic prefixes are not recognized under the dotted convention.
        assert!(!has_past_prefix(
            "past_key_self_0",
            KvNamingConvention::Dotted
        ));
        assert!(has_past_prefix(
            "past_key_self_0",
            KvNamingConvention::DottedAndGeneric
        ));
    }
}
