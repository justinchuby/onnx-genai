use super::*;

fn find_name(names: &[String], candidates: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        let lower = name.to_ascii_lowercase();
        candidates
            .iter()
            .any(|candidate| lower == *candidate || lower.ends_with(&format!(".{candidate}")))
            .then(|| name.clone())
    })
}

fn declared_or_detected_input(
    names: &[String],
    declared: Option<&str>,
    candidates: &[&str],
    field: &str,
) -> anyhow::Result<String> {
    if let Some(name) = declared {
        if names.iter().any(|candidate| candidate == name) {
            return Ok(name.to_owned());
        }
        bail!(
            "native graph metadata io.{field} declares input '{name}', but the graph exposes inputs {names:?}; fix the metadata port name"
        );
    }
    find_name(names, candidates).with_context(|| {
        format!(
            "native graph is missing {}; declare io.{field} explicitly or export one of {candidates:?}",
            candidates.first().copied().unwrap_or("the required input")
        )
    })
}

fn optional_declared_or_detected_input(
    names: &[String],
    declared: Option<&str>,
    candidates: &[&str],
    field: &str,
) -> anyhow::Result<Option<String>> {
    declared
        .map(|name| declared_or_detected_input(names, Some(name), candidates, field))
        .transpose()
        .map(|declared| declared.or_else(|| find_name(names, candidates)))
}

fn declared_or_detected_output(
    names: &[String],
    declared: Option<&str>,
    candidates: &[&str],
    field: &str,
) -> anyhow::Result<String> {
    if let Some(name) = declared {
        if names.iter().any(|candidate| candidate == name) {
            return Ok(name.to_owned());
        }
        bail!(
            "native graph metadata io.{field} declares output '{name}', but the graph exposes outputs {names:?}; fix the metadata port name"
        );
    }
    find_name(names, candidates).with_context(|| {
        format!(
            "native graph is missing {}; declare io.{field} explicitly or export one of {candidates:?}",
            candidates.first().copied().unwrap_or("the required output")
        )
    })
}

fn optional_declared_or_detected_output(
    names: &[String],
    declared: Option<&str>,
    candidates: &[&str],
    field: &str,
) -> anyhow::Result<Option<String>> {
    declared
        .map(|name| declared_or_detected_output(names, Some(name), candidates, field))
        .transpose()
        .map(|declared| declared.or_else(|| find_name(names, candidates)))
}

fn is_past_name(name: &str) -> bool {
    has_past_prefix(name, KvNamingConvention::Dotted)
}

fn is_present_name(name: &str) -> bool {
    has_present_prefix(name, KvNamingConvention::Dotted)
}

fn matching_past_name(output: &str, inputs: &[String]) -> Option<String> {
    matching_past_input(output, inputs, KvNamingConvention::Dotted).cloned()
}
