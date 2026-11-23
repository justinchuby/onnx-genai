//! Shared ONNX LinearAttention-27 gate contract.

/// Return an actionable explanation if the mode's optional operands are wrong.
pub fn gate_error(mode: &str, has_decay: bool, has_beta: bool) -> Option<String> {
    let (needs_decay, needs_beta) = match mode {
        "linear" => (false, false),
        "gated" => (true, false),
        "delta" => (false, true),
        "gated_delta" => (true, true),
        _ => {
            return Some(format!(
                "update_rule={mode:?} must be linear, gated, delta, or gated_delta"
            ));
        }
    };
    [
        ("decay", has_decay, needs_decay),
        ("beta", has_beta, needs_beta),
    ]
    .into_iter()
    .find(|&(_, present, required)| present != required)
    .map(|(input, _, required)| {
        let action = if required { "requires" } else { "forbids" };
        format!("update_rule={mode:?} {action} the {input} input")
    })
}
