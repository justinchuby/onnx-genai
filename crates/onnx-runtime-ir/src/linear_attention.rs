//! Shared LinearAttention head geometry and ONNX gate contract.

use crate::Node;

#[derive(Clone, Copy, Debug)]
pub struct Heads {
    pub query: usize,
    pub key: usize,
    pub state: usize,
    pub output: usize,
    /// Number of query heads served by one state; zero means one query
    /// head serves several consecutive states instead.
    pub queries_per_state: usize,
}

pub fn has_trajectory(opset: u64) -> bool {
    opset >= 29
}

impl Heads {
    /// `standard_opset` is absent for the contrib operator, which also allows
    /// inverse GQA and derives the number of packed key heads at execution.
    pub fn from_node(node: &Node, standard_opset: Option<u64>) -> Result<Self, String> {
        let positive = |name| {
            node.attr(name)
                .and_then(|a| a.as_int())
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v != 0)
                .ok_or_else(|| format!("{name} must be a positive integer"))
        };
        let query = positive("q_num_heads")?;
        let key = positive("kv_num_heads")?;
        let value = match node.attr("v_num_heads") {
            Some(attr) if standard_opset.is_some_and(has_trajectory) => attr
                .as_int()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| "v_num_heads must be a nonnegative integer".to_string())?,
            _ => 0,
        };
        let state = if value > 0 {
            if query != key || !value.is_multiple_of(key) {
                return Err(format!(
                    "v_num_heads={value} requires q_num_heads={query} to equal kv_num_heads={key} and must be a multiple of kv_num_heads"
                ));
            }
            value
        } else {
            key
        };
        let queries_per_state = if standard_opset.is_some() && value == 0 || query >= state {
            if !query.is_multiple_of(state) {
                return Err(format!(
                    "q_num_heads={query} must be a multiple of kv_num_heads={key}"
                ));
            }
            query / state
        } else {
            if !state.is_multiple_of(query) {
                return Err(format!(
                    "kv_num_heads={key} must be a multiple of q_num_heads={query} (inverse GQA)"
                ));
            }
            0
        };
        Ok(Self {
            query,
            key,
            state,
            output: query.max(state),
            queries_per_state,
        })
    }

    pub fn packed_key_heads(
        self,
        key_hidden: usize,
        d_k: usize,
        standard: bool,
    ) -> Result<usize, String> {
        if d_k == 0 || !key_hidden.is_multiple_of(d_k) {
            return Err(format!(
                "key hidden size {key_hidden} must be divisible by positive key head dimension; got d_k={d_k}"
            ));
        }
        let packed = key_hidden / d_k;
        if standard && packed != self.key {
            return Err(format!(
                "key has {packed} packed heads; expected kv_num_heads={}",
                self.key
            ));
        }
        if packed == 0 || !self.state.is_multiple_of(packed) {
            return Err(format!(
                "the number of packed key heads ({packed}) must be a positive divisor of the state head count ({})",
                self.state
            ));
        }
        Ok(packed)
    }
}

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
