//! Compiled parameter admission for calls handed to an external harness.
use anyhow::{Context, ensure};
use onnx_genai_engine::byte_grammar::{GrammarByteCursor, GrammarCandidate};
use onnx_genai_metadata::ToolCall;
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct ToolArguments(BTreeMap<String, GrammarByteCursor>);
impl ToolArguments {
    pub(super) fn compile(tools: &Value, max_bytes: usize) -> anyhow::Result<Self> {
        let mut compiled = BTreeMap::new();
        for tool in tools
            .as_array()
            .context("realtime tools must be an array")?
        {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .context("realtime function tool requires a name string")?;
            let default_schema = json!({"type":"object"});
            let schema = tool.get("parameters").unwrap_or(&default_schema);
            let cursor = GrammarByteCursor::json_schema(schema, max_bytes)
                .with_context(|| format!("compile realtime tool {name:?} parameters; use a schema supported by the constraint backend"))?;
            ensure!(
                compiled.insert(name.to_owned(), cursor).is_none(),
                "realtime tools contain duplicate function name {name:?}; declare each function once"
            );
        }
        Ok(Self(compiled))
    }
    // The entire batch is admitted before any call events reach the harness.
    pub(super) fn admit(&self, calls: &[ToolCall]) -> anyhow::Result<()> {
        for call in calls {
            let cursor = self.0.get(&call.name).with_context(|| format!(
                "model emitted undeclared realtime tool {:?}; repair model tool prompting/constraints", call.name))?;
            ensure!(
                matches!(
                    cursor
                        .preview_root(call.arguments.as_bytes())
                        .with_context(|| format!(
                            "validate realtime tool {:?} argument schema",
                            call.name
                        ))?,
                    GrammarCandidate::Viable { complete: true }
                ),
                "model emitted arguments that do not satisfy realtime tool {:?} parameters; repair model tool prompting/constraints",
                call.name
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_schema_and_duplicate_names_fail_during_configuration() {
        let invalid = ToolArguments::compile(
            &json!([{"name":"lookup","parameters":{"type":"not-a-json-type"}}]),
            4096,
        );
        assert!(
            invalid
                .err()
                .unwrap()
                .to_string()
                .contains("compile realtime tool \"lookup\" parameters")
        );
        let duplicate = ToolArguments::compile(&json!([{"name":"lookup"},{"name":"lookup"}]), 4096);
        assert!(
            duplicate
                .err()
                .unwrap()
                .to_string()
                .contains("duplicate function name")
        );
    }
    #[test]
    fn arguments_require_complete_nested_schema_and_reusable_batch_admission() {
        let tools = json!([{"name":"lookup","parameters":{"type":"object","properties":{"city":{"type":"string","enum":["Paris"]},"days":{"type":"array","items":{"type":"integer"},"minItems":1}},"required":["city","days"],"additionalProperties":false}}]);
        let admitted = ToolArguments::compile(&tools, 4096).unwrap();
        let call = |arguments: &str| ToolCall {
            id: "call1".into(),
            name: "lookup".into(),
            arguments: arguments.into(),
        };
        let valid = call(r#"{"city":"Paris","days":[1,2]}"#);
        admitted.admit(std::slice::from_ref(&valid)).unwrap();
        for bad in [
            r#"{"city":"Rome","days":[1]}"#,
            r#"{"city":"Paris"}"#,
            r#"{"city":"Paris","days":["x"]}"#,
            r#"{"city":"Paris","days":[]}"#,
            r#"{"city":"Paris","days":[1],"extra":true}"#,
            r#"{"city":"Paris","days":[1]"#,
        ] {
            assert!(
                admitted
                    .admit(&[valid.clone(), call(bad)])
                    .unwrap_err()
                    .to_string()
                    .contains("parameters"),
                "{bad}"
            );
        }
        admitted.admit(&[valid]).unwrap();
    }
}
