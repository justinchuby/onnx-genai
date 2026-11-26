//! Compiled parameter admission for calls handed to an external harness.
use anyhow::{Context, ensure};
use onnx_genai_engine::byte_grammar::{GrammarByteCursor, GrammarCandidate};
use onnx_genai_metadata::{ToolCall, ToolCallFragment};
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};

/// Immutable compiled catalog: active responses share its schemas and original
/// definitions; updating session defaults creates a new catalog, never mutates it.
#[derive(Clone, Default)]
pub(super) struct ToolArguments(Arc<CompiledTools>);
#[derive(Default)]
struct CompiledTools {
    definitions: Vec<Value>,
    arguments: BTreeMap<String, GrammarByteCursor>,
}

/// Proof of provisional schema admission, not a completed executable call.
/// The borrowed source cannot change while a renderer consumes this proof.
pub(super) struct AdmittedFragment<'a>(&'a ToolCallFragment);
impl AdmittedFragment<'_> {
    pub(super) fn fragment(&self) -> &ToolCallFragment {
        self.0
    }
}

impl ToolArguments {
    pub(super) fn compile(tools: &Value, max_bytes: usize) -> anyhow::Result<Self> {
        let mut compiled = BTreeMap::new();
        let definitions = tools
            .as_array()
            .context("realtime tools must be an array")?;
        for tool in definitions {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .context("realtime function tool requires a name string")?;
            let schema = crate::tool_protocol::object_argument_schema(tool.get("parameters"))
                .with_context(|| {
                    format!("resolve realtime tool {name:?} parameters object domain")
                })?;
            let cursor = GrammarByteCursor::json_schema(&schema, max_bytes)
                .with_context(|| format!("compile realtime tool {name:?} parameters; use a schema supported by the constraint backend"))?;
            ensure!(
                compiled.insert(name.to_owned(), cursor).is_none(),
                "realtime tools contain duplicate function name {name:?}; declare each function once"
            );
        }
        Ok(Self(Arc::new(CompiledTools {
            definitions: definitions.clone(),
            arguments: compiled,
        })))
    }
    pub(super) fn definitions(&self) -> &[Value] {
        &self.0.definitions
    }
    /// Admit the whole quantum before publishing any provisional item in it.
    /// Completed batch admission remains a separate operation below.
    pub(super) fn admit_fragments<'a>(
        &self,
        fragments: &'a [ToolCallFragment],
    ) -> anyhow::Result<Vec<AdmittedFragment<'a>>> {
        fragments
            .iter()
            .map(|fragment| {
                let cursor = self.0.arguments.get(&fragment.name).with_context(|| {
                    format!(
                        "model emitted undeclared realtime tool {:?}; repair model tool prompting/constraints",
                        fragment.name
                    )
                })?;
                let viable = cursor.preview_json_root(&fragment.arguments).with_context(|| {
                    format!("validate realtime tool {:?} argument prefix", fragment.name)
                })?;
                ensure!(
                    matches!(viable, GrammarCandidate::Viable { complete }
                        if !fragment.envelope_complete || complete),
                    "model emitted arguments that do not satisfy realtime tool {:?} parameter prefix; repair model tool prompting/constraints",
                    fragment.name
                );
                Ok(AdmittedFragment(fragment))
            })
            .collect()
    }
    /// Validate the actual streamed spelling and prove it describes the same
    /// values as the authoritative complete protocol parse. Re-serializing a
    /// JSON object can reorder keys in an otherwise valid schema prefix.
    pub(super) fn admit_projected(
        &self,
        calls: &[ToolCall],
        fragments: &[ToolCallFragment],
    ) -> anyhow::Result<()> {
        ensure!(
            calls.len() == fragments.len(),
            "realtime completed batch differs from its projected argument count; repair the model protocol parser"
        );
        let projected = calls.iter().zip(fragments).enumerate().map(|(index,(call,fragment))| {
            ensure!(fragment.index == index && fragment.envelope_complete && fragment.name == call.name
                && fragment.id.as_ref().is_none_or(|id| *id == call.id),
                "realtime completed call {index} differs from its projected identity; repair the model protocol parser");
            let raw: Value = serde_json::from_str(&fragment.arguments).context("parse completed realtime argument projection")?;
            let parsed: Value = serde_json::from_str(&call.arguments).context("parse completed realtime protocol arguments")?;
            ensure!(raw.is_object() && raw == parsed, "realtime completed call {index} changed its projected argument value; repair the model protocol parser");
            Ok(ToolCall {id:call.id.clone(),name:call.name.clone(),arguments:fragment.arguments.clone()})
        }).collect::<anyhow::Result<Vec<_>>>()?;
        self.admit(&projected)
    }

    // The entire batch is admitted before completed call events or harness-result admission.
    pub(super) fn admit(&self, calls: &[ToolCall]) -> anyhow::Result<()> {
        for call in calls {
            let cursor = self.0.arguments.get(&call.name).with_context(|| format!(
                "model emitted undeclared realtime tool {:?}; repair model tool prompting/constraints", call.name))?;
            ensure!(
                matches!(
                    cursor
                        .preview_json_root(&call.arguments)
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
    use serde_json::json;
    #[test]
    fn active_catalog_shares_immutable_schemas_while_defaults_are_replaced() {
        let definition = |city| json!([{"name":"lookup","parameters":{"type":"object","properties":{"city":{"const":city}},"required":["city"],"additionalProperties":false}}]);
        let mut defaults = ToolArguments::compile(&definition("Paris"), 4096).unwrap();
        let active = defaults.clone();
        assert!(Arc::ptr_eq(&defaults.0, &active.0));
        defaults = ToolArguments::compile(&definition("Rome"), 4096).unwrap();
        assert_eq!(
            active.definitions(),
            definition("Paris").as_array().unwrap()
        );
        assert_eq!(
            defaults.definitions(),
            definition("Rome").as_array().unwrap()
        );
        let call = |city| ToolCall {
            id: "call1".into(),
            name: "lookup".into(),
            arguments: format!("{{\"city\":\"{city}\"}}"),
        };
        for _ in 0..2 {
            active.admit(&[call("Paris")]).unwrap();
            assert!(active.admit(&[call("Rome")]).is_err());
            defaults.admit(&[call("Rome")]).unwrap();
            assert!(defaults.admit(&[call("Paris")]).is_err());
        }
    }

    #[test]
    fn provisional_arguments_require_viable_schema_and_complete_envelopes_require_eof() {
        let tools = ToolArguments::compile(
            &json!([{"name":"lookup","parameters":{"const":{"city":"Paris"}}}]),
            256,
        )
        .unwrap();
        let fragment = |arguments: &str, envelope_complete| ToolCallFragment {
            index: 0,
            id: None,
            name: "lookup".into(),
            arguments: arguments.into(),
            envelope_complete,
        };
        let prefixes = [fragment(r#"{"city":"Pa\u007"#, false)];
        assert_eq!(tools.admit_fragments(&prefixes).unwrap().len(), 1);
        assert!(
            tools
                .admit_fragments(&[fragment(r#"{"city":"Pa"#, true)])
                .is_err()
        );
        assert!(
            tools
                .admit_fragments(&[prefixes[0].clone(), fragment(r#"{"city":"Rome"}"#, true)])
                .is_err()
        );
        let complete = [fragment(r#"{"city":"Pa\u0072is"}"#, true)];
        assert_eq!(
            tools.admit_fragments(&complete).unwrap()[0].fragment(),
            &complete[0]
        );
        let forged = ToolCall {
            id: "call_0".into(),
            name: "lookup".into(),
            arguments: r#"{"city":"Rome"}"#.into(),
        };
        assert!(
            tools.admit_projected(&[forged], &complete).is_err(),
            "a viable projection cannot authorize a different completed value"
        );
    }

    #[test]
    fn unconstrained_realtime_parameters_still_require_protocol_objects() {
        for parameters in [
            json!(true),
            json!({}),
            json!({"type":["array","object"]}),
            Value::Null,
        ] {
            let tools =
                ToolArguments::compile(&json!([{"name":"lookup","parameters":parameters}]), 4096)
                    .unwrap();
            for arguments in ["[]", "1", "null", r#""text""#] {
                let fragments = [ToolCallFragment {
                    index: 0,
                    id: None,
                    name: "lookup".into(),
                    arguments: arguments.into(),
                    envelope_complete: false,
                }];
                assert!(tools.admit_fragments(&fragments).is_err(), "{arguments}");
            }
            let call = ToolCall {
                id: "call_0".into(),
                name: "lookup".into(),
                arguments: r#"{"x":1}"#.into(),
            };
            tools.admit(&[call]).unwrap();
        }
    }

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
