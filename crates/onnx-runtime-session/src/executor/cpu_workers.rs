//! Ownership admission for the initial external-mapped CPU worker profile.
use super::*;

impl Executor {
    pub(crate) fn cpu_worker_parts(&self) -> Result<(Graph, Arc<WeightStore>)> {
        self.validate_cpu_worker_ownership()?;
        Ok((self.graph.clone(), Arc::clone(&self.weights)))
    }

    pub(crate) fn validate_cpu_worker_ownership(&self) -> Result<()> {
        if self.heterogeneous.is_some() || self.ep.device_type() != DeviceType::Cpu {
            return Err(SessionError::UnsupportedCpuWorkerPlan {
                reason: "the loaded executor is heterogeneous or is not CPU; GPU/provider state cannot be shared by this CPU-only plan".into(),
            });
        }
        if self.captured_segment_count() != 0 {
            return Err(SessionError::UnsupportedCpuWorkerPlan {
                reason: "the loaded executor owns captured device graphs; capture ownership requires a separate worker implementation".into(),
            });
        }
        validate_graph(&self.graph)?;
        for (&value, weight) in &self.graph.initializers {
            if weight.dtype() == DataType::String
                || !super::kernel_cache::is_immutable_initializer(&self.graph, value)
            {
                return Err(SessionError::UnsupportedCpuWorkerPlan {
                    reason: format!(
                        "initializer value#{} is string data or mutable state; only immutable numeric mapped weights are admitted",
                        value.0
                    ),
                });
            }
            let WeightRef::External { .. } = weight else {
                return Err(SessionError::UnsupportedCpuWorkerPlan {
                    reason: format!(
                        "initializer value#{} owns inline payload; graph cloning would copy weights without a second admission, so export external tensor data",
                        value.0
                    ),
                });
            };
            if self.weights.external_mmap_provenance(weight).is_none() {
                return Err(SessionError::UnsupportedCpuWorkerPlan {
                    reason: format!(
                        "initializer value#{} has no validated live external mapping",
                        value.0
                    ),
                });
            }
            let buffer =
                self.buffers
                    .get(&value)
                    .ok_or_else(|| SessionError::UnsupportedCpuWorkerPlan {
                        reason: format!(
                            "initializer value#{} has no admitted CPU backing buffer",
                            value.0
                        ),
                    })?;
            if !buffer.is_borrowed() || !buffer.device().is_host_accessible() {
                return Err(SessionError::UnsupportedCpuWorkerPlan {
                    reason: format!(
                        "initializer value#{} uses an owned or non-host buffer; this profile shares only borrowed read-only external weights",
                        value.0
                    ),
                });
            }
        }
        // All initializer payloads are external descriptors. Tensor attributes,
        // nested bodies and functions were refused before this structural clone.
        Ok(())
    }
}

fn validate_graph(graph: &Graph) -> Result<()> {
    let refusal = |reason: String| SessionError::UnsupportedCpuWorkerPlan { reason };
    if !graph.subgraphs.is_empty()
        || !graph.model_functions.is_empty()
        || !graph.sparse_initializers.is_empty()
    {
        return Err(refusal("nested graphs, local functions or sparse initializers have payload/child-executor ownership not yet covered by this profile; lower them before requesting a CPU worker plan".into()));
    }
    if onnx_runtime_loader::ep_context_nodes(graph)
        .next()
        .is_some()
    {
        return Err(refusal("EPContext payloads require provider-specific rehydration and are not admitted by the CPU worker plan".into()));
    }
    for (_, node) in graph.nodes.iter() {
        if node.is_default_domain() && node.op_type == "Constant" {
            return Err(refusal(format!(
                "node {:?} is Constant; its compiled kernel owns a copied payload, so fold it into external initializers before requesting workers",
                node.name
            )));
        }
        for (name, attribute) in &node.attributes {
            if matches!(
                attribute,
                Attribute::Tensor(_)
                    | Attribute::Tensors(_)
                    | Attribute::SparseTensor(_)
                    | Attribute::SparseTensors(_)
                    | Attribute::Graph(_)
                    | Attribute::Graphs(_)
            ) {
                return Err(refusal(format!(
                    "node {:?} attribute {name:?} owns tensor or graph payload; this profile cannot clone that payload under the shared fixed-weight reservation",
                    node.name
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tensor_payload_guard_is_not_an_operator_name_allowlist() {
        let mut graph = Graph::new();
        let mut node = Node::new(NodeId(0), "Identity", vec![], vec![]);
        node.attributes.insert(
            "owned_payload".into(),
            Attribute::Tensor(onnx_runtime_ir::TensorData::from_raw(
                DataType::Float32,
                vec![1],
                1_f32.to_le_bytes().to_vec(),
            )),
        );
        graph.insert_node(node);
        let error = validate_graph(&graph).unwrap_err();
        assert!(error.to_string().contains("owned_payload"));
    }
}

#[cfg(test)]
mod constant_profile_tests {
    use super::*;
    #[test]
    fn constant_kernel_payload_is_refused_before_graph_clone() {
        let mut graph = Graph::new();
        let mut node = Node::new(NodeId(0), "Constant", vec![], vec![]);
        node.attributes
            .insert("value_float".into(), Attribute::Float(1.0));
        graph.insert_node(node);
        let error = validate_graph(&graph).unwrap_err();
        assert!(error.to_string().contains("Constant"));
        assert!(error.to_string().contains("external initializers"));
    }
}
