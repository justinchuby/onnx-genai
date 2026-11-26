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
            let canonical = self.weights.bytes(weight).ok_or_else(|| {
                SessionError::UnsupportedCpuWorkerPlan {
                    reason: format!(
                        "initializer value#{} cannot resolve its canonical mapped byte range",
                        value.0
                    ),
                }
            })?;
            if buffer.as_ptr().cast::<u8>() != canonical.as_ptr() || buffer.len() != canonical.len()
            {
                return Err(SessionError::UnsupportedCpuWorkerPlan {
                    reason: format!(
                        "initializer value#{} borrowed address/byte extent does not equal its canonical WeightStore mapping ({} buffer bytes versus {} canonical bytes); rebuild the source executor with exact read-only mapped initializer ranges",
                        value.0,
                        buffer.len(),
                        canonical.len()
                    ),
                });
            }
        }
        // All initializer payloads are external descriptors. Heap-backed value
        // attributes, nested bodies and functions were refused before cloning.
        // Scalar/type-descriptor IR metadata is not covered by a payload lease.
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
            // This is a value-kind boundary, not an operator/content heuristic.
            // In particular STRING can contain an opaque compiled blob, and
            // INTS/FLOATS can contain numerical model payloads. Even an empty
            // instance belongs to an unsupported ownership kind.
            let owns_value_payload = match attribute {
                Attribute::Int(_)
                | Attribute::Float(_)
                | Attribute::TypeProto(_)
                | Attribute::TypeProtos(_) => false,
                Attribute::Ref { .. } => {
                    return Err(refusal(format!(
                        "node {:?} attribute {name:?} retains an unresolved function-attribute reference; lower function attributes before requesting CPU workers",
                        node.name
                    )));
                }
                Attribute::String(_)
                | Attribute::Ints(_)
                | Attribute::Floats(_)
                | Attribute::Strings(_)
                | Attribute::Tensor(_)
                | Attribute::Tensors(_)
                | Attribute::SparseTensor(_)
                | Attribute::SparseTensors(_)
                | Attribute::Graph(_)
                | Attribute::Graphs(_) => true,
            };
            if owns_value_payload {
                return Err(refusal(format!(
                    "node {:?} attribute {name:?} of kind {:?} owns heap-backed value payload; this profile admits only scalar/type-descriptor attributes and cannot clone value payload under the shared fixed-weight reservation; keep one executor or use an ownership-admitted representation",
                    node.name,
                    attribute.kind()
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

#[cfg(test)]
mod mapped_range_tests {
    use super::*;

    struct FixtureFile(std::path::PathBuf);
    impl Drop for FixtureFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn borrowed_same_bytes_wrong_region_copy_or_extent_cannot_mint_worker_plan() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "nxrt-worker-mapped-range-{}-{unique}.bin",
            std::process::id()
        ));
        // Repeated values make a content-only comparison insufficient: the
        // second region has identical bytes but belongs to a different range.
        let values = [2_f32, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 3.0];
        std::fs::write(
            &path,
            values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let fixture = FixtureFile(path.clone());
        let mut weights = WeightStore::new();
        weights.map_external(&path).unwrap();
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let input = graph.create_named_value(
            "input",
            DataType::Float32,
            onnx_runtime_ir::static_shape([1, 2]),
        );
        let weight = graph.create_named_value(
            "weight",
            DataType::Float32,
            onnx_runtime_ir::static_shape([2, 2]),
        );
        let output = graph.create_named_value(
            "output",
            DataType::Float32,
            onnx_runtime_ir::static_shape([1, 2]),
        );
        graph.add_input(input);
        graph.add_output(output);
        graph.set_initializer(
            weight,
            WeightRef::External {
                path,
                offset: 0,
                length: 16,
                dtype: DataType::Float32,
                dims: vec![2, 2],
            },
        );
        graph.insert_node(Node::new(
            NodeId(0),
            "MatMul",
            vec![Some(input), Some(weight)],
            vec![output],
        ));
        let provider =
            Arc::new(CpuExecutionProvider::initialized_with_resources(Default::default()).unwrap());
        let mut executor = Executor::build(graph, Arc::new(weights), provider, None).unwrap();
        executor.validate_cpu_worker_ownership().unwrap();
        let descriptor = executor.graph.initializers.get(&weight).unwrap();
        let canonical = executor.weights.bytes(descriptor).unwrap();
        let canonical_ptr = canonical.as_ptr();
        let canonical_len = canonical.len();
        let same_contents_elsewhere = WeightRef::External {
            path: fixture.0.clone(),
            offset: 16,
            length: 16,
            dtype: DataType::Float32,
            dims: vec![2, 2],
        };
        assert_eq!(
            canonical,
            executor.weights.bytes(&same_contents_elsewhere).unwrap()
        );
        let original = executor.buffers.get(&weight).unwrap();
        let device = original.device();
        // SAFETY: the mmap spans 32 bytes. Each range stays within the live
        // read-only mapping, has float alignment, and is never executed/written.
        let wrong_region = unsafe {
            onnx_runtime_ep_api::DeviceBuffer::from_borrowed_parts(
                canonical_ptr.add(16).cast_mut().cast(),
                device,
                canonical_len,
                4,
            )
        };
        let copy = [2_f32, 0.0, 0.0, 3.0];
        // SAFETY: copy is live throughout all validation and handle disposal;
        // its f32 storage guarantees 4-byte alignment and is never written.
        let copied_payload = unsafe {
            onnx_runtime_ep_api::DeviceBuffer::from_borrowed_parts(
                copy.as_ptr().cast_mut().cast(),
                device,
                canonical_len,
                4,
            )
        };
        // SAFETY: both ranges stay within the live 32-byte mmap. They are
        // intentionally wrong logical extents and never reach execution.
        let short = unsafe {
            onnx_runtime_ep_api::DeviceBuffer::from_borrowed_parts(
                canonical_ptr.cast_mut().cast(),
                device,
                8,
                4,
            )
        };
        let long = unsafe {
            onnx_runtime_ep_api::DeviceBuffer::from_borrowed_parts(
                canonical_ptr.cast_mut().cast(),
                device,
                32,
                4,
            )
        };
        for invalid in [wrong_region, copied_payload, short, long] {
            assert!(invalid.is_borrowed());
            let original = executor.buffers.insert(weight, invalid).unwrap();
            let rejected = executor.cpu_worker_parts().map(|_| ());
            let invalid = executor.buffers.insert(weight, original).unwrap();
            executor.ep.deallocate(invalid).unwrap();
            let error = rejected
                .expect_err("a borrowed marker alone cannot prove canonical weight ownership");
            assert!(
                error.to_string().contains("canonical WeightStore mapping"),
                "{error}"
            );
            executor.validate_cpu_worker_ownership().unwrap();
        }
        // Restore the exact initializer after every negative and prove the
        // original executor still runs, rather than merely clearing an error.
        let input = crate::Tensor::from_f32(&[1, 2], &[2.0, 5.0]).unwrap();
        let result = executor.run(&[("input", &input)]).unwrap();
        assert_eq!(result[0].try_as_slice_f32().unwrap(), &[4.0, 15.0]);
        drop(result);
        drop(executor);
        drop(fixture);
    }
}

#[cfg(test)]
mod attribute_value_kind_tests {
    use super::*;

    #[test]
    fn opaque_and_numeric_value_lists_are_refused_by_kind_on_same_identity_operator() {
        for (name, attribute) in [
            ("opaque_blob", Attribute::String(vec![0, 0xff, 0x7f, 0x80])),
            (
                "numeric_coefficients",
                Attribute::Ints(vec![65_537, -32_769]),
            ),
            (
                "floating_coefficients",
                Attribute::Floats(vec![0.125, -3.25]),
            ),
            (
                "opaque_parts",
                Attribute::Strings(vec![b"compiled-payload".to_vec(), vec![0xff, 0]]),
            ),
            ("empty_blob", Attribute::String(vec![])),
            ("empty_integer_values", Attribute::Ints(vec![])),
            ("empty_float_values", Attribute::Floats(vec![])),
            ("empty_blob_list", Attribute::Strings(vec![])),
        ] {
            let mut graph = Graph::new();
            let mut node = Node::new(NodeId(0), "Identity", vec![], vec![]);
            node.attributes.insert(name.into(), attribute);
            graph.insert_node(node);
            let error =
                validate_graph(&graph).expect_err("owned value kinds must not reach Graph::clone");
            let text = error.to_string();
            assert!(
                text.contains(name) && text.contains("heap-backed value payload"),
                "{text}"
            );
        }
    }

    #[test]
    fn scalar_and_type_descriptors_remain_explicit_unmetered_ir_metadata() {
        let mut graph = Graph::new();
        let mut node = Node::new(NodeId(0), "Identity", vec![], vec![]);
        node.attributes
            .insert("integer_metadata".into(), Attribute::Int(42));
        node.attributes
            .insert("float_metadata".into(), Attribute::Float(0.125));
        node.attributes.insert(
            "type_metadata".into(),
            Attribute::TypeProto(onnx_runtime_ir::onnx::TypeProto::default()),
        );
        node.attributes.insert(
            "type_list_metadata".into(),
            Attribute::TypeProtos(vec![
                onnx_runtime_ir::onnx::TypeProto::default(),
                onnx_runtime_ir::onnx::TypeProto::default(),
            ]),
        );
        graph.insert_node(node);
        // Ownership admission is separate from operator/schema validation.
        // This does not claim Identity accepts these invented attribute names.
        validate_graph(&graph).unwrap();
    }
}
