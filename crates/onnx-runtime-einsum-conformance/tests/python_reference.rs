use onnx_runtime_einsum_conformance::{
    ComparisonMode, PythonEngine, PythonReferenceAdapter, ReferenceAdapterError, SchemaAuthority,
    compare, evaluate, materialize_inputs, named_cases,
};

#[test]
fn installed_onnx_reference_and_ort_adapters_match_the_direct_oracle_when_available() {
    SchemaAuthority::verify().unwrap();
    let adapter = PythonReferenceAdapter::default();
    let probe = match adapter.probe() {
        Ok(probe) => probe,
        Err(ReferenceAdapterError::Spawn { .. }) => {
            eprintln!("python3 is unavailable; optional reference adapter skipped");
            return;
        }
        Err(error) => panic!("adapter probe failed: {error}"),
    };
    if probe.status != onnx_runtime_einsum_conformance::AdapterStatus::Available {
        eprintln!("ONNX ReferenceEvaluator unavailable: {:?}", probe.reason);
        return;
    }
    assert!(probe.latest_einsum_schema.is_some());
    if probe.latest_einsum_schema.unwrap() < 28 {
        assert_eq!(SchemaAuthority::since_version(28).unwrap(), 28);
    }

    for id in ["bilinear-dot-f32", "bilinear-dot-f16"] {
        let case = named_cases()
            .into_iter()
            .find(|case| case.id == id)
            .unwrap();
        let inputs = materialize_inputs(&case).unwrap();
        let expected = evaluate(&case, &inputs).unwrap();
        let reference = adapter
            .run(PythonEngine::OnnxReference, &case, &inputs)
            .unwrap();
        compare(&case, &expected, &reference, ComparisonMode::ConditionAware).unwrap();
        if probe.onnxruntime_version.is_some() {
            let ort = adapter
                .run(PythonEngine::OnnxRuntime, &case, &inputs)
                .unwrap();
            compare(&case, &expected, &ort, ComparisonMode::ConditionAware).unwrap();
        }
    }
}

#[test]
fn stale_installed_onnx_never_silently_reinterprets_bf16_as_einsum_12() {
    let adapter = PythonReferenceAdapter::default();
    let Ok(probe) = adapter.probe() else {
        return;
    };
    if probe.latest_einsum_schema.unwrap_or(0) >= 28 {
        return;
    }
    let case = named_cases()
        .into_iter()
        .find(|case| case.id == "bf16-finite-identity")
        .unwrap();
    let inputs = materialize_inputs(&case).unwrap();
    let error = adapter
        .run(PythonEngine::OnnxReference, &case, &inputs)
        .unwrap_err();
    let ReferenceAdapterError::Unavailable(reason) = error else {
        panic!("stale ONNX should be rejected by schema probe, got {error}");
    };
    assert!(reason.contains("refusing to reinterpret"), "{reason}");
}
