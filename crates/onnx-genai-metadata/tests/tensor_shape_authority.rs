use onnx_genai_metadata::{TensorContract, TensorDimension};

fn contract(shape: Vec<TensorDimension>) -> TensorContract {
    TensorContract {
        dtype: "float32".to_string(),
        shape,
        optional: false,
        batch_layout: Default::default(),
        padding: Vec::new(),
    }
}

#[test]
fn scalar_shape_round_trips_and_derives_rank_zero() {
    let contract = contract(Vec::new());
    let json = serde_json::to_value(&contract).expect("contract serializes");

    assert_eq!(json["shape"], serde_json::json!([]));
    assert!(json.get("rank").is_none());

    let round_trip: TensorContract = serde_json::from_value(json).expect("contract deserializes");
    assert_eq!(round_trip, contract);
    assert_eq!(round_trip.rank(), 0);
}

#[test]
fn fixed_symbolic_and_independent_any_dimensions_round_trip() {
    let contract = contract(vec![
        TensorDimension::Fixed(2),
        TensorDimension::Symbol("sequence".to_string()),
        TensorDimension::Any,
        TensorDimension::Any,
    ]);
    let json = serde_json::to_value(&contract).expect("contract serializes");

    assert_eq!(
        json["shape"],
        serde_json::json!([2, "sequence", "Any", "Any"])
    );

    let round_trip: TensorContract = serde_json::from_value(json).expect("contract deserializes");
    assert_eq!(round_trip, contract);
    assert_eq!(round_trip.rank(), 4);
    assert!(matches!(round_trip.shape[2], TensorDimension::Any));
    assert!(matches!(round_trip.shape[3], TensorDimension::Any));
}

#[test]
fn omitted_shape_and_retired_rank_are_not_tensor_contracts() {
    let missing_shape = serde_json::json!({"dtype": "float32"});
    let retired_rank = serde_json::json!({"dtype": "float32", "rank": 1, "shape": ["Any"]});

    assert!(serde_json::from_value::<TensorContract>(missing_shape).is_err());
    assert!(serde_json::from_value::<TensorContract>(retired_rank).is_err());
}
