//! Regression test: batch>1 must not crash for CNN (vision) models.
//! Validates batch invariance: running N images in a batch produces the
//! same per-image output as running each image individually.

use onnx_runtime_session::{InferenceSession, Tensor};

#[test]
fn mobilenetv2_batch_invariance() {
    let model_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/mobilenetv2/model.onnx"
    );
    if !std::path::Path::new(model_path).exists() {
        eprintln!("SKIP: {model_path} not found");
        return;
    }
    let mut session = InferenceSession::load(model_path).expect("load model");
    let input_name = session.inputs()[0].name.clone();

    // Generate two distinct images
    let shape_1img = [1usize, 3, 224, 224];
    let elems_per_img: usize = shape_1img[1..].iter().product();
    let img_a: Vec<f32> = (0..elems_per_img)
        .map(|i| ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
        .collect();
    let img_b: Vec<f32> = (0..elems_per_img)
        .map(|i| ((i.wrapping_mul(53) % 257) as f32 - 64.0) / 128.0)
        .collect();

    // Run each image individually (batch=1)
    let input_a = Tensor::from_f32(&shape_1img, &img_a).unwrap();
    let out_a = session
        .run(&[(&input_name, &input_a)])
        .expect("batch=1 image A");
    let ref_a = out_a[0].to_vec_f32();
    eprintln!("batch=1 image A: output shape={:?}", out_a[0].shape);

    let input_b = Tensor::from_f32(&shape_1img, &img_b).unwrap();
    let out_b = session
        .run(&[(&input_name, &input_b)])
        .expect("batch=1 image B");
    let ref_b = out_b[0].to_vec_f32();

    // Run both in a batch=2
    let shape_2img = [2usize, 3, 224, 224];
    let mut batch_data = img_a.clone();
    batch_data.extend_from_slice(&img_b);
    let input_batch = Tensor::from_f32(&shape_2img, &batch_data).unwrap();
    let out_batch = session
        .run(&[(&input_name, &input_batch)])
        .expect("batch=2 must not crash");
    eprintln!("batch=2: output shape={:?}", out_batch[0].shape);
    let batch_data = out_batch[0].to_vec_f32();

    // Verify batch invariance: batch[0] == ref_a, batch[1] == ref_b
    let per_image_out = ref_a.len();
    assert_eq!(
        batch_data.len(),
        2 * per_image_out,
        "batch output size mismatch"
    );
    let batch_a = &batch_data[..per_image_out];
    let batch_b = &batch_data[per_image_out..];

    for (i, (&expected, &actual)) in ref_a.iter().zip(batch_a.iter()).enumerate() {
        let diff = (expected - actual).abs();
        assert!(
            diff < 1e-5,
            "Batch invariance violated for image A at index {i}: single={expected}, batch={actual}, diff={diff}"
        );
    }
    for (i, (&expected, &actual)) in ref_b.iter().zip(batch_b.iter()).enumerate() {
        let diff = (expected - actual).abs();
        assert!(
            diff < 1e-5,
            "Batch invariance violated for image B at index {i}: single={expected}, batch={actual}, diff={diff}"
        );
    }
    eprintln!("✓ batch invariance verified for MobileNetV2 at batch=2");
}

#[test]
fn mobilenetv2_batch4_invariance() {
    let model_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/mobilenetv2/model.onnx"
    );
    if !std::path::Path::new(model_path).exists() {
        eprintln!("SKIP: {model_path} not found");
        return;
    }
    let mut session = InferenceSession::load(model_path).expect("load model");
    let input_name = session.inputs()[0].name.clone();
    let elems_per_img: usize = 3 * 224 * 224;

    // Run 4 distinct images individually
    let images: Vec<Vec<f32>> = (0..4)
        .map(|seed| {
            (0..elems_per_img)
                .map(|i| ((i.wrapping_mul(37 + seed * 13) % 257) as f32 - 128.0) / 128.0)
                .collect()
        })
        .collect();

    let refs: Vec<Vec<f32>> = images
        .iter()
        .map(|img| {
            let input = Tensor::from_f32(&[1, 3, 224, 224], img).unwrap();
            let out = session.run(&[(&input_name, &input)]).unwrap();
            out[0].to_vec_f32()
        })
        .collect();

    // Run all 4 in a batch
    let mut batch_data: Vec<f32> = Vec::with_capacity(4 * elems_per_img);
    for img in &images {
        batch_data.extend_from_slice(img);
    }
    let input_batch = Tensor::from_f32(&[4, 3, 224, 224], &batch_data).unwrap();
    let out_batch = session
        .run(&[(&input_name, &input_batch)])
        .expect("batch=4 must not crash");
    let batch_out = out_batch[0].to_vec_f32();
    let per_image_out = refs[0].len();

    for (img_idx, reference) in refs.iter().enumerate() {
        let batch_slice = &batch_out[img_idx * per_image_out..(img_idx + 1) * per_image_out];
        for (i, (&expected, &actual)) in reference.iter().zip(batch_slice.iter()).enumerate() {
            let diff = (expected - actual).abs();
            assert!(
                diff < 1e-5,
                "Batch invariance violated: image {img_idx} index {i}: single={expected}, batch={actual}, diff={diff}"
            );
        }
    }
    eprintln!("✓ batch=4 invariance verified for MobileNetV2");
}

/// Performance characterization: measures native batch scaling.
/// Not a correctness gate — prints results for human review.
#[test]
fn mobilenetv2_batch_scaling_characterization() {
    let model_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/mobilenetv2/model.onnx"
    );
    if !std::path::Path::new(model_path).exists() {
        eprintln!("SKIP: {model_path} not found");
        return;
    }
    let mut session = InferenceSession::load(model_path).expect("load model");
    let input_name = session.inputs()[0].name.clone();
    let elems_per_img: usize = 3 * 224 * 224;
    let batch_sizes: &[usize] = &[1, 2, 4, 8, 16];
    let warmups = 3;
    let runs = 10;

    eprintln!("\nMobileNetV2 native batch scaling (warmups={warmups}, runs={runs}):");
    eprintln!("| batch | median ms | throughput (samples/s) | scaling vs b=1 |");
    eprintln!("|------:|----------:|----------------------:|---------------:|");

    let mut baseline_throughput = 0.0f64;
    for &batch in batch_sizes {
        let shape = vec![batch, 3, 224, 224];
        let data: Vec<f32> = (0..batch * elems_per_img)
            .map(|i| ((i.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
            .collect();
        let input = Tensor::from_f32(&shape, &data).unwrap();

        for _ in 0..warmups {
            session.run(&[(&input_name, &input)]).unwrap();
        }

        let mut latencies = Vec::with_capacity(runs);
        for _ in 0..runs {
            let start = std::time::Instant::now();
            session.run(&[(&input_name, &input)]).unwrap();
            latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = latencies[runs / 2];
        let throughput = batch as f64 / (median / 1000.0);

        if batch == 1 {
            baseline_throughput = throughput;
        }
        let scaling = throughput / baseline_throughput;
        eprintln!("| {batch:5} | {median:9.2} | {throughput:21.1} | {scaling:14.2}× |");
    }
}
