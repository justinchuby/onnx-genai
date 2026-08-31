//! Runtime conformance for a fixed-capacity ("static") KV cache workflow.
//!
//! The other workflow packages grow their cache: the buffer's extent along the
//! sequence axis *is* the length, so nothing has to be declared for a runtime to
//! know what is valid. A static cache separates the two. Its extent is a
//! capacity fixed when the graph was built, its valid region is a prefix named
//! by `logical_lengths`, and each step writes to a destination carried in data.
//!
//! That separation is the whole point — it is what lets rows of unequal length
//! share one rectangular buffer, lets an inactive row's slots be replaced
//! without disturbing what it still holds, and makes rewinding a cursor move
//! rather than buffer surgery. It is also what makes the buffer unsafe without a
//! declaration: an out-of-range destination is a silent out-of-bounds write, not
//! an error, on every execution provider.
//!
//! These tests pin both halves: that the declared contract executes, and that
//! the runtime rejects the writes the declaration makes illegal.

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};
use std::path::PathBuf;

const CAPACITY: usize = 8;
const WIDTH: usize = 4;

fn package() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/static_cache")
}

fn engine() -> anyhow::Result<Engine> {
    Engine::from_dir(&package(), EngineConfig::default())
}

fn request(
    prompts: &[i64],
    prompt_width: i64,
    write_indices: &[i64],
    active: &[bool],
    steps: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    let rows = i64::try_from(write_indices.len())?;
    let options = GenerateOptions {
        max_new_tokens: steps,
        ..GenerateOptions::default()
    };
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options,
    })
    .with_input(
        "request.input_ids",
        Value::from_slice_i64(prompts, &[rows, prompt_width])?,
    )
    .with_input(
        "request.write_indices",
        Value::from_slice_i64(write_indices, &[rows])?,
    )
    .with_input(
        "request.active",
        Value::from_raw_bytes(
            active.iter().map(|value| u8::from(*value)).collect(),
            &[rows],
            DataType::Bool,
        )?,
    )
    .with_input(
        "request.max_iterations",
        Value::from_slice_i64(&[i64::try_from(steps)?], &[1])?,
    ))
}

/// The slot-major view of one row of a returned cache.
fn row_slots(cache: &[f32], row: usize) -> Vec<&[f32]> {
    cache[row * CAPACITY * WIDTH..(row + 1) * CAPACITY * WIDTH]
        .chunks(WIDTH)
        .collect()
}

fn written_slots(cache: &[f32], row: usize) -> Vec<usize> {
    row_slots(cache, row)
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.iter().any(|value| *value != 0.0))
        .map(|(index, _)| index)
        .collect()
}

/// Two rows starting at different cursors share one rectangular buffer, and each
/// writes only its own slots.
///
/// This is the property a growing cache cannot express: with a shared extent and
/// no logical lengths, the shorter row would have to be padded into the longer
/// one's shape and the padding would be indistinguishable from history.
#[test]
fn static_cache_writes_unequal_rows_into_one_fixed_buffer() -> anyhow::Result<()> {
    let mut engine = engine()?;
    // Row 0 starts at slot 0, row 1 at slot 3. Both take a prefill plus 2 steps.
    let output = engine.run_pipeline_retained(request(
        &[1, 2, 3, 4, 5, 6],
        3,
        &[0, 3],
        &[true, true],
        2,
    )?)?;

    assert_eq!(
        output["key_cache"].shape(),
        [2, CAPACITY as i64, WIDTH as i64]
    );
    let keys = output["key_cache"].to_vec_f32()?;
    let values = output["value_cache"].to_vec_f32()?;

    // Prefill writes the starting slot; each of the two body steps writes the
    // next one. Nothing else in the row is touched.
    assert_eq!(written_slots(&keys, 0), vec![0, 1, 2]);
    assert_eq!(written_slots(&keys, 1), vec![3, 4, 5]);
    assert_eq!(written_slots(&values, 0), vec![0, 1, 2]);
    assert_eq!(written_slots(&values, 1), vec![3, 4, 5]);

    // The cursor and the valid length advance together, per row.
    assert_eq!(output["write_indices"].to_vec_i64()?, vec![2, 5]);
    assert_eq!(output["cache_lengths"].to_vec_i64()?, vec![3, 6]);

    // The two caches carry different updates, so a scatter that wrote the same
    // tensor into both would be caught here rather than passing silently.
    let key_slot = row_slots(&keys, 0)[0];
    let value_slot = row_slots(&values, 0)[0];
    assert_ne!(key_slot, value_slot);
    for (key, value) in key_slot.iter().zip(value_slot) {
        assert!((key * key - value).abs() < 1e-6);
    }
    Ok(())
}

/// A row that starts past slot zero leaves the slots below it untouched.
///
/// Capacity is physical and the valid prefix is logical: the runtime allocates
/// the whole buffer but the declaration says which part means anything.
#[test]
fn static_cache_leaves_slots_below_the_cursor_untouched() -> anyhow::Result<()> {
    let mut engine = engine()?;
    let output = engine.run_pipeline_retained(request(&[1, 2, 3], 3, &[4], &[true], 1)?)?;
    let keys = output["key_cache"].to_vec_f32()?;
    assert_eq!(written_slots(&keys, 0), vec![4, 5]);
    assert_eq!(output["cache_lengths"].to_vec_i64()?, vec![6]);
    Ok(())
}

/// An inactive row freezes its valid prefix while its slots above that prefix
/// stay writable.
///
/// That is what makes a static cache compactable in place: the runtime can hand
/// the free slots of an inactive row to something else without having to prove
/// the row is finished with them, because the row's own declaration says they
/// are outside what it still holds.
#[test]
fn static_cache_freezes_inactive_rows_and_frees_their_tail() -> anyhow::Result<()> {
    let mut engine = engine()?;
    let baseline = engine.run_pipeline_retained(request(
        &[1, 2, 3, 4, 5, 6],
        3,
        &[0, 0],
        &[true, true],
        0,
    )?)?;
    let baseline_keys = baseline["key_cache"].to_vec_f32()?;
    let baseline_lengths = baseline["cache_lengths"].to_vec_i64()?;
    assert_eq!(baseline_lengths, vec![1, 1]);

    // Same request, but row 1 is inactive for the loop.
    let output = engine.run_pipeline_retained(request(
        &[1, 2, 3, 4, 5, 6],
        3,
        &[0, 0],
        &[true, false],
        3,
    )?)?;
    let keys = output["key_cache"].to_vec_f32()?;
    let lengths = output["cache_lengths"].to_vec_i64()?;

    // The active row grows; the inactive row's valid length does not move.
    assert_eq!(lengths, vec![4, 1]);

    // Everything inside the inactive row's frozen prefix is byte-identical to
    // what it held when it went inactive.
    let frozen = usize::try_from(lengths[1])?;
    for slot in 0..frozen {
        assert_eq!(
            row_slots(&keys, 1)[slot],
            row_slots(&baseline_keys, 1)[slot],
            "inactive row lost slot {slot} from its valid prefix"
        );
    }
    Ok(())
}

/// Truncating a run reproduces the longer run's cache prefix exactly.
///
/// This is the invariant rewinding depends on: a step never disturbs a slot
/// below the cursor it wrote at, so restoring the cursor and the valid length
/// restores the state. Nothing has to be copied, cleared, or replayed.
#[test]
fn static_cache_rewind_is_a_cursor_move() -> anyhow::Result<()> {
    let mut engine = engine()?;
    let long = engine.run_pipeline_retained(request(&[4, 5, 6], 3, &[0], &[true], 4)?)?;
    let short = engine.run_pipeline_retained(request(&[4, 5, 6], 3, &[0], &[true], 2)?)?;

    let long_keys = long["key_cache"].to_vec_f32()?;
    let short_keys = short["key_cache"].to_vec_f32()?;
    let rewound = usize::try_from(short["cache_lengths"].to_vec_i64()?[0])?;
    assert_eq!(rewound, 3);

    for slot in 0..rewound {
        assert_eq!(
            row_slots(&short_keys, 0)[slot],
            row_slots(&long_keys, 0)[slot],
            "slot {slot} below the rewind point was disturbed by a later step"
        );
    }
    // The longer run really did go further, so the comparison above is not
    // vacuous.
    assert_eq!(long["cache_lengths"].to_vec_i64()?, vec![5]);
    assert_eq!(long["write_indices"].to_vec_i64()?, vec![4]);
    Ok(())
}

/// Repeating a request reproduces its cache and logits bit for bit.
#[test]
fn static_cache_runs_are_deterministic() -> anyhow::Result<()> {
    let mut engine = engine()?;
    let first = engine.run_pipeline_retained(request(
        &[1, 2, 3, 4, 5, 6],
        3,
        &[1, 2],
        &[true, true],
        3,
    )?)?;
    let second = engine.run_pipeline_retained(request(
        &[1, 2, 3, 4, 5, 6],
        3,
        &[1, 2],
        &[true, true],
        3,
    )?)?;
    assert_eq!(
        first["key_cache"].to_vec_f32()?,
        second["key_cache"].to_vec_f32()?
    );
    assert_eq!(
        first["logits"].to_vec_f32()?,
        second["logits"].to_vec_f32()?
    );
    assert_eq!(
        first["write_indices"].to_vec_i64()?,
        second["write_indices"].to_vec_i64()?
    );
    Ok(())
}

/// A destination outside the declared capacity is refused before the graph runs.
///
/// Without the declaration there is nothing to check against and the scatter is
/// undefined behaviour, so this is the test that justifies declaring capacity at
/// all.
#[test]
fn static_cache_rejects_a_destination_outside_capacity() -> anyhow::Result<()> {
    let mut engine = engine()?;
    let Err(error) =
        engine.run_pipeline_retained(request(&[1, 2, 3], 3, &[CAPACITY as i64], &[true], 1)?)
    else {
        anyhow::bail!("a write past capacity must not reach the graph");
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("outside its capacity"),
        "expected a capacity diagnostic, got: {message}"
    );
    assert!(
        message.contains("decoder_cache"),
        "the diagnostic must name the group, got: {message}"
    );
    Ok(())
}

/// A negative destination is refused for the same reason.
#[test]
fn static_cache_rejects_a_negative_destination() -> anyhow::Result<()> {
    let mut engine = engine()?;
    let Err(error) = engine.run_pipeline_retained(request(&[1, 2, 3], 3, &[-1], &[true], 1)?)
    else {
        anyhow::bail!("a negative write destination must not reach the graph");
    };
    assert!(
        format!("{error:#}").contains("outside its capacity"),
        "expected a capacity diagnostic, got: {error:#}"
    );
    Ok(())
}
