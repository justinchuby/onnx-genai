//! CPU reference checks for the corresponding CUDA fixtures; no device is needed.

mod common;
#[path = "fixtures/deform_conv.rs"]
mod deform_conv;
#[path = "fixtures/packed_validation_workspace.rs"]
mod packed_validation_workspace;

#[test]
fn deform_conv_cpu_matches_fixtures() {
    for c in deform_conv::cases() {
        let out = common::run_cpu(
            "DeformConv",
            "",
            c.opset,
            &c.inputs,
            std::slice::from_ref(&c.output),
            &c.attrs,
        );
        deform_conv::compare(&c, &out[0]);
    }
}

#[test]
fn shared_validation_gpu_fixture_expectations_match_cpu() {
    for (name, domain, opset, inputs, dtype, shape, attrs, expected) in
        packed_validation_workspace::cases()
    {
        let actual =
            common::try_run_cpu(name, domain, opset, &inputs, &[(dtype, shape)], &attrs).unwrap();
        assert_eq!(actual, [expected], "{name}");
    }
}
