use onnx_genai_metadata::{StatePortAccess, StatePortAlias};

#[test]
fn state_port_access_defaults_to_read_write_and_parses_read_only() {
    let writable: StatePortAlias =
        serde_yaml::from_str("input: past_key\noutput: present_key\nrole: key\nlayer: 0\n")
            .unwrap();
    assert_eq!(writable.access, StatePortAccess::ReadWrite);

    let read_only: StatePortAlias = serde_yaml::from_str(
        "input: past_key\noutput: present_key\naccess: read_only\nrole: key\nlayer: 0\n",
    )
    .unwrap();
    assert_eq!(read_only.access, StatePortAccess::ReadOnly);
}
