import errno
import os

import pytest

import onnx_std


TINY_MODEL = """\
<
  ir_version: 10,
  opset_import: ["" : 21]
>
main (float[1] X, float[1] Y) => (float[1] Z)
{
  Z = Add(X, Y)
}
"""


def test_missing_path_raises_file_not_found(tmp_path):
    missing = tmp_path / "missing.onnx"
    with pytest.raises(FileNotFoundError) as exc_info:
        onnx_std.load_model(missing)

    message = str(exc_info.value)
    assert str(missing) in message
    assert os.strerror(errno.ENOENT) in message
    assert "Pass a path to an existing, readable ONNX protobuf model" in message


def test_serialized_bytes_still_load(tmp_path):
    model = onnx_std.from_text(TINY_MODEL)
    path = tmp_path / "tiny.onnx"
    onnx_std.save_model(model, path)

    loaded = onnx_std.load_model(path.read_bytes())
    assert "nodes=1" in repr(loaded)


def test_non_path_argument_has_helpful_type_error():
    with pytest.raises(TypeError) as exc_info:
        onnx_std.load_model(42)

    message = str(exc_info.value)
    assert "str or os.PathLike" in message
    assert "bytes" in message
    assert "int" in message


def test_fspath_exception_is_preserved():
    class BrokenPath:
        def __fspath__(self):
            raise RuntimeError("path conversion exploded")

    with pytest.raises(RuntimeError, match="path conversion exploded"):
        onnx_std.load_model(BrokenPath())


@pytest.mark.skipif(os.name == "nt", reason="Unix filesystem-encoding test")
def test_non_utf8_unix_path_round_trips(tmp_path):
    model = onnx_std.from_text(TINY_MODEL)
    raw_path = os.fsencode(tmp_path) + b"/model-\xff.onnx"
    path = os.fsdecode(raw_path)

    onnx_std.save_model(model, path)
    loaded = onnx_std.load_model(path)

    assert "nodes=1" in repr(loaded)
    assert os.path.exists(raw_path)


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod permission semantics require a non-root Unix user",
)
def test_permission_denied_raises_permission_error(tmp_path):
    path = tmp_path / "unreadable.onnx"
    path.write_bytes(b"not read because permissions deny access")
    path.chmod(0)
    try:
        with pytest.raises(PermissionError) as exc_info:
            onnx_std.load_model(path)
        message = str(exc_info.value)
        assert str(path) in message
        assert os.strerror(errno.EACCES) in message
        assert "existing, readable ONNX protobuf model" in message
    finally:
        path.chmod(0o600)
