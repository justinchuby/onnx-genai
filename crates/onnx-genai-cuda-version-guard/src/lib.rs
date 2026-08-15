#[cfg(all(
    feature = "cuda",
    not(any(
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000"
    ))
))]
compile_error!(
    "onnx-genai CUDA build: no CUDA version selected. Enable exactly one of cuda-12060 | cuda-12080 | cuda-12090 | cuda-13000."
);

#[cfg(any(
    all(feature = "cuda-12060", feature = "cuda-12080"),
    all(feature = "cuda-12060", feature = "cuda-12090"),
    all(feature = "cuda-12060", feature = "cuda-13000"),
    all(feature = "cuda-12080", feature = "cuda-12090"),
    all(feature = "cuda-12080", feature = "cuda-13000"),
    all(feature = "cuda-12090", feature = "cuda-13000")
))]
compile_error!(
    "onnx-genai CUDA build: multiple CUDA versions selected; cudarc bindings cannot compile with more than one. Enable exactly one of cuda-12060 | cuda-12080 | cuda-12090 | cuda-13000 (and set default-features = false on inter-crate deps if you override the default)."
);
