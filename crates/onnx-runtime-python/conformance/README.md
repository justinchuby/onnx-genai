# Upstream `cbourjau/onnx-tests`

The adapter in `../tests/nxrt_runtime.py` implements the suite's
`(onnx.ModelProto) -> dict[str, np.ndarray]` runtime contract. It selects
`CPUExecutionProvider`, requests outputs in model order, and returns NumPy
arrays keyed by output name.

## Reproduce

These commands use plain pytest; pixi is not required. The upstream checkout
must remain outside this repository.

```bash
export PATH=/home/justinchu/.conda/envs/onnx/bin:$PATH
cd /path/to/onnx-genai/crates/onnx-runtime-python
maturin build --release
python -m pip install --force-reinstall \
  ../../target/wheels/nxrt-*cp310-abi3*.whl

git clone --depth 1 https://github.com/cbourjau/onnx-tests \
  /path/outside/onnx-tests-upstream
cd /path/outside/onnx-tests-upstream
python -m pip install -e . \
  "hypothesis>=6.130.4" "spox>=0.16" "ndonnx>=0.10.1" \
  "pytest-xdist>=3.8,<4"

PYTHONPATH=/path/to/onnx-genai/crates/onnx-runtime-python/tests \
RUN_CANDIDATE=nxrt_runtime.run_nxrt \
python -m pytest tests -q -n 8 --dist loadfile \
  --junitxml=nxrt-junit.xml

python /path/to/onnx-genai/crates/onnx-runtime-python/conformance/summarize_junit.py \
  nxrt-junit.xml
```

## 2026-07-14 CPU results

Tested upstream commit `856e89bcd3d2ee3cb31c7bf88b5706dec00eba5c`
with the default 100 Hypothesis examples. The 1,198 collected dtype/opset
parameter cases produced **158 passed, 1,038 failed, and 2 skipped** across
112 operator names. No operator passed every dtype case; 17 operators had at
least one passing case.

The detailed coverage and failure groups are recorded in
[`docs/EP_CONFORMANCE.md`](../../../docs/EP_CONFORMANCE.md).
