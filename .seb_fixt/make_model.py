import numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
K = N = 1024
LAYERS = 4
rng = np.random.default_rng(7)
inits, nodes = [], []
cur = "x"
for i in range(LAYERS):
    w = rng.standard_normal((K, N)).astype(np.float32) * 0.02
    inits.append(numpy_helper.from_array(w, f"w{i}"))
    nodes.append(helper.make_node("MatMul", [cur, f"w{i}"], [f"h{i}"]))
    cur = f"h{i}"
graph = helper.make_graph(nodes, "decode_shaped",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, K])],
    [helper.make_tensor_value_info(cur, TensorProto.FLOAT, [1, N])], inits)
m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 10
onnx.checker.check_model(m)
onnx.save(m, ".seb_fixt/decode_shaped.onnx")
