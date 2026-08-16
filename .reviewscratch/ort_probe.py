import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

def run_single(op_type, x, domain="", attrs=None, extra_inputs=None):
    attrs = attrs or {}
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [len(x)])]
    node_inputs = ["X"]
    node = helper.make_node(op_type, node_inputs, ["Y"], domain=domain, **attrs)
    graph = helper.make_graph(
        [node], "g", inputs,
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [len(x)])],
    )
    opset_imports = [helper.make_opsetid("", 17)]
    if domain and domain != "":
        opset_imports.append(helper.make_opsetid(domain, 1))
    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = 10
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    y = sess.run(None, {"X": np.array(x, dtype=np.float32)})[0]
    return y

def bits(v):
    return "0x%08X" % (np.float32(v).view(np.uint32))

# 1. Sigmoid(-Inf)
y = run_single("Sigmoid", [np.float32(-np.inf), np.float32(-18.0), np.float32(-17.999998)])
print("Sigmoid(-Inf)      =", repr(float(y[0])), bits(y[0]))
print("Sigmoid(-18.0)     =", repr(float(y[1])), bits(y[1]))
print("Sigmoid(-17.999998)=", repr(float(y[2])), bits(y[2]))

# 2. Tanh(8.442762) and tanh(9.0)
y = run_single("Tanh", [np.float32(8.442762), np.float32(9.0), np.float32(-9.0)])
print("Tanh(8.442762)     =", repr(float(y[0])), bits(y[0]))
print("Tanh(9.0)          =", repr(float(y[1])), bits(y[1]))
print("Tanh(-9.0)         =", repr(float(y[2])), bits(y[2]))

# 3. Gelu(-Inf) approximate=tanh  (opset 20 Gelu in default domain)
try:
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])]
    node = helper.make_node("Gelu", ["X"], ["Y"], approximate="tanh")
    graph = helper.make_graph([node], "g", inputs,
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 10
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    y = sess.run(None, {"X": np.array([np.float32(-np.inf)], dtype=np.float32)})[0]
    print("Gelu(-Inf,tanh)    =", repr(float(y[0])), bits(y[0]))
except Exception as e:
    print("Gelu approximate=tanh failed:", e)

# Also FastGelu (com.microsoft) at -Inf
try:
    y = run_single("FastGelu", [np.float32(-np.inf)], domain="com.microsoft")
    print("FastGelu(-Inf)     =", repr(float(y[0])), bits(y[0]))
except Exception as e:
    print("FastGelu failed:", e)

# 4. QuickGelu(-Inf, alpha=1.702) com.microsoft
try:
    y = run_single("QuickGelu", [np.float32(-np.inf)], domain="com.microsoft", attrs={"alpha": 1.702})
    print("QuickGelu(-Inf)    =", repr(float(y[0])), bits(y[0]))
except Exception as e:
    print("QuickGelu failed:", e)
