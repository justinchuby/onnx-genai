import numpy as np
from onnx import helper, TensorProto
import onnxruntime as ort

def run(op_type, xs, domain="", attrs=None):
    attrs = attrs or {}
    n = len(xs)
    inp = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [n])]
    node = helper.make_node(op_type, ["X"], ["Y"], domain=domain, **attrs)
    g = helper.make_graph([node], "g", inp,
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n])])
    opsets = [helper.make_opsetid("", 20)]
    if domain: opsets.append(helper.make_opsetid(domain, 1))
    m = helper.make_model(g, opset_imports=opsets); m.ir_version = 10
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return s.run(None, {"X": np.array(xs, dtype=np.float32)})[0]

def b(v): return "0x%08X" % (np.float32(v).view(np.uint32))

xs = [np.float32(v) for v in [-9.000001, 9.0, -9.0, -17.999998]]
for name, op, dom, at in [
    ("Sigmoid", "Sigmoid", "", None),
    ("Tanh", "Tanh", "", None),
    ("QuickGelu", "QuickGelu", "com.microsoft", {"alpha":1.702}),
    ("FastGelu", "FastGelu", "com.microsoft", None),
]:
    y = run(op, xs, dom, at)
    print(name, [(float(x), b(o)) for x,o in zip(xs,y)])
