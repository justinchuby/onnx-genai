import numpy as np
from onnx import helper, TensorProto
import onnxruntime as ort

def run(op, xs_bits, domain="", attrs=None):
    attrs = attrs or {}
    xs = np.array([np.uint32(b) for b in xs_bits], dtype=np.uint32).view(np.float32)
    n = len(xs)
    inp = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [n])]
    node = helper.make_node(op, ["X"], ["Y"], domain=domain, **attrs)
    g = helper.make_graph([node], "g", inp,
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n])])
    opsets = [helper.make_opsetid("", 20)]
    if domain: opsets.append(helper.make_opsetid(domain, 1))
    m = helper.make_model(g, opset_imports=opsets); m.ir_version = 10
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    y = s.run(None, {"X": xs})[0]
    return y.view(np.uint32)

for op, dom, at in [("Tanh","",None), ("Sigmoid","",None), ("QuickGelu","com.microsoft",{"alpha":1.702})]:
    out = run(op, [0x7FC01234, 0x7F800001], dom, at)
    print(f"{op}: qNaN 0x7FC01234 -> 0x{out[0]:08X} ; sNaN 0x7F800001 -> 0x{out[1]:08X}")
