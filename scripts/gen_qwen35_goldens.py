import numpy as np, onnx, struct
from onnx import helper, TensorProto
import onnxruntime as ort

def bits(a):
    a = np.asarray(a, dtype=np.float32).ravel()
    return [struct.unpack("<I", struct.pack("<f", float(x)))[0] for x in a]

def rust_arr(name, a):
    b = bits(a)
    items = ", ".join(f"0x{x:08x}" for x in b)
    return f"    pub const {name}: [u32; {len(b)}] = [{items}];\n"

lines = []
lines.append("//! Auto-generated ORT-1.26 golden fixtures for the Qwen3.5 hybrid linear-attention\n")
lines.append("//! kernels (CausalConvWithState, LinearAttention). Values are bit-exact f32\n")
lines.append("//! (`f32::from_bits`). Regenerate with scripts/gen_qwen35_goldens.py. DO NOT EDIT.\n")
lines.append("#![allow(clippy::all)]\n\n")

# ---------------- CausalConvWithState ----------------
def conv_sess(C,K,S,B,activation):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [B,C,S])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [C,1,K])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [C])
    st= helper.make_tensor_value_info("st",TensorProto.FLOAT, [B,C,K-1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [B,C,S])
    s2= helper.make_tensor_value_info("s2",TensorProto.FLOAT, [B,C,K-1])
    node=helper.make_node("CausalConvWithState",["x","w","b","st"],["y","s2"],
                          domain="com.microsoft",ndim=1,activation=activation)
    g=helper.make_graph([node],"g",[x,w,b,st],[y,s2])
    m=helper.make_model(g,opset_imports=[helper.make_opsetid("",17),helper.make_opsetid("com.microsoft",1)])
    m.ir_version=10; onnx.save(m,"scratch_g.onnx")
    return ort.InferenceSession("scratch_g.onnx",providers=["CPUExecutionProvider"])

conv_cases = [("CONVA",3,4,5,1,"silu"), ("CONVB",3,4,1,1,"silu"), ("CONVC",2,3,4,1,"none")]
lines.append("pub mod conv {\n")
lines.append("    /// (name, C, K, S, B, activation_is_silu)\n")
for name,C,K,S,B,act in conv_cases:
    rng=np.random.default_rng(hash(name)%2**31)
    x=rng.standard_normal((B,C,S)).astype(np.float32)
    w=rng.standard_normal((C,1,K)).astype(np.float32)
    bb=rng.standard_normal((C,)).astype(np.float32)
    st=rng.standard_normal((B,C,K-1)).astype(np.float32)
    sess=conv_sess(C,K,S,B,act)
    y,s2=sess.run(None,{"x":x,"w":w,"b":bb,"st":st})
    lines.append(f"    // case {name}: C={C} K={K} S={S} B={B} act={act}\n")
    lines.append(f"    pub const {name}_DIMS: [usize; 4] = [{B}, {C}, {K}, {S}];\n")
    lines.append(f"    pub const {name}_SILU: bool = {'true' if act=='silu' else 'false'};\n")
    lines.append(rust_arr(f"{name}_X", x))
    lines.append(rust_arr(f"{name}_W", w))
    lines.append(rust_arr(f"{name}_B", bb))
    lines.append(rust_arr(f"{name}_STATE", st))
    lines.append(rust_arr(f"{name}_Y", y))
    lines.append(rust_arr(f"{name}_PRESENT", s2))
lines.append("}\n\n")

# ---------------- LinearAttention ----------------
def la_sess(H,Dk,Dv,S,B,scale):
    q =helper.make_tensor_value_info("q", TensorProto.FLOAT,[B,S,H*Dk])
    k =helper.make_tensor_value_info("k", TensorProto.FLOAT,[B,S,H*Dk])
    v =helper.make_tensor_value_info("v", TensorProto.FLOAT,[B,S,H*Dv])
    st=helper.make_tensor_value_info("st",TensorProto.FLOAT,[B,H,Dk,Dv])
    g =helper.make_tensor_value_info("g", TensorProto.FLOAT,[B,S,H])
    be=helper.make_tensor_value_info("be",TensorProto.FLOAT,[B,S,H])
    o =helper.make_tensor_value_info("o", TensorProto.FLOAT,[B,S,H*Dv])
    s2=helper.make_tensor_value_info("s2",TensorProto.FLOAT,[B,H,Dk,Dv])
    node=helper.make_node("LinearAttention",["q","k","v","st","g","be"],["o","s2"],
            domain="com.microsoft",q_num_heads=H,kv_num_heads=H,update_rule="gated_delta",scale=float(scale))
    graph=helper.make_graph([node],"g",[q,k,v,st,g,be],[o,s2])
    m=helper.make_model(graph,opset_imports=[helper.make_opsetid("",17),helper.make_opsetid("com.microsoft",1)])
    m.ir_version=10; onnx.save(m,"scratch_g.onnx")
    return ort.InferenceSession("scratch_g.onnx",providers=["CPUExecutionProvider"])

la_cases=[("LAA",2,4,4,3,1,1.0),("LAB",2,4,4,1,1,1.0),("LAC",3,2,5,2,1,0.5)]
lines.append("pub mod la {\n")
for name,H,Dk,Dv,S,B,scale in la_cases:
    rng=np.random.default_rng(hash(name)%2**31)
    q=rng.standard_normal((B,S,H*Dk)).astype(np.float32)
    k=rng.standard_normal((B,S,H*Dk)).astype(np.float32)
    v=rng.standard_normal((B,S,H*Dv)).astype(np.float32)
    st=rng.standard_normal((B,H,Dk,Dv)).astype(np.float32)
    g=(-rng.random((B,S,H))).astype(np.float32)
    be=rng.random((B,S,H)).astype(np.float32)
    sess=la_sess(H,Dk,Dv,S,B,scale)
    o,s2=sess.run(None,{"q":q,"k":k,"v":v,"st":st,"g":g,"be":be})
    lines.append(f"    // case {name}: H={H} Dk={Dk} Dv={Dv} S={S} B={B} scale={scale}\n")
    lines.append(f"    pub const {name}_DIMS: [usize; 5] = [{B}, {H}, {Dk}, {Dv}, {S}];\n")
    lines.append(f"    pub const {name}_SCALE: f32 = {scale}f32;\n")
    lines.append(rust_arr(f"{name}_Q", q))
    lines.append(rust_arr(f"{name}_K", k))
    lines.append(rust_arr(f"{name}_V", v))
    lines.append(rust_arr(f"{name}_STATE", st))
    lines.append(rust_arr(f"{name}_G", g))
    lines.append(rust_arr(f"{name}_BETA", be))
    lines.append(rust_arr(f"{name}_O", o))
    lines.append(rust_arr(f"{name}_PRESENT", s2))
lines.append("}\n")

with open("crates/onnx-runtime-ep-cpu/src/kernels/qwen35_goldens.rs","w") as f:
    f.writelines(lines)
print("wrote goldens:", sum(len(l) for l in lines), "bytes")
