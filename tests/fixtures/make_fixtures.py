"""Generate small ONNX fixtures for the test-suite.   .venv/bin/python tests/fixtures/make_fixtures.py"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP, helper as h, numpy_helper as nh
from google.protobuf import text_format
from google.protobuf.json_format import MessageToJson

OUT = Path(__file__).resolve().parent
OPS = [h.make_opsetid("", 21)]


def save(model, name):
    model.ir_version = 10
    onnx.save(model, OUT / name)
    print("wrote", name)


def mk(graph, opsets=None, **kw):
    return h.make_model(graph, opset_imports=opsets or OPS, producer_name="onnxviz-tests", producer_version="1.0", **kw)


f32 = lambda name, shape: h.make_tensor_value_info(name, TP.FLOAT, shape)

# 1. If with then/else subgraphs capturing an outer value ------------------------------------
then_g = h.make_graph([h.make_node("Add", ["x", "w"], ["t"], name="then_add")], "then_branch", [], [f32("t", [2])])
else_g = h.make_graph([h.make_node("Sub", ["x", "w"], ["e"], name="else_sub")], "else_branch", [], [f32("e", [2])])
g = h.make_graph(
    [h.make_node("If", ["cond"], ["y"], name="if0", then_branch=then_g, else_branch=else_g)],
    "if_model", [f32("x", [2]), h.make_tensor_value_info("cond", TP.BOOL, [])], [f32("y", [2])],
    [nh.from_array(np.array([1, 2], np.float32), "w")],
)
save(mk(g), "if.onnx")

# 2. Loop with sequence-typed carried dependency ---------------------------------------------
body = h.make_graph(
    [h.make_node("SequenceInsert", ["seq_in", "item"], ["seq_out"], name="ins"),
     h.make_node("Identity", ["cond_in"], ["cond_out"], name="idc")],
    "loop_body",
    [h.make_tensor_value_info("iter", TP.INT64, []), h.make_tensor_value_info("cond_in", TP.BOOL, []),
     h.make_tensor_sequence_value_info("seq_in", TP.FLOAT, [3])],
    [h.make_tensor_value_info("cond_out", TP.BOOL, []), h.make_tensor_sequence_value_info("seq_out", TP.FLOAT, [3])],
)
g = h.make_graph(
    [h.make_node("SequenceEmpty", [], ["seq0"], name="empty", dtype=TP.FLOAT),
     h.make_node("Loop", ["trip", "cond0", "seq0"], ["seq_final"], name="loop0", body=body)],
    "loop_model",
    [h.make_tensor_value_info("trip", TP.INT64, []), h.make_tensor_value_info("cond0", TP.BOOL, [])],
    [h.make_tensor_sequence_value_info("seq_final", TP.FLOAT, [3])],
    [nh.from_array(np.zeros(3, np.float32), "item")],
    value_info=[h.make_tensor_sequence_value_info("seq0", TP.FLOAT, [3])],
)
save(mk(g), "loop_seq.onnx")

# 3. sparse initializer ----------------------------------------------------------------------
sp = h.make_sparse_tensor(
    nh.from_array(np.array([3.0, 4.0, 5.0], np.float32), "sp_w"),
    nh.from_array(np.array([1, 5, 10], np.int64), "sp_idx"), [4, 4],
)
g = h.make_graph([h.make_node("Identity", ["x"], ["y"], name="id")], "sparse_model", [f32("x", [4, 4])], [f32("y", [4, 4])])
g.sparse_initializer.append(sp)
save(mk(g), "sparse.onnx")
# 2-D coordinate indices variant
sp2 = h.make_sparse_tensor(
    nh.from_array(np.array([7.0, 8.0], np.float32), "sp2"),
    nh.from_array(np.array([[0, 1], [3, 3]], np.int64), "sp2_idx"), [4, 4],
)
g = h.make_graph([h.make_node("Identity", ["x"], ["y"], name="id")], "sparse2_model", [f32("x", [4, 4])], [f32("y", [4, 4])])
g.sparse_initializer.append(sp2)
save(mk(g), "sparse2d.onnx")

# 4. sequence / map / optional / sparse typed values ----------------------------------------
opt = h.make_tensor_type_proto  # noqa
g = h.make_graph(
    [h.make_node("Identity", ["seq"], ["seq_o"], name="n1"), h.make_node("Identity", ["m"], ["m_o"], name="n2"),
     h.make_node("Identity", ["o"], ["o_o"], name="n3")],
    "types_model",
    [h.make_tensor_sequence_value_info("seq", TP.FLOAT, ["N", 3]),
     h.make_value_info("m", h.make_map_type_proto(TP.STRING, h.make_tensor_type_proto(TP.FLOAT, []))),
     h.make_value_info("o", h.make_optional_type_proto(h.make_tensor_type_proto(TP.INT32, [2, "B"])))],
    [h.make_tensor_sequence_value_info("seq_o", TP.FLOAT, ["N", 3]),
     h.make_value_info("m_o", h.make_map_type_proto(TP.STRING, h.make_tensor_type_proto(TP.FLOAT, []))),
     h.make_value_info("o_o", h.make_optional_type_proto(h.make_tensor_type_proto(TP.INT32, [2, "B"])))],
)
save(mk(g), "types.onnx")

# 5. external data ---------------------------------------------------------------------------
g = h.make_graph(
    [h.make_node("MatMul", ["x", "W"], ["y"], name="mm")], "ext_model", [f32("x", [2, 4])], [f32("y", [2, 3])],
    [nh.from_array(np.arange(12, dtype=np.float32).reshape(4, 3), "W")],
)
m = mk(g)
m.ir_version = 10
onnx.save_model(m, OUT / "external.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
                location="external.bin", size_threshold=0)
print("wrote external.onnx + external.bin")

# 6. exotic dtypes --------------------------------------------------------------------------
inits = [
    h.make_tensor("f16", TP.FLOAT16, [4], [1.0, -2.0, 0.5, 65504.0]),
    h.make_tensor("bf16", TP.BFLOAT16, [3], [1.0, -2.5, 3.0]),
    h.make_tensor("i4", TP.INT4, [4], [-8, -1, 0, 7]),
    h.make_tensor("u4", TP.UINT4, [3], [0, 7, 15]),
    h.make_tensor("flag", TP.BOOL, [3], [True, False, True]),
    h.make_tensor("names", TP.STRING, [2], [b"alpha", b"beta"]),
    h.make_tensor("f8", TP.FLOAT8E4M3FN, [3], [1.0, -2.0, 0.5]),
    h.make_tensor("i64", TP.INT64, [3], [1, -2, 2**40]),
    h.make_tensor("big", TP.INT64, [1], [2**62]),
    h.make_tensor("u32", TP.UINT32, [2], [7, 4000000000]),
    nh.from_array(np.array([1 + 2j, 3 + 4j], np.complex64), "c64"),
    nh.from_array(np.array([1.5, -2.5], np.float16), "f16_raw"),
    nh.from_array(np.array([[1, 2, 3], [4, 5, 6]], np.int32), "i32_raw"),
]
g = h.make_graph([h.make_node("Identity", ["x"], ["y"], name="id")], "dtypes_model", [f32("x", [1])], [f32("y", [1])], inits)
save(mk(g), "dtypes.onnx")

# 7. Constant nodes with every value_* attribute ---------------------------------------------
cn = [
    h.make_node("Constant", [], ["c_value"], name="k0", value=nh.from_array(np.array([[1, 2], [3, 4]], np.float32))),
    h.make_node("Constant", [], ["c_float"], name="k1", value_float=2.5),
    h.make_node("Constant", [], ["c_floats"], name="k2", value_floats=[1.0, 2.0, 3.0]),
    h.make_node("Constant", [], ["c_int"], name="k3", value_int=7),
    h.make_node("Constant", [], ["c_ints"], name="k4", value_ints=[1, 2, 3, 4]),
    h.make_node("Constant", [], ["c_string"], name="k5", value_string=b"hello"),
    h.make_node("Constant", [], ["c_strings"], name="k6", value_strings=[b"a", b"bc"]),
    h.make_node("Constant", [], ["c_sparse"], name="k7", sparse_value=h.make_sparse_tensor(
        nh.from_array(np.array([9.0], np.float32), "v"), nh.from_array(np.array([2], np.int64), "i"), [4])),
    h.make_node("Add", ["x", "c_float"], ["y"], name="add"),
]
g = h.make_graph(cn, "const_model", [f32("x", [2])], [f32("y", [2])])
save(mk(g), "constants.onnx")

# 8. local function + call ------------------------------------------------------------------
fn = h.make_function(
    "local", "MyAddRelu", ["a", "b"], ["out"],
    [h.make_node("Add", ["a", "b"], ["s"], name="fadd"), h.make_node("Relu", ["s"], ["out"], name="frelu")],
    [h.make_opsetid("", 21)], [],
)
g = h.make_graph(
    [h.make_node("MyAddRelu", ["x", "y"], ["z"], name="call", domain="local")], "fn_model",
    [f32("x", [3]), f32("y", [3])], [f32("z", [3])],
)
save(mk(g, opsets=[h.make_opsetid("", 21), h.make_opsetid("local", 1)], functions=[fn]), "function.onnx")

# 9. QDQ ------------------------------------------------------------------------------------
g = h.make_graph(
    [h.make_node("QuantizeLinear", ["x", "s", "zp"], ["q"], name="Q"),
     h.make_node("DequantizeLinear", ["q", "s", "zp"], ["dq"], name="DQ"),
     h.make_node("Relu", ["dq"], ["y"], name="relu")],
    "qdq_model", [f32("x", [1, 4])], [f32("y", [1, 4])],
    [nh.from_array(np.array(0.1, np.float32), "s"), nh.from_array(np.array(3, np.uint8), "zp")],
)
save(mk(g), "qdq.onnx")

# 10. tensor-valued + graph-valued attributes ------------------------------------------------
cof = h.make_node("ConstantOfShape", ["shape"], ["filled"], name="cos",
                  value=nh.from_array(np.array([5.0], np.float32), "val"))
sub = h.make_graph([h.make_node("Identity", ["filled"], ["inner"], name="inner_id")], "sub_g", [], [f32("inner", [2, 2])])
custom = h.make_node("Custom", ["filled"], ["out"], name="custom", domain="my.dom", body=sub, ty=h.make_tensor_type_proto(TP.FLOAT, [1, 2]))
g = h.make_graph(
    [cof, custom], "attr_model", [], [f32("out", [2, 2])],
    [nh.from_array(np.array([2, 2], np.int64), "shape")],
)
save(mk(g, opsets=[h.make_opsetid("", 21), h.make_opsetid("my.dom", 1)]), "attrs.onnx")

# 11. unnamed nodes + initializer also listed as input + broadcasting without value_info ------
g = h.make_graph(
    [h.make_node("Conv", ["x", "w", "b"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
     h.make_node("Relu", ["c"], ["r"]),
     h.make_node("MaxPool", ["r"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
     h.make_node("Flatten", ["p"], ["f"]),
     h.make_node("Gemm", ["f", "fw"], ["y"], transB=1)],
    "unnamed_model", [f32("x", [1, 3, 8, 8]), f32("w", [4, 3, 3, 3])], [f32("y", [1, 5])],
    [nh.from_array(np.random.default_rng(0).standard_normal((4, 3, 3, 3)).astype(np.float32), "w"),
     nh.from_array(np.zeros(4, np.float32), "b"),
     nh.from_array(np.ones((5, 64), np.float32), "fw")],
)
save(mk(g), "unnamed.onnx")

# 12. text / json versions of the unnamed model ---------------------------------------------
m = onnx.load(OUT / "unnamed.onnx")
(OUT / "unnamed.onnx.prototxt").write_text(text_format.MessageToString(m))
(OUT / "unnamed.json").write_text(MessageToJson(m))
m = onnx.load(OUT / "dtypes.onnx")
(OUT / "dtypes.onnx.prototxt").write_text(text_format.MessageToString(m))
m = onnx.load(OUT / "if.onnx")
(OUT / "if.onnx.prototxt").write_text(text_format.MessageToString(m))
print("wrote text/json fixtures")
