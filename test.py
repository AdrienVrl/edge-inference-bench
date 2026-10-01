import onnx
from onnx import numpy_helper

m = onnx.load("models/mobilenet_v2.int8_trt.onnx")

# 1. does the graph's global input feed a QuantizeLinear, or go straight into a Conv?
graph_input = m.graph.input[0].name
print("graph input name:", graph_input)
print(
    "feeds into:", [(n.name, n.op_type) for n in m.graph.node if graph_input in n.input]
)

# 2. the scale/zero-point for the problem weight - actual dtype and values
initializers = {i.name: i for i in m.graph.initializer}
for name in ("onnx::Conv_539_quantized_scale", "onnx::Conv_539_quantized_zero_point"):
    t = initializers.get(name)
    if t is None:
        print(
            name,
            "-> NOT an initializer (must be computed at runtime - unusual for a weight DQ)",
        )
        continue
    arr = numpy_helper.to_array(t)
    print(name, "dtype:", arr.dtype, "shape:", arr.shape, "values:", arr.flatten()[:8])

# 3. sanity: how many Conv nodes in total have a weight DequantizeLinear, and does this
#    problem only affect the very first one, or all of them?
dq_to_conv = [
    n
    for n in m.graph.node
    if n.op_type == "DequantizeLinear"
    and any(c.op_type == "Conv" and n.output[0] in c.input for c in m.graph.node)
]
print("total weight-DQ-into-Conv nodes:", len(dq_to_conv))
print("first 3 names:", [n.name for n in dq_to_conv[:3]])
