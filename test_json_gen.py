#!/usr/bin/env python3
"""Test JSON generation directly"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onnx2tf.utils.json_auto_generator import analyze_conversion_error
import onnx_graphsurgeon as gs
import onnx

# Simulate the error
class MockError(Exception):
    def __init__(self):
        super().__init__("Dimensions must be equal, but are 32 and 2 for '{{node tf.math.multiply_48/Mul}} = Mul[T=DT_FLOAT](Placeholder, tf.math.multiply_48/Mul/y)' with input shapes: [1,2,1,256,32,1], [1,1,1,1,2,1].")
        self.onnx_op_name = "wa/lightglue/posenc/Expand"

# Load ONNX model
model = onnx.load("custom_spo2.onnx")
graph = gs.import_onnx(model)

# Analyze error
error = MockError()
error_info = analyze_conversion_error(error, graph)

print("Error info:")
print(f"  Type: {error_info['error_type']}")
print(f"  Problematic ops: {error_info['problematic_ops']}")
print(f"  Suggested types: {error_info['suggested_op_types']}")
print(f"  ONNX op name: {error_info.get('onnx_op_name', 'N/A')}")

# Find the Expand node
expand_nodes = [n for n in graph.nodes if n.op == "Expand"]
print(f"\nFound {len(expand_nodes)} Expand nodes:")
for i, node in enumerate(expand_nodes[:5]):
    print(f"  {i}: {node.name}")
    if node.inputs:
        for j, inp in enumerate(node.inputs[:2]):
            if hasattr(inp, 'shape'):
                print(f"     Input {j}: shape={inp.shape}")

# Check specifically for the problematic node
target_node = None
for node in graph.nodes:
    if node.name == "wa/lightglue/posenc/Expand":
        target_node = node
        break

if target_node:
    print(f"\nTarget node found: {target_node.name}")
    print(f"  Inputs: {len(target_node.inputs)}")
    for i, inp in enumerate(target_node.inputs):
        if hasattr(inp, 'shape'):
            print(f"    Input {i}: {inp.name} shape={inp.shape}")
else:
    print("\nTarget node NOT found!")