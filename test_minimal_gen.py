#!/usr/bin/env python3
"""Test minimal JSON generation for custom_spo2"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onnx2tf.utils.json_auto_generator import (
    analyze_conversion_error, generate_candidate_fixes, 
    ExpandFixer, save_auto_replacement_json
)
import onnx_graphsurgeon as gs
import onnx

# Simulate the error
class MockError(Exception):
    def __init__(self):
        super().__init__("Dimensions must be equal, but are 32 and 2 for '{{node tf.math.multiply_48/Mul}} = Mul[T=DT_FLOAT](Placeholder, tf.math.multiply_48/Mul/y)' with input shapes: [1,2,1,256,32,1], [1,1,1,1,2,1].")
        self.onnx_op_name = "wa/lightglue/posenc/Expand"

# Load model
model = onnx.load("custom_spo2.onnx")
graph = gs.import_onnx(model)

# Find the target node
target_node = None
for node in graph.nodes:
    if node.name == "wa/lightglue/posenc/Expand":
        target_node = node
        break

if target_node:
    print(f"Testing ExpandFixer for: {target_node.name}")
    
    # Create error info
    error = MockError()
    error_info = {
        'error_type': 'ValueError',
        'error_msg': str(error),
        'problematic_ops': ['wa/lightglue/posenc/Expand'],
        'suggested_op_types': ['Expand'],
        'onnx_op_name': 'wa/lightglue/posenc/Expand'
    }
    
    # Generate fixes
    fixer = ExpandFixer(target_node, error_info)
    fixes = fixer.generate_fixes()
    
    print(f"\nGenerated {len(fixes)} fixes:")
    
    # Look for the critical permutation
    critical_found = False
    for i, fix in enumerate(fixes):
        perm = fix.get('pre_process_transpose_perm')
        if perm == [0, 4, 2, 3, 1, 5]:
            print(f"  Fix {i}: CRITICAL PERMUTATION FOUND!")
            print(f"    op: {fix['op_name']}")
            print(f"    perm: {perm}")
            print(f"    confidence: {fix.get('confidence')}")
            critical_found = True
        elif i < 5:  # Show first 5
            print(f"  Fix {i}: {fix.get('op_name')} - perm={perm} conf={fix.get('confidence')}")
    
    if not critical_found:
        print("\n  WARNING: Critical permutation [0,4,2,3,1,5] NOT generated!")
        
    # Save a test JSON with just the critical fix
    test_json = {
        "format_version": 1,
        "operations": [{
            "op_name": "wa/lightglue/posenc/Expand",
            "param_target": "inputs",
            "param_name": "wa/lightglue/posenc/Unsqueeze_3_output_0",
            "pre_process_transpose_perm": [0, 4, 2, 3, 1, 5]
        }]
    }
    
    import json
    with open("test_critical_fix.json", "w") as f:
        json.dump(test_json, f, indent=2)
    print("\nSaved test_critical_fix.json for manual testing")