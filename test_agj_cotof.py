#!/usr/bin/env python3
"""Test the -agj and -cotof interaction"""
import subprocess
import os

print("Test 1: -agj alone")
print("=" * 50)
result = subprocess.run(
    ["python", "onnx2tf/onnx2tf.py", "-h"],
    capture_output=True,
    text=True
)
if "-agj" in result.stdout and "When used together with -cotof" in result.stdout:
    print("✓ Help text updated correctly")
else:
    print("✗ Help text not updated")

print("\nTest 2: Check that -agj and -cotof are no longer mutually exclusive")
print("=" * 50)
# This should not fail with mutual exclusivity error
result = subprocess.run(
    ["python", "onnx2tf/onnx2tf.py", "-i", "custom_spo2.onnx", "-agj", "-cotof", "-n"],
    capture_output=True,
    text=True,
    timeout=10
)
if "Cannot use -agj" not in result.stdout and "Cannot use -agj" not in result.stderr:
    print("✓ No mutual exclusivity error")
else:
    print("✗ Still showing mutual exclusivity error")
    print(f"Output: {result.stdout[:200]}")
    print(f"Error: {result.stderr[:200]}")

print("\nTest 3: Check function parameters")
print("=" * 50)
try:
    import onnx2tf
    from inspect import signature
    sig = signature(onnx2tf.convert)
    params = sig.parameters
    
    if 'auto_generate_json' in params and 'check_onnx_tf_outputs_elementwise_close_full' in params:
        print("✓ Both parameters exist in convert function")
        print(f"  auto_generate_json default: {params['auto_generate_json'].default}")
        print(f"  check_onnx_tf_outputs_elementwise_close_full default: {params['check_onnx_tf_outputs_elementwise_close_full'].default}")
    else:
        print("✗ Missing parameters in convert function")
except Exception as e:
    print(f"✗ Error: {e}")

print("\nAll tests completed!")