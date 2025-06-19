#!/usr/bin/env python3
"""Test the -agj option"""
import subprocess
import os
import sys

# Test 1: Check that -agj and -cotof are mutually exclusive
print("Test 1: Checking mutual exclusivity of -agj and -cotof...")
result = subprocess.run(
    ["python", "onnx2tf/onnx2tf.py", "-i", "custom_spo2.onnx", "-agj", "-cotof"],
    capture_output=True,
    text=True
)
if "Cannot use -agj" in result.stderr:
    print("✓ Mutual exclusivity check passed")
else:
    print("✗ Mutual exclusivity check failed")
    print(f"Error output: {result.stderr}")

# Test 2: Check that -agj option is recognized
print("\nTest 2: Checking -agj option is recognized...")
result = subprocess.run(
    ["python", "onnx2tf/onnx2tf.py", "-h"],
    capture_output=True,
    text=True
)
if "-agj" in result.stdout and "auto_generate_json" in result.stdout:
    print("✓ -agj option is recognized in help")
else:
    print("✗ -agj option not found in help")

# Test 3: Check convert function signature
print("\nTest 3: Checking convert function signature...")
try:
    import onnx2tf
    from inspect import signature
    sig = signature(onnx2tf.convert)
    if 'auto_generate_json' in sig.parameters:
        print("✓ auto_generate_json parameter found in convert function")
    else:
        print("✗ auto_generate_json parameter not found in convert function")
except Exception as e:
    print(f"✗ Error checking function signature: {e}")

print("\nAll tests completed!")