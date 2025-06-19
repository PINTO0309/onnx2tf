#!/usr/bin/env python3
"""Simple test of automatic generation finding the right fix"""
import subprocess
import json
import os

# First, clean up any existing auto JSON
if os.path.exists("saved_model/custom_spo2_auto.json"):
    os.remove("saved_model/custom_spo2_auto.json")

# Run conversion which should trigger auto generation
print("Running conversion to trigger auto JSON generation...")
result = subprocess.run(
    ["python", "onnx2tf/onnx2tf.py", "-i", "custom_spo2.onnx", "-cotof"],
    capture_output=True,
    text=True,
    timeout=30  # 30 second timeout
)

print(f"Return code: {result.returncode}")

# Check if JSON was generated
json_path = "saved_model/custom_spo2_auto.json"
if os.path.exists(json_path):
    print(f"\nJSON generated at: {json_path}")
    
    # Load and check contents
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Number of operations: {len(data.get('operations', []))}")
    
    # Check for critical permutation
    critical_found = False
    for op in data.get('operations', []):
        if (op.get('op_name') == 'wa/lightglue/posenc/Expand' and
            op.get('pre_process_transpose_perm') == [0, 4, 2, 3, 1, 5]):
            critical_found = True
            print("CRITICAL PERMUTATION FOUND in generated JSON!")
            break
    
    if not critical_found:
        print("WARNING: Critical permutation NOT found in generated JSON")
        
    # Show first few operations
    print("\nFirst 3 operations:")
    for i, op in enumerate(data.get('operations', [])[:3]):
        print(f"  {i}: {op.get('op_name')} - perm={op.get('pre_process_transpose_perm')}")
else:
    print("ERROR: No JSON file generated!")
    
# Show relevant error output
if "critical" in result.stdout:
    print("\nFound 'critical' in stdout - good sign!")
if "Selected high-confidence" in result.stdout:
    print("Found confidence selection message")