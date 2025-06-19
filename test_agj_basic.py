#!/usr/bin/env python3
"""Test the -agj option with custom_spo2.onnx"""
import subprocess
import os

# Clean up any existing auto JSON
if os.path.exists("saved_model/custom_spo2_auto.json"):
    os.remove("saved_model/custom_spo2_auto.json")

print("Testing -agj option with custom_spo2.onnx...")
print("This will attempt to auto-generate optimal JSON parameters.\n")

# Run with -agj option
result = subprocess.run(
    ["python", "onnx2tf/onnx2tf.py", "-i", "custom_spo2.onnx", "-agj", "-n"],
    capture_output=True,
    text=True,
    timeout=120  # 2 minute timeout
)

print("Exit code:", result.returncode)

# Check if JSON was generated
if os.path.exists("saved_model/custom_spo2_auto.json"):
    print("\n✓ Auto JSON file generated successfully!")
    with open("saved_model/custom_spo2_auto.json", "r") as f:
        import json
        data = json.load(f)
        print(f"  Contains {len(data.get('operations', []))} operations")
else:
    print("\n✗ No auto JSON file was generated")

# Check for key messages in output
if "Auto JSON generation started" in result.stdout:
    print("✓ Auto JSON generation was triggered")
else:
    print("✗ Auto JSON generation was not triggered")

if "Searching for optimal parameter" in result.stdout:
    print("✓ Optimization search started")
    
# Show relevant output
print("\nRelevant output:")
for line in result.stdout.split('\n'):
    if any(keyword in line for keyword in ["Auto JSON", "Searching", "Generated JSON", "All outputs"]):
        print(f"  {line}")

# Show any errors
if result.stderr:
    print("\nErrors:")
    print(result.stderr[:500])