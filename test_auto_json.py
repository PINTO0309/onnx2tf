#!/usr/bin/env python3
import sys
import onnx2tf

try:
    print("Starting conversion...")
    onnx2tf.convert(
        input_onnx_file_path='e2epose_1x3x512x512.onnx',
        output_folder_path='saved_model',
        non_verbose=False
    )
    print("Conversion succeeded!")
except Exception as e:
    print(f"Conversion failed with error: {type(e).__name__}")
    print(f"Error message: {str(e)[:500]}...")
    print("Checking if auto JSON was generated...")
    
    import os
    json_files = [f for f in os.listdir('saved_model') if f.endswith('_auto.json')]
    if json_files:
        print(f"Found auto-generated JSON files: {json_files}")
        for jf in json_files:
            print(f"\nContent of {jf}:")
            with open(os.path.join('saved_model', jf), 'r') as f:
                print(f.read())
    else:
        print("No auto-generated JSON files found")