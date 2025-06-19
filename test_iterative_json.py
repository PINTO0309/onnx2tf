#!/usr/bin/env python3
"""
Test iterative JSON generation with custom_spo2.onnx
This script implements the iterative validation loop for automatic JSON generation.
"""
import os
import sys
import json
import tempfile
import shutil
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import onnx
import onnx_graphsurgeon as gs

# Add onnx2tf to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onnx2tf import convert
from onnx2tf.utils.common_functions import dummy_onnx_inference
from onnx2tf.utils.json_auto_generator import (
    generate_auto_replacement_json, 
    save_auto_replacement_json,
    analyze_accuracy_errors,
    generate_candidate_fixes,
    combine_fixes
)
from onnx2tf.utils.logging import *


def run_conversion_with_json(model_path: str, json_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run conversion and return results including accuracy check
    """
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Run conversion
        kwargs = {
            'input_onnx_file_path': model_path,
            'output_folder_path': temp_dir,
            'copy_onnx_input_output_names_to_tflite': True,
            'non_verbose': True,
        }
        
        if json_path:
            kwargs['param_replacement_file'] = json_path
            
        # Suppress output during conversion
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            with contextlib.redirect_stderr(f):
                try:
                    result = convert(**kwargs)
                    conversion_success = True
                    conversion_error = None
                except Exception as e:
                    conversion_success = False
                    conversion_error = e
                    result = None
        
        # If conversion failed, return error
        if not conversion_success:
            return {
                'success': False,
                'error': conversion_error,
                'max_error': float('inf')
            }
        
        # Get accuracy results from conversion output
        output = f.getvalue()
        
        # Parse accuracy results
        max_error = 0.0
        check_results = {}
        
        # Look for accuracy validation results in output
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if 'Max Absolute Error' in line:
                try:
                    # Extract error value
                    parts = line.split('Max Absolute Error:')
                    if len(parts) > 1:
                        error_str = parts[1].strip().split()[0]
                        error_val = float(error_str)
                        max_error = max(max_error, error_val)
                except:
                    pass
        
        return {
            'success': True,
            'max_error': max_error,
            'check_results': check_results,
            'tf_layers_dict': {},
            'output': output
        }
        
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def iterative_json_generation(model_path: str, output_dir: str, max_iterations: int = 10):
    """
    Implement iterative JSON generation with actual conversion testing
    """
    info(f"Starting iterative JSON generation for: {model_path}")
    info(f"Maximum iterations: {max_iterations}")
    info(f"Target accuracy: 1e-2")
    
    # First, try conversion without JSON to get baseline
    info("\n=== Baseline conversion (no JSON) ===")
    baseline = run_conversion_with_json(model_path)
    
    if not baseline['success']:
        error(f"Baseline conversion failed: {baseline['error']}")
        
        # For conversion errors, generate JSON based on error analysis
        info("\n=== Generating JSON for conversion error ===")
        
        # Load ONNX model
        onnx_model = onnx.load(model_path)
        onnx_graph = gs.import_onnx(onnx_model)
        
        # Generate JSON for conversion error
        auto_json = generate_auto_replacement_json(
            onnx_graph=onnx_graph,
            tf_layers_dict={},
            check_results=None,
            conversion_error=baseline['error'],
            error_threshold=1e-2,
            model_path=model_path,
        )
        
        if auto_json.get('operations'):
            json_path = save_auto_replacement_json(
                replacement_json=auto_json,
                model_path=model_path,
                output_dir=output_dir,
            )
            info(f"Generated JSON for conversion error: {json_path}")
            
            # Test with generated JSON
            info("\n=== Testing with generated JSON ===")
            test_result = run_conversion_with_json(model_path, json_path)
            
            if test_result['success']:
                info(f"✓ Conversion succeeded with JSON!")
                info(f"Max error: {test_result['max_error']:.6f}")
            else:
                error(f"✗ Conversion still fails with JSON: {test_result['error']}")
        
        return
    
    info(f"Baseline max error: {baseline['max_error']:.6f}")
    
    if baseline['max_error'] <= 1e-2:
        info("✓ Model already meets accuracy target!")
        return
    
    # Load ONNX model for analysis
    onnx_model = onnx.load(model_path)
    onnx_graph = gs.import_onnx(onnx_model)
    
    # Initialize tracking
    best_json = None
    best_error = baseline['max_error']
    best_operations = []
    previous_attempts = set()
    
    # Iterative optimization
    for iteration in range(1, max_iterations + 1):
        info(f"\n=== Iteration {iteration}/{max_iterations} ===")
        info(f"Current best error: {best_error:.6f}")
        
        # Generate candidate fixes based on current error
        error_info = {
            'error_type': 'accuracy',
            'max_error': best_error,
            'error_nodes': [],  # Would be populated from actual analysis
        }
        
        candidate_fixes = generate_candidate_fixes(onnx_graph, error_info, previous_attempts)
        
        if not candidate_fixes:
            info("No more candidate fixes available")
            break
        
        info(f"Generated {len(candidate_fixes)} candidate fixes")
        
        # Generate combinations to test
        fix_combinations = combine_fixes(candidate_fixes, max_combinations=3)
        info(f"Testing {len(fix_combinations)} combinations")
        
        # Test each combination
        for i, combination in enumerate(fix_combinations):
            info(f"\nTesting combination {i+1}/{len(fix_combinations)}")
            
            # Merge with best operations so far
            test_operations = best_operations + combination
            
            # Create test JSON
            test_json = {
                "format_version": 1,
                "operations": test_operations
            }
            
            # Save temporary JSON
            temp_json_path = os.path.join(output_dir, f"test_{iteration}_{i}.json")
            with open(temp_json_path, 'w') as f:
                json.dump(test_json, f, indent=2)
            
            # Test conversion
            result = run_conversion_with_json(model_path, temp_json_path)
            
            if result['success']:
                info(f"  Max error: {result['max_error']:.6f}")
                
                # Check if improved
                if result['max_error'] < best_error:
                    best_error = result['max_error']
                    best_operations = test_operations
                    best_json = test_json
                    info(f"  ✓ New best! Error reduced to {best_error:.6f}")
                    
                    # Save intermediate best
                    intermediate_path = os.path.join(output_dir, f"best_iteration_{iteration}.json")
                    with open(intermediate_path, 'w') as f:
                        json.dump(best_json, f, indent=2)
                    
                    # Check if target reached
                    if best_error <= 1e-2:
                        info(f"  ✓ Target accuracy achieved!")
                        break
            else:
                info(f"  ✗ Conversion failed")
            
            # Clean up temp file
            os.remove(temp_json_path)
        
        # Add successful fixes to previous attempts
        for fix in combination:
            fix_key = f"{fix['op_name']}_{fix['op_type']}"
            previous_attempts.add(fix_key)
        
        # Check if target reached
        if best_error <= 1e-2:
            break
    
    # Save final JSON
    if best_json and best_operations:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        final_path = os.path.join(output_dir, f"{model_name}_auto.json")
        
        # Clean up confidence scores
        for op in best_operations:
            if "confidence" in op:
                del op["confidence"]
        
        best_json['operations'] = best_operations
        
        with open(final_path, 'w') as f:
            json.dump(best_json, f, indent=2, ensure_ascii=False)
        
        info(f"\n=== Final Results ===")
        info(f"Best error achieved: {best_error:.6f}")
        info(f"Total operations in JSON: {len(best_operations)}")
        info(f"Final JSON saved to: {final_path}")
        
        if best_error <= 1e-2:
            info("✓ Successfully achieved target accuracy!")
        else:
            warn("✗ Could not achieve target accuracy")
    else:
        warn("No improvement found through iterative optimization")


def main():
    """Main function"""
    model_path = "custom_spo2.onnx"
    output_dir = "saved_model"
    
    if not os.path.exists(model_path):
        error(f"Model not found: {model_path}")
        error("Please ensure custom_spo2.onnx is in the current directory")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run iterative JSON generation
    iterative_json_generation(model_path, output_dir, max_iterations=5)


if __name__ == "__main__":
    main()