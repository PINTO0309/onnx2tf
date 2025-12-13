#!/usr/bin/env python3
"""
Test the iterative JSON optimizer with custom_spo2.onnx
"""
import os
import sys
import json

# Add onnx2tf to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onnx2tf.utils.iterative_json_optimizer import IterativeJSONOptimizer
from onnx2tf.utils.json_auto_generator import generate_auto_replacement_json
from onnx2tf.utils.logging import *
import onnx
import onnx_graphsurgeon as gs


def main():
    model_path = "custom_spo2.onnx"
    output_dir = "saved_model"
    
    if not os.path.exists(model_path):
        error(f"Model not found: {model_path}")
        return
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # First generate initial JSON using existing logic
    info("Generating initial JSON...")
    
    # Load ONNX model
    onnx_model = onnx.load(model_path)
    onnx_graph = gs.import_onnx(onnx_model)
    
    # Create a dummy error for initial JSON generation
    dummy_error = ValueError(
        "Dimensions must be equal, but are 32 and 2 for '{{node tf.math.multiply_48/Mul}}'"
    )
    
    # Generate initial JSON
    initial_json = generate_auto_replacement_json(
        onnx_graph=onnx_graph,
        tf_layers_dict={},
        check_results=None,
        conversion_error=dummy_error,
        error_threshold=1e-2,
        model_path=model_path,
    )
    
    if not initial_json.get('operations'):
        error("Failed to generate initial JSON")
        return
        
    info(f"Initial JSON has {len(initial_json['operations'])} operations")
    
    # Create optimizer
    optimizer = IterativeJSONOptimizer(model_path, output_dir)
    
    # Run iterative optimization
    optimized_json = optimizer.optimize_iteratively(
        initial_json=initial_json,
        max_iterations=5,
        target_error=1e-2
    )
    
    # Save final JSON
    if optimized_json and optimized_json.get('operations'):
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        final_path = os.path.join(output_dir, f"{model_name}_optimized.json")
        
        with open(final_path, 'w') as f:
            json.dump(optimized_json, f, indent=2, ensure_ascii=False)
            
        info(f"\nFinal optimized JSON saved to: {final_path}")
        info(f"Best error achieved: {optimizer.best_error:.6f}")
    else:
        warn("Optimization did not produce a valid JSON")


if __name__ == "__main__":
    main()