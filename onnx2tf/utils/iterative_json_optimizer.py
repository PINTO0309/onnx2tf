"""
Iterative JSON optimizer for automatic parameter replacement.
This module implements an iterative optimization algorithm that repeatedly
tests different parameter modifications and evaluates their impact.
"""
import os
import json
import tempfile
import subprocess
import sys
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from onnx2tf.utils.logging import *
import onnx_graphsurgeon as gs


class IterativeJSONOptimizer:
    """
    Iteratively optimize JSON replacements by testing conversions
    """
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = output_dir
        self.best_json = None
        self.best_error = float('inf')
        self.tested_combinations = []
        
    def test_conversion(self, json_path: Optional[str] = None) -> Tuple[bool, float, str]:
        """
        Test conversion with a given JSON file
        Returns: (success, max_error, output_message)
        """
        # Build command
        cmd = [
            sys.executable,
            "-m", "onnx2tf",
            "-i", self.model_path,
            "-o", self.output_dir,
            "-cotof",
            "-n"  # non-verbose
        ]
        
        if json_path:
            cmd.extend(["-prf", json_path])
            
        # Run conversion
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Parse output for accuracy errors
            output = result.stdout + result.stderr
            max_error = 0.0
            
            # Look for max error in output
            for line in output.split('\n'):
                if 'Max Absolute Error:' in line:
                    try:
                        error_str = line.split('Max Absolute Error:')[1].strip().split()[0]
                        error_val = float(error_str)
                        max_error = max(max_error, error_val)
                    except:
                        pass
                        
            # Check if conversion succeeded
            success = result.returncode == 0
            
            return success, max_error, output
            
        except subprocess.TimeoutExpired:
            return False, float('inf'), "Conversion timed out"
        except Exception as e:
            return False, float('inf'), str(e)
            
    def optimize_iteratively(
        self, 
        initial_json: Dict[str, Any],
        max_iterations: int = 10,
        target_error: float = 1e-2
    ) -> Dict[str, Any]:
        """
        Iteratively optimize the JSON by testing different combinations
        """
        info(f"Starting iterative optimization for {self.model_path}")
        info(f"Target error: {target_error}")
        
        # Test baseline (no JSON)
        info("\n=== Testing baseline (no JSON) ===")
        base_success, base_error, base_output = self.test_conversion()
        info(f"Baseline: success={base_success}, error={base_error:.6f}")
        
        if base_success and base_error <= target_error:
            info("Model already meets target accuracy!")
            return {}
            
        # Initialize with initial JSON
        current_json = initial_json
        self.best_json = initial_json
        self.best_error = base_error
        
        # Iterative optimization
        for iteration in range(1, max_iterations + 1):
            info(f"\n=== Iteration {iteration}/{max_iterations} ===")
            
            # Save current JSON to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(current_json, f, indent=2)
                temp_json_path = f.name
                
            try:
                # Test current JSON
                success, error, output = self.test_conversion(temp_json_path)
                info(f"Test result: success={success}, error={error:.6f}")
                
                # Track this attempt
                self.tested_combinations.append({
                    'iteration': iteration,
                    'operations': len(current_json.get('operations', [])),
                    'success': success,
                    'error': error
                })
                
                # Update best if improved
                if success and error < self.best_error:
                    self.best_error = error
                    self.best_json = current_json.copy()
                    info(f"✓ New best! Error reduced to {error:.6f}")
                    
                    # Check if target reached
                    if error <= target_error:
                        info(f"✓ Target accuracy achieved!")
                        break
                        
                # Generate variations for next iteration
                if not success or error > target_error:
                    # Analyze the error from output
                    error_type = self._analyze_error_output(output)
                    
                    # Generate new variations based on error type
                    variations = self._generate_variations(current_json, error_type)
                    
                    # Test variations
                    best_variation = None
                    best_variation_error = float('inf')
                    
                    for i, variation in enumerate(variations[:3]):  # Test up to 3 variations
                        info(f"  Testing variation {i+1}/{len(variations[:3])}")
                        
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as vf:
                            json.dump(variation, vf, indent=2)
                            var_json_path = vf.name
                            
                        try:
                            var_success, var_error, _ = self.test_conversion(var_json_path)
                            if var_success and var_error < best_variation_error:
                                best_variation = variation
                                best_variation_error = var_error
                                info(f"    Variation error: {var_error:.6f}")
                        finally:
                            os.unlink(var_json_path)
                            
                    # Use best variation for next iteration
                    if best_variation and best_variation_error < error:
                        current_json = best_variation
                        info(f"  Using variation with error {best_variation_error:.6f}")
                    else:
                        # No improvement, try different strategy
                        current_json = self._modify_strategy(current_json, iteration)
                        
            finally:
                # Clean up temp file
                if os.path.exists(temp_json_path):
                    os.unlink(temp_json_path)
                    
        # Log summary
        info(f"\n=== Optimization Summary ===")
        info(f"Tested {len(self.tested_combinations)} configurations")
        info(f"Best error achieved: {self.best_error:.6f}")
        
        return self.best_json or {}
        
    def _analyze_error_output(self, output: str) -> str:
        """Analyze error output to determine error type"""
        if 'multiply' in output.lower() or 'mul' in output.lower():
            return 'multiply'
        elif 'concat' in output.lower():
            return 'concat'
        elif 'transpose' in output.lower():
            return 'transpose'
        elif 'dimension' in output.lower() or 'shape' in output.lower():
            return 'dimension'
        else:
            return 'unknown'
            
    def _generate_variations(self, json_data: Dict[str, Any], error_type: str) -> List[Dict[str, Any]]:
        """Generate variations based on error type"""
        variations = []
        operations = json_data.get('operations', [])
        
        if error_type == 'multiply':
            # Try different transpose permutations for Mul operations
            new_perms = [
                [0, 1, 2, 3, 4, 5],  # Identity
                [0, 4, 2, 3, 1, 5],  # Swap dims 1 and 4
                [0, 2, 1, 3, 4, 5],  # Swap dims 1 and 2
                [0, 1, 4, 3, 2, 5],  # Swap dims 2 and 4
            ]
            
            for perm in new_perms:
                variation = json_data.copy()
                variation['operations'] = []
                
                # Modify existing Mul operations
                for op in operations:
                    new_op = op.copy()
                    if 'Mul' in op.get('op_name', '') and 'pre_process_transpose_perm' in op:
                        new_op['pre_process_transpose_perm'] = perm
                    variation['operations'].append(new_op)
                    
                variations.append(variation)
                
        elif error_type == 'dimension':
            # Try removing some operations to see if simpler works better
            if len(operations) > 10:
                # Try with half the operations
                variation = json_data.copy()
                variation['operations'] = operations[:len(operations)//2]
                variations.append(variation)
                
        # Always include original as fallback
        variations.append(json_data)
        
        return variations
        
    def _modify_strategy(self, json_data: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Modify strategy when variations don't improve"""
        operations = json_data.get('operations', [])
        
        # Strategy 1: Reduce number of operations
        if iteration % 2 == 0 and len(operations) > 5:
            new_json = json_data.copy()
            new_json['operations'] = operations[:max(5, len(operations)//2)]
            return new_json
            
        # Strategy 2: Focus on different operation types
        if iteration % 3 == 0:
            new_json = json_data.copy()
            # Keep only Transpose operations
            new_json['operations'] = [op for op in operations if 'Transpose' in op.get('op_name', '')]
            return new_json
            
        # Default: return as-is
        return json_data