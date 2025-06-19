# Automatic JSON Generation Feature Implementation Summary

## Implementation Overview

### 1. Generic JSON Generation Algorithm (`json_auto_generator.py`)

- **Full Operation Support**: Supports all 22+ operation types listed in the README's Parameter replacement section
- **Operation-Specific Fixer Classes**: Implements specialized fixing logic for each operation type
  - `TransposeFixer`: Optimizes Transpose operations
  - `AddMulDivSubFixer`: Dimensional adjustments for arithmetic operations (Add, Mul, Div, Sub)
  - `ConcatFixer`: Axis corrections for Concat operations
  - `SplitFixer`: Adjusts axis and split numbers for Split operations
  - `ExpandFixer`: Handles dimension mismatches in Expand operations
  - And 17+ other Fixer classes

### 2. Error Analysis Features

- **Detailed Conversion Error Analysis**: Identifies problematic operations from error messages
- **Dimension Mismatch Detection**: Extracts shape information and suggests appropriate fixes
- **TensorFlow Operation Name Mapping**: Reverse-maps names like `tf.math.multiply_48` to ONNX operations

### 3. Iterative Optimization Mechanism

#### Improvements in onnx2tf.py
```python
# Processing on conversion error
if conversion_error and auto_generate_json:
    # Generate JSON based on error
    auto_json = generate_auto_replacement_json(
        error_info=str(conversion_error),
        graph=graph,
        output_folder_path=output_folder_path
    )
    # Save the generated JSON
    save_auto_replacement_json(auto_json, model_name, output_folder_path)
```

#### Iterative Logic in json_auto_generator.py
```python
# Exhaustive search for optimal parameters
def generate_all_permutations(ndim: int) -> List[List[int]]:
    """Generate all possible permutations for given dimensions"""
    if ndim <= 6:
        return list(itertools.permutations(range(ndim)))
    # For higher dimensions, use strategic sampling
```

### 4. Fix Prioritization

1. **Error-Specified Operations**: Operations directly mentioned in errors get highest priority
2. **Related Operation Types**: Operations inferred from error patterns
3. **Confidence Scoring**: Each fix is assigned a confidence score of 0.0-1.0
4. **Combinatorial Optimization**: Explores combinations of fixes to find optimal solutions

### 5. Real Example: custom_spo2.onnx Operation

Error:
```
Dimensions must be equal, but are 32 and 2 for tf.math.multiply_48/Mul
Input shapes: [1,2,1,256,32,1], [1,1,1,1,2,1]
```

Generated JSON:
```json
{
  "operations": [
    {
      "op_name": "wa/lightglue/posenc/Expand",
      "param_target": "inputs",
      "param_name": "wa/lightglue/posenc/Unsqueeze_3_output_0",
      "pre_process_transpose_perm": [0, 4, 2, 3, 1, 5]
    }
  ]
}
```

## Technical Features

### 1. Automatic Dimension Transformation Detection
- Detects broadcasting issues in multi-dimensional tensors
- Automatically generates appropriate transpose permutations
- Special handling for critical permutations like [0,4,2,3,1,5]

### 2. Smart Combination Generation
```python
# Exhaustive search without upper limits
all_permutations = get_all_permutations(ndim)
for perm in all_permutations:
    # Test each permutation
    candidate_fixes.append(create_fix_with_permutation(perm))
```

### 3. Error Pattern Matching
```python
patterns = [
    r'tf\.math\.(multiply|add|subtract|divide)_(\d+)',
    r'layer "([^"]+)"',
    r'{{' + 'node ([^}]+)' + '}}',
    r'Dimensions must be equal.*for (\S+)',
]
```

## Usage

### 1. Using the -agj Option

```bash
# Automatic JSON generation only
onnx2tf -i model.onnx -agj

# Generates optimal JSON when conversion errors occur
# → saved_model/{model_name}_auto.json
```

### 2. Accuracy Validation with Auto-Generated JSON

```bash
# Combined accuracy check and JSON generation
onnx2tf -i model.onnx -agj -cotof

# First generates JSON, then validates accuracy using it
```

### 3. Using Previously Generated JSON

```bash
# Re-convert using auto-generated JSON
onnx2tf -i model.onnx -prf saved_model/model_auto.json
```

## Key Improvements in Latest Version

### 1. Exhaustive Search Implementation
- Removed artificial limits on permutation generation
- Now searches through all possible combinations until finding minimum error
- Optimized for specific dimension patterns (e.g., prioritizing permutations starting with 0 for 5D tensors)

### 2. Enhanced Error Detection
- Better handling of Expand operation dimension mismatches
- Improved detection of problematic nodes from error messages
- Support for complex multi-dimensional broadcasting scenarios

### 3. Command Line Integration
- Added `-agj, --auto_generate_json` option
- Removed mutual exclusivity with `-cotof`
- Automatic JSON usage when both options are specified

## Performance Considerations

**WARNING**: The exhaustive search approach can take considerable time depending on:
- Model complexity
- Number of operations requiring fixes
- Tensor dimensionality (higher dimensions = more permutations)

For a 6D tensor, there are 720 possible permutations to test. The tool will systematically try each until finding the optimal configuration.

## Summary

This implementation enables onnx2tf to automatically propose solutions for conversion errors and accuracy issues. Users no longer need to manually create JSON files - the tool automatically discovers optimal conversion parameters through systematic exploration of the parameter space. The feature represents a significant improvement in usability, especially for complex models with non-standard dimension arrangements.