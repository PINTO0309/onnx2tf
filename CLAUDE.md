# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation
```bash
# Install all dependencies in the correct order
pip install -U pip
pip install -U onnx==1.17.0
pip install -U nvidia-pyindex
pip install -U onnx-graphsurgeon
pip install -U onnxruntime==1.18.1
pip install -U onnxsim==0.4.33
pip install -U simple_onnx_processing_tools
pip install -U sne4onnx>=1.0.13
pip install -U sng4onnx>=1.0.4
pip install -U ai_edge_litert==1.2.0
pip install -U tensorflow==2.19.0
pip install -U protobuf==3.20.3
pip install -U h5py==3.11.0
pip install -U psutil==5.9.5
pip install -U ml_dtypes==0.5.1
pip install -U tf-keras==2.19.0
pip install -U flatbuffers>=23.5.26
pip install -e .
```

### Docker Alternative
```bash
# Using pre-built Docker image
docker run --rm -it \
  -v `pwd`:/workdir \
  -w /workdir \
  docker.io/pinto0309/onnx2tf:1.27.10
```

### Testing
```bash
# Run model conversion tests
python tests/test_model_convert.py -m models -o wiki -v

# Test options:
# -m, --models: Specify ONNX model directory (default: models)
# -o, --output: Specify output directory (default: temp directory)
# -v, --verbose: Enable verbose output
# --dry-run: Process directory without actual conversion

# The test script downloads models from:
# https://s3.us-central-1.wasabisys.com/onnx2tf-en/models/resources.tar.gz
```

### Common Conversion Commands
```bash
# Basic conversion
onnx2tf -i model.onnx -o saved_model/

# Convert with INT8 quantization
onnx2tf -i model.onnx -oiqt -qt per-channel

# Convert with dynamic range quantization
onnx2tf -i model.onnx -odrqt

# Convert and output TFLite
onnx2tf -i model.onnx -o saved_model/ -osd

# Check GPU delegate compatibility
onnx2tf -i model.onnx -cgdc

# Automatic JSON generation for optimal conversion
onnx2tf -i model.onnx -agj

# Accuracy validation with auto-generated JSON
onnx2tf -i model.onnx -agj -cotof
```

### Building Package
```bash
# Build distribution packages
python setup.py sdist bdist_wheel

# The package entry point is:
# onnx2tf=onnx2tf:main
```

### Linting and Type Checking
```bash
# No specific linting commands are defined in the codebase
# Recommend using standard Python tools:
# - ruff check .
# - mypy onnx2tf/
```

### Environment Setup
```bash
# Disable CUDA for testing
export CUDA_VISIBLE_DEVICES=-1

# Minimize TensorFlow logging
export TF_CPP_MIN_LOG_LEVEL=3
```

## Code Architecture

### Core Conversion Flow
The main conversion logic is in `onnx2tf/onnx2tf.py` with the `convert()` function handling the ONNX to TensorFlow conversion. The conversion process follows NCHW (ONNX) to NHWC (TensorFlow) format transformation to solve transpose extrapolation issues.

### Operation Conversion Structure
Each ONNX operation has a corresponding converter in `onnx2tf/ops/`. The standard converter pattern:

```python
@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
```

Each converter follows these sections in order:

1. **Input Processing & Shape Transposition**: Handles dimension transposition (NCW→NWC, NCHW→NHWC, NCDHW→NDHWC)
2. **Graph Structure Preservation**: Maintains model graph metadata in dict format
3. **Parameter Replacement**: Applies user-defined parameter replacements from JSON
4. **Pre-process Transpose**: Uses `pre_process_transpose()` for input transformations
5. **TensorFlow Operation Generation**: Creates the equivalent TensorFlow operation
6. **Post-process Transpose**: Uses `post_process_transpose()` for output adjustments
7. **Debug Info Generation**: Produces conversion logs via `make_tf_node_info()`

Key decorators:
- `@print_node_info`: Outputs conversion logs and debug information
- `@inverted_operation_enable_disable`: Handles shape transposition logic and NCHW/NHWC conversions
- `@get_replacement_parameter`: Loads parameter replacements from JSON files

### Graph Structure Management
The conversion maintains the entire graph structure in `tf_layers_dict` with this format:
```python
tf_layers_dict[node_name] = {
    'optype': node.op,
    'shape': shape,
    'dtype': dtype,
    'tf_node': tf_operation,
    'before_op_output_shape_trans': bool  # Transposition flag
}
```

### Transposition Logic
The tool intelligently determines when to transpose tensors based on:
- Previous operation output shapes
- Current operation requirements
- The `before_op_output_shape_trans` flag tracking whether transposition is needed

### Parameter Replacement System
Users can provide JSON files to override ONNX parameters during conversion. The replacement targets can be:
- `'inputs'`: Input tensor values
- `'attributes'`: Operation attributes  
- `'outputs'`: Output configurations
- `'op'`: Operation type itself

### Automatic JSON Generation (-agj option)
The `onnx2tf/utils/json_auto_generator.py` module implements automatic parameter replacement JSON generation:
- `generate_auto_replacement_json()`: Main function that orchestrates the generation
- `OperationFixer` base class and specific fixer classes for each operation type
- Exhaustive search through parameter combinations to find minimal error
- Supports 22+ operation types from the README Parameter replacement section
- Uses iterative optimization with repeated accuracy validation

### Key Utility Functions
Common utilities from `onnx2tf/utils/common_functions.py`:
- `get_constant_or_variable()`: Retrieves tensor values from graph
- `replace_parameter()`: Applies JSON-based parameter replacements
- `pre_process_transpose()` / `post_process_transpose()`: Tensor transposition handling
- `transpose_with_flexing_deterrence()`: Smart transposition to minimize unnecessary transposes
- `explicit_broadcast()` / `pre_explicit_broadcast()`: Broadcasting operations
- `dummy_tf_inference()`: Test inference for shape validation
- `onnx_tf_tensor_validation()`: Validates conversion accuracy
- `make_tf_node_info()`: Generates debug information

### Testing Infrastructure
- Tests download models from S3: `https://s3.us-central-1.wasabisys.com/onnx2tf-en/models/resources.tar.gz`
- GitHub Actions runs conversion tests on pull requests via `.github/workflows/test-models.yml`
- Test results are posted to the wiki model status page at https://github.com/PINTO0309/onnx2tf/wiki/model_status
- Test script (`tests/test_model_convert.py`) validates both ONNX model integrity and conversion success

## Key Considerations

### Quantization
- INT8 quantization requires careful handling of activation functions
- SiLU/Swish activations cause catastrophic errors in INT8 - replace with ReLU
- HardSwish and ReLU6 should also be replaced with ReLU for INT8
- Non-zero constant padding can destroy quantization ranges

### Multiple Input Models
For models with multiple inputs, use the `-osd` option to generate a saved_model with proper signatures for correct INT8 calibration.

### Edge Cases and Limitations
- Some ONNX operations may not have direct TensorFlow equivalents
- The tool focuses on inference models (training operations may not be fully supported)
- Dynamic shapes are supported but may require shape hints via `-sh` parameter
- Custom operations can be added by creating new converters in `onnx2tf/ops/`

### Model Compatibility
- TensorFlow 2.13.0 and earlier: Use onnx2tf v1.17.5 or older
- TensorFlow 2.15.0 and later: Use latest onnx2tf (based on Keras API 3)

### GPU Delegate Optimization
Use `-cgdc` flag to check GPU delegate compatibility of generated TFLite models.

### Error Handling
- When conversion fails, check if `-agj` option can automatically generate optimal parameter replacements
- Use `-cotof` for full accuracy validation of all operations
- The tool raises exceptions instead of calling `sys.exit(1)` in utilities to allow error recovery