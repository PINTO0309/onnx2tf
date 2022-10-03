# [WIP] onnx2tf
Self-Created Tools to convert ONNX files (NCHW) to TensorFlow format (NHWC). The purpose of this tool is to solve the massive Transpose extrapolation problem in [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) ([onnx-tf](https://pypi.org/project/onnx-tf/)).

[![Downloads](https://static.pepy.tech/personalized-badge/onnx2tf?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/onnx2tf) ![GitHub](https://img.shields.io/github/license/PINTO0309/onnx2tf?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/onnx2tf?color=2BAF2B)](https://pypi.org/project/onnx2tf/) [![CodeQL](https://github.com/PINTO0309/onnx2tf/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/onnx2tf/actions?query=workflow%3ACodeQL)

## Key concept
- [ ] onnx-tensorflow is a very useful tool, but the performance of the generated TensorFlow models is significantly degraded due to the extrapolation of a large number of `Transpose` OPs before and after each OP during the format conversion from `NCHW` to `NHWC`. Therefore, I will make this tool myself as a derivative tool of [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) without extrapolating `Transpose`.
- [ ] Not only does it handle conversions of 4-dimensional inputs, such as `NCHW` to `NHWC`, but also the number of input dimensions in 3, 5, or even more dimensions. For example, `NCDHW` to `NDHWC`, etc. However, since 1-D, 2-D, 3-D and 6-D input may produce patterns that are mechanically difficult to convert, it should be possible to give parameters to externally modify the tool's behavior.
- [ ] Immediately following a `Reshape` OP with dimensional compression and dimensional decompression, there is a 95% probability that the model transformation operation will be disrupted and errors will occur. For example, patterns such as `[1,200,200,5]` -> `[1,200,-1]` or `[10,20,30,40,50]` -> `[10,2,10,30,10,4,50]`.
- [ ] Support conversion to TensorFlow saved model and TFLite (Float32/Float16).
- [ ] Does not support quantization to INT8. For quantization, use the official TensorFlow converter to convert from saved_model to your own.
- [ ] Files exceeding the Protocol Buffers file size limit of 2GB are not supported. Therefore, the external format is not supported at the initial stage of tool creation.
- [ ] If there are ONNX OPs that are not supported by TensorFlow, use [simple-onnx-processing-tools](https://github.com/PINTO0309/simple-onnx-processing-tools) to replace them with harmless OPs in advance and then use this tool to convert them. In other words, you can convert any model with your efforts.
- [x] `BatchNormalization` supports only inference mode.
- [x] Only for `opset=11` or higher

## Demo
![render1664767369339](https://user-images.githubusercontent.com/33194443/193496368-58cd9af9-e1fc-4d02-bf0e-1a92694c3e98.gif)

# Sample Usage
```
$ pip install -U onnx2tf
```
```
$ wget https://github.com/PINTO0309/onnx2tf/releases/download/0.0.2/resnet18-v1-7.onnx
$ onnx2tf -i resnet18-v1-7.onnx -o saved_model
```
## CLI Parameter
```
$ onnx2tf -h

usage: onnx2tf
[-h]
-i INPUT_ONNX_FILE_PATH
[-o OUTPUT_FOLDER_PATH]
[-k KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES [KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES ...]]
[-rari64 | -rarf32]
[-rasin]
[-racos]
[-n]

optional arguments:
  -h, --help
    show this help message and exit

  -i INPUT_ONNX_FILE_PATH, --input_onnx_file_path INPUT_ONNX_FILE_PATH
    Input onnx file path.

  -o OUTPUT_FOLDER_PATH, --output_folder_path OUTPUT_FOLDER_PATH
    Output folder path. Default: "saved_model"

  -k KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES [KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES ...], --keep_ncw_or_nchw_or_ncdhw_input_names KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES [KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES ...]
    Holds the NCW or NCHW or NCDHW of the input shape for the specified INPUT OP names.
    If a nonexistent INPUT OP name is specified, it is ignored. Valid only for 3D, 4D and 5D input tensors.
    e.g. --keep_ncw_or_nchw_or_ncdhw_input_names "input0" "input1" "input2"

  -rari64, --replace_argmax_to_reducemax_and_indicies_is_int64
    Replace ArgMax with a ReduceMax. The returned indicies are int64.
    Only one of replace_argmax_to_reducemax_and_indicies_is_int64
    and replace_argmax_to_reducemax_and_indicies_is_float32 can be specified.

  -rarf32, --replace_argmax_to_reducemax_and_indicies_is_float32
    Replace ArgMax with a ReduceMax. The returned indicies are float32.
    Only one of replace_argmax_to_reducemax_and_indicies_is_int64
    and replace_argmax_to_reducemax_and_indicies_is_float32 can be specified.

  -rasin, --replace_asin_to_pseudo_asin
    Replace Asin with a pseudo Asin.

  -racos, --replace_acos_to_pseudo_acos
    Replace Acos with a pseudo Acos.

  -n, --non_verbose
    Do not show all information logs. Only error logs are displayed.
```

## In-script Usage
```python
>>> from onnx2tf import convert
>>> help(convert)

Help on function convert in module onnx2tf.onnx2tf:

convert(
  input_onnx_file_path: Union[str, NoneType] = '',
  onnx_graph: Union[onnx.onnx_ml_pb2.ModelProto, NoneType] = None,
  output_folder_path: Union[str, NoneType] = 'saved_model',
  keep_ncw_or_nchw_or_ncdhw_input_names: Union[List[str], NoneType] = None,
  replace_argmax_to_reducemax_and_indicies_is_int64: Union[bool, NoneType] = False,
  replace_argmax_to_reducemax_and_indicies_is_float32: Union[bool, NoneType] = False,
  replace_asin_to_pseudo_asin: Union[bool, NoneType] = False,
  replace_acos_to_pseudo_acos: Union[bool, NoneType] = False,
  non_verbose: Union[bool, NoneType] = False
) -> keras.engine.training.Model

    Convert ONNX to TensorFlow models.
    
    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.
        Either input_onnx_file_path or onnx_graph must be specified.
    
    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.
        Either input_onnx_file_path or onnx_graph must be specified.
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.
    
    output_folder_path: Optional[str]
        Output tensorflow model folder path.
        Default: "saved_model"

    keep_ncw_or_nchw_or_ncdhw_input_names: Optional[List[str]]
        Holds the NCW or NCHW or NCDHW of the input shape for the specified INPUT OP names.
        If a nonexistent INPUT OP name is specified, it is ignored.
        Valid only for 3D, 4D and 5D input tensors.
        e.g. 
        --keep_ncw_or_nchw_or_ncdhw_input_names=['input0', 'input1', 'input2']
    
    replace_argmax_to_reducemax_and_indicies_is_int64: Optional[bool]
        Replace ArgMax with a ReduceMax. The returned indicies are int64.
        Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and 
        replace_argmax_to_reducemax_and_indicies_is_float32 can be specified.
        Default: False
    
    replace_argmax_to_reducemax_and_indicies_is_float32: Optional[bool]
        Replace ArgMax with a ReduceMax. The returned indicies are float32.
        Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and 
        replace_argmax_to_reducemax_and_indicies_is_float32 can be specified.
        Default: False
    
    replace_asin_to_pseudo_asin: Optional[bool]
        Replace Asin with a pseudo Asin.
    
    replace_acos_to_pseudo_acos: Optional[bool]
        Replace Acos with a pseudo Acos.
    
    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.
        Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and 
        replace_argmax_to_reducemax_and_indicies_is_float32 can be specified.
        Default: False
    
    Returns
    ----------
    model: tf.keras.Model
        Model
```
