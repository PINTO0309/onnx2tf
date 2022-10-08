# [WIP] onnx2tf
Self-Created Tools to convert ONNX files (NCHW) to TensorFlow format (NHWC). The purpose of this tool is to solve the massive Transpose extrapolation problem in [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) ([onnx-tf](https://pypi.org/project/onnx-tf/)).

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/193840307-fa69eace-05a9-4d93-9c5d-999cf88af28e.png" />
</p>

[![Downloads](https://static.pepy.tech/personalized-badge/onnx2tf?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/onnx2tf) ![GitHub](https://img.shields.io/github/license/PINTO0309/onnx2tf?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/onnx2tf?color=2BAF2B)](https://pypi.org/project/onnx2tf/) [![CodeQL](https://github.com/PINTO0309/onnx2tf/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/onnx2tf/actions?query=workflow%3ACodeQL)

## Key concept
- [x] [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) is a very useful tool, but the performance of the generated TensorFlow models is significantly degraded due to the extrapolation of a large number of `Transpose` OPs before and after each OP during the format conversion from `NCHW` to `NHWC`. Therefore, I will make this tool myself as a derivative tool of [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) without extrapolating `Transpose`.
- [x] Most of the internal processing of the tool is full-scratch, but some of the more complex OPs have been adapted from [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow). I am very grateful to the engineers at International Business Machines Corporation / LeapMind / Microsoft for developing [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow).
- [x] Not only does it handle conversions of 4-dimensional inputs, such as `NCHW` to `NHWC`, but also the number of input dimensions in 3, 5, or even more dimensions. For example, `NCDHW` to `NDHWC`, etc. However, since 1-D, 2-D, 3-D and 6-D input may produce patterns that are mechanically difficult to convert, it should be possible to give parameters to externally modify the tool's behavior.
- [x] If there are undefined dimensions in the input OP, the model structure is not fully optimized and conversion errors are very likely to occur.
- [ ] Immediately following a `Reshape` OP with dimensional compression and dimensional decompression, there is a 95% probability that the model transformation operation will be disrupted and errors will occur. For example, patterns such as `[1,200,200,5]` -> `[1,200,-1]` or `[10,20,30,40,50]` -> `[10,2,10,30,10,4,50]`.
- [x] TensorFlow's Convolution does not have an equivalent operation to ONNX's Padding operation. Therefore, a `Pad` OP is inserted immediately before a Convolution with Padding of size greater than 1.
- [x] Support conversion to TensorFlow saved model and TFLite (Float32/Float16).
- [x] Does not support quantization to INT8. For quantization, use the official TensorFlow converter to convert from saved_model to your own.
- [ ] Files exceeding the Protocol Buffers file size limit of 2GB are not supported. Therefore, the external format is not supported at the initial stage of tool creation.
- [x] If there are ONNX OPs that are not supported by TensorFlow, use [simple-onnx-processing-tools](https://github.com/PINTO0309/simple-onnx-processing-tools) to replace them with harmless OPs in advance and then use this tool to convert them. In other words, you can convert any model with your efforts.
- [x] `BatchNormalization` supports only inference mode.
- [x] Only for `opset=11` or higher
- [x] If you do not like the generated TFLite OP name, edit it using [tflite2json2tflite](https://github.com/PINTO0309/tflite2json2tflite).
- [x] The generated Keras models cannot be used for retraining. If you want to train, you must build your own model.
- [ ] Implement the `Resize` process for the 5D tensor.
- [x] Add process to replace `Asin` with `pseudo-Asin`.
- [x] Add process to replace `Acos` with `pseudo-Acos`.
- [ ] Add process to replace `GatherND` with `pseudo-GatherND`.
- [x] Add process to replace `HardSwish` with `pseudo-HardSwish`.
- [ ] Add process to replace `GridSample` with `pseudo-GridSample`.
- [x] Add process to replace `LeakyRelu` with `pseudo-LeakyRelu`.
- [x] Added option to fix dynamic batch size `N` to a specified number.

## Demo
![render1664767369339](https://user-images.githubusercontent.com/33194443/193496368-58cd9af9-e1fc-4d02-bf0e-1a92694c3e98.gif)

## Sample Usage
```
$ docker run --rm -it \
-v `pwd`:/workdir \
-w /workdir \
ghcr.io/pinto0309/onn2tf:0.0.18

or

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
[-b BATCH_SIZE]
[-k KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES [KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES ...]]
[-rari64 | -rarf32]
[-rasin]
[-racos]
[-rlr]
[-n]

optional arguments:
  -h, --help
    show this help message and exit

  -i INPUT_ONNX_FILE_PATH, --input_onnx_file_path INPUT_ONNX_FILE_PATH
    Input onnx file path.

  -o OUTPUT_FOLDER_PATH, --output_folder_path OUTPUT_FOLDER_PATH
    Output folder path. Default: "saved_model"

  -b BATCH_SIZE, --batch_size BATCH_SIZE
    Fixes the dynamic batch size to the specified numeric batch size.
    A value of 1 or more must be specified.

  -k KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES [KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES ...], \
      --keep_ncw_or_nchw_or_ncdhw_input_names KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES \
          [KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES ...]
    Holds the NCW or NCHW or NCDHW of the input shape for the specified INPUT OP names.
    If a nonexistent INPUT OP name is specified, it is ignored.
    Valid only for 3D, 4D and 5D input tensors.
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

  -rlr, --replace_leakyrelu_to_pseudo_leakyrelu
    Replace LeakyReLU with a pseudo LeakyReLU.

  -n, --non_verbose
    Do not show all information logs. Only error logs are displayed.
```

## In-script Usage
```python
>>> from onnx2tf import convert
>>> help(convert)

Help on function convert in module onnx2tf:

convert(
  input_onnx_file_path: Union[str, NoneType] = '',
  onnx_graph: Union[onnx.onnx_ml_pb2.ModelProto, NoneType] = None,
  output_folder_path: Union[str, NoneType] = 'saved_model',
  batch_size: Union[int, NoneType] = None,
  keep_ncw_or_nchw_or_ncdhw_input_names: Union[List[str], NoneType] = None,
  replace_argmax_to_reducemax_and_indicies_is_int64: Union[bool, NoneType] = False,
  replace_argmax_to_reducemax_and_indicies_is_float32: Union[bool, NoneType] = False,
  replace_asin_to_pseudo_asin: Union[bool, NoneType] = False,
  replace_acos_to_pseudo_acos: Union[bool, NoneType] = False,
  replace_leakyrelu_to_pseudo_leakyrelu: Union[bool, NoneType] = False,
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

    batch_size: Optional[int]
        Fixes the dynamic batch size to the specified numeric batch size.
        A value of 1 or more must be specified.

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

    replace_leakyrelu_to_pseudo_leakyrelu: Optional[bool]
        Replace LeakyReLU with a pseudo LeakyReLU.

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

## [WIP] Parameter replacement
This tool is used to convert `NCW` to `NWC`, `NCHW` to `NHWC`, `NCDHW` to `NDHWC`, `NCDDHW` to `NDDHWC`, `NCDDDDDDHW` to `NDDDDDDHWC`. Therefore, as stated in the Key Concepts, the conversion will inevitably break down at some point in the model. You need to look at the entire conversion log to see which OP transpositions are failing and correct them yourself. I dare to explain very little because I know that no matter how much detail I put in the README, you guys will not read it at all.

"A conversion error occurs." Please don't post such low level questions as issues.

- param_replacement.json
```yaml
{
  "format_version": 1,
  "operations": [
    {
      "op_name": "StatefulPartitionedCall/Tile_4",
      "param_target": "inputs", # attributes or inputs
      "param_name": "const_fold_opt__677",
      "values": [1,1,17] # Disable parameter transposition or overwrite parameters
    },
    {
      "op_name": "StatefulPartitionedCall/Sum_3",
      "param_target": "attributes", # attributes or inputs
      "param_name": "axes",
      "values": [2] # Disable parameter transposition or overwrite parameters
    }
  ]
}
```

## Generated Model
![model_float32 tflite_30](https://user-images.githubusercontent.com/33194443/193499566-cfed15d9-ccb7-479e-a538-105c938d9b79.png)
