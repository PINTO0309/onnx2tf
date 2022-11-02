# onnx2tf
Self-Created Tools to convert ONNX files (NCHW) to TensorFlow format (NHWC). The purpose of this tool is to solve the massive Transpose extrapolation problem in [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) ([onnx-tf](https://pypi.org/project/onnx-tf/)). I don't need a Star, but give me a pull request.

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/193840307-fa69eace-05a9-4d93-9c5d-999cf88af28e.png" />
</p>

[![Downloads](https://static.pepy.tech/personalized-badge/onnx2tf?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/onnx2tf) ![GitHub](https://img.shields.io/github/license/PINTO0309/onnx2tf?color=2BAF2B) [![Python](https://img.shields.io/badge/Python-3.8-2BAF2B)](https://img.shields.io/badge/Python-3.8-2BAF2B) [![PyPI](https://img.shields.io/pypi/v/onnx2tf?color=2BAF2B)](https://pypi.org/project/onnx2tf/) [![CodeQL](https://github.com/PINTO0309/onnx2tf/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/onnx2tf/actions?query=workflow%3ACodeQL) [![DOI](https://zenodo.org/badge/541831874.svg)](https://zenodo.org/badge/latestdoi/541831874)

## Key concept
- [x] [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) is a very useful tool, but the performance of the generated TensorFlow models is significantly degraded due to the extrapolation of a large number of `Transpose` OPs before and after each OP during the format conversion from `NCHW` to `NHWC`. Therefore, I will make this tool myself as a derivative tool of [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) without extrapolating `Transpose`.
- [x] Most of the internal processing of the tool is full-scratch, but some of the more complex OPs have been adapted from [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow). I am very grateful to the engineers at International Business Machines Corporation / LeapMind / Microsoft for developing [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow).
- [x] Supported layers list. [Supported layers](#supported-layers)
- [x] Contributors to this repository should first read **[Contribution Guide](https://github.com/PINTO0309/onnx2tf/blob/main/CONTRIBUTING.md)**.

  https://user-images.githubusercontent.com/33194443/197319770-e7ef7174-66cd-4bc2-84be-59e1a251151d.mp4

- [x] Not only does it handle conversions of 4-dimensional inputs, such as `NCHW` to `NHWC`, but also the number of input dimensions in 3, 5, or even more dimensions. For example, `NCDHW` to `NDHWC`, etc. However, since 1-D, 2-D, 3-D and 6-D input may produce patterns that are mechanically difficult to convert, it should be possible to give parameters to externally modify the tool's behavior. See [Parameter replacement](#parameter-replacement)
- [x] If there are undefined dimensions in the input OP, the model structure is not fully optimized and conversion errors are very likely to occur.
- [x] Immediately following a `Reshape` OP with dimensional compression and dimensional decompression, there is a 95% probability that the model transformation operation will be disrupted and errors will occur. For example, patterns such as `[1,200,200,5]` -> `[1,200,-1]` or `[10,20,30,40,50]` -> `[10,2,10,30,10,4,50]` or `Flatten`. See [#8 Not able to reshape input in replace.json](https://github.com/PINTO0309/onnx2tf/issues/8)
- [x] TensorFlow's Convolution does not have an equivalent operation to ONNX's Padding operation. Therefore, a `Pad` OP is inserted immediately before a Convolution with Padding of size greater than 1.
- [x] Support conversion to TensorFlow saved model and TFLite (Float32/Float16).
- [x] Does not support quantization to INT8. For quantization, use the official TensorFlow converter to convert from saved_model to your own.
- [ ] Files exceeding the Protocol Buffers file size limit of 2GB are not supported. Therefore, the external format is not supported at the initial stage of tool creation.
- [x] If there are ONNX OPs that are not supported by TensorFlow, use [simple-onnx-processing-tools](https://github.com/PINTO0309/simple-onnx-processing-tools) to replace them with harmless OPs in advance and then use this tool to convert them. In other words, you can convert any model with your efforts.
- [x] ONNX splitting, merging, generating OPs, rewriting OP attributes, BGR<->RGB conversion, converting to JSON and editing in the IDE, batch size changes for undefined dimensions, and various other processing can be done with the [simple-onnx-processing-tools](https://github.com/PINTO0309/simple-onnx-processing-tools). Therefore, it is recommended that models with very complex structures be converted to TFLite after modifying the structure beforehand.
- [x] `BatchNormalization` supports only inference mode.
- [x] `LayerNormalization` supports only inference mode.
- [x] Only for `opset=11` or higher
- [x] If you do not like the generated TFLite OP name, edit it using [tflite2json2tflite](https://github.com/PINTO0309/tflite2json2tflite).
- [x] The generated Keras models cannot be used for retraining. If you want to train, you must build your own model.
- [x] When converting to TensorFlow.js, CoreML, etc., please generate saved_model with the `--output_signaturedefs` option and use the generated saved_model to convert with various converters. [tensorflowjs_converter](https://github.com/tensorflow/tfjs), [coremltools](https://github.com/apple/coremltools), [edgetpu_compilier](https://coral.ai/docs/edgetpu/compiler/)
- [x] There are many OPs on ONNX that do not support EdgeTPU. Therefore, if you need to generate an EdgeTPU model, please specify `--replace_***_to_pseudo_***` to convert your model. onnx2tf will attempt to replace the OP with an EdgeTPU-compatible OP whenever possible.
- [x] The main factors that cause accuracy degradation after model conversion are as follows
1. differences in Padding specifications
2. difference in Python division specification in the process of model transformation (error due to even rounding)
3. Divide epsilon without consideration
4. deprecated TrueDivision
5. support difference of powers
6. differences in interpolation operation specifications during resizing
7. Difference in arithmetic precision supported by each operation
8. Calculation error due to scaling up or down by specifying a `scale` when resizing images

The above differences often cannot be dealt with by simply converting the model in a straightforward manner. Therefore, you need to replace the model yourself in advance with an operation that is less prone to errors.
- [x] TFLite does not support `TrueDiv`(INT), so `TrueDiv` is avoided if possible.
- [x] Implement the `Resize` process for the 5D tensor.
- [x] Add process to replace `Asin` with `pseudo-Asin`.
- [x] Add process to replace `Acos` with `pseudo-Acos`.
- [x] Add process to replace `GatherND` with `pseudo-GatherND`.
- [x] Add process to replace `HardSwish` with `pseudo-HardSwish`.
- [x] Add process to replace `GridSample` with `pseudo-GridSample`.
- [x] Add process to replace `LeakyRelu` with `pseudo-LeakyRelu`.
- [x] Add process to replace `Power` with `pseudo-Power`.
- [x] Add process to replace `Neg` with `pseudo-Neg`.
- [x] Add process to replace `ArgMax` with `pseudo-ArgMax`.
- [x] Added option to fix dynamic batch size `N` to a specified number.
- [x] Added option to overwrite dynamic shape input OPs with static shape. `--overwrite_input_shape`
- [x] Output in Keras H5 format.
- [x] Automatically run [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) (onnxsim) backend and optimize onnx files before model transformation.
- [x] Added the ability to automatically generate each OP name and assign OP names to ONNX files in the old format.

## Demo
Video speed is adjusted approximately 50 times slower than actual speed.
![render1665941718294](https://user-images.githubusercontent.com/33194443/196049928-57520fc2-842d-459c-9f28-7ee5f040c226.gif)

## Environment
- onnx
- onnx-simplifier
- onnx_graphsurgeon
- simple_onnx_processing_tools
- tensorflow>=2.10.0

## Sample Usage
- HostPC
  ```
  $ docker run --rm -it \
  -v `pwd`:/workdir \
  -w /workdir \
  ghcr.io/pinto0309/onnx2tf:1.0.49

  or

  $ pip install -U onnx \
  && pip install -U nvidia-pyindex \
  && pip install -U onnx-graphsurgeon \
  && pip install -U onnxsim \
  && pip install -U simple_onnx_processing_tools \
  && pip install -U onnx2tf

  or

  $ pip install -e .
  ```

or

- Google Colaboratory Python3.8+
  ```
  !sudo add-apt-repository -y ppa:deadsnakes/ppa
  !sudo apt-get -y update
  !sudo apt-get -y install python3.9
  !sudo apt-get -y install python3.9-dev
  !sudo apt-get -y install python3-pip
  !sudo apt-get -y install python3.9-distutils
  !python3.9 -m pip install -U setuptools \
    && python3.9 -m pip install -U pip \
    && python3.9 -m pip install -U distlib
  !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
  !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2
  !python3.9 -m pip install tensorflow==2.10.0 \
    && python3.9 -m pip install -U onnx \
    && python3.9 -m pip install -U nvidia-pyindex \
    && python3.9 -m pip install -U onnx-graphsurgeon \
    && python3.9 -m pip install -U onnxsim \
    && python3.9 -m pip install -U simple_onnx_processing_tools \
    && python3.9 -m pip install -U onnx2tf
  ```

Run test.
```bash
$ wget https://github.com/PINTO0309/onnx2tf/releases/download/0.0.2/resnet18-v1-7.onnx
$ onnx2tf -i resnet18-v1-7.onnx -o saved_model
```
## CLI Parameter
```
$ onnx2tf -h

usage: onnx2tf
[-h]
(-i INPUT_ONNX_FILE_PATH | -V)
[-o OUTPUT_FOLDER_PATH]
[-osd]
[-oh5]
[-nuo]
[-nuonag]
[-b BATCH_SIZE]
[-ois OVERWRITE_INPUT_SHAPE [OVERWRITE_INPUT_SHAPE ...]]
[-k KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES [KEEP_NCW_OR_NCHW_OR_NCDHW_INPUT_NAMES ...]]
[-rari64 | -rarf32 | -rafi64 | -raff32]
[-fasr FUSED_ARGMAX_SCALE_RATIO]
[-rasin]
[-racos]
[-rlr]
[-rpw]
[-rgn]
[-rng]
[-rhs]
[-me]
[-prf PARAM_REPLACEMENT_FILE]
[-n]

optional arguments:
  -h, --help
    show this help message and exit

  -i INPUT_ONNX_FILE_PATH, --input_onnx_file_path INPUT_ONNX_FILE_PATH
    Input onnx file path.

  -V, --version
    Show version and exit.

  -o OUTPUT_FOLDER_PATH, --output_folder_path OUTPUT_FOLDER_PATH
    Output folder path. Default: "saved_model"

  -osd, --output_signaturedefs
    Signature is added to the output for serving or for conversion
    to other model formats. However, this can significantly reduce the speed
    of model conversion and significant increase the size of the model.

  -oh5, --output_h5
    Output in Keras H5 format.

  -nuo, --not_use_onnxsim
    No optimization by onnx-simplifier is performed.
    If this option is used, the probability of a conversion error is very high.

  -nuonag, --not_use_opname_auto_generate
    Automatic generation of each OP name in the old format ONNX file
    and assignment of OP name are not performed.

  -b BATCH_SIZE, --batch_size BATCH_SIZE
    Fixes the dynamic batch size to the specified numeric batch size.
    A value of 1 or more must be specified.

  -ois OVERWRITE_INPUT_SHAPE [OVERWRITE_INPUT_SHAPE ...], \
      --overwrite_input_shape OVERWRITE_INPUT_SHAPE [OVERWRITE_INPUT_SHAPE ...]
    Overwrite the input shape.
    The format is
    "i1:dim0,...,dimN" "i2:dim0,...,dimN" "i3:dim0,...,dimN"
    When there is only one input, for example,
    "data:1,3,224,224"
    When there are multiple inputs, for example,
    "data1:1,3,224,224" "data2:1,3,112" "data3:5"
    A value of 1 or more must be specified.
    Numerical values other than dynamic dimensions are ignored.
    Ignores --batch_size if specified at the same time as --batch_size.

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
    and replace_argmax_to_reducemax_and_indicies_is_float32
    and replace_argmax_to_fused_argmax_and_indicies_is_int64
    and replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.

  -rarf32, --replace_argmax_to_reducemax_and_indicies_is_float32
    Replace ArgMax with a ReduceMax. The returned indicies are float32.
    Only one of replace_argmax_to_reducemax_and_indicies_is_int64
    and replace_argmax_to_reducemax_and_indicies_is_float32
    and replace_argmax_to_fused_argmax_and_indicies_is_int64
    and replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.

  -rafi64, --replace_argmax_to_fused_argmax_and_indicies_is_int64
    Replace ArgMax with a Fused_ArgMax. The returned indicies are int64.
    It improves inference speed at the cost of a small sacrifice in accuracy.
    See. https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/vision#argmax-fusion-to-improve-segmentation-model-latency
    Currently, only 4D tensors are supported.
    Only one of replace_argmax_to_reducemax_and_indicies_is_int64
    and replace_argmax_to_reducemax_and_indicies_is_float32
    and replace_argmax_to_fused_argmax_and_indicies_is_int64
    and replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.

  -raff32, --replace_argmax_to_fused_argmax_and_indicies_is_float32
    Replace ArgMax with a Fused_ArgMax. The returned indicies are float32.
    It improves inference speed at the cost of a small sacrifice in accuracy.
    See. https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/vision#argmax-fusion-to-improve-segmentation-model-latency
    Currently, only 4D tensors are supported.
    Only one of replace_argmax_to_reducemax_and_indicies_is_int64
    and replace_argmax_to_reducemax_and_indicies_is_float32
    and replace_argmax_to_fused_argmax_and_indicies_is_int64
    and replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.

  -fasr FUSED_ARGMAX_SCALE_RATIO, --fused_argmax_scale_ratio FUSED_ARGMAX_SCALE_RATIO
    For Fused ArgMax.
    Scale ratio when generating Fused ArgMax.
    0.0 < fused_argmax_scale_ratio <= 1.0
    Default: 0.5

  -rasin, --replace_asin_to_pseudo_asin
    Replace Asin with a pseudo Asin.

  -racos, --replace_acos_to_pseudo_acos
    Replace Acos with a pseudo Acos.

  -rlr, --replace_leakyrelu_to_pseudo_leakyrelu
    Replace LeakyReLU with a pseudo LeakyReLU.

  -rpw, --replace_power_to_pseudo_power
    Replace Power with a pseudo Power.

  -rgn, --replace_gathernd_to_pseudo_gathernd
    Replace GatherND with a pseudo GatherND.

  -rng, --replace_neg_to_pseudo_neg
    Replace Neg with a pseudo Neg.

  -rhs, --replace_hardswish_to_pseudo_hardswish
    Replace HardSwish with a pseudo HardSwish.

  -me, --mvn_epsilon
    For MeanVarianceNormalization.
    The number to be added to the variance to avoid division by zero
    when normalizing the value.
    (input_tensor - mean) / tf.sqrt(variance + mvn_epsilon)
    Default: 0.0000000001

  -prf PARAM_REPLACEMENT_FILE, --param_replacement_file PARAM_REPLACEMENT_FILE
    Parameter replacement file path. (.json)

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
  output_signaturedefs: Optional[bool] = False,
  output_h5: Optional[bool] = False,
  not_use_onnxsim: Optional[bool] = False,
  not_use_opname_auto_generate: Optional[bool] = False,
  batch_size: Union[int, NoneType] = None,
  overwrite_input_shape: Union[List[str], NoneType] = None,
  keep_ncw_or_nchw_or_ncdhw_input_names: Union[List[str], NoneType] = None,
  replace_argmax_to_reducemax_and_indicies_is_int64: Union[bool, NoneType] = False,
  replace_argmax_to_reducemax_and_indicies_is_float32: Union[bool, NoneType] = False,
  replace_argmax_to_fused_argmax_and_indicies_is_int64: Union[bool, NoneType] = False,
  replace_argmax_to_fused_argmax_and_indicies_is_float32: Union[bool, NoneType] = False,
  fused_argmax_scale_ratio: Union[float, NoneType] = 0.5,
  replace_asin_to_pseudo_asin: Union[bool, NoneType] = False,
  replace_acos_to_pseudo_acos: Union[bool, NoneType] = False,
  replace_leakyrelu_to_pseudo_leakyrelu: Union[bool, NoneType] = False,
  replace_power_to_pseudo_power: Optional[bool] = False,
  replace_gathernd_to_pseudo_gathernd: Optional[bool] = False,
  replace_neg_to_pseudo_neg: Optional[bool] = False,
  replace_hardswish_to_pseudo_hardswish: Optional[bool] = False,
  mvn_epsilon: Union[float, NoneType] = 0.0000000001,
  param_replacement_file: Optional[str] = '',
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

    output_signaturedefs: Optional[bool]
      Signature is added to the output for serving or for conversion
      to other model formats. However, this can significantly reduce the speed
      of model conversion and significant increase the size of the model.

    output_h5: Optional[bool]
      Output in Keras H5 format.

    not_use_onnxsim: Optional[bool]
      No optimization by onnx-simplifier is performed.
      If this option is used, the probability of a conversion error is very high.

    not_use_opname_auto_generate: Optional[bool]
      Automatic generation of each OP name in the old format ONNX file
      and assignment of OP name are not performed.

    batch_size: Optional[int]
      Fixes the dynamic batch size to the specified numeric batch size.
      A value of 1 or more must be specified.

    overwrite_input_shape: Optional[List[str]]
      Overwrite the input shape.
      The format is
      ['i1:dim0,dim1,...,dimN' 'i2:dim0,dim1,...,dimN' 'i3:dim0,dim1,...,dimN']
      When there is only one input, for example,
      ['data:1,3,224,224']
      When there are multiple inputs, for example,
      ['data1:1,3,224,224','data2:1,3,112','data3:5']
      A value of 1 or more must be specified.
      Numerical values other than dynamic dimensions are ignored.
      Ignores --batch_size if specified at the same time as --batch_size.

    keep_ncw_or_nchw_or_ncdhw_input_names: Optional[List[str]]
      Holds the NCW or NCHW or NCDHW of the input shape for the specified INPUT OP names.
      If a nonexistent INPUT OP name is specified, it is ignored.
      Valid only for 3D, 4D and 5D input tensors.
      e.g.
      --keep_ncw_or_nchw_or_ncdhw_input_names=['input0', 'input1', 'input2']

    replace_argmax_to_reducemax_and_indicies_is_int64: Optional[bool]
      Replace ArgMax with a ReduceMax. The returned indicies are int64.
      Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and
      replace_argmax_to_reducemax_and_indicies_is_float32 and
      replace_argmax_to_fused_argmax_and_indicies_is_int64 and
      replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.
      Default: False

    replace_argmax_to_reducemax_and_indicies_is_float32: Optional[bool]
      Replace ArgMax with a ReduceMax. The returned indicies are float32.
      Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and
      replace_argmax_to_reducemax_and_indicies_is_float32 and
      replace_argmax_to_fused_argmax_and_indicies_is_int64 and
      replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.
      Default: False

    replace_argmax_to_fused_argmax_and_indicies_is_int64: Optional[bool]
      Replace ArgMax with a ReduceMax. The returned indicies are int64.
      It improves inference speed at the cost of a small sacrifice in accuracy.
      See. https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/vision#argmax-fusion-to-improve-segmentation-model-latency
      Currently, only 4D tensors are supported.
      Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and
      replace_argmax_to_reducemax_and_indicies_is_float32 and
      replace_argmax_to_fused_argmax_and_indicies_is_int64 and
      replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.
      Default: False

    replace_argmax_to_fused_argmax_and_indicies_is_float32: Optional[bool]
      Replace ArgMax with a ReduceMax. The returned indicies are float32.
      It improves inference speed at the cost of a small sacrifice in accuracy.
      See. https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/vision#argmax-fusion-to-improve-segmentation-model-latency
      Currently, only 4D tensors are supported.
      Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and
      replace_argmax_to_reducemax_and_indicies_is_float32 and
      replace_argmax_to_fused_argmax_and_indicies_is_int64 and
      replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.
      Default: False

    fused_argmax_scale_ratio: Optional[float]
      For Fused ArgMax.
      Scale ratio when generating Fused ArgMax.
      0.0 < fused_argmax_scale_ratio <= 1.0
      Default: 0.5

    replace_asin_to_pseudo_asin: Optional[bool]
      Replace Asin with a pseudo Asin.

    replace_acos_to_pseudo_acos: Optional[bool]
      Replace Acos with a pseudo Acos.

    replace_leakyrelu_to_pseudo_leakyrelu: Optional[bool]
      Replace LeakyReLU with a pseudo LeakyReLU.

    replace_power_to_pseudo_power: Optional[bool]
      Replace Power with a pseudo Power.

    replace_gathernd_to_pseudo_gathernd: Optional[bool]
      Replace GatherND with a pseudo GatherND.

    replace_neg_to_pseudo_neg: Optional[bool]
      Replace Neg with a pseudo Neg.

    replace_hardswish_to_pseudo_hardswish: Optional[bool]
      Replace HardSwish with a pseudo HardSwish.

    mvn_epsilon: Optional[float]
      For MeanVarianceNormalization.
      The number to be added to the variance to avoid division by zero
      when normalizing the value.
      (input_tensor - mean) / tf.sqrt(variance + mvn_epsilon)
      Default: 0.0000000001

    param_replacement_file: Optional[str]
      Parameter replacement file path. (.json)

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

## Parameter replacement
This tool is used to convert `NCW` to `NWC`, `NCHW` to `NHWC`, `NCDHW` to `NDHWC`, `NCDDHW` to `NDDHWC`, `NCDDDDDDHW` to `NDDDDDDHWC`. Therefore, as stated in the Key Concepts, the conversion will inevitably break down at some point in the model. You need to look at the entire conversion log to see which OP transpositions are failing and correct them yourself. I dare to explain very little because I know that no matter how much detail I put in the README, you guys will not read it at all. `attribute` or `INPUT constant` or `INPUT Initializer` can be replaced with the specified value.

1. "A conversion error occurs."
2. "Output results are wrong."

Please don't post such low level questions as issues.

- convert option
  ```
  --param_replacement_file param_replacement.json
  ```

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
        "op_name": "StatefulPartitionedCall/Cast_3",
        "param_target": "attributes", # attributes or inputs
        "param_name": "to",
        "values": 1 # Disable parameter transposition or overwrite "to" parameters
      },
      {
        "op_name": "Resize__697",
        "param_target": "inputs",
        "param_name": "Concat__696:0",
        "values": [26,26] # Replacement of unk__x (Resize OP, sizes height/width parameter)
      },
      {
        "op_name": "Transpose__927",
        "param_target": "attributes",
        "param_name": "perm",
        "values": [0,1,2,3] # Disable parameter transposition or overwrite "perm" parameters
      },
      {
        "op_name": "StatefulPartitionedCall/functional_1/max_unpooling2d_2/Reshape_1",
        "param_target": "inputs",
        "param_name": "const_fold_opt__911",
        "values": [4,131072] # Overwrite "shape" parameters
      },
      {
        "op_name": "Reshape_25",
        "param_target": "outputs",
        "param_name": "onnx::InstanceNormalization_270",
        "post_process_transpose_perm": [0,2,1] # Extrapolate 3D Transpose after Reshape
      },
      {
        "op_name": "Reshape_30",
        "param_target": "outputs",
        "param_name": "onnx::Mul_275",
        "post_process_transpose_perm": [0,2,3,1] # Extrapolate 4D Transpose after Reshape
      },
      {
        "op_name": "flatten_1127",
        "param_target": "inputs",
        "param_name": "dropout0",
        "pre_process_transpose_perm": [0,3,1,2]
      }
    ]
  }
  ```
- Replacement Supported OPs
  |No.|OP type|Remarks|
  |:-:|:-|:-|
  |1|Cast|<table><thead><th>Type</th><th align="right">Values</th><th>Type</th><th align="right">Values</th></thead><tbody><tr><td>float16</td><td align="right">10</td><td>int8</td><td align="right">3</td></tr><tr><td>float32</td><td align="right">1</td><td>int16</td><td align="right">5</td></tr><tr><td>float64</td><td align="right">11</td><td>int32</td><td align="right">6</td></tr><tr><td>bool</td><td align="right">9</td><td>int64</td><td align="right">7</td></tr><tr><td>uint8</td><td align="right">2</td><td colspan="2" rowspan="4"></td></tr><tr><td>uint16</td><td align="right">4</td></tr><tr><td>uint32</td><td align="right">12</td></tr><tr><td>uint64</td><td align="right">13</td></tr></tbody></table>|
  |2|Div||
  |3|Flatten|1. "param_target": "attributes"<br>`axis`: Value of `axis`<br>2. "param_target": "inputs"<br>`pre_process_transpose_perm`: Transpose is applied to the tensor before the Reshape operation with the perm specified as pre-processing.<br>3. "param_target": "outputs"<br>`post_process_transpose_perm`: Transpose is applied to the tensor after the Reshape operation with the perm specified as post-processing.|
  |4|Gemm||
  |5|Mul||
  |6|Reshape|1. "param_target": "inputs"<br>`values`: Value of `shape`<br>`pre_process_transpose_perm`: Transpose is applied to the tensor before the Reshape operation with the perm specified as pre-processing.<br>2. "param_target": "outputs"<br>`post_process_transpose_perm`: Transpose is applied to the tensor after the Reshape operation with the perm specified as post-processing.|
  |7|Resize||
  |8|Softmax||
  |9|Sub||
  |10|Tile||
  |11|Transpose|1. "param_target": "attributes"<br>`perm`: Value of `perm`<br>2. "param_target": "inputs"<br>`values`: Value of `tensor`|
  |12|NonMaxSuppression||

## Supported layers
- https://github.com/onnx/onnx/blob/main/docs/Operators.md
- :heavy_check_mark:: Supported　**Help wanted**: Pull Request are welcome
  |OP|Status|
  |:-|:-:|
  |Abs|:heavy_check_mark:|
  |Acosh|:heavy_check_mark:|
  |Acos|:heavy_check_mark:|
  |Add|:heavy_check_mark:|
  |And|:heavy_check_mark:|
  |ArgMax|:heavy_check_mark:|
  |ArgMin|:heavy_check_mark:|
  |Asinh|:heavy_check_mark:|
  |Asin|:heavy_check_mark:|
  |Atanh|:heavy_check_mark:|
  |Atan|:heavy_check_mark:|
  |AveragePool|:heavy_check_mark:|
  |BatchNormalization|:heavy_check_mark:|
  |Bernoulli|:heavy_check_mark:|
  |BitShift|:heavy_check_mark:|
  |Cast|:heavy_check_mark:|
  |Ceil|:heavy_check_mark:|
  |Celu|:heavy_check_mark:|
  |Clip|:heavy_check_mark:|
  |Compress|:heavy_check_mark:|
  |ConcatFromSequence|:heavy_check_mark:|
  |Concat|:heavy_check_mark:|
  |ConstantOfShape|:heavy_check_mark:|
  |Constant|:heavy_check_mark:|
  |Conv|:heavy_check_mark:|
  |ConvTranspose|:heavy_check_mark:|
  |Cosh|:heavy_check_mark:|
  |Cos|:heavy_check_mark:|
  |CumSum|:heavy_check_mark:|
  |DepthToSpace|:heavy_check_mark:|
  |Det|:heavy_check_mark:|
  |DFT|**Help wanted**|
  |Div|:heavy_check_mark:|
  |Dropout|:heavy_check_mark:|
  |DynamicQuantizeLinear|:heavy_check_mark:|
  |Einsum|:heavy_check_mark:|
  |Elu|:heavy_check_mark:|
  |Equal|:heavy_check_mark:|
  |Erf|:heavy_check_mark:|
  |Expand|:heavy_check_mark:|
  |Exp|:heavy_check_mark:|
  |EyeLike|:heavy_check_mark:|
  |Flatten|:heavy_check_mark:|
  |Floor|:heavy_check_mark:|
  |GatherElements|:heavy_check_mark:|
  |GatherND|:heavy_check_mark:|
  |Gather|:heavy_check_mark:|
  |Gemm|:heavy_check_mark:|
  |GlobalAveragePool|:heavy_check_mark:|
  |GlobalLpPool|:heavy_check_mark:|
  |GlobalMaxPool|:heavy_check_mark:|
  |GreaterOrEqual|:heavy_check_mark:|
  |Greater|:heavy_check_mark:|
  |GridSample|:heavy_check_mark:|
  |GRU|**Help wanted**|
  |Hardmax|**Help wanted**|
  |HardSigmoid|:heavy_check_mark:|
  |HardSwish|:heavy_check_mark:|
  |Identity|:heavy_check_mark:|
  |If|**Help wanted**|
  |Input|:heavy_check_mark:|
  |InstanceNormalization|:heavy_check_mark:|
  |IsInf|:heavy_check_mark:|
  |IsNaN|:heavy_check_mark:|
  |LayerNormalization|:heavy_check_mark:|
  |LeakyRelu|:heavy_check_mark:|
  |LessOrEqual|:heavy_check_mark:|
  |Less|:heavy_check_mark:|
  |Log|:heavy_check_mark:|
  |LogSoftmax|:heavy_check_mark:|
  |Loop|**Help wanted**|
  |LpNormalization|:heavy_check_mark:|
  |LRN|:heavy_check_mark:|
  |LSTM|**Help wanted**|
  |MatMul|:heavy_check_mark:|
  |MatMulInteger|**Help wanted**|
  |MaxPool|:heavy_check_mark:|
  |Max|:heavy_check_mark:|
  |MaxRoiPool|**Help wanted**|
  |MaxUnpool|:heavy_check_mark:|
  |Mean|:heavy_check_mark:|
  |MeanVarianceNormalization|:heavy_check_mark:|
  |MelWeightMatrix|**Help wanted**|
  |Min|:heavy_check_mark:|
  |Mish|:heavy_check_mark:|
  |Mod|:heavy_check_mark:|
  |Mul|:heavy_check_mark:|
  |Multinomial|:heavy_check_mark:|
  |Neg|:heavy_check_mark:|
  |NonMaxSuppression|:heavy_check_mark:|
  |NonZero|:heavy_check_mark:|
  |Optional|**Help wanted**|
  |OptionalGetElement|**Help wanted**|
  |OptionalHasElement|**Help wanted**|
  |Not|:heavy_check_mark:|
  |OneHot|:heavy_check_mark:|
  |Or|:heavy_check_mark:|
  |Pad|:heavy_check_mark:|
  |Pow|:heavy_check_mark:|
  |PRelu|:heavy_check_mark:|
  |QLinearConv|**Help wanted**|
  |QLinearMatMul|**Help wanted**|
  |QuantizeLinear|:heavy_check_mark:|
  |RandomNormalLike|:heavy_check_mark:|
  |RandomNormal|:heavy_check_mark:|
  |RandomUniformLike|:heavy_check_mark:|
  |RandomUniform|:heavy_check_mark:|
  |Range|:heavy_check_mark:|
  |Reciprocal|:heavy_check_mark:|
  |ReduceL1|:heavy_check_mark:|
  |ReduceL2|:heavy_check_mark:|
  |ReduceLogSum|:heavy_check_mark:|
  |ReduceLogSumExp|:heavy_check_mark:|
  |ReduceMax|:heavy_check_mark:|
  |ReduceMean|:heavy_check_mark:|
  |ReduceMin|:heavy_check_mark:|
  |ReduceProd|:heavy_check_mark:|
  |ReduceSum|:heavy_check_mark:|
  |ReduceSumSquare|:heavy_check_mark:|
  |Relu|:heavy_check_mark:|
  |Reshape|:heavy_check_mark:|
  |Resize|:heavy_check_mark:|
  |ReverseSequence|:heavy_check_mark:|
  |RNN|**Help wanted**|
  |RoiAlign|:heavy_check_mark:|
  |Round|:heavy_check_mark:|
  |ScatterElements|:heavy_check_mark:|
  |ScatterND|:heavy_check_mark:|
  |Scan|**Help wanted**|
  |Selu|:heavy_check_mark:|
  |SequenceAt|:heavy_check_mark:|
  |SequenceConstruct|:heavy_check_mark:|
  |SequenceEmpty|:heavy_check_mark:|
  |SequenceErase|:heavy_check_mark:|
  |SequenceInsert|:heavy_check_mark:|
  |SequenceLength|:heavy_check_mark:|
  |Shape|:heavy_check_mark:|
  |Shrink|:heavy_check_mark:|
  |Sigmoid|:heavy_check_mark:|
  |Sign|:heavy_check_mark:|
  |Sinh|:heavy_check_mark:|
  |Sin|:heavy_check_mark:|
  |Size|:heavy_check_mark:|
  |Slice|:heavy_check_mark:|
  |Softmax|:heavy_check_mark:|
  |Softplus|:heavy_check_mark:|
  |Softsign|:heavy_check_mark:|
  |SpaceToDepth|:heavy_check_mark:|
  |Split|:heavy_check_mark:|
  |SplitToSequence|:heavy_check_mark:|
  |Sqrt|:heavy_check_mark:|
  |Squeeze|:heavy_check_mark:|
  |STFT|**Help wanted**|
  |StringNormalizer|**Help wanted**|
  |Sub|:heavy_check_mark:|
  |Sum|:heavy_check_mark:|
  |Tanh|:heavy_check_mark:|
  |Tan|:heavy_check_mark:|
  |TfIdfVectorizer|**Help wanted**|
  |ThresholdedRelu|:heavy_check_mark:|
  |Tile|:heavy_check_mark:|
  |TopK|:heavy_check_mark:|
  |Transpose|:heavy_check_mark:|
  |Trilu|:heavy_check_mark:|
  |Unique|**Help wanted**|
  |Unsqueeze|:heavy_check_mark:|
  |Upsample|:heavy_check_mark:|
  |Where|:heavy_check_mark:|
  |Xor|:heavy_check_mark:|

## Generated Model
- YOLOv7-tiny with Post-Process (NMS) ONNX to TFLite Float32
  https://github.com/PINTO0309/onnx2tf/releases/download/0.0.33/yolov7_tiny_head_0.768_post_480x640.onnx
  |onnx2tf|onnx-tensorflow<br>(Super redundant + Broken)|
  |:-:|:-:|
  |![image](https://user-images.githubusercontent.com/33194443/198160732-47ef9770-ecca-40dd-8502-be41c941f8e3.png)|![image](https://user-images.githubusercontent.com/33194443/195248761-9d4f4446-3fb4-41ad-a5d4-a7d211b527c0.png)|

- YOLACT-Edge MobileNetV2 with Post-Process (MultiClass-NMS) ONNX to TFLite Float32
  https://github.com/PINTO0309/onnx2tf/releases/download/1.0.11/yolact_edge_mobilenetv2_550x550.onnx
  ![model_float32 tflite](https://user-images.githubusercontent.com/33194443/196831377-891dfc49-6cdb-41e0-a723-8f0087817ddf.svg)

- MoveNet MultiPose ONNX to TFLite Float32 (`Cast` and `TrueDiv` standard OP support)
  https://github.com/PINTO0309/onnx2tf/releases/download/1.0.24/movenet_multipose_lightning_192x256_p6.onnx
  ![image](https://user-images.githubusercontent.com/33194443/198175219-b2db3ba3-65f8-464c-a0fd-411c4a62402e.png)

## Related tools
1. [tflite2tensorflow](https://github.com/PINTO0309/tflite2tensorflow)
2. [openvino2tensorflow](https://github.com/PINTO0309/openvino2tensorflow)
3. [tflite2json2tflite](https://github.com/PINTO0309/tflite2json2tflite)
4. [tensorflowjs_converter](https://github.com/tensorflow/tfjs)
5. [coremltools](https://github.com/apple/coremltools)
6. [simple-onnx-processing-tools](https://github.com/PINTO0309/simple-onnx-processing-tools)
7. [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
8. [onnx_graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon)
9. [onnx](https://github.com/onnx/onnx)
10. [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow)
11. [onnx2tflite](https://github.com/MPolaris/onnx2tflite)
12. [onnx2keras](https://github.com/gmalivenko/onnx2keras)
