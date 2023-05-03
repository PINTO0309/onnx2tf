#! /usr/bin/env python

import os
import re
__path__ = (os.path.dirname(__file__), )
with open(os.path.join(__path__[0], '__init__.py')) as f:
    init_text = f.read()
    __version__ = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
import sys
sys.setrecursionlimit(10000)
import ast
import json
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import subprocess
import random
random.seed(0)
import numpy as np
np.random.seed(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.random.set_seed(0)
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.FATAL)
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

import onnx
import onnx_graphsurgeon as gs
from typing import Optional, List, Any, Dict
from argparse import ArgumentParser

import importlib
from onnx2tf.utils.common_functions import (
    dummy_onnx_inference,
    dummy_tf_inference,
    onnx_tf_tensor_validation,
    weights_export,
    download_test_image_data,
    get_tf_model_inputs,
    get_tf_model_outputs,
    rewrite_tflite_inout_opname,
)
from onnx2tf.utils.colors import Color
from sng4onnx import generate as op_name_auto_generate

def convert(
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_folder_path: Optional[str] = 'saved_model',
    output_signaturedefs: Optional[bool] = False,
    output_h5: Optional[bool] = False,
    output_keras_v3: Optional[bool] = False,
    output_tfv1_pb: Optional[bool] = False,
    output_weights: Optional[bool] = False,
    copy_onnx_input_output_names_to_tflite: Optional[bool] = False,
    output_integer_quantized_tflite: Optional[bool] = False,
    quant_type: Optional[str] = 'per-channel',
    custom_input_op_name_np_data_path: Optional[List] = None,
    input_output_quant_dtype: Optional[str] = 'int8',
    not_use_onnxsim: Optional[bool] = False,
    not_use_opname_auto_generate: Optional[bool] = False,
    batch_size: Optional[int] = None,
    overwrite_input_shape: Optional[List[str]] = None,
    no_large_tensor: Optional[bool] = False,
    output_nms_with_dynamic_tensor: Optional[bool] = False,
    keep_ncw_or_nchw_or_ncdhw_input_names: Optional[List[str]] = None,
    keep_nwc_or_nhwc_or_ndhwc_input_names: Optional[List[str]] = None,
    keep_shape_absolutely_input_names: Optional[List[str]] = None,
    output_names_to_interrupt_model_conversion: Optional[List[str]] = None,
    disable_group_convolution: Optional[bool] = False,
    enable_batchmatmul_unfold: Optional[bool] = False,
    enable_rnn_unroll: Optional[bool] = False,
    disable_suppression_flextranspose: Optional[bool] = False,
    number_of_dimensions_after_flextranspose_compression: Optional[int] = 6,
    disable_suppression_flexstridedslice: Optional[bool] = False,
    number_of_dimensions_after_flexstridedslice_compression: Optional[int] = 5,
    optimization_for_gpu_delegate: Optional[bool] = False,
    replace_argmax_to_reducemax_and_indicies_is_int64: Optional[bool] = False,
    replace_argmax_to_reducemax_and_indicies_is_float32: Optional[bool] = False,
    replace_argmax_to_fused_argmax_and_indicies_is_int64: Optional[bool] = False,
    replace_argmax_to_fused_argmax_and_indicies_is_float32: Optional[bool] = False,
    fused_argmax_scale_ratio: Optional[float] = 0.5,
    replace_to_pseudo_operators: List[str] = None,
    param_replacement_file: Optional[str] = '',
    check_gpu_delegate_compatibility: Optional[bool] = False,
    check_onnx_tf_outputs_elementwise_close: Optional[bool] = False,
    check_onnx_tf_outputs_elementwise_close_full: Optional[bool] = False,
    check_onnx_tf_outputs_sample_data_normalization: Optional[str] = 'norm',
    check_onnx_tf_outputs_elementwise_close_rtol: Optional[float] = 0.0,
    check_onnx_tf_outputs_elementwise_close_atol: Optional[float] = 1e-4,
    mvn_epsilon: Optional[float] = 0.0000000001,
    non_verbose: Optional[bool] = False,
) -> tf.keras.Model:
    """Convert ONNX to TensorFlow models.

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.\n
        Either input_onnx_file_path or onnx_graph must be specified.

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n
        Either input_onnx_file_path or onnx_graph must be specified.\n
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_folder_path: Optional[str]
        Output tensorflow model folder path.\n
        Default: "saved_model"

    output_signaturedefs: Optional[bool]
        Signature is added to the output for serving or for conversion\n
        to other model formats. However, this can significantly reduce the speed\n
        of model conversion and significant increase the size of the model.

    output_h5: Optional[bool]
        Output model in Keras (hdf5) format.

    output_keras_v3: Optional[bool]
        Output model in Keras (keras_v3) format.

    output_tfv1_pb: Optional[bool]
        Output model in TF v1 (.pb) format.

    output_weights: Optional[bool]
        Output weights in hdf5 format.

    copy_onnx_input_output_names_to_tflite: Optional[bool]
        Copy the input/output OP name of ONNX to the input/output OP name of tflite.\n
        Due to Tensorflow internal operating specifications,\n
        the input/output order of ONNX does not necessarily match\n
        the input/output order of tflite.\n
        Be sure to check that the input/output OP names in the generated\n
        tflite file have been converted as expected.\n
        Also, this option generates a huge JSON file as a temporary file for processing.\n
        Therefore, it is strongly discouraged to use it on large models of hundreds\n
        of megabytes or more.

    output_integer_quantized_tflite: Optional[bool]
        Output of integer quantized tflite.

    quant_type: Optional[str]
        Selects whether "per-channel" or "per-tensor" quantization is used.\n
        Default: "per-channel"

    custom_input_op_name_np_data_path: Optional[List]
        INPUT Name of OP and path of calibration data file (Numpy) for quantization\n
        and mean and std.\n
        The specification can be omitted only when the input OP is a single 4D tensor image data.\n
        If omitted, it is automatically calibrated using 20 normalized MS-COCO images.\n
        The type of the input OP must be Float32.\n
        Data for calibration must be pre-normalized to a range of 0 to 1.\n
        [\n
            [{input_op_name: str} {numpy_file_path: str} {mean: np.ndarray} {std: np.ndarray}],\n
            [{input_op_name: str} {numpy_file_path: str} {mean: np.ndarray} {std: np.ndarray}],\n
            [{input_op_name: str} {numpy_file_path: str} {mean: np.ndarray} {std: np.ndarray}],\n
            :\n
        ]\n
        Numpy file paths must be specified the same number of times as the number of input OPs.\n
        Normalize the value of the input OP based on the tensor specified in mean and std.\n
        (input_value - mean) / std\n
        Tensors in Numpy file format must be in dimension order after conversion to TF.\n
        Note that this is intended for deployment on low-resource devices,\n
        so the batch size is limited to 1 only.\n\n
        e.g.\n
        The example below shows a case where there are three input OPs.\n
        Assume input0 is 128x128 RGB image data.\n
        In addition, input0 should be a value that has been divided by 255\n
        in the preprocessing and normalized to a range between 0 and 1.\n
        input1 and input2 assume the input of something that is not an image.\n
        Because input1 and input2 assume something that is not an image,\n
        the divisor is not 255 when normalizing from 0 to 1.\n
        "n" is the number of calibration data.\n\n
        ONNX INPUT shapes:\n
            input0: [n,3,128,128]\n
                mean: [1,3,1,1] -> [[[[0.485]],[[0.456]],[[0.406]]]]\n
                std:  [1,3,1,1] -> [[[[0.229]],[[0.224]],[[0.225]]]]\n
            input1: [n,64,64]\n
                mean: [1,64] -> [0.1, ..., 0.64]\n
                std:  [1,64] -> [0.05, ..., 0.08]\n
            input2: [n,5]\n
                mean: [1] -> [0.3]\n
                std:  [1] -> [0.07]\n
        TensorFlow INPUT shapes (Numpy file ndarray shapes):\n
            input0: [n,128,128,3]\n
                mean: [1,1,1,3] -> [[[[0.485, 0.456, 0.406]]]]\n
                std:  [1,1,1,3] -> [[[[0.229, 0.224, 0.225]]]]\n
            input1: [n,64,64]\n
                mean: [1,64] -> [0.1, ..., 0.64]\n
                std:  [1,64] -> [0.05, ..., 0.08]\n
            input2: [n,5]\n
                mean: [1] -> [0.3]\n
                std:  [1] -> [0.07]\n
        cind=[
            ["input0","../input0.npy",[[[[0.485, 0.456, 0.406]]]],[[[[0.229, 0.224, 0.225]]]]],\n
            ["input1","./input1.npy",[0.1, ..., 0.64],[0.05, ..., 0.08]],\n
            ["input2","input2.npy",[0.3],[0.07]],\n
        ]

    input_output_quant_dtype: Optional[str]
        Input and Output dtypes when doing Full INT8 Quantization.\n
        "int8"(default) or "uint8"

    not_use_onnxsim: Optional[bool]
        No optimization by onnx-simplifier is performed.\n
        If this option is used, the probability of a conversion error is very high.

    not_use_opname_auto_generate: Optional[bool]
        Automatic generation of each OP name in the old format ONNX file\n
        and assignment of OP name are not performed.

    batch_size: Optional[int]
        Fixes the dynamic batch size to the specified numeric batch size.\n
        A value of 1 or more must be specified.

    overwrite_input_shape: Optional[List[str]]
        Overwrite the input shape.\n
        The format is\n
        ["input_name_1:dim0,...,dimN","input_name_2:dim0,...,dimN","input_name_3:dim0,...,dimN"].\n
        When there is only one input, for example,\n
        ['data:1,3,224,224']\n
        When there are multiple inputs, for example,\n
        ['data1:1,3,224,224','data2:1,3,112,112','data3:5']\n
        A value of 1 or more must be specified.\n
        Numerical values other than dynamic dimensions are ignored.\n
        Ignores batch_size if specified at the same time as batch_size.

    no_large_tensor: Optional[bool]
        Suppresses constant bloat caused by Tile OP when optimizing models in onnxsim.\n
        See: https://github.com/daquexian/onnx-simplifier/issues/178

    output_nms_with_dynamic_tensor: Optional[bool]
        The number of bounding boxes in the NMS output results is\n
        not fixed at the maximum number of max_output_boxes_per_class,\n
        but rather at the smallest possible number of dynamic tensors.\n
        If this option is disabled, NMS output is padded to the number\n
        set in the max_output_boxes_per_class attribute.\n
        e.g.\n
        disable --output_nms_with_dynamic_tensor:\n
            output_tensor_shape: [100, 7]\n
        enable --output_nms_with_dynamic_tensor:\n
            output_tensor_shape: [N, 7]

    keep_ncw_or_nchw_or_ncdhw_input_names: Optional[List[str]]
        Holds the NCW or NCHW or NCDHW of the input shape for the specified INPUT OP names.\n
        If a nonexistent INPUT OP name is specified, it is ignored.\n
        Valid only for 3D, 4D and 5D input tensors.\n\n
        e.g. \n
        keep_ncw_or_nchw_or_ncdhw_input_names=['input0','input1','input2']

    keep_nwc_or_nhwc_or_ndhwc_input_names: Optional[List[str]]
        Holds the NWC or NHWC or NDHWC of the input shape for the specified INPUT OP names.\n
        If a nonexistent INPUT OP name is specified, it is ignored.\n
        If the input OP name is the same as the input OP name specified\n
        in the keep_ncw_or_nchw_or_ncdhw_input_names option, it is ignored.\n
        Valid only for 3D, 4D and 5D input tensors.\n\n
        e.g. \n
        keep_nwc_or_nhwc_or_ndhwc_input_names=['input0','input1','input2']

    keep_shape_absolutely_input_names: Optional[List[str]]
        Name of the INPUT that unconditionally maintains its shape.\n
        If a nonexistent INPUT OP name is specified, it is ignored.\n\n
        e.g.\n
        keep_shape_absolutely_input_names=['input0','input1','input2']

    output_names_to_interrupt_model_conversion: Optional[List[str]]
        Output names that interrupt model conversion.\n
        Interrupts model transformation at the specified output name\n
        and outputs the model partitioned into subgraphs.\n\n
        e.g.\n
        output_names_to_interrupt_model_conversion=['output0','output1','output2']

    disable_group_convolution: Optional[bool]
        Disable GroupConvolution and replace it with SeparableConvolution\n
        for output to saved_model format.

    enable_batchmatmul_unfold: Optional[bool]
        BatchMatMul is separated batch by batch to generate a primitive MatMul.

    enable_rnn_unroll: Optional[bool]
        Instead of increasing inference speed by expanding all symbolic loops of the RNN (LSTM, GRU, RNN),\n
        RAM consumption will increase because all tensors are expanded and embedded in the model.\n
        https://keras.io/api/layers/recurrent_layers/

    disable_suppression_flextranspose: Optional[bool]
        Disables FlexTranspose generation suppression.

    number_of_dimensions_after_flextranspose_compression: Optional[int]
        Number of Transpose OP dimensions generated after avoiding FlexTranspose generation.\n
        Also suppress the creation of the Transpose itself by specifying 2.\n
        Default: 6

    disable_suppression_flexstridedslice: Optional[bool]
        Disables FlexStridedSlice generation suppression.

    number_of_dimensions_after_flexstridedslice_compression: Optional[int]
        Number of StridedSlice OP dimensions generated after avoiding FlexStridedSlice generation.\n
        Default: 5

    optimization_for_gpu_delegate: Optional[bool]
        Replace operations that do not support gpu delegate with those\n
        that do as much as possible.

    replace_argmax_to_reducemax_and_indicies_is_int64: Optional[bool]
        Replace ArgMax with a ReduceMax. The returned indicies are int64.\n
        Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and \n
        replace_argmax_to_reducemax_and_indicies_is_float32 and \n
        replace_argmax_to_fused_argmax_and_indicies_is_int64 and \n
        replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.\n
        Default: False

    replace_argmax_to_reducemax_and_indicies_is_float32: Optional[bool]
        Replace ArgMax with a ReduceMax. The returned indicies are float32.\n
        Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and \n
        replace_argmax_to_reducemax_and_indicies_is_float32 and \n
        replace_argmax_to_fused_argmax_and_indicies_is_int64 and \n
        replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.\n
        Default: False

    replace_argmax_to_fused_argmax_and_indicies_is_int64: Optional[bool]
        Replace ArgMax with a ReduceMax. The returned indicies are int64.\n
        It improves inference speed at the cost of a small sacrifice in accuracy.\n
        See. https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/vision#argmax-fusion-to-improve-segmentation-model-latency\n
        Currently, only 4D tensors are supported.\n
        Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and \n
        replace_argmax_to_reducemax_and_indicies_is_float32 and \n
        replace_argmax_to_fused_argmax_and_indicies_is_int64 and \n
        replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.\n
        Default: False

    replace_argmax_to_fused_argmax_and_indicies_is_float32: Optional[bool]
        Replace ArgMax with a ReduceMax. The returned indicies are float32.\n
        It improves inference speed at the cost of a small sacrifice in accuracy.\n
        See. https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/vision#argmax-fusion-to-improve-segmentation-model-latency\n
        Currently, only 4D tensors are supported.\n
        Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and \n
        replace_argmax_to_reducemax_and_indicies_is_float32 and \n
        replace_argmax_to_fused_argmax_and_indicies_is_int64 and \n
        replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.\n
        Default: False

    fused_argmax_scale_ratio: Optional[float]
        For Fused ArgMax.\n
        Scale ratio when generating Fused ArgMax.\n
        0.0 < fused_argmax_scale_ratio <= 1.0\n
        Default: 0.5

    replace_to_pseudo_operators: List[str]
        Replace list of operators to pseudo operators. \n
        Full name of the target operators should be given. \n
        Currently supported operators : \n
        Asin, Acos, Atan, Abs, PReLU, LeakyReLU, Power, GatherND, Neg, HardSwish, Erf

    mvn_epsilon: Optional[float]
        For MeanVarianceNormalization.\n
        The number to be added to the variance to avoid division by zero when normalizing the value.\n
        (input_tensor - mean) / tf.sqrt(variance + mvn_epsilon)\n
        Default: 0.0000000001

    param_replacement_file: Optional[str]
        Parameter replacement file path. (.json)

    check_gpu_delegate_compatibility: Optional[bool]
        Run TFLite ModelAnalyzer on the generated Float16 tflite model\n
        to check if the model can be supported by GPU Delegate.

    check_onnx_tf_outputs_elementwise_close: Optional[bool]
        Returns "Matches" if the output of onnx and the output of TF are\n
        within acceptable proximity element by element.\n
        Returns "Unmatched" if the output of onnx and the output of TF are\n
        not within acceptable proximity element by element.\n
        If the output of onnx is 1D, it returns "Skipped" and skips the comparison\n
        between the output of onnx and that of TF. This is because when undefined\n
        dimensions are present, a situation often arises where very large index\n
        values are compared, causing OutOfMemory.\n
        Only the output content of the models final output OP is checked.

    check_onnx_tf_outputs_elementwise_close_full: Optional[bool]
        Returns "Matches" if the output of onnx and the output of TF are\n
        within acceptable proximity element by element.\n
        Check the output of all OPs in sequence from the beginning,\n
        including all but the final output OP of the model.\n
        Returns "Unmatched" if the output of onnx and the output of TF are\n
        not within acceptable proximity element by element.\n
        If the output of onnx is 1D, it returns "Skipped" and skips the comparison\n
        between the output of onnx and that of TF. This is because when undefined\n
        dimensions are present, a situation often arises where very large index\n
        values are compared, causing OutOfMemory.\n
        It is very time consuming because it performs as many inferences as\n
        there are operations.

    check_onnx_tf_outputs_sample_data_normalization: Optional[str]
        norm: Validate using random data normalized to the range 0.0 to 1.0\n
        denorm: Validate using random data in the range 0.0 to 255.0\n
        If there is a normalization layer at the model's entry point,\n
        or if the model was trained on denormalized data, "denorm" must be specified.\n
        Default: "norm"

    check_onnx_tf_outputs_elementwise_close_rtol: Optional[float]
        The relative tolerance parameter.\n
        Default: 0.0

    check_onnx_tf_outputs_elementwise_close_atol: Optional[float]
        The absolute tolerance parameter.\n
        Default: 1e-4

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n
        Default: False

    Returns
    ----------
    model: tf.keras.Model
        Model
    """
    # Either designation required
    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    # If output_folder_path is empty, set the initial value
    if not output_folder_path:
        output_folder_path = 'saved_model'

    # Escape
    input_onnx_file_path = fr'{input_onnx_file_path}'
    output_folder_path = fr'{output_folder_path}'

    # Input file existence check
    if not os.path.exists(input_onnx_file_path) and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'The specified *.onnx file does not exist. ' +
            f'input_onnx_file_path: {input_onnx_file_path}'
        )
        sys.exit(1)

    # Extracting onnx filenames
    output_file_name = ''
    if input_onnx_file_path:
        output_file_name = os.path.splitext(
            os.path.basename(input_onnx_file_path)
        )[0]
    else:
        output_file_name = 'model'

    # batch_size
    if batch_size is not None and batch_size <= 0:
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'batch_size must be greater than or equal to 1. batch_size: {batch_size}'
        )
        sys.exit(1)

    # overwrite_input_shape
    if overwrite_input_shape is not None \
        and not isinstance(overwrite_input_shape, list):
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'overwrite_input_shape must be specified by list.'
        )
        sys.exit(1)

    # determination of errors in custom input
    if custom_input_op_name_np_data_path is not None:
        for param in custom_input_op_name_np_data_path:
            if len(param) not in [2, 4]:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} ' +
                    f"'-cind' option must have INPUT_NAME, NUMPY_FILE_PATH, MEAN(optional), STD(optional)"
                )
                sys.exit(1)

    # replace_argmax_to_reducemax_and_indicies_is_int64
    # replace_argmax_to_reducemax_and_indicies_is_float32
    # replace_argmax_to_fused_argmax_and_indicies_is_int64
    # replace_argmax_to_fused_argmax_and_indicies_is_float32
    ra_option_list = [
        replace_argmax_to_reducemax_and_indicies_is_int64,
        replace_argmax_to_reducemax_and_indicies_is_float32,
        replace_argmax_to_fused_argmax_and_indicies_is_int64,
        replace_argmax_to_fused_argmax_and_indicies_is_float32,
    ]
    if ra_option_list.count(True) > 1:
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and ' +
            f'replace_argmax_to_reducemax_and_indicies_is_float32 and ' +
            f'replace_argmax_to_fused_argmax_and_indicies_is_int64 and ' +
            f'replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.'
        )
        sys.exit(1)

    # fused_argmax_scale_ratio
    if ra_option_list.count(True) > 0 and not (0.0 < fused_argmax_scale_ratio <= 1.0):
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'fused_argmax_scale_ratio must be specified in the range '+
            f'0.0 < fused_argmax_scale_ratio <= 1.0. '+
            f'fused_argmax_scale_ratio: {fused_argmax_scale_ratio}'
        )
        sys.exit(1)

    # number_of_dimensions_after_flextranspose_compression
    if number_of_dimensions_after_flextranspose_compression < 2:
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'number_of_dimensions_after_flextranspose_compression must be at least 2. '+
            f'number_of_dimensions_after_flextranspose_compression: ' +
            f'{number_of_dimensions_after_flextranspose_compression}'
        )
        sys.exit(1)

    # number_of_dimensions_after_flexstridedslice_compression
    if number_of_dimensions_after_flexstridedslice_compression < 1:
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'number_of_dimensions_after_flexstridedslice_compression must be at least 1. '+
            f'number_of_dimensions_after_flexstridedslice_compression: ' +
            f'{number_of_dimensions_after_flexstridedslice_compression}'
        )
        sys.exit(1)

    replacement_parameters = None
    if param_replacement_file:
        if not os.path.isfile(param_replacement_file):
            print(
                f'{Color.RED}ERROR:{Color.RESET} ' +
                f'File specified in param_replacement_file not found. \n' +
                f'param_replacement_file: {param_replacement_file}'
            )
            sys.exit(1)
        try:
            with open(param_replacement_file, 'r') as f:
                replacement_parameters = json.load(f)['operations']
                for operations in replacement_parameters:
                    operations['op_name'] = operations['op_name'].replace(':','_')
                    if output_signaturedefs or output_integer_quantized_tflite:
                        operations['op_name'] = re.sub('^/', 'wa/', operations['op_name'])
        except json.decoder.JSONDecodeError as ex:
            print(
                f'{Color.RED}ERROR:{Color.RESET} ' +
                f'The file specified in param_replacement_file is not in JSON format. \n' +
                f'param_replacement_file: {param_replacement_file}'
            )
            sys.exit(1)

    # onnx-simplifier
    # To fully optimize the model, run onnxsim three times in a row.
    # Due to unstable script execution of onnxsim in v0.4.8,
    # I have no choice but to use subprocesses that we do not want to use.
    if not not_use_onnxsim:
        try:
            if not non_verbose:
                print('')
                print(f'{Color.REVERCE}Model optimizing started{Color.RESET}', '=' * 60)
            for _ in range(3):
                append_param = list(['--overwrite-input-shape'] + overwrite_input_shape) \
                    if overwrite_input_shape is not None else []
                append_param = append_param + ['--no-large-tensor', '10MB'] \
                    if no_large_tensor else append_param
                result = subprocess.check_output(
                    [
                        'onnxsim',
                        f'{input_onnx_file_path}',
                        f'{input_onnx_file_path}'
                    ] + append_param,
                    stderr=subprocess.PIPE
                ).decode('utf-8')
                if not non_verbose:
                    print(result)
            if not non_verbose:
                print(f'{Color.GREEN}Model optimizing complete!{Color.RESET}')
        except Exception as e:
            if not non_verbose:
                import traceback
                traceback.print_exc()
                print(
                    f'{Color.YELLOW}WARNING:{Color.RESET} '+
                    'Failed to optimize the onnx file.'
                )

    # Automatic generation of each OP name - sng4onnx
    if not not_use_opname_auto_generate:
        if not non_verbose:
            print('')
            print(f'{Color.REVERCE}Automatic generation of each OP name started{Color.RESET}', '=' * 40)
        try:
            op_name_auto_generate(
                input_onnx_file_path=f'{input_onnx_file_path}',
                onnx_graph=onnx_graph,
                output_onnx_file_path=f'{input_onnx_file_path}',
                non_verbose=True,
            )
            if not non_verbose:
                print(f'{Color.GREEN}Automatic generation of each OP name complete!{Color.RESET}')
        except Exception as e:
            if not non_verbose:
                import traceback
                traceback.print_exc()
                print(
                    f'{Color.YELLOW}WARNING:{Color.RESET} '+
                    'Failed to automatic generation of each OP name.'
                )

    # quantization_type
    disable_per_channel = False \
        if quant_type is not None and quant_type == 'per-channel' else True

    # Output model in TF v1 (.pb) format
    # Output signatures to saved_model
    if output_tfv1_pb:
        output_signaturedefs = True

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)

    graph = gs.import_onnx(onnx_graph)

    # List Output
    # Cut the ONNX graph when an output name is specified that interrupts the conversion
    if not output_names_to_interrupt_model_conversion:
        output_names = [
            graph_output.name for graph_output in graph.outputs
        ]
    else:
        try:
            from sne4onnx import extraction
        except Exception as ex:
            print(
                f'{Color.RED}ERROR:{Color.RESET} ' +\
                f'If --output_names_to_interrupt_model_conversion is specified, ' +\
                f'you must install sne4onnx. pip install sne4onnx'
            )
            sys.exit(1)
        # Cut ONNX graph at specified output position
        output_names = [
            output_op_name \
                for output_op_name in output_names_to_interrupt_model_conversion
        ]
        onnx_graph: onnx.ModelProto = extraction(
            input_op_names=[graph_input.name for graph_input in graph.inputs],
            output_op_names=output_names,
            onnx_graph=onnx_graph,
        )
        # Re-import of onnx_graph
        del graph
        graph = gs.import_onnx(onnx_graph)

    def sanitizing(node):
        if hasattr(node, 'name'):
            node.name = node.name.replace(':','_')
            if output_signaturedefs or output_integer_quantized_tflite:
                node.name = re.sub('^/', 'wa/', node.name)
        elif hasattr(node, '_name'):
            node._name = node._name.replace(':','_')
            if output_signaturedefs or output_integer_quantized_tflite:
                node._name = re.sub('^/', 'wa/', node._name)

    # sanitizing ':', '/'
    _ = [sanitizing(graph_input) for graph_input in graph.inputs]
    _ = [sanitizing(graph_node) for graph_node in graph.nodes]
    _ = [sanitizing(graph_output) for graph_output in graph.outputs]
    try:
        onnx_graph = gs.export_onnx(graph)
    except Exception as ex:
        # Workaround for SequenceConstruct terminating abnormally with onnx_graphsurgeon
        pass

    if not non_verbose:
        print('')
        print(f'{Color.REVERCE}Model loaded{Color.RESET}', '=' * 72)

    # Create Output folder
    os.makedirs(output_folder_path, exist_ok=True)

    if replace_to_pseudo_operators is None:
        replace_to_pseudo_operators = []

    # Define additional parameters
    additional_parameters = {
        'input_onnx_file_path': input_onnx_file_path if input_onnx_file_path is not None else None,
        'onnx_graph': onnx_graph,
        'opset': graph.opset,
        'batch_size': batch_size,
        'non_verbose': non_verbose,
        'disable_group_convolution': disable_group_convolution,
        'enable_rnn_unroll': enable_rnn_unroll,
        'disable_suppression_flextranspose': disable_suppression_flextranspose,
        'number_of_dimensions_after_flextranspose_compression': number_of_dimensions_after_flextranspose_compression,
        'disable_suppression_flexstridedslice': disable_suppression_flexstridedslice,
        'number_of_dimensions_after_flexstridedslice_compression': number_of_dimensions_after_flexstridedslice_compression,
        'optimization_for_gpu_delegate': optimization_for_gpu_delegate,
        'replace_argmax_to_reducemax_and_indicies_is_int64': replace_argmax_to_reducemax_and_indicies_is_int64,
        'replace_argmax_to_reducemax_and_indicies_is_float32': replace_argmax_to_reducemax_and_indicies_is_float32,
        'replace_argmax_to_fused_argmax_and_indicies_is_int64': replace_argmax_to_fused_argmax_and_indicies_is_int64,
        'replace_argmax_to_fused_argmax_and_indicies_is_float32': replace_argmax_to_fused_argmax_and_indicies_is_float32,
        'fused_argmax_scale_ratio': fused_argmax_scale_ratio,
        'replace_to_pseudo_operators': replace_to_pseudo_operators,
        'replacement_parameters': replacement_parameters,
        'mvn_epsilon': mvn_epsilon,
        'output_signaturedefs': output_signaturedefs,
        'output_nms_with_dynamic_tensor': output_nms_with_dynamic_tensor,
    }

    tf_layers_dict = {}

    if not non_verbose:
        print('')
        print(f'{Color.REVERCE}Model convertion started{Color.RESET}', '=' * 60)

    with graph.node_ids():

        onnx_graph_input_names: List[str] = [
            inputop.name for inputop in graph.inputs
        ]
        onnx_graph_output_names: List[str] = [
            outputop.name for outputop in graph.outputs
        ]

        # Inputs
        for graph_input in graph.inputs:
            """
            graph_input.shape: [1]
            graph_input.dtype: dtype('float32')
            graph_input.name: 'abs6_input'

            graph_input.shape: [1, 3, 192, 256]
            graph_input.dtype: dtype('float32')
            graph_input.name: 'input'

            graph_input.shape: [1, 3, 'height', 'width']
            graph_input.dtype: dtype('float32')
            graph_input.name: 'input'
            """
            # AUTO calib 4D check
            if output_integer_quantized_tflite \
                and custom_input_op_name_np_data_path is None \
                and (graph_input.dtype != np.float32 or len(graph_input.shape) != 4):
                print(
                    f'{Color.RED}ERROR:{Color.RESET} ' +
                    f'For INT8 quantization, the input data type must be Float32. ' +
                    f'Also, if --custom_input_op_name_np_data_path is not specified, ' +
                    f'all input OPs must assume 4D tensor image data. ' +
                    f'INPUT Name: {graph_input.name} INPUT Shape: {graph_input.shape} INPUT dtype: {graph_input.dtype}'
                )
                sys.exit(1)

            # make input
            op = importlib.import_module(f'onnx2tf.ops.Input')

            # substitution because saved_model does not allow colons
            # Substitution because saved_model does not allow leading slashes in op names
            sanitizing(graph_input)

            op.make_node(
                graph_input=graph_input,
                tf_layers_dict=tf_layers_dict,
                keep_ncw_or_nchw_or_ncdhw_input_names=keep_ncw_or_nchw_or_ncdhw_input_names,
                keep_nwc_or_nhwc_or_ndhwc_input_names=keep_nwc_or_nhwc_or_ndhwc_input_names,
                keep_shape_absolutely_input_names=keep_shape_absolutely_input_names,
                **additional_parameters,
            )

        # Get Inputs
        inputs = get_tf_model_inputs(
            tf_layers_dict=tf_layers_dict,
        )

        # download test data
        all_four_dim = sum(
            [
                1 for input in inputs \
                    if len(input.shape) == 4 \
                        and input.shape[0] is not None \
                        and input.shape[0] <= 20 \
                        and input.shape[-1] == 3 \
                        and input.shape[1] is not None \
                        and input.shape[2] is not None
            ]
        ) == len(inputs)
        same_batch_dim = False
        if all_four_dim:
            batch_size = inputs[0].shape[0]
            for input in inputs:
                same_batch_dim = batch_size == input.shape[0]
        test_data_nhwc = None
        if all_four_dim and same_batch_dim:
            test_data: np.ndarray = download_test_image_data()
            test_data_nhwc = test_data[:inputs[0].shape[0], ...]
            if check_onnx_tf_outputs_sample_data_normalization == "norm":
                pass
            elif check_onnx_tf_outputs_sample_data_normalization == "denorm":
                test_data_nhwc = test_data_nhwc * 255.0

        # ONNX dummy inference
        # Generate output for all OPs.
        # Used to verify the output error of each OP in the TensorFlow model.
        full_ops_output_names = []
        onnx_tensor_infos_for_validation = None
        for graph_node in graph.nodes:
            full_ops_output_names_sub = []
            for graph_node_output in graph_node.outputs:
                full_ops_output_names_sub.append(graph_node_output.name)
            full_ops_output_names.extend(full_ops_output_names_sub)
        # Models with errors during inference in onnxruntime skip dummy inference.
        try:
            onnx_outputs_for_validation: List[np.ndarray] = dummy_onnx_inference(
                onnx_graph=onnx_graph,
                output_names=full_ops_output_names,
                test_data_nhwc=test_data_nhwc,
                custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                tf_layers_dict=tf_layers_dict,
            )
            """
            onnx_tensor_infos_for_validation:
                {
                    onnx_output_name: np.ndarray,
                    onnx_output_name: np.ndarray,
                    onnx_output_name: np.ndarray,
                                :
                }
            """
            onnx_tensor_infos_for_validation = {
                ops_output_name: onnx_output_for_validation \
                    for ops_output_name, onnx_output_for_validation \
                        in zip(full_ops_output_names, onnx_outputs_for_validation)
            }
        except Exception as ex:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} ' +\
                f'The optimization process for shape estimation is skipped ' +
                f'because it contains OPs that cannot be inferred by the standard onnxruntime.'
            )
            print(f'{Color.YELLOW}WARNING:{Color.RESET} {ex}')
        additional_parameters['onnx_tensor_infos_for_validation'] = onnx_tensor_infos_for_validation
        additional_parameters['test_data_nhwc'] = test_data_nhwc
        additional_parameters['custom_input_op_name_np_data_path'] = custom_input_op_name_np_data_path

        # Nodes
        # https://github.com/onnx/onnx/blob/main/docs/Operators.md
        for graph_node in graph.nodes:
            optype = graph_node.op
            try:
                op = importlib.import_module(f'onnx2tf.ops.{optype}')
            except ModuleNotFoundError as ex:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} {optype} OP is not yet implemented.'
                )
                sys.exit(1)

            # substitution because saved_model does not allow colons
            # Substitution because saved_model does not allow leading slashes in op names
            sanitizing(graph_node)

            op.make_node(
                graph_node=graph_node,
                tf_layers_dict=tf_layers_dict,
                **additional_parameters,
            )

        del additional_parameters['onnx_tensor_infos_for_validation']
        del onnx_tensor_infos_for_validation

        # Get Outputs
        outputs = get_tf_model_outputs(
            tf_layers_dict=tf_layers_dict,
            output_names=output_names,
        )

        # Bring back output names from ONNX model
        for output, name in zip(outputs, output_names):
            output.node.layer._name = name.replace(':','_')
            if output_signaturedefs or output_integer_quantized_tflite:
                output.node.layer._name = re.sub('^/', '', output.node.layer._name)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        if not non_verbose:
            print('')
            model.summary(line_length=140)
            print('')

        # Output in Keras h5 format
        if output_h5:
            if not non_verbose:
                print(f'{Color.REVERCE}h5 output started{Color.RESET}', '=' * 67)
            model.save(f'{output_folder_path}/{output_file_name}_float32.h5')
            if not non_verbose:
                print(f'{Color.GREEN}h5 output complete!{Color.RESET}')

        # Output in Keras keras_v3 format
        if output_keras_v3:
            if not non_verbose:
                print(f'{Color.REVERCE}keras_v3 output started{Color.RESET}', '=' * 61)
            model.save(f'{output_folder_path}/{output_file_name}_float32.keras', save_format="keras_v3")
            if not non_verbose:
                print(f'{Color.GREEN}keras_v3 output complete!{Color.RESET}')

        # Create concrete func
        run_model = tf.function(lambda *inputs : model(inputs))
        concrete_func = run_model.get_concrete_function(
            *[tf.TensorSpec(tensor.shape, tensor.dtype) for tensor in model.inputs]
        )

        SIGNATURE_KEY = 'serving_default'

        # saved_model
        try:
            # concrete_func
            if not non_verbose:
                print(f'{Color.REVERCE}saved_model output started{Color.RESET}', '=' * 58)
            if not output_signaturedefs and not output_integer_quantized_tflite:
                tf.saved_model.save(concrete_func, output_folder_path)
            else:
                tf.saved_model.save(model, output_folder_path)
            if not non_verbose:
                print(f'{Color.GREEN}saved_model output complete!{Color.RESET}')
        except TypeError as e:
            # Switch to .pb
            if not non_verbose:
                print(f'{Color.GREEN}Switch to the output of an optimized protocol buffer file (.pb).{Color.RESET}')
        except KeyError as e:
            msg_list = [s for s in e.args if isinstance(s, str)]
            if len(msg_list) > 0:
                for s in msg_list:
                    if 'Failed to add concrete function' in s:
                        print(
                            f'{Color.YELLOW}WARNING:{Color.RESET} ' +\
                            f'This model contains GroupConvolution and is automatically optimized for TFLite, ' +
                            f'but is not output because saved_model does not support GroupConvolution. ' +
                            f'If saved_model is needed, specify --disable_group_convolution to retransform the model.'
                        )
                        break
            else:
                print(f'{Color.RED}ERROR:{Color.RESET}', e)
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f'{Color.RED}ERROR:{Color.RESET}', e)
            import traceback
            traceback.print_exc()

        # TFv1 .pb
        if output_tfv1_pb:
            try:
                from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
                imported = tf.saved_model.load(output_folder_path)
                f = imported.signatures[SIGNATURE_KEY]
                frozen_func = convert_variables_to_constants_v2(f)
                frozen_func.graph.as_graph_def()
                tf.io.write_graph(
                    graph_or_graph_def=frozen_func.graph,
                    logdir=output_folder_path,
                    name=f'{output_file_name}_float32.pb',
                    as_text=False,
                )
                if not non_verbose:
                    print(f'{Color.GREEN}TFv1 .pb output complete!{Color.RESET}')
            except Exception as e:
                print(f'{Color.RED}ERROR:{Color.RESET}', e)
                import traceback
                traceback.print_exc()

        # TFLite
        """
        TypeError: EndVector() missing 1 required positional argument: 'vectorNumElems'
        https://stackoverflow.com/questions/73442005/tflite-model-maker-in-colab-typeerror-endvector-missing-1-required-positional

        pip show flatbuffers
        Name: flatbuffers
        Version: 1.12

        pip install -U flatbuffers
        pip show flatbuffers
        Name: flatbuffers
        Version: 22.10.26
        """
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [concrete_func]
        )
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_disable_batchmatmul_unfold = not enable_batchmatmul_unfold
        tflite_model = converter.convert()
        with open(f'{output_folder_path}/{output_file_name}_float32.tflite', 'wb') as w:
            w.write(tflite_model)
        if copy_onnx_input_output_names_to_tflite:
            rewrite_tflite_inout_opname(
                output_folder_path=output_folder_path,
                tflite_file_name=f'{output_file_name}_float32.tflite',
                onnx_input_names=onnx_graph_input_names,
                onnx_output_names=onnx_graph_output_names,
            )
        if output_weights:
            weights_export(
                extract_target_tflite_file_path=f'{output_folder_path}/{output_file_name}_float32.tflite',
                output_weights_file_path=f'{output_folder_path}/{output_file_name}_float32_weights.h5',
            )
        if not non_verbose:
            print(f'{Color.GREEN}Float32 tflite output complete!{Color.RESET}')

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter.convert()
        with open(f'{output_folder_path}/{output_file_name}_float16.tflite', 'wb') as w:
            w.write(tflite_model)
        if copy_onnx_input_output_names_to_tflite:
            rewrite_tflite_inout_opname(
                output_folder_path=output_folder_path,
                tflite_file_name=f'{output_file_name}_float16.tflite',
                onnx_input_names=onnx_graph_input_names,
                onnx_output_names=onnx_graph_output_names,
            )
        if output_weights:
            weights_export(
                extract_target_tflite_file_path=f'{output_folder_path}/{output_file_name}_float16.tflite',
                output_weights_file_path=f'{output_folder_path}/{output_file_name}_float16_weights.h5',
            )
        if not non_verbose:
            print(f'{Color.GREEN}Float16 tflite output complete!{Color.RESET}')

        # Run TFLite ModelAnalyzer on the generated Float16 tflite model
        # to check if the model can be supported by GPU Delegate.
        if check_gpu_delegate_compatibility:
            print('')
            try:
                tf.lite.experimental.Analyzer.analyze(
                    model_content=tflite_model,
                    gpu_compatibility=True,
                )
            except Exception as ex:
                if not non_verbose:
                    import traceback
                    traceback.print_exc()
                    print(
                        f'{Color.YELLOW}WARNING:{Color.RESET} '+
                        'TFLite ModelAnalyzer failed.'
                    )

        # Quantized TFLite
        MEAN = np.asarray([[[[0.485, 0.456, 0.406]]]], dtype=np.float32)
        STD = np.asarray([[[[0.229, 0.224, 0.225]]]], dtype=np.float32)
        if output_integer_quantized_tflite:
            # Get signatures/input keys
            if not non_verbose:
                loaded_saved_model = tf.saved_model.load(
                    output_folder_path
                ).signatures[SIGNATURE_KEY]
                input_keys = list(loaded_saved_model.structured_input_signature[1].keys())
                input_shapes = [v.shape for v in loaded_saved_model.structured_input_signature[1].values()]
                input_dtypes = [v.dtype for v in loaded_saved_model.structured_input_signature[1].values()]
                print(f'{Color.BLUE}Input signature information for quantization{Color.RESET}')
                print(f'{Color.BLUE}signature_name{Color.RESET}: {SIGNATURE_KEY}')
                for idx, (input_key, input_shape, input_dtype) in enumerate(zip(input_keys, input_shapes, input_dtypes)):
                    print(
                        f'{Color.BLUE}input_name.{idx}{Color.RESET}: {input_key} '+
                        f'{Color.BLUE}shape{Color.RESET}: {input_shape} '+
                        f'{Color.BLUE}dtype{Color.RESET}: {input_dtype}'
                    )

            # INT8 Converter
            converter = tf.lite.TFLiteConverter.from_saved_model(
                output_folder_path,
            )
            # Dynamic Range Quantization
            try:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = []
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter._experimental_disable_per_channel = disable_per_channel
                converter._experimental_disable_batchmatmul_unfold = not enable_batchmatmul_unfold
                tflite_model = converter.convert()
                with open(f'{output_folder_path}/{output_file_name}_dynamic_range_quant.tflite', 'wb') as w:
                    w.write(tflite_model)
                if copy_onnx_input_output_names_to_tflite:
                    rewrite_tflite_inout_opname(
                        output_folder_path=output_folder_path,
                        tflite_file_name=f'{output_file_name}_dynamic_range_quant.tflite',
                        onnx_input_names=onnx_graph_input_names,
                        onnx_output_names=onnx_graph_output_names,
                    )
                if output_weights:
                    weights_export(
                        extract_target_tflite_file_path=f'{output_folder_path}/{output_file_name}_dynamic_range_quant.tflite',
                        output_weights_file_path=f'{output_folder_path}/{output_file_name}_dynamic_range_quant_weights.h5',
                    )
                if not non_verbose:
                    print(f'{Color.GREEN}Dynamic Range Quantization tflite output complete!{Color.RESET}')
            except RuntimeError as ex:
                if not non_verbose:
                    import traceback
                    traceback.print_exc()
                    print(
                        f'{Color.YELLOW}WARNING:{Color.RESET} '+
                        'Dynamic Range Quantization tflite output failed.'
                    )

            # Download sample calibration data - MS-COCO x20 images
            # Used only when there is only one input OP, a 4D tensor image,
            # and --quant_calib_input_op_name_np_data_path is not specified.
            # Otherwise, calibrate using the data specified in --quant_calib_input_op_name_np_data_path.
            calib_data_dict = {}
            model_input_name_list = [
                model_input.name for model_input in model.inputs
            ]
            data_count = 0
            if custom_input_op_name_np_data_path is None \
                and model.inputs[0].dtype == tf.float32 \
                and len(model.inputs[0].shape) == 4:

                # AUTO calib 4D images
                calib_data: np.ndarray = download_test_image_data()
                data_count = calib_data.shape[0]
                for model_input in model.inputs:
                    if model_input.dtype != tf.float32 \
                        or len(model_input.shape) != 4 \
                        or model_input.shape[-1] != 3:
                        print(
                            f'{Color.RED}ERROR:{Color.RESET} ' +
                            f'For models that have multiple input OPs and need to perform INT8 quantization calibration '+
                            f'using non-rgb-image input tensors, specify the calibration data with '+
                            f'--quant_calib_input_op_name_np_data_path. '+
                            f'model_input[n].shape: {model_input.shape}'
                        )
                        sys.exit(1)

                    calib_data_dict[model_input.name] = \
                        [
                            tf.image.resize(
                                calib_data.copy(),
                                (model_input.shape[1], model_input.shape[2])
                            ),
                            MEAN,
                            STD,
                        ]
            elif custom_input_op_name_np_data_path is not None:
                for param in custom_input_op_name_np_data_path:
                    if len(param) != 4:
                        print(
                            f"{Color.RED}ERROR:{Color.RESET} " +
                            "If you want to use custom input with the '-oiqt' option, " +
                            "{input_op_name}, {numpy_file_path}, {mean}, and {std} must all be entered. " +
                            f"However, you have only entered {len(param)} options. "
                        )
                        sys.exit(1)

                    input_op_name = str(param[0])
                    numpy_file_path = str(param[1])
                    calib_data = np.load(numpy_file_path)
                    if data_count == 0:
                        data_count = calib_data.shape[0]
                    mean = param[2]
                    std = param[3]

                    calib_data_dict[input_op_name] = \
                        [
                            calib_data.copy(),
                            mean,
                            std,
                        ]

            # representative_dataset_gen
            def representative_dataset_gen():
                for idx in range(data_count):
                    yield_data_dict = {}
                    for model_input_name in model_input_name_list:
                        calib_data, mean, std = calib_data_dict[model_input_name]
                        normalized_calib_data = (calib_data[idx] - mean) / std
                        yield_data_dict[model_input_name] = normalized_calib_data
                    yield yield_data_dict

            # INT8 Quantization
            try:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter._experimental_disable_per_channel = disable_per_channel
                converter._experimental_disable_batchmatmul_unfold = not enable_batchmatmul_unfold
                converter.representative_dataset = representative_dataset_gen
                tflite_model = converter.convert()
                with open(f'{output_folder_path}/{output_file_name}_integer_quant.tflite', 'wb') as w:
                    w.write(tflite_model)
                if copy_onnx_input_output_names_to_tflite:
                    rewrite_tflite_inout_opname(
                        output_folder_path=output_folder_path,
                        tflite_file_name=f'{output_file_name}_integer_quant.tflite',
                        onnx_input_names=onnx_graph_input_names,
                        onnx_output_names=onnx_graph_output_names,
                    )
                if output_weights:
                    weights_export(
                        extract_target_tflite_file_path=f'{output_folder_path}/{output_file_name}_integer_quant.tflite',
                        output_weights_file_path=f'{output_folder_path}/{output_file_name}_integer_quant_weights.h5',
                    )
                if not non_verbose:
                    print(f'{Color.GREEN}INT8 Quantization tflite output complete!{Color.RESET}')

                # Full Integer Quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter._experimental_disable_per_channel = disable_per_channel
                converter._experimental_disable_batchmatmul_unfold = not enable_batchmatmul_unfold
                converter.representative_dataset = representative_dataset_gen
                inf_type = None
                if input_output_quant_dtype == 'int8':
                    inf_type = tf.int8
                elif input_output_quant_dtype == 'uint8':
                    inf_type = tf.uint8
                else:
                    inf_type = tf.int8
                converter.inference_input_type = inf_type
                converter.inference_output_type = inf_type
                tflite_model = converter.convert()
                with open(f'{output_folder_path}/{output_file_name}_full_integer_quant.tflite', 'wb') as w:
                    w.write(tflite_model)
                if copy_onnx_input_output_names_to_tflite:
                    rewrite_tflite_inout_opname(
                        output_folder_path=output_folder_path,
                        tflite_file_name=f'{output_file_name}_full_integer_quant.tflite',
                        onnx_input_names=onnx_graph_input_names,
                        onnx_output_names=onnx_graph_output_names,
                    )
                if output_weights:
                    weights_export(
                        extract_target_tflite_file_path=f'{output_folder_path}/{output_file_name}_full_integer_quant.tflite',
                        output_weights_file_path=f'{output_folder_path}/{output_file_name}_full_integer_quant_weights.h5',
                    )
                if not non_verbose:
                    print(f'{Color.GREEN}Full INT8 Quantization tflite output complete!{Color.RESET}')
            except RuntimeError as ex:
                if not non_verbose:
                    import traceback
                    traceback.print_exc()
                    print(
                        f'{Color.YELLOW}WARNING:{Color.RESET} '+
                        'Full INT8 Quantization tflite output failed.'
                    )

            # Integer quantization with int16 activations
            try:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = []
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter._experimental_disable_per_channel = disable_per_channel
                converter._experimental_disable_batchmatmul_unfold = not enable_batchmatmul_unfold
                converter.representative_dataset = representative_dataset_gen
                converter.inference_input_type = tf.float32
                converter.inference_output_type = tf.float32
                tflite_model = converter.convert()
                with open(f'{output_folder_path}/{output_file_name}_integer_quant_with_int16_act.tflite', 'wb') as w:
                    w.write(tflite_model)
                if copy_onnx_input_output_names_to_tflite:
                    rewrite_tflite_inout_opname(
                        output_folder_path=output_folder_path,
                        tflite_file_name=f'{output_file_name}_integer_quant_with_int16_act.tflite',
                        onnx_input_names=onnx_graph_input_names,
                        onnx_output_names=onnx_graph_output_names,
                    )
                if not non_verbose:
                    print(f'{Color.GREEN}INT8 Quantization with int16 activations tflite output complete!{Color.RESET}')
            except RuntimeError as ex:
                if not non_verbose:
                    import traceback
                    traceback.print_exc()
                    print(
                        f'{Color.YELLOW}WARNING:{Color.RESET} '+
                        'INT8 Quantization with int16 activations tflite output failed.'
                    )

            # Full Integer quantization with int16 activations
            try:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = []
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
                converter._experimental_disable_per_channel = disable_per_channel
                converter._experimental_disable_batchmatmul_unfold = not enable_batchmatmul_unfold
                converter.representative_dataset = representative_dataset_gen
                converter.inference_input_type = tf.int16
                converter.inference_output_type = tf.int16
                tflite_model = converter.convert()
                with open(f'{output_folder_path}/{output_file_name}_full_integer_quant_with_int16_act.tflite', 'wb') as w:
                    w.write(tflite_model)
                if copy_onnx_input_output_names_to_tflite:
                    rewrite_tflite_inout_opname(
                        output_folder_path=output_folder_path,
                        tflite_file_name=f'{output_file_name}_full_integer_quant_with_int16_act.tflite',
                        onnx_input_names=onnx_graph_input_names,
                        onnx_output_names=onnx_graph_output_names,
                    )
                if not non_verbose:
                    print(f'{Color.GREEN}Full INT8 Quantization with int16 activations tflite output complete!{Color.RESET}')
            except RuntimeError as ex:
                if not non_verbose:
                    import traceback
                    traceback.print_exc()
                    print(
                        f'{Color.YELLOW}WARNING:{Color.RESET} '+
                        'Full INT8 Quantization with int16 activations tflite output failed.'
                    )

        # Returns true if the two arrays, the output of onnx and the output of TF,
        # are elementwise equal within an acceptable range.
        # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html#numpy-allclose
        # numpy.allclose(a, b, rtol=1e-05, atol=1e-05, equal_nan=True)
        if check_onnx_tf_outputs_elementwise_close or check_onnx_tf_outputs_elementwise_close_full:
            try:
                import onnxruntime
                import sne4onnx
            except Exception as ex:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} ' +\
                    f'If --check_onnx_tf_outputs_elementwise_close is specified, ' +\
                    f'you must install onnxruntime and sne4onnx. pip install sne4onnx onnxruntime'
                )
                sys.exit(1)

            if not non_verbose:
                print('')
                print(f'{Color.REVERCE}ONNX and TF output value validation started{Color.RESET}', '=' * 41)
                print(
                    f'{Color.GREEN}INFO:{Color.RESET} {Color.GREEN}validation_conditions{Color.RESET}: '+
                    f'np.allclose(onnx_outputs, tf_outputs, '+
                    f'rtol={check_onnx_tf_outputs_elementwise_close_rtol}, '+
                    f'atol={check_onnx_tf_outputs_elementwise_close_atol}, '+
                    f'equal_nan=True)')

            # Listing of output_names
            # --check_onnx_tf_outputs_elementwise_close_full lists all output names
            # in the ONNX graph when enabled.
            # ops_output_names is a list of output_names.
            #
            # output_names: List[str]
            #
            # ops_output_names = [
            #   output_names,
            #   output_names,
            #   output_names,
            #         :
            # ]

            # Output OP extended to all ONNX nodes
            ops_output_names = []
            if check_onnx_tf_outputs_elementwise_close_full:
                ops_output_names = full_ops_output_names
            else:
                ops_output_names = output_names

            # Rebuild model for validation
            del model
            outputs = [
                layer_info['tf_node'] \
                    for opname, layer_info in tf_layers_dict.items() \
                        if opname in ops_output_names \
                            and not hasattr(layer_info['tf_node'], 'numpy')
            ]
            exclude_output_names = [
                opname \
                    for opname, layer_info in tf_layers_dict.items() \
                        if opname in ops_output_names \
                            and hasattr(layer_info['tf_node'], 'numpy')
            ]
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Exclude output OPs not subject to validation
            ops_output_names = [
                ops_output_name for ops_output_name in ops_output_names \
                    if ops_output_name not in exclude_output_names
            ]

            dummy_onnx_outputs = None
            try:
                # ONNX dummy inference
                dummy_onnx_outputs: List[np.ndarray] = dummy_onnx_inference(
                    onnx_graph=onnx_graph,
                    output_names=ops_output_names,
                    test_data_nhwc=test_data_nhwc,
                    custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                    tf_layers_dict=tf_layers_dict,
                )
            except Exception as ex:
                print(
                    f'{Color.YELLOW}WARNING:{Color.RESET} ' +\
                    f'The accuracy error measurement process was skipped ' +
                    f'because the standard onnxruntime contains OPs that cannot be inferred.'
                )
                print(f'{Color.YELLOW}WARNING:{Color.RESET} {ex}')
            else:
                # TF dummy inference
                tf_tensor_infos: Dict[Any] = dummy_tf_inference(
                    model=model,
                    inputs=inputs,
                    test_data_nhwc=test_data_nhwc,
                    custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                )
                # Validation
                onnx_tensor_infos = {
                    output_name: dummy_onnx_output \
                        for output_name, dummy_onnx_output in zip(ops_output_names, dummy_onnx_outputs)
                }
                """
                np.allclose(
                    dummy_onnx_outputs,
                    dummy_tf_outputs,
                    rtol=0.0,
                    atol=1e-04,
                    equal_nan=True,
                )

                check_results: Dict[str, List[np.ndarray, int, float|int]]
                    {
                        onnx_output_name: [
                            onnx_tensor,
                            matched_flg, <--- 0: Unmatched, 1: Matched, 2: Skipped (Deleted or Shape Unmatched)
                            max_abs_err,
                        ]
                    }
                """
                input_names = [k.name for k in inputs]
                for k, v in tf_layers_dict.items():
                    if 'tf_node_info' in v:
                        if v['tf_node_info']['tf_op_type'] == 'identity':
                            tf_tensor_infos[v['tf_node'].name] = np.ndarray([0], dtype=np.int64)
                onnx_tf_output_pairs = {
                    (k, v['tf_node'].name): (onnx_tensor_infos[k], tf_tensor_infos[v['tf_node'].name])
                        for k, v in tf_layers_dict.items() if k not in input_names and not hasattr(v['tf_node'], 'numpy')
                }

                check_results = onnx_tf_tensor_validation(
                    output_pairs=onnx_tf_output_pairs,
                    rtol=check_onnx_tf_outputs_elementwise_close_rtol,
                    atol=check_onnx_tf_outputs_elementwise_close_atol,
                )
                for (onnx_output_name, tf_output_name), checked_value in check_results.items():
                    validated_onnx_tensor: np.ndarray = checked_value[0]
                    matched_flg: int = checked_value[1]
                    max_abs_err: Any = checked_value[2]
                    message = ''
                    if matched_flg == 0:
                        message = \
                            f'{Color.GREEN}validate_result{Color.RESET}: ' +\
                            f'{Color.REVERCE}{Color.YELLOW} Unmatched {Color.RESET} ' +\
                            f'{Color.GREEN}max_abs_error{Color.RESET}: {max_abs_err}'
                    elif matched_flg == 1:
                        message = \
                            f'{Color.GREEN}validate_result{Color.RESET}: ' +\
                            f'{Color.REVERCE}{Color.GREEN} Matches {Color.RESET}'
                    elif matched_flg == 2:
                        message = \
                            f'{Color.GREEN}validate_result{Color.RESET}: ' +\
                            f'{Color.REVERCE}{Color.BLUE} Skipped (Deleted or Shape Unmatched) {Color.RESET}'
                    print(
                        f'{Color.GREEN}INFO:{Color.RESET} '+
                        f'{Color.GREEN}onnx_output_name{Color.RESET}: {onnx_output_name} '+
                        f'{Color.GREEN}tf_output_name{Color.RESET}: {tf_output_name} '+
                        f'{Color.GREEN}shape{Color.RESET}: {validated_onnx_tensor.shape} '+
                        f'{Color.GREEN}dtype{Color.RESET}: {validated_onnx_tensor.dtype} '+
                        f'{message}'
                    )

        return model


def main():
    parser = ArgumentParser()
    iV_group = parser.add_mutually_exclusive_group(required=True)
    iV_group.add_argument(
        '-i',
        '--input_onnx_file_path',
        type=str,
        help='Input onnx file path.'
    )
    iV_group.add_argument(
        '-V',
        '--version',
        action='store_true',
        help='Show version and exit.'
    )
    parser.add_argument(
        '-o',
        '--output_folder_path',
        type=str,
        help=\
            'Output folder path. \n' +
            'Default: "saved_model"'
    )
    parser.add_argument(
        '-osd',
        '--output_signaturedefs',
        action='store_true',
        help=\
            'Signature is added to the output for serving or for conversion \n' +
            'to other model formats. However, this can significantly reduce the speed \n' +
            'of model conversion and significant increase the size of the model.'
    )
    parser.add_argument(
        '-oh5',
        '--output_h5',
        action='store_true',
        help=\
            'Output model in Keras (hdf5) format.'
    )
    parser.add_argument(
        '-okv3',
        '--output_keras_v3',
        action='store_true',
        help=\
            'Output model in Keras (keras_v3) format.'
    )
    parser.add_argument(
        '-otfv1pb',
        '--output_tfv1_pb',
        action='store_true',
        help=\
            'Output model in TF v1 (.pb) format.'
    )
    parser.add_argument(
        '-ow',
        '--output_weights',
        action='store_true',
        help=\
            'Output weights in hdf5 format.'
    )
    parser.add_argument(
        '-coion',
        '--copy_onnx_input_output_names_to_tflite',
        action='store_true',
        help=\
            'Copy the input/output OP name of ONNX to the input/output OP name of tflite. \n' +
            'Due to Tensorflow internal operating specifications, \n' +
            'the input/output order of ONNX does not necessarily match \n' +
            'the input/output order of tflite. \n' +
            'Be sure to check that the input/output OP names in the generated \n' +
            'tflite file have been converted as expected. \n' +
            'Also, this option generates a huge JSON file as a temporary file for processing. \n' +
            'Therefore, it is strongly discouraged to use it on large models of hundreds \n'
            'of megabytes or more.'
    )
    parser.add_argument(
        '-oiqt',
        '--output_integer_quantized_tflite',
        action='store_true',
        help=\
            'Output of integer quantized tflite.'
    )
    parser.add_argument(
        '-qt',
        '--quant_type',
        type=str,
        choices=['per-channel', 'per-tensor'],
        default='per-channel',
        help=\
            'Selects whether "per-channel" or "per-tensor" quantization is used. \n' +
            'Default: "per-channel"'
    )
    parser.add_argument(
        '-cind',
        '--custom_input_op_name_np_data_path',
        type=str,
        action='append',
        nargs='+',
        help=\
            'Input name of OP and path of data file (Numpy) for custom input for -cotof or -oiqt, \n' +
            'and mean (optional) and std (optional). \n' +

            '\n<Usage in -cotof> \n' +
            'When using -cotof, custom input defined by the user, instead of dummy data, is used. \n' +
            'In this case, mean and std are omitted from the input. \n' +
            '-cind {input_op_name} {numpy_file_path} \n' +
            'ex) -cind onnx::Equal_0 test_cind/x_1.npy -cind onnx::Add_1 test_cind/x_2.npy -cotof \n' +
            'The input_op_name must be the same as in ONNX, \n' +
            'and it may not work if the input format is different between ONNX and TF. \n' +

            '\n<Usage in -oiqt> \n' +
            'INPUT Name of OP and path of calibration data file (Numpy) for quantization \n' +
            'and mean and std. \n' +
            'The specification can be omitted only when the input OP is a single 4D tensor image data. \n' +
            'If omitted, it is automatically calibrated using 20 normalized MS-COCO images. \n' +
            'The type of the input OP must be Float32. \n' +
            'Data for calibration must be pre-normalized to a range of 0 to 1. \n' +
            '-cind {input_op_name} {numpy_file_path} {mean} {std} \n' +
            'Numpy file paths must be specified the same number of times as the number of input OPs. \n' +
            'Normalize the value of the input OP based on the tensor specified in mean and std. \n' +
            '(input_value - mean) / std \n' +
            'Tensors in Numpy file format must be in dimension order after conversion to TF. \n' +
            'Note that this is intended for deployment on low-resource devices, \n' +
            'so the batch size is limited to 1 only. \n\n' +
            'e.g. \n' +
            'The example below shows a case where there are three input OPs. \n' +
            'Assume input0 is 128x128 RGB image data. \n' +
            'In addition, input0 should be a value that has been divided by 255 \n' +
            'in the preprocessing and normalized to a range between 0 and 1. \n' +
            'input1 and input2 assume the input of something that is not an image. \n' +
            'Because input1 and input2 assume something that is not an image, \n' +
            'the divisor is not 255 when normalizing from 0 to 1. \n' +
            '"n" is the number of calibration data. \n\n' +
            'ONNX INPUT shapes: \n' +
            '   input0: [n,3,128,128] \n' +
            '       mean: [1,3,1,1] -> [[[[0.485]],[[0.456]],[[0.406]]]] \n' +
            '       std:  [1,3,1,1] -> [[[[0.229]],[[0.224]],[[0.225]]]] \n' +
            '   input1: [n,64,64] \n' +
            '       mean: [1,64] -> [0.1, ..., 0.64] \n' +
            '       std:  [1,64] -> [0.05, ..., 0.08] \n' +
            '   input2: [n,5] \n' +
            '       mean: [1] -> [0.3] \n' +
            '       std:  [1] -> [0.07] \n' +
            'TensorFlow INPUT shapes (Numpy file ndarray shapes): \n' +
            '   input0: [n,128,128,3] \n' +
            '       mean: [1,1,1,3] -> [[[[0.485, 0.456, 0.406]]]] \n' +
            '       std:  [1,1,1,3] -> [[[[0.229, 0.224, 0.225]]]] \n' +
            '   input1: [n,64,64] \n' +
            '       mean: [1,64] -> [0.1, ..., 0.64] \n' +
            '       std:  [1,64] -> [0.05, ..., 0.08] \n' +
            '   input2: [n,5] \n' +
            '       mean: [1] -> [0.3] \n' +
            '       std:  [1] -> [0.07] \n' +
            '-cind "input0" "../input0.npy" [[[[0.485, 0.456, 0.406]]]] [[[[0.229, 0.224, 0.225]]]] \n' +
            '-cind "input1" "./input1.npy" [0.1, ..., 0.64] [0.05, ..., 0.08] \n' +
            '-cind "input2" "input2.npy" [0.3] [0.07] \n\n' +
            '\n<Using -cotof and -oiqt at the same time> \n' +
            'To use -cotof and -oiqt simultaneously, \n' +
            'you need to enter the Input name of OP, path of data file, mean, and std all together. \n' +
            'And the data file must be in Float32 format, \n' +
            'and {input_op_name}, {numpy_file_path}, {mean}, and {std} must all be entered. \n' +
            'Otherwise, an error will occur during the -oiqt stage.'
    )
    parser.add_argument(
        '-ioqd',
        '--input_output_quant_dtype',
        type=str,
        choices=['int8', 'uint8'],
        default='int8',
        help=\
            'Input and Output dtypes when doing Full INT8 Quantization. \n' +
            '"int8"(default) or "uint8"'
    )
    parser.add_argument(
        '-nuo',
        '--not_use_onnxsim',
        action='store_true',
        help=\
            'No optimization by onnx-simplifier is performed. \n' +
            'If this option is used, the probability of a conversion error is very high.'
    )
    parser.add_argument(
        '-nuonag',
        '--not_use_opname_auto_generate',
        action='store_true',
        help=\
            'Automatic generation of each OP name in the old format ONNX file '+
            'and assignment of OP name are not performed.'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help=\
            'Fixes the dynamic batch size to the specified numeric batch size. \n' +
            'A value of 1 or more must be specified.'
    )
    parser.add_argument(
        '-ois',
        '--overwrite_input_shape',
        type=str,
        nargs='+',
        help=\
            'Overwrite the input shape. \n' +
            'The format is\n' +
            '"input_name_1:dim0,...,dimN" "input_name_2:dim0,...,dimN" "input_name_3:dim0,...,dimN". \n' +
            'When there is only one input, for example, \n' +
            '"data:1,3,224,224" \n' +
            'When there are multiple inputs, for example, \n' +
            '"data1:1,3,224,224" "data2:1,3,112,112" "data3:5" \n' +
            'A value of 1 or more must be specified. \n' +
            'Numerical values other than dynamic dimensions are ignored. \n' +
            'Ignores --batch_size if specified at the same time as --batch_size.'
    )
    parser.add_argument(
        '-nlt',
        '--no_large_tensor',
        action='store_true',
        help=\
            'Suppresses constant bloat caused by Tile OP when optimizing models in onnxsim. \n' +
            'See: https://github.com/daquexian/onnx-simplifier/issues/178'
    )
    parser.add_argument(
        '-onwdt',
        '--output_nms_with_dynamic_tensor',
        action='store_true',
        help=\
            'The number of bounding boxes in the NMS output results is \n' +
            'not fixed at the maximum number of max_output_boxes_per_class, \n' +
            'but rather at the smallest possible number of dynamic tensors. \n' +
            'If this option is disabled, NMS output is padded to the number \n' +
            'set in the max_output_boxes_per_class attribute. \n' +
            'e.g. \n' +
            'disable --output_nms_with_dynamic_tensor: \n' +
            '    output_tensor_shape: [100, 7] \n' +
            'enable --output_nms_with_dynamic_tensor: \n' +
            '    output_tensor_shape: [N, 7]'
    )
    parser.add_argument(
        '-k',
        '--keep_ncw_or_nchw_or_ncdhw_input_names',
        type=str,
        nargs='+',
        help=\
            'Holds the NCW or NCHW or NCDHW of the input shape for the specified INPUT OP names. \n' +
            'If a nonexistent INPUT OP name is specified, it is ignored. \n' +
            'Valid only for 3D, 4D and 5D input tensors. \n\n' +
            'e.g. \n' +
            '--keep_ncw_or_nchw_or_ncdhw_input_names "input0" "input1" "input2"'
    )
    parser.add_argument(
        '-kt',
        '--keep_nwc_or_nhwc_or_ndhwc_input_names',
        type=str,
        nargs='+',
        help=\
            'Holds the NWC or NHWC or NDHWC of the input shape for the specified INPUT OP names. \n' +
            'If a nonexistent INPUT OP name is specified, it is ignored. \n' +
            'If the input OP name is the same as the input OP name specified \n' +
            'in the keep_ncw_or_nchw_or_ncdhw_input_names option, it is ignored. \n' +
            'Valid only for 3D, 4D and 5D input tensors. \n\n' +
            'e.g. \n' +
            '--keep_nwc_or_nhwc_or_ndhwc_input_names "input0" "input1" "input2"'
    )
    parser.add_argument(
        '-kat',
        '--keep_shape_absolutely_input_names',
        type=str,
        nargs='+',
        help=\
            'Name of the INPUT that unconditionally maintains its shape. \n' +
            'If a nonexistent INPUT OP name is specified, it is ignored. \n\n' +
            'e.g. \n' +
            '--keep_shape_absolutely_input_names "input0" "input1" "input2"'
    )
    parser.add_argument(
        '-onimc',
        '--output_names_to_interrupt_model_conversion',
        type=str,
        nargs='+',
        help=\
            'Output names that interrupt model conversion. \n' +
            'Interrupts model transformation at the specified output name \n' +
            'and outputs the model partitioned into subgraphs. \n\n' +
            'e.g. \n' +
            '--output_names_to_interrupt_model_conversion "output0" "output1" "output2"'
    )
    parser.add_argument(
        '-dgc',
        '--disable_group_convolution',
        action='store_true',
        help=\
            'Disable GroupConvolution and replace it with SeparableConvolution \n' +
            'for output to saved_model format.'
    )
    parser.add_argument(
        '-ebu',
        '--enable_batchmatmul_unfold',
        action='store_true',
        help=\
            'BatchMatMul is separated batch by batch to generate a primitive MatMul.'
    )
    parser.add_argument(
        '-eru',
        '--enable_rnn_unroll',
        action='store_true',
        help=\
            'Instead of increasing inference speed by expanding all symbolic loops of the RNN (LSTM, GRU, RNN), \n' +
            'RAM consumption will increase because all tensors are expanded and embedded in the model. \n' +
            'https://keras.io/api/layers/recurrent_layers/'
    )
    parser.add_argument(
        '-dsft',
        '--disable_suppression_flextranspose',
        action='store_true',
        help=\
            'Disables FlexTranspose generation suppression.'
    )
    parser.add_argument(
        '-nodaftc',
        '--number_of_dimensions_after_flextranspose_compression',
        type=int,
        default=6,
        help=\
            'Number of Transpose OP dimensions generated after avoiding FlexTranspose generation. \n' +
            'Also suppress the creation of the Transpose itself by specifying 2. \n' +
            'Default: 6'
    )
    parser.add_argument(
        '-dsfs',
        '--disable_suppression_flexstridedslice',
        action='store_true',
        help=\
            'Disables FlexStridedSlice generation suppression.'
    )
    parser.add_argument(
        '-nodafsc',
        '--number_of_dimensions_after_flexstridedslice_compression',
        type=int,
        default=5,
        help=\
            'Number of StricedSlice OP dimensions generated after avoiding FlexStridedSlice generation. \n' +
            'Default: 5'
    )
    parser.add_argument(
        '-ofgd',
        '--optimization_for_gpu_delegate',
        action='store_true',
        help=\
            'Replace operations that do not support gpu delegate with those \n' +
            'that do as much as possible.'
    )
    rar_group = parser.add_mutually_exclusive_group()
    rar_group.add_argument(
        '-rari64',
        '--replace_argmax_to_reducemax_and_indicies_is_int64',
        action='store_true',
        help=\
            'Replace ArgMax with a ReduceMax. The returned indicies are int64. \n' +
            'Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and \n' +
            'replace_argmax_to_reducemax_and_indicies_is_float32 and \n'+
            'replace_argmax_to_fused_argmax_and_indicies_is_int64 and \n'+
            'replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.'
    )
    rar_group.add_argument(
        '-rarf32',
        '--replace_argmax_to_reducemax_and_indicies_is_float32',
        action='store_true',
        help=\
            'Replace ArgMax with a ReduceMax. The returned indicies are float32. \n' +
            'Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and \n' +
            'replace_argmax_to_reducemax_and_indicies_is_float32 and \n'+
            'replace_argmax_to_fused_argmax_and_indicies_is_int64 and \n'+
            'replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.'
    )
    rar_group.add_argument(
        '-rafi64',
        '--replace_argmax_to_fused_argmax_and_indicies_is_int64',
        action='store_true',
        help=\
            'Replace ArgMax with a Fused_ArgMax. The returned indicies are int64. \n' +
            'It improves inference speed at the cost of a small sacrifice in accuracy. \n' +
            'See. https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/vision#argmax-fusion-to-improve-segmentation-model-latency \n' +
            'Currently, only 4D tensors are supported. \n' +
            'Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and \n' +
            'replace_argmax_to_reducemax_and_indicies_is_float32 and \n'+
            'replace_argmax_to_fused_argmax_and_indicies_is_int64 and \n'+
            'replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.'
    )
    rar_group.add_argument(
        '-raff32',
        '--replace_argmax_to_fused_argmax_and_indicies_is_float32',
        action='store_true',
        help=\
            'Replace ArgMax with a Fused_ArgMax. The returned indicies are float32. \n' +
            'It improves inference speed at the cost of a small sacrifice in accuracy. \n' +
            'See. https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/vision#argmax-fusion-to-improve-segmentation-model-latency \n' +
            'Currently, only 4D tensors are supported. \n' +
            'Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and \n' +
            'replace_argmax_to_reducemax_and_indicies_is_float32 and \n'+
            'replace_argmax_to_fused_argmax_and_indicies_is_int64 and \n'+
            'replace_argmax_to_fused_argmax_and_indicies_is_float32 can be specified.'
    )
    parser.add_argument(
        '-fasr',
        '--fused_argmax_scale_ratio',
        type=float,
        default=0.5,
        help=\
            'For Fused ArgMax. \n' +
            'Scale ratio when generating Fused ArgMax. \n' +
            '0.0 < fused_argmax_scale_ratio <= 1.0 \n' +
            'Default: 0.5'
    )
    parser.add_argument(
        '-rtpo',
        '--replace_to_pseudo_operators',
        nargs='*',
        default=[],
        help=\
            'Replace list of operators to pseudo operators. \n ' +
            'Full name of the target operators should be given. \n ' +
            'Currently supported operators : \n' +
            'Asin, Acos, Atan, Abs, PReLU, LeakyReLU, Power, GatherND, Neg, HardSwish, Erf'
    )
    parser.add_argument(
        '-me',
        '--mvn_epsilon',
        type=float,
        default=0.0000000001,
        help=\
            'For MeanVarianceNormalization. \n' +
            'The number to be added to the variance to avoid division by zero when normalizing the value. \n' +
            '(input_tensor - mean) / tf.sqrt(variance + mvn_epsilon) \n' +
            'Default: 0.0000000001'
    )
    parser.add_argument(
        '-prf',
        '--param_replacement_file',
        type=str,
        default='',
        help='Parameter replacement file path. (.json)'
    )
    parser.add_argument(
        '-cgdc',
        '--check_gpu_delegate_compatibility',
        action='store_true',
        help=\
            'Run TFLite ModelAnalyzer on the generated Float16 tflite model ' +
            'to check if the model can be supported by GPU Delegate.'
    )
    coto_group = parser.add_mutually_exclusive_group()
    coto_group.add_argument(
        '-coto',
        '--check_onnx_tf_outputs_elementwise_close',
        action='store_true',
        help=\
            'Returns "Matches" if the output of onnx and the output of TF are'+
            'within acceptable proximity element by element. '+
            'Returns "Unmatched" if the output of onnx and the output of TF are '+
            'not within acceptable proximity element by element. '+
            'If the output of onnx is 1D, it returns "Skipped" and skips the comparison '+
            'between the output of onnx and that of TF. This is because when undefined '+
            'dimensions are present, a situation often arises where very large index '+
            'values are compared, causing OutOfMemory. '+
            'Only the output content of the models final output OP is checked.'
    )
    coto_group.add_argument(
        '-cotof',
        '--check_onnx_tf_outputs_elementwise_close_full',
        action='store_true',
        help=\
            'Returns "Matches" if the output of onnx and the output of TF are '+
            'within acceptable proximity element by element. '+
            'Check the output of all OPs in sequence from the beginning, '+
            'including all but the final output OP of the model. '+
            'Returns "Unmatched" if the output of onnx and the output of TF are '+
            'not within acceptable proximity element by element. '+
            'If the output of onnx is 1D, it returns "Skipped" and skips the comparison '+
            'between the output of onnx and that of TF. This is because when undefined '+
            'dimensions are present, a situation often arises where very large index '+
            'values are compared, causing OutOfMemory. ' +
            'It is very time consuming because it performs as many inferences as '+
            'there are operations.'
    )
    parser.add_argument(
        '-coton',
        '--check_onnx_tf_outputs_sample_data_normalization',
        type=str,
        choices=['norm', 'denorm'],
        default='norm',
        help=\
            'norm: Validate using random data normalized to the range 0.0 to 1.0 ' +
            'denorm: Validate using random data in the range 0.0 to 255.0 ' +
            'If there is a normalization layer at the models entry point, ' +
            'or if the model was trained on denormalized data, "denorm" must be specified. ' +
            'Default: "norm"'
    )
    parser.add_argument(
        '-cotor',
        '--check_onnx_tf_outputs_elementwise_close_rtol',
        type=float,
        default=0.0,
        help=\
            'The relative tolerance parameter \n' +
            'Default: 0.0'
    )
    parser.add_argument(
        '-cotoa',
        '--check_onnx_tf_outputs_elementwise_close_atol',
        type=float,
        default=1e-4,
        help=\
            'The absolute tolerance parameter \n' +
            'Default: 1e-4'
    )
    parser.add_argument(
        '-n',
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    # Print version
    if args.version:
        print(__version__)
        sys.exit(0)

    # convert quant_calib_input_op_name_np_data_path
    # [
    #   [{input_op_name} {numpy_file_path} {mean} {std}],
    #   [{input_op_name} {numpy_file_path} {mean} {std}],
    #   [{input_op_name} {numpy_file_path} {mean} {std}],
    # ]
    custom_params = []
    if args.custom_input_op_name_np_data_path is not None:
        for param in args.custom_input_op_name_np_data_path:
            tmp = []
            if len(param) == 2:
                tmp.append(str(param[0])) # input_op_name
                tmp.append(str(param[1])) # numpy_file_path

            if len(param) == 4:
                tmp.append(str(param[0])) # input_op_name
                tmp.append(str(param[1])) # numpy_file_path
                tmp.append(np.asarray(ast.literal_eval(param[2]), dtype=np.float32)) # mean
                tmp.append(np.asarray(ast.literal_eval(param[3]), dtype=np.float32)) # std

            custom_params.append(
                tmp
            )

    if len(custom_params) == 0:
        custom_params = None

    args.replace_to_pseudo_operators = [
        name.lower() for name in args.replace_to_pseudo_operators
    ]

    # Convert
    model = convert(
        input_onnx_file_path=args.input_onnx_file_path,
        output_folder_path=args.output_folder_path,
        output_signaturedefs=args.output_signaturedefs,
        output_h5=args.output_h5,
        output_keras_v3=args.output_keras_v3,
        output_tfv1_pb=args.output_tfv1_pb,
        output_weights=args.output_weights,
        copy_onnx_input_output_names_to_tflite=args.copy_onnx_input_output_names_to_tflite,
        output_integer_quantized_tflite=args.output_integer_quantized_tflite,
        quant_type=args.quant_type,
        custom_input_op_name_np_data_path=custom_params,
        input_output_quant_dtype=args.input_output_quant_dtype,
        not_use_onnxsim=args.not_use_onnxsim,
        not_use_opname_auto_generate=args.not_use_opname_auto_generate,
        batch_size=args.batch_size,
        overwrite_input_shape=args.overwrite_input_shape,
        no_large_tensor=args.no_large_tensor,
        output_nms_with_dynamic_tensor=args.output_nms_with_dynamic_tensor,
        keep_ncw_or_nchw_or_ncdhw_input_names=args.keep_ncw_or_nchw_or_ncdhw_input_names,
        keep_nwc_or_nhwc_or_ndhwc_input_names=args.keep_nwc_or_nhwc_or_ndhwc_input_names,
        keep_shape_absolutely_input_names=args.keep_shape_absolutely_input_names,
        output_names_to_interrupt_model_conversion=args.output_names_to_interrupt_model_conversion,
        disable_group_convolution=args.disable_group_convolution,
        enable_batchmatmul_unfold=args.enable_batchmatmul_unfold,
        enable_rnn_unroll=args.enable_rnn_unroll,
        disable_suppression_flextranspose=args.disable_suppression_flextranspose,
        number_of_dimensions_after_flextranspose_compression=args.number_of_dimensions_after_flextranspose_compression,
        disable_suppression_flexstridedslice=args.disable_suppression_flexstridedslice,
        number_of_dimensions_after_flexstridedslice_compression=args.number_of_dimensions_after_flexstridedslice_compression,
        optimization_for_gpu_delegate=args.optimization_for_gpu_delegate,
        replace_argmax_to_reducemax_and_indicies_is_int64=args.replace_argmax_to_reducemax_and_indicies_is_int64,
        replace_argmax_to_reducemax_and_indicies_is_float32=args.replace_argmax_to_reducemax_and_indicies_is_float32,
        replace_argmax_to_fused_argmax_and_indicies_is_int64=args.replace_argmax_to_fused_argmax_and_indicies_is_int64,
        replace_argmax_to_fused_argmax_and_indicies_is_float32=args.replace_argmax_to_fused_argmax_and_indicies_is_float32,
        fused_argmax_scale_ratio=args.fused_argmax_scale_ratio,
        replace_to_pseudo_operators=args.replace_to_pseudo_operators,
        param_replacement_file=args.param_replacement_file,
        check_gpu_delegate_compatibility=args.check_gpu_delegate_compatibility,
        check_onnx_tf_outputs_elementwise_close=args.check_onnx_tf_outputs_elementwise_close,
        check_onnx_tf_outputs_elementwise_close_full=args.check_onnx_tf_outputs_elementwise_close_full,
        check_onnx_tf_outputs_sample_data_normalization=args.check_onnx_tf_outputs_sample_data_normalization,
        check_onnx_tf_outputs_elementwise_close_rtol=args.check_onnx_tf_outputs_elementwise_close_rtol,
        check_onnx_tf_outputs_elementwise_close_atol=args.check_onnx_tf_outputs_elementwise_close_atol,
        mvn_epsilon=args.mvn_epsilon,
        non_verbose=args.non_verbose,
    )


if __name__ == '__main__':
    main()

