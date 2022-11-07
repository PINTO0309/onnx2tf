#! /usr/bin/env python

import os
import re
__path__ = (os.path.dirname(__file__), )
with open(os.path.join(__path__[0], '__init__.py')) as f:
    init_text = f.read()
    __version__ = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
import io
import sys
import ast
import json
import logging
import requests
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
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

import onnx
import onnx_graphsurgeon as gs
from typing import Optional, List
from argparse import ArgumentParser

import importlib
from onnx2tf.utils.colors import Color
from sng4onnx import generate as op_name_auto_generate


def convert(
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_folder_path: Optional[str] = 'saved_model',
    output_signaturedefs: Optional[bool] = False,
    output_h5: Optional[bool] = False,
    output_integer_quantized_tflite: Optional[bool] = False,
    quant_type: Optional[str] = 'per-channel',
    quant_calib_input_op_name_np_data_path: Optional[List] = None,
    input_output_quant_dtype: Optional[str] = 'int8',
    not_use_onnxsim: Optional[bool] = False,
    not_use_opname_auto_generate: Optional[bool] = False,
    batch_size: Optional[int] = None,
    overwrite_input_shape: Optional[List[str]] = None,
    keep_ncw_or_nchw_or_ncdhw_input_names: Optional[List[str]] = None,
    keep_nwc_or_nhwc_or_ndhwc_input_names: Optional[List[str]] = None,
    replace_argmax_to_reducemax_and_indicies_is_int64: Optional[bool] = False,
    replace_argmax_to_reducemax_and_indicies_is_float32: Optional[bool] = False,
    replace_argmax_to_fused_argmax_and_indicies_is_int64: Optional[bool] = False,
    replace_argmax_to_fused_argmax_and_indicies_is_float32: Optional[bool] = False,
    fused_argmax_scale_ratio: Optional[float] = 0.5,
    replace_asin_to_pseudo_asin: Optional[bool] = False,
    replace_acos_to_pseudo_acos: Optional[bool] = False,
    replace_prelu_to_pseudo_prelu: Optional[bool] = False,
    replace_leakyrelu_to_pseudo_leakyrelu: Optional[bool] = False,
    replace_power_to_pseudo_power: Optional[bool] = False,
    replace_gathernd_to_pseudo_gathernd: Optional[bool] = False,
    replace_neg_to_pseudo_neg: Optional[bool] = False,
    replace_hardswish_to_pseudo_hardswish: Optional[bool] = False,
    param_replacement_file: Optional[str] = '',
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
        Output in Keras H5 format.

    output_integer_quantized_tflite: Optional[bool]
        Output of integer quantized tflite.

    quant_type: Optional[str]
        Selects whether "per-channel" or "per-tensor" quantization is used.\n
        Default: "per-channel"

    quant_calib_input_op_name_np_data_path: Optional[List]
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
        -qcind "input0" "../input0.npy" [[[[0.485, 0.456, 0.406]]]] [[[[0.229, 0.224, 0.225]]]]\n
        -qcind "input1" "./input1.npy" [0.1, ..., 0.64] [0.05, ..., 0.08]\n
        -qcind "input2" "input2.npy" [0.3] [0.07]

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
        Ignores --batch_size if specified at the same time as --batch_size.

    keep_ncw_or_nchw_or_ncdhw_input_names: Optional[List[str]]
        Holds the NCW or NCHW or NCDHW of the input shape for the specified INPUT OP names.\n
        If a nonexistent INPUT OP name is specified, it is ignored.\n
        Valid only for 3D, 4D and 5D input tensors.\n\n
        e.g. \n
        --keep_ncw_or_nchw_or_ncdhw_input_names=['input0', 'input1', 'input2']

    keep_nwc_or_nhwc_or_ndhwc_input_names: Optional[List[str]]
        Holds the NWC or NHWC or NDHWC of the input shape for the specified INPUT OP names.\n
        If a nonexistent INPUT OP name is specified, it is ignored.\n
        If the input OP name is the same as the input OP name specified\n
        in the keep_ncw_or_nchw_or_ncdhw_input_names option, it is ignored.\n
        Valid only for 3D, 4D and 5D input tensors.\n\n
        e.g. \n
        --keep_nwc_or_nhwc_or_ndhwc_input_names=['input0', 'input1', 'input2']

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

    replace_asin_to_pseudo_asin: Optional[bool]
        Replace Asin with a pseudo Asin.

    replace_acos_to_pseudo_acos: Optional[bool]
        Replace Acos with a pseudo Acos.

    replace_prelu_to_pseudo_prelu: Optional[bool]
        Replace PReLU with a pseudo PReLU.

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
        For MeanVarianceNormalization.\n
        The number to be added to the variance to avoid division by zero when normalizing the value.\n
        (input_tensor - mean) / tf.sqrt(variance + mvn_epsilon)\n
        Default: 0.0000000001

    param_replacement_file: Optional[str]
        Parameter replacement file path. (.json)

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n
        Only one of replace_argmax_to_reducemax_and_indicies_is_int64 and \n
        replace_argmax_to_reducemax_and_indicies_is_float32 can be specified.\n
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
    if not os.path.exists(input_onnx_file_path):
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'The specified *.onnx file does not exist. ' +
            f'input_onnx_file_path: {input_onnx_file_path}'
        )
        sys.exit(1)

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
            f'fused_argmax_scale_ratio must be specified in the range '+
            f'0.0 < fused_argmax_scale_ratio <= 1.0. '+
            f'fused_argmax_scale_ratio: {fused_argmax_scale_ratio}'
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
                print(
                    f'{Color.YELLOW}WARNING:{Color.RESET} '+
                    'Failed to optimize the onnx file.'
                )
                import traceback
                traceback.print_exc()

    # Automatic generation of each OP name - sng4onnx
    if not not_use_opname_auto_generate:
        if not non_verbose:
            print('')
            print(f'{Color.REVERCE}Automatic generation of each OP name started{Color.RESET}', '=' * 40)
        op_name_auto_generate(
            input_onnx_file_path=f'{input_onnx_file_path}',
            output_onnx_file_path=f'{input_onnx_file_path}',
            non_verbose=True,
        )
        if not non_verbose:
            print(f'{Color.GREEN}Automatic generation of each OP name complete!{Color.RESET}')

    # quantization_type
    disable_per_channel = False \
        if quant_type is not None and quant_type == 'per-channel' else True

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)

    graph = gs.import_onnx(onnx_graph)

    if not non_verbose:
        print('')
        print(f'{Color.REVERCE}Model loaded{Color.RESET}', '=' * 72)

    # Create Output folder
    os.makedirs(output_folder_path, exist_ok=True)

    # Define additional parameters
    additional_parameters = {
        'opset': graph.opset,
        'batch_size': batch_size,
        'non_verbose': non_verbose,
        'replace_argmax_to_reducemax_and_indicies_is_int64': replace_argmax_to_reducemax_and_indicies_is_int64,
        'replace_argmax_to_reducemax_and_indicies_is_float32': replace_argmax_to_reducemax_and_indicies_is_float32,
        'replace_argmax_to_fused_argmax_and_indicies_is_int64': replace_argmax_to_fused_argmax_and_indicies_is_int64,
        'replace_argmax_to_fused_argmax_and_indicies_is_float32': replace_argmax_to_fused_argmax_and_indicies_is_float32,
        'fused_argmax_scale_ratio': fused_argmax_scale_ratio,
        'replace_asin_to_pseudo_asin': replace_asin_to_pseudo_asin,
        'replace_acos_to_pseudo_acos': replace_acos_to_pseudo_acos,
        'replace_prelu_to_pseudo_prelu': replace_prelu_to_pseudo_prelu,
        'replace_leakyrelu_to_pseudo_leakyrelu': replace_leakyrelu_to_pseudo_leakyrelu,
        'replace_power_to_pseudo_power': replace_power_to_pseudo_power,
        'replace_gathernd_to_pseudo_gathernd': replace_gathernd_to_pseudo_gathernd,
        'replace_neg_to_pseudo_neg': replace_neg_to_pseudo_neg,
        'replace_hardswish_to_pseudo_hardswish': replace_hardswish_to_pseudo_hardswish,
        'replacement_parameters': replacement_parameters,
        'mvn_epsilon': mvn_epsilon,
    }

    tf_layers_dict = {}

    if not non_verbose:
        print('')
        print(f'{Color.REVERCE}Model convertion started{Color.RESET}', '=' * 60)

    with graph.node_ids():
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
                and quant_calib_input_op_name_np_data_path is None \
                and (graph_input.dtype != np.float32 or len(graph_input.shape) != 4):
                print(
                    f'{Color.RED}ERROR:{Color.RESET} ' +
                    f'For INT8 quantization, the input data type must be Float32. ' +
                    f'Also, if --quant_calib_input_op_name_np_data_path is not specified, ' +
                    f'all input OPs must assume 4D tensor image data. ' +
                    f'INPUT Name: {graph_input.name} INPUT Shape: {graph_input.shape} INPUT dtype: {graph_input.dtype}'
                )
                sys.exit(1)

            # make input
            op = importlib.import_module(f'onnx2tf.ops.Input')
            op.make_node(
                graph_input=graph_input,
                tf_layers_dict=tf_layers_dict,
                keep_ncw_or_nchw_or_ncdhw_input_names=keep_ncw_or_nchw_or_ncdhw_input_names,
                keep_nwc_or_nhwc_or_ndhwc_input_names=keep_nwc_or_nhwc_or_ndhwc_input_names,
                **additional_parameters,
            )

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

            op.make_node(
                graph_node=graph_node,
                tf_layers_dict=tf_layers_dict,
                **additional_parameters,
            )

        # List "optype"="Input"
        inputs = [
            layer_info['op'] \
                for layer_info in tf_layers_dict.values() \
                    if layer_info['optype'] == 'Input'
        ]

        # List Output
        output_names = [
            graph_output.name for graph_output in graph.outputs
        ]
        outputs = [
            layer_info['tf_node'] \
                for opname, layer_info in tf_layers_dict.items() \
                    if opname in output_names
        ]

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        if not non_verbose:
            print('')
            model.summary(line_length=140)
            print('')

        # Output in Keras H5 format
        if output_h5:
            if not non_verbose:
                print(f'{Color.REVERCE}h5 output started{Color.RESET}', '=' * 67)
            model.save(f'{output_folder_path}/model_float32.h5')
            if not non_verbose:
                print(f'{Color.GREEN}h5 output complete!{Color.RESET}')

        # Create concrete func
        run_model = tf.function(lambda *inputs : model(inputs))
        concrete_func = run_model.get_concrete_function(
            *[tf.TensorSpec(tensor.shape, tensor.dtype) for tensor in model.inputs]
        )

        # saved_model
        try:
            # concrete_func
            if not non_verbose:
                print(f'{Color.REVERCE}saved_model output started{Color.RESET}', '=' * 58)
            if not output_signaturedefs:
                tf.saved_model.save(concrete_func, output_folder_path)
            else:
                tf.saved_model.save(model, output_folder_path)
            if not non_verbose:
                print(f'{Color.GREEN}saved_model output complete!{Color.RESET}')
        except TypeError as e:
            # Switch to .pb
            if not non_verbose:
                print(f'{Color.GREEN}Switch to the output of an optimized protocol buffer file (.pb).{Color.RESET}')
            output_pb = True
            flag_for_output_switching_from_saved_model_to_pb_due_to_error = True
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
        tflite_model = converter.convert()
        with open(f'{output_folder_path}/model_float32.tflite', 'wb') as w:
            w.write(tflite_model)
        if not non_verbose:
            print(f'{Color.GREEN}Float32 tflite output complete!{Color.RESET}')

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter.convert()
        with open(f'{output_folder_path}/model_float16.tflite', 'wb') as w:
            w.write(tflite_model)
        if not non_verbose:
            print(f'{Color.GREEN}Float16 tflite output complete!{Color.RESET}')

        # Quantized TFLite
        if output_integer_quantized_tflite:
            # Dynamic Range Quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = []
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_disable_per_channel = disable_per_channel
            tflite_model = converter.convert()
            with open(f'{output_folder_path}/model_dynamic_range_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            if not non_verbose:
                print(f'{Color.GREEN}Dynamic Range Quantization tflite output complete!{Color.RESET}')

            # Download sample calibration data - MS-COCO x20 images
            # Used only when there is only one input OP, a 4D tensor image,
            # and --quant_calib_input_op_name_np_data_path is not specified.
            # Otherwise, calibrate using the data specified in --quant_calib_input_op_name_np_data_path.
            calib_data_dict = {}
            model_input_name_list = [
                model_input.name for model_input in model.inputs
            ]
            data_count = 0
            if output_integer_quantized_tflite \
                and quant_calib_input_op_name_np_data_path is None \
                and model.inputs[0].dtype == tf.float32 \
                and len(model.inputs[0].shape) == 4:

                # AUTO calib 4D images
                FILE_NAME = 'calibration_image_sample_data_20x128x128x3_float32.npy'
                DOWNLOAD_VER = '1.0.49'
                URL = f'https://github.com/PINTO0309/onnx2tf/releases/download/{DOWNLOAD_VER}/{FILE_NAME}'
                MEAN = np.asarray([[[[0.485, 0.456, 0.406]]]], dtype=np.float32)
                STD = np.asarray([[[[0.229, 0.224, 0.225]]]], dtype=np.float32)
                calib_sample_images_npy = requests.get(URL).content
                with io.BytesIO(calib_sample_images_npy) as f:
                    calib_data: np.ndarray = np.load(f)
                    data_count = calib_data.shape[0]
                    for model_input in model.inputs:
                        calib_data_dict[model_input.name] = \
                            [
                                tf.image.resize(
                                    calib_data.copy(),
                                    (model_input.shape[1], model_input.shape[2])
                                ),
                                MEAN,
                                STD,
                            ]
            else:
                if output_integer_quantized_tflite \
                    and quant_calib_input_op_name_np_data_path is not None:
                    for param in quant_calib_input_op_name_np_data_path:
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
                    calib_data_list = []
                    for model_input_name in model_input_name_list:
                        calib_data, mean, std = calib_data_dict[model_input_name]
                        normalized_calib_data = (calib_data[idx] - mean) / std
                        calib_data_list.append(normalized_calib_data)
                    yield calib_data_list

            # INT8 Quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_disable_per_channel = disable_per_channel
            converter.representative_dataset = representative_dataset_gen
            tflite_model = converter.convert()
            with open(f'{output_folder_path}/model_integer_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            if not non_verbose:
                print(f'{Color.GREEN}INT8 Quantization tflite output complete!{Color.RESET}')

            # Full Integer Quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_disable_per_channel = disable_per_channel
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
            with open(f'{output_folder_path}/model_full_integer_quant.tflite', 'wb') as w:
                w.write(tflite_model)
            if not non_verbose:
                print(f'{Color.GREEN}Full INT8 Quantization tflite output complete!{Color.RESET}')

            # Integer quantization with int16 activations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = []
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_disable_per_channel = disable_per_channel
            converter.representative_dataset = representative_dataset_gen
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
            tflite_model = converter.convert()
            with open(f'{output_folder_path}/model_integer_quant_with_int16_act.tflite', 'wb') as w:
                w.write(tflite_model)
            if not non_verbose:
                print(f'{Color.GREEN}INT8 Quantization with int16 activations tflite output complete!{Color.RESET}')

            # Full Integer quantization with int16 activations
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = []
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter._experimental_disable_per_channel = disable_per_channel
            converter.representative_dataset = representative_dataset_gen
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
            tflite_model = converter.convert()
            with open(f'{output_folder_path}/model_full_integer_quant_with_int16_act.tflite', 'wb') as w:
                w.write(tflite_model)
            if not non_verbose:
                print(f'{Color.GREEN}Full INT8 Quantization with int16 activations tflite output complete!{Color.RESET}')

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
            'Output in Keras H5 format.'
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
        '-qcind',
        '--quant_calib_input_op_name_np_data_path',
        type=str,
        action='append',
        nargs=4,
        help=\
            'INPUT Name of OP and path of calibration data file (Numpy) for quantization \n' +
            'and mean and std. \n' +
            'The specification can be omitted only when the input OP is a single 4D tensor image data. \n' +
            'If omitted, it is automatically calibrated using 20 normalized MS-COCO images. \n' +
            'The type of the input OP must be Float32. \n' +
            'Data for calibration must be pre-normalized to a range of 0 to 1. \n' +
            '-qcind {input_op_name} {numpy_file_path} {mean} {std} \n' +
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
            '-qcind "input0" "../input0.npy" [[[[0.485, 0.456, 0.406]]]] [[[[0.229, 0.224, 0.225]]]] \n' +
            '-qcind "input1" "./input1.npy" [0.1, ..., 0.64] [0.05, ..., 0.08] \n' +
            '-qcind "input2" "input2.npy" [0.3] [0.07]'
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
        '-rasin',
        '--replace_asin_to_pseudo_asin',
        action='store_true',
        help='Replace Asin with a pseudo Asin.'
    )
    parser.add_argument(
        '-racos',
        '--replace_acos_to_pseudo_acos',
        action='store_true',
        help='Replace Acos with a pseudo Acos.'
    )
    parser.add_argument(
        '-rpr',
        '--replace_prelu_to_pseudo_prelu',
        action='store_true',
        help='Replace PReLU with a pseudo PReLU.'
    )
    parser.add_argument(
        '-rlr',
        '--replace_leakyrelu_to_pseudo_leakyrelu',
        action='store_true',
        help='Replace LeakyReLU with a pseudo LeakyReLU.'
    )
    parser.add_argument(
        '-rpw',
        '--replace_power_to_pseudo_power',
        action='store_true',
        help='Replace Power with a pseudo Power.'
    )
    parser.add_argument(
        '-rgn',
        '--replace_gathernd_to_pseudo_gathernd',
        action='store_true',
        help='Replace GatherND with a pseudo GatherND.'
    )
    parser.add_argument(
        '-rng',
        '--replace_neg_to_pseudo_neg',
        action='store_true',
        help='Replace Neg with a pseudo Neg.'
    )
    parser.add_argument(
        '-rhs',
        '--replace_hardswish_to_pseudo_hardswish',
        action='store_true',
        help='Replace HardSwish with a pseudo HardSwish.'
    )
    parser.add_argument(
        '-me',
        '--mvn_epsilon',
        type=float,
        default=0.0000000001,
        help=\
            'For MeanVarianceNormalization. \n' +
            'The number to be added to the variance to avoid division by zero when normalizing the value. \n' +
            '(input_tensor - mean) / tf.sqrt(variance + mvn_epsilon) \n'
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
    calib_params = []
    if args.quant_calib_input_op_name_np_data_path is not None:
        for param in args.quant_calib_input_op_name_np_data_path:
            input_op_name = str(param[0])
            numpy_file_path = str(param[1])
            mean = np.asarray(ast.literal_eval(param[2]), dtype=np.float32)
            std = np.asarray(ast.literal_eval(param[3]), dtype=np.float32)
            calib_params.append(
                [input_op_name, numpy_file_path, mean, std]
            )
    if len(calib_params) == 0:
        calib_params = None

    # Convert
    model = convert(
        input_onnx_file_path=args.input_onnx_file_path,
        output_folder_path=args.output_folder_path,
        output_signaturedefs=args.output_signaturedefs,
        output_h5=args.output_h5,
        output_integer_quantized_tflite=args.output_integer_quantized_tflite,
        quant_type=args.quant_type,
        quant_calib_input_op_name_np_data_path=calib_params,
        input_output_quant_dtype=args.input_output_quant_dtype,
        not_use_onnxsim=args.not_use_onnxsim,
        not_use_opname_auto_generate=args.not_use_opname_auto_generate,
        batch_size=args.batch_size,
        overwrite_input_shape=args.overwrite_input_shape,
        keep_ncw_or_nchw_or_ncdhw_input_names=args.keep_ncw_or_nchw_or_ncdhw_input_names,
        keep_nwc_or_nhwc_or_ndhwc_input_names=args.keep_nwc_or_nhwc_or_ndhwc_input_names,
        replace_argmax_to_reducemax_and_indicies_is_int64=args.replace_argmax_to_reducemax_and_indicies_is_int64,
        replace_argmax_to_reducemax_and_indicies_is_float32=args.replace_argmax_to_reducemax_and_indicies_is_float32,
        replace_argmax_to_fused_argmax_and_indicies_is_int64=args.replace_argmax_to_fused_argmax_and_indicies_is_int64,
        replace_argmax_to_fused_argmax_and_indicies_is_float32=args.replace_argmax_to_fused_argmax_and_indicies_is_float32,
        fused_argmax_scale_ratio=args.fused_argmax_scale_ratio,
        replace_asin_to_pseudo_asin=args.replace_asin_to_pseudo_asin,
        replace_acos_to_pseudo_acos=args.replace_acos_to_pseudo_acos,
        replace_prelu_to_pseudo_prelu=args.replace_prelu_to_pseudo_prelu,
        replace_leakyrelu_to_pseudo_leakyrelu=args.replace_leakyrelu_to_pseudo_leakyrelu,
        replace_power_to_pseudo_power=args.replace_power_to_pseudo_power,
        replace_gathernd_to_pseudo_gathernd=args.replace_gathernd_to_pseudo_gathernd,
        replace_neg_to_pseudo_neg=args.replace_neg_to_pseudo_neg,
        replace_hardswish_to_pseudo_hardswish=args.replace_hardswish_to_pseudo_hardswish,
        param_replacement_file=args.param_replacement_file,
        mvn_epsilon=args.mvn_epsilon,
        non_verbose=args.non_verbose,
    )


if __name__ == '__main__':
    main()
