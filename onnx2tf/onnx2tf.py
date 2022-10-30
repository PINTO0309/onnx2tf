#! /usr/bin/env python

import os
import re
__path__ = (os.path.dirname(__file__), )
with open(os.path.join(__path__[0], '__init__.py')) as f:
    init_text = f.read()
    __version__ = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
import sys
import json
import logging
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
    not_use_onnxsim: Optional[bool] = False,
    not_use_opname_auto_generate: Optional[bool] = False,
    batch_size: Optional[int] = None,
    overwrite_input_shape: Optional[List[str]] = None,
    keep_ncw_or_nchw_or_ncdhw_input_names: Optional[List[str]] = None,
    replace_argmax_to_reducemax_and_indicies_is_int64: Optional[bool] = False,
    replace_argmax_to_reducemax_and_indicies_is_float32: Optional[bool] = False,
    replace_argmax_to_fused_argmax_and_indicies_is_int64: Optional[bool] = False,
    replace_argmax_to_fused_argmax_and_indicies_is_float32: Optional[bool] = False,
    fused_argmax_scale_ratio: Optional[float] = 0.5,
    replace_asin_to_pseudo_asin: Optional[bool] = False,
    replace_acos_to_pseudo_acos: Optional[bool] = False,
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
        0.0 < fused_argmax_scale_ratio < 1.0\n
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
    if ra_option_list.count(True) > 0 and not (0.0 < fused_argmax_scale_ratio < 1.0):
        print(
            f'fused_argmax_scale_ratio must be specified in the range '+
            f'0.0 < fused_argmax_scale_ratio < 1.0. '+
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
            op = importlib.import_module(f'onnx2tf.ops.Input')
            op.make_node(
                graph_input=graph_input,
                tf_layers_dict=tf_layers_dict,
                keep_ncw_or_nchw_or_ncdhw_input_names=keep_ncw_or_nchw_or_ncdhw_input_names,
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
            '0.0 < fused_argmax_scale_ratio < 1.0 \n' +
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

    # Convert
    model = convert(
        input_onnx_file_path=args.input_onnx_file_path,
        output_folder_path=args.output_folder_path,
        output_signaturedefs=args.output_signaturedefs,
        output_h5=args.output_h5,
        not_use_onnxsim=args.not_use_onnxsim,
        not_use_opname_auto_generate=args.not_use_opname_auto_generate,
        batch_size=args.batch_size,
        overwrite_input_shape=args.overwrite_input_shape,
        keep_ncw_or_nchw_or_ncdhw_input_names=args.keep_ncw_or_nchw_or_ncdhw_input_names,
        replace_argmax_to_reducemax_and_indicies_is_int64=args.replace_argmax_to_reducemax_and_indicies_is_int64,
        replace_argmax_to_reducemax_and_indicies_is_float32=args.replace_argmax_to_reducemax_and_indicies_is_float32,
        replace_argmax_to_fused_argmax_and_indicies_is_int64=args.replace_argmax_to_fused_argmax_and_indicies_is_int64,
        replace_argmax_to_fused_argmax_and_indicies_is_float32=args.replace_argmax_to_fused_argmax_and_indicies_is_float32,
        fused_argmax_scale_ratio=args.fused_argmax_scale_ratio,
        replace_asin_to_pseudo_asin=args.replace_asin_to_pseudo_asin,
        replace_acos_to_pseudo_acos=args.replace_acos_to_pseudo_acos,
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
