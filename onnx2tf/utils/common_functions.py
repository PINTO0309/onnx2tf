import math
import os
import io
import sys
import copy
import json
import psutil
import random
random.seed(0)
import requests
import flatbuffers
import itertools
import collections
import traceback
import subprocess
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tf_keras
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.utils import conv_utils
import onnx
from onnx.serialization import ProtoSerializer
import onnx_graphsurgeon as gs
try:
    import onnxruntime as ort
except Exception as ex:
    pass
from onnx2tf.utils.logging import *
from typing import Any, List, Optional, Union, Tuple, Dict
from functools import wraps
from collections import namedtuple
from onnx2tf.utils.enums import (
    TF_DTYPES_TO_NUMPY_DTYPES,
    NUMPY_DTYPES_TO_TF_DTYPES,
)


INF_INDEX_VALUE: int = 4294967296
ONNX_INF_INDEX_VALUE = sys.maxsize # 9223372036854775807


def get_replacement_parameter(func):
    @wraps(func)
    def get_replacement_parameter_wrapper_func(*args, **kwargs):
        op_name = kwargs['graph_node'].name
        replacement_parameters = kwargs.get('replacement_parameters', None)
        kwargs['op_rep_params'] = []
        if replacement_parameters is not None:
            kwargs['op_rep_params'] = [
                replacement_parameter \
                    for replacement_parameter in replacement_parameters \
                        if replacement_parameter['op_name'] == op_name
            ]
        func(*args, **kwargs)
    return get_replacement_parameter_wrapper_func


def replace_parameter(
    *,
    value_before_replacement: Any,
    param_target: str,
    param_name: str,
    **kwargs: Dict,
):
    """Replace attributes, INPUT constants, and INPUT initializers with the specified values.

    Parameters
    ----------
    value_before_replacement: Any
    param_target: str
    param_name: str
    **kwargs: Dict

    Returns
    ----------
    replace_value: Any
    """
    replace_value = value_before_replacement
    op_rep_params = kwargs.get('op_rep_params', [])
    for op_rep_param in op_rep_params:
        if op_rep_param['param_target'] == param_target \
            and op_rep_param['param_name'] == param_name \
            and 'values' in op_rep_param:
            replace_value = op_rep_param.get('values', value_before_replacement)
            if isinstance(value_before_replacement, np.ndarray):
                replace_value = np.asarray(
                    replace_value,
                    dtype=value_before_replacement.dtype,
                )
            elif isinstance(value_before_replacement, list):
                replace_value = list(replace_value)
            elif isinstance(value_before_replacement, bool):
                replace_value = \
                    bool(replace_value) if isinstance(replace_value, int) and replace_value in [0, 1] else \
                    bool(int(replace_value)) if isinstance(replace_value, str) and replace_value in ["0", "1"] else \
                    False if isinstance(replace_value, str) and replace_value.lower() == "false" else \
                    True if isinstance(replace_value, str) and replace_value.lower() == "True" else \
                    replace_value
            elif isinstance(value_before_replacement, int):
                replace_value = int(replace_value)
            elif isinstance(value_before_replacement, float):
                replace_value = float(replace_value)
            elif isinstance(value_before_replacement, str):
                replace_value = str(replace_value)
            elif tf_keras.backend.is_keras_tensor(value_before_replacement):
                replace_value = np.asarray(
                    replace_value,
                    dtype=TF_DTYPES_TO_NUMPY_DTYPES[value_before_replacement.dtype],
                )
            break
    return replace_value


def pre_process_transpose(
    *,
    value_before_transpose: Any,
    param_target: str,
    param_name: str,
    **kwargs: Dict,
):
    """Add Transpose as a post-processing step for Reshape OP.

    Parameters
    ----------
    value_before_transpose: tf_op
    param_target: str
    param_name: str
    **kwargs: Dict

    Returns
    ----------
    transposed_value: tf_op
    """
    transposed_value = value_before_transpose
    op_rep_params = kwargs.get('op_rep_params', [])
    for op_rep_param in op_rep_params:
        if op_rep_param['param_target'] == param_target \
            and op_rep_param['param_name'] == param_name:
            transpose_perm = op_rep_param.get('pre_process_transpose_perm', None)
            if transpose_perm is not None:
                transposed_value = transpose_with_flexing_deterrence(
                    input_tensor=value_before_transpose,
                    perm=transpose_perm,
                    **kwargs,
                )
            break
    return transposed_value


def post_process_transpose(
    *,
    value_before_transpose: Any,
    param_target: str,
    param_name: str,
    graph_node: gs.Node = None,
    **kwargs: Dict,
):
    """Add Transpose as a post-processing step for Reshape OP.

    Parameters
    ----------
    value_before_transpose: tf_op
    param_target: dict
    param_name: dict
    **kwargs: dict

    Returns
    ----------
    transposed_value: tf_op
    """
    transposed_value = value_before_transpose
    op_rep_params = kwargs.get('op_rep_params', [])
    for op_rep_param in op_rep_params:
        if op_rep_param['param_target'] == param_target \
            and op_rep_param['param_name'] == param_name:
            transpose_perm = op_rep_param.get('post_process_transpose_perm', None)
            if transpose_perm is not None:
                if graph_node is not None \
                    and graph_node.op != "Concat":
                    transposed_value = \
                        transpose_with_flexing_deterrence(
                            input_tensor=value_before_transpose,
                            perm=transpose_perm,
                            **kwargs,
                        )
                else:
                    if value_before_transpose.shape is not None \
                        and len(value_before_transpose.shape) == 1 \
                        and value_before_transpose.shape[0] is not None:
                        # Gather
                        transposed_value = tf.gather(
                            params=value_before_transpose,
                            indices=tf.convert_to_tensor(transpose_perm)
                        )
                    else:
                        # Normal
                        transposed_value = \
                            transpose_with_flexing_deterrence(
                                input_tensor=value_before_transpose,
                                perm=transpose_perm,
                                **kwargs,
                            )
            break
    return transposed_value


def make_tf_node_info(**kwargs):
    """Generate information for debug log output.

    Parameters
    ----------
    tf_op_type: dict
    tf_attrs: dict
    tf_inputs: dict
    tf_outputs: dict

    Returns
    ----------
    tf_node_info: dict
    """
    tf_node_info = {}
    node_info: dict = kwargs.get('node_info', None)
    if node_info is not None:
        tf_op_type = node_info.get('tf_op_type', None)
        tf_node_info['tf_op_type'] = \
            tf_op_type.__name__ if hasattr(tf_op_type, '__name__') else \
            tf_op_type if isinstance(tf_op_type, str) else ''
        tf_attrs: dict = node_info.get('tf_attrs', None)
        if tf_attrs is not None:
            tf_node_info['tf_attrs'] = {
                attr_key: {
                    'shape': attr_val.shape if hasattr(attr_val, 'shape') else None,
                    'dtype': attr_val.dtype if hasattr(attr_val, 'dtype') else None,
                    'val': attr_val,
                } for attr_key, attr_val in tf_attrs.items()
            }
        tf_inputs: dict = node_info.get('tf_inputs', None)
        if tf_inputs is not None:
            tf_node_info['tf_inputs'] = {
                input_key: {
                    'name': input_val.name if hasattr(input_val, 'name') else None,
                    'shape': input_val.shape if hasattr(input_val, 'shape') else None,
                    'dtype': input_val.dtype if hasattr(input_val, 'dtype') else None,
                    'val': input_val \
                        if isinstance(input_val, list) \
                            or isinstance(input_val, str) \
                            or isinstance(input_val, bool) \
                            or isinstance(input_val, int) \
                            or isinstance(input_val, float) else None,
                } for input_key, input_val in tf_inputs.items()
            }
        tf_outputs: dict = node_info.get('tf_outputs', None)
        if tf_outputs is not None:
            tf_node_info['tf_outputs'] = {
                output_key: {
                    'name': output_val.name if hasattr(output_val, 'name') else None,
                    'shape': output_val.shape if hasattr(output_val, 'shape') else None,
                    'dtype': output_val.dtype if hasattr(output_val, 'dtype') else None,
                    'val': output_val \
                        if isinstance(output_val, list) \
                            or isinstance(output_val, str) \
                            or isinstance(output_val, bool) \
                            or isinstance(output_val, int) \
                            or isinstance(output_val, float) else None,
                } for output_key, output_val in tf_outputs.items()
            }
    return tf_node_info


def print_node_info(func):
    @wraps(func)
    def print_wrapper_func(*args, **kwargs):
        input_onnx_file_path: str = kwargs.get('input_onnx_file_path', None)
        graph_input: gs.Variable = kwargs.get('graph_input', None)
        graph_node: gs.Variable = kwargs.get('graph_node', None)
        tf_layers_dict: dict = kwargs.get('tf_layers_dict', None)
        if get_log_level() <= LOG_LEVELS['debug']:
            if graph_input is not None:
                debug(
                    Color.GREEN(f'INFO:') + ' '+
                    Color.GREEN(f'input_op_name') + f': {graph_input.name} '+
                    Color.GREEN(f'shape') + f': {graph_input.shape} '+
                    Color.GREEN(f'dtype') + f': {graph_input.dtype}'
                )
            elif graph_node is not None:
                debug('')
                op_counta: int = kwargs.get('op_counta', 1)
                total_op_count: int = kwargs.get('total_op_count', 0)
                debug(
                    Color.GREEN(f'INFO:') + ' '+
                    Color.GREEN(f'{op_counta} / {total_op_count}')
                )
                debug(
                    Color.GREEN(f'INFO:') + ' ' + Color.MAGENTA(f'onnx_op_type') + ': '+
                    f'{graph_node.op}' + Color.MAGENTA(' onnx_op_name') + f': {graph_node.name}')
                for idx, graph_node_input in enumerate(graph_node.inputs):
                    debug(
                        Color.GREEN(f'INFO:') + ' '+
                        Color.CYAN(f' input_name.{idx+1}') + f': {graph_node_input.name} '+
                        Color.CYAN(f'shape') + f': {graph_node_input.shape} '+
                        Color.CYAN(f'dtype') + f': {graph_node_input.dtype}'
                    )
                for idx, graph_node_output in enumerate(graph_node.outputs):
                    debug(
                        Color.GREEN(f'INFO:') + ' '+
                        Color.CYAN(f' output_name.{idx+1}') + f': {graph_node_output.name} '+
                        Color.CYAN(f'shape') + f': {graph_node_output.shape} '+
                        Color.CYAN(f'dtype') + f': {graph_node_output.dtype}'
                    )
        try:
            result = func(*args, **kwargs)

            if get_log_level() <= LOG_LEVELS['debug']:
                if graph_node is not None and tf_layers_dict is not None:
                    for idx, graph_node_output in enumerate(graph_node.outputs):
                        tf_layer_info: dict = tf_layers_dict.get(graph_node_output.name, None)
                        if tf_layer_info is not None:
                            tf_node_info = tf_layer_info.get('tf_node_info', None)
                            if tf_node_info is not None:
                                tf_op_type = tf_node_info.get('tf_op_type', None)
                                debug(
                                    Color.GREEN(f'INFO:') + ' ' + \
                                    Color.MAGENTA(f'tf_op_type') + f': {tf_op_type}'
                                )

                                tf_inputs = tf_node_info.get('tf_inputs', None)
                                if tf_inputs is not None:
                                    for input_idx, (input_key, input_values) in enumerate(tf_inputs.items()):
                                        input_info_text = \
                                            Color.GREEN(f'INFO:') + ' ' + \
                                            Color.BLUE(f' input.{input_idx+1}.{input_key}') + ': '
                                        for input_attr_name, input_attr_value in input_values.items():
                                            input_info_text += \
                                                Color.BLUE(f'{input_attr_name}') + f': {input_attr_value} ' \
                                                if input_attr_value  is not None else ''
                                        debug(input_info_text)

                                tf_outputs = tf_node_info.get('tf_outputs', None)
                                if tf_outputs is not None:
                                    for output_idx, (output_key, output_values) in enumerate(tf_outputs.items()):
                                        output_info_text = \
                                            Color.GREEN(f'INFO:') + ' ' + \
                                            Color.BLUE(f' output.{output_idx+1}.{output_key}') + ': '
                                        for output_attr_name, output_attr_value in output_values.items():
                                            output_info_text += \
                                                Color.BLUE(f'{output_attr_name}') + f': {output_attr_value} ' \
                                                if output_attr_value  is not None else ''
                                        debug(output_info_text)
            return result
        except:
            error(f'The trace log is below.')
            import traceback
            error(traceback.format_exc(), prefix=False)
            if input_onnx_file_path is not None:
                error(
                    f'input_onnx_file_path: {input_onnx_file_path}'
                )
            if graph_node is not None:
                error(
                    f'onnx_op_name: {graph_node.name}'
                )
            error(
                f'Read this and deal with it. https://github.com/PINTO0309/onnx2tf#parameter-replacement'
            )
            error(
                f'Alternatively, if the input OP has a dynamic dimension, ' +
                f'use the -b or -ois option to rewrite it to a static shape and try again.'
            )
            error(
                f'If the input OP of ONNX before conversion is NHWC or ' +
                f'an irregular channel arrangement other than NCHW, use the -kt or -kat option.'
            )
            error(
                f'Also, for models that include NonMaxSuppression in the post-processing, ' +
                f'try the -onwdt option.'
            )
            sys.exit(1)
    return print_wrapper_func


def inverted_operation_enable_disable(func):
    @wraps(func)
    def inverted_operation_enable_disable_wrapper_func(*args, **kwargs):
        result = func(*args, **kwargs)
        """
        The output_shape_trans stores the result of determining
        whether the final output shape of the connected OP differs between ONNX and TensorFlow.
        before_op_output_shape_trans is used to determine
        if the input tensor needs to be transposed within the processing body of each OP.

        True: Transpose the input tensor from NCHW to NHWC and so on
        False: No transposition
        """
        graph_node = kwargs.get('graph_node', None)
        tf_layers_dict = kwargs.get('tf_layers_dict', None)
        batch_size = kwargs.get('batch_size', None)
        output_shape_trans = False
        for graph_node_output in graph_node.outputs:
            onnx_node_output: gs.Variable = graph_node_output
            onnx_node_output_shape = onnx_node_output.shape
            onnx_node_output_shape = [
                shape if not isinstance(shape, str) else None for shape in onnx_node_output_shape
            ] if onnx_node_output_shape is not None else None
            if onnx_node_output_shape is not None \
                and len(onnx_node_output_shape) > 0 \
                and onnx_node_output_shape.count(None) != len(onnx_node_output_shape) \
                and batch_size is not None:
                onnx_node_output_shape[0] = batch_size
            if onnx_node_output_shape is not None:
                onnx_node_output_shape = [
                    None if s is not None and not isinstance(s, str) and s < 1 else s \
                        for s in onnx_node_output_shape
                ]
            tf_node_output_shape = tf_layers_dict[onnx_node_output.name]['tf_node'].shape

            trans_judge = (onnx_node_output_shape != tf_node_output_shape)
            # Avoiding patterns of misjudgment when the second and subsequent dimensions are all the same value
            if tf_node_output_shape != tf.TensorShape(None) \
                and len(tf_node_output_shape) >= 3:
                base_shape = tf_node_output_shape[1]
                if len(tf_node_output_shape)-1 == sum([1 if base_shape == s else 0 for s in tf_node_output_shape[1:]]) \
                    and (onnx_node_output_shape == tf_node_output_shape) \
                    and graph_node.op != 'MatMul':
                    trans_judge = True
            output_shape_trans = output_shape_trans or trans_judge
            tf_layers_dict[onnx_node_output.name]['before_op_output_shape_trans'] = output_shape_trans


        return result
    return inverted_operation_enable_disable_wrapper_func


def auto_cast(func):
    @wraps(func)
    def auto_cast_wrapper_func(*args, **kwargs):
        const_or_var = func(*args, **kwargs)
        if isinstance(const_or_var, np.ndarray) \
            and const_or_var.dtype == np.float16:
            const_or_var = const_or_var.astype(np.float32)
        elif isinstance(const_or_var, tf.Tensor) \
            and const_or_var.dtype == tf.float16:
            const_or_var = tf.cast(const_or_var, dtype=tf.float32)
        elif not isinstance(const_or_var, np.ndarray) \
            and not isinstance(const_or_var, gs.Variable) \
            and not isinstance(const_or_var, gs.Constant) \
            and tf_keras.backend.is_keras_tensor(const_or_var) \
            and const_or_var.dtype == tf.float16:
            const_or_var = tf.cast(const_or_var, dtype=tf.float32)
        else:
            pass
        return const_or_var
    return auto_cast_wrapper_func


@auto_cast
def get_constant_or_variable(
    const_or_var: Any,
    before_op_output_shape_trans: bool,
    is_bias: Optional[bool] = False,
) -> Any:
    """Get a Numpy constant or gs.Variable from graph_surgeon node.

    Parameters
    ----------
    const_or_var: gs.Variable
        gs.Variable

    Returns
    ----------
    const_or_var:
        Numpy array or gs.Variable
    """
    if hasattr(const_or_var, 'values'):
        values = const_or_var.values
        if not before_op_output_shape_trans:
            return values
        tensor_rank = values.ndim
        if tensor_rank > 2:
            convertion_table = [0] + [i for i in range(2, tensor_rank)] + [1]
            values = values.transpose(convertion_table)
        elif tensor_rank == 1 and values.size > 2 and not is_bias:
            convertion_table = [0] + [i for i in range(2, values.size)] + [1]
            new_values = np.zeros(values.size, dtype=values.dtype)
            for new_idx, idx in enumerate(convertion_table):
                new_values[new_idx] = values[idx]
            values = copy.deepcopy(new_values)
        return values
    else:
        return const_or_var


@auto_cast
def get_weights_constant_or_variable(
    const_or_var: Any,
    kernel_size: int,
) -> Any:
    """For obtaining transposed weights.

    Parameters
    ----------
    const_or_var: gs.Variable
        gs.Variable

    kernel_size: int
        Number of elements in kernel_shape\n
        Conv1D: 1\n
        Conv2D: 2\n
        Conv3D: 3

    Returns
    ----------
    const_or_var:
        Transposed weights. Numpy array or gs.Variable
    """
    if hasattr(const_or_var, 'values'):
        values = const_or_var.values
        """
        e.g.
        Conv1D
            ONNX: [C_OUT, C_IN,     X] = [8,1,3]
            tf  : [    X, C_IN, C_OUT] = [3,1,8]

        Conv2D
            ONNX: [C_OUT, C_IN,     Y,     X] = [8,1,3,3]
            tf  : [    Y,    X,  C_IN, C_OUT] = [3,3,1,8]

        Conv3D
            ONNX: [C_OUT, C_IN, Z,    Y,     X] = [8,1,3,3,3]
            tf  : [    Z,    Y, X, C_IN, C_OUT] = [3,3,3,1,8]
        """
        convertion_table = [i for i in range(2, kernel_size + 2)] + [1, 0]
        values = values.transpose(convertion_table)
        return tf.convert_to_tensor(values)
    elif hasattr(const_or_var, 'inputs') \
        and hasattr(const_or_var.inputs[0], 'attrs') \
        and 'value' in const_or_var.inputs[0].attrs \
        and hasattr(const_or_var.inputs[0].attrs['value'], 'values'):
        values = const_or_var.inputs[0].attrs['value'].values
        """
        e.g.
        Conv1D
            ONNX: [C_OUT, C_IN,     X] = [8,1,3]
            tf  : [    X, C_IN, C_OUT] = [3,1,8]

        Conv2D
            ONNX: [C_OUT, C_IN,     Y,     X] = [8,1,3,3]
            tf  : [    Y,    X,  C_IN, C_OUT] = [3,3,1,8]

        Conv3D
            ONNX: [C_OUT, C_IN, Z,    Y,     X] = [8,1,3,3,3]
            tf  : [    Z,    Y, X, C_IN, C_OUT] = [3,3,3,1,8]
        """
        convertion_table = [i for i in range(2, kernel_size + 2)] + [1, 0]
        values = values.transpose(convertion_table).astype(np.float32)
        if isinstance(values, np.ndarray) and values.dtype in (tf.int8, tf.uint8):
            values = values.astype(np.float32)
        return tf.convert_to_tensor(values)
    elif isinstance(const_or_var, tf.Tensor):
        values = const_or_var
        """
        e.g.
        Conv1D
            ONNX: [C_OUT, C_IN,     X] = [8,1,3]
            tf  : [    X, C_IN, C_OUT] = [3,1,8]

        Conv2D
            ONNX: [C_OUT, C_IN,     Y,     X] = [8,1,3,3]
            tf  : [    Y,    X,  C_IN, C_OUT] = [3,3,1,8]

        Conv3D
            ONNX: [C_OUT, C_IN, Z,    Y,     X] = [8,1,3,3,3]
            tf  : [    Z,    Y, X, C_IN, C_OUT] = [3,3,3,1,8]
        """
        convertion_table = [i for i in range(2, kernel_size + 2)] + [1, 0]
        values = tf.cast(tf.transpose(values, perm=convertion_table), tf.float32)
        return tf.convert_to_tensor(values)
    elif isinstance(const_or_var.i(), gs.Constant) \
        and hasattr(const_or_var.i(), 'values'):
        values = const_or_var.i().values
        """
        e.g.
        Conv1D
            ONNX: [C_OUT, C_IN,     X] = [8,1,3]
            tf  : [    X, C_IN, C_OUT] = [3,1,8]

        Conv2D
            ONNX: [C_OUT, C_IN,     Y,     X] = [8,1,3,3]
            tf  : [    Y,    X,  C_IN, C_OUT] = [3,3,1,8]

        Conv3D
            ONNX: [C_OUT, C_IN, Z,    Y,     X] = [8,1,3,3,3]
            tf  : [    Z,    Y, X, C_IN, C_OUT] = [3,3,3,1,8]
        """
        convertion_table = [i for i in range(2, kernel_size + 2)] + [1, 0]
        values = values.transpose(convertion_table).astype(np.float32)
        if isinstance(values, np.ndarray) and values.dtype in (tf.int8, tf.uint8):
            values = values.astype(np.float32)
        return tf.convert_to_tensor(values)
    else:
        return const_or_var


def check_cuda_enabled() -> bool:
    try:
        output = subprocess.check_output('nvidia-smi', shell=True)
        if 'nvidia-smi' in output.decode().lower():
            return True
        else:
            return False
    except Exception as ex:
        return False


def convert_axis(
    *,
    axis: int,
    tensor_rank: int,
    before_op_output_shape_trans: bool,
) -> int:
    """Convert axis from NCW to NWC or NCHW to NHWC or NCDHW to NDHWC.

    Parameters
    ----------
    axis: int
        Axis value to be replaced

    tensor_rank: int
        Number of ranks of ex-tensors specified by axis

    Returns
    ----------
    converted_axis: int
        Converted axis
    """
    # Convert a negative number of axis to a positive number
    converted_axis = axis if axis >= 0 else axis + tensor_rank

    if not before_op_output_shape_trans:
        return converted_axis

    # 3D and 4D and 5D axis conversion table
    """
    convertion_table_3d = [0,2,1]
    convertion_table_4d = [0,3,1,2]
    convertion_table_5d = [0,4,1,2,3]
    convertion_table_6d = [0,5,1,2,3,4]
        :
    """
    if tensor_rank > 2:
        convertion_table = [0] + [tensor_rank - 1] + [i for i in range(1, tensor_rank - 1)]
        converted_axis = convertion_table[converted_axis]

    return converted_axis


def convert_reverse_axis(
    *,
    axis: int,
    tensor_rank: int,
    before_op_output_shape_trans: bool,
) -> int:
    """Convert axis from NWC to NCW or NHWC to NCHW or NDHWC to NCDHW.

    Parameters
    ----------
    axis: int
        Axis value to be replaced

    tensor_rank: int
        Number of ranks of ex-tensors specified by axis

    Returns
    ----------
    converted_axis: int
        Converted axis
    """
    # Convert a negative number of axis to a positive number
    converted_axis = axis if axis >= 0 else axis + tensor_rank

    if not before_op_output_shape_trans:
        return converted_axis

    # 3D and 4D and 5D axis conversion table
    """
    convertion_table_3d = [0,2,1]
    convertion_table_4d = [0,3,1,2]
    convertion_table_5d = [0,4,1,2,3]
    convertion_table_6d = [0,5,1,2,3,4]
        :
    """
    if tensor_rank > 2:
        convertion_table = [0] + [tensor_rank - 1] + [i for i in range(1, tensor_rank - 1)]
        converted_axis = convertion_table.index(converted_axis)

    return converted_axis


def broadcast_validity_check(
    shape1: Union[np.ndarray, List],
    shape2: Union[np.ndarray, List],
):
    """Check the validity of dimension shape for same length of tensors.

    Parameters
    ----------
    shape1: Union[np.ndarray, List]
        1d list or ndarray.

    shape2: Union[np.ndarray, List]
        1d list or ndarray.

    Returns
    -------
    result: bool
        True if shape1 and shape2 is valid for broadcasting, else False
    """
    result = False

    if shape1 is None or shape2 is None:
        return result
    elif len(shape1) != len(shape2):
        return result
    else:
        for i, j in zip(shape1, shape2):
            if i == j or i == 1 or j == 1:
                result = True
            else:
                result = False
                break

    return result


def pre_explicit_broadcast(
    *,
    input_tensor_1: Any,
    input_tensor_2: Any,
) -> Tuple[Any, Any]:
    """Shrink a tensor whose input_tensor_1 and input_tensor_2
    have the same rank and all but one dimension is 1.

    Parameters
    ----------
    input_tensor_1: Any
        gs.Variable or np.ndarray

    input_tensor_2: Any
        gs.Variable or np.ndarray

    Returns
    ----------
    input_tensor_1: Any
        gs.Variable or np.ndarray

    input_tensor_2: Any
        gs.Variable or np.ndarray
    """
    # e.g.1
    # input:
    #   input_tensor_1: [1,80,80,12]
    #   input_tensor_2: [1,12,1,1]
    # output:
    #   input_tensor_1: [1,80,80,12]
    #   input_tensor_2: [12]
    #
    # e.g.2
    # input:
    #   input_tensor_1: [1,2,3,4]
    #   input_tensor_2: [1,3,1,1]
    # output:
    #   input_tensor_1: [1,2,3,4]
    #   input_tensor_2: [3,1]
    #
    # e.g.3
    # input:
    #   input_tensor_1: [1,2,3,4]
    #   input_tensor_2: [1,1,1]
    # output:
    #   input_tensor_1: [1,2,3,4]
    #   input_tensor_2: [1,1,1,1]
    if input_tensor_1.shape is not None \
        and input_tensor_1.shape != tf.TensorShape(None) \
        and input_tensor_2.shape is not None \
        and input_tensor_2.shape != tf.TensorShape(None) \
        and None not in input_tensor_1.shape \
        and None not in input_tensor_2.shape \
        and len(input_tensor_1.shape) == len(input_tensor_2.shape):

        # Broadcasting of the operation Checks whether the operation functions normally,
        # and if so, terminates the process without doing anything.
        try:
            dummy_mul = input_tensor_1 * input_tensor_2
            max_shape_prod = max(np.prod(input_tensor_1.shape), np.prod(input_tensor_2.shape))
            if np.prod(dummy_mul.shape) <= max_shape_prod:
                return input_tensor_1, input_tensor_2
        except Exception as e:
            pass
        else:
            return input_tensor_1, input_tensor_2

        input_tensor_2_shape = input_tensor_2.shape
        squeezed_input_tensor_2_shape = [idx for idx in input_tensor_2_shape if idx != 1]
        squeezed_input_tensor_2_shape_rank = len(squeezed_input_tensor_2_shape)
        input_tensor_1_shape = input_tensor_1.shape
        if squeezed_input_tensor_2_shape_rank == 1 \
            and squeezed_input_tensor_2_shape[0] in input_tensor_1_shape:
            input_tensor_2 = tf.squeeze(input_tensor_2)
            reversed_input_tensor_1_shape = []
            if isinstance(input_tensor_1_shape, list):
                reversed_input_tensor_1_shape = input_tensor_1_shape.reverse()
            elif isinstance(input_tensor_1_shape, tuple):
                reversed_input_tensor_1_shape = list(input_tensor_1_shape[::-1])
            elif isinstance(input_tensor_1_shape, np.ndarray):
                reversed_input_tensor_1_shape = input_tensor_1_shape[::-1].tolist()
            elif isinstance(input_tensor_1_shape, tf.TensorShape):
                reversed_input_tensor_1_shape = list(input_tensor_1_shape[::-1])
            expand_count = reversed_input_tensor_1_shape.index(squeezed_input_tensor_2_shape[0])
            for _ in range(expand_count):
                input_tensor_2 = tf.expand_dims(
                    input=input_tensor_2,
                    axis=-1,
                )
        else:
            input_tensor_1_shape = input_tensor_1.shape
            squeezed_input_tensor_1_shape = [idx for idx in input_tensor_1_shape if idx != 1]
            squeezed_input_tensor_1_shape_rank = len(squeezed_input_tensor_1_shape)
            input_tensor_2_shape = input_tensor_2.shape
            if squeezed_input_tensor_1_shape_rank == 1 \
                and squeezed_input_tensor_1_shape[0] in input_tensor_2_shape:
                input_tensor_1 = tf.squeeze(input_tensor_1)
                reversed_input_tensor_2_shape = []
                if isinstance(input_tensor_2_shape, list):
                    reversed_input_tensor_2_shape = input_tensor_2_shape.reverse()
                elif isinstance(input_tensor_2_shape, tuple):
                    reversed_input_tensor_2_shape = list(input_tensor_2_shape[::-1])
                elif isinstance(input_tensor_2_shape, np.ndarray):
                    reversed_input_tensor_2_shape = input_tensor_2_shape[::-1].tolist()
                elif isinstance(input_tensor_2_shape, tf.TensorShape):
                    reversed_input_tensor_2_shape = list(input_tensor_2_shape[::-1])
                expand_count = reversed_input_tensor_2_shape.index(squeezed_input_tensor_1_shape[0])
                for _ in range(expand_count):
                    input_tensor_1 = tf.expand_dims(
                        input=input_tensor_1,
                        axis=-1,
                    )
    elif input_tensor_1.shape is not None \
        and input_tensor_1.shape != tf.TensorShape(None) \
        and input_tensor_2.shape is not None \
        and input_tensor_2.shape != tf.TensorShape(None) \
        and None not in input_tensor_1.shape \
        and None not in input_tensor_2.shape \
        and len(input_tensor_1.shape) > len(input_tensor_2.shape) \
        and sum([1 if not isinstance(dim, str) and dim == 1 else 0 for dim in input_tensor_2.shape]) == len(input_tensor_2.shape):
        expand_count = len(input_tensor_1.shape) - len(input_tensor_2.shape)
        for _ in range(expand_count):
            input_tensor_2 = tf.expand_dims(
                input=input_tensor_2,
                axis=-1,
            )

    return input_tensor_1, input_tensor_2


def explicit_broadcast(
    *,
    const_or_var_1: Any,
    const_or_var_2: Any,
    graph_node: Optional[gs.Node] = None,
    tf_layers_dict: dict = None,
) -> Tuple[Any, Any]:
    """Of the two tensors in the argument, the one with the lower dimensionality
    is broadcast to match the one with the higher dimensionality.

    Parameters
    ----------
    const_or_var_1: Any
        gs.Variable or np.ndarray

    const_or_var_2: Any
        gs.Variable or np.ndarray

    Returns
    ----------
    const_or_var_1: Any
        gs.Variable or np.ndarray

    const_or_var_2: Any
        gs.Variable or np.ndarray
    """
    graph_node_input_name1 = None
    graph_node_input_name2 = None
    graph_node_input_shape1 = []
    graph_node_input_shape2 = []
    if graph_node is not None:
        graph_node_input_name1 = graph_node.inputs[0].name
        graph_node_input_name2 = graph_node.inputs[1].name
        graph_node_input_shape1 = list(graph_node.inputs[0].shape) \
            if graph_node.inputs[0].shape is not None else None
        graph_node_input_shape2 = list(graph_node.inputs[1].shape) \
            if graph_node.inputs[1].shape is not None else None

    # If shape is empty (scalar value), return it without doing anything.
    if graph_node_input_shape1 == [] or graph_node_input_shape2 == []:
        return const_or_var_1, const_or_var_2

    # If const_or_var_1 and const_or_var_2 of TF have exactly the same shape, skip processing
    if list(const_or_var_1.shape) == list(const_or_var_2.shape):
        return const_or_var_1, const_or_var_2

    # If all dimensions are undefined dimensions, return without doing anything
    if graph_node_input_shape1 is not None \
        and sum([1 if isinstance(s, str) else 0 for s in graph_node_input_shape1]) == len(graph_node_input_shape1):
        return const_or_var_1, const_or_var_2
    if graph_node_input_shape2 is not None \
        and sum([1 if isinstance(s, str) else 0 for s in graph_node_input_shape2]) == len(graph_node_input_shape2):
        return const_or_var_1, const_or_var_2

    # If the input shape of ONNX is None, return without doing anything.
    if graph_node_input_shape1 is None or graph_node_input_shape2 is None:
        return const_or_var_1, const_or_var_2

    # If either operand have shape of all 1's, do not broadcast and return as is
    shape_for_judging_skip_processing_1 = [
        i if i is not None else INF_INDEX_VALUE for i in const_or_var_1.shape
    ]
    shape_for_judging_skip_processing_2 = [
        i if i is not None else INF_INDEX_VALUE for i in const_or_var_2.shape
    ]
    if np.prod(shape_for_judging_skip_processing_1) == 1 or np.prod(shape_for_judging_skip_processing_2) == 1:
        return const_or_var_1, const_or_var_2

    # If the two tensors to be processed have exactly the same number of axes, skip the process.
    if len(const_or_var_1.shape) == len(const_or_var_2.shape):
        return const_or_var_1, const_or_var_2

    # Dealing with tricky BatchNormalization
    #   X: [1,1024,1]
    #   scale: [1024]
    #   B: [1024]
    #   mean: [1024]
    #   var: [1024]
    #       ↓
    #   scale: [1,1024,1]
    #   B: [1,1024,1]
    #   mean: [1,1024,1]
    #   var: [1,1024,1]
    if len(const_or_var_1.shape) > len(const_or_var_2.shape) \
        and None not in const_or_var_1.shape \
        and sum([0 if isinstance(dim, str) else 1 for dim in const_or_var_1.shape]) == len(const_or_var_1.shape) \
        and None not in const_or_var_2.shape \
        and sum([0 if isinstance(dim, str) else 1 for dim in const_or_var_2.shape]) == len(const_or_var_2.shape) \
        and len(const_or_var_2.shape) == 1 \
        and np.prod(shape_for_judging_skip_processing_1) == np.prod(shape_for_judging_skip_processing_2):

        var2_rehsape_possible = False
        try:
            _ = tf.reshape(const_or_var_2, shape=shape_for_judging_skip_processing_1)
            var2_rehsape_possible = True
        except:
            var2_rehsape_possible = False
        if var2_rehsape_possible:
            const_or_var_2 = tf.reshape(const_or_var_2, shape=shape_for_judging_skip_processing_1)
            return const_or_var_1, const_or_var_2
    # https://github.com/PINTO0309/onnx2tf/issues/394
    elif len(const_or_var_1.shape) > len(const_or_var_2.shape) \
        and None not in const_or_var_1.shape \
        and sum([0 if isinstance(dim, str) else 1 for dim in const_or_var_1.shape]) == len(const_or_var_1.shape) \
        and None not in const_or_var_2.shape \
        and sum([0 if isinstance(dim, str) else 1 for dim in const_or_var_2.shape]) == len(const_or_var_2.shape) \
        and len(const_or_var_2.shape) == 1 \
        and graph_node.op == 'BatchNormalization':

        const_or_var_2_shape_dim = const_or_var_2.shape[0]
        for idx, shape_for_judging_skip_processing_1_dim in enumerate(shape_for_judging_skip_processing_1[::-1]):
            if shape_for_judging_skip_processing_1_dim == const_or_var_2_shape_dim:
                for _ in range(idx):
                    const_or_var_2 = tf.expand_dims(const_or_var_2, -1)
                return const_or_var_1, const_or_var_2

    # Swap: len(const_or_var_1.shape) > len(const_or_var_2.shape)
    swapped = 0
    if len(const_or_var_1.shape) < len(const_or_var_2.shape) and not graph_node.op in ['Sub', 'Div', 'Mod']:
        const_or_var_1, const_or_var_2 = const_or_var_2, const_or_var_1
        graph_node_input_name1, graph_node_input_name2 = graph_node_input_name2, graph_node_input_name1
        graph_node_input_shape1, graph_node_input_shape2 = graph_node_input_shape2, graph_node_input_shape1
        swapped += 1

        # Skip subsequent processing in the following patterns.
        #   const_or_var_1: [1,1,5000]
        #   const_or_var_2: [5000]
        if len(const_or_var_1.shape) >= 1 \
            and len(const_or_var_2.shape) == 1 \
            and const_or_var_1.shape[-1] == const_or_var_2.shape[-1]:
            return const_or_var_1, const_or_var_2

    """
    UnSqueeze 1 at the beginning of const_or_var_2_shape until const_or_var_1.shape
    and const_or_var_2.shape have the same rank
    e.g.
        const_or_var_1.shape (TF)  : [1,64,128,128,3], onnx[1,3,64,128,128]
        const_or_var_2.shape (ONNX const pettern): [3,64,128,128]
        new_const_or_var_2.shape (ONNX): [1,3,64,128,128] -> [1,64,128,128,3]

        const_or_var_1.shape (TF)  : [1,64,128,128,3]
        const_or_var_2.shape (TF ver pettern): [128,128,3]
        new_const_or_var_2.shape (ONNX): [1,1,128,128,3]

        const_or_var_1.shape (TF)  : [1,128,3], onnx[1,3,128]
        const_or_var_2.shape (ONNX const pettern): [3,128]
        new_const_or_var_2.shape (ONNX): [1,3,128] -> [1,128,3]
    """
    tmp_graph_node_input_shape2 = list(const_or_var_2.shape)
    for _ in range(len(const_or_var_1.shape) - len(const_or_var_2.shape)):
        if isinstance(const_or_var_2, np.ndarray):
            const_or_var_2 = const_or_var_2[np.newaxis, ...]
        elif isinstance(const_or_var_2, tf.Tensor):
            const_or_var_2 = tf.expand_dims(
                input=const_or_var_2,
                axis=0,
            )
        elif not isinstance(const_or_var_2, np.ndarray) \
            and tf_keras.backend.is_keras_tensor(const_or_var_2):
            const_or_var_2 = tf.expand_dims(
                input=const_or_var_2,
                axis=0,
            )
        tmp_graph_node_input_shape2 = [1] + tmp_graph_node_input_shape2
    if len(tmp_graph_node_input_shape2) == len(const_or_var_1.shape):
        graph_node_input_shape2 = tmp_graph_node_input_shape2

    # Swap operands to apply transpose to correct target if needed
    # second operand is always target of transpose
    if broadcast_validity_check(list(const_or_var_1.shape), graph_node_input_shape1) and \
            not broadcast_validity_check(list(const_or_var_2.shape), graph_node_input_shape2):
        const_or_var_1, const_or_var_2 = const_or_var_2, const_or_var_1
        graph_node_input_name1, graph_node_input_name2 = graph_node_input_name2, graph_node_input_name1
        graph_node_input_shape1, graph_node_input_shape2 = graph_node_input_shape2, graph_node_input_shape1
        swapped += 1

    # Check if operands need transpose
    # CAUTION: this part may occur problem when there are more than two same numbers in tensor shape.
    #          please consider manual debugging if output is differ with onnx.
    if broadcast_validity_check(list(const_or_var_1.shape), list(const_or_var_2.shape)) and \
            broadcast_validity_check(graph_node_input_shape1, graph_node_input_shape2):
        pass
    else:
        transpose_perm = [0] + [i+2 for i in range(len(const_or_var_1.shape)-2)] + [1]

        if isinstance(const_or_var_2, np.ndarray):
            const_or_var_2: np.ndarray = const_or_var_2.transpose(transpose_perm)

        elif isinstance(const_or_var_2, tf.Tensor) \
            or (
                not isinstance(const_or_var_2, np.ndarray) \
                and tf_keras.backend.is_keras_tensor(const_or_var_2)
            ):
            if graph_node_input_name2 is not None \
                and tf_layers_dict is not None \
                and graph_node_input_name2 in tf_layers_dict \
                and tf_layers_dict[graph_node_input_name2]['optype'] == 'Input':
                const_or_var_2: np.ndarray = tf.transpose(
                    a=const_or_var_2,
                    perm=transpose_perm
                )
        else:
            pass

    # Re-swap operand if swapped in early steps to match shapes. order of operands is important for Sub and Div.
    if swapped == 1:
        const_or_var_1, const_or_var_2 = const_or_var_2, const_or_var_1

    return const_or_var_1, const_or_var_2


# https://github.com/onnx/onnx-tensorflow/blob/main/onnx_tf/common/tf_helper.py
def tf_shape(
    *,
    input_tensor: tf.Tensor,
    dtype: tf.dtypes=tf.int64,
) -> Any:
    """Helper function returning the shape of a Tensor.

    Parameters
    ----------
    input_tensor: tf.Tensor
        A Tensor

    dtype: tf.dtypes
        The output dtype (tf.int32 or tf.int64).
        Defaults: tf.int64.

    Returns
    ----------
    shape: Any
        The function will check for fully defined shape and will return numpy array or \n
        if the shape is not fully defined will use tf.shape() to return the shape as a Tensor.
    """
    if isinstance(input_tensor, np.ndarray):
        return input_tensor.shape
    if not isinstance(input_tensor, np.ndarray) and input_tensor.shape.is_fully_defined():
        return np.array(input_tensor.shape.as_list(), dtype=dtype.as_numpy_dtype)
    else:
        return tf.shape(input_tensor, out_type=dtype)


def upsampling2d_bilinear(
    input_tensor,
    new_size,
    align_corners,
    half_pixel_centers,
    name,
):
    return tf.compat.v1.image.resize_bilinear(
        images=input_tensor,
        size=new_size,
        align_corners=align_corners,
        half_pixel_centers=half_pixel_centers,
        name=name,
    )

def upsampling2d_bicubic(
    input_tensor,
    new_size,
    align_corners,
    half_pixel_centers,
    name,
):
    return tf.compat.v1.image.resize_bicubic(
        images=input_tensor,
        size=new_size,
        align_corners=align_corners,
        half_pixel_centers=half_pixel_centers,
        name=name,
    )

def upsampling2d_nearest(
    input_tensor,
    new_size,
    align_corners,
    half_pixel_centers,
    name,
):
    return tf.compat.v1.image.resize_nearest_neighbor(
        images=input_tensor,
        size=new_size,
        align_corners=align_corners,
        half_pixel_centers=half_pixel_centers,
        name=name,
    )


def upsampling3d_bilinear(
    input_tensor,
    new_size,
    align_corners,
    half_pixel_centers,
    name,
):
    d = new_size[0]
    h = new_size[1]
    w = new_size[2]
    # Dpeth (height x width)
    resized_list = []
    unstack_img_list = tf.unstack(input_tensor, axis=1)
    for img in unstack_img_list:
        resized_list.append(
            tf.compat.v1.image.resize_bilinear(
                images=img,
                size=[h, w],
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
                name=name,
            )
        )
    stack_img_hw = tf.stack(resized_list, axis=1)
    # Width (depth x Height)
    resized_list = []
    unstack_img_list = tf.unstack(stack_img_hw, axis=3)
    for img in unstack_img_list:
        resized_list.append(
            tf.compat.v1.image.resize_bilinear(
                images=img,
                size=[d, h],
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
                name=name,
            )
            )
    stack_img_dh = tf.stack(resized_list, axis=3)
    return stack_img_dh


def upsampling3d_bicubic(
    input_tensor,
    new_size,
    align_corners,
    half_pixel_centers,
    name,
):
    d = new_size[0]
    h = new_size[1]
    w = new_size[2]

    # Dpeth (height x width)
    resized_list = []
    unstack_img_list = tf.unstack(input_tensor, axis=1)
    for img in unstack_img_list:
        resized_list.append(
            tf.compat.v1.image.resize_bicubic(
                images=img,
                size=[h, w],
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
                name=name,
            )
        )
    stack_img_hw = tf.stack(resized_list, axis=1)
    # Width (depth x Height)
    resized_list = []
    unstack_img_list = tf.unstack(stack_img_hw, axis=3)
    for img in unstack_img_list:
        resized_list.append(
            tf.compat.v1.image.resize_bicubic(
                images=img,
                size=[d, h],
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
                name=name,
            )
            )
    stack_img_dh = tf.stack(resized_list, axis=3)
    return stack_img_dh


def upsampling3d_nearest(
    input_tensor,
    new_size,
    align_corners,
    half_pixel_centers,
    name,
):
    d = new_size[0]
    h = new_size[1]
    w = new_size[2]

    # Dpeth (height x width)
    resized_list = []
    unstack_img_list = tf.unstack(input_tensor, axis=1)
    for img in unstack_img_list:
        resized_list.append(
            tf.compat.v1.image.resize_nearest_neighbor(
                images=img,
                size=[h, w],
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
                name=name,
            )
        )
    stack_img_hw = tf.stack(resized_list, axis=1)
    # Width (depth x Height)
    resized_list = []
    unstack_img_list = tf.unstack(stack_img_hw, axis=3)
    for img in unstack_img_list:
        resized_list.append(
            tf.compat.v1.image.resize_nearest_neighbor(
                images=img,
                size=[d, h],
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
                name=name,
            )
        )
    stack_img_dh = tf.stack(resized_list, axis=3)
    return stack_img_dh


def _nnapi_scalar(
    value,
    dtype: tf.dtypes,
) -> Any:
    """Scalar to constant of 1D array.

    Parameters
    ----------
    value: Tensor
        Tensor to be processed

    dtype: tf.dtypes
        Tensor type

    Returns
    ----------
    tensor: Tensor
        Tensor converted from Scalar to constant of 1D array
    """
    return tf.constant(value, dtype=dtype, shape=(1,))


def alternative_argmax(
    *,
    input_tensor,
    axis: int = -1,
    output_type: tf.dtypes = tf.dtypes.float32,
    name: str = None,
    keepdims: bool = False,
    epsilon: float = None,
    replace_argmax_to_reducemax_and_indices_is_int64: bool = False,
    replace_argmax_to_reducemax_and_indices_is_float32: bool = False,
) -> Any:
    """Replace ArgMax with a ReduceMax.

    Parameters
    ----------
    input_tensor: Tensor
        Tensor to be processed

    axis: int
        The axis to reduce across
        Default: -1

    output_type: tf.dtypes
        Data type of the final OP
        Default: tf.dtypes.float32

    name: str
        OP name to be assigned to the final OP
        Default: None

    keepdims: bool
        True: Array dimensionality is preserved after ArgMax
        False: Number of array dimensions not maintained after ArgMax
        Default: False

    epsilon: float
        Very small numbers added to avoid division by zero
        Default: None

    replace_argmax_to_reducemax_and_indices_is_int64: bool
        True: Convert final output to int64
        False: Do not convert final output to int64
        Default: False

    replace_argmax_to_reducemax_and_indices_is_float32: bool
        True: Convert final output to float32
        False: Do not convert final output to float32
        Default: False

    Returns
    ----------
    pseudo_argmax: Tensor
        Converted ArgMax
    """
    safe_axis = axis
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    if safe_axis < 0:
        safe_axis = input_tensor_rank + safe_axis
    reduction_size = input_tensor.shape[axis]
    axis_max = tf.math.reduce_max(
        input_tensor,
        axis=axis,
        keepdims=True,
    )
    zero_if_max = tf.subtract(
        axis_max,
        input_tensor,
    )
    eps = epsilon if epsilon else 1e-6

    if input_tensor.dtype.is_floating:
        zero_if_max_else_eps = tf.math.minimum(
            _nnapi_scalar(eps, input_tensor.dtype),
            zero_if_max,
        )
        zero_if_max_else_one = \
            zero_if_max_else_eps * _nnapi_scalar(1 / eps, input_tensor.dtype)
    elif input_tensor.dtype.is_integer:
        zero_if_max_else_one = tf.math.minimum(
            _nnapi_scalar(1, input_tensor.dtype),
            zero_if_max,
        )
    else:
        error_msg = f''+\
            Color.RED(f'ERROR:') + ' ' +\
            f'Please specify epsilon for unknown input data type. '
        print(error_msg)
        assert False, error_msg

    zero_if_max_else_one = tf.cast(
        zero_if_max_else_one,
        dtype=output_type,
    )
    zero_if_max_else_one = zero_if_max_else_one
    one_if_max_else_zero = tf.math.subtract(
        _nnapi_scalar(1, output_type),
        zero_if_max_else_one,
    )
    rev_index = tf.range(
        reduction_size,
        0,
        -1,
        dtype=output_type,
    )
    for index in range(safe_axis + 1, len(input_tensor.shape)):
        rev_index = tf.expand_dims(
            rev_index,
            axis=index - safe_axis,
        )
    rev_index = rev_index
    rev_index_if_max_else_zero = tf.math.multiply(
        one_if_max_else_zero,
        rev_index,
    )
    reverse_argmax = tf.math.reduce_max(
        rev_index_if_max_else_zero,
        axis=axis,
        keepdims=keepdims,
    )

    if replace_argmax_to_reducemax_and_indices_is_int64:
        return tf.cast(
            tf.math.subtract(
                _nnapi_scalar(reduction_size, output_type),
                reverse_argmax,
                name=name,
            ),
            dtype=tf.dtypes.int64,
        )
    elif replace_argmax_to_reducemax_and_indices_is_float32:
        return tf.cast(
            tf.math.subtract(
                _nnapi_scalar(reduction_size, output_type),
                reverse_argmax,
                name=name,
            ),
            dtype=tf.dtypes.float32,
        )
    else:
        return tf.math.subtract(
            _nnapi_scalar(reduction_size, output_type),
            reverse_argmax,
            name=name,
        )


def alternative_fused_argmax(
    *,
    input_tensor,
    original_shape,
    axis: int = -1,
    output_type: tf.dtypes = tf.dtypes.float32,
    name: str = None,
    keepdims: bool = True,
    replace_argmax_to_fused_argmax_and_indices_is_int64: bool = False,
    replace_argmax_to_fused_argmax_and_indices_is_float32: bool = False,
) -> Any:
    """Replace ArgMax with a ReduceMax.

    Parameters
    ----------
    input_tensor: Tensor
        Tensor to be processed

    original_shape: list
        Input shape of ONNX graph before machining

    axis: int
        The axis to reduce across
        Default: -1

    output_type: tf.dtypes
        Data type of the final OP
        Default: tf.dtypes.float32

    name: str
        OP name to be assigned to the final OP
        Default: None

    keepdims: bool
        True: Array dimensionality is preserved after ArgMax
        False: Number of array dimensions not maintained after ArgMax
        Default: True

    replace_argmax_to_fused_argmax_and_indices_is_int64: bool
        True: Convert final output to int64
        False: Do not convert final output to int64
        Default: False

    replace_argmax_to_fused_argmax_and_indices_is_float32: bool
        True: Convert final output to float32
        False: Do not convert final output to float32
        Default: False

    Returns
    ----------
    pseudo_fused_argmax: Tensor
        Converted ArgMax
    """
    safe_axis = axis
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    final_tensor = None

    if safe_axis < 0:
        safe_axis = input_tensor_rank + safe_axis

    # Currently, only 4D tensors are supported
    if input_tensor_rank != 4:
        # Not 4D Tensor
        argmaxed_tensor = tf.math.argmax(
            input=input_tensor,
            axis=axis,
            output_type=output_type,
            name=f'{name}_fused_argmax',
        )
        if keepdims:
            final_tensor = \
                tf.expand_dims(
                    input=argmaxed_tensor,
                    axis=axis,
                    name=f'{name}_expand_dims',
                )
        else:
            final_tensor = argmaxed_tensor
        return final_tensor

    else:
        # 4D Tensor
        input_height, input_width = original_shape[2], original_shape[3]
        align_corners = True
        half_pixel_centers = False
        argmaxed_tensor = tf.math.argmax(
            input=input_tensor,
            axis=axis,
            output_type=output_type,
            name=f'{name}_fused_argmax',
        )
        expanded_tensor = \
            tf.expand_dims(
                input=argmaxed_tensor,
                axis=axis,
                name=f'{name}_expand_dims',
            )
        expanded_tensor_dtype = expanded_tensor.dtype
        casted_tensor = tf.cast(
            x=expanded_tensor,
            dtype=tf.float32,
        )
        align_corners = True
        half_pixel_centers = False
        upscaled_tensor = Lambda(
            upsampling2d_nearest,
            arguments={
                'new_size': np.asarray([input_height, input_width], dtype=np.int32),
                'align_corners': align_corners,
                'half_pixel_centers': half_pixel_centers,
                'name': f'{name}_resize_nearest',
            }
        )(casted_tensor)
        recasted_tensor = tf.cast(upscaled_tensor, dtype=expanded_tensor_dtype)
        if keepdims:
            final_tensor = recasted_tensor
        else:
            final_tensor = \
                tf.squeeze(
                    input=recasted_tensor,
                    axis=axis,
                    name=f'{name}_squeeze',
                )
        if replace_argmax_to_fused_argmax_and_indices_is_int64:
            final_tensor = tf.cast(
                x=final_tensor,
                dtype=tf.int64,
                name=f'{name}_cast',
            )
        elif replace_argmax_to_fused_argmax_and_indices_is_float32:
            final_tensor = tf.cast(
                x=final_tensor,
                dtype=tf.float32,
                name=f'{name}_cast',
            )
        return final_tensor


# https://zenn.dev/pinto0309/articles/8f6df1d2304395
def alternative_asin(
    *,
    input_tensor,
) -> Any:
    """Replace Asin with a pseudo_Asin.

    Parameters
    ----------
    input_tensor: Tensor
        Tensor to be processed

    Returns
    ----------
    pseudo_asin: Tensor
        Converted Asin
    """
    x_abs = None
    x_abs = tf.abs(input_tensor)
    x_dtype = input_tensor.dtype
    neg = tf.math.divide(
        tf.math.multiply(
            tf.minimum(input_tensor, tf.convert_to_tensor(0.0, dtype=x_dtype)),
            tf.convert_to_tensor(-1.0, dtype=x_dtype)
        ),
        x_abs
    )
    x = x_abs
    y = tf.constant(-0.0187293)
    y = tf.math.multiply(y, x)
    y = tf.math.add(y, 0.0742610)
    y = tf.math.multiply(y, x)
    y = tf.math.subtract(y, 0.2121144)
    y = tf.math.multiply(y, x)
    y = tf.math.add(y, 1.5707288)
    y = tf.math.subtract(
        tf.math.multiply(3.14159265358979, 0.5),
        tf.math.multiply(
            tf.sqrt(tf.math.subtract(tf.convert_to_tensor(1.0, dtype=x.dtype), x)),
            y
        )
    )
    pseudo_asin = tf.math.subtract(
        y,
        tf.math.multiply(
            tf.math.multiply(tf.convert_to_tensor(2.0, dtype=neg.dtype), neg),
            y
        )
    )
    return pseudo_asin


# https://zenn.dev/pinto0309/articles/8f6df1d2304395
def alternative_acos(
    *,
    input_tensor,
) -> Any:
    """Replace Acos with a pseudo_Acos.

    Parameters
    ----------
    input_tensor: Tensor
        Tensor to be processed

    Returns
    ----------
    pseudo_acos: Tensor
        Converted Acos
    """
    x_abs = None
    x_abs = tf.abs(input_tensor)
    x_dtype = input_tensor.dtype
    neg = tf.math.divide(
        tf.math.multiply(
            tf.minimum(input_tensor, tf.convert_to_tensor(0.0, dtype=x_dtype)),
            tf.convert_to_tensor(-1.0, dtype=x_dtype)
        ),
        x_abs
    )
    x = x_abs
    y = tf.constant(-0.0187293)
    y = tf.math.multiply(y, x)
    y = tf.math.add(y, 0.0742610)
    y = tf.math.multiply(y, x)
    y = tf.math.subtract(y, 0.2121144)
    y = tf.math.multiply(y, x)
    y = tf.math.add(y, 1.5707288)
    y = tf.math.multiply(
        y,
        tf.sqrt(tf.math.subtract(tf.convert_to_tensor(1.0, dtype=x.dtype), x))
    )
    y = tf.math.multiply(
        y,
        tf.math.subtract(
            tf.convert_to_tensor(1.0, dtype=neg.dtype),
            tf.math.multiply(tf.convert_to_tensor(2.0, dtype=neg.dtype), neg),
        )
    )
    pseudo_acos = tf.math.add(
        tf.math.multiply(
            neg,
            3.14159265358979
        ),
        y
    )
    return pseudo_acos


# https://developer.download.nvidia.com/cg/atan2.html
def alternative_atan2(
    *,
    input_tensor_y,
    input_tensor_x,
) -> Any:
    """Replace Atan2 with a pseudo_Atan2.

    Parameters
    ----------
    input_tensor_y: Tensor
        Tensor to be processed.
        Vector or scalar for numerator of ratio of which to determine the arctangent.

    input_tensor_x: Tensor
        Tensor to be processed.
        Vector or scalar of denominator of ratio of which to determine the arctangent.

    Returns
    ----------
    pseudo_atan2: Tensor
        Converted Atan2
    """
    pseudo_atan2 = tf.math.atan2(
        y=input_tensor_y,
        x=input_tensor_x,
    )
    return pseudo_atan2


# https://developer.download.nvidia.com/cg/atan.html
def alternative_atan(
    *,
    input_tensor,
) -> Any:
    """Replace Atan with a pseudo_Atan.

    Parameters
    ----------
    input_tensor_x: Tensor
        Tensor to be processed.
        Vector or scalar of which to determine the arctangent.

    Returns
    ----------
    pseudo_atan: Tensor
        Converted Atan
    """
    return alternative_atan2(
        input_tensor_y=input_tensor,
        input_tensor_x=tf.broadcast_to(
            tf.convert_to_tensor(1.0, dtype=input_tensor.dtype),
            shape=input_tensor.shape
        ),
    )


# https://github.com/onnx/onnx-tensorflow/blob/main/onnx_tf/common/pooling_helper.py
pad_ops = namedtuple(
    "pad_ops",
    ["max_op", "ceil_op", "floor_op", "cast_int_op"]
)
pad_numpy_ops = pad_ops(
    np.maximum,
    np.ceil,
    np.floor,
    lambda arr: arr.astype(np.int64)
)
pad_tf_ops = pad_ops(
    tf.maximum,
    tf.math.ceil,
    tf.math.floor,
    lambda tensor: tf.cast(tensor, tf.int64)
)

def _calc_pads_same_pooling(
    *,
    in_spatial_shape,
    kernel_shape,
    strides,
    dilations,
    padding,
    padding_ops=pad_numpy_ops,
    pads_order=1
) -> List[int]:
    """Calculates the SAME paddings that need to be added to the input.

    Parameters
    ----------
    in_spatial_shape:
        input spatial shape

    kernel_shape:
        the size of the kernel along each axis

    strides:
        stride along each spatial axis

    dilations:
        dilations value along each spatial axis

    padding:
        padding to calculate: SAME_UPPER orSAME_LOWER

    padding_ops:
        namedtuple with ops to be used during calculations.\n
        there are two sets of ops defined pad_numpy_ops and pad_tf_ops with numpy and tensorflow ops

    pads_order:
        order of returned pads.\n
        possible options are:\n
            1 - b1, b2, ..., bn, e1, e2, ..., en\n
            2 - b1, e1, b2, e2, ..., bn, en\n
        where n = len(kernel_shape) * 2, b1, b2, ..., bn\n
        define pads at the begging of axis e1, e2, ..., en define pads at the end of axis

    Returns
    ----------
    pads:
        array with calculated pads. the order of the values is determined by `pads_order`
    """
    spatial_size = len(kernel_shape)
    pads = [0] * (spatial_size * 2)
    for i in range(spatial_size):
        in_size = in_spatial_shape[i]
        filter_size = (kernel_shape[i] - 1) * dilations[i] + 1

        out_size = padding_ops.ceil_op(in_size / strides[i])
        out_size = padding_ops.cast_int_op(out_size)
        pad_along_axis = \
            padding_ops.max_op((out_size - 1) * strides[i] + filter_size - in_size, 0)
        if padding.lower() == "same_lower":
            pad_op = padding_ops.ceil_op
        else:
            pad_op = padding_ops.floor_op
        pad_begin = pad_op(pad_along_axis / 2)

        pad_begin = padding_ops.cast_int_op(pad_begin)
        pad_along_axis = padding_ops.cast_int_op(pad_along_axis)

        pad_end = pad_along_axis - pad_begin

        pads[i * pads_order] = pad_begin
        pads[i * pads_order + (spatial_size if pads_order == 1 else 1)] = pad_end

    return pads


def calc_pads_explicit_pooling(
    *,
    padding,
    spatial_size,
):
    """
    Calculate explicit padding
    """
    assert type(padding) is list

    pads = []
    for i in range(spatial_size):
        pads += [padding[i], padding[i + spatial_size]]
    return pads


def calc_pads_ceil_mode_pooling(
    *,
    in_spatial_shape,
    spatial_size,
    kernel_shape,
    dilations,
    strides,
    is_known_shape,
):
    """
    Calculate padding in ceil_mode
    """
    pads = []
    for i in range(spatial_size):
        dim_size = in_spatial_shape[i]
        filter_size = (kernel_shape[i] - 1) * dilations[i] + 1
        out_size = (dim_size - filter_size) / strides[i]
        if is_known_shape:
            pad_size = (
                np.ceil(out_size) - np.floor(out_size)
            ).astype(np.int64)
        else:
            pad_size = tf.cast(
                tf.math.ceil(out_size) - tf.math.floor(out_size),
                tf.int64,
            )

        pads += [0, pad_size * strides[i]]
    return pads


def calc_pads_same_pooling(
    *,
    kernel_shape,
    strides,
    dilations,
    padding,
    in_spatial_shape,
    is_known_shape,
):
    """
    Calculate SAME_* paddings.
    """
    pad_ops = pad_numpy_ops if is_known_shape else pad_tf_ops

    return _calc_pads_same_pooling(
        in_spatial_shape=in_spatial_shape,
        kernel_shape=kernel_shape,
        strides=strides,
        dilations=dilations,
        padding=padding,
        padding_ops=pad_ops,
        pads_order=2,
    )


def calc_pads_pooling(
    *,
    kernel_shape,
    strides,
    dilations,
    padding,
    is_known_shape,
    spatial_size,
    in_spatial_shape,
    ceil_mode,
):
    if is_known_shape:
        pads = np.zeros([spatial_size * 2], np.int64)
    else:
        pads = tf.zeros([spatial_size * 2], tf.int64)

    # check for explicit padding
    if type(padding) is list:
        pads += calc_pads_explicit_pooling(
            padding=padding,
            spatial_size=spatial_size,
        )
    elif padding.lower().startswith("same"):
        pads += calc_pads_same_pooling(
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
            padding=padding,
            in_spatial_shape=in_spatial_shape,
            is_known_shape=is_known_shape,
        )

    # when padding is set to SAME, ceil_mode will not do anything
    # because output sizes will be multiple of the strides
    if ceil_mode and (type(padding) is list or not padding.lower().startswith("same")):
        new_spatial_shape = [
            in_spatial_shape[i] + pads[i * 2] + pads[i * 2 + 1]
            for i in range(spatial_size)
        ]
        pads += calc_pads_ceil_mode_pooling(
            in_spatial_shape=new_spatial_shape,
            spatial_size=spatial_size,
            kernel_shape=kernel_shape,
            dilations=dilations,
            strides=strides,
            is_known_shape=is_known_shape,
        )
    return pads


def pad_input(
    *,
    input_tensor,
    is_known_shape,
    kernel_shape,
    ceil_mode,
    spatial_size,
    strides,
    dilations,
    padding,
    padding_constant,
):
    """
    Pad the input according to the parameters
    """
    # check if we need to do any padding at all
    if not ceil_mode \
        and ((type(padding) is list and padding == [0] * spatial_size * 2) or padding == "VALID"):
        return input_tensor

    # in_spatial_shape = self.input_shape[2:]
    input_shape = tf_shape(
        input_tensor=input_tensor,
    )
    in_spatial_shape = input_shape[1:len(kernel_shape)+1]
    pads = calc_pads_pooling(
        kernel_shape=kernel_shape,
        strides=strides,
        dilations=dilations,
        padding=padding,
        is_known_shape=is_known_shape,
        spatial_size=spatial_size,
        in_spatial_shape=in_spatial_shape,
        ceil_mode=ceil_mode,
    )

    if is_known_shape and np.count_nonzero(pads) == 0:
        return input_tensor

    # no padding on the NC dimensions
    tf_paddings = [[0, 0]]
    # padding for the (D)HW dimensions
    for i in range(spatial_size):
        tf_paddings += [[pads[i * 2], pads[i * 2 + 1]]]
    tf_paddings += [[0, 0]]

    padded_tensor = tf.pad(
        input_tensor,
        tf_paddings,
        mode='CONSTANT',
        constant_values=padding_constant,
    )
    return padded_tensor


def get_padding_as_op(
    *,
    x,
    pads,
):
    num_dim = int(len(pads) / 2)
    tf_pads = np.transpose(np.array(pads).reshape([2, num_dim]))
    # tf_pads = [0, 0, 0, 0] + tf_pads.flatten().tolist()
    tf_pads = [0, 0] + tf_pads.flatten().tolist() + [0, 0]
    padding = tf.constant(
        np.array(tf_pads).reshape([num_dim + 2, 2]).astype(np.int32)
    )  # tf requires int32 paddings
    return tf.pad(x, padding)


def tf_product(
    *,
    a,
    b,
):
    """
            Calculates the cartesian product of two column vectors a and b
            Example:
            a = [[1]
                [2]
                [3]]
            b = [[0]
                [1]]
            result = [[1 0]
                    [1 1]
                    [2 0]
                    [2 1]
                    [3 0]
                    [3 1]]
    """
    tile_a = tf.tile(a, [1, tf.shape(b)[0]])
    tile_a = tf.expand_dims(tile_a, 2)
    tile_a = tf.reshape(tile_a, [-1, 1])
    b = tf.tile(b, [tf.shape(a)[0], 1])
    b = tf.concat([tile_a, b], axis=1)
    return b


def _calc_input_ind(
    *,
    output_ind,
    kernel,
    dilation,
    stride
):
    return \
        (output_ind // kernel) * \
        (stride - kernel * dilation) + \
        output_ind * dilation


def remove_dilations(
    *,
    input_tensor,
    kernel_shape,
    spatial_size,
    strides,
    dilations,
):
    input_shape = tf_shape(input_tensor=input_tensor)
    in_spatial_shape = input_shape[1:len(kernel_shape)+1]
    channels_count = input_shape[-1]

    # initilize the output_shape with zeros
    # self.output_shape will contain the shape of the
    # output tensor after the loop below is executed
    output_shape = [0] * (spatial_size + 2)
    output_shape[0] = input_shape[0]

    for dim in range(spatial_size - 1, -1, -1):
        filter_size = (kernel_shape[dim] - 1) * dilations[dim] + 1
        output_size = (
            ((in_spatial_shape[dim] - filter_size) // strides[dim]) + 1
        ) * kernel_shape[dim]
        output_shape[dim + 1] = output_size

        # initialize the output dimension index with the range of the
        # dimension output size (e.g. 4): [0, 1, 2, 3]
        dim_ind = tf.range(output_size)

        # calculate the matching indices in the input data
        # [0, 1, 2, 3] will calculate to [0, 2, 1, 3]
        # from the above example
        dim_ind = _calc_input_ind(
            output_ind=dim_ind,
            kernel=kernel_shape[dim],
            dilation=dilations[dim],
            stride=strides[dim],
        )
        # convert to column vector
        dim_ind = tf.expand_dims(dim_ind, 1)

        if dim == spatial_size - 1:
            gather_ind = dim_ind
        else:
            # "combine" current dimension indices with the previous dimensions
            # using cartesian product
            gather_ind = tf_product(
                a=dim_ind,
                b=gather_ind,
            )

    # The result from the above loop for 2D data will be:
    # [[y1, x1], [y2, x2], ..., [yn, xm]] where n is the height,
    # m is the width.

    # set the channels count in the output_shape
    output_shape[-1] = channels_count
    # create the channel indices
    channel_ind = tf.range(channels_count, dtype=tf.int64)
    # convert to column vector
    channel_ind = tf.expand_dims(channel_ind, 1)
    # "combine" channel indices with the result from the loop
    gather_ind = tf_product(
        a=gather_ind,
        b=channel_ind,
    )

    # expand the dimensions to match the input dimensions + 1
    for x in range(spatial_size):
        gather_ind = tf.expand_dims(gather_ind, 0)
    # dublicate the indices for every batch
    gather_ind = tf.tile(
        gather_ind,
        [input_shape[0]] + [1] * (spatial_size + 1),
    )

    # extract the selected values from the input
    output = tf.gather_nd(input_tensor, gather_ind, batch_dims=1)
    # reshape the output to the correct shape calculated earlier
    output = tf.reshape(output, output_shape)

    return output


def process_neg_idx(
    *,
    data,
    indices,
    batch_dims=0,
):
    """ Convert all the negative indices to positive
    GatherND and ScatterND/TensorScatterNDUpdate in Tensorflow
    doesn't support negative indices. Therefore need to run this
    function to convert all the negative indices to positive before
    send it to Tensorflow.
    """
    data_shape = data.shape
    if None not in data_shape:
        indices_shape = indices.shape
    else:
        indices_shape = tf_shape(input_tensor=indices)
    if batch_dims > 0:
        max_i = tf.cast(
            data_shape[batch_dims:indices_shape[-1] + batch_dims],
            indices.dtype,
        )
    else:
        if not isinstance(indices_shape[-1], int) \
            and not isinstance(indices_shape[-1], np.ndarray) \
            and not isinstance(indices_shape[-1], tf.Tensor) \
            and tf_keras.backend.is_keras_tensor(indices_shape[-1]):
            if tf.TensorShape(None) not in data_shape :
                max_i = tf.cast(
                    tf.strided_slice(
                        input_=data_shape,
                        begin=0,
                        end=indices_shape[-1],
                        begin_mask=1,
                    ),
                    indices.dtype,
                )
            else:
                return indices
        else:
            max_i = tf.cast(
                data_shape[:indices_shape[-1]],
                indices.dtype,
            )
    return tf.math.floormod(tf.add(indices, max_i), max_i)


def process_neg_idx_along_axis(
    *,
    data,
    axis,
    indices,
):
    """ Convert all the negative indices to positive
    ScatterND/TensorScatterNDUpdate in Tensorflow doesn't support
    negative indices. Therefore need to run this function to convert
    all the negative indices to positive before send it to Tensorflow.
    """
    data_shape = tf_shape(input_tensor=data)
    max_i = tf.cast(data_shape[axis], indices.dtype)
    return tf.math.floormod(tf.add(indices, max_i), max_i)


def is_integer_num(
    *,
    x: Any,
) -> bool:
    """Determines whether an integer or not.

    Parameters
    ----------
    x: Any

    Returns
    ----------
    Result: bool
        True: integer
        False: non-integer
    """
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return x.is_integer()
    elif isinstance(x, np.ndarray) \
        and x.dtype in [np.int8, np.int16, np.int32, np.int64]:
        return True
    elif isinstance(x, np.ndarray) \
        and x.squeeze().ndim == 0 and int(x) == x:
        return True
    return False


def disable_unnecessary_transpose(
    *,
    graph_node_input_1: Any,
    graph_node_input_2: Any,
    input_tensor_1: Any,
    input_tensor_2: Any,
    **kwargs: Dict,
) -> Tuple[Any, Any, Any, Any]:
    """Remove unnecessary Transpose to NHWC.

    Parameters
    ----------
    graph_node_input_1: Any
        Input Node X of ONNX

    graph_node_input_2: Any
        Input Node Y of ONNX

    input_tensor_1: Any
        Input Node X of TensorFlow

    input_tensor_2: Any
        Input Node Y of TensorFlow

    Returns
    ----------
    graph_node_input_1: Any
        Input Node X of ONNX

    graph_node_input_2: Any
        Input Node Y of ONNX

    input_tensor_1: Any
        Input shape-corrected TensorFlow input node X

    input_tensor_2: Any
        Input shape-corrected TensorFlow input node Y
    """
    if isinstance(graph_node_input_1, gs.Variable) \
        and isinstance(graph_node_input_2, gs.Variable):

        # Skip special processing if the operation does not result
        # in an error even if special processing is not performed
        try:
            _ = input_tensor_1 * input_tensor_2
            return graph_node_input_1, graph_node_input_2, input_tensor_1, input_tensor_2
        except Exception as ex:
            pass

        node_x_op_type = graph_node_input_1.inputs[0].op \
            if len(graph_node_input_1.inputs) > 0 else 'Input'
        node_y_op_type = graph_node_input_2.inputs[0].op \
            if len(graph_node_input_2.inputs) > 0 else 'Input'

        if ((node_x_op_type == 'Transpose' and not node_y_op_type == 'Transpose') \
            or (not node_x_op_type == 'Transpose' and node_y_op_type == 'Transpose')) \
            and graph_node_input_1.shape is not None \
            and graph_node_input_2.shape is not None \
            and (len(graph_node_input_1.shape) == len(graph_node_input_2.shape)):

            if (node_x_op_type == 'Transpose' and not node_y_op_type == 'Transpose'):
                input_tensor_1, input_tensor_2 = input_tensor_2, input_tensor_1
                graph_node_input_1, graph_node_input_2 = graph_node_input_2, graph_node_input_1

            node_y_perm: list = graph_node_input_2.inputs[0].attrs['perm']
            input_tensor_1_shape = [
                dim if isinstance(dim, int) else None for dim in input_tensor_1.shape
            ]
            input_tensor_2_shape = [
                dim if isinstance(dim, int) else None for dim in input_tensor_2.shape
            ]
            tensor_rank = len(input_tensor_1_shape)
            perm = [
                convert_axis(
                    axis=idx,
                    tensor_rank=tensor_rank,
                    before_op_output_shape_trans=True,
                ) for idx in range(tensor_rank)
            ]
            reverse_perm = [
                convert_reverse_axis(
                    axis=idx,
                    tensor_rank=tensor_rank,
                    before_op_output_shape_trans=True,
                ) for idx in range(tensor_rank)
            ]
            if node_y_perm == perm:
                unmatch = False
                for dim1, dim2 in zip(input_tensor_1_shape, input_tensor_2_shape):
                    if isinstance(dim1, int) and dim1 != 1 and isinstance(dim2, int) and dim2 != 1:
                        if dim1 != dim2:
                            unmatch = True
                            break
                if unmatch:
                    input_tensor_2 = transpose_with_flexing_deterrence(
                        input_tensor=input_tensor_2,
                        perm=reverse_perm,
                        **kwargs,
                    )
    return graph_node_input_1, graph_node_input_2, input_tensor_1, input_tensor_2


def shape_unmatched_special_avoidance_workaround(
    *,
    graph_node_input_1: Any,
    graph_node_input_2: Any,
    input_tensor_1: Any,
    input_tensor_2: Any,
    tf_layers_dict: Dict,
    **kwargs: Dict,
) -> Tuple[Any, Any]:
    """Force correction of the shape mismatch between input X and input Y to NHWC format
    only if the output of the immediately preceding OP is definitively NHWC.

    Parameters
    ----------
    graph_node_input_1: Any
        Input Node X of ONNX

    graph_node_input_2: Any
        Input Node Y of ONNX

    input_tensor_1: Any
        Input Node X of TensorFlow

    input_tensor_2: Any
        Input Node Y of TensorFlow

    Returns
    ----------
    input_tensor_1: Any
        Input shape-corrected TensorFlow input node X

    input_tensor_2: Any
        Input shape-corrected TensorFlow input node Y
    """
    try:
        if hasattr(input_tensor_1, "shape") \
            and hasattr(input_tensor_2, "shape") \
            and input_tensor_1.shape is not None \
            and input_tensor_2.shape is not None \
            and input_tensor_1.shape == input_tensor_2.shape:
            return input_tensor_1, input_tensor_2
    except:
        pass
    # At least one True value for same_input_shape_as_onnx
    # At least one True value in nhwc_flags
    # same_input_shape_as_onnx == True and nhwc_flags == False and 3D or 4D or 5D tensor is NHWC transposed
    nhwc_flag_1 = False
    same_input_shape_as_onnx_1 = False
    if isinstance(graph_node_input_1, gs.Variable):
        nhwc_flag_1 = tf_layers_dict[graph_node_input_1.name]['nhwc'] \
            if 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False
        if graph_node_input_1.shape is not None:
            graph_node_input_1_shape = [
                dim if not isinstance(dim, str) else None for dim in graph_node_input_1.shape
            ]
        else:
            graph_node_input_1_shape = []
        same_input_shape_as_onnx_1 = True if len(graph_node_input_1_shape) > 0 \
            and graph_node_input_1_shape == input_tensor_1.shape else False
    else:
        nhwc_flag_1 = False
        if graph_node_input_1.shape is not None:
            graph_node_input_1_shape = [
                dim if not isinstance(dim, str) else None for dim in graph_node_input_1.shape
            ]
        else:
            graph_node_input_1_shape = []
        same_input_shape_as_onnx_1 = True if len(graph_node_input_1_shape) > 0 \
            and graph_node_input_1_shape == input_tensor_1.shape else False
    nhwc_flag_2 = False
    same_input_shape_as_onnx_2 = False
    if isinstance(graph_node_input_2, gs.Variable):
        nhwc_flag_2 = tf_layers_dict[graph_node_input_2.name]['nhwc'] \
            if 'nhwc' in tf_layers_dict[graph_node_input_2.name].keys() else False
        if graph_node_input_2.shape is not None:
            graph_node_input_2_shape = [
                dim if not isinstance(dim, str) else None for dim in graph_node_input_2.shape
            ]
        else:
            graph_node_input_2_shape = []
        same_input_shape_as_onnx_2 = True if len(graph_node_input_2_shape) > 0 \
            and graph_node_input_2_shape == input_tensor_2.shape else False
    else:
        nhwc_flag_2 = False
        if graph_node_input_2.shape is not None:
            graph_node_input_2_shape = [
                dim if not isinstance(dim, str) else None for dim in graph_node_input_2.shape
            ]
        else:
            graph_node_input_2_shape = []
        same_input_shape_as_onnx_2 = True if len(graph_node_input_2_shape) > 0 \
            and graph_node_input_2_shape == input_tensor_2.shape else False

    same_input_shape_as_onnxs = [same_input_shape_as_onnx_1, same_input_shape_as_onnx_2]
    nhwc_flags = [nhwc_flag_1, nhwc_flag_2]
    if True in same_input_shape_as_onnxs and True in nhwc_flags:
        values = [input_tensor_1, input_tensor_2]
        for idx, (same_input_shape_as_onnx, nhwc_flag) in enumerate(zip(same_input_shape_as_onnxs, nhwc_flags)):
            if same_input_shape_as_onnx and not nhwc_flag:
                if len(values[idx].shape) == 3:
                    values[idx] = \
                        transpose_with_flexing_deterrence(
                            input_tensor=values[idx],
                            perm=[0,2,1],
                            **kwargs,
                        )
                elif len(values[idx].shape) == 4:
                    values[idx] = \
                        transpose_with_flexing_deterrence(
                            input_tensor=values[idx],
                            perm=[0,2,3,1],
                            **kwargs,
                        )
                elif len(values[idx].shape) == 5:
                    values[idx] = \
                        transpose_with_flexing_deterrence(
                            input_tensor=values[idx],
                            perm=[0,2,3,4,1],
                            **kwargs,
                        )

        # Transpose until the nhwc flag matches the shape toward True
        #   1. Either one of the nhwc flags is True and either one is False
        #   2. len(A.shape) == len(B.shape)
        #   3. None not in A.shape
        #   4. None not in B.shape
        #   5. No two or more identical values exist in the A and B shapes.
        #   6. Number of shape matches < len(A.shape)-1
        if True in nhwc_flags and False in nhwc_flags:
            input_tensor_1_shape = input_tensor_1.shape
            input_tensor_2_shape = input_tensor_2.shape

            if len(input_tensor_1_shape) == len(input_tensor_2_shape) \
                and None not in input_tensor_1_shape \
                and None not in input_tensor_2_shape:

                input_tensor_1_shape_wo_one = [dim for dim in input_tensor_1.shape if dim != 1]
                input_tensor_2_shape_wo_one = [dim for dim in input_tensor_2.shape if dim != 1]

                if  len(input_tensor_1_shape_wo_one) == len(set(input_tensor_1_shape_wo_one)) \
                    and len(input_tensor_2_shape_wo_one) == len(set(input_tensor_2_shape_wo_one)) \
                    and sum(1 if s1 == s2 else 0 for s1, s2 in zip(input_tensor_1_shape, input_tensor_2_shape)) < len(input_tensor_1_shape) - 1:

                    false_indices = nhwc_flags.index(False)
                    no_transpose_target_tensor = values[1 - false_indices]
                    transpose_target_tensor = values[false_indices]
                    seq = [i for i in range(len(transpose_target_tensor.shape))]
                    perms = list(itertools.permutations(seq))
                    for perm in perms:
                        try:
                            tmp_trans_value = \
                                transpose_with_flexing_deterrence(
                                    input_tensor=transpose_target_tensor,
                                    perm=perm,
                                    **kwargs,
                                )
                            dummy_mul = no_transpose_target_tensor * tmp_trans_value
                            values[false_indices] = tmp_trans_value
                            break
                        except:
                            pass

        input_tensor_1 = values[0]
        input_tensor_2 = values[1]

    return input_tensor_1, input_tensor_2


def calc_output_shape_conv_transpose(
    *,
    input_shape: List[Any],
    kernel: List[int],
    pad_mode: str,
    output_padding: List[int],
    stride: List[int],
    dilation: List[int],
) -> List[int]:
    """Calculation of ConvTranspose output geometry.

    Parameters
    ----------
    input_shape: List[Any]
        INPUT Node Shape

    kernel: List[int]
        kernel size

    pad_mode: str
        pad mode. "valid" or "same"

    output_padding: List[int]
        output paddings

    stride: List[int]
        strides

    dilation: List[int]
        dilations

    Returns
    ----------
    output_shape: List[int]
        Accurately calculated ConvTranspose output shape
    """
    assert len(input_shape) == len(kernel) == len(output_padding) == len(stride) == len(dilation),\
        "All parameters should have same length"

    output_shape = []

    for i, k, p, s, d in zip(input_shape, kernel, output_padding, stride, dilation):
        output_shape.append(
            conv_utils.deconv_output_length(
                input_length=i,
                filter_size=k,
                padding=pad_mode.lower(),
                output_padding=p,
                stride=s,
                dilation=d,
            )
        )

    return output_shape


def replace_max_values_negative_values(
    *,
    input_tensor_shape: np.asarray,
    index_list: np.asarray,
    axes: np.asarray,
) -> List[int]:
    """Replacement of maximum index value and negative index value for ONNX.
    For Slice OP.

    Parameters
    ----------
    input_tensor_shape: np.asarray
        INPUT Node Shape

    index_list: np.asarray
        Index list of starts or ends of Slice OP

    axes: np.asarray
        Slice OP axes

    Returns
    ----------
    index_list: List[int]
        List of ONNX maximum index values and negative index values replaced
        with acceptable positive integers
    """
    if axes is None:
        return index_list

    for axis in axes:
        data_shape_length = input_tensor_shape[axis]
        if data_shape_length is None:
            continue

        # Max Value
        """
        9223372036854775807 = -1
        9223372036854775806 = -2
        9223372036854775805 = -3
        9223372036854775804 = -4
        9223372036854775803 = -5
        """
        maxvalue_index_list = [
            ONNX_INF_INDEX_VALUE - i \
                for i in range(data_shape_length)
        ]
        maxvalue_substitution_index_list = [
            i - ONNX_INF_INDEX_VALUE + data_shape_length \
                for i in maxvalue_index_list
        ]
        """
        maxvalue_index_dict
            9223372036854775807: 4
            9223372036854775806: 3
            9223372036854775805: 2
            9223372036854775804: 1
            9223372036854775803: 0
        """
        maxvalue_index_dict = {
            i: j for i,j in zip(maxvalue_index_list, maxvalue_substitution_index_list)
        }
        # Negative Value
        negativevalue_substitution_index_list = [
            -i - 1 for i in range(data_shape_length)
        ]
        """
        negativevalue_index_dict
            -1: 4
            -2: 3
            -3: 2
            -4: 1
            -5: 0
        """
        negativevalue_index_dict = {
            i: i+data_shape_length for i in negativevalue_substitution_index_list
        }

        # replace max values
        index_list[axis] = index_list[axis] \
            if index_list[axis] not in maxvalue_index_dict.keys() \
                else maxvalue_index_dict[index_list[axis]]

        # replace negative values
        index_list[axis] = index_list[axis] \
            if index_list[axis] not in negativevalue_index_dict.keys() \
                else negativevalue_index_dict[index_list[axis]]
    return index_list


# https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0
# Transpose v5->v6, 5D->6D
def transpose_with_flexing_deterrence(
    *,
    input_tensor: Any,
    perm: List[int],
    output_shape: List[int] = None,
    name: str = None,
    **kwargs: Dict,
) -> Any:
    """Transpose tensors of 7 or more dimensions while suppressing the transformation to FlexTranspose.
    Suppress FlexTranspose generation only if the enable_suppression_flextranspose option is enabled when the tool is started.

    Parameters
    ----------
    input_tensor: Any
        Tensor to be transposed

    perm: List[int]
        inverted perm

    output_shape: List[int]
        Shape of tensor after transposition.
        The shape of the tensor in TensorFlow format after transposition must be specified.
        This value may produce the most optimized Transpose with Special Transpose.1 applied.
        If this value is not specified, the redundant Special Transpose.2 is applied.

    name: str
        graph_node.name

    Returns
    ----------
    tensor_after_transposition: Any
        Tensor after transposition
    """
    disable_suppression_flextranspose: bool = \
        kwargs['disable_suppression_flextranspose']
    number_of_dimensions_after_flextranspose_compression: int = \
        kwargs['number_of_dimensions_after_flextranspose_compression']
    COMPRESSION_DEFAULT_VALUE = 6

    tensor_after_transposition = input_tensor

    # If no transposition is necessary, skip all processing.
    if perm is not None \
        and (isinstance(perm, list) or isinstance(perm, Tuple) or isinstance(perm, np.ndarray)) \
        and list(perm) == list(range(len(perm))):
        return tensor_after_transposition

    elif perm is not None \
        and not (isinstance(perm, list) or isinstance(perm, Tuple) or isinstance(perm, np.ndarray)) \
        and tf_keras.backend.is_keras_tensor(perm) \
        and hasattr(perm, '_inferred_value') \
        and isinstance(perm._inferred_value, list) \
        and perm._inferred_value == list(range(len(perm._inferred_value))):
        return tensor_after_transposition

    if disable_suppression_flextranspose:
        # Normal Transpose
        tensor_after_transposition = tf.transpose(
            a=input_tensor,
            perm=perm,
            name=name,
        )
    else:
        # Special Transpose
        # https://zenn.dev/pinto0309/scraps/cfb59856ac0453
        # Get dimension with 1 element
        input_tensor_shape: List[int] = input_tensor.shape
        input_tensor_rank = len(input_tensor_shape)
        x_shape_one_dims = [
            idx for idx in range(len(input_tensor_shape)) \
                if isinstance(input_tensor_shape[idx], int) and input_tensor_shape[idx]==1
        ]
        x_shape_none_dims_count = len(
            [dim for dim in input_tensor_shape if not isinstance(dim, int) or dim < 1]
        )
        # Delete dimension with 1 element
        squeezed_original_x = tf.squeeze(input_tensor, x_shape_one_dims)
        # Obtain a shape with the dimension with 1 element removed
        squeezed_original_shapes = squeezed_original_x.shape

        if input_tensor_rank >= (COMPRESSION_DEFAULT_VALUE + 1) \
            and len(squeezed_original_shapes) <= COMPRESSION_DEFAULT_VALUE \
            and x_shape_none_dims_count < 2 \
            and output_shape is not None:
            # Special Transpose.1
            #   Suppresses as much as possible the conversion of transposes
            #   of 7 or more dimensions into FlexTransposes.
            #   Compresses dimensions with a numerical value of 1
            #   to suppress the generation of redundant Transpose.
            remove_one_target_perm = [
                idx for idx in perm if idx not in x_shape_one_dims
            ]
            sorted_remove_one_target_perm = sorted(remove_one_target_perm)
            replaced_remove_one_target_perm = [
                sorted_remove_one_target_perm.index(idx) \
                    for idx in remove_one_target_perm
            ]
            transposed_no_one_data = \
                tf.transpose(
                    a=squeezed_original_x,
                    perm=replaced_remove_one_target_perm,
                )
            tensor_after_transposition = \
                tf.reshape(
                    tensor=transposed_no_one_data,
                    shape=[
                        dim if not isinstance(dim, str) else -1 for dim in output_shape
                    ],
                )
        elif \
                (
                    input_tensor_rank >= (COMPRESSION_DEFAULT_VALUE + 1) and x_shape_none_dims_count == 0
                ) \
            or \
                (
                    number_of_dimensions_after_flextranspose_compression < COMPRESSION_DEFAULT_VALUE \
                        and number_of_dimensions_after_flextranspose_compression >= 2 \
                        and x_shape_none_dims_count == 0
                ):
            # Special Transpose.2
            #   Suppresses as much as possible the conversion of transposes
            #   of 7 or more dimensions into FlexTransposes.
            #   Decompose and transpose the tensor to be less than 6 dimensions.
            #   Compress in order from the dimension with the smallest value.
            #   https://github.com/PINTO0309/onnx2tf/issues/93

            # Overall process flow
            #   1. Extract the dimension with the smallest number needed to be less than 5 dimensions
            #   2. Split the tensor in the extracted dimension
            #   3. Transpose a divided tensor
            #   4. Concat the transposed tensor

            """
            e.g.
                data:
                    shape = [2,8,8,3,4,5,4,5]
                    x = torch.arange(1, np.prod(shape)+1)
                    x = x.reshape(shape)
                    target_transpose_perm = [6,0,1,4,7,2,5,3]

                result:
                    shape = [4,2,8,4,5,8,5,3]
            """
            # 1. Extract the dimension with the smallest number needed to be less than 5 dimensions
            np_input_tensor_shape = np.asarray(input_tensor_shape)
            num_of_dim_requiring_compression = \
                input_tensor_rank - number_of_dimensions_after_flextranspose_compression
            """
            np_input_tensor_shape:
                Shape of input data before transposition
                [2, 8, 8, 3, 4, 5, 4, 5]

            sorted_minimum_idxs:
                List of extracted dimension numbers with small numbers
                [0, 3, 4]

            removed_split_perm:

                [6, 1, 7, 2, 5]

            target_transpose_perm:
                perm after transposition
                [6, 0, 1, 4, 7, 2, 5, 3]

            target_sorted_minimum_idxs:
                Dimension to be restored at the end of processing
                [1, 7, 3]

            target_minimum_dims:
                Number of dimensions to be finally re-expanded
                [2, 3, 4]

            target_transpose_shape:
                [4, 2, 8, 4, 5, 8, 5, 3]
            """
            sorted_minimum_idxs = np.argsort(np_input_tensor_shape)[:num_of_dim_requiring_compression].tolist()
            target_minimum_dims = [
                np_input_tensor_shape[sorted_idx] for sorted_idx in sorted_minimum_idxs
            ]
            removed_split_perm = [
                dim for dim in perm if dim not in sorted_minimum_idxs
            ]
            sorted_removed_split_perm = sorted(removed_split_perm)
            removed_splited_transpose_perm = [
                sorted_removed_split_perm.index(idx) \
                    for idx in removed_split_perm
            ]
            target_transpose_perm = perm
            target_sorted_minimum_idxs = [
                target_transpose_perm.index(idx) for idx in sorted_minimum_idxs
            ]

            # 2. Split the tensor in the extracted dimension
            def split_squeeze_tensor(
                *,
                input_tensors: List[Any],
                axis: int,
            ):
                result_tensor_list = []
                for input_tensor in input_tensors:
                    splited_squeezed_tensors = []
                    for i in range(input_tensor.shape[axis]):
                        splited_squeezed_tensors.append(
                            tf.gather(
                                params=input_tensor,
                                indices=i,
                                axis=axis,
                            )
                        )
                    result_tensor_list = result_tensor_list + splited_squeezed_tensors
                return result_tensor_list

            splited_squeezed_tensors = [input_tensor]
            axeses = copy.deepcopy(sorted_minimum_idxs)
            axeses_idx = 0
            while True:
                axis = axeses[axeses_idx]
                splited_squeezed_tensors = split_squeeze_tensor(
                    input_tensors=splited_squeezed_tensors,
                    axis=axis,
                )
                axeses_idx += 1
                if axeses_idx > len(axeses)-1:
                    break
                new_axeses = []
                for axes in axeses:
                    if axes <= axis:
                        new_axeses.append(axes)
                    else:
                        new_axeses.append(axes-1)
                axeses = new_axeses

            # 3. Transpose a divided tensor (splited_squeezed_tensors)
            """
            splited_squeezed_tensors:
                [
                    [8, 8, 5, 4, 5],
                    [8, 8, 5, 4, 5],
                    [8, 8, 5, 4, 5],
                            :
                ]

            shrink_transposed_tensors:
                [
                    [4, 8, 5, 8, 5],
                    [4, 8, 5, 8, 5],
                    [4, 8, 5, 8, 5],
                            :
                ]
            """
            shrink_transposed_tensors = []
            for splited_squeezed_tensor in splited_squeezed_tensors:
                shrink_transposed_tensors.append(
                    tf.transpose(
                        a=splited_squeezed_tensor,
                        perm=removed_splited_transpose_perm,
                    )
                )

            # 4. Concat the transposed tensor
            """
            target_sorted_minimum_idxs:
                [1, 7, 3]

            asc_target_idxs_for_expand:
                [1, 3, 7]

            target_minimum_dims:
                [2, 3, 4]

            len(shrink_transposed_tensors):
                24

            ##########################################
            shrink_transposed_tensors:
                [
                    [4, 8, 5, 8, 5],
                    [4, 8, 5, 8, 5],
                    [4, 8, 5, 8, 5],
                            :
                ]

            ########################################## step.1 - expand
            [1, 7, 3] -> [1, 3, 7]
            shrink_transposed_tensors:
                [
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                            :
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                ]

            ########################################## step.2 - grouping
            target_concat_axes: [1, 7, 3] -> [3, 7, 1]
            gorouping_dims: [2, 3, 4] -> [4, 3, 2]
            grouped_total_tensors:
            [
                [
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                ],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                ],
                                :
                [
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                    [4, 1, 8, 1, 5, 8, 5, 1],
                ],
            ]

            ########################################## step.3 - concat
            concated_part_tensors:
            [
                [4, 1, 8, 4, 5, 8, 5, 1],
                [4, 1, 8, 4, 5, 8, 5, 1],
                            :
                [4, 1, 8, 4, 5, 8, 5, 1],
                [4, 1, 8, 4, 5, 8, 5, 1],
            ]

            ########################################## step.final
            final_transposed_tensors:
                [4, 2, 8, 4, 5, 8, 5, 3]
            """

            ########################################## step.1 - expand
            asc_target_idxs_for_expand = sorted(target_sorted_minimum_idxs)
            for target_sorted_minimum_idx in asc_target_idxs_for_expand:
                transposed_expanded_tensors = []
                for shrink_transposed_tensor in shrink_transposed_tensors:
                    transposed_expanded_tensors.append(
                        tf.expand_dims(
                            input=shrink_transposed_tensor,
                            axis=target_sorted_minimum_idx,
                        )
                    )
                shrink_transposed_tensors = transposed_expanded_tensors

            ########################################## step.2 - grouping
            target_concat_axes = reversed(target_sorted_minimum_idxs)
            gorouping_dims = reversed(target_minimum_dims)
            for concat_axis, target_concat_dim in zip(target_concat_axes, gorouping_dims):
                grouped_part_tensors = []
                grouped_total_tensors = []
                for idx, shrink_transposed_tensor in enumerate(shrink_transposed_tensors):
                    if idx > 0 and (idx % target_concat_dim) == 0:
                        grouped_total_tensors.append(grouped_part_tensors)
                        grouped_part_tensors = []
                    grouped_part_tensors.append(shrink_transposed_tensor)
                grouped_total_tensors.append(grouped_part_tensors)

                ########################################## step.3 - concat
                concated_part_tensors = []
                for tensors in grouped_total_tensors:
                    concated_part_tensors.append(
                        tf.concat(
                            values=tensors,
                            axis=concat_axis,
                        )
                    )
                shrink_transposed_tensors = concated_part_tensors

            ########################################## step.final
            tensor_after_transposition = shrink_transposed_tensors[0]

        else:
            # Normal Transpose
            tensor_after_transposition = tf.transpose(
                a=input_tensor,
                perm=perm,
                name=name,
            )

    return tensor_after_transposition


def stridedslice_with_flexing_deterrence(
    *,
    input_tensor: Any,
    begin: List[int],
    end: List[int],
    strides: List[int],
    begin_mask: List[int],
    end_mask: List[int],
    ignore_axes: List[int],
    compression_defult_value: int,
    onnx_slice_dims_count: int,
    output_shape: List[int] = None,
    name: str = None,
    **kwargs: Dict,
) -> Any:
    """StridedSlice tensors of 6 or more dimensions while suppressing the transformation to FlexTranspose.
    Suppress FlexTranspose generation only if the enable_suppression_flextranspose option is enabled when the tool is started.

    Parameters
    ----------
    input_tensor: Any
        Tensor to be Slice

    begin: List[int]
        begin

    end: List[int]
        end

    strides: List[int]
        strides

    begin_mask: List[int]
        begin_mask

    end_mask: List[int]
        end_mask

    ignore_axes: List[int]
        List of dimensions to slice

    compression_defult_value: int
        Default minimum number of dimensions to enable or disable dimensional compression processing.

    output_shape: List[int]
        Shape of tensor after transposition.
        The shape of the tensor in TensorFlow format after transposition must be specified.
        This value may produce the most optimized StridedSlice with Special StridedSlice.1 applied.
        If this value is not specified, the redundant Special StridedSlice.2 is applied.

    onnx_slice_dims_count: int
        Number of dimensions to slice.
        Used as a decision condition to skip processing of Special StridedSlice.2
        when multiple dimensions are targeted for Slice.

    name: str
        graph_node.name

    Returns
    ----------
    tensor_after_stridedslice: Any
        Tensor after stridedslice
    """
    disable_suppression_flexstridedslice: bool = \
        kwargs['disable_suppression_flexstridedslice']
    number_of_dimensions_after_flexstridedslice_compression: int = \
        kwargs['number_of_dimensions_after_flexstridedslice_compression']

    tensor_after_stridedslice = input_tensor

    if disable_suppression_flexstridedslice:
        # Normal StridedSlice
        tensor_after_stridedslice = \
            tf.strided_slice(
                input_=input_tensor,
                begin=begin,
                end=end,
                strides=strides,
                begin_mask=begin_mask,
                end_mask=end_mask,
                name=name,
            )
    else:
        # Special StridedSlice
        # Get dimension with 1 element
        input_tensor_shape: List[int] = input_tensor.shape
        input_tensor_rank = len(input_tensor_shape)
        # Only dimensions for which both begin_mask and end_mask are
        # ignored (==1) are selected for compression
        remove_one_taget_axes = [
            axis for axis in range(input_tensor_rank) \
                if axis not in ignore_axes
        ]
        # Extract dimensions of size 1
        x_shape_one_dims = [
            idx for idx in (remove_one_taget_axes) \
                if isinstance(input_tensor_shape[idx], int) and input_tensor_shape[idx]==1
        ]
        # Get the number of undefined dimensions
        x_shape_none_dims_count = len(
            [dim for dim in input_tensor_shape if not isinstance(dim, int) or dim < 1]
        )
        # Delete dimension with 1 element
        squeezed_original_x = tf.squeeze(input_tensor, x_shape_one_dims)

        # Obtain a shape with the dimension with 1 element removed
        squeezed_original_shapes = squeezed_original_x.shape

        if input_tensor_rank >= (compression_defult_value + 1) \
            and len(squeezed_original_shapes) <= compression_defult_value \
            and x_shape_none_dims_count == 0 \
            and output_shape is not None:
            # Special StridedSlice.1
            #   Suppresses as much as possible the conversion of StridedSlice
            #   of 6 or more dimensions into FlexStridedSlice.
            #   Compresses dimensions with a numerical value of 1
            #   to suppress the generation of redundant StridedSlice.

            # Adjust begin, end, strides, begin_mask, end_mask
            begin_ = []
            end_ = []
            strides_ = []
            begin_mask_ = 0
            end_mask_ = 0
            for axis in range(input_tensor_rank):
                if axis not in x_shape_one_dims:
                    begin_.append(int(begin[axis]))
            for axis in range(input_tensor_rank):
                if axis not in x_shape_one_dims:
                    end_.append(int(end[axis]))
            for axis in range(input_tensor_rank):
                if axis not in x_shape_one_dims:
                    strides_.append(int(strides[axis]))
            begin_mask_bit = list(reversed([int(i) for i in list(bin(begin_mask)[2:].zfill(len(begin_)))]))
            idx = 0
            for axis in range(input_tensor_rank):
                if axis not in x_shape_one_dims:
                    begin_mask_ += 2**idx*begin_mask_bit[axis]
                    idx +=1
            end_mask_bit = list(reversed([int(i) for i in list(bin(end_mask)[2:].zfill(len(begin_)))]))
            idx = 0
            for axis in range(input_tensor_rank):
                if axis not in x_shape_one_dims:
                    end_mask_ += 2**idx*end_mask_bit[axis]
                    idx +=1
            if len(begin_) > 0:
                # Special StridedSlice
                stridedslice_no_one_data = \
                    tf.strided_slice(
                        input_=squeezed_original_x,
                        begin=tf.convert_to_tensor(begin_),
                        end=tf.convert_to_tensor(end_),
                        strides=tf.convert_to_tensor(strides_),
                        begin_mask=tf.convert_to_tensor(begin_mask_),
                        end_mask=tf.convert_to_tensor(end_mask_),
                        name=name,
                    )
                tensor_after_stridedslice = \
                    tf.reshape(
                        tensor=stridedslice_no_one_data,
                        shape=\
                            tf.convert_to_tensor(
                                [
                                    dim if not isinstance(dim, str) and dim is not None else -1 for dim in output_shape
                                ]
                            ),
                    )
            else:
                # Normal StridedSlice
                tensor_after_stridedslice = \
                    tf.strided_slice(
                        input_=input_tensor,
                        begin=begin,
                        end=end,
                        strides=strides,
                        begin_mask=begin_mask,
                        end_mask=end_mask,
                        name=name,
                    )

        elif onnx_slice_dims_count == 1 \
            and \
                (
                    (
                        input_tensor_rank >= (compression_defult_value + 1) \
                            and x_shape_none_dims_count == 0
                    ) \
                or \
                    (
                        number_of_dimensions_after_flexstridedslice_compression < compression_defult_value \
                            and number_of_dimensions_after_flexstridedslice_compression >= 1 \
                            and x_shape_none_dims_count == 0
                    )
                ):
            # Special StridedSlice.2
            #   Suppresses as much as possible the conversion of StridedSlice
            #   of 6 or more dimensions into FlexStridedSlice.
            #   Decompose and StridedSlice the tensor to be less than 5 dimensions.
            #   Compress in order from the dimension with the smallest value.
            #   This special process is performed only when there is only one dimension to slice.

            # Overall process flow
            #   1. Extract the dimension with the smallest number needed to be less than 5 dimensions
            #   2. Split the tensor in the extracted dimension
            #   3. StridedSlice a divided tensor
            #   4. Concat the sliced tensor

            """
            e.g.
                data:
                    shape = [2,8,8,3,4,5,4,5]
                    x = torch.arange(1, np.prod(shape)+1)
                    x = x.reshape(shape)
                    pattern.1: target_slice = axis=0, [0:1] (smallest axis)
                    pattern.2: target_slice = axis=3, [1:2] (second smallest axis)
                    pattern.3: target_slice = axis=7, [4:5]

                result:
                    pattern.1: shape = [1,8,8,3,4,5,4,5]
                    pattern.2: shape = [2,8,8,2,4,5,4,5]
                    pattern.3: shape = [2,8,8,3,4,5,4,1]
            """
            # 1. Extract the dimension with the smallest number needed to be less than 5 dimensions
            np_input_tensor_shape = np.asarray(input_tensor_shape)
            num_of_dim_requiring_compression = \
                input_tensor_rank - number_of_dimensions_after_flexstridedslice_compression
            """
            pattern.1: axis=0

            np_input_tensor_shape:
                Shape of input data before transposition
                [2, 8, 8, 3, 4, 5, 4, 5]

            taget_axes:
                List of indices excluding the dimension to be slice
                [1, 2, 3, 4, 5, 6, 7]

            sorted_minimum_idxs:
                List of extracted dimension numbers with small numbers
                Split is performed on the index extracted here.
                    0  1  2  3  4  5  6              2, 3, 5, 4, 6, 0, 1                3, 4, 4
                [   8, 8, 3, 4, 5, 4, 5] -> sort -> [3, 4, 4, 5, 5, 8, 8] -> 3picks -> [2, 3, 5]

            target_minimum_dims:
                Get the dim size corresponding to sorted_minimum_idxs.
                        2  3     5
                [ 8, 8, 3, 4, 5, 4, 5] -> [3, 4, 4]

            sorted_minimum_idxs_correction:
                Convert the list to take into account indexes that were initially excluded from processing.
                sorted_minimum_idxs           : [2, 3, 5]
                sorted_minimum_idxs_correction: [3, 4, 6]
                    0, 1, 2, 3, 4, 5, 6
                [   8, 8, 3, 4, 5, 4, 5]

                    0, 1, 2, 3, 4, 5, 6, 7
                [   2, 8, 8, 3, 4, 5, 4, 5]

            splited_squeezed_tensors:
                [
                    [2, 8, 8, 5, 5],
                    [2, 8, 8, 5, 5],
                    [2, 8, 8, 5, 5],
                            :
                ]
            """
            # Extract dimensions other than the dimension to be Slice
            taget_axes = [
                axis for axis in range(input_tensor_rank) \
                    if axis not in ignore_axes
            ]
            ignore_axis = ignore_axes[0]
            if len(taget_axes) < num_of_dim_requiring_compression:
                # Normal StridedSlice
                tensor_after_stridedslice = tf.strided_slice(
                    input_=input_tensor,
                    begin=begin,
                    end=end,
                    strides=strides,
                    begin_mask=begin_mask,
                    end_mask=end_mask,
                    name=name,
                )
            else:
                np_input_tensor_shape = np_input_tensor_shape[taget_axes]
                # Obtain axis for the number of dimensions that need to be compressed,
                # starting from the one with the smallest dimensional value.
                sorted_minimum_idxs = np.argsort(np_input_tensor_shape)[:num_of_dim_requiring_compression].tolist()
                target_minimum_dims = [
                    np_input_tensor_shape[sorted_idx] for sorted_idx in sorted_minimum_idxs
                ]

                """
                sorted_minimum_idxs           : [2, 3, 5]
                sorted_minimum_idxs_correction: [3, 4, 6]

                    0, 1, 2, 3, 4, 5, 6
                [   8, 8, 3, 4, 5, 4, 5]

                    0, 1, 2, 3, 4, 5, 6, 7
                [   2, 8, 8, 3, 4, 5, 4, 5]
                """
                sorted_minimum_idxs_correction = [
                    idx+1 if idx+1 > ignore_axis else idx for idx in sorted_minimum_idxs
                ]

                # 2. Split the tensor in the extracted dimension
                def split_squeeze_tensor(
                    *,
                    input_tensors: List[Any],
                    axis: int,
                ):
                    result_tensor_list = []
                    for input_tensor in input_tensors:
                        splited_squeezed_tensors = []
                        for i in range(input_tensor.shape[axis]):
                            splited_squeezed_tensors.append(
                                tf.gather(
                                    params=input_tensor,
                                    indices=i,
                                    axis=axis,
                                )
                            )
                        result_tensor_list = result_tensor_list + splited_squeezed_tensors
                    return result_tensor_list

                splited_squeezed_tensors = [input_tensor]
                axeses = copy.deepcopy(sorted_minimum_idxs_correction)
                axeses_idx = 0
                while True:
                    axis = axeses[axeses_idx]
                    splited_squeezed_tensors = split_squeeze_tensor(
                        input_tensors=splited_squeezed_tensors,
                        axis=axis,
                    )
                    axeses_idx += 1
                    if axeses_idx > len(axeses)-1:
                        break
                    new_axeses = []
                    for axes in axeses:
                        if axes <= axis:
                            new_axeses.append(axes)
                        else:
                            new_axeses.append(axes-1)
                    axeses = new_axeses

                # 3. Slice a divided tensor (splited_squeezed_tensors)
                """
                sorted_minimum_idxs_correction:
                    [3, 4, 6]

                pickup_idx:
                    [0, 1, 2, 5, 7]

                splited_squeezed_tensors:
                    [
                        [2, 8, 8, 5, 5],
                        [2, 8, 8, 5, 5],
                        [2, 8, 8, 5, 5],
                                :
                    ]
                            0  1  2  3  4  5  6  7
                begin     :[1, 0, 0, 0, 0, 0, 0, 0]
                end       :[2, 0, 0, 0, 0, 0, 0, 0]
                strides   :[1, 1, 1, 1, 1, 1, 1, 1]
                begin_mask:254 -> 11111110
                end_mask  :254 -> 11111110
                """
                # Adjust begin, end, strides, begin_mask, end_mask
                pickup_idx = np.asarray(
                    [idx for idx in range(input_tensor_rank) \
                        if idx not in sorted_minimum_idxs_correction]
                )
                begin_ = None
                if hasattr(begin, 'numpy'):
                    begin_ = begin.numpy()
                elif isinstance(begin, int):
                    begin_ = np.asarray([begin])
                else:
                    begin_ = np.asarray(begin)
                if hasattr(end, 'numpy'):
                    end_ = end.numpy()
                elif isinstance(end, int):
                    end_ = np.asarray([end])
                else:
                    end_ = np.asarray(end)
                if hasattr(strides, 'numpy'):
                    strides_ = strides.numpy()
                elif isinstance(strides, int):
                    strides_ = np.asarray([strides])
                else:
                    strides_ = np.asarray(strides)
                begin_mask_bit = np.asarray(list(reversed([int(val) for val in list(bin(begin_mask)[2:].zfill(len(begin_)))])))
                end_mask_bit = np.asarray(list(reversed([int(val) for val in list(bin(end_mask)[2:].zfill(len(begin_)))])))

                begin_ = begin_[pickup_idx]
                end_ = end_[pickup_idx]
                strides_ = strides_[pickup_idx]
                begin_mask_bit = begin_mask_bit[pickup_idx]
                begin_mask_ = 0
                for idx, mask_bit in enumerate(begin_mask_bit):
                    begin_mask_ += 2**idx*mask_bit
                end_mask_bit = end_mask_bit[pickup_idx]
                end_mask_ = 0
                for idx, mask_bit in enumerate(end_mask_bit):
                    end_mask_ += 2**idx*mask_bit

                """
                shrink_sliced_tensors:
                    [
                        [1, 8, 8, 5, 5],
                        [1, 8, 8, 5, 5],
                        [1, 8, 8, 5, 5],
                                :
                    ]
                """
                shrink_sliced_tensors = []
                for splited_squeezed_tensor in splited_squeezed_tensors:
                    shrink_sliced_tensors.append(
                        tf.strided_slice(
                            input_=splited_squeezed_tensor,
                            begin=begin_,
                            end=end_,
                            strides=strides_,
                            begin_mask=begin_mask_,
                            end_mask=end_mask_,
                        )
                    )

                # 4. Concat the sliced tensor
                """
                target_minimum_dims:
                    [3, 4, 4]

                sorted_minimum_idxs_correction:
                    [3, 4, 6]

                len(shrink_sliced_tensors):
                    48

                ##########################################
                shrink_sliced_tensors:
                    [
                        [1, 8, 8, 5, 5],
                        [1, 8, 8, 5, 5],
                        [1, 8, 8, 5, 5],
                                :
                    ]

                ########################################## step.1 - expand
                shrink_sliced_tensors:
                    [
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                                :
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                    ]

                ########################################## step.2 - grouping
                sorted_minimum_idxs_correction: [3, 4, 6] -> target_concat_axes: [6, 4, 3]
                target_minimum_dims: [3, 4, 4] -> gorouping_dims: [4, 4, 3]
                grouped_total_tensors:
                [
                    [
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                    ],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                    ],
                                    :
                    [
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                        [1, 8, 8, 1, 1, 5, 1, 5],
                    ],
                ]

                ########################################## step.3 - concat
                concated_part_tensors:
                [
                    [1, 8, 8, 1, 1, 5, 4, 5],
                    [1, 8, 8, 1, 1, 5, 4, 5],
                                :
                    [1, 8, 8, 1, 1, 5, 4, 5],
                    [1, 8, 8, 1, 1, 5, 4, 5],
                ]
                    ↓
                [
                    [1, 8, 8, 1, 4, 5, 4, 5],
                    [1, 8, 8, 1, 4, 5, 4, 5],
                                :
                    [1, 8, 8, 1, 4, 5, 4, 5],
                    [1, 8, 8, 1, 4, 5, 4, 5],
                ]
                    ↓
                [
                    [1, 8, 8, 3, 4, 5, 4, 5],
                ]

                ########################################## step.final
                final_transposed_tensors:
                    [1, 8, 8, 3, 4, 5, 4, 5]
                """

                ########################################## step.1 - expand
                for sorted_minimum_idx in sorted_minimum_idxs_correction:
                    sliced_expanded_tensors = []
                    for shrink_sliced_tensor in shrink_sliced_tensors:
                        sliced_expanded_tensors.append(
                            tf.expand_dims(
                                input=shrink_sliced_tensor,
                                axis=sorted_minimum_idx,
                            )
                        )
                    shrink_sliced_tensors = sliced_expanded_tensors

                ########################################## step.2 - grouping
                target_concat_axes = reversed(sorted_minimum_idxs_correction)
                gorouping_dims = reversed(target_minimum_dims)

                for concat_axis, target_concat_dim in zip(target_concat_axes, gorouping_dims):
                    grouped_part_tensors = []
                    grouped_total_tensors = []
                    for idx, shrink_sliced_tensor in enumerate(shrink_sliced_tensors):
                        if idx > 0 and (idx % target_concat_dim) == 0:
                            grouped_total_tensors.append(grouped_part_tensors)
                            grouped_part_tensors = []
                        grouped_part_tensors.append(shrink_sliced_tensor)
                    grouped_total_tensors.append(grouped_part_tensors)

                    ########################################## step.3 - concat
                    concated_part_tensors = []
                    for tensors in grouped_total_tensors:
                        concated_part_tensors.append(
                            tf.concat(
                                values=tensors,
                                axis=concat_axis,
                            )
                        )
                    shrink_sliced_tensors = concated_part_tensors

                ########################################## step.final
                tensor_after_stridedslice = shrink_sliced_tensors[0]

        else:
            # Normal StridedSlice
            tensor_after_stridedslice = \
                tf.strided_slice(
                    input_=input_tensor,
                    begin=begin,
                    end=end,
                    strides=strides,
                    begin_mask=begin_mask,
                    end_mask=end_mask,
                    name=name,
                )

    return tensor_after_stridedslice


def dummy_onnx_inference(
    *,
    onnx_graph: onnx.ModelProto,
    output_names: List[str],
    test_data_nhwc: Optional[np.ndarray] = None,
    custom_input_op_name_np_data_path: Optional[str] = None,
    tf_layers_dict: Optional[Dict] = None,
    use_cuda: bool = False,
    disable_strict_mode: bool = False,
) -> List[np.ndarray]:
    """Perform inference on ONNX subgraphs with an all-1 dummy tensor.

    Parameters
    ----------
    onnx_graph: onnx.ModelProto
        ONNX subgraphs

    output_names: List[str]
        List of output names to be checked for output values

    test_data_nhwc: Optional[np.ndarray]
        Test Image Data

    custom_input_op_name_np_data_path: Optional[str]
        Path to Numpy file for custom data used for dummy inference

    tf_layers_dict: Optional[Dict]
        TensorFlow Model Structure Dictionary.
        Used to determine if the input OP needs to be transposed
        from NHWC to NCHW when checking accuracy.

    use_cuda: Optional[bool]
        True if CUDA is used for inference, False if not.

    disable_strict_mode: Optional[bool]
        True to disable strict inference mode, False to enable it.

    Returns
    ----------
    outputs: List[np.ndarray]
        Results of inference using dummy tensor
    """
    # Separate onnx at specified output_names position
    domain: str = onnx_graph.domain
    ir_version: int = onnx_graph.ir_version
    meta_data = {'domain': domain, 'ir_version': ir_version}
    metadata_props = None
    if hasattr(onnx_graph, 'metadata_props'):
        metadata_props = onnx_graph.metadata_props
    gs_graph = gs.import_onnx(onnx_graph)

    # reduce all axes except batch axis
    for i, node in enumerate(gs_graph.nodes):
        if gs_graph.opset <= 17 \
            and gs_graph.nodes[i].op in ['ReduceMax', 'ReduceMean', 'ReduceMin', 'ReduceProd'] \
            and 'axes' not in node.attrs:
            gs_graph.nodes[i].attrs['axes'] = [
                i for i in range(1, len(gs_graph.nodes[i].inputs[0].shape))
            ] if len(gs_graph.nodes[i].inputs[0].shape) > 1 else [0]

        elif gs_graph.opset > 17 \
            and gs_graph.nodes[i].op in ['ReduceMax', 'ReduceMean', 'ReduceMin', 'ReduceProd'] \
            and len(gs_graph.nodes[i].inputs) == 1:
            const_axes = [
                i for i in range(1, len(gs_graph.nodes[i].inputs[0].shape))
            ] if len(gs_graph.nodes[i].inputs[0].shape) > 1 else [0]
            gs_graph.nodes[i].inputs.append(
                gs.Constant(
                    f'{gs_graph.nodes[i].name}_axes',
                    values=np.asarray(const_axes, dtype=np.int64)
                )
            )

        elif gs_graph.opset <= 12 \
            and gs_graph.nodes[i].op in ['ReduceSum'] \
            and 'axes' not in node.attrs:
            gs_graph.nodes[i].attrs['axes'] = [
                i for i in range(1, len(gs_graph.nodes[i].inputs[0].shape))
            ] if len(gs_graph.nodes[i].inputs[0].shape) > 1 else [0]

        elif gs_graph.opset > 12 \
            and gs_graph.nodes[i].op in ['ReduceSum'] \
            and len(gs_graph.nodes[i].inputs) == 1:
            const_axes = [
                i for i in range(1, len(gs_graph.nodes[i].inputs[0].shape))
            ] if len(gs_graph.nodes[i].inputs[0].shape) > 1 else [0]
            gs_graph.nodes[i].inputs.append(
                gs.Constant(
                    f'{gs_graph.nodes[i].name}_axes',
                    values=np.asarray(const_axes, dtype=np.int64)
                )
            )

    # instead, modify onnx graph manually
    gs_graph.outputs = []
    for graph_node in gs_graph.nodes:
        for node_output in graph_node.outputs:
            if node_output.name in output_names:
                if node_output.dtype is not None:
                    gs_graph.outputs.append(node_output)

    new_onnx_graph = gs.export_onnx(graph=gs_graph, do_type_check=False, **meta_data)
    if metadata_props is not None:
        new_onnx_graph.metadata_props.extend(metadata_props)
    tmp_onnx_path = ''
    tmp_onnx_external_weights_path =''
    try:
        serializer: ProtoSerializer = onnx._get_serializer(fmt='protobuf')
        serialized_graph = serializer.serialize_proto(proto=new_onnx_graph)
    except ValueError as ve:
        tmp_onnx_path = 'tmp.onnx'
        tmp_onnx_external_weights_path ='tmp_external.weights'
        onnx.save(
            proto=new_onnx_graph,
            f=tmp_onnx_path,
            save_as_external_data=True,
            location=tmp_onnx_external_weights_path
        )
        serialized_graph = tmp_onnx_path
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True) - 1

    if use_cuda:
        if check_cuda_enabled():
            try:
                onnx_session = ort.InferenceSession(
                    path_or_bytes=serialized_graph,
                    sess_options=sess_options,
                    providers=['CUDAExecutionProvider','CPUExecutionProvider'],
                )
            except Exception as ex:
                onnx_session = ort.InferenceSession(
                    path_or_bytes=serialized_graph,
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider'],
                )
        else:
            onnx_session = ort.InferenceSession(
                path_or_bytes=serialized_graph,
                sess_options=sess_options,
                providers=['CPUExecutionProvider'],
            )
    else:
        onnx_session = ort.InferenceSession(
            path_or_bytes=serialized_graph,
            sess_options=sess_options,
            providers=['CPUExecutionProvider'],
        )
    onnx_inputs = gs_graph.inputs
    input_names: List[str] = [inp.name for inp in onnx_inputs]
    input_sizes: List[int] = [inp.shape for inp in onnx_inputs]
    new_input_sizes = []
    for input_size in input_sizes:
        new_input_size = []
        for idx, dim in enumerate(input_size):
            if idx == 0 and input_sizes[0][0] is not None \
                and not isinstance(input_sizes[0][0], str) \
                and len(input_sizes[0]) == len(input_size) \
                and (dim is None or isinstance(dim, str)):
                # Batch size assignment for input OPs
                new_input_size.append(input_sizes[0][0])
            elif dim is None or isinstance(dim, str):
                # Fixed and assigned 1
                new_input_size.append(1)
            else:
                # Assign input shape as is
                new_input_size.append(dim)
        new_input_sizes.append(new_input_size)
    input_sizes = new_input_sizes
    input_dtypes: List[Any] = [inp.dtype for inp in onnx_inputs]
    input_datas = {}

    # -cid
    if custom_input_op_name_np_data_path:
        for param in custom_input_op_name_np_data_path:
            input_op_name = str(param[0])
            numpy_file_path = str(param[1])
            custom_input_data: np.ndarray = np.load(numpy_file_path)
            # NHWC -> NCHW
            input_op_info: Dict = tf_layers_dict.get(input_op_name, None)
            if input_op_info is not None:
                ncw_nchw_ncdhw_perm: List = input_op_info.get('ncw_nchw_ncdhw_perm', None)
                if ncw_nchw_ncdhw_perm is not None:
                    custom_input_data = custom_input_data.transpose(ncw_nchw_ncdhw_perm)
                onnx_batch_size = input_op_info['shape'][0]
                cdata_batch_size = custom_input_data.shape[0]
                if isinstance(onnx_batch_size, int) and onnx_batch_size != cdata_batch_size and cdata_batch_size > 1:
                    custom_input_data = custom_input_data[0:onnx_batch_size, ...]
                elif isinstance(onnx_batch_size, str) and cdata_batch_size > 1:
                    custom_input_data = custom_input_data[0:1, ...]

            input_datas[input_op_name] = custom_input_data

    else:
        for input_name, input_size, input_dtype in zip(input_names, input_sizes, input_dtypes):
            if test_data_nhwc is None:
                input_datas[input_name] = np.ones(
                    input_size,
                    dtype=input_dtype,
                )
            else:
                input_datas[input_name] = \
                    tf.transpose(
                        a=tf.image.resize(
                            images=test_data_nhwc,
                            size=[input_size[2],input_size[3]],
                        ),
                        perm=[0,3,1,2],
                    ).numpy().astype(input_dtype)

    dtype_sizes = {
        np.dtype('float16'): 2,
        np.dtype('float32'): 4,
        np.dtype('float64'): 8,

        np.dtype('uint8'): 1,
        np.dtype('uint16'): 2,
        np.dtype('uint32'): 4,
        np.dtype('uint64'): 8,

        np.dtype('int8'): 1,
        np.dtype('int16'): 2,
        np.dtype('int32'): 4,
        np.dtype('int64'): 8,

        np.dtype('bool_'): 1,
    }
    total_output_size: int = 0
    for gs_graph_output in gs_graph.outputs:
        op_output_size: int = 1
        if gs_graph_output.shape is not None:
            for s in gs_graph_output.shape:
                if isinstance(s, int):
                    op_output_size *= s
            # Total bytes
            total_output_size += op_output_size * dtype_sizes.get(gs_graph_output.dtype, 4)

    # When exact inference mode is enabled and the total size of the tensor of inference results exceeds approximately 80% of available RAM
    mem_available = psutil.virtual_memory().available * 0.80 // 1024 // 1024 //1024
    total_output_size_gb = (total_output_size // 1024 // 1024 //1024)
    if (not disable_strict_mode and total_output_size_gb > mem_available):
        if tmp_onnx_path:
            os.remove(tmp_onnx_path)
            os.remove(tmp_onnx_external_weights_path)
        raise Exception(
            f'The tool skipped dummy inference to avoid SWAP processing because the total size of the tensor of inference results exceeded about {mem_available} GB. (results: {total_output_size_gb} GB)'
        )

    outputs = onnx_session.run(None, input_datas)
    if tmp_onnx_path:
        os.remove(tmp_onnx_path)
        os.remove(tmp_onnx_external_weights_path)
    return outputs


def dummy_tf_inference(
    *,
    model: tf_keras.Model,
    inputs: List[tf_keras.Input],
    test_data_nhwc: Optional[np.ndarray] = None,
    verification_datas: Optional[List[np.ndarray]] = None,
    custom_input_op_name_np_data_path: Optional[str] = None,
) -> Any:
    """Perform inference on TF subgraphs with an all-1 dummy tensor.

    Parameters
    ----------
    model: tf_keras.Model
        Keras model

    inputs: List[tf_keras.Input]
        List of tf_keras.Input

    test_data_nhwc: Optional[np.ndarray]
        Test Image Data

    verification_datas: Optional[List[np.ndarray]]
        Test Data

    custom_input_op_name_np_data_path
        Path to Numpy file for custom data used for dummy inference

    Returns
    ----------
    outputs: Dict[np.ndarray]
        Results of inference using dummy tensor.
        Dict of tensorflow node and corresponding ndarray output.
    """
    input_names: List[str] = [inp.name for inp in inputs]
    input_sizes: List[int] = [inp.shape for inp in inputs]
    new_input_sizes = []
    for input_size in input_sizes:
        new_input_size = []
        for idx, dim in enumerate(input_size):
            if idx == 0 and input_sizes[0][0] is not None \
                and len(input_sizes[0]) == len(input_size) \
                and dim is None:
                # Batch size assignment for input OPs
                new_input_size.append(input_sizes[0][0])
            elif dim is None:
                # Fixed and assigned 1
                new_input_size.append(1)
            else:
                # Assign input shape as is
                new_input_size.append(dim)
        new_input_sizes.append(new_input_size)
    input_sizes = new_input_sizes
    input_dtypes: List[Any] = [inp.dtype for inp in inputs]
    input_datas = {}

    # -cid
    if custom_input_op_name_np_data_path:
        for idx, param in enumerate(custom_input_op_name_np_data_path):
            numpy_file_path = str(param[1])
            custom_input_data = np.load(numpy_file_path)
            input_size = input_sizes[idx]

            tf_batch_size = input_size[0]
            cdata_batch_size = custom_input_data.shape[0]
            if isinstance(tf_batch_size, int) and tf_batch_size != cdata_batch_size and cdata_batch_size > 1:
                custom_input_data = custom_input_data[0:tf_batch_size, ...]
            elif tf_batch_size is None and cdata_batch_size > 1:
                custom_input_data = custom_input_data[0:1, ...]

            if list(custom_input_data.shape) != input_size:
                error_msg = f'' + \
                    Color.RED(f'ERROR:') + ' ' + \
                    f"The format of custom input data is different from Tensorflow's format. " + \
                    f"Therefore, you cannot use custom input. "

                raise ValueError(error_msg)

            input_datas[input_names[idx]] = custom_input_data

    else:
        if verification_datas is None:
            for input_name, input_size, input_dtype in zip(input_names, input_sizes, input_dtypes):
                if test_data_nhwc is None:
                    input_datas[input_name] = np.ones(
                        input_size,
                        dtype=TF_DTYPES_TO_NUMPY_DTYPES[input_dtype],
                    )
                else:
                    input_datas[input_name] = \
                        tf.image.resize(
                            images=test_data_nhwc,
                            size=[input_size[1],input_size[2]],
                        ).numpy().astype(TF_DTYPES_TO_NUMPY_DTYPES[input_dtype])
        else:
            for input_name, input_size, input_dtype, verification_data \
                in zip(input_names, input_sizes, input_dtypes, verification_datas):

                if verification_data is not None:
                    verification_data = verification_data.numpy() \
                        if hasattr(verification_data, "numpy") else verification_data
                    if len(input_size) != len(verification_data.shape):
                        if len(verification_data.shape) <= 1:
                            input_datas[input_name] = verification_data
                        else:
                            input_datas[input_name] = verification_data.reshape(input_size)
                    else:
                        input_datas[input_name] = verification_data
                else:
                    input_datas[input_name] = np.ones(
                        input_size,
                        dtype=TF_DTYPES_TO_NUMPY_DTYPES[input_dtype],
                    )
    outputs = model(
        inputs={
            input.name: input_datas[input.name] for input in inputs
        },
        training=False,
    )

    if not isinstance(outputs, list):
        outputs = [outputs]

    tf_output_dict = {
        tensor.name: output.numpy() for tensor, output in zip(model.outputs, outputs)
    }

    return tf_output_dict


def onnx_tf_tensor_validation(
    *,
    output_pairs: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    rtol: float=1e-05,
    atol: float=1e-05,
) -> Dict[str, List]:
    """Check if the ONNX tensor and the TF tensor are approximate.

    Parameters
    ----------
    output_pairs: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]
        ONNX tensor to be verified
        {
            (onnx_output_name, tf_output_name): (onnx_tensor, tf_tensor),
            (onnx_output_name, tf_output_name): (onnx_tensor, tf_tensor),
                    :
        }

    rtol: float=1e-05
        The relative tolerance parameter

    atol: float=1e-05
        The absolute tolerance parameter

    Returns
    ----------
    check_results: Dict[str, List[np.ndarray, int, float|int]]
        Tensor Comparison Results
        {
            onnx_output_name: [
                onnx_tensor,
                matched_flg, <--- 0: Unmatched, 1: Matched, 2: Skipped (Deleted or Shape Unmatched),
                max_abs_err,
            ]
        }
    """
    check_results = {
        k: [v[0], False, 0.0] \
            for k, v in output_pairs.items()
    }

    for names_pair, (onnx_tensor, tf_tensor) in output_pairs.items():

        onnx_tensor_shape = onnx_tensor.shape
        max_abs_err = ONNX_INF_INDEX_VALUE
        """
        onnx_dummy_data: np.random.random_sample([1,3,224,224])
        tf_dummy_data  : onnx_dummy_data.transpose([0,2,3,1]), len(tf_tensor.shape) == 4

        tf_shape_transpose_perms:
            [
                (0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2),
                (0, 3, 2, 1), (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 0, 3), (1, 2, 3, 0),
                (1, 3, 0, 2), (1, 3, 2, 0), (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3),
                (2, 1, 3, 0), (2, 3, 0, 1), (2, 3, 1, 0), (3, 0, 1, 2), (3, 0, 2, 1),
                (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0)
            ]

        tf_target_transpose_perms:
            [(0, 3, 1, 2), (0, 3, 2, 1)]
        """
        tf_shape_transpose_perms = list(itertools.permutations(range(len(tf_tensor.shape))))
        tf_target_transpose_perms = [
            tf_shape_transpose_perm \
            for tf_shape_transpose_perm in tf_shape_transpose_perms \
            if tf_tensor.transpose(tf_shape_transpose_perm).shape == onnx_tensor_shape
        ]
        # Validation
        """
        tf_check_infos:
            {
                [
                    tf_target_transpose_perm, <--- tf_target_transpose_perms[idx]
                    matched_flg, <--- True: Matched, False: Unmatched
                ]
            }
        """
        validate_result = False
        tf_check_infos = [
            [tf_target_transpose_perm, 0] for tf_target_transpose_perm in tf_target_transpose_perms
        ]
        for tf_check_info in tf_check_infos:
            if len(onnx_tensor_shape) > 1:
                tf_transposed_tensor = tf_tensor.transpose(tf_check_info[0])
                if np.allclose(a=onnx_tensor, b=tf_transposed_tensor, rtol=rtol, atol=atol, equal_nan=True):
                    # Matched
                    tf_check_info[1] = 1
                    max_abs_err = 0.0
                    break
                else:
                    # Unmatched
                    dtype = NUMPY_DTYPES_TO_TF_DTYPES[tf_transposed_tensor.dtype] \
                        if isinstance(tf_transposed_tensor.dtype, np.dtype) else tf_transposed_tensor.dtype
                    if onnx_tensor.shape == tf_transposed_tensor.shape and dtype != tf.bool:
                        error_value = np.max(np.abs(onnx_tensor - tf_transposed_tensor))
                        max_abs_err = error_value if error_value < max_abs_err else max_abs_err
            else:
                tf_check_info[1] = 2
                max_abs_err = 0.0

        # Validation results check
        for tf_check_info in tf_check_infos:
            if tf_check_info[1]:
                validate_result = tf_check_info[1]
                break

        if not validate_result and max_abs_err == ONNX_INF_INDEX_VALUE:
            # Tensors deleted from the TensorFlow model structure during
            # the model optimization process are not comparable,
            # so the status is rewritten to Skip.
            # If there was no match between ONNX and TensorFlow output shapes.
            check_results[names_pair][1] = 2
            check_results[names_pair][2] = max_abs_err
        else:
            check_results[names_pair][1] = validate_result
            check_results[names_pair][2] = max_abs_err

    return check_results


def weights_export(
    *,
    extract_target_tflite_file_path: str,
    output_weights_file_path: str,
):
    """Extract only the weights from the generated TFLite file and save it to a file in hdf5 format.
    Note that the INT16 format is not supported.

    Parameters
    ----------
    extract_target_tflite_file_path: str
        Path of the tflite file from which the weights are extracted

    output_weights_file_path: str
        Path to file in hdf5 format to save the extracted weights
    """
    import h5py
    from tensorflow.lite.python import interpreter as interpreter_wrapper
    interpreter = interpreter_wrapper.Interpreter(
        model_path=extract_target_tflite_file_path,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_indexes = [
        input_detail['index'] for input_detail in input_details
    ]
    output_details = interpreter.get_output_details()
    output_indexes = [
        output_detail['index'] for output_detail in output_details
    ]
    tensor_details = interpreter.get_tensor_details()
    with h5py.File(output_weights_file_path, 'w') as f:
        for tensor_detail in tensor_details:
            tensor_index = tensor_detail['index']
            if tensor_index not in input_indexes \
                and tensor_index not in output_indexes:
                try:
                    d = f.create_dataset(
                        name=tensor_detail['name'],
                        data=interpreter.get_tensor(tensor_index)
                    )
                    del d
                except Exception as e:
                    pass


def download_test_image_data() -> np.ndarray:
    """Download dummy data for testing.

    Returns
    ----------
    test_image_data: np.ndarray
    """
    DATA_COUNT = 20
    FILE_NAME = f'calibration_image_sample_data_{DATA_COUNT}x128x128x3_float32.npy'
    LOCAL_FILE_PATH = os.path.join(os.getcwd(), FILE_NAME)

    if not os.path.isfile(LOCAL_FILE_PATH):
        URL = f'https://github.com/PINTO0309/onnx2tf/releases/download/1.20.4/{FILE_NAME}'
        try:
            # GitHub releases
            test_sample_images_npy = requests.get(URL, timeout=(1.0, 5.0)).content
        except requests.exceptions.Timeout:
            # Wasabi Storage
            URL = f'https://s3.us-central-1.wasabisys.com/onnx2tf-en/datas/{FILE_NAME}'
            test_sample_images_npy = requests.get(URL).content
    else:
        with open(LOCAL_FILE_PATH, 'rb') as test_sample_images_npy_file:
            test_sample_images_npy = test_sample_images_npy_file.read()
    test_image_data = None
    with io.BytesIO(test_sample_images_npy) as f:
        test_image_data: np.ndarray = np.load(f)
    return test_image_data


def broadcast_for_gpu_delegate(
    *,
    input_tensor_1: Any,
    input_tensor_2: Any,
    **kwargs: Dict,
):
    """Tensor broadcast when optimizing to GPU Delegate.
    'MUL requires one tensor that not less than second in all dimensions.'

    Returns
    ----------
    tiled_x: Any
        Broadcasted input_tensor1

    tiled_y: Any
        Broadcasted input_tensor2
    """
    optimization_for_gpu_delegate: bool = \
        kwargs['optimization_for_gpu_delegate']
    if not optimization_for_gpu_delegate:
        return input_tensor_1, input_tensor_2
    xshapes = input_tensor_1.shape
    xshape_list = [int(dim) for dim in input_tensor_1.shape]
    xshapes_rank = len(xshapes)
    yshapes = input_tensor_2.shape
    yshape_list = [int(dim) for dim in input_tensor_2.shape]
    yshapes_rank = len(yshape_list)

    try:
        if xshapes_rank > 0 and yshapes_rank > 0:

            def x_tile(
                *,
                input_tensor_1,
                input_tensor_2,
                xshapes,
                xshapes_rank,
                yshapes_rank,
                yshape_list,
            ):
                tile_counts = np.asarray([1] * xshapes_rank, dtype=np.int64)
                tile_counts_part_list = list(tile_counts)
                tile_counts_part_list = list(tile_counts[-yshapes_rank:])
                xshape_part_list = list(xshapes[-yshapes_rank:])
                for axis, (xshape, yshape) in enumerate(zip(xshape_part_list, yshape_list)):
                    if xshape is not None and yshape is not None and xshape < yshape and xshape == 1:
                        tile_counts_part_list[axis] = yshape
                tile_counts[-yshapes_rank:] = tile_counts_part_list
                tiled_x = tf.tile(input_tensor_1, list(tile_counts)) * -1 * -1
                return tiled_x, input_tensor_2

            def y_tile(
                *,
                input_tensor_1,
                input_tensor_2,
                yshapes,
                yshapes_rank,
                xshapes_rank,
                xshape_list,
            ):
                tile_counts = np.asarray([1] * yshapes_rank, dtype=np.int64)
                tile_counts_part_list = list(tile_counts)
                tile_counts_part_list = list(tile_counts[-xshapes_rank:])
                yshape_part_list = list(yshapes[-xshapes_rank:])
                for axis, (xshape, yshape) in enumerate(zip(xshape_list, yshape_part_list)):
                    if xshape is not None and yshape is not None and xshape > yshape and yshape == 1:
                        tile_counts_part_list[axis] = xshape
                tile_counts[-xshapes_rank:] = tile_counts_part_list
                tiled_y = tf.tile(input_tensor_2, list(tile_counts)) * -1 * -1
                return input_tensor_1, tiled_y

            if xshapes_rank > yshapes_rank:
                tiled_x, input_tensor_2 = x_tile(
                    input_tensor_1=input_tensor_1,
                    input_tensor_2=input_tensor_2,
                    xshapes=xshapes,
                    xshapes_rank=xshapes_rank,
                    yshapes_rank=yshapes_rank,
                    yshape_list=yshape_list,
                )
                return tiled_x, input_tensor_2

            elif xshapes_rank < yshapes_rank:
                input_tensor_1, tiled_y =  y_tile(
                    input_tensor_1=input_tensor_1,
                    input_tensor_2=input_tensor_2,
                    yshapes=yshapes,
                    yshapes_rank=yshapes_rank,
                    xshapes_rank=xshapes_rank,
                    xshape_list=xshape_list,
                )
                return tiled_y, input_tensor_1

            elif xshapes_rank == yshapes_rank:
                # 1. Compare xshape_list from the end to get the position where [-(n-1)] > [-n].
                # 2. Compare yshape_list from the end to get the position where [-(n-1)] > [-n].
                # 3. Tile the dimension for which [-(n-1)] > [-n] first holds.
                x_mn2_large_mn1_index = -1
                y_mn2_large_mn1_index = -1
                for axis, dim in reversed(list(enumerate(xshape_list))):
                    if dim is not None and dim > x_mn2_large_mn1_index:
                        x_mn2_large_mn1_index = axis
                for axis, dim in reversed(list(enumerate(yshape_list))):
                    if dim is not None and dim > y_mn2_large_mn1_index:
                        y_mn2_large_mn1_index = axis

                if x_mn2_large_mn1_index == xshapes_rank - 1 and y_mn2_large_mn1_index == yshapes_rank - 1:
                    return input_tensor_1, input_tensor_2
                elif x_mn2_large_mn1_index != xshapes_rank - 1 and y_mn2_large_mn1_index == yshapes_rank - 1:
                    tiled_x, input_tensor_2 = x_tile(
                        input_tensor_1=input_tensor_1,
                        input_tensor_2=input_tensor_2,
                        xshapes=xshapes,
                        xshapes_rank=xshapes_rank,
                        yshapes_rank=yshapes_rank,
                        yshape_list=yshape_list,
                    )
                    return tiled_x, input_tensor_2
                elif x_mn2_large_mn1_index == xshapes_rank - 1 and y_mn2_large_mn1_index != yshapes_rank - 1:
                    input_tensor_1, tiled_y =  y_tile(
                        input_tensor_1=input_tensor_1,
                        input_tensor_2=input_tensor_2,
                        yshapes=yshapes,
                        yshapes_rank=yshapes_rank,
                        xshapes_rank=xshapes_rank,
                        xshape_list=xshape_list,
                    )
                    return tiled_y, input_tensor_1
                elif x_mn2_large_mn1_index != xshapes_rank - 1 and y_mn2_large_mn1_index != yshapes_rank - 1:
                    tiled_x, input_tensor_2 = x_tile(
                        input_tensor_1=input_tensor_1,
                        input_tensor_2=input_tensor_2,
                        xshapes=xshapes,
                        xshapes_rank=xshapes_rank,
                        yshapes_rank=yshapes_rank,
                        yshape_list=yshape_list,
                    )
                    tiled_x, tiled_y = y_tile(
                        input_tensor_1=tiled_x,
                        input_tensor_2=input_tensor_2,
                        yshapes=yshapes,
                        yshapes_rank=yshapes_rank,
                        xshapes_rank=xshapes_rank,
                        xshape_list=xshape_list,
                    )
                    return tiled_x, tiled_y
    except Exception as ex:
        pass
    return input_tensor_1, input_tensor_2


def calc_tf_pooling_pads(input_shape, kernel, strides, input_tensor):
    """Calculate how much padding is needed for tensorflow mode 'SAME'.

    Parameters
    ----------
    input_shape: Union[np.ndarray, List]
        input tensor shape of pooling layer
    kernel: List
        kernel shape from onnx
    strides: List
        strides from onnx
    input_tensor: tf.Tensor
        input_tensor

    Returns
    -------
    same_pads: List
        onnx formatted padding, [x1_begin, x2_begin, ..., xn_begin, x1_end, x2_end, ..., xn_end]
    """

    same_pads = []
    same_pads_end = []

    undefined_dim_count = sum([1 if i is None else 0 for i in input_shape[1:-1]])

    if undefined_dim_count == 0:
        # calculate how much padding is needed except batch and channel dimension
        for i, k, s in zip(input_shape[1:-1], kernel, strides):
            same_output_shape = math.floor((i - 1) / s) + 1
            axis_pads = np.max((same_output_shape - 1) * s + k - i, 0)

            padded_valid_output_shape = math.floor((i + axis_pads - k) / s) + 1
            error_msg = Color.RED(f'ERROR:') + ' ' + f'Wrong padding calculation.'
            assert same_output_shape == padded_valid_output_shape, error_msg

            same_pads.append(axis_pads // 2)
            # pads to end more for odd number padding
            if axis_pads % 2:
                same_pads_end.append(axis_pads // 2 + 1)
            else:
                same_pads_end.append(axis_pads // 2)

        same_pads.extend(same_pads_end)

    else:
        # calculate how much padding is needed except batch and channel dimension
        input_shape_tensor = tf.shape(input_tensor)[1:-1]
        for i, k, s in zip(input_shape_tensor, kernel, strides):
            same_output_shape = tf.cast(tf.math.floor((i - 1) / s) + 1, dtype=tf.int32)
            axis_pads = tf.cast(tf.math.maximum((same_output_shape - 1) * s + k - i, 0), dtype=tf.int32)
            padded_valid_output_shape = tf.cast(tf.math.floor((i + axis_pads - k) / s) + 1, dtype=tf.int32)

            same_pads.append(axis_pads // 2)

            # pads to end more for odd number padding
            mod_padding = tf.math.mod(axis_pads, 2)
            same_pads.append(
                tf.cond(
                    pred=mod_padding > 0,
                    true_fn=lambda: same_pads_end.append(tf.math.floordiv(axis_pads, 2) + 1),
                    false_fn=lambda: same_pads_end.append(tf.math.floordiv(axis_pads, 2)),
                )
            )

    return same_pads


def calc_extra_padding_with_ceil(
        input_shape: Union[np.ndarray, List],
        kernel: Union[np.ndarray, List],
        pads: Union[np.ndarray, List],
        dilations: Union[np.ndarray, List],
        strides: Union[np.ndarray, List],
) -> List:
    """
    Calculate extra padding for ceil_mode enabled pooling layer

    Parameters
    ----------
    input_shape: Union[np.ndarray, List]
        input tensor shape of pooling layer except batch and channel dimension
    kernel: Union[np.ndarray, List]
        kernel size of pooling layer
    pads: Union[np.ndarray, List]
        onnx formatted padding, [x1_begin, x2_begin, ..., xn_begin, x1_end, x2_end, ..., xn_end]
    dilations: Union[np.ndarray, List]
        dilations of pooling layer
    strides: Union[np.ndarray, List]
        strides of pooling layer

    Returns
    -------
    extra_pads: List
        extra padding value to match output shape between onnx and tensorflow when ceil_mode == 1

    """

    if not len(input_shape) == len(kernel) == len(pads) // 2 == len(dilations) == len(strides):
        error_msg = f'' + \
                    Color.RED(f'ERROR:') + ' ' + \
                    f'Wrong input shapes for extra padding calculation.'
        print(error_msg)
        raise ValueError(error_msg)

    warn(
        f'Current pooling with ceil_mode = 1 follows pytorch implementation ' \
        f'since current onnx implementation generates nan values. ' \
        f'Please refer to https://github.com/PINTO0309/onnx2tf/issues/207.'
    )

    pads_begin = pads[:len(pads) // 2]
    pads_end = pads[len(pads) // 2:]
    pads_along_axis = [i + j for i, j in zip(pads_begin, pads_end)]

    output_spatial_shape = [
        (i + p - d * (k - 1) - 1) / s + 1
        for i, p, k, d, s in zip(input_shape, pads_along_axis, kernel, dilations, strides)
    ]

    output_shape_ceil = [math.ceil(output_shape) for output_shape in output_spatial_shape]
    last_stride_starts = [(o - 1) * s for o, s in zip(output_shape_ceil, strides)]

    # If last step is smaller than padding (no valid tensor in kernel), it is dropped.
    # This follows pytorch implementation since current onnx implementation generates nan values.
    # Please refer to https://github.com/PINTO0309/onnx2tf/issues/207
    # Determine whether the last stride contains any input or not
    last_stride_validity = [
        ls < (i + pb)
        for ls, k, i, pb
        in zip(last_stride_starts, kernel, input_shape, pads_begin)
    ]

    # Calculate extra pads to conduct one more stride if there is difference in output shape
    extra_pads = [
        ls + (k - 1) * d + 1 - (i + p)
        if valid else 0
        for valid, s, i, p, ls, k, d
        in zip(last_stride_validity, strides, input_shape, pads_along_axis, last_stride_starts, kernel, dilations)
    ]

    return extra_pads


def get_tf_model_inputs(
    *,
    tf_layers_dict: dict,
) -> List[Any]:
    """Get a list of input OPs for a TensorFlow model.

    Parameters
    ----------
    tf_layers_dict: dict
        Graph structure of TensorFlow models

    Returns
    -------
    tf_model_inputs: List
        List of input OPs for TensorFlow model
    """
    tf_model_inputs = [
        layer_info['op'] \
            for layer_info in tf_layers_dict.values() \
                if layer_info['optype'] == 'Input'
    ]
    return tf_model_inputs


def get_tf_model_outputs(
    *,
    tf_layers_dict: dict,
    output_names: List[str],
) -> List[Any]:
    """Get a list of output OPs for a TensorFlow model.

    Parameters
    ----------
    tf_layers_dict: dict
        Graph structure of TensorFlow models

    output_names: List[str]
        Name of ONNX output OP to be extracted

    Returns
    -------
    tf_model_outputs: List
        List of output OPs for TensorFlow model
    """
    tf_model_outputs = []
    for name in output_names:
        if name in tf_layers_dict:
            op = tf_layers_dict[name]
            tf_model_outputs.append(op['tf_node'])
    return tf_model_outputs


def rewrite_tflite_inout_opname(
    *,
    output_folder_path: str,
    tflite_file_name: str,
    onnx_input_names: List[str],
    onnx_output_names: List[str],
    onnx_graph_input_shapes: List[List[int |str]],
    onnx_graph_output_shapes: List[List[int |str]],
):
    """Rewrite the input/output OP name of tflite to the input/output OP name of ONNX.
    Pre-installation of flatc is required.

    Parameters
    ----------
    output_folder_path: str
        Path of the folder where the tflite file to be rewritten is stored

    tflite_file_name: str
        Name of tflite file to be rewritten

    onnx_input_names: List[str]
        List of ONNX input OP names

    onnx_output_names: List[str]
        List of ONNX output OP names

    onnx_graph_input_shapes: List[List[int |str]]
        List of ONNX input OP shapes

    onnx_graph_output_shapes: List[List[int |str]]
        List of ONNX output OP shapes
    """
    try:
        # Check to see if flatc is installed
        result = subprocess.check_output(
            [
                'flatc', '--version'
            ],
            stderr=subprocess.PIPE
        ).decode('utf-8')

        # Download schema.fbs if it does not exist
        if not os.path.isfile(f'{output_folder_path}/schema.fbs'):
            result = subprocess.check_output(
                [
                    'curl',
                    'https://raw.githubusercontent.com/tensorflow/tensorflow/v2.17.0-rc1/tensorflow/compiler/mlir/lite/schema/schema.fbs',
                    '-o',
                    f'{output_folder_path}/schema.fbs'
                ],
                stderr=subprocess.PIPE
            ).decode('utf-8')

        # Generate Python API from schema
        result = subprocess.check_output(
            [
                'flatc', '-t',
                '--python',
                '--gen-object-api',
                '--gen-onefile',
                'schema.fbs',
            ],
            stderr=subprocess.PIPE,
            cwd=output_folder_path
        ).decode('utf-8')
        schema_tflite = {}
        with open(f'{output_folder_path}/schema_generated.py') as f:
            exec(f.read(), schema_tflite)

        tflite_file_path = f'{output_folder_path}/{tflite_file_name}'
        with open(tflite_file_path, 'rb') as tflite_file:
            model_bytes = tflite_file.read()
        flat_model = schema_tflite['ModelT'].InitFromObj(
            schema_tflite['Model'].GetRootAs(model_bytes))

        flat_subgraphs = flat_model.subgraphs[0]
        flat_tensors = flat_subgraphs.tensors
        flat_input_nums: List[int] = flat_subgraphs.inputs
        flat_output_nums: List[int] = flat_subgraphs.outputs
        flat_input_infos = [flat_tensors[idx] for idx in flat_input_nums]
        flat_output_infos = [flat_tensors[idx] for idx in flat_output_nums]

        # Determination of the number of inputs/outputs of the same shape
        # Correct name discrepancies based on shape if multiple inputs/outputs shapes do not overlap
        # However, if there are inputs/outputs containing undefined dimensions,
        # workaround is skipped because correction is not possible.
        # https://github.com/PINTO0309/onnx2tf/issues/650
        inputs_second_dim_elements = [
            tuple(onnx_graph_input_shape) \
                for onnx_graph_input_shape in onnx_graph_input_shapes
        ]
        inputs_has_duplicates = len(inputs_second_dim_elements) != len(set(inputs_second_dim_elements))
        inputs_has_undefined_dim = any(isinstance(item, str) for onnx_graph_input_shape in onnx_graph_input_shapes for item in onnx_graph_input_shape)

        outputs_second_dim_elements = [
            tuple(onnx_graph_output_shape) \
                for onnx_graph_output_shape in onnx_graph_output_shapes
        ]
        outputs_has_duplicates = len(outputs_second_dim_elements) != len(set(outputs_second_dim_elements))
        outputs_has_undefined_dim = any(isinstance(item, str) for onnx_graph_output_shape in onnx_graph_output_shapes for item in onnx_graph_output_shape)

        # INPUT
        if not inputs_has_duplicates and not inputs_has_undefined_dim:
            for onnx_input_name, onnx_input_shape in zip(onnx_input_names, onnx_graph_input_shapes):
                for flat_input_info in flat_input_infos:
                    if np.prod(onnx_input_shape) == np.prod(list(flat_input_info.shape)):
                        flat_input_info.name = onnx_input_name
                        break
        else:
            for idx, flat_input_info in enumerate(flat_input_infos):
                flat_input_info.name = onnx_input_names[idx]

        # OUTPUT
        if not outputs_has_duplicates and not outputs_has_undefined_dim:
            for onnx_output_name, onnx_output_shape in zip(onnx_output_names, onnx_graph_output_shapes):
                for flat_output_info in flat_output_infos:
                    if np.prod(onnx_output_shape) == np.prod(list(flat_output_info.shape)):
                        flat_output_info.name = onnx_output_name
                        break
        else:
            for idx, flat_output_info in enumerate(flat_output_infos):
                flat_output_info.name = onnx_output_names[idx]

        if inputs_has_duplicates or inputs_has_undefined_dim or outputs_has_duplicates or outputs_has_undefined_dim:
            warn('Carefully check the output .tflite as the order of input OP names and output OP names may have been corrupted by TensorFlow.')

        # make signature_defs
        """
        "signature_defs": [
            {
                "inputs": [
                    {
                        "name": "input",
                        "tensor_index": 0
                    }
                ],
                "outputs": [
                    {
                        "name": "boxes",
                        "tensor_index": 208
                    },
                    {
                        "name": "scores",
                        "tensor_index": 190
                    }
                ],
                "signature_key": "serving_default",
                "subgraph_index": 0
            }
        ]
        """
        signature_defs = schema_tflite['SignatureDefT']()
        # signature_defs_inputs
        signature_defs_inputs = []
        for idx, flat_input_info in enumerate(flat_input_infos):
            tm = schema_tflite["TensorMapT"]()
            tm.name = onnx_input_names[idx]
            tm.tensorIndex = flat_input_info.buffer - 1
            signature_defs_inputs.append(tm)
        signature_defs.inputs = signature_defs_inputs
        # signature_defs_outputs
        signature_defs_outputs = []
        for idx, flat_output_info in enumerate(flat_output_infos):
            tm = schema_tflite["TensorMapT"]()
            tm.name = onnx_output_names[idx]
            tm.tensorIndex = flat_output_info.buffer - 1
            signature_defs_outputs.append(tm)
        signature_defs.outputs = signature_defs_outputs
        # signature_defs_inputs
        signature_defs.signatureKey = 'serving_default'
        # subgraph_index
        signature_defs.subgraphIndex = 0
        # update model
        flat_model.signatureDefs = [signature_defs]

        if flat_model is not None:
            builder = flatbuffers.Builder()
            model_offset = flat_model.Pack(builder)
            builder.Finish(model_offset, file_identifier=b'TFL3')
            model_bytes = bytes(builder.Output())
            with open(tflite_file_path, 'wb') as f:
                f.write(model_bytes)

    except Exception as ex:
        warn(
            'If you want tflite input OP name and output OP name ' +
            'to match ONNX input and output names, ' +
            'convert them after installing "flatc". ' +
            'Also, do not use symbols such as slashes in input/output OP names. ' +
            'To install flatc, run the following command:\n' +
            'wget https://github.com/PINTO0309/onnx2tf/releases/download/1.16.31/flatc.tar.gz' +
            ' && tar -zxvf flatc.tar.gz && sudo chmod +x flatc && sudo mv flatc /usr/bin/'
        )


def make_tf_partial_model_inputs(
    *,
    input_tensors: List[Any],
) -> List[tf_keras.Input]:
    """Generate input OPs for TensorFlow subgraph generation.

    Parameters
    ----------
    input_tensors: List[Any]
        List of input tensor

    Returns
    -------
    inputs: List[tf_keras.Input]
        List of tf_keras.Input
    """
    # Generate input OPs for TensorFlow subgraphs
    # For inference testing on OP stand-alone
    tf_partial_model_input_shapes = []
    tf_partial_model_input_dtypes = []
    for input_tensor in input_tensors:
        if input_tensor.shape is None \
            or input_tensor.shape == tf.TensorShape(None):
            return None
        else:
            tf_partial_model_input_shape = [dim for dim in input_tensor.shape]
            if None in tf_partial_model_input_shape:
                return None
            tf_partial_model_input_shapes.append(
                tf_partial_model_input_shape
            )
            tf_partial_model_input_dtypes.append(
                NUMPY_DTYPES_TO_TF_DTYPES[input_tensor.dtype] \
                    if isinstance(input_tensor.dtype, np.dtype) else input_tensor.dtype
            )

    inputs: List[tf_keras.Input] = []
    input = None
    for idx, input_shape in enumerate(tf_partial_model_input_shapes):
        if isinstance(input_shape, list) and len(input_shape) == 0:
            tf_partial_model_input_shapes[idx] = [1]
    for input_shape, input_dtype in zip(tf_partial_model_input_shapes, tf_partial_model_input_dtypes):
        if len(input_shape) == 1:
            input = tf_keras.Input(
                shape=input_shape[0] if isinstance(input_shape[0], int) else None,
                batch_size=1,
                dtype=input_dtype,
            )
        elif len(input_shape) >= 2:
            input = tf_keras.Input(
                shape=[
                    inp if isinstance(inp, int) else None for inp in input_shape[1:]
                ],
                batch_size=input_shape[0] if isinstance(input_shape[0], int) else None,
                dtype=input_dtype,
            )
        inputs.append(input)
    return inputs


def merge_two_consecutive_identical_ops_into_one(
    graph_node_input_1: Any,
    graph_node_input_2: Any,
    graph_node_output: gs.Variable,
    before_op_output_shape_trans: bool,
    input_tensor_1: Any,
    input_tensor_2: Any,
    graph_node: gs.Node,
    tf_layers_dict: Dict,
    tf_func: str,
) -> Tuple[dict, Any]:
    """Merges two consecutive add/subtract operations into one.

    Parameters
    ----------
    graph_node_input_1: Any
        Input Node X of ONNX

    graph_node_input_2: Any
        Input Node Y of ONNX

    graph_node_output: gs.Variable
        Output of ONNX

    before_op_output_shape_trans: bool
        Flag to determine if the input tensor needs to be transposed

    input_tensor_1: Any
        Input Node X of TensorFlow

    input_tensor_2: Any
        Input Node Y of TensorFlow

    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: Dict
        TensorFlow pre-built graphs

    tf_func: str
        'Mul' or 'Div' or 'Add' or 'Sub'

    Returns
    -------
    inputs: Tuple[dict, Any]
        tf_layers_dict
        tf_type: tf.identity or tf.math.multiply or tf.math.divide or tf.math.add or tf.math.subtract
    """
    # Merge two consecutive identical OPs into one
    #   A constant is calculated in advance only
    #   when one of the operations of the current OP
    #   is a constant and one of the operations of
    #   the next OP is also a constant.
    # By merging two OPs into one, an accuracy error always occurs
    # in the merged OP during the accuracy check.
    # 1. `Mul` -> `Mul` to `Single-Mul` : `10 * 5 * 8 -> 10 * 40`
    # 2. `Mul` -> `Div` to `Single-Mul` : `10 * 5 / 8 -> 10 * 0.625`
    # 3. `Div` -> `Mul` to `Single-Mul` : `10 / 5 * 8 -> 10 * 1.6`
    # 4. `Div` -> `Div` to `Single-Mul` : `10 / 5 / 8 -> 10 * 0.025`
    # 5. `Sub` -> `Sub` to `Single-Sub` : `10 - 5 - 8 -> 10 - 13`
    # 6. `Sub` -> `Add` to `Single-Sub` : `10 - 5 + 8 -> 10 + 3`
    # 7. `Add` -> `Add` to `Single-Add`  : `10 + 5 + 8 -> 10 + 13`
    # 8. `Add` -> `Sub` to `Single-Add`  : `10 + 5 - 8 -> 10 - 3`

    tf_type = None
    if tf_func == 'Mul':
        if (
            not isinstance(graph_node_input_1, np.ndarray) \
                and 'merge_mul' in tf_layers_dict[graph_node_input_1.name] \
                and tf_layers_dict[graph_node_input_1.name]['merge_mul']
            ) \
            or (
                not isinstance(graph_node_input_2, np.ndarray) \
                and 'merge_mul' in tf_layers_dict[graph_node_input_2.name] \
                    and tf_layers_dict[graph_node_input_2.name]['merge_mul']
            ) \
            or (
            not isinstance(graph_node_input_1, np.ndarray) \
                and 'merge_div' in tf_layers_dict[graph_node_input_1.name] \
                and tf_layers_dict[graph_node_input_1.name]['merge_div']
            ) \
            or (
                not isinstance(graph_node_input_2, np.ndarray) \
                and 'merge_div' in tf_layers_dict[graph_node_input_2.name] \
                    and tf_layers_dict[graph_node_input_2.name]['merge_div']
            ):
            if not isinstance(input_tensor_1, np.ndarray) \
                and not hasattr(input_tensor_1, 'numpy'):
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.identity(input=input_tensor_1)
            elif not isinstance(input_tensor_2, np.ndarray) \
                and not hasattr(input_tensor_2, 'numpy'):
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.identity(input=input_tensor_2)
            tf_type = tf.math.multiply
        else:
            if isinstance(input_tensor_1, np.ndarray) \
                or hasattr(input_tensor_1, 'numpy') \
                or isinstance(input_tensor_2, np.ndarray) \
                or hasattr(input_tensor_2, 'numpy'):
                try:
                    workaround_exec_flg = False
                    try:
                        # Check whether there is one, two, or more OPs to connect next.
                        graph_node.o(consumer_idx=1)
                    except Exception as ex:
                        # Error == Only one node connected next
                        workaround_exec_flg = True
                    if workaround_exec_flg:
                        next_graph_node_o = graph_node.o()
                        next_graph_node_o_op = next_graph_node_o.op
                        if next_graph_node_o_op in ['Mul', 'Div']:
                            next_graph_node_input_1 = get_constant_or_variable(
                                next_graph_node_o.inputs[0],
                                before_op_output_shape_trans,
                            )
                            next_graph_node_input_2 = get_constant_or_variable(
                                next_graph_node_o.inputs[1],
                                before_op_output_shape_trans,
                            )
                        if next_graph_node_o_op == 'Mul':
                            # 1. `Mul` -> `Mul` to `Single-Mul` : `10 * 5 * 8 -> 10 * 40`
                            if isinstance(next_graph_node_input_1, np.ndarray) or hasattr(next_graph_node_input_1, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = input_tensor_1 * next_graph_node_input_1
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = input_tensor_2 * next_graph_node_input_1
                                tf_layers_dict[graph_node_output.name]['merge_mul'] = True
                                tf_type = tf.identity
                            elif isinstance(next_graph_node_input_2, np.ndarray) or hasattr(next_graph_node_input_2, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = input_tensor_1 * next_graph_node_input_2
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = input_tensor_2 * next_graph_node_input_2
                                tf_layers_dict[graph_node_output.name]['merge_mul'] = True
                                tf_type = tf.identity
                            else:
                                tf_type = tf.math.multiply

                        elif next_graph_node_o_op == 'Div':
                            # 2. `Mul` -> `Div` to `Single-Mul` : `10 * 5 / 8 -> 10 * 0.625`
                            if isinstance(next_graph_node_input_1, np.ndarray) or hasattr(next_graph_node_input_1, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = input_tensor_1 / next_graph_node_input_1
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = input_tensor_2 / next_graph_node_input_1
                                tf_layers_dict[graph_node_output.name]['merge_mul'] = True
                                tf_type = tf.identity
                            elif isinstance(next_graph_node_input_2, np.ndarray) or hasattr(next_graph_node_input_2, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = input_tensor_1 / next_graph_node_input_2
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = input_tensor_2 / next_graph_node_input_2
                                tf_layers_dict[graph_node_output.name]['merge_mul'] = True
                                tf_type = tf.identity
                            else:
                                tf_type = tf.math.multiply
                        else:
                            tf_type = tf.math.multiply
                    else:
                        tf_type = tf.math.multiply

                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.math.multiply(
                            x=input_tensor_1 \
                                if not isinstance(input_tensor_1, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_1),
                            y=input_tensor_2 \
                                if not isinstance(input_tensor_2, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_2),
                            name=graph_node.name,
                        )

                except Exception as ex:
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.math.multiply(
                            x=input_tensor_1 \
                                if not isinstance(input_tensor_1, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_1),
                            y=input_tensor_2 \
                                if not isinstance(input_tensor_2, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_2),
                            name=graph_node.name,
                        )
                    tf_type = tf.math.multiply
            else:
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.math.multiply(
                        x=input_tensor_1 \
                            if not isinstance(input_tensor_1, np.ndarray) \
                                else tf.convert_to_tensor(input_tensor_1),
                        y=input_tensor_2 \
                            if not isinstance(input_tensor_2, np.ndarray) \
                                else tf.convert_to_tensor(input_tensor_2),
                        name=graph_node.name,
                    )
                tf_type = tf.math.multiply

    elif tf_func == 'Div':
        if (
            not isinstance(graph_node_input_1, np.ndarray) \
                and 'merge_div' in tf_layers_dict[graph_node_input_1.name] \
                and tf_layers_dict[graph_node_input_1.name]['merge_div']
            ) \
            or (
                not isinstance(graph_node_input_2, np.ndarray) \
                and 'merge_div' in tf_layers_dict[graph_node_input_2.name] \
                    and tf_layers_dict[graph_node_input_2.name]['merge_div']
            ) \
            or (
            not isinstance(graph_node_input_1, np.ndarray) \
                and 'merge_mul' in tf_layers_dict[graph_node_input_1.name] \
                and tf_layers_dict[graph_node_input_1.name]['merge_mul']
            ) \
            or (
                not isinstance(graph_node_input_2, np.ndarray) \
                and 'merge_mul' in tf_layers_dict[graph_node_input_2.name] \
                    and tf_layers_dict[graph_node_input_2.name]['merge_mul']
            ):
            if not isinstance(input_tensor_1, np.ndarray) \
                and not hasattr(input_tensor_1, 'numpy'):
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.identity(input=input_tensor_1)
            elif not isinstance(input_tensor_2, np.ndarray) \
                and not hasattr(input_tensor_2, 'numpy'):
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.identity(input=input_tensor_2)
            tf_type = tf.multiply
        else:
            if isinstance(input_tensor_1, np.ndarray) \
                or hasattr(input_tensor_1, 'numpy') \
                or isinstance(input_tensor_2, np.ndarray) \
                or hasattr(input_tensor_2, 'numpy'):
                try:
                    workaround_exec_flg = False
                    try:
                        # Check whether there is one, two, or more OPs to connect next.
                        graph_node.o(consumer_idx=1)
                    except Exception as ex:
                        # Error == Only one node connected next
                        workaround_exec_flg = True
                    if workaround_exec_flg:
                        next_graph_node_o = graph_node.o()
                        next_graph_node_o_op = next_graph_node_o.op
                        if next_graph_node_o_op in ['Mul', 'Div']:
                            next_graph_node_input_1 = get_constant_or_variable(
                                next_graph_node_o.inputs[0],
                                before_op_output_shape_trans,
                            )
                            next_graph_node_input_2 = get_constant_or_variable(
                                next_graph_node_o.inputs[1],
                                before_op_output_shape_trans,
                            )
                        if next_graph_node_o_op == 'Mul':
                            # 3. `Div` -> `Mul` to `Single-Mul` : `10 / 5 * 8 -> 10 * 1.6`
                            if isinstance(next_graph_node_input_1, np.ndarray) or hasattr(next_graph_node_input_1, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = \
                                        np.asarray(1.0, dtype=next_graph_node_input_1.dtype) / (input_tensor_1 / next_graph_node_input_1)
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = \
                                        np.asarray(1.0, dtype=next_graph_node_input_1.dtype) / (input_tensor_2 / next_graph_node_input_1)
                                tf_layers_dict[graph_node_output.name]['merge_div'] = True
                                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                                    tf.math.multiply(
                                        x=input_tensor_1 \
                                            if not isinstance(input_tensor_1, np.ndarray) \
                                                else tf.convert_to_tensor(input_tensor_1),
                                        y=input_tensor_2 \
                                            if not isinstance(input_tensor_2, np.ndarray) \
                                                else tf.convert_to_tensor(input_tensor_2),
                                        name=graph_node.name,
                                    )
                                tf_type = tf.identity
                            elif isinstance(next_graph_node_input_2, np.ndarray) or hasattr(next_graph_node_input_2, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = \
                                        np.asarray(1.0, dtype=next_graph_node_input_2.dtype) / (input_tensor_1 / next_graph_node_input_2)
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = \
                                        np.asarray(1.0, dtype=next_graph_node_input_2.dtype) / (input_tensor_2 / next_graph_node_input_2)
                                tf_layers_dict[graph_node_output.name]['merge_div'] = True
                                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                                    tf.math.multiply(
                                        x=input_tensor_1 \
                                            if not isinstance(input_tensor_1, np.ndarray) \
                                                else tf.convert_to_tensor(input_tensor_1),
                                        y=input_tensor_2 \
                                            if not isinstance(input_tensor_2, np.ndarray) \
                                                else tf.convert_to_tensor(input_tensor_2),
                                        name=graph_node.name,
                                    )
                                tf_type = tf.identity
                            else:
                                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                                    tf.math.multiply(
                                        x=input_tensor_1 \
                                            if not isinstance(input_tensor_1, np.ndarray) \
                                                else tf.convert_to_tensor(input_tensor_1),
                                        y=input_tensor_2 \
                                            if not isinstance(input_tensor_2, np.ndarray) \
                                                else tf.convert_to_tensor(input_tensor_2),
                                        name=graph_node.name,
                                    )
                                tf_type = tf.identity

                        elif next_graph_node_o_op == 'Div':
                            # 4. `Div` -> `Div` to `Single-Nul` : `10 / 5 / 8 -> 10 * 0.025`
                            if isinstance(next_graph_node_input_1, np.ndarray) or hasattr(next_graph_node_input_1, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = \
                                        np.asarray(1.0, dtype=next_graph_node_input_1.dtype) / (input_tensor_1 * next_graph_node_input_1)
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = \
                                        np.asarray(1.0, dtype=next_graph_node_input_1.dtype) / (input_tensor_2 * next_graph_node_input_1)
                                tf_layers_dict[graph_node_output.name]['merge_div'] = True
                                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                                    tf.math.multiply(
                                        x=input_tensor_1 \
                                            if not isinstance(input_tensor_1, np.ndarray) \
                                                else tf.convert_to_tensor(input_tensor_1),
                                        y=input_tensor_2 \
                                            if not isinstance(input_tensor_2, np.ndarray) \
                                                else tf.convert_to_tensor(input_tensor_2),
                                        name=graph_node.name,
                                    )
                                tf_type = tf.identity
                            elif isinstance(next_graph_node_input_2, np.ndarray) or hasattr(next_graph_node_input_2, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = \
                                        np.asarray(1.0, dtype=next_graph_node_input_2.dtype) / (input_tensor_1 * next_graph_node_input_2)
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = \
                                        np.asarray(1.0, dtype=next_graph_node_input_2.dtype) / (input_tensor_2 * next_graph_node_input_2)
                                tf_layers_dict[graph_node_output.name]['merge_div'] = True
                                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                                    tf.math.multiply(
                                        x=input_tensor_1 \
                                            if not isinstance(input_tensor_1, np.ndarray) \
                                                else tf.convert_to_tensor(input_tensor_1),
                                        y=input_tensor_2 \
                                            if not isinstance(input_tensor_2, np.ndarray) \
                                                else tf.convert_to_tensor(input_tensor_2),
                                        name=graph_node.name,
                                    )
                                tf_type = tf.identity
                            else:
                                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                                    tf.math.divide(
                                        x=input_tensor_1 \
                                            if not isinstance(input_tensor_1, np.ndarray) \
                                                else tf.convert_to_tensor(input_tensor_1),
                                        y=input_tensor_2 \
                                            if not isinstance(input_tensor_2, np.ndarray) \
                                                else tf.convert_to_tensor(input_tensor_2),
                                        name=graph_node.name,
                                    )
                                tf_type = tf.math.divide
                        else:
                            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                                tf.math.divide(
                                    x=input_tensor_1 \
                                        if not isinstance(input_tensor_1, np.ndarray) \
                                            else tf.convert_to_tensor(input_tensor_1),
                                    y=input_tensor_2 \
                                        if not isinstance(input_tensor_2, np.ndarray) \
                                            else tf.convert_to_tensor(input_tensor_2),
                                    name=graph_node.name,
                                )
                            tf_type = tf.math.divide
                    else:
                        tf_layers_dict[graph_node_output.name]['tf_node'] = \
                            tf.math.divide(
                                x=input_tensor_1 \
                                    if not isinstance(input_tensor_1, np.ndarray) \
                                        else tf.convert_to_tensor(input_tensor_1),
                                y=input_tensor_2 \
                                    if not isinstance(input_tensor_2, np.ndarray) \
                                        else tf.convert_to_tensor(input_tensor_2),
                                name=graph_node.name,
                            )
                        tf_type = tf.math.divide

                except Exception as ex:
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.math.divide(
                            x=input_tensor_1 \
                                if not isinstance(input_tensor_1, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_1),
                            y=input_tensor_2 \
                                if not isinstance(input_tensor_2, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_2),
                            name=graph_node.name,
                        )
                    tf_type = tf.math.divide
            else:
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.math.divide(
                        x=input_tensor_1 \
                            if not isinstance(input_tensor_1, np.ndarray) \
                                else tf.convert_to_tensor(input_tensor_1),
                        y=input_tensor_2 \
                            if not isinstance(input_tensor_2, np.ndarray) \
                                else tf.convert_to_tensor(input_tensor_2),
                        name=graph_node.name,
                    )
                tf_type = tf.math.divide

    elif tf_func == 'Sub':
        if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.math.subtract(
                    x=input_tensor_1 \
                        if not isinstance(input_tensor_1, np.ndarray) \
                            else tf.convert_to_tensor(input_tensor_1),
                    y=input_tensor_2 \
                        if not isinstance(input_tensor_2, np.ndarray) \
                            else tf.convert_to_tensor(input_tensor_2),
                    name=graph_node.name,
                )
            tf_type = tf.math.subtract
        elif (
            not isinstance(graph_node_input_1, np.ndarray) \
                and 'merge_sub' in tf_layers_dict[graph_node_input_1.name] \
                and tf_layers_dict[graph_node_input_1.name]['merge_sub']
            ) \
            or (
                not isinstance(graph_node_input_2, np.ndarray) \
                and 'merge_sub' in tf_layers_dict[graph_node_input_2.name] \
                    and tf_layers_dict[graph_node_input_2.name]['merge_sub']
            ) \
            or (
                not isinstance(graph_node_input_1, np.ndarray) \
                and 'merge_add' in tf_layers_dict[graph_node_input_1.name] \
                and tf_layers_dict[graph_node_input_1.name]['merge_add']
            ) \
            or (
                not isinstance(graph_node_input_2, np.ndarray) \
                and 'merge_add' in tf_layers_dict[graph_node_input_2.name] \
                    and tf_layers_dict[graph_node_input_2.name]['merge_add']
            ):
            if not isinstance(input_tensor_1, np.ndarray) \
                and not hasattr(input_tensor_1, 'numpy'):
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.identity(input=input_tensor_1)
            elif not isinstance(input_tensor_2, np.ndarray) \
                and not hasattr(input_tensor_2, 'numpy'):
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.identity(input=input_tensor_2)
            tf_type = tf.math.subtract
        else:
            if isinstance(input_tensor_1, np.ndarray) \
                or hasattr(input_tensor_1, 'numpy') \
                or isinstance(input_tensor_2, np.ndarray) \
                or hasattr(input_tensor_2, 'numpy'):
                try:
                    workaround_exec_flg = False
                    try:
                        # Check whether there is one, two, or more OPs to connect next.
                        graph_node.o(consumer_idx=1)
                    except Exception as ex:
                        # Error == Only one node connected next
                        workaround_exec_flg = True
                    if workaround_exec_flg:
                        next_graph_node_o = graph_node.o()
                        next_graph_node_o_op = next_graph_node_o.op
                        if next_graph_node_o_op in ['Sub', 'Add']:
                            next_graph_node_input_1 = get_constant_or_variable(
                                next_graph_node_o.inputs[0],
                                before_op_output_shape_trans,
                            )
                            next_graph_node_input_2 = get_constant_or_variable(
                                next_graph_node_o.inputs[1],
                                before_op_output_shape_trans,
                            )
                        if next_graph_node_o_op == 'Sub':
                            # 5. `Sub` -> `Sub` to `Single-Sub` : `10 - 5 - 8 -> 10 - 13`
                            if isinstance(next_graph_node_input_1, np.ndarray) or hasattr(next_graph_node_input_1, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = (input_tensor_1 + next_graph_node_input_1)
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = (input_tensor_2 + next_graph_node_input_1)
                                tf_layers_dict[graph_node_output.name]['merge_sub'] = True
                                tf_type = tf.identity
                            elif isinstance(next_graph_node_input_2, np.ndarray) or hasattr(next_graph_node_input_2, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = (input_tensor_1 + next_graph_node_input_2)
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = (input_tensor_2 + next_graph_node_input_2)
                                tf_layers_dict[graph_node_output.name]['merge_sub'] = True
                                tf_type = tf.identity
                            else:
                                tf_type = tf.math.subtract

                        elif next_graph_node_o_op == 'Add':
                            # 6. `Sub` -> `Add` to `Single-Sub` : `10 - 5 + 8 -> 10 + 3`
                            if isinstance(next_graph_node_input_1, np.ndarray) or hasattr(next_graph_node_input_1, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = (input_tensor_1 - next_graph_node_input_1)
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = (input_tensor_2 - next_graph_node_input_1)
                                tf_layers_dict[graph_node_output.name]['merge_sub'] = True
                                tf_type = tf.identity
                            elif isinstance(next_graph_node_input_2, np.ndarray) or hasattr(next_graph_node_input_2, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = (input_tensor_1 - next_graph_node_input_2)
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = (input_tensor_2 - next_graph_node_input_2)
                                tf_layers_dict[graph_node_output.name]['merge_sub'] = True
                                tf_type = tf.identity
                            else:
                                tf_type = tf.math.subtract
                        else:
                            tf_type = tf.math.subtract
                    else:
                        tf_type = tf.math.subtract

                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.math.subtract(
                            x=input_tensor_1 \
                                if not isinstance(input_tensor_1, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_1),
                            y=input_tensor_2 \
                                if not isinstance(input_tensor_2, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_2),
                            name=graph_node.name,
                        )

                except Exception as ex:
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.math.subtract(
                            x=input_tensor_1 \
                                if not isinstance(input_tensor_1, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_1),
                            y=input_tensor_2 \
                                if not isinstance(input_tensor_2, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_2),
                            name=graph_node.name,
                        )
                    tf_type = tf.math.subtract
            else:
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.math.subtract(
                        x=input_tensor_1 \
                            if not isinstance(input_tensor_1, np.ndarray) \
                                else tf.convert_to_tensor(input_tensor_1),
                        y=input_tensor_2 \
                            if not isinstance(input_tensor_2, np.ndarray) \
                                else tf.convert_to_tensor(input_tensor_2),
                        name=graph_node.name,
                    )
                tf_type = tf.math.subtract

    elif tf_func == 'Add':
        if (
            not isinstance(graph_node_input_1, np.ndarray) \
                and 'merge_add' in tf_layers_dict[graph_node_input_1.name] \
                and tf_layers_dict[graph_node_input_1.name]['merge_add']
            ) \
            or (
                not isinstance(graph_node_input_2, np.ndarray) \
                and 'merge_add' in tf_layers_dict[graph_node_input_2.name] \
                    and tf_layers_dict[graph_node_input_2.name]['merge_add']
            ) \
            or (
                not isinstance(graph_node_input_1, np.ndarray) \
                and 'merge_sub' in tf_layers_dict[graph_node_input_1.name] \
                and tf_layers_dict[graph_node_input_1.name]['merge_sub']
            ) \
            or (
                not isinstance(graph_node_input_2, np.ndarray) \
                and 'merge_sub' in tf_layers_dict[graph_node_input_2.name] \
                    and tf_layers_dict[graph_node_input_2.name]['merge_sub']
            ):
            if not isinstance(input_tensor_1, np.ndarray) \
                and not hasattr(input_tensor_1, 'numpy'):
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.identity(input=input_tensor_1)
            elif not isinstance(input_tensor_2, np.ndarray) \
                and not hasattr(input_tensor_2, 'numpy'):
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.identity(input=input_tensor_2)
            tf_type = tf.math.add
        else:
            if isinstance(input_tensor_1, np.ndarray) \
                or hasattr(input_tensor_1, 'numpy') \
                or isinstance(input_tensor_2, np.ndarray) \
                or hasattr(input_tensor_2, 'numpy'):
                try:
                    workaround_exec_flg = False
                    try:
                        # Check whether there is one, two, or more OPs to connect next.
                        graph_node.o(consumer_idx=1)
                    except Exception as ex:
                        # Error == Only one node connected next
                        workaround_exec_flg = True
                    if workaround_exec_flg:
                        next_graph_node_o = graph_node.o()
                        next_graph_node_o_op = next_graph_node_o.op
                        if next_graph_node_o_op in ['Sub', 'Add']:
                            next_graph_node_input_1 = get_constant_or_variable(
                                next_graph_node_o.inputs[0],
                                before_op_output_shape_trans,
                            )
                            next_graph_node_input_2 = get_constant_or_variable(
                                next_graph_node_o.inputs[1],
                                before_op_output_shape_trans,
                            )
                        if next_graph_node_o_op == 'Add':
                            # 7. `Add` -> `Add` to `Single-Add`  : `10 + 5 + 8 -> 10 + 13`
                            if isinstance(next_graph_node_input_1, np.ndarray) or hasattr(next_graph_node_input_1, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = (input_tensor_1 + next_graph_node_input_1)
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = (input_tensor_2 + next_graph_node_input_1)
                                tf_layers_dict[graph_node_output.name]['merge_add'] = True
                                tf_type = tf.identity
                            elif isinstance(next_graph_node_input_2, np.ndarray) or hasattr(next_graph_node_input_2, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = (input_tensor_1 + next_graph_node_input_2)
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = (input_tensor_2 + next_graph_node_input_2)
                                tf_layers_dict[graph_node_output.name]['merge_add'] = True
                                tf_type = tf.identity
                            else:
                                tf_type = tf.math.add

                        elif next_graph_node_o_op == 'Sub':
                            # 8. `Add` -> `Sub` to `Single-Add`  : `10 + 5 - 8 -> 10 - 3`
                            if isinstance(next_graph_node_input_1, np.ndarray) or hasattr(next_graph_node_input_1, 'numpy'):
                                tf_type = tf.math.add
                            elif isinstance(next_graph_node_input_2, np.ndarray) or hasattr(next_graph_node_input_2, 'numpy'):
                                if isinstance(input_tensor_1, np.ndarray) or hasattr(input_tensor_1, 'numpy'):
                                    input_tensor_1 = (input_tensor_1 - next_graph_node_input_2)
                                elif isinstance(input_tensor_2, np.ndarray) or hasattr(input_tensor_2, 'numpy'):
                                    input_tensor_2 = (input_tensor_2 - next_graph_node_input_2)
                                tf_layers_dict[graph_node_output.name]['merge_add'] = True
                                tf_type = tf.identity
                            else:
                                tf_type = tf.math.add
                        else:
                            tf_type = tf.math.add
                    else:
                        tf_type = tf.math.add

                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.math.add(
                            x=input_tensor_1 \
                                if not isinstance(input_tensor_1, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_1),
                            y=input_tensor_2 \
                                if not isinstance(input_tensor_2, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_2),
                            name=graph_node.name,
                        )

                except Exception as ex:
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.math.add(
                            x=input_tensor_1 \
                                if not isinstance(input_tensor_1, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_1),
                            y=input_tensor_2 \
                                if not isinstance(input_tensor_2, np.ndarray) \
                                    else tf.convert_to_tensor(input_tensor_2),
                            name=graph_node.name,
                        )
                    tf_type = tf.math.add
            else:
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.math.add(
                        x=input_tensor_1 \
                            if not isinstance(input_tensor_1, np.ndarray) \
                                else tf.convert_to_tensor(input_tensor_1),
                        y=input_tensor_2 \
                            if not isinstance(input_tensor_2, np.ndarray) \
                                else tf.convert_to_tensor(input_tensor_2),
                        name=graph_node.name,
                    )
                tf_type = tf.math.add

    return tf_layers_dict[graph_node_output.name]['tf_node'], tf_type


def deterring_shape_corruption_due_to_broadcast(
    graph_node_output_shape: List,
    input_tensor_1: Any,
    input_tensor_2: Any,
):
    """Deterring shape corruption due to broadcast.
    Transformer conversion stability improvement. (3D and below only)

    Parameters
    ----------
    graph_node_output_shape: List
        Output shape of ONNX OP

    input_tensor_1: Any
        Input Node X of TensorFlow

    input_tensor_2: Any
        Input Node Y of TensorFlow

    Returns
    -------
    input_tensor_1: Any
        Input Node X of TensorFlow

    input_tensor_2: Any
        Input Node Y of TensorFlow
    """
    # If the shape contains a string or None, skip all processing
    if graph_node_output_shape is None \
        or None in graph_node_output_shape \
        or sum([1 for dim in graph_node_output_shape if isinstance(dim, str)]) > 0:
        return input_tensor_1, input_tensor_2
    input_tensor_1_shape = input_tensor_1.shape
    if input_tensor_1_shape is None \
        or None in input_tensor_1_shape \
        or sum([1 for dim in input_tensor_1_shape if isinstance(dim, str)]) > 0:
        return input_tensor_1, input_tensor_2
    input_tensor_2_shape = input_tensor_2.shape
    if input_tensor_2_shape is None \
        or None in input_tensor_2_shape \
        or sum([1 for dim in input_tensor_2_shape if isinstance(dim, str)]) > 0:
        return input_tensor_1, input_tensor_2
    if len(input_tensor_1_shape) > 3 or len(input_tensor_2_shape) > 3:
        return input_tensor_1, input_tensor_2

    # Calculate the total number of elements in the onnx tensor
    onnx_output_elements = np.prod(graph_node_output_shape)

    # Perform dummy calculations to find out
    # the total number of tf tensor elements after the calculation
    try:
        dummy_tensor = input_tensor_1 * input_tensor_2
    except Exception as ex1:
        # To avoid Abort on shape mismatch error.
        tensor_1_candidate_for_transpositions = \
            list(itertools.permutations(range(len(input_tensor_1_shape)))) \
                if input_tensor_1_shape not in [tf.TensorShape([]), tf.TensorShape(None)] else [None]
        tensor_2_candidate_for_transpositions = \
            list(itertools.permutations(range(len(input_tensor_2_shape)))) \
                if input_tensor_2_shape not in [tf.TensorShape([]), tf.TensorShape(None)] else [None]
        for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
            for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
                try:
                    dummy_tensor = \
                        tf.transpose(a=input_tensor_1, perm=tensor_1_candidate_for_transposition) * \
                            tf.transpose(a=input_tensor_2, perm=tensor_2_candidate_for_transposition)
                    input_tensor_1 = tf.transpose(a=input_tensor_1, perm=tensor_1_candidate_for_transposition)
                    input_tensor_2 = tf.transpose(a=input_tensor_2, perm=tensor_2_candidate_for_transposition)
                    break
                except Exception as ex2:
                    pass
            else:
                continue
            break

    dummy_tensor_elements = np.prod(dummy_tensor.shape)

    # If the total number of output elements in ONNX matches the total number of
    # output elements in TensorFlow, the process is terminated
    if onnx_output_elements == dummy_tensor_elements:
        return input_tensor_1, input_tensor_2

    # If the total number of output elements in ONNX does not match the total number of
    # output elements in TensorFlow, search for an arrangement with matching geometry
    input_tensor_1_shape_perms = \
        list(itertools.permutations(range(len(input_tensor_1_shape)))) \
            if input_tensor_1_shape not in [tf.TensorShape([]), tf.TensorShape(None)] else [None]
    input_tensor_2_shape_perms = \
        list(itertools.permutations(range(len(input_tensor_2_shape)))) \
            if input_tensor_2_shape not in [tf.TensorShape([]), tf.TensorShape(None)] else [None]
    input_tensor_1_final_perm = None
    input_tensor_2_final_perm = None
    for input_tensor_1_shape_perm in input_tensor_1_shape_perms:
        for input_tensor_2_shape_perm in input_tensor_2_shape_perms:
            try:
                dummy_tensor = \
                    tf.transpose(a=input_tensor_1, perm=input_tensor_1_shape_perm) \
                        * tf.transpose(a=input_tensor_2, perm=input_tensor_2_shape_perm)
                dummy_tensor_elements = np.prod(dummy_tensor.shape)
                if onnx_output_elements == dummy_tensor_elements \
                    and graph_node_output_shape == list(dummy_tensor.shape):
                    input_tensor_1_final_perm = input_tensor_1_shape_perm
                    input_tensor_2_final_perm = input_tensor_2_shape_perm
                    break
            except Exception as ex:
                pass
        else:
            continue
        break
    # Transposition to error-free geometry
    if input_tensor_1_final_perm is not None \
        and input_tensor_2_final_perm is not None:
        input_tensor_1 = tf.transpose(
            a=input_tensor_1,
            perm=input_tensor_1_final_perm,
        )
        input_tensor_2 = tf.transpose(
            a=input_tensor_2,
            perm=input_tensor_2_final_perm,
        )

    return input_tensor_1, input_tensor_2



def acquisition_of_validation_data(
    *,
    input_tensor_1: Any,
    input_tensor_2: Any,
    graph_node_output: gs.Variable,
    tf_layers_dict: Dict,
    **kwargs: Dict,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Acquisition of Validation Data.

    Parameters
    ----------
    input_tensor_1: Any
        The output OP immediately before the OP to be verified or a constant.

    input_tensor_2: Any
        The output OP immediately before the OP to be verified or a constant.

    graph_node_output: gs.Variable
        ONNX OP output information

    tf_layers_dict: Dict
        TensorFlow Model Structure Dictionary

    Returns
    -------
    total_perm_combination_list: List[List[int]]
        Shape transposition pattern perms
    """
    onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = \
        kwargs['onnx_tensor_infos_for_validation']
    test_data_nhwc: np.ndarray = \
        kwargs['test_data_nhwc']
    custom_input_op_name_np_data_path: str = \
        kwargs['custom_input_op_name_np_data_path']

    # Get the output tensor of one previous OP of TensorFlow only once
    tf_model_inputs = get_tf_model_inputs(
        tf_layers_dict=tf_layers_dict,
    )
    val_model = None
    if not isinstance(input_tensor_1, np.ndarray) \
        and not hasattr(input_tensor_1, 'numpy') \
        and not isinstance(input_tensor_2, np.ndarray) \
        and not hasattr(input_tensor_2, 'numpy'):
        val_model = tf_keras.Model(
            inputs=tf_model_inputs,
            outputs=[
                input_tensor_1,
                input_tensor_2,
            ],
        )
    elif not isinstance(input_tensor_1, np.ndarray) \
        and not hasattr(input_tensor_1, 'numpy') \
        and isinstance(input_tensor_2, np.ndarray):
        val_model = tf_keras.Model(
            inputs=tf_model_inputs,
            outputs=[
                input_tensor_1
            ],
        )
    elif isinstance(input_tensor_1, np.ndarray) \
        and not isinstance(input_tensor_2, np.ndarray) \
        and not hasattr(input_tensor_2, 'numpy'):
        val_model = tf_keras.Model(
            inputs=tf_model_inputs,
            outputs=[
                input_tensor_2
            ],
        )

    else:
        # TODO: Error
        pass

    # TF dummy inference
    #   Get the output tensor of the previous layer of MatMul
    #   If input.1 and input.2 are both layers, tf_pre_tensor_infos is 2 cases
    #   If one of input.1 or input.2 is np.ndarray, tf_pre_tensor_infos is 1 case
    tf_pre_tensor_infos = {}
    try:
        tf_pre_tensor_infos: Dict[Any] = dummy_tf_inference(
            model=val_model,
            inputs=tf_model_inputs,
            test_data_nhwc=test_data_nhwc,
            custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
        )
    except Exception as ex:
        pass
    del val_model

    # Get np.ndarray for validation
    validation_data_1 = None
    validation_data_2 = None
    if len(tf_pre_tensor_infos) == 2:
        validation_data_1 = list(tf_pre_tensor_infos.values())[0]
        validation_data_2 = list(tf_pre_tensor_infos.values())[1]
    elif len(tf_pre_tensor_infos) == 1:
        if not isinstance(input_tensor_1, np.ndarray):
            validation_data_1 = list(tf_pre_tensor_infos.values())[0]
            validation_data_2 = copy.deepcopy(input_tensor_2)
        else:
            validation_data_1 = copy.deepcopy(input_tensor_1)
            validation_data_2 = list(tf_pre_tensor_infos.values())[0]

    # Get ONNX inference results
    onnx_tensor_infos = {}
    if onnx_tensor_infos_for_validation is not None \
        and onnx_tensor_infos_for_validation.get(graph_node_output.name, None) is not None:
        onnx_tensor_infos = {
            graph_node_output.name: onnx_tensor_infos_for_validation[graph_node_output.name]
        }
        del onnx_tensor_infos_for_validation

    return onnx_tensor_infos, validation_data_1, validation_data_2


def obtaining_an_inverted_pattern_for_brute_force_validation(
    *,
    tensor_shape: List[int],
) -> List[List[int]]:
    """Obtaining reversal patterns for brute force verification.

    Parameters
    ----------
    tensor_shape: List[int]
        Shape of tensor for transposed pattern acquisition

    Returns
    -------
    total_perm_combination_list: List[List[int]]
        Shape transposition pattern perms
    """
    # Finding parts of a dimension with the same value
    unique_dims = list(set(tensor_shape))
    groups = {}
    for dim in unique_dims:
        groups[dim] = [i for i, x in enumerate(tensor_shape) if x == dim]

    # Get the permutation of the index for each group
    index_permutations = {}
    for key, group in groups.items():
        index_permutations[key] = list(itertools.permutations(group))

    # For each combination, swap the shape and generate a new array
    index_permutations_key_list = list(index_permutations.keys())
    index_combinations = [i for i in itertools.product(*index_permutations.values())]
    total_perm_combination_list = []
    for index_combination in index_combinations:
        partial_combination_list = [-1] * len(tensor_shape)
        for target_key, indexes in zip(index_permutations_key_list, index_combination):
            target_key_indexes_idx = 0
            for idx, s in enumerate(tensor_shape):
                if s == target_key:
                    partial_combination_list[idx] = indexes[target_key_indexes_idx]
                    target_key_indexes_idx += 1
        total_perm_combination_list.append(partial_combination_list)

    return total_perm_combination_list


def correction_process_for_accuracy_errors(
    *,
    input_tensor_1: Any,
    input_tensor_2: Any,
    tf_func: Any,
    np_func: Any,
    graph_node_output_shape: List,
    graph_node_output: gs.Variable,
    tf_layers_dict: Dict,
    **kwargs,
) -> Tuple[Any, Any]:
    """Correction process for accuracy errors.

    Parameters
    ----------
    input_tensor_1: Any
        Tensor subject to accuracy check and accuracy correction

    input_tensor_2: Any
        Tensor subject to accuracy check and accuracy correction

    tf_func: Any
        TensorFlow OP to be generated after accuracy correction

    np_func: Any
        Numpy functions used for accuracy checks and accuracy corrections

    graph_node_output_shape: List
        ONNX OP output shape

    tf_layers_dict: Dict
        TensorFlow Model Structure Dictionary

    Returns
    -------
    input_tensor_1: Any
        TensorFlow tensor after accuracy check and accuracy correction

    input_tensor_2: Any
        TensorFlow tensor after accuracy check and accuracy correction
    """
    onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = kwargs['onnx_tensor_infos_for_validation']
    onnx_tensor_infos = None
    validation_data_1 = None
    validation_data_2 = None
    if onnx_tensor_infos_for_validation is None:
        if np_func is not None \
            and None not in input_tensor_1.shape \
            and None not in input_tensor_2.shape:
            try:
                dummy_tensor = np_func(
                    np.ones(input_tensor_1.shape, dtype=np.float32),
                    np.ones(input_tensor_2.shape, dtype=np.float32),
                )
                if np.prod(list(dummy_tensor.shape)) == np.prod(graph_node_output_shape):
                    return input_tensor_1, input_tensor_2
            except:
                pass
            # No accuracy correction is performed, but only unmatching of output shapes is avoided.
            tensor_1_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_1.shape))))
            tensor_2_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_2.shape))))
            min_abs_err_perm_1 = [i for i in range(len(input_tensor_1.shape))]
            pre_min_abs_err_perm_1 = min_abs_err_perm_1
            min_abs_err_perm_2 = [i for i in range(len(input_tensor_2.shape))]
            pre_min_abs_err_perm_2 = min_abs_err_perm_2
            test_output_shape = graph_node_output_shape
            for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
                for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
                    try:
                        test_tensor_1 = np.ones(input_tensor_1.shape, dtype=np.float32).transpose(tensor_1_candidate_for_transposition)
                        test_tensor_2 = np.ones(input_tensor_2.shape, dtype=np.float32).transpose(tensor_2_candidate_for_transposition)
                        test_result_tensor = np_func(test_tensor_1, test_tensor_2)
                        test_result_tensor_shape = list(test_result_tensor.shape)
                        if test_result_tensor_shape == test_output_shape:
                            min_abs_err_perm_1 = tensor_1_candidate_for_transposition
                            min_abs_err_perm_2 = tensor_2_candidate_for_transposition
                            break
                    except Exception as ex1:
                        pass
                else:
                    continue
                break
            if pre_min_abs_err_perm_1 != min_abs_err_perm_1:
                input_tensor_1 = tf.transpose(a=input_tensor_1, perm=min_abs_err_perm_1)
            if pre_min_abs_err_perm_2 != min_abs_err_perm_2:
                input_tensor_2 = tf.transpose(a=input_tensor_2, perm=min_abs_err_perm_2)
        return input_tensor_1, input_tensor_2
    else:
        del onnx_tensor_infos_for_validation
    if graph_node_output_shape is not None:
        onnx_output_shape = [dim if not isinstance(dim, str) else -1 for dim in graph_node_output_shape]
        onnx_output_same_shape_counts = collections.Counter(onnx_output_shape)
        if sum([1 if dim > 1 and cnt > 1 else 0 for dim, cnt in onnx_output_same_shape_counts.items()]) >= 1:
            # Generate dummy op
            dummy_op = None
            tensor_2_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_2.shape))))
            for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
                try:
                    dummy_op = tf_func(
                        input_tensor_1,
                        tf.transpose(a=input_tensor_2, perm=tensor_2_candidate_for_transposition),
                    )
                    break
                except Exception as ex:
                    pass
            if dummy_op is not None and dummy_op.shape != tf.TensorShape(None):
                tf_output_shape = [dim if dim is not None else -1 for dim in dummy_op.shape]
                number_of_dim_other_than_1 = sum([1 if i != 1 else 0 for i in onnx_output_shape])
                # Processing continues only if there are two or more dimensions other than 1
                if number_of_dim_other_than_1 >= 2:
                    # Processing continues only when ONNX output shape and TensorFlow output shape match
                    if onnx_output_shape == tf_output_shape:
                        # Obtain ONNX inference results and
                        # TensorFlow inference results up to the previous layer of TensorFlow
                        onnx_tensor_infos, validation_data_1, validation_data_2 = \
                            acquisition_of_validation_data(
                                input_tensor_1=input_tensor_1 \
                                    if not hasattr(input_tensor_1, 'numpy') else input_tensor_1.numpy(),
                                input_tensor_2=input_tensor_2 \
                                    if not hasattr(input_tensor_2, 'numpy') else input_tensor_2.numpy(),
                                graph_node_output=graph_node_output,
                                tf_layers_dict=tf_layers_dict,
                                **kwargs,
                            )
                        # Perform simple accuracy verification
                        # Terminate when the error is less than 1e-3
                        if onnx_tensor_infos:
                            min_abs_err = sys.maxsize
                            min_abs_err_perm_1: int = [idx for idx in range(len(validation_data_1.shape))]
                            min_abs_err_perm_2: int = [idx for idx in range(len(validation_data_2.shape))]
                            tensor_1_candidate_for_transpositions = \
                                obtaining_an_inverted_pattern_for_brute_force_validation(tensor_shape=validation_data_1.shape)
                            tensor_2_candidate_for_transpositions = \
                                list(itertools.permutations(range(len(validation_data_2.shape))))
                            for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
                                for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
                                    try:
                                        tf_tensor_infos: Dict[Any] = \
                                            {
                                                'dummy_result': \
                                                    np_func(
                                                        validation_data_1.transpose(tensor_1_candidate_for_transposition), \
                                                        validation_data_2.transpose(tensor_2_candidate_for_transposition)
                                                    )
                                            }
                                        # Validation
                                        onnx_tf_output_pairs = {
                                            (oi[0], ti[0]): (oi[1], ti[1]) \
                                                for oi, ti in zip(onnx_tensor_infos.items(), tf_tensor_infos.items())
                                        }
                                        del tf_tensor_infos
                                        """
                                        check_results: Dict[str, List[np.ndarray, int, float|int]]
                                            {
                                                onnx_output_name: [
                                                    onnx_tensor,
                                                    matched_flg, <--- 0: Unmatched, 1: Matched, 2: Skipped (Deleted or Shape Unmatched)
                                                    max_abs_err,
                                                ]
                                            }
                                        """
                                        check_results = onnx_tf_tensor_validation(
                                            output_pairs=onnx_tf_output_pairs,
                                            rtol=0.0,
                                            atol=0.0,
                                        )
                                        result_err = sum([val[2] for val in check_results.values()])
                                        if result_err < min_abs_err:
                                            min_abs_err = result_err
                                            min_abs_err_perm_1 = list(tensor_1_candidate_for_transposition)
                                            min_abs_err_perm_2 = list(tensor_2_candidate_for_transposition)
                                            if min_abs_err < 1e-3:
                                                break
                                    except Exception as ex1:
                                        pass
                                else:
                                    continue
                                break
                            if len(min_abs_err_perm_1) > 1:
                                input_tensor_1 = tf.transpose(
                                    a=input_tensor_1 \
                                        if not isinstance(input_tensor_1, np.ndarray) else tf.convert_to_tensor(input_tensor_1),
                                    perm=min_abs_err_perm_1,
                                )
                            if len(min_abs_err_perm_2) > 1:
                                input_tensor_2 = tf.transpose(
                                    a=input_tensor_2 \
                                        if not isinstance(input_tensor_2, np.ndarray) else tf.convert_to_tensor(input_tensor_2),
                                    perm=min_abs_err_perm_2,
                                )
                        del onnx_tensor_infos
                        del validation_data_1
                        del validation_data_2
    return input_tensor_1, input_tensor_2


def nhwc_determination_of_output_value_of_binary_input_op(
    *,
    graph_node_input_1: Any,
    graph_node_input_2: Any,
    tf_layers_dict: Dict,
):
    """NHWC determination of output value of binary input OP.

    Parameters
    ----------
    graph_node_input_1: Any
        Input variable to be verified

    graph_node_input_2: Any
        Input variable to be verified

    tf_layers_dict: Dict
        TensorFlow Model Structure Dictionary

    Returns
    -------
    NHWC or not NHWC: bool
        True: "NHWC"
        False: not "NHWC"
    """
    is_output_nhwc_1 = \
        tf_layers_dict[graph_node_input_1.name]['nhwc'] \
            if isinstance(graph_node_input_1, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False

    is_output_nhwc_2 = \
        tf_layers_dict[graph_node_input_2.name]['nhwc'] \
            if isinstance(graph_node_input_2, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input_2.name].keys() else False

    return is_output_nhwc_1 or is_output_nhwc_2


def shape_is_equal_ignore_order(
    shape_list_1: List[int],
    shape_list_2: List[int],
) -> bool:
    """Verify that all axis size combinations match.

    Parameters
    ----------
    shape_list_1: List[int]
        List of shapes to be verified

    shape_list_2: List[int]
        List of shapes to be verified

    Returns
    -------
    True: Matches
    False: Unmatches
    """
    shape_list_1 = [-1 if isinstance(s, str) or s is None else s for s in shape_list_1]
    shape_list_2 = [-1 if isinstance(s, str) or s is None else s for s in shape_list_2]
    return sorted(shape_list_1) == sorted(shape_list_2)

# ReduceL1
# ReduceL2
# ReduceLogSum
# ReduceLogSumExp
# ReduceMax
# ReduceMean
# ReduceMin
# ReduceProd
# ReduceSum
# ReduceSumSquare
def define_reduceXXX(
    *,
    tf_func: str,
    target_input_tensor: Any,
    target_axes: List[int],
    target_keepdims: bool,
):
    reduced_tensor = None
    axes = target_axes if len(target_axes) > 1 else target_axes[0] if target_axes is not None else None

    if tf_func == 'ReduceL1':
        reduced_tensor = tf.norm(
            tensor=target_input_tensor,
            ord=1,
            axis=axes,
            keepdims=target_keepdims,
        )
    elif tf_func == 'ReduceL2':
        reduced_tensor = tf.norm(
            tensor=target_input_tensor,
            ord=2,
            axis=axes,
            keepdims=target_keepdims,
        )
    elif tf_func == 'ReduceLogSum':
        reduced_tensor = \
            tf.math.log(
                x=tf.reduce_sum(
                    input_tensor=target_input_tensor,
                    axis=axes,
                    keepdims=target_keepdims,
                )
            )
    elif tf_func == 'ReduceLogSumExp':
        reduced_tensor = tf.math.reduce_logsumexp(
            input_tensor=target_input_tensor,
            axis=axes,
            keepdims=target_keepdims,
        )
    elif tf_func == 'ReduceMax':
        reduced_tensor = tf.math.reduce_max(
            input_tensor=target_input_tensor,
            axis=axes,
            keepdims=target_keepdims,
        )
    elif tf_func == 'ReduceMean':
        reduced_tensor = tf.math.reduce_mean(
            input_tensor=target_input_tensor,
            axis=axes,
            keepdims=target_keepdims,
        )
    elif tf_func == 'ReduceMin':
        reduced_tensor = tf.math.reduce_min(
            input_tensor=target_input_tensor,
            axis=axes,
            keepdims=target_keepdims,
        )
    elif tf_func == 'ReduceProd':
        reduced_tensor = tf.math.reduce_prod(
            input_tensor=target_input_tensor,
            axis=axes,
            keepdims=target_keepdims,
        )
    elif tf_func == 'ReduceSum':
        reduced_tensor = tf.reduce_sum(
            input_tensor=target_input_tensor,
            axis=axes,
            keepdims=target_keepdims,
        )
    elif tf_func == 'ReduceSumSquare':
        reduced_tensor = tf.reduce_sum(
            input_tensor=tf.square(x=target_input_tensor),
            axis=axes,
            keepdims=target_keepdims,
        )
    return reduced_tensor
