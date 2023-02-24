import os
import io
import sys
import copy
import json
import random
import requests
random.seed(0)
import itertools
import traceback
import subprocess
import numpy as np
np.random.seed(0)
import tensorflow as tf
from tensorflow.keras.layers import Lambda # type: ignore
from tensorflow.python.keras.utils import conv_utils
import onnx
import onnx_graphsurgeon as gs
try:
    import onnxruntime as ort
except Exception as ex:
    pass
from onnx2tf.utils.colors import Color
from typing import Any, List, Optional, Union, Tuple, Dict, Callable
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
    **kwargs: dict,
):
    """Replace attributes, INPUT constants, and INPUT initializers with the specified values.

    Parameters
    ----------
    value_before_replacement: Any
    param_target: dict
    param_name: dict
    **kwargs: dict

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
            elif tf.keras.backend.is_keras_tensor(value_before_replacement):
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
    **kwargs: dict,
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
    **kwargs: dict,
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
                transposed_value = transpose_with_flexing_deterrence(
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
        non_verbose: bool = kwargs.get('non_verbose', False)
        if not non_verbose:
            if graph_input is not None:
                print(
                    f'{Color.GREEN}INFO:{Color.RESET} '+
                    f'{Color.GREEN}input_op_name{Color.RESET}: {graph_input.name} '+
                    f'{Color.GREEN}shape{Color.RESET}: {graph_input.shape} '+
                    f'{Color.GREEN}dtype{Color.RESET}: {graph_input.dtype}'
                )
            elif graph_node is not None:
                print('')
                print(
                    f'{Color.GREEN}INFO:{Color.RESET} {Color.MAGENTA}onnx_op_type{Color.RESET}: '+
                    f'{graph_node.op} {Color.MAGENTA}onnx_op_name{Color.RESET}: {graph_node.name}')
                for idx, graph_node_input in enumerate(graph_node.inputs):
                    print(
                        f'{Color.GREEN}INFO:{Color.RESET} '+
                        f'{Color.CYAN} input_name.{idx+1}{Color.RESET}: {graph_node_input.name} '+
                        f'{Color.CYAN}shape{Color.RESET}: {graph_node_input.shape} '+
                        f'{Color.CYAN}dtype{Color.RESET}: {graph_node_input.dtype}'
                    )
                for idx, graph_node_output in enumerate(graph_node.outputs):
                    print(
                        f'{Color.GREEN}INFO:{Color.RESET} '+
                        f'{Color.CYAN} output_name.{idx+1}{Color.RESET}: {graph_node_output.name} '+
                        f'{Color.CYAN}shape{Color.RESET}: {graph_node_output.shape} '+
                        f'{Color.CYAN}dtype{Color.RESET}: {graph_node_output.dtype}'
                    )
        try:
            result = func(*args, **kwargs)

            if not non_verbose:
                if graph_node is not None and tf_layers_dict is not None:
                    for idx, graph_node_output in enumerate(graph_node.outputs):
                        tf_layer_info: dict = tf_layers_dict.get(graph_node_output.name, None)
                        if tf_layer_info is not None:
                            tf_node_info = tf_layer_info.get('tf_node_info', None)
                            if tf_node_info is not None:
                                tf_op_type = tf_node_info.get('tf_op_type', None)
                                print(
                                    f'{Color.GREEN}INFO:{Color.RESET} ' + \
                                    f'{Color.MAGENTA}tf_op_type{Color.RESET}: {tf_op_type}'
                                )

                                tf_inputs = tf_node_info.get('tf_inputs', None)
                                if tf_inputs is not None:
                                    for input_idx, (input_key, input_values) in enumerate(tf_inputs.items()):
                                        input_info_text = \
                                            f'{Color.GREEN}INFO:{Color.RESET} ' + \
                                            f'{Color.BLUE} input.{input_idx+1}.{input_key}{Color.RESET}: '
                                        for input_attr_name, input_attr_value in input_values.items():
                                            input_info_text += \
                                                f'{Color.BLUE}{input_attr_name}{Color.RESET}: {input_attr_value} ' \
                                                if input_attr_value  is not None else ''
                                        print(input_info_text)

                                tf_outputs = tf_node_info.get('tf_outputs', None)
                                if tf_outputs is not None:
                                    for output_idx, (output_key, output_values) in enumerate(tf_outputs.items()):
                                        output_info_text = \
                                            f'{Color.GREEN}INFO:{Color.RESET} ' + \
                                            f'{Color.BLUE} output.{output_idx+1}.{output_key}{Color.RESET}: '
                                        for output_attr_name, output_attr_value in output_values.items():
                                            output_info_text += \
                                                f'{Color.BLUE}{output_attr_name}{Color.RESET}: {output_attr_value} ' \
                                                if output_attr_value  is not None else ''
                                        print(output_info_text)
            return result
        except:
            print(f'{Color.RED}ERROR:{Color.RESET} The trace log is below.')
            traceback.print_exc()
            if input_onnx_file_path is not None:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} ' +
                    f'input_onnx_file_path: {input_onnx_file_path}'
                )
            if graph_node is not None:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} ' +
                    f'onnx_op_name: {graph_node.name}'
                )
            print(
                f'{Color.RED}ERROR:{Color.RESET} ' +
                f'Read this and deal with it. https://github.com/PINTO0309/onnx2tf#parameter-replacement'
            )
            print(
                f'{Color.RED}ERROR:{Color.RESET} ' +
                f'Alternatively, if the input OP has a dynamic dimension, ' +
                f'use the -b or -ois option to rewrite it to a static shape and try again.'
            )
            print(
                f'{Color.RED}ERROR:{Color.RESET} ' +
                f'If the input OP of ONNX before conversion is NHWC or ' +
                f'an irregular channel arrangement other than NCHW, use the -kt or -kat option.'
            )
            print(
                f'{Color.RED}ERROR:{Color.RESET} ' +
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
        and input_tensor_2.shape is not None \
        and None not in input_tensor_1.shape \
        and None not in input_tensor_2.shape \
        and len(input_tensor_1.shape) == len(input_tensor_2.shape):

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
        and input_tensor_2.shape is not None \
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

    # If either operand have shape of all 1's, do not broadcast and return as is
    shape_for_judging_skip_processing_1 = [
        i if i is not None else INF_INDEX_VALUE for i in const_or_var_1.shape
    ]
    shape_for_judging_skip_processing_2 = [
        i if i is not None else INF_INDEX_VALUE for i in const_or_var_2.shape
    ]
    if np.prod(shape_for_judging_skip_processing_1) == 1 or np.prod(shape_for_judging_skip_processing_2) == 1:
        return const_or_var_1, const_or_var_2

    # Swap: len(const_or_var_1.shape) > len(const_or_var_2.shape)
    swapped = 0
    if len(const_or_var_1.shape) < len(const_or_var_2.shape):
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
    for _ in range(len(const_or_var_1.shape) - len(const_or_var_2.shape)):
        if isinstance(const_or_var_2, np.ndarray):
            const_or_var_2 = const_or_var_2[np.newaxis, ...]
        elif isinstance(const_or_var_2, tf.Tensor):
            const_or_var_2 = tf.expand_dims(
                input=const_or_var_2,
                axis=0,
            )
        elif not isinstance(const_or_var_2, np.ndarray) \
            and tf.keras.backend.is_keras_tensor(const_or_var_2):
            const_or_var_2 = tf.expand_dims(
                input=const_or_var_2,
                axis=0,
            )
        graph_node_input_shape2 = [1] + graph_node_input_shape2

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
                and tf.keras.backend.is_keras_tensor(const_or_var_2)
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
    if input_tensor.shape.is_fully_defined():
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
    d = new_size.shape[0]
    h = new_size.shape[1]
    w = new_size.shape[2]
    # Dpeth (height x width)
    resized_list = []
    unstack_img_list = tf.unstack(input_tensor, axis=1)
    for i in unstack_img_list:
        resized_list.append(
            tf.compat.v1.image.resize_bicubic(
                images=input_tensor,
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
    for i in unstack_img_list:
        resized_list.append(
            tf.compat.v1.image.resize_bicubic(
                images=input_tensor,
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
    d = new_size.shape[0]
    h = new_size.shape[1]
    w = new_size.shape[2]
    # Dpeth (height x width)
    resized_list = []
    unstack_img_list = tf.unstack(input_tensor, axis=1)
    for i in unstack_img_list:
        resized_list.append(
            tf.compat.v1.image.resize_nearest_neighbor(
                images=input_tensor,
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
    for i in unstack_img_list:
        resized_list.append(
            tf.compat.v1.image.resize_nearest_neighbor(
                images=input_tensor,
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
    replace_argmax_to_reducemax_and_indicies_is_int64: bool = False,
    replace_argmax_to_reducemax_and_indicies_is_float32: bool = False,
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

    replace_argmax_to_reducemax_and_indicies_is_int64: bool
        True: Convert final output to int64
        False: Do not convert final output to int64
        Default: False

    replace_argmax_to_reducemax_and_indicies_is_float32: bool
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
            f'{Color.RED}ERROR:{Color.RESET} ' +\
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

    if replace_argmax_to_reducemax_and_indicies_is_int64:
        return tf.cast(
            tf.math.subtract(
                _nnapi_scalar(reduction_size, output_type),
                reverse_argmax,
                name=name,
            ),
            dtype=tf.dtypes.int64,
        )
    elif replace_argmax_to_reducemax_and_indicies_is_float32:
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
    replace_argmax_to_fused_argmax_and_indicies_is_int64: bool = False,
    replace_argmax_to_fused_argmax_and_indicies_is_float32: bool = False,
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

    replace_argmax_to_fused_argmax_and_indicies_is_int64: bool
        True: Convert final output to int64
        False: Do not convert final output to int64
        Default: False

    replace_argmax_to_fused_argmax_and_indicies_is_float32: bool
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
        if replace_argmax_to_fused_argmax_and_indicies_is_int64:
            final_tensor = tf.cast(
                x=final_tensor,
                dtype=tf.int64,
                name=f'{name}_cast',
            )
        elif replace_argmax_to_fused_argmax_and_indicies_is_float32:
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
    neg = tf.math.divide(
        tf.math.multiply(
            tf.minimum(input_tensor, 0),
            -1
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
            tf.sqrt(tf.math.subtract(1.0, x)),
            y
        )
    )
    pseudo_asin = tf.math.subtract(
        y,
        tf.math.multiply(
            tf.math.multiply(2, neg),
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
    neg = tf.math.divide(
        tf.math.multiply(
            tf.minimum(input_tensor, 0),
            -1
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
        tf.sqrt(tf.math.subtract(1.0, x))
    )
    y = tf.math.multiply(
        y,
        tf.math.subtract(
            1.0,
            tf.math.multiply(2.0, neg)
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
        input_tensor_x=tf.broadcast_to(1.0, shape=input_tensor.shape),
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
            and tf.keras.backend.is_keras_tensor(indices_shape[-1]):
            if data_shape != tf.TensorShape([None]):
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
    **kwargs: dict,
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
    tf_layers_dict: dict,
    **kwargs: dict,
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
    # At least one True value for same_input_shape_as_onnx
    # At least one True value in nhwc_flags
    # same_input_shape_as_onnx == True and nhwc_flags == False and 3D or 4D or 5D tensor is NHWC transposed
    nhwc_flag_1 = False
    same_input_shape_as_onnx_1 = False
    if isinstance(input_tensor_1, gs.Variable):
        nhwc_flag_1 = tf_layers_dict[input_tensor_1.name]['nhwc'] \
            if 'nhwc' in tf_layers_dict[input_tensor_1.name].keys() else False
        if graph_node_input_1.shape is not None:
            graph_node_input_1_shape = [
                dim if not isinstance(dim, str) else None for dim in graph_node_input_1.shape
            ]
        else:
            graph_node_input_1_shape = []
        same_input_shape_as_onnx_1 = True if len(graph_node_input_1_shape) > 0 \
            and graph_node_input_1_shape == tf_layers_dict[input_tensor_1.name]['tf_node'].shape else False
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
    if isinstance(input_tensor_2, gs.Variable):
        nhwc_flag_2 = tf_layers_dict[input_tensor_2.name]['nhwc'] \
            if 'nhwc' in tf_layers_dict[input_tensor_2.name].keys() else False
        if graph_node_input_2.shape is not None:
            graph_node_input_2_shape = [
                dim if not isinstance(dim, str) else None for dim in graph_node_input_2.shape
            ]
        else:
            graph_node_input_2_shape = []
        same_input_shape_as_onnx_2 = True if len(graph_node_input_2_shape) > 0 \
            and graph_node_input_2_shape == tf_layers_dict[input_tensor_2.name]['tf_node'].shape else False
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
                    values[idx] = transpose_with_flexing_deterrence(
                        input_tensor=values[idx],
                        perm=[0,2,1],
                        **kwargs,
                    )
                elif len(values[idx].shape) == 4:
                    values[idx] = transpose_with_flexing_deterrence(
                        input_tensor=values[idx],
                        perm=[0,2,3,1],
                        **kwargs,
                    )
                elif len(values[idx].shape) == 5:
                    values[idx] = transpose_with_flexing_deterrence(
                        input_tensor=values[idx],
                        perm=[0,2,3,4,1],
                        **kwargs,
                    )
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


# https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0-rc0
# Transpose v5->v6, 5D->6D
def transpose_with_flexing_deterrence(
    *,
    input_tensor: Any,
    perm: List[int],
    output_shape: List[int] = None,
    name: str = None,
    **kwargs: dict,
) -> Any:
    """Transpose tensors of 6 or more dimensions while suppressing the transformation to FlexTranspose.
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
            #   of 6 or more dimensions into FlexTransposes.
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
        elif input_tensor_rank >= (COMPRESSION_DEFAULT_VALUE + 1) and x_shape_none_dims_count == 0 \
            or number_of_dimensions_after_flextranspose_compression < COMPRESSION_DEFAULT_VALUE \
                and number_of_dimensions_after_flextranspose_compression >= 2 \
                and x_shape_none_dims_count == 0:
            # Special Transpose.2
            #   Suppresses as much as possible the conversion of transposes
            #   of 6 or more dimensions into FlexTransposes.
            #   Decompose and transpose the tensor to be less than 5 dimensions.
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
                    splited_tensors = tf.split(
                        value=input_tensor,
                        num_or_size_splits=input_tensor.shape[axis],
                        axis=axis,
                    )
                    splited_squeezed_tensors = []
                    for splited_tensor in splited_tensors:
                        splited_squeezed_tensors.append(
                            tf.squeeze(
                                input=splited_tensor,
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


def dummy_onnx_inference(
    *,
    onnx_graph: onnx.ModelProto,
    output_names: List[str],
    test_data_nhwc: Optional[np.ndarray] = None,
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

    Returns
    ----------
    outputs: List[np.ndarray]
        Results of inference using dummy tensor
    """
    # Separate onnx at specified output_names position
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

    new_onnx_graph = gs.export_onnx(gs_graph)
    serialized_graph = onnx._serialize(new_onnx_graph)
    onnx_session = ort.InferenceSession(
        path_or_bytes=serialized_graph,
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
                and not isinstance(input_sizes[0][0], str):
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
    dummy_datas = {}
    for input_name, input_size, input_dtype in zip(input_names, input_sizes, input_dtypes):
        if test_data_nhwc is None:
            dummy_datas[input_name] = np.ones(
                input_size,
                dtype=input_dtype,
            )
        else:
            dummy_datas[input_name] = \
                tf.transpose(
                    a=tf.image.resize(
                        images=test_data_nhwc,
                        size=[input_size[2],input_size[3]],
                    ),
                    perm=[0,3,1,2],
                ).numpy().astype(input_dtype)
    outputs = onnx_session.run(None, dummy_datas)
    return outputs


def dummy_tf_inference(
    *,
    model: tf.keras.Model,
    inputs: List[tf.keras.Input],
    test_data_nhwc: Optional[np.ndarray] = None,
    verification_datas: Optional[List[np.ndarray]] = None,
) -> Any:
    """Perform inference on TF subgraphs with an all-1 dummy tensor.

    Parameters
    ----------
    model: tf.keras.Model
        Keras model

    inputs: List[tf.keras.Input]
        List of tf.keras.Input

    test_data_nhwc: Optional[np.ndarray]
        Test Image Data

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
            if idx == 0 and input_sizes[0][0] is not None:
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
    dummy_datas = {}
    if verification_datas is None:
        for input_name, input_size, input_dtype in zip(input_names, input_sizes, input_dtypes):
            if test_data_nhwc is None:
                dummy_datas[input_name] = np.ones(
                    input_size,
                    dtype=TF_DTYPES_TO_NUMPY_DTYPES[input_dtype],
                )
            else:
                dummy_datas[input_name] = \
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
                        dummy_datas[input_name] = verification_data
                    else:
                        dummy_datas[input_name] = verification_data.reshape(input_size)
                else:
                    dummy_datas[input_name] = verification_data
            else:
                dummy_datas[input_name] = np.ones(
                    input_size,
                    dtype=TF_DTYPES_TO_NUMPY_DTYPES[input_dtype],
                )
    outputs = model(
        inputs={
            input.name: dummy_datas[input.name] for input in inputs
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
    URL = f'https://s3.us-central-1.wasabisys.com/onnx2tf-en/datas/{FILE_NAME}'
    test_sample_images_npy = requests.get(URL).content
    test_image_data = None
    with io.BytesIO(test_sample_images_npy) as f:
        test_image_data: np.ndarray = np.load(f)
    return test_image_data


def broadcast_for_gpu_delegate(
    *,
    input_tensor_1: Any,
    input_tensor_2: Any,
    **kwargs: dict,
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


def calc_tf_pooling_pads(input_shape, kernel, strides, func):
    """Calculate how much padding is needed for tensorflow mode 'SAME'.

    Parameters
    ----------
    input_shape: Union[np.ndarray, List]
        input tensor shape of pooling layer
    kernel: List
        kernel shape from onnx
    strides: List
        strides from onnx
    func: Callable
        function for ceil or floor, depends on onnx option ceil_mode

    Returns
    -------
    same_pads: List
        onnx formatted padding, [x1_begin, x2_begin, ..., xn_begin, x1_end, x2_end, ..., xn_end]
    """

    same_pads = []
    same_pads_end = []

    # calculate how much padding is needed except batch and channel dimension
    for i, k, s in zip(input_shape[1:-1], kernel, strides):
        same_output_shape = func((i - 1) / s) + 1
        axis_pads = np.max((same_output_shape - 1) * s + k - i, 0)

        padded_valid_output_shape = func((i + axis_pads - k) / s) + 1
        error_msg = f'{Color.RED}ERROR:{Color.RESET} ' + \
                    f'Wrong padding calculation.'
        assert same_output_shape == padded_valid_output_shape, error_msg

        same_pads.append(axis_pads // 2)
        # pads to end more for odd number padding
        if axis_pads % 2:
            same_pads_end.append(axis_pads // 2 + 1)
        else:
            same_pads_end.append(axis_pads // 2)

    same_pads.extend(same_pads_end)

    return same_pads


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
    non_verbose: bool = False,
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

    non_verbose: bool
        Do not show all information logs. Only error logs are displayed.
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
                    'https://raw.githubusercontent.com/tensorflow/tensorflow/v2.11.0/tensorflow/lite/schema/schema.fbs',
                    '-o',
                    f'{output_folder_path}/schema.fbs'
                ],
                stderr=subprocess.PIPE
            ).decode('utf-8')

        # tflite -> JSON
        result = subprocess.check_output(
            [
                'flatc', '-t',
                '--strict-json',
                '--defaults-json',
                '-o', f'{output_folder_path}',
                f'{output_folder_path}/schema.fbs',
                '--',
                f'{output_folder_path}/{tflite_file_name}'
            ],
            stderr=subprocess.PIPE
        ).decode('utf-8')

        # Rewrite input OP name and output OP name
        json_file_name = f'{os.path.splitext(os.path.basename(tflite_file_name))[0]}.json'
        json_file_path = f'{output_folder_path}/{json_file_name}'
        flat_json = None

        with open(json_file_path, 'r') as f:
            flat_json = json.load(f)
            flat_subgraphs = flat_json['subgraphs'][0]
            flat_tensors: List[Dict] = flat_subgraphs['tensors']
            flat_input_nums: List[int] = flat_subgraphs['inputs']
            flat_output_nums: List[int] = flat_subgraphs['outputs']
            flat_input_infos = [flat_tensors[idx] for idx in flat_input_nums]
            flat_output_infos = [flat_tensors[idx] for idx in flat_output_nums]
            # INPUT
            for idx, flat_input_info in enumerate(flat_input_infos):
                flat_input_info['name'] = onnx_input_names[idx]
            # OUTPUT
            for idx, flat_output_info in enumerate(flat_output_infos):
                flat_output_info['name'] = onnx_output_names[idx]

        if flat_json is not None:
            with open(json_file_path, 'w') as f:
                json.dump(flat_json, f)
            # JSON -> tflite
            result = subprocess.check_output(
                [
                    'flatc',
                    '-o', f'{output_folder_path}',
                    '-b', f'{output_folder_path}/schema.fbs',
                    f'{json_file_path}'
                ],
                stderr=subprocess.PIPE
            ).decode('utf-8')
            # Delete JSON
            os.remove(f'{json_file_path}')

    except Exception as ex:
        if not non_verbose:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} '+
                'If you want tflite input OP name and output OP name ' +
                'to match ONNX input and output names, ' +
                'convert them after installing "flatc". ' +
                'Also, do not use symbols such as slashes in input/output OP names. ' +
                'debian/ubuntu: apt install -y flatbuffers-compiler ' +
                'Other than debian/ubuntu: https://github.com/google/flatbuffers/releases'
            )


def make_tf_partial_model_inputs(
    *,
    input_tensors: List[Any],
) -> List[tf.keras.Input]:
    """Generate input OPs for TensorFlow subgraph generation.

    Parameters
    ----------
    input_tensors: List[Any]
        List of input tensor

    Returns
    -------
    inputs: List[tf.keras.Input]
        List of tf.keras.Input
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

    inputs: List[tf.keras.Input] = []
    input = None
    for idx, input_shape in enumerate(tf_partial_model_input_shapes):
        if isinstance(input_shape, list) and len(input_shape) == 0:
            tf_partial_model_input_shapes[idx] = [1]
    for input_shape, input_dtype in zip(tf_partial_model_input_shapes, tf_partial_model_input_dtypes):
        if len(input_shape) == 1:
            input = tf.keras.Input(
                shape=input_shape[0] if isinstance(input_shape[0], int) else None,
                batch_size=1,
                dtype=input_dtype,
            )
        elif len(input_shape) >= 2:
            input = tf.keras.Input(
                shape=[
                    inp if isinstance(inp, int) else None for inp in input_shape[1:]
                ],
                batch_size=input_shape[0] if isinstance(input_shape[0], int) else None,
                dtype=input_dtype,
            )
        inputs.append(input)
    return inputs
