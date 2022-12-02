import sys
import copy
import random
random.seed(0)
import itertools
import traceback
import numpy as np
np.random.seed(0)
import tensorflow as tf
from tensorflow.keras.layers import Lambda # type: ignore
import onnx_graphsurgeon as gs
from onnx2tf.utils.colors import Color
from typing import Any, List, Optional, Union
from functools import wraps
from collections import namedtuple
from onnx2tf.utils.enums import (
    TF_DTYPES_TO_NUMPY_DTYPES,
)

INF_INDEX_VALUE: int = 4294967296

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
                transposed_value = tf.transpose(
                    a=value_before_transpose,
                    perm=transpose_perm,
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
                transposed_value = tf.transpose(
                    a=value_before_transpose,
                    perm=transpose_perm,
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
                f'If the input OP of ONNX before conversion is NHWC, use the -kt option.'
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
            if len(tf_node_output_shape) >= 3:
                base_shape = tf_node_output_shape[1]
                if len(tf_node_output_shape)-1 == sum([1 if base_shape == s else 0 for s in tf_node_output_shape[1:]]) \
                    and (onnx_node_output_shape == tf_node_output_shape):
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
        return values
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
        return values
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
        return values
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


def simple_arithmetic_validity_check(
    *,
    op_type: str,
    onnx_x: np.ndarray,
    onnx_y: np.ndarray,
    tf_x: np.ndarray,
    tf_y: np.ndarray,
):
    copy_onnx_x = onnx_x.copy()
    copy_onnx_y = onnx_y.copy()
    copy_tf_x = tf_x.copy()
    copy_tf_y = tf_y.copy()

    tf_y_shape = copy_tf_y.shape
    tf_y_rank = len(tf_y_shape)
    # All permutations of inverted patterns used for brute force checks
    # Transpose and compute tf_y repeatedly in the pattern of all permutations,
    # and repeat until the results match the results of onnx_x and onnx_y operations
    test_perm_list = list(itertools.permutations(range(tf_y_rank)))

    NP_CALC_FUNCS = {
        'Add': np.add,
        'Sub': np.subtract,
        'Mul': np.multiply,
        'Div': np.divide,
    }

    matched_perm = None
    for perm in test_perm_list:
        trans_copy_tf_y = copy_tf_y.transpose(perm)
        onnx_result: np.ndarray = NP_CALC_FUNCS[op_type](copy_onnx_x, copy_onnx_y)
        try:
            tf_result: np.ndarray = NP_CALC_FUNCS[op_type](copy_tf_x, trans_copy_tf_y)
            flat_onnx_result = np.sort(onnx_result.flatten())
            flat_tf_result = np.sort(tf_result.flatten())
            if flat_onnx_result.shape[0] == flat_tf_result.shape[0] \
                and np.isclose(flat_onnx_result, flat_tf_result, equal_nan=True).all():
                matched_perm = perm
                break
        except:
            pass
    if matched_perm is None:
        # Exception throw
        pass


def broadcast_validity_check(shape1: Union[np.ndarray, List], shape2: Union[np.ndarray, List]):
    """
    Check the validity of dimension shape for same length of tensors.
    Parameters
    ----------
    shape1: 1d list or ndarray.
    shape2: 1d list or ndarray.

    Returns
    -------
    result: True if shape1 and shape2 is valid for broadcasting, else False
    """
    result = False

    if len(shape1) != len(shape2):
        return result
    else:
        for i, j in zip(shape1, shape2):
            if i == j or i == 1 or j == 1:
                result = True
            else:
                result = False
                break

    return result


def explicit_broadcast(
    *,
    const_or_var_1: Any,
    const_or_var_2: Any,
    graph_node: Optional[gs.Node] = None,
    tf_layers_dict: dict = None,
):
    graph_node_input_name1 = None
    graph_node_input_name2 = None
    graph_node_input_shape1 = []
    graph_node_input_shape2 = []
    if graph_node is not None:
        graph_node_input_name1 = graph_node.inputs[0].name
        graph_node_input_name2 = graph_node.inputs[1].name
        graph_node_input_shape1 = list(graph_node.inputs[0].shape)
        graph_node_input_shape2 = list(graph_node.inputs[1].shape)

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
            # # Check output values by brute force only if the shape of const_or_var_2
            # # is different between const_or_var_1 and const_or_var_2 in dimensions other than 1
            # const_or_var_1_shape_not_none = [
            #     i if not isinstance(i, str) else 1 for i in const_or_var_1_shape
            # ]
            # for var1_shape, var2_shape in zip(const_or_var_1_shape_not_none, const_or_var_2.shape):
            #     if var2_shape != 1 and var1_shape != 1 and var1_shape != var2_shape:
            #         dummy_data_onnx = np.random.random_sample(graph_node.inputs[0].shape)
            #         dummy_data_onnx = dummy_data_onnx.astype(graph_node.inputs[0].dtype)
            #         dummy_data_tf = dummy_data_onnx.copy()
            #         dummy_data_tf = dummy_data_tf.reshape(const_or_var_1_shape).astype(graph_node.inputs[0].dtype)
            #         simple_arithmetic_validity_check(
            #             op_type=graph_node.op,
            #             onnx_x=dummy_data_onnx,
            #             onnx_y=graph_node.inputs[1].values,
            #             tf_x=dummy_data_tf,
            #             tf_y=const_or_var_2,
            #         )
            #         break

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
    shape:
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
    d = new_size.shape[0]
    h = new_size.shape[1]
    w = new_size.shape[2]
    # Dpeth (height x width)
    resized_list = []
    unstack_img_list = tf.unstack(input_tensor, axis=1)
    for i in unstack_img_list:
        resized_list.append(
            tf.compat.v1.image.resize_bilinear(
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
            tf.compat.v1.image.resize_bilinear(
                images=input_tensor,
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
    fused_argmax_scale_ratio: float = 0.5,
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

    fused_argmax_scale_ratio: float
        Scale ratio when generating Fused ArgMax
        Default: 0.5

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
    input_shape = tf_shape(input_tensor)
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
        output_shape[dim + 2] = output_size

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

        if (dim == spatial_size - 1):
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
    output_shape[1] = channels_count
    # create the channel indices
    channel_ind = tf.range(channels_count, dtype=tf.int64)
    # convert to column vector
    channel_ind = tf.expand_dims(channel_ind, 1)
    # "combine" channel indices with the result from the loop
    gather_ind = tf_product(
        a=channel_ind,
        b=gather_ind,
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
    data_shape = tf_shape(input_tensor=data)
    if data.get_shape().is_fully_defined():
        if not isinstance(indices, np.ndarray):
            indices_shape = indices.get_shape().as_list()
        else:
            indices_shape = indices.shape
    else:
        indices_shape = tf_shape(input_tensor=indices)
    if batch_dims > 0:
        max_i = tf.cast(
            data_shape[batch_dims:indices_shape[-1] + batch_dims],
            indices.dtype,
        )
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
):
    if isinstance(graph_node_input_1, gs.Variable) \
        and isinstance(graph_node_input_2, gs.Variable):

        node_x_op_type = graph_node_input_1.inputs[0].op \
            if len(graph_node_input_1.inputs) > 0 else 'Input'
        node_y_op_type = graph_node_input_2.inputs[0].op \
            if len(graph_node_input_2.inputs) > 0 else 'Input'

        if ((node_x_op_type == 'Transpose' and not node_y_op_type == 'Transpose') \
            or (not node_x_op_type == 'Transpose' and node_y_op_type == 'Transpose')) \
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
                    input_tensor_2 = tf.transpose(
                        a=input_tensor_2,
                        perm=reverse_perm,
                    )
    return graph_node_input_1, graph_node_input_2, input_tensor_1, input_tensor_2


def shape_unmatched_special_avoidance_workaround(
    *,
    graph_node_input_1,
    graph_node_input_2,
    input_tensor_1,
    input_tensor_2,
    tf_layers_dict,
):
    # At least one True value for same_input_shape_as_onnx
    # At least one True value in nhwc_flags
    # same_input_shape_as_onnx == True and nhwc_flags == False and 3D or 4D or 5D tensor is NHWC transposed
    nhwc_flag_1 = False
    same_input_shape_as_onnx_1 = False
    if isinstance(input_tensor_1, gs.Variable):
        nhwc_flag_1 =tf_layers_dict[input_tensor_1.name]['nhwc'] \
            if 'nhwc' in tf_layers_dict[input_tensor_1.name].keys() else False
        graph_node_input_1_shape = [
            dim if not isinstance(dim, str) else None for dim in graph_node_input_1.shape
        ]
        same_input_shape_as_onnx_1 = True if len(graph_node_input_1_shape) > 0 \
            and graph_node_input_1_shape == tf_layers_dict[input_tensor_1.name]['tf_node'].shape else False
    else:
        nhwc_flag_1 = False
        graph_node_input_1_shape = [
            dim if not isinstance(dim, str) else None for dim in graph_node_input_1.shape
        ]
        same_input_shape_as_onnx_1 = True if len(graph_node_input_1_shape) > 0 \
            and graph_node_input_1_shape == input_tensor_1.shape else False
    nhwc_flag_2 = False
    same_input_shape_as_onnx_2 = False
    if isinstance(input_tensor_2, gs.Variable):
        nhwc_flag_2 =tf_layers_dict[input_tensor_2.name]['nhwc'] \
            if 'nhwc' in tf_layers_dict[input_tensor_2.name].keys() else False
        graph_node_input_2_shape = [
            dim if not isinstance(dim, str) else None for dim in graph_node_input_1.shape
        ]
        same_input_shape_as_onnx_2 = True if len(graph_node_input_2_shape) > 0 \
            and graph_node_input_2_shape == tf_layers_dict[input_tensor_2.name]['tf_node'].shape else False
    else:
        nhwc_flag_2 = False
        graph_node_input_2_shape = [
            dim if not isinstance(dim, str) else None for dim in graph_node_input_1.shape
        ]
        same_input_shape_as_onnx_2 = True if len(graph_node_input_2_shape) > 0 \
            and graph_node_input_2_shape == input_tensor_2.shape else False

    same_input_shape_as_onnxs = [same_input_shape_as_onnx_1, same_input_shape_as_onnx_2]
    nhwc_flags = [nhwc_flag_1, nhwc_flag_2]
    if True in same_input_shape_as_onnxs and True in nhwc_flags:
        values = [input_tensor_1, input_tensor_2]
        for idx, (same_input_shape_as_onnx, nhwc_flag) in enumerate(zip(same_input_shape_as_onnxs, nhwc_flags)):
            if same_input_shape_as_onnx and not nhwc_flag:
                if len(values[idx].shape) == 3:
                    values[idx] = tf.transpose(a=values[idx], perm=[0,2,1])
                elif len(values[idx].shape) == 4:
                    values[idx] = tf.transpose(a=values[idx], perm=[0,2,3,1])
                elif len(values[idx].shape) == 5:
                    values[idx] = tf.transpose(a=values[idx], perm=[0,2,3,4,1])
        input_tensor_1 = values[0]
        input_tensor_2 = values[1]

    return input_tensor_1, input_tensor_2
