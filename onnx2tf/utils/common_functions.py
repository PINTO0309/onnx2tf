import sys
import copy
import random
random.seed(0)
import traceback
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.colors import Color
from typing import Any, List
from functools import wraps
from collections import namedtuple


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
                print(f'{Color.GREEN}INFO:{Color.RESET} {Color.MAGENTA}onnx_op_type{Color.RESET}: {graph_node.op} {Color.MAGENTA}onnx_op_name{Color.RESET}: {graph_node.name}')
                for idx, graph_node_input in enumerate(graph_node.inputs):
                    print(
                        f'{Color.GREEN}INFO:{Color.RESET} '+
                        f'{Color.CYAN}input_name.{idx+1}{Color.RESET}: {graph_node_input.name} '+
                        f'{Color.CYAN}shape{Color.RESET}: {graph_node_input.shape} '+
                        f'{Color.CYAN}dtype{Color.RESET}: {graph_node_input.dtype}'
                    )
                for idx, graph_node_output in enumerate(graph_node.outputs):
                    print(
                        f'{Color.GREEN}INFO:{Color.RESET} '+
                        f'{Color.CYAN}output_name.{idx+1}{Color.RESET}: {graph_node_output.name} '+
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
                            tf_layer = tf_layer_info.get('tf_node', None)
                            if tf_layer is not None:
                                if hasattr(tf_layer, 'node') and isinstance(tf_layer.node.input_tensors, list):
                                    for in_idx, input_tensor in enumerate(tf_layer.node.input_tensors):
                                        print(
                                            f'{Color.GREEN}INFO:{Color.RESET} '+
                                            f'{Color.BLUE}tf_op_type{Color.RESET}: {tf_layer_info.get("optype", "")} '+
                                            f'{Color.BLUE}input_name.{in_idx+1}{Color.RESET}: {input_tensor.name if hasattr(input_tensor, "name") else "np.ndarray"} '+
                                            f'{Color.BLUE}shape{Color.RESET}: {input_tensor.shape} '+
                                            f'{Color.BLUE}dtype{Color.RESET}: {input_tensor.dtype}'
                                        )
                                else:
                                    print(
                                        f'{Color.GREEN}INFO:{Color.RESET} '+
                                        f'{Color.BLUE}tf_op_type{Color.RESET}: {tf_layer_info.get("optype", "")} '+
                                        f'{Color.BLUE}input_name.{idx+1}{Color.RESET}: {tf_layer.node.input_tensors.name if hasattr(tf_layer, "node") and hasattr(tf_layer.node.input_tensors, "name") else "np.ndarray"} '+
                                        f'{Color.BLUE}shape{Color.RESET}: {tf_layer.node.input_tensors.shape if hasattr(tf_layer, "node") else ""} '+
                                        f'{Color.BLUE}dtype{Color.RESET}: {tf_layer.node.input_tensors.dtype if hasattr(tf_layer, "node") else ""}'
                                    )

                    for idx, graph_node_output in enumerate(graph_node.outputs):
                        tf_layer_info: dict = tf_layers_dict.get(graph_node_output.name, None)
                        if tf_layer_info is not None:
                            tf_layer = tf_layer_info.get('tf_node', None)
                            if tf_layer is not None:
                                type_spec_info = ''
                                if hasattr(tf_layer, 'type_spec'):
                                    type_spec_info = tf_layer.type_spec
                                else:
                                    type_spec_info = graph_node_output
                                print(
                                    f'{Color.GREEN}INFO:{Color.RESET} '+
                                    f'{Color.BLUE}tf_op_type{Color.RESET}: {tf_layer_info.get("optype", "")} '+
                                    f'{Color.BLUE}output_name.{idx+1}{Color.RESET}: {graph_node_output.name} '+
                                    f'{Color.BLUE}shape{Color.RESET}: {type_spec_info.shape} '+
                                    f'{Color.BLUE}dtype{Color.RESET}: {type_spec_info.dtype}'
                                )
            return result
        except:
            print(f'{Color.RED}ERROR:{Color.RESET} The trace log is below.')
            traceback.print_exc()
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
            tf_node_output_shape = tf_layers_dict[onnx_node_output.name]['tf_node'].shape
            output_shape_trans = output_shape_trans or (onnx_node_output_shape != tf_node_output_shape)
            tf_layers_dict[onnx_node_output.name]['before_op_output_shape_trans'] = output_shape_trans
        return result
    return inverted_operation_enable_disable_wrapper_func


def get_constant_or_variable(
    const_or_var: Any,
    before_op_output_shape_trans: bool,
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
        elif tensor_rank == 1 and values.size > 2:
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

    if safe_axis < 0:
        safe_axis = len(input_tensor.shape) + safe_axis
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
        zero_if_max_else_one = zero_if_max_else_eps * _nnapi_scalar(1 / eps, input_tensor.dtype)
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
            pad_size = (np.ceil(out_size) - np.floor(out_size)).astype(np.int64)
        else:
            pad_size = tf.cast(tf.math.ceil(out_size) - tf.math.floor(out_size), tf.int64)

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
    if not ceil_mode and ((type(padding) is list and padding == [0] * spatial_size * 2) or padding == "VALID"):
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
    tf_paddings = [[0, 0], [0, 0]]
    # padding for the (D)HW dimensions
    for i in range(spatial_size):
        tf_paddings += [[pads[i * 2], pads[i * 2 + 1]]]

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
    return (output_ind // kernel) * (stride - kernel * dilation) + output_ind * dilation


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
        output_size = (((in_spatial_shape[dim] - filter_size) // strides[dim]) + 1) * kernel_shape[dim]
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
    gather_ind = tf.tile(gather_ind, [input_shape[0]] + [1] * (spatial_size + 1))

    # extract the selected values from the input
    output = tf.gather_nd(input_tensor, gather_ind, batch_dims=1)
    # reshape the output to the correct shape calculated earlier
    output = tf.reshape(output, output_shape)

    return output


def explicit_broadcast(
    *,
    x,
    y,
    axis=None,
):
    if np.prod(y.shape) == 1:
        return y

    if axis is None:
        return y

    total_num_dim = len(x.shape)
    if axis < 0:
        axis += total_num_dim

    y_rank = len(y.shape)

    if axis + y_rank == total_num_dim:
        return y

    dims = [axis + i for i in range(y_rank)]
    new_y = y
    for i in range(total_num_dim):
        if i not in dims:
            new_y = tf.expand_dims(new_y, i)
    return new_y
