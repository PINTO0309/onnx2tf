import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from utils.colors import Color
from utils.common_functions import convert_axis


def _nnapi_scalar(value, dtype):
    return tf.constant(value, dtype=dtype, shape=(1,))


def _alternative_argmax(
    input_tensor,
    axis = -1,
    output_type = tf.dtypes.float32,
    name = None,
    keepdims = False,
    epsilon = None,
    replace_argmax_to_reducemax_and_indicies_is_int64 = False,
    replace_argmax_to_reducemax_and_indicies_is_float32 = False,
):
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
        name=name,
    )

    if replace_argmax_to_reducemax_and_indicies_is_int64:
        return tf.cast(
            tf.math.subtract(
                _nnapi_scalar(reduction_size, output_type),
                reverse_argmax, name=name
            ),
            dtype=tf.dtypes.int64,
        )
    elif replace_argmax_to_reducemax_and_indicies_is_float32:
        return tf.cast(
            tf.math.subtract(
                _nnapi_scalar(reduction_size, output_type),
                reverse_argmax, name=name
            ),
            dtype=tf.dtypes.float32,
        )
    else:
        return tf.math.subtract(
            _nnapi_scalar(reduction_size, output_type),
            reverse_argmax,
            name=name,
        )


def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """ArgMax

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    graph_node_input: gs.Variable = graph_node.inputs[0]
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    replace_argmax_to_reducemax_and_indicies_is_int64 = \
        kwargs['replace_argmax_to_reducemax_and_indicies_is_int64']
    replace_argmax_to_reducemax_and_indicies_is_float32 = \
        kwargs['replace_argmax_to_reducemax_and_indicies_is_float32']

    axis = 0
    keepdims = True
    select_last_index = False

    if 'axis' in graph_node.attrs:
        axis = int(graph_node.attrs['axis'])
        # NCHW->NHWC, NCDHW->NDHWC
        axis = convert_axis(
            axis=axis,
            tensor_rank=len(shape),
        )

    if 'keepdims' in graph_node.attrs:
        # 0: False, 1: True
        keepdims = True if int(graph_node.attrs['keepdims']) == 1 else False

    if 'select_last_index' in graph_node.attrs:
        # 0: False, 1: True
        select_last_index = True if int(graph_node.attrs['select_last_index']) == 1 else False

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    if not select_last_index:
        if keepdims:
            argmaxed_tensor = tf.math.argmax(
                input=tf_layers_dict[graph_node_input.name]['tf_node'],
                axis=axis,
                output_type=dtype,
                name=graph_node.name,
            )
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.expand_dims(
                    input=argmaxed_tensor,
                    axis=axis,
                    name=f'{graph_node.name}_expand_dims',
                )
        else:
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.math.argmax(
                    input=tf_layers_dict[graph_node_input.name]['tf_node'],
                    axis=axis,
                    output_type=dtype,
                    name=graph_node.name,
                )

    else:
        if keepdims:
            reversed_tensor = \
                tf.reverse(
                    tensor=tf_layers_dict[graph_node_input.name]['tf_node'],
                    axis=axis,
                    name=f'{graph_node.name}_reverse',
                )
            argmaxed_tensor = \
                tf.math.argmax(
                    input=reversed_tensor,
                    axis=axis,
                    output_type=dtype,
                    name=f'{graph_node.name}_argmax',
                )
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.expand_dims(
                    input=argmaxed_tensor,
                    axis=axis,
                    name=f'{graph_node.name}_expand_dims',
                )
        else:
            reversed_tensor = \
                tf.reverse(
                    tensor=tf_layers_dict[graph_node_input.name]['tf_node'],
                    axis=axis,
                    name=f'{graph_node.name}_reverse',
                )
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.math.argmax(
                    input=reversed_tensor,
                    axis=axis,
                    output_type=dtype,
                    name=f'{graph_node.name}_argmax',
                )
