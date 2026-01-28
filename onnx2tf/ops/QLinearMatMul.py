import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
)

def _get_qmin_qmax(dtype: tf.dtypes.DType):
    if dtype == tf.uint8:
        return 0.0, 255.0
    if dtype == tf.int8:
        return -128.0, 127.0
    if dtype == tf.uint16:
        return 0.0, 65535.0
    if dtype == tf.int16:
        return -32768.0, 32767.0
    return None, None


def _reshape_for_axis(
    *,
    value,
    input_tensor,
    axis: int,
):
    value_rank = len(value.shape)
    input_rank = len(input_tensor.shape)
    if value_rank == 1 and input_rank is not None:
        shape = [1] * input_rank
        shape[axis] = -1
        return tf.reshape(value, shape)
    return value


def _reshape_for_output(
    *,
    value,
    output_tensor,
):
    value_rank = len(value.shape)
    output_rank = len(output_tensor.shape)
    if value_rank == 1 and output_rank is not None and output_rank >= 2:
        if output_tensor.shape[-2] == value.shape[0]:
            shape = [1] * output_rank
            shape[-2] = -1
            return tf.reshape(value, shape)
    return value


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """QLinearMatMul

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_2 = \
        tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans,
    )
    graph_node_input_4 = get_constant_or_variable(
        graph_node.inputs[3],
        before_op_output_shape_trans,
    )
    graph_node_input_5 = get_constant_or_variable(
        graph_node.inputs[4],
        before_op_output_shape_trans,
    )
    graph_node_input_6 = get_constant_or_variable(
        graph_node.inputs[5],
        before_op_output_shape_trans,
    )
    graph_node_input_7 = get_constant_or_variable(
        graph_node.inputs[6],
        before_op_output_shape_trans,
    )
    graph_node_input_8 = get_constant_or_variable(
        graph_node.inputs[7],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    a = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    a_is_dequantized = False
    if isinstance(graph_node_input_1, gs.Variable):
        a_is_dequantized = tf_layers_dict.get(graph_node_input_1.name, {}).get('is_dequantized', False)
    a_scale = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    a_zero_point = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    b = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4
    b_is_dequantized = False
    if isinstance(graph_node_input_4, gs.Variable):
        b_is_dequantized = tf_layers_dict.get(graph_node_input_4.name, {}).get('is_dequantized', False)
    b_scale = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) else graph_node_input_5
    b_zero_point = tf_layers_dict[graph_node_input_6.name]['tf_node'] \
        if isinstance(graph_node_input_6, gs.Variable) else graph_node_input_6
    y_scale = tf_layers_dict[graph_node_input_7.name]['tf_node'] \
        if isinstance(graph_node_input_7, gs.Variable) else graph_node_input_7
    y_zero_point = tf_layers_dict[graph_node_input_8.name]['tf_node'] \
        if isinstance(graph_node_input_8, gs.Variable) else graph_node_input_8
    y_dtype = y_zero_point.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'is_dequantized': True,
    }

    # Generation of TF OP

    # reshape a_scale and a_zero_point to broadcast on row axis (second last)
    a_scale = _reshape_for_axis(value=a_scale, input_tensor=a, axis=-2)
    a_zero_point = _reshape_for_axis(value=a_zero_point, input_tensor=a, axis=-2)
    # reshape b_scale and b_zero_point to broadcast on column axis (last)
    b_scale = _reshape_for_axis(value=b_scale, input_tensor=b, axis=-1)
    b_zero_point = _reshape_for_axis(value=b_zero_point, input_tensor=b, axis=-1)

    # cast all inputs to float32
    a = tf.cast(a, tf.float32)
    a_scale = tf.cast(a_scale, tf.float32)
    a_zero_point = tf.cast(a_zero_point, tf.float32)
    b = tf.cast(b, tf.float32)
    b_scale = tf.cast(b_scale, tf.float32)
    b_zero_point = tf.cast(b_zero_point, tf.float32)
    y_scale = tf.cast(y_scale, tf.float32)
    y_zero_point = tf.cast(y_zero_point, tf.float32)

    # dequantize a and b
    if a_is_dequantized:
        dequantized_a = tf.cast(a, tf.float32)
    else:
        dequantized_a = tf.multiply(tf.subtract(a, a_zero_point), a_scale)

    if b_is_dequantized:
        dequantized_b = tf.cast(b, tf.float32)
    else:
        dequantized_b = tf.multiply(tf.subtract(b, b_zero_point), b_scale)

    # matmul
    x = tf.matmul(dequantized_a, dequantized_b)

    # broadcast output scale/zero_point if needed
    y_scale = _reshape_for_output(value=y_scale, output_tensor=x)
    y_zero_point = _reshape_for_output(value=y_zero_point, output_tensor=x)

    # quantize then dequantize to float32
    y = tf.round(tf.divide(x, y_scale))
    y = tf.add(y, y_zero_point)
    qmin, qmax = _get_qmin_qmax(y_dtype)
    if qmin is not None and qmax is not None:
        y = tf.clip_by_value(y, qmin, qmax)
    y = tf.multiply(tf.subtract(y, y_zero_point), y_scale)

    tf_layers_dict[graph_node_output.name]['tf_node'] = y

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'QLinearMatMul',
                'tf_inputs': {
                    'a': a,
                    'a_scale': a_scale,
                    'a_zero_point': a_zero_point,
                    'b': b,
                    'b_scale': b_scale,
                    'b_zero_point': b_zero_point,
                    'y': y,
                    'y_zero_point': y_zero_point,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
