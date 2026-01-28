import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)

def _expand_scale_or_zero_point(
    *,
    value,
    input_tensor,
    axis: int,
    block_size: int,
):
    value_rank = len(value.shape)
    input_rank = len(input_tensor.shape)

    if value_rank == 0:
        return value

    if input_rank <= 0:
        return value

    if axis < 0 or axis >= input_rank:
        axis = 0

    # Blocked quantization: expand along axis then slice to input shape
    if block_size > 0 and value_rank == input_rank:
        if value.shape[axis] is None \
            or input_tensor.shape[axis] is None \
            or value.shape[axis] != input_tensor.shape[axis]:
            expanded = tf.repeat(value, repeats=block_size, axis=axis)
            expanded = tf.slice(expanded, [0] * input_rank, tf.shape(input_tensor))
            return expanded
        return value

    # Per-axis quantization: reshape 1-D to broadcast
    if value_rank == 1 and input_rank is not None:
        shape = [1] * input_rank
        shape[axis] = -1
        return tf.reshape(value, shape)

    return value


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """DequantizeLinear

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
    graph_node_input_3 = None
    if len(graph_node.inputs) >= 3:
        graph_node_input_3 = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_is_dequantized = False
    input_nhwc = False
    if isinstance(graph_node_input_1, gs.Variable):
        input_is_dequantized = tf_layers_dict.get(graph_node_input_1.name, {}).get('is_dequantized', False)
        input_nhwc = tf_layers_dict.get(graph_node_input_1.name, {}).get('nhwc', False)

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    input_tensor_rank = len(input_tensor.shape)
    input_tensor_dtype = input_tensor.dtype
    x_scale = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    x_zero_point = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    axis = graph_node.attrs.get('axis', 1)
    axis = convert_axis(
        axis=axis,
        tensor_rank=input_tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )
    if input_tensor_rank == 1:
        axis = 0

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'is_dequantized': True,
        'nhwc': input_nhwc,
    }

    # Generation of TF OP

    input_tensor = tf.cast(input_tensor, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)

    block_size = int(graph_node.attrs.get('block_size', 0))
    x_scale = _expand_scale_or_zero_point(
        value=x_scale,
        input_tensor=input_tensor,
        axis=axis,
        block_size=block_size,
    )

    if input_is_dequantized:
        tf_layers_dict[graph_node_output.name]['tf_node'] = input_tensor
    else:
        if x_zero_point is None or input_tensor_dtype == tf.int32:
            x_zero_point = tf.zeros_like(x_scale)
        else:
            x_zero_point = tf.cast(x_zero_point, tf.float32)
            x_zero_point = _expand_scale_or_zero_point(
                value=x_zero_point,
                input_tensor=input_tensor,
                axis=axis,
                block_size=block_size,
            )

        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.multiply(
                x=tf.subtract(input_tensor, x_zero_point),
                y=x_scale,
            )

    if hasattr(tf_layers_dict[graph_node_output.name]['tf_node'], 'numpy'):
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.convert_to_tensor(tf_layers_dict[graph_node_output.name]['tf_node'].numpy())

    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'DequantizeLinear',
                'tf_inputs': {
                    'input': input_tensor,
                    'x_scale': x_scale,
                    'x_zero_point': x_zero_point,
                    'axis': axis,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
