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
)


@print_node_info
@inverted_operation_enable_disable
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
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)
    x_scale = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    x_scale_shape = x_scale.shape
    x_scale_rank = len(x_scale_shape)
    x_zero_point = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    axis = graph_node.attrs.get('axis', 1)
    axis = convert_axis(
        axis=axis,
        tensor_rank=input_tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP

    input_tensor = tf.cast(input_tensor, tf.float32)

    # Reshape process is needed for per-axis dequantization
    # when scale is a 1-D tensor
    if x_scale_rank == 1:
        shape_broadcast = list([1 for _ in range(axis)] + [input_tensor_shape[axis]] + [1 for _ in range(axis + 1, input_tensor_rank)])
        x_scale = tf.reshape(
            tensor=x_scale,
            shape=shape_broadcast,
        )

    subed_tensor = input_tensor
    if len(graph_node.inputs) >= 3 and input_tensor.dtype != tf.int32:
        x_zero_point = tf.cast(
            x=x_zero_point,
            dtype=tf.float32,
        )
        x_zero_point = tf.reshape(
            tensor=x_zero_point,
            shape=shape_broadcast,
        ) if x_scale_rank == 1 else x_zero_point
        subed_tensor = tf.subtract(
            x=input_tensor,
            y=x_zero_point,
        )
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.multiply(
            x=subed_tensor,
            y=x_scale,
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
