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
    convert_axis,
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """QuantizeLinear

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
    y_scale = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    y_scale_shape = y_scale.shape
    y_scale_rank = len(y_scale_shape)
    y_zero_point = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
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
    input_tensor = tf.cast(
        x=input_tensor,
        dtype=tf.float32,
    )
    x_shape = input_tensor_shape
    x_rank = input_tensor_rank
    y_scale_shape = y_scale_shape

    # Reshape process is needed for per-axis quantization
    # when scale is a 1-D tensor
    if y_scale_rank == 1:
        shape_broadcast = list(
            [1 for _ in range(axis)] \
            + [x_shape[axis]] \
            + [1 for _ in range(axis + 1, x_rank)]
        )
        y_scale = tf.reshape(
            tensor=y_scale,
            shape=shape_broadcast,
        )
    y = tf.divide(
        x=input_tensor,
        y=y_scale,
    )
    y = tf.round(y)

    if y_zero_point is not None:
        y_dtype = y_zero_point.dtype if y_zero_point.dtype not in [tf.int8, tf.uint8] else tf.float32
        y_zero_point = tf.cast(
            x=y_zero_point,
            dtype=tf.float32,
        )
        y_zero_point = tf.reshape(
            tensor=y_zero_point,
            shape=shape_broadcast,
        ) if y_scale_rank == 1 else y_zero_point
        y = tf.add(
            x=y,
            y=y_zero_point,
        )
    else:  # y_zero_point default dtype = uint8
        y_dtype = tf.uint8

    # Generation of TF OP
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.saturate_cast(
            value=y,
            dtype=y_dtype,
            name=graph_node.name,
        )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'QuantizeLinear',
                'tf_inputs': {
                    'x': input_tensor,
                    'y_scale': y_scale,
                    'y_zero_point': y_zero_point,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
