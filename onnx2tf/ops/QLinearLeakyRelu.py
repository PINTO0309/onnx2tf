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


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """QLinearLeakyRelu

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1

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
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    alpha = graph_node.attrs.get('alpha', 0.01)
    replace_leakyrelu_to_pseudo_leakyrelu = \
        kwargs['replace_leakyrelu_to_pseudo_leakyrelu']

    x = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    x_scale = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    x_zero_point = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    y_scale = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4
    y_zero_point = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) else graph_node_input_5
    output_dtype = y_zero_point.dtype if y_zero_point.dtype not in [tf.int8, tf.uint8] else tf.float32

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP

    # cast all inputs to float32
    x = tf.cast(x, tf.float32)
    x_zero_point = tf.cast(x_zero_point, tf.float32)
    y_zero_point = tf.cast(y_zero_point, tf.float32)

    # dequantize x
    dequantized_x = tf.multiply(
        x=x_scale,
        y=tf.subtract(x, x_zero_point),
    )
    tf_op_type = None
    dequantized_leakyrelu = None
    if not replace_leakyrelu_to_pseudo_leakyrelu:
        dequantized_leakyrelu = \
            tf.nn.leaky_relu(
                features=dequantized_x,
                alpha=alpha,
                name=graph_node.name,
            )
        tf_op_type = tf.nn.leaky_relu
    else:
        dequantized_leakyrelu = \
            tf.maximum(0.0, dequantized_x) + \
                tf.minimum(0.0, alpha * dequantized_x)
        tf_op_type = 'tf.maximum + tf.minimum'
    dequantized_leakyrelu = tf.add(
        x=tf.divide(
            x=dequantized_leakyrelu,
            y=y_scale,
        ),
        y=y_zero_point,
    )

    casted_sigmoid = tf.cast(dequantized_leakyrelu, output_dtype)

    tf_layers_dict[graph_node_output.name]['tf_node'] = casted_sigmoid

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'x': x,
                    'x_scale': x_scale,
                    'x_zero_point': x_zero_point,
                    'y_scale': y_scale,
                    'y_zero_point': y_zero_point,
                    'alpha': alpha,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
