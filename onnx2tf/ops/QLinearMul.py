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
    """QLinearMul

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
    a_scale = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    a_zero_point = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    b = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4
    b_scale = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) else graph_node_input_5
    b_zero_point = tf_layers_dict[graph_node_input_6.name]['tf_node'] \
        if isinstance(graph_node_input_6, gs.Variable) else graph_node_input_6
    c_scale = tf_layers_dict[graph_node_input_7.name]['tf_node'] \
        if isinstance(graph_node_input_7, gs.Variable) else graph_node_input_7
    c_zero_point = tf_layers_dict[graph_node_input_8.name]['tf_node'] \
        if isinstance(graph_node_input_8, gs.Variable) else graph_node_input_8
    output_dtype = c_zero_point.dtype if c_zero_point.dtype not in [tf.int8, tf.uint8] else tf.float32

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP

    # cast all inputs to float32
    a = tf.cast(a, tf.float32)
    a_zero_point = tf.cast(a_zero_point, tf.float32)
    b = tf.cast(b, tf.float32)
    b_zero_point = tf.cast(b_zero_point, tf.float32)
    c_zero_point = tf.cast(c_zero_point, tf.float32)

    # dequantize a and b
    ab_scale = tf.multiply(a_scale, b_scale)
    dequantized_a = tf.subtract(a, a_zero_point)
    dequantized_b = tf.subtract(b, b_zero_point)
    dequantized_ab = tf.multiply(tf.multiply(dequantized_a, dequantized_b), ab_scale)
    dequantized_abc = tf.add(tf.divide(dequantized_ab, c_scale), c_zero_point)

    casted_mul = tf.cast(dequantized_abc, output_dtype)

    tf_layers_dict[graph_node_output.name]['tf_node'] = casted_mul

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'QLinearMul',
                'tf_inputs': {
                    'a': a,
                    'a_scale': a_scale,
                    'a_zero_point': a_zero_point,
                    'b': b,
                    'b_scale': b_scale,
                    'b_zero_point': b_zero_point,
                    'c_scale': c_scale,
                    'c_zero_point': c_zero_point,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
