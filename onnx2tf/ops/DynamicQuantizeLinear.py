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
    """DynamicQuantizeLinear

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
    graph_node_output_1: gs.Variable = graph_node.outputs[0]
    o1_shape = graph_node_output_1.shape
    o1_dtype = graph_node_output_1.dtype
    graph_node_output_2: gs.Variable = graph_node.outputs[1]
    o2_shape = graph_node_output_2.shape
    o2_dtype = graph_node_output_2.dtype
    graph_node_output_3: gs.Variable = graph_node.outputs[2]
    o3_shape = graph_node_output_3.shape
    o3_dtype = graph_node_output_3.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output_1.name] = {
        'optype': graph_node.op,
        'shape': o1_shape,
        'dtype': o1_dtype,
    }
    tf_layers_dict[graph_node_output_2.name] = {
        'optype': graph_node.op,
        'shape': o2_shape,
        'dtype': o2_dtype,
    }
    tf_layers_dict[graph_node_output_3.name] = {
        'optype': graph_node.op,
        'shape': o3_shape,
        'dtype': o3_dtype,
    }

    input_tensor_1 = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1

    # Generation of TF OP
    dtype = tf.uint8
    qmin = dtype.min
    qmax = dtype.max
    min_x = tf.math.minimum(0., tf.math.reduce_min(input_tensor_1))
    max_x = tf.math.maximum(0., tf.math.reduce_max(input_tensor_1))
    y_scale = (max_x - min_x) / (qmax - qmin)
    intermediate_zero_point = qmin - (min_x / y_scale)
    y_zero_point = tf.clip_by_value(
        tf.round(
            x=intermediate_zero_point
        ),
        clip_value_min=qmin,
        clip_value_max=qmax,
    )
    y = tf.cast(
        tf.clip_by_value(
            (tf.round(input_tensor_1 / y_scale) + y_zero_point),
            clip_value_min=qmin,
            clip_value_max=qmax,
        ),
        dtype=dtype,
    )

    tf_layers_dict[graph_node_output_1.name]['tf_node'] = y
    tf_layers_dict[graph_node_output_2.name]['tf_node'] = y_scale
    tf_layers_dict[graph_node_output_3.name]['tf_node'] = \
        tf.cast(
            x=y_zero_point,
            dtype=dtype,
        )

    # Generation of Debug Info
    tf_outputs = {
        'y': tf_layers_dict[graph_node_output_1.name]['tf_node'],
        'y_scale': tf_layers_dict[graph_node_output_2.name]['tf_node'],
        'y_zero_point': tf_layers_dict[graph_node_output_3.name]['tf_node'],
    }
    tf_layers_dict[graph_node_output_1.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'DynamicQuantizeLinear',
                'tf_inputs': {
                    'x': input_tensor_1,
                },
                'tf_outputs': tf_outputs,
            }
        )
