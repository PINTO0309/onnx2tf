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
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Flatten

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

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    input_tensor_shape = tf.shape(input_tensor)

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    axis = graph_node.attrs.get("axis", 0)
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(graph_node_input.shape),
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    cal_shape = None
    if axis == 0:
        cal_shape = (1, -1)
    else:
        cal_shape = (
            tf.reduce_prod(input_tensor_shape[0:axis]),
            tf.reduce_prod(input_tensor_shape[axis:tf.size(input_tensor_shape)]),
        )
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.reshape(
            tensor=input_tensor,
            shape=cal_shape,
            name=graph_node.name,
        )
