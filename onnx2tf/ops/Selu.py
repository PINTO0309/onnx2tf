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
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Selu

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
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    alpha = graph_node.attrs.get('alpha', 1.67326319217681884765625)
    gamma = graph_node.attrs.get('gamma', 1.05070102214813232421875)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.clip_by_value(
            input_tensor,
            0,
            tf.reduce_max(input_tensor)) * gamma + \
                (
                    tf.exp(
                        tf.clip_by_value(
                            input_tensor,
                            tf.reduce_min(input_tensor),
                            0
                        )
                    ) - 1
                ) * alpha * gamma
