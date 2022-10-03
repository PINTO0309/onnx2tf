import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
)


@print_node_info
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Clip

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    graph_node_input = get_constant_or_variable(graph_node.inputs[0])
    min_value_node = get_constant_or_variable(graph_node.inputs[1])
    max_value_node = get_constant_or_variable(graph_node.inputs[2])
    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    features = None
    if isinstance(graph_node_input, gs.Variable):
        features = tf_layers_dict[graph_node_input.name]['tf_node']
    else:
        features = graph_node_input
    min_value = None
    if isinstance(min_value_node, gs.Variable) and min_value_node.shape is not None:
        min_value = tf_layers_dict[min_value_node.name]['tf_node']
    else:
        min_value = min_value_node
    max_value = None
    if isinstance(max_value_node, gs.Variable) and max_value_node.shape is not None:
        max_value = tf_layers_dict[max_value_node.name]['tf_node']
    else:
        max_value = max_value_node

    if isinstance(min_value, np.ndarray) and min_value == 0.0 \
        and isinstance(max_value, np.ndarray) and max_value == 6.0:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.nn.relu6(features=features)
    elif isinstance(min_value, np.ndarray) and min_value == 0.0 \
        and max_value.shape is None:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.nn.relu(features=features)
    else:
        if min_value.shape is not None and max_value.shape is not None:
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.clip_by_value(
                    t=features,
                    clip_value_min=min_value,
                    clip_value_max=max_value,
                )
        elif min_value.shape is not None and max_value.shape is None:
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.maximum(
                    x=features,
                    y=min_value,
                )
        elif min_value.shape is None and max_value.shape is not None:
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.minimum(
                    x=features,
                    y=max_value,
                )
