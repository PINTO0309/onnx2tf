import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from utils.common_functions import get_constant_or_variable


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
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.clip_by_value(
            t=tf_layers_dict[graph_node_input.name]['tf_node'] \
                if isinstance(graph_node_input, gs.Variable) else graph_node_input,
            clip_value_min=tf_layers_dict[min_value_node.name]['tf_node'] \
                if isinstance(min_value_node, gs.Variable) else min_value_node,
            clip_value_max=tf_layers_dict[max_value_node.name]['tf_node'] \
                if isinstance(max_value_node, gs.Variable) else max_value_node,
        )
