import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
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
    """Min

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    values = []
    for graph_node_input in graph_node.inputs:
        const_or_var = get_constant_or_variable(graph_node_input)
        if isinstance(const_or_var, gs.Variable):
            values.append(tf_layers_dict[const_or_var.name]['tf_node'])
        else:
            values.append(const_or_var)

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
    minimumed_tesnor = values[0]
    for i in range(1, len(values)):
        minimumed_tesnor = tf.minimum(minimumed_tesnor, values[i])

    tf_layers_dict[graph_node_output.name]['tf_node'] = minimumed_tesnor
