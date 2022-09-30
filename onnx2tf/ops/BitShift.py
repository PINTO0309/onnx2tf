import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs


def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """BitShift

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    graph_node_input_1: gs.Variable = graph_node.inputs[0]
    graph_node_input_2: gs.Variable = graph_node.inputs[1]
    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    direction = graph_node.attrs.get('direction', 'RIGHT')

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    if direction == 'RIGHT':
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.bitwise.right_shift(
                x=tf_layers_dict[graph_node_input_1.name]['tf_node'],
                y=tf_layers_dict[graph_node_input_2.name]['tf_node'],
                name=graph_node.name,
            )
    elif direction == 'LEFT':
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.bitwise.left_shift(
                x=tf_layers_dict[graph_node_input_1.name]['tf_node'],
                y=tf_layers_dict[graph_node_input_2.name]['tf_node'],
                name=graph_node.name,
            )
