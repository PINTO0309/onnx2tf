import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from utils.common_functions import (
    get_constant_or_variable,
    convert_axis,
    print_node_info,
)


@print_node_info
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Transpose

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    graph_node_input = get_constant_or_variable(graph_node.inputs[0])
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    tensor_rank = len(input_tensor.shape)

    perm = graph_node.attrs.get('perm', [idx for idx in reversed(range(tensor_rank))])

    if isinstance(perm, list) or (isinstance(perm, np.ndarray) and len(perm.shape) > 0):
        perm = [convert_axis(axis=idx, tensor_rank=tensor_rank) for idx in perm]
    elif perm is not None and isinstance(perm, np.ndarray) and len(perm.shape) == 0:
        perm = convert_axis(axis=perm, tensor_rank=tensor_rank)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.transpose(
            a=input_tensor,
            perm=list(perm) if perm is not None else None,
            name=graph_node.name,
        )
