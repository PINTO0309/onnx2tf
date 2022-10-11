import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.enums import ONNX_DTYPES_TO_TF_DTYPES
from onnx2tf.utils.common_functions import (
    print_node_info,
    make_tf_node_info,
)


@print_node_info
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """RandomUniform

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    rdtype = graph_node.attrs.get('dtype', 1)
    rhigh = graph_node.attrs.get('high', 1.0)
    rlow = graph_node.attrs.get('low', 0.0)
    rseed = graph_node.attrs.get('seed', 0)
    rshape = graph_node.attrs['shape']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.random.uniform(
            shape=rshape,
            minval=rlow,
            maxval=rhigh,
            dtype=ONNX_DTYPES_TO_TF_DTYPES[rdtype],
            seed=rseed,
            name=graph_node.name,
        )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.random.uniform,
                'tf_inputs': {
                    'shape': shape,
                    'minval': rlow,
                    'maxval': rhigh,
                    'dtype': ONNX_DTYPES_TO_TF_DTYPES[rdtype],
                    'seed': rseed,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
