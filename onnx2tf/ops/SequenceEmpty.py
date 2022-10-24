import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
)
from onnx2tf.utils.enums import ONNX_DTYPES_TO_TF_DTYPES


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """SequenceEmpty

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    sequence_dtype = ONNX_DTYPES_TO_TF_DTYPES(graph_node.attrs.get('dtype', 1)) # Float32

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
    ragged = tf.RaggedTensor.from_row_lengths(values=[], row_lengths=[])
    sparse = tf.cast(ragged.to_sparse(), sequence_dtype)

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.RaggedTensor.from_sparse(sparse)

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'SequenceEmpty',
                'tf_inputs': {
                    'dtype': sequence_dtype,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
