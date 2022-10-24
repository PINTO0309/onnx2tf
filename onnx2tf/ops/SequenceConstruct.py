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
    """SequenceConstruct

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

    input_sequence = tf.ragged.constant([], dtype=graph_node.inputs[0].dtype)

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
    output_seq = None
    for input in graph_node.inputs:
        graph_node_input = get_constant_or_variable(
            input,
            before_op_output_shape_trans,
        )
        input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
            if isinstance(graph_node_input, gs.Variable) else graph_node_input
        expanded_input_tensor = tf.expand_dims(input_tensor, 0)
        if input_sequence.shape[0] == 0:
            output_seq = tf.RaggedTensor.from_tensor(expanded_input_tensor)
        else:
            output_seq = tf.concat([input_sequence, expanded_input_tensor], axis=0)
        input_sequence = output_seq

    tf_layers_dict[graph_node_output.name]['tf_node'] = output_seq

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'SequenceConstruct',
                'tf_inputs': {
                    'graph_node.inputs[0]': graph_node.inputs[0],
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
