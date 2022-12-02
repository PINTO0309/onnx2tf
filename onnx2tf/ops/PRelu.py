import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from tensorflow.python.keras.layers import PReLU
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    explicit_broadcast,
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
    """PRelu

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_2 = \
        tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    slope = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    replace_prelu_to_pseudo_prelu = kwargs['replace_prelu_to_pseudo_prelu']

    _, slope = explicit_broadcast(
        const_or_var_1=input_tensor,
        const_or_var_2=slope,
        graph_node=graph_node,
        tf_layers_dict= tf_layers_dict
    )
    slope_rank = len(slope.shape)

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input_1.name]['nhwc'] \
            if isinstance(graph_node_input_1, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False
    }

    # Generation of TF OP
    if replace_prelu_to_pseudo_prelu:
        pos = tf.nn.relu(input_tensor)
        neg = (input_tensor - abs(input_tensor)) * (slope * 0.5)
        tf_layers_dict[graph_node_output.name]['tf_node'] = pos + neg
    else:
        shared_axes = [val + 1 for val in range(len(input_tensor.shape) - 2)]
        tf_layers_dict[graph_node_output.name]['tf_node'] = PReLU(
            weights=slope,
            shared_axes=shared_axes,
        )(input_tensor)

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'PReLU',
                'tf_inputs': {
                    'x': input_tensor,
                    'slope': slope,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
