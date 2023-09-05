import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """LayerNormalization

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

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    graph_node_input_2 = None
    if hasattr(graph_node.inputs[1], 'values'):
        graph_node_input_2 = graph_node.inputs[1].values
    else:
        graph_node_input_2 = graph_node.inputs[1]
    graph_node_input_3 = None
    if len(graph_node.inputs) >= 3:
        if hasattr(graph_node.inputs[2], 'values'):
            graph_node_input_3 = graph_node.inputs[2].values
        else:
            graph_node_input_3 = graph_node.inputs[2]

    scale = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    bias = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    graph_node_output_1: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output_1.shape
    dtype = graph_node_output_1.dtype

    # graph_node_output_2: gs.Variable = None
    # if len(graph_node.outputs) >= 2:
    #     graph_node_output_2: gs.Variable = graph_node.outputs[1]
    # graph_node_output_3: gs.Variable = None
    # if len(graph_node.outputs) >= 3:
    #     graph_node_output_3: gs.Variable = graph_node.outputs[2]


    axis = graph_node.attrs.get('axis', -1)
    axis = convert_axis(
        axis=axis,
        tensor_rank=input_tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )
    epsilon = graph_node.attrs.get('epsilon', 1e-05)
    stash_type = bool(graph_node.attrs.get('stash_type', 1))

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output_1.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input_1.name]['nhwc'] \
            if isinstance(graph_node_input_1, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False
    }

    # Generation of TF OP
    tf_layers_dict[graph_node_output_1.name]['tf_node'] = \
        tf.keras.layers.LayerNormalization(
            axis=[axis],
            epsilon=epsilon,
            gamma_initializer=tf.keras.initializers.constant(scale) if scale is not None else 'ones',
            beta_initializer=tf.keras.initializers.constant(bias) if bias is not None else 'zeros',
        )(input_tensor)

    # Post-process transpose
    tf_layers_dict[graph_node_output_1.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output_1.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output_1.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'LayerNormalization',
                'tf_inputs': {
                    'input': input_tensor,
                    'scale': scale,
                    'bias': bias,
                    'axis': axis,
                    'epsilon': epsilon,
                    'stash_type': stash_type,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output_1.name]['tf_node'],
                },
            }
        )
