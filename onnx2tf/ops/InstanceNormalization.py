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
    """InstanceNormalization

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
    before_op_output_shape_trans_3 = \
        tf_layers_dict.get(graph_node.inputs[2].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2 \
        and before_op_output_shape_trans_3

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor.shape)

    scale = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
        is_bias=True,
    )
    B = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans,
        is_bias=True,
    )

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    epsilon = graph_node.attrs.get('epsilon', 1e-05)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    moments_axes = list(range(input_tensor_rank))[1:-1]
    channel_size = input_tensor_shape[-1] \
        if input_tensor_shape[-1] is not None else scale.shape[0]
    params_shape_broadcast = \
        list([1] + [1 for _ in range(2, input_tensor_rank)] + [channel_size])
    if input_tensor_rank == 3 and graph_node.i().op == 'Reshape':
        params_shape_broadcast = [1, len(B), 1]
        B = tf.reshape(tensor=B, shape=[1] + [len(B)] + [1])
        scale = tf.reshape(tensor=scale, shape=[1] + [len(scale)] + [1])
    beta = tf.reshape(B, params_shape_broadcast)
    gamma = tf.reshape(scale, params_shape_broadcast)

    # Calculate the moments (instance activations).
    mean, variance = tf.nn.moments(
        x=input_tensor,
        axes=moments_axes,
        keepdims=True,
    )

    # Compute instance normalization
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.nn.batch_normalization(
            x=input_tensor,
            mean=mean,
            variance=variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=epsilon,
            name=graph_node.name,
        )

    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    if len(shape) == 3 and graph_node.o().op == 'Reshape':
        tf_layers_dict[graph_node_output.name]['is_instance_norm_3d'] = True

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.nn.batch_normalization,
                'tf_inputs': {
                    'x': input_tensor,
                    'mean': mean,
                    'variance': variance,
                    'offset': beta,
                    'scale': gamma,
                    'variance_epsilon': epsilon,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
