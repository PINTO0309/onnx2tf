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
    pre_explicit_broadcast,
    explicit_broadcast,
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
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans \
            if graph_node.inputs[1].shape is not None \
                and input_tensor_rank == len(graph_node.inputs[1].shape) else False,
    )
    slope = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    replace_prelu_to_pseudo_prelu = "prelu" in kwargs['replace_to_pseudo_operators']

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

    input_tensor, slope = \
        pre_explicit_broadcast(
            input_tensor_1=input_tensor,
            input_tensor_2=slope,
        )
    input_tensor, slope = \
        explicit_broadcast(
            const_or_var_1=input_tensor,
            const_or_var_2=slope,
            graph_node=graph_node,
            tf_layers_dict= tf_layers_dict,
        )

    # input_tensor: [1, 4, 4, 4]
    # slope: [4, 1, 1] -> [1, 4, 1, 1] -> [1, 1, 1, 4]
    # https://github.com/PINTO0309/onnx2tf/issues/418
    if tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and input_tensor_shape is not None \
        and input_tensor_rank >= 3 \
        and sum([1 if isinstance(s, int) and s == input_tensor_shape[1] else 0 for s in input_tensor_shape]) == input_tensor_rank - 1 \
        and slope.shape is not None \
        and len(slope.shape) >= 3 \
        and input_tensor_rank == len(slope.shape) \
        and isinstance(slope, np.ndarray) \
        and slope.shape[-1] == 1:
        convertion_table = [0] + [i for i in range(2, input_tensor_rank)] + [1]
        slope = slope.transpose(convertion_table)

    slope = tf.convert_to_tensor(slope)

    # Pre-process transpose
    before_trans_shape = input_tensor_shape
    input_tensor = \
        pre_process_transpose(
            value_before_transpose=input_tensor,
            param_target='inputs',
            param_name=graph_node.inputs[0].name,
            **kwargs,
        )
    after_trans_shape = input_tensor.shape
    if 'nhwc' in tf_layers_dict[graph_node_output.name].keys() \
        and tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and before_trans_shape != after_trans_shape:
        tf_layers_dict[graph_node_output.name].pop('nhwc')

    # Generation of TF OP
    if replace_prelu_to_pseudo_prelu:
        pos = tf.nn.relu(input_tensor)
        neg = (input_tensor - abs(input_tensor)) * (slope * 0.5)
        tf_layers_dict[graph_node_output.name]['tf_node'] = pos + neg
    else:
        if slope.shape is not None \
            and len(slope.shape) > 0 \
            and sum([1 if dim is not None and dim == 1 else 0 for dim in slope.shape]) == len(slope.shape):
            shared_axes = [val + 1 for val in range(len(input_tensor.shape) - 1)]
        else:
            shared_axes = [val + 1 for val in range(len(input_tensor.shape) - 2)]
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            PReLU(
                weights=slope,
                shared_axes=shared_axes,
            )(input_tensor)

    # Post-process transpose
    before_trans_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )
    after_trans_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    if 'nhwc' in tf_layers_dict[graph_node_output.name].keys() \
        and tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and before_trans_shape != after_trans_shape:
        tf_layers_dict[graph_node_output.name].pop('nhwc')

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
