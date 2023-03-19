import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
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
    """Expand

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
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_tensor_shape = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP

    # Param replacement
    input_tensor = replace_parameter(
        value_before_replacement=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    input_tensor_shape = replace_parameter(
        value_before_replacement=input_tensor_shape,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    tf_type = None
    if \
        (
            isinstance(graph_node_input_1, gs.Variable) \
            and 'unnecessary_reshape' in tf_layers_dict[graph_node_input_1.name] \
            and tf_layers_dict[graph_node_input_1.name]['unnecessary_reshape'] == True
        ) or \
        (
            isinstance(graph_node_input_2, gs.Variable) \
            and 'unnecessary_reshape' in tf_layers_dict[graph_node_input_2.name] \
            and tf_layers_dict[graph_node_input_2.name]['unnecessary_reshape'] == True
        ):
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.identity(input=input_tensor)
        tf_type = tf.identity
    else:
        # tf.math.multiply does not support bool therefore use int8
        expanded_tensor = None
        if input_tensor.dtype is tf.bool:
            ones = tf.ones(input_tensor_shape, dtype=tf.int8)
            r = tf.cast(input_tensor, tf.int8) * ones
            expanded_tensor = tf.cast(r, tf.bool)
        else:
            ones = tf.ones(input_tensor_shape, dtype=input_tensor.dtype)
            expanded_tensor = input_tensor * ones
        tf_layers_dict[graph_node_output.name]['tf_node'] = expanded_tensor
        tf_type = 'Expand'

    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_type,
                'tf_inputs': {
                    'input_tensor': input_tensor,
                    'input_tensor_shape': input_tensor_shape,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
