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
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Shrink

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

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

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

    lambd = graph_node.attrs.get('lambd', 0.5)
    bias = graph_node.attrs.get('bias', 0.0)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    output_dtype = NUMPY_DTYPES_TO_TF_DTYPES[input_tensor.dtype] \
        if isinstance(input_tensor.dtype, np.dtype) else input_tensor.dtype
    lambd_tensor = tf.fill(
        dims=input_tensor_shape,
        value=tf.constant(lambd, output_dtype),
    )
    lambd_neg_tensor = tf.fill(
        dims=input_tensor_shape,
        value=tf.constant(lambd * -1, output_dtype),
    )
    bias_tensor = tf.fill(
        dims=input_tensor_shape,
        value=tf.constant(bias, output_dtype),
    )
    zeros_tensor = tf.zeros(
        shape=input_tensor_shape,
        dtype=output_dtype,
    )

    # prepare return values and conditions
    input_plus = tf.add(
        x=input_tensor,
        y=bias_tensor,
    )
    input_minus = tf.subtract(
        x=input_tensor,
        y=bias_tensor,
    )
    greater_cond = tf.greater(
        x=input_tensor,
        y=lambd_tensor,
    )
    less_cond = tf.less(
        x=input_tensor,
        y=lambd_neg_tensor,
    )

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.where(
            condition=less_cond,
            x=input_plus,
            y=tf.where(
                condition=greater_cond,
                x=input_minus,
                y=zeros_tensor,
            )
        )

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
                'tf_op_type': 'Shrink',
                'tf_inputs': {
                    'input': input_tensor,
                    'bias': bias,
                    'lambd': lambd,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
