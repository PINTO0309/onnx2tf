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
    """Hardmax

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

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

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
    tensor_rank = len(input_tensor_shape)

    opset = kwargs['opset']
    axis = -1
    if opset >= 13:
        axis = graph_node.attrs.get('axis', axis)
    else:
        axis = graph_node.attrs.get('axis', 1)
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(input_tensor_shape),
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    x = None
    if axis == tensor_rank - 1:
        x = input_tensor
    else:
        if opset >= 13:
            perm1 = tf.range(
                start=0,
                limit=axis,
            )
            perm2 = tf.range(
                start=axis + 1,
                limit=tensor_rank - 1
            )
            perm = tf.concat(
                values=[perm1, [tensor_rank - 1], perm2, [axis]],
                axis=-1,
            )
            x = tf.transpose(
                a=input_tensor,
                perm=perm,
            )
        else:
            cal_shape = (
                tf.reduce_prod(input_tensor=input_tensor_shape[0:axis]),
                tf.reduce_prod(input_tensor=input_tensor_shape[axis:tensor_rank]),
            )
            x = tf.reshape(
                tensor=input_tensor,
                shape=cal_shape,
            )

    x_shape = x.shape
    depth = x_shape[-1]
    onehoted_tensor = tf.one_hot(
        indices=tf.argmax(
            input=x,
            axis=-1,
        ),
        depth=depth,
        dtype=x.dtype,
        name=graph_node.name,
    )

    tf_layers_dict[graph_node_output.name]['tf_node'] = onehoted_tensor

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
                'tf_op_type': 'Hardmax',
                'tf_inputs': {
                    'input': input_tensor,
                    'axis': axis,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
