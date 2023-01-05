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
    convert_axis,
    convert_reverse_axis,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
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
    """Inverse

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

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # https://pytorch.org/docs/stable/generated/torch.linalg.inv.html#torch.linalg.inv
    # https://www.tensorflow.org/api_docs/python/tf/linalg/inv
    # The input is a tensor of shape [..., M, M] whose inner-most 2 dimensions form square matrices.
    # The output is a tensor of the same shape as the input containing the inverse for all input submatrices [..., :, :].
    nhwc_flag = tf_layers_dict[graph_node_input.name]['nhwc'] \
        if 'nhwc' in tf_layers_dict[graph_node_input.name].keys() else False
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)
    inv_transpose = \
        nhwc_flag == True \
        and input_tensor_rank >= 4 \
        and input_tensor_shape[input_tensor_rank-3] == input_tensor_shape[input_tensor_rank-2]
    if inv_transpose:
        perm = [
            convert_axis(
                axis=axis,
                tensor_rank=input_tensor_rank,
                before_op_output_shape_trans=True
            ) for axis in range(input_tensor_rank)
        ]
        input_tensor = transpose_with_flexing_deterrence(
            input_tensor=input_tensor,
            perm=perm,
            **kwargs,
        )

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.linalg.inv(
            input=input_tensor,
            name=graph_node.name,
        )

    if inv_transpose:
        perm = [
            convert_reverse_axis(
                axis=axis,
                tensor_rank=input_tensor_rank,
                before_op_output_shape_trans=True
            ) for axis in range(input_tensor_rank)
        ]
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            transpose_with_flexing_deterrence(
                input_tensor=tf_layers_dict[graph_node_output.name]['tf_node'],
                perm=perm,
                **kwargs,
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
                'tf_op_type': tf.linalg.inv,
                'tf_inputs': {
                    'x': input_tensor,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
