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
    process_neg_idx_along_axis,
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """GatherElements

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
    indices_tensor = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    tensor_rank = len(input_tensor.shape)

    axis = graph_node.attrs.get('axis', 0)
    axis = convert_axis(
        axis=axis,
        tensor_rank=tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    indices_tensor = process_neg_idx_along_axis(
        data=input_tensor,
        axis=axis,
        indices=indices_tensor,
    )

    if axis == 0:
        axis_perm = tf.range(tf.rank(input_tensor))
        data_swaped = input_tensor
        index_swaped = indices_tensor
    else:
        axis_perm = tf.tensor_scatter_nd_update(
            tf.range(tf.rank(input_tensor)),
            tf.constant([[0], [axis]]),
            tf.constant([axis, 0])
        )
        data_swaped = tf.transpose(input_tensor, perm=axis_perm)
        index_swaped = tf.transpose(indices_tensor, perm=axis_perm)

    idx_tensors_per_axis = [
        tf.range(tf.shape(index_swaped, index_swaped.dtype)[i]) \
            for i in range(index_swaped.shape.rank)
    ]

    idx_tensors_per_axis = tf.meshgrid(
        *idx_tensors_per_axis,
        indexing='ij',
    )
    idx_tensors_per_axis[0] = index_swaped
    dim_expanded_idx_tensors_per_axis = [
        tf.expand_dims(idx_tensor, axis=-1)
        for idx_tensor in idx_tensors_per_axis
    ]
    index_expanded = tf.concat(dim_expanded_idx_tensors_per_axis, axis=-1)

    gathered = tf.gather_nd(data_swaped, index_expanded)

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.transpose(gathered, perm=axis_perm)

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'GatherElements',
                'tf_inputs': {
                    'data': input_tensor,
                    'indices': indices_tensor,
                    'axis': axis,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
