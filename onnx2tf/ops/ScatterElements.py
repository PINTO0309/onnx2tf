import sys
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
    process_neg_idx_along_axis,
    make_tf_node_info,
    convert_axis,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
)
from onnx2tf.utils.colors import Color
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
    """ScatterElements

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

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
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
    input_tensor_rank = len(input_tensor_shape)
    indices_tensor = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    # Pre-process transpose
    indices_tensor = pre_process_transpose(
        value_before_transpose=indices_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    updates_tensor = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    # Pre-process transpose
    updates_tensor = pre_process_transpose(
        value_before_transpose=updates_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[2].name,
        **kwargs,
    )
    updates_tensor_shape = updates_tensor.shape
    updates_tensor_rank = len(updates_tensor_shape)

    axis = graph_node.attrs.get('axis', 0)
    axis = convert_axis(
        axis=axis,
        tensor_rank=input_tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )
    reduction = graph_node.attrs.get('reduction', 'none')
    enable_reductions = ['none']
    if reduction not in enable_reductions:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'ScatterElements currently supports only reduction={enable_reductions}. '+
            f'Pull requests are welcome. \n'+
            f'graph_node.name: {graph_node.name} reduction: {reduction}'
        )
        sys.exit(1)

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
    sparsified_dense_idx_shape = updates_tensor_shape

    if None not in sparsified_dense_idx_shape:
        idx_tensors_per_axis = [
            tf.range(sparsified_dense_idx_shape[i])
            for i in range(updates_tensor_rank)
        ]
    else:
        idx_tensors_per_axis = [
            tf.range(tf.shape(updates_tensor)[i])
            for i in range(updates_tensor_rank)
        ]

    idx_tensors_per_axis = tf.meshgrid(*idx_tensors_per_axis, indexing='ij')
    idx_tensors_per_axis[axis] = indices_tensor
    dim_expanded_idx_tensors_per_axis = [
        tf.expand_dims(idx_tensor, axis=-1)
        for idx_tensor in idx_tensors_per_axis
    ]
    new_dim_expanded_idx_tensors_per_axis = []
    for dim in dim_expanded_idx_tensors_per_axis:
        new_dim_expanded_idx_tensors_per_axis.append(tf.cast(dim, dtype=tf.int64))
    dim_expanded_idx_tensors_per_axis = new_dim_expanded_idx_tensors_per_axis

    # If the tensor to be concat contains different shapes, transpose the shapes using majority logic
    # However, only tensors whose geometry is all None are considered.
    # This is a special process for Faster-RCNN
    # TODO: Correct any other appropriate error workarounds.
    new_dim_expanded_idx_tensors_per_axis = []
    for dim in dim_expanded_idx_tensors_per_axis:
        if dim is not None and len([i for i in dim.shape if i is None]) == len(dim.shape)-1 and dim.shape[-1] == 1 and len(dim.shape) == 5:
            new_dim_expanded_idx_tensors_per_axis.append(
                transpose_with_flexing_deterrence(
                    input_tensor=dim,
                    perm=[0,2,3,1,4],
                    **kwargs,
                )
            )
        else:
            new_dim_expanded_idx_tensors_per_axis.append(dim)
    dim_expanded_idx_tensors_per_axis = new_dim_expanded_idx_tensors_per_axis

    coordinate = tf.concat(
        dim_expanded_idx_tensors_per_axis,
        axis=-1,
    )

    indices = tf.reshape(coordinate, [-1, input_tensor_rank])
    updates = tf.reshape(updates_tensor, [-1])
    output = tf.tensor_scatter_nd_update(
        tensor=input_tensor,
        indices=indices,
        updates=updates,
    )
    output_dtype = NUMPY_DTYPES_TO_TF_DTYPES[input_tensor.dtype] \
        if isinstance(input_tensor.dtype, np.dtype) else input_tensor.dtype
    output = tf.cast(output, output_dtype)

    tf_layers_dict[graph_node_output.name]['tf_node'] = output

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
                'tf_op_type': tf.tensor_scatter_nd_update,
                'tf_inputs': {
                    'tensor': input_tensor,
                    'indices': indices_tensor,
                    'updates': updates_tensor,
                    'axis': axis,
                    'reduction': reduction,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
