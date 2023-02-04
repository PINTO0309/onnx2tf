import copy
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from typing import List
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
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
    """Gather

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

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    indices = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    before_cast_indices = None
    if isinstance(indices, np.ndarray) and indices.ndim == 1 and indices[0] is not None:
        if indices[0] >= 0:
            before_cast_indices = indices[0]
    elif isinstance(indices, np.ndarray) and indices.ndim == 0 and indices is not None:
        if indices >= 0:
            before_cast_indices = int(indices)

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    axis = graph_node.attrs.get("axis", 0)
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(input_tensor.shape),
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    optimization_for_gpu_delegate: bool = \
        kwargs['optimization_for_gpu_delegate']

    # tensorflow gather supports only positive indices
    out_dtype = NUMPY_DTYPES_TO_TF_DTYPES[indices.dtype] \
        if isinstance(indices.dtype, np.dtype) else indices.dtype
    if not optimization_for_gpu_delegate:
        cond = tf.cast(indices < 0, dtype=out_dtype)
        indices = tf.cast(
            indices + cond * tf.shape(input_tensor, out_type=out_dtype)[axis],
            dtype=out_dtype,
        )
    else:
        cond = indices < 0
        indices = indices + tf.cast(
            tf.where(
                cond,
                tf.convert_to_tensor(1, dtype=out_dtype),
                tf.convert_to_tensor(0, dtype=out_dtype)
            ) * tf.shape(input_tensor, out_type=out_dtype)[axis],
            dtype=out_dtype,
        )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Param replacement
    input_tensor = replace_parameter(
        value_before_replacement=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    indices = replace_parameter(
        value_before_replacement=indices,
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

    # Gather + Unsqueeze -> strided_slice
    # Replace combination of Gather and Unsqueeze
    # with strided_slice if available
    consumer_count = 0
    consumer_nodes: List[gs.Node] = []
    while True:
        try:
            consumer_node =  graph_node.o(consumer_count, 0)
            consumer_nodes.append(consumer_node)
            consumer_count += 1
        except:
            break
    unsqueeze_count = 0
    for consumer_node in consumer_nodes:
        if consumer_node.op == 'Unsqueeze' \
            and hasattr(consumer_node, 'attrs') \
            and 'axes' in consumer_node.attrs \
            and len(consumer_node.attrs['axes']) == 1 \
            and consumer_node.attrs['axes'][0] == axis:
            unsqueeze_count += 1

    # Generation of TF OP
    if unsqueeze_count == consumer_count \
        and before_cast_indices is not None:
        # Replace
        ind = before_cast_indices
        begin_ = [
            0 if idx != axis else ind \
                for idx in range(len(input_tensor.shape))
        ]
        end_ = [
            0 if idx != axis else ind + 1 \
                for idx in range(len(input_tensor.shape))
        ]
        begin_mask_ = sum(
            [
                2**idx if idx != axis else 0 \
                    for idx in range(len(input_tensor.shape))
            ]
        )
        end_mask_ = begin_mask_
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.strided_slice(
                input_=input_tensor,
                begin=begin_,
                end=end_,
                begin_mask=begin_mask_,
                end_mask=end_mask_,
            )
        tf_layers_dict[graph_node_output.name]['unnecessary_gather'] = True
    else:
        # No-replace
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.gather(
                params=input_tensor,
                indices=indices,
                axis=axis,
                name=graph_node.name,
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
                'tf_op_type': tf.gather,
                'tf_inputs': {
                    'params': input_tensor,
                    'indices': indices,
                    'axis': axis,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
