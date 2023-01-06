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
    transpose_with_flexing_deterrence,
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
    """TopK

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

    X = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    K = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    Values: gs.Variable = graph_node.outputs[0]
    Indices: gs.Variable = graph_node.outputs[1]
    Values_shape = Values.shape
    Values_dtype = Values.dtype
    Indices_shape = Indices.shape
    Indices_dtype = Indices.dtype

    input_tensor = tf_layers_dict[X.name]['tf_node'] \
        if isinstance(X, gs.Variable) else X
    k_tensor = tf_layers_dict[K.name]['tf_node'] \
        if isinstance(K, gs.Variable) else K
    k_tensor = int(k_tensor) \
        if isinstance(k_tensor, np.ndarray) else tf.cast(k_tensor, dtype=tf.int32)
    tensor_rank = len(input_tensor.shape)

    axis = graph_node.attrs.get('axis', -1)
    axis = convert_axis(
        axis=axis,
        tensor_rank=tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )
    largest = bool(graph_node.attrs.get('largest', 1))
    sorted = bool(graph_node.attrs.get('sorted', 1))

    # Preserving Graph Structure (Dict)
    tf_layers_dict[Values.name] = {
        'optype': graph_node.op,
        'shape': Values_shape,
        'dtype': Values_dtype,
    }
    tf_layers_dict[Indices.name] = {
        'optype': graph_node.op,
        'shape': Indices_shape,
        'dtype': Indices_dtype,
    }

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    topked_values = None
    topked_indices = None
    perm = None
    if axis != (tensor_rank-1):
        perm = [idx for idx in range(tensor_rank) if idx != axis] + [axis]
        input_tensor = \
            transpose_with_flexing_deterrence(
                input_tensor=input_tensor,
                perm=perm,
                **kwargs,
            )

    # MLIR only accepts scalar values for k-values, thus compressing the dimension
    if isinstance(k_tensor, int):
        pass
    elif k_tensor.shape is not None and len(k_tensor.shape) >= 1:
        k_tensor = tf.squeeze(k_tensor)

    if largest:
        topked_values, topked_indices = \
            tf.math.top_k(
                input=input_tensor,
                k=k_tensor,
                sorted=sorted,
                name=graph_node.name,
            )
    else:
        topked_values, topked_indices = \
            tf.math.top_k(
                input=tf.negative(input_tensor),
                k=k_tensor,
                sorted=sorted,
                name=graph_node.name,
            )
        topked_values = tf.negative(topked_values)
    topked_indices = tf.cast(
        x=topked_indices,
        dtype=NUMPY_DTYPES_TO_TF_DTYPES[Indices_dtype],
    )

    if axis != (tensor_rank-1):
        perm = [perm.index(idx) for idx in range(tensor_rank)]
        topked_values = \
            transpose_with_flexing_deterrence(
                input_tensor=topked_values,
                perm=perm,
                **kwargs,
            )
        topked_indices = \
            transpose_with_flexing_deterrence(
                input_tensor=topked_indices,
                perm=perm,
                **kwargs,
            )

    tf_layers_dict[Values.name]['tf_node'] = topked_values
    tf_layers_dict[Indices.name]['tf_node'] = topked_indices

    # Post-process transpose
    tf_layers_dict[Values.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[Values.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )
    tf_layers_dict[Indices.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[Indices.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[1].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[Values.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.math.top_k,
                'tf_inputs': {
                    'input': input_tensor,
                    'k': k_tensor,
                    'sorted': sorted,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[Values.name]['tf_node'],
                },
            }
        )
    tf_layers_dict[Indices.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.math.top_k,
                'tf_inputs': {
                    'input': input_tensor,
                    'k': k_tensor,
                    'sorted': sorted,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[Indices.name]['tf_node'],
                },
            }
        )