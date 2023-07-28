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
    make_tf_node_info,
    convert_axis,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)
from onnx2tf.utils.logging import *
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
    """SplitToSequence

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
    graph_node_input_2 = None
    if len(graph_node.inputs) >= 2:
        graph_node_input_2 = get_constant_or_variable(
            graph_node.inputs[1],
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
    split = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    axis = graph_node.attrs.get('axis', 0)
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(graph_node_input_1.shape),
        before_op_output_shape_trans=before_op_output_shape_trans,
    )
    keepdims = bool(graph_node.attrs.get('keepdims', 1))

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    if split is not None:
        split_shape = split.shape
        # check if the split is 1-d or scalar
        if split_shape.shape[0] == 1:
            split_sizes = split
        else:
            error(
                f'Split to sequence with scalar split is not supported due to API limitations.\n'+
                f'graph_node.name: {graph_node.name} split.shape: {split_shape}'
            )
            sys.exit(1)
        split_inputs = tf.split(
            value=input_tensor,
            num_or_size_splits=split_sizes,
            axis=axis,
        )

    else:
        # split is not provided, use default 1
        split_sizes = tf.tile(
            input=[1],
            multiples=tf.reshape(input_tensor_shape[axis], [1])
        )
        split_inputs = tf.split(
            value=input_tensor,
            num_or_size_splits=split_sizes,
            axis=axis,
        )
        if not keepdims:
            split_inputs = [
                tf.squeeze(split_input) for split_input in split_inputs
            ]

    output_dtype = NUMPY_DTYPES_TO_TF_DTYPES[dtype] \
        if isinstance(dtype, np.dtype) else dtype

    # create an empty sequence next
    input_sequence = tf.ragged.constant([], dtype=output_dtype)

    # insert tensors at the end of sequence
    for i in range(len(split_inputs)):
        input_tensor = tf.expand_dims(split_inputs[i], 0)
        if input_sequence.shape[0] == 0:
            output_seq = tf.RaggedTensor.from_tensor(input_tensor)
        else:
            output_seq = tf.concat([input_sequence, input_tensor], axis=0)
        input_sequence = output_seq

    tf_layers_dict[graph_node_output.name]['tf_node'] = output_seq

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
                'tf_op_type': 'SplitToSequence',
                'tf_inputs': {
                    'input': input_tensor,
                    'axis': axis,
                    'keepdims': keepdims,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
