import sys
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
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES
from onnx2tf.utils.logging import *


def _as_tensor(value):
    if isinstance(value, np.ndarray):
        return tf.convert_to_tensor(value)
    if isinstance(value, (np.generic, int, float, bool, str, bytes)):
        return tf.convert_to_tensor(value)
    return value


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """RMSNormalization

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
    output_tf_dtype = NUMPY_DTYPES_TO_TF_DTYPES[dtype] \
        if isinstance(dtype, np.dtype) else dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    scale_tensor = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input_1.name]['nhwc'] \
            if isinstance(graph_node_input_1, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False
    }

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    scale_tensor = pre_process_transpose(
        value_before_transpose=scale_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )

    # Generation of TF OP
    input_tensor = _as_tensor(input_tensor)
    scale_tensor = _as_tensor(scale_tensor)

    input_rank = input_tensor.shape.rank
    if input_rank is None and graph_node.inputs[0].shape is not None:
        input_rank = len(graph_node.inputs[0].shape)
    if input_rank is None:
        error(
            f'RMSNormalization requires known input rank.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    axis = graph_node.attrs.get('axis', -1)
    axis = convert_axis(
        axis=axis,
        tensor_rank=input_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )
    epsilon = graph_node.attrs.get('epsilon', 1e-05)
    stash_type = int(graph_node.attrs.get('stash_type', 1))

    axes = list(range(axis, input_rank))

    compute_dtype = input_tensor.dtype
    if stash_type == 1 and input_tensor.dtype != tf.float32:
        compute_dtype = tf.float32

    x = tf.cast(input_tensor, compute_dtype)
    xsquared = tf.math.square(x)
    xsquared_mean = tf.reduce_mean(xsquared, axis=axes, keepdims=True)
    rms = tf.sqrt(xsquared_mean + tf.cast(epsilon, compute_dtype))
    normalized = x / rms

    if compute_dtype != input_tensor.dtype:
        normalized = tf.cast(normalized, input_tensor.dtype)
    if scale_tensor.dtype != normalized.dtype:
        scale_tensor = tf.cast(scale_tensor, normalized.dtype)

    output_tensor = tf.math.multiply(
        normalized,
        scale_tensor,
        name=graph_node.name,
    )

    if output_tf_dtype is not None and output_tensor.dtype != output_tf_dtype:
        output_tensor = tf.cast(output_tensor, output_tf_dtype)

    tf_layers_dict[graph_node_output.name]['tf_node'] = output_tensor

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
                'tf_op_type': 'RMSNormalization',
                'tf_inputs': {
                    'input': input_tensor,
                    'scale': scale_tensor,
                    'axis': axis,
                    'epsilon': epsilon,
                    'stash_type': stash_type,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
