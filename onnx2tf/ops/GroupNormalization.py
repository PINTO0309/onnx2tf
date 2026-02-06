import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tf_keras
import onnx2tf.gs as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
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
    """GroupNormalization

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

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    scale = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans \
            if graph_node.inputs[1].shape is not None and len(graph_node.inputs[1].shape) != 1 else False,
        is_bias=True,
    )
    scale_dtype = NUMPY_DTYPES_TO_TF_DTYPES[scale.dtype] \
        if isinstance(scale.dtype, np.dtype) else scale.dtype
    scale = tf.convert_to_tensor(scale, dtype=scale_dtype) \
        if isinstance(scale, np.ndarray) else scale

    bias = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans \
            if graph_node.inputs[2].shape is not None and len(graph_node.inputs[2].shape) != 1 else False,
        is_bias=True,
    )
    bias_dtype = NUMPY_DTYPES_TO_TF_DTYPES[bias.dtype] \
        if isinstance(bias.dtype, np.dtype) else bias.dtype
    bias = tf.convert_to_tensor(bias, dtype=bias_dtype) \
        if isinstance(bias, np.ndarray) else bias

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    epsilon = graph_node.attrs.get('epsilon', 1e-05)
    num_groups = int(graph_node.attrs.get('num_groups', 1))
    stash_type = int(graph_node.attrs.get('stash_type', 1))
    opset = kwargs.get('opset', None)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input.name]['nhwc'] \
            if isinstance(graph_node_input, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input.name].keys() else False
    }

    input_rank = input_tensor.shape.rank
    if input_rank is None:
        input_rank = tf.rank(input_tensor)

    channel_axis = -1 if before_op_output_shape_trans else 1
    channel_axis_idx = channel_axis
    if isinstance(input_rank, int):
        channel_axis_idx = channel_axis if channel_axis >= 0 else input_rank + channel_axis

    internal_perm = None
    internal_inverse_perm = None
    if isinstance(input_rank, int) and channel_axis_idx != (input_rank - 1):
        perm = [i for i in range(input_rank) if i != channel_axis_idx] + [channel_axis_idx]
        internal_perm = perm
        internal_inverse_perm = [0] * input_rank
        for i, p in enumerate(perm):
            internal_inverse_perm[p] = i
    elif not isinstance(input_rank, int) and channel_axis != -1:
        rank_t = tf.cast(input_rank, tf.int32)
        perm = tf.concat([
            tf.range(channel_axis),
            tf.range(channel_axis + 1, rank_t),
            [channel_axis],
        ], axis=0)
        internal_perm = perm
        internal_inverse_perm = tf.argsort(perm)

    x = input_tensor
    if internal_perm is not None:
        x = transpose_with_flexing_deterrence(
            input_tensor=x,
            perm=internal_perm,
            **kwargs,
        )

    input_dtype = x.dtype
    calc_dtype = tf.float32 if stash_type == 1 else input_dtype
    x = tf.cast(x, calc_dtype)

    x_shape = tf.shape(x, out_type=tf.int32)
    channels = x_shape[-1]
    group_size = tf.math.floordiv(channels, num_groups)

    group_shape = tf.stack([num_groups, group_size], axis=0)
    new_shape = tf.concat([x_shape[:-1], group_shape], axis=0)
    x_grouped = tf.reshape(x, new_shape)

    rank_with_group = tf.rank(x_grouped)
    spatial_axes = tf.range(1, rank_with_group - 2)
    reduce_axes = tf.concat(
        [spatial_axes, tf.expand_dims(rank_with_group - 1, axis=0)],
        axis=0,
    )

    mean, variance = tf.nn.moments(x_grouped, axes=reduce_axes, keepdims=True)
    x_norm = (x_grouped - mean) * tf.math.rsqrt(variance + tf.cast(epsilon, calc_dtype))
    x_norm = tf.cast(x_norm, input_dtype)

    if opset is not None and opset < 21:
        rank_with_group = x_grouped.shape.rank
        if rank_with_group is not None:
            scale_shape = [1] * (rank_with_group - 2) + [num_groups, 1]
            scale_group = tf.reshape(scale, scale_shape)
            bias_group = tf.reshape(bias, scale_shape)
        else:
            rank_with_group = tf.rank(x_grouped)
            prefix_ones = tf.fill([rank_with_group - 2], 1)
            scale_shape = tf.concat(
                [prefix_ones, tf.constant([num_groups, 1], dtype=tf.int32)],
                axis=0,
            )
            scale_group = tf.reshape(scale, scale_shape)
            bias_group = tf.reshape(bias, scale_shape)
        x_norm = x_norm * tf.cast(scale_group, input_dtype) + tf.cast(bias_group, input_dtype)

    x_norm = tf.reshape(x_norm, x_shape)

    if opset is None or opset >= 21:
        rank_out = x_norm.shape.rank
        if rank_out is not None:
            scale_reshape = tf.reshape(scale, [1] * (rank_out - 1) + [-1])
            bias_reshape = tf.reshape(bias, [1] * (rank_out - 1) + [-1])
        else:
            rank_out = tf.rank(x_norm)
            prefix_ones = tf.fill([rank_out - 1], 1)
            scale_shape = tf.concat(
                [prefix_ones, tf.constant([-1], dtype=tf.int32)],
                axis=0,
            )
            scale_reshape = tf.reshape(scale, scale_shape)
            bias_reshape = tf.reshape(bias, scale_shape)
        x_norm = x_norm * tf.cast(scale_reshape, input_dtype) + tf.cast(bias_reshape, input_dtype)

    if internal_inverse_perm is not None:
        x_norm = transpose_with_flexing_deterrence(
            input_tensor=x_norm,
            perm=internal_inverse_perm,
            **kwargs,
        )

    tf_layers_dict[graph_node_output.name]['tf_node'] = x_norm

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
                'tf_op_type': 'GroupNormalization',
                'tf_inputs': {
                    'x': input_tensor,
                    'scale': scale,
                    'bias': bias,
                    'num_groups': num_groups,
                    'epsilon': epsilon,
                    'stash_type': stash_type,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
