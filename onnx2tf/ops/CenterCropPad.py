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
    convert_axis,
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
    """CenterCropPad

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
    graph_node_shape = get_constant_or_variable(
        graph_node.inputs[1],
        False,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    target_shape = tf_layers_dict[graph_node_shape.name]['tf_node'] \
        if isinstance(graph_node_shape, gs.Variable) else graph_node_shape

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input.name]['nhwc'] \
            if isinstance(graph_node_input, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input.name].keys() else False
    }

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    input_rank = input_tensor.shape.rank
    if input_rank is None:
        input_rank = tf.rank(input_tensor)

    axes = graph_node.attrs.get('axes', None)
    if isinstance(axes, np.ndarray):
        axes = axes.tolist()

    if axes is None:
        if isinstance(input_rank, int):
            axes_list = list(range(input_rank))
            if before_op_output_shape_trans:
                axes_list = [
                    convert_axis(
                        axis=axis,
                        tensor_rank=input_rank,
                        before_op_output_shape_trans=before_op_output_shape_trans,
                    ) for axis in axes_list
                ]
            axes_tensor = tf.constant(axes_list, dtype=tf.int32)
        else:
            rank_t = tf.cast(input_rank, tf.int32)
            axes_tensor = tf.range(rank_t)
            if before_op_output_shape_trans:
                axes_tensor = tf.where(
                    tf.equal(axes_tensor, 0),
                    0,
                    tf.where(tf.equal(axes_tensor, 1), rank_t - 1, axes_tensor - 1),
                )
    else:
        if not isinstance(axes, list):
            axes = [axes]
        if isinstance(input_rank, int):
            axes_conv = [
                convert_axis(
                    axis=axis,
                    tensor_rank=input_rank,
                    before_op_output_shape_trans=before_op_output_shape_trans,
                ) for axis in axes
            ]
            axes_tensor = tf.constant(axes_conv, dtype=tf.int32)
        else:
            axes_tensor = tf.convert_to_tensor(axes, dtype=tf.int32)
            if before_op_output_shape_trans:
                rank_t = tf.cast(input_rank, tf.int32)
                axes_tensor = tf.where(axes_tensor < 0, axes_tensor + rank_t, axes_tensor)
                axes_tensor = tf.where(
                    tf.equal(axes_tensor, 0),
                    0,
                    tf.where(tf.equal(axes_tensor, 1), rank_t - 1, axes_tensor - 1),
                )

    if isinstance(target_shape, list):
        target_shape = tf.constant(np.asarray(target_shape, dtype=np.int32))
    elif isinstance(target_shape, np.ndarray):
        target_shape = tf.convert_to_tensor(target_shape.astype(np.int32))
    else:
        target_shape = tf.cast(target_shape, tf.int32)

    input_shape = tf.shape(input_tensor, out_type=tf.int32)
    target_shape_full = tf.tensor_scatter_nd_update(
        input_shape,
        tf.expand_dims(axes_tensor, axis=1),
        target_shape,
    )

    diff = target_shape_full - input_shape

    pad_before = tf.where(diff > 0, tf.math.floordiv(diff, 2), 0)
    pad_after = tf.where(diff > 0, diff - tf.math.floordiv(diff, 2), 0)
    crop_before = tf.where(diff < 0, tf.math.floordiv(-diff, 2), 0)
    crop_after = tf.where(diff < 0, (-diff) - tf.math.floordiv(-diff, 2), 0)

    begin = crop_before
    size = input_shape - crop_before - crop_after
    cropped = tf.slice(input_tensor, begin, size)

    paddings = tf.stack([pad_before, pad_after], axis=1)
    if input_tensor.dtype == tf.string:
        pad_value = tf.constant('', dtype=tf.string)
    else:
        pad_value = tf.cast(0, input_tensor.dtype)

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.pad(
            tensor=cropped,
            paddings=paddings,
            constant_values=pad_value,
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
                'tf_op_type': 'CenterCropPad',
                'tf_inputs': {
                    'input': input_tensor,
                    'shape': target_shape,
                    'axes': axes,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
