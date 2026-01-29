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
    """TensorScatter

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
    if len(graph_node.inputs) >= 3:
        before_op_output_shape_trans_3 = \
            tf_layers_dict.get(graph_node.inputs[2].name, {}).get('before_op_output_shape_trans', True)
        before_op_output_shape_trans = \
            before_op_output_shape_trans \
            and before_op_output_shape_trans_3

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_input_3 = None
    if len(graph_node.inputs) >= 3:
        graph_node_input_3 = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans=False,
        )

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    past_cache = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    update = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    write_indices = None
    if graph_node_input_3 is not None:
        write_indices = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
            if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

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
    past_cache = pre_process_transpose(
        value_before_transpose=past_cache,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    update = pre_process_transpose(
        value_before_transpose=update,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    if write_indices is not None:
        write_indices = pre_process_transpose(
            value_before_transpose=write_indices,
            param_target='inputs',
            param_name=graph_node.inputs[2].name,
            **kwargs,
        )

    # Generation of TF OP
    past_cache = _as_tensor(past_cache)
    update = _as_tensor(update)
    if write_indices is not None:
        write_indices = _as_tensor(write_indices)

    cache_rank = past_cache.shape.rank
    if cache_rank is None and graph_node.inputs[0].shape is not None:
        cache_rank = len(graph_node.inputs[0].shape)
    if cache_rank is None:
        error(
            f'TensorScatter requires known input rank.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)
    axis = graph_node.attrs.get('axis', -2)
    axis = convert_axis(
        axis=axis,
        tensor_rank=cache_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )
    mode = graph_node.attrs.get('mode', 'linear')
    if mode not in ['linear', 'circular']:
        error(
            f'TensorScatter supports mode=linear or mode=circular only. mode={mode}\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    past_shape = tf.shape(past_cache)
    update_shape = tf.shape(update)

    if write_indices is None:
        write_indices = tf.zeros(
            [past_shape[0]],
            dtype=tf.int64,
        )
    else:
        write_indices = tf.cast(write_indices, tf.int64)

    max_sequence_length = past_shape[axis]
    sequence_length = update_shape[axis]

    idx_tensors_per_axis = [
        tf.range(update_shape[i]) for i in range(cache_rank)
    ]
    idx_tensors_per_axis = tf.meshgrid(*idx_tensors_per_axis, indexing='ij')

    axis_idx = idx_tensors_per_axis[axis]
    batch_idx = idx_tensors_per_axis[0]
    write_offsets = tf.gather(write_indices, batch_idx)
    axis_idx = axis_idx + tf.cast(write_offsets, axis_idx.dtype)
    if mode == 'circular':
        axis_idx = tf.math.floormod(
            axis_idx,
            tf.cast(max_sequence_length, axis_idx.dtype),
        )
    idx_tensors_per_axis[axis] = axis_idx

    coordinate = tf.stack(idx_tensors_per_axis, axis=-1)
    indices = tf.reshape(coordinate, [-1, cache_rank])
    indices = tf.cast(indices, tf.int64)
    updates = tf.reshape(update, [-1])

    output = tf.tensor_scatter_nd_update(
        tensor=past_cache,
        indices=indices,
        updates=updates,
        name=graph_node.name,
    )
    output_dtype = NUMPY_DTYPES_TO_TF_DTYPES[past_cache.dtype] \
        if isinstance(past_cache.dtype, np.dtype) else past_cache.dtype
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
                    'tensor': past_cache,
                    'indices': indices,
                    'updates': update,
                    'axis': axis,
                    'mode': mode,
                    'write_indices': write_indices,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
