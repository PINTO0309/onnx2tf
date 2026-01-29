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


def _move_class_to_last(tensor, class_axis):
    rank = tensor.shape.rank
    if rank is None:
        rank = tf.rank(tensor)
    if isinstance(rank, int):
        if class_axis == rank - 1:
            return tensor, None
        perm = [i for i in range(rank) if i != class_axis] + [class_axis]
        return tf.transpose(tensor, perm=perm), perm
    return tensor, None


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """SoftmaxCrossEntropyLoss

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
            before_op_output_shape_trans,
        )

    graph_node_output_1: gs.Variable = graph_node.outputs[0]
    output_1_shape = graph_node_output_1.shape
    output_1_dtype = graph_node_output_1.dtype
    output_1_tf_dtype = NUMPY_DTYPES_TO_TF_DTYPES[output_1_dtype] \
        if isinstance(output_1_dtype, np.dtype) else output_1_dtype

    graph_node_output_2 = None
    if len(graph_node.outputs) >= 2:
        graph_node_output_2 = graph_node.outputs[1]
        output_2_shape = graph_node_output_2.shape
        output_2_dtype = graph_node_output_2.dtype
        output_2_tf_dtype = NUMPY_DTYPES_TO_TF_DTYPES[output_2_dtype] \
            if isinstance(output_2_dtype, np.dtype) else output_2_dtype
    else:
        output_2_tf_dtype = None

    scores_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    labels_tensor = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    weight_tensor = None
    if graph_node_input_3 is not None:
        weight_tensor = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
            if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    reduction = graph_node.attrs.get('reduction', 'mean')
    ignore_index = graph_node.attrs.get('ignore_index', None)

    input_rank = len(scores_tensor.shape)
    class_axis = convert_axis(
        axis=1,
        tensor_rank=input_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # Preserving Graph Structure (Dict)
    output_entry = {
        'optype': graph_node.op,
        'shape': output_1_shape,
        'dtype': output_1_dtype,
    }
    if reduction == 'none':
        output_entry['nhwc'] = tf_layers_dict[graph_node_input_1.name]['nhwc'] \
            if isinstance(graph_node_input_1, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False
    tf_layers_dict[graph_node_output_1.name] = output_entry
    if graph_node_output_2 is not None:
        tf_layers_dict[graph_node_output_2.name] = {
            'optype': graph_node.op,
            'shape': output_2_shape,
            'dtype': output_2_dtype,
            'nhwc': tf_layers_dict[graph_node_input_1.name]['nhwc'] \
                if isinstance(graph_node_input_1, gs.Variable) \
                    and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False
        }

    # Pre-process transpose
    scores_tensor = pre_process_transpose(
        value_before_transpose=scores_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    labels_tensor = pre_process_transpose(
        value_before_transpose=labels_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    if weight_tensor is not None:
        weight_tensor = pre_process_transpose(
            value_before_transpose=weight_tensor,
            param_target='inputs',
            param_name=graph_node.inputs[2].name,
            **kwargs,
        )

    # Generation of TF OP
    scores_tensor = _as_tensor(scores_tensor)
    labels_tensor = _as_tensor(labels_tensor)
    if weight_tensor is not None:
        weight_tensor = _as_tensor(weight_tensor)

    log_prob = tf.nn.log_softmax(
        logits=scores_tensor,
        axis=class_axis,
    )

    log_prob_for_loss, _ = _move_class_to_last(log_prob, class_axis)

    depth = log_prob_for_loss.shape[-1]
    if depth is None:
        depth = tf.shape(log_prob_for_loss)[-1]
    depth = tf.cast(depth, tf.int32)

    labels = tf.cast(labels_tensor, tf.int32)
    if ignore_index is not None:
        ignore_index_val = tf.cast(ignore_index, labels.dtype)
        mask = tf.equal(labels, ignore_index_val)
        labels_safe = tf.where(mask, tf.zeros_like(labels), labels)
    else:
        mask = None
        labels_safe = labels

    one_hot = tf.one_hot(
        indices=labels_safe,
        depth=depth,
        axis=-1,
        dtype=log_prob_for_loss.dtype,
    )
    selected = tf.reduce_sum(log_prob_for_loss * one_hot, axis=-1)
    loss = -selected

    weight_per_label = None
    if weight_tensor is not None:
        weight_per_label = tf.gather(weight_tensor, labels_safe)
        weight_per_label = tf.cast(weight_per_label, loss.dtype)
        if mask is not None:
            weight_per_label = tf.where(mask, tf.zeros_like(weight_per_label), weight_per_label)
        loss = loss * weight_per_label

    if mask is not None:
        loss = tf.where(mask, tf.zeros_like(loss), loss)

    if reduction == 'none':
        output_tensor = loss
    elif reduction == 'sum':
        output_tensor = tf.reduce_sum(loss)
    elif reduction == 'mean':
        if weight_per_label is None:
            output_tensor = tf.reduce_mean(loss)
        else:
            denom = tf.reduce_sum(weight_per_label)
            output_tensor = tf.math.divide_no_nan(tf.reduce_sum(loss), denom)
    else:
        error(
            f'SoftmaxCrossEntropyLoss reduction={reduction} is not supported.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    if output_1_tf_dtype is not None and output_tensor.dtype != output_1_tf_dtype:
        output_tensor = tf.cast(output_tensor, output_1_tf_dtype)

    tf_layers_dict[graph_node_output_1.name]['tf_node'] = output_tensor

    if graph_node_output_2 is not None:
        log_prob_out = log_prob
        if output_2_tf_dtype is not None and log_prob_out.dtype != output_2_tf_dtype:
            log_prob_out = tf.cast(log_prob_out, output_2_tf_dtype)
        tf_layers_dict[graph_node_output_2.name]['tf_node'] = log_prob_out

    # Post-process transpose
    tf_layers_dict[graph_node_output_1.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output_1.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )
    if graph_node_output_2 is not None:
        tf_layers_dict[graph_node_output_2.name]['tf_node'] = post_process_transpose(
            value_before_transpose=tf_layers_dict[graph_node_output_2.name]['tf_node'],
            param_target='outputs',
            param_name=graph_node.outputs[1].name,
            **kwargs,
        )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output_1.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'SoftmaxCrossEntropyLoss',
                'tf_inputs': {
                    'scores': scores_tensor,
                    'labels': labels_tensor,
                    'weights': weight_tensor,
                    'reduction': reduction,
                    'ignore_index': ignore_index,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output_1.name]['tf_node'],
                    'log_prob': tf_layers_dict[graph_node_output_2.name]['tf_node'] \
                        if graph_node_output_2 is not None else None,
                },
            }
        )
    if graph_node_output_2 is not None:
        tf_layers_dict[graph_node_output_2.name]['tf_node_info'] = \
            make_tf_node_info(
                node_info={
                    'tf_op_type': tf.nn.log_softmax,
                    'tf_inputs': {
                        'logits': scores_tensor,
                        'axis': class_axis,
                    },
                    'tf_outputs': {
                        'output': tf_layers_dict[graph_node_output_2.name]['tf_node'],
                    },
                }
            )
