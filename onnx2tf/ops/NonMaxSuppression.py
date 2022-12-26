import sys
import random

random.seed(0)
import numpy as np

np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)
from onnx2tf.utils.colors import Color

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.util import dispatch


@dispatch.add_dispatch_support
def non_max_suppression(
    boxes,
    scores,
    max_output_size,
    iou_threshold=0.5,
    score_threshold=float('-inf'),
    name=None,
):
    with ops.name_scope(name, 'non_max_suppression'):
        iou_threshold = ops.convert_to_tensor(iou_threshold, name='iou_threshold')
        score_threshold = ops.convert_to_tensor(
            score_threshold, name='score_threshold')
        selected_indices, num_valid = gen_image_ops.non_max_suppression_v4(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pad_to_max_output_size=False,
        )
        return selected_indices[:num_valid]


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """NonMaxSuppression

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
    boxes = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    scores = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    # Pre-process transpose
    boxes = pre_process_transpose(
        value_before_transpose=boxes,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    scores = pre_process_transpose(
        value_before_transpose=scores,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )

    graph_node_input_3 = None
    if len(graph_node.inputs) >= 3:
        graph_node_input_3 = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )
    graph_node_input_4 = None
    if len(graph_node.inputs) >= 4:
        graph_node_input_4 = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
        )
    graph_node_input_5 = None
    if len(graph_node.inputs) >= 5:
        graph_node_input_5 = get_constant_or_variable(
            graph_node.inputs[4],
            before_op_output_shape_trans,
        )
    max_output_boxes_per_class = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    iou_threshold = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4
    score_threshold = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) else graph_node_input_5

    try:
        max_output_boxes_per_class = tf.cast(
            max_output_boxes_per_class,
            tf.int32,
        )
    except:
        max_output_boxes_per_class = tf.constant(0, tf.int32)

    max_output_boxes_per_class = tf.where(
        condition=max_output_boxes_per_class <= 0,
        x=tf.shape(boxes)[1],
        y=max_output_boxes_per_class,
    )

    max_output_boxes_per_class = tf.squeeze(max_output_boxes_per_class) \
        if len(max_output_boxes_per_class.shape) == 1 else max_output_boxes_per_class

    iou_threshold = iou_threshold \
        if (iou_threshold is not None and iou_threshold != "") else tf.constant(0, tf.float32)
    iou_threshold = tf.squeeze(iou_threshold) \
        if len(iou_threshold.shape) == 1 else iou_threshold

    score_threshold = score_threshold \
        if (score_threshold is not None and score_threshold != "") else tf.constant(-np.inf)
    score_threshold = tf.squeeze(score_threshold) \
        if len(score_threshold.shape) == 1 else score_threshold

    center_point_box = graph_node.attrs.get('center_point_box', 0)

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Param replacement
    boxes = replace_parameter(
        value_before_replacement=boxes,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    scores = replace_parameter(
        value_before_replacement=scores,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    if len(graph_node.inputs) >= 3:
        max_output_boxes_per_class = replace_parameter(
            value_before_replacement=max_output_boxes_per_class,
            param_target='inputs',
            param_name=graph_node.inputs[2].name,
            **kwargs,
        )
    if len(graph_node.inputs) >= 4:
        iou_threshold = replace_parameter(
            value_before_replacement=iou_threshold,
            param_target='inputs',
            param_name=graph_node.inputs[3].name,
            **kwargs,
        )
    if len(graph_node.inputs) >= 5:
        score_threshold = replace_parameter(
            value_before_replacement=score_threshold,
            param_target='inputs',
            param_name=graph_node.inputs[4].name,
            **kwargs,
        )
    center_point_box = replace_parameter(
        value_before_replacement=center_point_box,
        param_target='attributes',
        param_name='center_point_box',
        **kwargs,
    )

    # Generation of TF OP
    if center_point_box == 1:
        boxes_t = tf.transpose(boxes, perm=[0, 2, 1])
        x_centers = tf.slice(boxes_t, [0, 0, 0], [-1, 1, -1])
        y_centers = tf.slice(boxes_t, [0, 1, 0], [-1, 1, -1])
        widths = tf.slice(boxes_t, [0, 2, 0], [-1, 1, -1])
        heights = tf.slice(boxes_t, [0, 3, 0], [-1, 1, -1])
        y1 = tf.subtract(y_centers, tf.divide(heights, 2))
        x1 = tf.subtract(x_centers, tf.divide(widths, 2))
        y2 = tf.add(y_centers, tf.divide(heights, 2))
        x2 = tf.add(x_centers, tf.divide(widths, 2))
        boxes_t = tf.concat([y1, x1, y2, x2], 1)
        boxes = tf.transpose(boxes_t, perm=[0, 2, 1])

    num_batches = boxes.shape[0]

    if num_batches is None:
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'It is not possible to specify a dynamic shape ' +
            f'for the batch size of the input tensor in NonMaxSuppression. ' +
            f'Use the --batch_size option to change the batch size to a fixed size. \n' +
            f'graph_node.name: {graph_node.name} boxes.shape: {boxes.shape} scores.shape: {scores.shape}'
        )
        sys.exit(1)

    for batch_i in tf.range(num_batches):
        # get boxes in batch_i only
        tf_boxes = tf.squeeze(tf.gather(boxes, [batch_i]), axis=0)
        # get scores of all classes in batch_i only
        batch_i_scores = tf.squeeze(tf.gather(scores, [batch_i]), axis=0)
        # get number of classess in batch_i only
        num_classes = batch_i_scores.shape[0]
        for class_j in tf.range(num_classes):
            # get scores in class_j for batch_i only
            tf_scores = tf.squeeze(tf.gather(batch_i_scores, [class_j]), axis=0)
            # get the selected boxes indices

            selected_indices = non_max_suppression(
                boxes=tf_boxes,
                scores=tf_scores,
                max_output_size=max_output_boxes_per_class,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )

            # add batch and class information into the indices
            output = tf.transpose([tf.cast(selected_indices, dtype=tf.int64)])
            paddings = tf.constant([[0, 0], [1, 0]])
            output = tf.pad(
                output,
                paddings,
                constant_values=tf.cast(class_j, dtype=tf.int64),
            )
            output = tf.pad(
                output,
                paddings,
                constant_values=tf.cast(batch_i, dtype=tf.int64),
            )
            # tf.function will auto convert "result" from variable to placeholder
            # therefore don't need to use assign here
            result = output if tf.equal(batch_i, 0) and tf.equal(class_j, 0) else tf.concat([result, output], 0)

    tf_layers_dict[graph_node_output.name]['tf_node'] = result

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
                'tf_op_type': tf.image.non_max_suppression,
                'tf_inputs': {
                    'boxes': tf_boxes,
                    'scores': tf_scores,
                    'max_output_boxes_per_class': max_output_boxes_per_class,
                    'iou_threshold': iou_threshold,
                    'score_threshold': score_threshold,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
