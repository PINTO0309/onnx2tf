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
)
from onnx2tf.utils.colors import Color


@print_node_info
@inverted_operation_enable_disable
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

    max_output_boxes_per_class = tf.cast(
        max_output_boxes_per_class,
        tf.int32,
    ) if (max_output_boxes_per_class is not None and max_output_boxes_per_class != "") else tf.constant(0, tf.int32)

    max_output_boxes_per_class = tf.squeeze(max_output_boxes_per_class) \
        if len(max_output_boxes_per_class.shape) == 1 else max_output_boxes_per_class

    iou_threshold = iou_threshold \
        if (iou_threshold is not None and iou_threshold != "") else tf.constant(0, tf.float32)
    iou_threshold = tf.squeeze(iou_threshold) \
        if len(iou_threshold.shape) == 1 else iou_threshold

    score_threshold = score_threshold \
        if (score_threshold is not None and score_threshold != "") else tf.constant(float('-inf'))
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
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'It is not possible to specify a dynamic shape '+
            f'for the batch size of the input tensor in NonMaxSuppression. '+
            f'Use the --batch_size option to change the batch size to a fixed size. \n'+
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
            selected_indices = tf.image.non_max_suppression(
                tf_boxes,
                tf_scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
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
