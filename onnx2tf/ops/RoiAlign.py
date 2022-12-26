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
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
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
    """RoiAlign

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

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    rois = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    batch_indices = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    coordinate_transformation_mode = graph_node.attrs.get('coordinate_transformation_mode', 'half_pixel') # half_pixel or output_half_pixel
    mode = graph_node.attrs.get('mode', 'avg') # avg or max
    output_height = graph_node.attrs.get('output_height', 1)
    output_width = graph_node.attrs.get('output_width', 1)
    sampling_ratio = graph_node.attrs.get('sampling_ratio', 0)
    spatial_scale = graph_node.attrs.get('spatial_scale', 1.0)
    adaptive_ratio = False
    if sampling_ratio <= 0:
        sampling_ratio = int((output_height + output_width) / 2)
        adaptive_ratio = True

    rois = rois * spatial_scale

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    def crop_and_resize(
        *,
        image,
        boxes,
        box_ind,
        crop_size,
        sampling_ratio,
        adaptive_ratio=False,
        pad_border=False,
    ):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        # TF's crop_and_resize produces zeros on border
        if pad_border:
            image = tf.pad(
                tensor=image,
                paddings=[
                    [0, 0],
                    [1, 1],
                    [1, 1],
                    [0, 0],
                ],
                mode='SYMMETRIC',
            )
            boxes = boxes + 1

        def transform_fpcoor_for_tf(
            *,
            boxes,
            image_shape,
            crop_size,
            sampling_ratio,
            adaptive_ratio,
        ):
            x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)
            if not adaptive_ratio:
                crop_shape = (
                    crop_size[0] * sampling_ratio,
                    crop_size[1] * sampling_ratio,
                )
                spacing_w = (x1 - x0) / tf.cast(crop_shape[1], dtype=tf.float32)
                spacing_h = (y1 - y0) / tf.cast(crop_shape[0], dtype=tf.float32)
                nx0 = (x0 + spacing_w / 2) / tf.cast(image_shape[1] - 1, dtype=tf.float32)
                ny0 = (y0 + spacing_h / 2) / tf.cast(image_shape[0] - 1, dtype=tf.float32)

                nw = spacing_w * tf.cast(
                    crop_shape[1] - 1,
                    dtype=tf.float32,
                ) / tf.cast(
                    image_shape[1] - 1,
                    dtype=tf.float32,
                )
                nh = spacing_h * tf.cast(
                    crop_shape[0] - 1,
                    dtype=tf.float32,
                ) / tf.cast(
                    image_shape[0] - 1,
                    dtype=tf.float32,
                )
            else:
                roi_width = x1 - x0
                roi_height = y1 - y0
                nx0 = x0 / tf.cast(image_shape[1] - 1, dtype=tf.float32)
                ny0 = y0 / tf.cast(image_shape[0] - 1, dtype=tf.float32)
                nw = (roi_width - 1) / tf.cast(image_shape[1] - 1, dtype=tf.float32)
                nh = (roi_height - 1) / tf.cast(image_shape[0] - 1, dtype=tf.float32)
            return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

        image_shape = tf.shape(image)[1:3]
        boxes = transform_fpcoor_for_tf(
            boxes=boxes,
            image_shape=image_shape,
            crop_size=crop_size,
            sampling_ratio=sampling_ratio,
            adaptive_ratio=adaptive_ratio,
        )
        ret = tf.image.crop_and_resize(
            image,
            boxes,
            tf.cast(box_ind, dtype=tf.int32),
            crop_size=(
                crop_size[0] * sampling_ratio,
                crop_size[1] * sampling_ratio,
            ),
        )
        return ret

    croped_tensor = crop_and_resize(
        image=input_tensor,
        boxes=rois,
        box_ind=tf.cast(batch_indices, tf.int32),
        crop_size=(output_height, output_width),
        sampling_ratio=sampling_ratio,
        adaptive_ratio=adaptive_ratio,
    )

    pooled_tensor = None
    if mode.lower() == 'avg':
        pooled_tensor = tf.nn.avg_pool(
            input=croped_tensor,
            ksize=[1, sampling_ratio, sampling_ratio, 1],
            strides=[1, sampling_ratio, sampling_ratio, 1],
            padding='SAME',
            name=graph_node.name,
        )
    elif mode.lower() == 'max':
        pooled_tensor = tf.nn.max_pool(
            input=croped_tensor,
            ksize=[1, sampling_ratio, sampling_ratio, 1],
            strides=[1, sampling_ratio, sampling_ratio, 1],
            padding='SAME',
            name=graph_node.name,
        )

    tf_layers_dict[graph_node_output.name]['tf_node'] = pooled_tensor

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
                'tf_op_type': 'RoiAlign',
                'tf_inputs': {
                    'input': input_tensor,
                    'rois': rois,
                    'batch_indices': batch_indices,
                    'coordinate_transformation_mode': coordinate_transformation_mode,
                    'mode': mode,
                    'output_height': output_height,
                    'output_width': output_width,
                    'sampling_ratio': sampling_ratio,
                    'spatial_scale': spatial_scale,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
