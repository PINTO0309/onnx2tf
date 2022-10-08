import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from tensorflow.keras.layers import Lambda # type: ignore
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Resize

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

    input_tensor = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    roi = None
    if len(graph_node.inputs) >= 2:
        roi = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
    scales = None
    if len(graph_node.inputs) >= 3:
        scales = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )
    sizes = None
    if len(graph_node.inputs) >= 4:
        sizes = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
        )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[input_tensor.name]['tf_node'] \
        if isinstance(input_tensor, gs.Variable) else input_tensor
    input_tensor_shape = input_tensor.shape
    roi = tf_layers_dict[roi.name]['tf_node'] \
        if isinstance(roi, gs.Variable) else roi
    scales = tf_layers_dict[scales.name]['tf_node'] \
        if isinstance(scales, gs.Variable) else scales
    sizes = tf_layers_dict[sizes.name]['tf_node'] \
        if isinstance(sizes, gs.Variable) else sizes

    coordinate_transformation_mode = graph_node.attrs.get('coordinate_transformation_mode', 'half_pixel')
    extrapolation_value = graph_node.attrs.get('extrapolation_value', 0.0)
    mode = graph_node.attrs.get('mode', 'nearest')

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }



    def upsampling2d_bilinear(input_tensor, new_size, align_corners, half_pixel_centers, name):
        return tf.compat.v1.image.resize_bilinear(
            images=input_tensor,
            size=new_size,
            align_corners=align_corners,
            half_pixel_centers=half_pixel_centers,
            name=name,
        )

    def upsampling2d_bicubic(input_tensor, new_size, align_corners, half_pixel_centers, name):
        return tf.compat.v1.image.resize_bicubic(
            images=input_tensor,
            size=new_size,
            align_corners=align_corners,
            half_pixel_centers=half_pixel_centers,
            name=name,
        )

    def upsampling2d_nearest(input_tensor, new_size, align_corners, half_pixel_centers, name):
        return tf.compat.v1.image.resize_nearest_neighbor(
            images=input_tensor,
            size=new_size,
            align_corners=align_corners,
            half_pixel_centers=half_pixel_centers,
            name=name,
        )


    # Generation of TF OP
    if mode.lower() == "linear":
        mode = tf.image.ResizeMethod.BILINEAR
        tf_resize = upsampling2d_bilinear
    elif mode.lower() == "cubic":
        mode = tf.image.ResizeMethod.BICUBIC
        tf_resize = upsampling2d_bicubic
    else:
        mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        tf_resize = upsampling2d_nearest

    if sizes is not None:
        # sizes is defined
        # The number of elements of 'sizes' should be the same as the rank of input 'X'
        sizes = sizes.set_shape(input_tensor_shape.shape) if isinstance(sizes, gs.Variable) else sizes
        new_size = tf.cast(sizes[1:3], tf.int32)
    elif scales is not None:
        # only scales is defined
        h_w_scale = scales[1:3]
        h_w_shape = input_tensor_shape[1:3]
        new_size = tf.cast(h_w_scale * tf.cast(h_w_shape, scales.dtype), tf.int32)

    # Tensorflow require the shape of "size" in the "tf.image.resize" must be known at
    # graph creation time. However in the dynamic shape situation, the shape of "new_size"
    # will be "None", the actual shape can only be determine at runtime. But we know
    # "new_size" should always contain [h, w], therefore the shape must be 2.
    new_size.set_shape([2])

    if hasattr(new_size, '_inferred_value'):
        new_size_values = new_size._inferred_value
        if new_size_values.count(None) == len(new_size_values):
            tensor_rank = len(graph_node_output.shape)
            convertion_table = [0] + [i for i in range(2, tensor_rank)] + [1]
            new_values = [0] * tensor_rank
            for new_idx, idx in enumerate(convertion_table):
                new_values[new_idx] = graph_node_output.shape[idx]
            new_size = new_values[-3:-1]

    # TODO: upsampling2d_bilinear_5d
    # TODO: upsampling2d_nearest_5d
    resized_tensor = None
    if coordinate_transformation_mode == "tf_crop_and_resize":
        # get boxes for crop
        indices = [1,2,5,6]
        boxes = tf.expand_dims(tf.gather(roi, indices, axis=0), 0)
        # get box_indices for crop
        box_indices = tf.cast(tf.range(0, input_tensor_shape[0]), dtype=tf.int32)
        # run crop and resize
        resized_tensor = tf.image.crop_and_resize(
            images=input_tensor,
            boxes=boxes,
            box_indices=box_indices,
            crop_size=new_size,
            method=mode,
            extrapolation_value=extrapolation_value,
            name=graph_node.name,
        )
    elif coordinate_transformation_mode == "align_corners":
        resized_tensor = Lambda(
            tf_resize,
            arguments={
                'new_size': new_size,
                'align_corners': True,
                'half_pixel_centers': False,
                'name': graph_node.name,
            }
        )(input_tensor)
    elif coordinate_transformation_mode == "asymmetric":
        resized_tensor = Lambda(
            tf_resize,
            arguments={
                'new_size': new_size,
                'align_corners': False,
                'half_pixel_centers': False,
                'name': graph_node.name,
            }
        )(input_tensor)
    else:
        resized_tensor = tf.image.resize(
            images=input_tensor,
            size=new_size,
            method=mode,
            name=graph_node.name,
        )

    tf_layers_dict[graph_node_output.name]['tf_node'] = resized_tensor
