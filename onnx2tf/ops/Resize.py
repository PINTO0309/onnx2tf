import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from tensorflow.keras.layers import Lambda # type: ignore
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    upsampling2d_bilinear,
    upsampling2d_bicubic,
    upsampling2d_nearest,
    upsampling3d_bilinear,
    upsampling3d_bicubic,
    upsampling3d_nearest,
)
from onnx2tf.utils.colors import Color


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
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
    input_tensor_rank = len(input_tensor_shape)
    roi = tf_layers_dict[roi.name]['tf_node'] \
        if isinstance(roi, gs.Variable) else roi
    scales = tf_layers_dict[scales.name]['tf_node'] \
        if isinstance(scales, gs.Variable) else scales
    sizes = tf_layers_dict[sizes.name]['tf_node'] \
        if isinstance(sizes, gs.Variable) else sizes

    coordinate_transformation_mode = graph_node.attrs.get('coordinate_transformation_mode', 'half_pixel')
    extrapolation_value = graph_node.attrs.get('extrapolation_value', 0.0)
    mode = graph_node.attrs.get('mode', 'nearest')

    replace_argmax_to_fused_argmax_and_indicies_is_int64 = \
        kwargs['replace_argmax_to_fused_argmax_and_indicies_is_int64']
    replace_argmax_to_fused_argmax_and_indicies_is_float32 = \
        kwargs['replace_argmax_to_fused_argmax_and_indicies_is_float32']
    fused_argmax_scale_ratio = \
        kwargs['fused_argmax_scale_ratio']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    if mode.lower() == "linear":
        mode = tf.image.ResizeMethod.BILINEAR
        if input_tensor_rank == 4:
            tf_resize = upsampling2d_bilinear
        elif input_tensor_rank == 5:
            tf_resize = upsampling3d_bilinear
        else:
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'Currently, Resize operations other than 4D and 5D are not supported. '+
                'Pull requests are welcome. \n'+
                f'graph_node.name: {graph_node.name} shape: {input_tensor_shape}'
            )
            sys.exit(1)
    elif mode.lower() == "cubic":
        mode = tf.image.ResizeMethod.BICUBIC
        if input_tensor_rank == 4:
            tf_resize = upsampling2d_bicubic
        elif input_tensor_rank == 5:
            tf_resize = upsampling3d_bicubic
        else:
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'Currently, Resize operations other than 4D and 5D are not supported. '+
                'Pull requests are welcome. \n'+
                f'graph_node.name: {graph_node.name} shape: {input_tensor_shape}'
            )
            sys.exit(1)
    else:
        mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        if input_tensor_rank == 4:
            tf_resize = upsampling2d_nearest
        elif input_tensor_rank == 5:
            tf_resize = upsampling3d_nearest
        else:
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'Currently, Resize operations other than 4D and 5D are not supported. '+
                'Pull requests are welcome. \n'+
                f'graph_node.name: {graph_node.name} shape: {input_tensor_shape}'
            )
            sys.exit(1)

    if sizes is not None:
        # sizes is defined
        # The number of elements of 'sizes' should be the same as the rank of input 'X'
        if isinstance(sizes, gs.Variable):
            sizes = sizes.set_shape(input_tensor_shape.shape)
            new_size = tf.cast(sizes[1:3], tf.int32)
        elif isinstance(sizes, np.ndarray):
            new_size = tf.cast(sizes[1:3], tf.int32)
        elif tf.keras.backend.is_keras_tensor(sizes):
            new_size = tf.cast(tf.slice(sizes, [1], [2]), tf.int32)
    elif scales is not None:
        # only scales is defined
        if hasattr(graph_node.outputs[0], 'shape') \
            and graph_node.outputs[0].shape is not None \
            and isinstance(graph_node.outputs[0].shape[-2], int) \
            and isinstance(graph_node.outputs[0].shape[-1], int):
            new_size = graph_node.outputs[0].shape[-2:len(graph_node.outputs[0].shape)] # Estimated from ONNX output shape
        else:
            h_w_scale = scales[1:3]
            h_w_shape = input_tensor_shape[1:3]
            new_size = tf.cast(h_w_scale * tf.cast(h_w_shape, scales.dtype), tf.int32)

    # Tensorflow require the shape of "size" in the "tf.image.resize" must be known at
    # graph creation time. However in the dynamic shape situation, the shape of "new_size"
    # will be "None", the actual shape can only be determine at runtime. But we know
    # "new_size" should always contain [h, w], therefore the shape must be 2.
    if hasattr(new_size, 'set_shape'):
        new_size.set_shape([2])


    if (replace_argmax_to_fused_argmax_and_indicies_is_int64 \
        or replace_argmax_to_fused_argmax_and_indicies_is_float32) \
        and graph_node.o().op == 'ArgMax' \
        and input_tensor_rank == 4:
        new_size = tf.cast(
            tf.cast(new_size, dtype=tf.float32) * fused_argmax_scale_ratio,
            dtype=tf.int32,
        )

    if hasattr(new_size, '_inferred_value'):
        new_size_values = new_size._inferred_value
        if (new_size_values is None or new_size_values.count(None) == len(new_size_values)) \
            and sum([1 if isinstance(s, str) else 0 for s in graph_node_output.shape[1:3]]) == 0:
            tensor_rank = len(graph_node_output.shape)
            convertion_table = [0] + [i for i in range(2, tensor_rank)] + [1]
            new_values = [0] * tensor_rank
            for new_idx, idx in enumerate(convertion_table):
                new_values[new_idx] = graph_node_output.shape[idx]
            new_size = new_values[-3:-1]

    # Param replacement
    input_tensor = replace_parameter(
        value_before_replacement=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    if len(graph_node.inputs) >= 2:
        roi = replace_parameter(
            value_before_replacement=roi,
            param_target='inputs',
            param_name=graph_node.inputs[1].name,
            **kwargs,
        )
    if len(graph_node.inputs) >= 3:
        scales = replace_parameter(
            value_before_replacement=scales,
            param_target='inputs',
            param_name=graph_node.inputs[2].name,
            **kwargs,
        )
    if len(graph_node.inputs) >= 4:
        new_size = replace_parameter(
            value_before_replacement=new_size,
            param_target='inputs',
            param_name=graph_node.inputs[3].name,
            **kwargs,
        )

    coordinate_transformation_mode = replace_parameter(
        value_before_replacement=coordinate_transformation_mode,
        param_target='attributes',
        param_name='coordinate_transformation_mode',
        **kwargs,
    )
    extrapolation_value = replace_parameter(
        value_before_replacement=extrapolation_value,
        param_target='attributes',
        param_name='extrapolation_value',
        **kwargs,
    )
    mode = replace_parameter(
        value_before_replacement=mode,
        param_target='attributes',
        param_name='mode',
        **kwargs,
    )

    resized_tensor = None
    boxes = None
    box_indices = None
    tf_op_type = None
    align_corners = None
    half_pixel_centers = None
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
        tf_op_type = tf.image.crop_and_resize
    elif coordinate_transformation_mode == "align_corners":
        align_corners = True
        half_pixel_centers = False
        resized_tensor = Lambda(
            tf_resize,
            arguments={
                'new_size': new_size,
                'align_corners': align_corners,
                'half_pixel_centers': half_pixel_centers,
                'name': graph_node.name,
            }
        )(input_tensor)
        tf_op_type = tf_resize
    elif coordinate_transformation_mode == "asymmetric":
        align_corners = False
        half_pixel_centers = False
        resized_tensor = Lambda(
            tf_resize,
            arguments={
                'new_size': new_size,
                'align_corners': align_corners,
                'half_pixel_centers': half_pixel_centers,
                'name': graph_node.name,
            }
        )(input_tensor)
        tf_op_type = tf_resize
    else:
        resized_tensor = tf.image.resize(
            images=input_tensor,
            size=new_size,
            method=mode,
            name=graph_node.name,
        )
        tf_op_type = tf.image.resize

    tf_layers_dict[graph_node_output.name]['tf_node'] = resized_tensor

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'images': input_tensor,
                    'boxes': boxes,
                    'box_indices': box_indices,
                    'new_size/crop_size': new_size,
                    'method': mode,
                    'extrapolation_value': extrapolation_value,
                    'align_corners': align_corners,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
