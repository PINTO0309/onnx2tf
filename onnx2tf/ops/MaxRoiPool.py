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
    transpose_with_flexing_deterrence,
)
from onnx2tf.utils.logging import *

INF_INDEX_VALUE: int = 4294967296


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """MaxRoiPool

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
        before_op_output_shape_trans_1 and before_op_output_shape_trans_2

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

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    rois = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Workaround to avoid as many conversion failures as possible
    # for models with useless Transpose immediately before them.
    # If the input geometry of the ONNX and the input geometry of the TF model match,
    # the input geometry on the TF model side is forcibly transposed to the NHWC format.
    # However, if all dimensions of CHW have the same value,
    # the forced transposition process is skipped because it may destroy the structure of the model.
    onnx_input_shape = [
        dim if isinstance(dim, int) else None for dim in graph_node.inputs[0].shape
    ] if graph_node.inputs[0].shape is not None else None
    tf_input_shape = [
        dim if isinstance(dim, int) else None for dim in input_tensor.shape
    ]
    if onnx_input_shape is not None \
        and len(onnx_input_shape) > 1 and len(tf_input_shape) > 1 \
        and onnx_input_shape == tf_input_shape:

        shape_for_judging_skip = [
            dim if dim is not None else INF_INDEX_VALUE for dim in onnx_input_shape[1:]
        ]
        if shape_for_judging_skip.count(shape_for_judging_skip[0]) != len(shape_for_judging_skip):
            if len(onnx_input_shape) == 4:
                # 2D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,1],
                    **kwargs,
                )

    pooled_shape = graph_node.attrs.get('pooled_shape', None)
    if pooled_shape is None or len(pooled_shape) != 2:
        error_msg = \
            Color.RED(f'ERROR:') + ' ' + \
            f'pooled_shape is required for MaxRoiPool. ' \
            f'graph_node.name: {graph_node.name}, pooled_shape: {pooled_shape}'
        print(error_msg)
        raise ValueError(error_msg)

    pooled_h = int(pooled_shape[0])
    pooled_w = int(pooled_shape[1])
    spatial_scale = float(graph_node.attrs.get('spatial_scale', 1.0))

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # Generation of TF OP
    rois = tf.cast(rois, tf.float32)
    if rois.shape.rank == 1:
        rois = tf.expand_dims(rois, axis=0)

    channels_static = input_tensor.shape[-1]
    channel_spec = tf.TensorSpec(
        shape=(channels_static,) if channels_static is not None else (None,),
        dtype=input_tensor.dtype,
    )
    row_spec = tf.TensorSpec(
        shape=(pooled_w, channels_static) if channels_static is not None else (pooled_w, None),
        dtype=input_tensor.dtype,
    )
    roi_spec = tf.TensorSpec(
        shape=(pooled_h, pooled_w, channels_static) if channels_static is not None else (pooled_h, pooled_w, None),
        dtype=input_tensor.dtype,
    )

    def roi_pool_single(roi):
        batch_idx = tf.cast(roi[0], tf.int32)
        x1, y1, x2, y2 = tf.unstack(roi[1:5])
        x1 = x1 * spatial_scale
        y1 = y1 * spatial_scale
        x2 = x2 * spatial_scale
        y2 = y2 * spatial_scale

        roi_start_w = tf.cast(tf.round(x1), tf.int32)
        roi_start_h = tf.cast(tf.round(y1), tf.int32)
        roi_end_w = tf.cast(tf.round(x2), tf.int32)
        roi_end_h = tf.cast(tf.round(y2), tf.int32)

        height = tf.shape(input_tensor)[1]
        width = tf.shape(input_tensor)[2]

        roi_start_w = tf.clip_by_value(roi_start_w, 0, width)
        roi_start_h = tf.clip_by_value(roi_start_h, 0, height)
        roi_end_w = tf.clip_by_value(roi_end_w, 0, width)
        roi_end_h = tf.clip_by_value(roi_end_h, 0, height)

        roi_width = tf.maximum(roi_end_w - roi_start_w + 1, 1)
        roi_height = tf.maximum(roi_end_h - roi_start_h + 1, 1)

        bin_size_h = tf.cast(roi_height, tf.float32) / tf.cast(pooled_h, tf.float32)
        bin_size_w = tf.cast(roi_width, tf.float32) / tf.cast(pooled_w, tf.float32)

        channels_dynamic = tf.shape(input_tensor)[-1]
        zero = tf.zeros([channels_dynamic], dtype=input_tensor.dtype)

        def pool_bin(ph, pw):
            ph_f = tf.cast(ph, tf.float32)
            pw_f = tf.cast(pw, tf.float32)
            hstart = tf.cast(tf.floor(ph_f * bin_size_h), tf.int32) + roi_start_h
            hend = tf.cast(tf.ceil((ph_f + 1.0) * bin_size_h), tf.int32) + roi_start_h
            wstart = tf.cast(tf.floor(pw_f * bin_size_w), tf.int32) + roi_start_w
            wend = tf.cast(tf.ceil((pw_f + 1.0) * bin_size_w), tf.int32) + roi_start_w

            hstart = tf.clip_by_value(hstart, 0, height)
            hend = tf.clip_by_value(hend, 0, height)
            wstart = tf.clip_by_value(wstart, 0, width)
            wend = tf.clip_by_value(wend, 0, width)

            is_empty = tf.logical_or(hend <= hstart, wend <= wstart)

            def do_max():
                region = input_tensor[batch_idx, hstart:hend, wstart:wend, :]
                return tf.reduce_max(region, axis=[0,1])

            return tf.cond(is_empty, lambda: zero, do_max)

        def pool_row(ph):
            return tf.map_fn(
                lambda pw: pool_bin(ph, pw),
                tf.range(pooled_w),
                fn_output_signature=channel_spec,
            )

        return tf.map_fn(
            pool_row,
            tf.range(pooled_h),
            fn_output_signature=row_spec,
        )

    pooled_tensor = tf.map_fn(
        roi_pool_single,
        rois,
        fn_output_signature=roi_spec,
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
                'tf_op_type': 'MaxRoiPool',
                'tf_inputs': {
                    'input': input_tensor,
                    'rois': rois,
                    'pooled_shape': pooled_shape,
                    'spatial_scale': spatial_scale,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
