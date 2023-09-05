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
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
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
    """GridSample

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
        False \
            if hasattr(graph_node.inputs[1], 'values') \
                and isinstance(graph_node.inputs[1].values, np.ndarray) \
                    else before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    image = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    grid = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    input_tensor_shape = image.shape

    # Pre-process transpose
    image = pre_process_transpose(
        value_before_transpose=image,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    grid = pre_process_transpose(
        value_before_transpose=grid,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )

    align_corners = bool(graph_node.attrs.get('align_corners', 0))
    mode = graph_node.attrs.get('mode', 'bilinear')
    padding_mode = graph_node.attrs.get('padding_mode', 'zeros')

    ENABLE_MODES = ['bilinear']
    if mode not in ENABLE_MODES:
        error(
            f'The current implementation of GridSample supports only mode={ENABLE_MODES}. '+
            f'Pull requests are welcome. \n'+
            f'mode: {mode}'
        )
        sys.exit(1)

    ENABLE_PADDING_MODES = ['zeros']
    if padding_mode not in ENABLE_PADDING_MODES:
        error(
            f'The current implementation of GridSample supports only mode={ENABLE_PADDING_MODES}. '+
            f'Pull requests are welcome. \n'+
            f'mode: {padding_mode}'
        )
        sys.exit(1)

    # Workaround to avoid as many conversion failures as possible
    # for models with useless Transpose immediately before them.
    # If the input geometry of the ONNX and the input geometry of the TF model match,
    # the input geometry on the TF model side is forcibly transposed to the NWC or NHWC or NDHWC format.
    # However, if all dimensions of CW or CHW or CDHW have the same value,
    # the forced transposition process is skipped because it may destroy the structure of the model.
    onnx_input_shape = [
        dim if isinstance(dim, int) else None for dim in graph_node.inputs[0].shape
    ] if graph_node.inputs[0].shape is not None else None
    tf_input_shape = [
        dim if isinstance(dim, int) else None for dim in input_tensor_shape
    ]
    if onnx_input_shape is not None \
        and len(onnx_input_shape) > 1 and len(tf_input_shape) > 1 \
        and onnx_input_shape == tf_input_shape:

        shape_for_judging_skip = [
            dim if dim is not None else INF_INDEX_VALUE for dim in onnx_input_shape[1:]
        ]
        if shape_for_judging_skip.count(shape_for_judging_skip[0]) != len(shape_for_judging_skip):
            if len(onnx_input_shape) == 3:
                # 1D
                image = transpose_with_flexing_deterrence(
                    input_tensor=image,
                    perm=[0,2,1],
                    **kwargs,
                )
            elif len(onnx_input_shape) == 4:
                # 2D
                image = transpose_with_flexing_deterrence(
                    input_tensor=image,
                    perm=[0,2,3,1],
                    **kwargs,
                )
            elif len(onnx_input_shape) == 5:
                # 3D
                image = transpose_with_flexing_deterrence(
                    input_tensor=image,
                    perm=[0,2,3,4,1],
                    **kwargs,
                )

    # Generation of TF OP
    """
    image
        [N, H, W, C]
    grid
        [N, grid_H, grid_W, 2]
    """
    _, h_in, w_in, _ = image.shape

    if align_corners:
        pixs = tf.math.multiply(grid + 1.0, tf.convert_to_tensor([(w_in - 1) * 0.5, (h_in - 1) * 0.5], dtype=tf.float32))
    else:
        pixs = (tf.math.multiply(grid + 1.0, tf.convert_to_tensor([w_in, h_in], dtype=tf.float32)) - 1.0) * 0.5
    
    # x/y coordinate map dimension: [N, H, W, 1]
    x, y = tf.split(pixs, num_or_size_splits=2, axis=-1)

    x0 = tf.clip_by_value(tf.math.floor(x), clip_value_min=0, clip_value_max=w_in - 1)
    y0 = tf.clip_by_value(tf.math.floor(y), clip_value_min=0, clip_value_max=h_in - 1)

    x1 = tf.clip_by_value(x0 + 1, clip_value_min=0, clip_value_max=w_in - 1)
    y1 = tf.clip_by_value(y0 + 1, clip_value_min=0, clip_value_max=h_in - 1)

    dx = tf.math.subtract(x, x0)
    dy = tf.math.subtract(y, y0)

    # bilinear interpolation
    #   image[x, y] = \
    #       (1 - dy) * (1 - dx) * image[y0, x0] + \
    #           dy   * (1 - dx) * image[y1, x0] + \
    #           dy   *    dx    * image[y1, x1] + \
    #       (1 - dy) *    dx    * image[y0, x1]
    w_y0_x0 = tf.math.multiply(1.0 - dy, 1.0 - dx)
    w_y1_x0 = tf.math.multiply(dy, 1.0 - dx)
    w_y1_x1 = tf.math.multiply(dy, dx)
    w_y0_x1 = tf.math.multiply(1.0 - dy, dx)

    # input - [N, H_in, W_in, C]
    # grid - [N, H_out, W_out, 2]
    # output - [N, H_out, W_out, C]
    v_y0_x0 = tf.gather_nd(params=image, indices=tf.cast(tf.concat([y0, x0], axis=-1), dtype=tf.int64), batch_dims=1)
    v_y1_x0 = tf.gather_nd(params=image, indices=tf.cast(tf.concat([y1, x0], axis=-1), dtype=tf.int64), batch_dims=1)
    v_y1_x1 = tf.gather_nd(params=image, indices=tf.cast(tf.concat([y1, x1], axis=-1), dtype=tf.int64), batch_dims=1)
    v_y0_x1 = tf.gather_nd(params=image, indices=tf.cast(tf.concat([y0, x1], axis=-1), dtype=tf.int64), batch_dims=1)

    output = w_y0_x0 * v_y0_x0 + w_y1_x0 * v_y1_x0 + w_y1_x1 * v_y1_x1 + w_y0_x1 * v_y0_x1
    
    x_invalid = tf.math.logical_or(
        tf.math.less(x, tf.convert_to_tensor(0.0, dtype=tf.float32)),
        tf.math.greater(x, tf.convert_to_tensor(w_in - 1.0, dtype=tf.float32)))
    y_invalid = tf.math.logical_or(
        tf.math.less(y, tf.convert_to_tensor(0.0, dtype=tf.float32)),
        tf.math.greater(y, tf.convert_to_tensor(h_in - 1.0, dtype=tf.float32)))
    invalid = tf.math.logical_or(x_invalid, y_invalid)

    output = tf.where(
        condition=invalid,
        x=tf.convert_to_tensor(0.0, dtype=tf.float32),
        y=output)

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
                'tf_op_type': 'GridSample',
                'tf_inputs': {
                    'image': image,
                    'grid': grid,
                    'align_corners': align_corners,
                    'mode': mode,
                    'padding_mode': padding_mode,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
