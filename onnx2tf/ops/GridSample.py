import sys
import copy
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tf_keras
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
    onnx_tf_tensor_validation,
    dummy_tf_inference,
    acquisition_of_validation_data,
)
from onnx2tf.utils.logging import *
from typing import Any, Dict

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

    nhwc = tf_layers_dict[graph_node_input_1.name]['nhwc'] \
        if isinstance(graph_node_input_1, gs.Variable) \
            and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False

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
    mode = graph_node.attrs.get('mode', 'linear')
    padding_mode = graph_node.attrs.get('padding_mode', 'zeros')

    if mode == 'bilinear':
        mode = 'linear'
    ENABLE_MODES = ['linear', 'nearest', 'cubic']
    if mode not in ENABLE_MODES:
        error(
            f'The current implementation of GridSample supports only mode={ENABLE_MODES}. '+
            f'Pull requests are welcome. \n'+
            f'mode: {mode}'
        )
        sys.exit(1)

    ENABLE_PADDING_MODES = ['zeros', 'border', 'reflection']
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
    if not nhwc \
        and onnx_input_shape is not None \
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
    use_linear_gather_2d = padding_mode in ['zeros', 'border', 'reflection'] \
        and mode in ['linear', 'nearest', 'cubic']
    use_linear_gather_3d = padding_mode in ['zeros', 'border', 'reflection'] \
        and mode in ['linear', 'nearest', 'cubic']
    """
    image
        [N, H, W, C]
    grid
        [N, grid_H, grid_W, 2]
    """
    def _reflect_coord(coord, size, align_corners):
        size = tf.cast(size, coord.dtype)
        if align_corners:
            size_minus_one = tf.maximum(size - 1.0, 1.0)
            coord = (size - 1.0) - tf.abs(
                tf.math.floormod(coord, 2.0 * size_minus_one) - (size - 1.0)
            )
        else:
            coord = size - tf.abs(
                tf.math.floormod(coord + 0.5, 2.0 * size) - size
            ) - 0.5
            coord = tf.clip_by_value(coord, 0.0, size - 1.0)
        return coord

    def _normalize_grid(grid_coord, size, align_corners):
        size = tf.cast(size, grid_coord.dtype)
        if align_corners:
            return (grid_coord + 1.0) * (size - 1.0) * 0.5
        return (grid_coord + 1.0) * size * 0.5 - 0.5

    def _cubic_kernel(x, a=-0.75):
        absx = tf.abs(x)
        absx2 = absx * absx
        absx3 = absx2 * absx
        f1 = (a + 2.0) * absx3 - (a + 3.0) * absx2 + 1.0
        f2 = a * absx3 - 5.0 * a * absx2 + 8.0 * a * absx - 4.0 * a
        return tf.where(
            absx <= 1.0,
            f1,
            tf.where(absx < 2.0, f2, tf.zeros_like(x)),
        )

    def _prepare_linear_gather_2d(input_tensor):
        shape = tf.shape(input_tensor)
        h = shape[1]
        w = shape[2]
        c = shape[3]
        input_flat = tf.reshape(input_tensor, tf.stack([shape[0], h * w, c]))
        return input_flat, h, w

    def _prepare_linear_gather_3d(input_tensor):
        shape = tf.shape(input_tensor)
        d = shape[1]
        h = shape[2]
        w = shape[3]
        c = shape[4]
        input_flat = tf.reshape(input_tensor, tf.stack([shape[0], d * h * w, c]))
        return input_flat, d, h, w

    def _gather_1d(input_tensor, x_idx):
        idx = tf.cast(x_idx, tf.int64)
        return tf.gather_nd(input_tensor, idx, batch_dims=1)

    def _gather_2d(input_tensor, y_idx, x_idx, linear_cache=None):
        if use_linear_gather_2d:
            if linear_cache is None:
                input_flat, _, w = _prepare_linear_gather_2d(input_tensor)
            else:
                input_flat, _, w = linear_cache
            w_f = tf.cast(w, y_idx.dtype)
            linear = tf.cast(y_idx * w_f + x_idx, tf.int32)
            linear = tf.squeeze(linear, axis=-1)
            return tf.gather(params=input_flat, indices=linear, batch_dims=1)
        idx = tf.cast(tf.concat([y_idx, x_idx], axis=-1), tf.int64)
        return tf.gather_nd(input_tensor, idx, batch_dims=1)

    def _gather_3d(input_tensor, z_idx, y_idx, x_idx, linear_cache=None):
        if use_linear_gather_3d:
            if linear_cache is None:
                input_flat, d, h, w = _prepare_linear_gather_3d(input_tensor)
            else:
                input_flat, d, h, w = linear_cache
            w_f = tf.cast(w, z_idx.dtype)
            h_f = tf.cast(h, z_idx.dtype)
            linear = z_idx * (h_f * w_f) + y_idx * w_f + x_idx
            linear = tf.cast(linear, tf.int32)
            linear = tf.squeeze(linear, axis=-1)
            return tf.gather(params=input_flat, indices=linear, batch_dims=1)
        idx = tf.cast(tf.concat([z_idx, y_idx, x_idx], axis=-1), tf.int64)
        return tf.gather_nd(input_tensor, idx, batch_dims=1)

    def _grid_sample_1d(image, grid, target_name):
        w_in = tf.shape(image)[1]
        w_in_f = tf.cast(w_in, grid.dtype)
        x = _normalize_grid(grid, w_in_f, align_corners)

        if padding_mode == 'border':
            x = tf.clip_by_value(x, 0.0, w_in_f - 1.0)
            max_x = w_in_f - 1.0
        elif padding_mode == 'reflection':
            x = _reflect_coord(x, w_in_f, align_corners)
            max_x = w_in_f - 1.0
        else:
            pad = 2 if mode == 'cubic' else 1
            x = tf.clip_by_value(x, -float(pad), w_in_f - 1.0 + float(pad)) + float(pad)
            image = tf.pad(image, paddings=[[0,0],[pad,pad],[0,0]])
            max_x = tf.cast(w_in + 2 * pad - 1, grid.dtype)

        if mode == 'nearest':
            x0 = tf.round(x)
            if padding_mode == 'reflection':
                x0 = _reflect_coord(x0, w_in_f, align_corners)
            x0 = tf.clip_by_value(x0, 0.0, max_x)
            output = _gather_1d(image, x0)
            return tf.identity(output, name=target_name)

        if mode == 'cubic':
            x1 = tf.floor(x)
            dx = x - x1
            x0 = x1 - 1.0
            x2 = x1 + 1.0
            x3 = x1 + 2.0

            if padding_mode == 'reflection':
                x0 = _reflect_coord(x0, w_in_f, align_corners)
                x1 = _reflect_coord(x1, w_in_f, align_corners)
                x2 = _reflect_coord(x2, w_in_f, align_corners)
                x3 = _reflect_coord(x3, w_in_f, align_corners)

            x0 = tf.clip_by_value(x0, 0.0, max_x)
            x1 = tf.clip_by_value(x1, 0.0, max_x)
            x2 = tf.clip_by_value(x2, 0.0, max_x)
            x3 = tf.clip_by_value(x3, 0.0, max_x)

            w0 = _cubic_kernel(dx + 1.0)
            w1 = _cubic_kernel(dx)
            w2 = _cubic_kernel(dx - 1.0)
            w3 = _cubic_kernel(dx - 2.0)

            v0 = _gather_1d(image, x0)
            v1 = _gather_1d(image, x1)
            v2 = _gather_1d(image, x2)
            v3 = _gather_1d(image, x3)
            output = w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3
            return tf.identity(output, name=target_name)

        x0 = tf.floor(x)
        x1 = x0 + 1.0
        dx = x - x0

        if padding_mode == 'reflection':
            x0 = _reflect_coord(x0, w_in_f, align_corners)
            x1 = _reflect_coord(x1, w_in_f, align_corners)

        x0 = tf.clip_by_value(x0, 0.0, max_x)
        x1 = tf.clip_by_value(x1, 0.0, max_x)

        w0 = 1.0 - dx
        w1 = dx
        v0 = _gather_1d(image, x0)
        v1 = _gather_1d(image, x1)
        output = w0 * v0 + w1 * v1
        return tf.identity(output, name=target_name)

    def _grid_sample_2d(image, grid, target_name):
        h_in = tf.shape(image)[1]
        w_in = tf.shape(image)[2]
        h_in_f = tf.cast(h_in, grid.dtype)
        w_in_f = tf.cast(w_in, grid.dtype)
        grid_x, grid_y = tf.split(grid, num_or_size_splits=2, axis=-1)
        x = _normalize_grid(grid_x, w_in_f, align_corners)
        y = _normalize_grid(grid_y, h_in_f, align_corners)

        if padding_mode == 'border':
            x = tf.clip_by_value(x, 0.0, w_in_f - 1.0)
            y = tf.clip_by_value(y, 0.0, h_in_f - 1.0)
            max_x = w_in_f - 1.0
            max_y = h_in_f - 1.0
        elif padding_mode == 'reflection':
            x = _reflect_coord(x, w_in_f, align_corners)
            y = _reflect_coord(y, h_in_f, align_corners)
            max_x = w_in_f - 1.0
            max_y = h_in_f - 1.0
        else:
            pad = 2 if mode == 'cubic' else 1
            x = tf.clip_by_value(x, -float(pad), w_in_f - 1.0 + float(pad)) + float(pad)
            y = tf.clip_by_value(y, -float(pad), h_in_f - 1.0 + float(pad)) + float(pad)
            image = tf.pad(image, paddings=[[0,0],[pad,pad],[pad,pad],[0,0]])
            max_x = tf.cast(w_in + 2 * pad - 1, grid.dtype)
            max_y = tf.cast(h_in + 2 * pad - 1, grid.dtype)

        linear_cache = _prepare_linear_gather_2d(image) if use_linear_gather_2d else None

        if mode == 'nearest':
            x0 = tf.round(x)
            y0 = tf.round(y)
            if padding_mode == 'reflection':
                x0 = _reflect_coord(x0, w_in_f, align_corners)
                y0 = _reflect_coord(y0, h_in_f, align_corners)
            x0 = tf.clip_by_value(x0, 0.0, max_x)
            y0 = tf.clip_by_value(y0, 0.0, max_y)
            output = _gather_2d(image, y0, x0, linear_cache)
            return tf.identity(output, name=target_name)

        if mode == 'cubic':
            x1 = tf.floor(x)
            y1 = tf.floor(y)
            dx = x - x1
            dy = y - y1
            x0 = x1 - 1.0
            x2 = x1 + 1.0
            x3 = x1 + 2.0
            y0 = y1 - 1.0
            y2 = y1 + 1.0
            y3 = y1 + 2.0

            if padding_mode == 'reflection':
                x0 = _reflect_coord(x0, w_in_f, align_corners)
                x1 = _reflect_coord(x1, w_in_f, align_corners)
                x2 = _reflect_coord(x2, w_in_f, align_corners)
                x3 = _reflect_coord(x3, w_in_f, align_corners)
                y0 = _reflect_coord(y0, h_in_f, align_corners)
                y1 = _reflect_coord(y1, h_in_f, align_corners)
                y2 = _reflect_coord(y2, h_in_f, align_corners)
                y3 = _reflect_coord(y3, h_in_f, align_corners)

            x0 = tf.clip_by_value(x0, 0.0, max_x)
            x1 = tf.clip_by_value(x1, 0.0, max_x)
            x2 = tf.clip_by_value(x2, 0.0, max_x)
            x3 = tf.clip_by_value(x3, 0.0, max_x)
            y0 = tf.clip_by_value(y0, 0.0, max_y)
            y1 = tf.clip_by_value(y1, 0.0, max_y)
            y2 = tf.clip_by_value(y2, 0.0, max_y)
            y3 = tf.clip_by_value(y3, 0.0, max_y)

            wx0 = _cubic_kernel(dx + 1.0)
            wx1 = _cubic_kernel(dx)
            wx2 = _cubic_kernel(dx - 1.0)
            wx3 = _cubic_kernel(dx - 2.0)
            wy0 = _cubic_kernel(dy + 1.0)
            wy1 = _cubic_kernel(dy)
            wy2 = _cubic_kernel(dy - 1.0)
            wy3 = _cubic_kernel(dy - 2.0)

            out = 0.0
            for x_idx, wx in [(x0, wx0), (x1, wx1), (x2, wx2), (x3, wx3)]:
                for y_idx, wy in [(y0, wy0), (y1, wy1), (y2, wy2), (y3, wy3)]:
                    out = out + _gather_2d(image, y_idx, x_idx, linear_cache) * wx * wy
            return tf.identity(out, name=target_name)

        x0 = tf.floor(x)
        y0 = tf.floor(y)
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        dx = x - x0
        dy = y - y0

        if padding_mode == 'reflection':
            x0 = _reflect_coord(x0, w_in_f, align_corners)
            x1 = _reflect_coord(x1, w_in_f, align_corners)
            y0 = _reflect_coord(y0, h_in_f, align_corners)
            y1 = _reflect_coord(y1, h_in_f, align_corners)

        x0 = tf.clip_by_value(x0, 0.0, max_x)
        x1 = tf.clip_by_value(x1, 0.0, max_x)
        y0 = tf.clip_by_value(y0, 0.0, max_y)
        y1 = tf.clip_by_value(y1, 0.0, max_y)

        w_y0_x0 = (1.0 - dy) * (1.0 - dx)
        w_y1_x0 = dy * (1.0 - dx)
        w_y1_x1 = dy * dx
        w_y0_x1 = (1.0 - dy) * dx

        v_y0_x0 = _gather_2d(image, y0, x0, linear_cache)
        v_y1_x0 = _gather_2d(image, y1, x0, linear_cache)
        v_y1_x1 = _gather_2d(image, y1, x1, linear_cache)
        v_y0_x1 = _gather_2d(image, y0, x1, linear_cache)

        output = w_y0_x0 * v_y0_x0 + w_y1_x0 * v_y1_x0 + w_y1_x1 * v_y1_x1 + w_y0_x1 * v_y0_x1
        return tf.identity(output, name=target_name)

    def _grid_sample_3d(image, grid, target_name):
        d_in = tf.shape(image)[1]
        h_in = tf.shape(image)[2]
        w_in = tf.shape(image)[3]
        d_in_f = tf.cast(d_in, grid.dtype)
        h_in_f = tf.cast(h_in, grid.dtype)
        w_in_f = tf.cast(w_in, grid.dtype)
        grid_x, grid_y, grid_z = tf.split(grid, num_or_size_splits=3, axis=-1)
        x = _normalize_grid(grid_x, w_in_f, align_corners)
        y = _normalize_grid(grid_y, h_in_f, align_corners)
        z = _normalize_grid(grid_z, d_in_f, align_corners)

        if padding_mode == 'border':
            x = tf.clip_by_value(x, 0.0, w_in_f - 1.0)
            y = tf.clip_by_value(y, 0.0, h_in_f - 1.0)
            z = tf.clip_by_value(z, 0.0, d_in_f - 1.0)
            max_x = w_in_f - 1.0
            max_y = h_in_f - 1.0
            max_z = d_in_f - 1.0
        elif padding_mode == 'reflection':
            x = _reflect_coord(x, w_in_f, align_corners)
            y = _reflect_coord(y, h_in_f, align_corners)
            z = _reflect_coord(z, d_in_f, align_corners)
            max_x = w_in_f - 1.0
            max_y = h_in_f - 1.0
            max_z = d_in_f - 1.0
        else:
            pad = 2 if mode == 'cubic' else 1
            x = tf.clip_by_value(x, -float(pad), w_in_f - 1.0 + float(pad)) + float(pad)
            y = tf.clip_by_value(y, -float(pad), h_in_f - 1.0 + float(pad)) + float(pad)
            z = tf.clip_by_value(z, -float(pad), d_in_f - 1.0 + float(pad)) + float(pad)
            image = tf.pad(image, paddings=[[0,0],[pad,pad],[pad,pad],[pad,pad],[0,0]])
            max_x = tf.cast(w_in + 2 * pad - 1, grid.dtype)
            max_y = tf.cast(h_in + 2 * pad - 1, grid.dtype)
            max_z = tf.cast(d_in + 2 * pad - 1, grid.dtype)

        linear_cache = _prepare_linear_gather_3d(image) if use_linear_gather_3d else None

        if mode == 'nearest':
            x0 = tf.round(x)
            y0 = tf.round(y)
            z0 = tf.round(z)
            if padding_mode == 'reflection':
                x0 = _reflect_coord(x0, w_in_f, align_corners)
                y0 = _reflect_coord(y0, h_in_f, align_corners)
                z0 = _reflect_coord(z0, d_in_f, align_corners)
            x0 = tf.clip_by_value(x0, 0.0, max_x)
            y0 = tf.clip_by_value(y0, 0.0, max_y)
            z0 = tf.clip_by_value(z0, 0.0, max_z)
            output = _gather_3d(image, z0, y0, x0, linear_cache)
            return tf.identity(output, name=target_name)

        if mode == 'cubic':
            x1 = tf.floor(x)
            y1 = tf.floor(y)
            z1 = tf.floor(z)
            dx = x - x1
            dy = y - y1
            dz = z - z1
            x0 = x1 - 1.0
            x2 = x1 + 1.0
            x3 = x1 + 2.0
            y0 = y1 - 1.0
            y2 = y1 + 1.0
            y3 = y1 + 2.0
            z0 = z1 - 1.0
            z2 = z1 + 1.0
            z3 = z1 + 2.0

            if padding_mode == 'reflection':
                x0 = _reflect_coord(x0, w_in_f, align_corners)
                x1 = _reflect_coord(x1, w_in_f, align_corners)
                x2 = _reflect_coord(x2, w_in_f, align_corners)
                x3 = _reflect_coord(x3, w_in_f, align_corners)
                y0 = _reflect_coord(y0, h_in_f, align_corners)
                y1 = _reflect_coord(y1, h_in_f, align_corners)
                y2 = _reflect_coord(y2, h_in_f, align_corners)
                y3 = _reflect_coord(y3, h_in_f, align_corners)
                z0 = _reflect_coord(z0, d_in_f, align_corners)
                z1 = _reflect_coord(z1, d_in_f, align_corners)
                z2 = _reflect_coord(z2, d_in_f, align_corners)
                z3 = _reflect_coord(z3, d_in_f, align_corners)

            x0 = tf.clip_by_value(x0, 0.0, max_x)
            x1 = tf.clip_by_value(x1, 0.0, max_x)
            x2 = tf.clip_by_value(x2, 0.0, max_x)
            x3 = tf.clip_by_value(x3, 0.0, max_x)
            y0 = tf.clip_by_value(y0, 0.0, max_y)
            y1 = tf.clip_by_value(y1, 0.0, max_y)
            y2 = tf.clip_by_value(y2, 0.0, max_y)
            y3 = tf.clip_by_value(y3, 0.0, max_y)
            z0 = tf.clip_by_value(z0, 0.0, max_z)
            z1 = tf.clip_by_value(z1, 0.0, max_z)
            z2 = tf.clip_by_value(z2, 0.0, max_z)
            z3 = tf.clip_by_value(z3, 0.0, max_z)

            wx0 = _cubic_kernel(dx + 1.0)
            wx1 = _cubic_kernel(dx)
            wx2 = _cubic_kernel(dx - 1.0)
            wx3 = _cubic_kernel(dx - 2.0)
            wy0 = _cubic_kernel(dy + 1.0)
            wy1 = _cubic_kernel(dy)
            wy2 = _cubic_kernel(dy - 1.0)
            wy3 = _cubic_kernel(dy - 2.0)
            wz0 = _cubic_kernel(dz + 1.0)
            wz1 = _cubic_kernel(dz)
            wz2 = _cubic_kernel(dz - 1.0)
            wz3 = _cubic_kernel(dz - 2.0)

            out = 0.0
            for x_idx, wx in [(x0, wx0), (x1, wx1), (x2, wx2), (x3, wx3)]:
                for y_idx, wy in [(y0, wy0), (y1, wy1), (y2, wy2), (y3, wy3)]:
                    for z_idx, wz in [(z0, wz0), (z1, wz1), (z2, wz2), (z3, wz3)]:
                        out = out + _gather_3d(image, z_idx, y_idx, x_idx, linear_cache) * wx * wy * wz
            return tf.identity(out, name=target_name)

        x0 = tf.floor(x)
        y0 = tf.floor(y)
        z0 = tf.floor(z)
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        z1 = z0 + 1.0
        dx = x - x0
        dy = y - y0
        dz = z - z0

        if padding_mode == 'reflection':
            x0 = _reflect_coord(x0, w_in_f, align_corners)
            x1 = _reflect_coord(x1, w_in_f, align_corners)
            y0 = _reflect_coord(y0, h_in_f, align_corners)
            y1 = _reflect_coord(y1, h_in_f, align_corners)
            z0 = _reflect_coord(z0, d_in_f, align_corners)
            z1 = _reflect_coord(z1, d_in_f, align_corners)

        x0 = tf.clip_by_value(x0, 0.0, max_x)
        x1 = tf.clip_by_value(x1, 0.0, max_x)
        y0 = tf.clip_by_value(y0, 0.0, max_y)
        y1 = tf.clip_by_value(y1, 0.0, max_y)
        z0 = tf.clip_by_value(z0, 0.0, max_z)
        z1 = tf.clip_by_value(z1, 0.0, max_z)

        w000 = (1.0 - dz) * (1.0 - dy) * (1.0 - dx)
        w001 = (1.0 - dz) * (1.0 - dy) * dx
        w010 = (1.0 - dz) * dy * (1.0 - dx)
        w011 = (1.0 - dz) * dy * dx
        w100 = dz * (1.0 - dy) * (1.0 - dx)
        w101 = dz * (1.0 - dy) * dx
        w110 = dz * dy * (1.0 - dx)
        w111 = dz * dy * dx

        v000 = _gather_3d(image, z0, y0, x0, linear_cache)
        v001 = _gather_3d(image, z0, y0, x1, linear_cache)
        v010 = _gather_3d(image, z0, y1, x0, linear_cache)
        v011 = _gather_3d(image, z0, y1, x1, linear_cache)
        v100 = _gather_3d(image, z1, y0, x0, linear_cache)
        v101 = _gather_3d(image, z1, y0, x1, linear_cache)
        v110 = _gather_3d(image, z1, y1, x0, linear_cache)
        v111 = _gather_3d(image, z1, y1, x1, linear_cache)

        output = (
            w000 * v000 + w001 * v001 + w010 * v010 + w011 * v011 +
            w100 * v100 + w101 * v101 + w110 * v110 + w111 * v111
        )
        return tf.identity(output, name=target_name)

    def define_fast_gridsample(image, grid, align_corners, target_name):
        is_string = image.dtype == tf.string
        if is_string and mode != 'nearest':
            error('GridSample supports string input types only with nearest mode.')
            sys.exit(1)

        compute_dtype = tf.float64 \
            if image.dtype == tf.float64 or grid.dtype == tf.float64 else tf.float32
        image_compute = image if (is_string or image.dtype.is_complex) else tf.cast(image, compute_dtype)
        grid_compute = tf.cast(grid, compute_dtype)

        image_rank = len(image_compute.shape)
        if image_rank == 3:
            output = _grid_sample_1d(image_compute, grid_compute, target_name)
        elif image_rank == 4:
            output = _grid_sample_2d(image_compute, grid_compute, target_name)
        elif image_rank == 5:
            output = _grid_sample_3d(image_compute, grid_compute, target_name)
        else:
            error(f'GridSample supports only 1D/2D/3D inputs. Got rank={image_rank}.')
            sys.exit(1)

        if output.dtype != image.dtype:
            output = tf.cast(output, image.dtype)
        return output

    def define_accurate_gridsample(image, grid, align_corners, target_name):
        return define_fast_gridsample(
            image=image,
            grid=grid,
            align_corners=align_corners,
            target_name=target_name,
        )

    disable_strict_mode: bool = kwargs['disable_strict_mode']
    enable_fast_gridsample = True
    min_abs_err = sys.maxsize

    # Workaround define_accurate_gridsample to a problem where the accuracy of
    # define_fast_gridsample degrades significantly when more than 3 channels of tensors are input.
    # Instead of maintaining accuracy, inference is sacrificed.
    if not disable_strict_mode:
        # Obtain ONNX inference results and
        # TensorFlow inference results up to the previous layer of TensorFlow
        onnx_tensor_infos, validation_data_1, validation_data_2 = \
            acquisition_of_validation_data(
                input_tensor_1=image,
                input_tensor_2=grid,
                graph_node_output=graph_node_output,
                tf_layers_dict=tf_layers_dict,
                **kwargs,
            )
        try:
            # Build TF dummy model
            input_1 = tf_keras.Input(
                shape=validation_data_1.shape[1:],
                batch_size=validation_data_1.shape[0] \
                    if isinstance(validation_data_1.shape[0], int) else None,
                name='dummy_input_1',
                dtype=validation_data_1.dtype,
            )
            input_2 = tf_keras.Input(
                shape=validation_data_2.shape[1:],
                batch_size=validation_data_2.shape[0] \
                    if isinstance(validation_data_2.shape[0], int) else None,
                name='dummy_input_2',
                dtype=validation_data_2.dtype,
            )
            dummy_gridsample = \
                define_fast_gridsample(
                    image=input_1,
                    grid=input_2,
                    align_corners=align_corners,
                    target_name=graph_node.name,
                )
            # Perform simple accuracy verification
            # Terminate when the error is less than 1e-3
            if onnx_tensor_infos:
                try:
                    # Search for the axis with the smallest error
                    val_model = tf_keras.Model(
                        inputs=[
                            input_1,
                            input_2,
                        ],
                        outputs=[
                            dummy_gridsample,
                        ],
                    )

                    # TF dummy inference
                    tf_tensor_infos: Dict[Any] = \
                        dummy_tf_inference(
                            model=val_model,
                            inputs=[
                                input_1,
                                input_2,
                            ],
                            verification_datas=[
                                validation_data_1,
                                validation_data_2,
                            ],
                        )
                    del input_1
                    del input_2
                    del dummy_gridsample
                    del val_model

                    # Validation
                    onnx_tf_output_pairs = {
                        (oi[0], ti[0]): (oi[1], ti[1]) \
                            for oi, ti in zip(onnx_tensor_infos.items(), tf_tensor_infos.items())
                    }
                    """
                    check_results: Dict[str, List[np.ndarray, int, float|int]]
                        {
                            onnx_output_name: [
                                onnx_tensor,
                                matched_flg, <--- 0: Unmatched, 1: Matched, 2: Skipped (Deleted or Shape Unmatched)
                                max_abs_err,
                            ]
                        }
                    """
                    check_results = \
                        onnx_tf_tensor_validation(
                            output_pairs=onnx_tf_output_pairs,
                            rtol=0.0,
                            atol=0.0,
                        )
                    result_err = sum([val[2] for val in check_results.values()])
                    if result_err < min_abs_err:
                        min_abs_err = result_err
                        if min_abs_err < 1e-3:
                            enable_fast_gridsample = True
                        else:
                            enable_fast_gridsample = False
                except Exception as ex1:
                    pass
        except Exception as ex2:
            pass

    if enable_fast_gridsample:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            define_fast_gridsample(
                image=image,
                grid=grid,
                align_corners=align_corners,
                target_name=graph_node.name,
            )
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            define_accurate_gridsample(
                image=image,
                grid=grid,
                align_corners=align_corners,
                target_name=graph_node.name,
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
