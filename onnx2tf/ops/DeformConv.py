import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    get_weights_constant_or_variable,
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


def _to_int_tensor(value, name=None):
    if isinstance(value, tf.Tensor):
        return tf.cast(value, tf.int32)
    return tf.constant(value, dtype=tf.int32, name=name)


def _bilinear_sample_2d(
    image,
    coords,
):
    """
    image: [N, H, W, C]
    coords: [N, oH, oW, kH, kW, 2] in absolute coords (y, x)
    """
    coord_dtype = coords.dtype
    h = tf.shape(image)[1]
    w = tf.shape(image)[2]
    h_f = tf.cast(h, coord_dtype)
    w_f = tf.cast(w, coord_dtype)
    max_y = h_f - 1.0
    max_x = w_f - 1.0

    y, x = tf.split(coords, num_or_size_splits=2, axis=-1)

    y0 = tf.floor(y)
    x0 = tf.floor(x)
    y1 = y0 + 1.0
    x1 = x0 + 1.0

    dy = y - y0
    dx = x - x0

    w00 = (1.0 - dy) * (1.0 - dx)
    w10 = dy * (1.0 - dx)
    w11 = dy * dx
    w01 = (1.0 - dy) * dx

    def _in_bounds(y_idx, x_idx):
        return tf.logical_and(
            tf.logical_and(y_idx >= 0.0, y_idx <= max_y),
            tf.logical_and(x_idx >= 0.0, x_idx <= max_x),
        )

    m00 = _in_bounds(y0, x0)
    m10 = _in_bounds(y1, x0)
    m11 = _in_bounds(y1, x1)
    m01 = _in_bounds(y0, x1)

    y0c = tf.clip_by_value(y0, 0.0, max_y)
    x0c = tf.clip_by_value(x0, 0.0, max_x)
    y1c = tf.clip_by_value(y1, 0.0, max_y)
    x1c = tf.clip_by_value(x1, 0.0, max_x)

    y0i = tf.cast(y0c, tf.int32)
    x0i = tf.cast(x0c, tf.int32)
    y1i = tf.cast(y1c, tf.int32)
    x1i = tf.cast(x1c, tf.int32)

    input_flat = tf.reshape(image, tf.stack([tf.shape(image)[0], h * w, tf.shape(image)[3]]))

    def _gather(y_idx, x_idx):
        linear = y_idx * w + x_idx
        linear = tf.squeeze(linear, axis=-1)
        return tf.gather(input_flat, linear, batch_dims=1)

    v00 = _gather(y0i, x0i)
    v10 = _gather(y1i, x0i)
    v11 = _gather(y1i, x1i)
    v01 = _gather(y0i, x1i)

    m00 = tf.cast(m00, image.dtype)
    m10 = tf.cast(m10, image.dtype)
    m11 = tf.cast(m11, image.dtype)
    m01 = tf.cast(m01, image.dtype)

    output = w00 * m00 * v00 + w10 * m10 * v10 + w11 * m11 * v11 + w01 * m01 * v01
    return output


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """DeformConv

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_3 = \
        tf_layers_dict.get(graph_node.inputs[2].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_4 = \
        tf_layers_dict.get(graph_node.inputs[3].name, {}).get('before_op_output_shape_trans', True) \
            if len(graph_node.inputs) >= 4 else True
    before_op_output_shape_trans_5 = \
        tf_layers_dict.get(graph_node.inputs[4].name, {}).get('before_op_output_shape_trans', True) \
            if len(graph_node.inputs) >= 5 else True

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans_1,
    )

    kernel_shape = graph_node.attrs.get('kernel_shape', [])
    if kernel_shape == [] and graph_node.inputs[1].shape is not None:
        kernel_shape = graph_node.inputs[1].shape[2:]
    kernel_size = len(kernel_shape) if kernel_shape != [] else 2

    graph_node_input_2 = get_weights_constant_or_variable(
        const_or_var=graph_node.inputs[1],
        kernel_size=kernel_size,
    )
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans_3,
    )
    graph_node_input_4 = get_constant_or_variable(
        graph_node.inputs[3],
        before_op_output_shape_trans_4,
    ) if len(graph_node.inputs) >= 4 else None
    graph_node_input_5 = get_constant_or_variable(
        graph_node.inputs[4],
        before_op_output_shape_trans_5,
    ) if len(graph_node.inputs) >= 5 else None

    graph_node_output: gs.Variable = graph_node.outputs[0]
    output_tensor_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    weights = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    offset = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    bias = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4
    mask = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) else graph_node_input_5

    input_tensor_shape = input_tensor.shape

    if input_tensor_shape is not None and len(input_tensor_shape) != 4:
        error('DeformConv currently supports only 2D inputs (N, C, H, W).')
        sys.exit(1)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': output_tensor_shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    offset = pre_process_transpose(
        value_before_transpose=offset,
        param_target='inputs',
        param_name=graph_node.inputs[2].name,
        **kwargs,
    )
    if mask is not None:
        mask = pre_process_transpose(
            value_before_transpose=mask,
            param_target='inputs',
            param_name=graph_node.inputs[4].name,
            **kwargs,
        )

    input_dtype = input_tensor.dtype
    if weights is not None and weights.dtype != input_dtype:
        weights = tf.cast(weights, input_dtype)
    if offset is not None and offset.dtype != input_dtype:
        offset = tf.cast(offset, input_dtype)
    if bias is not None and bias.dtype != input_dtype:
        bias = tf.cast(bias, input_dtype)
    if mask is not None and mask.dtype != input_dtype:
        mask = tf.cast(mask, input_dtype)

    # Workaround to avoid as many conversion failures as possible
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
            input_tensor = transpose_with_flexing_deterrence(
                input_tensor=input_tensor,
                perm=[0,2,3,1],
                **kwargs,
            )
            offset = transpose_with_flexing_deterrence(
                input_tensor=offset,
                perm=[0,2,3,1],
                **kwargs,
            )
            if mask is not None:
                mask = transpose_with_flexing_deterrence(
                    input_tensor=mask,
                    perm=[0,2,3,1],
                    **kwargs,
                )

    # Attributes
    dilations = graph_node.attrs.get('dilations', [1, 1])
    group = graph_node.attrs.get('group', 1)
    offset_group = graph_node.attrs.get('offset_group', 1)
    pads = graph_node.attrs.get('pads', [0, 0, 0, 0])
    strides = graph_node.attrs.get('strides', [1, 1])

    dilation_h, dilation_w = dilations
    stride_h, stride_w = strides
    pad_top, pad_left, pad_bottom, pad_right = pads

    # Input prep
    if pad_top != 0 or pad_bottom != 0 or pad_left != 0 or pad_right != 0:
        input_tensor = tf.pad(
            input_tensor,
            paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        )

    batch = tf.shape(input_tensor)[0]
    in_h = tf.shape(input_tensor)[1]
    in_w = tf.shape(input_tensor)[2]
    in_c = tf.shape(input_tensor)[3]

    offset_shape = tf.shape(offset)
    out_h = offset_shape[1]
    out_w = offset_shape[2]

    # Kernel shape
    if kernel_shape != []:
        kh = _to_int_tensor(kernel_shape[0])
        kw = _to_int_tensor(kernel_shape[1])
    else:
        kh = _to_int_tensor(tf.shape(weights)[0])
        kw = _to_int_tensor(tf.shape(weights)[1])

    # Base grid: [oH, oW, kH, kW, 2]
    oy = tf.range(out_h, dtype=input_dtype) * tf.cast(stride_h, input_dtype)
    ox = tf.range(out_w, dtype=input_dtype) * tf.cast(stride_w, input_dtype)
    ky = tf.range(kh, dtype=input_dtype) * tf.cast(dilation_h, input_dtype)
    kx = tf.range(kw, dtype=input_dtype) * tf.cast(dilation_w, input_dtype)

    oy = tf.reshape(oy, tf.stack([out_h, 1, 1, 1]))
    ox = tf.reshape(ox, tf.stack([1, out_w, 1, 1]))
    ky = tf.reshape(ky, tf.stack([1, 1, kh, 1]))
    kx = tf.reshape(kx, tf.stack([1, 1, 1, kw]))

    y = oy + ky
    x = ox + kx
    target_shape = tf.stack([out_h, out_w, kh, kw])
    y = tf.broadcast_to(y, target_shape)
    x = tf.broadcast_to(x, target_shape)
    base_grid = tf.stack([y, x], axis=-1)

    # Offset reshape: [N, oH, oW, Goff, kH, kW, 2]
    offset = tf.reshape(
        offset,
        tf.stack([batch, out_h, out_w, offset_group, kh, kw, 2]),
    )

    coords = base_grid[None, :, :, None, :, :, :] + offset
    coords = tf.transpose(coords, [0, 3, 1, 2, 4, 5, 6])
    coords = tf.reshape(coords, tf.stack([batch * offset_group, out_h, out_w, kh, kw, 2]))

    # Input grouping for offset_group
    c_per_offset = tf.math.floordiv(in_c, offset_group)
    input_tensor = tf.reshape(
        input_tensor,
        tf.stack([batch, in_h, in_w, offset_group, c_per_offset]),
    )
    input_tensor = tf.transpose(input_tensor, [0, 3, 1, 2, 4])
    input_tensor = tf.reshape(
        input_tensor,
        tf.stack([batch * offset_group, in_h, in_w, c_per_offset]),
    )

    sampled = _bilinear_sample_2d(input_tensor, coords)
    sampled = tf.reshape(
        sampled,
        tf.stack([batch, offset_group, out_h, out_w, kh, kw, c_per_offset]),
    )
    sampled = tf.transpose(sampled, [0, 2, 3, 1, 4, 5, 6])

    if mask is not None:
        mask = tf.reshape(
            mask,
            tf.stack([batch, out_h, out_w, offset_group, kh, kw, 1]),
        )
        sampled = sampled * tf.cast(mask, sampled.dtype)

    # Merge offset_group back to channel dim: [N, oH, oW, kH, kW, C]
    sampled = tf.reshape(
        sampled,
        tf.stack([batch, out_h, out_w, kh, kw, in_c]),
    )

    # Grouped convolution via batched matmul
    out_c = tf.shape(weights)[3]
    c_per_group = tf.math.floordiv(in_c, group)
    out_c_per_group = tf.math.floordiv(out_c, group)

    cols = tf.reshape(sampled, tf.stack([batch * out_h * out_w, kh * kw * in_c]))
    cols = tf.reshape(cols, tf.stack([batch * out_h * out_w, group, kh * kw * c_per_group]))
    cols = tf.transpose(cols, [1, 0, 2])

    weights = tf.reshape(weights, tf.stack([kh, kw, c_per_group, group, out_c_per_group]))
    weights = tf.transpose(weights, [3, 0, 1, 2, 4])
    weights = tf.reshape(weights, tf.stack([group, kh * kw * c_per_group, out_c_per_group]))

    output = tf.matmul(cols, weights)
    output = tf.transpose(output, [1, 0, 2])
    output = tf.reshape(output, tf.stack([batch, out_h, out_w, out_c]))

    if bias is not None:
        output += tf.reshape(bias, tf.stack([1, 1, 1, out_c]))

    if output.dtype != input_dtype:
        output = tf.cast(output, input_dtype)

    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=output,
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'DeformConv',
                'tf_inputs': {
                    'input_tensor': input_tensor,
                    'weights': weights,
                    'offset': offset,
                    'bias': bias,
                    'mask': mask,
                    'strides': strides,
                    'dilations': dilations,
                    'pads': pads,
                    'group': group,
                    'offset_group': offset_group,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
