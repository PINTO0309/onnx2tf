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
)
from onnx2tf.utils.logging import *


def _build_col2im_kernel(
    *,
    k_h,
    k_w,
    dilation_h,
    dilation_w,
    dtype,
):
    k_h = tf.cast(k_h, tf.int32)
    k_w = tf.cast(k_w, tf.int32)
    eff_k_h = (k_h - 1) * dilation_h + 1
    eff_k_w = (k_w - 1) * dilation_w + 1

    ky = tf.reshape(tf.repeat(tf.range(k_h), k_w), tf.stack([k_h, k_w]))
    kx = tf.reshape(tf.tile(tf.range(k_w), [k_h]), tf.stack([k_h, k_w]))

    positions = ky * dilation_h * eff_k_w + kx * dilation_w
    positions = tf.reshape(positions, [-1])
    one_hot = tf.one_hot(positions, depth=eff_k_h * eff_k_w, dtype=dtype)
    kernel = tf.reshape(one_hot, tf.stack([k_h * k_w, eff_k_h, eff_k_w]))
    kernel = tf.transpose(kernel, [1, 2, 0])
    kernel = tf.expand_dims(kernel, axis=2)
    return kernel, eff_k_h, eff_k_w


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Col2Im

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = before_op_output_shape_trans_1

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans=False,
    )
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans=False,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_image_shape = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    input_block_shape = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    dilations = graph_node.attrs.get('dilations', [1, 1])
    pads = graph_node.attrs.get('pads', [0, 0, 0, 0])
    strides = graph_node.attrs.get('strides', [1, 1])

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # Pre-process transpose
    before_trans_shape = input_tensor.shape
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    after_trans_shape = input_tensor.shape
    if 'nhwc' in tf_layers_dict[graph_node_output.name].keys() \
        and tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and before_trans_shape != after_trans_shape:
        tf_layers_dict[graph_node_output.name].pop('nhwc')

    # Generation of TF OP
    original_dtype = input_tensor.dtype
    compute_dtype = original_dtype
    if original_dtype not in (tf.float16, tf.float32, tf.float64, tf.bfloat16):
        if original_dtype in (tf.complex64, tf.complex128):
            error('Col2Im does not support complex types in non-Flex implementation.')
            sys.exit(1)
        compute_dtype = tf.float32
        input_tensor = tf.cast(input_tensor, compute_dtype)

    input_image_shape = tf.cast(input_image_shape, tf.int32)
    input_block_shape = tf.cast(input_block_shape, tf.int32)

    if input_image_shape.shape is not None \
        and input_image_shape.shape.rank is not None \
        and input_image_shape.shape.rank != 1:
        error('Col2Im supports only 2D image_shape input.')
        sys.exit(1)

    if input_block_shape.shape is not None \
        and input_block_shape.shape.rank is not None \
        and input_block_shape.shape.rank != 1:
        error('Col2Im supports only 2D block_shape input.')
        sys.exit(1)

    k_h = input_block_shape[0]
    k_w = input_block_shape[1]
    h_img = input_image_shape[0]
    w_img = input_image_shape[1]

    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations
    pad_top, pad_left, pad_bottom, pad_right = pads

    kernel, eff_k_h, eff_k_w = _build_col2im_kernel(
        k_h=k_h,
        k_w=k_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        dtype=compute_dtype,
    )

    h_pad = h_img + pad_top + pad_bottom
    w_pad = w_img + pad_left + pad_right

    out_h = tf.math.floordiv(h_pad - eff_k_h, stride_h) + 1
    out_w = tf.math.floordiv(w_pad - eff_k_w, stride_w) + 1

    input_shape = tf.shape(input_tensor)
    n = input_shape[0]
    ck = input_shape[1]
    c = tf.math.floordiv(ck, k_h * k_w)

    cols = tf.reshape(
        input_tensor,
        tf.stack([n, c, k_h * k_w, out_h, out_w]),
    )
    cols = tf.transpose(cols, [0, 1, 3, 4, 2])
    cols = tf.reshape(cols, tf.stack([n * c, out_h, out_w, k_h * k_w]))

    output_shape = tf.stack([n * c, h_pad, w_pad, 1])
    output = tf.nn.conv2d_transpose(
        cols,
        kernel,
        output_shape=output_shape,
        strides=[1, stride_h, stride_w, 1],
        padding='VALID',
    )

    output = tf.reshape(output, tf.stack([n, c, h_pad, w_pad]))
    output = tf.transpose(output, [0, 2, 3, 1])

    output = tf.slice(
        output,
        tf.stack([0, pad_top, pad_left, 0]),
        tf.stack([-1, h_img, w_img, -1]),
    )

    if output.dtype != original_dtype:
        output = tf.cast(output, original_dtype)

    tf_layers_dict[graph_node_output.name]['tf_node'] = output

    # Post-process transpose
    before_trans_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )
    after_trans_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    if 'nhwc' in tf_layers_dict[graph_node_output.name].keys() \
        and tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and before_trans_shape != after_trans_shape:
        tf_layers_dict[graph_node_output.name].pop('nhwc')

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'Col2Im',
                'tf_inputs': {
                    'input': input_tensor,
                    'image_shape': input_image_shape,
                    'block_shape': input_block_shape,
                    'dilations': dilations,
                    'pads': pads,
                    'strides': strides,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
