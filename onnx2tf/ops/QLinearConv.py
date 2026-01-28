import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    get_weights_constant_or_variable,
    get_padding_as_op,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
)
from onnx2tf.utils.logging import *


def _dequantize_tensor(
    *,
    base,
    zero_point,
    scale,
):
    # Do computation in float32
    base = tf.cast(base, tf.float32)
    zero_point = tf.cast(zero_point, tf.float32)
    scale = tf.cast(scale, tf.float32)
    return (base - zero_point) * scale


def _reshape_per_output_channel(
    *,
    value,
    weights,
):
    value_rank = len(value.shape)
    weights_rank = len(weights.shape)
    if value_rank == 0:
        return value
    if value_rank == 1 and weights_rank is not None:
        shape = [1] * weights_rank
        shape[-1] = -1
        return tf.reshape(value, shape)
    return value


def _dequantize_weights(
    *,
    base,
    zero_point,
    scale,
):
    # Do computation in float32
    casted_base = tf.cast(base, tf.float32)
    casted_zero_point = tf.cast(zero_point, tf.float32)
    casted_scale = tf.cast(scale, tf.float32)
    casted_zero_point = _reshape_per_output_channel(
        value=casted_zero_point,
        weights=casted_base,
    )
    casted_scale = _reshape_per_output_channel(
        value=casted_scale,
        weights=casted_base,
    )
    return (casted_base - casted_zero_point) * casted_scale


def _get_qmin_qmax(dtype: tf.dtypes.DType):
    if dtype == tf.uint8:
        return 0.0, 255.0
    if dtype == tf.int8:
        return -128.0, 127.0
    if dtype == tf.uint16:
        return 0.0, 65535.0
    if dtype == tf.int16:
        return -32768.0, 32767.0
    return None, None


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """QLinearConv

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
    kernel_shape = graph_node.attrs.get('kernel_shape', [])
    kernel_size = len(kernel_shape)
    graph_node_input_4 = get_weights_constant_or_variable(
        const_or_var=graph_node.inputs[3],
        kernel_size=kernel_size,
    )
    graph_node_input_5 = get_constant_or_variable(
        graph_node.inputs[4],
        before_op_output_shape_trans,
    )
    graph_node_input_6 = get_constant_or_variable(
        graph_node.inputs[5],
        before_op_output_shape_trans,
    )
    graph_node_input_7 = get_constant_or_variable(
        graph_node.inputs[6],
        before_op_output_shape_trans,
    )
    graph_node_input_8 = get_constant_or_variable(
        graph_node.inputs[7],
        before_op_output_shape_trans,
    )
    graph_node_input_9 = None
    if len(graph_node.inputs) >= 9:
        graph_node_input_9 = get_constant_or_variable(
            graph_node.inputs[8],
            before_op_output_shape_trans,
        )

    graph_node_output: gs.Variable = graph_node.outputs[0]
    try:
        output_tensor_shape = graph_node.o().outputs[0].shape
    except:
        output_tensor_shape = graph_node.outputs[0].shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_is_dequantized = False
    input_nhwc = False
    if isinstance(graph_node_input_1, gs.Variable):
        input_is_dequantized = tf_layers_dict.get(graph_node_input_1.name, {}).get('is_dequantized', False)
        input_nhwc = tf_layers_dict.get(graph_node_input_1.name, {}).get('nhwc', False)
    input_tensor_scale = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    input_tensor_zero_point = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    input_weights = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4
    input_weights_scale = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) else graph_node_input_5
    input_weights_zero_point = tf_layers_dict[graph_node_input_6.name]['tf_node'] \
        if isinstance(graph_node_input_6, gs.Variable) else graph_node_input_6
    y_scale = tf_layers_dict[graph_node_input_7.name]['tf_node'] \
        if isinstance(graph_node_input_7, gs.Variable) else graph_node_input_7
    y_zero_point = tf_layers_dict[graph_node_input_8.name]['tf_node'] \
        if isinstance(graph_node_input_8, gs.Variable) else graph_node_input_8
    input_bias = tf_layers_dict[graph_node_input_9.name]['tf_node'] \
        if isinstance(graph_node_input_9, gs.Variable) else graph_node_input_9
    output_quant_dtype = y_zero_point.dtype

    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)
    spatial_size = input_tensor_rank - 2
    input_weights_shape = input_weights.shape
    auto_pad = graph_node.attrs.get('auto_pad', 'NOTSET')
    dilations = graph_node.attrs.get('dilations', [1] * spatial_size)
    group = graph_node.attrs.get('group', 1)
    pads = graph_node.attrs.get('pads', [0, 0] * spatial_size)
    strides = graph_node.attrs.get('strides', [1] * spatial_size)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': output_tensor_shape,
        'dtype': dtype,
        'is_dequantized': True,
        'nhwc': input_nhwc,
    }

    # Generation of TF OP

    # Dequantize variables to float32
    if input_is_dequantized:
        input_tensor = tf.cast(input_tensor, tf.float32)
    else:
        input_tensor = _dequantize_tensor(
            base=input_tensor,
            zero_point=input_tensor_zero_point,
            scale=input_tensor_scale,
        )
    input_weights = _dequantize_weights(
        base=input_weights,
        zero_point=input_weights_zero_point,
        scale=input_weights_scale,
    )

    # if bias is defined save it here
    if input_bias is not None:
        input_bias = tf.cast(input_bias, tf.float32)
        input_bias_scale = tf.cast(input_tensor_scale, tf.float32) * tf.cast(input_weights_scale, tf.float32)
        input_bias = input_bias * input_bias_scale

    """
    Conv1D
    Conv2D
    Conv3D
    DepthwiseConv2D
    SeparableConv2D
    """
    # Check auto_pad nonexistent or NOTSET first
    pad_mode = 'VALID'
    if auto_pad == 'NOTSET':
        if input_tensor_rank >=2 \
            and graph_node.i().inputs[0].shape[2:] == output_tensor_shape[2:]:
            pad_mode = "SAME"
        elif pads != [0, 0] * spatial_size:
            input_tensor = get_padding_as_op(
                x=input_tensor,
                pads=pads,
            )
            pad_mode = 'VALID'
        else:
            pad_mode = 'VALID'
    # Then we use auto_pad to setup pad_mode
    elif auto_pad == "SAME_UPPER":
        pad_mode = "SAME"
    elif auto_pad == "VALID":
        pad_mode = "VALID"
    elif auto_pad == "SAME_LOWER":
        error_msg = f'' +\
            Color.RED(f'ERROR:') + ' ' +\
            f'Invalid auto_pad attribute: {auto_pad}'
        print(error_msg)
        assert False, error_msg
    else:
        error_msg = f'' +\
            Color.RED(f'ERROR:') + ' ' +\
            f'Invalid auto_pad attribute: {auto_pad}'
        print(error_msg)
        assert False, error_msg


    depthwise = (input_tensor_rank == 4 and len(input_weights_shape) == 4 and group != 1 and not (None in input_weights_shape))
    if depthwise and input_tensor_shape[-1] != None:
        depthwise = bool(group == input_tensor_shape[-1])

    if depthwise is True:
        depthwise_filter_shape = list(input_weights_shape[0:2]) + [input_weights_shape[2], input_weights_shape[3] // group]
        input_weights = tf.reshape(input_weights, depthwise_filter_shape)

    # Conv
    tf_op_type = None
    if not depthwise:
        if group == 1:
            # Conv1D, Conv2D, Conv3D - No Bias
            conv_node = \
                tf.nn.convolution(
                    input=input_tensor,
                    filters=input_weights,
                    strides=strides,
                    padding=pad_mode,
                    dilations=dilations,
                )
            tf_op_type = tf.nn.convolution
        else:
            # SeparableConv
            input_tensor_splits = tf.split(input_tensor, num_or_size_splits=group, axis=-1)
            weight_splits = tf.split(input_weights, num_or_size_splits=group, axis=-1)
            conv_node = \
                tf.concat(
                    values=[
                        tf.nn.convolution(
                            input=input_tensor_split,
                            filters=weight_split,
                            padding=pad_mode,
                            strides=strides,
                            dilations=dilations,
                        ) for (input_tensor_split, weight_split) in zip(input_tensor_splits, weight_splits)
                    ],
                    axis=-1
                )
            tf_op_type = tf.nn.convolution
    else:
        # DepthwiseConv2D
        strides = [1] + strides + [1]
        conv_node = \
            tf.nn.depthwise_conv2d(
                input=input_tensor,
                filter=input_weights,
                padding=pad_mode,
                strides=strides,
                dilations=dilations,
            )
        tf_op_type = tf.nn.depthwise_conv2d

    # Add bias to the convolution (float)
    if input_bias is not None:
        conv_node = tf.add(
            x=conv_node,
            y=input_bias,
        )

    # quantize then dequantize to float32
    y_scale = tf.cast(y_scale, tf.float32)
    y_zero_point = tf.cast(y_zero_point, tf.float32)
    quantized = tf.round(tf.divide(conv_node, y_scale)) + y_zero_point
    qmin, qmax = _get_qmin_qmax(output_quant_dtype)
    if qmin is not None and qmax is not None:
        quantized = tf.clip_by_value(quantized, qmin, qmax)
    dequantized = tf.multiply(tf.subtract(quantized, y_zero_point), y_scale)

    tf_layers_dict[graph_node_output.name]['tf_node'] = dequantized

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'input': input_tensor,
                    'weights': input_weights,
                    'bias': input_bias,
                    'strides': strides,
                    'dilations': dilations,
                    'padding': pad_mode,
                    'group': group,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
