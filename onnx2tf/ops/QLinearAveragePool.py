import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx2tf.gs as gs
from tensorflow.python.keras.layers import (
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
)
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    calc_tf_pooling_pads,
    calc_extra_padding_with_ceil,
)


def _apply_avg_pool(
    *,
    input_tensor: tf.Tensor,
    kernel_shape: list,
    strides: list,
    padding: str,
):
    spatial_size = len(kernel_shape)
    if spatial_size == 1:
        if input_tensor.shape[1] is not None and kernel_shape[0] > input_tensor.shape[1]:
            return AveragePooling1D(
                pool_size=[input_tensor.shape[1]],
                strides=[input_tensor.shape[1]],
                padding=padding.upper(),
            )(input_tensor), AveragePooling1D
        return AveragePooling1D(
            pool_size=kernel_shape,
            strides=strides,
            padding=padding.upper(),
        )(input_tensor), AveragePooling1D
    if spatial_size == 2:
        return AveragePooling2D(
            pool_size=kernel_shape,
            strides=strides,
            padding=padding.upper(),
        )(input_tensor), AveragePooling2D
    if spatial_size == 3:
        return AveragePooling3D(
            pool_size=kernel_shape,
            strides=strides,
            padding=padding.upper(),
        )(input_tensor), AveragePooling3D
    raise ValueError(
        f'QLinearAveragePool supports only 1D, 2D, and 3D. Type: AveragePool{spatial_size}D'
    )


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """QLinearAveragePool

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
    graph_node_input_4 = get_constant_or_variable(
        graph_node.inputs[3],
        before_op_output_shape_trans,
    )
    graph_node_input_5 = get_constant_or_variable(
        graph_node.inputs[4],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    x = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    x_is_dequantized = False
    x_nhwc = False
    if isinstance(graph_node_input_1, gs.Variable):
        x_is_dequantized = tf_layers_dict.get(graph_node_input_1.name, {}).get('is_dequantized', False)
        x_nhwc = tf_layers_dict.get(graph_node_input_1.name, {}).get('nhwc', False)

    x_scale = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    x_zero_point = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    y_scale = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4
    y_zero_point = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) else graph_node_input_5
    output_dtype = y_zero_point.dtype if y_zero_point.dtype not in [tf.int8, tf.uint8] else tf.float32

    auto_pad = graph_node.attrs.get('auto_pad', 'NOTSET')
    ceil_mode = bool(graph_node.attrs.get('ceil_mode', 0))
    count_include_pad = bool(graph_node.attrs.get('count_include_pad', 0))
    kernel_shape = graph_node.attrs['kernel_shape']
    spatial_size = len(kernel_shape)
    dilations = graph_node.attrs.get('dilations', [1] * spatial_size)
    pads = graph_node.attrs.get('pads', [0] * spatial_size * 2)
    strides = graph_node.attrs.get('strides', [1] * spatial_size)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # Generation of TF OP
    x = tf.cast(x, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    x_zero_point = tf.cast(x_zero_point, tf.float32)
    y_scale = tf.cast(y_scale, tf.float32)
    y_zero_point = tf.cast(y_zero_point, tf.float32)

    if x_is_dequantized:
        dequantized_x = x
    else:
        dequantized_x = tf.multiply(
            x=x_scale,
            y=tf.subtract(x, x_zero_point),
        )

    input_tensor_shape = dequantized_x.shape.as_list()
    is_known_shape = None not in input_tensor_shape[1:]
    tf_pad_mode = 'VALID'
    is_explicit_padding = False
    tf_pads = calc_tf_pooling_pads(
        input_shape=input_tensor_shape,
        kernel=kernel_shape,
        strides=strides,
        input_tensor=dequantized_x,
    )

    if auto_pad == 'NOTSET':
        if is_known_shape and pads != [0] * spatial_size * 2 and tf_pads == pads:
            tf_pad_mode = 'SAME'
        else:
            is_explicit_padding = True
            if ceil_mode and is_known_shape:
                extra_pads = calc_extra_padding_with_ceil(
                    input_shape=input_tensor_shape[1:-1],
                    kernel=kernel_shape,
                    pads=pads,
                    dilations=dilations,
                    strides=strides,
                )
                pads = pads[:len(pads) // 2] + [p + e for p, e in zip(pads[len(pads) // 2:], extra_pads)]
            tf_pads = pads
    elif auto_pad == 'SAME_UPPER':
        tf_pad_mode = 'SAME'
    elif auto_pad == 'SAME_LOWER':
        is_explicit_padding = True
    elif auto_pad == 'VALID':
        tf_pads = [0] * spatial_size * 2
    else:
        raise ValueError(f'Wrong auto_pad parameter in QLinearAveragePool: {auto_pad}.')

    tf_pads_as_tf = None
    pooled_input = dequantized_x
    if is_explicit_padding and tf_pads != [0] * spatial_size * 2:
        if auto_pad == 'SAME_LOWER':
            # switch the order of pads
            tf_pads = [i for tup in zip(tf_pads[len(tf_pads) // 2:], tf_pads[:len(tf_pads) // 2]) for i in tup]

        tf_pads_as_tf = \
            [[0, 0]] + \
            [list(i) for i in zip(tf_pads[:len(tf_pads) // 2], tf_pads[len(tf_pads) // 2:])] + \
            [[0, 0]]
        pooled_input = tf.pad(
            tensor=dequantized_x,
            paddings=tf_pads_as_tf,
            mode='CONSTANT',
        )

    pooled, tf_op_type = _apply_avg_pool(
        input_tensor=pooled_input,
        kernel_shape=kernel_shape,
        strides=strides,
        padding=tf_pad_mode,
    )

    # Match ONNX count_include_pad=False for explicit paddings.
    if is_explicit_padding and not count_include_pad and tf_pads_as_tf is not None:
        mask = tf.ones_like(dequantized_x, dtype=pooled.dtype)
        mask = tf.pad(
            tensor=mask,
            paddings=tf_pads_as_tf,
            mode='CONSTANT',
        )
        mask_pooled, _ = _apply_avg_pool(
            input_tensor=mask,
            kernel_shape=kernel_shape,
            strides=strides,
            padding=tf_pad_mode,
        )
        kernel_volume = float(np.prod(kernel_shape))
        count_valid = mask_pooled * tf.cast(kernel_volume, dtype=mask_pooled.dtype)
        multiplier = tf.math.divide_no_nan(
            tf.cast(kernel_volume, dtype=mask_pooled.dtype),
            count_valid,
        )
        pooled = pooled * multiplier

    requantized = tf.add(
        x=tf.divide(
            x=pooled,
            y=y_scale,
        ),
        y=y_zero_point,
    )

    tf_layers_dict[graph_node_output.name]['tf_node'] = tf.cast(requantized, output_dtype)

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'x': x,
                    'x_scale': x_scale,
                    'x_zero_point': x_zero_point,
                    'y_scale': y_scale,
                    'y_zero_point': y_zero_point,
                    'kernel_shape': kernel_shape,
                    'strides': strides,
                    'pads': tf_pads if tf_pad_mode != 'SAME' else tf_pad_mode,
                    'count_include_pad': count_include_pad,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
