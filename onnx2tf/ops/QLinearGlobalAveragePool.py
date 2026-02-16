import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx2tf.gs as gs
from onnx2tf.utils.common_functions import (
    convert_axis,
    get_constant_or_variable,
    inverted_operation_enable_disable,
    make_tf_node_info,
    print_node_info,
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """QLinearGlobalAveragePool

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
    x_nhwc = False
    if isinstance(graph_node_input_1, gs.Variable):
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

    tensor_rank = len(x.shape)
    channels_last = int(graph_node.attrs.get('channels_last', 0))
    axis_transpose_required = before_op_output_shape_trans
    if channels_last == 0 and x_nhwc:
        # Quantized branches can carry NHWC tensors even when shape metadata is unknown.
        axis_transpose_required = True
    onnx_spatial_axes = \
        [dim for dim in range(1, tensor_rank - 1)] \
        if channels_last != 0 else [dim for dim in range(2, tensor_rank)]
    reduce_axes = [
        convert_axis(
            axis=axis,
            tensor_rank=tensor_rank,
            before_op_output_shape_trans=axis_transpose_required if channels_last == 0 else False,
        )
        for axis in onnx_spatial_axes
    ]

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

    dequantized_x = tf.multiply(
        x=x_scale,
        y=tf.subtract(x, x_zero_point),
    )
    pooled = tf.reduce_mean(
        input_tensor=dequantized_x,
        axis=reduce_axes,
        keepdims=True,
        name=graph_node.name,
    )
    requantized = tf.add(
        x=tf.divide(
            x=pooled,
            y=y_scale,
        ),
        y=y_zero_point,
    )

    tf_layers_dict[graph_node_output.name]['tf_node'] = tf.cast(requantized, output_dtype)

    input_shape_onnx = graph_node_input_1.shape
    if isinstance(input_shape_onnx, (list, tuple)) and len(input_shape_onnx) == tensor_rank:
        inferred_shape = [dim if dim is not None else None for dim in input_shape_onnx]
    else:
        batch_dim = x.shape[0] if len(x.shape) > 0 else None
        channel_dim = None
        if channels_last != 0:
            channel_dim = x.shape[-1] if len(x.shape) > 1 else None
            inferred_shape = [batch_dim] + [None for _ in range(max(0, tensor_rank - 2))] + [channel_dim]
        else:
            if axis_transpose_required:
                channel_dim = x.shape[-1] if len(x.shape) > 1 else None
            else:
                channel_dim = x.shape[1] if len(x.shape) > 1 else None
            inferred_shape = [batch_dim, channel_dim] + [None for _ in range(max(0, tensor_rank - 2))]

    for axis in onnx_spatial_axes:
        if axis < len(inferred_shape):
            inferred_shape[axis] = 1
    graph_node_output.shape = inferred_shape
    tf_layers_dict[graph_node_output.name]['shape'] = inferred_shape

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'QLinearGlobalAveragePool',
                'tf_inputs': {
                    'x': x,
                    'x_scale': x_scale,
                    'x_zero_point': x_zero_point,
                    'y_scale': y_scale,
                    'y_zero_point': y_zero_point,
                    'reduce_axes': reduce_axes,
                    'channels_last': channels_last,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
