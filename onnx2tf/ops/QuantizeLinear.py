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
    convert_axis,
)
from onnx2tf.utils.enums import ONNX_DTYPES_TO_TF_DTYPES


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


def _expand_scale_or_zero_point(
    *,
    value,
    input_tensor,
    axis: int,
    block_size: int,
):
    value_rank = len(value.shape)
    input_rank = len(input_tensor.shape)

    if value_rank == 0:
        return value

    if block_size > 0 and value_rank == input_rank:
        if value.shape[axis] is None \
            or input_tensor.shape[axis] is None \
            or value.shape[axis] != input_tensor.shape[axis]:
            expanded = tf.repeat(value, repeats=block_size, axis=axis)
            expanded = tf.slice(expanded, [0] * input_rank, tf.shape(input_tensor))
            return expanded
        return value

    if value_rank == 1 and input_rank is not None:
        shape = [1] * input_rank
        shape[axis] = -1
        return tf.reshape(value, shape)

    return value


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """QuantizeLinear

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
        before_op_output_shape_trans,
    )
    graph_node_input_3 = None
    if len(graph_node.inputs) >= 3:
        graph_node_input_3 = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_nhwc = False
    if isinstance(graph_node_input_1, gs.Variable):
        input_nhwc = tf_layers_dict.get(graph_node_input_1.name, {}).get('nhwc', False)
    input_tensor_rank = len(input_tensor.shape)
    y_scale = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    y_zero_point = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    axis = graph_node.attrs.get('axis', 1)
    axis = convert_axis(
        axis=axis,
        tensor_rank=input_tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'is_dequantized': True,
        'nhwc': input_nhwc,
    }

    # Generation of TF OP
    input_tensor = tf.cast(
        x=input_tensor,
        dtype=tf.float32,
    )

    # If QuantizeLinear is immediately followed by Cast -> DequantizeLinear
    # or DequantizeLinear only, bypass fake-quant to avoid generating
    # Mul/Round/Min/Relu/Mul chains in TF/TFLite.
    bypass_fake_quant = False
    if graph_node.outputs and len(graph_node.outputs) > 0:
        consumers = graph_node.outputs[0].outputs
        if consumers:
            bypass_fake_quant = True
            for consumer in consumers:
                if consumer.op == 'DequantizeLinear':
                    continue
                if consumer.op == 'Cast':
                    cast_outs = consumer.outputs[0].outputs if consumer.outputs else []
                    if not cast_outs or any(grand.op != 'DequantizeLinear' for grand in cast_outs):
                        bypass_fake_quant = False
                        break
                else:
                    bypass_fake_quant = False
                    break

    if bypass_fake_quant:
        tf_layers_dict[graph_node_output.name]['tf_node'] = input_tensor
        tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
            make_tf_node_info(
                node_info={
                    'tf_op_type': 'QuantizeLinear',
                    'tf_inputs': {
                        'x': input_tensor,
                    },
                    'tf_outputs': {
                        'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                    },
                }
            )
        return
    y_scale = tf.cast(y_scale, tf.float32)

    block_size = int(graph_node.attrs.get('block_size', 0))
    y_scale = _expand_scale_or_zero_point(
        value=y_scale,
        input_tensor=input_tensor,
        axis=axis,
        block_size=block_size,
    )

    output_dtype_attr = int(graph_node.attrs.get('output_dtype', 0))
    if y_zero_point is None:
        output_dtype = ONNX_DTYPES_TO_TF_DTYPES.get(output_dtype_attr, tf.uint8) \
            if output_dtype_attr != 0 else tf.uint8
        y_zero_point = tf.zeros_like(y_scale)
    else:
        output_dtype = y_zero_point.dtype
        y_zero_point = tf.cast(y_zero_point, tf.float32)
        y_zero_point = _expand_scale_or_zero_point(
            value=y_zero_point,
            input_tensor=input_tensor,
            axis=axis,
            block_size=block_size,
        )

    y = tf.round(tf.divide(input_tensor, y_scale))
    y = tf.add(y, y_zero_point)

    qmin, qmax = _get_qmin_qmax(output_dtype)
    if qmin is not None and qmax is not None:
        y = tf.clip_by_value(y, qmin, qmax)

    # dequantize to float32 output
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.multiply(
            x=tf.subtract(y, y_zero_point),
            y=y_scale,
        )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'QuantizeLinear',
                'tf_inputs': {
                    'x': input_tensor,
                    'y_scale': y_scale,
                    'y_zero_point': y_zero_point,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
