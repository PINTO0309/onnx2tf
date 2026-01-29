import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
import cv2
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


def _as_tensor(value):
    if isinstance(value, np.ndarray):
        return tf.convert_to_tensor(value)
    if isinstance(value, (np.generic, int, float, bool, str, bytes)):
        return tf.convert_to_tensor(value)
    return value


def _decode_image_np(encoded_stream, pixel_format):
    if encoded_stream is None:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    if encoded_stream.dtype != np.uint8:
        encoded_stream = encoded_stream.astype(np.uint8)
    if encoded_stream.size == 0:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    if encoded_stream.ndim != 1:
        encoded_stream = encoded_stream.reshape(-1)
    try:
        if pixel_format == 'Grayscale':
            flag = cv2.IMREAD_GRAYSCALE
        else:
            flag = cv2.IMREAD_COLOR
        decoded = cv2.imdecode(encoded_stream, flag)
        if decoded is None:
            raise ValueError('cv2.imdecode failed')
        if pixel_format == 'RGB':
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        if pixel_format == 'Grayscale' and decoded.ndim == 2:
            decoded = decoded[..., np.newaxis]
        return decoded.astype(np.uint8)
    except Exception:
        if pixel_format == 'Grayscale':
            return np.zeros((0, 0, 1), dtype=np.uint8)
        return np.zeros((0, 0, 3), dtype=np.uint8)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """ImageDecoder

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

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    input_tensor = _as_tensor(input_tensor)
    pixel_format = graph_node.attrs.get('pixel_format', 'RGB')
    if pixel_format not in ['RGB', 'BGR', 'Grayscale']:
        error(
            f'ImageDecoder pixel_format={pixel_format} is not supported.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        pixel_format = 'RGB'

    decoded = tf.numpy_function(
        func=lambda x: _decode_image_np(x, pixel_format),
        inp=[input_tensor],
        Tout=tf.uint8,
        name=graph_node.name,
    )
    channels = 1 if pixel_format == 'Grayscale' else 3
    decoded = tf.ensure_shape(decoded, [None, None, channels])
    tf_layers_dict[graph_node_output.name]['tf_node'] = decoded

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
                'tf_op_type': 'ImageDecoder',
                'tf_inputs': {
                    'encoded_stream': input_tensor,
                    'pixel_format': pixel_format,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
