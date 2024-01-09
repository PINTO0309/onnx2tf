import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from functools import partial

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
from onnx2tf.utils.enums import ONNX_DTYPES_TO_TF_DTYPES
from onnx2tf.utils.logging import *

# from tensorflow.python.framework import ops

def get_window_onnx(window_length, onnx_window, dtype=tf.float32):
    return tf.cast(onnx_window, dtype=dtype)

def custom_stft(signal, frame_length, frame_step, window, graph_node_name):
    spec = tf.signal.stft(
                signals=signal,
                frame_length=frame_length,
                frame_step=frame_step,
                window_fn=partial(get_window_onnx, onnx_window=window),
                name=graph_node_name,
            )

    # view_as_real
    extended_bin = spec[..., None]
    spec = tf.concat([tf.math.real(extended_bin), tf.math.imag(extended_bin)], axis=-1)
    spec = tf.cast(spec, dtype=signal.dtype)

    return spec


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """STFT

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """

    # ONNX parameter read section
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)


    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans=before_op_output_shape_trans_1,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans=False,
    )
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans=False,
    )
    graph_node_input_4 = get_constant_or_variable(
        graph_node.inputs[3],
        before_op_output_shape_trans=False,
    )

    signal = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    frame_step = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    window = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    frame_length = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4
    onesided = bool(graph_node.attrs.get('onesided', 1))


    # Preserving Graph Structure (Dict)
    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    if onesided:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            custom_stft(
                signal, frame_length, frame_step, window, graph_node.name
            )
    else:
        error(f"STFT onesided={onesided} not yet implemented")
        sys.exit(1)

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
                'tf_op_type': tf.signal.stft,
                'tf_inputs': {
                    'onesided': onesided,
                    'signal': signal,
                    'frame_step': frame_step,
                    'window': window,
                    'frame_length': frame_length,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
