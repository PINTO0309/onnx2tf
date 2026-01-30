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


def _normalize_axis(axis, rank):
    axis = tf.cast(axis, tf.int32)
    rank = tf.cast(rank, tf.int32)
    return tf.where(axis < 0, axis + rank, axis)


def _move_axis_to_last(x, axis):
    rank = tf.rank(x)
    axis = _normalize_axis(axis, rank)
    range0 = tf.range(axis)
    range1 = tf.range(axis + 1, rank)
    perm = tf.concat([range0, range1, [axis]], axis=0)
    x_t = tf.transpose(x, perm)
    inv_perm = tf.argsort(perm)
    return x_t, inv_perm


def _pad_or_slice_last(x, length):
    length = tf.cast(length, tf.int32)
    current = tf.shape(x)[-1]

    def _pad():
        pad_amount = length - current
        pad = tf.concat(
            [
                tf.zeros([tf.rank(x) - 1, 2], dtype=tf.int32),
                tf.stack([[0, pad_amount]]),
            ],
            axis=0,
        )
        return tf.pad(x, pad)

    def _slice():
        return x[..., :length]

    x = tf.cond(current < length, _pad, lambda: x)
    x = tf.cond(current > length, _slice, lambda: x)
    return x


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """DFT

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans_1,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans=False,
    ) if len(graph_node.inputs) >= 2 else None
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans=False,
    ) if len(graph_node.inputs) >= 3 else None

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    dft_length = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    axis = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    onesided = bool(graph_node.attrs.get('onesided', 0))
    inverse = bool(graph_node.attrs.get('inverse', 0))

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input_1.name]['nhwc'] \
            if isinstance(graph_node_input_1, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False,
    }

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    input_rank = tf.rank(input_tensor)
    axis_attr = graph_node.attrs.get('axis', None)
    if axis is None:
        if axis_attr is not None:
            axis = tf.constant(axis_attr, dtype=tf.int32)
        else:
            axis = tf.cast(input_rank - 2, tf.int32)
    else:
        axis = tf.cast(tf.reshape(axis, []), tf.int32)
    axis = _normalize_axis(axis, input_rank)

    dft_length_value = None
    if dft_length is not None:
        dft_length_value = tf.cast(tf.reshape(dft_length, []), tf.int32)

    input_dtype = input_tensor.dtype
    if input_dtype in (tf.float64,):
        float_dtype = tf.float64
        complex_dtype = tf.complex128
    elif input_dtype in (tf.float32,):
        float_dtype = tf.float32
        complex_dtype = tf.complex64
    elif input_dtype in (tf.float16, tf.bfloat16):
        float_dtype = tf.float32
        complex_dtype = tf.complex64
    else:
        error('DFT supports float/bfloat16 types only.')
        sys.exit(1)

    # Convert to complex tensor (drop last dim)
    last_dim_static = input_tensor.shape[-1]
    if last_dim_static is not None:
        if last_dim_static == 1:
            real = tf.squeeze(input_tensor, axis=-1)
            imag = tf.zeros_like(real)
        elif last_dim_static == 2:
            real, imag = tf.unstack(input_tensor, axis=-1)
        else:
            error('DFT input last dimension must be 1 or 2.')
            sys.exit(1)
    else:
        last_dim = tf.shape(input_tensor)[-1]
        def _real_case():
            real = tf.squeeze(input_tensor, axis=-1)
            imag = tf.zeros_like(real)
            return real, imag
        def _complex_case():
            real, imag = tf.unstack(input_tensor, axis=-1)
            return real, imag
        real, imag = tf.cond(
            tf.equal(last_dim, 1),
            _real_case,
            _complex_case,
        )

    real = tf.cast(real, float_dtype)
    imag = tf.cast(imag, float_dtype)
    signal = tf.complex(real, imag)

    signal_t, inv_perm = _move_axis_to_last(signal, axis)
    signal_len = tf.shape(signal_t)[-1]

    if onesided and inverse:
        if last_dim_static == 1:
            error('DFT onesided inverse supports only complex input.')
            sys.exit(1)
        if dft_length_value is None:
            dft_length_value = tf.cast(signal_len * 2 - 2, tf.int32)
        out_real = tf.signal.irfft(signal_t, fft_length=[dft_length_value])
        out_real = tf.transpose(out_real, inv_perm)
        output = tf.expand_dims(out_real, axis=-1)
    elif onesided and not inverse:
        if last_dim_static == 2:
            error('DFT onesided forward supports only real input.')
            sys.exit(1)
        if dft_length_value is None:
            dft_length_value = tf.cast(signal_len, tf.int32)
        real_signal = tf.math.real(signal_t)
        out_complex = tf.signal.rfft(real_signal, fft_length=[dft_length_value])
        out_complex = tf.transpose(out_complex, inv_perm)
        output = tf.stack([tf.math.real(out_complex), tf.math.imag(out_complex)], axis=-1)
    else:
        if dft_length_value is not None:
            signal_t = _pad_or_slice_last(signal_t, dft_length_value)
        if inverse:
            out_complex = tf.signal.ifft(signal_t)
        else:
            out_complex = tf.signal.fft(signal_t)
        out_complex = tf.transpose(out_complex, inv_perm)
        output = tf.stack([tf.math.real(out_complex), tf.math.imag(out_complex)], axis=-1)

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
                'tf_op_type': 'DFT',
                'tf_inputs': {
                    'input': input_tensor,
                    'axis': axis,
                    'dft_length': dft_length_value,
                    'onesided': onesided,
                    'inverse': inverse,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
