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


def _as_tensor(value):
    if isinstance(value, np.ndarray):
        return tf.convert_to_tensor(value)
    if isinstance(value, (np.generic, int, float, bool, str, bytes)):
        return tf.convert_to_tensor(value)
    return value


def _split_rotary(input_tensor, rotary_dim):
    if isinstance(rotary_dim, int):
        x_rotate = input_tensor[:, :, :, :rotary_dim]
        x_not_rotate = input_tensor[:, :, :, rotary_dim:]
        return x_rotate, x_not_rotate
    rotary_dim = tf.cast(rotary_dim, tf.int32)
    input_shape = tf.shape(input_tensor)
    head_size = input_shape[-1]
    x_rotate = tf.slice(
        input_tensor,
        [0, 0, 0, 0],
        [-1, -1, -1, rotary_dim],
    )
    x_not_rotate = tf.slice(
        input_tensor,
        [0, 0, 0, rotary_dim],
        [-1, -1, -1, head_size - rotary_dim],
    )
    return x_rotate, x_not_rotate


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """RotaryEmbedding

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
    before_op_output_shape_trans_3 = \
        tf_layers_dict.get(graph_node.inputs[2].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2 \
        and before_op_output_shape_trans_3
    if len(graph_node.inputs) >= 4:
        before_op_output_shape_trans_4 = \
            tf_layers_dict.get(graph_node.inputs[3].name, {}).get('before_op_output_shape_trans', True)
        before_op_output_shape_trans = \
            before_op_output_shape_trans \
            and before_op_output_shape_trans_4

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
    graph_node_input_4 = None
    if len(graph_node.inputs) >= 4:
        graph_node_input_4 = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans=False,
        )

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    cos_cache = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    sin_cache = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    position_ids = None
    if graph_node_input_4 is not None:
        position_ids = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
            if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4

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
    cos_cache = pre_process_transpose(
        value_before_transpose=cos_cache,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    sin_cache = pre_process_transpose(
        value_before_transpose=sin_cache,
        param_target='inputs',
        param_name=graph_node.inputs[2].name,
        **kwargs,
    )
    if position_ids is not None:
        position_ids = pre_process_transpose(
            value_before_transpose=position_ids,
            param_target='inputs',
            param_name=graph_node.inputs[3].name,
            **kwargs,
        )

    # Generation of TF OP
    input_tensor = _as_tensor(input_tensor)
    cos_cache = _as_tensor(cos_cache)
    sin_cache = _as_tensor(sin_cache)
    if position_ids is not None:
        position_ids = _as_tensor(position_ids)

    input_dtype = input_tensor.dtype
    if cos_cache.dtype != input_dtype:
        cos_cache = tf.cast(cos_cache, input_dtype)
    if sin_cache.dtype != input_dtype:
        sin_cache = tf.cast(sin_cache, input_dtype)

    input_rank = input_tensor.shape.rank
    if input_rank is None:
        error(
            f'RotaryEmbedding only supports 3D/4D input with known rank.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    original_input_tensor = input_tensor
    original_input_shape = tf.shape(original_input_tensor)

    num_heads = graph_node.attrs.get('num_heads', None)
    rotary_embedding_dim = graph_node.attrs.get('rotary_embedding_dim', 0)
    interleaved = bool(graph_node.attrs.get('interleaved', 0))

    if input_rank == 4:
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 1, 3])
    elif input_rank == 3:
        if num_heads is None or int(num_heads) == 0:
            error(
                f'num_heads attribute is required for 3D input in RotaryEmbedding.\n' +
                f'graph_node.name: {graph_node.name}'
            )
            sys.exit(1)
        num_heads = int(num_heads)
        input_shape = tf.shape(input_tensor)
        head_size = tf.math.floordiv(
            input_shape[-1],
            tf.constant(num_heads, dtype=input_shape.dtype),
        )
        input_tensor = tf.reshape(
            input_tensor,
            tf.stack(
                [
                    input_shape[0],
                    input_shape[1],
                    tf.constant(num_heads, dtype=input_shape.dtype),
                    head_size,
                ]
            ),
        )
    else:
        error(
            f'RotaryEmbedding only supports 3D/4D input.\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    head_size = input_tensor.shape[-1]
    if head_size is None:
        head_size = tf.shape(input_tensor)[-1]

    if rotary_embedding_dim is None or int(rotary_embedding_dim) == 0:
        rotary_embedding_dim = head_size
    else:
        rotary_embedding_dim = int(rotary_embedding_dim)

    x_rotate, x_not_rotate = _split_rotary(input_tensor, rotary_embedding_dim)

    if position_ids is not None:
        cos_cache = tf.gather(cos_cache, position_ids)
        sin_cache = tf.gather(sin_cache, position_ids)

    cos_cache = tf.expand_dims(cos_cache, axis=2)
    sin_cache = tf.expand_dims(sin_cache, axis=2)

    if interleaved:
        x1 = x_rotate[:, :, :, 0::2]
        x2 = x_rotate[:, :, :, 1::2]
    else:
        x1, x2 = tf.split(x_rotate, num_or_size_splits=2, axis=-1)

    real = (cos_cache * x1) - (sin_cache * x2)
    imag = (sin_cache * x1) + (cos_cache * x2)

    if interleaved:
        real = tf.expand_dims(real, axis=-1)
        imag = tf.expand_dims(imag, axis=-1)
        x_rotate = tf.reshape(
            tf.concat([real, imag], axis=-1),
            tf.shape(x_rotate),
        )
    else:
        x_rotate = tf.concat([real, imag], axis=-1)

    output_tensor = tf.concat([x_rotate, x_not_rotate], axis=-1)
    if input_rank == 3:
        output_tensor = tf.reshape(output_tensor, original_input_shape)
    else:
        output_tensor = tf.transpose(output_tensor, perm=[0, 2, 1, 3])

    tf_layers_dict[graph_node_output.name]['tf_node'] = output_tensor

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
                'tf_op_type': 'RotaryEmbedding',
                'tf_inputs': {
                    'input': input_tensor,
                    'cos_cache': cos_cache,
                    'sin_cache': sin_cache,
                    'position_ids': position_ids,
                    'interleaved': interleaved,
                    'rotary_embedding_dim': rotary_embedding_dim,
                    'num_heads': num_heads,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
