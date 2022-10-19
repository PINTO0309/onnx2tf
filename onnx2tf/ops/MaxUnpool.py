import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """MaxUnpool

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
    input_tensor_shape = input_tensor.shape
    ind = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    output_shape = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    kernel_shape = graph_node.attrs['kernel_shape']
    spatial_size = len(kernel_shape)
    strides = graph_node.attrs.get("strides", [1] * spatial_size)
    pads = graph_node.attrs.get("pads", None)

    default_shape = []
    for d in range(len(kernel_shape)):
        default_shape.append(
            (input_tensor_shape[d + 1] - 1) * int(strides[d]) + int(kernel_shape[d])
        )
    default_shape = \
        [input_tensor_shape[0]] \
        + default_shape \
        + [input_tensor_shape[-1]]


    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    flat_input_size = tf.reduce_prod(input_tensor=input_tensor_shape)
    flat_output_shape = [
        tf.reduce_prod(input_tensor=default_shape)
    ]
    pool_ = tf.reshape(
        tensor=input_tensor,
        shape=[flat_input_size],
    )
    ind_ = tf.reshape(
        tensor=ind,
        shape=[flat_input_size, 1],
    )
    ret = tf.scatter_nd(
        indices=ind_,
        updates=pool_,
        shape=tf.cast(flat_output_shape, tf.int64),
    )
    unpooled = tf.reshape(
        tensor=ret,
        shape=default_shape,
    )
    unpooled_shape = unpooled.get_shape() \
        if hasattr(unpooled, 'get_shape') else unpooled.shape

    if output_shape is not None:
        unpool_shape = tf.cast(unpooled_shape, dtype=tf.int32)
        new_shape = tf.cast(output_shape, dtype=tf.int32)
        pads_begin = []
        pads_end = []
        for d in range(len(unpool_shape)):
            pad_total = new_shape[d] - unpool_shape[d]
            pad_begin = tf.cast(pad_total / 2, tf.int32)
            pad_end = pad_total - pad_begin
            pads_begin = pads_begin + [pad_begin]
            pads_end = pads_end + [pad_end]
        pads = pads_begin + pads_end

    if pads is not None:
        unpool_shape = unpooled_shape
        paddings = []
        paddings = paddings + [0, 0]
        for d in range(0, spatial_size):
            paddings = paddings + [[pads[d], pads[d + spatial_size]]]
        paddings = paddings + [0, 0]
        unpooled = tf.pad(
            tensor=unpooled,
            paddings=paddings,
            mode='CONSTANT',
            constant_values=0,
        )

    tf_layers_dict[graph_node_output.name]['tf_node'] = unpooled

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'MaxUnpool',
                'tf_inputs': {
                    'input': input_tensor,
                    'indices': ind,
                    'output_shape': output_shape,
                    'kernel_shape': kernel_shape,
                    'pads': pads,
                    'strides': strides,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
