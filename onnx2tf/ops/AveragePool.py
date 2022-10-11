import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from tensorflow.python.keras.layers import (
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
)
from onnx2tf.utils.colors import Color
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    calc_pads_same_pooling,
    pad_input,
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
    """AveragePool

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

    # 0: False, 1: True
    ceil_mode = bool(graph_node.attrs.get('ceil_mode', 0))
    # 0: False, 1: True
    count_include_pad = bool(graph_node.attrs.get('count_include_pad', 0))
    kernel_shape = graph_node.attrs['kernel_shape']
    spatial_size = len(kernel_shape)
    x_rank = spatial_size + 2
    strides = graph_node.attrs.get('strides', [1] * spatial_size)
    dilations = graph_node.attrs.get('dilations', [1] * spatial_size)
    is_known_shape = input_tensor.shape.is_fully_defined()

    pads = graph_node.attrs.get('auto_pad', 'NOTSET')
    if pads == 'NOTSET':
        pads = graph_node.attrs.get('pads', [0] * spatial_size * 2)
        if is_known_shape and pads != [0] * spatial_size * 2:
            in_shape = input_tensor.get_shape()
            same_paddings = calc_pads_same_pooling(
                in_spatial_shape=in_shape[1:x_rank - 1],
                kernel_shape=kernel_shape,
                strides=strides,
                dilations=dilations,
                padding='SAME_UPPER',
            )
            if pads == same_paddings:
                pads = 'SAME_UPPER'

    is_explicit_padding = type(pads) is list
    padding_ = ''

    if is_explicit_padding or pads == 'SAME_LOWER' or (pads == 'SAME_UPPER' and count_include_pad):
        # pad the input
        padded_tensor = pad_input(
            input_tensor=input_tensor,
            is_known_shape=is_known_shape,
            kernel_shape=kernel_shape,
            ceil_mode=ceil_mode,
            spatial_size=spatial_size,
            strides=strides,
            dilations=dilations,
            padding=pads,
            padding_constant=0,
        )
        padding_ = 'valid'

    elif pads == 'SAME_UPPER':
        padded_tensor = input_tensor
        padding_ = 'same'

    else:
        padded_tensor = input_tensor
        padding_ = 'same'

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    tf_op_type = None
    if len(kernel_shape) == 1:
        pooled_tensor = AveragePooling1D(
            pool_size=kernel_shape,
            strides=strides,
            padding=padding_,
        )(padded_tensor)
        tf_op_type = AveragePooling1D

    elif len(kernel_shape) == 2:
        pooled_tensor = AveragePooling2D(
            pool_size=kernel_shape,
            strides=strides,
            padding=padding_,
        )(padded_tensor)
        tf_op_type = AveragePooling2D

    elif len(kernel_shape) == 3:
        pooled_tensor = AveragePooling3D(
            pool_size=kernel_shape,
            strides=strides,
            padding=padding_,
        )(padded_tensor)
        tf_op_type = AveragePooling3D

    else:
        error_msg = f'' +\
            f'{Color.RED}ERROR:{Color.RESET} ' +\
            f'AveragePool supports only 1D, 2D, and 3D. ' +\
            f'opname: {graph_node.name} Type: AveragePool{len(kernel_shape)}D'
        print(error_msg)
        assert False, error_msg

    tf_layers_dict[graph_node_output.name]['tf_node'] = pooled_tensor

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'x': input_tensor,
                    'pool_size': kernel_shape,
                    'strides': strides,
                    'padding': padding_,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
