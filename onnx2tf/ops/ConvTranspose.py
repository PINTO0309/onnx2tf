import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    get_weights_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    convert_reverse_axis,
    tf_shape,
    make_tf_node_info,
)
from onnx2tf.utils.colors import Color


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """ConvTranspose

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

    input_tensor = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    kernel_shape = graph_node.attrs.get('kernel_shape', [])
    kernel_size = len(kernel_shape)
    input_weights = get_weights_constant_or_variable(
        const_or_var=graph_node.inputs[1],
        kernel_size=kernel_size,
    )
    input_bias = None
    if len(graph_node.inputs) >= 3:
        input_bias = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
            is_bias=True,
        )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[input_tensor.name]['tf_node'] \
        if isinstance(input_tensor, gs.Variable) else input_tensor
    input_tensor_shape = input_tensor.shape
    input_weights = tf_layers_dict[input_weights.name]['tf_node'] \
        if isinstance(input_weights, gs.Variable) else input_weights
    input_bias = tf_layers_dict[input_weights.name]['tf_node'] \
        if isinstance(input_bias, gs.Variable) else input_bias

    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)
    spatial_size = input_tensor_rank - 2
    input_weights_shape = input_weights.shape
    auto_pad = graph_node.attrs.get('auto_pad', 'NOTSET')
    dilations = graph_node.attrs.get('dilations', [1] * spatial_size)
    group = graph_node.attrs.get('group', 1)
    output_padding = graph_node.attrs.get('output_padding', None)
    output_shape = graph_node.attrs.get('output_shape', None)
    pads = graph_node.attrs.get('pads', [0, 0] * spatial_size)
    strides = graph_node.attrs.get('strides', [1] * spatial_size)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    # Check auto_pad nonexistent or NOTSET first
    pad_mode = 'VALID'
    if auto_pad == 'NOTSET':
        pad_mode = 'NOTSET'
    # Then we use auto_pad to setup pad_mode
    elif auto_pad == "SAME_UPPER":
        pad_mode = "SAME"
    elif auto_pad == "VALID":
        pad_mode = "VALID"
    elif auto_pad == "SAME_LOWER":
        error_msg = f'' +\
            f'{Color.RED}ERROR:{Color.RESET} ' +\
            f'Invalid auto_pad attribute: {auto_pad}'
        print(error_msg)
        assert False, error_msg
    else:
        error_msg = f'' +\
            f'{Color.RED}ERROR:{Color.RESET} ' +\
            f'Invalid auto_pad attribute: {auto_pad}'
        print(error_msg)
        assert False, error_msg


    # ConvTranspose
    # if dilations != [1] * spatial_size:
    #     error_msg = f'' +\
    #         f'{Color.RED}ERROR:{Color.RESET} ' +\
    #         f'Cannot set non-1 dilation for ConvTranspose.'
    #     print(error_msg)
    #     assert False, error_msg

    if group == 1:
        input_tensor_splits = [input_tensor]
        weight_splits = [input_weights]
    else:
        input_tensor_splits = tf.split(input_tensor, num_or_size_splits=group, axis=-1)
        weight_splits = tf.split(input_weights, num_or_size_splits=group, axis=-1)
    convolved = []

    # get corresponding function in tf
    conv_func = None
    if spatial_size == 1:
        conv_func = tf.nn.conv1d_transpose
        strides = strides[0]
    elif spatial_size == 2:
        conv_func = tf.nn.conv2d_transpose
    elif spatial_size == 3:
        conv_func = tf.nn.conv3d_transpose
    else:
        error_msg = f'' +\
            f'{Color.RED}ERROR:{Color.RESET} ' +\
            f'Transposed convolution for {spatial_size}d is not implemented in Tensorflow.'
        print(error_msg)
        assert False, error_msg

    for (input_tensor_split, weight_split) in zip(input_tensor_splits, weight_splits):
        input_tensor_split_shape = input_tensor_split.shape
        input_tensor_split_spatial_shape = input_tensor_split_shape[1:-1]
        conv_output_shape = None

        if pad_mode == "NOTSET":
            if graph_node_output.shape is not None:
                converted_axis = []
                for idx in range(len(graph_node_output.shape)):
                    converted_axis.append(
                        convert_reverse_axis(
                            axis=idx,
                            tensor_rank=input_tensor_rank,
                            before_op_output_shape_trans=True,
                        )
                    )
                conv_output_shape = [0] * input_tensor_rank
                for idx in range(len(graph_node_output.shape)):
                    conv_output_shape[idx] = graph_node_output.shape[converted_axis[idx]]
            elif output_shape is None:
                conv_output_shape = \
                input_tensor_split_shape[0] + \
                [
                    strides[i] * input_tensor_split_spatial_shape[i] - strides[i] + (kernel_shape[i] - 1) * dilations[i] + 1 \
                        for i in list(range(spatial_size))
                ] + \
                input_tensor_split_shape[-1]
            elif output_shape is not None:
                conv_output_shape = [
                    s + pads[i] + pads[spatial_size + i] \
                        if s is not None else None \
                            for i, s in enumerate(output_shape)
                ]

            # use raw input x to do transposed conv
            conv_rs = conv_func(
                input=input_tensor_split,
                filters=weight_split,
                output_shape=conv_output_shape,
                strides=strides,
                padding="VALID",
            )

            # pad output first by output_padding attr
            if output_padding is not None and output_shape is None:
                output_padding = \
                [
                    [0, 0]
                ] + \
                [
                    [0, p] for p in output_padding
                ] + \
                [
                    [0, 0]
                ]
                conv_rs = tf.pad(conv_rs, output_padding)

            # remove pads set in pads attr
            conv_rs_shape = tf_shape(
                input_tensor=conv_rs,
                dtype=tf.int32,
            )
            conv_rs_shape_list = [
                conv_rs_shape[i] for i in range(conv_rs.shape.rank)
            ]
            begin = [0] + pads[:spatial_size] + [0]
            size = \
            [-1] + \
            [
                s - pads[idx] - pads[idx + spatial_size] for idx, s in enumerate(conv_rs_shape_list[1:-1])
            ] + \
            [-1]
            conv_rs = tf.slice(conv_rs, begin=begin, size=size)
            convolved.append(conv_rs)

        else:
            if graph_node_output.shape is not None:
                converted_axis = []
                for idx in range(len(graph_node_output.shape)):
                    converted_axis.append(
                        convert_reverse_axis(
                            axis=idx,
                            tensor_rank=input_tensor_rank,
                            before_op_output_shape_trans=True,
                        )
                    )
                conv_output_shape = [0] * input_tensor_rank
                for idx in range(len(graph_node_output.shape)):
                    conv_output_shape[idx] = graph_node_output.shape[converted_axis[idx]]
            elif pad_mode == "VALID":
                conv_output_shape = \
                input_tensor_split_shape[0] + \
                [
                    strides[i] * (input_tensor_split_spatial_shape[i] - 1) + input_weights_shape[i] \
                        for i in list(range(spatial_size))
                ] + \
                input_tensor_split_shape[-1]
            else:
                conv_output_shape = \
                input_tensor_split_shape[0] + \
                [
                    strides[i] * input_tensor_split_spatial_shape[i] \
                        for i in list(range(spatial_size))
                ] + \
                input_tensor_split_shape[-1]

            # use raw input x to do transposed conv
            conv_rs = conv_func(
                input=input_tensor_split,
                filters=weight_split,
                output_shape=conv_output_shape,
                strides=strides,
                padding=pad_mode,
            )
            convolved.append(conv_rs)

    if input_bias is not None:
        if len(convolved) == 1:
            # Conv1D_Transpose, Conv2D_Transpose, Conv3D_Transpose - Bias Add
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.add(
                    convolved[0],
                    input_bias,
                )
        else:
            # Conv1D_Transpose, Conv2D_Transpose, Conv3D_Transpose - Bias Add
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.add(
                    tf.concat(
                        values=convolved,
                        axis=-1
                    ),
                    input_bias,
                )
    else:
        if len(convolved) == 1:
            # Conv1D_Transpose, Conv2D_Transpose, Conv3D_Transpose - No Bias
            tf_layers_dict[graph_node_output.name]['tf_node'] = convolved[0]
        else:
            # Conv1D_Transpose, Conv2D_Transpose, Conv3D_Transpose - No Bias
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.concat(
                    values=convolved,
                    axis=-1
                )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': conv_func,
                'tf_inputs': {
                    'input': input_tensor_split,
                    'filters': weight_split,
                    'output_shape': conv_output_shape,
                    'strides': strides,
                    'padding': pad_mode,
                    'group': group,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
