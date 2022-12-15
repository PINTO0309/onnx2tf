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
    graph_node_input: gs.Variable = graph_node.inputs[0]
    graph_node_output: gs.Variable = graph_node.outputs[0]

    before_op_output_shape_trans = \
        tf_layers_dict.get(graph_node_input.name, {}).get('before_op_output_shape_trans', True)

    # ONNX activation input
    input_tensor = get_constant_or_variable(
        graph_node_input,
        before_op_output_shape_trans,
    )
    input_tensor = tf_layers_dict[input_tensor.name]['tf_node'] \
        if isinstance(input_tensor, gs.Variable) else input_tensor
    input_tensor_shape = input_tensor.shape
    graph_node_input_shape = graph_node_input.shape
    input_tensor_rank = len(input_tensor_shape)
    spatial_size = input_tensor_rank - 2

    # ONNX weight input
    kernel_shape = graph_node.attrs.get('kernel_shape', [])
    input_weights = get_weights_constant_or_variable(
        const_or_var=graph_node.inputs[1],
        kernel_size=len(kernel_shape),
    )
    input_weights = tf_layers_dict[input_weights.name]['tf_node'] \
        if isinstance(input_weights, gs.Variable) else input_weights

    # ONNX bias input
    input_bias = None
    if len(graph_node.inputs) >= 3:
        input_bias = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
            is_bias=True,
        )
        input_bias = tf_layers_dict[input_bias.name]['tf_node'] \
            if isinstance(input_bias, gs.Variable) else input_bias

    pads = graph_node.attrs.get('pads', [0, 0] * spatial_size)
    strides = graph_node.attrs.get('strides', [1] * spatial_size)
    dilations = graph_node.attrs.get('dilations', [1] * spatial_size)

    # get ONNX convolution output shape
    graph_node_output_shape = graph_node.attrs.get('output_shape', graph_node_output.shape)
    if graph_node_output_shape is None:
        output_padding = graph_node.attrs.get('output_padding', [0] * spatial_size)
        graph_node_output_shape = [graph_node_input_shape[0]] + [graph_node.inputs[1].shape[0]] + \
            [ (strides[i] * (graph_node_input_shape[i+2] - 1) + dilations[i] * (kernel_shape[i] - 1) + \
              1 + output_padding[i] - pads[2*i] - pads[2*i+1]) for i in range(spatial_size)]

    # convert ONNX convolution output shape to TF convolution output shape
    converted_axis = []
    for idx in range(input_tensor_rank):
        converted_axis.append(
            convert_reverse_axis(
                axis=idx,
                tensor_rank=input_tensor_rank,
                before_op_output_shape_trans=True,
            )
        )
    conv_output_shape = []
    for idx in range(input_tensor_rank):
        conv_output_shape.append(graph_node_output_shape[converted_axis[idx]])

    # Generation of TF OP
    # select TF padding mode
    auto_pad = graph_node.attrs.get('auto_pad', 'NOTSET')
    pad_mode = 'VALID'
    if auto_pad == 'NOTSET':
        if graph_node_input_shape[2:] == graph_node_output_shape[2:]:
            pad_mode = "SAME"
        else:
            # TODO: check for valid explicit pads.
            pad_mode = 'VALID'
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

   # get corresponding function in TF
    if spatial_size == 1:
        conv_func = tf.nn.conv1d_transpose
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

    # deal with grouped convolution (TF-Lite does not support grouped transposed convolution)
    group = graph_node.attrs.get('group', 1)
    if group == 1:
        input_tensor_splits = [input_tensor]
        weight_splits = [input_weights]
    else:
        input_tensor_splits = tf.split(input_tensor, num_or_size_splits=group, axis=-1)
        weight_splits = tf.split(input_weights, num_or_size_splits=group, axis=-1)

    convolved = []
    for (input_tensor_split, weight_split) in zip(input_tensor_splits, weight_splits):
        split_conv_output_shape = conv_output_shape[:-1] + [weight_split.shape[spatial_size]]

        conv_rs = conv_func(
            input=input_tensor_split,
            filters=weight_split,
            output_shape=split_conv_output_shape,
            strides=strides,
            dilations=dilations,
            padding=pad_mode,
        )
        convolved.append(conv_rs)

    if group > 1:
        # concatenate in case of grouped convolution 
        conv_rs = tf.concat(values=convolved, axis=-1)
    if input_bias is not None:
        # add bias to combined convolution
        conv_rs = tf.add(conv_rs, input_bias)
    
    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': graph_node_output_shape,
        'dtype': graph_node_output.dtype,
        'nhwc': True,
        'tf_node': conv_rs,
    }
    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': conv_func,
                'tf_inputs': {
                    'input': input_tensor,
                    'filters': input_weights,
                    'output_shape': conv_output_shape,
                    'strides': strides,
                    'dilations': dilations,
                    'padding': pad_mode,
                    'group': group,
                    'bias': input_bias,
                },
                'tf_outputs': {
                    'output': conv_rs,
                },
            }
        )
