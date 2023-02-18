import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import itertools
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    get_constant_or_variable,
    get_weights_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    convert_reverse_axis,
    make_tf_node_info,
    calc_output_shape_conv_transpose,
    dummy_onnx_inference,
    transpose_with_flexing_deterrence,
)
from onnx2tf.utils.colors import Color


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
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
    output_padding = graph_node.attrs.get('output_padding', [0] * spatial_size)
    if graph_node_output_shape is None:
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

    # need to calculate output shape for valid mode
    disable_calc_output_shape_conv_transpose = len([
        dim for dim in graph_node_input_shape[2:] \
            if isinstance(dim, str) or dim is None
    ]) > 0
    # If the spartial shape is an undefined dimension, skip the process.
    # Instead, run inference with dummy input tensors in onnxruntime to try to estimate the output shape.
    tf_output_shape = None
    if not disable_calc_output_shape_conv_transpose:
        # The TensorFlow API is used to estimate the output shape.
        tf_output_shape = calc_output_shape_conv_transpose(
            input_shape=graph_node_input_shape[2:],
            kernel=kernel_shape,
            pad_mode='valid',
            output_padding=output_padding,
            stride=strides,
            dilation=dilations,
        )
        tf_output_shape = [graph_node_output_shape[0], *tf_output_shape, graph_node_output_shape[1]]
    else:
        # Perform inference using a dummy input tensor to attempt to estimate the output shape.
        try:
            import onnxruntime as ort
        except Exception as ex:
            print(
                f'{Color.RED}ERROR:{Color.RESET} ' +\
                f'The information needed to estimate the output shape of the ConvTranspose ' +\
                f'was missing and must be estimated using onnxruntime. ' +\
                f'Install onnxruntime. pip install onnxruntime'
            )
            sys.exit(1)
        onnx_graph = kwargs['onnx_graph']
        convtranspose_output = dummy_onnx_inference(
            onnx_graph=onnx_graph,
            output_names=[graph_node_output.name],
        )[0]
        onnx_output_shape = list(convtranspose_output.shape)
        tf_output_shape = []
        for idx in range(input_tensor_rank):
            tf_output_shape.append(onnx_output_shape[converted_axis[idx]])

    if auto_pad == 'NOTSET':
        # pad_mode SAME generates flex operation, use VALID always
        pad_mode = 'VALID'
    elif auto_pad == "SAME_UPPER":
        # TODO: this may generates flex operation, need to check
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

        if input_bias is not None:
            bias_splits = [input_bias]
    else:
        input_tensor_splits = tf.split(input_tensor, num_or_size_splits=group, axis=-1)
        weight_splits = tf.split(input_weights, num_or_size_splits=group, axis=-1)

        if input_bias is not None:
            bias_splits = tf.split(input_bias, num_or_size_splits=group, axis=-1)

    # Param replacement - OP replacement
    op_rep_params = kwargs.get('op_rep_params', [])
    output_shape_ = None
    for op_rep_param in op_rep_params:
        if op_rep_param['param_target'] == 'op':
            output_shape_ = op_rep_param.get('output_shape', None)
            strides_ = op_rep_param.get('strides', None)
            padding_ = op_rep_param.get('padding', 'SAME')
            dilations_ = op_rep_param.get('dilations', None)

            if output_shape_ is None or strides_ is None:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} ' +
                    f'When replacing ConvTranspose OP, "filters", "output_shape" and "strides" must be specified in replace.json. ' +
                    f'Check the specification of tf.nn.convXd_transpose in TensorFlow and specify the appropriate parameters. ' +
                    f'https://www.tensorflow.org/api_docs/python/tf/nn/conv1d_transpose or ' +
                    f'https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose or ' +
                    f'https://www.tensorflow.org/api_docs/python/tf/nn/conv3d_transpose'
                )
                sys.exit(1)

    conv_rs = None
    convolved = []
    for i, (input_tensor_split, weight_split) in enumerate(zip(input_tensor_splits, weight_splits)):
        split_conv_output_shape = tf_output_shape[:-1] + [weight_split.shape[spatial_size]]
        if output_shape_ is None:
            # Normal ConvTranspose
            try:
                conv_rs = conv_func(
                    input=input_tensor_split,
                    filters=weight_split,
                    output_shape=split_conv_output_shape,
                    strides=strides,
                    padding=pad_mode,
                    dilations=dilations,
                )
            except Exception as ex1:
                # Shape Unmatch Error Mitigation Measures
                # Search for and transpose shapes that do not cause shape unmatch errors
                tensor_1_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_split.shape))))
                tensor_2_candidate_for_transpositions = list(itertools.permutations(range(len(weight_split.shape))))
                for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
                    for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
                        try:
                            conv_rs = conv_func(
                                input=transpose_with_flexing_deterrence(
                                    input_tensor=input_tensor_split,
                                    perm=tensor_1_candidate_for_transposition,
                                    **kwargs,
                                ),
                                filters=transpose_with_flexing_deterrence(
                                    input_tensor=weight_split,
                                    perm=tensor_2_candidate_for_transposition,
                                    **kwargs,
                                ),
                                output_shape=split_conv_output_shape,
                                strides=strides,
                                padding=pad_mode,
                                dilations=dilations,
                            )
                            break
                        except Exception as ex2:
                            pass
                    else:
                        continue
                    break
                if conv_rs is None:
                    raise ex1

        else:
            # OP replacement
            conv_rs = conv_func(
                input=input_tensor_splits,
                filters=weight_splits,
                output_shape=output_shape_,
                strides=strides_,
                padding=padding_,
                dilations=dilations_,
            )

        # add split bias to combined convolution for 1d and 2d
        if input_bias is not None and spatial_size != 3:
            conv_rs = tf.add(conv_rs, bias_splits[i])

        convolved.append(conv_rs)

    if group > 1:
        # concatenate in case of grouped convolution
        conv_rs = tf.concat(values=convolved, axis=-1)

        # add bias after concat for 3d since Conv3dTranspose in tensorflow does not support bias in layer level
        if input_bias is not None and spatial_size == 3:
            conv_rs = tf.add(conv_rs, input_bias)

    if pad_mode == "VALID":
        # TODO: this part may not work properly when OP replacement is used, need to check
        # remove pads
        begin = [0] + pads[:spatial_size] + [0]

        # add slice if needed
        if max(begin) > 0:
            conv_rs = tf.slice(conv_rs, begin=begin, size=conv_output_shape)

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
