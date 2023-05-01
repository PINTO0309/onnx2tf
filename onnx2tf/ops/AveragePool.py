import math
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
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    calc_tf_pooling_pads,
    calc_extra_padding_with_ceil,
    transpose_with_flexing_deterrence,
)

INF_INDEX_VALUE: int = 4294967296


def summarize_multiplier(arr):
    """
    summarize consecutive numbers in average multiplier
    Parameters
    ----------
    arr: List
        average multiplier

    Returns
    -------
    summary: List
        summarized average multiplier
    """
    summary = []
    for sub_arr in arr:
        sub_summary = []
        i = 0
        while i < len(sub_arr):
            start_index = i
            value = sub_arr[i]
            while i < len(sub_arr) - 1 and sub_arr[i] == sub_arr[i+1]:
                i += 1
            end_index = i
            sub_summary.append((start_index, end_index, value))
            i += 1
        summary.append(sub_summary)
    return summary


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
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
    input_tensor_shape = input_tensor.shape

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Workaround to avoid as many conversion failures as possible
    # for models with useless Transpose immediately before them.
    # If the input geometry of the ONNX and the input geometry of the TF model match,
    # the input geometry on the TF model side is forcibly transposed to the NWC or NHWC or NDHWC format.
    # However, if all dimensions of CW or CHW or CDHW have the same value,
    # the forced transposition process is skipped because it may destroy the structure of the model.
    onnx_input_shape = [
        dim if isinstance(dim, int) else None for dim in graph_node.inputs[0].shape
    ] if graph_node.inputs[0].shape is not None else None
    tf_input_shape = [
        dim if isinstance(dim, int) else None for dim in input_tensor_shape
    ]
    if onnx_input_shape is not None \
        and len(onnx_input_shape) > 1 and len(tf_input_shape) > 1 \
        and onnx_input_shape == tf_input_shape:

        shape_for_judging_skip = [
            dim if dim is not None else INF_INDEX_VALUE for dim in onnx_input_shape[1:]
        ]
        if shape_for_judging_skip.count(shape_for_judging_skip[0]) != len(shape_for_judging_skip):
            if len(onnx_input_shape) == 3:
                # 1D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,1],
                    **kwargs,
                )
            elif len(onnx_input_shape) == 4:
                # 2D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,1],
                    **kwargs,
                )
            elif len(onnx_input_shape) == 5:
                # 3D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,4,1],
                    **kwargs,
                )

    auto_pad = graph_node.attrs.get('auto_pad', 'NOTSET')
    ceil_mode = bool(graph_node.attrs.get('ceil_mode', 0))
    count_include_pad = bool(graph_node.attrs.get('count_include_pad', 0))
    kernel_shape = graph_node.attrs['kernel_shape']
    spatial_size = len(kernel_shape)
    dilations = graph_node.attrs.get('dilations', [1] * spatial_size)
    pads = graph_node.attrs.get('pads', [0] * spatial_size * 2)
    strides = graph_node.attrs.get('strides', [1] * spatial_size)

    input_tensor_shape = input_tensor.shape.as_list()
    is_known_shape = None not in input_tensor_shape[1:]
    extra_pads = [0] * spatial_size
    average_multiplier = None

    # default tensorflow action is 'SAME_UPPER' mode (extra padding in the end for odd numbers)
    # explicit pad layer is added for tensorflow incompatible cases
    tf_pad_mode = 'VALID'
    is_explicit_padding = False
    tf_pads = calc_tf_pooling_pads(
        input_shape=input_tensor_shape,
        kernel=kernel_shape,
        strides=strides
    )

    func = math.ceil if ceil_mode else math.floor
    output_spatial_shape = [
        func((i + pb + pe - d * (k - 1) - 1) / s + 1)
        for i, pb, pe, k, d, s in zip(input_tensor_shape[1:-1], pads[:len(pads) // 2], pads[len(pads) // 2:], kernel_shape, dilations, strides)
    ]

    # onnx padding value is ignored if auto_pad is not 'NOTSET'
    if auto_pad == 'NOTSET':

        # check if onnx padding is same with tensorflow padding mode 'SAME'
        # this is to avoid flex operations since tflite builtin pooling operator does not support manual padding
        if is_known_shape and pads != [0] * spatial_size * 2 and tf_pads == pads:
            auto_pad = 'SAME_UPPER'
            tf_pad_mode = 'SAME'

        else:
            auto_pad = 'VALID'
            is_explicit_padding = True

            # extra padding to end side (right, bottom) may be needed when ceil_mode is True
            # this extra padding should not be counted as padding when count_include_pad is True
            if ceil_mode:
                extra_pads = \
                    calc_extra_padding_with_ceil(
                        input_shape=input_tensor_shape[1:-1],
                        kernel=kernel_shape,
                        pads=pads,
                        dilations=dilations,
                        strides=strides,
                    )
                pads = pads[:len(pads) // 2] + [p + e for p, e in zip(pads[len(pads) // 2:], extra_pads)]

            tf_pads = pads

    elif auto_pad == 'SAME_UPPER':
        tf_pad_mode = 'SAME'

    elif auto_pad == 'SAME_LOWER':
        is_explicit_padding = True

    elif auto_pad == 'VALID':
        tf_pads = [0] * spatial_size * 2

    else:
        error_msg = f'{Color.RED}ERROR:{Color.RESET} ' + \
                    f'Wrong auto_pad parameter in AveragePool: {auto_pad}.'
        raise ValueError(error_msg)

    # count nonzero elements in kernel each strides for the case count_include_pad is False
    non_zero_counts = []

    for input_spatial_shape, output_size, kernel, dilation, stride, pads_begin, pads_end \
            in zip(input_tensor_shape[1:-1], output_spatial_shape, kernel_shape,
                   dilations, strides, pads[:len(pads) // 2], pads[len(pads) // 2:]):
        sample_target = np.concatenate([
            np.zeros(pads_begin),
            np.ones(input_spatial_shape),
            np.zeros(pads_end)]
        )
        sample_kernel = np.zeros((kernel - 1) * dilation + 1)
        sample_kernel[::dilation] = 1

        counts = []
        for i in range(output_size):
            start = i * stride
            stride_target = sample_target[start:start+len(sample_kernel)]
            # pad target to match size
            stride_target = np.concatenate([stride_target, np.zeros(len(sample_kernel) - len(stride_target))])
            counts.extend(np.convolve(stride_target, sample_kernel, mode='valid'))

        non_zero_counts.append(counts)

    need_multiplier = len(set([i for sublist in non_zero_counts for i in sublist])) != 1

    # default tensorflow option for count_include_pad is True and cannot control
    # average value should be compensated in cases below
    # 1. when extra padding layer is added and count_include_pad is False
    # 2. when extra padding layer is not added and count_include_pad is True
    # 3. when last stride has extra padding due to ceil_mode and count_include_pad is True
    if is_explicit_padding and tf_pads != [0] * spatial_size * 2:
        warning_msg = \
            f'{Color.YELLOW}WARNING:{Color.RESET} ' \
            f'Tensorflow incompatible padding detected. ' \
            f'Extra pad layer is inserted automatically. '
        print(warning_msg)

        if auto_pad == 'SAME_LOWER':
            # switch the order of pads
            tf_pads = [i for tup in zip(tf_pads[len(tf_pads) // 2:], tf_pads[:len(tf_pads) // 2]) for i in tup]

        if not count_include_pad and need_multiplier:
            average_multiplier = []
            for k, non_zero_count in zip(kernel_shape, non_zero_counts):
                multiplier = [k / n if n != 0 else 1 for n in non_zero_count]
                average_multiplier.append(multiplier)

        # convert to tensorflow padding format
        tf_pads = \
            [[0, 0]] + \
            [list(i) for i in zip(tf_pads[:len(tf_pads) // 2], tf_pads[len(tf_pads) // 2:])] + \
            [[0, 0]]

        padded_tensor = tf.pad(
            tensor=input_tensor,
            paddings=tf_pads,
            mode='CONSTANT',
        )

    else:
        padded_tensor = input_tensor

        if count_include_pad and need_multiplier:
            average_multiplier = []
            for k, non_zero_count in zip(kernel_shape, non_zero_counts):
                multiplier = [n / k for n in non_zero_count]
                average_multiplier.append(multiplier)

    if count_include_pad and extra_pads != [0] * spatial_size:
        # extra padding in last stride should not be included in averaging
        if average_multiplier is None:
            average_multiplier = []
            for k, non_zero_count, extra_pad in zip(kernel_shape, non_zero_counts, extra_pads):
                multiplier = [1 for _ in non_zero_count]
                multiplier[-1] = k / (k - extra_pad)
                average_multiplier.append(multiplier)
        else:
            for i, k, non_zero_count, extra_pad in enumerate(zip(kernel_shape, non_zero_counts, extra_pads)):
                average_multiplier[i][-1] = k / (k - extra_pad)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # Generation of TF OP
    tf_op_type = None
    if len(kernel_shape) == 1:
        pooled_tensor = AveragePooling1D(
            pool_size=kernel_shape,
            strides=strides,
            padding=tf_pad_mode.upper(),
        )(padded_tensor)
        tf_op_type = AveragePooling1D

    elif len(kernel_shape) == 2:
        pooled_tensor = AveragePooling2D(
            pool_size=kernel_shape,
            strides=strides,
            padding=tf_pad_mode.upper(),
        )(padded_tensor)
        tf_op_type = AveragePooling2D

    elif len(kernel_shape) == 3:
        pooled_tensor = AveragePooling3D(
            pool_size=kernel_shape,
            strides=strides,
            padding=tf_pad_mode.upper(),
        )(padded_tensor)
        tf_op_type = AveragePooling3D

    else:
        error_msg = f'' +\
            f'{Color.RED}ERROR:{Color.RESET} ' +\
            f'AveragePool supports only 1D, 2D, and 3D. ' +\
            f'opname: {graph_node.name} Type: AveragePool{len(kernel_shape)}D'
        print(error_msg)
        raise AssertionError(error_msg)

    # tensorflow average pooling needs extra process to get same output with onnx
    # https://github.com/PINTO0309/onnx2tf/issues/124
    if average_multiplier is not None:
        warning_msg = \
            f'{Color.YELLOW}WARNING:{Color.RESET} ' \
            f'Tensorflow incompatible action detected. ' \
            f'Some additional layers are inserted to reproduce same output. ' \
            f'Please refer to the following link for more information: ' \
            f'https://github.com/PINTO0309/onnx2tf/issues/124'
        print(warning_msg)

        average_multiplier = summarize_multiplier(average_multiplier)

        for i, multiplier in enumerate(average_multiplier, start=1):
            slice_list = [slice(None) for _ in range(spatial_size * 2)]
            multiplied_slices = []

            for m in multiplier:
                start, stop, value = m
                slice_list[i] = slice(start, stop + 1)
                multiplied_slices.append(pooled_tensor[slice_list] * value)

            pooled_tensor = tf.concat(multiplied_slices, axis=i)

    tf_layers_dict[graph_node_output.name]['tf_node'] = pooled_tensor

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
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'x': input_tensor,
                    'pool_size': kernel_shape,
                    'strides': strides,
                    'padding': tf_pads if tf_pad_mode != 'same' else tf_pad_mode,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
