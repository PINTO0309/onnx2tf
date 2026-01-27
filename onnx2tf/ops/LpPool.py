import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    remove_dilations,
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
from onnx2tf.utils.logging import *

INF_INDEX_VALUE: int = 4294967296


def _kernel_size_const(kernel_shape, dtype):
    if isinstance(kernel_shape, (list, tuple, np.ndarray)):
        size = 1
        for k in kernel_shape:
            size *= int(k)
        return tf.cast(size, dtype)
    if tf.is_tensor(kernel_shape):
        return tf.cast(tf.reduce_prod(kernel_shape), dtype)
    return tf.cast(int(kernel_shape), dtype)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """LpPool

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
    input_tensor_rank = len(input_tensor_shape)

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
    kernel_shape = graph_node.attrs['kernel_shape']
    spatial_size = len(kernel_shape)
    dilations = graph_node.attrs.get('dilations', [1] * spatial_size)
    pads = graph_node.attrs.get('pads', [0] * spatial_size * 2)
    strides = graph_node.attrs.get('strides', [1] * spatial_size)
    p = float(graph_node.attrs.get('p', 2))

    input_tensor_shape = input_tensor.shape.as_list()
    is_known_shape = None not in input_tensor_shape[1:]

    # default tensorflow action is 'SAME_UPPER' mode (extra padding in the end for odd numbers)
    # explicit pad layer is added for tensorflow incompatible cases
    tf_pad_mode = 'VALID'
    is_explicit_padding = False
    dilated_kernel_shape = kernel_shape
    if dilations != [1] * spatial_size:
        dilated_kernel_shape = [(k - 1) * d for k, d in zip(kernel_shape, dilations)]

    tf_pads = calc_tf_pooling_pads(
        input_shape=input_tensor_shape,
        kernel=dilated_kernel_shape,
        strides=strides,
        input_tensor=input_tensor,
    )

    # onnx padding value is ignored if auto_pad is not 'NOTSET'
    if auto_pad == 'NOTSET':

        # check if onnx padding is same with tensorflow padding mode 'SAME'
        # this is to avoid flex operations since tflite has no builtin pooling with manual padding value
        if is_known_shape and pads != [0] * spatial_size * 2 and tf_pads == pads:
            auto_pad = 'SAME_UPPER'
            tf_pad_mode = 'SAME'

        else:
            auto_pad = 'VALID'
            is_explicit_padding = True

            # extra padding may be needed for ceiling
            # this padding is added to end side (right, bottom) only
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
        error_msg = Color.RED(f'ERROR:') + ' ' + \
                    f'Wrong auto_pad parameter in LpPool: {auto_pad}.'
        raise ValueError(error_msg)

    # add extra pad layer if needed
    if is_explicit_padding and tf_pads != [0] * spatial_size * 2:
        warn(
            f'Tensorflow incompatible padding detected. ' \
            f'Extra pad layer is inserted automatically. '
        )

        if auto_pad == 'SAME_LOWER':
            # switch the order of pads
            tf_pads = [i for tup in zip(tf_pads[len(tf_pads) // 2:], tf_pads[:len(tf_pads) // 2]) for i in tup]

        # convert to tensorflow padding format
        tf_pads = \
            [[0, 0]] + \
            [list(i) for i in zip(tf_pads[:len(tf_pads) // 2], tf_pads[len(tf_pads) // 2:])] + \
            [[0, 0]]

        padded_tensor = tf.pad(
            tensor=input_tensor,
            paddings=tf_pads,
            mode='CONSTANT',
            constant_values=0.0,
        )

    else:
        padded_tensor = input_tensor

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # Generation of TF OP
    tf_op_type = None
    abs_p_tensor = tf.pow(tf.abs(padded_tensor), p) if p != 1.0 else tf.abs(padded_tensor)
    kernel_size_const = _kernel_size_const(kernel_shape, abs_p_tensor.dtype)

    if spatial_size < 4 and (strides == [1] * spatial_size or dilations == [1] * spatial_size):
        pooled_tensor = tf.nn.pool(
            input=abs_p_tensor,
            window_shape=kernel_shape,
            dilations=dilations,
            strides=strides,
            padding=tf_pad_mode.upper(),
            pooling_type='AVG',
        )
        tf_op_type = tf.nn.pool
    else:
        # TODO: Dilated LpPool with strides is broken for 3D and above, need to be fixed
        if spatial_size >= 3:
            error_msg = Color.RED(f'ERROR:') + ' ' \
                        f'Dilated LpPool with strides is not supported for 3D and above for now. '
            print(error_msg)
            raise NotImplementedError(error_msg)

        abs_p_tensor = remove_dilations(
            input_tensor=abs_p_tensor,
            kernel_shape=kernel_shape,
            spatial_size=spatial_size,
            strides=strides,
            dilations=dilations,
        )
        pooled_tensor = tf.nn.pool(
            input=abs_p_tensor,
            window_shape=kernel_shape,
            strides=kernel_shape,
            padding='VALID',
            pooling_type='AVG',
        )
        tf_op_type = tf.nn.pool

    pooled_tensor = pooled_tensor * kernel_size_const
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.pow(pooled_tensor, 1.0 / p) if p != 1.0 else pooled_tensor

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
                    'input': input_tensor,
                    'kernel_shape': kernel_shape,
                    'strides': strides,
                    'dilations': dilations,
                    'padding': tf_pads if tf_pad_mode != 'same' else tf_pad_mode,
                    'ceil_mode': ceil_mode,
                    'p': p,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
