import sys
import math
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
    """MaxPool

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

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    if len(graph_node.outputs) > 1:
        error_msg = f'{Color.RED}ERROR:{Color.RESET} ' \
                    f'MaxPoolWithArgmax is not yet implemented. ' \
                    f'Pull requests are welcome. \n' \
                    f'https://github.com/onnx/onnx-tensorflow/blob/f9ebc35dba8a9555112a8d0b84f5a3d51278cca9/onnx_tf/handlers/backend/dilated_pooling.py#L544 \n' \
                    f'graph_node.name: {graph_node.name}'
        print(error_msg)
        raise NotImplementedError(error_msg)

    filter = None

    auto_pad = graph_node.attrs.get('auto_pad', 'NOTSET')
    ceil_mode = bool(graph_node.attrs.get('ceil_mode', 0))
    kernel_shape = graph_node.attrs['kernel_shape']
    spatial_size = len(kernel_shape)
    dilations = graph_node.attrs.get('dilations', [1] * spatial_size)
    pads = graph_node.attrs.get('pads', [0] * spatial_size * 2)
    storage_order = graph_node.attrs.get('storage_order', 0)
    strides = graph_node.attrs.get('strides', [1] * spatial_size)

    input_tensor_shape = input_tensor.shape.as_list()
    is_known_shape = None not in input_tensor_shape[1:]
    input_tensor_dtype = input_tensor.dtype

    if storage_order:
        error_msg = f'{Color.RED}ERROR:{Color.RESET} ' + \
                    f'storage_order option is not implemented yet.'
        print(error_msg)
        raise NotImplementedError(error_msg)

    # default tensorflow action is 'SAME_UPPER' mode (extra padding in the end for odd numbers)
    # explicit pad layer is added for tensorflow incompatible cases
    tf_pad_mode = 'VALID'
    is_explicit_padding = False
    func = math.ceil if ceil_mode else math.floor
    dilated_kernel_shape = kernel_shape
    if dilations != [1] * spatial_size:
        dilated_kernel_shape = [(k - 1) * d for k, d in zip(kernel_shape, dilations)]

    tf_pads = calc_tf_pooling_pads(input_shape=input_tensor_shape,
                                   kernel=dilated_kernel_shape,
                                   strides=strides,
                                   func=func)

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
            tf_pads = pads

    elif auto_pad == 'SAME_UPPER':
        tf_pad_mode = 'SAME'

    elif auto_pad == 'SAME_LOWER':
        is_explicit_padding = True

    elif auto_pad == 'VALID':
        tf_pads = [0] * spatial_size * 2

    else:
        error_msg = f'{Color.RED}ERROR:{Color.RESET} ' + \
                    f'Wrong auto_pad parameter in MaxPool: {auto_pad}.'
        raise ValueError(error_msg)

    # add extra pad layer if needed
    if is_explicit_padding and tf_pads != [0] * spatial_size * 2:
        warning_msg = f'{Color.YELLOW}WARNING:{Color.RESET} ' \
                      f'Tensorflow incompatible padding detected. ' \
                      f'Extra pad layer is inserted automatically. '
        print(warning_msg)

        if auto_pad == 'SAME_LOWER':
            # switch the order of pads
            tf_pads = [i for tup in zip(tf_pads[len(tf_pads) // 2:], tf_pads[:len(tf_pads) // 2]) for i in tup]

        # convert to tensorflow padding format
        tf_pads = [[0, 0]] + \
                  [list(i) for i in zip(tf_pads[:len(tf_pads) // 2], tf_pads[len(tf_pads) // 2:])] + \
                  [[0, 0]]

        # explicit padding value should be negative infinite since this is max pooling
        padded_tensor = tf.pad(
            tensor=input_tensor,
            paddings=tf_pads,
            mode='CONSTANT',
            constant_values=-np.inf
        )

    else:
        padded_tensor = input_tensor

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    tf_op_type = None

    # tf.nn.dilation2d
    if spatial_size == 2 and dilations != [1] * spatial_size:
        strides = [1] + list(strides) + [1]
        dilations = [1] + list(dilations) + [1]

        # tf.nn.dilation2d only support data_format='NHWC'
        filter = tf.zeros(
            [kernel_shape[0], kernel_shape[1], input_tensor_shape[-1]],
            input_tensor_dtype,
        )
        pooled_tensor = tf.nn.dilation2d(
            input=padded_tensor,
            filters=filter,
            strides=strides,
            dilations=dilations,
            padding=tf_pad_mode.upper(),
            data_format="NHWC",
        )
        tf_op_type = tf.nn.dilation2d

    # if spatial_size < 4 and strides == 1 or dilation == 1 use tf.nn.pool
    elif spatial_size < 4 and (strides == [1] * spatial_size or dilations == [1] * spatial_size):
        # if strides == 1 and not LpPool use tf.nn.pool directly
        if strides == [1] * spatial_size:
            pooled_tensor = tf.nn.pool(
                input=padded_tensor,
                window_shape=kernel_shape,
                dilations=dilations,
                strides=strides,
                padding=tf_pad_mode.upper(),
                pooling_type='MAX',
            )
            tf_op_type = tf.nn.pool
        else:
            # otherwise check the pooling_type and use the correct op
            pooled_tensor = tf.nn.max_pool(
                input=padded_tensor,
                ksize=kernel_shape,
                strides=strides,
                padding=tf_pad_mode.upper(),
            )
            tf_op_type = tf.nn.max_pool
    # in any other case we use custom implementation _remove_dilations
    # to reduce atrous/dilated pooling into regular pooling and selecting
    # only the values of the input that should have been selected by
    # applying the strides and dilations. Then use tf.nn.pool with
    # strides = kernel_shape and no dilations
    else:
        # TODO: Dilated MaxPool with strides is broken for 3D and above, need to be fixed
        if spatial_size >= 3:
            error_msg = f'{Color.RED}ERROR:{Color.RESET} ' \
                        f'Dilated MaxPool with strides is not supported for 3D and above for now. '
            print(error_msg)
            raise NotImplementedError(error_msg)

        input_tensor = remove_dilations(
            input_tensor=padded_tensor,
            kernel_shape=kernel_shape,
            spatial_size=spatial_size,
            strides=strides,
            dilations=dilations,
        )
        tf_pad_mode = 'VALID'
        pooled_tensor = tf.nn.pool(
            input=input_tensor,
            window_shape=kernel_shape,
            strides=kernel_shape,
            padding=tf_pad_mode.upper(),
            pooling_type='MAX',
        )
        tf_op_type = tf.nn.pool

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
                    'input': input_tensor,
                    'filters': filter,
                    'kernel_shape': kernel_shape,
                    'strides': strides,
                    'dilations': dilations,
                    'padding': tf_pads if tf_pad_mode != 'same' else tf_pad_mode,
                    'ceil_mode': ceil_mode,
                },
                'tf_outputs': {
                    'output': pooled_tensor,
                },
            }
        )
