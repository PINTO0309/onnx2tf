import copy
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    pre_process_transpose,
    post_process_transpose,
)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Unsqueeze

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

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = None
    if len(graph_node.inputs) >= 2:
        graph_node_input_2 = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1

    axes = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    if axes is not None and axes.shape is None:
        axes = None
    axes = graph_node.attrs.get('axes', axes)

    nhwc: bool = tf_layers_dict[graph_node_input_1.name]['nhwc'] \
        if isinstance(graph_node_input_1, gs.Variable) \
            and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False

    if input_tensor.shape != tf.TensorShape(None):
        input_tensor_shape = list(input_tensor.shape)
        tensor_rank = len(input_tensor_shape)
    elif graph_node_output.shape is not None:
        input_tensor_shape = [
            dim for idx, dim in enumerate(graph_node_output.shape) if idx not in axes
        ]
        input_tensor_shape = [
            dim if not isinstance(dim, str) else None for dim in input_tensor_shape
        ]
        tensor_rank = len(input_tensor_shape)

    if isinstance(axes, list) or (isinstance(axes, np.ndarray) and len(axes.shape) > 0):
        if nhwc:
            axes = [
                convert_axis(
                    axis=idx,
                    tensor_rank=tensor_rank+len(axes),
                    before_op_output_shape_trans=True,
                ) for idx in axes
            ]
        elif not nhwc and (isinstance(axes, list) and len(axes) == 1 or isinstance(axes, np.ndarray) and len(axes.shape) == 1) and axes[0] == -1:
            axes = [
                convert_axis(
                    axis=idx,
                    tensor_rank=tensor_rank+len(axes),
                    before_op_output_shape_trans=before_op_output_shape_trans,
                ) for idx in axes
            ]
        else:
            axes = [idx for idx in axes]
    elif axes is not None and isinstance(axes, np.ndarray) and len(axes.shape) == 0:
        if nhwc:
            axes = convert_axis(
                axis=axes,
                tensor_rank=tensor_rank+1,
                before_op_output_shape_trans=True,
            )
        elif not nhwc and (isinstance(axes, list) and len(axes) == 1 or isinstance(axes, np.ndarray) and len(axes.shape) == 1) and axes[0] == -1:
            axes = [
                convert_axis(
                    axis=idx,
                    tensor_rank=tensor_rank+len(axes),
                    before_op_output_shape_trans=before_op_output_shape_trans,
                ) for idx in axes
            ]
        axes = list(axes[np.newaxis])

    if axes is not None and isinstance(axes, list) and len(axes) > 0:
        axes.sort()

    new_shape = copy.deepcopy(input_tensor_shape)
    for idx in axes:
        new_shape.insert(idx, 1)

    new_shape = [dim if dim is not None else -1 for dim in new_shape]

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input_1.name]['nhwc'] \
            if nhwc and len(axes) == 1 and tensor_rank is not None and axes not in [0, -1, tensor_rank] else False
    }

    # Param replacement - OP replacement
    shape_replaced = False
    op_rep_params = kwargs.get('op_rep_params', [])
    for op_rep_param in op_rep_params:
        if op_rep_param['param_target'] == 'op':
            new_shape = op_rep_param.get('new_shape', None)
            shape_replaced = True

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#unsqueeze-13
    """
    [2,3,4,5,6,7]
    test pattern.1 : axes=0, [1,2,3,4,5,6,7]
    test pattern.2 : axes=1, [2,1,3,4,5,6,7]
    test pattern.3 : axes=5, [2,3,4,5,6,1,7]
    test pattern.4 : axes=6, [2,3,4,5,6,7,1]
    test pattern.5 : axes=[0,1], [1,1,2,3,4,5,6,7]
    test pattern.6 : axes=[1,4], [2,1,3,4,1,5,6,7]
    test pattern.7 : axes=[6,7], [2,3,4,5,6,7,1,1]
    test pattern.8 : axes=[3,6], [2,3,4,1,5,6,1,7]
    test pattern.9 : axes=[3,-1], [2,3,4,1,5,6,1,7]
    """
    tf_type = None
    if isinstance(graph_node_input_1, gs.Variable) \
        and 'simple_resize' in tf_layers_dict[graph_node_input_1.name] \
        and tf_layers_dict[graph_node_input_1.name]['simple_resize'] == True:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.identity(input=input_tensor)
        tf_layers_dict[graph_node_output.name]['simple_resize'] = True
        tf_layers_dict[graph_node_output.name]['simple_resize_shape_op'] = tf_layers_dict[graph_node_input_1.name]['simple_resize_shape_op']
        tf_type = tf.identity
    elif \
        (
            isinstance(graph_node_input_1, gs.Variable) \
            and 'unnecessary_squeeze' in tf_layers_dict[graph_node_input_1.name] \
            and tf_layers_dict[graph_node_input_1.name]['unnecessary_squeeze'] == True
        ) or \
        (
            isinstance(graph_node_input_2, gs.Variable) \
            and 'unnecessary_squeeze' in tf_layers_dict[graph_node_input_2.name] \
            and tf_layers_dict[graph_node_input_2.name]['unnecessary_squeeze'] == True
        ):
        # Remove useless squeeze/unsqueeze combinations
        #   Only when squeeze and unsqueeze are consecutive
        #   and each is performing a useless process of
        #   compressing and decompressing the same axis,
        #   the two operations are disabled at the same time.
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.identity(input=input_tensor)
        tf_type = tf.identity

    elif \
        (
            isinstance(graph_node_input_1, gs.Variable) \
            and 'unnecessary_gather' in tf_layers_dict[graph_node_input_1.name] \
            and tf_layers_dict[graph_node_input_1.name]['unnecessary_gather'] == True
        ) or \
        (
            isinstance(graph_node_input_2, gs.Variable) \
            and 'unnecessary_gather' in tf_layers_dict[graph_node_input_2.name] \
            and tf_layers_dict[graph_node_input_2.name]['unnecessary_gather'] == True
        ):
        # Remove useless gather/unsqueeze combinations
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.identity(input=input_tensor)
        tf_type = tf.identity

    elif \
        (
            isinstance(graph_node_input_1, gs.Variable) \
            and 'unnecessary_reshape' in tf_layers_dict[graph_node_input_1.name] \
            and tf_layers_dict[graph_node_input_1.name]['unnecessary_reshape'] == True
        ) or \
        (
            isinstance(graph_node_input_2, gs.Variable) \
            and 'unnecessary_reshape' in tf_layers_dict[graph_node_input_2.name] \
            and tf_layers_dict[graph_node_input_2.name]['unnecessary_reshape'] == True
        ):
        # Remove useless reshape/unsqueeze combinations
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.identity(input=input_tensor)
        tf_type = tf.identity

    elif not shape_replaced \
        and nhwc \
        and len(axes) == 1 \
        and not isinstance(axes, int):
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.expand_dims(
                input=input_tensor,
                axis=axes[0],
                name=graph_node.name,
            )
        tf_type = tf.expand_dims

    elif not shape_replaced \
        and len(new_shape) >= 2 \
        and len([dim for dim in new_shape if dim is None or dim == -1]) >= 2 \
        and not isinstance(axes, int) \
        and len(axes) == 1:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.expand_dims(
                input=input_tensor,
                axis=axes[0],
                name=graph_node.name,
            )
        tf_type = tf.expand_dims

    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.reshape(
                tensor=input_tensor,
                shape=new_shape,
                name=graph_node.name,
            )
        tf_type = tf.reshape

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
                'tf_op_type': tf_type,
                'tf_inputs': {
                    'tensor': input_tensor,
                    'shape': new_shape,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
