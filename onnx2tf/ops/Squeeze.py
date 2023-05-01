import copy
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    replace_parameter,
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
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
    """Squeeze

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
    input_tensor_shape = input_tensor.shape
    tensor_rank = len(input_tensor_shape) \
        if input_tensor_shape != tf.TensorShape(None) else 1

    axes = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    if axes is not None and axes.shape is None:
        axes = None

    axes = graph_node.attrs.get('axes', axes)
    # axes for determining the deletion of unnecessary squeeze/unsqueeze combinations
    non_transpose_axes = copy.deepcopy(axes)

    if isinstance(axes, list) or (isinstance(axes, np.ndarray) and len(axes.shape) > 0):
        axes = [
            convert_axis(
                axis=idx,
                tensor_rank=tensor_rank,
                before_op_output_shape_trans=before_op_output_shape_trans,
            ) for idx in axes
        ]
    elif axes is not None and isinstance(axes, np.ndarray) and len(axes.shape) == 0:
        axes = convert_axis(
            axis=axes,
            tensor_rank=tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Param replacement
    axes = replace_parameter(
        value_before_replacement=axes,
        param_target='attributes',
        param_name='axes',
        **kwargs,
    )
    if len(graph_node.inputs) >= 2:
        axes = replace_parameter(
            value_before_replacement=axes,
            param_target='inputs',
            param_name=graph_node.inputs[1].name,
            **kwargs,
        )

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    axes = list(axes) if axes is not None else None
    tf_type = None

    if \
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
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.identity(input=input_tensor)
        tf_type = tf.identity
    else:
        try:
            workaround_exec_flg = False
            try:
                graph_node.o(consumer_idx=1)
            except Exception as ex:
                # Error == Only one node connected next
                workaround_exec_flg = True
            if workaround_exec_flg \
                and graph_node.o().op == 'Unsqueeze' \
                and ((hasattr(graph_node.o(), 'attrs') and 'axes' in graph_node.o().attrs) or len(graph_node.o().inputs) >= 2):
                # Remove useless squeeze/unsqueeze combinations
                #   Only when squeeze and unsqueeze are consecutive
                #   and each is performing a useless process of
                #   compressing and decompressing the same axis,
                #   the two operations are disabled at the same time.
                next_unsqueeze_node = graph_node.o()
                next_node_axes = None
                if len(next_unsqueeze_node.inputs) >= 2:
                    next_node_axes = get_constant_or_variable(
                        next_unsqueeze_node.inputs[1],
                        before_op_output_shape_trans,
                    )
                    next_node_axes = tf_layers_dict[next_node_axes.name]['tf_node'] \
                        if isinstance(next_node_axes, gs.Variable) else next_node_axes
                    if next_node_axes is not None and next_node_axes.shape is None:
                        next_node_axes = None
                next_unsqueezed_axes = next_node_axes \
                    if next_node_axes is not None else next_unsqueeze_node.attrs['axes']
                if next_unsqueezed_axes == non_transpose_axes:
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.identity(input=input_tensor)
                    tf_layers_dict[graph_node_output.name]['unnecessary_squeeze'] = True
                    tf_type = tf.identity
                else:
                    # Normal squeeze
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.squeeze(
                            input=input_tensor,
                            axis=axes,
                            name=graph_node.name,
                        )
                    tf_type = tf.squeeze
            else:
                # Normal squeeze
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.squeeze(
                        input=input_tensor,
                        axis=axes,
                        name=graph_node.name,
                    )
                tf_type = tf.squeeze
        except Exception as ex:
            # Normal squeeze
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.squeeze(
                    input=input_tensor,
                    axis=axes,
                    name=graph_node.name,
                )
            tf_type = tf.squeeze

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
                    'input': input_tensor,
                    'axis': axes,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
