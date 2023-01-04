import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
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
    """Flatten

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

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    output_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    axis = graph_node.attrs.get("axis", 0)
    if graph_node_input.shape is not None:
        axis = convert_axis(
            axis=axis,
            tensor_rank=len(graph_node_input.shape),
            before_op_output_shape_trans=before_op_output_shape_trans,
        )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': output_shape,
        'dtype': dtype,
    }

    # Param replacement
    axis = replace_parameter(
        value_before_replacement=axis,
        param_target='attributes',
        param_name='axis',
        **kwargs,
    )

    # Generation of TF OP
    cal_shape = None
    if axis == 0:
        cal_shape = (1, -1)
    elif graph_node_output.shape is not None and len(graph_node_output.shape) == 2 and axis == input_tensor_rank - 1:
        cal_shape = (1, -1)
    elif input_tensor_rank >= 2 \
        and input_tensor_shape[0] is None \
        and len([idx for idx in input_tensor_shape[1:] if idx is not None]) == input_tensor_rank - 1 \
        and axis == 1:
        cal_shape = (-1, np.prod(input_tensor_shape[1:]))
    elif input_tensor_rank >= 2 \
        and input_tensor_shape[0] is None \
        and len([idx for idx in input_tensor_shape[1:] if idx is not None]) != input_tensor_rank - 1 \
        and axis == 1:
        # Use Keras Flatten() if there are two or more undefined dimensions
        cal_shape = None
    else:
        cal_shape = (
            tf.reduce_prod(input_tensor_shape[0:axis]),
            tf.reduce_prod(input_tensor_shape[axis:tf.size(input_tensor_shape)]),
        )

    # If the output geometry is clear, overwrite with ONNX output geometry
    has_undefined_outputshape = output_shape is None
    if not has_undefined_outputshape:
        has_none_outputshape = None in output_shape
        has_str_outputshape = True in [True for dim in output_shape if isinstance(dim, str)]
        has_undefined_outputshape = has_none_outputshape or has_str_outputshape
    cal_shape = cal_shape if has_undefined_outputshape else output_shape

    # Param replacement
    input_tensor = replace_parameter(
        value_before_replacement=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    perm = [
        convert_axis(
            axis=idx,
            tensor_rank=input_tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        ) for idx in range(input_tensor_rank)
    ]
    input_tensor = tf.transpose(
        a=input_tensor,
        perm=list(perm) if perm is not None else None,
    )

    if cal_shape is not None:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.reshape(
                tensor=input_tensor,
                shape=cal_shape,
                name=graph_node.name,
            )
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.keras.layers.Flatten()(input_tensor)

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
                'tf_op_type': tf.reshape,
                'tf_inputs': {
                    'tensor': input_tensor,
                    'shape': cal_shape,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
