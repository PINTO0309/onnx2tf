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
    """Reshape

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
    tensor_rank = len(input_tensor.shape)

    reshape_shape = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    if reshape_shape is not None and reshape_shape.shape is None:
        reshape_shape = []
    if reshape_shape is None:
        reshape_shape = []

    # Ignore
    allowzero = bool(graph_node.attrs.get('allowzero', 0))

    # If Reshape's shape contains zeros, get the deformed shape from the output shape
    if isinstance(reshape_shape, list) and reshape_shape.count(0) > 0:
        new_shape = [-1 if isinstance(s, str) else int(s) for s in shape]
        reshape_shape = new_shape
    elif isinstance(reshape_shape, np.ndarray) and np.count_nonzero(reshape_shape == 0) > 0:
        new_shape = [-1 if isinstance(s, str) else int(s) for s in shape]
        reshape_shape = new_shape

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    # NWC->NCW, NHWC->NCHW, NDHWC->NCDHW Transpose
    perm = [
        convert_axis(
            axis=idx,
            tensor_rank=tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        ) for idx in range(tensor_rank)
    ]
    transposed_tensor = tf.transpose(
        a=input_tensor,
        perm=list(perm) if perm is not None else None,
    )
    if isinstance(reshape_shape, np.ndarray):
        perm_shape = [
            convert_axis(
                axis=idx,
                tensor_rank=len(reshape_shape),
                before_op_output_shape_trans=before_op_output_shape_trans,
            ) for idx in range(len(reshape_shape))
        ]
        transposed_reshape_shape = list(reshape_shape)
        if before_op_output_shape_trans:
            transposed_reshape_shape = [
                transposed_reshape_shape[perm_shape_dim] for perm_shape_dim in list(perm_shape)
            ]
    else:
        transposed_reshape_shape = reshape_shape

    # Param replacement
    transposed_tensor = replace_parameter(
        value_before_replacement=transposed_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    transposed_reshape_shape = replace_parameter(
        value_before_replacement=transposed_reshape_shape,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )

    # Pre-process transpose
    transposed_tensor = pre_process_transpose(
        value_before_transpose=transposed_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Reshape
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.reshape(
            tensor=transposed_tensor,
            shape=transposed_reshape_shape,
            name=graph_node.name,
        )

    # Reshape of the GroupNormalization (InstanceNormalization) process,
    # if applicable, is re-transposed to NHWC.
    if 'is_instance_norm_3d' in tf_layers_dict[graph_node_input_1.name] \
        and tf_layers_dict[graph_node_input_1.name]['is_instance_norm_3d'] == True:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.transpose(
                a=tf_layers_dict[graph_node_output.name]['tf_node'],
                perm=[0,2,3,1],
            )
    else:
        # Special support for ShuffleNet patterns
        # 5D Reshape -> 5D Transpose -> 4D Reshape
        # 1,2,72,16,16 -> 1,72,2,16,16 -> 1,144,16,16
        # At this time, only the channel shuffling pattern of image processing is supported.
        try:
            two_previous_op: gs.Node = graph_node.i().i()
            two_previous_op_type = two_previous_op.op
            one_previous_op: gs.Node = graph_node.i()
            one_previous_op_type = one_previous_op.op
            if two_previous_op_type == 'Reshape' and one_previous_op_type == 'Transpose':
                two_previous_op_output_shape = two_previous_op.outputs[0].shape
                one_previous_op_output_shape = one_previous_op.outputs[0].shape
                two_previous_op_output_rank = len(two_previous_op_output_shape)
                one_previous_op_output_rank = len(one_previous_op_output_shape)
                current_op_output_shape = graph_node.outputs[0].shape
                current_op_output_rank = len(current_op_output_shape)
                if two_previous_op_output_rank == 5 \
                    and one_previous_op_output_rank == 5 \
                    and current_op_output_rank == 4 \
                    and two_previous_op_output_shape[1] == one_previous_op_output_shape[2] \
                    and two_previous_op_output_shape[2] == one_previous_op_output_shape[1] \
                    and current_op_output_shape[1] == (one_previous_op_output_shape[1] * one_previous_op_output_shape[2]):
                    # ShuffleNet patterns - 4D only
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.transpose(
                            a=tf_layers_dict[graph_node_output.name]['tf_node'],
                            perm=[0,2,3,1],
                        )
                else:
                    pass
        except:
            pass


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
                    'tensor': transposed_tensor,
                    'shape': transposed_reshape_shape,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
