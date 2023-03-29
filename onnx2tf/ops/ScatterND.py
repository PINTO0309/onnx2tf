import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    process_neg_idx,
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
    """ScatterND

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_2 = \
        tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_3 = \
        tf_layers_dict.get(graph_node.inputs[2].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2 \
        and before_op_output_shape_trans_3

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    indices_tensor = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    updates_tensor = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    # Inverted workaround to avoid shape errors as much as possible
    if not before_op_output_shape_trans \
        and list(graph_node.inputs[0].shape) != list(input_tensor.shape) \
        and list(graph_node.inputs[1].shape) == list(indices_tensor.shape) \
        and list(graph_node.inputs[2].shape) == list(updates_tensor.shape):

        input_tensor_shape = input_tensor.shape
        input_tensor_rank = len(input_tensor_shape)
        indices_tensor_shape = indices_tensor.shape
        updates_tensor_shape = updates_tensor.shape
        updates_tensor_rank = len(updates_tensor_shape)
        # Obtaining the number of ranks to be operated
        number_of_ranks_operate = indices_tensor_shape[-1]
        # 1. The number of ranks to be operated must be numeric
        # 2. The number of ranks to be operated on must match the number of ranks in the input tensor
        # 3. The number of ranks to be operated on must match the number of ranks in the tensor for updating
        # 4. Forced NHWC conversion if all the above conditions are met
        if isinstance(number_of_ranks_operate, int) \
            and input_tensor_rank == number_of_ranks_operate \
            and updates_tensor_rank == number_of_ranks_operate:

            indices_convertion_table = [0] + [i for i in range(2, input_tensor_rank)] + [1, input_tensor_rank]
            indices_gather_table = [0] + [i for i in range(2, input_tensor_rank)] + [1]
            updates_convertion_table = [0] + [i for i in range(2, input_tensor_rank)] + [1]
            # Corrects tensor shape discrepancies
            if isinstance(indices_tensor, np.ndarray):
                indices_tensor = indices_tensor.transpose(indices_convertion_table)
            elif tf.keras.backend.is_keras_tensor(indices_tensor):
                indices_tensor = tf.transpose(
                    a=indices_tensor,
                    perm=indices_convertion_table,
                )
            if isinstance(updates_tensor, np.ndarray):
                updates_tensor = updates_tensor.transpose(updates_convertion_table)
            elif tf.keras.backend.is_keras_tensor(updates_tensor):
                updates_tensor = tf.transpose(
                    a=updates_tensor,
                    perm=updates_convertion_table,
                )
            # Transposition of indices
            if isinstance(indices_tensor, np.ndarray):
                indices_tensor = indices_tensor[..., indices_gather_table]
            elif tf.keras.backend.is_keras_tensor(indices_tensor):
                indices_tensor = tf.gather(
                    params=indices_tensor,
                    indices=indices_gather_table,
                    axis=-1,
                )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    indices_tensor = pre_process_transpose(
        value_before_transpose=indices_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    updates_tensor = pre_process_transpose(
        value_before_transpose=updates_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[2].name,
        **kwargs,
    )

    # Complex ScatterND -> Simple ScatterND
    simple_scatternd = False
    # Verify if negative numbers need to be converted to positive numbers
    if isinstance(indices_tensor, np.ndarray) and None not in indices_tensor:
        flatten_indices_tensor = indices_tensor.flatten()
        if np.sum(np.where(flatten_indices_tensor < 0, 1, 0)) > 0:
            simple_scatternd = False
        else:
            simple_scatternd = True
    elif hasattr(indices_tensor, 'numpy') and None not in indices_tensor.numpy():
        flatten_indices_tensor = indices_tensor.numpy().flatten()
        if np.sum(np.where(flatten_indices_tensor < 0, 1, 0)) > 0:
            simple_scatternd = False
        else:
            simple_scatternd = True
    elif isinstance(indices_tensor, int) and indices_tensor >= 0:
        simple_scatternd = True
    else:
        simple_scatternd = False

    # Generation of TF OP
    if not simple_scatternd:
        indices_tensor = process_neg_idx(
            data=input_tensor,
            indices=indices_tensor,
        )

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.tensor_scatter_nd_update(
            tensor=input_tensor \
                if not isinstance(input_tensor, np.ndarray) \
                    else tf.convert_to_tensor(input_tensor),
            indices=indices_tensor \
                if not isinstance(indices_tensor, np.ndarray) \
                    else tf.convert_to_tensor(indices_tensor),
            updates=updates_tensor \
                if not isinstance(updates_tensor, np.ndarray) \
                    else tf.convert_to_tensor(updates_tensor),
            name=graph_node.name,
        )

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
                'tf_op_type': tf.tensor_scatter_nd_update,
                'tf_inputs': {
                    'tensor': input_tensor,
                    'indices': indices_tensor,
                    'updates': updates_tensor,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
