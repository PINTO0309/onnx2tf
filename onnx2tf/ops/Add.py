import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    replace_parameter,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    pre_explicit_broadcast,
    explicit_broadcast,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    disable_unnecessary_transpose,
    shape_unmatched_special_avoidance_workaround,
    merge_two_consecutive_identical_ops_into_one,
    deterring_shape_corruption_due_to_broadcast,
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
    """Add

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
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    graph_node_output_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': graph_node_output_shape,
        'dtype': dtype,
    }

    input_tensor_1 = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_tensor_2 = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    # Param replacement
    input_tensor_1 = replace_parameter(
        value_before_replacement=input_tensor_1,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    input_tensor_2 = replace_parameter(
        value_before_replacement=input_tensor_2,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )

    # Pre-process transpose
    input_tensor_1 = pre_process_transpose(
        value_before_transpose=input_tensor_1,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    input_tensor_2 = pre_process_transpose(
        value_before_transpose=input_tensor_2,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )

    # Disable unnecessary Transpose
    #   1. If both x and y are gs.Variable
    #   2. If only one of the two is the output of Transpose
    #   3. If the perm of the Transpose is [0,2,1] or [0,3,1,2] or [0,4,1,2,3]
    #   4. Furthermore, if the shape of x and y are mismatched
    graph_node_input_1, graph_node_input_2, input_tensor_1, input_tensor_2 = \
        disable_unnecessary_transpose(
            graph_node_input_1=graph_node_input_1,
            graph_node_input_2=graph_node_input_2,
            input_tensor_1=input_tensor_1,
            input_tensor_2=input_tensor_2,
            **kwargs,
        )

    # Shape Unmatched Special Avoidance Workaround
    # At least one True value for same_input_shape_as_onnx
    # At least one True value in nhwc_flags
    # same_input_shape_as_onnx == True and nhwc_flags == False and 3D or 4D or 5D tensor is NHWC transposed
    input_tensor_1, input_tensor_2 = shape_unmatched_special_avoidance_workaround(
        graph_node_input_1=graph_node_input_1,
        graph_node_input_2=graph_node_input_2,
        input_tensor_1=input_tensor_1,
        input_tensor_2=input_tensor_2,
        tf_layers_dict=tf_layers_dict,
        **kwargs,
    )

    input_tensor_1, input_tensor_2 = pre_explicit_broadcast(
        input_tensor_1=input_tensor_1,
        input_tensor_2=input_tensor_2,
    )

    input_tensor_1, input_tensor_2 = explicit_broadcast(
        const_or_var_1=input_tensor_1,
        const_or_var_2=input_tensor_2,
        graph_node=graph_node,
        tf_layers_dict= tf_layers_dict,
    )

    # Deterring shape corruption due to broadcast
    input_tensor_1, input_tensor_2 = \
        deterring_shape_corruption_due_to_broadcast(
            graph_node_output_shape=graph_node_output_shape,
            input_tensor_1=input_tensor_1,
            input_tensor_2=input_tensor_2,
        )

    # Generation of TF OP
    # Merge two consecutive identical OPs into one
    # https://github.com/PINTO0309/onnx2tf/issues/230
    #   A constant is calculated in advance only
    #   when one of the operations of the current OP
    #   is a constant and one of the operations of
    #   the next OP is also a constant.
    # By merging two OPs into one, an accuracy error always occurs
    # in the merged OP during the accuracy check.
    # 1. `Mul` -> `Mul` to `Single-Mul` : `10 * 5 * 8 -> 10 * 40`
    # 2. `Mul` -> `Div` to `Single-Mul` : `10 * 5 / 8 -> 10 * 0.625`
    # 3. `Div` -> `Mul` to `Single-Mul` : `10 / 5 * 8 -> 10 * 1.6`
    # 4. `Div` -> `Div` to `Single-Mul` : `10 / 5 / 8 -> 10 * 0.025`
    # 5. `Sub` -> `Sub` to `Single-Sub` : `10 - 5 - 8 -> 10 - 13`
    # 6. `Sub` -> `Add` to `Single-Sub` : `10 - 5 + 8 -> 10 + 3`
    # 7. `Add` -> `Add` to `Single-Add`  : `10 + 5 + 8 -> 10 + 13`
    # 8. `Add` -> `Sub` to `Single-Add`  : `10 + 5 - 8 -> 10 - 3`
    _, tf_type = merge_two_consecutive_identical_ops_into_one(
        graph_node_input_1=graph_node_input_1,
        graph_node_input_2=graph_node_input_2,
        graph_node_output=graph_node_output,
        before_op_output_shape_trans=before_op_output_shape_trans,
        input_tensor_1=input_tensor_1,
        input_tensor_2=input_tensor_2,
        graph_node=graph_node,
        tf_layers_dict=tf_layers_dict,
        tf_func='Add'
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
                'tf_op_type': tf_type,
                'tf_inputs': {
                    'x': input_tensor_1,
                    'y': input_tensor_2,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
