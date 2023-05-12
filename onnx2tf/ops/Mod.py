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
    disable_unnecessary_transpose,
    shape_unmatched_special_avoidance_workaround,
    pre_explicit_broadcast,
    explicit_broadcast,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    deterring_shape_corruption_due_to_broadcast,
    correction_process_for_accuracy_errors,
    nhwc_determination_of_output_value_of_binary_input_op,
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
    """Mod

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
        'nhwc': \
            nhwc_determination_of_output_value_of_binary_input_op(
                graph_node_input_1=graph_node_input_1,
                graph_node_input_2=graph_node_input_2,
                tf_layers_dict=tf_layers_dict
            )
    }

    # Generation of TF OP
    input_tensor_1 = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_tensor_2 = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    disable_strict_mode: bool = kwargs['disable_strict_mode']

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

    # Correction process for accuracy errors
    if not disable_strict_mode:
        input_tensor_1, input_tensor_2 = correction_process_for_accuracy_errors(
            input_tensor_1=input_tensor_1,
            input_tensor_2=input_tensor_2,
            tf_func=tf.math.mod,
            np_func=np.mod,
            graph_node_output_shape=graph_node_output_shape,
            graph_node_output=graph_node_output,
            tf_layers_dict=tf_layers_dict,
            **kwargs,
        )

    # Generation of TF OP
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.math.mod(
            x=input_tensor_1 \
                if not isinstance(input_tensor_1, np.ndarray) \
                    else tf.convert_to_tensor(input_tensor_1),
            y=input_tensor_2 \
                if not isinstance(input_tensor_2, np.ndarray) \
                    else tf.convert_to_tensor(input_tensor_2),
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
                'tf_op_type': tf.math.mod,
                'tf_inputs': {
                    'x': input_tensor_1,
                    'y': input_tensor_2,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
