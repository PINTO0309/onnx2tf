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
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    pre_explicit_broadcast,
    explicit_broadcast,
    pre_process_transpose,
    post_process_transpose,
    disable_unnecessary_transpose,
    shape_unmatched_special_avoidance_workaround,
    make_tf_partial_model_inputs,
    dummy_tf_inference,
)
from typing import Any, Dict, List


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Sub

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
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    input_tensor_1 = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_tensor_2 = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    # Acquisition of test data for validation
    if kwargs['acc_check']:
        if not isinstance(graph_node_input_1, np.ndarray) \
            and graph_node_input_1.name in tf_layers_dict \
            and 'verification_data' in tf_layers_dict[graph_node_input_1.name].keys():
            test_data1: np.ndarray = tf_layers_dict[graph_node_input_1.name]['verification_data']
        elif isinstance(graph_node_input_1, np.ndarray):
            test_data1: np.ndarray = graph_node_input_1
        else:
            test_data1 = None
        if not isinstance(graph_node_input_2, np.ndarray) \
            and graph_node_input_2.name in tf_layers_dict \
            and 'verification_data' in tf_layers_dict[graph_node_input_2.name].keys():
            test_data2: np.ndarray = tf_layers_dict[graph_node_input_2.name]['verification_data']
        elif isinstance(graph_node_input_2, np.ndarray):
            test_data2: np.ndarray = graph_node_input_2
        else:
            test_data2 = None

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

    # Generate input OPs for TensorFlow subgraphs
    # For inference testing on OP stand-alone
    if kwargs['acc_check']:
        tf_partial_model_inputs: List[tf.keras.Input] = \
            make_tf_partial_model_inputs(
                input_tensors=[
                    input_tensor_1,
                    input_tensor_2,
                ]
            )
        tf_partial_model_outputs = None

    # Generation of TF OP
    ### Overall model
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.math.subtract(
            x=input_tensor_1,
            y=input_tensor_2,
            name=graph_node.name,
        )
    ### Partial model
    if kwargs['acc_check'] and tf_partial_model_inputs is not None:
        tf_partial_model_outputs = \
            [
                tf.math.subtract(
                    x=tf_partial_model_inputs[0],
                    y=tf_partial_model_inputs[1],
                )
            ]
        tf_partial_model = tf.keras.Model(
            inputs=tf_partial_model_inputs,
            outputs=tf_partial_model_outputs,
        )
        test_data1 = None
        if isinstance(input_tensor_1, np.ndarray):
            test_data1 = input_tensor_1
        test_data2 = None
        if isinstance(input_tensor_2, np.ndarray):
            test_data2 = input_tensor_2
        tf_partial_model_result_infos: Dict[Any] = dummy_tf_inference(
            model=tf_partial_model,
            inputs=tf_partial_model_inputs,
            verification_datas=[
                test_data1,
                test_data2,
            ]
        )
        tf_layers_dict[graph_node_output.name]['verification_data'] = \
            list(tf_partial_model_result_infos.values())[0]
        del tf_partial_model
        del tf_partial_model_inputs
        del tf_partial_model_outputs
        del test_data1
        del test_data2

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
                'tf_op_type': tf.math.subtract,
                'tf_inputs': {
                    'x': input_tensor_1,
                    'y': input_tensor_2,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
