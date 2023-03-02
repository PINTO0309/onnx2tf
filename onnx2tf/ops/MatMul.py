import random
random.seed(0)
import numpy as np
np.random.seed(0)
import itertools
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    make_tf_partial_model_inputs,
    dummy_tf_inference,
    transpose_with_flexing_deterrence,
)
from typing import Any, Dict, List
from onnx2tf.utils.enums import (
    NUMPY_DTYPES_TO_TF_DTYPES,
    TF_DTYPES_TO_NUMPY_DTYPES,
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
    """MatMul

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

    # Generation of TF OP
    input_tensor_1 = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_tensor_2 = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

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

    output_dtype = NUMPY_DTYPES_TO_TF_DTYPES[dtype] \
        if isinstance(dtype, np.dtype) else dtype

    try:
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
        ### Overall model
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.matmul(
                a=input_tensor_1 \
                    if not isinstance(input_tensor_1, np.ndarray) \
                        else tf.convert_to_tensor(input_tensor_1),
                b=input_tensor_2 \
                    if not isinstance(input_tensor_2, np.ndarray) \
                        else tf.convert_to_tensor(input_tensor_2),
                output_type=output_dtype,
                name=graph_node.name,
            )
        ### Partial model
        if kwargs['acc_check'] and tf_partial_model_inputs is not None:
            tf_partial_model_outputs = \
                [
                    tf.matmul(
                        a=tf_partial_model_inputs[0],
                        b=tf_partial_model_inputs[1],
                        output_type=output_dtype,
                    )
                ]
            tf_partial_model = tf.keras.Model(
                inputs=tf_partial_model_inputs,
                outputs=tf_partial_model_outputs,
            )
            test_data1 = None
            if not isinstance(input_tensor_1, np.ndarray):
                if not isinstance(graph_node_input_1, np.ndarray) \
                    and graph_node_input_1.name in tf_layers_dict \
                    and 'verification_data' in tf_layers_dict[graph_node_input_1.name].keys():
                    test_data1: np.ndarray = tf_layers_dict[graph_node_input_1.name]['verification_data']
                elif isinstance(graph_node_input_1, np.ndarray):
                    test_data1: np.ndarray = graph_node_input_1
                else:
                    test_data1 = None
            else:
                test_data1 = input_tensor_1
            test_data2 = None
            if not isinstance(input_tensor_2, np.ndarray):
                if not isinstance(graph_node_input_2, np.ndarray) \
                    and graph_node_input_2.name in tf_layers_dict \
                    and 'verification_data' in tf_layers_dict[graph_node_input_2.name].keys():
                    test_data2: np.ndarray = tf_layers_dict[graph_node_input_2.name]['verification_data']
                elif isinstance(graph_node_input_2, np.ndarray):
                    test_data2: np.ndarray = graph_node_input_2
                else:
                    test_data2 = None
            else:
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

    except Exception as ex1:
        # Shape Unmatch Error Mitigation Measures
        # Search for and transpose shapes that do not cause shape unmatch errors
        tensor_1_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_1.shape))))
        tensor_2_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_2.shape))))
        for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
            for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
                try:
                    # Generate input OPs for TensorFlow subgraphs
                    # For inference testing on OP stand-alone
                    if kwargs['acc_check']:
                        tf_partial_model_inputs: List[tf.keras.Input] = \
                                make_tf_partial_model_inputs(
                                    input_tensors=[
                                        np.zeros(
                                            list(input_tensor_1.shape),
                                            dtype=input_tensor_1.dtype \
                                                if isinstance(input_tensor_1, np.ndarray) \
                                                    else TF_DTYPES_TO_NUMPY_DTYPES[input_tensor_1.dtype],
                                        ).transpose(tensor_1_candidate_for_transposition),
                                        np.zeros(
                                            list(input_tensor_2.shape),
                                            dtype=input_tensor_2.dtype \
                                                if isinstance(input_tensor_2, np.ndarray) \
                                                    else TF_DTYPES_TO_NUMPY_DTYPES[input_tensor_2.dtype],
                                        ).transpose(tensor_2_candidate_for_transposition),
                                    ]
                                )
                        tf_partial_model_outputs = None
                    ### Overall model
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.matmul(
                            a=transpose_with_flexing_deterrence(
                                input_tensor=input_tensor_1 \
                                    if not isinstance(input_tensor_1, np.ndarray) \
                                        else tf.convert_to_tensor(input_tensor_1),
                                perm=list(tensor_1_candidate_for_transposition),
                                **kwargs,
                            ),
                            b=transpose_with_flexing_deterrence(
                                input_tensor=input_tensor_2 \
                                    if not isinstance(input_tensor_2, np.ndarray) \
                                        else tf.convert_to_tensor(input_tensor_2),
                                perm=list(tensor_2_candidate_for_transposition),
                                **kwargs,
                            ),
                            output_type=output_dtype,
                            name=graph_node.name,
                        )
                    ### Partial model
                    if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                        tf_partial_model_outputs = \
                            [
                                tf.matmul(
                                    a=tf_partial_model_inputs[0] \
                                        if not isinstance(tf_partial_model_inputs[0], np.ndarray) \
                                            else tf.convert_to_tensor(tf_partial_model_inputs[0]),
                                    b=tf_partial_model_inputs[1] \
                                        if not isinstance(tf_partial_model_inputs[1], np.ndarray) \
                                            else tf.convert_to_tensor(tf_partial_model_inputs[1]),
                                    output_type=output_dtype,
                                )
                            ]
                        tf_partial_model = tf.keras.Model(
                            inputs=tf_partial_model_inputs,
                            outputs=tf_partial_model_outputs,
                        )
                        test_data1 = None
                        if not isinstance(input_tensor_1, np.ndarray):
                            if not isinstance(graph_node_input_1, np.ndarray) \
                                and graph_node_input_1.name in tf_layers_dict \
                                and 'verification_data' in tf_layers_dict[graph_node_input_1.name].keys():
                                test_data1: np.ndarray = transpose_with_flexing_deterrence(
                                    input_tensor=tf_layers_dict[graph_node_input_1.name]['verification_data'],
                                    perm=list(tensor_1_candidate_for_transposition),
                                    **kwargs,
                                )
                            elif isinstance(graph_node_input_1, np.ndarray):
                                test_data1: np.ndarray = transpose_with_flexing_deterrence(
                                    input_tensor=graph_node_input_1,
                                    perm=list(tensor_1_candidate_for_transposition),
                                    **kwargs,
                                )
                            else:
                                test_data1 = None
                        else:
                            test_data1 = transpose_with_flexing_deterrence(
                                input_tensor=input_tensor_1,
                                perm=list(tensor_1_candidate_for_transposition),
                                **kwargs,
                            )
                        test_data2 = None
                        if not isinstance(input_tensor_2, np.ndarray):
                            if not isinstance(graph_node_input_2, np.ndarray) \
                                and graph_node_input_2.name in tf_layers_dict \
                                and 'verification_data' in tf_layers_dict[graph_node_input_2.name].keys():
                                test_data2: np.ndarray = transpose_with_flexing_deterrence(
                                    input_tensor=tf_layers_dict[graph_node_input_2.name]['verification_data'],
                                    perm=list(tensor_2_candidate_for_transposition),
                                    **kwargs,
                                )
                            elif isinstance(graph_node_input_2, np.ndarray):
                                test_data2: np.ndarray = transpose_with_flexing_deterrence(
                                    input_tensor=graph_node_input_2,
                                    perm=list(tensor_2_candidate_for_transposition),
                                    **kwargs,
                                )
                            else:
                                test_data2 = None
                        else:
                            test_data2 = transpose_with_flexing_deterrence(
                                input_tensor=input_tensor_2,
                                perm=list(tensor_2_candidate_for_transposition),
                                **kwargs,
                            )
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
                    break
                except Exception as ex2:
                    pass
            else:
                continue
            break
        if 'tf_node' not in tf_layers_dict[graph_node_output.name]:
            raise ex1

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
                'tf_op_type': tf.matmul,
                'tf_inputs': {
                    'a': input_tensor_1,
                    'b': input_tensor_2,
                    'output_type': dtype,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
