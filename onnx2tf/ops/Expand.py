import sys
import copy
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import itertools
import tensorflow as tf
import tf_keras
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
    get_tf_model_inputs,
    dummy_tf_inference,
    onnx_tf_tensor_validation,
)
from typing import List, Dict, Any


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Expand

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

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_tensor_rank = len(input_tensor.shape)
    input_tensor_shape = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = kwargs['onnx_tensor_infos_for_validation']
    test_data_nhwc: np.ndarray = kwargs['test_data_nhwc']
    custom_input_op_name_np_data_path: str = kwargs['custom_input_op_name_np_data_path']
    disable_strict_mode: bool = kwargs['disable_strict_mode']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP

    # Param replacement
    input_tensor = replace_parameter(
        value_before_replacement=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    input_tensor_shape = replace_parameter(
        value_before_replacement=input_tensor_shape,
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
        # tf.math.multiply does not support bool therefore use int32
        expanded_tensor = None
        if input_tensor.dtype is tf.bool:
            ones = tf.ones(input_tensor_shape, dtype=tf.int32)
            r = tf.cast(input_tensor, tf.int32) * ones
            expanded_tensor = tf.cast(r, tf.bool)
        else:
            ones = tf.ones(input_tensor_shape, dtype=input_tensor.dtype)
            expanded_tensor = input_tensor * ones
        tf_layers_dict[graph_node_output.name]['tf_node'] = expanded_tensor
        tf_type = 'Expand'

    if tf_type == 'Expand':
        graph_node_input_1_shape = graph_node_input_1.shape
        graph_node_input_2_shape = graph_node_input_2.shape

        # Get the output tensor of one previous OP of TensorFlow only once
        if not disable_strict_mode:
            tf_model_inputs = get_tf_model_inputs(
                tf_layers_dict=tf_layers_dict,
            )
            val_model = None
            if not isinstance(input_tensor, np.ndarray):
                expand_shape = []
                if not isinstance(input_tensor_shape, np.ndarray):
                    expand_shape = [input_tensor_shape]
                val_model = tf_keras.Model(
                    inputs=tf_model_inputs,
                    outputs=[
                        input_tensor,
                    ] + expand_shape,
                )
            else:
                pass

        # TF dummy inference
        #   Get the output tensor of the previous layer of MatMul
        #   If input.1 and input.2 are both layers, tf_pre_tensor_infos is 2 cases
        #   If one of input.1 or input.2 is np.ndarray, tf_pre_tensor_infos is 1 case
        tf_pre_tensor_infos = {}
        if not disable_strict_mode:
            try:
                tf_pre_tensor_infos: Dict[Any] = \
                    dummy_tf_inference(
                        model=val_model,
                        inputs=tf_model_inputs,
                        test_data_nhwc=test_data_nhwc,
                        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                    )
            except Exception as ex:
                pass
            del val_model

        # Get np.ndarray for validation
        validation_data_1 = None
        validation_data_2 = None

        if not disable_strict_mode:
            if len(tf_pre_tensor_infos) == 1:
                if not isinstance(input_tensor, np.ndarray):
                    validation_data_1 = list(tf_pre_tensor_infos.values())[0]
                else:
                    validation_data_1 = copy.deepcopy(input_tensor)
            elif len(tf_pre_tensor_infos) == 2:
                if not isinstance(input_tensor, np.ndarray):
                    validation_data_1 = list(tf_pre_tensor_infos.values())[0]
                else:
                    validation_data_1 = copy.deepcopy(input_tensor)
                if not isinstance(input_tensor_shape, np.ndarray):
                    validation_data_2 = list(tf_pre_tensor_infos.values())[1]
                else:
                    validation_data_2 = copy.deepcopy(input_tensor_shape)

            # Get ONNX inference results
            onnx_tensor_infos = None
            if onnx_tensor_infos_for_validation is not None \
                and onnx_tensor_infos_for_validation.get(graph_node_output.name, None) is not None:
                onnx_tensor_infos = {
                    graph_node_output.name: onnx_tensor_infos_for_validation[graph_node_output.name]
                }
                del onnx_tensor_infos_for_validation

        # ONNX   : N,C,W
        # TF     : N,W,C
        # TF-axes: [1]
        #
        # ONNX: N,C,H,W
        # TF  : N,H,W,C
        # TF-axes: [1,2]
        #
        # ONNX: N,C,D,H,W
        # TF  : N,D,H,W,C
        # TF-axes: [1,2,3]

        # Automatic correction of accuracy degradation
        min_abs_err = sys.maxsize
        min_abs_err_perm_1: List[int] = [idx for idx in range(input_tensor_rank)]
        min_abs_err_perm_2: List[int] = [idx for idx, val in enumerate(input_tensor_shape)]

        if not disable_strict_mode:
            if onnx_tensor_infos is not None and validation_data_1 is not None and validation_data_2 is not None:
                tensor_1_candidate_for_transpositions = list(itertools.permutations(range(input_tensor_rank)))
                tensor_2_candidate_for_transpositions = list(itertools.permutations(range(len(min_abs_err_perm_2))))
                # Search for the axis with the smallest error
                for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
                    try:
                        for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
                            try:
                                # Build TF dummy model
                                input_1 = tf_keras.Input(
                                    shape=validation_data_1.shape[1:],
                                    batch_size=validation_data_1.shape[0] \
                                        if isinstance(validation_data_1.shape[0], int) else None,
                                    name='dummy_input_1',
                                    dtype=validation_data_1.dtype,
                                )
                                expand_shape = [validation_data_2[pos] for pos in tensor_2_candidate_for_transposition]
                                input_2 = tf_keras.Input(
                                    shape=[len(expand_shape)],
                                    batch_size=1,
                                    name='dummy_input_2',
                                    dtype=validation_data_2.dtype,
                                )
                                a=0

                                ones = tf.ones(input_2[0], dtype=input_tensor.dtype)
                                expanded_tensor = input_1 * ones
                                a=0

                                val_model = tf_keras.Model(
                                    inputs=[
                                        input_1,
                                        input_2,
                                    ],
                                    outputs=[
                                        expanded_tensor,
                                    ],
                                )
                                a=0
                                # TF dummy inference
                                tf_tensor_infos: Dict[Any] = \
                                    dummy_tf_inference(
                                        model=val_model,
                                        inputs=[
                                            input_1,
                                            input_2,
                                        ],
                                        verification_datas=[
                                            validation_data_1,
                                            tf.convert_to_tensor([expand_shape], dtype=tf.int64) if isinstance(expand_shape, list) else tf.expand_dims(expand_shape, axis=0),
                                        ],
                                    )
                                del input_1
                                del input_2
                                del val_model

                                # Validation
                                onnx_tf_output_pairs = {
                                    (oi[0], ti[0]): (oi[1], ti[1]) \
                                        for oi, ti in zip(onnx_tensor_infos.items(), tf_tensor_infos.items())
                                }
                                """
                                check_results: Dict[str, List[np.ndarray, int, float|int]]
                                    {
                                        onnx_output_name: [
                                            onnx_tensor,
                                            matched_flg, <--- 0: Unmatched, 1: Matched, 2: Skipped (Deleted or Shape Unmatched)
                                            max_abs_err,
                                        ]
                                    }
                                """
                                check_results = \
                                    onnx_tf_tensor_validation(
                                        output_pairs=onnx_tf_output_pairs,
                                        rtol=0.0,
                                        atol=0.0,
                                    )
                                result_err = sum([val[2] for val in check_results.values()])
                                if result_err < min_abs_err:
                                    min_abs_err = result_err
                                    min_abs_err_perm_1 = list(tensor_1_candidate_for_transposition)
                                    min_abs_err_perm_2 = list(tensor_2_candidate_for_transposition)
                                    if min_abs_err < 1e-3:
                                        break
                            except Exception as ex:
                                pass
                    except Exception as ex:
                        pass

                input_tensor = \
                    transpose_with_flexing_deterrence(
                        input_tensor=input_tensor,
                        perm=min_abs_err_perm_1,
                        output_shape=input_tensor_shape \
                            if None not in input_tensor.shape and input_tensor.shape != [] else None,
                        **kwargs,
                    )
                input_tensor_shape = [input_tensor_shape[pos] for pos in min_abs_err_perm_2]
                ones = tf.ones(input_tensor_shape, dtype=input_tensor.dtype)
                expanded_tensor = input_tensor * ones

                tf_layers_dict[graph_node_output.name]['tf_node'] = expanded_tensor
                tf_type = tf.expand_dims

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
                    'input_tensor': input_tensor,
                    'input_tensor_shape': input_tensor_shape,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
