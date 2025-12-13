import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import itertools
import tensorflow as tf
import tf_keras
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
    dummy_tf_inference,
    onnx_tf_tensor_validation,
    acquisition_of_validation_data,
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
    """Einsum

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans = True
    for graph_node_input in graph_node.inputs:
        before_op_output_shape_trans_n = \
            tf_layers_dict.get(graph_node_input.name, {}).get('before_op_output_shape_trans', True)
        before_op_output_shape_trans = \
            before_op_output_shape_trans and before_op_output_shape_trans_n

    values = []
    for graph_node_input in graph_node.inputs:
        graph_node_input_perm = get_constant_or_variable(
            graph_node_input,
            before_op_output_shape_trans,
        )
        input_tensor = tf_layers_dict[graph_node_input_perm.name]['tf_node'] \
            if isinstance(graph_node_input_perm, gs.Variable) else graph_node_input_perm
        values.append(input_tensor)
    graph_node_output: gs.Variable = graph_node.outputs[0]
    onnx_output_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    equation = graph_node.attrs['equation']
    onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = kwargs['onnx_tensor_infos_for_validation']
    if onnx_tensor_infos_for_validation is not None \
        and onnx_tensor_infos_for_validation.get(graph_node_output.name, None) is not None \
        and graph_node_output.name in onnx_tensor_infos_for_validation:
        onnx_output_shape = list(onnx_tensor_infos_for_validation[graph_node_output.name].shape)
        graph_node_output.shape = onnx_output_shape

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': onnx_output_shape,
        'dtype': dtype,
    }

    # Pre-process transpose
    new_values = []
    for graph_node_input, value in zip(graph_node.inputs, values):
        value = pre_process_transpose(
            value_before_transpose=value,
            param_target='inputs',
            param_name=graph_node_input.name,
            **kwargs,
        )
        new_values.append(value)
    values = new_values

    # Generation of TF OP
    if len(values) == 2 \
        and sum([1 if s is None else 0 for s in values[0].shape]) == 0 \
        and sum([1 if s is None else 0 for s in values[1].shape]) == 0 \
        and onnx_output_shape is not None:

        # Obtain ONNX inference results and
        # TensorFlow inference results up to the previous layer of TensorFlow
        input_tensor_1 = values[0]
        input_tensor_2 = values[1]
        onnx_tensor_infos, validation_data_1, validation_data_2 = \
            acquisition_of_validation_data(
                input_tensor_1=input_tensor_1,
                input_tensor_2=input_tensor_2,
                graph_node_output=graph_node_output,
                tf_layers_dict=tf_layers_dict,
                **kwargs,
            )

        min_abs_err = sys.maxsize
        min_abs_err_perm_1: int = [idx for idx in range(len(input_tensor_1.shape))]
        min_abs_err_perm_2: int = [idx for idx in range(len(input_tensor_2.shape))]

        def define_einsum(
            *,
            target_input_tensor_1: Any,
            target_perm_1: List,
            target_input_tensor_2: Any,
            target_perm_2: List,
            target_name: str,
            equation: str,
            **kwargs: Dict,
        ):
            return \
                tf.einsum(
                    equation,
                    *[
                        transpose_with_flexing_deterrence(
                            input_tensor=target_input_tensor_1 \
                                if not isinstance(target_input_tensor_1, np.ndarray) \
                                    else tf.convert_to_tensor(target_input_tensor_1),
                            perm=target_perm_1,
                            **kwargs,
                        ),
                        transpose_with_flexing_deterrence(
                            input_tensor=target_input_tensor_2 \
                                if not isinstance(target_input_tensor_2, np.ndarray) \
                                    else tf.convert_to_tensor(target_input_tensor_2),
                            perm=target_perm_2,
                            **kwargs,
                        ),
                    ],
                    name=target_name,
                )

        tensor_1_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_1.shape))))
        tensor_2_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_2.shape))))

        for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
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
                    input_2 = tf_keras.Input(
                        shape=validation_data_2.shape[1:],
                        batch_size=validation_data_2.shape[0] \
                            if isinstance(validation_data_2.shape[0], int) else None,
                        name='dummy_input_2',
                        dtype=validation_data_2.dtype,
                    )
                    dummy_concat = define_einsum(
                        target_input_tensor_1=input_1,
                        target_perm_1=list(tensor_1_candidate_for_transposition),
                        target_input_tensor_2=input_2,
                        target_perm_2=list(tensor_2_candidate_for_transposition),
                        target_name=graph_node.name,
                        equation=equation,
                        **kwargs
                    )
                    # Verify that the output shape matches that of ONNX
                    # If the combination of each value of a dimension is not correct,
                    # invalidate the normal processing judgment.
                    onnx_output_shape_prod = np.prod([dim if not isinstance(dim, str) else -1 for dim in onnx_output_shape])
                    concat_output_shapes = list(dummy_concat.shape)
                    concat_output_shape_prod = np.prod([dim if dim is not None else -1 for dim in concat_output_shapes])
                    if onnx_output_shape_prod != concat_output_shape_prod:
                        del input_1
                        del input_2
                        del dummy_concat
                        continue

                    # Perform simple accuracy verification
                    # Terminate when the error is less than 1e-3
                    if onnx_tensor_infos:
                        try:
                            # Search for the axis with the smallest error
                            val_model = tf_keras.Model(
                                inputs=[
                                    input_1,
                                    input_2,
                                ],
                                outputs=[
                                    dummy_concat,
                                ],
                            )

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
                                        validation_data_2,
                                    ],
                                )
                            del input_1
                            del input_2
                            del dummy_concat
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
                        except Exception as ex1:
                            pass
                except Exception as ex2:
                    pass
            else:
                continue
            break

        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            define_einsum(
                target_input_tensor_1=input_tensor_1,
                target_perm_1=min_abs_err_perm_1,
                target_input_tensor_2=input_tensor_2,
                target_perm_2=min_abs_err_perm_2,
                target_name=graph_node.name,
                equation=equation,
                **kwargs
            )
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.einsum(
                equation,
                *values,
                name=graph_node.name,
            )

    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        post_process_transpose(
            value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
            param_target='outputs',
            param_name=graph_node.outputs[0].name,
            **kwargs,
        )

    # Generation of Debug Info
    tf_inputs = {f"input{idx}": input for idx, input in enumerate(values)}
    tf_inputs['equation'] = equation
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.einsum,
                'tf_inputs': tf_inputs,
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
