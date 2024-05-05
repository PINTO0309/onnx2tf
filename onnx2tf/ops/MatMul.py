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
from onnx2tf.utils.enums import (
    NUMPY_DTYPES_TO_TF_DTYPES,
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
    onnx_output_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    disable_strict_mode: bool = kwargs['disable_strict_mode']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': onnx_output_shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    input_tensor_1 = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_tensor_2 = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    # Response to the case where the first tensor is 1-dimensional
    # Response to the case where the second tensor is 1-dimensional
    # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    input_tensor_1_is_one_d = False
    input_tensor_2_is_one_d = False
    if input_tensor_1.shape is not None \
        and len(input_tensor_1.shape) == 1:
        input_tensor_1 = tf.expand_dims(input_tensor_2, axis=0)
        input_tensor_1_is_one_d = True
    elif input_tensor_2.shape is not None \
            and len(input_tensor_2.shape) == 1:
            input_tensor_2 = tf.expand_dims(input_tensor_2, axis=-1)
            input_tensor_2_is_one_d = True

    # Obtain ONNX inference results and
    # TensorFlow inference results up to the previous layer of TensorFlow
    onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = kwargs['onnx_tensor_infos_for_validation']
    onnx_tensor_infos = None
    validation_data_1 = None
    validation_data_2 = None
    if onnx_tensor_infos_for_validation is not None \
        and onnx_tensor_infos_for_validation.get(graph_node_output.name, None) is not None:
        onnx_tensor_infos, validation_data_1, validation_data_2 = \
            acquisition_of_validation_data(
                input_tensor_1=input_tensor_1,
                input_tensor_2=input_tensor_2,
                graph_node_output=graph_node_output,
                tf_layers_dict=tf_layers_dict,
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

    output_dtype = NUMPY_DTYPES_TO_TF_DTYPES[dtype] \
        if isinstance(dtype, np.dtype) else dtype

    # Shape Unmatch Error Mitigation Measures
    # Search for and transpose shapes that do not cause shape unmatch errors
    min_abs_err = sys.maxsize
    min_abs_err_perm_1: List[int] = [idx for idx in range(len(input_tensor_1.shape))]
    min_abs_err_perm_2: List[int] = [idx for idx in range(len(input_tensor_2.shape))]

    def define_matmul(
        *,
        target_input_tensor_1: Any,
        target_perm_1: List,
        target_input_tensor_2: Any,
        target_perm_2: List,
        taget_output_dtype: tf.dtypes.DType,
        target_name: str,
        **kwargs: Dict,
    ):
        return \
            tf.matmul(
                a=transpose_with_flexing_deterrence(
                    input_tensor=target_input_tensor_1 \
                        if not isinstance(target_input_tensor_1, np.ndarray) \
                            else tf.convert_to_tensor(target_input_tensor_1),
                    perm=target_perm_1,
                    **kwargs,
                ),
                b=transpose_with_flexing_deterrence(
                    input_tensor=target_input_tensor_2 \
                        if not isinstance(target_input_tensor_2, np.ndarray) \
                            else tf.convert_to_tensor(target_input_tensor_2),
                    perm=target_perm_2,
                    **kwargs,
                ),
                output_type=taget_output_dtype,
                name=target_name,
            )


    if onnx_tensor_infos:
        tensor_1_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_1.shape))))
        tensor_2_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_2.shape))))

        # Verify that the output shape matches that of ONNX
        # If the combination of each value of a dimension is not correct,
        # invalidate the normal processing judgment.
        if graph_node.outputs[0].name is not None \
            and graph_node.outputs[0].name != '' \
            and graph_node.outputs[0].name in onnx_tensor_infos:
            target_onnx_output: np.ndarray = onnx_tensor_infos[graph_node.outputs[0].name]
            target_onnx_output_shape = target_onnx_output.shape
        else:
            target_onnx_output_shape = onnx_output_shape

        for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
            for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
                try:
                    # Calc dummy numpy matmul
                    # Patterns that raise exceptions are not checked.
                    # Deliberately raise an exception and do not capture it.
                    # Workaround to speed up the conversion process.
                    if None not in input_tensor_1.shape \
                        and None not in input_tensor_2.shape \
                        and None not in target_onnx_output_shape \
                        and sum([1 if isinstance(s, str) else 0 for s in target_onnx_output_shape]) == 0:
                        dummy_np_1 = np.ones(list(input_tensor_1.shape), dtype=np.float32).transpose(tensor_1_candidate_for_transposition)
                        dummy_np_2 = np.ones(list(input_tensor_2.shape), dtype=np.float32).transpose(tensor_2_candidate_for_transposition)
                        dummy_np_result: np.ndarray = np.matmul(dummy_np_1, dummy_np_2)
                        if np.prod(dummy_np_result.shape) != np.prod(target_onnx_output_shape):
                            continue

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
                    dummy_matmul = define_matmul(
                        target_input_tensor_1=input_1,
                        target_perm_1=list(tensor_1_candidate_for_transposition),
                        target_input_tensor_2=input_2,
                        target_perm_2=list(tensor_2_candidate_for_transposition),
                        taget_output_dtype=output_dtype,
                        target_name=graph_node.name,
                        **kwargs
                    )
                    onnx_output_shape_prod = np.prod([dim if not isinstance(dim, str) else -1 for dim in target_onnx_output_shape])
                    matmul_output_shapes = list(dummy_matmul.shape)
                    matmul_output_shape_prod = np.prod([dim if dim is not None else -1 for dim in matmul_output_shapes])
                    if onnx_output_shape_prod != matmul_output_shape_prod:
                        del input_1
                        del input_2
                        del dummy_matmul
                        continue

                    # Perform simple accuracy verification
                    # Terminate when the error is less than 1e-3
                    try:
                        # Search for the axis with the smallest error
                        val_model = tf_keras.Model(
                            inputs=[
                                input_1,
                                input_2,
                            ],
                            outputs=[
                                dummy_matmul,
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
                        del dummy_matmul
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
                        check_results = onnx_tf_tensor_validation(
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

    try:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            define_matmul(
                target_input_tensor_1=input_tensor_1,
                target_perm_1=min_abs_err_perm_1,
                target_input_tensor_2=input_tensor_2,
                target_perm_2=min_abs_err_perm_2,
                taget_output_dtype=output_dtype,
                target_name=graph_node.name,
                **kwargs
            )
    except:
        # Workaround when data for validation cannot be obtained.
        # Verify only the certainty of the output shape, not the degradation of accuracy.
        # However, verify only if there are no undefined dimensions.
        if not disable_strict_mode \
            and None not in input_tensor_1.shape \
            and len(input_tensor_1.shape) >= 2 \
            and None not in input_tensor_2.shape \
            and len(input_tensor_2.shape) >= 2 \
            and graph_node.outputs[0].shape is not None \
            and sum([1 if isinstance(s, str) else 0 for s in graph_node.outputs[0].shape]) == 0:

            input_tensor_1_shape_last_two = input_tensor_1.shape[-2:]
            input_tensor_2_shape_last_two = input_tensor_2.shape[-2:]
            tensor_1_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_1_shape_last_two))))
            tensor_2_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_2_shape_last_two))))

            min_abs_err_perm_1 = [i for i in range(len(input_tensor_1.shape))]
            min_abs_err_perm_2 = [i for i in range(len(input_tensor_2.shape))]
            test_output_shape = graph_node.outputs[0].shape[-2:]

            for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
                for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
                    try:
                        test_tensor_1 = np.ones(input_tensor_1_shape_last_two, dtype=np.float32).transpose(tensor_1_candidate_for_transposition)
                        test_tensor_2 = np.ones(input_tensor_2_shape_last_two, dtype=np.float32).transpose(tensor_2_candidate_for_transposition)
                        test_result_tensor = np.matmul(test_tensor_1, test_tensor_2)
                        test_result_tensor_shape = list(test_result_tensor.shape)
                        if test_result_tensor_shape == test_output_shape:
                            addition_axis_1 = len(input_tensor_1.shape) - 2
                            addition_axis_2 = len(input_tensor_2.shape) - 2
                            min_abs_err_perm_1 = [i for i in range(addition_axis_1)] + [i + addition_axis_1 for i in tensor_1_candidate_for_transposition]
                            min_abs_err_perm_2 = [i for i in range(addition_axis_2)] + [i + addition_axis_2 for i in tensor_2_candidate_for_transposition]
                            break
                    except Exception as ex1:
                        pass
                else:
                    continue
                break
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                define_matmul(
                    target_input_tensor_1=input_tensor_1,
                    target_perm_1=min_abs_err_perm_1,
                    target_input_tensor_2=input_tensor_2,
                    target_perm_2=min_abs_err_perm_2,
                    taget_output_dtype=output_dtype,
                    target_name=graph_node.name,
                    **kwargs
                )
        else:
            raise

    # Response to the case where the first tensor is 1-dimensional
    # Response to the case where the second tensor is 1-dimensional
    # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    if input_tensor_1_is_one_d:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.squeeze(input=tf_layers_dict[graph_node_output.name]['tf_node'], axis=-2)
    elif input_tensor_2_is_one_d:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.squeeze(input=tf_layers_dict[graph_node_output.name]['tf_node'], axis=-1)

    # Transpose to match ONNX output shape
    post_matmul_shape = list(tf_layers_dict[graph_node_output.name]['tf_node'].shape)
    post_matmul_shape_none_count = sum([1 if dim is None else 0 for dim in post_matmul_shape])
    if post_matmul_shape_none_count <= 1 \
        and onnx_output_shape is not None \
        and post_matmul_shape != list(onnx_output_shape):

        onnx_output_shape = [
            dim if not isinstance(dim, str) else None for dim in onnx_output_shape
        ]
        onnx_output_shape_none_count = sum([1 if dim is None else 0 for dim in onnx_output_shape])
        if post_matmul_shape_none_count == onnx_output_shape_none_count:
            post_transpose_perm = []
            for dim in onnx_output_shape:
                idx = post_matmul_shape.index(dim)
                post_transpose_perm.append(idx)
                post_matmul_shape[idx] = -999

            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.transpose(
                    a=tf_layers_dict[graph_node_output.name]['tf_node'],
                    perm=post_transpose_perm,
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
