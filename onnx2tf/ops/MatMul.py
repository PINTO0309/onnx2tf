import sys
import copy
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
    transpose_with_flexing_deterrence,
    get_tf_model_inputs,
    dummy_tf_inference,
    onnx_tf_tensor_validation,
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

    onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = \
        kwargs['onnx_tensor_infos_for_validation']
    test_data_nhwc: np.ndarray = \
        kwargs['test_data_nhwc']
    custom_input_op_name_np_data_path: str = \
        kwargs['custom_input_op_name_np_data_path']

    # Get the output tensor of one previous OP of TensorFlow only once
    tf_model_inputs = get_tf_model_inputs(
        tf_layers_dict=tf_layers_dict,
    )
    val_model = None
    if not isinstance(input_tensor_1, np.ndarray) \
        and not isinstance(input_tensor_2, np.ndarray):
        val_model = tf.keras.Model(
            inputs=tf_model_inputs,
            outputs=[
                input_tensor_1,
                input_tensor_2,
            ],
        )
    elif not isinstance(input_tensor_1, np.ndarray) \
        and isinstance(input_tensor_2, np.ndarray):
        val_model = tf.keras.Model(
            inputs=tf_model_inputs,
            outputs=[
                input_tensor_1
            ],
        )
    elif isinstance(input_tensor_1, np.ndarray) \
        and not isinstance(input_tensor_2, np.ndarray):
        val_model = tf.keras.Model(
            inputs=tf_model_inputs,
            outputs=[
                input_tensor_2
            ],
        )

    else:
        # TODO: Error
        pass

    # TF dummy inference
    #   Get the output tensor of the previous layer of MatMul
    #   If input.1 and input.2 are both layers, tf_pre_tensor_infos is 2 cases
    #   If one of input.1 or input.2 is np.ndarray, tf_pre_tensor_infos is 1 case
    tf_pre_tensor_infos = {}
    try:
        tf_pre_tensor_infos: Dict[Any] = dummy_tf_inference(
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
    if len(tf_pre_tensor_infos) == 2:
        validation_data_1 = list(tf_pre_tensor_infos.values())[0]
        validation_data_2 = list(tf_pre_tensor_infos.values())[1]
    elif len(tf_pre_tensor_infos) == 1:
        if not isinstance(input_tensor_1, np.ndarray):
            validation_data_1 = list(tf_pre_tensor_infos.values())[0]
            validation_data_2 = copy.deepcopy(input_tensor_2)
        else:
            validation_data_1 = copy.deepcopy(input_tensor_1)
            validation_data_2 = list(tf_pre_tensor_infos.values())[0]

    # Get ONNX inference results
    onnx_tensor_infos = None
    if onnx_tensor_infos_for_validation is not None:
        onnx_tensor_infos = {
            graph_node_output.name: onnx_tensor_infos_for_validation[graph_node_output.name]
        }
        del onnx_tensor_infos_for_validation

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
    min_abs_err_perm_1: int = [idx for idx in range(len(input_tensor_1.shape))]
    min_abs_err_perm_2: int = [idx for idx in range(len(input_tensor_2.shape))]

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

    tensor_1_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_1.shape))))
    if len(input_tensor_1.shape) == 3:
        tensor_1_candidate_for_transpositions = tensor_1_candidate_for_transpositions[:2]
    tensor_2_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_2.shape))))
    if len(input_tensor_2.shape) == 3:
        tensor_2_candidate_for_transpositions = tensor_2_candidate_for_transpositions[:2]

    for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
        for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
            try:
                # Build TF dummy model
                input_1 = tf.keras.Input(
                    shape=validation_data_1.shape[1:],
                    batch_size=validation_data_1.shape[0] \
                        if isinstance(validation_data_1.shape[0], int) else None,
                    name='dummy_input_1',
                    dtype=validation_data_1.dtype,
                )
                input_2 = tf.keras.Input(
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
                # Verify that the output shape matches that of ONNX
                # If the combination of each value of a dimension is not correct,
                # invalidate the normal processing judgment.
                onnx_output_shape_prod = np.prod([dim if not isinstance(dim, str) else -1 for dim in onnx_output_shape])
                matmul_output_shapes = list(dummy_matmul.shape)
                matmul_output_shape_prod = np.prod([dim if dim is not None else -1 for dim in matmul_output_shapes])
                if onnx_output_shape_prod != matmul_output_shape_prod:
                    del input_1
                    del input_2
                    del dummy_matmul
                    continue

                # Perform simple accuracy verification
                # Terminate when the error is less than 1e-3
                if onnx_tensor_infos is not None:
                    try:
                        # Search for the axis with the smallest error
                        val_model = tf.keras.Model(
                            inputs=[
                                input_1,
                                input_2,
                            ],
                            outputs=[
                                dummy_matmul,
                            ],
                        )

                        # TF dummy inference
                        tf_tensor_infos: Dict[Any] = dummy_tf_inference(
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

    # Transpose to match ONNX output shape
    post_matmul_shape = list(tf_layers_dict[graph_node_output.name]['tf_node'].shape)
    if sum([1 if dim is None else 0 for dim in post_matmul_shape]) <= 1 \
        and post_matmul_shape != list(onnx_output_shape):

        onnx_output_shape = [
            dim if not isinstance(dim, str) else None for dim in onnx_output_shape
        ]
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
