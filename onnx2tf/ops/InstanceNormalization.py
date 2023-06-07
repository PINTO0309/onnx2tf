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
    dummy_tf_inference,
    get_tf_model_inputs,
    onnx_tf_tensor_validation,
)
from typing import Any, Dict
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """InstanceNormalization

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

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    scale = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
        is_bias=True,
    )
    scale_dtype = NUMPY_DTYPES_TO_TF_DTYPES[scale.dtype] \
        if isinstance(scale.dtype, np.dtype) else scale.dtype
    scale = tf.convert_to_tensor(scale, dtype=scale_dtype) \
        if isinstance(scale, np.ndarray) else scale

    B = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans,
        is_bias=True,
    )
    B_dtype = NUMPY_DTYPES_TO_TF_DTYPES[B.dtype] \
        if isinstance(B.dtype, np.dtype) else B.dtype
    B = tf.convert_to_tensor(B, dtype=B_dtype) \
        if isinstance(B, np.ndarray) else B

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    epsilon = graph_node.attrs.get('epsilon', 1e-05)
    epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)

    onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = kwargs['onnx_tensor_infos_for_validation']
    test_data_nhwc: np.ndarray = kwargs['test_data_nhwc']
    custom_input_op_name_np_data_path: str = kwargs['custom_input_op_name_np_data_path']
    disable_strict_mode: bool = kwargs['disable_strict_mode']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input.name]['nhwc'] \
            if isinstance(graph_node_input, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input.name].keys() else False
    }

    # Get the output tensor of one previous OP of TensorFlow only once
    if not disable_strict_mode:
        tf_model_inputs = get_tf_model_inputs(
            tf_layers_dict=tf_layers_dict,
        )
        val_model = None
        if not isinstance(input_tensor, np.ndarray):
            val_model = tf.keras.Model(
                inputs=tf_model_inputs,
                outputs=[
                    input_tensor,
                ],
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
    validation_data = None
    if not disable_strict_mode:
        if len(tf_pre_tensor_infos) == 1:
            if not isinstance(input_tensor, np.ndarray):
                validation_data = list(tf_pre_tensor_infos.values())[0]
            else:
                validation_data = copy.deepcopy(input_tensor)

        # Get ONNX inference results
        onnx_tensor_infos = None
        if onnx_tensor_infos_for_validation is not None:
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
    min_abs_err_perm_1: int = [idx for idx in range(input_tensor_rank)]
    axes = [idx for idx in range(1, input_tensor_rank - 1)]

    if not disable_strict_mode:
        if onnx_tensor_infos is not None:
            tensor_1_candidate_for_transpositions = list(itertools.permutations(range(input_tensor_rank)))
            tensor_1_candidate_for_transpositions = [
                trans_perm for trans_perm in tensor_1_candidate_for_transpositions \
                    if trans_perm[0] == 0
            ]
            # Search for the axis with the smallest error
            for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
                try:
                    target_validation_data = validation_data.transpose(tensor_1_candidate_for_transposition)
                    # Build TF dummy model
                    input = tf.keras.Input(
                        shape=target_validation_data.shape[1:],
                        batch_size=target_validation_data.shape[0] \
                            if isinstance(target_validation_data.shape[0], int) else None,
                        name='dummy_input',
                        dtype=target_validation_data.dtype,
                    )
                    mean, variance = tf.nn.moments(input, axes=axes, keepdims=True)
                    val_model = tf.keras.Model(
                        inputs=[
                            input,
                        ],
                        outputs=[
                            scale * (input - mean) * tf.math.rsqrt(variance + epsilon) + B
                        ],
                    )
                    # TF dummy inference
                    tf_tensor_infos: Dict[Any] = dummy_tf_inference(
                        model=val_model,
                        inputs=[
                            input,
                        ],
                        verification_datas=[
                            target_validation_data,
                        ],
                    )
                    del input
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
                        if min_abs_err < 1e-2:
                            break
                except Exception as ex:
                    pass

    # Generation of TF OP
    input_tensor = tf.transpose(a=input_tensor, perm=min_abs_err_perm_1)
    mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=True)

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        scale * (input_tensor - mean) * tf.math.rsqrt(variance + epsilon) + B

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
                'tf_op_type': 'InstanceNormalization',
                'tf_inputs': {
                    'x': input_tensor,
                    'mean': mean,
                    'variance': variance,
                    'offset': B,
                    'scale': scale,
                    'variance_epsilon': epsilon,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
