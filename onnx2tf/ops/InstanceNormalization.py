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
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    make_tf_partial_model_inputs,
    dummy_tf_inference,
    transpose_with_flexing_deterrence,
)
from typing import Any, Dict, List
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES

INF_INDEX_VALUE: int = 4294967296


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

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # transpose
    tf_transposed_perm = None
    test_data_transposed_perm = None
    try:
        if graph_node.i().op == 'Reshape':
            onnx_input_shape = [
                dim if isinstance(dim, int) else None for dim in graph_node.inputs[0].shape
            ]
            tf_input_shape = [
                dim if isinstance(dim, int) else None for dim in input_tensor_shape
            ]
            if len(onnx_input_shape) > 1 and len(tf_input_shape) > 1 \
                and onnx_input_shape == tf_input_shape:

                shape_for_judging_skip = [
                    dim if dim is not None else INF_INDEX_VALUE for dim in onnx_input_shape[1:]
                ]
                if shape_for_judging_skip.count(shape_for_judging_skip[0]) != len(shape_for_judging_skip):
                    if len(onnx_input_shape) == 3:
                        # 1D
                        input_tensor = transpose_with_flexing_deterrence(
                            input_tensor=input_tensor,
                            perm=[0,2,1],
                            **kwargs,
                        )
                        tf_transposed_perm = [0,2,1]
                    elif len(onnx_input_shape) == 4:
                        # 2D
                        input_tensor = transpose_with_flexing_deterrence(
                            input_tensor=input_tensor,
                            perm=[0,2,3,1],
                            **kwargs,
                        )
                        tf_transposed_perm = [0,2,3,1]
                else:
                    if len(onnx_input_shape) == 3:
                        test_data_transposed_perm = [0,2,1]
                    elif len(onnx_input_shape) == 4:
                        test_data_transposed_perm = [0,2,3,1]
    except:
        pass

    # Generate input OPs for TensorFlow subgraphs
    # For inference testing on OP stand-alone
    if kwargs['acc_check']:
        tf_partial_model_inputs: List[tf.keras.Input] = \
            make_tf_partial_model_inputs(
                input_tensors=[input_tensor]
            )
        tf_partial_model_outputs = None


    # Generation of TF OP
    axes = [idx for idx in range(1, input_tensor_rank - 1)]
    ### Overall model
    mean = tf.reduce_mean(
        input_tensor=input_tensor,
        axis=axes,
        keepdims=True,
    )
    variance = tf.math.reduce_variance(
        input_tensor=input_tensor,
        axis=axes,
        keepdims=True,
    )
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        (input_tensor - mean) / tf.math.sqrt(variance + epsilon) * scale + B
    ### Partial model
    if kwargs['acc_check'] and tf_partial_model_inputs is not None:
        mean = tf.reduce_mean(
            input_tensor=tf_partial_model_inputs[0],
            axis=axes,
            keepdims=True,
        )
        variance = tf.math.reduce_variance(
            input_tensor=tf_partial_model_inputs[0],
            axis=axes,
            keepdims=True,
        )
        pre_instance_norm = (tf_partial_model_inputs[0] - mean) / tf.math.sqrt(variance + epsilon)
        tf_partial_model_outputs = \
            [
                pre_instance_norm * scale + B
            ]
        tf_partial_model = tf.keras.Model(
            inputs=tf_partial_model_inputs,
            outputs=tf_partial_model_outputs,
        )
        test_data = None
        if not isinstance(input_tensor, np.ndarray):
            if not isinstance(graph_node_input, np.ndarray) \
                and graph_node_input.name in tf_layers_dict \
                and 'verification_data' in tf_layers_dict[graph_node_input.name].keys():
                test_data: np.ndarray = tf_layers_dict[graph_node_input.name]['verification_data']
            elif isinstance(graph_node_input, np.ndarray):
                test_data: np.ndarray = graph_node_input
            else:
                test_data = None
        else:
            test_data = input_tensor
        if tf_transposed_perm is not None \
            and isinstance(test_data, np.ndarray):
            test_data = test_data.transpose(tf_transposed_perm)
        elif test_data_transposed_perm is not None \
            and isinstance(test_data, np.ndarray) \
            and tf_input_shape != list(test_data.shape):
            test_data = test_data.transpose(test_data_transposed_perm)
        tf_partial_model_result_infos: Dict[Any] = dummy_tf_inference(
            model=tf_partial_model,
            inputs=tf_partial_model_inputs,
            verification_datas=[
                test_data
            ]
        )
        tf_layers_dict[graph_node_output.name]['verification_data'] = \
            list(tf_partial_model_result_infos.values())[0]
        del tf_partial_model
        del tf_partial_model_inputs
        del tf_partial_model_outputs
        del test_data

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
