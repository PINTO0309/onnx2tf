import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    make_tf_partial_model_inputs,
    dummy_tf_inference,
)
from typing import Any, Dict, List
from onnx2tf.utils.enums import (
    TF_DTYPES_TO_NUMPY_DTYPES,
    NUMPY_DTYPES_TO_TF_DTYPES,
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
    """Gemm

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
    if len(graph_node.inputs) > 2:
        graph_node_input_3 = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )
    else:
        graph_node_input_3 = 0
    graph_node_output: gs.Variable = graph_node.outputs[0]

    x = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    y = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    z = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

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
            and not (isinstance(graph_node_input_2, int) and graph_node_input_2 == 0) \
            and graph_node_input_2.name in tf_layers_dict \
            and 'verification_data' in tf_layers_dict[graph_node_input_2.name].keys():
            test_data2: np.ndarray = tf_layers_dict[graph_node_input_2.name]['verification_data']
        elif isinstance(graph_node_input_2, np.ndarray):
            test_data2: np.ndarray = graph_node_input_2
        else:
            test_data2 = None
        if not isinstance(graph_node_input_3, np.ndarray) \
            and not (isinstance(graph_node_input_3, int) and graph_node_input_3 == 0) \
            and graph_node_input_3.name in tf_layers_dict \
            and 'verification_data' in tf_layers_dict[graph_node_input_3.name].keys():
            test_data3: np.ndarray = tf_layers_dict[graph_node_input_3.name]['verification_data']
        elif isinstance(graph_node_input_3, np.ndarray):
            test_data3: np.ndarray = graph_node_input_3
        else:
            test_data3 = None

    # Pre-process transpose
    x = pre_process_transpose(
        value_before_transpose=x,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    y = pre_process_transpose(
        value_before_transpose=y,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    if len(graph_node.inputs) > 2:
        z = pre_process_transpose(
            value_before_transpose=z,
            param_target='inputs',
            param_name=graph_node.inputs[2].name,
            **kwargs,
        )

    input_tensor_x_dtype = NUMPY_DTYPES_TO_TF_DTYPES[x.dtype] \
        if isinstance(x.dtype, np.dtype) else x.dtype
    x = tf.keras.layers.Flatten()(x)

    if kwargs['acc_check']:
        if test_data1 is not None and isinstance(test_data1, np.ndarray):
            test_data1 = tf.keras.layers.Flatten()(test_data1).numpy()

    # The Flatten API changes data type from tf.float64 to tf.float32
    # so we need the following line to get the original type back
    x = tf.cast(x, input_tensor_x_dtype) \
        if input_tensor_x_dtype is tf.float64 else x

    if kwargs['acc_check']:
        if test_data1 is not None and isinstance(test_data1, np.ndarray):
            test_data1 = test_data1.astype(TF_DTYPES_TO_NUMPY_DTYPES[input_tensor_x_dtype]) \
                if input_tensor_x_dtype is tf.float64 else test_data1

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    transA = bool(graph_node.attrs.get('transA', 0))
    transB = bool(graph_node.attrs.get('transB', 0))
    alpha = graph_node.attrs.get('alpha', 1.0)
    beta = graph_node.attrs.get('beta', 1.0)

    optimization_for_gpu_delegate: bool = \
        kwargs['optimization_for_gpu_delegate']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Param replacement
    x = replace_parameter(
        value_before_replacement=x,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    y = replace_parameter(
        value_before_replacement=y,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    if len(graph_node.inputs) > 2:
        z = replace_parameter(
            value_before_replacement=z,
            param_target='inputs',
            param_name=graph_node.inputs[2].name,
            **kwargs,
        )
    alpha = replace_parameter(
        value_before_replacement=alpha,
        param_target='attributes',
        param_name='alpha',
        **kwargs,
    )
    beta = replace_parameter(
        value_before_replacement=beta,
        param_target='attributes',
        param_name='beta',
        **kwargs,
    )
    transA = replace_parameter(
        value_before_replacement=transA,
        param_target='attributes',
        param_name='transA',
        **kwargs,
    )
    transB = replace_parameter(
        value_before_replacement=transB,
        param_target='attributes',
        param_name='transB',
        **kwargs,
    )

    # Generation of TF OP
    if transA == True:
        x = tf.transpose(x)
        if kwargs['acc_check'] and test_data1 is not None:
            test_data1 = tf.transpose(test_data1)
    if transB == True:
        y = tf.transpose(y)
        if kwargs['acc_check'] and test_data2 is not None:
            test_data2 = tf.transpose(test_data2)

    # Generate input OPs for TensorFlow subgraphs
    # For inference testing on OP stand-alone
    if kwargs['acc_check']:
        tf_partial_model_inputs: List[tf.keras.Input] = \
            make_tf_partial_model_inputs(
                input_tensors=[
                    x,
                    y,
                ] + ([z] if len(graph_node.inputs) > 2 else [])
            )
        tf_partial_x = tf_partial_model_inputs[0] \
            if tf_partial_model_inputs is not None else None
        tf_partial_y = tf_partial_model_inputs[1] \
            if tf_partial_model_inputs is not None else None
        tf_partial_z = tf_partial_model_inputs[2] \
            if tf_partial_model_inputs is not None \
                and len(tf_partial_model_inputs) > 2 else None
        if tf_partial_model_inputs is not None:
            if isinstance(test_data3, np.ndarray) \
                and len(test_data3.shape) == 1 \
                and tf_partial_z is not None \
                and len(tf_partial_z.shape) == 2:
                test_data3 = np.expand_dims(test_data3, 0)
        tf_partial_model_outputs = None

    # We cast to either input or attribute data type to preserve precision
    if input_tensor_x_dtype in [tf.float64]:
        # cast to input data type
        alpha = tf.cast(alpha, input_tensor_x_dtype)
        beta = tf.cast(beta, input_tensor_x_dtype)
        ### Overall model
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            alpha * tf.matmul(x, y) + beta * z
        ### Partial model
        if kwargs['acc_check'] and tf_partial_model_inputs is not None:
            tf_partial_model_outputs = \
                [
                    alpha * tf.matmul(tf_partial_x, tf_partial_y) + beta * tf_partial_z
                ]
            tf_partial_model = tf.keras.Model(
                inputs=tf_partial_model_inputs,
                outputs=tf_partial_model_outputs,
            )
            tf_partial_model_result_infos: Dict[Any] = dummy_tf_inference(
                model=tf_partial_model,
                inputs=tf_partial_model_inputs,
                verification_datas=[
                    test_data1,
                    test_data2,
                ] + ([test_data3] if len(graph_node.inputs) > 2 else [])
            )
            tf_layers_dict[graph_node_output.name]['verification_data'] = \
                list(tf_partial_model_result_infos.values())[0]
            del tf_partial_model
            del tf_partial_model_inputs
            del tf_partial_model_outputs
            del test_data1
            del test_data2
            del test_data3

    else:
        # cast to attribute data type
        x = tf.cast(x, tf.float32)
        if kwargs['acc_check'] and test_data1 is not None:
            test_data1 = tf.cast(test_data1, tf.float32)
        y = tf.cast(y, tf.float32)
        if kwargs['acc_check'] and test_data2 is not None:
            test_data2 = tf.cast(test_data2, tf.float32)
        z = tf.cast(z, tf.float32)
        if kwargs['acc_check'] and test_data3 is not None:
            test_data3 = tf.cast(test_data3, tf.float32)
        if not optimization_for_gpu_delegate:
            ### Overall model
            if z is not None:
                result = alpha * tf.matmul(x, y) + beta * z
            else:
                result = alpha * tf.matmul(x, y) + beta
            ### Partial model
            if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                if tf_partial_z is not None:
                    tf_partial_model_outputs = \
                        [
                            alpha * tf.matmul(tf_partial_x, tf_partial_y) + beta * tf_partial_z
                        ]
                else:
                    tf_partial_model_outputs = \
                        [
                            alpha * tf.matmul(tf_partial_x, tf_partial_y) + beta
                        ]
                tf_partial_model = tf.keras.Model(
                    inputs=tf_partial_model_inputs,
                    outputs=tf_partial_model_outputs,
                )
                tf_partial_model_result_infos: Dict[Any] = dummy_tf_inference(
                    model=tf_partial_model,
                    inputs=tf_partial_model_inputs,
                    verification_datas=[
                        test_data1,
                        test_data2,
                    ] + ([test_data3] if len(graph_node.inputs) > 2 else [])
                )
                tf_layers_dict[graph_node_output.name]['verification_data'] = \
                    list(tf_partial_model_result_infos.values())[0]
                del tf_partial_model
                del tf_partial_model_inputs
                del tf_partial_model_outputs
                del test_data1
                del test_data2
                del test_data3

        else:
            ### Overall model
            result = alpha * tf.matmul(x, y) - (beta * z) * -1
            ### Partial model
            if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                tf_partial_model_outputs = \
                    [
                        alpha * tf.matmul(tf_partial_x, tf_partial_y) - (beta * tf_partial_z) * -1
                    ]
                tf_partial_model = tf.keras.Model(
                    inputs=tf_partial_model_inputs,
                    outputs=tf_partial_model_outputs,
                )
                tf_partial_model_result_infos: Dict[Any] = dummy_tf_inference(
                    model=tf_partial_model,
                    inputs=tf_partial_model_inputs,
                    verification_datas=[
                        test_data1,
                        test_data2,
                        test_data3,
                    ]
                )
                tf_layers_dict[graph_node_output.name]['verification_data'] = \
                    list(tf_partial_model_result_infos.values())[0]
                del tf_partial_model
                del tf_partial_model_inputs
                del tf_partial_model_outputs
                del test_data1
                del test_data2
                del test_data3

        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.cast(result, input_tensor_x_dtype)
        if kwargs['acc_check']:
            if 'verification_data' in tf_layers_dict[graph_node_output.name].keys():
                tf_layers_dict[graph_node_output.name]['verification_data'] = \
                    tf_layers_dict[graph_node_output.name]['verification_data'].astype(
                        TF_DTYPES_TO_NUMPY_DTYPES[input_tensor_x_dtype]
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
                    'x': x,
                    'y': y,
                    'z': z,
                    'alpha': alpha,
                    'beta': beta,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
