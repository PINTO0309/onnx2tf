import sys
import copy
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tf_keras
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    dummy_tf_inference,
    onnx_tf_tensor_validation,
    get_tf_model_inputs,
)
from typing import Any, Dict


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """LayerNormalization

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
    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    graph_node_input_2 = None
    if hasattr(graph_node.inputs[1], 'values'):
        graph_node_input_2 = graph_node.inputs[1].values
    else:
        graph_node_input_2 = graph_node.inputs[1]
    graph_node_input_3 = None
    if len(graph_node.inputs) >= 3:
        if hasattr(graph_node.inputs[2], 'values'):
            graph_node_input_3 = graph_node.inputs[2].values
        else:
            graph_node_input_3 = graph_node.inputs[2]

    scale = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    bias = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    graph_node_output_1: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output_1.shape
    dtype = graph_node_output_1.dtype

    axis = graph_node.attrs.get('axis', -1)
    pre_convert_axis = axis
    axis = convert_axis(
        axis=axis,
        tensor_rank=input_tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )
    epsilon = graph_node.attrs.get('epsilon', 1e-05)
    stash_type = bool(graph_node.attrs.get('stash_type', 1))

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output_1.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input_1.name]['nhwc'] \
            if isinstance(graph_node_input_1, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False
    }

    onnx_tensor_infos_for_validation: Dict[str:np.ndarray] = kwargs['onnx_tensor_infos_for_validation']
    test_data_nhwc: np.ndarray = kwargs['test_data_nhwc']
    custom_input_op_name_np_data_path: str = kwargs['custom_input_op_name_np_data_path']
    disable_strict_mode: bool = kwargs['disable_strict_mode']
    onnx_tensor_infos = None
    validation_data = None

    # Generation of TF OP
    tf_layers_dict[graph_node_output_1.name]['tf_node'] = \
        tf_keras.layers.LayerNormalization(
            axis=[axis],
            epsilon=epsilon,
            gamma_initializer=tf_keras.initializers.constant(scale) if scale is not None else 'ones',
            beta_initializer=tf_keras.initializers.constant(bias) if bias is not None else 'zeros',
        )(input_tensor)

    # Detect conversion errors in axis and identify the axis
    # with the smallest possible error and replace it.
    min_abs_err = sys.maxsize
    min_abs_err_axis: int = axis

    # If all axes are of different sizes and the axis sizes specified in axis are the same
    # in onnx and sensorflow, skip the accuracy check.
    acc_check_pass_flg = False
    if graph_node.inputs[0].shape is not None \
        and input_tensor.shape is not None:
        onnx_input_shapes = list(graph_node.inputs[0].shape)
        tf_input_shapes = list(input_tensor.shape)
        if onnx_input_shapes is not None \
            and tf_input_shapes is not None \
            and len(onnx_input_shapes) >= 1 \
            and len(tf_input_shapes) >= 1 \
            and len(onnx_input_shapes) == len(set(onnx_input_shapes)) \
            and not isinstance(onnx_input_shapes[pre_convert_axis], str) \
            and tf_input_shapes[axis] is not None \
            and onnx_input_shapes[pre_convert_axis] == tf_input_shapes[axis]:
            acc_check_pass_flg = True

    if not disable_strict_mode and not acc_check_pass_flg:
        # Get the output tensor of one previous OP of TensorFlow only once
        tf_model_inputs = get_tf_model_inputs(tf_layers_dict=tf_layers_dict)
        val_model = None
        if not isinstance(input_tensor, np.ndarray):
            val_model = tf_keras.Model(
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
        try:
            tf_pre_tensor_infos: Dict[Any] = \
                dummy_tf_inference(
                    model=val_model,
                    inputs=tf_model_inputs,
                    test_data_nhwc=test_data_nhwc,
                    custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                )
        except:
            pass

        # Get np.ndarray for validation
        if len(tf_pre_tensor_infos) == 1:
            if not isinstance(input_tensor, np.ndarray):
                validation_data = list(tf_pre_tensor_infos.values())[0]
            else:
                validation_data = copy.deepcopy(input_tensor)

        # Get ONNX inference results
        onnx_tensor_infos = None
        if onnx_tensor_infos_for_validation is not None \
            and onnx_tensor_infos_for_validation.get(graph_node_output_1.name, None) is not None:
            onnx_tensor_infos = {
                graph_node_output_1.name:
                onnx_tensor_infos_for_validation[graph_node_output_1.name]
            }
            del onnx_tensor_infos_for_validation

        if onnx_tensor_infos is not None and validation_data is not None:
            check_axes = reversed([idx for idx in range(input_tensor_rank)])
            # Search for the axis with the smallest error
            for check_axis in check_axes:
                try:
                    # Build TF dummy model
                    input = tf_keras.Input(
                        shape=validation_data.shape[1:],
                        batch_size=validation_data.shape[0] \
                            if isinstance(validation_data.shape[0], int) else None,
                        name='dummy_input',
                        dtype=validation_data.dtype,
                    )
                    val_model = tf_keras.Model(
                        inputs=[
                            input,
                        ],
                        outputs=[
                            tf_keras.layers.LayerNormalization(
                                axis=[check_axis],
                                epsilon=epsilon,
                                gamma_initializer=tf_keras.initializers.constant(scale) if scale is not None else 'ones',
                                beta_initializer=tf_keras.initializers.constant(bias) if bias is not None else 'zeros',
                            )(input)
                        ],
                    )
                    # TF dummy inference
                    tf_tensor_infos: Dict[Any] = \
                        dummy_tf_inference(
                            model=val_model,
                            inputs=[
                                input,
                            ],
                            verification_datas=[
                                validation_data,
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
                        min_abs_err_axis = check_axis
                        if min_abs_err < 1e-3:
                            break
                except Exception as ex:
                    pass

            tf_layers_dict[graph_node_output_1.name]['tf_node'] = \
                tf_keras.layers.LayerNormalization(
                    axis=[min_abs_err_axis],
                    epsilon=epsilon,
                    gamma_initializer=tf_keras.initializers.constant(scale) if scale is not None else 'ones',
                    beta_initializer=tf_keras.initializers.constant(bias) if bias is not None else 'zeros',
                )(input_tensor)

    # Post-process transpose
    tf_layers_dict[graph_node_output_1.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output_1.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output_1.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'LayerNormalization',
                'tf_inputs': {
                    'input': input_tensor,
                    'scale': scale,
                    'bias': bias,
                    'axis': axis,
                    'epsilon': epsilon,
                    'stash_type': stash_type,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output_1.name]['tf_node'],
                },
            }
        )
