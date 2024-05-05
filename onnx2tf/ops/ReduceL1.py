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
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    pre_process_transpose,
    post_process_transpose,
    dummy_tf_inference,
    get_tf_model_inputs,
    onnx_tf_tensor_validation,
    define_reduceXXX,
)
from onnx2tf.utils.logging import *
from typing import Any, Dict, List, Tuple


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """ReduceL1

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = None
    if len(graph_node.inputs) >= 2:
        graph_node_input_2 = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    onnx_output_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    tensor_rank = len(input_tensor.shape)

    axes = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    if axes is not None and axes.shape is None:
        axes = None

    axes = graph_node.attrs.get('axes', axes)
    noop_with_empty_axes = bool(graph_node.attrs.get('noop_with_empty_axes', 0))
    if noop_with_empty_axes and axes is None:
        error_msg = f'' +\
            Color.RED(f'ERROR:') + ' ' +\
            f'TensorFlow does not support noop_with_empty_axes=1 (True).'
        print(error_msg)
        assert not noop_with_empty_axes, error_msg

    if isinstance(axes, list) or (isinstance(axes, np.ndarray) and len(axes.shape) > 0):
        axes = [
            convert_axis(
                axis=idx,
                tensor_rank=tensor_rank,
                before_op_output_shape_trans=before_op_output_shape_trans,
            ) for idx in axes
        ]
    elif axes is not None and isinstance(axes, np.ndarray) and len(axes.shape) == 0:
        axes = convert_axis(
            axis=axes,
            tensor_rank=tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        )

    # 0: False, 1: True
    keepdims = bool(graph_node.attrs.get('keepdims', 1))

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': onnx_output_shape,
        'dtype': dtype,
    }

    onnx_tensor_infos_for_validation: Dict[str:np.ndarray] = kwargs['onnx_tensor_infos_for_validation']
    test_data_nhwc: np.ndarray = kwargs['test_data_nhwc']
    custom_input_op_name_np_data_path: str = kwargs['custom_input_op_name_np_data_path']
    disable_strict_mode: bool = kwargs['disable_strict_mode']
    onnx_tensor_infos = None
    validation_data = None

    if onnx_tensor_infos_for_validation is not None \
        and onnx_tensor_infos_for_validation.get(graph_node_output.name, None) is not None:
        # Get the output tensor of one previous OP of TensorFlow only once
        if not disable_strict_mode:
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
            except:
                pass

        # Get np.ndarray for validation
        if not disable_strict_mode:
            if len(tf_pre_tensor_infos) == 1:
                if not isinstance(input_tensor, np.ndarray):
                    validation_data = list(tf_pre_tensor_infos.values())[0]
                else:
                    validation_data = copy.deepcopy(input_tensor)

            # Get ONNX inference results
            onnx_tensor_infos = None
            if onnx_tensor_infos_for_validation is not None \
                and onnx_tensor_infos_for_validation.get(graph_node_output.name, None) is not None:
                onnx_tensor_infos = {
                    graph_node_output.name:
                    onnx_tensor_infos_for_validation[graph_node_output.name]
                }
                del onnx_tensor_infos_for_validation

    if not disable_strict_mode:
        if onnx_tensor_infos is not None and validation_data is not None and axes is not None:
            # Shape Unmatch Error Mitigation Measures
            # Search for and transpose shapes that do not cause shape unmatch errors
            min_abs_err = sys.maxsize
            min_abs_err_axes: List[int] = None
            if isinstance(axes, list):
                min_abs_err_axes = copy.deepcopy(axes)
            elif isinstance(axes, int):
                min_abs_err_axes = [axes]
            elif isinstance(axes, np.ndarray):
                min_abs_err_axes = list(axes)
            else:
                min_abs_err_axes = axes

            check_axes_tuples: List[Tuple] = list(itertools.combinations(list(range(tensor_rank)), len(axes)))
            if tuple(axes) in check_axes_tuples:
                check_axes_tuples.remove(tuple(axes))
                check_axes_tuples.insert(0, tuple(axes))
            check_axes = [list(check_axes_tuple) for check_axes_tuple in check_axes_tuples]

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

            # Search for the axis with the smallest error
            for check_axis in check_axes:
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
                        define_reduceXXX(
                            tf_func='ReduceL1',
                            target_input_tensor=input,
                            target_axes=check_axis,
                            target_keepdims=keepdims,
                        )
                    ],
                )

                onnx_output_shape_prod = np.prod([dim if not isinstance(dim, str) else -1 for dim in target_onnx_output_shape])
                val_model_output_shapes = list(val_model.output.shape)
                val_model_output_shape_prod = np.prod([dim if dim is not None else -1 for dim in val_model_output_shapes])
                if onnx_output_shape_prod != val_model_output_shape_prod:
                    del input
                    del val_model
                    continue

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
                    min_abs_err_axes = check_axis
                    if min_abs_err < 1e-3:
                        break
            axes = min_abs_err_axes


    # Param replacement
    axes = replace_parameter(
        value_before_replacement=axes,
        param_target='attributes',
        param_name='axes',
        **kwargs,
    )
    keepdims = replace_parameter(
        value_before_replacement=keepdims,
        param_target='attributes',
        param_name='keepdims',
        **kwargs,
    )

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    axes = list(axes) if axes is not None else None
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.norm(
            tensor=input_tensor,
            ord=1,
            axis=axes if len(axes) > 1 else axes[0],
            keepdims=keepdims,
            name=graph_node.name,
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
                'tf_op_type': tf.norm,
                'tf_inputs': {
                    'tensor': input_tensor,
                    'ord': 'euclidean',
                    'axis': axes,
                    'keepdims': keepdims,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
