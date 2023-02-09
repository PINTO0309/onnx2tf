import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    replace_parameter,
    get_constant_or_variable,
    convert_axis,
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


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Softmax

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

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    tensor_rank = len(input_tensor.shape)

    axis = graph_node.attrs.get('axis', tensor_rank - 1)
    axis = convert_axis(
        axis=axis,
        tensor_rank=tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input.name]['nhwc'] \
            if isinstance(graph_node_input, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input.name].keys() else False,
        'nwc_nhwc_ndhwc_keep': tf_layers_dict[graph_node_input.name]['nwc_nhwc_ndhwc_keep'] \
            if isinstance(graph_node_input, gs.Variable) \
                and 'nwc_nhwc_ndhwc_keep' in tf_layers_dict[graph_node_input.name].keys() else False,
    }

    # Param replacement
    input_tensor = replace_parameter(
        value_before_replacement=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    axis = replace_parameter(
        value_before_replacement=axis,
        param_target='attributes',
        param_name='axis',
        **kwargs,
    )

    # Pre-process transpose
    before_trans_shape = input_tensor.shape
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    after_trans_shape = input_tensor.shape
    if 'nhwc' in tf_layers_dict[graph_node_output.name].keys() \
        and tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and before_trans_shape != after_trans_shape:
        tf_layers_dict[graph_node_output.name].pop('nhwc')

    # Detect conversion errors in axis and identify the axis
    # with the smallest possible error and replace it.
    min_abs_err = sys.maxsize
    min_abs_err_axis: int = axis
    try:
        onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = \
            kwargs['onnx_tensor_infos_for_validation']
        if onnx_tensor_infos_for_validation is not None:
            onnx_tensor_infos = {
                graph_node_output.name: onnx_tensor_infos_for_validation[graph_node_output.name]
            }
            del onnx_tensor_infos_for_validation
            check_axes = reversed([idx for idx in range(tensor_rank)])
            # Search for the axis with the smallest error
            tf_model_inputs = get_tf_model_inputs(
                tf_layers_dict=tf_layers_dict,
            )
            for check_axis in check_axes:
                # TF dummy inference
                val_model = tf.keras.Model(
                    inputs=tf_model_inputs,
                    outputs=[
                        tf.nn.softmax(
                            logits=input_tensor,
                            axis=check_axis,
                            name=graph_node.name,
                        )
                    ],
                )
                tf_tensor_infos: Dict[Any] = dummy_tf_inference(
                    model=val_model,
                    inputs=tf_model_inputs,
                )
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
    except tf.errors.InvalidArgumentError as ex:
        pass

    # Generation of TF OP
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.nn.softmax(
            logits=input_tensor,
            axis=min_abs_err_axis,
            name=graph_node.name,
        )

    # Post-process transpose
    before_trans_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )
    after_trans_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    if 'nhwc' in tf_layers_dict[graph_node_output.name].keys() \
        and tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and before_trans_shape != after_trans_shape:
        tf_layers_dict[graph_node_output.name].pop('nhwc')

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.nn.softmax,
                'tf_inputs': {
                    'logits': input_tensor,
                    'axis': axis,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
