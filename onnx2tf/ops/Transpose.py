import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    transpose_with_flexing_deterrence,
    make_tf_partial_model_inputs,
    dummy_tf_inference,
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
    """Transpose

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
    output_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    input_tensor_shape = input_tensor.shape
    tensor_rank = len(input_tensor_shape)

    perm = graph_node.attrs.get('perm', [idx for idx in reversed(range(tensor_rank))])
    if 'nwc_nhwc_ndhwc_keep' in tf_layers_dict[graph_node_input.name] \
        and tf_layers_dict[graph_node_input.name]['nwc_nhwc_ndhwc_keep'] == True:
        perm = [i for i in range(tensor_rank)]

    nwc_nhwc_ndhwc_keep = False
    if isinstance(perm, list) or (isinstance(perm, np.ndarray) and len(perm.shape) > 0):
        if perm[0] == 0:
            try:
                if graph_node.o().op == 'Softmax' \
                    and graph_node.o().inputs[0].shape == input_tensor_shape:
                    perm = [idx for idx in range(tensor_rank)]
                    nwc_nhwc_ndhwc_keep = True
                else:
                    perm = [
                        convert_axis(
                            axis=idx,
                            tensor_rank=tensor_rank,
                            before_op_output_shape_trans=before_op_output_shape_trans,
                        ) for idx in perm
                    ]
            except:
                perm = [
                    convert_axis(
                        axis=idx,
                        tensor_rank=tensor_rank,
                        before_op_output_shape_trans=before_op_output_shape_trans,
                    ) for idx in perm
                ]
        elif output_shape is not None:
            # When a zero-dimensional transposition occurs, compare the shape
            # of the final output tensor of ONNX with the shape
            # of the input tensor of TF and transpose to match the shape
            # of the final output tensor on the ONNX side
            onnx_output_shape = [s if not isinstance(s, str) else None for s in output_shape]
            onnx_output_shape_none_count = onnx_output_shape.count(None)
            tf_input_shape = input_tensor_shape
            tf_input_shape_none_count = [s for s in tf_input_shape].count(None)
            new_perm = [-1] * len(onnx_output_shape)
            if onnx_output_shape_none_count > 0 and tf_input_shape_none_count == 0:
                pass
            else:
                for tf_shape_idx, tf_shape_value in enumerate(tf_input_shape):
                    matched_idxs = [
                        idx for idx, onnx_shape_value in enumerate(onnx_output_shape) \
                            if onnx_shape_value == tf_shape_value
                    ]
                    if len(matched_idxs) == 0 and onnx_output_shape_none_count <= 1:
                        new_perm[tf_shape_idx] = onnx_output_shape.index(tf_shape_value)
                    elif len(matched_idxs) == 0 and onnx_output_shape_none_count > 1:
                        new_perm = perm
                    elif len(matched_idxs) == 1:
                        new_perm[matched_idxs[0]] = tf_shape_idx
                    else:
                        for matched_idx in matched_idxs:
                            if new_perm[matched_idx] == -1:
                                new_perm[matched_idx] = tf_shape_idx
                                break
                perm = new_perm

    elif perm is not None and isinstance(perm, np.ndarray) and len(perm.shape) == 0:
        if perm[0] == 0:
            perm = convert_axis(
                axis=perm,
                tensor_rank=tensor_rank,
                before_op_output_shape_trans=before_op_output_shape_trans,
            )
        elif output_shape is not None:
            # When a zero-dimensional transposition occurs, compare the shape
            # of the final output tensor of ONNX with the shape of the input tensor
            # of TF and transpose to match the shape of the final output tensor on the ONNX side
            onnx_output_shape = [s if not isinstance(s, str) else None for s in output_shape]
            onnx_output_shape_none_count = onnx_output_shape.count(None)
            tf_input_shape = input_tensor_shape
            tf_input_shape_none_count = [s for s in tf_input_shape].count(None)
            new_perm = [-1] * len(onnx_output_shape)
            if onnx_output_shape_none_count > 0 and tf_input_shape_none_count == 0:
                pass
            else:
                for tf_shape_idx, tf_shape_value in enumerate(tf_input_shape):
                    matched_idxs = [
                        idx for idx, onnx_shape_value in enumerate(onnx_output_shape) \
                            if onnx_shape_value == tf_shape_value
                    ]
                    if len(matched_idxs) == 0 and onnx_output_shape_none_count <= 1:
                        new_perm[tf_shape_idx] = onnx_output_shape.index(tf_shape_value)
                    elif len(matched_idxs) == 0 and onnx_output_shape_none_count > 1:
                        new_perm = perm
                    elif len(matched_idxs) == 1:
                        new_perm[matched_idxs[0]] = tf_shape_idx
                    else:
                        for matched_idx in matched_idxs:
                            if new_perm[matched_idx] == -1:
                                new_perm[matched_idx] = tf_shape_idx
                                break
                perm = new_perm

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': output_shape,
        'dtype': dtype,
        'nwc_nhwc_ndhwc_keep': nwc_nhwc_ndhwc_keep,
    }

    perm = list(perm) if perm is not None else None

    # Param replacement
    input_tensor = replace_parameter(
        value_before_replacement=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    perm = replace_parameter(
        value_before_replacement=perm,
        param_target='attributes',
        param_name='perm',
        **kwargs,
    )

    # Generate input OPs for TensorFlow subgraphs
    # For inference testing on OP stand-alone
    if kwargs['acc_check']:
        tf_partial_model_inputs: List[tf.keras.Input] = \
            make_tf_partial_model_inputs(
                input_tensors=[input_tensor]
            )
        tf_partial_model_outputs = None

    # Generation of TF OP
    ### Overall model
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        transpose_with_flexing_deterrence(
            input_tensor=input_tensor,
            perm=perm,
            output_shape=output_shape,
            name=graph_node.name,
            **kwargs,
        )
    ### Partial model
    if kwargs['acc_check'] and tf_partial_model_inputs is not None:
        tf_partial_model_outputs = \
            [
                transpose_with_flexing_deterrence(
                    input_tensor=tf_partial_model_inputs[0],
                    perm=perm,
                    output_shape=output_shape,
                    **kwargs,
                )
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

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.transpose,
                'tf_inputs': {
                    'a': input_tensor,
                    'perm': perm,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
