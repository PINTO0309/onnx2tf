import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    convert_axis,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    pre_process_transpose,
    post_process_transpose,
    make_tf_partial_model_inputs,
    dummy_tf_inference,
    transpose_with_flexing_deterrence,
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
    """Concat

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
    nhwc_flags = []
    same_input_shape_as_onnxs = []
    for graph_node_input in graph_node.inputs:
        const_or_var = get_constant_or_variable(
            graph_node_input,
            before_op_output_shape_trans,
        )
        if isinstance(const_or_var, gs.Variable):
            values.append(tf_layers_dict[const_or_var.name]['tf_node'])
            nhwc_flags.append(
                tf_layers_dict[const_or_var.name]['nhwc'] \
                    if 'nhwc' in tf_layers_dict[const_or_var.name].keys() else False
            )
            same_input_shape_as_onnxs.append(
                True if graph_node_input.shape is not None and len(graph_node_input.shape) > 0 \
                    and graph_node_input.shape == tf_layers_dict[const_or_var.name]['tf_node'].shape else False
            )
        else:
            values.append(const_or_var)
            nhwc_flags.append(False)
            same_input_shape_as_onnxs.append(
                True if graph_node_input.shape is not None and len(graph_node_input.shape) > 0 \
                    and graph_node_input.shape == const_or_var.shape else False
            )

    # Shape Unmatched Special Avoidance Workaround
    # At least one True value for same_input_shape_as_onnx
    # At least one True value in nhwc_flags
    # same_input_shape_as_onnx == True and nhwc_flags == False and 3D or 4D or 5D tensor is NHWC transposed
    if True in same_input_shape_as_onnxs and True in nhwc_flags:
        before_op_output_shape_trans = True
        new_values = []
        for same_input_shape_as_onnx, nhwc_flag, value in zip(same_input_shape_as_onnxs, nhwc_flags, values):
            if same_input_shape_as_onnx and not nhwc_flag:
                if len(value.shape) == 3:
                    new_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0,2,1],
                            **kwargs,
                        )
                    )
                elif len(value.shape) == 4:
                    new_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0,2,3,1],
                            **kwargs,
                        )
                    )
                elif len(value.shape) == 5:
                    new_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0,2,3,4,1],
                            **kwargs,
                        )
                    )
                else:
                    new_values.append(value)
            else:
                new_values.append(value)
        values = new_values

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    axis = graph_node.attrs.get('axis', 0)
    # NCHW->NHWC, NCDHW->NDHWC
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(shape) if shape is not None else len(values[0].shape),
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # Param replacement
    axis = replace_parameter(
        value_before_replacement=axis,
        param_target='attributes',
        param_name='axis',
        **kwargs,
    )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP

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

    # TensorFlow does not support Concat for scalar values, so convert to tensor
    values = [
        value if len(value.shape) > 0 else tf.reshape(value, [1]) for value in values
    ]

    # Generate input OPs for TensorFlow subgraphs
    # For inference testing on OP stand-alone
    if kwargs['acc_check']:
        tf_partial_model_inputs: List[tf.keras.Input] = \
            make_tf_partial_model_inputs(
                input_tensors=values
            )
        tf_partial_model_outputs = None

    # Generation of TF OP
    ### Overall model
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.concat(
            values=values,
            axis=axis,
            name=graph_node.name,
        )
    ### Partial model
    if kwargs['acc_check'] and tf_partial_model_inputs is not None:
        test_datas = []
        for graph_node_input, value in zip(graph_node.inputs, values):
            test_data = None
            if not isinstance(value, np.ndarray):
                if not isinstance(graph_node_input, np.ndarray) \
                    and graph_node_input.name in tf_layers_dict \
                    and 'verification_data' in tf_layers_dict[graph_node_input.name].keys():
                    test_data: np.ndarray = tf_layers_dict[graph_node_input.name]['verification_data']
                elif isinstance(graph_node_input, np.ndarray):
                    test_data: np.ndarray = graph_node_input
                else:
                    test_data = None
            else:
                test_data = value
            test_datas.append(test_data)
        new_test_datas = []
        is_expanded = False
        for tf_partial_model_input, test_data in zip(tf_partial_model_inputs, test_datas):
            if isinstance(test_data, np.ndarray) \
                and len(test_data.shape) == 1 \
                and len(tf_partial_model_input.shape) == 2:
                test_data = np.expand_dims(test_data, 0)
                is_expanded = True
            new_test_datas.append(test_data)
        test_datas = new_test_datas
        tf_partial_model_outputs = \
            [
                tf.concat(
                    values=tf_partial_model_inputs,
                    axis=axis,
                ) if not is_expanded else \
                tf.squeeze(
                    input=tf.concat(
                        values=tf_partial_model_inputs,
                        axis=axis if not is_expanded else axis+1,
                    ),
                    axis=0,
                )
            ]
        tf_partial_model = tf.keras.Model(
            inputs=tf_partial_model_inputs,
            outputs=tf_partial_model_outputs,
        )
        tf_partial_model_result_infos: Dict[Any] = dummy_tf_inference(
            model=tf_partial_model,
            inputs=tf_partial_model_inputs,
            verification_datas=test_datas
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
    tf_inputs = {f"input{idx}": value for idx, value in enumerate(values)}
    tf_inputs['axis'] = axis
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.concat,
                'tf_inputs': tf_inputs,
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
