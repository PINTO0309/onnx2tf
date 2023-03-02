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
)
from typing import Any, Dict, List
from onnx2tf.utils.colors import Color


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """ReduceMax

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
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    tensor_rank = len(graph_node_input_1.shape)

    axes = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    if axes is not None and axes.shape is None:
        axes = None

    axes = graph_node.attrs.get('axes', axes)
    noop_with_empty_axes = bool(graph_node.attrs.get('noop_with_empty_axes', 0))
    if noop_with_empty_axes and axes is None:
        error_msg = f'' +\
            f'{Color.RED}ERROR:{Color.RESET} ' +\
            f'TensorFlow does not support noop_with_empty_axes=1 (True).'
        print(error_msg)
        assert not noop_with_empty_axes, error_msg

    # NCHW->NHWC, NCDHW->NDHWC
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
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1

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
    reducemaxed_tensor = tf.math.reduce_max(
        input_tensor=input_tensor,
        axis=axes,
        keepdims=keepdims,
        name=f'{graph_node.name}',
    )
    tf_layers_dict[graph_node_output.name]['tf_node'] = reducemaxed_tensor
    ### Partial model
    if kwargs['acc_check'] and tf_partial_model_inputs is not None:
        tf_partial_model_outputs = \
            [
                tf.math.reduce_max(
                    input_tensor=tf_partial_model_inputs[0],
                    axis=axes,
                    keepdims=keepdims,
                )
            ]
        tf_partial_model = tf.keras.Model(
            inputs=tf_partial_model_inputs,
            outputs=tf_partial_model_outputs,
        )
        test_data = None
        if not isinstance(input_tensor, np.ndarray):
            if not isinstance(graph_node_input_1, np.ndarray) \
                and graph_node_input_1.name in tf_layers_dict \
                and 'verification_data' in tf_layers_dict[graph_node_input_1.name].keys():
                test_data: np.ndarray = tf_layers_dict[graph_node_input_1.name]['verification_data']
            elif isinstance(graph_node_input_1, np.ndarray):
                test_data: np.ndarray = graph_node_input_1
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


    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_inputs = {f"axis{idx}": value for idx, value in enumerate(axes)} if axes is not None else {"axis": None}
    tf_inputs['input_tensor'] = input_tensor
    tf_inputs['keepdims'] = keepdims

    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.reduce_max,
                'tf_inputs': tf_inputs,
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
