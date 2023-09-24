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
    """Dropout

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
    data = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1

    # Pre-process transpose
    data = pre_process_transpose(
        value_before_transpose=data,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    graph_node_input_2 = None
    if len(graph_node.inputs) >= 2:
        graph_node_input_2 = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
    graph_node_input_3 = None
    if len(graph_node.inputs) >= 3:
        graph_node_input_3 = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )
    ratio = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    training_mode = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    training_mode = False if training_mode is None else True

    opset = kwargs['opset']
    seed = graph_node.attrs.get('seed', None)
    is_test = bool(graph_node.attrs.get('is_test', 0))
    ratio = graph_node.attrs.get('ratio', ratio)

    if opset < 7 and not is_test:
        ratio = 1.0 - ratio

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }
    if len(graph_node.outputs) >= 2:
        tf_layers_dict[graph_node.outputs[1].name] = {
            'optype': graph_node.op,
            'shape': shape,
            'dtype': dtype,
        }

    # Generation of TF OP
    dropout_result = None
    mask = None
    if opset < 7:
        tf_layers_dict[graph_node_output.name]['tf_node'] = data
        return_mask = len(graph_node.outputs) >= 2 # if there are 2 outputs, mask is requested
        if return_mask:
            mask = tf.ones(data.shape, dtype=tf.bool)
            tf_layers_dict[graph_node.outputs[1].name]['tf_node'] = mask
    elif opset < 12:
        tf_layers_dict[graph_node_output.name]['tf_node'] = data
        return_mask = len(graph_node.outputs) >= 2 # if there are 2 outputs, mask is requested
        if return_mask:
            mask = tf.ones(data.shape, dtype=tf.bool)
            tf_layers_dict[graph_node.outputs[1].name]['tf_node'] = mask
        dropout_result = data
    else:
        # ratio and training_mode are optional and passed as inputs
        return_mask = len(graph_node.outputs) >= 2 # if there are 2 outputs, mask is requested
        if ratio == 0.0 or training_mode is False:
            # Inferencing
            if return_mask:
                tf_layers_dict[graph_node_output.name]['tf_node'] = data
                mask = tf.ones(data.shape, dtype=tf.bool)
                tf_layers_dict[graph_node.outputs[1].name]['tf_node'] = mask
            else:
                tf_layers_dict[graph_node_output.name]['tf_node'] = data
            dropout_result = data
        else:
            # Training
            dropout_result = tf.nn.dropout(
                x=data,
                rate=ratio,
                noise_shape=None,
                seed=seed,
                name=graph_node.name,
            )
            if return_mask:
                # Create the mask based on the result of the Dropout
                mask = tf.dtypes.cast(
                    dropout_result,
                    tf.bool,
                )
                tf_layers_dict[graph_node_output.name]['tf_node'] = dropout_result
                tf_layers_dict[graph_node.outputs[1].name]['tf_node'] = mask
            else:
                tf_layers_dict[graph_node_output.name]['tf_node'] = dropout_result

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
                'tf_op_type': tf.nn.dropout,
                'tf_inputs': {
                    'x': data,
                    'rate': ratio,
                    'noise_shape': None,
                    'seed': seed,
                },
                'tf_outputs': {
                    'dropout_result': dropout_result,
                    'mask': mask,
                },
            }
        )
