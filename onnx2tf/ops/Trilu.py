from typing import Any
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx2tf.gs as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: Any,
):
    """Trilu

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
    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    tensor_shape = input_tensor.shape

    graph_node_input_2 = None
    k = None
    if len(graph_node.inputs) >= 2:
        graph_node_input_2 = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
        k = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
            if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    k_value = 0
    if k is not None:
        if isinstance(k, np.ndarray):
            if k.size > 0:
                k_value = int(k.reshape(-1)[0])
        elif isinstance(k, np.generic):
            k_value = int(k.item())
        elif tf.is_tensor(k):
            k_static = tf.get_static_value(k)
            if isinstance(k_static, np.ndarray):
                if k_static.size > 0:
                    k_value = int(k_static.reshape(-1)[0])
            elif isinstance(k_static, np.generic):
                k_value = int(k_static.item())
            elif isinstance(k_static, (int, np.integer)):
                k_value = int(k_static)
        elif isinstance(k, (int, np.integer)):
            k_value = int(k)

    if isinstance(tensor_shape[-1], int) and k_value > tensor_shape[-1]:
        k_value = int(tensor_shape[-1])
    if isinstance(tensor_shape[-2], int) and k_value < -int(tensor_shape[-2]):
        k_value = -int(tensor_shape[-2])

    k = int(k_value)
    keep_triangle = -1

    upper = bool(graph_node.attrs.get('upper', 1))

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    if upper:
        if k > 0:
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.subtract(
                    x=input_tensor,
                    y=tf.linalg.band_part(
                        input=input_tensor,
                        num_lower=keep_triangle,
                        num_upper=k - 1,
                    ),
                    name=graph_node.name,
                )
        else:
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.linalg.band_part(
                    input=input_tensor,
                    num_lower=-k,
                    num_upper=keep_triangle,
                    name=graph_node.name,
                )
    else:
        if k >= 0:
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.linalg.band_part(
                    input=input_tensor,
                    num_lower=keep_triangle,
                    num_upper=k,
                    name=graph_node.name,
                )
        else:
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.subtract(
                    x=input_tensor,
                    y=tf.linalg.band_part(
                        input=input_tensor,
                        num_lower=-1 - k,
                        num_upper=keep_triangle,
                    ),
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
                'tf_op_type': 'Trilu',
                'tf_inputs': {
                    'input': input_tensor,
                    'k': k,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
