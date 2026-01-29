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


def _as_tensor(value):
    if isinstance(value, np.ndarray):
        return tf.convert_to_tensor(value)
    if isinstance(value, (np.generic, str, bytes)):
        return tf.convert_to_tensor(value)
    return value


def _normalize_delimiter(delimiter):
    if delimiter is None:
        return None
    if isinstance(delimiter, bytes):
        delimiter = delimiter.decode('utf-8')
    if delimiter == "":
        return None
    return delimiter


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """StringSplit

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
    graph_node_output_1: gs.Variable = graph_node.outputs[0]
    graph_node_output_2: gs.Variable = graph_node.outputs[1]
    output_1_shape = graph_node_output_1.shape
    output_1_dtype = graph_node_output_1.dtype
    output_2_shape = graph_node_output_2.shape
    output_2_dtype = graph_node_output_2.dtype

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output_1.name] = {
        'optype': graph_node.op,
        'shape': output_1_shape,
        'dtype': output_1_dtype,
    }
    tf_layers_dict[graph_node_output_2.name] = {
        'optype': graph_node.op,
        'shape': output_2_shape,
        'dtype': output_2_dtype,
    }

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    input_tensor = _as_tensor(input_tensor)
    delimiter = _normalize_delimiter(graph_node.attrs.get('delimiter', None))
    maxsplit = graph_node.attrs.get('maxsplit', None)
    if maxsplit is None:
        maxsplit = -1

    split_rt = tf.strings.split(
        input=input_tensor,
        sep=delimiter,
        maxsplit=maxsplit,
    )
    output_strings = split_rt.to_tensor(default_value="")
    output_counts = split_rt.row_lengths()
    output_counts = tf.reshape(output_counts, tf.shape(input_tensor))
    output_counts = tf.cast(output_counts, tf.int64)

    tf_layers_dict[graph_node_output_1.name]['tf_node'] = output_strings
    tf_layers_dict[graph_node_output_2.name]['tf_node'] = output_counts

    # Post-process transpose
    tf_layers_dict[graph_node_output_1.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output_1.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )
    tf_layers_dict[graph_node_output_2.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output_2.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[1].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output_1.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.strings.split,
                'tf_inputs': {
                    'input': input_tensor,
                    'sep': delimiter,
                    'maxsplit': maxsplit,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output_1.name]['tf_node'],
                },
            }
        )
    tf_layers_dict[graph_node_output_2.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.strings.split,
                'tf_inputs': {
                    'input': input_tensor,
                    'sep': delimiter,
                    'maxsplit': maxsplit,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output_2.name]['tf_node'],
                },
            }
        )
