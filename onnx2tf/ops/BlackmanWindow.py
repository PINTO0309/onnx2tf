import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from onnx import TensorProto
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
from onnx2tf.utils.enums import ONNX_DTYPES_TO_TF_DTYPES


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """BlackmanWindow

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans=False,
    )
    size = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1

    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    output_datatype = int(graph_node.attrs.get('output_datatype', TensorProto.FLOAT))
    output_datatype = ONNX_DTYPES_TO_TF_DTYPES[output_datatype]
    periodic = bool(graph_node.attrs.get('periodic', 1))

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Pre-process transpose
    size = pre_process_transpose(
        value_before_transpose=size,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    size_fp = tf.cast(size, tf.float32)
    periodic_size_fp = size_fp
    symmetric_size_fp = size_fp - tf.constant(1.0, dtype=tf.float32)
    is_periodic_fp = tf.cast(periodic, tf.float32)
    size_fp = periodic_size_fp * is_periodic_fp + symmetric_size_fp * (1.0 - is_periodic_fp)

    two_pi = tf.constant(6.28319, dtype=tf.float32)
    angular_increment = tf.math.divide_no_nan(two_pi, size_fp)
    range_vals = tf.range(tf.cast(periodic_size_fp, tf.int32), dtype=tf.float32)
    range_angular = range_vals * angular_increment

    a0 = tf.constant(0.42, dtype=tf.float32)
    a1 = tf.constant(0.5, dtype=tf.float32)
    a2 = tf.constant(0.08, dtype=tf.float32)

    temp0 = a0 - a1 * tf.cos(range_angular)
    temp1 = temp0 + a2 * tf.cos(range_angular * 2.0)
    tf_layers_dict[graph_node_output.name]['tf_node'] = tf.cast(
        temp1,
        dtype=output_datatype,
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
                'tf_op_type': 'BlackmanWindow',
                'tf_inputs': {
                    'size': size,
                    'periodic': periodic,
                    'dtype': output_datatype,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
