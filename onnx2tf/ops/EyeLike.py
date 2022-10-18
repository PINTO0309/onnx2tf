import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
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
    """EyeLike

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

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    input_tensor_shape = tf.shape(input_tensor)

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    dtype = graph_node.attrs.get('dtype', 1)
    offset = graph_node.attrs.get('k', 0)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    if None not in input_tensor_shape:
        max_eye_shape_ub = input_tensor_shape[1] \
            if offset > 0 else input_tensor_shape[0]
        max_eye_shape_lb = input_tensor_shape[0] \
            if offset > 0 else input_tensor_shape[1]
        offset = max_eye_shape_ub * np.sign(offset) \
            if abs(offset) > max_eye_shape_ub else offset
        abs_offset = abs(offset)
        eye_shape = min(max_eye_shape_ub - abs_offset, max_eye_shape_lb)
        tensor = tf.eye(
            eye_shape,
            num_columns=eye_shape,
            dtype=dtype,
        )
        if offset > 0:
            tb_paddings = [
                0,
                input_tensor_shape[0] - eye_shape
            ]
            lr_paddings = [
                offset,
                input_tensor_shape[1] - offset - eye_shape
            ]
        else:
            tb_paddings = [
                abs_offset,
                input_tensor_shape[0] - abs_offset - eye_shape
            ]
            lr_paddings = [
                0,
                input_tensor_shape[1] - eye_shape
            ]
        paddings = tf.constant([tb_paddings, lr_paddings], dtype=tf.int32)

        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.pad(
                tensor=tensor,
                paddings=paddings,
                name=graph_node.name,
            )
    else:
        def create_nodes(inp, offset):
            new_shape = tf.shape(inp, out_type=tf.int32)
            max_eye_shape_ub = new_shape[1] \
                if offset > 0 else new_shape[0]
            max_eye_shape_lb = new_shape[0] \
                if offset > 0 else new_shape[1]
            offset = max_eye_shape_ub * np.sign(offset) \
                if abs(offset) > max_eye_shape_ub else offset
            abs_offset = abs(offset)
            eye_shape = tf.minimum(
                max_eye_shape_ub - abs_offset,
                max_eye_shape_lb,
            )
            tensor = tf.eye(
                eye_shape,
                num_columns=eye_shape,
                dtype=dtype,
            )
            if offset > 0:
                tb_paddings = [
                    0,
                    new_shape[0] - eye_shape
                ]
                lr_paddings = [
                    offset,
                    new_shape[1] - offset - eye_shape
                ]
            else:
                tb_paddings = [
                    abs_offset,
                    new_shape[0] - abs_offset - eye_shape
                ]
                lr_paddings = [
                    0,
                    new_shape[1] - eye_shape
                ]
            paddings = [tb_paddings, lr_paddings]
            return tensor, paddings

        tensor, paddings = create_nodes(input_tensor, offset)

        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.pad(
                tensor=tensor,
                paddings=paddings,
                name=graph_node.name,
            )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'EyeLike',
                'tf_inputs': {
                    'tensor': input_tensor,
                    'dtype': dtype,
                    'offset': offset,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
