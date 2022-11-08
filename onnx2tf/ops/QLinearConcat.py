import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    convert_axis,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """QLinearConcat

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    y_scale_list = [i for i in graph_node.inputs[0::3]]
    y_zero_point_list = [i for i in graph_node.inputs[1::3]]
    input_list = [i for i in graph_node.inputs[2::3]]

    input_tensor_shape = input_list[0].shape
    input_tensor_rank = len(input_tensor_shape)

    before_op_output_shape_trans = True
    for graph_node_input in input_list:
        before_op_output_shape_trans_n = \
            tf_layers_dict.get(graph_node_input.name, {}).get('before_op_output_shape_trans', True)
        before_op_output_shape_trans = \
            before_op_output_shape_trans and before_op_output_shape_trans_n

    got_values = []
    got_y_scale_list = []
    got_y_zero_point_list = []
    for input, y_scale, y_zero_point  in zip(input_list, y_scale_list, y_zero_point_list):
        const_or_var = get_constant_or_variable(
            input,
            before_op_output_shape_trans,
        )
        if isinstance(const_or_var, gs.Variable):
            got_values.append(tf_layers_dict[const_or_var.name]['tf_node'])
        else:
            got_values.append(const_or_var)

        const_or_var = get_constant_or_variable(
            y_scale,
            before_op_output_shape_trans,
        )
        if isinstance(const_or_var, gs.Variable):
            got_y_scale_list.append(tf_layers_dict[const_or_var.name]['tf_node'])
        else:
            got_y_scale_list.append(const_or_var)

        const_or_var = get_constant_or_variable(
            y_zero_point,
            before_op_output_shape_trans,
        )
        if isinstance(const_or_var, gs.Variable):
            got_y_zero_point_list.append(tf_layers_dict[const_or_var.name]['tf_node'])
        else:
            got_y_zero_point_list.append(const_or_var)

    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    axis = graph_node.attrs.get('axis', 0)
    # NCHW->NHWC, NCDHW->NDHWC
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(shape) if shape is not None else input_tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP

    # TensorFlow does not support Concat for scalar values, so convert to tensor
    values = [
        value if len(value.shape) > 0 else tf.reshape(value, [1]) for value in got_values
    ]
    # cast all inputs to float32
    casted_x_list = []
    casted_y_zero_point_list = []
    for x, y_zero_point in zip(values, got_y_zero_point_list):
        casted_x_list.append(tf.cast(x, tf.float32))
        casted_y_zero_point_list.append(tf.cast(y_zero_point, tf.float32))
    # dequantize x with y_scale, y_zero_point
    dequantized_x_list = []
    for x, y_scale, y_zero_point in zip(casted_x_list, got_y_scale_list, casted_y_zero_point_list):
        dequantized_value = tf.add(
            x=tf.divide(
                x=x,
                y=y_scale,
            ),
            y=y_zero_point,
        )
        dequantized_x_list.append(dequantized_value)

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.concat(
            values=dequantized_x_list,
            axis=axis,
            name=graph_node.name,
        )

    # Generation of Debug Info
    tf_inputs = {f"input{idx}": dequantized_x for idx, dequantized_x in enumerate(dequantized_x_list)}
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
