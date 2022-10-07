import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """LSTM

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_2 = \
        tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_3 = \
        tf_layers_dict.get(graph_node.inputs[2].name, {}).get('before_op_output_shape_trans', True)

    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2 \
        and before_op_output_shape_trans_3

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans,
    )

    graph_node_input_4 = None
    if len(graph_node.inputs) >= 4:
        graph_node_input_4 = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
        )
    graph_node_input_5 = None
    if len(graph_node.inputs) >= 5:
        graph_node_input_5 = get_constant_or_variable(
            graph_node.inputs[4],
            before_op_output_shape_trans,
        )
    graph_node_input_6 = None
    if len(graph_node.inputs) >= 6:
        graph_node_input_6 = get_constant_or_variable(
            graph_node.inputs[5],
            before_op_output_shape_trans,
        )
    graph_node_input_7 = None
    if len(graph_node.inputs) >= 7:
        graph_node_input_7 = get_constant_or_variable(
            graph_node.inputs[6],
            before_op_output_shape_trans,
        )
    graph_node_input_8 = None
    if len(graph_node.inputs) >= 8:
        graph_node_input_8 = get_constant_or_variable(
            graph_node.inputs[7],
            before_op_output_shape_trans,
        )

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    X = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    X_rank = len(X.shape)
    W = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    R = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    B = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4

    sequence_lens = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) else graph_node_input_5
    initial_h = tf_layers_dict[graph_node_input_6.name]['tf_node'] \
        if isinstance(graph_node_input_6, gs.Variable) else graph_node_input_6
    initial_c = tf_layers_dict[graph_node_input_7.name]['tf_node'] \
        if isinstance(graph_node_input_7, gs.Variable) else graph_node_input_7
    P = tf_layers_dict[graph_node_input_8.name]['tf_node'] \
        if isinstance(graph_node_input_8, gs.Variable) else graph_node_input_8



    activation_alpha = graph_node.attrs.get('activation_alpha', None)
    activation_beta = graph_node.attrs.get('activation_beta', None)
    activations = graph_node.attrs.get('activations', None)
    clip = graph_node.attrs.get('clip', None)
    direction = graph_node.attrs.get('direction', 'forward')
    hidden_size = graph_node.attrs.get('hidden_size', None)
    input_forget = graph_node.attrs.get('input_forget', 0)
    layout = graph_node.attrs.get('layout', 0)





    # if isinstance(axes, list) or (isinstance(axes, np.ndarray) and len(axes.shape) > 0):
    #     axes = [
    #         convert_axis(
    #             axis=idx,
    #             tensor_rank=tensor_rank,
    #             before_op_output_shape_trans=before_op_output_shape_trans,
    #         ) for idx in axes
    #     ]
    # elif axes is not None and isinstance(axes, np.ndarray) and len(axes.shape) == 0:
    #     axes = convert_axis(
    #         axis=axes,
    #         tensor_rank=tensor_rank,
    #         before_op_output_shape_trans=before_op_output_shape_trans,
    #     )

    # # Preserving Graph Structure (Dict)
    # tf_layers_dict[graph_node_output.name] = {
    #     'optype': graph_node.op,
    #     'shape': shape,
    #     'dtype': dtype,
    # }

    # # Generation of TF OP
    # tf_layers_dict[graph_node_output.name]['tf_node'] = \
    #     tf.squeeze(
    #         input=input_tensor,
    #         axis=list(axes) if axes is not None else None,
    #         name=graph_node.name,
    #     )
