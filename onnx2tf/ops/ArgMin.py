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
    **kwargs: dict,
):
    """ArgMin

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
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    axis = graph_node.attrs.get('axis', 0)
    # NCHW->NHWC, NCDHW->NDHWC
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(graph_node_input.shape),
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # 0: False, 1: True
    keepdims = bool(graph_node.attrs.get('keepdims', 0))

    # 0: False, 1: True
    select_last_index = bool(graph_node.attrs.get('select_last_index', 0))

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    reversed_tensor = None
    if not select_last_index:
        reversed_tensor = input_tensor
    else:
        reversed_tensor = \
            tf.reverse(
                tensor=input_tensor,
                axis=axis,
                name=f'{graph_node.name}_reverse',
            )

    final_tensor = None
    argmined_tensor = tf.math.argmin(
        input=reversed_tensor,
        axis=axis,
        output_type=NUMPY_DTYPES_TO_TF_DTYPES[dtype] \
            if isinstance(dtype, np.dtype) else dtype,
        name=f'{graph_node.name}_argmin',
    )
    if keepdims:
        final_tensor = \
            tf.expand_dims(
                input=argmined_tensor,
                axis=axis,
                name=f'{graph_node.name}_expand_dims',
            )
    else:
        final_tensor = argmined_tensor

    tf_layers_dict[graph_node_output.name]['tf_node'] = final_tensor

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
                'tf_op_type': tf.math.argmin,
                'tf_inputs': {
                    'input': input_tensor,
                    'axis': axis,
                    'output_type': dtype,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
