import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from tensorflow.python.keras.layers import (
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
)
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
)

INF_INDEX_VALUE: int = 4294967296


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """GlobalAveragePool

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
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Workaround to avoid as many conversion failures as possible
    # for models with useless Transpose immediately before them.
    # If the input geometry of the ONNX and the input geometry of the TF model match,
    # the input geometry on the TF model side is forcibly transposed to the NWC or NHWC or NDHWC format.
    # However, if all dimensions of CW or CHW or CDHW have the same value,
    # the forced transposition process is skipped because it may destroy the structure of the model.
    onnx_input_shape = [
        dim if isinstance(dim, int) else None for dim in graph_node.inputs[0].shape
    ] if graph_node.inputs[0].shape is not None else None
    tf_input_shape = [
        dim if isinstance(dim, int) else None for dim in input_tensor_shape
    ]
    if onnx_input_shape is not None \
        and len(onnx_input_shape) > 1 and len(tf_input_shape) > 1 \
        and onnx_input_shape == tf_input_shape:

        shape_for_judging_skip = [
            dim if dim is not None else INF_INDEX_VALUE for dim in onnx_input_shape[1:]
        ]
        if shape_for_judging_skip.count(shape_for_judging_skip[0]) != len(shape_for_judging_skip):
            if len(onnx_input_shape) == 3:
                # 1D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,1],
                    **kwargs,
                )
            elif len(onnx_input_shape) == 4:
                # 2D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,1],
                    **kwargs,
                )
            elif len(onnx_input_shape) == 5:
                # 3D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,4,1],
                    **kwargs,
                )

    # Generation of TF OP
    axis = [dim for dim in range(1, input_tensor_rank - 1)]
    if len(axis) == 1:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            GlobalAveragePooling1D(
                keepdims=True,
            )(input_tensor)
    elif len(axis) == 2:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            GlobalAveragePooling2D(
                keepdims=True,
            )(input_tensor)
    elif len(axis) == 3:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            GlobalAveragePooling3D(
                keepdims=True,
            )(input_tensor)
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.reduce_mean(
                input_tensor=input_tensor,
                axis=axis,
                keepdims=True,
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
                'tf_op_type': tf.reduce_mean,
                'tf_inputs': {
                    'input_tensor': input_tensor,
                    'axis': axis,
                    'keepdims': True,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
