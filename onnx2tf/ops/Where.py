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
    nhwc_determination_of_output_value_of_binary_input_op,
    transpose_with_flexing_deterrence,
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
    """Where

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

    values = [
        tf_layers_dict[graph_node_input_1.name]['tf_node'] \
            if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1,
        tf_layers_dict[graph_node_input_2.name]['tf_node'] \
            if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2,
        tf_layers_dict[graph_node_input_3.name]['tf_node'] \
            if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3,
    ]

    # Shape Unmatched Special Avoidance Workaround
    # At least one True value for same_input_shape_as_onnx
    # At least one True value in nhwc_flags
    # same_input_shape_as_onnx == True and nhwc_flags == False and 3D or 4D or 5D tensor is NHWC transposed
    if hasattr(values[0].shape, '__len__') \
        and hasattr(values[1].shape, '__len__') \
        and hasattr(values[2].shape, '__len__') \
        and len(values[0].shape) == len(values[1].shape) == len(values[2].shape):

        nhwc_flags = []
        same_input_shape_as_onnxs = []

        def nhwc_judge(const_or_var, graph_node_input):
            if isinstance(const_or_var, gs.Variable):
                nhwc_flags.append(
                    tf_layers_dict[const_or_var.name]['nhwc'] \
                        if 'nhwc' in tf_layers_dict[const_or_var.name].keys() else False
                )
                same_input_shape_as_onnxs.append(
                    True if graph_node_input.shape is not None and len(graph_node_input.shape) > 0 \
                        and graph_node_input.shape == tf_layers_dict[const_or_var.name]['tf_node'].shape else False
                )
            else:
                nhwc_flags.append(False)
                same_input_shape_as_onnxs.append(
                    True if graph_node_input.shape is not None and len(graph_node_input.shape) > 0 \
                        and graph_node_input.shape == const_or_var.shape else False
                )

        nhwc_judge(graph_node_input_1, graph_node.inputs[0])
        nhwc_judge(graph_node_input_2, graph_node.inputs[1])
        nhwc_judge(graph_node_input_3, graph_node.inputs[2])

        if True in same_input_shape_as_onnxs and True in nhwc_flags:
            before_op_output_shape_trans = True
            for idx, (same_input_shape_as_onnx, nhwc_flag, value) in enumerate(zip(same_input_shape_as_onnxs, nhwc_flags, values)):
                if same_input_shape_as_onnx and not nhwc_flag:
                    if len(value.shape) == 3:
                        value = \
                            transpose_with_flexing_deterrence(
                                input_tensor=value,
                                perm=[0,2,1],
                                **kwargs,
                            )
                    elif len(value.shape) == 4:
                        value = \
                            transpose_with_flexing_deterrence(
                                input_tensor=value,
                                perm=[0,2,3,1],
                                **kwargs,
                            )
                    elif len(value.shape) == 5:
                        value = \
                            transpose_with_flexing_deterrence(
                                input_tensor=value,
                                perm=[0,2,3,4,1],
                                **kwargs,
                            )
                    else:
                        pass
                else:
                    pass
                values[idx] = value

    condition = values[0]
    X = values[1]
    Y = values[2]

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': \
            nhwc_determination_of_output_value_of_binary_input_op(
                graph_node_input_1=graph_node_input_1,
                graph_node_input_2=graph_node_input_2,
                tf_layers_dict=tf_layers_dict
            ) \
            or \
            nhwc_determination_of_output_value_of_binary_input_op(
                graph_node_input_1=graph_node_input_2,
                graph_node_input_2=graph_node_input_3,
                tf_layers_dict=tf_layers_dict
            )
    }

    # Pre-process transpose
    condition = pre_process_transpose(
        value_before_transpose=condition,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    X = pre_process_transpose(
        value_before_transpose=X,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    Y = pre_process_transpose(
        value_before_transpose=Y,
        param_target='inputs',
        param_name=graph_node.inputs[2].name,
        **kwargs,
    )

    # Generation of TF OP
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.where(
            condition=condition,
            x=X,
            y=Y,
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
                'tf_op_type': tf.where,
                'tf_inputs': {
                    'condition': condition,
                    'x': X,
                    'y': Y,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
