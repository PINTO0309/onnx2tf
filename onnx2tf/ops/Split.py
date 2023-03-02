import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from typing import List
from onnx2tf.utils.common_functions import (
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    convert_axis,
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
    """Split

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = before_op_output_shape_trans_1

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    input_tensor_rank = len(graph_node_input_1.shape) \
        if graph_node_input_1.shape is not None \
            else len(tf_layers_dict[graph_node_input_1.name]['tf_node'].shape)

    graph_node_input_2 = None
    if len(graph_node.inputs) >= 2:
        graph_node_input_2 = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
    # graph_node_output: gs.Variable = graph_node.outputs[0]
    graph_node_outputs: List[gs.Variable] = [
        graph_node_output for graph_node_output in graph_node.outputs
    ]

    shape = graph_node_outputs[0].shape
    dtype = graph_node_outputs[0].dtype

    axis = graph_node.attrs.get('axis', 0)
    # NCHW->NHWC, NCDHW->NDHWC
    axis = convert_axis(
        axis=axis,
        tensor_rank=input_tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    num_outputs = graph_node.attrs.get('num_outputs', None)

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1

    split = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    if split is not None and split.shape is None:
        split = len(graph_node_outputs)
    if split is None:
        split = len(graph_node_outputs)
    split = graph_node.attrs.get('split', split)

    for graph_node_output in graph_node_outputs:
        # Preserving Graph Structure (Dict)
        tf_layers_dict[graph_node_output.name] = {
            'optype': graph_node.op,
            'shape': shape,
            'dtype': dtype,
        }

    # Param replacement
    axis = replace_parameter(
        value_before_replacement=axis,
        param_target='attributes',
        param_name='axis',
        **kwargs,
    )
    num_outputs = replace_parameter(
        value_before_replacement=num_outputs,
        param_target='attributes',
        param_name='num_outputs',
        **kwargs,
    )
    split = replace_parameter(
        value_before_replacement=split,
        param_target='inputs',
        param_name='split',
        **kwargs,
    )

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    splited_tensors = None
    if (
            (isinstance(split, int) and split == 1) \
                or \
            (isinstance(split, np.ndarray) and len(list(split)) == 1 and split[0] == 1)
        ) \
        and isinstance(input_tensor.shape[axis], int) \
        and input_tensor.shape[axis] == 1:
        # Disable unnecessary splits
        splited_tensors = \
            [
                tf.identity(
                    input=input_tensor,
                    name=graph_node.name,
                )
            ]
    elif isinstance(split, np.ndarray) \
        and len(list(split)) > 1 \
        and np.prod(split) == 1 \
        and isinstance(input_tensor.shape[axis], int) \
        and input_tensor.shape[axis] == len(list(split)):
        # strided_slice - Slice everything in size 1
        # Suppression of FlexSplitV generation
        splited_tensors = []
        for split_idx in range(len(list(split))):
            begin_ = [
                split_idx if idx == axis else 0 for idx in range(input_tensor_rank)
            ]
            end_ = []
            for idx in range(input_tensor_rank):
                if idx == axis:
                    end_.append(split_idx + 1)
                elif input_tensor.shape[idx] is None:
                    end_.append(-1)
                else:
                    end_.append(input_tensor.shape[idx])

            splited_tensors.append(
                tf.strided_slice(
                    input_=input_tensor,
                    begin=begin_,
                    end=end_,
                )
            )
    else:
        splited_tensors = \
            tf.split(
                value=input_tensor,
                num_or_size_splits=split,
                axis=axis,
                num=num_outputs,
                name=graph_node.name,
            )
    for splited_tensor, graph_node_output in zip(splited_tensors, graph_node_outputs):
        tf_layers_dict[graph_node_output.name]['tf_node'] = splited_tensor
        # Post-process transpose
        tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
            value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
            param_target='outputs',
            param_name=graph_node_output.name,
            **kwargs,
        )

    # Generation of Debug Info
    tf_outputs = {f"output{idx}": value for idx, value in enumerate(splited_tensors)}
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.split,
                'tf_inputs': {
                    'value': input_tensor,
                    'num_or_size_splits': split,
                    'axis': axis,
                    'num': num_outputs,
                },
                'tf_outputs': tf_outputs,
            }
        )
