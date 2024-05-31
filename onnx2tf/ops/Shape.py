import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from typing import List
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    convert_axis,
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
    """Shape

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
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    nhwc = tf_layers_dict[graph_node_input.name]['nhwc'] \
        if isinstance(graph_node_input, gs.Variable) \
            and 'nhwc' in tf_layers_dict[graph_node_input.name].keys() else False

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': nhwc,
    }

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Generation of TF OP
    input_tensor_shape = input_tensor.shape
    tensor_rank = len(input_tensor_shape)

    start = graph_node.attrs.get('start', None)
    if start is not None:
        start = convert_axis(
            axis=start,
            tensor_rank=tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        )
        if start < 0:
            start += tensor_rank
            # Clip if start is still < 0
            start = 0 if start < 0 else start

    end = graph_node.attrs.get('end', None)
    if end is not None:
        end = convert_axis(
            axis=end,
            tensor_rank=tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        )
        if end < 0:
            end += tensor_rank
            # Clip if end is still < 0
            end = 0 if end < 0 else end

    out_dtype = NUMPY_DTYPES_TO_TF_DTYPES[dtype] \
        if isinstance(dtype, np.dtype) else dtype
    if start is not None and end is not None:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.slice(
                tf.shape(
                    input=input_tensor,
                    out_type=out_dtype,
                    name=graph_node.name,
                ),
                [start],
                [end - start],
            )
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.shape(
                input=input_tensor,
                out_type=out_dtype,
                name=graph_node.name,
            )

    # Simple Resize
    simple_resize = False
    if nhwc:
        # 1. Gather -> Unsqueeze -> Concat -> Resize
        consumer_count = 0
        consumer_nodes: List[gs.Node] = []
        while True:
            try:
                consumer_node = graph_node.o(consumer_count, 0)
                consumer_nodes.append(consumer_node)
                consumer_count += 1
            except:
                break
        gather_count = 0
        gather_nodes: List[gs.Node] = []
        if consumer_count == 2:
            for consumer_node in consumer_nodes:
                if consumer_node.op == 'Gather':
                    gather_nodes.append(consumer_node)
                    gather_count += 1
        if gather_count == 2:
            consumer_count_total = 0
            consumer_nodes: List[gs.Node] = []
            for gather_node in gather_nodes:
                consumer_count = 0
                while True:
                    try:
                        consumer_node = gather_node.o(consumer_count, 0)
                        consumer_nodes.append(consumer_node)
                        consumer_count += 1
                    except:
                        break
                consumer_count_total += consumer_count
            unsqueeze_count = 0
            unsqueeze_nodes: List[gs.Node] = []
            if consumer_count_total == 2:
                for consumer_node in consumer_nodes:
                    if consumer_node.op == 'Unsqueeze':
                        unsqueeze_nodes.append(consumer_node)
                        unsqueeze_count += 1
                if unsqueeze_count == 2:
                    consumer_count_total = 0
                    consumer_nodes: List[gs.Node] = []
                    for unsqueeze_node in unsqueeze_nodes:
                        consumer_count = 0
                        while True:
                            try:
                                consumer_node = unsqueeze_node.o(consumer_count, 0)
                                consumer_nodes.append(consumer_node)
                                consumer_count += 1
                            except:
                                break
                        consumer_count_total += consumer_count
                    if consumer_count_total == 2:
                        target_concat_node: gs.Node = None
                        simple_resize_concat: bool = True
                        for consumer_node in consumer_nodes:
                            if target_concat_node is None \
                                and consumer_node.op == 'Concat':
                                target_concat_node = consumer_node
                                simple_resize_concat = simple_resize_concat and True
                            elif target_concat_node is not None \
                                and consumer_node.op == 'Concat' \
                                and consumer_node.name == target_concat_node.name:
                                simple_resize_concat = simple_resize_concat and True
                            else:
                                simple_resize_concat = simple_resize_concat and False
                        if simple_resize_concat:
                            consumer_count = 0
                            consumer_nodes: List[gs.Node] = []
                            while True:
                                try:
                                    consumer_node: gs.Node = target_concat_node.o(consumer_count, 0)
                                    if consumer_node.op == 'Resize':
                                        consumer_nodes.append(consumer_node)
                                        consumer_count += 1
                                except:
                                    break
                            if consumer_count == 1:
                                simple_resize = True
                                tf_layers_dict[graph_node_output.name]['simple_resize'] = True
                                tf_layers_dict[graph_node_output.name]['simple_resize_shape_op'] = tf_layers_dict[graph_node_output.name]['tf_node']

        if not simple_resize:
            # 2. Slice -> Concat -> Resize
            consumer_count = 0
            consumer_nodes: List[gs.Node] = []
            while True:
                try:
                    consumer_node = graph_node.o(consumer_count, 0)
                    consumer_nodes.append(consumer_node)
                    consumer_count += 1
                except:
                    break
            slice_count = 0
            slice_nodes: List[gs.Node] = []
            if consumer_count == 1:
                for consumer_node in consumer_nodes:
                    if consumer_node.op == 'Slice':
                        slice_nodes.append(consumer_node)
                        slice_count += 1
            if slice_count == 1:
                consumer_count_total = 0
                consumer_nodes: List[gs.Node] = []
                for slice_node in slice_nodes:
                    consumer_count = 0
                    while True:
                        try:
                            consumer_node = slice_node.o(consumer_count, 0)
                            consumer_nodes.append(consumer_node)
                            consumer_count += 1
                        except:
                            break
                    consumer_count_total += consumer_count
                if consumer_count_total == 1:
                    target_concat_node: gs.Node = None
                    simple_resize_concat: bool = True
                    for consumer_node in consumer_nodes:
                        if target_concat_node is None \
                            and consumer_node.op == 'Concat':
                            target_concat_node = consumer_node
                            simple_resize_concat = simple_resize_concat and True
                        elif target_concat_node is not None \
                            and consumer_node.op == 'Concat' \
                            and consumer_node.name == target_concat_node.name:
                            simple_resize_concat = simple_resize_concat and True
                        else:
                            simple_resize_concat = simple_resize_concat and False
                    if simple_resize_concat:
                        consumer_count = 0
                        consumer_nodes: List[gs.Node] = []
                        while True:
                            try:
                                consumer_node: gs.Node = target_concat_node.o(consumer_count, 0)
                                if consumer_node.op == 'Resize':
                                    consumer_nodes.append(consumer_node)
                                    consumer_count += 1
                            except:
                                break
                        if consumer_count == 1:
                            simple_resize = True
                            tf_layers_dict[graph_node_output.name]['simple_resize2'] = True
                            tf_layers_dict[graph_node_output.name]['simple_resize_shape_op'] = tf_layers_dict[graph_node_output.name]['tf_node']

    # if simple_resize:
    #     tf_layers_dict[graph_node_output.name]['simple_resize'] = True
    #     tf_layers_dict[graph_node_output.name]['simple_resize_shape_op'] = tf_layers_dict[graph_node_output.name]['tf_node']


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
                'tf_op_type': tf.shape,
                'tf_inputs': {
                    'x': input_tensor,
                    'out_type': dtype,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
