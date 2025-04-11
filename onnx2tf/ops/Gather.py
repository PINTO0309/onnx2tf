import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tf_keras
import onnx_graphsurgeon as gs
from typing import List
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
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
    """Gather

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
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
        is_bias=True \
            if graph_node.inputs[1].shape is None or len(graph_node.inputs[1].shape) == 1 else False
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    indices = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    axis = graph_node.attrs.get("axis", 0)
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(input_tensor.shape),
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # Param replacement - axis
    axis = replace_parameter(
        value_before_replacement=axis,
        param_target='attributes',
        param_name='axis',
        **kwargs,
    )

    nhwc = tf_layers_dict[graph_node_input_1.name]['nhwc'] \
        if isinstance(graph_node_input_1, gs.Variable) \
            and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False

    before_cast_indices = None
    if isinstance(indices, np.ndarray) and indices.ndim == 1 and len(indices) == 1 and indices[0] is not None:
        if indices[0] >= 0:
            before_cast_indices = indices[0]
            # 直前が Shape だった場合のみの特別なワークアラウンドで、入力がNHWCで確定しているときはindicesを変換する
            # 1. ind=0 のときはそのまま
            # 2. ind=1 のときは末尾
            # 3. ind=2 のときは1
            # 4. ind=3 のときは2
            # 5. ind=4 のときは3
            #
            # ONNX: ind=2
            #   0,1,2 -> 0,2,1
            #   0,1,2,3 -> 0,2,3,1
            #   0,1,2,3,4 -> 0,2,3,4,1
            #   0,1,2,3,4,5 -> 0,2,3,4,5,1
            if nhwc and graph_node.i().op == 'Shape':
                input_tensor_rank = input_tensor.shape[0]
                if before_cast_indices == 0:
                    # batch
                    pass
                elif before_cast_indices == 1:
                    # channel
                    before_cast_indices = input_tensor_rank - 1
                else:
                    # spartial dim
                    before_cast_indices = before_cast_indices - 1

    elif isinstance(indices, np.ndarray) and indices.ndim == 0 and indices is not None:
        if indices >= 0:
            before_cast_indices = int(indices)
            # 直前が Shape だった場合のみの特別なワークアラウンドで、入力がNHWCで確定しているときはindicesを変換する
            # 1. ind=0 のときはそのまま
            # 2. ind=1 のときは末尾
            # 3. ind=2 のときは1
            # 4. ind=3 のときは2
            # 5. ind=4 のときは3
            #
            # ONNX: ind=2
            #   0,1,2 -> 0,2,1
            #   0,1,2,3 -> 0,2,3,1
            #   0,1,2,3,4 -> 0,2,3,4,1
            #   0,1,2,3,4,5 -> 0,2,3,4,5,1
            if nhwc and graph_node.i().op == 'Shape':
                input_tensor_rank = input_tensor.shape[0]
                if before_cast_indices == 0:
                    # batch
                    pass
                elif before_cast_indices == 1:
                    # channel
                    before_cast_indices = input_tensor_rank - 1
                else:
                    # spartial dim
                    before_cast_indices = before_cast_indices - 1

    simple_indices = None
    if isinstance(indices, np.ndarray) and indices.ndim == 1 and None not in indices:
        simple_indices = indices.copy()
    elif isinstance(indices, np.ndarray) and indices.ndim == 0 and indices is not None:
        simple_indices = int(indices)

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    optimization_for_gpu_delegate: bool = \
        kwargs['optimization_for_gpu_delegate']

    # tensorflow gather supports only positive indices
    out_dtype = NUMPY_DTYPES_TO_TF_DTYPES[indices.dtype] \
        if isinstance(indices.dtype, np.dtype) else indices.dtype

    if not optimization_for_gpu_delegate:
        cond = tf.cast(indices < 0, dtype=out_dtype)
        plus_values = cond * tf.shape(input_tensor, out_type=out_dtype)[axis]
        if hasattr(plus_values, "_inferred_value") \
            and plus_values._inferred_value is not None:
            plus_values_inferred_value = np.asarray(plus_values._inferred_value)
            plus_values = plus_values_inferred_value.flatten()
            if None not in plus_values.shape \
                and len(plus_values) != sum([1 if plus_value is not None else 0 for plus_value in plus_values]):
                plus_values = tf.constant([0], dtype=out_dtype)
        data = indices + plus_values
        if data.dtype != out_dtype:
            indices = tf.cast(
                data,
                dtype=out_dtype,
            )
        else:
            indices = data
    else:
        cond = indices < 0
        data = tf.where(
            cond,
            tf.convert_to_tensor(1, dtype=out_dtype),
            tf.convert_to_tensor(0, dtype=out_dtype)
        ) * tf.shape(input_tensor, out_type=out_dtype)[axis]
        if data.dtype != out_dtype:
            indices = indices + tf.cast(
                data,
                dtype=out_dtype,
            )
        else:
            indices = indices + data

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': nhwc,
    }

    # Param replacement
    input_tensor = replace_parameter(
        value_before_replacement=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    indices = replace_parameter(
        value_before_replacement=indices,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    if simple_indices is not None:
        simple_indices = replace_parameter(
            value_before_replacement=simple_indices,
            param_target='inputs',
            param_name=graph_node.inputs[1].name,
            **kwargs,
        )

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Gather + Unsqueeze -> strided_slice
    # Replace combination of Gather and Unsqueeze
    # with strided_slice if available
    consumer_count = 0
    consumer_nodes: List[gs.Node] = []
    while True:
        try:
            consumer_node = graph_node.o(consumer_count, 0)
            consumer_nodes.append(consumer_node)
            consumer_count += 1
        except:
            break
    unsqueeze_count = 0
    for consumer_node in consumer_nodes:
        if consumer_node.op == 'Unsqueeze' \
            and hasattr(consumer_node, 'attrs') \
            and 'axes' in consumer_node.attrs \
            and len(consumer_node.attrs['axes']) == 1 \
            and consumer_node.attrs['axes'][0] == axis:
            unsqueeze_count += 1

    # Complex Gather -> Simple Gather
    # https://github.com/PINTO0309/onnx2tf/issues/261
    simple_gather = False
    if isinstance(simple_indices, np.ndarray) \
        or isinstance(simple_indices, int):

        if isinstance(simple_indices, int):
            simple_gather = True
        elif isinstance(simple_indices, np.ndarray) \
            and simple_indices.ndim == 1:
            int_check_sum = sum(
                [
                    1 for dim in simple_indices \
                        if (isinstance(dim, int) or isinstance(dim, np.int32) or isinstance(dim, np.int64)) and dim >= 0
                ]
            )
            if len(simple_indices) == int_check_sum:
                simple_gather = True
            else:
                simple_gather = False
        else:
            simple_gather = False

    # Generation of TF OP
    tf_type = None
    if isinstance(graph_node_input_1, gs.Variable) \
        and 'simple_resize' in tf_layers_dict[graph_node_input_1.name] \
        and tf_layers_dict[graph_node_input_1.name]['simple_resize'] == True:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.identity(input=input_tensor)
        tf_layers_dict[graph_node_output.name]['simple_resize'] = True
        tf_layers_dict[graph_node_output.name]['simple_resize_shape_op'] = tf_layers_dict[graph_node_input_1.name]['simple_resize_shape_op']
        tf_type = tf.identity

    elif unsqueeze_count > 0 \
        and unsqueeze_count == consumer_count \
        and before_cast_indices is not None:
        # Replace
        ind = before_cast_indices
        begin_ = [
            0 if idx != axis else ind \
                for idx in range(len(input_tensor.shape))
        ]
        end_ = [
            0 if idx != axis else ind + 1 \
                for idx in range(len(input_tensor.shape))
        ]
        begin_mask_ = sum(
            [
                2**idx if idx != axis else 0 \
                    for idx in range(len(input_tensor.shape))
            ]
        )
        end_mask_ = begin_mask_
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.strided_slice(
                input_=input_tensor \
                    if not isinstance(input_tensor, np.ndarray) \
                        else tf.convert_to_tensor(input_tensor),
                begin=begin_,
                end=end_,
                begin_mask=begin_mask_,
                end_mask=end_mask_,
            )
        tf_layers_dict[graph_node_output.name]['unnecessary_gather'] = True
        tf_type = tf.strided_slice

    elif \
        (
            isinstance(simple_indices, np.ndarray) \
                or isinstance(simple_indices, int)
        ) and simple_gather:
        # Complex Gather -> Simple Gather
        # https://github.com/PINTO0309/onnx2tf/issues/261
        # No-replace
        indices_values = indices._inferred_value \
            if hasattr(indices, "_inferred_value") \
                and indices._inferred_value is not None else simple_indices

        # Disable negative indexes
        if isinstance(indices_values, np.ndarray) \
            and None not in indices_values \
            and input_tensor.shape[axis] is not None:
            maximum_number_of_elements = input_tensor.shape[axis]
            indices_values = np.where(
                indices_values < 0,
                indices_values+maximum_number_of_elements,
                indices_values
            )
        elif isinstance(indices_values, int) \
            and indices_values < 0 \
            and input_tensor.shape[axis] is not None:
            maximum_number_of_elements = input_tensor.shape[axis]
            indices_values = indices_values + maximum_number_of_elements

        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.gather(
                params=input_tensor,
                indices=indices_values \
                    if not isinstance(indices_values, np.ndarray) \
                        else tf.convert_to_tensor(indices_values),
                axis=axis,
                name=graph_node.name,
            )
        tf_type = tf.gather

    else:
        # No-replace
        indices_values = indices._inferred_value \
            if hasattr(indices, "_inferred_value") \
                and indices._inferred_value is not None else indices

        # Disable negative indexes
        process_flatten_gather = False
        if isinstance(indices_values, np.ndarray) \
            and None not in indices_values \
            and input_tensor.shape[axis] is not None:
            maximum_number_of_elements = input_tensor.shape[axis]
            indices_values = np.where(
                indices_values < 0,
                indices_values+maximum_number_of_elements,
                indices_values
            )
        elif isinstance(indices_values, int) \
            and indices_values < 0 \
            and input_tensor.shape[axis] is not None:
            maximum_number_of_elements = input_tensor.shape[axis]
            indices_values = indices_values + maximum_number_of_elements
        elif tf_keras.backend.is_keras_tensor(indices_values) \
            and indices_values.shape == tf.TensorShape(None):
            indices_values = tf.reshape(indices_values, [-1])
        # https://github.com/PINTO0309/onnx2tf/issues/751
        elif hasattr(input_tensor, 'shape') \
            and hasattr(indices_values, 'shape') \
            and input_tensor.shape != tf.TensorShape(None) \
            and indices_values.shape != tf.TensorShape(None) \
            and len(input_tensor.shape) > 0 \
            and len(indices_values.shape) > 0 \
            and len(input_tensor.shape) == len(indices_values.shape) \
            and input_tensor.shape[axis] is not None \
            and indices_values.shape[axis] is not None \
            and None in input_tensor.shape \
            and None in indices_values.shape:
                process_flatten_gather = True

        if not process_flatten_gather:
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.gather(
                    params=input_tensor,
                    indices=indices_values \
                        if not isinstance(indices_values, np.ndarray) \
                            else tf.convert_to_tensor(indices_values),
                    axis=axis,
                    name=graph_node.name,
                )
        else:
            # https://github.com/PINTO0309/onnx2tf/issues/751
            # input_tensor.shape: TensorShape([1500, None])
            # indices_values.shape: TensorShape([300, None])
            # axis: 0
            # output.shape: TensorShape([300, None])
            flattened_indices = tf.reshape(indices_values, [-1])  # shape: [?,]
            gathered = tf.gather(input_tensor, flattened_indices, axis=0)  # shape: [?, None]
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.reshape(
                    tensor=gathered,
                    shape=tf.shape(indices_values),
                )
        tf_type = tf.gather

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
                'tf_op_type': tf_type,
                'tf_inputs': {
                    'params': input_tensor,
                    'indices': indices,
                    'axis': axis,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
