import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    convert_axis,
)
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Slice

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_2 = True
    if len(graph_node.inputs) >= 2:
        before_op_output_shape_trans_2 = \
            tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_3 = True
    if len(graph_node.inputs) >= 3:
        before_op_output_shape_trans_3 = \
            tf_layers_dict.get(graph_node.inputs[2].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_4 = True
    if len(graph_node.inputs) >= 4:
        before_op_output_shape_trans_4 = \
            tf_layers_dict.get(graph_node.inputs[3].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_5 = True
    if len(graph_node.inputs) >= 5:
        before_op_output_shape_trans_5 = \
            tf_layers_dict.get(graph_node.inputs[4].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2 \
        and before_op_output_shape_trans_3 \
        and before_op_output_shape_trans_4 \
        and before_op_output_shape_trans_5

    input_tensor = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    input_tensor = tf_layers_dict[input_tensor.name]['tf_node'] \
        if isinstance(input_tensor, gs.Variable) else input_tensor
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    starts = None
    if len(graph_node.inputs) >= 2:
        starts = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
        starts = tf_layers_dict[starts.name]['tf_node'] \
            if isinstance(starts, gs.Variable) else starts

    ends = None
    if len(graph_node.inputs) >= 3:
        ends = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )
        ends = tf_layers_dict[ends.name]['tf_node'] \
            if isinstance(ends, gs.Variable) else ends

    axes = None
    if len(graph_node.inputs) >= 4:
        axes = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
        )
    axes = tf_layers_dict[axes.name]['tf_node'] \
        if isinstance(axes, gs.Variable) else axes
    if isinstance(axes, np.ndarray):
        axes = axes if len(graph_node.inputs) >= 4 else tf.range(tf.shape(starts)[0], dtype=ends.dtype)
    elif isinstance(axes, list):
        axes = np.asarray(axes, dtype=ends.dtype) if len(graph_node.inputs) >= 4 else tf.range(tf.shape(starts)[0], dtype=ends.dtype)
    elif axes is not None:
        axes = axes if len(graph_node.inputs) >= 4 else tf.range(tf.shape(starts)[0], dtype=ends.dtype)

    steps = None
    if len(graph_node.inputs) >= 5:
        steps = get_constant_or_variable(
            graph_node.inputs[4],
            before_op_output_shape_trans,
        )
    steps = tf_layers_dict[steps.name]['tf_node'] \
        if isinstance(steps, gs.Variable) else steps
    if isinstance(steps, np.ndarray):
        steps = tf.constant(steps, dtype=NUMPY_DTYPES_TO_TF_DTYPES[steps.dtype])

    axes = graph_node.attrs.get('axes', axes)

    if isinstance(axes, list) or (isinstance(axes, np.ndarray) and len(axes.shape) > 0):
        axes = [
            convert_axis(
                axis=idx,
                tensor_rank=input_tensor_rank,
                before_op_output_shape_trans=before_op_output_shape_trans,
            ) for idx in axes
        ]
    elif axes is not None and isinstance(axes, np.ndarray) and len(axes.shape) == 0:
        axes = convert_axis(
            axis=axes,
            tensor_rank=input_tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        )
    if isinstance(axes, list):
        axes = np.asarray(axes)

    starts = graph_node.attrs.get('starts', starts)
    if isinstance(starts, list):
        starts = np.asarray(starts)
    ends = graph_node.attrs.get('ends', ends)
    if isinstance(ends, list):
        ends = np.asarray(ends)

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    if None not in input_tensor_shape:
        # first of all, get the input tensor shape
        if axes is not None:
            is_axes_negative = tf.less(axes, tf.zeros_like(axes))
            axes = tf.where(is_axes_negative, axes + tf.cast(input_tensor_rank, axes.dtype), axes)
        else:
            axes = [i for i in range(input_tensor_rank)]

        # expand a dimension of 1 at the end
        sparse_indices = tf.cast(tf.expand_dims(axes, -1), tf.int64)

        # build the indexed dimension sizes as sparse_shape
        sparse_shape = tf.gather_nd(
            params=input_tensor_shape,
            indices=sparse_indices,
        )
        sparse_shape = tf.cast(sparse_shape, tf.int64)

        # take care of starts, ends that are larger than the dim size.
        starts_min = tf.minimum(starts, sparse_shape)
        ends_min = tf.minimum(ends, sparse_shape)

        # need to densify everything for the inputs to slice
        # the output shape is the input_tensor rank
        output_shape = tf.reshape(input_tensor_rank, [1])
        output_shape = tf.cast(output_shape, tf.int64)

        def tf_SparseTensor(sparse_indices, pos, sparse_shape, start_or_end):
            dense = None
            # take care of starts, ends that are negative
            if start_or_end == 'start':
                is_starts_negative = tf.less(pos, tf.zeros_like(pos))
                final = tf.where(is_starts_negative, pos + sparse_shape, pos)
                sparse_tensor = tf.sparse.reorder(
                    sp_input=tf.sparse.SparseTensor(
                        indices=sparse_indices,
                        values=final,
                        dense_shape=sparse_shape,
                    )
                )
                dense = tf.sparse.to_dense(sparse_tensor)
            elif start_or_end == 'end':
                is_ends_negative = tf.less(pos, tf.zeros_like(pos))
                final = tf.where(is_ends_negative, pos + sparse_shape, pos)
                sparse_tensor = tf.sparse.reorder(
                    sp_input=tf.sparse.SparseTensor(
                        indices=sparse_indices,
                        values=final,
                        dense_shape=sparse_shape,
                    )
                )
                dense_ends = tf.sparse.to_dense(
                    sparse_tensor,
                    default_value=tf.constant(-1, dtype=dense_begins.dtype)
                )
                dense = tf.where(
                    tf.equal(dense_ends, tf.constant(-1, dtype=dense_begins.dtype)),
                    input_tensor_shape,
                    dense_ends,
                )
            return dense

        # create dense tensor, pad 0 as default begins
        dense_begins = Lambda(
            tf_SparseTensor,
            arguments={
                'pos': starts_min,
                'sparse_shape': output_shape,
                'start_or_end': 'start',
            }
        )(sparse_indices)
        # create dense tensor, pad -1 for next step
        dense_ends = Lambda(
            tf_SparseTensor,
            arguments={
                'pos': ends_min,
                'sparse_shape': output_shape,
                'start_or_end': 'end'
            }
        )(sparse_indices)

        # create dense tensor for steps if not already so
        if steps is not None:
            dense_steps = tf.sparse.reorder(
                sp_input=tf.sparse.SparseTensor(
                    sparse_indices,
                    steps,
                    output_shape,
                )
            )
            dense_steps = tf.sparse.to_dense(
                dense_steps,
                default_value=tf.constant(1, dtype=steps.dtype)
            )
        else:
            dense_steps = tf.ones(input_tensor_rank, ends.dtype)

        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.strided_slice(
                input_=input_tensor,
                begin=dense_begins,
                end=dense_ends,
                strides=tf.cast(dense_steps, dtype=dense_begins.dtype),
            )
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.strided_slice(
                input_=input_tensor,
                begin=starts,
                end=ends,
                strides=steps,
                name=graph_node.name,
            )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.strided_slice,
                'tf_inputs': {
                    'input_': input_tensor,
                    'begin': starts,
                    'end': ends,
                    'strides': steps,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
