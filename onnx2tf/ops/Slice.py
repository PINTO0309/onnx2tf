import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
)


@print_node_info
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
    input_tensor = get_constant_or_variable(graph_node.inputs[0])
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)
    starts = get_constant_or_variable(graph_node.inputs[1])
    ends = get_constant_or_variable(graph_node.inputs[2])
    axes = None
    if len(graph_node.inputs) >= 4:
        axes = get_constant_or_variable(graph_node.inputs[3])
        if input_tensor_rank <= 3:
            axes = tf.range(tf.shape(starts)[0], dtype=ends.dtype)
    steps = None
    if len(graph_node.inputs) >= 5:
        steps = get_constant_or_variable(graph_node.inputs[4])

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[input_tensor.name]['tf_node'] \
        if isinstance(input_tensor, gs.Variable) else input_tensor
    starts = tf_layers_dict[starts.name]['tf_node'] \
        if isinstance(starts, gs.Variable) else starts
    ends = tf_layers_dict[ends.name]['tf_node'] \
        if isinstance(ends, gs.Variable) else ends
    axes = tf_layers_dict[axes.name]['tf_node'] \
        if isinstance(axes, gs.Variable) else axes
    steps = tf_layers_dict[steps.name]['tf_node'] \
        if isinstance(steps, gs.Variable) else steps

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    is_axes_negative = tf.less(
        axes,
        tf.zeros_like(axes),
    )
    axes = tf.where(
        is_axes_negative,
        axes + tf.cast(tf.rank(input_tensor), axes.dtype),
        axes,
    )

    # expand a dimension of 1 at the end
    sparse_indices = tf.cast(
        tf.expand_dims(axes, -1),
        tf.int64,
    )

    # build the indexed dimension sizes as sparse_shape
    sparse_shape = tf.gather_nd(
        params=input_tensor_shape,
        indices=sparse_indices,
    )
    sparse_shape = tf.cast(
        sparse_shape,
        ends.dtype,
    )

    # take care of starts, ends that are larger than the dim size.
    starts_min = tf.minimum(
        starts,
        sparse_shape,
    )
    ends_min = tf.minimum(
        ends,
        sparse_shape,
    )

    # take care of starts, ends that are negative
    is_starts_negative = tf.less(
        starts_min,
        tf.zeros_like(starts_min),
    )
    starts_final = tf.where(
        is_starts_negative,
        starts_min + sparse_shape,
        starts_min,
    )
    is_ends_negative = tf.less(
        ends_min,
        tf.zeros_like(ends_min),
    )
    ends_final = tf.where(
        is_ends_negative,
        ends_min + sparse_shape,
        ends_min,
    )

    # need to densify everything for the inputs to slice
    # the output shape is the input_tensor rank
    output_shape = tf.reshape(
        tf.rank(input_tensor),
        [1],
    )
    output_shape = tf.cast(
        output_shape,
        tf.int64,
    )

    # create dense tensor, pad 0 as default begins
    dense_begins = tf.sparse.to_dense(
        tf.sparse.SparseTensor(
            sparse_indices,
            starts_final,
            output_shape,
        )
    )

    # create dense tensor, pad -1 for next step
    dense_ends = tf.sparse.SparseTensor(
        sparse_indices,
        ends_final,
        output_shape,
    )
    dense_ends = tf.sparse.to_dense(
        dense_ends,
        default_value=tf.constant(-1, dtype=dense_begins.dtype)
    )
    dense_ends = tf.where(
        tf.equal(dense_ends, tf.constant(-1, dtype=dense_begins.dtype)),
        input_tensor_shape,
        dense_ends,
    )

    # create dense tensor for steps if not already so
    if len(graph_node.inputs) >= 5:
        dense_steps = tf.sparse.SparseTensor(
            sparse_indices,
            steps,
            output_shape
        )
        dense_steps = tf.sparse.to_dense(
            dense_steps,
            default_value=tf.constant(1, dtype=steps.dtype)
        )
    else:
        dense_steps = tf.ones(tf.shape(input_tensor_shape), ends.dtype)

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.strided_slice(
            input_=input_tensor,
            begin=dense_begins,
            end=dense_ends,
            strides=dense_steps,
            name=graph_node.name,
        )
