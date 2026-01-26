import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)
from typing import Any


def _make_coords(
    size_dim: Any,
    align_corners: bool,
    dtype: Any,
) -> Any:
    size_dim = tf.cast(size_dim, tf.int32)
    size_f = tf.cast(size_dim, dtype)

    if align_corners:
        denom = size_f - tf.constant(1.0, dtype=dtype)
        step = tf.where(
            condition=size_dim > 1,
            x=tf.constant(2.0, dtype=dtype) / denom,
            y=tf.constant(0.0, dtype=dtype),
        )
        start = tf.constant(-1.0, dtype=dtype)
    else:
        step = tf.constant(2.0, dtype=dtype) / size_f
        start = tf.constant(-1.0, dtype=dtype) + step / tf.constant(2.0, dtype=dtype)

    return start + tf.range(size_dim, dtype=dtype) * step


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """AffineGrid

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

    graph_node_input_theta = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_size = get_constant_or_variable(
        graph_node.inputs[1],
        False \
            if hasattr(graph_node.inputs[1], 'values') \
                and isinstance(graph_node.inputs[1].values, np.ndarray) \
                    else before_op_output_shape_trans,
    )

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    theta = tf_layers_dict[graph_node_input_theta.name]['tf_node'] \
        if isinstance(graph_node_input_theta, gs.Variable) else graph_node_input_theta
    size = tf_layers_dict[graph_node_input_size.name]['tf_node'] \
        if isinstance(graph_node_input_size, gs.Variable) else graph_node_input_size

    # Pre-process transpose
    theta = pre_process_transpose(
        value_before_transpose=theta,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    align_corners = bool(graph_node.attrs.get('align_corners', 0))
    align_corners = replace_parameter(
        value_before_replacement=align_corners,
        param_target='attributes',
        param_name='align_corners',
        **kwargs,
    )

    theta_dtype = theta.dtype
    size_tensor = tf.cast(size, tf.int32)

    size_rank = size_tensor.shape[0] if size_tensor.shape.rank == 1 else None

    def _build_grid_2d(size_vec):
        N, _, H, W = tf.unstack(size_vec)
        h_coords = _make_coords(H, align_corners, theta_dtype)
        w_coords = _make_coords(W, align_corners, theta_dtype)
        grid_h, grid_w = tf.meshgrid(h_coords, w_coords, indexing='ij')
        ones = tf.ones_like(grid_w, dtype=theta_dtype)
        grid = tf.stack([grid_w, grid_h, ones], axis=-1)
        grid_flat = tf.reshape(grid, shape=[-1, 3])
        grid_flat_t = tf.transpose(grid_flat)
        grid_flat_t = tf.cast(grid_flat_t, theta_dtype)
        out = tf.matmul(theta, grid_flat_t)
        out = tf.transpose(out, perm=[0, 2, 1])
        out = tf.reshape(out, shape=tf.stack([N, H, W, 2]))
        return out

    def _build_grid_3d(size_vec):
        N, _, D, H, W = tf.unstack(size_vec)
        d_coords = _make_coords(D, align_corners, theta_dtype)
        h_coords = _make_coords(H, align_corners, theta_dtype)
        w_coords = _make_coords(W, align_corners, theta_dtype)
        grid_d, grid_h, grid_w = tf.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        ones = tf.ones_like(grid_w, dtype=theta_dtype)
        grid = tf.stack([grid_w, grid_h, grid_d, ones], axis=-1)
        grid_flat = tf.reshape(grid, shape=[-1, 4])
        grid_flat_t = tf.transpose(grid_flat)
        grid_flat_t = tf.cast(grid_flat_t, theta_dtype)
        out = tf.matmul(theta, grid_flat_t)
        out = tf.transpose(out, perm=[0, 2, 1])
        out = tf.reshape(out, shape=tf.stack([N, D, H, W, 3]))
        return out

    if size_rank == 4:
        grid = _build_grid_2d(size_tensor)
    elif size_rank == 5:
        grid = _build_grid_3d(size_tensor)
    else:
        size_dim = tf.shape(size_tensor)[0]
        grid = tf.cond(
            pred=tf.equal(size_dim, 4),
            true_fn=lambda: _build_grid_2d(size_tensor),
            false_fn=lambda: _build_grid_3d(size_tensor),
        )

    tf_layers_dict[graph_node_output.name]['tf_node'] = grid

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
                'tf_op_type': 'AffineGrid',
                'tf_inputs': {
                    'theta': theta,
                    'size': size,
                    'align_corners': align_corners,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
