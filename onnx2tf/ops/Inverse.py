from typing import Any, Optional, cast
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx2tf.gs as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    convert_axis,
    convert_reverse_axis,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
)


def _build_pseudo_inverse_2x2(
    *,
    x: tf.Tensor,
) -> tf.Tensor:
    x_any = cast(Any, x)
    a00 = x_any[..., 0:1, 0:1]
    a01 = x_any[..., 0:1, 1:2]
    a10 = x_any[..., 1:2, 0:1]
    a11 = x_any[..., 1:2, 1:2]

    det = a00 * a11 - a01 * a10
    row0 = tf.concat([a11, -a01], axis=-1)
    row1 = tf.concat([-a10, a00], axis=-1)
    adj = tf.concat([row0, row1], axis=-2)
    det_abs = tf.math.abs(det)
    eps = tf.cast(1.0e-3 if x.dtype == tf.float16 else 1.0e-6, x.dtype)
    det_safe = tf.where(det_abs < eps, tf.ones_like(det), det)
    return adj / det_safe


def _build_pseudo_inverse_3x3(
    *,
    x: tf.Tensor,
) -> tf.Tensor:
    x_any = cast(Any, x)
    a00 = x_any[..., 0:1, 0:1]
    a01 = x_any[..., 0:1, 1:2]
    a02 = x_any[..., 0:1, 2:3]
    a10 = x_any[..., 1:2, 0:1]
    a11 = x_any[..., 1:2, 1:2]
    a12 = x_any[..., 1:2, 2:3]
    a20 = x_any[..., 2:3, 0:1]
    a21 = x_any[..., 2:3, 1:2]
    a22 = x_any[..., 2:3, 2:3]

    c00 = a11 * a22 - a12 * a21
    c01 = a12 * a20 - a10 * a22
    c02 = a10 * a21 - a11 * a20

    det = a00 * c00 + a01 * c01 + a02 * c02

    adj00 = c00
    adj01 = a02 * a21 - a01 * a22
    adj02 = a01 * a12 - a02 * a11
    adj10 = c01
    adj11 = a00 * a22 - a02 * a20
    adj12 = a02 * a10 - a00 * a12
    adj20 = c02
    adj21 = a01 * a20 - a00 * a21
    adj22 = a00 * a11 - a01 * a10

    row0 = tf.concat([adj00, adj01, adj02], axis=-1)
    row1 = tf.concat([adj10, adj11, adj12], axis=-1)
    row2 = tf.concat([adj20, adj21, adj22], axis=-1)
    adj = tf.concat([row0, row1, row2], axis=-2)
    det_abs = tf.math.abs(det)
    eps = tf.cast(1.0e-3 if x.dtype == tf.float16 else 1.0e-6, x.dtype)
    det_safe = tf.where(det_abs < eps, tf.ones_like(det), det)
    return adj / det_safe


def _build_pseudo_inverse(
    *,
    x: tf.Tensor,
    op_name: str,
) -> tf.Tensor:
    shape = x.shape
    rank = len(shape)
    if rank < 2:
        return tf.linalg.inv(input=x, name=op_name)
    shape_dims = cast(Any, shape)
    rows = cast(Optional[int], tf.compat.dimension_value(shape_dims[-2]))
    cols = cast(Optional[int], tf.compat.dimension_value(shape_dims[-1]))
    if rows is None or cols is None or rows != cols:
        return tf.linalg.inv(input=x, name=op_name)

    n = rows
    if n == 2:
        return _build_pseudo_inverse_2x2(x=x)
    if n == 3:
        return _build_pseudo_inverse_3x3(x=x)
    return tf.linalg.inv(input=x, name=op_name)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: Any,
):
    """Inverse

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

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }
    # Inverse is pseudo-lowered by default to avoid FlexMatrixInverse.
    # If `-rtpo Inverse` is explicitly specified, use tf.linalg.inv instead.
    use_flex_matrix_inverse = \
        "inverse" in kwargs['replace_to_pseudo_operators']

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

    # https://pytorch.org/docs/stable/generated/torch.linalg.inv.html#torch.linalg.inv
    # https://www.tensorflow.org/api_docs/python/tf/linalg/inv
    # The input is a tensor of shape [..., M, M] whose inner-most 2 dimensions form square matrices.
    # The output is a tensor of the same shape as the input containing the inverse for all input submatrices [..., :, :].
    nhwc_flag = tf_layers_dict[graph_node_input.name]['nhwc'] \
        if not isinstance(graph_node_input, np.ndarray) \
            and 'nhwc' in tf_layers_dict[graph_node_input.name].keys() else False
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)
    inv_transpose = \
        nhwc_flag == True \
        and input_tensor_rank >= 4 \
        and input_tensor_shape[input_tensor_rank-3] == input_tensor_shape[input_tensor_rank-2]
    if inv_transpose:
        perm = [
            convert_axis(
                axis=axis,
                tensor_rank=input_tensor_rank,
                before_op_output_shape_trans=True
            ) for axis in range(input_tensor_rank)
        ]
        input_tensor = transpose_with_flexing_deterrence(
            input_tensor=input_tensor,
            perm=perm,
            **kwargs,
        )

    input_tensor = \
        tf.convert_to_tensor(input_tensor) \
            if isinstance(input_tensor, np.ndarray) else input_tensor
    if use_flex_matrix_inverse:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.linalg.inv(
                input=input_tensor,
                name=graph_node.name,
            )
        tf_op_type = tf.linalg.inv
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = _build_pseudo_inverse(
            x=input_tensor,
            op_name=graph_node.name,
        )
        tf_op_type = 'pseudo_inverse'

    if inv_transpose:
        perm = [
            convert_reverse_axis(
                axis=axis,
                tensor_rank=input_tensor_rank,
                before_op_output_shape_trans=True
            ) for axis in range(input_tensor_rank)
        ]
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            transpose_with_flexing_deterrence(
                input_tensor=tf_layers_dict[graph_node_output.name]['tf_node'],
                perm=perm,
                **kwargs,
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
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'x': input_tensor,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
