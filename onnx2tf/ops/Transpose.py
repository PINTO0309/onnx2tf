import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Transpose

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

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    tensor_rank = len(input_tensor.shape)

    perm = graph_node.attrs.get('perm', [idx for idx in reversed(range(tensor_rank))])

    if isinstance(perm, list) or (isinstance(perm, np.ndarray) and len(perm.shape) > 0):
        if perm[0] == 0:
            perm = [
                convert_axis(
                    axis=idx,
                    tensor_rank=tensor_rank,
                    before_op_output_shape_trans=before_op_output_shape_trans,
                ) for idx in perm
            ]
        else:
            # ゼロ次元目の転置が発生しているときは、ONNXの最終出力テンソルの形状とTFの入力テンソルの形状を比較して
            # ONNX側の最終出力テンソルの形状に合うように転置する
            onnx_output_shape = shape
            tf_input_shape = input_tensor.shape
            new_perm = [-1] * len(onnx_output_shape)
            for tf_shape_idx, tf_shape_value in enumerate(tf_input_shape):
                matched_idxs = [
                    idx for idx, onnx_shape_value in enumerate(onnx_output_shape) \
                        if onnx_shape_value == tf_shape_value
                ]
                if len(matched_idxs) == 0:
                    new_perm[tf_shape_idx] = onnx_output_shape.index(tf_shape_value)
                elif len(matched_idxs) == 1:
                    new_perm[matched_idxs[0]] = tf_shape_idx
                else:
                    for matched_idx in matched_idxs:
                        if new_perm[matched_idx] == -1:
                            new_perm[matched_idx] = tf_shape_idx
                            break
            perm = new_perm

    elif perm is not None and isinstance(perm, np.ndarray) and len(perm.shape) == 0:
        if perm[0] == 0:
            perm = convert_axis(
                axis=perm,
                tensor_rank=tensor_rank,
                before_op_output_shape_trans=before_op_output_shape_trans,
            )
        else:
            # ゼロ次元目の転置が発生しているときは、ONNXの最終出力テンソルの形状とTFの入力テンソルの形状を比較して
            # ONNX側の最終出力テンソルの形状に合うように転置する
            onnx_output_shape = shape
            tf_input_shape = input_tensor.shape
            new_perm = [-1] * len(onnx_output_shape)
            for tf_shape_idx, tf_shape_value in enumerate(tf_input_shape):
                matched_idxs = [
                    idx for idx, onnx_shape_value in enumerate(onnx_output_shape) \
                        if onnx_shape_value == tf_shape_value
                ]
                if len(matched_idxs) == 0:
                    new_perm[tf_shape_idx] = onnx_output_shape.index(tf_shape_value)
                elif len(matched_idxs) == 1:
                    new_perm[matched_idxs[0]] = tf_shape_idx
                else:
                    for matched_idx in matched_idxs:
                        if new_perm[matched_idx] == -1:
                            new_perm[matched_idx] = tf_shape_idx
                            break
            perm = new_perm

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.transpose(
            a=input_tensor,
            perm=list(perm) if perm is not None else None,
            name=graph_node.name,
        )
