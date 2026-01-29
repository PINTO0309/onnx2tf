import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import itertools
import collections
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    replace_parameter,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    pre_explicit_broadcast,
    explicit_broadcast,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    disable_unnecessary_transpose,
    shape_unmatched_special_avoidance_workaround,
    merge_two_consecutive_identical_ops_into_one,
    transpose_with_flexing_deterrence,
    deterring_shape_corruption_due_to_broadcast,
    acquisition_of_validation_data,
    onnx_tf_tensor_validation,
    obtaining_an_inverted_pattern_for_brute_force_validation,
    correction_process_for_accuracy_errors,
    nhwc_determination_of_output_value_of_binary_input_op,
)
from typing import Any, Dict


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Add

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

    graph_node_input_1 = \
        get_constant_or_variable(
            graph_node.inputs[0],
            before_op_output_shape_trans \
                if graph_node.inputs[0].shape is not None and len(graph_node.inputs[0].shape) != 1 else False,
        )
    graph_node_input_2 = \
        get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans \
                if graph_node.inputs[1].shape is not None and len(graph_node.inputs[1].shape) != 1 else False,
        )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    graph_node_output_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': graph_node_output_shape,
        'dtype': dtype,
        'nhwc': \
            nhwc_determination_of_output_value_of_binary_input_op(
                graph_node_input_1=graph_node_input_1,
                graph_node_input_2=graph_node_input_2,
                tf_layers_dict=tf_layers_dict
            )
    }

    input_tensor_1 = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_tensor_2 = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    # issue: https://github.com/PINTO0309/onnx2tf/issues/698
    if isinstance(input_tensor_1, np.ndarray) and not isinstance(input_tensor_2, np.ndarray):
        input_tensor_1, input_tensor_2 = input_tensor_2, input_tensor_1

    disable_strict_mode: bool = kwargs['disable_strict_mode']
    gelu_replace_op_names: dict = kwargs['gelu_replace_op_names']

    # Param replacement
    input_tensor_1 = \
        replace_parameter(
            value_before_replacement=input_tensor_1,
            param_target='inputs',
            param_name=graph_node.inputs[0].name,
            **kwargs,
        )
    input_tensor_2 = \
        replace_parameter(
            value_before_replacement=input_tensor_2,
            param_target='inputs',
            param_name=graph_node.inputs[1].name,
            **kwargs,
        )

    # Pre-process transpose
    input_tensor_1 = pre_process_transpose(
        value_before_transpose=input_tensor_1,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    input_tensor_2 = pre_process_transpose(
        value_before_transpose=input_tensor_2,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )

    # Workaround for ConvInteger
    if input_tensor_1.dtype == tf.float32 and input_tensor_2.dtype in [tf.int32, tf.int64, tf.float16]:
        input_tensor_2 = tf.cast(input_tensor_2, dtype=tf.float32)
    elif input_tensor_1.dtype in [tf.int32, tf.int64, tf.float16] and input_tensor_2.dtype == tf.float32:
        input_tensor_1 = tf.cast(input_tensor_1, dtype=tf.float32)

    # Disable unnecessary Transpose
    #   1. If both x and y are gs.Variable
    #   2. If only one of the two is the output of Transpose
    #   3. If the perm of the Transpose is [0,2,1] or [0,3,1,2] or [0,4,1,2,3]
    #   4. Furthermore, if the shape of x and y are mismatched
    graph_node_input_1, graph_node_input_2, input_tensor_1, input_tensor_2 = \
        disable_unnecessary_transpose(
            graph_node_input_1=graph_node_input_1,
            graph_node_input_2=graph_node_input_2,
            input_tensor_1=input_tensor_1,
            input_tensor_2=input_tensor_2,
            **kwargs,
        )

    # Shape Unmatched Special Avoidance Workaround
    # At least one True value for same_input_shape_as_onnx
    # At least one True value in nhwc_flags
    # same_input_shape_as_onnx == True and nhwc_flags == False and 3D or 4D or 5D tensor is NHWC transposed
    input_tensor_1, input_tensor_2 = \
        shape_unmatched_special_avoidance_workaround(
            graph_node_input_1=graph_node_input_1,
            graph_node_input_2=graph_node_input_2,
            input_tensor_1=input_tensor_1,
            input_tensor_2=input_tensor_2,
            tf_layers_dict=tf_layers_dict,
            **kwargs,
        )

    try:
        is_scalar_1 = False
        is_scalar_2 = False
        is_scalar_1_rank = tf.rank(input_tensor_1) == 0
        if hasattr(is_scalar_1_rank, 'numpy'):
            is_scalar_1 = is_scalar_1_rank.numpy()
        is_scalar_2_rank = tf.rank(input_tensor_2) == 0
        if hasattr(is_scalar_2_rank, 'numpy'):
            is_scalar_2 = is_scalar_2_rank.numpy()

        if (is_scalar_1 or is_scalar_2) and graph_node.i().op == 'Gemm':
            pass
        elif (is_scalar_1 or is_scalar_2) and graph_node.i().op != 'Gemm':
            first_tensor = None
            second_tensor = None
            if is_scalar_1:
                first_tensor = input_tensor_2
                second_tensor = input_tensor_1
            elif is_scalar_2:
                first_tensor = input_tensor_1
                second_tensor = input_tensor_2
            tmp_result = tf.math.add(first_tensor, second_tensor)
            tmp_result_shape = tmp_result.shape
            if first_tensor.shape == tmp_result_shape:
                pass
            else:
                input_tensor_1, input_tensor_2 = \
                    pre_explicit_broadcast(
                        input_tensor_1=input_tensor_1,
                        input_tensor_2=input_tensor_2,
                    )

                input_tensor_1, input_tensor_2 = \
                    explicit_broadcast(
                        const_or_var_1=input_tensor_1,
                        const_or_var_2=input_tensor_2,
                        graph_node=graph_node,
                        tf_layers_dict= tf_layers_dict,
                    )

        else:
            input_tensor_1, input_tensor_2 = \
                pre_explicit_broadcast(
                    input_tensor_1=input_tensor_1,
                    input_tensor_2=input_tensor_2,
                )

            input_tensor_1, input_tensor_2 = \
                explicit_broadcast(
                    const_or_var_1=input_tensor_1,
                    const_or_var_2=input_tensor_2,
                    graph_node=graph_node,
                    tf_layers_dict= tf_layers_dict,
                )
    except Exception as ex:
        input_tensor_1, input_tensor_2 = \
            pre_explicit_broadcast(
                input_tensor_1=input_tensor_1,
                input_tensor_2=input_tensor_2,
            )

        input_tensor_1, input_tensor_2 = \
            explicit_broadcast(
                const_or_var_1=input_tensor_1,
                const_or_var_2=input_tensor_2,
                graph_node=graph_node,
                tf_layers_dict= tf_layers_dict,
            )

    # Deterring shape corruption due to broadcast
    input_tensor_1, input_tensor_2 = \
        deterring_shape_corruption_due_to_broadcast(
            graph_node_output_shape=graph_node_output_shape,
            input_tensor_1=input_tensor_1,
            input_tensor_2=input_tensor_2,
        )

    # Correction process for accuracy errors
    if not disable_strict_mode:
        input_tensor_1, input_tensor_2 = \
            correction_process_for_accuracy_errors(
                input_tensor_1=input_tensor_1,
                input_tensor_2=input_tensor_2,
                tf_func=tf.math.add,
                np_func=np.add,
                graph_node_output_shape=graph_node_output_shape,
                graph_node_output=graph_node_output,
                tf_layers_dict=tf_layers_dict,
                **kwargs,
            )

    # Generation of TF OP

    # Replace with GeLU if available.
    gelu_op_names = [op_name for op_names in gelu_replace_op_names.values() for op_name in op_names]
    enable_gelu = graph_node.name in gelu_op_names

    if not enable_gelu:
        # Merge two consecutive identical OPs into one
        # https://github.com/PINTO0309/onnx2tf/issues/230
        #   A constant is calculated in advance only
        #   when one of the operations of the current OP
        #   is a constant and one of the operations of
        #   the next OP is also a constant.
        # By merging two OPs into one, an accuracy error always occurs
        # in the merged OP during the accuracy check.
        # 1. `Mul` -> `Mul` to `Single-Mul` : `10 * 5 * 8 -> 10 * 40`
        # 2. `Mul` -> `Div` to `Single-Mul` : `10 * 5 / 8 -> 10 * 0.625`
        # 3. `Div` -> `Mul` to `Single-Mul` : `10 / 5 * 8 -> 10 * 1.6`
        # 4. `Div` -> `Div` to `Single-Mul` : `10 / 5 / 8 -> 10 * 0.025`
        # 5. `Sub` -> `Sub` to `Single-Sub` : `10 - 5 - 8 -> 10 - 13`
        # 6. `Sub` -> `Add` to `Single-Sub` : `10 - 5 + 8 -> 10 + 3`
        # 7. `Add` -> `Add` to `Single-Add`  : `10 + 5 + 8 -> 10 + 13`
        # 8. `Add` -> `Sub` to `Single-Add`  : `10 + 5 - 8 -> 10 - 3`
        _, tf_type = \
            merge_two_consecutive_identical_ops_into_one(
                graph_node_input_1=graph_node_input_1,
                graph_node_input_2=graph_node_input_2,
                graph_node_output=graph_node_output,
                before_op_output_shape_trans=before_op_output_shape_trans,
                input_tensor_1=input_tensor_1,
                input_tensor_2=input_tensor_2,
                graph_node=graph_node,
                tf_layers_dict=tf_layers_dict,
                tf_func='Add'
            )
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.identity(
                input=input_tensor_1 if graph_node.i(0).name in gelu_op_names else input_tensor_2,
                name=graph_node.name,
            )
        tf_type = tf.identity

    def _normalize_dim(dim):
        return int(dim) if isinstance(dim, (int, np.integer)) else None

    def _get_static_shape(tensor):
        shape = getattr(tensor, 'shape', None)
        if shape is None or shape == tf.TensorShape(None):
            return None
        return [_normalize_dim(dim) for dim in list(shape)]

    def _shape_match_with_none(expected, actual):
        if expected is None or actual is None:
            return False
        if len(expected) != len(actual):
            return False
        for e_dim, a_dim in zip(expected, actual):
            e_dim = _normalize_dim(e_dim)
            a_dim = _normalize_dim(a_dim)
            if e_dim is None or a_dim is None:
                continue
            if e_dim != a_dim:
                return False
        return True

    def _perm_shape(shape, perm):
        return [shape[i] for i in perm] if shape is not None else None

    def _limited_perms(rank):
        identity = list(range(rank))
        perms = [identity]
        if rank == 3:
            perms.append([0, 2, 1])
        elif rank == 4:
            perms.extend([[0, 2, 3, 1], [0, 3, 1, 2]])
        elif rank == 5:
            perms.extend([[0, 2, 3, 4, 1], [0, 4, 1, 2, 3]])
        return perms

    def _ranked_perms(perms, input_shape, onnx_shape):
        if input_shape is None or onnx_shape is None:
            return perms
        scored = []
        for perm in perms:
            score = 0
            for out_idx, in_idx in enumerate(perm):
                if out_idx >= len(onnx_shape) or in_idx >= len(input_shape):
                    continue
                o_dim = _normalize_dim(onnx_shape[out_idx])
                i_dim = input_shape[in_idx]
                if isinstance(o_dim, int) and isinstance(i_dim, int) and o_dim == i_dim:
                    score += o_dim
            scored.append((score, 1 if perm == list(range(len(perm))) else 0, perm))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [p for _, _, p in scored]

    # Rescue guard for unexpected broadcasted shapes
    if not enable_gelu:
        expected_shape = None
        if graph_node_output_shape is not None:
            expected_shape = [_normalize_dim(dim) for dim in list(graph_node_output_shape)]
        output_shape = _get_static_shape(tf_layers_dict[graph_node_output.name]['tf_node'])
        input_shape_1 = _get_static_shape(input_tensor_1)
        input_shape_2 = _get_static_shape(input_tensor_2)
        if expected_shape is not None \
            and output_shape is not None \
            and not _shape_match_with_none(expected_shape, output_shape) \
            and input_shape_1 is not None \
            and input_shape_2 is not None \
            and len(input_shape_1) == len(expected_shape) \
            and len(input_shape_2) == len(expected_shape):

            rank = len(expected_shape)
            perms = _limited_perms(rank)
            perm_list_1 = _ranked_perms(perms, input_shape_1, expected_shape)
            perm_list_2 = _ranked_perms(perms, input_shape_2, expected_shape)
            rescue_done = False
            for perm_1 in perm_list_1:
                for perm_2 in perm_list_2:
                    try_input_1 = transpose_with_flexing_deterrence(
                        input_tensor=input_tensor_1,
                        perm=perm_1,
                        **kwargs,
                    )
                    try_input_2 = transpose_with_flexing_deterrence(
                        input_tensor=input_tensor_2,
                        perm=perm_2,
                        **kwargs,
                    )
                    try:
                        rescue_tensor = tf.math.add(
                            x=try_input_1 \
                                if not isinstance(try_input_1, np.ndarray) \
                                    else tf.convert_to_tensor(try_input_1),
                            y=try_input_2 \
                                if not isinstance(try_input_2, np.ndarray) \
                                    else tf.convert_to_tensor(try_input_2),
                            name=graph_node.name,
                        )
                    except Exception as ex:
                        continue

                    rescue_shape = _get_static_shape(rescue_tensor)
                    if _shape_match_with_none(expected_shape, rescue_shape):
                        input_tensor_1 = try_input_1
                        input_tensor_2 = try_input_2
                        tf_layers_dict[graph_node_output.name]['tf_node'] = rescue_tensor
                        tf_type = tf.math.add
                        rescue_done = True
                        break
                if rescue_done:
                    break

    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        post_process_transpose(
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
                    'x': input_tensor_1,
                    'y': input_tensor_2,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
