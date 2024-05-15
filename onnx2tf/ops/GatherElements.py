import sys
import copy
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import itertools
import tensorflow as tf
import tf_keras
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    process_neg_idx_along_axis,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
    get_tf_model_inputs,
    dummy_tf_inference,
    onnx_tf_tensor_validation,
)
from typing import List, Dict, Any


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """GatherElements

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
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    indices_tensor = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    indices_tensor = pre_process_transpose(
        value_before_transpose=indices_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )

    tensor_rank = len(input_tensor.shape)

    axis = graph_node.attrs.get('axis', 0)
    axis = convert_axis(
        axis=axis,
        tensor_rank=tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = kwargs['onnx_tensor_infos_for_validation']
    test_data_nhwc: np.ndarray = kwargs['test_data_nhwc']
    custom_input_op_name_np_data_path: str = kwargs['custom_input_op_name_np_data_path']
    disable_strict_mode: bool = kwargs['disable_strict_mode']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    input_tensor = input_tensor \
        if not isinstance(input_tensor, np.ndarray) \
            else tf.convert_to_tensor(input_tensor)
    indices_tensor = indices_tensor \
        if not isinstance(indices_tensor, np.ndarray) \
            else tf.convert_to_tensor(indices_tensor)
    indices_tensor = process_neg_idx_along_axis(
        data=input_tensor,
        axis=axis,
        indices=indices_tensor,
    )

    def define_gather_elements(axis: int, target_tensor, target_indices):
        if axis == 0:
            axis_perm = tf.range(tf.rank(target_tensor))
            data_swaped = target_tensor
            index_swaped = target_indices
        else:
            axis_perm = tf.tensor_scatter_nd_update(
                tf.range(tf.rank(target_tensor)),
                tf.constant([[0], [axis]]),
                tf.constant([axis, 0])
            )
            data_swaped = transpose_with_flexing_deterrence(
                input_tensor=target_tensor,
                perm=axis_perm,
                **kwargs,
            )
            index_swaped = transpose_with_flexing_deterrence(
                input_tensor=target_indices,
                perm=axis_perm,
                **kwargs,
            )

        idx_tensors_per_axis = [
            tf.range(tf.shape(index_swaped, index_swaped.dtype)[i]) \
                for i in range(index_swaped.shape.rank)
        ]

        idx_tensors_per_axis = tf.meshgrid(
            *idx_tensors_per_axis,
            indexing='ij',
        )
        idx_tensors_per_axis[0] = index_swaped
        dim_expanded_idx_tensors_per_axis = [
            tf.expand_dims(idx_tensor, axis=-1)
            for idx_tensor in idx_tensors_per_axis
        ]
        index_expanded = tf.concat(dim_expanded_idx_tensors_per_axis, axis=-1)
        gathered = tf.gather_nd(data_swaped, index_expanded)
        transposed = \
            transpose_with_flexing_deterrence(
                input_tensor=gathered,
                perm=axis_perm,
                **kwargs,
            )
        return transposed

    # Workaround to special patterns with wrong transposition when all axes except batch size have the same value.
    # Examine which combination of axis configurations reduces the error in output values the most,
    # and apply the transposition with the best performance.
    # https://github.com/PINTO0309/onnx2tf/issues/629
    # convnext-det.onnx
    graph_node_input_1_shape = graph_node_input_1.shape
    graph_node_input_2_shape = graph_node_input_2.shape
    if graph_node_input_1_shape is not None \
        and graph_node_input_2_shape is not None \
        and len(graph_node_input_1_shape) >= 3 \
        and len(graph_node_input_2_shape) >= 2 \
        and sum([1 if isinstance(s, str) else 0 for s in graph_node_input_1_shape]) == 0 \
        and sum([1 if isinstance(s, str) else 0 for s in graph_node_input_2_shape]) == 0:

        # Get the output tensor of one previous OP of TensorFlow only once
        if not disable_strict_mode:
            tf_model_inputs = get_tf_model_inputs(
                tf_layers_dict=tf_layers_dict,
            )
            val_model = None
            if not isinstance(input_tensor, np.ndarray):
                val_model = tf_keras.Model(
                    inputs=tf_model_inputs,
                    outputs=[
                        input_tensor,
                        indices_tensor,
                    ],
                )
            else:
                pass

        # TF dummy inference
        #   Get the output tensor of the previous layer of MatMul
        #   If input.1 and input.2 are both layers, tf_pre_tensor_infos is 2 cases
        #   If one of input.1 or input.2 is np.ndarray, tf_pre_tensor_infos is 1 case
        tf_pre_tensor_infos = {}
        if not disable_strict_mode:
            try:
                tf_pre_tensor_infos: Dict[Any] = \
                    dummy_tf_inference(
                        model=val_model,
                        inputs=tf_model_inputs,
                        test_data_nhwc=test_data_nhwc,
                        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                    )
            except Exception as ex:
                pass
            del val_model

        # Get np.ndarray for validation
        validation_data_1 = None
        validation_data_2 = None
        if not disable_strict_mode:
            if len(tf_pre_tensor_infos) == 2:
                if not isinstance(input_tensor, np.ndarray):
                    validation_data_1 = list(tf_pre_tensor_infos.values())[0]
                else:
                    validation_data_1 = copy.deepcopy(input_tensor)
                if not isinstance(indices_tensor, np.ndarray):
                    validation_data_2 = list(tf_pre_tensor_infos.values())[1]
                else:
                    validation_data_2 = copy.deepcopy(indices_tensor)

            # Get ONNX inference results
            onnx_tensor_infos = None
            if onnx_tensor_infos_for_validation is not None \
                and onnx_tensor_infos_for_validation.get(graph_node_output.name, None) is not None:
                onnx_tensor_infos = {
                    graph_node_output.name: onnx_tensor_infos_for_validation[graph_node_output.name]
                }
                del onnx_tensor_infos_for_validation

        # ONNX   : N,C,W
        # TF     : N,W,C
        # TF-axes: [1]
        #
        # ONNX: N,C,H,W
        # TF  : N,H,W,C
        # TF-axes: [1,2]
        #
        # ONNX: N,C,D,H,W
        # TF  : N,D,H,W,C
        # TF-axes: [1,2,3]

        # Automatic correction of accuracy degradation
        min_abs_err = sys.maxsize
        indices_tensor_rank = len(graph_node_input_2_shape)
        min_abs_err_perm_2: List[int] = [idx for idx in range(indices_tensor_rank)]

        if not disable_strict_mode:
            if onnx_tensor_infos is not None \
                and validation_data_1 is not None \
                and validation_data_2 is not None:

                tensor_2_candidate_for_transpositions = list(itertools.permutations(range(indices_tensor_rank)))
                # Search for the axis with the smallest error
                for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
                    try:
                        target_validation_data_1 = validation_data_1
                        target_validation_data_2 = validation_data_2
                        # Build TF dummy model
                        input_1 = tf_keras.Input(
                            shape=validation_data_1.shape[1:],
                            batch_size=validation_data_1.shape[0] \
                                if isinstance(validation_data_1.shape[0], int) else None,
                            name='dummy_input_1',
                            dtype=validation_data_1.dtype,
                        )
                        input_2 = tf_keras.Input(
                            shape=validation_data_2.shape[1:],
                            batch_size=validation_data_2.shape[0] \
                                if isinstance(validation_data_2.shape[0], int) else None,
                            name='dummy_input_2',
                            dtype=validation_data_2.dtype,
                        )
                        val_model = tf_keras.Model(
                            inputs=[
                                input_1,
                                input_2,
                            ],
                            outputs=[
                                define_gather_elements(
                                    axis=axis,
                                    target_tensor=input_1,
                                    target_indices=tf.transpose(a=input_2, perm=tensor_2_candidate_for_transposition),
                                )
                            ],
                        )
                        # TF dummy inference
                        tf_tensor_infos: Dict[Any] = \
                            dummy_tf_inference(
                                model=val_model,
                                inputs=[
                                    input_1,
                                    input_2,
                                ],
                                verification_datas=[
                                    target_validation_data_1,
                                    target_validation_data_2,
                                ],
                            )
                        del input_1
                        del input_2
                        del val_model

                        # Validation
                        onnx_tf_output_pairs = {
                            (oi[0], ti[0]): (oi[1], ti[1]) \
                                for oi, ti in zip(onnx_tensor_infos.items(), tf_tensor_infos.items())
                        }
                        """
                        check_results: Dict[str, List[np.ndarray, int, float|int]]
                            {
                                onnx_output_name: [
                                    onnx_tensor,
                                    matched_flg, <--- 0: Unmatched, 1: Matched, 2: Skipped (Deleted or Shape Unmatched)
                                    max_abs_err,
                                ]
                            }
                        """
                        check_results = \
                            onnx_tf_tensor_validation(
                                output_pairs=onnx_tf_output_pairs,
                                rtol=0.0,
                                atol=0.0,
                            )
                        result_err = sum([val[2] for val in check_results.values()])
                        if result_err < min_abs_err:
                            min_abs_err = result_err
                            min_abs_err_perm_2 = list(tensor_2_candidate_for_transposition)
                            if min_abs_err < 1e-3:
                                break
                    except Exception as ex:
                        pass

                transposed_tensor_shape = list(tf.transpose(a=indices_tensor, perm=min_abs_err_perm_2).shape)
                indices_tensor = \
                    transpose_with_flexing_deterrence(
                        input_tensor=indices_tensor,
                        perm=min_abs_err_perm_2,
                        output_shape=transposed_tensor_shape \
                            if None not in transposed_tensor_shape and transposed_tensor_shape != [] else None,
                        **kwargs,
                    )

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        define_gather_elements(
            axis=axis,
            target_tensor=input_tensor,
            target_indices=indices_tensor,
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
                'tf_op_type': 'GatherElements',
                'tf_inputs': {
                    'data': input_tensor,
                    'indices': indices_tensor,
                    'axis': axis,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
