import sys
import copy
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import itertools
import tensorflow as tf
import onnx_graphsurgeon as gs
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
    """Reshape

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

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = None
    if len(graph_node.inputs) >= 2:
        graph_node_input_2 = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    output_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    tensor_rank = len(input_tensor.shape)

    reshape_shape = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    if reshape_shape is not None and reshape_shape.shape is None:
        reshape_shape = []
    if reshape_shape is None:
        reshape_shape = []

    # Ignore
    allowzero = bool(graph_node.attrs.get('allowzero', 0))

    # If Reshape's shape contains zeros, get the deformed shape from the output shape
    if isinstance(reshape_shape, list) and reshape_shape.count(0) > 0:
        new_shape = [-1 if isinstance(s, str) else int(s) for s in output_shape]
        reshape_shape = new_shape
    elif isinstance(reshape_shape, np.ndarray) and np.count_nonzero(reshape_shape == 0) > 0:
        new_shape = [-1 if isinstance(s, str) else int(s) for s in output_shape]
        reshape_shape = new_shape

    onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = kwargs['onnx_tensor_infos_for_validation']
    test_data_nhwc: np.ndarray = kwargs['test_data_nhwc']
    custom_input_op_name_np_data_path: str = kwargs['custom_input_op_name_np_data_path']
    disable_strict_mode: bool = kwargs['disable_strict_mode']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': output_shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    # NWC->NCW, NHWC->NCHW, NDHWC->NCDHW Transpose
    perm = [
        convert_axis(
            axis=idx,
            tensor_rank=tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        ) for idx in range(tensor_rank)
    ]

    # NHWC -> NCHW
    try:
        if graph_node.i().op not in ['LSTM', 'RNN', 'GRU']:
            transposed_tensor = \
                transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=list(perm) if perm is not None else None,
                    output_shape=output_shape,
                    name=graph_node.name,
                    **kwargs,
                )
        else:
            transposed_tensor = input_tensor
    except:
        transposed_tensor = \
            transpose_with_flexing_deterrence(
                input_tensor=input_tensor,
                perm=list(perm) if perm is not None else None,
                output_shape=output_shape,
                name=graph_node.name,
                **kwargs,
            )
    test_data = None
    if not isinstance(input_tensor, np.ndarray):
        if not isinstance(graph_node_input_1, np.ndarray) \
            and graph_node_input_1.name in tf_layers_dict \
            and 'verification_data' in tf_layers_dict[graph_node_input_1.name].keys():
            test_data: np.ndarray = tf_layers_dict[graph_node_input_1.name]['verification_data']
            test_data = test_data.transpose(list(perm) if perm is not None else None)
        elif isinstance(graph_node_input_1, np.ndarray):
            test_data: np.ndarray = input_tensor
            test_data = test_data.transpose(list(perm) if perm is not None else None)
        else:
            test_data = None
    else:
        test_data = input_tensor.transpose(list(perm) if perm is not None else None)

    if isinstance(reshape_shape, np.ndarray):
        perm_shape = [
            convert_axis(
                axis=idx,
                tensor_rank=len(reshape_shape),
                before_op_output_shape_trans=before_op_output_shape_trans,
            ) for idx in range(len(reshape_shape))
        ]
        transposed_reshape_shape = list(reshape_shape)
        if before_op_output_shape_trans:
            transposed_reshape_shape = [
                transposed_reshape_shape[perm_shape_dim] for perm_shape_dim in list(perm_shape)
            ]
    else:
        transposed_reshape_shape = reshape_shape

    # Param replacement
    transposed_tensor = replace_parameter(
        value_before_replacement=transposed_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    replaced_shape = replace_parameter(
        value_before_replacement=transposed_reshape_shape,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    shape_replaced_flg = False
    if ((isinstance(transposed_reshape_shape, list) and isinstance(replaced_shape, list)) \
        or (isinstance(transposed_reshape_shape, np.ndarray) and isinstance(replaced_shape, np.ndarray))) \
        and transposed_reshape_shape != replaced_shape:
        shape_replaced_flg = True
    elif (not isinstance(transposed_reshape_shape, list) and not isinstance(transposed_reshape_shape, np.ndarray)) \
        and tf.keras.backend.is_keras_tensor(transposed_reshape_shape) \
        and (isinstance(replaced_shape, list) or isinstance(replaced_shape, np.ndarray)):
        shape_replaced_flg = True
    if shape_replaced_flg:
        transposed_reshape_shape = replaced_shape

    # Pre-process transpose
    transposed_tensor = pre_process_transpose(
        value_before_transpose=transposed_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Reshape
    has_undefined_outputshape = output_shape is None
    if not has_undefined_outputshape:
        has_none_outputshape = None in output_shape
        has_str_outputshape = True in [True for dim in output_shape if isinstance(dim, str)]
        has_undefined_outputshape = has_none_outputshape or has_str_outputshape
    final_shape = transposed_reshape_shape \
        if (has_undefined_outputshape or shape_replaced_flg) else output_shape
    final_shape = final_shape \
        if not isinstance(final_shape, np.ndarray) \
            else tf.convert_to_tensor(final_shape)


    # Elimination of unnecessary OP sequential processing
    # [1,81,52,52] -> Reshape [81,52,52] -> Expand [1,81,52,52]
    # [1,81,52,52] -> Reshape [81,52,52] -> Unsqueeze [1,81,52,52]
    # [81,52,52] -> Reshape [1,81,52,52] -> Squeeze [81,52,52]
    consumer_count = 0
    consumer_nodes: List[gs.Node] = []
    while True:
        try:
            consumer_node =  graph_node.o(consumer_count, 0)
            consumer_nodes.append(consumer_node)
            consumer_count += 1
        except:
            break
    expand_count = 0
    unsqueeze_count = 0
    squeeze_count = 0
    for consumer_node in consumer_nodes:
        if consumer_node.op == 'Expand' \
            and consumer_node.outputs[0].shape is not None \
            and consumer_node.outputs[0].shape == transposed_tensor.shape:
            expand_count += 1

        elif consumer_node.op == 'Unsqueeze' \
            and consumer_node.outputs[0].shape is not None \
            and consumer_node.outputs[0].shape == transposed_tensor.shape:
            unsqueeze_count += 1

        elif consumer_node.op == 'Squeeze' \
            and consumer_node.outputs[0].shape is not None \
            and consumer_node.outputs[0].shape == transposed_tensor.shape:
            squeeze_count += 1
    if (expand_count > 0 and expand_count == consumer_count) \
        or (unsqueeze_count > 0 and unsqueeze_count == consumer_count) \
        or (squeeze_count > 0 and squeeze_count == consumer_count):
        # Replace
        final_shape = consumer_nodes[0].outputs[0].shape
        tf_layers_dict[graph_node_output.name]['unnecessary_reshape'] = True
    else:
        # No-replace
        pass

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.reshape(
            tensor=transposed_tensor,
            shape=final_shape,
            name=graph_node.name,
        )

    # Special support for ShuffleNet patterns
    # 5D Reshape -> 5D Transpose -> 4D Reshape
    # 1,2,72,16,16 -> 1,72,2,16,16 -> 1,144,16,16
    # At this time, only the channel shuffling pattern of image processing is supported.
    try:
        two_previous_op: gs.Node = graph_node.i().i()
        two_previous_op_type = two_previous_op.op
        one_previous_op: gs.Node = graph_node.i()
        one_previous_op_type = one_previous_op.op
        if two_previous_op_type == 'Reshape' and one_previous_op_type == 'Transpose':
            two_previous_op_output_shape = two_previous_op.outputs[0].shape
            one_previous_op_output_shape = one_previous_op.outputs[0].shape
            two_previous_op_output_rank = len(two_previous_op_output_shape)
            one_previous_op_output_rank = len(one_previous_op_output_shape)
            current_op_output_shape = graph_node.outputs[0].shape
            current_op_output_rank = len(current_op_output_shape)
            if two_previous_op_output_rank == 5 \
                and one_previous_op_output_rank == 5 \
                and current_op_output_rank == 4 \
                and two_previous_op_output_shape[1] == one_previous_op_output_shape[2] \
                and two_previous_op_output_shape[2] == one_previous_op_output_shape[1] \
                and current_op_output_shape[1] == (one_previous_op_output_shape[1] * one_previous_op_output_shape[2]):
                # ShuffleNet patterns - 4D only
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    transpose_with_flexing_deterrence(
                        input_tensor=tf_layers_dict[graph_node_output.name]['tf_node'],
                        perm=[0,2,3,1],
                        **kwargs,
                    )
            else:
                pass
    except:
        pass


    # Workaround to special patterns with wrong transposition when all axes except batch size have the same value.
    # Examine which combination of axis configurations reduces the error in output values the most,
    # and apply the transposition with the best performance.
    # Input: [1, 20, 20, 20], Output: [1, 800, 10]
    # https://github.com/PINTO0309/onnx2tf/issues/478
    graph_node_input_1_shape = graph_node_input_1.shape
    if graph_node_input_1_shape is not None \
        and len(graph_node_input_1_shape) >= 3 \
        and sum([1 if isinstance(s, str) else 0 for s in graph_node_input_1_shape]) == 0 \
        and len(set(graph_node_input_1_shape[1:])) == 1:

        # Get the output tensor of one previous OP of TensorFlow only once
        if not disable_strict_mode:
            tf_model_inputs = get_tf_model_inputs(
                tf_layers_dict=tf_layers_dict,
            )
            val_model = None
            if not isinstance(input_tensor, np.ndarray):
                val_model = tf.keras.Model(
                    inputs=tf_model_inputs,
                    outputs=[
                        input_tensor,
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
        validation_data = None
        if not disable_strict_mode:
            if len(tf_pre_tensor_infos) == 1:
                if not isinstance(input_tensor, np.ndarray):
                    validation_data = list(tf_pre_tensor_infos.values())[0]
                else:
                    validation_data = copy.deepcopy(input_tensor)

            # Get ONNX inference results
            onnx_tensor_infos = None
            if onnx_tensor_infos_for_validation is not None:
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
        min_abs_err_perm_1: int = [idx for idx in range(tensor_rank)]

        if not disable_strict_mode:
            if onnx_tensor_infos is not None and validation_data is not None:
                tensor_1_candidate_for_transpositions = list(itertools.permutations(range(tensor_rank)))
                tensor_1_candidate_for_transpositions = [
                    trans_perm for trans_perm in tensor_1_candidate_for_transpositions \
                        if trans_perm[0] == 0
                ]
                # Search for the axis with the smallest error
                for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
                    try:
                        target_validation_data = validation_data.transpose(tensor_1_candidate_for_transposition)
                        # Build TF dummy model
                        input = tf.keras.Input(
                            shape=target_validation_data.shape[1:],
                            batch_size=target_validation_data.shape[0] \
                                if isinstance(target_validation_data.shape[0], int) else None,
                            name='dummy_input',
                            dtype=target_validation_data.dtype,
                        )
                        val_model = tf.keras.Model(
                            inputs=[
                                input,
                            ],
                            outputs=[
                                tf.reshape(
                                    tensor=tf.transpose(a=input, perm=tensor_1_candidate_for_transposition),
                                    shape=final_shape,
                                    name=graph_node.name,
                                )
                            ],
                        )
                        # TF dummy inference
                        tf_tensor_infos: Dict[Any] = \
                            dummy_tf_inference(
                                model=val_model,
                                inputs=[
                                    input,
                                ],
                                verification_datas=[
                                    target_validation_data,
                                ],
                            )
                        del input
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
                            min_abs_err_perm_1 = list(tensor_1_candidate_for_transposition)
                            # if min_abs_err < 1e-3:
                            #     break
                    except Exception as ex:
                        pass
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.reshape(
                        tensor=tf.transpose(a=input_tensor, perm=min_abs_err_perm_1),
                        shape=final_shape,
                        name=graph_node.name,
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
                'tf_op_type': tf.reshape,
                'tf_inputs': {
                    'tensor': transposed_tensor,
                    'shape': transposed_reshape_shape,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
