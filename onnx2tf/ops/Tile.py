import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import itertools
import tensorflow as tf
import tf_keras
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    pre_process_transpose,
    post_process_transpose,
    dummy_tf_inference,
    onnx_tf_tensor_validation,
    acquisition_of_validation_data,
    transpose_with_flexing_deterrence,
)
from typing import Any, Dict, List


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Tile

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
    before_op_output_shape_trans = before_op_output_shape_trans_1 and before_op_output_shape_trans_2

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    onnx_output_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': onnx_output_shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input_1.name]['nhwc'] \
            if isinstance(graph_node_input_1, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False
    }

    input_tensor_1 = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_tensor_2 = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    # Obtain ONNX inference results and
    # TensorFlow inference results up to the previous layer of TensorFlow
    onnx_tensor_infos, validation_data_1, validation_data_2 = \
        acquisition_of_validation_data(
            input_tensor_1=input_tensor_1,
            input_tensor_2=input_tensor_2,
            graph_node_output=graph_node_output,
            tf_layers_dict=tf_layers_dict,
            **kwargs,
        )

    # Param replacement
    input_tensor_1 = replace_parameter(
        value_before_replacement=input_tensor_1,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    input_tensor_2 = replace_parameter(
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

    # Shape Unmatch Error Mitigation Measures
    # Search for and transpose shapes that do not cause shape unmatch errors
    min_abs_err = sys.maxsize
    min_abs_err_perm_1: int = [idx for idx in range(len(input_tensor_1.shape))]
    min_abs_err_perm_2: int = [idx for idx in range(len(input_tensor_2.shape))]

    def define_tile(
        *,
        target_input_tensor_1: Any,
        target_perm_1: List,
        target_input_tensor_2: Any,
        target_perm_2: List,
        target_name: str,
        **kwargs: Dict,
    ):
        multiples = None
        if not isinstance(target_input_tensor_2, np.ndarray):
            multiples = \
                tf.gather(
                    params=target_input_tensor_2,
                    indices=target_perm_2,
                    axis=0,
                )
        else:
            multiples = target_input_tensor_2[target_perm_2]

        return \
            tf.tile(
                input=transpose_with_flexing_deterrence(
                    input_tensor=target_input_tensor_1 \
                        if not isinstance(target_input_tensor_1, np.ndarray) \
                            else tf.convert_to_tensor(target_input_tensor_1),
                    perm=target_perm_1,
                    **kwargs,
                ),
                multiples=multiples,
                name=target_name,
            )

    tensor_1_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_1.shape))))
    tensor_2_candidate_for_transpositions = None

    if isinstance(input_tensor_2, int):
        tensor_2_candidate_for_transpositions = list(itertools.permutations(range(input_tensor_2)))
    elif isinstance(input_tensor_2, np.ndarray) and hasattr(input_tensor_2, '__len__'):
        tiles_tensor_length = len(input_tensor_2)
        if tiles_tensor_length > 1:
            tensor_2_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_2))))
        else:
            tensor_2_candidate_for_transpositions = list(itertools.permutations(input_tensor_2))
    elif tf_keras.backend.is_keras_tensor(input_tensor_2) and hasattr(input_tensor_2.shape, '__len__'):
        tiles_tensor_length = len(input_tensor_2.shape)
        if tiles_tensor_length > 1:
            tensor_2_candidate_for_transpositions = list(itertools.permutations(range(tiles_tensor_length)))
        else:
            # Dynamic Tensor
            pass
    else:
        # Unknown
        pass

    if tensor_2_candidate_for_transpositions is not None:
        for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
            for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
                try:
                    # Build TF dummy model
                    input_1 = tf_keras.Input(
                        shape=validation_data_1.shape[1:],
                        batch_size=validation_data_1.shape[0] \
                            if isinstance(validation_data_1.shape[0], int) else None,
                        name='dummy_input_1',
                        dtype=validation_data_1.dtype,
                    )
                    input_2 = validation_data_2
                    dummy_tile = define_tile(
                        target_input_tensor_1=input_1,
                        target_perm_1=list(tensor_1_candidate_for_transposition),
                        target_input_tensor_2=input_2,
                        target_perm_2=list(tensor_2_candidate_for_transposition),
                        target_name=graph_node.name,
                        **kwargs
                    )
                    # Verify that the output shape matches that of ONNX
                    # If the combination of each value of a dimension is not correct,
                    # invalidate the normal processing judgment.
                    onnx_output_shape_prod = np.prod([dim if not isinstance(dim, str) else -1 for dim in onnx_output_shape])
                    tile_output_shapes = list(dummy_tile.shape)
                    tile_output_shape_prod = np.prod([dim if dim is not None else -1 for dim in tile_output_shapes])
                    if onnx_output_shape_prod != tile_output_shape_prod:
                        del input_1
                        del input_2
                        del dummy_tile
                        continue

                    # Perform simple accuracy verification
                    # Terminate when the error is less than 1e-3
                    if onnx_tensor_infos:
                        try:
                            # Search for the axis with the smallest error
                            val_model = tf_keras.Model(
                                inputs=[
                                    input_1,
                                ],
                                outputs=[
                                    dummy_tile,
                                ],
                            )

                            # TF dummy inference
                            tf_tensor_infos: Dict[Any] = dummy_tf_inference(
                                model=val_model,
                                inputs=[
                                    input_1,
                                ],
                                verification_datas=[
                                    validation_data_1,
                                ],
                            )
                            del input_1
                            del input_2
                            del dummy_tile
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
                            check_results = onnx_tf_tensor_validation(
                                output_pairs=onnx_tf_output_pairs,
                                rtol=0.0,
                                atol=0.0,
                            )
                            result_err = sum([val[2] for val in check_results.values()])
                            if result_err < min_abs_err:
                                min_abs_err = result_err
                                min_abs_err_perm_1 = list(tensor_1_candidate_for_transposition)
                                min_abs_err_perm_2 = list(tensor_2_candidate_for_transposition)
                                if min_abs_err < 1e-3:
                                    break
                        except Exception as ex1:
                            pass
                except Exception as ex2:
                    pass
            else:
                continue
            break

    # Generation of TF OP
    if tensor_2_candidate_for_transpositions is not None:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            define_tile(
                target_input_tensor_1=input_tensor_1,
                target_perm_1=min_abs_err_perm_1,
                target_input_tensor_2=input_tensor_2,
                target_perm_2=min_abs_err_perm_2,
                target_name=graph_node.name,
                **kwargs
            )
    else:
        # Dynamic Tensor
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.tile(
                input=input_tensor_1 \
                    if not isinstance(input_tensor_1, np.ndarray) \
                        else tf.convert_to_tensor(input_tensor_1),
                multiples=tf.convert_to_tensor([dim for dim in input_tensor_2]),
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
                'tf_op_type': tf.tile,
                'tf_inputs': {
                    'input': input_tensor_1,
                    'multiples': input_tensor_2,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )

