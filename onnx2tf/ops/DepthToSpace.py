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
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
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
    """DepthToSpace

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
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    input_tensor_shape = input_tensor.shape

    blocksize = graph_node.attrs.get('blocksize', 1)
    mode = graph_node.attrs.get('mode', 'DCR')

    nhwc = tf_layers_dict[graph_node_input_1.name]['nhwc'] \
        if isinstance(graph_node_input_1, gs.Variable) \
            and 'nhwc' in tf_layers_dict[graph_node_input_1.name].keys() else False

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # Generation of TF OP
    if mode == "DCR":
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.nn.depth_to_space(
                input=input_tensor,
                block_size=blocksize,
                name=graph_node.name,
            )

    elif mode == "CRD":
        batch, channel = input_tensor_shape[0], input_tensor_shape[-1]
        height, width = input_tensor_shape[1], input_tensor_shape[2]
        csize = channel // (blocksize**2)

        reshape_node = tf.reshape(
            tensor=input_tensor,
            shape=[batch, height, width, csize, blocksize, blocksize]
        )
        transpose_node = transpose_with_flexing_deterrence(
            input_tensor=reshape_node,
            perm=[0,1,4,2,5,3],
            **kwargs,
        )
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.reshape(
                tensor=transpose_node,
                shape=[batch, height * blocksize, width * blocksize, csize],
                name=graph_node.name,
        )

    onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = kwargs['onnx_tensor_infos_for_validation']
    test_data_nhwc: np.ndarray = kwargs['test_data_nhwc']
    custom_input_op_name_np_data_path: str = kwargs['custom_input_op_name_np_data_path']
    disable_strict_mode: bool = kwargs['disable_strict_mode']

    # Workaround to special patterns with wrong transposition when all axes except batch size have the same value.
    # Examine which combination of axis configurations reduces the error in output values the most,
    # and apply the transposition with the best performance.
    graph_node_input_1_shape = graph_node_input_1.shape
    if not nhwc \
        and graph_node_input_1_shape is not None \
        and len(graph_node_input_1_shape) >= 4 \
        and sum([1 if isinstance(s, str) else 0 for s in graph_node_input_1_shape]) == 0:

        tensor_rank = len(graph_node_input_1_shape)

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
            if onnx_tensor_infos_for_validation is not None \
                and onnx_tensor_infos_for_validation.get(graph_node_output.name, None) is not None:
                onnx_tensor_infos = {
                    graph_node_output.name: onnx_tensor_infos_for_validation[graph_node_output.name]
                }
                del onnx_tensor_infos_for_validation

        # Automatic correction of accuracy degradation
        min_abs_err = sys.maxsize
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
                        input = tf_keras.Input(
                            shape=target_validation_data.shape[1:],
                            batch_size=target_validation_data.shape[0] \
                                if isinstance(target_validation_data.shape[0], int) else None,
                            name='dummy_input',
                            dtype=target_validation_data.dtype,
                        )
                        val_model = None
                        if mode == "DCR":
                            val_model = tf_keras.Model(
                                inputs=[
                                    input,
                                ],
                                outputs=[
                                    tf.nn.depth_to_space(
                                        input=tf.transpose(a=input, perm=tensor_1_candidate_for_transposition),
                                        block_size=blocksize,
                                        name=graph_node.name,
                                    )
                                ],
                            )
                        elif mode == "CRD":
                            input_shape = input.shape
                            batch, channel = input_shape[0], input_shape[-1]
                            height, width = input_shape[1], input_shape[2]
                            csize = channel // (blocksize**2)
                            reshape_node = tf.reshape(
                                tensor=input,
                                shape=[batch, height, width, csize, blocksize, blocksize]
                            )
                            transpose_node = transpose_with_flexing_deterrence(
                                input_tensor=reshape_node,
                                perm=[0,1,4,2,5,3],
                                **kwargs,
                            )
                            val_model = tf_keras.Model(
                                inputs=[
                                    input,
                                ],
                                outputs=[
                                    tf.reshape(
                                        tensor=transpose_node,
                                        shape=[batch, height * blocksize, width * blocksize, csize],
                                        name=graph_node.name,
                                    )
                                ]
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
                            if min_abs_err < 1e-3:
                                break
                    except Exception as ex:
                        pass

                if mode == "DCR":
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.nn.depth_to_space(
                            input=tf.transpose(a=input_tensor, perm=min_abs_err_perm_1),
                            block_size=blocksize,
                            name=graph_node.name,
                        )
                elif mode == "CRD":
                    transposed_input = tf.transpose(a=input_tensor, perm=min_abs_err_perm_1)
                    input_shape = transposed_input.shape
                    batch, channel = input_shape[0], input_shape[-1]
                    height, width = input_shape[1], input_shape[2]
                    csize = channel // (blocksize**2)
                    reshape_node = tf.reshape(
                        tensor=transposed_input,
                        shape=[batch, height, width, csize, blocksize, blocksize]
                    )
                    transpose_node = transpose_with_flexing_deterrence(
                        input_tensor=reshape_node,
                        perm=[0,1,4,2,5,3],
                        **kwargs,
                    )
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.reshape(
                            tensor=transpose_node,
                            shape=[batch, height * blocksize, width * blocksize, csize],
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
                'tf_op_type': tf.nn.depth_to_space,
                'tf_inputs': {
                    'input': input_tensor,
                    'block_size': blocksize,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
