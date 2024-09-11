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
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    explicit_broadcast,
    pre_explicit_broadcast,
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
    """BatchNormalization

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    # Inputs
    X: gs.Variable = graph_node.inputs[0]
    scale = graph_node.inputs[1]
    B = graph_node.inputs[2]
    input_mean = graph_node.inputs[3]
    input_var = graph_node.inputs[4]
    # Outputs
    Y: gs.Variable = graph_node.outputs[0]
    if len(graph_node.outputs) > 1:
        graph_node.outputs = [graph_node.outputs[0]]

    if hasattr(scale, 'inputs') \
        and len(scale.inputs) > 0 \
        and hasattr(scale.inputs[0], 'attrs') \
        and 'value' in scale.inputs[0].attrs \
        and hasattr(scale.inputs[0].attrs['value'], 'values'):
        scale = scale.inputs[0].attrs['value']

    if hasattr(B, 'inputs') \
        and len(B.inputs) > 0 \
        and hasattr(B.inputs[0], 'attrs') \
        and 'value' in B.inputs[0].attrs \
        and hasattr(B.inputs[0].attrs['value'], 'values'):
        B = B.inputs[0].attrs['value']

    if hasattr(input_mean, 'inputs') \
        and len(input_mean.inputs) > 0 \
        and hasattr(input_mean.inputs[0], 'attrs') \
        and 'value' in input_mean.inputs[0].attrs \
        and hasattr(input_mean.inputs[0].attrs['value'], 'values'):
        input_mean = input_mean.inputs[0].attrs['value']

    if hasattr(input_var, 'inputs') \
        and len(input_var.inputs) > 0 \
        and hasattr(input_var.inputs[0], 'attrs') \
        and 'value' in input_var.inputs[0].attrs \
        and hasattr(input_var.inputs[0].attrs['value'], 'values'):
        input_var = input_var.inputs[0].attrs['value']

    shape = Y.shape
    dtype = Y.dtype

    epsilon = graph_node.attrs.get('epsilon', 1e-05)
    momentum = graph_node.attrs.get('momentum', 0.9)
    training_mode = bool(graph_node.attrs.get('training_mode', 0)) # disuse

    nhwc: bool = tf_layers_dict[X.name]['nhwc'] \
        if isinstance(X, gs.Variable) and 'nhwc' in tf_layers_dict[X.name].keys() else False

    onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = kwargs['onnx_tensor_infos_for_validation']
    test_data_nhwc: np.ndarray = kwargs['test_data_nhwc']
    custom_input_op_name_np_data_path: str = kwargs['custom_input_op_name_np_data_path']
    disable_strict_mode: bool = kwargs['disable_strict_mode']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[Y.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': nhwc,
    }

    # Generation of TF OP
    input_tensor = tf_layers_dict[X.name]['tf_node']
    input_tensor_rank = len(input_tensor.shape)

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    mean = input_mean.values \
        if not isinstance(input_mean, gs.Variable) \
            else tf_layers_dict[input_mean.name]['tf_node']
    var = input_var.values \
        if not isinstance(input_var, gs.Variable) \
            else tf_layers_dict[input_var.name]['tf_node']
    offset = B.values \
        if not isinstance(B, gs.Variable) \
            else tf_layers_dict[B.name]['tf_node']
    scale = scale.values \
        if not isinstance(scale, gs.Variable) \
            else tf_layers_dict[scale.name]['tf_node']

    # Broadcast
    if len(input_tensor.shape) == 3:
        input_tensor, mean = pre_explicit_broadcast(
            input_tensor_1=input_tensor,
            input_tensor_2=mean,
        )
        input_tensor, var = pre_explicit_broadcast(
            input_tensor_1=input_tensor,
            input_tensor_2=var,
        )
        input_tensor, offset = pre_explicit_broadcast(
            input_tensor_1=input_tensor,
            input_tensor_2=offset,
        )
        input_tensor, scale = pre_explicit_broadcast(
            input_tensor_1=input_tensor,
            input_tensor_2=scale,
        )

    try:
        tf_layers_dict[Y.name]['tf_node'] = \
            tf.nn.batch_normalization(
                x=input_tensor,
                mean=mean \
                    if not isinstance(mean, np.ndarray) \
                        else tf.convert_to_tensor(mean),
                variance=var \
                    if not isinstance(var, np.ndarray) \
                        else tf.convert_to_tensor(var),
                offset=offset \
                    if not isinstance(offset, np.ndarray) \
                        else tf.convert_to_tensor(offset),
                scale=scale \
                    if not isinstance(scale, np.ndarray) \
                        else tf.convert_to_tensor(scale),
                variance_epsilon=epsilon,
            )
        # Shape correction workaround when final output shapes do not match
        if not nhwc \
            and graph_node.outputs[0].shape is not None \
            and sum([1 if isinstance(s, str) else 0 for s in graph_node.outputs[0].shape]) == 0 \
            and np.prod(graph_node.outputs[0].shape) != np.prod(tf_layers_dict[Y.name]['tf_node'].shape) \
            and (
                (mean.shape is not None and len(mean.shape) == 1) \
                    or (var.shape is not None and len(var.shape) == 1) \
                    or (offset.shape is not None and len(offset.shape) == 1) \
                    or (scale.shape is not None and len(scale.shape) == 1)
            ):
                if len(input_tensor.shape) == 3:
                    input_tensor = \
                        transpose_with_flexing_deterrence(
                            input_tensor=input_tensor,
                            perm=[0,2,1],
                            **kwargs,
                        )
                    tf_layers_dict[Y.name]['tf_node'] = \
                        tf.nn.batch_normalization(
                            x=input_tensor,
                            mean=mean \
                                if not isinstance(mean, np.ndarray) \
                                    else tf.convert_to_tensor(mean),
                            variance=var \
                                if not isinstance(var, np.ndarray) \
                                    else tf.convert_to_tensor(var),
                            offset=offset \
                                if not isinstance(offset, np.ndarray) \
                                    else tf.convert_to_tensor(offset),
                            scale=scale \
                                if not isinstance(scale, np.ndarray) \
                                    else tf.convert_to_tensor(scale),
                            variance_epsilon=epsilon,
                        )
                elif len(input_tensor.shape) == 4:
                    input_tensor = \
                        transpose_with_flexing_deterrence(
                            input_tensor=input_tensor,
                            perm=[0,2,3,1],
                            **kwargs,
                        )
                    tf_layers_dict[Y.name]['tf_node'] = \
                        tf.nn.batch_normalization(
                            x=input_tensor,
                            mean=mean \
                                if not isinstance(mean, np.ndarray) \
                                    else tf.convert_to_tensor(mean),
                            variance=var \
                                if not isinstance(var, np.ndarray) \
                                    else tf.convert_to_tensor(var),
                            offset=offset \
                                if not isinstance(offset, np.ndarray) \
                                    else tf.convert_to_tensor(offset),
                            scale=scale \
                                if not isinstance(scale, np.ndarray) \
                                    else tf.convert_to_tensor(scale),
                            variance_epsilon=epsilon,
                        )
                elif len(input_tensor.shape) == 5:
                    input_tensor = \
                        transpose_with_flexing_deterrence(
                            input_tensor=input_tensor,
                            perm=[0,2,3,4,1],
                            **kwargs,
                        )
                    tf_layers_dict[Y.name]['tf_node'] = \
                        tf.nn.batch_normalization(
                            x=input_tensor,
                            mean=mean \
                                if not isinstance(mean, np.ndarray) \
                                    else tf.convert_to_tensor(mean),
                            variance=var \
                                if not isinstance(var, np.ndarray) \
                                    else tf.convert_to_tensor(var),
                            offset=offset \
                                if not isinstance(offset, np.ndarray) \
                                    else tf.convert_to_tensor(offset),
                            scale=scale \
                                if not isinstance(scale, np.ndarray) \
                                    else tf.convert_to_tensor(scale),
                            variance_epsilon=epsilon,
                        )

    except Exception as e:
        # Workaround for inconsistent "C" position
        input_tensor_rank = len(input_tensor.shape)
        if input_tensor_rank >= 3 \
            and input_tensor.shape[1] is not None \
            and input_tensor.shape[1] == offset.shape[0]:

            perm = [0] + [i for i in range(2, input_tensor_rank)] + [1]
            tf_layers_dict[Y.name]['tf_node'] = \
                tf.nn.batch_normalization(
                    x=tf.transpose(input_tensor, perm=perm),
                    mean=mean \
                        if not isinstance(mean, np.ndarray) \
                            else tf.convert_to_tensor(mean),
                    variance=var \
                        if not isinstance(var, np.ndarray) \
                            else tf.convert_to_tensor(var),
                    offset=offset \
                        if not isinstance(offset, np.ndarray) \
                            else tf.convert_to_tensor(offset),
                    scale=scale \
                        if not isinstance(scale, np.ndarray) \
                            else tf.convert_to_tensor(scale),
                    variance_epsilon=epsilon,
                )
        else:
            raise

    # Automatic accuracy compensation
    graph_node_input_1_shape = X.shape
    if graph_node_input_1_shape is not None:

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
                and onnx_tensor_infos_for_validation.get(Y.name, None) is not None:
                onnx_tensor_infos = {
                    Y.name: onnx_tensor_infos_for_validation[Y.name]
                }
                del onnx_tensor_infos_for_validation

        # Automatic correction of accuracy degradation
        min_abs_err = sys.maxsize
        min_abs_err_perm_1: List[int] = []
        check_length = 0
        if input_tensor.shape is not None and mean.shape is not None and len(input_tensor.shape) >= len(mean.shape):
            check_length = len(input_tensor.shape)
        else:
            check_length = len(mean.shape)
        min_abs_err_perm_1: List[int] = [idx for idx in range(check_length)]

        if not disable_strict_mode:
            if onnx_tensor_infos is not None and validation_data is not None:
                tensor_1_candidate_for_transpositions = list(itertools.permutations(range(check_length)))
                # Search for the axis with the smallest error
                for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
                    try:
                        target_validation_data = validation_data
                        # Build TF dummy model
                        input = tf_keras.Input(
                            shape=validation_data.shape[1:],
                            batch_size=validation_data.shape[0] \
                                if isinstance(validation_data.shape[0], int) else None,
                            name='dummy_input',
                            dtype=validation_data.dtype,
                        )
                        val_model = tf_keras.Model(
                            inputs=[
                                input,
                            ],
                            outputs=[
                                tf.nn.batch_normalization(
                                    x=input,
                                    mean=\
                                        transpose_with_flexing_deterrence(
                                            input_tensor=mean,
                                            perm=min_abs_err_perm_1,
                                            output_shape=Y.shape \
                                                if None not in Y.shape and Y.shape != [] else None,
                                            **kwargs,
                                        ) if not isinstance(mean, np.ndarray) else \
                                        transpose_with_flexing_deterrence(
                                            input_tensor=tf.convert_to_tensor(mean),
                                            perm=min_abs_err_perm_1,
                                            output_shape=Y.shape \
                                                if None not in Y.shape and Y.shape != [] else None,
                                            **kwargs,
                                        ),
                                    variance=\
                                        transpose_with_flexing_deterrence(
                                            input_tensor=var,
                                            perm=min_abs_err_perm_1,
                                            output_shape=Y.shape \
                                                if None not in Y.shape and Y.shape != [] else None,
                                            **kwargs,
                                        ) if not isinstance(var, np.ndarray) else \
                                        transpose_with_flexing_deterrence(
                                            input_tensor=tf.convert_to_tensor(var),
                                            perm=min_abs_err_perm_1,
                                            output_shape=Y.shape \
                                                if None not in Y.shape and Y.shape != [] else None,
                                            **kwargs,
                                        ),
                                    offset=\
                                        transpose_with_flexing_deterrence(
                                            input_tensor=offset,
                                            perm=min_abs_err_perm_1,
                                            output_shape=Y.shape \
                                                if None not in Y.shape and Y.shape != [] else None,
                                            **kwargs,
                                        ) if not isinstance(offset, np.ndarray) else \
                                        transpose_with_flexing_deterrence(
                                            input_tensor=tf.convert_to_tensor(offset),
                                            perm=min_abs_err_perm_1,
                                            output_shape=Y.shape \
                                                if None not in Y.shape and Y.shape != [] else None,
                                            **kwargs,
                                        ),
                                    scale=\
                                        transpose_with_flexing_deterrence(
                                            input_tensor=scale,
                                            perm=min_abs_err_perm_1,
                                            output_shape=Y.shape \
                                                if None not in Y.shape and Y.shape != [] else None,
                                            **kwargs,
                                        ) if not isinstance(scale, np.ndarray) else \
                                        transpose_with_flexing_deterrence(
                                            input_tensor=tf.convert_to_tensor(scale),
                                            perm=min_abs_err_perm_1,
                                            output_shape=Y.shape \
                                                if None not in Y.shape and Y.shape != [] else None,
                                            **kwargs,
                                        ),
                                    variance_epsilon=epsilon,
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
                            if min_abs_err < 1e-3:
                                break
                    except Exception as ex:
                        pass

                if min_abs_err_perm_1 != [idx for idx in range(check_length)]:
                    tf_layers_dict[Y.name]['tf_node'] = \
                        tf.nn.batch_normalization(
                            x=input_tensor,
                            mean=\
                                transpose_with_flexing_deterrence(
                                    input_tensor=mean,
                                    perm=min_abs_err_perm_1,
                                    output_shape=Y.shape \
                                        if None not in Y.shape and Y.shape != [] else None,
                                    **kwargs,
                                ) if not isinstance(mean, np.ndarray) else \
                                transpose_with_flexing_deterrence(
                                    input_tensor=tf.convert_to_tensor(mean),
                                    perm=min_abs_err_perm_1,
                                    output_shape=Y.shape \
                                        if None not in Y.shape and Y.shape != [] else None,
                                    **kwargs,
                                ),
                            variance=\
                                transpose_with_flexing_deterrence(
                                    input_tensor=var,
                                    perm=min_abs_err_perm_1,
                                    output_shape=Y.shape \
                                        if None not in Y.shape and Y.shape != [] else None,
                                    **kwargs,
                                ) if not isinstance(var, np.ndarray) else \
                                transpose_with_flexing_deterrence(
                                    input_tensor=tf.convert_to_tensor(var),
                                    perm=min_abs_err_perm_1,
                                    output_shape=Y.shape \
                                        if None not in Y.shape and Y.shape != [] else None,
                                    **kwargs,
                                ),
                            offset=\
                                transpose_with_flexing_deterrence(
                                    input_tensor=offset,
                                    perm=min_abs_err_perm_1,
                                    output_shape=Y.shape \
                                        if None not in Y.shape and Y.shape != [] else None,
                                    **kwargs,
                                ) if not isinstance(offset, np.ndarray) else \
                                transpose_with_flexing_deterrence(
                                    input_tensor=tf.convert_to_tensor(offset),
                                    perm=min_abs_err_perm_1,
                                    output_shape=Y.shape \
                                        if None not in Y.shape and Y.shape != [] else None,
                                    **kwargs,
                                ),
                            scale=\
                                transpose_with_flexing_deterrence(
                                    input_tensor=scale,
                                    perm=min_abs_err_perm_1,
                                    output_shape=Y.shape \
                                        if None not in Y.shape and Y.shape != [] else None,
                                    **kwargs,
                                ) if not isinstance(scale, np.ndarray) else \
                                transpose_with_flexing_deterrence(
                                    input_tensor=tf.convert_to_tensor(scale),
                                    perm=min_abs_err_perm_1,
                                    output_shape=Y.shape \
                                        if None not in Y.shape and Y.shape != [] else None,
                                    **kwargs,
                                ),
                            variance_epsilon=epsilon,
                        )
                tf_type = tf.nn.batch_normalization

    # Post-process transpose
    tf_layers_dict[Y.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[Y.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[Y.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'BatchNormalization',
                'tf_inputs': {
                    'X': tf_layers_dict[X.name]['tf_node'],
                    'mean': input_mean,
                    'variance': input_var,
                    'offset': B,
                    'scale': scale,
                    'variance_epsilon': epsilon,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[Y.name]['tf_node'],
                },
            }
        )
