import sys
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
    convert_axis,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
    shape_is_equal_ignore_order,
    dummy_tf_inference,
    onnx_tf_tensor_validation,
    acquisition_of_validation_data,
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
    """Concat

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans = True
    for graph_node_input in graph_node.inputs:
        before_op_output_shape_trans_n = \
            tf_layers_dict.get(graph_node_input.name, {}).get('before_op_output_shape_trans', True)
        before_op_output_shape_trans = \
            before_op_output_shape_trans and before_op_output_shape_trans_n

    values = []
    nhwc_flags = []
    same_input_shape_as_onnxs = []
    for graph_node_input in graph_node.inputs:
        const_or_var = get_constant_or_variable(
            graph_node_input,
            before_op_output_shape_trans,
        )
        if isinstance(const_or_var, gs.Variable):
            values.append(tf_layers_dict[const_or_var.name]['tf_node'])
            nhwc_flags.append(
                tf_layers_dict[const_or_var.name]['nhwc'] \
                    if 'nhwc' in tf_layers_dict[const_or_var.name].keys() else False
            )
            same_input_shape_as_onnxs.append(
                True if graph_node_input.shape is not None and len(graph_node_input.shape) > 0 \
                    and graph_node_input.shape == tf_layers_dict[const_or_var.name]['tf_node'].shape else False
            )
        else:
            values.append(const_or_var)
            nhwc_flags.append(False)
            same_input_shape_as_onnxs.append(
                True if graph_node_input.shape is not None and len(graph_node_input.shape) > 0 \
                    and graph_node_input.shape == const_or_var.shape else False
            )

    # Shape Unmatched Special Avoidance Workaround
    # At least one True value for same_input_shape_as_onnx
    # At least one True value in nhwc_flags
    # same_input_shape_as_onnx == True and nhwc_flags == False and 3D or 4D or 5D tensor is NHWC transposed
    if True in same_input_shape_as_onnxs and True in nhwc_flags:
        before_op_output_shape_trans = True
        new_values = []
        for same_input_shape_as_onnx, nhwc_flag, value in zip(same_input_shape_as_onnxs, nhwc_flags, values):
            if same_input_shape_as_onnx and not nhwc_flag:
                if len(value.shape) == 3:
                    new_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0,2,1],
                            **kwargs,
                        )
                    )
                elif len(value.shape) == 4:
                    new_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0,2,3,1],
                            **kwargs,
                        )
                    )
                elif len(value.shape) == 5:
                    new_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0,2,3,4,1],
                            **kwargs,
                        )
                    )
                else:
                    new_values.append(value)
            else:
                new_values.append(value)
        values = new_values

    graph_node_output: gs.Variable = graph_node.outputs[0]
    onnx_output_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    axis = graph_node.attrs.get('axis', 0)

    if len(values) == 2 \
        and ((not isinstance(values[0], np.ndarray) and isinstance(values[1], np.ndarray)) or (isinstance(values[0], np.ndarray) and not isinstance(values[1], np.ndarray))) \
        and sum([f for f in nhwc_flags]) == 0:

        variable_tensor = values[0] if not isinstance(values[0], np.ndarray) else values[1]
        constant_tensor = values[0] if isinstance(values[0], np.ndarray) else values[1]
        if hasattr(constant_tensor, '__len__'):
            tensor_candidate_for_transpositions = list(itertools.permutations(range(len(constant_tensor.shape))))
            new_values = []
            for tensor_candidate_for_transposition in tensor_candidate_for_transpositions:
                try:
                    _ = tf.concat(
                        values=[variable_tensor, constant_tensor.transpose(tensor_candidate_for_transposition)],
                        axis=axis
                    )
                    before_op_output_shape_trans = True
                    if not isinstance(values[0], np.ndarray):
                        new_values.append(values[0])
                        new_values.append(values[1].transpose(tensor_candidate_for_transposition))
                    else:
                        new_values.append(values[0].transpose(tensor_candidate_for_transposition))
                        new_values.append(values[1])
                    break
                except Exception as ex:
                    pass
            if new_values:
                values = new_values

    # NCHW->NHWC, NCDHW->NDHWC
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(onnx_output_shape) if onnx_output_shape is not None else len(values[0].shape),
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # Param replacement
    before_axis = axis
    axis = replace_parameter(
        value_before_replacement=axis,
        param_target='attributes',
        param_name='axis',
        **kwargs,
    )

    # Preserving Graph Structure (Dict)
    nhwc_judge = True
    for graph_node_input in graph_node.inputs:
        if isinstance(graph_node_input, gs.Variable) \
            and 'nhwc' in tf_layers_dict[graph_node_input.name].keys() \
            and tf_layers_dict[graph_node_input.name]['nhwc'] == True:
                nhwc_judge = nhwc_judge and True
        elif isinstance(graph_node_input, gs.Constant) \
            and hasattr(graph_node_input, 'values') \
            and isinstance(graph_node_input.values, np.ndarray):
                nhwc_judge = nhwc_judge or True
        else:
            nhwc_judge = nhwc_judge and False

    # Set NHWC flag to True if all input tensors are determined by NHWC
    if nhwc_judge:
        tf_layers_dict[graph_node_output.name] = {
            'optype': graph_node.op,
            'shape': onnx_output_shape,
            'dtype': dtype,
            'nhwc': True,
        }
    else:
        tf_layers_dict[graph_node_output.name] = {
            'optype': graph_node.op,
            'shape': onnx_output_shape,
            'dtype': dtype,
        }

    # Generation of TF OP

    # Pre-process transpose
    new_values = []
    for graph_node_input, value in zip(graph_node.inputs, values):
        value = pre_process_transpose(
            value_before_transpose=value,
            param_target='inputs',
            param_name=graph_node_input.name,
            **kwargs,
        )
        new_values.append(
            value \
                if not isinstance(value, np.ndarray) \
                    else tf.convert_to_tensor(value)
        )
    values = new_values

    # TensorFlow does not support Concat for scalar values, so convert to tensor
    values = [
        value if len(value.shape) > 0 else tf.reshape(value, [1]) for value in values
    ]

    # Generation of TF OP
    try:
        # normal concat attempt
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.concat(
                values=values,
                axis=axis,
                name=graph_node.name,
            )
    except:
        # Workaround to reduce error rate when merging tensors with undefined dimensions
        try:
            # Attempts to bind with the axis specified in ONNX
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.concat(
                    values=values,
                    axis=int(graph_node.attrs.get('axis', 0)),
                    name=graph_node.name,
                )
            axis = int(graph_node.attrs.get('axis', 0))
        except:
            # If not successful with the same axis as ONNX, try to combine by other axes
            # Trial in reverse order from the axis at the end
            value_rank = len(values[0].shape)
            succeed = False
            for idx in reversed(range(value_rank)):
                try:
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.concat(
                            values=values,
                            axis=idx,
                            name=graph_node.name,
                        )
                    axis = idx
                    succeed = True
                    break
                except:
                    pass
            if not succeed:
                raise

    # Attempts to force axis correction when the number of axes in the combined tensor do not exactly match.
    # However, if more than 2 patterns of correct answers exist, give up the correction.
    # This workaround is useful when automatic axis correction is practically difficult,
    # such as when all tensors to be combined originate from Transpose or Reshape.
    # https://github.com/PINTO0309/onnx2tf/issues/473
    output_tensor_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    if output_tensor_shape != tf.TensorShape(None):
        output_tensor_rank = len(output_tensor_shape)
        if graph_node.outputs[0].shape is not None \
            and axis != 0 \
            and output_tensor_rank >= 2 \
            and before_axis == axis:

            # Search for valid Concat patterns
            if not shape_is_equal_ignore_order(list(graph_node.outputs[0].shape), list(output_tensor_shape)):
                matched_axes = []
                for dummy_axis in range(1, output_tensor_rank):
                    try:
                        dummy_concat_tensor = \
                            tf.concat(
                                values=values,
                                axis=dummy_axis,
                                name=graph_node.name,
                            )
                        dummy_output_shape = dummy_concat_tensor.shape
                        if shape_is_equal_ignore_order(list(graph_node.outputs[0].shape), list(dummy_output_shape)):
                            matched_axes.append(dummy_axis)
                    except:
                        pass
                # Review Concat axes only if there is one valid join pattern
                if len(matched_axes) == 1:
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.concat(
                            values=values,
                            axis=matched_axes[0],
                            name=graph_node.name,
                        )
                    axis = matched_axes[0]

    # Workaround for post-concat accuracy degradation issues
    # Process only in the case of a Concat of two tensors because the process is too redundant.
    # Input1: [1, 64, 64], Input2: [1, 256, 64], Output: [1, 320, 64]
    if len(values) == 2 \
        and len(values[0].shape) == len(values[1].shape) \
        and len(values[0].shape) >= 3 \
        and sum([1 if isinstance(s, str) else 0 for s in values[0].shape]) == 0 \
        and sum([1 if isinstance(s, str) else 0 for s in values[1].shape]) == 0 \
        and (len(set(values[0].shape[1:])) == 1 or len(set(values[1].shape[1:])) == 1):

        # Obtain ONNX inference results and
        # TensorFlow inference results up to the previous layer of TensorFlow
        input_tensor_1 = values[0]
        input_tensor_2 = values[1]
        onnx_tensor_infos, validation_data_1, validation_data_2 = \
            acquisition_of_validation_data(
                input_tensor_1=input_tensor_1,
                input_tensor_2=input_tensor_2,
                graph_node_output=graph_node_output,
                tf_layers_dict=tf_layers_dict,
                **kwargs,
            )

        min_abs_err = sys.maxsize
        min_abs_err_perm_1: int = [idx for idx in range(len(input_tensor_1.shape))]
        min_abs_err_perm_2: int = [idx for idx in range(len(input_tensor_2.shape))]

        def define_concat(
            *,
            target_input_tensor_1: Any,
            target_perm_1: List,
            target_input_tensor_2: Any,
            target_perm_2: List,
            target_name: str,
            axis: int,
            **kwargs: Dict,
        ):
            return \
                tf.concat(
                    [
                        transpose_with_flexing_deterrence(
                            input_tensor=target_input_tensor_1 \
                                if not isinstance(target_input_tensor_1, np.ndarray) \
                                    else tf.convert_to_tensor(target_input_tensor_1),
                            perm=target_perm_1,
                            **kwargs,
                        ),
                        transpose_with_flexing_deterrence(
                            input_tensor=target_input_tensor_2 \
                                if not isinstance(target_input_tensor_2, np.ndarray) \
                                    else tf.convert_to_tensor(target_input_tensor_2),
                            perm=target_perm_2,
                            **kwargs,
                        ),
                    ],
                    axis=axis,
                    name=target_name,
                )

        tensor_1_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_1.shape))))
        tensor_2_candidate_for_transpositions = list(itertools.permutations(range(len(input_tensor_2.shape))))

        for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
            for tensor_2_candidate_for_transposition in tensor_2_candidate_for_transpositions:
                try:
                    # Build TF dummy model
                    input_1 = tf.keras.Input(
                        shape=validation_data_1.shape[1:],
                        batch_size=validation_data_1.shape[0] \
                            if isinstance(validation_data_1.shape[0], int) else None,
                        name='dummy_input_1',
                        dtype=validation_data_1.dtype,
                    )
                    input_2 = tf.keras.Input(
                        shape=validation_data_2.shape[1:],
                        batch_size=validation_data_2.shape[0] \
                            if isinstance(validation_data_2.shape[0], int) else None,
                        name='dummy_input_2',
                        dtype=validation_data_2.dtype,
                    )
                    dummy_concat = define_concat(
                        target_input_tensor_1=input_1,
                        target_perm_1=list(tensor_1_candidate_for_transposition),
                        target_input_tensor_2=input_2,
                        target_perm_2=list(tensor_2_candidate_for_transposition),
                        target_name=graph_node.name,
                        axis=axis,
                        **kwargs
                    )
                    # Verify that the output shape matches that of ONNX
                    # If the combination of each value of a dimension is not correct,
                    # invalidate the normal processing judgment.
                    onnx_output_shape_prod = np.prod([dim if not isinstance(dim, str) else -1 for dim in onnx_output_shape])
                    concat_output_shapes = list(dummy_concat.shape)
                    concat_output_shape_prod = np.prod([dim if dim is not None else -1 for dim in concat_output_shapes])
                    if onnx_output_shape_prod != concat_output_shape_prod:
                        del input_1
                        del input_2
                        del dummy_concat
                        continue

                    # Perform simple accuracy verification
                    # Terminate when the error is less than 1e-3
                    if onnx_tensor_infos:
                        try:
                            # Search for the axis with the smallest error
                            val_model = tf.keras.Model(
                                inputs=[
                                    input_1,
                                    input_2,
                                ],
                                outputs=[
                                    dummy_concat,
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
                                        validation_data_1,
                                        validation_data_2,
                                    ],
                                )
                            del input_1
                            del input_2
                            del dummy_concat
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

        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            define_concat(
                target_input_tensor_1=input_tensor_1,
                target_perm_1=min_abs_err_perm_1,
                target_input_tensor_2=input_tensor_2,
                target_perm_2=min_abs_err_perm_2,
                target_name=graph_node.name,
                axis=axis,
                **kwargs
            )

    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_inputs = {f"input{idx}": value for idx, value in enumerate(values)}
    tf_inputs['axis'] = axis
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.concat,
                'tf_inputs': tf_inputs,
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
