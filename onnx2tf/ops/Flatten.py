import random
random.seed(0)
import numpy as np
import itertools
np.random.seed(0)
import tensorflow as tf
import tf_keras
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    dummy_tf_inference,
    get_tf_model_inputs,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Flatten

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

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    output_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    axis = graph_node.attrs.get("axis", None)
    if axis is not None:
        if graph_node_input.shape is not None \
            and axis < input_tensor_rank:
            axis = convert_axis(
                axis=axis,
                tensor_rank=len(graph_node_input.shape),
                before_op_output_shape_trans=before_op_output_shape_trans,
            )
    else:
        axis = input_tensor_rank - 1

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': output_shape,
        'dtype': dtype,
    }

    # Param replacement
    axis = replace_parameter(
        value_before_replacement=axis,
        param_target='attributes',
        param_name='axis',
        **kwargs,
    )

    # Param replacement
    input_tensor = replace_parameter(
        value_before_replacement=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    perm = [
        convert_axis(
            axis=idx,
            tensor_rank=input_tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        ) for idx in range(input_tensor_rank)
    ]

    # Brute-force transpose to match ONNX dummy inference outputs when available.
    onnx_tensor_infos_for_validation = kwargs.get('onnx_tensor_infos_for_validation', None)
    test_data_nhwc: np.ndarray = kwargs.get('test_data_nhwc', None)
    custom_input_op_name_np_data_path: str = kwargs.get('custom_input_op_name_np_data_path', None)
    disable_strict_mode: bool = kwargs.get('disable_strict_mode', False)
    if not disable_strict_mode \
        and onnx_tensor_infos_for_validation is not None \
        and onnx_tensor_infos_for_validation.get(graph_node_output.name, None) is not None:
        validation_input = None
        if isinstance(input_tensor, np.ndarray):
            validation_input = input_tensor
        elif hasattr(input_tensor, 'numpy'):
            try:
                validation_input = input_tensor.numpy()
            except Exception:
                validation_input = None
        else:
            try:
                tf_model_inputs = get_tf_model_inputs(tf_layers_dict=tf_layers_dict)
                val_model = tf_keras.Model(
                    inputs=tf_model_inputs,
                    outputs=[input_tensor],
                )
                tf_pre_tensor_infos = dummy_tf_inference(
                    model=val_model,
                    inputs=tf_model_inputs,
                    test_data_nhwc=test_data_nhwc,
                    custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                )
                if len(tf_pre_tensor_infos) >= 1:
                    validation_input = list(tf_pre_tensor_infos.values())[0]
                del val_model
            except Exception:
                validation_input = None
        if validation_input is None:
            onnx_input_name = graph_node.inputs[0].name
            if onnx_tensor_infos_for_validation.get(onnx_input_name, None) is not None:
                validation_input = onnx_tensor_infos_for_validation[onnx_input_name]

        onnx_output = onnx_tensor_infos_for_validation.get(graph_node_output.name, None)
        if validation_input is not None and onnx_output is not None:
            rank = len(validation_input.shape)
            if rank <= 6:
                perm_candidates = itertools.permutations(range(rank))
            else:
                perm_candidates = [perm]

            def _flatten_np(arr, axis):
                if axis == 0:
                    return arr.reshape(1, -1)
                if axis >= arr.ndim:
                    return arr.reshape(-1, 1)
                return arr.reshape(
                    int(np.prod(arr.shape[:axis])),
                    int(np.prod(arr.shape[axis:])),
                )

            matched_perm = None
            matched_axis = None
            for cand in perm_candidates:
                try:
                    cand_arr = np.transpose(validation_input, cand)
                    for axis_candidate in range(0, rank + 1):
                        cand_flat = _flatten_np(cand_arr, axis_candidate)
                        if cand_flat.shape != onnx_output.shape:
                            continue
                        if np.allclose(cand_flat, onnx_output, rtol=0.0, atol=0.0, equal_nan=True):
                            matched_perm = list(cand)
                            matched_axis = axis_candidate
                            break
                    if matched_perm is not None:
                        break
                except Exception:
                    continue
            if matched_perm is not None:
                perm = matched_perm
                if matched_axis is not None:
                    axis = matched_axis

    # Generation of TF OP
    cal_shape = None
    if axis == 0:
        cal_shape = (1, -1)
    elif axis >= input_tensor_rank:
        cal_shape = (-1, 1)
    elif graph_node_output.shape is not None \
        and len(graph_node_output.shape) == 2 \
        and axis == input_tensor_rank - 1 \
        and not isinstance(graph_node_output.shape[0], str):
        cal_shape = (graph_node_output.shape[0], -1)
    elif graph_node_output.shape is not None \
        and len(graph_node_output.shape) == 2 \
        and axis == input_tensor_rank - 1 \
        and isinstance(graph_node_output.shape[0], str):
        try:
            dim_prod = int(np.prod(graph_node_output.shape[1:]))
            cal_shape = (-1, dim_prod)
        except:
            cal_shape = (1, -1)
    elif input_tensor_rank >= 2 \
        and input_tensor_shape[0] is None \
        and len([idx for idx in input_tensor_shape[1:] if idx is not None]) == input_tensor_rank - 1 \
        and axis == 1:
        cal_shape = (-1, np.prod(input_tensor_shape[1:]))
    elif input_tensor_rank >= 2 \
        and input_tensor_shape[0] is None \
        and len([idx for idx in input_tensor_shape[1:] if idx is not None]) != input_tensor_rank - 1 \
        and axis == 1:
        # Use Keras Flatten() if there are two or more undefined dimensions
        cal_shape = None
    elif input_tensor_rank >= 2 \
        and input_tensor_shape[0] is None \
        and len([idx for idx in input_tensor_shape[1:] if idx is not None]) != input_tensor_rank - 1 \
        and axis == 2:
        # Use Keras Flatten() if there are two or more undefined dimensions
        cal_shape = None
    else:
        cal_shape = (
            tf.reduce_prod(input_tensor_shape[0:axis]),
            tf.reduce_prod(input_tensor_shape[axis:tf.size(input_tensor_shape)]),
        )

    # If the output geometry is clear, overwrite with ONNX output geometry
    has_undefined_outputshape = output_shape is None
    if not has_undefined_outputshape:
        has_none_outputshape = None in output_shape
        has_str_outputshape = True in [True for dim in output_shape if isinstance(dim, str)]
        has_undefined_outputshape = has_none_outputshape or has_str_outputshape
    cal_shape = cal_shape if has_undefined_outputshape else output_shape
    input_tensor = transpose_with_flexing_deterrence(
        input_tensor=input_tensor,
        perm=list(perm) if perm is not None else None,
        **kwargs,
    )

    if cal_shape is not None:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.reshape(
                tensor=input_tensor,
                shape=cal_shape,
                name=graph_node.name,
            )
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf_keras.layers.Flatten()(input_tensor)

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
                    'tensor': input_tensor,
                    'shape': cal_shape,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
