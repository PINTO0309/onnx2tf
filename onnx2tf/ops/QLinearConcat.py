import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    convert_axis,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    pre_process_transpose,
    replace_parameter,
    shape_is_equal_ignore_order,
    transpose_with_flexing_deterrence,
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """QLinearConcat

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    y_scale_list = [i for i in graph_node.inputs[0::3]]
    y_zero_point_list = [i for i in graph_node.inputs[1::3]]
    input_list = [i for i in graph_node.inputs[2::3]]

    input_tensor_rank = len(input_list[0].shape)

    before_op_output_shape_trans = True
    for graph_node_input in input_list:
        before_op_output_shape_trans_n = \
            tf_layers_dict.get(graph_node_input.name, {}).get('before_op_output_shape_trans', True)
        before_op_output_shape_trans = \
            before_op_output_shape_trans and before_op_output_shape_trans_n

    got_values = []
    nhwc_flags = []
    same_input_shape_as_onnxs = []
    input_is_dequantized_list = []
    got_y_scale_list = []
    got_y_zero_point_list = []
    for input, y_scale, y_zero_point  in zip(input_list, y_scale_list, y_zero_point_list):
        const_or_var = get_constant_or_variable(
            input,
            before_op_output_shape_trans,
        )
        if isinstance(const_or_var, gs.Variable):
            got_values.append(tf_layers_dict[const_or_var.name]['tf_node'])
            nhwc_flags.append(
                tf_layers_dict[const_or_var.name].get('nhwc', False)
            )
            same_input_shape_as_onnxs.append(
                True if input.shape is not None and len(input.shape) > 0 \
                    and input.shape == tf_layers_dict[const_or_var.name]['tf_node'].shape else False
            )
            input_is_dequantized_list.append(
                tf_layers_dict[const_or_var.name].get('is_dequantized', False)
            )
        else:
            got_values.append(const_or_var)
            nhwc_flags.append(False)
            same_input_shape_as_onnxs.append(
                True if input.shape is not None and len(input.shape) > 0 \
                    and input.shape == const_or_var.shape else False
            )
            input_is_dequantized_list.append(False)

        const_or_var = get_constant_or_variable(
            y_scale,
            before_op_output_shape_trans,
        )
        if isinstance(const_or_var, gs.Variable):
            got_y_scale_list.append(tf_layers_dict[const_or_var.name]['tf_node'])
        else:
            got_y_scale_list.append(const_or_var)

        const_or_var = get_constant_or_variable(
            y_zero_point,
            before_op_output_shape_trans,
        )
        if isinstance(const_or_var, gs.Variable):
            got_y_zero_point_list.append(tf_layers_dict[const_or_var.name]['tf_node'])
        else:
            got_y_zero_point_list.append(const_or_var)

    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    axis = graph_node.attrs.get('axis', 0)

    # Shape Unmatched Special Avoidance Workaround
    if True in same_input_shape_as_onnxs and True in nhwc_flags:
        before_op_output_shape_trans = True
        new_values = []
        for same_input_shape_as_onnx, nhwc_flag, value in zip(same_input_shape_as_onnxs, nhwc_flags, got_values):
            if same_input_shape_as_onnx and not nhwc_flag:
                if len(value.shape) == 3:
                    new_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0, 2, 1],
                            **kwargs,
                        )
                    )
                elif len(value.shape) == 4:
                    new_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0, 2, 3, 1],
                            **kwargs,
                        )
                    )
                elif len(value.shape) == 5:
                    new_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0, 2, 3, 4, 1],
                            **kwargs,
                        )
                    )
                else:
                    new_values.append(value)
            else:
                new_values.append(value)
        got_values = new_values

    # Preserving Graph Structure (Dict)
    nhwc_judge = True
    for graph_node_input in input_list:
        if isinstance(graph_node_input, gs.Variable) \
            and tf_layers_dict.get(graph_node_input.name, {}).get('nhwc', False):
            nhwc_judge = nhwc_judge and True
        elif isinstance(graph_node_input, gs.Constant) \
            and hasattr(graph_node_input, 'values') \
            and isinstance(graph_node_input.values, np.ndarray):
            nhwc_judge = nhwc_judge or False
        else:
            nhwc_judge = nhwc_judge and False

    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'is_dequantized': True,
    }
    if nhwc_judge:
        tf_layers_dict[graph_node_output.name]['nhwc'] = True

    # Generation of TF OP

    # NCHW->NHWC, NCDHW->NDHWC
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(shape) if shape is not None else input_tensor_rank,
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

    # TensorFlow does not support Concat for scalar values, so convert to tensor
    values = []
    for graph_node_input, value in zip(input_list, got_values):
        value = pre_process_transpose(
            value_before_transpose=value,
            param_target='inputs',
            param_name=graph_node_input.name,
            **kwargs,
        )
        values.append(value if len(value.shape) > 0 else tf.reshape(value, [1]))

    def _infer_concat_axis(values, output_shape):
        if not values:
            return None
        ranks = []
        shapes = []
        for val in values:
            if val.shape is None or val.shape == tf.TensorShape(None):
                return None
            shape_list = list(val.shape)
            ranks.append(len(shape_list))
            shapes.append(shape_list)
        if len(set(ranks)) != 1:
            return None
        rank = ranks[0]
        candidates = []
        for ax in range(rank):
            ok = True
            for dim in range(rank):
                if dim == ax:
                    continue
                base = shapes[0][dim]
                for s in shapes[1:]:
                    if base is None or s[dim] is None:
                        continue
                    if base != s[dim]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            if output_shape is not None and len(output_shape) == rank:
                out_dim = output_shape[ax]
                if out_dim is not None:
                    sum_dim = 0
                    for s in shapes:
                        if s[ax] is None:
                            sum_dim = None
                            break
                        sum_dim += s[ax]
                    if sum_dim is None or sum_dim != out_dim:
                        continue
            candidates.append(ax)
        if len(candidates) == 1:
            return candidates[0]
        return None

    inferred_axis = _infer_concat_axis(values, shape if shape is not None else None)
    if inferred_axis is not None:
        axis = inferred_axis
    # cast all inputs to float32
    casted_x_list = []
    casted_y_zero_point_list = []
    casted_y_scale_list = []
    for x, y_scale, y_zero_point in zip(values, got_y_scale_list, got_y_zero_point_list):
        casted_x_list.append(tf.cast(x, tf.float32))
        casted_y_scale_list.append(tf.cast(y_scale, tf.float32))
        casted_y_zero_point_list.append(tf.cast(y_zero_point, tf.float32))
    # dequantize x with y_scale, y_zero_point
    dequantized_x_list = []
    for x, y_scale, y_zero_point, is_dequantized in zip(
        casted_x_list,
        casted_y_scale_list,
        casted_y_zero_point_list,
        input_is_dequantized_list,
    ):
        if is_dequantized:
            dequantized_x_list.append(x)
        else:
            dequantized_value = tf.multiply(
                x=tf.subtract(x, y_zero_point),
                y=y_scale,
            )
            dequantized_x_list.append(dequantized_value)

    try:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.concat(
                values=dequantized_x_list,
                axis=axis,
                name=graph_node.name,
            )
    except:
        try:
            onnx_axis = int(graph_node.attrs.get('axis', 0))
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.concat(
                    values=dequantized_x_list,
                    axis=onnx_axis,
                    name=graph_node.name,
                )
            axis = onnx_axis
        except:
            value_rank = len(dequantized_x_list[0].shape)
            succeed = False
            for idx in reversed(range(value_rank)):
                try:
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.concat(
                            values=dequantized_x_list,
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

    output_tensor_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    if output_tensor_shape != tf.TensorShape(None):
        output_tensor_rank = len(output_tensor_shape)
        if graph_node.outputs[0].shape is not None \
            and axis != 0 \
            and output_tensor_rank >= 2 \
            and before_axis == axis:
            if not shape_is_equal_ignore_order(list(graph_node.outputs[0].shape), list(output_tensor_shape)):
                matched_axes = []
                for dummy_axis in range(1, output_tensor_rank):
                    try:
                        dummy_concat_tensor = \
                            tf.concat(
                                values=dequantized_x_list,
                                axis=dummy_axis,
                                name=graph_node.name,
                            )
                        dummy_output_shape = dummy_concat_tensor.shape
                        if shape_is_equal_ignore_order(list(graph_node.outputs[0].shape), list(dummy_output_shape)):
                            matched_axes.append(dummy_axis)
                    except:
                        pass
                if len(matched_axes) == 1:
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.concat(
                            values=dequantized_x_list,
                            axis=matched_axes[0],
                            name=graph_node.name,
                        )
                    axis = matched_axes[0]
                elif not nhwc_judge:
                    onnx_axis = int(graph_node.attrs.get('axis', 0))
                    onnx_axis = output_tensor_rank - 1 if onnx_axis == -1 else onnx_axis
                    if onnx_axis == output_tensor_rank - 1 \
                        and onnx_axis in matched_axes:
                        tf_layers_dict[graph_node_output.name]['tf_node'] = \
                            tf.concat(
                                values=dequantized_x_list,
                                axis=onnx_axis,
                                name=graph_node.name,
                            )
                        axis = onnx_axis

    # Generation of Debug Info
    tf_inputs = {f"input{idx}": dequantized_x for idx, dequantized_x in enumerate(dequantized_x_list)}
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
