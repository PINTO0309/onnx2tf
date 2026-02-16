import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx2tf.gs as gs
from onnx2tf.utils.common_functions import (
    convert_axis,
    _is_output_nhwc_assumed,
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

    def get_rank(shape):
        if shape is None:
            return None
        if isinstance(shape, tf.TensorShape):
            if shape == tf.TensorShape(None):
                return None
            return shape.rank
        if hasattr(shape, 'rank'):
            try:
                rank = shape.rank
                if rank is not None:
                    return int(rank)
            except Exception:
                pass
        if hasattr(shape, 'as_list'):
            try:
                shape_list = shape.as_list()
                if shape_list is not None:
                    return len(shape_list)
            except Exception:
                pass
        try:
            return len(shape)
        except Exception:
            return None

    input_tensor_rank = None
    for graph_node_input in input_list:
        input_tensor_rank = get_rank(getattr(graph_node_input, 'shape', None))
        if input_tensor_rank is not None:
            break

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

    def normalize_shape(shape):
        if shape is None:
            return None
        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()
        elif hasattr(shape, 'as_list'):
            try:
                shape = shape.as_list()
            except Exception:
                pass
        if shape is None:
            return None
        if not isinstance(shape, (list, tuple)):
            try:
                shape = list(shape)
            except TypeError:
                return None
        normalized_shape = []
        for dim in shape:
            if hasattr(dim, 'value'):
                dim = dim.value
            if isinstance(dim, np.generic):
                dim = dim.item()
            normalized_shape.append(dim)
        return normalized_shape

    def is_same_shape(shape_1, shape_2):
        normalized_shape_1 = normalize_shape(shape_1)
        normalized_shape_2 = normalize_shape(shape_2)
        return normalized_shape_1 is not None \
            and normalized_shape_2 is not None \
            and len(normalized_shape_1) > 0 \
            and normalized_shape_1 == normalized_shape_2

    for input, y_scale, y_zero_point  in zip(input_list, y_scale_list, y_zero_point_list):
        const_or_var = get_constant_or_variable(
            input,
            before_op_output_shape_trans,
        )
        if isinstance(const_or_var, gs.Variable):
            got_values.append(tf_layers_dict[const_or_var.name]['tf_node'])
            nhwc_flags.append(
                _is_output_nhwc_assumed(
                    graph_node_input=const_or_var,
                    tf_layers_dict=tf_layers_dict,
                )
            )
            same_input_shape_as_onnxs.append(
                is_same_shape(
                    input.shape,
                    tf_layers_dict[const_or_var.name]['tf_node'].shape,
                )
            )
            input_is_dequantized_list.append(
                tf_layers_dict[const_or_var.name].get('is_dequantized', False)
            )
        else:
            got_values.append(const_or_var)
            nhwc_flags.append(False)
            same_input_shape_as_onnxs.append(
                is_same_shape(
                    input.shape,
                    const_or_var.shape,
                )
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

    onnx_axis = int(graph_node.attrs.get('axis', 0))
    axis = onnx_axis

    # Shape Unmatched Special Avoidance Workaround
    if True in same_input_shape_as_onnxs and True in nhwc_flags:
        before_op_output_shape_trans = True
        new_values = []
        for same_input_shape_as_onnx, nhwc_flag, value in zip(same_input_shape_as_onnxs, nhwc_flags, got_values):
            if same_input_shape_as_onnx and not nhwc_flag:
                value_rank = get_rank(value.shape)
                if value_rank == 3:
                    new_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0, 2, 1],
                            **kwargs,
                        )
                    )
                elif value_rank == 4:
                    new_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0, 2, 3, 1],
                            **kwargs,
                        )
                    )
                elif value_rank == 5:
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
            and _is_output_nhwc_assumed(
                graph_node_input=graph_node_input,
                tf_layers_dict=tf_layers_dict,
            ):
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
    axis_transpose_required = before_op_output_shape_trans or nhwc_judge
    tensor_rank = len(shape) if shape is not None else input_tensor_rank
    if tensor_rank is None:
        for value in got_values:
            tensor_rank = get_rank(getattr(value, 'shape', None))
            if tensor_rank is not None:
                break
    if tensor_rank is not None:
        axis = convert_axis(
            axis=axis,
            tensor_rank=tensor_rank,
            before_op_output_shape_trans=axis_transpose_required,
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
        value_rank = get_rank(getattr(value, 'shape', None))
        values.append(value if value_rank != 0 else tf.reshape(value, [1]))

    def _shape_list(tensor):
        shape = getattr(tensor, 'shape', None)
        if shape is None or shape == tf.TensorShape(None):
            return None
        shape = list(shape)
        normalized = []
        for dim in shape:
            try:
                normalized.append(int(dim) if dim is not None else None)
            except Exception:
                normalized.append(None)
        return normalized

    def _shape_matches_with_perm(src_shape, dst_shape, perm):
        if src_shape is None or dst_shape is None:
            return False
        if len(src_shape) != len(dst_shape) or len(src_shape) != len(perm):
            return False
        for dst_axis, src_axis in enumerate(perm):
            src_dim = src_shape[src_axis]
            dst_dim = dst_shape[dst_axis]
            if src_dim is not None and dst_dim is not None and src_dim != dst_dim:
                return False
        return True

    # If mixed NCHW/NHWC tensors are fed into one Concat, align them to the
    # first tensor layout when the transpose relation is unambiguous.
    if len(values) >= 2:
        ref_shape = _shape_list(values[0])
        if ref_shape is not None and len(ref_shape) == 4:
            aligned_values = [values[0]]
            aligned = False
            for value in values[1:]:
                value_shape = _shape_list(value)
                if value_shape is None or len(value_shape) != 4:
                    aligned_values.append(value)
                    continue
                if _shape_matches_with_perm(value_shape, ref_shape, [0, 1, 2, 3]):
                    aligned_values.append(value)
                    continue
                if _shape_matches_with_perm(value_shape, ref_shape, [0, 3, 1, 2]):
                    aligned_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0, 3, 1, 2],
                            **kwargs,
                        )
                    )
                    aligned = True
                    continue
                if _shape_matches_with_perm(value_shape, ref_shape, [0, 2, 3, 1]):
                    aligned_values.append(
                        transpose_with_flexing_deterrence(
                            input_tensor=value,
                            perm=[0, 2, 3, 1],
                            **kwargs,
                        )
                    )
                    aligned = True
                    continue
                aligned_values.append(value)
            if aligned:
                values = aligned_values

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

    # Apply delayed axis conversion when early conversion was skipped because
    # rank information was unavailable at attribute parsing time.
    if axis_transpose_required and axis == onnx_axis and len(dequantized_x_list) > 0:
        dequantized_rank = None
        for dequantized_x in dequantized_x_list:
            dequantized_rank = get_rank(getattr(dequantized_x, 'shape', None))
            if dequantized_rank is not None:
                break
        if dequantized_rank is not None:
            axis = convert_axis(
                axis=onnx_axis,
                tensor_rank=dequantized_rank,
                before_op_output_shape_trans=True,
            )

    try:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.concat(
                values=dequantized_x_list,
                axis=axis,
                name=graph_node.name,
            )
    except:
        try:
            fallback_axis = onnx_axis
            fallback_rank = get_rank(getattr(dequantized_x_list[0], 'shape', None)) \
                if len(dequantized_x_list) > 0 else None
            if axis_transpose_required and fallback_rank is not None:
                fallback_axis = convert_axis(
                    axis=onnx_axis,
                    tensor_rank=fallback_rank,
                    before_op_output_shape_trans=True,
                )
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.concat(
                    values=dequantized_x_list,
                    axis=fallback_axis,
                    name=graph_node.name,
                )
            axis = fallback_axis
        except:
            value_rank = get_rank(getattr(dequantized_x_list[0], 'shape', None))
            if value_rank is None:
                raise
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
