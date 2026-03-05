import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tf_keras
from typing import Any, Optional, cast
from tensorflow.python.keras.layers import Lambda
import onnx2tf.gs as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    upsampling2d_bilinear,
    upsampling2d_bicubic,
    upsampling2d_nearest,
    upsampling3d_bilinear,
    upsampling3d_bicubic,
    upsampling3d_nearest,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
)
from onnx2tf.utils.logging import *
from onnx2tf.utils.enums import (
    NUMPY_DTYPES_TO_TF_DTYPES,
)

INF_INDEX_VALUE: int = 4294967296
_BICUBIC_MATRIX_CACHE: dict = {}


def _normalize_resize_coordinate_transformation_mode(
    *,
    coordinate_transformation_mode: str,
) -> Optional[str]:
    ctm = str(coordinate_transformation_mode).lower()
    if ctm in {"align_corners", "asymmetric", "half_pixel", "pytorch_half_pixel"}:
        return ctm
    return None


def _compute_resize_source_index(
    *,
    out_index: int,
    input_size: int,
    output_size: int,
    coordinate_transformation_mode: str,
) -> float:
    ctm = str(coordinate_transformation_mode).lower()
    in_size = int(input_size)
    out_size = int(output_size)
    i = float(out_index)
    if ctm == "align_corners":
        if out_size <= 1:
            return 0.0
        return i * float(in_size - 1) / float(out_size - 1)
    if ctm == "asymmetric":
        return i * float(in_size) / float(out_size)
    if ctm == "half_pixel":
        return (i + 0.5) * float(in_size) / float(out_size) - 0.5
    if ctm == "pytorch_half_pixel":
        if out_size > 1:
            return (i + 0.5) * float(in_size) / float(out_size) - 0.5
        return 0.0
    raise ValueError(f"Unsupported coordinate_transformation_mode for Resize(cubic): {coordinate_transformation_mode}")


def _cubic_kernel_weight(*, distance: float, cubic_coeff_a: float) -> float:
    t = float(abs(distance))
    a = float(cubic_coeff_a)
    if t <= 1.0:
        return ((a + 2.0) * t * t * t) - ((a + 3.0) * t * t) + 1.0
    if t < 2.0:
        return (a * t * t * t) - (5.0 * a * t * t) + (8.0 * a * t) - (4.0 * a)
    return 0.0


def _const_hw_from_tensor(
    *,
    values: Optional[Any],
) -> Optional[tuple[int, int]]:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        arr = np.asarray(values).reshape(-1)
    elif isinstance(values, (list, tuple)):
        arr = np.asarray(values).reshape(-1)
    else:
        arr = tf.get_static_value(values)
        if arr is None:
            try:
                arr = values.numpy()
            except Exception:
                return None
        arr = np.asarray(arr).reshape(-1)
    if arr.size != 2:
        return None
    out_h = int(arr[0])
    out_w = int(arr[1])
    if out_h <= 0 or out_w <= 0:
        return None
    return out_h, out_w


def _to_resize_float_attr(
    *,
    value: Any,
    default: float,
) -> float:
    scalar = value
    if isinstance(scalar, np.ndarray):
        arr = np.asarray(scalar).reshape(-1)
        if arr.size == 0:
            return float(default)
        scalar = arr[0]
    elif isinstance(scalar, (list, tuple)):
        if len(scalar) == 0:
            return float(default)
        scalar = scalar[0]
    elif tf.is_tensor(scalar):
        resolved = tf.get_static_value(scalar)
        if resolved is None:
            return float(default)
        arr = np.asarray(resolved).reshape(-1)
        if arr.size == 0:
            return float(default)
        scalar = arr[0]
    try:
        return float(scalar)
    except Exception:
        return float(default)


def _to_resize_bool_attr(
    *,
    value: Any,
    default: bool,
) -> bool:
    scalar = value
    if isinstance(scalar, np.ndarray):
        arr = np.asarray(scalar).reshape(-1)
        if arr.size == 0:
            return bool(default)
        scalar = arr[0]
    elif isinstance(scalar, (list, tuple)):
        if len(scalar) == 0:
            return bool(default)
        scalar = scalar[0]
    elif tf.is_tensor(scalar):
        resolved = tf.get_static_value(scalar)
        if resolved is None:
            return bool(default)
        arr = np.asarray(resolved).reshape(-1)
        if arr.size == 0:
            return bool(default)
        scalar = arr[0]
    try:
        return bool(int(scalar))
    except Exception:
        return bool(scalar)


def _build_resize_cubic_matrix_from_onnx(
    *,
    input_size: int,
    output_size: int,
    coordinate_transformation_mode: str,
    cubic_coeff_a: float,
    exclude_outside: bool,
) -> np.ndarray:
    key = (
        int(input_size),
        int(output_size),
        str(coordinate_transformation_mode).lower(),
        float(cubic_coeff_a),
        bool(exclude_outside),
    )
    cached = _BICUBIC_MATRIX_CACHE.get(key, None)
    if cached is not None:
        return np.asarray(cached, dtype=np.float32)

    in_size = int(input_size)
    out_size = int(output_size)
    if in_size <= 0 or out_size <= 0:
        raise ValueError(
            f"Resize(cubic) requires positive input/output size. input_size={in_size} output_size={out_size}"
        )
    ctm = _normalize_resize_coordinate_transformation_mode(
        coordinate_transformation_mode=coordinate_transformation_mode,
    )
    if ctm is None:
        raise ValueError(
            "Unsupported coordinate_transformation_mode for Resize(cubic). "
            f"coordinate_transformation_mode={coordinate_transformation_mode}"
        )

    matrix = np.zeros((out_size, in_size), dtype=np.float32)
    for out_idx in range(out_size):
        src = _compute_resize_source_index(
            out_index=out_idx,
            input_size=in_size,
            output_size=out_size,
            coordinate_transformation_mode=ctm,
        )
        src_floor = int(np.floor(src))
        row_weight_sum = 0.0
        for offset in [-1, 0, 1, 2]:
            src_idx = int(src_floor + offset)
            weight = _cubic_kernel_weight(
                distance=src - float(src_idx),
                cubic_coeff_a=float(cubic_coeff_a),
            )
            if bool(exclude_outside) and (src_idx < 0 or src_idx >= in_size):
                continue
            if src_idx < 0:
                src_idx = 0
            elif src_idx >= in_size:
                src_idx = in_size - 1
            matrix[out_idx, src_idx] += np.float32(weight)
            row_weight_sum += float(weight)
        if bool(exclude_outside) and abs(row_weight_sum) > 0.0:
            matrix[out_idx, :] = matrix[out_idx, :] / np.float32(row_weight_sum)

    _BICUBIC_MATRIX_CACHE[key] = matrix
    return np.asarray(matrix, dtype=np.float32)


def _try_resize_cubic_without_flex(
    *,
    input_tensor: tf.Tensor,
    new_size: Optional[Any],
    coordinate_transformation_mode: str,
    cubic_coeff_a: float,
    exclude_outside: bool,
    name: str,
) -> Optional[tf.Tensor]:
    input_shape = input_tensor.shape
    if input_shape is None or len(input_shape) != 4:
        return None

    in_h = input_shape[1]
    in_w = input_shape[2]
    in_c = input_shape[3]
    if not isinstance(in_h, int) or not isinstance(in_w, int) or not isinstance(in_c, int):
        return None
    if int(in_h) <= 0 or int(in_w) <= 0 or int(in_c) <= 0:
        return None

    out_hw = _const_hw_from_tensor(values=new_size)
    if out_hw is None:
        return None
    out_h, out_w = out_hw

    ctm = _normalize_resize_coordinate_transformation_mode(
        coordinate_transformation_mode=coordinate_transformation_mode,
    )
    if ctm is None:
        return None

    h_matrix = _build_resize_cubic_matrix_from_onnx(
        input_size=int(in_h),
        output_size=int(out_h),
        coordinate_transformation_mode=ctm,
        cubic_coeff_a=float(cubic_coeff_a),
        exclude_outside=bool(exclude_outside),
    )
    w_matrix = _build_resize_cubic_matrix_from_onnx(
        input_size=int(in_w),
        output_size=int(out_w),
        coordinate_transformation_mode=ctm,
        cubic_coeff_a=float(cubic_coeff_a),
        exclude_outside=bool(exclude_outside),
    )
    h_matrix_const = tf.constant(
        np.asarray(h_matrix, dtype=np.float32).reshape(1, int(out_h), int(in_h)),
        dtype=tf.float32,
        name=f"{name}_resize_cubic_h_matrix",
    )
    w_matrix_const = tf.constant(
        np.asarray(w_matrix, dtype=np.float32).reshape(1, 1, int(out_w), int(in_w)),
        dtype=tf.float32,
        name=f"{name}_resize_cubic_w_matrix",
    )

    original_dtype = input_tensor.dtype
    work = input_tensor
    if original_dtype != tf.float32:
        work = tf.cast(work, tf.float32, name=f"{name}_resize_cubic_input_f32")

    flatten_wc = int(in_w) * int(in_c)
    h_in = tf.reshape(
        work,
        [-1, int(in_h), flatten_wc],
        name=f"{name}_resize_cubic_h_in",
    )
    h_out = tf.matmul(
        h_matrix_const,
        h_in,
        name=f"{name}_resize_cubic_h_out",
    )
    h_nhwc = tf.reshape(
        h_out,
        [-1, int(out_h), int(in_w), int(in_c)],
        name=f"{name}_resize_cubic_h_nhwc",
    )
    y = tf.matmul(
        w_matrix_const,
        h_nhwc,
        name=f"{name}_resize_cubic_output_f32",
    )

    if original_dtype != y.dtype:
        y = tf.cast(y, original_dtype, name=f"{name}_resize_cubic_output_cast")
    return tf.convert_to_tensor(y)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: Any,
):
    """Resize

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

    opset = kwargs['opset']

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    # Workaround to avoid as many Resize failures as possible
    # for models with useless Transpose immediately before them.
    # If the input geometry of the ONNX and the input geometry of the TF model match,
    # the input geometry on the TF model side is forcibly transposed to the NWC or NHWC or NDHWC format.
    # However, if all dimensions of CW or CHW or CDHW have the same value,
    # the forced transposition process is skipped because it may destroy the structure of the model.
    onnx_input_shape = [
        dim if isinstance(dim, int) else None for dim in graph_node.inputs[0].shape
    ] if graph_node.inputs[0].shape is not None else None
    tf_input_shape = [
        dim if isinstance(dim, int) else None for dim in input_tensor_shape
    ]
    if onnx_input_shape is not None \
        and len(onnx_input_shape) > 1 and len(tf_input_shape) > 1 \
        and onnx_input_shape == tf_input_shape:

        shape_for_judging_skip = [
            dim if dim is not None else INF_INDEX_VALUE for dim in onnx_input_shape[1:]
        ]
        if shape_for_judging_skip.count(shape_for_judging_skip[0]) != len(shape_for_judging_skip):
            if len(onnx_input_shape) == 3:
                # 1D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,1],
                    **kwargs,
                )
                before_op_output_shape_trans = True

            elif len(onnx_input_shape) == 4:
                # 2D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,1],
                    **kwargs,
                )
                before_op_output_shape_trans = True

            elif len(onnx_input_shape) == 5:
                # 3D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,4,1],
                    **kwargs,
                )
                before_op_output_shape_trans = True

    roi = None
    scales = None
    if len(graph_node.inputs) >= 2:
        if opset > 10:
            roi = get_constant_or_variable(
                graph_node.inputs[1],
                before_op_output_shape_trans,
            )
        else:
            scales = get_constant_or_variable(
                graph_node.inputs[1],
                before_op_output_shape_trans,
            )
    if len(graph_node.inputs) >= 3:
        scales = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )
    sizes = None
    if len(graph_node.inputs) >= 4:
        sizes = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
        )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    roi = tf_layers_dict[roi.name]['tf_node'] \
        if (isinstance(roi, gs.Variable) and roi.name != '') else roi
    scales = tf_layers_dict[scales.name]['tf_node'] \
        if (isinstance(scales, gs.Variable) and scales.name != '') else scales
    sizes = tf_layers_dict[sizes.name]['tf_node'] \
        if isinstance(sizes, gs.Variable) else sizes

    # Dynamic-shape QAT models can lose layout hints and keep NCHW-ordered
    # Resize constants. Repair common rank-4 constants to NHWC order when
    # patterns are unambiguous.
    if input_tensor_rank == 4:
        if isinstance(scales, np.ndarray):
            scales_arr = np.asarray(scales).reshape(-1)
            if scales_arr.size == 4:
                if np.isclose(float(scales_arr[1]), 1.0) and not np.isclose(float(scales_arr[3]), 1.0):
                    scales = scales_arr[[0, 2, 3, 1]]
        if isinstance(sizes, np.ndarray):
            sizes_arr = np.asarray(sizes).reshape(-1)
            if sizes_arr.size == 4 \
                and input_tensor_shape[-1] is not None \
                and int(sizes_arr[1]) == int(input_tensor_shape[-1]) \
                and int(sizes_arr[3]) != int(input_tensor_shape[-1]):
                sizes = sizes_arr[[0, 2, 3, 1]]

    coordinate_transformation_mode = graph_node.attrs.get('coordinate_transformation_mode', 'half_pixel')
    extrapolation_value = graph_node.attrs.get('extrapolation_value', 0.0)
    mode = graph_node.attrs.get('mode', 'nearest')
    cubic_coeff_a = graph_node.attrs.get('cubic_coeff_a', -0.75)
    exclude_outside = graph_node.attrs.get('exclude_outside', 0)
    antialias = bool(graph_node.attrs.get('antialias', 0))
    keep_aspect_ratio_policy = graph_node.attrs.get('keep_aspect_ratio_policy', 'stretch')
    preserve_aspect_ratio = False
    if keep_aspect_ratio_policy == 'stretch':
        preserve_aspect_ratio = False
    elif keep_aspect_ratio_policy == 'not_larger':
        preserve_aspect_ratio = True
    elif keep_aspect_ratio_policy == 'not_smaller':
        preserve_aspect_ratio = True
    else:
        preserve_aspect_ratio = False

    replace_argmax_to_fused_argmax_and_indices_is_int64 = \
        kwargs['replace_argmax_to_fused_argmax_and_indices_is_int64']
    replace_argmax_to_fused_argmax_and_indices_is_float32 = \
        kwargs['replace_argmax_to_fused_argmax_and_indices_is_float32']
    fused_argmax_scale_ratio = \
        kwargs['fused_argmax_scale_ratio']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # 1D Resize workaround
    # https://github.com/PINTO0309/onnx2tf/issues/283
    resize_one_d = False
    if input_tensor_rank == 3:
        # Reshape 4D N,W,C -> N,H,W,C
        # H scale is fixed at 1x.
        input_tensor = tf.expand_dims(input=input_tensor, axis=1)
        input_tensor_rank = 4
        resize_one_d = True

        if isinstance(sizes, np.ndarray):
            sizes = np.insert(arr=sizes, obj=1, values=1)
        elif sizes is not None and getattr(sizes, 'shape', None) is not None and tf.is_tensor(sizes) and hasattr(sizes, 'numpy'):
            sizes = np.insert(arr=cast(Any, sizes).numpy(), obj=1, values=1)
        elif sizes is not None and sizes.shape is not None and tf_keras.backend.is_keras_tensor(sizes):
            sizes = tf.concat([sizes[:1], [1], sizes[1:]], axis=0)

        if isinstance(scales, np.ndarray):
            scales = np.insert(arr=scales, obj=1, values=1)
        elif scales is not None and getattr(scales, 'shape', None) is not None and tf.is_tensor(scales) and hasattr(scales, 'numpy'):
            scales = np.insert(arr=cast(Any, scales).numpy(), obj=1, values=1)
        elif scales is not None and scales.shape is not None and tf_keras.backend.is_keras_tensor(scales):
            scales = tf.concat([cast(Any, scales)[:1], [1], cast(Any, scales)[1:]], axis=0)

    # Generation of TF OP
    if mode.lower() == "linear":
        mode = tf.image.ResizeMethod.BILINEAR
        if input_tensor_rank == 4:
            tf_resize = upsampling2d_bilinear
        elif input_tensor_rank == 5:
            tf_resize = upsampling3d_bilinear
        else:
            error(
                f'Currently, Resize operations other than 4D and 5D are not supported. '+
                'Pull requests are welcome. \n'+
                f'graph_node.name: {graph_node.name} shape: {input_tensor_shape}'
            )
            sys.exit(1)
    elif mode.lower() == "cubic":
        mode = tf.image.ResizeMethod.BICUBIC
        if input_tensor_rank == 4:
            tf_resize = upsampling2d_bicubic
        elif input_tensor_rank == 5:
            tf_resize = upsampling3d_bicubic
        else:
            error(
                f'Currently, Resize operations other than 4D and 5D are not supported. '+
                'Pull requests are welcome. \n'+
                f'graph_node.name: {graph_node.name} shape: {input_tensor_shape}'
            )
            sys.exit(1)
    else:
        mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        if input_tensor_rank == 4:
            tf_resize = upsampling2d_nearest
        elif input_tensor_rank == 5:
            tf_resize = upsampling3d_nearest
        else:
            error(
                f'Currently, Resize operations other than 4D and 5D are not supported. '+
                'Pull requests are welcome. \n'+
                f'graph_node.name: {graph_node.name} shape: {input_tensor_shape}'
            )
            sys.exit(1)

    new_size = None
    if sizes is not None:
        # sizes is defined
        # The number of elements of 'sizes' should be the same as the rank of input 'X'
        sizes_shape = getattr(sizes, 'shape', None)
        sizes_rank = sizes_shape.rank if sizes_shape is not None and hasattr(sizes_shape, 'rank') else None
        sizes_dim0 = None
        if sizes_shape is not None and hasattr(sizes_shape, 'as_list') and sizes_shape.rank is not None:
            sizes_shape_list = sizes_shape.as_list()
            if sizes_shape_list is not None and len(sizes_shape_list) > 0:
                sizes_dim0 = sizes_shape_list[0]
        if isinstance(sizes, np.ndarray):
            new_size = tf.cast(tf.convert_to_tensor(sizes[1:input_tensor_rank-1]), tf.int32)
        elif tf.is_tensor(sizes) and hasattr(sizes, 'numpy'):
            new_size = tf.cast(tf.convert_to_tensor(cast(Any, sizes).numpy()[1:input_tensor_rank-1]), tf.int32)
        elif tf_keras.backend.is_keras_tensor(sizes) and sizes_rank is not None and sizes_rank > 1:
            new_size = tf.cast(tf.slice(sizes, [1], [input_tensor_rank-2]), tf.int32)
        elif tf_keras.backend.is_keras_tensor(sizes) and sizes_rank == 1 and sizes_dim0 == 2:
            new_size = tf.cast(sizes, tf.int32)
        elif tf_keras.backend.is_keras_tensor(sizes) and sizes_rank == 1 and sizes_dim0 == 4:
            new_size = tf.cast(tf.slice(sizes, [1], [2]), tf.int32)

    elif scales is not None:
        # only scales is defined
        if hasattr(graph_node_output, 'shape') \
            and graph_node_output.shape is not None:
            numeric_bools = np.asarray([isinstance(graph_node_output.shape[-(idx+1)], int) for idx in range(input_tensor_rank-2)])
            if numeric_bools.all():
                new_size = graph_node_output.shape[-len(numeric_bools):len(graph_node_output.shape)] # Estimated from ONNX output shape
            else:
                h_w_scale = cast(Any, scales)[1:input_tensor_rank-1]
                h_w_shape = input_tensor_shape[1:input_tensor_rank-1]
                scales_dtype = cast(Any, scales).dtype
                try:
                    new_size = tf.cast(
                        h_w_scale * tf.cast(
                            h_w_shape,
                            NUMPY_DTYPES_TO_TF_DTYPES[scales_dtype] \
                                if isinstance(scales_dtype, np.dtype) else scales_dtype,
                        ),
                        tf.int32,
                    )
                except:
                    # Workaround when h_w_shape contains undefined dimensions
                    new_size = tf.cast(
                        h_w_scale * tf.cast(
                            tf.slice(
                                tf.shape(input_tensor),
                                begin=[1],
                                size=[input_tensor_rank - 2],
                            ),
                            NUMPY_DTYPES_TO_TF_DTYPES[scales_dtype] \
                                if isinstance(scales_dtype, np.dtype) else scales_dtype,
                        ),
                        tf.int32,
                    )
        else:
            h_w_scale = cast(Any, scales)[1:input_tensor_rank-1]
            h_w_shape = input_tensor_shape[1:input_tensor_rank-1]
            scales_dtype = cast(Any, scales).dtype
            if None not in h_w_shape:
                new_size = tf.cast(
                    h_w_scale * tf.cast(
                        h_w_shape,
                        NUMPY_DTYPES_TO_TF_DTYPES[scales_dtype] \
                            if isinstance(scales_dtype, np.dtype) else scales_dtype,
                    ),
                    tf.int32,
                )
            else:
                h_w_shape = tf.slice(
                    tf.shape(input_tensor),
                    begin=[1],
                    size=[input_tensor_rank - 2],
                )
                new_size = tf.cast(
                    h_w_scale * tf.cast(
                        h_w_shape,
                        NUMPY_DTYPES_TO_TF_DTYPES[scales_dtype] \
                            if isinstance(scales_dtype, np.dtype) else scales_dtype,
                    ),
                    tf.int32,
                )

    if tf.is_tensor(new_size) and hasattr(new_size, '_inferred_value'):
        new_size_values = cast(Any, new_size)._inferred_value
        if (new_size_values is None or new_size_values.count(None) == len(new_size_values)) \
            and graph_node_output.shape is not None \
            and sum([1 if isinstance(s, str) else 0 for s in graph_node_output.shape[1:input_tensor_rank-1]]) == 0:
            tensor_rank = len(graph_node_output.shape)
            convertion_table = [0] + [i for i in range(2, tensor_rank)] + [1]
            new_values = [0] * tensor_rank
            for new_idx, idx in enumerate(convertion_table):
                dim = graph_node_output.shape[idx]
                new_values[new_idx] = int(dim) if isinstance(dim, int) else 0
            new_size = new_values[-(input_tensor_rank-1):-1]

    if (replace_argmax_to_fused_argmax_and_indices_is_int64 \
        or replace_argmax_to_fused_argmax_and_indices_is_float32) \
        and graph_node.o().op == 'ArgMax' \
        and input_tensor_rank == 4:
        new_size = tf.cast(
            tf.cast(tf.convert_to_tensor(cast(Any, new_size)), dtype=tf.float32) * fused_argmax_scale_ratio,
            dtype=tf.int32,
        )

    # Param replacement
    input_tensor = replace_parameter(
        value_before_replacement=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    if len(graph_node.inputs) >= 2:
        roi = replace_parameter(
            value_before_replacement=roi,
            param_target='inputs',
            param_name=graph_node.inputs[1].name,
            **kwargs,
        )
    if len(graph_node.inputs) >= 3:
        scales = replace_parameter(
            value_before_replacement=scales,
            param_target='inputs',
            param_name=graph_node.inputs[2].name,
            **kwargs,
        )
    if len(graph_node.inputs) >= 4:
        new_size = replace_parameter(
            value_before_replacement=new_size,
            param_target='inputs',
            param_name=graph_node.inputs[3].name,
            **kwargs,
        )

    coordinate_transformation_mode = replace_parameter(
        value_before_replacement=coordinate_transformation_mode,
        param_target='attributes',
        param_name='coordinate_transformation_mode',
        **kwargs,
    )
    extrapolation_value = replace_parameter(
        value_before_replacement=extrapolation_value,
        param_target='attributes',
        param_name='extrapolation_value',
        **kwargs,
    )
    mode = replace_parameter(
        value_before_replacement=mode,
        param_target='attributes',
        param_name='mode',
        **kwargs,
    )
    cubic_coeff_a = replace_parameter(
        value_before_replacement=cubic_coeff_a,
        param_target='attributes',
        param_name='cubic_coeff_a',
        **kwargs,
    )
    exclude_outside = replace_parameter(
        value_before_replacement=exclude_outside,
        param_target='attributes',
        param_name='exclude_outside',
        **kwargs,
    )

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    resized_tensor = None
    boxes = None
    box_indices = None
    tf_op_type = None
    align_corners = None
    half_pixel_centers = None
    org_dtype = input_tensor.dtype
    resolved_cubic_coeff_a = _to_resize_float_attr(
        value=cubic_coeff_a,
        default=-0.75,
    )
    resolved_exclude_outside = _to_resize_bool_attr(
        value=exclude_outside,
        default=False,
    )

    # Prefer non-Flex cubic lowering when static spatial sizes are available.
    # This path uses RESHAPE + BATCH_MATMUL + RESHAPE + BATCH_MATMUL and keeps
    # ONNX cubic semantics by deriving interpolation matrices from ONNX attributes.
    if mode == tf.image.ResizeMethod.BICUBIC \
        and input_tensor_rank == 4 \
        and not resize_one_d:
        resolved_ctm = _normalize_resize_coordinate_transformation_mode(
            coordinate_transformation_mode=coordinate_transformation_mode,
        )
        if resolved_ctm is not None:
            resized_tensor = _try_resize_cubic_without_flex(
                input_tensor=input_tensor,
                new_size=new_size,
                coordinate_transformation_mode=resolved_ctm,
                cubic_coeff_a=float(resolved_cubic_coeff_a),
                exclude_outside=bool(resolved_exclude_outside),
                name=graph_node.name,
            )
            if resized_tensor is not None:
                align_corners = bool(resolved_ctm == "align_corners")
                half_pixel_centers = bool(resolved_ctm in {"half_pixel", "pytorch_half_pixel"})
                tf_op_type = tf.matmul

    if coordinate_transformation_mode == "tf_crop_and_resize":
        # get boxes for crop
        indices = [1,2,5,6]
        boxes = tf.expand_dims(tf.gather(roi, indices, axis=0), 0)
        # get box_indices for crop
        box_indices = tf.cast(tf.range(0, input_tensor_shape[0]), dtype=tf.int32)
        # run crop and resize
        resized_tensor = tf.image.crop_and_resize(
            image=input_tensor,
            boxes=boxes,
            box_indices=box_indices,
            crop_size=new_size,
            method=mode,
            extrapolation_value=extrapolation_value,
            name=graph_node.name,
        )
        tf_op_type = tf.image.crop_and_resize

    elif resized_tensor is None and coordinate_transformation_mode == "align_corners" and opset <= 17:
        align_corners = True
        half_pixel_centers = False
        resized_tensor = Lambda(
            tf_resize,
            arguments={
                'new_size': new_size,
                'align_corners': align_corners,
                'half_pixel_centers': half_pixel_centers,
                'name': graph_node.name,
            }
        )(input_tensor)
        tf_op_type = tf_resize

    elif resized_tensor is None and coordinate_transformation_mode == "asymmetric" and opset <= 17:
        align_corners = False
        half_pixel_centers = False
        resized_tensor = Lambda(
            tf_resize,
            arguments={
                'new_size': new_size,
                'align_corners': align_corners,
                'half_pixel_centers': half_pixel_centers,
                'name': graph_node.name,
            }
        )(input_tensor)
        tf_op_type = tf_resize

    elif resized_tensor is None and coordinate_transformation_mode == "half_pixel" and opset <= 17:
        align_corners = False
        half_pixel_centers = True
        resized_tensor = Lambda(
            tf_resize,
            arguments={
                'new_size': new_size,
                'align_corners': align_corners,
                'half_pixel_centers': half_pixel_centers,
                'name': graph_node.name,
            }
        )(input_tensor)
        tf_op_type = tf_resize

    elif resized_tensor is None:
        resized_tensor = tf.image.resize(
            images=input_tensor,
            size=new_size,
            method=mode,
            preserve_aspect_ratio=preserve_aspect_ratio,
            antialias=antialias,
            name=graph_node.name,
        )
        tf_op_type = tf.image.resize

    # 1D Resize workaround
    # https://github.com/PINTO0309/onnx2tf/issues/283
    # Reshape N,H,W,C -> N,W,C
    if resize_one_d:
        resized_tensor = tf.gather(params=resized_tensor, indices=0, axis=1)
    resized_tensor = tf.convert_to_tensor(cast(Any, resized_tensor))

    # TensorFlow's Resize operation casts to Float32 on its own,
    # so we have to change it back to the original type.
    if org_dtype != resized_tensor.dtype:
        resized_tensor = tf.cast(
            x=resized_tensor,
            dtype=org_dtype,
        )

    tf_layers_dict[graph_node_output.name]['tf_node'] = resized_tensor

    # Post-process transpose
    before_trans_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )
    after_trans_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    if 'nhwc' in tf_layers_dict[graph_node_output.name].keys() \
        and tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and before_trans_shape != after_trans_shape:
        tf_layers_dict[graph_node_output.name].pop('nhwc')

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'images': input_tensor,
                    'boxes': boxes,
                    'box_indices': box_indices,
                    'new_size/crop_size': new_size,
                    'method': mode,
                    'extrapolation_value': extrapolation_value,
                    'cubic_coeff_a': resolved_cubic_coeff_a,
                    'exclude_outside': resolved_exclude_outside,
                    'align_corners': align_corners,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
