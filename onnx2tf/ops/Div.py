import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    pre_explicit_broadcast,
    explicit_broadcast,
    pre_process_transpose,
    post_process_transpose,
    disable_unnecessary_transpose,
    shape_unmatched_special_avoidance_workaround,
    merge_two_consecutive_identical_ops_into_one,
    deterring_shape_corruption_due_to_broadcast,
    correction_process_for_accuracy_errors,
    nhwc_determination_of_output_value_of_binary_input_op,
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
    """Div

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
        before_op_output_shape_trans \
            if graph_node.inputs[0].shape is not None and len(graph_node.inputs[0].shape) != 1 else False,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans \
            if graph_node.inputs[1].shape is not None and len(graph_node.inputs[1].shape) != 1 else False,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    graph_node_output_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': graph_node_output_shape,
        'dtype': dtype,
        'nhwc': \
            nhwc_determination_of_output_value_of_binary_input_op(
                graph_node_input_1=graph_node_input_1,
                graph_node_input_2=graph_node_input_2,
                tf_layers_dict=tf_layers_dict
            )
    }

    input_tensor_1 = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_tensor_2 = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    disable_strict_mode: bool = kwargs['disable_strict_mode']
    replace_erf_to_pseudo_gelu = "gelu" in kwargs['replace_to_pseudo_operators']
    mul_div_replace_op_names: dict = kwargs['mul_div_replace_op_names']

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

    # Workaround for ConvInteger
    if input_tensor_1.dtype == tf.float32 and input_tensor_2.dtype in [tf.int32, tf.int64]:
        input_tensor_2 = tf.cast(input_tensor_2, dtype=tf.float32)
    elif input_tensor_1.dtype in [tf.int32, tf.int64] and input_tensor_2.dtype == tf.float32:
        input_tensor_1 = tf.cast(input_tensor_1, dtype=tf.float32)

    # Disable unnecessary Transpose
    #   1. If both x and y are gs.Variable
    #   2. If only one of the two is the output of Transpose
    #   3. If the perm of the Transpose is [0,2,1] or [0,3,1,2] or [0,4,1,2,3]
    #   4. Furthermore, if the shape of x and y are mismatched
    graph_node_input_1, graph_node_input_2, input_tensor_1, input_tensor_2 = \
        disable_unnecessary_transpose(
            graph_node_input_1=graph_node_input_1,
            graph_node_input_2=graph_node_input_2,
            input_tensor_1=input_tensor_1,
            input_tensor_2=input_tensor_2,
            **kwargs,
        )

    # Shape Unmatched Special Avoidance Workaround
    # At least one True value for same_input_shape_as_onnx
    # At least one True value in nhwc_flags
    # same_input_shape_as_onnx == True and nhwc_flags == False and 3D or 4D or 5D tensor is NHWC transposed
    input_tensor_1, input_tensor_2 = \
        shape_unmatched_special_avoidance_workaround(
            graph_node_input_1=graph_node_input_1,
            graph_node_input_2=graph_node_input_2,
            input_tensor_1=input_tensor_1,
            input_tensor_2=input_tensor_2,
            tf_layers_dict=tf_layers_dict,
            **kwargs,
        )

    try:
        is_scalar_1 = False
        is_scalar_2 = False
        is_scalar_1_rank = tf.rank(input_tensor_1) == 0
        if hasattr(is_scalar_1_rank, 'numpy'):
            is_scalar_1 = is_scalar_1_rank.numpy()
        is_scalar_2_rank = tf.rank(input_tensor_2) == 0
        if hasattr(is_scalar_2_rank, 'numpy'):
            is_scalar_2 = is_scalar_2_rank.numpy()
        if (is_scalar_1 or is_scalar_2) and graph_node.i().op == 'Gemm':
            pass
        else:
            input_tensor_1, input_tensor_2 = \
                pre_explicit_broadcast(
                    input_tensor_1=input_tensor_1,
                    input_tensor_2=input_tensor_2,
                )

            input_tensor_1, input_tensor_2 = \
                explicit_broadcast(
                    const_or_var_1=input_tensor_1,
                    const_or_var_2=input_tensor_2,
                    graph_node=graph_node,
                    tf_layers_dict= tf_layers_dict,
                )
    except Exception as ex:
        input_tensor_1, input_tensor_2 = \
            pre_explicit_broadcast(
                input_tensor_1=input_tensor_1,
                input_tensor_2=input_tensor_2,
            )

        input_tensor_1, input_tensor_2 = \
            explicit_broadcast(
                const_or_var_1=input_tensor_1,
                const_or_var_2=input_tensor_2,
                graph_node=graph_node,
                tf_layers_dict= tf_layers_dict,
            )

    # Deterring shape corruption due to broadcast
    input_tensor_1, input_tensor_2 = \
        deterring_shape_corruption_due_to_broadcast(
            graph_node_output_shape=graph_node_output_shape,
            input_tensor_1=input_tensor_1,
            input_tensor_2=input_tensor_2,
        )

    # Correction process for accuracy errors
    if not disable_strict_mode:
        input_tensor_1, input_tensor_2 = \
            correction_process_for_accuracy_errors(
                input_tensor_1=input_tensor_1,
                input_tensor_2=input_tensor_2,
                tf_func=tf.math.divide,
                np_func=np.divide,
                graph_node_output_shape=graph_node_output_shape,
                graph_node_output=graph_node_output,
                tf_layers_dict=tf_layers_dict,
                **kwargs,
            )

    # Generation of TF OP
    target_cast_dtype = [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ]

    # TFLite does not support TrueDiv(INT), so TrueDiv is avoided if possible
    x_is_integer = input_tensor_1.dtype in target_cast_dtype
    y_is_integer = input_tensor_2.dtype in target_cast_dtype
    xy_is_integer = x_is_integer and y_is_integer
    input_tensor_1 = input_tensor_1 \
        if not xy_is_integer else tf.cast(input_tensor_1, dtype=tf.float32)
    input_tensor_2 = input_tensor_2 \
        if not xy_is_integer else tf.cast(input_tensor_2, dtype=tf.float32)

    # Generation of TF OP
    # TODO: Temporarily Revert due to missing decision conditions
    # # Merge two consecutive identical OPs into one
    # # https://github.com/PINTO0309/onnx2tf/issues/230
    # #   A constant is calculated in advance only
    # #   when one of the operations of the current OP
    # #   is a constant and one of the operations of
    # #   the next OP is also a constant.
    # # By merging two OPs into one, an accuracy error always occurs
    # # in the merged OP during the accuracy check.
    # # 1. `Mul` -> `Mul` to `Single-Mul` : `10 * 5 * 8 -> 10 * 40`
    # # 2. `Mul` -> `Div` to `Single-Mul` : `10 * 5 / 8 -> 10 * 0.625`
    # # 3. `Div` -> `Mul` to `Single-Mul` : `10 / 5 * 8 -> 10 * 1.6`
    # # 4. `Div` -> `Div` to `Single-Mul` : `10 / 5 / 8 -> 10 * 0.025`
    # # 5. `Sub` -> `Sub` to `Single-Sub` : `10 - 5 - 8 -> 10 - 13`
    # # 6. `Sub` -> `Add` to `Single-Sub` : `10 - 5 + 8 -> 10 + 3`
    # # 7. `Add` -> `Add` to `Single-Add`  : `10 + 5 + 8 -> 10 + 13`
    # # 8. `Add` -> `Sub` to `Single-Add`  : `10 + 5 - 8 -> 10 - 3`
    # divided_tensor, tf_type = merge_two_consecutive_identical_ops_into_one(
    #     graph_node_input_1=graph_node_input_1,
    #     graph_node_input_2=graph_node_input_2,
    #     graph_node_output=graph_node_output,
    #     before_op_output_shape_trans=before_op_output_shape_trans,
    #     input_tensor_1=input_tensor_1,
    #     input_tensor_2=input_tensor_2,
    #     graph_node=graph_node,
    #     tf_layers_dict=tf_layers_dict,
    #     tf_func='Div'
    # )

    # GeLU replace check
    # https://github.com/PINTO0309/onnx2tf/issues/465
    # a. x -> b.Div (a*1.4142135381698608) -> c.Erf (b) -> d.Add (c+1.0) -> e.Mul (a*d) -> f.Mul (f*0.5)
    enable_gelu = False
    if hasattr(input_tensor_2, 'numpy') and np.prod(input_tensor_2.numpy().shape) <= 1 and input_tensor_2.numpy() == 1.4142135:
        try:
            input_node_name = graph_node_input_1.name
            erf_node: gs.Node = None
            add_node: gs.Node = None
            mul_1_node: gs.Node = None
            mul_2_node: gs.Node = None
            if graph_node.o().op == 'Erf':

                erf_node = graph_node.o()

                if erf_node.o().op == 'Add' \
                    and len(erf_node.o().inputs) == 2 \
                    and isinstance(graph_node.o().o().inputs[1], gs.Constant) \
                    and hasattr(erf_node.o().inputs[1], 'values') \
                    and np.prod(erf_node.o().inputs[1].values.shape) <= 1 \
                    and erf_node.o().inputs[1].values == 1.0:

                    add_node = erf_node.o()

                    if add_node.o().op == 'Mul' \
                        and len(add_node.o().inputs) == 2 \
                        and add_node.o().inputs[0].name == input_node_name:

                        mul_1_node = add_node.o()

                        if mul_1_node.o().op == 'Mul' \
                            and isinstance(mul_1_node.o().inputs[1], gs.Constant) \
                            and hasattr(mul_1_node.o().inputs[1], 'values') \
                            and np.prod(mul_1_node.o().inputs[1].values.shape) <= 1 \
                            and mul_1_node.o().inputs[1].values == 0.5:

                            mul_2_node = mul_1_node.o()

                            # Save the name of the OP set to be replaced to GeLU
                            # Div, Erf, Add, Mul, Mul
                            if 'gelu_replace_op_names' not in kwargs:
                                kwargs['gelu_replace_op_names'] = {}
                            kwargs['gelu_replace_op_names'][graph_node.name] = [
                                erf_node.name,
                                add_node.name,
                                mul_1_node.name,
                                mul_2_node.name,
                            ]
                            enable_gelu = True
        except:
            pass

    # Replace with GeLU if available.
    mul_div_op_names = [op_name for op_names in mul_div_replace_op_names.values() for op_name in op_names]
    disable_div = graph_node.name in mul_div_op_names

    # Replace with GeLU if available.
    if not enable_gelu:
        if not disable_div:
            divided_tensor = tf.math.divide(
                x=input_tensor_1,
                y=input_tensor_2,
                name=graph_node.name,
            )
            tf_type = tf.math.divide
        else:
            if len(graph_node.inputs) == 2 \
                and graph_node.inputs[0].name in tf_layers_dict \
                and tf_layers_dict[graph_node.inputs[0].name]['optype'] == 'Input':
                divided_tensor = \
                    tf.identity(
                        input=tf_layers_dict[graph_node.inputs[1].name]['tf_node'],
                        name=graph_node.name,
                    )
            elif len(graph_node.inputs) == 2 \
                and graph_node.inputs[1].name in tf_layers_dict \
                and tf_layers_dict[graph_node.inputs[1].name]['optype'] == 'Input':
                divided_tensor = \
                    tf.identity(
                        input=tf_layers_dict[graph_node.inputs[0].name]['tf_node'],
                        name=graph_node.name,
                    )
            else:
                divided_tensor = \
                    tf.identity(
                        input=input_tensor_1 if graph_node.i(0).name in mul_div_replace_op_names else input_tensor_2,
                        name=graph_node.name,
                    )
            tf_type = tf.identity
    else:
        divided_tensor = tf.nn.gelu(
            features=input_tensor_1,
            approximate=replace_erf_to_pseudo_gelu,
        )
        tf_type = tf.identity

    # Post-process transpose
    divided_tensor = post_process_transpose(
        value_before_transpose=divided_tensor,
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    if xy_is_integer and dtype in target_cast_dtype:
        divided_tensor = tf.cast(divided_tensor, dtype=dtype)

    tf_layers_dict[graph_node_output.name]['tf_node'] = divided_tensor

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_type,
                'tf_inputs': {
                    'x': input_tensor_1,
                    'y': input_tensor_2,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
