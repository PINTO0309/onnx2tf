import re
import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
)
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES
import importlib
from onnx2tf.utils.colors import Color


class While_Loop_CustomLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(While_Loop_CustomLayer, self).__init__()

    def call(self, cond, body, loop_vars, shape_invariants, maximum_iterations):
        return tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=loop_vars,
            shape_invariants=shape_invariants,
            maximum_iterations=maximum_iterations,
        )


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Loop

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
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_input_n_list = []
    for graph_node_input in graph_node.inputs[2:]:
        graph_node_input_n = get_constant_or_variable(
            graph_node_input,
            before_op_output_shape_trans,
        )
        graph_node_input_n_list.append(graph_node_input_n)

    # M はループ終了条件 (ループ回数)
    # ONNXは純粋にループカンタとして処理可能だが、TensorFlowは M に類する引数が無い
    # したがって、グラフの一部として事前に引き渡す必要が有る
    # 1. 引き渡されたループ回数が前のOPからの入力変数の場合はすでにグラフの中に埋め込まれているので何もしない
    # 2. 引き渡されたループ回数が定数(np.ndarray)の場合はグラフの中にまだ埋め込まれていないので埋め込む
    M = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    M = None if isinstance(M, str) and M == "" else M
    M = tf.convert_to_tensor(M) if isinstance(M, np.ndarray) else M
    M = tf.where(
        tf.greater(M, tf.int32.max),
        tf.constant(tf.int32.max, tf.int32),
        tf.cast(M, tf.int32)
    ) if M is not None else tf.constant(tf.int32.max, tf.int32)
    M_name = None
    if not isinstance(graph_node_input_1, np.ndarray):
        graph_node_input_1.name = graph_node_input_1.name.replace(':','_')
        M_name = graph_node_input_1.name
    else:
        M_name = graph_node.inputs[0].name.replace(':','_')
    M_name = f'{M_name}_M'
    if kwargs['output_signaturedefs']:
        M_name = re.sub('^/', 'wa/', M_name)
    tf_layers_dict[f'{M_name}'] = {
        'optype': 'Constant' if hasattr(M, 'numpy') else 'Variable',
        'shape': M.shape,
        'dtype': M.dtype,
    }
    tf_layers_dict[f'{M_name}']['tf_node'] = M

    # ループ１周目に渡すループ継続条件判定値
    # ループ継続判定用bool値
    # デフォルト: True
    cond = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    cond_init = None if isinstance(cond, str) and cond == "" else tf.cast(cond, tf.bool)
    cond_init = tf.convert_to_tensor(cond_init) if isinstance(cond_init, np.ndarray) else cond_init

    cond_init_name = None
    if not isinstance(graph_node_input_2, np.ndarray):
        graph_node_input_2.name = graph_node_input_2.name.replace(':','_')
        cond_init_name = graph_node_input_2.name
    else:
        cond_init_name = graph_node.inputs[0].name.replace(':','_')
    cond_init_name = f'{cond_init_name}_cond_init'
    if kwargs['output_signaturedefs']:
        cond_init_name = re.sub('^/', 'wa/', cond_init_name)
    tf_layers_dict[f'{cond_init_name}'] = {
        'optype': 'Constant' if hasattr(cond_init, 'numpy') else 'Variable',
        'shape': cond_init.shape,
        'dtype': cond_init.dtype,
    }
    tf_layers_dict[f'{cond_init_name}']['tf_node'] = cond_init




    # ボディ部で処理対象とするループカウンタを除いた変数のリスト
    # 前のOPの出力 あるいは 定数
    v_init = [
        tf_layers_dict[graph_node_input_n.name]['tf_node'] \
            if isinstance(graph_node_input_n, gs.Variable) else graph_node_input_n \
                for graph_node_input_n in graph_node_input_n_list
    ]

    # ボディ部で処理対象とするループカウンタを除いた全変数の出力形状
    # shape_invariants (ループのボディ部内部で形状が変化する可能性が有る場合に指定が必要となる出力形状のヒント) に使用する
    v_shapes = [
        tf.TensorShape([None for i in range(len(v.shape))]) for v in v_init
    ]

    # ボディ部のグラフ
    body: gs.Graph = graph_node.attrs["body"]

    # ループカウンタの初期値 ゼロ固定
    iter_cnt_init = tf.convert_to_tensor(np.int32(0))

    scan_outputs_start_index = 1 + len(v_init)
    scan_outputs_init = [
        tf.TensorArray(
            dtype=body.outputs[i].dtype,
            size=0,
            dynamic_size=True
        ) for i in range(scan_outputs_start_index, len(body.outputs))
    ]
    scan_outputs_shapes = [tf.TensorShape(None) for o in scan_outputs_init]

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    # def run_subgraph(iter_cnt,  ):
    #     for body_input in body.inputs:
    #         try:
    #             op = importlib.import_module(f'onnx2tf.ops.Input')
    #         except ModuleNotFoundError as ex:
    #             print(
    #                 f'{Color.RED}ERROR:{Color.RESET} {optype} OP is not yet implemented.'
    #             )
    #             sys.exit(1)
    #         # substitution because saved_model does not allow colons
    #         body_input.name = body_input.name.replace(':','_')
    #         # Substitution because saved_model does not allow leading slashes in op names
    #         if kwargs['output_signaturedefs']:
    #             body_input.name = re.sub('^/', 'wa/', body_input.name)
    #         op.make_node(
    #             graph_input=body_input,
    #             tf_layers_dict=tf_layers_dict,
    #             keep_ncw_or_nchw_or_ncdhw_input_names=[],
    #             keep_nwc_or_nhwc_or_ndhwc_input_names=[],
    #             keep_shape_absolutely_input_names=[],
    #             **kwargs,
    #         )
    #     for body_node in body.nodes:
    #         optype = body_node.op
    #         try:
    #             op = importlib.import_module(f'onnx2tf.ops.{optype}')
    #         except ModuleNotFoundError as ex:
    #             print(
    #                 f'{Color.RED}ERROR:{Color.RESET} {optype} OP is not yet implemented.'
    #             )
    #             sys.exit(1)
    #         # substitution because saved_model does not allow colons
    #         body_node.name = body_node.name.replace(':','_')
    #         # Substitution because saved_model does not allow leading slashes in op names
    #         if kwargs['output_signaturedefs']:
    #             body_node.name = re.sub('^/', 'wa/', body_node.name)
    #         op.make_node(
    #             graph_node=body_node,
    #             tf_layers_dict=tf_layers_dict,
    #             **kwargs,
    #         )
    #         # Resister constant
    #         for output in body.outputs:
    #             if output.name not in tf_layers_dict and isinstance(output, gs.Constant):
    #                 tf_layers_dict[output.name] = {
    #                     'optype': 'Constant',
    #                     'shape': output.values.shape,
    #                     'dtype': output.values.dtype,
    #                 }
    #                 tf_layers_dict[output.name]['tf_node'] = \
    #                     tf.constant(
    #                         output.values,
    #                         dtype=NUMPY_DTYPES_TO_TF_DTYPES[output.values.dtype],
    #                     )
    #     outputs = [tf_layers_dict[output.name]['tf_node'] for output in body.outputs]
    #     for i in range(scan_outputs_start_index, len(outputs)):
    #         s_index = i - scan_outputs_start_index
    #         insert_index = scan_outputs[s_index].size()
    #         scan_outputs[s_index] = scan_outputs[s_index].write(insert_index, outputs[i])
    #     iter_cnt += 1
    #     return iter_cnt, outputs[0], outputs[1:scan_outputs_start_index], scan_outputs
    def run_subgraph(iter_cnt,  ):
        for body_input in body.inputs:
            try:
                op = importlib.import_module(f'onnx2tf.ops.Input')
            except ModuleNotFoundError as ex:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} {optype} OP is not yet implemented.'
                )
                sys.exit(1)
            # substitution because saved_model does not allow colons
            body_input.name = body_input.name.replace(':','_')
            # Substitution because saved_model does not allow leading slashes in op names
            if kwargs['output_signaturedefs']:
                body_input.name = re.sub('^/', 'wa/', body_input.name)
            op.make_node(
                graph_input=body_input,
                tf_layers_dict=tf_layers_dict,
                keep_ncw_or_nchw_or_ncdhw_input_names=[],
                keep_nwc_or_nhwc_or_ndhwc_input_names=[],
                keep_shape_absolutely_input_names=[],
                **kwargs,
            )
        for body_node in body.nodes:
            optype = body_node.op
            try:
                op = importlib.import_module(f'onnx2tf.ops.{optype}')
            except ModuleNotFoundError as ex:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} {optype} OP is not yet implemented.'
                )
                sys.exit(1)
            # substitution because saved_model does not allow colons
            body_node.name = body_node.name.replace(':','_')
            # Substitution because saved_model does not allow leading slashes in op names
            if kwargs['output_signaturedefs']:
                body_node.name = re.sub('^/', 'wa/', body_node.name)
            op.make_node(
                graph_node=body_node,
                tf_layers_dict=tf_layers_dict,
                **kwargs,
            )
            # Resister constant
            for output in body.outputs:
                if output.name not in tf_layers_dict and isinstance(output, gs.Constant):
                    tf_layers_dict[output.name] = {
                        'optype': 'Constant',
                        'shape': output.values.shape,
                        'dtype': output.values.dtype,
                    }
                    tf_layers_dict[output.name]['tf_node'] = \
                        tf.constant(
                            output.values,
                            dtype=NUMPY_DTYPES_TO_TF_DTYPES[output.values.dtype],
                        )
        outputs = [tf_layers_dict[output.name]['tf_node'] for output in body.outputs]
        iter_cnt += 1
        return iter_cnt, outputs[0], outputs[1:scan_outputs_start_index]




    # Regiter v_initial
    # 1. Loop OP で処理対象とする変数のリスト
    # 2. 変数のリストは直前のOPそのもの、あるいは、定数のどちらか


    # Register body - Inputs
    for body_input in body.inputs:
        try:
            op = importlib.import_module(f'onnx2tf.ops.Input')
        except ModuleNotFoundError as ex:
            print(
                f'{Color.RED}ERROR:{Color.RESET} {optype} OP is not yet implemented.'
            )
            sys.exit(1)
        # substitution because saved_model does not allow colons
        body_input.name = body_input.name.replace(':','_')
        # Substitution because saved_model does not allow leading slashes in op names
        if kwargs['output_signaturedefs']:
            body_input.name = re.sub('^/', 'wa/', body_input.name)
        op.make_node(
            graph_input=body_input,
            tf_layers_dict=tf_layers_dict,
            keep_ncw_or_nchw_or_ncdhw_input_names=[],
            keep_nwc_or_nhwc_or_ndhwc_input_names=[],
            keep_shape_absolutely_input_names=[],
            **kwargs,
        )
    # Register body - Nodes
    for body_node in body.nodes:
        optype = body_node.op
        try:
            op = importlib.import_module(f'onnx2tf.ops.{optype}')
        except ModuleNotFoundError as ex:
            print(
                f'{Color.RED}ERROR:{Color.RESET} {optype} OP is not yet implemented.'
            )
            sys.exit(1)
        # substitution because saved_model does not allow colons
        body_node.name = body_node.name.replace(':','_')
        # Substitution because saved_model does not allow leading slashes in op names
        if kwargs['output_signaturedefs']:
            body_node.name = re.sub('^/', 'wa/', body_node.name)
        op.make_node(
            graph_node=body_node,
            tf_layers_dict=tf_layers_dict,
            **kwargs,
        )
    # Register body - Constant outputs
    for output in body.outputs:
        if output.name not in tf_layers_dict and isinstance(output, gs.Constant):
            tf_layers_dict[output.name] = {
                'optype': 'Constant',
                'shape': output.values.shape,
                'dtype': output.values.dtype,
            }
            tf_layers_dict[output.name]['tf_node'] = \
                tf.constant(
                    output.values,
                    dtype=NUMPY_DTYPES_TO_TF_DTYPES[output.values.dtype],
                )

    def condition(i, x):
        return tf.less(i, tf_layers_dict[f'{M_name}']['tf_node'])

    def body_graph(i, x):
        return tf.add(i, 1),

    a = tf.while_loop(
        cond=condition,
        body=body_graph,
        loop_vars=[iter_cnt_init],
    )







    # for loop
    # https://stackoverflow.com/questions/71635459/how-to-use-keras-symbolic-inputs-with-tf-while-loop
    if M is not None and cond_init is None:
        condition = lambda iter_cnt, cond, v, scan_outputs: True
        while_loop_layer = While_Loop_CustomLayer()
        iter_cnt_final, _, v_final, scan_outputs_final = while_loop_layer(
            cond=condition,
            body=run_subgraph,
            loop_vars=[
                iter_cnt_init,
                "",
                v_init,
                scan_outputs_init,
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape(None),
                v_shapes,
                scan_outputs_shapes,
            ],
            maximum_iterations=M,
        )
    # while and do-while loop
    # https://stackoverflow.com/questions/71635459/how-to-use-keras-symbolic-inputs-with-tf-while-loop
    elif M is None and cond_init is not None:
        condition = lambda iter_cnt, cond, v, scan_outputs: tf.reduce_all(tf.equal(cond, True))
        while_loop_layer = While_Loop_CustomLayer()
        iter_cnt_final, cond_final, v_final, scan_outputs_final = while_loop_layer(
            cond=condition,
            body=run_subgraph,
            loop_vars=[
                iter_cnt_init,
                cond_init,
                v_init,
                scan_outputs_init,
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape(None),
                v_shapes,
                scan_outputs_shapes,
            ],
        )
    # combine for loop and while loop together
    # https://stackoverflow.com/questions/71635459/how-to-use-keras-symbolic-inputs-with-tf-while-loop
    elif M is not None and cond_init is not None:
        # condition = lambda iter_cnt, cond, v, scan_outputs: tf.reduce_all(tf.equal(cond, True))
        # while_loop_layer = While_Loop_CustomLayer()
        # iter_cnt_final, cond_final, v_final, scan_outputs_final = while_loop_layer(
        #     cond=condition,
        #     body=run_subgraph,
        #     loop_vars=[
        #         tf.constant(iter_cnt_init, dtype=iter_cnt_init.dtype),
        #         cond_init,
        #         v_init,
        #         scan_outputs_init,
        #     ],
        #     shape_invariants=[
        #         tf.TensorShape([]),
        #         tf.TensorShape(None),
        #         v_shapes,
        #         scan_outputs_shapes,
        #     ],
        #     maximum_iterations=M,
        # )


        test = tf.while_loop(
            cond=cond_init.numpy() \
                if hasattr(cond_init, 'numpy') else cond_init,
            body=run_subgraph,
            loop_vars=[iter_cnt_init, *v_init]
        )
        a = 0



    # M is None and cond is None
    else:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'Both M and cond in Loop are not set at the same time ' +
            f'Tensorflow.(PS. if you want to create a do-while loop ' +
            f'then please set cond to True or 1)\n' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)


    if scan_outputs_start_index == len(body.outputs):
        # there is no scan_output in the body graph
        tf_layers_dict[graph_node_output.name]['tf_node'] =  v_final

    else:
        def true_fn():
            return scan_outputs_final

        def false_fn():
            new_scan_outputs = []
            for i in range(scan_outputs_start_index, len(body.outputs)):
                exp_elem_shape = scan_outputs_init[i-scan_outputs_start_index].element_shape
            elem_shape = []
            for j in range(exp_elem_shape.rank):
                shape_j = 0 if exp_elem_shape[j] is None else exp_elem_shape[j]
                elem_shape.append(shape_j)
            new_scan_outputs.append(
                tf.TensorArray(
                    dtype=body.outputs[i].dtype,
                    size=0,
                    element_shape=tf.TensorShape(elem_shape)
                )
            )
            return new_scan_outputs

        scan_out_final = tf.cond(tf.greater(iter_cnt_final, 0), true_fn, false_fn)
        scan_outputs_tensors = [o.stack() for o in scan_out_final]
        tf_layers_dict[graph_node_output.name]['tf_node'] = v_final + scan_outputs_tensors

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.while_loop,
                'tf_inputs': {
                    'condition': condition,
                    'M': M,
                    'cond': cond_init,
                    'v_initial': v_init,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
