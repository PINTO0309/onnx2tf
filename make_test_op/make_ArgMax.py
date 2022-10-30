# https://qiita.com/natsutan/items/b13429cdb855ee7d77d0

import onnx
from onnx import helper
from onnx import TensorProto
from onnxsim import simplify

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1,3,128,128])
Y = helper.make_tensor_value_info('Y', TensorProto.INT64, [1,3,128,128])


relu = helper.make_node(
    'Relu',
    inputs = ['X'],
    outputs = ['relu_out']
)

argmax_0 = helper.make_node(
    'ArgMax',
    inputs = ['relu_out'],
    outputs = ['argmax0_out'],
    axis=1
)

softmax = helper.make_node(
    'Softmax',
    inputs = ['X'],
    outputs = ['softmax_out'],
    axis = 0
)

argmax_1 = helper.make_node(
    'ArgMax',
    inputs = ['softmax_out'],
    outputs = ['argmax1_out'],
    axis = 1
)

log = helper.make_node(
    'Log',
    inputs = ['X'],
    outputs = ['log_out'],
)

argmax_2 = helper.make_node(
    'ArgMax',
    inputs = ['log_out'],
    outputs = ['argmax2_out'],
    axis = 1
)


concat = helper.make_node(
    'Concat',
    inputs = ['argmax0_out', 'argmax1_out', 'argmax2_out'],
    outputs = ['Y'],
    axis = 1
)

graph_def = helper.make_graph(
    [relu, argmax_0, softmax, argmax_1, log, argmax_2, concat],
    'test-model',
    [X],
    [Y]
)

OPSET=17
model_def = helper.make_model(
    graph_def,
    producer_name='onnx_example',
)

onnx_file = f'ops/ArgMax_{OPSET}.onnx'
onnx.save(model_def, onnx_file)

model_onnx1 = onnx.load(onnx_file)
model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
onnx.save(model_onnx1, onnx_file)

onnx_file_opt = f'ops/ArgMax_{OPSET}_opt.onnx'
model_onnx2 = onnx.load(onnx_file)
model_simp, check = simplify(model_onnx2)
onnx.save(model_simp, onnx_file_opt)
