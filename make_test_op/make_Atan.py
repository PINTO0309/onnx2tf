import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        ret = torch.atan(x)
        return ret

model = Model()

x = torch.randn([1,3,224,224])
onnx_file = f'Atan_11.onnx'
torch.onnx.export(
    model,
    args=(x),
    f=onnx_file,
    opset_version=11,
    input_names=[
        'input',
    ],
    output_names=[
        'output',
    ],
)
import onnx
from onnxsim import simplify
model_onnx2 = onnx.load(onnx_file)
model_simp, check = simplify(model_onnx2)
onnx.save(model_simp, onnx_file)
