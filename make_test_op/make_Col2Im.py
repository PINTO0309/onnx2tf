import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 2))

    def forward(self, x):
        return self.fold(x)


x = torch.randn(1, 3 * 2 * 2, 12)
mvn_model = Model()
onnx_file1 = f'col2im.onnx'
torch.onnx.export(
    mvn_model,
    args=(x),
    f=onnx_file1,
    opset_version=18,
    input_names=[
        'input',
    ],
    output_names=[
        'output',
    ],
)
import onnx
from onnxsim import simplify
model_onnx2 = onnx.load(onnx_file1)
model_simp, check = simplify(model_onnx2)
onnx.save(model_simp, onnx_file1)
