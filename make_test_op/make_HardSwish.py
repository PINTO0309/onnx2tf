#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
import numpy as np
np.random.seed(0)


class Model(nn.Module):
    def __init__(
        self,
    ):
        super(Model, self).__init__()
        self.n = nn.Hardswish()

    def forward(self, x):
        y = self.n(x)
        return y


if __name__ == "__main__":
    OPSET=14
    MODEL = f'HardSwish'
    model = Model()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            f'{MODEL}_input',
        ],
        output_names=[
            f'{MODEL}_output',
        ],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
