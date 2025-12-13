#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
import numpy as np
np.random.seed(0)
from torchvision.ops import deform_conv2d

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        kh, kw = 3, 3
        weight = torch.rand(5, 3, kh, kw)
        offset = torch.rand(4, 2 * kh * kw, 8, 8)
        mask = torch.rand(4, kh * kw, 8, 8)
        return deform_conv2d(input, offset, weight, mask=mask)

if __name__ == "__main__":
    OPSET=19
    MODEL = f'DeformConv'
    model = Model()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.randn(4, 3, 10, 10)
    onnx_program = torch.onnx.dynamo_export(model, x)
    onnx_program.save(onnx_file)

    # model_onnx1 = onnx.load(onnx_file)
    # model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    # onnx.save(model_onnx1, onnx_file)
    # model_onnx2 = onnx.load(onnx_file)
    # model_simp, check = simplify(model_onnx2)
    # onnx.save(model_simp, onnx_file)