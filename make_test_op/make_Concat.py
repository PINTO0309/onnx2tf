#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
import numpy as np
np.random.seed(0)


class Model1(nn.Module):
    def __init__(
        self,
    ):
        super(Model1, self).__init__()

    def forward(self, x1, x2, x3):
        return torch.cat([x1, x2, x3], dim=-1)

class Model2(nn.Module):
    def __init__(
        self,
    ):
        super(Model2, self).__init__()

    def forward(self, x1):
        return torch.cat(
            [
                x1,
                torch.reshape(torch.arange(0, 49152, dtype=torch.float32), [1,3,128,128]),
                torch.reshape(torch.arange(0, 49152, dtype=torch.float32), [1,3,128,128]),
            ],
            dim=-1
        )


if __name__ == "__main__":
    OPSET=11
    MODEL = f'Concat_var'
    model = Model1()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x1 = torch.randn(1, 3, 128, 128)
    x2 = torch.randn(1, 3, 128, 128)
    x3 = torch.randn(1, 3, 128, 128)
    torch.onnx.export(
        model,
        args=(x1,x2,x3),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            f'{MODEL}_input',
        ],
        output_names=[
            f'{MODEL}_output',
        ],
        do_constant_folding=False,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    MODEL = f'Concat'
    model = Model2()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x1 = torch.randn(1, 3, 128, 128)
    torch.onnx.export(
        model,
        args=(x1),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            f'{MODEL}_input',
        ],
        output_names=[
            f'{MODEL}_output',
        ],
        do_constant_folding=False,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
