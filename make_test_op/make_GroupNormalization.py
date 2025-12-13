#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
import numpy as np
np.random.seed(0)


class pseudo_GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=3, eps=1e-5):
        super(pseudo_GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias


class Model(nn.Module):
    def __init__(
        self,
    ):
        super(Model, self).__init__()
        self.gn = nn.GroupNorm(3, 6)

    def forward(self, x):
        return self.gn(x)


if __name__ == "__main__":
    OPSET=11
    MODEL = f'GroupNormalization'
    model = Model()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.randn(20, 6, 10, 10)
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
        do_constant_folding=False,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    OPSET=18
    MODEL = f'GroupNormalization'
    model = Model()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.randn(20, 6, 10, 10)
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
        do_constant_folding=False,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    onnx_file = f"{MODEL}_{OPSET}_dynamo.onnx"
    onnx_program = torch.onnx.dynamo_export(model, x)
    onnx_program.save(onnx_file)