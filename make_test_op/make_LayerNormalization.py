#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
import numpy as np
np.random.seed(0)


class LayerNormalization(nn.Module):
    def __init__(
        self,
        embedding_dim,
        weight,
        bias,
    ):
        super(LayerNormalization, self).__init__()
        self.embedding_dim = embedding_dim
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        layernormed = nn.functional.layer_norm(
            input=x,
            normalized_shape=self.embedding_dim,
            weight=self.weight,
            bias=self.bias,
            eps=1e-05,
        )
        return layernormed

if __name__ == "__main__":
    OPSET=[11, 17]

    for opset in OPSET:
        MODEL = f'LayerNormalization1D'
        batch, sentence_length, embedding_dim = 20, 5, 10
        input = torch.randn(
            size=batch,
            generator=sentence_length,
            out=embedding_dim,
        )
        embedding_dim_tensor = torch.zeros(
            size=embedding_dim,
        )
        model = LayerNormalization(
            embedding_dim=[embedding_dim],
            weight=torch.tensor(
                data=torch.full_like(
                    input=embedding_dim_tensor,
                    fill_value=0.1
                ),
                dtype=torch.float32,
            ),
            bias=torch.tensor(
                torch.full_like(
                    input=embedding_dim_tensor,
                    fill_value=0.2,
                ),
                dtype=torch.float32,
            ),
        )
        onnx_file = f"{MODEL}_{opset}.onnx"
        torch.onnx.export(
            model,
            args=(input),
            f=onnx_file,
            opset_version=opset,
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

        MODEL = f'LayerNormalization2D'
        N, C, H, W = 20, 5, 10, 10
        embedding_dim = [C, H, W]
        embedding_dim_tensor = torch.zeros(
            size=[C, H, W],
        )
        input = torch.randn(N, C, H, W)
        model = LayerNormalization(
            embedding_dim=embedding_dim,
            weight=torch.tensor(
                data=torch.full_like(
                    input=embedding_dim_tensor,
                    fill_value=0.1,
                ),
                dtype=torch.float32,
            ),
            bias=torch.tensor(
                data=torch.full_like(
                    input=embedding_dim_tensor,
                    fill_value=0.2,
                ),
                dtype=torch.float32,
            ),
        )
        onnx_file = f"{MODEL}_{opset}.onnx"
        torch.onnx.export(
            model,
            args=(input),
            f=onnx_file,
            opset_version=opset,
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

        MODEL = f'LayerNormalization3D'
        N, C, D, H, W = 20, 5, 10, 10, 10
        embedding_dim = [C, D, H, W]
        embedding_dim_tensor = torch.zeros(
            size=[C, D, H, W],
        )
        input = torch.randn(N, C, D, H, W)
        model = LayerNormalization(
            embedding_dim=embedding_dim,
            weight=torch.tensor(
                data=torch.full_like(
                    input=embedding_dim_tensor,
                    fill_value=0.1,
                ),
                dtype=torch.float32,
            ),
            bias=torch.tensor(
                data=torch.full_like(
                    input=embedding_dim_tensor,
                    fill_value=0.2,
                ),
                dtype=torch.float32,
            ),
        )
        onnx_file = f"{MODEL}_{opset}.onnx"
        torch.onnx.export(
            model,
            args=(input),
            f=onnx_file,
            opset_version=opset,
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
