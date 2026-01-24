"""
Primitive OPs Attention: torch-2.8.0 onnxscript-0.2.2
Single OP Attention: torch-2.10.0 onnx_ir-0.1.15 onnxscript-0.5.7
"""

import torch

class SDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V)

query = torch.rand(32, 8, 128, 64, dtype=torch.float32, device="cpu")
key = torch.rand(32, 8, 128, 64, dtype=torch.float32, device="cpu")
value = torch.rand(32, 8, 128, 64, dtype=torch.float32, device="cpu")

model = SDPA()
model(query, key, value)

ep = torch.export.export(model, (query, key, value),)

OPSET = 23
torch.onnx.export(
    ep,
    (query, key, value),
    f"Attention_{OPSET}.onnx",
    dynamo=False,
    opset_version=OPSET,
)