import torch
import torch.nn as nn


class Model16(nn.Module):
    def __init__(self):
        super(Model16, self).__init__()

    def forward(self, x, grid):
        ret = torch.nn.functional.grid_sample(input=x, grid=grid)
        return ret

class Model11(nn.Module):
    def __init__(self):
        super(Model11, self).__init__()

    def forward(self, x, grid):
        Nt, C, H, W = x.shape
        grid_H = grid.shape[1]
        grid_W = grid.shape[2]
        xgrid, ygrid = torch.split(
            tensor=grid,
            split_size_or_sections=[1, 1],
            dim=-1,
        )
        mask = (
            (xgrid >= 0) & (ygrid >= 0) & (xgrid < W - 1) & (ygrid < H - 1)
        ).float()
        x0 = torch.floor(xgrid)
        x1 = x0 + 1
        y0 = torch.floor(ygrid)
        y1 = y0 + 1
        wa = ((x1 - xgrid) * (y1 - ygrid)).permute(3, 0, 1, 2)
        wb = ((x1 - xgrid) * (ygrid - y0)).permute(3, 0, 1, 2)
        wc = ((xgrid - x0) * (y1 - ygrid)).permute(3, 0, 1, 2)
        wd = ((xgrid - x0) * (ygrid - y0)).permute(3, 0, 1, 2)
        x0 = (x0 * mask).view(Nt, grid_H, grid_W).long()
        y0 = (y0 * mask).view(Nt, grid_H, grid_W).long()
        x1 = (x1 * mask).view(Nt, grid_H, grid_W).long()
        y1 = (y1 * mask).view(Nt, grid_H, grid_W).long()
        ind = torch.arange(Nt)
        ind = ind\
            .view(Nt, 1)\
            .expand(-1, grid_H)\
            .view(Nt, grid_H, 1)\
            .expand(-1, -1, grid_W)\
            .long()
        x = x.permute(1, 0, 2, 3)
        output_tensor = (
            x[:, ind, y0, x0] * wa \
            + x[:, ind, y1, x0] * wb \
            + x[:, ind, y0, x1] * wc \
            + x[:, ind, y1, x1] * wd
        ).permute(1, 0, 2, 3)
        ret = output_tensor * mask.permute(0, 3, 1, 2).expand(-1, C, -1, -1)
        return ret


model = Model16()
x = torch.randn([1,3,224,224])
grid = torch.randn([1,32,32,2])
onnx_file = f'GridSample_16.onnx'
torch.onnx.export(
    model,
    args=(x, grid),
    f=onnx_file,
    opset_version=16,
    input_names=[
        'image',
        'grid',
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


model = Model11()
x = torch.randn([1,3,224,224])
grid = torch.randn([1,32,32,2])
onnx_file = f'GridSample_11.onnx'
torch.onnx.export(
    model,
    args=(x, grid),
    f=onnx_file,
    opset_version=11,
    input_names=[
        'image',
        'grid',
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
