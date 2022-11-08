import torch
from torchvision import models
import onnx
from onnxsim import simplify

model = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(
    pretrained=True,
    min_size=200,
    max_size=300,
)
model.cuda()
model.eval()


MODEL = 'faster_rcnn'
H = 100
W = 100
onnx_file = f"{MODEL}_{H}x{W}.onnx"
x = torch.randn(1, 3, H, W).cuda()
torch.onnx.export(
    model,
    args=(x),
    f=onnx_file,
    opset_version=11,
    input_names=['input'],
    # output_names=['output'],
)
model_onnx1 = onnx.load(onnx_file)
model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
onnx.save(model_onnx1, onnx_file)

model_onnx2 = onnx.load(onnx_file)
model_simp, check = simplify(model_onnx2)
onnx.save(model_simp, onnx_file)
model_onnx2 = onnx.load(onnx_file)
model_simp, check = simplify(model_onnx2)
onnx.save(model_simp, onnx_file)
model_onnx2 = onnx.load(onnx_file)
model_simp, check = simplify(model_onnx2)
onnx.save(model_simp, onnx_file)
