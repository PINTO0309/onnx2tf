import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf
from onnx2tf import onnx2tf
from onnx2tf.utils.logging import *


class SoftmaxOperator(torch.nn.Module):

    def forward(self, input: torch.Tensor, axis: torch.Tensor):
        return F.softmax(input=input, dim=axis.item())


def run_torch_softmax(input, axis):
    softmax = SoftmaxOperator()
    torch_output = softmax(torch.tensor(input, dtype=torch.float32),
                           torch.tensor(axis, dtype=torch.int64))
    torch_output = torch_output.cpu().detach().numpy()
    return torch_output


def run_tflite_softmax(input, tflite_model_file):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"],
                           tf.convert_to_tensor(input, dtype=tf.float32))

    interpreter.invoke()

    tflite_output = interpreter.get_tensor(output_details[0]["index"])
    return tflite_output


def test_softmax_convert(n_dims, axis):
    shape = np.random.randint(low=1, high=32, size=[n_dims])

    input = np.random.randn(*shape).astype(np.float32)
    info(
        Color.GREEN(
            f"[SoftmaxTest] create input with dimension {shape} at axis <{axis}>."
        ))

    with tempfile.TemporaryDirectory() as tmp_path:
        model_name = "softmax_reproduction"

        # 1. export pytorch to onnx model
        onnx_model_file = os.path.join(tmp_path, f"{model_name}.onnx")
        torch.onnx.export(SoftmaxOperator(),
                          (torch.tensor(input, dtype=torch.float32),
                           torch.tensor(axis, dtype=torch.int64)),
                          onnx_model_file,
                          opset_version=16,
                          input_names=["input", "axis"],
                          output_names=["output"],
                          do_constant_folding=False,
                          export_params=True)
        info(
            Color.GREEN(
                f"[SoftmaxTest] pytorch model export to onnx: {onnx_model_file}."
            ))

        # 2. convert onnx to tflite model
        tf_model_path = os.path.join(tmp_path, f"{model_name}.tf")
        tflite_model_file = os.path.join(tf_model_path,
                                         f"{model_name}_float32.tflite")
        onnx2tf.convert(
            input_onnx_file_path=onnx_model_file,
            output_folder_path=tf_model_path,
            keep_shape_absolutely_input_names=["input", "axis"],
        )
        info(
            Color.GREEN(
                f"[SoftmaxTest] onnx model convert to tflite: {tflite_model_file}."
            ))

        # 3. compare pytorch and tflite softmax output
        torch_output = run_torch_softmax(input, axis)
        tflite_output = run_tflite_softmax(input, tflite_model_file)
        assert np.allclose(
            torch_output, tflite_output, atol=1e-6, rtol=1e-6), error(
                Color.RED(
                    f"[SoftmaxTest] tflite runtime consistency check failed!"))


def main():
    test_softmax_convert(n_dims=6, axis=0)
    test_softmax_convert(n_dims=6, axis=3)
    test_softmax_convert(n_dims=6, axis=-1)
    test_softmax_convert(n_dims=6, axis=-5)


if __name__ == "__main__":
    main()