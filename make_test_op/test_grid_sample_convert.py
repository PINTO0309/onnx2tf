import os
import time
import random
import tempfile
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
import onnxruntime
from onnx2tf import onnx2tf
from onnx2tf.utils.logging import *

class GridSampleOperator(torch.nn.Module):
    def forward(self, input: torch.Tensor, grid: torch.Tensor):
        return F.grid_sample(
            input=input,
            grid=grid,
            padding_mode="zeros",
            mode="bilinear",
            align_corners=True,
        )


def run_torch_grid_sampler(input, grid):
    grid_sampler = GridSampleOperator()
    torch_output = grid_sampler(
        torch.tensor(input, dtype=torch.float32),
        torch.tensor(grid, dtype=torch.float32),
    )
    torch_output = torch_output.cpu().detach().numpy()
    return torch_output


def run_onnx_grid_sampler(input, grid, onnx_model_file):
    session = onnxruntime.InferenceSession(
        path_or_bytes=onnx_model_file,
        providers=['CPUExecutionProvider'],
    )

    onnx_output = session.run(None, {"input": input, "grid": grid})
    return onnx_output[0]


def run_tflite_grid_sampler(input, grid, tflite_model_file):
    # Load the TFLite model and allocate tensors
    interpreter = Interpreter(model_path=tflite_model_file)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]["index"], tf.convert_to_tensor(input, dtype=tf.float32)
    )
    interpreter.set_tensor(
        input_details[1]["index"], tf.convert_to_tensor(grid, dtype=tf.float32)
    )

    interpreter.invoke()

    tflite_output = interpreter.get_tensor(output_details[0]["index"])
    # TFLite dimension: NHWC -> NCHW
    tflite_output = np.transpose(tflite_output, [0, 3, 1, 2])
    return tflite_output


def test_grid_sample_convert(n, c, h_in, w_in, h_out, w_out):
    with tempfile.TemporaryDirectory() as tmp_path:
        model_name = "grid_sample_reproduction"

        # 1. pytorch sample input i.e. 4D feature map dimension: NCHW
        input = np.random.randn(n, c, h_in, w_in).astype(np.float32)
        grid = np.random.random(size=[n, h_out, w_out, 2]).astype(np.float32)
        grid = 2.0 * grid - 1.0
        info(
            Color.GREEN(
                f"[GridSampleTest] create sample input: {n}x{c}x{h_in}x{w_in}, grid: {n}x{h_out}x{w_out}x2."
            )
        )

        # 2. add randomly Out-Of-Range grids
        ratio = np.random.uniform(low=0.0, high=0.1)
        n_grid_pixs = n * h_out * w_out
        n_grid_outs = int(ratio * n_grid_pixs)
        outs = random.sample(population=list(range(n_grid_pixs)), k=n_grid_outs)
        i, j, k = np.unravel_index(outs, shape=[n, h_out, w_out])
        grid[i, j, k, :] = np.random.uniform(low=2.0, high=100.0, size=[n_grid_outs, 2])
        grid[i, j, k, :] *= np.where(
            np.random.uniform(low=0.0, high=1.0, size=[n_grid_outs, 2]) < 0.5,
            -1.0,
            1.0,
        )
        info(
            Color.GREEN(
                f"[GridSampleTest] add random Out-Of-Range outliers, {ratio * 100.0:.2f}% i.e. {n_grid_outs} grids."
            )
        )

        # 3. pytorch model export to onnx
        onnx_model_file = os.path.join(tmp_path, f"{model_name}.onnx")
        torch.onnx.export(
            GridSampleOperator(),
            {
                "input": torch.tensor(input, dtype=torch.float32),
                "grid": torch.tensor(grid, dtype=torch.float32),
            },
            onnx_model_file,
            opset_version=16,
            input_names=["input", "grid"],
            output_names=["output"],
        )
        info(
            Color.GREEN(
                f"[GridSampleTest] pytorch model export to onnx: {onnx_model_file}."
            )
        )

        # 4. onnx model convert to tflite
        tf_model_path = os.path.join(tmp_path, f"{model_name}.tf")
        tflite_model_file = os.path.join(tf_model_path, f"{model_name}_float32.tflite")
        onnx2tf.convert(
            input_onnx_file_path=onnx_model_file,
            output_folder_path=tf_model_path,
            keep_shape_absolutely_input_names=["input", "grid"],
        )
        info(
            Color.GREEN(
                f"[GridSampleTest] onnx model convert to tflite: {tflite_model_file}."
            )
        )

        # 5. compute model output for pytorch / onnx / tflite
        grid_sampler_funcs = {
            "PYTORCH": run_torch_grid_sampler,
            "ONNX": partial(run_onnx_grid_sampler, onnx_model_file=onnx_model_file),
            "TFLITE": partial(
                run_tflite_grid_sampler, tflite_model_file=tflite_model_file
            ),
        }
        grid_sampler_outputs = {}
        for framework, grid_sampler_func in grid_sampler_funcs.items():
            start = time.time()
            grid_sampler_outputs[framework] = grid_sampler_func(input, grid)
            end = time.time()
            info(
                Color.GREEN(
                    f"[GridSampleTest] {framework} elapsed time: {float(end - start):.4f} seconds."
                )
            )

        # 6. check output consistency for pytorch / onnx / tflite
        groundtruth = "PYTORCH"
        for framework in grid_sampler_outputs.keys():
            if framework == groundtruth:
                continue

            assert np.allclose(
                grid_sampler_outputs[framework],
                grid_sampler_outputs[groundtruth],
                atol=1e-6,
                rtol=1e-6,
            ), error(
                Color.RED(
                    f"[GridSampleTest] {framework} runtime consistency check failed!"
                )
            )


if __name__ == "__main__":
    test_grid_sample_convert(32, 16, 32, 64, 48, 54)

    test_grid_sample_convert(32, 16, 1, 64, 48, 54)
    test_grid_sample_convert(32, 16, 32, 1, 48, 54)
    test_grid_sample_convert(32, 16, 1, 1, 48, 54)

    test_grid_sample_convert(32, 16, 32, 64, 1, 54)
    test_grid_sample_convert(32, 16, 32, 64, 48, 1)
    test_grid_sample_convert(32, 16, 32, 64, 1, 1)
