#! /usr/bin/env python

from pathlib import Path

import onnx
from onnx import TensorProto, helper
from onnxsim import simplify


def normalize_axis(axis: int, rank: int) -> int:
    if axis < 0:
        axis += rank
    return axis


def compute_output_shape(
    input_shape,
    axis,
    dft_length,
    onesided,
    inverse,
):
    rank = len(input_shape)
    axis_index = normalize_axis(axis, rank)
    length = dft_length if dft_length is not None else input_shape[axis_index]
    if onesided and not inverse:
        out_len = length // 2 + 1
        last_dim = 2
    elif onesided and inverse:
        out_len = length
        last_dim = 1
    else:
        out_len = length
        last_dim = 2
    output_shape = list(input_shape)
    output_shape[axis_index] = out_len
    output_shape[-1] = last_dim
    return output_shape


def build_dft_model(
    *,
    opset,
    name,
    input_shape,
    axis,
    dft_length,
    onesided,
    inverse,
    axis_input=False,
    dft_length_input=False,
):
    output_shape = compute_output_shape(
        input_shape=input_shape,
        axis=axis,
        dft_length=dft_length,
        onesided=onesided,
        inverse=inverse,
    )
    input_info = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        input_shape,
    )
    input_infos = [input_info]
    output_info = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        output_shape,
    )

    inputs = ["input"]
    initializers = []

    if dft_length_input:
        input_infos.append(
            helper.make_tensor_value_info(
                "dft_length",
                TensorProto.INT64,
                [],
            )
        )
        inputs.append("dft_length")
    elif dft_length is not None:
        dft_init = helper.make_tensor(
            "dft_length",
            TensorProto.INT64,
            [],
            [int(dft_length)],
        )
        initializers.append(dft_init)
        inputs.append("dft_length")

    attrs = {
        "inverse": int(inverse),
        "onesided": int(onesided),
    }

    if opset >= 20:
        if axis_input:
            input_infos.append(
                helper.make_tensor_value_info(
                    "axis",
                    TensorProto.INT64,
                    [],
                )
            )
            inputs.append("axis")
        elif axis is not None:
            axis_init = helper.make_tensor(
                "axis",
                TensorProto.INT64,
                [],
                [int(axis)],
            )
            initializers.append(axis_init)
            inputs.append("axis")
    else:
        if axis is not None:
            attrs["axis"] = int(axis)

    node = helper.make_node(
        "DFT",
        inputs=inputs,
        outputs=["output"],
        **attrs,
    )

    graph = helper.make_graph(
        nodes=[node],
        name=name,
        inputs=input_infos,
        outputs=[output_info],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="onnx2tf-dft-test",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    return model


def save_model(model, path: Path):
    model = onnx.shape_inference.infer_shapes(model)
    try:
        model_simp, _ = simplify(model)
        onnx.save(model_simp, str(path))
    except Exception:
        onnx.save(model, str(path))


def main():
    base_dir = Path(__file__).resolve().parent
    ops_dir = base_dir / "ops"
    ops_dir.mkdir(exist_ok=True)

    cases = [
        {
            "name": "DFT_17",
            "opset": 17,
            "input_shape": [1, 8, 1],
            "axis": 1,
            "dft_length": 8,
            "onesided": 0,
            "inverse": 0,
            "dft_length_input": True,
        },
        {
            "name": "DFT_17_onesided",
            "opset": 17,
            "input_shape": [1, 8, 1],
            "axis": 1,
            "dft_length": 8,
            "onesided": 1,
            "inverse": 0,
            "dft_length_input": True,
        },
        {
            "name": "DFT_17_irfft",
            "opset": 17,
            "input_shape": [1, 5, 2],
            "axis": 1,
            "dft_length": 8,
            "onesided": 1,
            "inverse": 1,
            "dft_length_input": True,
        },
        {
            "name": "DFT_20",
            "opset": 20,
            "input_shape": [2, 4, 1],
            "axis": -2,
            "dft_length": 4,
            "onesided": 0,
            "inverse": 0,
            "axis_input": True,
            "dft_length_input": True,
        },
        {
            "name": "DFT_20_onesided",
            "opset": 20,
            "input_shape": [2, 4, 1],
            "axis": -2,
            "dft_length": 4,
            "onesided": 1,
            "inverse": 0,
            "axis_input": True,
            "dft_length_input": True,
        },
        {
            "name": "DFT_20_irfft",
            "opset": 20,
            "input_shape": [2, 3, 2],
            "axis": -2,
            "dft_length": 4,
            "onesided": 1,
            "inverse": 1,
            "axis_input": True,
            "dft_length_input": True,
        },
    ]

    for case in cases:
        model = build_dft_model(**case)
        onnx_file = ops_dir / f"{case['name']}.onnx"
        save_model(model, onnx_file)


if __name__ == "__main__":
    main()
