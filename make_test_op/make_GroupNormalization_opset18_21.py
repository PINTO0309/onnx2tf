#! /usr/bin/env python

from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto


def _make_groupnorm_model(opset: int, out_path: Path) -> None:
    n, c, h, w = 1, 4, 2, 2
    num_groups = 2

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [n, c, h, w])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n, c, h, w])

    if opset >= 21:
        scale_shape = [c]
        bias_shape = [c]
    else:
        scale_shape = [num_groups]
        bias_shape = [num_groups]

    scale_data = np.linspace(1.0, 1.0, num=np.prod(scale_shape), dtype=np.float32)
    bias_data = np.zeros(bias_shape, dtype=np.float32)

    scale_init = helper.make_tensor(
        name="scale",
        data_type=TensorProto.FLOAT,
        dims=scale_shape,
        vals=scale_data.flatten().tolist(),
    )
    bias_init = helper.make_tensor(
        name="bias",
        data_type=TensorProto.FLOAT,
        dims=bias_shape,
        vals=bias_data.flatten().tolist(),
    )

    attrs = {
        "epsilon": 1e-5,
        "num_groups": num_groups,
    }
    if opset >= 21:
        attrs["stash_type"] = 1

    node = helper.make_node(
        "GroupNormalization",
        inputs=["X", "scale", "bias"],
        outputs=["Y"],
        **attrs,
    )

    graph = helper.make_graph(
        nodes=[node],
        name=f"GroupNormalization_{opset}",
        inputs=[x_info],
        outputs=[y_info],
        initializer=[scale_init, bias_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    onnx.save(model, out_path.as_posix())


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "ops"
    out_dir.mkdir(parents=True, exist_ok=True)

    for opset in (18, 21):
        out_path = out_dir / f"GroupNormalization_{opset}.onnx"
        _make_groupnorm_model(opset, out_path)


if __name__ == "__main__":
    main()
