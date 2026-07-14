from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.pytorch_export_errors import ModelIRPyTorchExportError
from onnx2tf.tflite_builder import pytorch_state_dict_support as state_support


class _FakeTensor:
    def __init__(self, value: object):
        self.array = np.asarray(value)

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    def to(self, *, dtype: np.dtype) -> "_FakeTensor":
        return _FakeTensor(self.array.astype(dtype))

    def numel(self) -> int:
        return int(self.array.size)

    def reshape(self, shape: object) -> "_FakeTensor":
        return _FakeTensor(self.array.reshape(shape))

    def permute(self, *perm: int) -> "_FakeTensor":
        return _FakeTensor(self.array.transpose(perm))

    def contiguous(self) -> "_FakeTensor":
        return self

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def clone(self) -> "_FakeTensor":
        return _FakeTensor(self.array.copy())


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    def _as_tensor(value: object) -> _FakeTensor:
        return value if isinstance(value, _FakeTensor) else _FakeTensor(value)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(as_tensor=_as_tensor),
    )


def test_generated_package_import_sanitizes_name_and_replaces_stale_modules(
    tmp_path,
) -> None:
    package_dir = tmp_path / "123 generated-package"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("VALUE = 17\n", encoding="utf-8")
    expected_module_name = "_onnx2tf_generated_123_generated_package"
    sys.modules[f"{expected_module_name}.stale"] = SimpleNamespace()

    module = state_support._import_generated_package_from_output(str(package_dir))

    assert module.VALUE == 17
    assert module.__name__ == expected_module_name
    assert f"{expected_module_name}.stale" not in sys.modules
    sys.modules.pop(expected_module_name, None)


def test_prepare_exported_state_tensor_preserves_dtype_and_exact_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch)
    target = _FakeTensor(np.zeros((2, 2), dtype=np.float16))

    prepared = state_support._prepare_exported_state_tensor(
        np.arange(4, dtype=np.float32).reshape(2, 2),
        target,
    )

    assert prepared.shape == (2, 2)
    assert prepared.dtype == np.dtype(np.float16)
    np.testing.assert_array_equal(prepared.array, [[0, 1], [2, 3]])


def test_prepare_exported_state_tensor_reshapes_equal_element_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch)

    prepared = state_support._prepare_exported_state_tensor(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        _FakeTensor(np.zeros((3, 2), dtype=np.float32)),
    )

    assert prepared.shape == (3, 2)
    np.testing.assert_array_equal(prepared.array, [[0, 1], [2, 3], [4, 5]])


def test_prepare_exported_state_tensor_rejects_incompatible_element_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch)

    with pytest.raises(ModelIRPyTorchExportError, match="could not align"):
        state_support._prepare_exported_state_tensor(
            np.arange(6, dtype=np.float32),
            _FakeTensor(np.zeros((4,), dtype=np.float32)),
        )


class _FakeGeneratedValue(_FakeTensor):
    pass


class _FakeGeneratedModel:
    def __init__(self, *, load_weights: bool):
        assert load_weights is False

    def state_dict(self) -> dict[str, _FakeGeneratedValue]:
        return {"layer.weight": _FakeGeneratedValue(np.zeros((2,), dtype=np.float32))}


def test_build_native_state_dict_reconciles_and_populates_mapped_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        state_support,
        "_import_generated_package_from_output",
        lambda _: SimpleNamespace(Model=_FakeGeneratedModel),
    )
    monkeypatch.setattr(
        state_support,
        "_prepare_exported_state_tensor",
        lambda source, target: np.asarray(source).copy(),
    )
    model_ir = ModelIR(name="state_dict")
    model_ir.tensors["weight"] = TensorIR(
        name="weight",
        dtype="FLOAT32",
        shape=[2],
        data=np.asarray([3.0, 4.0], dtype=np.float32),
    )

    state_dict = state_support._build_native_generated_state_dict(
        package_path="unused",
        model_ir=model_ir,
        load_specs=[("layer.weight", "weight")],
    )

    np.testing.assert_array_equal(state_dict["layer.weight"], [3.0, 4.0])


def test_build_native_state_dict_rejects_key_mismatch_and_missing_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        state_support,
        "_import_generated_package_from_output",
        lambda _: SimpleNamespace(Model=_FakeGeneratedModel),
    )
    model_ir = ModelIR(name="state_dict_errors")

    with pytest.raises(ModelIRPyTorchExportError, match="reconcile"):
        state_support._build_native_generated_state_dict(
            package_path="unused",
            model_ir=model_ir,
            load_specs=[("other.weight", "weight")],
        )

    with pytest.raises(ModelIRPyTorchExportError, match="concrete tensor data"):
        state_support._build_native_generated_state_dict(
            package_path="unused",
            model_ir=model_ir,
            load_specs=[("layer.weight", "weight")],
        )
