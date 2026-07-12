from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class ArtifactPlan:
    """Normalized set of artifacts requested from one ModelIR."""

    float32_tflite: bool = True
    float16_tflite: bool = True
    dynamic_range_quantized_tflite: bool = False
    integer_quantized_tflite: bool = False
    weights: bool = False
    saved_model: bool = False
    pytorch: bool = False
    torchscript: bool = False
    dynamo_onnx: bool = False
    exported_program: bool = False
    split_manifest: bool = False
    op_coverage_report: bool = False

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "ArtifactPlan":
        torchscript = bool(options.get("output_torchscript_from_model_ir", False))
        dynamo_onnx = bool(options.get("output_dynamo_onnx_from_model_ir", False))
        exported_program = bool(
            options.get("output_exported_program_from_model_ir", False)
        )
        return cls(
            dynamic_range_quantized_tflite=bool(
                options.get("output_dynamic_range_quantized_tflite", False)
            ),
            integer_quantized_tflite=bool(
                options.get("output_integer_quantized_tflite", False)
            ),
            weights=bool(options.get("output_weights", False)),
            saved_model=bool(options.get("output_saved_model_from_model_ir", False)),
            pytorch=bool(options.get("output_pytorch_from_model_ir", False))
            or torchscript
            or dynamo_onnx
            or exported_program,
            torchscript=torchscript,
            dynamo_onnx=dynamo_onnx,
            exported_program=exported_program,
            split_manifest=bool(options.get("force_split_manifest", False)),
            op_coverage_report=bool(options.get("report_op_coverage", False)),
        )


@dataclass(frozen=True)
class ConversionRequest:
    """Immutable boundary between the public compatibility layer and core."""

    output_folder_path: str
    output_file_name: str
    onnx_graph: Any
    artifacts: ArtifactPlan
    options: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any]) -> "ConversionRequest":
        normalized: Dict[str, Any] = dict(kwargs)
        normalized.setdefault("output_folder_path", "saved_model")
        normalized.setdefault("output_file_name", "model")
        normalized.setdefault("flatbuffer_direct_show_progress", True)
        options = MappingProxyType(normalized)
        return cls(
            output_folder_path=str(normalized["output_folder_path"]),
            output_file_name=str(normalized["output_file_name"]),
            onnx_graph=normalized.get("onnx_graph"),
            artifacts=ArtifactPlan.from_options(options),
            options=options,
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)


@dataclass(frozen=True)
class ConversionResult:
    """Typed internal result with an exact legacy dictionary adapter."""

    artifacts: Mapping[str, Any]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy_dict(
        cls,
        artifacts: Mapping[str, Any],
        *,
        diagnostics: Optional[Mapping[str, Any]] = None,
    ) -> "ConversionResult":
        return cls(
            artifacts=MappingProxyType(dict(artifacts)),
            diagnostics=MappingProxyType(dict(diagnostics or {})),
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        return dict(self.artifacts)
