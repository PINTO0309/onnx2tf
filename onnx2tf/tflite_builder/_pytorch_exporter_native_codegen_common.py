from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from onnx2tf.tflite_builder.ir import ModelIR


@dataclass
class _NativeModelFileWriterContext:
    output_folder_path: str
    model_ir: ModelIR
    metadata: Dict[str, Any]
    tensor_storage_name_map: Dict[str, str]
    package_dir: Path
    preserve_channel_last_tensor_names: Set[str]
    tensor_var_names: Dict[str, str]
    producer_index: Dict[str, int]
    consumer_index: Dict[str, List[int]]
    module_init_lines: List[str] = field(default_factory=list)
    load_specs: List[Tuple[str, str]] = field(default_factory=list)
    runtime_imports: Set[str] = field(default_factory=set)
    forward_lines: List[str] = field(default_factory=list)


@dataclass
class _NativeCodegenBindings:
    module_globals: Dict[str, Any] = field(default_factory=dict)
    compiled_impl: Optional[Callable[..., Any]] = None
    canonicalize_generated_model_source_fn: Optional[Callable[[Path], None]] = None


@dataclass
class _NativeCodegenState:
    context: _NativeModelFileWriterContext
    exec_env: Dict[str, Any] = field(default_factory=dict)
    module_param_tensor_names: Set[str] = field(default_factory=set)
    submodule_state_tensor_names: Set[str] = field(default_factory=set)
    op_module_attr_names: Dict[int, str] = field(default_factory=dict)
    fused_module_specs: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    affine_layer_norm_specs: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    nms_method_specs: List[Dict[str, Any]] = field(default_factory=list)
    module_attr_counts: Dict[str, int] = field(default_factory=dict)
    inlined_constant_tensor_names: Set[str] = field(default_factory=set)
    skipped_op_indices: Set[int] = field(default_factory=set)
    conv_module_pad_specs: Dict[int, Optional[List[int]]] = field(default_factory=dict)
    tensor_expr_aliases: Dict[str, str] = field(default_factory=dict)
    channel_first_tensor_expr_aliases: Dict[str, str] = field(default_factory=dict)
    synthetic_local_var_names: Dict[str, str] = field(default_factory=dict)
    used_local_var_names: Set[str] = field(default_factory=set)
    synthetic_tensor_serial_ref: List[int] = field(default_factory=lambda: [0])
    public_input_names: Set[str] = field(default_factory=set)
    public_layout_bridge_tensor_names: Set[str] = field(default_factory=set)
    load_specs_result: Optional[List[Tuple[str, str]]] = None
