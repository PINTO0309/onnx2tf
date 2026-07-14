# `flatbuffer_direct` PyTorch exporter regression check — 2026-07-14

## Outcome

The `fb-refactor4` PyTorch exporter split had real integration regressions: the
dynamically compiled legacy native-codegen body still resolves a small set of
helpers from `pytorch_exporter.py` module globals, but four helpers were no
longer bound after their implementations moved to dedicated owner modules.
This caused broad `NameError` cascades even though the new owner-module unit
tests passed.

The exporter now explicitly imports the required bindings while leaving their
implementations in their single owner modules:

- `_fold_single_use_static_reshape_chains` from `pytorch_codegen_stages.py`;
- `_constant_int_list` from `pytorch_codegen_utils.py`;
- `_torch_pad_literal_for_constant_tensor` from
  `pytorch_codegen_values.py`;
- `logical_layout_permutation` from `ir.py`.

Architecture tests now enforce these compatibility bindings in addition to
enforcing implementation ownership. No algorithm, public API, artifact name,
dependency, or TensorFlow boundary was changed.

The focused regression gates pass, and one lightweight real model generated
and numerically validated the TFLite, native PyTorch package, TorchScript,
Dynamo ONNX, and ExportedProgram artifacts. No SWAP was detected.

## Environment

The repository environment was audited with `uv sync --extra torch`; no
package or lock-file change was made. The environment contains Python 3.12.12
and CPU-only PyTorch 2.11.0.

The host exports `LD_LIBRARY_PATH` and `PYTHONPATH` entries for a Python 3.10
user installation. Without sanitization, Python 3.12 loads that installation's
`libtorch_python.so` and fails with an undefined `_PyCode_GetExtra` symbol.
All successful PyTorch checks therefore used:

```text
env -u LD_LIBRARY_PATH -u PYTHONPATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync <command>
```

This is an environment-path conflict, not a missing dependency. Tests remained
strictly sequential; no `xdist`, process pool, or parallel inference worker was
used.

## Test results

### Modular and architecture gates

- The 39 modular PyTorch exporter/policy test files plus
  `test_pytorch_layout_utils.py`: **359 passed** in 1.31 seconds.
- `test_flatbuffer_direct_architecture.py` and
  `test_flatbuffer_direct_bulk_runner.py`: **132 passed** in 34.98 seconds.
- Focused static-reshape, fused-Conv, constant-list, and constant-Pad codegen
  checks passed after the bindings were restored. One exact-source assertion
  reached its assertion instead of raising `NameError`; the same assertion
  fails on `fb-refactor3`, so it is not an `fb-refactor4` regression.

### Lightweight real-model artifact gate

`rfdn_64x64.onnx` was the only real model used. The bulk runner invoked
`flatbuffer_direct -cotof -fdopt -fdots -fdodo -fdoep` with a 180-second model
ceiling and a 60-second native-generation ceiling.

| Result | Value |
| --- | ---: |
| Classification | pass |
| Duration | 18.554 s |
| TFLite maximum absolute error | 0.0000371933 |
| PyTorch maximum absolute error | 0.0000433922 |
| TFLite accuracy gate | pass |
| PyTorch accuracy gate | pass |
| SWAP detected | no |

Generated artifacts included float32 and float16 TFLite files, the native
PyTorch package and state dict, TorchScript `.pt`, Dynamo `.onnx` plus external
data, and ExportedProgram `.pt2`. Both maximum absolute errors are far below
the required `1e-1` ceiling.

## Central exporter-suite investigation

The monolithic `test_pytorch_exporter.py` remains useful as a characterization
suite, but it is not currently a clean fast gate.

Before restoring any binding, a complete central/architecture/bulk run
reported **970 passed and 282 failed** in 681.74 seconds. The dominant new
failure was the missing `_fold_single_use_static_reshape_chains` binding.

After restoring that first binding, a second run reached 1,025 of 1,252 tests
before it was manually interrupted after about 20 minutes to honor the
minimal-test policy and avoid further RSS growth. Its completed portion
reported 883 passes and 142 failures. Seventy-three failures were `NameError`
cases; 66 were the three additional `fb-refactor4` binding omissions fixed in
this checkpoint. The other observed missing names are inherited:

- `_parse_binary_sub_args` has no definition in either `fb-refactor3` or the
  current branch;
- runtime-wrapper construction refers to `_GeneratedModel` outside the lazy
  class factory in both branches.

Many remaining failures are exact generated-source expectations or legacy
model-specific characterizations. A representative PIDNet canonicalization
failure and the NHWC-Conv/rank-3-reshape source assertion were reproduced on
`fb-refactor3`. The remainder was not exhaustively classified during this
minimal regression check.

The interrupted test was
`test_export_pytorch_package_uses_shape_signature_for_dynamic_gather_nd_targets`,
inside ExportedProgram inverse-permute archive cleanup. The cleanup function's
AST differs from `fb-refactor3` only by the new local `import torch`; its pass
algorithm is otherwise identical. The test used CPU continuously, reached
approximately 2.6 GiB RSS, and had zero SWAP. It should be treated as a long
test or investigated for independent performance improvement rather than used
as a fast `fb-refactor4` gate.

## Remaining known issues

- The full monolithic PyTorch exporter characterization suite is not green and
  was not completed after the binding fixes.
- The two inherited undefined-name defects above remain outside this
  branch-specific compatibility fix.
- Legacy exact-source expectations need a separate baseline audit before they
  can become a reliable branch regression gate.
- The long ExportedProgram archive-cleanup case needs a bounded performance
  test or optimization before routine full-suite execution.

The recommended fast regression gate is the modular 359-test selection, the
132 architecture/bulk-runner tests, focused codegen representatives for any
touched binding, and one lightweight real model only when artifact integration
needs verification.
