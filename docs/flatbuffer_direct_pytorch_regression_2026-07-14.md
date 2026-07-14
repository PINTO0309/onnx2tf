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
enforcing implementation ownership. That initial compatibility-binding
checkpoint changed no algorithm, public API, artifact name, dependency, or
TensorFlow boundary; the bounded inherited-failure repairs performed
afterward are documented below.

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

## Follow-up repairs for inherited failures

After the exact branch comparison was recorded, the inherited non-timeout
failures were reviewed for defects that could be corrected without broad
layout-policy or numerical changes. The following bounded repairs were made:

- the source parser now owns `_parse_binary_sub_args`, and the channel-first
  softmax-mask rewrite accepts the generated rank-three mask target as well as
  the legacy rank-four form; non-literal shapes remain a no-op;
- the generated-package loader obtains its runtime model class through the
  existing lazy `_get_generated_model_cls()` factory instead of referring to
  the undefined historical `_GeneratedModel` symbol;
- the legacy compiled codegen environment explicitly binds
  `_compose_axis_permutations` and `_perm_cf_to_cl` from their existing
  `pytorch_layout_utils.py` owner;
- the TFLite-input path imports SavedModel exporters only when a SavedModel or
  a TensorFlow-dependent bridge is actually requested, so direct PyTorch,
  Dynamo ONNX, ExportedProgram, and TFLite/PyTorch accuracy output no longer
  import TensorFlow or tf-keras;
- CONCAT code generation materializes an inlined scalar constant as a tensor
  on the non-constant input's dtype and device before `_apply_concat` handles
  scalar rank normalization.

These changes add no dependency and do not alter public APIs or artifact
names. Twenty distinct focused cases that had failed during this investigation
now pass. This is a targeted confirmation, not a recomputation of the complete
1,120-test accounting table below. A full per-test 60-second rerun was avoided
because the seven known inherited long-running tests alone add at least seven
minutes and the requested validation policy prioritizes minimal tests.

The follow-up validation results are:

- all 39 modular PyTorch policy/exporter files plus
  `test_pytorch_layout_utils.py`: **360 passed**;
- architecture, bulk-runner, and TensorFlow-optional boundary tests:
  **144 passed**;
- lazy runtime, softmax-mask parser, TFLite-input, and accuracy focused set:
  **14 passed**;
- three additional previously failing central cases covering TFLite-input
  auxiliary artifacts and scalar-CONCAT generated packages: **3 passed**;
- three compatibility-binding cases that previously stopped at missing layout
  helpers: **3 passed**.

Ten representative layout/code-shape cases reached their historical
assertion or runtime failures after the missing helper bindings were restored.
They require broader layout-policy work and were deliberately left unchanged
rather than risking already passing models.

## Historical central exporter-suite investigation

The monolithic `test_pytorch_exporter.py` remains useful as a characterization
suite, but it is not currently a clean fast gate.

Before restoring any binding, a complete central/architecture/bulk run
reported **970 passed and 282 failed** in 681.74 seconds. The dominant new
failure was the missing `_fold_single_use_static_reshape_chains` binding.

After restoring that first binding, an intermediate run reached 1,025 of 1,252
combined tests before it was manually interrupted after about 20 minutes. Its
completed portion reported 883 passes and 142 failures and exposed the three
additional compatibility bindings fixed in this checkpoint. Those partial
numbers are retained only as discovery history; the exact post-fix comparison
below supersedes the earlier estimate of unclassified failures.

### Exact post-binding comparison with `fb-refactor3`

The complete `test_pytorch_exporter.py` collection contains 1,120 tests. The
current test file and expectations were used for both branches. The comparison
ran current commit `bdb95903d330e717900a8550cff4c20071009acc` and detached
`fb-refactor3` commit `c52bc1699b4c7a11a03a535e0b7f10315e1292bd` in separate
processes. Root ONNX files were made visible to the baseline worktree through
temporary symlinks so model-availability checks used the same corpus.

Execution was strictly sequential. A test that did not complete in 60 seconds
was removed from the normal-result set and classified separately as a
long-running test. The exact accounting is:

| Classification | Tests |
| --- | ---: |
| Collected | 1,120 |
| Current branch pass | 995 |
| Current branch non-timeout fail | 118 |
| Long-running, over 60 seconds | 7 |
| Comparable non-timeout failures reproduced on `fb-refactor3` | 112 |
| User-excluded `bread` model tests | 6 |
| Confirmed unresolved `fb-refactor4` regressions | **0** |

All 112 comparable current failures also failed on `fb-refactor3`; no baseline
pass became a current failure. Their normalized first-line failure signatures
matched 112 out of 112. Raw JUnit messages matched exactly for 102 tests. The
other ten differed only in volatile temporary paths, process IDs, timestamps,
object addresses, or assertion-detail ordering while retaining the same
normalized failure signature.

The 112 inherited failures comprise:

| Failure kind | Tests |
| --- | ---: |
| Assertion failure | 69 |
| `NameError` | 23 |
| `RuntimeError` | 8 |
| `ModelIRPyTorchExportError` | 5 |
| `AttributeError` | 5 |
| `RecursionError` | 1 |
| `TorchExportError` | 1 |

At that snapshot, the inherited undefined names were
`_parse_binary_sub_args` in two tests,
`_GeneratedModel` in twelve, `_compose_axis_permutations` in seven, and
`_perm_cf_to_cl` in two. The six user-excluded `bread`/`bread_nonfm` tests also
reached `_perm_cf_to_cl` `NameError`, but were deliberately not run on the
baseline, have not been retested, and are not counted as regression candidates.

The seven long-running tests were individually rerun on `fb-refactor3` with
the same 60-second ceiling. All seven timed out on the baseline as well:

- `test_export_pytorch_package_uses_shape_signature_for_dynamic_gather_nd_targets`;
- `test_scatter_nd_with_constant_shape_supports_dynamo_onnx_and_exported_program`;
- `test_scatter_nd_with_dynamic_prefix_supports_dynamo_onnx_and_exported_program`;
- `test_export_pytorch_package_supports_dynamic_gather_nd_params_for_torch_export`;
- `test_export_artifacts_handle_non_max_suppression_v4`;
- `test_export_artifacts_handle_shape_derived_non_max_suppression_v4`;
- `test_convert_flatbuffer_direct_birdnet_preserves_permute_optimizations_and_pytorch_parity_when_model_is_available`.

The first six spend their time in ExportedProgram archive cleanup. That cleanup
function differs from `fb-refactor3` only by its local `import torch`; its pass
algorithm is otherwise identical. The current long-running discovery runs
used CPU continuously and had no observed SWAP. These results establish that
the long-running behavior is inherited rather than introduced by
`fb-refactor4`.

## Remaining known issues

- No unresolved non-timeout regression specific to `fb-refactor4` is confirmed
  in the 1,120-test exporter collection.
- The 112 reproduced failures are the exact pre-follow-up snapshot. A focused
  subset is now fixed, but the complete remaining count was intentionally not
  recomputed; the historical failures are not evidence that the current
  exporter is fully correct.
- Six `bread` model tests remain outside comparison by explicit user exclusion.
- The seven inherited long-running cases need bounded performance tests or
  independent optimization before routine full-suite execution.

The recommended fast regression gate is the modular 359-test selection, the
132 architecture/bulk-runner tests, focused codegen representatives for any
touched binding, and one lightweight real model only when artifact integration
needs verification.
