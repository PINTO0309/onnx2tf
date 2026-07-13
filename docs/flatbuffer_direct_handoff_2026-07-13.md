# flatbuffer_direct refactor handoff — 2026-07-13

## Pause checkpoint

- Branch: `fb-refactor3`
- Latest implementation checkpoint: `73f96ca` (`extract add concat suffix layout pass`)
- Remote: after this handoff is pushed, `origin/fb-refactor3` contains
  `73f96ca` and the documentation checkpoint
- Pull request: none; do not create one on resume
- The final handoff commit contains documentation only. After it is pushed,
  the expected working-tree state is clean and local/remote divergence is
  `0 0`.

## Completed work

This resumed interval completed eleven adjacent semantic layout families using
the staged characterization → mechanical extraction → indexed runner process,
then characterized the next Add/Concat/constant-suffix family.

1. Mean layout
   - Characterized the long Mean/Mul/Reshape/Add/Conv success path and Mean
     fan-out rejection in `c99418a`.
   - Moved both Mean layout matchers to `passes/mean_layout.py` in `efb15cd`.
   - Added differential graph-index/layout-state mutation and stable ordered
     runners in `06a9dbd`.
2. LayerNorm statistics layout
   - Characterized pre-Transpose removal, existing post-Transpose reuse, and
     centered-value fan-out rejection in `d7866d2`.
   - Moved both matchers to `passes/layernorm_layout.py` in `267126a`.
   - Replaced both adjacent raw call pairs with one shared-state two-spec
     runner in `bffde62`.
3. Terminal unary/Mean layout
   - Characterized shared-pre retention, unary fan-out rejection, and
     inverse-Transpose-tail deferral in `8bce913`.
   - Moved the complete matcher to `passes/terminal_mean_layout.py` with an
     identical AST in `92446d7`.
   - Added stable runner `layout.terminal_unary_mean_reshape`, differential
     indexing, indexed preconditions, layout synchronization, and all six
     production runner calls in `a872774`.
4. EfficientNet-style SE layout
   - Characterized shared-input, gate-fan-out, and public-boundary behavior in
     `5f5de07`, then moved both complete matchers to `passes/se_layout.py` in
     `ce519b6`.
   - Indexed SE-Conv and SE-FC independently in `a9c6971` and `817bfa0`, using
     shared graph/layout state and stable ordered runners at all production
     call sites.
5. Elementwise gate layout
   - Added missing SUM/Logistic/Sub/Mul/Add boundary characterization in
     `1623762` and mechanically extracted four rules to
     `passes/elementwise_gate_layout.py` in `2095a01`.
   - Indexed the four-rule group under stable ordered pass IDs, replacing five
     repeated raw call groups in `9832355`.
6. Generic multi-branch gate layout
   - Replaced model-specific coverage with a compact generic two-branch
     success/rejection corpus in `13ec048`.
   - Mechanically extracted the complete matcher to
     `passes/multi_branch_gate_layout.py` with AST equivalence in `42bb3e8`.
   - Migrated all reads and mutations to shared graph/layout state, added the
     ordered runner, and replaced the single production call in `b0d1248`.
7. Complementary dual-postconv gate layout
   - Added a generic two-output success fixture and gate-fan-out,
     data-adapter-fan-out, and public-intermediate no-op boundaries in
     `ed6d8c1`.
   - Mechanically moved the complete implementation to
     `passes/dual_postconv_gate_layout.py` with AST equivalence in `8d149cb`.
   - Migrated the matcher and all five production positions to shared indexed
     state and a stable ordered runner in `cc828c8`.
8. Complementary postadd gate layout
   - Added generic Add/Conv-tail success and gate-fan-out,
     data-adapter-fan-out, and public-intermediate no-op characterization in
     `ea78747`.
   - Mechanically moved the complete matcher beside its complementary-gate
     sibling with AST equivalence in `f961413`.
   - Integrated it as the second ordered spec over a shared complementary-gate
     prefix and removed all five raw calls in `78b0742`.
9. Rank-five Leaky/Logistic gate layout
   - Replaced the 177-line central inline fixture with a dedicated compact
     success graph and five unsafe-boundary cases in `ee3d2fd`.
   - Mechanically moved the complete matcher to
     `passes/ndhwc_gate_layout.py` with AST equivalence in `332612f`.
   - Migrated all mutation and six production calls to a stable indexed runner
     in `2871ade`.
10. Conv3D/Leaky/Unsqueeze gate layout
    - Replaced a 176-line central fixture with compact 4D/5D success variants
      and six unsafe-boundary cases in `49c72b9`.
    - Mechanically moved the complete matcher beside its NDHWC sibling with AST
      equivalence in `ae3c00b`.
    - Migrated producer/consumer reads, rewrites, structural removals, pruning,
      and layout synchronization to shared indexed state in `a470cce`.
    - Registered stable ordered pass ID
      `layout.ndhwc_conv3d_leaky_unsqueeze_gate` after the rank-five gate,
      removed all six raw production calls, and retained the compatibility
      wrapper.
11. Cost-volume/ScatterND layout
    - Replaced the 177-line central success fixture with a dedicated compact
      corpus in `4b6f297`.
    - Added six whole-ModelIR no-op boundaries covering leading-adapter
      fan-out, pre/post sides of the trailing adapter, a public intermediate,
      invalid leading permutation, and an invalid downstream operator.
    - Preserved production behavior at the characterization checkpoint.
    - Moved the complete matcher to
      `passes/cost_volume_scatter_layout.py` with AST equivalence in `d62e77d`;
      the compatibility wrapper and all six raw call positions remain.
    - Added pure indexed topology and constant planning, transactional runner
      `layout.cost_volume_scatter_ndhwc`, shared graph/layout mutation, and six
      production runner calls in `56516ef`.
    - Fixed late-validation partial mutation for invalid ScatterND shape,
      coordinate rank, and out-of-bounds coordinates; all now reject before a
      snapshot and preserve the complete ModelIR.
12. Add/Concat/constant-suffix layout
    - Added the first dedicated success corpus for the previously untested
      central matcher in `fcf24b2`.
    - Fixed nine complete no-op boundaries covering branch/Add/Concat/Mul
      fan-out, public intermediate/post output, invalid permutation/axis, and
      missing suffix constant.
    - Preserved production behavior and all five raw call positions;
      mechanical extraction remains the next checkpoint.
    - Moved the complete matcher to `passes/add_concat_suffix_layout.py` with
      AST equivalence in `73f96ca`; the compatibility wrapper and all five raw
      production positions remain.

Compatibility wrappers remain in `lower_from_onnx2tf.py` for all extracted
families. Every implementation migrated through the indexed-runner stage
contains no whole-graph producer/consumer map construction and no direct
operator-list insertion/deletion. No dependency or TensorFlow import path was
added.

The 2,000 threshold is only the Tier 5 ONNX node-count boundary. It is not a
source-file line limit and no source-line gate should be introduced.

## Unfinished work

The overall Goal is not complete. In particular:

- Continue staged extraction/indexing of the remaining legacy layout rules.
  The immediate next unit is the extracted
  `_optimize_transpose_add_concat_const_suffix_nhwc_chains`: move it
  to shared graph/layout state, protect shared suffix constants, and replace
  all five raw production positions with a stable transactional runner.
- Complete the remaining central lowerer/registry decomposition and consolidate
  op-family validation, capability selection, and lowering.
- Reconnect and exhaustively validate quantization, split/crop, custom/pseudo
  ops, weights, and requested-artifact-only execution on the fixed ModelIR
  contract.
- Complete the planned PyTorch, TorchScript, Dynamo ONNX, and ExportedProgram
  exporter decomposition.
- Run the fixed corpus gates sequentially through Tier 0–Tier 5. Tier 5 means
  models with at least 2,000 ONNX nodes and remains a late-stage gate.
- Complete the artifact matrix, optional TensorFlow exporter compatibility,
  full public CLI/Python contract audit, and final baseline/failure-signature
  comparison.
- Measure three-run median conversion time and peak RSS by tier, enforce the
  +10% non-regression limit, and evaluate the Tier 4 15% speedup target.
- Produce the final requirement-by-requirement completion audit and developer
  documentation. Do not mark the Goal complete until every original plan item
  has direct evidence.

## Branch and changed files

Current branch is `fb-refactor3`. Before this handoff-document update, the
implementation working tree is clean at `73f96ca`; after the documentation
commit is pushed, local/remote divergence must be `0 0`. The implementation
checkpoints since the previous pause changed:

- `docs/flatbuffer_direct_architecture.md`
- `docs/flatbuffer_direct_handoff_2026-07-12.md`
- `docs/flatbuffer_direct_handoff_2026-07-13.md`
- `onnx2tf/tflite_builder/core/model_ir_utils.py`
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`
- `onnx2tf/tflite_builder/passes/elementwise_gate_layout.py`
- `onnx2tf/tflite_builder/passes/dual_postconv_gate_layout.py`
- `onnx2tf/tflite_builder/passes/multi_branch_gate_layout.py`
- `onnx2tf/tflite_builder/passes/ndhwc_gate_layout.py`
- `onnx2tf/tflite_builder/passes/se_layout.py`
- `tests/test_flatbuffer_direct_architecture.py`
- `tests/test_flatbuffer_direct_elementwise_gate_layout.py`
- `tests/test_flatbuffer_direct_dual_postconv_gate_layout.py`
- `tests/test_flatbuffer_direct_3d_gate_layout.py`
- `tests/test_flatbuffer_direct_osnet_gate_layout.py`
- `tests/test_flatbuffer_direct_pass_efficiency.py`
- `tests/test_flatbuffer_direct_se_layout.py`
- `tests/test_tflite_builder_direct.py`

The final handoff checkpoint updates documentation only. No implementation
file remains uncommitted after it.

## Tests executed

All commands ran in the existing `uv` environment. Inference was strictly
sequential with only one model/process active at a time.

- Terminal Mean characterization plus legacy tests: `5 passed`.
- Terminal Mean extraction/ownership focus: `6 passed`.
- Terminal Mean indexed runner, architecture, and efficiency focus:
  `40 passed`.
- Full direct selection after mechanical extraction:
  `1167 passed, 5 deselected, 2 warnings in 151.84s`.
- Tier 1 `superpoint.onnx`, sequential
  `-tb flatbuffer_direct -cotof` after indexed migration:
  `evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, cosine similarity `1.0`.
- Full direct selection after indexed migration:
  `1170 passed, 5 deselected, 2 warnings in 162.39s`.
- SE characterization/extraction/indexed focused gates passed; the final SE
  full direct selection was
  `1178 passed, 5 deselected, 2 warnings in 158.98s`.
- Elementwise-gate characterization/extraction/indexed focused gates passed;
  the final elementwise full direct selection was
  `1183 passed, 5 deselected, 2 warnings in 152.05s`.
- Multi-branch gate characterization and extraction focused gate passed
  `3 tests`; its full direct selection was
  `1186 passed, 5 deselected, 2 warnings in 151.88s`.
- Pause verification of multi-branch characterization plus architecture:
  `35 passed in 17.23s`.
- Indexed multi-branch runner, architecture, and efficiency focus:
  `41 passed in 18.59s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed multi-branch migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed multi-branch migration:
  `1188 passed, 5 deselected, 2 warnings in 161.24s`.
- Dual-postconv gate compact characterization: `4 passed`.
- Dual-postconv extraction and ownership focus: `38 passed in 18.99s`.
- Full direct selection after mechanical extraction:
  `1193 passed, 5 deselected, 2 warnings in 169.93s`.
- Indexed dual-postconv runner, architecture, and efficiency focus:
  `46 passed in 19.29s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed dual-postconv migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed dual-postconv migration:
  `1197 passed, 5 deselected, 2 warnings in 161.48s`.
- Postadd complementary-gate compact characterization: `4 passed`.
- Postadd extraction, sibling, and ownership focus:
  `46 passed in 18.04s`.
- Full direct selection after mechanical extraction:
  `1201 passed, 5 deselected, 2 warnings in 157.71s`.
- Indexed postadd/complementary-gate runner, architecture, and efficiency
  focus: `54 passed in 17.73s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed postadd migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed postadd migration:
  `1205 passed, 5 deselected, 2 warnings in 154.34s`.
- Dedicated rank-five 3D gate characterization: `6 passed`.
- 3D extraction and ownership focus: `41 passed in 17.69s`.
- Full direct selection after 3D mechanical extraction:
  `1211 passed, 5 deselected, 2 warnings in 157.23s`.
- Indexed 3D runner, architecture, and efficiency focus:
  `51 passed in 18.61s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed 3D migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed 3D migration:
  `1217 passed, 5 deselected, 2 warnings in 156.83s`.
- Dedicated Conv3D gate characterization: `8 passed`.
- Conv3D extraction, sibling, and ownership focus:
  `55 passed in 17.93s`.
- Full direct selection after Conv3D mechanical extraction:
  `1224 passed, 5 deselected, 2 warnings in 157.08s`.
- Indexed Conv3D runner, rank-five sibling, architecture, and efficiency
  focus: `67 passed in 17.27s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed Conv3D migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed Conv3D migration:
  `1232 passed, 5 deselected, 2 warnings in 154.69s`.
- Dedicated cost-volume/ScatterND characterization: `7 passed in 0.29s`.
- Cost-volume/ScatterND focused selection with the central module present:
  `7 passed, 758 deselected in 2.72s`.
- Full direct selection after moving the fixture and adding six boundaries:
  `1238 passed, 5 deselected, 2 warnings in 153.83s`.
- Cost-volume/ScatterND extraction, characterization, and ownership focus:
  `43 passed in 18.64s`; the extracted function AST exactly matched
  `4b6f297`.
- Full direct selection after mechanical extraction:
  `1239 passed, 5 deselected, 2 warnings in 156.55s`.
- Indexed cost-volume/ScatterND runner, late-validation, architecture, and
  efficiency focus: `60 passed in 18.58s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed cost-volume/ScatterND migration:
  `1252 passed, 5 deselected, 2 warnings in 164.88s`.
- Dedicated Add/Concat/constant-suffix characterization: `10 passed in 0.32s`.
- Full direct selection after adding the previously missing success and nine
  unsafe-boundary cases:
  `1262 passed, 5 deselected, 2 warnings in 195.24s`.
- Add/Concat/constant-suffix extraction, characterization, and ownership focus:
  `47 passed in 18.70s`; the extracted function AST exactly matched
  `fcf24b2`.
- Full direct selection after mechanical extraction:
  `1263 passed, 5 deselected, 2 warnings in 158.07s`.
- Tier 1 `superpoint.onnx` was run sequentially after both indexed SE units and
  indexed elementwise gates. Every run retained `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The full direct command intentionally deselected the same five
optional/environment-sensitive cases used by the established gate:

- `test_tflite_backend_matrix_add`
- `test_tflite_backend_matrix_hardswish_rewrite_on_off`
- `test_tf_converter_resize_cubic_avoids_flex_resize_bicubic`
- `test_tf_converter_resize_cubic_honors_cubic_coeff_a`
- `test_flatbuffer_direct_group_norm_alias_builtin_conversion`

## Failing tests and known issues

- No newly failing test is known at this checkpoint.
- The two full-suite warnings are the established FLOAT16 overflow warnings in
  `ir.py` from ArgMax/ReduceMax and negative-infinity Where coverage.
- The five tests listed above remain explicitly outside the current core gate;
  they were not silently treated as passes.
- A full Tier 0–Tier 5 corpus run has not been performed after this checkpoint,
  so corpus-wide non-regression is not yet proven.
- Performance/RSS targets are not yet proven for the current architecture.
- Baseline-invalid `vit_h_encoder.onnx` remains classified as `invalid_onnx`.
- Per user direction, DEIM is considered a successful conversion family.

## First work on resume

1. Verify `git status --short --branch`, local/remote divergence, and the two
   latest commits; do not create a pull request.
2. Add shared-constant fan-out characterization for both suffix constants and
   define copy-on-write behavior so unrelated consumers remain NCHW.
3. Migrate topology reads, rewrites, aliasing, structural removals, pruning,
   and layout reconciliation to one shared index/state and stable runner.
4. Replace all five raw calls, then run focused, sequential Tier 1, and full
   direct gates before committing and pushing the indexed checkpoint.

Resume constraints remain: commit and push at coherent checkpoints only; no
pull request; no new dependency; default direct TFLite and `-cotof` must remain
TensorFlow-free; use `uv`; and run inference validation sequentially with one
process.

## Resumed SE layout checkpoint

Checkpoint `5f5de07` added a dedicated compact SE corpus without duplicating
the large legacy fixtures. It fixes three important boundaries: an SE-Conv
gate with an additional consumer rejects unchanged, a public SE-FC gate
rejects unchanged, and an SE-FC target branch sharing its leading NCHW adapter
rewrites while retaining that adapter for the side branch. Together with the
six existing success variants, focused characterization passed 9 tests.

The complete `_optimize_transpose_se_conv_mul_prepost_nhwc_chains` and
`_optimize_transpose_se_fc_mul_prepost_nhwc_chains` implementations then moved
mechanically to `passes/se_layout.py`. Their ASTs, including docstrings, match
`5f5de07`. The lowerer retains signature-compatible wrappers; all six SE-Conv
and nine SE-FC production positions remain unchanged until the separate
indexed migration checkpoint. An architecture test fixes the single-owner
boundary.

Focused success, rejection, and ownership validation passed 10 tests. The
complete sequential direct selection passed:

```text
1174 passed, 5 deselected, 2 warnings in 149.62s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Next work is the GraphIndex/ordered-runner migration of this
extracted family.

### Indexed SE-Conv checkpoint

The extracted SE-Conv implementation now accepts the shared
`ModelIRGraphIndex` and `LayoutState`. Consumer/producer reads, Swish and gate
input rewrites, Mean/post adapter alias rewrites, canonical output rewrite,
structural removals, pruning, metadata updates, and layout reconciliation use
differential state. Its implementation contains no whole-graph map builder or
direct operator-list deletion.

All six production positions call `run_se_conv_layout_cleanup`, with stable
`LAYOUT_PLAN` ID `layout.se_conv_gate_nhwc`. A cheap model-only capability scan
precedes an indexed common-region guard covering the leading
Transpose/Logistic/Mul, Mean branch and accepted adapter class, exclusive
second gate, and terminal inverse-Transpose fan-out. The existing deep matcher
continues to validate Logistic, affine, and Squeeze/Reshape gate details.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 40 tests. The compact positive fixture uses one initial
index refresh and one snapshot; gate fan-out rejects before snapshotting. Tier
1 `superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1176 passed, 5 deselected, 2 warnings in 150.22s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_se_conv_superpoint` artifacts were removed after metrics
inspection. SE-FC remains the next separate indexed migration unit.

### Indexed SE-FC checkpoint

The 977-line SE-FC implementation now accepts the shared
`ModelIRGraphIndex` and `LayoutState`. All normal and alternate-path
consumer/producer reads, cloned Mean-axis input replacement, pool/Mul/Conv/gate
rewrites, canonical output and aliases, structural removals, pruning, metadata,
and layout reconciliation use differential state. With SE-Conv already
indexed, `passes/se_layout.py` now contains no whole-graph map builder or
direct operator-list deletion.

`run_se_fc_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.se_fc_gate_nhwc`. Eight main-model positions receive the session layout
state and diagnostics. The ninth fallback-IR position receives diagnostics but
creates its own layout state because it operates on a distinct ModelIR.
Model-only Transpose/Mul/dense-or-Conv capability preflight skips irrelevant
graphs. The indexed guard proves public boundaries, a normal gate Reshape and
inverse output bridge, or the common ADD/MUL/inverse-bridge prefix of the
alternate float path; the existing matcher retains all deep topology checks.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 42 tests. A normal SE-FC rewrite uses one initial index
refresh and one snapshot; a public gate rejects before snapshotting. The
shared-pre runner fixture retains its leading Transpose for the side branch.
Tier 1 `superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1178 passed, 5 deselected, 2 warnings in 158.98s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_se_fc_superpoint` artifacts were removed after metrics
inspection.

### Elementwise gate characterization and mechanical extraction

Checkpoint `1623762` added the missing compact SUM/Logistic/Sub/Mul/Add
characterization. The successful graph proves three leading adapter removals,
SUM axis remapping from NCHW axis 1 to NHWC axis 3, canonical ADD output, and
downstream rewiring. A reduction-input side consumer proves a complete no-op.
Together with existing Logistic/Mul/Add, weighted Swish, nested weighted
Swish, and legacy-user fixtures, focused characterization passed 6 tests.

The four complete implementations moved mechanically to
`passes/elementwise_gate_layout.py`. Their ASTs, including docstrings, match
`1623762`; the lowerer keeps signature-compatible wrappers and all five raw
production positions per rule until the separate indexed migration. The
previous lowerer-local `_is_scalar_like_tensor` helper moved unchanged to
`core/model_ir_utils.py`, while remaining a compatibility import from the
lowerer. Ownership tests fix both boundaries.

Focused characterization, legacy, and ownership validation passed 7 tests.
The complete sequential direct selection passed:

```text
1181 passed, 5 deselected, 2 warnings in 166.23s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Next work is the indexed migration of these four extracted
rules.

### Indexed elementwise gate checkpoint

All four extracted matchers now accept one shared `ModelIRGraphIndex` and
active `LayoutState`. Consumer/producer traversal, SUM and branch input
rewrites, canonical output and aliases, conditional legacy-adapter rewrites,
structural removals, pruning, metadata, and layout reconciliation use
differential state. The module contains no whole-graph map builder or direct
operator-list deletion.

The five repeated raw call groups became five calls to
`run_elementwise_gate_layout_cleanup`. Each invocation owns four ordered
`LAYOUT_PLAN` specs with stable IDs `layout.sum_logistic_muladd_nhwc`,
`layout.weighted_add_swish_nhwc`,
`layout.nested_weighted_add_swish_nhwc`, and
`layout.logistic_muladd_nhwc`. Model-only common capability preflight and
indexed per-pattern guards preserve the legacy order while sharing one state.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 40 tests. The SUM success fixture produces four diagnostics,
one initial index refresh, and exactly one snapshot/change. Reduction fan-out
produces four skips and zero snapshots. Existing Logistic/MulAdd and both
weighted-Swish success fixtures now execute through the grouped runner. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1183 passed, 5 deselected, 2 warnings in 152.05s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_elementwise_gate_superpoint` artifacts were removed after
metrics inspection.

### Multi-branch gate characterization and mechanical extraction

Checkpoint `13ec048` replaced the missing model-specific coverage with a
compact generic two-branch graph. It proves branch adapter removal, independent
Mean-axis constant cloning/remapping, Logistic/Mul leaf propagation, Add-root
output canonicalization, and a complete gate-fan-out rejection. Focused
characterization passed 2 tests.

The complete 518-line matcher moved mechanically to
`passes/multi_branch_gate_layout.py`, with an AST including docstrings that
matches `13ec048`. Despite its historical OSNet name, the test and matcher are
defined only by generic topology. The lowerer keeps a signature-compatible
wrapper and the one production call remains unchanged until indexed migration.
An architecture test fixes ownership.

Focused characterization and ownership validation passed 3 tests. The
complete sequential direct selection passed:

```text
1186 passed, 5 deselected, 2 warnings in 151.88s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently.

### Indexed multi-branch gate checkpoint

Checkpoint `b0d1248` migrated the extracted matcher to shared
`ModelIRGraphIndex` and `LayoutState`. Producer/consumer traversal, cloned
Mean-axis input replacement, Relu and Logistic input rewrites, Add-root output
canonicalization, alias rewrites, structural removals, pruning, metadata, and
layout reconciliation now use differential state. The implementation contains
no whole-graph producer/consumer map builder and no direct operator-list
deletion.

`run_multi_branch_gate_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.multi_branch_gate_add_tree_nhwc`. Its model-only required-op preflight
avoids state construction for irrelevant graphs. Its indexed guard proves an
inverse output bridge, a nested Add tree with at least two Mul leaves, guarded
Relu branches with keep-dims Mean users, exclusive Logistic gates, and accepted
gate adapters before the complete matcher performs deeper validation. The
single production position supplies session layout state and diagnostics; the
lowerer compatibility wrapper remains available.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 41 tests. The success fixture uses one initial index refresh
and one snapshot; gate fan-out rejects before snapshotting. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1188 passed, 5 deselected, 2 warnings in 161.24s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_multi_branch_gate_superpoint` artifacts were removed after
metrics inspection.

### Dual-postconv gate characterization and mechanical extraction

Checkpoint `ed6d8c1` added the missing compact generic corpus for the
complementary Logistic/Sub gate feeding two Mul/Add outputs and two downstream
Conv branches. The positive fixture proves removal of all three leading and
both trailing layout adapters, direct NHWC inputs to the elementwise graph,
canonical Add outputs, and unchanged Conv inputs. Parameterized no-op coverage
fixes Logistic gate fan-out, a data-adapter side consumer, and a public Add
intermediate. Focused characterization passed 4 tests.

The complete 323-line matcher moved mechanically to
`passes/dual_postconv_gate_layout.py`. Its AST, including docstrings, matches
`ed6d8c1`; the lowerer retains a signature-compatible wrapper and all five raw
production positions until the separate indexed migration. An architecture
test fixes single ownership.

Focused characterization and ownership validation passed 38 tests. The
complete sequential direct selection passed:

```text
1193 passed, 5 deselected, 2 warnings in 169.93s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently.

### Indexed dual-postconv gate checkpoint

Checkpoint `cc828c8` migrated the extracted complementary-gate matcher to
shared `ModelIRGraphIndex` and `LayoutState`. Producer/consumer traversal,
Logistic/Mul/Add input rewrites, Add output canonicalization, post-output alias
rewrites, structural removals, pruning, metadata, and layout reconciliation
now use differential state. The implementation contains no whole-graph map
builder and no direct operator-list deletion.

`run_dual_postconv_gate_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.dual_postconv_complementary_gate_nhwc`. Its model-only required-op
preflight skips irrelevant graphs without state construction. Its indexed
guard proves the exclusive Logistic/Sub complementary gate, distinct data
adapters, two Mul/Add branches, inverse output adapters, public boundaries, and
allowed data fan-out before the complete matcher retains deeper checks. All
five production positions now supply session layout state and diagnostics; the
lowerer compatibility wrapper remains available.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 46 tests. The successful two-Conv fixture uses one initial
index refresh and one snapshot. Gate fan-out, data-adapter fan-out, and public
intermediate variants all reject before snapshotting. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1197 passed, 5 deselected, 2 warnings in 161.48s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_dual_postconv_superpoint` artifacts were removed after metrics
inspection.

### Postadd complementary-gate characterization and extraction

Checkpoint `ea78747` added a compact generic graph for the complementary
Logistic/Sub gate whose two Mul outputs cross inverse adapters before an NHWC
Add and downstream Conv. The success fixture proves removal of all five layout
adapters, direct NHWC elementwise inputs, canonical Mul outputs, and unchanged
downstream Add inputs. Parameterized no-op coverage fixes Logistic gate
fan-out, a data-adapter side consumer, and a public Mul intermediate. Focused
characterization passed 4 tests.

The complete 272-line matcher moved mechanically beside the dual-postconv
matcher in `passes/dual_postconv_gate_layout.py`. Its AST, including
docstrings, matches `ea78747`; the lowerer retains a signature-compatible
wrapper and all five raw production positions until the separate indexed
migration. The family ownership test covers both matchers.

Focused postadd, indexed sibling, and ownership validation passed 46 tests.
The complete sequential direct selection passed:

```text
1201 passed, 5 deselected, 2 warnings in 157.71s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently.

### Indexed postadd complementary-gate checkpoint

Checkpoint `78b0742` migrated the postadd matcher to shared
`ModelIRGraphIndex` and `LayoutState`. Producer/consumer traversal,
Logistic/Mul input rewrites, Mul output canonicalization, post-output alias
rewrites, structural removals, pruning, metadata, and layout reconciliation
now use differential state. Both complementary-gate matchers contain no
whole-graph map builder and no direct operator-list deletion.

The existing family runner now registers a second stable `LAYOUT_PLAN` ID,
`layout.postadd_complementary_gate_nhwc`, after
`layout.dual_postconv_complementary_gate_nhwc`. Both indexed guards reuse one
resolver for the three input adapters, Logistic/Sub gate, and two Mul branches;
only their Add-before-post versus post-before-Add output contracts remain
separate. Each of the five production groups invokes the runner once and shares
one graph/layout state while preserving the legacy rule order.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 54 tests. The postadd success fixture creates one initial
index and one snapshot across both ordered specs. Gate fan-out, data-adapter
fan-out, and public-intermediate variants all reject with zero snapshots. Tier
1 `superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1205 passed, 5 deselected, 2 warnings in 154.34s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_postadd_superpoint` artifacts were removed after metrics
inspection.

### Rank-five 3D gate characterization and extraction

Checkpoint `ee3d2fd` replaced the 177-line success fixture embedded in the
central direct test module with a dedicated compact rank-five corpus. The
positive graph proves base Reshape remapping, skip LeakyRelu and gate Logistic
adapter removal, both Add output canonicalizations, and all five Transpose
removals. Parameterized no-op coverage fixes shared-base fan-out, gate fan-out,
a public Add intermediate, an invalid NDHWC-to-NCDHW permutation, and an
invalid Reshape-constant rank. Focused characterization passed 6 tests while
reducing the central test module by 177 lines.

The complete 378-line matcher moved mechanically to
`passes/ndhwc_gate_layout.py`. Its AST, including docstrings, matches
`ee3d2fd`; the lowerer retains a signature-compatible wrapper and all six raw
production positions until the separate indexed migration. An architecture
test fixes single ownership.

Focused characterization and ownership validation passed 41 tests. The
complete sequential direct selection passed:

```text
1211 passed, 5 deselected, 2 warnings in 157.23s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently.

### Indexed rank-five 3D gate checkpoint

Checkpoint `2871ade` migrated the extracted rank-five matcher to shared
`ModelIRGraphIndex` and `LayoutState`. Producer/consumer traversal, base
Reshape, skip LeakyRelu, and gate Logistic input rewrites, both Add output
canonicalizations, constant-shape remapping, structural removals, pruning,
metadata, and layout reconciliation now use differential state. Two duplicate
legacy permutation assignments were also removed. The implementation contains
no whole-graph map builder and no direct operator-list deletion.

`run_ndhwc_gate_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.ndhwc_leaky_logistic_gate`. Its model-only required-op preflight skips
irrelevant graphs without state construction. Its indexed guard proves both
inverse Add outputs, the shared base and skip branches, exclusive gate and
Mul, exact rank-four/rank-five permutations, public boundaries, and the
rank-five Reshape constant before the complete matcher performs the rewrite.
All six production positions now supply session layout state and diagnostics;
the lowerer compatibility wrapper remains available.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 51 tests. The success fixture uses one initial index refresh
and one snapshot. Shared-base fan-out, gate fan-out, public intermediate,
invalid permutation, and invalid reshape-rank variants all reject before
snapshotting. Tier 1 `superpoint.onnx` passed sequential
`-tb flatbuffer_direct -cotof` with `evaluation_pass=true`,
`max_abs=1.6666017472743988e-06`, `rmse=1.6207873294228388e-07`, and cosine
similarity `1.0`.

The complete sequential direct selection passed:

```text
1217 passed, 5 deselected, 2 warnings in 156.83s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_ndhwc_gate_superpoint` artifacts were removed after metrics
inspection.

### Conv3D gate characterization and extraction

Checkpoint `49c72b9` replaced the 176-line central inline fixture with a
dedicated compact Conv3D/LeakyRelu/Reshape gate corpus. Separate positive cases
fix both accepted semantic adapter ranks: rank-four NHWC-to-NCHW and rank-five
NDHWC-to-NCDHW. They prove semantic Reshape remapping, Conv-side LeakyRelu
adapter removal, gated Mul output canonicalization, and unchanged downstream
Conv3D input. Six no-op boundaries cover Conv-adapter fan-out, LeakyRelu
fan-out, gate-Reshape fan-out, a public Mul intermediate, invalid permutation,
and invalid reshape rank. Focused characterization passed 8 tests.

The complete 226-line matcher moved mechanically beside the rank-five sibling
in `passes/ndhwc_gate_layout.py`. Its AST, including docstrings, matches
`49c72b9`; the lowerer retains a signature-compatible wrapper and all six raw
production positions until the separate indexed migration. Family ownership
coverage includes both matchers.

Focused Conv3D, indexed sibling, and ownership validation passed 55 tests. The
complete sequential direct selection passed:

```text
1224 passed, 5 deselected, 2 warnings in 157.08s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently.

### Indexed Conv3D gate checkpoint

Checkpoint `a470cce` migrated the extracted Conv3D gate matcher to shared
`ModelIRGraphIndex` and `LayoutState`. Producer/consumer traversal, semantic
Reshape and Conv-side LeakyRelu rewrites, gated Mul output canonicalization,
constant-shape remapping, structural removals, pruning, metadata, and layout
reconciliation now use differential state. Both matchers in
`passes/ndhwc_gate_layout.py` contain no whole-graph map builder and no direct
operator-list deletion.

`run_ndhwc_gate_layout_cleanup` now registers a second stable `LAYOUT_PLAN` ID,
`layout.ndhwc_conv3d_leaky_unsqueeze_gate`, after
`layout.ndhwc_leaky_logistic_gate`. Its indexed guard proves the inverse Mul
output adapter, exclusive LeakyRelu and Reshape branches, accepted rank-four or
rank-five semantic adapter, rank-five Conv adapter, public boundaries, and
rank-five remappable Reshape constant before snapshotting. All six production
groups invoke the shared runner once; the legacy raw calls were removed while
the compatibility wrapper remains available.

Focused runner, sibling, ownership, architecture, and irrelevant-graph
efficiency validation passed 67 tests. Both accepted semantic-rank fixtures
use one initial index refresh and one snapshot. Conv-adapter, LeakyRelu, and
Reshape fan-out, public intermediate, invalid permutation, and invalid
reshape-rank variants all reject before snapshotting. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1232 passed, 5 deselected, 2 warnings in 154.69s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_conv3d_gate_superpoint` artifacts were removed after metrics
inspection.

### Cost-volume/ScatterND characterization checkpoint

Checkpoint `4b6f297` moved the embedded cost-volume success fixture out of the
central direct test module into
`tests/test_flatbuffer_direct_cost_volume_scatter_layout.py`. The compact graph
retains both descriptor adapters, shared Slice constants, Mean-axis mapping,
Reshape, casted five-coordinate ScatterND indices, ScatterND shape mapping,
the inverse rank-five adapter, and the downstream Conv3D contract. It proves
the same constant values, tensor metadata, operator removal, and Conv3D input
as the former fixture while reducing the central module by 177 lines.

Six parameterized boundaries prove a complete ModelIR no-op for a leading
adapter side consumer, ScatterND-result side consumer, post-adapter side
consumer, public ScatterND intermediate, invalid leading permutation, and
non-Conv3D downstream consumer. The snapshots compare operator topology and
options plus every tensor dtype, shape, shape signature, and constant value.
Production code was intentionally unchanged at this checkpoint.

Focused characterization passed 7 tests. The complete sequential direct
selection passed:

```text
1238 passed, 5 deselected, 2 warnings in 153.83s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. The next unit is a strictly mechanical matcher extraction with
AST-equivalence and single-owner gates.

### Cost-volume/ScatterND mechanical extraction checkpoint

Checkpoint `d62e77d` moved the complete 536-line matcher mechanically to
`passes/cost_volume_scatter_layout.py`. Its function AST, including the
docstring and nested helpers, exactly matches checkpoint `4b6f297`. The central
lowerer now keeps only a signature-compatible wrapper. All six raw production
call positions remain unchanged so extraction does not alter rule ordering or
retry behavior.

The architecture gate fixes the focused module as the single implementation
owner, the lowerer alias and compatibility wrapper, and exactly six production
calls. Focused characterization and ownership validation passed 43 tests. The
complete sequential direct selection passed:

```text
1239 passed, 5 deselected, 2 warnings in 156.55s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Indexed planning, transactional late-validation safety, shared
graph/layout state, and runner integration remain a separate next checkpoint.

### Indexed cost-volume/ScatterND checkpoint

Checkpoint `56516ef` replaced repeated producer/consumer map construction with
one shared `ModelIRGraphIndex` and introduced a pure candidate plan that proves
the complete upstream island and every mutable constant before rewriting.
Slice ranges, reduction axes, Concat axes, ScatterND output shape, casted index
coordinates, coordinate rank, and bounds are validated without modifying the
model. This closes a legacy failure mode where an invalid late ScatterND
constant could leave earlier Slice or Mean constants partially remapped even
though the matcher reported no rewrite.

Input and alias rewrites now update the differential index, structural removal
uses indexed operators, and pruning/metadata reconciliation synchronize the
shared `LayoutState`. `run_cost_volume_scatter_layout_cleanup` registers stable
`LAYOUT_PLAN` ID `layout.cost_volume_scatter_ndhwc`; all six production
positions call it with session state and diagnostics. The lowerer compatibility
wrapper remains available.

Focused success, nine complete no-op boundaries, runner instrumentation,
ownership, architecture, and irrelevant-graph efficiency validation passed 60
tests. The success graph uses one initial index refresh and one transactional
snapshot. All nine rejection cases, including invalid ScatterND shape,
coordinate rank, and out-of-bounds coordinates, reject before snapshotting.
Tier 1 `superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1252 passed, 5 deselected, 2 warnings in 164.88s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_cost_volume_superpoint` artifacts were removed after metrics
inspection.

### Add/Concat/constant-suffix characterization checkpoint

Checkpoint `fcf24b2` added the first dedicated coverage for
`_optimize_transpose_add_concat_const_suffix_nhwc_chains`. The compact success
graph includes two independent branch adapters, one shared base adapter, both
Add fan-ins, channel Concat, strict MUL(const) then ADD(const) suffix, inverse
output adapter, and a downstream Conv consumer. It proves all four adapters
are removed, both Add inputs become NHWC, Concat moves to axis 3, both rank-four
constants are transposed to NHWC, metadata follows the rewrite, and the suffix
Add directly owns the canonical post-adapter tensor name.

Nine parameterized boundaries prove a complete ModelIR no-op for branch
adapter fan-out, Add output fan-out, Concat fan-out, Mul output fan-out, public
suffix intermediate, public post output, invalid leading permutation, invalid
Concat axis, and missing suffix constant. Snapshots compare operator topology
and options plus every tensor dtype, shape, shape signature, and constant
value. Production code and all five raw call positions were unchanged.

Focused characterization passed 10 tests. The complete sequential direct
selection passed:

```text
1262 passed, 5 deselected, 2 warnings in 195.24s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Mechanical extraction with AST-equivalence and ownership gates
is the next separate unit.

### Add/Concat/constant-suffix mechanical extraction checkpoint

Checkpoint `73f96ca` moved the complete 271-line matcher mechanically to
`passes/add_concat_suffix_layout.py`. Its function AST, including docstring,
exactly matches checkpoint `fcf24b2`. The central lowerer now retains only a
signature-compatible wrapper, while all five raw production positions remain
unchanged so rule order and retry behavior are identical.

The architecture gate fixes the focused module as the single implementation
owner, the lowerer import alias and wrapper, and exactly five production calls.
Focused characterization and ownership validation passed 47 tests. The
complete sequential direct selection passed:

```text
1263 passed, 5 deselected, 2 warnings in 158.07s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Shared-constant copy-on-write, indexed mutation, transactional
runner integration, and raw-call replacement remain the next checkpoint.
