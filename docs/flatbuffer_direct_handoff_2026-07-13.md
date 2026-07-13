# flatbuffer_direct refactor handoff — 2026-07-13

## Pause checkpoint

- Branch: `fb-refactor3`
- Latest implementation checkpoint: `42bb3e8` (`extract multi branch gate layout pass`)
- Remote: `origin/fb-refactor3` contains `42bb3e8`
- Pull request: none; do not create one on resume
- The final handoff commit contains documentation only. After it is pushed,
  the expected working-tree state is clean and local/remote divergence is
  `0 0`.

## Completed work

This resumed interval completed five adjacent semantic layout families using
the staged characterization → mechanical extraction → indexed runner process,
and stopped after mechanically extracting the next family at a clean boundary.

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
     Indexed mutation and an ordered runner are intentionally deferred to the
     next checkpoint; the code is not left partially migrated.

Compatibility wrappers remain in `lower_from_onnx2tf.py` for all extracted
families. Every implementation migrated through the indexed-runner stage
contains no whole-graph producer/consumer map construction and no direct
operator-list insertion/deletion. The newly extracted multi-branch matcher
still preserves its legacy internals until the next indexed checkpoint. No
dependency or TensorFlow import path was added.

The 2,000 threshold is only the Tier 5 ONNX node-count boundary. It is not a
source-file line limit and no source-line gate should be introduced.

## Unfinished work

The overall Goal is not complete. In particular:

- Continue staged extraction/indexing of the remaining legacy layout rules.
  The immediate next unit is the already extracted generic multi-branch gate
  matcher in `passes/multi_branch_gate_layout.py`: migrate it to shared
  `ModelIRGraphIndex`/`LayoutState`, add a stable ordered runner, and replace
  its single raw production call. After that, audit the adjacent
  `_optimize_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains` family.
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
working tree is clean and `HEAD` equals `origin/fb-refactor3` at `42bb3e8`
(local/remote divergence `0 0`). The implementation checkpoints since the
previous pause changed:

- `docs/flatbuffer_direct_architecture.md`
- `docs/flatbuffer_direct_handoff_2026-07-12.md`
- `docs/flatbuffer_direct_handoff_2026-07-13.md`
- `onnx2tf/tflite_builder/core/model_ir_utils.py`
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`
- `onnx2tf/tflite_builder/passes/elementwise_gate_layout.py`
- `onnx2tf/tflite_builder/passes/multi_branch_gate_layout.py`
- `onnx2tf/tflite_builder/passes/se_layout.py`
- `tests/test_flatbuffer_direct_architecture.py`
- `tests/test_flatbuffer_direct_elementwise_gate_layout.py`
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
- The generic multi-branch gate matcher is mechanically extracted but not yet
  indexed. This is planned work, not a failing or partially applied change;
  its legacy production call remains intact and the full direct gate passes.

## First work on resume

1. Verify `git status --short --branch`, local/remote divergence, and the two
   latest commits; do not create a pull request.
2. Audit every read and mutation in
   `passes/multi_branch_gate_layout.py`, then add optional `graph_index` and
   `layout_state` parameters without changing matching semantics.
3. Add `run_multi_branch_gate_layout_cleanup` with a stable `LAYOUT_PLAN` pass
   ID and an indexed cheap guard. Replace the single production raw call while
   retaining the lowerer compatibility wrapper.
4. Extend the compact success/rejection tests to prove one initial index
   refresh, one success snapshot, and zero rejection snapshots; update ordered
   runner and efficiency expectations.
5. Run focused tests, sequential Tier 1 `superpoint.onnx -cotof`, and the full
   direct selection. Commit and push this indexed migration as one coherent
   checkpoint.

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
