# `fb-refactor7` improvement summary

## Purpose and compatibility boundary

`fb-refactor7` improves the internal efficiency, ownership, and observability
of the TensorFlow-free `flatbuffer_direct` pipeline. It deliberately preserves
the public CLI and Python API, default backend, output names, ModelIR and
artifact semantics, pass order, optional TensorFlow boundary, dependency set,
and sequential-inference policy.

The branch follows a characterize-first workflow. Every production change is
preceded by a contract that freezes the existing behavior, including
zero-mutation and cleanup-only paths. The implementation is then admitted only
after the focused contract and a broader sequential regression gate pass.

## Diagnostic bookkeeping efficiency

- ModelIR pass diagnostic numbering no longer repeatedly scans the complete
  and growing diagnostic history for every emitted event.
- A single scan initializes event, group, and per-pass invocation counters for
  arbitrary caller-owned diagnostic lists.
- `ConversionSession`-owned diagnostic ledgers retain this numbering state
  across pass groups, avoiding one full history scan per group while preserving
  the exact external diagnostic schema and numbering rules.
- Ordinary lists remain supported through the safe one-scan fallback, so this
  optimization does not leak a new internal type through the public boundary.

## Stable-reconciliation and convergence fast paths

Several late and terminal layout paths previously reran broad shape/layout
reconciliation even when the immediately preceding owner reported no graph
mutation. The branch now uses explicit mutation results to skip those stable
rescans. Covered paths include:

- mixed singleton and final Reshape reconciliation;
- final PReLU and SINet reconciliation;
- terminal SE/Gather reconciliation;
- shared late reconciliation and late binary repair;
- binary-layout convergence rounds;
- final shape reconciliation and both final convergence scans;
- post-fusion and placeholder-MatMul repair follow-up scans.

The fast paths are conservative. Cleanup-only owners include clamped net
tensor-count deltas, and reconciliation is skipped only when all declared
mutation evidence is zero. Pass order and mutation-positive behavior remain
unchanged.

## Explicit mutation evidence and result propagation

Late and terminal orchestration now returns or stages ordered, fixed-schema
mutation evidence instead of silently discarding child results. This makes
graph-stability decisions auditable and prevents future optimizations from
using incomplete proxy counters.

The covered boundaries include:

- recovery orchestration and singleton consecutive-Reshape results;
- late layout, hard-activation, SPP, QKV, and binary-layout clusters;
- terminal Hardswish/SE, split/Conv/Concat bridges, shape extraction, and
  Expand/Squeeze/Reshape recovery;
- terminal and pre-terminal slice/pad/concat and affine recovery;
- channel-slice/pad-Mul and composite pre-ADD owners;
- indexed affine post-ADD and InstanceNorm post-bias,
  residual/Mul/Concat, and dual-statistics owners;
- the separate pre-terminal and absolute-final post-bias observation points.

Raw counters are used only when the owner prunes after a positive rewrite. An
owner that can prune with zero rewrites additionally reports the clamped net
tensor reduction. Non-mutating iteration counters are excluded from mutation
summaries. These rules make every stability guard conservative without adding
ModelIR copies or additional graph passes.

## Complete static-shape reconciliation evidence

Static-shape reconciliation historically exposed only the number of output
tensor shape updates. That legacy counter intentionally omitted valid
parameter-only repairs, such as correcting a stale Reshape `newShape`, so it
could not safely be used as complete graph-mutation evidence.

The reconciler now offers an opt-in complete counter. Its existing fixed-point
walk records operator-option changes, constant shape-parameter writes, direct
tensor metadata updates, and ordinary output shape updates while performing
the original work. The default return dictionary remains byte-for-byte
compatible; only the selected very-late call requests the additional
`reconciled_static_shape_mutations` key and stages the result as
`_very_late_static_shape_stats`. No fingerprint, graph copy, pre/post scan, or
new dependency is required.

The guarded reconciliation after unsupported-dtype Split fallback uses the
same complete evidence contract. Its owner counter already covers every
rewrite and cleanup path, so the existing positive guard remains unchanged.
The no-rewrite path exposes a stable two-key zero value; the rewrite path
stages the opt-in result as `_post_split_fallback_static_shape_stats`.

Safety-fallback norm cleanup now reports cleanup-only tensor pruning alongside
its legacy rewrite counter. A before/after tensor count is sampled around the
existing owner, and the clamped delta is stored as `pruned_unused_tensors`.
The established rewrite-only reconciliation guard remains unchanged because
unused-tensor deletion alone does not require shape propagation.

The fallback-only dynamic rank-one Unsqueeze/Reshape-shape result is also
staged instead of discarded. Its following topological and logical-layout
refreshes remain unconditional, preserving the recursive fallback contract
while making later scan-elision analysis evidence-based.

The fallback broadcast repair now exposes a stable zero reconciliation result
and replaces it with complete opt-in evidence only after a positive rewrite.
Its complete owner counter, guard, topological refresh, and layout inference
remain unchanged.

The fallback SINet-shuffle and SE/FC/Gather aggregate now stages the same
complete reconciliation evidence. Its three rewrite counters, cleanup-only
tensor delta, combined guard, and following fallback order are unchanged.

The fallback placeholder-MatMul restore is no longer embedded in its guard.
Its single invocation is staged, and a stable zero reconciliation value is
replaced by opt-in complete evidence only after a positive restore. The
positive-only pruning contract and following sort remain unchanged.

Fallback unbound-input repair no longer performs a redundant second
static-shape reconciliation. Its compatibility wrapper already reconciles once
after a positive indexed repair and reuses the live GraphIndex; the caller now
continues directly to Conv-input repair, eliminating one full-graph scan on
that positive fallback path.

## Smaller internal owners

- Typed ONNX `Constant` lowering is isolated in its op-family module while
  preserving supported attribute forms, dtypes, provenance, and errors.
- Shape-readiness decisions are extracted into a demand-driven helper so the
  lowerer does not duplicate readiness logic.
- Late binary recovery has a dedicated runner with an explicit result schema.
- Late hard-activation, SPP, QKV, channel-slice/pad-Mul, affine recovery, and
  related orchestration modules expose their ordered results directly.
- `NodeView` value-list annotations were clarified for static type checking;
  runtime behavior is unchanged.

These extractions reduce implicit coupling in the main lowerer and make a pass
owner, its mutation contract, and its regression tests discoverable together.

## TensorFlow-free and artifact guarantees

No dependency was added. Normal `flatbuffer_direct` lowering and its focused
tests continue to run with `PYTHONNOUSERSITE=1 uv run --no-sync`. Optional
TensorFlow import-blocking coverage remains in every broad gate. The changes do
not add an exporter invocation, quantization path, split path, model-inference
worker, or TensorFlow import.

## Validation

The latest expanded sequential regression gate covers the changed indexed
owners, late/terminal orchestrators, static-shape reconciliation, shared
ModelIR pass context, core contracts, pass efficiency, architecture
constraints, and the optional TensorFlow import boundary. Result: `1339
passed`. The dedicated reconciliation/convergence gate was repeated for the
current post-Split checkpoint and produced `433 passed in 26.91s`.

The subsequent safety-fallback norm-evidence checkpoint extends that gate to
`446 passed in 27.35s`; dynamic-rank-one result staging extends it to `447
passed in 27.34s`; and broadcast reconciliation staging extends it to `451
passed in 27.03s`. The SE/FC/Gather reconciliation checkpoint extends the
focused branch gate to `463 passed in 27.00s`; placeholder-MatMul staging
extends it to `464 passed in 26.93s`; duplicate unbound reconciliation removal
extends it to `470 passed in 27.49s`.

Focused Ruff, Python bytecode compilation, and `git diff --check` also pass.
These results are contract and orchestration tests; they do not claim a new
full model-corpus run for this observation and accounting unit.

## Remaining work

The broader `flatbuffer_direct` refactor remains active. The next characterized
unit should determine whether the fallback dynamic-rank-one evidence can safely
guard either following refresh. If equivalence is not locally provable, it
should leave both refreshes unchanged. The next local audit is the combined
fallback placeholder-MatMul restore predicate plus its guarded reconciliation.
The next local audit is the fallback unbound-input repair and its guarded
reconciliation, followed by the cleanup-capable Conv-input aggregate. Any new
mutation evidence must preserve the recursive fallback boundary, current pass
order, TensorFlow-free boundary, dependency set, and sequential validation
policy.
