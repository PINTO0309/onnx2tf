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

The fallback NCHW Concat/Transpose/Conv-axis repair now retains complete
reconciliation evidence under its unchanged positive guard. Its indexed
matching, axis/metadata writes, other production occurrences, and following
binary-layout owner remain unchanged.

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

Fallback Conv-input repair now merges cleanup-only tensor pruning into its two
existing rewrite counters. The stale-Transpose-only reconciliation guard is
unchanged, but its result is retained with the complete opt-in schema instead
of being discarded.

The fallback mixed-NHWC-input repair for NCHW Concat now retains complete
reconciliation evidence under its unchanged positive guard. Its adapter
insertion, Concat rewire, output metadata, and following Concat-axis owner are
unchanged.

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
extends it to `470 passed in 27.49s`; and complete fallback Conv-input evidence
extends it to `480 passed in 27.81s`. Mixed-Concat reconciliation staging
extends the current focused branch gate to `491 passed in 29.82s`.

Fallback Concat-axis reconciliation staging extends that gate to `520 passed
in 27.79s`.

Complete fallback stale-binary-layout evidence extends the focused gate to
`534 passed in 27.64s`. This checkpoint records cleanup-only tensor pruning,
initializes a stable two-key reconciliation result, and captures the opt-in
complete reconciliation result without changing the repair-only guard or the
following topological sort.

Terminal fallback layout validation extends the current gate, including the
existing rank-6 BatchMatMul structure and numeric-parity tests, to `538 passed
in 27.90s`. The pure validator now observes the graph after all fallback
mutations and clears only validation errors inherited from recursive lowering
when the terminal graph is valid. The error-list schema is unchanged.

Fallback high-rank BatchMatMul result staging extends that gate to `540 passed
in 27.98s`. The owner counter is proven complete, including a true no-op path,
and the fallback now retains a stable complete reconciliation result without
changing compression eligibility, reshape construction, pruning, sorting, or
the following convergence owner.

The fallback now also retains the complete indexed binary-layout convergence
dictionary, extending the gate to `541 passed in 27.89s`. No extra
reconciliation or scan is added: the owner already aggregates every broadcast,
stale-Transpose, and shape mutation across its bounded rounds.

Primary-path terminal layout validation extends the expanded gate to `544
passed in 27.95s`. Validation now observes indexed binary convergence,
high-rank binary coalescing, boundary-signature realignment, and the final
topological sort. Valid terminal graphs clear only a stale validation-error
key; progress reporting and artifact behavior are unchanged.

The three primary terminal mutation dictionaries are now retained, extending
the expanded gate to `564 passed in 28.36s`. This adds no scans or
reconciliation: bounded binary convergence, high-rank binary coalescing, and
boundary-signature realignment already return complete evidence.

Primary final high-rank BatchMatMul reconciliation staging extends the gate to
`565 passed in 28.72s`. The primary guard now retains the same stable complete
shape evidence as the recursive fallback without changing compression,
pruning, layout sync, sorting, or the following Pad owner.

Primary final Pad reconciliation staging extends the expanded gate to `570
passed in 28.52s`. Its exact adapter-insertion counter controls the unchanged
guard, and the complete shape result is retained without changing Pad matching,
lineage, layout sync, sorting, or the following Conv-input repair.

Complete primary final Conv-input evidence extends the gate to `572 passed in
28.31s`. The caller now records cleanup-only tensor pruning and retains the
complete guarded shape result without broadening the rewrite-only guard or
changing the following mixed-Concat owner.

Primary final mixed-Concat reconciliation staging extends the gate to `573
passed in 28.97s`. The complete counter and unchanged guard now retain the
opt-in shape result without changing adapter insertion, rewiring, output
metadata, sorting, or the following Concat-axis owner.

Complete final Concat-axis and stale-binary evidence extends the gate to `574
passed in 28.45s`. Concat-axis retains its complete guarded shape result, while
stale-binary cleanup-only pruning is recorded explicitly without broadening
its rewrite-only guard or moving the progress boundary.

Primary final SiNet Concat/Resize reconciliation staging extends the expanded
gate to `633 passed in 28.54s`. Its transactional counter controls the same
guard, which now retains complete shape evidence without changing matching,
rewiring, pruning, layout sync, or the following high-rank BMM owner.

The remaining five primary final SiNet guards now retain complete
reconciliation evidence, extending the expanded gate to `1090 passed in 29.25s`.
All transactional owners, exact counters, positive-only prune/layout sync, and
the late-residual-to-Concat/Resize order are unchanged.

The absolute-final consecutive-Reshape guard now retains the same stable,
complete reconciliation evidence. Its three existing counters already cover
every no-op removal, single-consumer chain removal, and fan-out bypass, while
pruning and layout synchronization remain positive-only. The aggregate guard,
runner schema, and following final SiNet boundary are unchanged.

This checkpoint extends the expanded related gate to `1094 passed in 29.69s`;
the focused consecutive-Reshape, terminal-orchestration, and architecture gate
is `282 passed in 18.99s`.

The immediately preceding final PReLU path now retains its complete guarded
reconciliation result as well. Its existing tensor-count sample remains
necessary because the owner preserves a legacy unconditional-prune contract;
the rewrite-or-net-reduction guard and following consecutive-Reshape boundary
are unchanged.

This checkpoint extends the expanded related gate to `1124 passed in 30.38s`;
the focused indexed-PReLU, terminal-orchestration, and architecture gate is
`298 passed in 19.18s`.

The primary final SiNet-shuffle plus SE/FC/Gather aggregate now mirrors the
recursive fallback by retaining stable, complete guarded reconciliation
evidence. Its three counters and aggregate tensor-count sample continue to
cover positive rewrites and zero-rewrite pruning without changing orchestration
order or adding a scan.

This checkpoint extends the expanded related gate to `1136 passed in 30.02s`;
the focused three-owner, orchestration, core-guard, and architecture gate is
`543 passed in 17.93s`.

The final placeholder-MatMul conditional block now retains complete results for
both of its existing reconciliation calls. A one-key in-memory projection keeps
the nested binary-repair guard on its exact legacy output-shape semantics, so
the additional complete mutation key cannot cause an extra second scan.

This checkpoint extends the expanded related gate to `1181 passed in 30.44s`;
the focused restore, binary-adapter, terminal, core-guard, and architecture gate
is `382 passed in 19.28s`.

The preceding final mixed-singleton Concat path now retains stable, complete
reconciliation evidence under its unchanged exact repair counter. Adapter
planning, rewiring, positive-only pruning/layout synchronization, and the
following placeholder-MatMul boundary are unchanged.

This checkpoint extends the expanded related gate to `1213 passed in 29.96s`;
the focused indexed-owner, terminal, core-guard, and architecture gate is
`357 passed in 19.46s`.

The preceding final rank-four channelwise broadcast block now retains complete
reconciliation evidence under its unchanged exact repair counter. The existing
reconciliation→sort→layout-inference order and the following mixed-singleton
Concat boundary are unchanged.

This checkpoint extends the expanded related gate to `1214 passed in 30.53s`;
the focused binary-owner, convergence, terminal, and architecture gate is
`286 passed in 17.42s`.

The preceding final decomposed-InstanceNorm block now retains complete
reconciliation evidence under its unchanged exact repair counter. Plan
validation/application and the reconciliation→sort→layout-inference order are
unchanged.

This checkpoint extends the expanded related gate to `1252 passed in 30.10s`;
the focused indexed-InstanceNorm, terminal, and architecture gate is
`310 passed in 18.35s`.

The preceding final ConvInteger structural-repair path now retains complete
reconciliation evidence without broadening its guard to self-contained channel-
last hint propagation. The repair-only reconciliation→sort→layout-inference
order remains unchanged.

This checkpoint extends the expanded related gate to `1256 passed in 30.30s`;
the focused quantized-layout, terminal, and architecture gate is
`277 passed in 17.28s`.

The absolute-final dynamic rank-one Unsqueeze/Reshape-shape call now retains its
exact raw result, completing result capture across its very-late, recursive-
fallback, and absolute-final occurrences without adding a guard or
reconciliation.

This checkpoint extends the expanded related gate to `1257 passed in 30.51s`;
the focused dynamic-Reshape, terminal-orchestration, architecture, and
occurrence gate is `311 passed in 18.45s`.

The immediately preceding absolute-final normalization/attention pair now
propagates its two ordered mutation dictionaries through the orchestration
runner and lowerer helper into one primary-path result assignment. This reuses
the existing shared recovery result tuple and adds no pass execution, scan,
guard, reconciliation, or summary traversal.

This checkpoint extends the expanded related gate to `1268 passed in 31.17s`;
the focused pair-owner, pass-efficiency, architecture, and terminal-
orchestration gate is `317 passed in 20.12s`.

The final boundary-signature restore now retains both the dynamic-map
realignment dictionary and the complete static-signature
repair/preservation dictionary. All earlier and terminal occurrences retain
their existing targets, and the realign→sanitize→affine order is unchanged.

This checkpoint extends the expanded related gate to `1269 passed in 30.78s`;
the focused signature-owner, terminal, absolute-final orchestration, and
architecture gate is `306 passed in 17.88s`.

The guarded no-layout final cleanup now retains the raw SE/FC propagation and
indexed affine pre/post dictionaries inside the existing fallback-only branch.
The branch condition, callback arguments, surrounding topological sorts, and
following signature restore are unchanged.

This checkpoint extends the expanded related gate to `1372 passed in 30.75s`;
the focused SE/FC, indexed-affine, terminal, architecture, and pass-efficiency
gate is `417 passed in 17.82s`.

The primary final precision sequence now retains the raw divisor-rewrite,
consecutive-Mul-fold, and precision-sensitive divisor-restore dictionaries.
Only the adjacent final trio changed; earlier core-cleanup and recursive-
fallback occurrences and the following progress/sort boundary remain fixed.

This checkpoint extends the expanded related gate to `1396 passed in 30.84s`;
the focused precision, graph-cleanup, terminal, architecture, and pass-
efficiency gate is `332 passed in 17.67s`.

The recursive safety fallback now retains the same three precision dictionaries
under fallback-specific targets while preserving its deliberate omission of
layout-state handoff. The preceding fallback sort and following unbound-input
repair remain unchanged.

This checkpoint extends the expanded related gate to `1397 passed in 31.63s`;
the focused safety-fallback, precision, graph-cleanup, terminal, and
architecture gate is `319 passed in 18.49s`.

The earlier primary core-cleanup consecutive-Mul call now retains its raw
dictionary as well, completing result capture for all three production
occurrences of that transactional owner. Its pseudo-LeakyReLU/YOLO predecessors
and terminal-sanitizer successor are unchanged.

This checkpoint extends the expanded related gate to `1398 passed in 31.86s`;
the focused graph-cleanup, terminal, architecture, and pass-efficiency gate is
`330 passed in 18.00s`.

The two preceding primary core-cleanup fusions now retain their pseudo-
LeakyReLU and YOLO-decode mutation dictionaries. Their single occurrences,
zero-argument option surface, ordering, and captured consecutive-Mul successor
are unchanged.

This checkpoint extends the expanded related gate to `1432 passed in 31.30s`;
the focused indexed-fusion, graph-cleanup, terminal, architecture, and pass-
efficiency gate is `364 passed in 18.37s`.

Both terminal quantization-cleanup pairs now retain their raw Transpose/
Dequantize sanitizer and transactional Quantize/Dequantize dictionaries under
phase-specific targets. Pair order, callback contracts, progress boundaries,
and Conv-affine successors are unchanged.

This checkpoint extends the expanded related gate to `1480 passed in 31.26s`;
the focused quantization-cleanup, terminal, architecture, and pass-efficiency
gate is `359 passed in 17.79s`.

The Conv MUL/ADD affine fold and Conv/binary activation fusion immediately
following each terminal quantization-cleanup pair now retain all four raw
result dictionaries under phase-specific targets. Pass order, arguments,
mutation behavior, and distinct dynamic-Reshape/ArgMax successors are
unchanged; the third later Conv-affine occurrence remains outside this unit.

This checkpoint extends the expanded related gate to `1508 passed in 30.41s`;
the focused Conv-affine/activation, terminal convergence/orchestration, and
architecture gate is `322 passed in 19.48s`.

The third and final direct Conv MUL/ADD affine fold now retains its four-counter
result after the shared NDHWC-gate/cost-volume scope. The exact owner arguments,
both surrounding state-scope boundaries, and all three production occurrences
remain fixed; no result consumer or additional graph work was introduced.

This checkpoint extends the expanded related gate to `1557 passed in 32.00s`;
the focused Conv-affine, NDHWC/cost-volume, pass-efficiency, terminal-
orchestration, and architecture gate is `373 passed in 18.53s`.

The four adjacent late Concat shared-scope runners now retain their axis-3
constant-Concat, Dequantize/Concat/Quantize, LayerNorm-statistics, and generic
Transpose-cleanup dictionaries. The state scope, transactional execution,
diagnostics, call order, following optimize-layout guard, and the other two
lowerer Transpose-cleanup occurrences are unchanged.

This checkpoint extends the expanded related gate to `1630 passed in 32.11s`;
the focused four-owner, shared-scope efficiency, terminal-orchestration, and
architecture gate is `387 passed in 17.89s`.

Both guarded elementwise NHWC→NCHW fanout-roundtrip calls now retain their
one-counter dictionaries under late-Concat and terminal phase targets. Values
remain guard-local; no default, consumer, call, or additional graph work was
introduced, and both distinct outer boundaries are unchanged.

This checkpoint extends the expanded related gate to `1645 passed in 32.11s`;
the focused elementwise-fanout, terminal-singleton, layout-recovery, terminal-
orchestration, and architecture gate is `299 passed in 19.15s`.

The adjacent late ExpandDims and flatten-HW Transpose/Reshape compatibility
calls now retain their one-counter dictionaries. Their indexed-first/fallback
behavior, live LayoutState handoff, pruning synchronization, exact adjacency,
captured fanout predecessor, and following NHWC-Reshape owner are unchanged.

This checkpoint extends the expanded related gate to `1661 passed in 32.15s`;
the focused indexed/compatibility-owner, terminal-orchestration, and
architecture gate is `301 passed in 18.71s`.

The immediately following private rank-three layout-shim collapse now retains
its one-counter NHWC-Reshape dictionary. Its internal GraphIndex mutation and
positive-only pruning, model-only callback, captured flatten-HW predecessor,
and channel-shuffle/Gather policy boundary are unchanged.

This checkpoint extends the expanded related gate to `1734 passed in 32.13s`;
the focused collapse-owner, channel-shuffle boundary, layout-recovery,
terminal-orchestration, and architecture gate is `365 passed in 20.07s`.

Channel-shuffle/Gather orchestration now returns the ordered two-to-seven child
result tuple already produced by its recovery runner. The local helper
propagates it, and the guarded full-post plus unguarded late-base calls retain
raw tuples under phase targets. All eight policy combinations, pass order,
shared scope, diagnostics, and boundaries remain unchanged.

This checkpoint extends the expanded related gate to `1736 passed in 31.42s`;
the focused all-policy runner, helper, pass-efficiency, layout-recovery,
terminal-orchestration, and architecture gate is `348 passed in 19.69s`.

The immediately following attention-QKV Reshape/Transpose compatibility call
now retains its existing one-counter dictionary as
`_late_attention_qkv_reshape_stats`. Indexed-first/fallback behavior, the live
Session LayoutState, exact pass order, channel-shuffle predecessor, cleanup
successor, and TensorFlow-free boundary are unchanged. The value has no
consumer and therefore introduces no additional graph traversal or mutation.

This checkpoint passes the focused QKV owner and orchestration gate with
`326 passed in 19.68s`, plus the branch-changed broad related suite with
`1379 passed in 23.78s`.

The direct late attention Gather/Transpose/Reshape cleanup call now retains its
existing pattern-A/pattern-B rewrite dictionary as
`_late_attention_gather_cleanup_stats`. The separate recovery-runner selection,
model-only callback, GraphIndex/pruning behavior, QKV predecessor,
live-LayoutState Gather-axis0 successor, and TensorFlow-free boundary are
unchanged. The retained value has no consumer or additional graph work.

This checkpoint passes the focused cleanup/QKV/Gather-axis0 orchestration gate
with `437 passed in 19.04s`, plus the branch-changed broad related suite with
`1514 passed in 24.34s`.

The direct late Gather-axis0 singleton-to-Reshape compatibility call now
retains its existing one-counter dictionary as
`_late_gather_axis0_reshape_stats`. The separate recovery-runner selection,
indexed GraphIndex behavior, live Session LayoutState, cleanup predecessor,
preprojection successor, and TensorFlow-free boundary are unchanged. The value
has no consumer and adds no traversal or mutation.

This checkpoint passes the focused Gather/cleanup/preprojection orchestration
gate with `496 passed in 18.93s`, plus the branch-changed broad related suite
with `1581 passed in 24.33s`.

The direct late attention-preprojection Reshape-to-BatchMatMul rank-lift call
now retains its existing one-counter dictionary as
`_late_attention_preproj_ranklift_stats`. The separate recovery-runner
selection, model-only owner, GraphIndex/pruning behavior, Gather-axis0
predecessor, live-LayoutState window-partition successor, and TensorFlow-free
boundary are unchanged. The value has no consumer or additional graph work.

This checkpoint passes the focused preprojection/Gather/window-partition
orchestration gate with `451 passed in 18.88s`, plus the branch-changed broad
related suite with `1633 passed in 24.31s`.

The direct late window-partition Reshape/Transpose-to-SpaceToDepth call now
retains its existing one-counter dictionary as `_late_window_partition_stats`.
The separate recovery-runner selection, indexed GraphIndex behavior, live
Session LayoutState, preprojection predecessor, window-reverse successor, and
TensorFlow-free boundary are unchanged. The value has no consumer and adds no
traversal or mutation.

This checkpoint passes the focused window-owner orchestration gate with
`415 passed in 18.79s`, plus the branch-changed broad related suite with
`1634 passed in 25.07s`.

The adjacent direct late window-reverse Reshape/Transpose-to-DepthToSpace call
now retains its existing one-counter dictionary as `_late_window_reverse_stats`.
The separate recovery-runner selection, indexed GraphIndex behavior, live
Session LayoutState, window-partition predecessor, indexed final-convergence
successor, and TensorFlow-free boundary are unchanged. The value has no
consumer and adds no traversal or mutation.

This checkpoint passes the focused window/final-convergence orchestration gate
with `411 passed in 18.65s`, plus the branch-changed broad related suite with
`1683 passed in 24.57s`.

The sole late indexed final shape/activation convergence call now retains its
existing aggregate mutation dictionary as
`_late_final_shape_activation_convergence_stats`. Its one-index internal
convergence, conditional reconciliation, activation fusion, live LayoutState,
window-reverse predecessor, boundary-normalization successor, and TensorFlow-
free boundary are unchanged. The retained value has no consumer.

This checkpoint passes the focused convergence/window/boundary orchestration
gate with `390 passed in 20.13s`, plus the branch-changed broad related suite
with `1688 passed in 24.36s`.

Of the two boundary-input normalization calls, the final occurrence now retains
its existing one-counter dictionary as
`_final_boundary_input_normalization_stats`. The earlier occurrence remains an
unchanged raw call. Transaction/preflight behavior, live Session LayoutState,
diagnostics, final-convergence predecessor, internal channel-slice successor,
and TensorFlow-free boundary are unchanged. The retained value has no consumer.

This checkpoint passes the focused two-occurrence boundary contract with
`343 passed in 18.16s`, plus the branch-changed broad related suite with
`1689 passed in 24.15s`.

The earlier boundary-input normalization call now also retains its existing
one-counter dictionary as `_terminal_boundary_input_normalization_stats`.
Together with the already captured final occurrence, both production calls now
retain raw results independently. Owner behavior, live Session LayoutState,
diagnostics, terminal-softmax/boundary-channel-slice neighbors, final-
convergence boundaries, and TensorFlow-free behavior are unchanged.

This checkpoint passes the focused two-occurrence/terminal-neighbor gate with
`360 passed in 18.72s`, plus the branch-changed broad related suite with
`1719 passed in 24.61s`.

The sole terminal Softmax/Transpose-after-NHWC-propagation call now retains its
existing one-counter dictionary as `_terminal_softmax_transpose_stats`.
Indexed GraphIndex behavior, live Session LayoutState, Gather-channel-fanout
predecessor, captured boundary-normalization successor, and TensorFlow-free
boundary are unchanged. The value has no consumer or extra graph work.

This checkpoint passes the focused terminal Softmax/Gather/boundary gate with
`365 passed in 18.32s`, plus the branch-changed broad related suite with
`1724 passed in 25.28s`.

The direct terminal Gather-channel-fanout call now retains its existing one-
counter dictionary as `_terminal_transpose_gather_channel_fanout_stats`. The
same callback's two orchestrated selections, transaction/preflight behavior,
live Session LayoutState, diagnostics, ArgMax predecessor, captured Softmax
successor, and TensorFlow-free boundary are unchanged. The retained value has
no consumer.

This checkpoint passes the focused ArgMax/fanout/Softmax accounting gate with
`400 passed in 18.38s`, plus the branch-changed broad related suite with
`1763 passed in 24.86s`.

The sole terminal pre-ArgMax production call now retains its existing one-
counter dictionary as `_terminal_pre_argmax_stats`. This assignment-only
orchestration change preserves the live Session LayoutState, terminal Conv-
activation predecessor, captured Gather-channel-fanout successor, pass
implementation, schema, and TensorFlow-free boundary. The retained value has
no consumer.

This checkpoint passes the focused Conv-activation/ArgMax/fanout/Softmax
accounting gate with `417 passed in 19.64s`, plus the branch-changed broad
related suite with `1764 passed in 25.32s`.

The sole boundary-input Transpose/channel-slice production call now retains its
existing four-counter dictionary as
`_terminal_boundary_input_channel_slice_stats`. A small zero-mutation test fixes
the exact schema, while existing rewrite coverage continues to validate shared
GraphIndex/LayoutState synchronization. The assignment-only orchestration
change preserves the captured normalization predecessor, internal channel-
slice successor, pass implementation, and TensorFlow-free boundary. The value
has no consumer.

This checkpoint passes the focused boundary-input/channel-slice accounting
gate with `339 passed in 19.11s`, plus the branch-changed broad related suite
with `1383 passed in 24.43s`.

The first of two internal Transpose/channel-slice propagation calls now retains
its existing four-counter dictionary as `_terminal_internal_channel_slice_stats`.
The later model-only occurrence remains raw. A small zero-mutation test fixes
the exact schema, and existing rewrite coverage retains shared GraphIndex/
LayoutState validation. The assignment-only orchestration change preserves the
captured boundary owner, live-LayoutState MulAdd-bridge successor, pass
implementation, and TensorFlow-free boundary. The value has no consumer.

This checkpoint passes the focused boundary/internal channel-slice accounting
gate with `341 passed in 18.70s`, plus the branch-changed broad related suite
with `1385 passed in 24.19s`.

The later model-only internal Transpose/channel-slice propagation call now
retains the same four-counter schema as `_final_internal_channel_slice_stats`,
while the first live-LayoutState occurrence keeps
`_terminal_internal_channel_slice_stats`. The assignment-only orchestration
change preserves final boundary normalization, the later model-only MulAdd-
bridge successor, owner behavior, and the TensorFlow-free boundary. Neither
value has a consumer.

This checkpoint passes the focused boundary/internal channel-slice accounting
gate with `342 passed in 18.84s`, plus the branch-changed broad related suite
with `1386 passed in 23.44s`.

The later model-only Transpose/channel-slice MulAdd-bridge call now retains its
existing one-counter dictionary as `_final_channel_slice_muladd_bridge_stats`.
The first live-LayoutState occurrence remains raw. Recovery-boundary contracts
now explicitly distinguish those two forms, while preserving both ordered
recovery calls, owner behavior, and the TensorFlow-free boundary. The retained
value has no consumer.

This checkpoint passes the focused MulAdd-bridge/recovery accounting gate with
`346 passed in 20.08s`, plus the branch-changed broad related suite with
`1391 passed in 24.78s`.

The first live-LayoutState Transpose/channel-slice MulAdd-bridge call now
retains the same one-counter schema as
`_terminal_channel_slice_muladd_bridge_stats`, while the later model-only call
keeps `_final_channel_slice_muladd_bridge_stats`. Recovery and architecture
contracts now require both targets and preserve both ordered recovery calls.
The assignment-only orchestration change leaves owner behavior and the
TensorFlow-free boundary unchanged. Neither value has a consumer.

This checkpoint passes the focused MulAdd-bridge/recovery accounting gate with
`347 passed in 20.85s`, plus the branch-changed broad related suite with
`1392 passed in 24.26s`.

The sole boundary-input Transpose/StridedSlice/QDQ/Concat production call now
retains its existing four-counter dictionary as
`_terminal_boundary_stridedslice_qdq_concat_stats`. Recovery outer-boundary
contracts now distinguish that assignment from the later raw successor while
preserving the live Session LayoutState, Swish-residual closure, owner behavior,
and TensorFlow-free boundary. The retained value has no consumer.

This checkpoint passes the focused boundary-StridedSlice/recovery accounting
gate with `349 passed in 20.54s`, plus the branch-changed broad related suite
with `1394 passed in 24.34s`.

Two stale indexed quantized-Swish tests no longer monkeypatch compatibility map
helpers that are absent from the lowerer. They now assert the actual owner
module's indexed-only contract while preserving runtime GraphIndex refresh,
currentness, mutation-counter, metadata, and fixed-point checks. The complete
owner module passes `21 passed in 0.53s`; the related terminal orchestration,
architecture, and pass-efficiency gate passes `364 passed in 18.52s`.

The sole model-only Swish-residual-closure production call now retains its
existing four-counter dictionary as
`_terminal_swish_residual_concat_closure_stats`. The assignment-only
orchestration change preserves fixed owner options, indexed phase order,
captured boundary-StridedSlice/dequant-logistic boundaries, and the TensorFlow-
free core. The retained value has no consumer.

This checkpoint passes the focused indexed-Swish/orchestration gate with
`365 passed in 19.09s`, plus the branch-changed broad related suite with
`1416 passed in 24.00s`.

The sole model-only dequant-logistic-Mul-quantize bridge call now retains its
existing one-counter dictionary as
`_terminal_dequant_logistic_mul_quantize_bridge_stats`. The assignment-only
orchestration change preserves the optional indexed owner contract, captured
Swish-closure/Swish-QDQ-island boundaries, and TensorFlow-free core. The
retained value has no consumer.

This checkpoint passes the focused indexed logistic/Swish orchestration gate
with `382 passed in 19.29s`, plus the branch-changed broad related suite with
`1433 passed in 24.66s`.

The sole model-only Swish-QDQ-island production call now retains its existing
five-counter dictionary as `_terminal_swish_qdq_island_stats`. The assignment-
only orchestration change preserves default options, indexed phase order,
captured dequant-logistic/InstanceNorm-bias boundaries, and TensorFlow-free
core. The retained value has no consumer.

This checkpoint passes the focused indexed logistic/Swish orchestration gate
with `376 passed in 18.74s`, plus the branch-changed broad related suite with
`1434 passed in 24.48s`.

All four direct InstanceNorm post-Transpose bias/add calls now retain their
existing one-counter dictionaries. Their distinct targets are
`_terminal_instancenorm_post_bias_stats`,
`_very_late_instancenorm_post_bias_stats`,
`_pre_terminal_affine_instancenorm_post_bias_stats`, and
`_absolute_final_instancenorm_post_bias_stats`. The nested convergence call
continues to consume its counter separately. Occurrence-shape contracts now
distinguish all five forms. The direct retained values have no consumers.

This checkpoint passes the focused InstanceNorm/Swish/terminal-final accounting
gate with `386 passed in 19.95s`, plus the branch-changed broad related suite
with `1419 passed in 23.89s`.

The second direct call now retains its result as
`_very_late_instancenorm_post_bias_stats` between diagnostics-aware pad-layout
cleanup and the live-LayoutState InstanceNorm residual/Mul/Concat owner. This
assignment-only orchestration change leaves the wrapper, indexed owner,
one-key schema, graph mutation, pruning, pass order, callback arguments,
dependencies, TensorFlow-free boundary, and the other four occurrence forms
unchanged.

This checkpoint passes the focused owner/orchestration gate with
`387 passed in 20.17s`, plus the branch-changed broad related suite with
`1420 passed in 23.78s`.

The second of three direct InstanceNorm residual/Mul/Concat/Conv calls now
retains its existing one-counter dictionary as
`_very_late_instancenorm_residual_mul_concat_stats`. It remains between the
captured very-late post-bias result and the live-LayoutState dual-statistics
owner. The first terminal call remains raw, the third retains its pre-terminal
target, and the nested convergence call continues to consume its counter. This
assignment-only change adds no graph work or consumer.

This checkpoint passes the focused indexed owner/orchestration gate with
`469 passed in 19.78s`, plus the branch-changed broad related suite with
`1421 passed in 24.65s`.

The second of three direct dual-statistics InstanceNorm residual/add/resize
calls now retains its existing one-counter dictionary as
`_very_late_instancenorm_dualstats_stats`. It remains between the captured
very-late residual/Mul/Concat result and singleton consecutive-Reshape cluster.
The first terminal call remains raw, the third keeps its pre-terminal target,
and the nested convergence call continues to consume its counter. The retained
value has no consumer and adds no graph work.

The broad gate exposed one stale singleton-boundary test that required its
predecessor to be an expression. That contract now requires the intentional
assignment and exact target instead; the singleton implementation and its
other boundaries are unchanged. The focused gate including that module passes
`686 passed in 20.88s`, and the branch-changed broad related suite passes
`1422 passed in 24.93s`.

The first model-level singleton/consecutive-Reshape cluster call now retains
its existing ordered three-dictionary tuple as
`_very_late_singleton_consecutive_reshape_results`. The later model-level call
continues to destructure its results for the shared reconciliation guard, and
the conditional fallback call remains raw. This assignment-only change
preserves all child-owner ordering, shared state, diagnostics, and graph work.

This checkpoint passes the focused cluster and all three child-owner families
with `380 passed in 18.95s`, plus the branch-changed broad related suite with
`1445 passed in 24.34s`.

The guarded very-late layout-Transpose cleanup now retains its existing five-
key dictionary as `_very_late_layout_transpose_cleanup_stats`. Because the
owner can prune unused tensors with zero rewrite counters, this result remains
observation-only and does not control reconciliation or scan elision. The
earlier guarded occurrence remains raw and the late-Concat occurrence keeps
its existing shared-scope target.

The implementation gate exposed a characterization selector that inspected
only the first statement in each guard. It now scans every direct statement,
so both guarded occurrences are fixed before selecting the very-late one by
its captured predecessor. The targeted contract passes `1 passed in 0.55s`,
the focused gate passes `364 passed in 18.87s`, and the branch-changed broad
related suite passes `1428 passed in 24.87s`.

The very-late rank-four channelwise broadcast-constant repair now retains its
complete one-counter result as `_very_late_broadcast_repair_stats`. Indexed
binary convergence still consumes the module-level occurrence, and the
fallback/final lowerer occurrences keep their existing targets and positive
guards. The new target is observation-only and leaves the immediate static-
shape reconciliation unconditional.

The implementation gate corrected the characterization inventory from four
lowerer calls to four module-wide calls: three are inside `lower_onnx_to_ir`
and one belongs to indexed convergence. The targeted contract passes
`1 passed in 0.60s`, the focused gate passes `377 passed in 18.74s`, and the
branch-changed broad related suite passes `1425 passed in 24.42s`.

The immediate post-broadcast static-shape reconciliation remains unconditional
but now requests its established complete mutation counter and retains the two-
key result as `_very_late_broadcast_static_shape_stats`. This captures
parameter-, option-, metadata-, and ordinary shape writes during the existing
fixed-point walk without a copy or additional traversal. The result has no
consumer.

The implementation gate updated the preceding broadcast-boundary contract to
require the new assignment and opt-in keyword. Those two boundary contracts
pass `2 passed in 0.64s`, the focused reconciliation/owner gate passes
`357 passed in 19.30s`, and the branch-changed broad related suite passes
`1430 passed in 25.27s`.

The existing shared-late reconciliation predicate remains unchanged over nine
mutation dictionaries plus its tensor-count decrease. When that guard fires,
the reconciliation now requests the complete mutation counter and retains
`_shared_late_static_shape_stats`. Runtime fixtures still prove one additional
scan for every positive dictionary and prune-only outcome; the retained result
has no consumer.

The implementation gate updated the matching architecture contract from a raw
expression to the exact assignment. A similar late-binary guard remains a raw
expression and was explicitly rechecked after applying the update with
function-scoped context. The three structural contracts pass
`3 passed in 2.22s`, the focused gate passes `365 passed in 18.89s`, and the
branch-changed broad related suite passes `1427 passed in 24.69s`.

The guarded late-binary static-shape reconciliation now requests the complete
mutation counter and retains `_late_binary_repair_static_shape_stats`. Its
three-counter plus tensor-delta predicate is unchanged, and the existing
runtime fixture still covers each positive and prune-only path. The complete
result has no consumer.

This checkpoint passes the focused runtime/reconciler/architecture gate with
`353 passed in 21.19s`, plus the branch-changed broad related suite with
`1428 passed in 24.96s`.

The nested reconciliation after `late_binary_layout_recovery_stats` now
requests the complete mutation counter and retains
`_late_binary_layout_recovery_static_shape_stats`. The complete recovery
aggregate, outer option guard, inner positive-count predicate, and runtime
rewrite/prune/stable behavior are unchanged. The retained result has no
consumer.

This checkpoint passes the focused recovery/runtime/architecture gate with
`299 passed in 19.68s`, plus the branch-changed broad related suite with
`1429 passed in 24.81s`.

All three direct InstanceNorm residual/Mul/Concat results are now retained at
distinct terminal, very-late, and pre-terminal targets. The new first target is
`_terminal_instancenorm_residual_mul_concat_stats`; the nested convergence call
continues to consume its counter with the shared GraphIndex. This assignment-
only change adds no graph work or result consumer.

The implementation gate updated the very-late cross-occurrence contract from
the former raw terminal expression to the exact target. Those two contracts
pass `2 passed in 0.62s`, the focused owner/orchestration gate passes
`465 passed in 19.59s`, and the branch-changed broad related suite passes
`1430 passed in 25.07s`.

All three direct dual-statistics InstanceNorm residual/add/resize results are
now retained at distinct terminal, very-late, and pre-terminal targets. The new
first target is `_terminal_instancenorm_dualstats_stats`; the nested convergence
call continues to consume its counter with the shared GraphIndex. The terminal
boundary cluster now explicitly requires that assignment as its predecessor.

The boundary and cross-occurrence contracts pass `3 passed in 2.27s`, the
focused owner/orchestration gate passes `569 passed in 20.57s`, and the branch-
changed broad related suite passes `1431 passed in 25.37s`.

Focused Ruff, Python bytecode compilation, and `git diff --check` also pass.
These results are contract and orchestration tests; they do not claim a new
full model-corpus run for this observation and accounting unit.

The terminal direct InstanceNorm residual-add-to-single-post-adapter result is
now retained as `_terminal_instancenorm_residual_add_stats`. Its only other
production occurrence remains the nested convergence call that consumes the
same one-counter dictionary with the shared GraphIndex. The diagnostics-aware
normalization/pad predecessor and retained terminal residual/Mul/Concat
successor remain adjacent.

This assignment-only change passes the focused owner/orchestration gate with
`447 passed in 20.01s` and the branch-changed broad related suite with
`1515 passed in 25.58s`. It adds no graph traversal, result consumer,
dependency, or TensorFlow import path.

The diagnostics-aware terminal normalization/pad aggregate is now retained as
`_terminal_normalization_pad_stats` between the captured InstanceNorm post-bias
and residual-add results. Its loop-local result consumer and two flatten-only
orchestrated selections remain unchanged.

The aggregate's two rewrite counters deliberately remain observation-only:
both child owners can prune unused tensors while returning zero, so no new
guard or result consumer was added. The focused gate passes `375 passed in
19.73s`, concrete normalization/pad fixtures pass `2 passed, 739 deselected in
0.60s`, and the branch-changed broad suite passes `1433 passed in 24.60s`.

The terminal boundary-layout phase now propagates its existing ordered
five-result tuple through `run_terminal_boundary_layout()` and the local helper
to `_terminal_boundary_layout_results`. Child order, single execution, shared
pass-state scope, arguments, diagnostics, and graph mutation are unchanged;
the tuple has no consumer.

The focused owner/orchestration gate passes `375 passed in 21.39s`, and the
branch-changed broad suite passes `1456 passed in 25.15s`. Four stale AST
contracts were updated from raw-expression assumptions to the exact return and
assignment boundaries.

Mean/attention orchestration now propagates its existing policy-selected tuple
through `run_mean_attention()` and the local helper. The two direct primary
calls retain `_layout_pass_set_1_mean_attention_results` and
`_terminal_mean_attention_results`, while two callback contexts continue to
ignore the return value.

All four policy combinations, child order, shared scope, option guards, and
callback references remain fixed. The focused gate passes `330 passed in
20.05s`, and the branch-changed broad suite passes `1464 passed in 26.61s`.
Four stale raw-expression contracts were updated to exact return/assignment
boundaries.

Both BatchMatMul affine-transpose-input results are now retained as
`_terminal_batchmatmul_affine_input_stats` and
`_post_sinet_batchmatmul_affine_input_stats`. The owner and its unconditional
unused-tensor prune are unchanged, so both one-counter dictionaries remain
observation-only and have no consumers.

The focused owner/two-boundary gate passes `369 passed in 18.78s`, and the
branch-changed broad suite passes `1459 passed in 25.79s`. No graph traversal,
guard, dependency, or TensorFlow import path was added.

Both adjacent BatchMatMul reshape/SE results are now retained as
`_terminal_batchmatmul_reshape_se_stats` and
`_post_sinet_batchmatmul_reshape_se_stats`. Their one-counter dictionaries
remain observation-only because the owner can prune while reporting zero; no
consumers or guards were added.

The focused reshape/SE and affine-input gate passes `356 passed in 19.80s`, and
the branch-changed broad suite passes `1461 passed in 25.43s`. Both adj-flags
successors and surrounding option policy remain fixed.

Both BatchMatMul transpose-input-to-adj-flags results are now retained as
`_terminal_batchmatmul_adj_flags_stats` and
`_post_sinet_batchmatmul_adj_flags_stats`. The one-counter result is complete
owner mutation evidence, but remains unconsumed in this unit.

The focused owner/QKV-boundary gate passes `372 passed in 19.41s`, and the
branch-changed broad suite passes `1463 passed in 25.65s`. No child execution,
guard, graph traversal, dependency, or TensorFlow import path was added.

The two default-policy QKV tuples are now retained as
`_terminal_qkv_attention_results` and
`_post_sinet_qkv_attention_results`. The existing late policy tuple and its
complete summary with net tensor pruning remain unchanged; the two new tuples
have no consumers.

The focused QKV/adj-flags gate passes `371 passed in 19.64s`, and the
branch-changed broad suite passes `1464 passed in 25.62s`. A test-only call
extractor was made tolerant of non-call lowerer statements after the retained
assignment exposed that stale assumption.

The two earlier indexed Split/Conv/Concat bridge results are now retained as
`_terminal_qkv_split_conv_concat_bridge_stats` and
`_post_sinet_split_conv_concat_bridge_stats`; the existing late target remains
unchanged. All three complete one-counter dictionaries are distinct and the two
new values have no consumers.

The focused indexed-owner/boundary gate passes `401 passed in 18.59s`, and the
branch-changed broad suite passes `1502 passed in 25.50s`. Singleton boundary
tests now accept an assigned predecessor without changing cluster execution.

Singleton/Reshape orchestration now propagates its existing policy-selected
result tuple through `run_singleton_reshape()` and the local helper. The two
direct primary calls retain `_terminal_singleton_reshape_results` and
`_post_terminal_singleton_reshape_results`; neither tuple has a consumer.

All sixteen policy combinations, exact child order, shared pass-state scope,
and both terminal boundaries remain fixed. The focused gate passes
`392 passed in 19.36s`, and the branch-changed broad suite passes
`1503 passed in 25.34s`. The change adds no child execution, graph traversal,
dependency, or TensorFlow import path.

The raw top-level indexed shape-convergence result is now retained as
`_post_terminal_indexed_shape_convergence_stats`. Its complete three-key
mutation dictionary has no new consumer; the existing nested
`convergence_stats` form remains consumed by final convergence.

The focused schema/two-form/boundary gate passes `362 passed in 20.57s`, and
the branch-changed broad suite passes `1512 passed in 26.73s`. One-index reuse,
conditional final reconciliation, the live Session LayoutState, and both
surrounding recovery boundaries are unchanged.

SiNet terminal-layout orchestration now propagates its existing three-element
`Tuple[Any, ...]` through the phase runner and local helper. The two direct
calls retain `_terminal_sinet_layout_recovery_results` and
`_very_late_sinet_layout_recovery_results`; neither tuple has a consumer.

The injected pre-add/resize callback remains the middle child and its return is
preserved without changing execution. The focused gate passes
`357 passed in 19.62s`, and the branch-changed broad suite passes
`1533 passed in 27.45s`. Child order, context wiring, boundaries, dependencies,
and TensorFlow behavior remain fixed.

SiNet pre-add/resize orchestration now returns its ordered six-dictionary tuple
through the phase runner and local helper. The three direct calls retain
`_terminal_sinet_preadd_resize_results`,
`_very_late_sinet_preadd_resize_results`, and
`_post_cleanup_sinet_preadd_resize_results`; none has a consumer.

Both retained terminal-layout tuples now carry that six-result tuple as their
middle callback result while remaining unconsumed. The focused gate passes
`351 passed in 18.37s`, and the branch-changed broad suite passes
`1534 passed in 26.35s`. Six-child order, arguments, live LayoutState wiring,
callback identity, dependencies, and TensorFlow behavior remain unchanged.

The sole post-cleanup CSP-attention result is now retained as
`_post_cleanup_csp_attention_stats`. Its one-counter dictionary remains
observation-only because the owner prunes unused tensors unconditionally; no
consumer or guard was added.

The focused contract gate passes `307 passed in 19.19s`, both concrete CSP
owner fixtures pass, and the branch-changed broad suite passes
`1530 passed in 26.74s`. The indexed owner, live LayoutState, SiNet predecessor,
SA/PA successor, dependencies, and TensorFlow behavior remain fixed.

The two direct SA/PA MirrorPad results are now retained as
`_layout_opt_sa_pa_mirrorpad_stats` and
`_post_cleanup_sa_pa_mirrorpad_stats`. Their one-counter dictionaries are
complete owner mutation evidence but remain unconsumed in this unit.

The focused owner/orchestration/boundary gate passes `357 passed in 17.70s`,
and the branch-changed broad suite passes `1584 passed in 26.18s`. Positive-only
pruning and LayoutState sync, attention-recovery selection, option policy,
dependencies, and TensorFlow behavior remain unchanged.

Both direct ReLU/Split all-outputs results are now retained as
`_post_sinet_relu_split_all_outputs_stats` and
`_terminal_relu_split_all_outputs_stats`. Their one-counter dictionaries are
complete owner mutation evidence but remain unconsumed in this unit.

The focused owner/QKV/Split-Conv-Concat boundary gate passes
`387 passed in 18.81s`, and the branch-changed broad suite passes
`1575 passed in 27.30s`. Successful-plan counting, positive-only pruning, live
LayoutState, both surrounding sequences, dependencies, and TensorFlow behavior
remain unchanged.

Both adjacent ReLU/Split/Conv/ReLU/Concat post-transpose results are now
retained as `_post_sinet_relu_split_conv_concat_stats` and
`_terminal_relu_split_conv_concat_stats`. Their one-counter dictionaries are
complete owner mutation evidence but remain unconsumed in this unit.

The focused owner and boundary gate passes `389 passed in 18.39s`, the concrete
rewrite fixture passes, and the branch-changed broad suite passes
`1577 passed in 26.79s`. Successful-plan counting, positive-only pruning, live
LayoutState, both predecessors and successors, dependencies, and TensorFlow
behavior remain unchanged.

Both direct Split/mixed pre-Concat adapter results are now retained as
`_layout_opt_split_mixed_pre_concat_stats` and
`_terminal_split_mixed_pre_concat_stats`. Their one-counter dictionaries are
complete owner mutation evidence but remain unconsumed in this unit.

The focused indexed-owner/orchestration/boundary gate passes
`408 passed in 18.71s`, and the branch-changed broad suite passes
`1663 passed in 28.60s`. The option guard, public-owner orchestration selection,
positive-only pruning, live LayoutState, dependencies, and TensorFlow behavior
remain unchanged.

Both direct Concat input-adapter results are now retained as
`_layout_opt_concat_input_adapter_stats` and
`_terminal_concat_input_adapter_stats`. Their owner prunes unused tensors
unconditionally, so both one-counter dictionaries remain observation-only and
have no consumers.

The focused indexed-owner/selection/boundary gate passes
`406 passed in 18.66s`, and the branch-changed broad suite passes
`1663 passed in 27.31s`. The option guard, public-owner orchestration and
private-wrapper safe-reduction selections, live LayoutState, dependencies, and
TensorFlow behavior remain unchanged.

The guarded Slice/Logistic/Concat/Reshape-tail result is now retained as
`_layout_opt_slice_logistic_concat_tail_stats`. Its owner prunes unused tensors
unconditionally, so the one-counter dictionary remains observation-only and
has no consumer.

The focused indexed-owner/orchestration/policy gate passes
`363 passed in 18.58s`, and the branch-changed broad suite passes
`1598 passed in 27.99s`. The option guard, live LayoutState, captured Concat
input-adapter predecessor, channel-shuffle policy, dependencies, and TensorFlow
behavior remain unchanged.

The terminal Concat/unary/Conv cleanup result is now retained as
`_terminal_concat_unary_conv_stats`. Its transactional one-counter dictionary
remains observation-only, while diagnostics continue to record changed and
precondition-skipped outcomes.

The focused runner/transaction/boundary gate passes `401 passed in 20.06s`, and
the branch-changed broad suite passes `1585 passed in 28.42s`. Positive-only
pruning and LayoutState sync, preflight/default details, live diagnostics,
dependencies, and TensorFlow behavior remain unchanged.

The two previously raw shape-extract results are now retained as
`_terminal_shape_extract_stats` and
`_late_pre_layout_cluster_shape_extract_stats`; the existing
`_late_pre_qkv_shape_extract_stats` target remains unchanged. All three complete
one-counter dictionaries remain unconsumed.

The focused three-form owner/boundary gate passes `412 passed in 23.37s`, and
the branch-changed broad suite passes `1556 passed in 27.91s`. Positive-only
pruning, terminal and late-layout boundaries, QKV policy, dependencies, and
TensorFlow behavior remain unchanged.

The guarded Conv/Pool output-transpose result is now retained as
`_terminal_convpool_output_passthrough_stats`. Its owner prunes unused tensors
unconditionally, so the one-counter dictionary remains observation-only and
has no consumer.

The focused owner/branch/boundary gate passes `398 passed in 20.60s`, and the
branch-changed broad suite passes `1656 passed in 28.18s`. The terminal layout
guard, no-layout safe-reduction fallback, HardSigmoid successor, dependencies,
and TensorFlow behavior remain unchanged.

All three direct dequantized HardSigmoid bridge results are now retained as
`_terminal_dequant_hardsigmoid_bridge_stats`,
`_post_sinet_dequant_hardsigmoid_bridge_stats`, and
`_late_dequant_hardsigmoid_bridge_stats`. The owner can prune with a zero
counter when Transposes exist, so all three remain observation-only.

The focused indexed-owner/selection/boundary gate passes
`405 passed in 21.05s`, and the branch-changed broad suite passes
`1654 passed in 29.28s`. Early return, graph-index behavior, SiNet/cost-volume/
late-dequant boundaries, dependencies, and TensorFlow behavior remain
unchanged.

## Remaining work

The broader `flatbuffer_direct` refactor remains active. The next characterized
unit is the raw terminal-SiNet
`_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains()`
result. The owner has exactly two production forms; the later form already
retains prune-aware evidence, while the earlier discarded result is now
characterized for observation-only assignment. The new terminal
dequant-HardSigmoid result, existing late HardSwish-SE evidence, SiNet recovery,
and cleanup evidence semantics must remain fixed. All retained SiNet callback
results, Singleton/Reshape policies, three QKV result forms, and the late
summary must also remain fixed.
The focused characterization gate passes `323 passed, 1 xfailed in 17.83s`;
the sole strict xfail is the selected earlier result assignment.

The earlier call now retains its unchanged dictionary as
`_terminal_sinet_hardswish_se_stats`. It remains observation-only because a
zero rewrite counter does not exclude the owner's unconditional unused-tensor
pruning. The later prune-aware form and all pass behavior remain unchanged.
The focused implementation gate passes `324 passed in 20.03s`, and the
branch-changed broad suite passes `1539 passed in 27.43s`.

The next unit should characterize raw result propagation for the preceding
three-owner `_run_terminal_clamp_unary_relu_pass_cluster()`, which currently
returns `None`. Shared state scope, cleanup-only mutations, and both outer
boundaries must be fixed before either runner layer changes.

That cluster is now characterized for ordered raw result propagation. Its
shared executor already returns the three one-key dictionaries, but both runner
layers discard them. Clamp and unary callbacks can prune while their rewrite
counters are zero, so the selected `_terminal_clamp_unary_relu_results` tuple
must remain observation-only. The shared scope, exact child order, and guarded
singleton/terminal-SiNet boundaries are frozen before implementation.
The focused characterization gate passes
`370 passed, 1 xfailed in 17.98s`; the sole strict xfail is the selected
two-layer propagation and production assignment.

Both runner layers now return the shared executor's unchanged ordered tuple,
and the sole production call retains it as
`_terminal_clamp_unary_relu_results`. It remains observation-only. The focused
implementation gate passes `371 passed in 19.72s`, and the branch-changed broad
suite passes `1541 passed in 27.26s`; targeted static validation also passes.

The next unit should audit both discarded calls to
`_run_terminal_slice_concat_layout_recovery_sequence()`, including every child
schema and cleanup behavior, shared state scope, call multiplicity, and the two
distinct boundary pairs before either runner layer changes.

The terminal slice/concat recovery is now characterized as fourteen ordered
slots at two production sites. Its nested child schemas, final non-mutating
`iterations`, unconditional cleanup evidence, exact ordering, and two boundary
pairs are fixed. Both runner layers still discard the shared executor's tuple;
the selected observation-only targets are
`_terminal_slice_concat_recovery_results` and
`_final_slice_concat_recovery_results`.
The focused characterization gate passes
`465 passed, 1 xfailed in 20.37s`; the sole strict xfail covers the missing
two-layer propagation and both assignments.

Both runner layers now propagate the unchanged fourteen-slot nested tuple, and
the two production calls retain it as
`_terminal_slice_concat_recovery_results` and
`_final_slice_concat_recovery_results`. Both remain observation-only. The
focused implementation gate passes `466 passed in 20.96s`, the branch-changed
broad suite passes `1543 passed in 27.94s`, and targeted static validation
passes.

The next unit should audit the sole raw
`_optimize_transpose_slice_prepost_nhwc_passthrough_chains()` result immediately
after the final slice/concat recovery, including cleanup semantics and the
final-slice/pre-Concat boundaries.

That sole Slice pre/post result is now characterized for assignment-only
retention as `_final_slice_prepost_passthrough_stats`. Its one counter is
complete mutation evidence because unused-tensor cleanup runs only after a
positive rewrite. The compatibility wrapper, exact call count, and adjacent
final-slice/pre-Concat boundaries are frozen before implementation.
The focused characterization gate passes
`371 passed, 1 xfailed in 19.73s`; the sole strict xfail is the selected
assignment.

The sole production call now retains its unchanged dictionary as
`_final_slice_prepost_passthrough_stats`, without adding a consumer or guard.
The focused implementation gate passes `372 passed in 22.41s`, the
branch-changed broad suite passes `1545 passed in 28.53s`, and targeted static
validation passes.

The next unit should audit all three production forms of
`_optimize_transpose_pre_concat_nhwc_chains()`, including distinct boundaries,
layout/diagnostics routing, owner schema, cleanup behavior, and current result
policies before changing any call.

The three direct pre-Concat calls are now characterized separately from the
independent layout-recovery callback form. Selected observation-only targets
are `_layout_opt_pre_concat_stats`, `_final_pre_concat_stats`, and
`_absolute_final_pre_concat_stats`. The one-key composite schema, indexed/
quantized/legacy dispatch order, unconditional legacy cleanup, identical
layout/diagnostics routing, and all three boundary pairs are frozen before
implementation.
The focused characterization gate passes
`430 passed, 1 xfailed in 19.00s`; the sole strict xfail covers the three
selected direct assignments.

The three direct calls now retain their unchanged dictionaries as
`_layout_opt_pre_concat_stats`, `_final_pre_concat_stats`, and
`_absolute_final_pre_concat_stats`. All remain observation-only; the
layout-recovery callback form is unchanged. The focused implementation gate
passes `431 passed in 19.60s`, the branch-changed broad suite passes
`1548 passed in 29.18s`, and targeted static validation passes.

The next unit should audit the sole direct
`run_ndhwc_concat_layout_cleanup()` result and its independent layout-recovery
occurrence, including schema, cleanup semantics, layout/diagnostics routing,
and pre-Concat/strided-Slice boundaries.

That sole direct NDHWC Concat result is now characterized separately from its
layout-recovery occurrence. `_layout_opt_ndhwc_concat_stats` is the selected
observation-only target. The one-key transactional schema, positive-only
cleanup, exact arguments, direct-call count, independent orchestration
selection, and pre-Concat/strided-Slice boundaries are frozen before
implementation.
The focused characterization gate passes
`375 passed, 1 xfailed in 18.42s`; the sole strict xfail is the selected direct
assignment.

The sole direct call now retains its unchanged dictionary as
`_layout_opt_ndhwc_concat_stats`, without changing the independent
layout-recovery occurrence. The focused implementation gate passes
`376 passed in 18.50s`, the branch-changed broad suite passes
`1551 passed in 28.84s`, and targeted static validation passes.

The next unit should audit the sole direct indexed strided-Slice pre-Concat
result and its independent layout-recovery occurrence, including schema,
cleanup semantics, graph-index/layout routing, and NDHWC/split-mixed
boundaries.

That indexed owner is now characterized with one direct call and one separate
layout-recovery occurrence. `_layout_opt_stridedslice_pre_concat_stats` is the
selected observation-only target. Optional graph-index/layout/bound/candidate
forwarding, the one-key schema, unconditional cleanup with a proven
zero-rewrite prune, direct arguments, and NDHWC/split-mixed boundaries are
frozen before implementation.
The focused characterization gate passes
`360 passed, 1 xfailed in 18.98s`; the sole strict xfail is the selected direct
assignment.

The sole direct call now retains its unchanged dictionary as
`_layout_opt_stridedslice_pre_concat_stats`; it remains observation-only, and
the public-owner layout-recovery occurrence is unchanged. The focused
implementation gate passes `361 passed in 18.38s`, the branch-changed broad
suite passes `1554 passed in 29.49s`, and targeted static validation passes.

The next unit should audit the sole direct SPP layout-cleanup result and its
independent layout-recovery occurrence, including transactional schema,
positive-only cleanup, layout/diagnostics routing, and elementwise-Concat/
pre-Concat boundaries.

The SPP runner is now characterized with one direct call and three independent
orchestration selections. `_layout_opt_spp_stats` is the selected direct
target. Its one-key schema, positive-only cleanup, layout/diagnostics routing,
direct-call count, elementwise-Concat/pre-Concat boundaries, and existing
layout-recovery/late-layout/late-SPP result policies are frozen before
implementation.
The focused characterization gate passes
`363 passed, 1 xfailed in 18.83s`; the sole strict xfail is the selected direct
assignment.

The sole direct SPP call now retains its unchanged dictionary as
`_layout_opt_spp_stats`; the three orchestration policies remain unchanged.
The focused implementation gate passes `364 passed in 18.10s`, the
branch-changed broad suite passes `1556 passed in 28.08s`, and targeted static
validation passes.

The next unit should audit the sole direct elementwise-Concat/Conv group result
and its independent layout-recovery owner occurrence, including optional
graph-index/layout/bound/candidate forwarding, schema, cleanup semantics, and
binary-bridge/SPP boundaries.

That indexed owner is now characterized with one direct private-wrapper call
and one independent public-owner layout-recovery occurrence.
`_layout_opt_elementwise_concat_conv_stats` is the selected observation-only
target. Optional graph-index/layout/bound/candidate forwarding, the one-key
schema, unconditional cleanup with a proven zero-rewrite prune, exact direct
arguments, and binary-bridge/SPP boundaries are frozen before implementation.
The focused characterization gate passes
`358 passed, 1 xfailed in 19.01s`; the sole strict xfail is the selected direct
assignment.

The sole direct call now retains its unchanged dictionary as
`_layout_opt_elementwise_concat_conv_stats`; it remains observation-only, and
the public-owner layout-recovery occurrence is unchanged. The focused
implementation gate passes `359 passed in 19.54s`, the branch-changed broad
suite passes `1559 passed in 27.59s`, and targeted static validation passes.

The next unit should audit both calls to the quantized-activation binary-bridge
recovery helper and the runner result they currently discard, while preserving
their distinct conditional-binary-bridge and elementwise-Concat/Conv
successors.

That nested recovery boundary is now characterized as six ordered outer
results, with the safe-binary owner's five-key dictionary retained inside a
one-slot nested tuple. Selected observation-only targets are
`_layout_pass_set_1_quantized_activation_binary_results` and
`_layout_pass_set_2_quantized_activation_binary_results`. Exact child schemas,
shared context routing, cleanup-only zero-counter mutation, both distinct
boundaries, and the unconsumed policy are frozen before implementation.
The focused characterization gate passes
`665 passed, 1 xfailed in 18.31s`; the sole strict xfail covers nested runner,
helper, and two-call result propagation.

Both phase runners and the lowerer helper now return the existing nested tuple,
and the two calls retain
`_layout_pass_set_1_quantized_activation_binary_results` and
`_layout_pass_set_2_quantized_activation_binary_results`. Both remain
observation-only. The focused implementation gate passes
`666 passed in 20.24s`, the branch-changed broad suite passes
`1570 passed in 28.87s`, and targeted static validation passes.

The next unit should audit the two independent direct safe-binary helper calls,
whose phase runner now returns a one-slot tuple while the lowerer helper still
discards it.

Those direct calls are now characterized with distinct observation-only
targets `_layout_pass_set_1_safe_binary_results` and
`_layout_pass_set_1_final_safe_binary_results`. Transparent helper return,
one-slot/five-key schema, unconditional cleanup, both exact boundary pairs, and
the unconsumed policy are frozen before implementation. The corrected focused
gate passes `410 passed, 1 xfailed in 18.93s`; the strict xfail is the selected
two-call retention contract. The branch-changed broad characterization gate
passes `1570 passed, 1 xfailed in 29.04s`.

The safe-binary lowerer helper now returns its unchanged one-slot tuple, and
the two direct calls retain `_layout_pass_set_1_safe_binary_results` and
`_layout_pass_set_1_final_safe_binary_results`. Both remain observation-only.
The focused implementation gate passes `411 passed in 20.67s`, the
branch-changed broad suite passes `1571 passed in 29.20s`, and targeted static
validation passes.

The next unit should audit both layout-attention/quantized suffix calls and the
ordered runner results they currently discard.

That suffix is now characterized as thirteen ordered results, including three
nested cluster slots. Selected observation-only targets are
`_layout_pass_set_1_attention_quantized_suffix_results` and
`_layout_pass_set_1_final_attention_quantized_suffix_results`. Exact pass-ID
order, instrumented result identity, shared context and boolean policy routing,
both distinct boundary pairs, and the unconsumed policy are frozen before
implementation. The focused gate passes
`726 passed, 1 xfailed in 19.28s`; the strict xfail is the two-result
propagation contract.

The suffix runner and helper now return the unchanged thirteen-slot tuple, and
the two calls retain
`_layout_pass_set_1_attention_quantized_suffix_results` and
`_layout_pass_set_1_final_attention_quantized_suffix_results`. Both remain
observation-only. The focused implementation gate passes
`736 passed in 20.88s`, the branch-changed broad suite passes
`1588 passed in 28.15s`, and targeted static validation passes.

The next unit should audit the post-QDQ direct Transpose/unary-fanout cluster
result and inventory the helper's other occurrences before changing any call.

That helper is now characterized as two three-slot policy variants: the sole
direct post-QDQ result and the attention-gate/QDQ callback result. Selected
direct target `_layout_pass_set_1_transpose_unary_fanout_results` remains
observation-only. Exact active pass-ID orders, result identities, shared scope,
policy arguments, callback identity, final-suffix/final-safe-binary boundaries,
and the unconsumed policy are frozen before implementation. The focused gate
passes `370 passed, 1 xfailed in 18.07s`; the strict xfail is transparent
runner/helper/direct propagation.

The runner and helper now return the active three-dictionary tuple. The direct
post-QDQ call retains `_layout_pass_set_1_transpose_unary_fanout_results`; the
attention-gate/QDQ callback now contributes its default-policy tuple to the
parent's still-discarded result. The focused implementation gate passes
`371 passed in 20.83s`, the branch-changed broad suite passes
`1589 passed in 28.35s`, and targeted static validation passes.

The next unit should audit every direct attention-gate/QDQ parent result now
that its Transpose/unary-fanout callback slot is populated.

That ten-slot parent is now characterized with the nested three-dictionary
unary-fanout tuple at slot five. Selected observation-only direct targets are
`_layout_pass_set_1_attention_gate_qdq_results` and
`_layout_pass_set_2_attention_gate_qdq_results`. Exact instrumented result
identity, child order, shared context, two direct boundary pairs, suffix
nesting, and the unconsumed policy are frozen before implementation. The
focused gate passes `571 passed, 1 xfailed in 18.81s`; the strict xfail is
parent runner/helper/two-call propagation.

The parent runner/helper now return the unchanged ten-slot tuple. Direct calls
retain `_layout_pass_set_1_attention_gate_qdq_results` and
`_layout_pass_set_2_attention_gate_qdq_results`; both retained suffix tuples
also receive the nested parent result. All remain observation-only. The focused
implementation gate passes `572 passed in 20.79s`, the branch-changed broad
suite passes `1590 passed in 28.69s`, and targeted static validation passes.

The next unit should audit both direct pre-add/mean/attention recovery results
and preserve the new pass-set-2 attention result immediately following the
second call.

That seven-slot parent is now characterized with its nested mean-attention
result. Selected observation-only targets are
`_layout_pass_set_2_preadd_mean_attention_results` and
`_layout_opt_preadd_mean_attention_results`. Exact instrumented result
identity, shared context, zero-argument calls, layout-prefix/attention and
channel-shuffle/SA-PA boundaries, and the unconsumed policy are frozen before
implementation. The focused gate passes
`355 passed, 1 xfailed in 18.12s`; the strict xfail is runner/helper/two-call
propagation.

The parent runner/helper now return the unchanged seven-slot tuple. The two
direct calls retain `_layout_pass_set_2_preadd_mean_attention_results` and
`_layout_opt_preadd_mean_attention_results`; both remain observation-only. The
focused implementation gate passes `356 passed in 20.18s`, the branch-changed
broad suite passes `1591 passed in 29.45s`, and targeted static validation
passes.

The next unit should audit the layout-recovery prefix result immediately before
the first retained pre-add parent result and inventory every other occurrence.
Mean/attention tuples and the preceding BatchMatMul results must remain
observation-only and policy guarded. The retained
`_terminal_normalization_pad_stats` also remains observation-only because it
omits cleanup-only pruning. Any change must preserve current pass order,
TensorFlow-free boundary, dependency set, and sequential validation policy.

That layout-recovery prefix is now characterized as nineteen ordered results,
including the three heterogeneous lowerer callback slots. The sole direct
target is `_layout_pass_set_2_layout_recovery_prefix_results`; the same runner
also remains the first nested attention-prefix callback. Exact result identity,
shared context, QLinear/pre-add boundary, nested selection, and the unconsumed
observation-only policy are frozen before implementation. The focused gate
passes `414 passed, 1 xfailed in 18.88s`; the strict xfail is the transparent
runner/helper/direct result propagation contract. The branch-changed broad
characterization gate passes `1597 passed, 1 xfailed in 28.73s`.

The layout-recovery runner/helper now return the unchanged nineteen-slot tuple,
and the sole direct call retains
`_layout_pass_set_2_layout_recovery_prefix_results`. It remains unconsumed and
observation-only. The first nested attention callback receives the same tuple,
while the attention parent still discards its aggregate result. The focused
implementation gate passes `422 passed in 21.02s`, the branch-changed broad
suite passes `1605 passed in 30.11s`, and targeted static validation passes.

The next unit should audit all three direct layout/reshape/attention prefix
results, including the newly populated nested layout-recovery slot, without
changing their distinct surrounding pass boundaries or consuming incomplete
child summaries.

That fifteen-slot parent is now characterized with the complete nested
nineteen-slot layout-recovery result at slot zero. Selected observation-only
targets are `_layout_pass_set_1_initial_attention_recovery_results`,
`_layout_pass_set_1_post_binary_attention_recovery_results`, and
`_layout_pass_set_1_final_attention_recovery_results`. Exact result identity,
shared context, three zero-argument calls, layout-cleanup/affine,
duplicate-fanout/affine, and QLinear/InstanceNorm boundaries, and the
unconsumed policy are frozen. The focused gate passes
`303 passed, 1 xfailed in 18.24s`; the branch-changed broad gate passes
`1605 passed, 1 xfailed in 29.09s`.

The parent runner/helper now return the unchanged fifteen-slot tuple. The three
direct calls retain `_layout_pass_set_1_initial_attention_recovery_results`,
`_layout_pass_set_1_post_binary_attention_recovery_results`, and
`_layout_pass_set_1_final_attention_recovery_results`, all observation-only
and unconsumed. The nested nineteen-slot layout tuple remains intact. The
focused implementation gate passes `304 passed in 19.89s`, the branch-changed
broad suite passes `1606 passed in 28.89s`, and targeted static validation
passes.

The next unit should audit both direct QLinear/mean/Concat recovery results,
preserving the newly retained final attention result that follows the second
call and both distinct production boundaries.

That five-slot parent is now characterized with two observation-only targets:
`_layout_pass_set_1_qlinear_mean_concat_results` and
`_layout_pass_set_2_qlinear_mean_concat_results`. Exact result identity,
shared context, zero-argument calls, dequant-mean/final-attention and
progress/layout-recovery boundaries, and the unconsumed policy are frozen.
The focused gate passes `424 passed, 1 xfailed in 18.09s`; the branch-changed
broad gate passes `1606 passed, 1 xfailed in 28.82s`.

The QLinear runner/helper now return the unchanged five-slot tuple. Its two
direct calls retain `_layout_pass_set_1_qlinear_mean_concat_results` and
`_layout_pass_set_2_qlinear_mean_concat_results`, both unconsumed and
observation-only. The focused implementation gate passes
`425 passed in 19.97s`, the branch-changed broad suite passes
`1607 passed in 29.54s`, and targeted static validation passes.

The next unit should audit the raw primary layout-transpose cleanup result
immediately before the newly retained initial attention result, without
changing its other retained/nested occurrences or live layout/diagnostic
arguments.

The primary layout-transpose result is now characterized separately from the
already retained late-ConCat and very-late calls and the independent
late-binary nested call. Selected observation-only target is
`_layout_pass_set_1_layout_transpose_cleanup_stats`. The fixed five-key schema,
cleanup/LayoutState sync semantics, exact occurrence counts, live arguments,
shared-scope difference, duplicate-fanout-policy/initial-attention boundary,
and unconsumed policy are frozen. The focused gate passes
`370 passed, 1 xfailed in 19.20s`; the branch-changed broad gate passes
`1608 passed, 1 xfailed in 29.60s`.

The primary call now retains
`_layout_pass_set_1_layout_transpose_cleanup_stats`, making all three direct
lowerer occurrences explicit while leaving the late-binary nested consumer
unchanged. The result remains observation-only. The focused implementation
gate passes `371 passed in 20.03s`, the branch-changed broad suite passes
`1609 passed in 29.42s`, and targeted static validation passes.

The next unit should audit the sole raw duplicate-fanout cleanup result before
the newly retained post-binary attention result, preserving its QDQ-dependent
transpose policy and live layout/diagnostic context.

The duplicate-fanout owner is now characterized with its reshape-only one-key
and Transpose-enabled two-key schemas, two transactional pass IDs, three
independent orchestration selections, and the sole direct QDQ-dependent call.
Selected observation-only target is
`_layout_pass_set_1_duplicate_fanout_stats`. Exact policy/context arguments,
conditional binary-bridge/post-binary-attention boundary, and unconsumed
contract are frozen. The focused gate passes
`378 passed, 1 xfailed in 18.47s`; the branch-changed broad gate passes
`1610 passed, 1 xfailed in 28.74s`.

The sole direct call now retains
`_layout_pass_set_1_duplicate_fanout_stats`, which remains unconsumed and
observation-only. The QDQ-derived schema/policy and all nested selections are
unchanged. The focused implementation gate passes
`379 passed in 18.50s`, the branch-changed broad suite passes
`1611 passed in 29.45s`, and targeted static validation passes.

The next unit should audit the raw dequantize/mean/quantize bridge result before
the newly retained pass-set-1 QLinear parent result.

That sole bridge result is now characterized as the one-key
`moved_transpose_dequantize_mean_quantize_bridges` dictionary. Selected
observation-only target is `_layout_pass_set_1_dequant_mean_quantize_stats`.
Wrapper forwarding, early and final prune paths, sole exact call,
safe-binary/QLinear boundaries, and unconsumed policy are frozen. The focused
gate passes `357 passed, 1 xfailed in 17.88s`; the branch-changed broad gate
passes `1612 passed, 1 xfailed in 29.37s`.

The sole call now retains `_layout_pass_set_1_dequant_mean_quantize_stats`,
which remains unconsumed and observation-only because its counter omits
cleanup-only pruning. The focused implementation gate passes
`358 passed in 18.10s`, the branch-changed broad suite passes
`1613 passed in 29.32s`, and targeted static validation passes.

The next unit should audit both InstanceNorm pre/post NHWC recovery occurrences
and their distinct direct/conditional boundaries.

The InstanceNorm compatibility dispatcher is now characterized as four
graph-ordered indexed owners, one result counter, and a 32-rewrite cap. Its
later two-iteration convergence form remains an existing counter consumer;
only the raw pass-set-1 result is selected as
`_layout_pass_set_1_instancenorm_prepost_stats`. Exact arguments, call count,
loop `.get()` expression, final-attention/squeeze-cleanup boundary, and
unconsumed direct policy are frozen. The focused gate passes
`536 passed, 1 xfailed in 18.21s`; the branch-changed broad gate passes
`1614 passed, 1 xfailed in 29.38s`.

The direct pass-set-1 call now retains
`_layout_pass_set_1_instancenorm_prepost_stats`; the later convergence consumer
is unchanged. One stale raw-successor architecture assertion was updated after
the initial `536 passed, 1 failed` gate. The corrected focused gate passes
`537 passed in 20.50s`, the branch-changed broad suite passes
`1615 passed in 28.89s`, and targeted static validation passes.

The next unit should audit the direct squeeze/unary/Reshape identity cleanup
result and all of that runner's other policy-selected occurrences before
changing its boundary.

That cleanup runner is now characterized with an identity-only one-key schema,
a unary-enabled two-key schema, three direct unary-enabled calls, one nested
attention unary-enabled selection, and one nested Singleton identity-only
selection. Selected observation-only targets are
`_layout_pass_set_1_squeeze_reshape_identity_stats`,
`_core_cleanup_squeeze_reshape_identity_stats`, and
`_layout_pass_set_2_squeeze_reshape_identity_stats`. Exact direct arguments,
all three boundary pairs, nested policies, and unconsumed results are frozen.
The focused gate passes `366 passed, 1 xfailed in 18.27s`; the branch-changed
broad gate passes `1616 passed, 1 xfailed in 29.63s`.

All three direct calls now retain their unchanged two-key dictionaries in the
selected observation-only targets. The initial focused implementation gate
reported `366 passed, 1 failed` because the final-attention suffix contract
still required its cleanup predecessor to be a raw expression. That contract
now accepts either a direct assignment or expression while preserving the
exact call identity and boundary. The corrected focused gate passes
`367 passed in 18.11s`, the branch-changed broad gate passes
`1617 passed in 29.27s`, and targeted static validation passes.

The quantized-PReLU cleanup is now characterized as a fixed four-key result,
four transactional default passes in priorities 10 through 40, one direct
lowerer call, and one nested duplicate-fanout orchestration selection with a
shared pass-state scope. The selected observation-only direct target is
`_layout_pass_set_1_quantized_prelu_stats`; exact arguments, the attention-
gate/dequant-TransposeConv boundary, the nested invocation, and absence of a
consumer are frozen. The focused gate passes
`325 passed, 1 xfailed in 17.98s`; the branch-changed broad gate passes
`1618 passed, 1 xfailed in 29.14s`.

The direct quantized-PReLU call now retains its unchanged four-key dictionary
as `_layout_pass_set_1_quantized_prelu_stats`. It remains unconsumed and
observation-only; the nested duplicate-fanout selection and shared state scope
are unchanged. The focused gate passes `326 passed in 17.98s`, the branch-
changed broad gate passes `1619 passed in 29.96s`, and targeted static
validation passes.

The quantized-Reshape cleanup is now characterized as one transactional pass
with a fixed `folded_dequant_reshape_quantize_chains` result, one direct
lowerer call, and one nested quantized-suffix selection. The selected
observation-only direct target is
`_layout_pass_set_1_quantized_reshape_stats`; exact arguments, the dequant-
TransposeConv/quantized-activation boundary, nested layout/diagnostics routing,
and absence of a consumer are frozen. The focused gate passes
`314 passed, 1 xfailed in 18.59s`; the branch-changed broad gate passes
`1620 passed, 1 xfailed in 29.58s`.

The direct quantized-Reshape call now retains its unchanged one-key dictionary
as `_layout_pass_set_1_quantized_reshape_stats`. The initial focused gate
reported `314 passed, 1 failed` because an activation-recovery architecture
contract still required its predecessor to be a raw expression. That contract
now accepts either direct-call statement form while preserving the exact call
identity. The corrected focused gate passes `315 passed in 20.48s`, the branch-
changed broad gate passes `1621 passed in 30.14s`, and targeted static
validation passes.

The dequantize/TransposeConv/quantize cleanup is now characterized as a fixed
one-key result, two direct lowerer calls, and one nested quantized-suffix
selection. Selected observation-only targets are
`_layout_pass_set_1_dequant_transposeconv_quantize_stats` and
`_layout_pass_set_2_dequant_transposeconv_quantize_stats`. Exact arguments,
both retained-neighbor boundaries, the nested layout route, and absence of
consumers are frozen. Because candidate-missing exits can still prune unused
tensors while returning zero, the counter is not complete mutation evidence.
The focused gate passes `386 passed, 1 xfailed in 18.24s`; the branch-changed
broad gate passes `1622 passed, 1 xfailed in 30.56s`.

Both direct dequant-TransposeConv calls now retain their unchanged one-key
dictionaries in the selected observation-only targets. The nested suffix call
and its layout route are unchanged, and zero counters still must not be treated
as proof of no mutation. The focused gate passes `387 passed in 18.36s`, the
branch-changed broad gate passes `1623 passed in 29.53s`, and targeted static
validation passes.

The floating-point MUL/ADD/MUL affine-chain fold is now characterized as a
fixed one-key result, a default 32-rewrite indexed owner, two direct lowerer
calls, and one terminal-orchestration selection. The nested result remains
consumed by the existing terminal mutation summary; only the direct results
are selected as observation-only targets
`_layout_pass_set_1_initial_affine_chain_fold_stats` and
`_layout_pass_set_1_post_binary_affine_chain_fold_stats`. Exact arguments,
both attention-adjacent boundaries, nested layout routing, and absence of
direct consumers are frozen. The focused gate passes
`409 passed, 1 xfailed in 19.22s`; the branch-changed broad gate passes
`1624 passed, 1 xfailed in 31.32s`.

Both direct affine-chain fold calls now retain their unchanged one-key
dictionaries in the selected observation-only targets. The terminal nested
result and mutation summary are unchanged. The initial focused implementation
gate reported `409 passed, 1 failed` because the attention-prefix architecture
contract still expected no successor targets. Its corrected target contract
passes `410 passed in 20.86s`; the branch-changed broad gate passes
`1625 passed in 30.08s`, and targeted static validation passes.

The Transpose/MUL/ADD/Transpose affine pre/post owner is now characterized as
a fixed one-key result with a default 32-rewrite cap, three lowerer calls, three
declarative orchestration selections, and one consumed late-binary composite
call. The final lowerer call already retains
`_no_layout_final_affine_prepost_stats`; selected targets for the two raw forms
are `_layout_pass_set_1_affine_prepost_stats` and
`_no_layout_fallback_affine_prepost_stats`. Exact arguments, conditional and
final boundaries, nested layout routes, and consumed late-binary form are
frozen. Unconditional pruning means a zero counter is not complete mutation
evidence. The focused gate passes `495 passed, 1 xfailed in 19.89s`; the
branch-changed broad gate passes `1626 passed, 1 xfailed in 30.38s`.

The initial and no-layout fallback affine pre/post calls now retain their
unchanged dictionaries in the selected observation-only targets. The existing
final target, three declarative selections, and consumed late-binary form are
unchanged; zero counters remain insufficient to prove no mutation. The focused
gate passes `496 passed in 20.88s`, the branch-changed broad gate passes
`1627 passed in 30.99s`, and targeted static validation passes.

The pre-unary/MUL-ADD/transpose fan-out owner is now characterized as a fixed
one-key model-only result, one direct lowerer call, and two attention-related
declarative selections. The selected observation-only target is
`_layout_pass_set_1_pre_unary_affine_fanout_stats`; exact model-only arguments,
the affine-pre/post/mean-affine boundary, both nested indices, and absence of a
consumer are frozen. Unconditional final pruning means a zero counter is not
complete mutation evidence. The focused gate passes
`323 passed, 1 xfailed in 18.08s`; the branch-changed broad gate passes
`1628 passed, 1 xfailed in 30.84s`.

The direct pre-unary affine fan-out call now retains its unchanged one-key
dictionary as `_layout_pass_set_1_pre_unary_affine_fanout_stats`. It remains
unconsumed and observation-only; both model-only attention selections and the
owner's pruning behavior are unchanged. The focused gate passes
`324 passed in 18.41s`, the branch-changed broad gate passes
`1629 passed in 30.91s`, and targeted static validation passes.

The mean/MUL-ADD affine pre/post owner is now characterized as a fixed one-key
model-only result, one direct lowerer call, and two attention-related
declarative selections. The selected observation-only target is
`_layout_pass_set_1_mean_affine_prepost_stats`; exact model-only arguments, the
pre-unary-fan-out/mean-attention boundary, both nested indices, and absence of a
consumer are frozen. Unconditional final pruning means a zero counter is not
complete mutation evidence. The focused gate passes
`328 passed, 1 xfailed in 18.26s`; the branch-changed broad gate passes
`1630 passed, 1 xfailed in 30.30s`.

The direct mean-affine pre/post call now retains its unchanged one-key
dictionary as `_layout_pass_set_1_mean_affine_prepost_stats`. It remains
unconsumed and observation-only; both model-only attention selections and the
owner's pruning behavior are unchanged. The focused gate passes
`329 passed in 18.61s`, the branch-changed broad gate passes
`1631 passed in 30.12s`, and targeted static validation passes.
