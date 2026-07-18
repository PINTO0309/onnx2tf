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

Focused Ruff, Python bytecode compilation, and `git diff --check` also pass.
These results are contract and orchestration tests; they do not claim a new
full model-corpus run for this observation and accounting unit.

## Remaining work

The broader `flatbuffer_direct` refactor remains active. The next characterized
unit should audit the immediately following window-partition
Reshape/Transpose-to-SpaceToDepth result, its live LayoutState contract, and its
preprojection/window-reverse boundaries. Any new mutation evidence must
preserve current pass order, TensorFlow-free boundary, dependency set, and
sequential validation policy.
