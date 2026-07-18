# `fb-refactor7` handoff

## Current state

- Branch: `fb-refactor7`.
- Starting checkpoint: `d3369d9a`, the merge of the completed
  `fb-refactor6` work into `main`.
- Pull requests are outside the continuation workflow. Continue with coherent
  commits and pushes to `origin/fb-refactor7` only.
- Inference remains strictly sequential. No model-inference ProcessPool or
  parallel worker is permitted.

## Diagnostic-numbering characterization checkpoint

The first `fb-refactor7` unit inventories a non-context core efficiency issue
without changing production code. `run_model_ir_pass_group()` calculates the
next diagnostic group with one full history scan, but `_record_event()` then
rescans the complete and growing history for every result to derive global
`sequence` and per-pass `invocation`. A group with `P` results therefore uses
`P + 1` diagnostic-list iterations and accumulates quadratic bookkeeping cost
across a conversion with many pass events.

The new semantic fixture freezes all externally visible numbering rules:

- non-`model_ir_pass` records are preserved and excluded from numbering;
- the next `group_sequence` is one greater than the maximum existing group;
- `sequence` counts all existing ModelIR-pass records, independent of stored
  legacy sequence values;
- `invocation` counts only records with the same pass ID;
- every event produced by one group receives the same group number.

A strict expected-failure efficiency fixture supplies a list subclass that
counts history iterations. The current implementation performs four scans for
one three-result group; the required implementation performs exactly one. The
focused characterization result is `2 passed, 1 xfailed`.

No production source, public API, artifact, pass order, graph index, layout,
diagnostic schema, dependency, TensorFlow boundary, corpus policy, or model
conversion changed in this checkpoint.

## Next action

In `run_model_ir_pass_group()`, scan existing diagnostics once to initialize:

1. the existing ModelIR-pass event count;
2. the maximum group sequence;
3. per-pass invocation counts.

Update those counters locally as events are appended. Preserve exact behavior
for ordinary, skipped, cycle-stopped, and invariant-failure events. Remove the
strict xfail only after the efficiency fixture passes, then run the complete
core, pass-efficiency, architecture, and TensorFlow-import-blocked gates. Keep
real-model conversion minimal because this unit cannot affect ModelIR or
artifacts. Commit and push only; do not create or update a pull request.

## Diagnostic-numbering implementation checkpoint

`run_model_ir_pass_group()` now scans an existing diagnostics list exactly
once. That scan initializes the total ModelIR-pass event count, maximum group
sequence, and a per-pass invocation-count map. `_record_event()` increments
those local counters while appending, rather than iterating over the complete
and growing history again for every pass result.

The implementation preserves the characterized behavior for unrelated
diagnostics, missing or negative legacy group values, repeated pass IDs,
multiple results in one group, skipped passes, cycles, and invariant failures.
It does not change pass execution, ModelIR, LayoutState, GraphIndex, result
details, diagnostic fields, summary schema, or the identity of the caller's
list.

Sequential validation completed as follows:

- focused numbering and pass-group contracts: `7 passed`;
- complete core, pass-efficiency, architecture, and TensorFlow-import-blocked
  gate: `324 passed in 27.74s`;
- the scan-count fixture changed from one strict xfail to pass and observes one
  history iteration for a three-result group;
- focused Ruff, Python compilation, and whitespace checks: passed.

No real-model conversion was run because this unit cannot change ModelIR or an
artifact. No public API, CLI behavior, dependency, pass order, corpus policy,
inference concurrency, or TensorFlow boundary changed.

At resume, return to the remaining direct lowerer-to-pass and core contract
inventory. Select another non-context unit only when it has an explicit
ownership or measurable scan/allocation cost, and characterize its behavior
before production changes. Continue with coherent commits and pushes only; do
not create or update a pull request.

## Cross-group diagnostic-ledger characterization checkpoint

The first implementation removes repeated history scans inside one pass group,
but an ordinary diagnostics list must still be scanned once at the start of
every group. That safe fallback is required for arbitrary caller-owned lists,
which may have changed between calls. Production is different:
`ConversionSession` owns one diagnostics object for the complete conversion and
only appends through the pass runner or `record_diagnostic()`.

A new strict expected-failure contract requires the Session-owned object to
retain its ModelIR-pass event count, maximum group, and per-pass invocation
counts across group calls. The fixture deliberately performs slice assignment
first, proving that arbitrary list mutation invalidates the ledger and causes
exactly one lazy rebuild. Two subsequent groups must reuse the rebuilt state
while preserving sequences `2, 3`, invocations `2, 3`, and groups `5, 6`.

No production source changed. At implementation, introduce a private
list-compatible diagnostic ledger owned only by `ConversionSession`. Normal
append/extend/insert operations may update valid counters incrementally;
replacement or removal operations must invalidate them. Arbitrary external
lists passed to `run_model_ir_pass_group()` must retain the existing one-scan
fallback. Do not expose the internal type through the public core API.

## Cross-group diagnostic-ledger implementation checkpoint

`ConversionSession.diagnostics` now defaults to the private list-compatible
`ModelIRPassDiagnostics` type. It maintains the ModelIR-pass event count,
maximum group sequence, and pass-ID invocation counts incrementally for append-
only production use. `run_model_ir_pass_group()` reads a copied numbering
snapshot from this ledger in constant time; arbitrary external lists continue
to use the characterized one-scan fallback.

List compatibility is explicit. Append, extend, and insert update valid state.
Slice/item replacement, deletion, multiplication, pop, and remove invalidate
state without changing the list operation; the next group lazily rebuilds it
once. Clear resets an empty valid ledger. Malformed externally appended entries
invalidate rather than raising early, preserving error timing until the next
numbering read. The type is deliberately absent from `core.__all__`.

Sequential validation completed as follows:

- focused diagnostic/pass-group contracts, including zero rebuilds for the
  ordinary Session append path and one rebuild after slice replacement: pass;
- complete core, pass-efficiency, architecture, and TensorFlow-import-blocked
  gate: `328 passed in 26.75s`;
- focused Ruff, Python compilation, and whitespace checks: passed.

No model conversion ran because pass execution and ModelIR are unaffected. No
public API, CLI behavior, artifact, dependency, pass ID/order, GraphIndex,
LayoutState, corpus policy, inference concurrency, or TensorFlow boundary
changed.

At resume, move beyond diagnostic bookkeeping and inventory the remaining
direct lowerer-to-pass/core boundaries again. Select a non-context unit with a
measurable graph scan, state allocation, or duplicated ownership contract;
characterize it before changing production. Commit and push coherent units
only, and do not create or update a pull request.

## Absolute-final SINet reconciliation characterization checkpoint

The next non-context graph-scan cost is the terminal sequence of six SINet
owners:

- late residual Add/Mul/Add/PReLU;
- deep-skip pre-Add/Concat/PReLU fan-out;
- deep-skip dual-Resize affine recovery;
- shared-post PReLU fan-out;
- deep-skip Concat/Resize affine tail;
- final Concat/Resize affine bridge.

Each owner is already bounded and transactional, returns one exact rewrite
counter, and has positive/no-op/idempotence coverage. The lowerer nevertheless
runs `_reconcile_static_tensor_shapes()` unconditionally after every owner.
Thus the zero-owner path performs six unnecessary full reconciliation scans.

A strict expected-failure architecture fixture freezes the exact owner order,
counter key, immediate guard, and single reconciliation body for all six
owners. Production is unchanged in this checkpoint. At implementation, assign
each existing result and run its existing immediate reconciliation only when
the corresponding counter is greater than zero. Do not merge, reorder, skip,
or generalize any SINet pass, and do not alter LayoutState handling.

Characterization validation is `514 passed` across the six complete owner
suites, plus one strict architecture xfail for the currently unconditional
reconciliations. Focused Ruff and whitespace checks pass.

## Absolute-final SINet reconciliation implementation checkpoint

All six owners remain in their original terminal order and execute with the
same main ModelIR and Session LayoutState. Each return value is now assigned,
and only a positive value under its established mutation counter triggers the
existing immediately following `_reconcile_static_tensor_shapes()` call. No
owner was merged, skipped, generalized, or moved.

The architecture xfail is green. A synthetic Add lowerer fixture replaces the
six owners with controlled counters: the all-zero case establishes the base
reconciliation count, and each of the six one-changed cases adds exactly one
call. The fixture confirms both zero-owner scan elimination and positive-owner
maintenance wiring without requiring real-model inference.

Sequential validation completed as follows:

- six complete SINet owner suites, core, pass efficiency, architecture, and
  TensorFlow-import-blocked optional boundary: `844 passed in 29.04s`;
- focused runtime counter wiring: passed for the all-zero baseline and all six
  individual changed counters;
- focused Ruff, Python compilation, and whitespace checks: passed, excluding
  only the lowerer's two pre-existing F401 findings.

No public API, CLI behavior, artifact, dependency, pass call/order, LayoutState,
GraphIndex, corpus exclusion, operation-count tier, inference concurrency, or
TensorFlow boundary changed. No real-model conversion ran because the 514
positive/no-op owner fixtures plus lowerer wiring test cover this maintenance-
scan-only change.

At resume, inventory the other unconditional static-shape reconciliations near
this terminal block. Do not assume their runners expose complete mutation
counters: characterize each return contract and positive/no-op coverage before
guarding another scan. Commit and push only; do not create or update a pull
request.

## NodeView value-list typing checkpoint

`NodeView.inputs` and `NodeView.outputs` now explicitly carry
`list[_ValueView]` annotations. This resolves the editor-side protobuf type
misinference that reported `.name` as unavailable at the demand-driven input
rank recovery in `lower_from_onnx2tf.py`. The produced lists and their values
are unchanged at runtime.

Sequential validation covered the complete NodeView, flatbuffer-direct core,
and TensorFlow-import-blocked optional-boundary suites: `52 passed`. Focused
Ruff, Python compilation, and whitespace checks also pass. Continue with the
remaining unconditional static-shape reconciliation inventory; commit and push
only, with no pull request.

## Mixed-singleton Concat reconciliation characterization checkpoint

The first safe candidate in the remaining terminal inventory is
`_repair_mixed_singleton_nchw_inputs_for_nhwc_concat()`. Its indexed owner
returns one exact mutation counter and performs no mutation when that counter is
zero. Existing tests cover positive rewrites, complete no-op preservation,
maintained GraphIndex/LayoutState, fan-out, name allocation, and the fast path
without a Concat. The immediately following shape reconciliation is still
unconditional in production.

A strict expected-failure architecture contract now requires that exact call to
be assigned and its immediate reconciliation to be guarded by
`repaired_mixed_singleton_nchw_inputs_for_nhwc_concat > 0`. This checkpoint does
not change production. Do not apply the same assumption to the nearby PReLU
owner, which intentionally prunes unused tensors even when its rewrite count is
zero, or to the SE/FC/Gather cluster, whose wrapper currently returns no
aggregate mutation result.

At implementation, change only this one call site, add lowerer-level zero/one
counter wiring coverage, and preserve its order and Session LayoutState. Run the
complete mixed-singleton owner, core, architecture, pass-efficiency, and
TensorFlow-import-blocked gates sequentially. Commit and push only; do not
create or update a pull request.

## Mixed-singleton Concat reconciliation implementation checkpoint

The terminal mixed-singleton Concat owner still runs exactly once in its
original position with the Session LayoutState. Its result is now assigned, and
the immediately following `_reconcile_static_tensor_shapes()` runs only when
`repaired_mixed_singleton_nchw_inputs_for_nhwc_concat` is positive. No matcher,
rewrite, GraphIndex update, layout update, or neighboring owner changed.

The strict architecture expectation is green. A synthetic lowerer fixture
proves that the positive counter adds exactly one reconciliation relative to
the zero-counter path. Sequential validation across the complete owner, core,
pass-efficiency, architecture, and TensorFlow-import-blocked optional-boundary
suites is `363 passed in 25.91s`. Focused owner coverage is `31 passed`.

At resume, the absolute-final consecutive-Reshape runner is the next possible
complete-counter candidate: it exposes three mutation counters and mutates only
when their sum is positive. Characterize the exact terminal call before
changing it. Do not guard the PReLU or SE/FC/Gather boundaries without first
making their mutation accounting complete. Commit and push only; do not create
or update a pull request.

## Consecutive-Reshape reconciliation characterization checkpoint

The absolute-final `run_consecutive_reshape_cleanup()` is now characterized as
the next complete-counter boundary. Its three results cover no-op removal,
consecutive-chain rewrites, and fan-out bypass rewrites. The fan-out path also
increments the aggregate rewrite counter, and cleanup/synchronization only run
after a positive mutation. A new no-candidate fixture freezes the exact
all-zero result, byte-for-byte-equivalent ModelIR representation, skipped
diagnostic, and no pass-state construction.

Production remains unchanged. A strict expected-failure architecture contract
requires the terminal runner result to be assigned and its immediately
following shape reconciliation to be guarded by all three exact counters. At
implementation, keep the runner in place with the same LayoutState and
diagnostics. Add zero-counter and one-counter-per-key lowerer wiring coverage,
then run the consecutive-Reshape, core, pass-efficiency, architecture, and
TensorFlow-import-blocked suites sequentially. Commit and push only; do not
create or update a pull request.

## Consecutive-Reshape reconciliation implementation checkpoint

The absolute-final `run_consecutive_reshape_cleanup()` remains in place with
the same Session LayoutState and diagnostic ledger. Its result is now assigned,
and the immediately following static-shape reconciliation runs only when the
sum of its three exact mutation counters is positive. No reshape matching,
rewriting, pruning, diagnostics, ordering, or neighboring pass changed.

The strict architecture expectation is green. A synthetic lowerer fixture
establishes the all-zero baseline and proves that each individual positive
counter adds exactly one reconciliation. Sequential validation across the
complete consecutive-Reshape, core, pass-efficiency, architecture, and
TensorFlow-import-blocked suites is `337 passed in 25.69s`; the focused runner
suite is `3 passed`.

At resume, leave the absolute-final PReLU reconciliation unchanged until its
zero-rewrite tensor-pruning side effect has explicit mutation accounting. The
SE/FC/Gather cluster likewise needs an aggregate result contract before its
reconciliation can be guarded. Inventory another bounded scan or allocation
with a complete owner contract, characterize it first, then commit and push
only. Do not create or update a pull request.

## Absolute-final PReLU reconciliation characterization checkpoint

The absolute-final PReLU owner has one rewrite counter but also intentionally
calls `_prune_unused_tensors()` for zero-match invocations. A new owner fixture
uses `max_rewrites=0` with one unused tensor and freezes the complete behavior:
the rewrite counter remains zero, the tensor count decreases by one, and the
LayoutState stays synchronized.

A strict expected-failure architecture contract therefore does not permit a
rewrite-only guard. It requires the terminal lowerer to record the tensor-table
size immediately before the owner and reconcile when either the exact rewrite
counter is positive or the tensor count decreased. This preserves the existing
zero-rewrite cleanup contract without paying for reconciliation when neither a
rewrite nor pruning occurred. Production is unchanged at this checkpoint.

At implementation, change only the absolute-final call site, retain the owner
call and Session LayoutState, and add lowerer wiring coverage for zero-change,
rewrite, and prune-only outcomes. Run the complete PReLU owner, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites
sequentially. Commit and push only; do not create or update a pull request.

## Absolute-final PReLU reconciliation implementation checkpoint

The absolute-final PReLU owner remains in its original position with the same
Session LayoutState and its intentional unconditional tensor prune. The lowerer
now records the tensor-table size immediately before the owner, assigns its
result, and runs the existing following reconciliation only after a positive
rewrite counter or a tensor-count decrease.

The strict architecture expectation is green. A synthetic lowerer fixture
separately proves unchanged, rewrite, and prune-only outcomes; both changed
outcomes add exactly one reconciliation over the unchanged baseline. Sequential
validation across the complete PReLU owner, core, pass-efficiency, architecture,
and TensorFlow-import-blocked suites is `365 passed in 26.66s`; focused owner
coverage is `29 passed`.

At resume, the remaining absolute-final SE/FC/Gather reconciliation still has
no aggregate mutation result. Characterize the two-runner orchestration return
contracts and decide whether aggregation can be introduced without changing
their public runner results or diagnostics. If not, leave it unconditional and
inventory another bounded scan/allocation. Commit and push only; do not create
or update a pull request.

## Recovery-result propagation characterization checkpoint

`run_recovery_invocations()` currently validates ordered pass IDs and invokes
each callback but discards its return value. Consequently,
`run_se_fc_gather_channel_fanout()` cannot expose the two existing result
dictionaries needed for a complete terminal reconciliation guard.

Two strict expected-failure contracts now require ordered result tuples from
the generic recovery utility and the SE/FC/Gather wrapper. They preserve the
same callback order, arguments, shared `ModelIRPassStateScope`, diagnostics, ID
drift failure before execution, and exception timing. Production is unchanged.

At implementation, return a tuple from the generic runner and forward a typed
two-dictionary tuple from the SE/FC/Gather wrapper. Existing orchestrators may
continue ignoring results. Do not yet change either main/fallback terminal
reconciliation; first validate all recovery-orchestration and SE/FC/Gather
contracts sequentially. Commit and push only; do not create or update a pull
request.

## Recovery-result propagation implementation checkpoint

`run_recovery_invocations()` now returns the callback values as an ordered
tuple after the same pass-ID validation. Existing orchestrators continue to
ignore it. `run_se_fc_gather_channel_fanout()` returns its two typed result
dictionaries, and the lowerer's private helper forwards them while preserving
the same context, Session diagnostics, and shared `ModelIRPassStateScope`.

Sequential validation across every
`test_flatbuffer_direct_*orchestration.py` file is `292 passed in 4.37s`.
The core, pass-efficiency, architecture, and TensorFlow-import-blocked gate is
`336 passed in 25.72s`. Focused return-order, ID-drift, SE/FC/Gather, and helper
contracts pass.

At resume, characterize the main and fallback terminal reconciliation guards as
one combined ownership boundary: SINet shuffle result, both ordered cluster
results, and tensor-count reduction for zero-rewrite pruning. Do not change the
two call sites until both boundary forms have strict structural and lowerer
wiring coverage. Commit and push only; do not create or update a pull request.

## Terminal SE/FC/Gather reconciliation characterization checkpoint

The main ModelIR and fallback ModelIR have the same terminal boundary: the
SINet shuffle residual owner, the two-runner SE/FC/Gather cluster, then an
unconditional static-shape reconciliation. The SINet owner and both runner
results expose exact rewrite counters, while the two runner implementations may
also prune unused tensors on a zero rewrite count.

A strict expected-failure structural contract now covers both targets. It
requires each boundary to record the tensor count, assign the SINet result,
unpack both ordered cluster results, and guard the immediate reconciliation by
the three exact counters or a tensor-count decrease. Production is unchanged.

At implementation, preserve the same owner order, target LayoutState forms,
diagnostics, and fallback/main separation. Add lowerer-level unchanged,
per-counter, and prune-only wiring coverage before running the complete SINet
shuffle, SE layout, orchestration, core, architecture, pass-efficiency, and
TensorFlow-import-blocked gates sequentially. Commit and push only; do not
create or update a pull request.

## Terminal SE/FC/Gather reconciliation implementation checkpoint

Both main and fallback boundaries now record tensor count, assign the SINet
shuffle result, unpack the ordered SE-FC and Gather results, and guard the
existing immediate reconciliation by the three exact counters or a tensor-count
decrease. The owner order, main Session LayoutState, fallback `None` LayoutState,
diagnostics, and recursive fallback separation are unchanged.

The strict two-boundary architecture contract is green. A synthetic main-path
lowerer fixture covers unchanged, each of the three positive counters, and
prune-only outcomes; every changed outcome adds exactly one reconciliation.
Sequential validation across the complete related owners/orchestrator, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites is
`566 passed in 27.36s`.

At resume, re-inventory the remaining unconditional static-shape
reconciliations outside this absolute-final block. Prefer a boundary with a
complete returned mutation contract; where a runner prunes on zero rewrite,
preserve that behavior with explicit accounting. Characterize before changing
production, then commit and push only. Do not create or update a pull request.

## Late binary-repair reconciliation characterization checkpoint

The late boundary after the first post-repair reconciliation consists of
static shape-signature sanitization, exact rank-four binary adapter repair,
singleton-broadcast adapter repair, and another unconditional reconciliation.
Their exact mutation counters are respectively
`sanitized_static_shape_signature_consistency`,
`inserted_rank4_binary_layout_fix_transpose`, and
`repaired_rank4_binary_singleton_broadcast_layout_mismatch`.

The exact adapter owner also prunes unused tensors even when its rewrite count
is zero. A new owner fixture freezes that behavior. A strict expected-failure
architecture contract therefore requires a pre-boundary tensor count, all
three assigned results, and an immediate guard covering every counter plus a
tensor-count decrease. Production is unchanged.

At implementation, change only this second late repair boundary. Add lowerer
wiring coverage for unchanged, all three positive counters, and prune-only
outcomes. Run the complete binary-adapter, static-signature, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites
sequentially. Commit and push only; do not create or update a pull request.

## Late binary-repair reconciliation implementation checkpoint

The second late repair boundary now records tensor count and assigns the static
signature, exact binary adapter, and singleton adapter results. Its immediate
reconciliation runs only after one of the three exact mutation counters is
positive or the exact adapter's zero-rewrite prune reduced the tensor table.
The first repair boundary and final boundary-signature sanitizer are unchanged.

The strict architecture expectation is green. A synthetic lowerer fixture
covers unchanged, all three individual positive counters, and prune-only
outcomes; every changed outcome adds exactly one reconciliation. Sequential
validation across the complete binary-adapter, static-signature, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites is
`402 passed in 26.49s`; focused owner coverage is `63 passed`.

At resume, inspect the earlier shared reconciliation after static sanitizers,
two binary repairs, and the singleton-consecutive-Reshape cluster. It needs a
combined aggregate result from that cluster before it can be guarded safely.
Do not infer no-op status from only the two repair counters. Characterize first,
then commit and push only; do not create or update a pull request.

## Singleton/consecutive-Reshape result characterization checkpoint

The earlier shared reconciliation boundary includes the three-runner
singleton-channel transpose, duplicate-fan-out, and consecutive-Reshape
cluster. The generic recovery utility already returns ordered callback values,
but `run_singleton_consecutive_reshape()` and its private lowerer helper still
discard all three dictionaries.

A strict expected-failure contract now requires an ordered three-result tuple
while preserving owner order, the shared `ModelIRPassStateScope`, all three
target forms, arguments, diagnostics, and exception behavior. Production and
the shared reconciliation remain unchanged.

At implementation, forward a typed three-dictionary tuple from the runner and
private helper. Existing call sites may continue ignoring it. Validate every
orchestration suite and the core/architecture/TensorFlow boundary before
changing the shared reconciliation. Commit and push only; do not create or
update a pull request.

## Singleton/consecutive-Reshape result implementation checkpoint

`run_singleton_consecutive_reshape()` now returns the ordered result
dictionaries for singleton-channel transpose, duplicate fan-out, and
consecutive Reshape. The lowerer's private helper forwards the typed triple.
All three existing call sites continue ignoring it, so ModelIR and
reconciliation behavior remain unchanged in this checkpoint.

Sequential validation across every orchestration suite is
`294 passed in 3.93s`. The core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `339 passed in 25.88s`. Focused helper,
ordered-result, target-form, and shared-state-scope contracts pass.

At resume, characterize the earlier shared reconciliation using all preceding
sanitizer/repair results, the three cluster result dictionaries, and tensor-count
prune accounting. Do not use an incomplete subset of cluster counters. Commit
and push only; do not create or update a pull request.

## Shared late-reconciliation characterization checkpoint

The earlier shared reconciliation is owned by six direct results and the three
singleton/consecutive cluster results: dynamic boundary-signature realignment,
HardSwish shape sanitization, Squeeze sanitization, wrong-way Conv transpose
sanitization, exact and singleton binary repair, singleton-channel transpose,
duplicate Reshape fan-out, and consecutive Reshape cleanup.

All returned values in these nine dictionaries are mutation counts. The new
empty-cluster fixture freezes its exact zero-only dictionaries. A strict
expected-failure architecture contract requires all nine results to be assigned
and passed to one compact `_stats_have_positive_count()` predicate, with a
pre-boundary tensor count covering zero-rewrite pruning. Production remains
unchanged.

At implementation, add the private pure mutation-count helper, capture all nine
results in their existing order, and guard only the immediate shared
reconciliation. Add helper unit tests and lowerer wiring for every dictionary
plus prune-only behavior. Validate all owner/orchestration/core gates
sequentially, then commit and push only; do not create or update a pull request.

## Shared late-reconciliation implementation checkpoint

The earlier shared boundary now records its tensor count, assigns the six
direct sanitizer/repair results, and unpacks the three ordered
singleton/consecutive cluster results without changing their execution order.
Its immediate static-shape reconciliation runs only when one of those nine
pure mutation-count dictionaries contains a positive value or zero-rewrite
pruning reduces the tensor table.

`_stats_have_positive_count()` is deliberately narrow: it receives only pure
mutation dictionaries and treats zero or negative values as unchanged. A
synthetic lowerer fixture covers the all-zero path, every individual result,
and prune-only behavior; each changed outcome adds exactly one reconciliation.
The singleton/consecutive boundary contract was updated to recognize the
result-capturing assignment. Three older indexed-pass tests were also aligned
with the extracted helper ownership in `core.model_ir_utils`, removing stale
references to lowerer-private aliases without changing production behavior.

Sequential validation across the complete related owner/orchestration, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites is
`432 passed in 27.29s`. The focused helper, lowerer wiring, extracted-owner,
and structural checks are `7 passed in 0.70s`.

At resume, re-inventory the remaining unconditional static-shape
reconciliations and select the next boundary only when every mutating owner has
a complete returned result or explicit prune accounting. Characterize it
before changing production, then commit and push only. Do not create or update
a pull request.

## Indexed binary-layout convergence characterization checkpoint

`_run_indexed_binary_layout_convergence()` still executes its complete
three-round sequence after reaching a stable round. Each round runs rank-four
broadcast-constant repair, stale NCHW-to-NHWC binary-Transpose repair, and
static-shape reconciliation with one shared `ModelIRGraphIndex`.

The three returned dictionaries contain only mutation counts. The two repair
owners prune only after a positive rewrite, and reconciliation changes metadata
only, so an all-zero round is a complete convergence signal. Strict
expected-failure cases require immediate zero-change convergence to stop after
one round and a reconciliation-only first change to stop after the following
zero round. A passing contract keeps the existing maximum at three rounds when
reconciliation reports a change every time and verifies the shared index.

At implementation, preserve owner order, one-index ownership, aggregate return
statistics, and the three-round cap. Break only after adding all three current
round results and confirming every mutation dictionary is zero. Run the full
indexed binary-layout owner suite plus core, pass-efficiency, architecture, and
TensorFlow-import-blocked gates sequentially. Commit and push only; do not
create or update a pull request.

## Indexed binary-layout convergence implementation checkpoint

`_run_indexed_binary_layout_convergence()` now accumulates broadcast repair,
stale binary-Transpose repair, and static-shape reconciliation results for the
current round, then stops when all three pure mutation dictionaries are zero.
Changing rounds continue with the same `ModelIRGraphIndex`, owner order, and
aggregate counters, and the existing three-round maximum remains unchanged.

The immediate-stability and second-round-stability characterizations are now
passing. The always-changing fixture still executes exactly three rounds, and
the original multi-repair fixture remains ModelIR/stat identical to the former
fixed three-round sequence. Two adjacent binary-owner tests were updated to
monkeypatch the extracted owner modules rather than aliases that no longer
exist in the lowerer; this changes test ownership only.

Sequential validation across the complete indexed convergence, binary layout,
binary adapter, shape reconciliation, core, pass-efficiency, architecture, and
TensorFlow-import-blocked suites is `402 passed in 26.66s`. The focused
convergence owner suite is `11 passed in 0.54s`.

At resume, continue the reconciliation inventory outside fixed-point helpers.
Choose a boundary only when all mutations since the preceding reconciliation
are represented by returned counters or explicit prune accounting. Preserve
phase order, characterize before production changes, and commit/push only. Do
not create or update a pull request.

## Indexed shape-convergence stable-scan characterization checkpoint

`_run_indexed_shape_convergence_cleanup()` owns dead-operator pruning, an
initial static-shape reconciliation, dynamic-Reshape resolution, and a final
static-shape reconciliation under one `ModelIRGraphIndex`. The final scan still
runs when all three preceding mutation dictionaries are zero.

A strict expected-failure fixture now requires that complete stable path to
execute only one reconciliation. Three passing fixtures preserve the final
reconciliation independently after a prune mutation, a first-reconciliation
metadata mutation, or a dynamic-Reshape mutation. This deliberately does not
assume that one changing reconciliation is already a fixed point.

At implementation, initialize the final result to the exact zero dictionary
and invoke the second reconciliation only when one of the three preceding pure
mutation dictionaries is positive. Preserve aggregate statistics, the shared
index, LayoutState forwarding, and both production call boundaries. Validate
dynamic Reshape, shape reconciliation, final convergence, core, architecture,
pass-efficiency, and TensorFlow-import-blocked suites sequentially. Commit and
push only; do not create or update a pull request.

## Indexed shape-convergence stable-scan implementation checkpoint

`_run_indexed_shape_convergence_cleanup()` now initializes the final
reconciliation result with the exact zero counter. It runs the second static
shape scan only when dead pruning, the first reconciliation, or dynamic-Reshape
resolution reports a positive mutation count.

The all-zero path therefore uses one reconciliation. Each independently
changing owner still triggers the second scan, preserving possible second-order
metadata convergence. Aggregate statistics, the shared `ModelIRGraphIndex`,
LayoutState forwarding, and both production call boundaries are unchanged. The
architecture contract directly verifies the three-result guard and the guarded
call's shared index.

Sequential validation across dynamic Reshape, shape reconciliation, indexed
final convergence, graph cleanup, core, pass-efficiency, architecture, and
TensorFlow-import-blocked suites is `380 passed in 25.82s`. Focused stable,
per-owner mutation, structure, and end-to-end equivalence coverage is
`6 passed in 2.07s`.

At resume, inspect the larger final shape/activation convergence coordinator.
Do not guard any of its three remaining reconciliation boundaries until the
preceding sanitizer, Reshape resolver, or fusion result completely accounts for
the relevant mutation interval. Characterize each candidate independently,
then commit and push only. Do not create or update a pull request.

## First final-convergence reconciliation characterization checkpoint

The indexed final shape/activation coordinator currently runs an additional
static-shape reconciliation immediately after
`_run_indexed_shape_convergence_cleanup()` and HardSwish shape sanitation, even
when both returned mutation dictionaries are entirely zero.

A strict expected-failure event-order fixture requires that one scan to be
absent on the complete stable path. Two passing cases preserve it after either
the convergence aggregate or HardSwish sanitizer reports a mutation. Every
stage receives the same `ModelIRGraphIndex`; the later Reshape reconciliation
and final post-fusion reconciliation remain in their original order.

At implementation, initialize only `first_reconcile_stats` with the exact zero
counter and guard that call with both predecessor dictionaries. Do not change
the second or final reconciliation in this checkpoint. Preserve aggregate
statistics, one-index ownership, LayoutState forwarding, and full legacy
ModelIR equality. Validate indexed final convergence, dynamic Reshape, shape
reconciliation, core, architecture, pass-efficiency, and
TensorFlow-import-blocked suites sequentially. Commit and push only; do not
create or update a pull request.

## First final-convergence reconciliation implementation checkpoint

The first extra reconciliation in
`_run_indexed_final_shape_activation_convergence()` now initializes its result
with the exact zero counter and runs only when the preceding indexed
shape-convergence aggregate or HardSwish sanitizer contains a positive
mutation count.

The complete stable path skips that scan. Either predecessor mutation retains
it in the same position with the same `ModelIRGraphIndex`. The later
dynamic-Reshape reconciliation and final post-fusion reconciliation are
unchanged. The architecture contract verifies the exact two-result guard,
guarded-call position, and shared index.

Sequential validation across indexed final convergence, dynamic Reshape,
shape reconciliation, graph cleanup, HardSwish/SE layout, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites is
`391 passed in 25.95s`. Focused stable, per-predecessor mutation, structure,
and legacy-equivalence coverage is `6 passed in 2.08s`.

At resume, characterize the dynamic-Reshape-to-second-reconciliation boundary.
The guard must include both `first_reconcile_stats` and `reshape_stats` so a
first reconciliation that changed metadata can still receive a convergence
scan even when Reshape resolution is a no-op. Leave the final post-fusion scan
unchanged until its own interval is characterized. Commit and push only; do
not create or update a pull request.

## Second final-convergence reconciliation characterization checkpoint

The second additional scan in
`_run_indexed_final_shape_activation_convergence()` follows the optional first
reconciliation and dynamic-Reshape resolution. It still runs when both of
those mutation dictionaries are zero.

A strict expected-failure event-order fixture requires that stable path to
proceed directly to activation fusion. Two passing paths preserve the scan
after a first-reconciliation metadata mutation and after a dynamic-Reshape
rewrite. The shared `ModelIRGraphIndex`, aggregate statistics, already-guarded
first scan, and final post-fusion scan remain explicit in every fixture.

At implementation, initialize only `second_reconcile_stats` with the exact
zero counter and guard its call with `first_reconcile_stats` and
`reshape_stats`. Do not broaden the guard to earlier aggregate results and do
not change the final reconciliation. Update the structural contract to verify
both guards in source order, then validate indexed final convergence, dynamic
Reshape, shape reconciliation, core, architecture, pass-efficiency, and
TensorFlow-import-blocked suites sequentially. Commit and push only; do not
create or update a pull request.

## Second final-convergence reconciliation implementation checkpoint

`second_reconcile_stats` now initializes with the exact zero counter. The
second additional static-shape scan runs only when `first_reconcile_stats` or
`reshape_stats` reports a positive mutation count.

The complete stable path and predecessor-only changes whose first scan is
already stable now proceed directly to activation fusion. A changing first
reconciliation or dynamic-Reshape rewrite retains the second scan in its
original position. The structural contract verifies both reconciliation guards
in source order and confirms every direct and guarded owner receives the same
`ModelIRGraphIndex`. The final post-fusion scan is unchanged.

Sequential validation across indexed final convergence, dynamic Reshape,
shape reconciliation, graph cleanup, HardSwish/SE layout, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites is
`394 passed in 26.57s`. Focused stable, mutation-source, ordered-guard, and
legacy-equivalence coverage is `9 passed in 0.56s`.

At resume, characterize the final post-fusion reconciliation. Its guard must
cover both `second_reconcile_stats` and the complete activation-fusion result,
so metadata changes from the second scan retain one convergence opportunity
even when fusion is a no-op. Characterize fusion counter completeness and any
zero-count prune behavior before changing production. Commit and push only; do
not create or update a pull request.

## Post-fusion reconciliation characterization checkpoint

The final scan in `_run_indexed_final_shape_activation_convergence()` follows
the optional second reconciliation and activation fusion. Fusion exposes
complete rewrite counters, but its owner unconditionally calls
`_prune_unused_tensors()` even when every counter is zero.

A new owner fixture freezes zero-fusion pruning, its lineage event, and
LayoutState synchronization. A strict expected-failure event-order fixture
requires the final scan to be skipped only for zero second-reconciliation and
fusion results with no tensor-count reduction. Passing fixtures preserve it
after a second-reconciliation metadata change, a fusion rewrite, and a
zero-rewrite prune-only mutation.

At implementation, record tensor count immediately before fusion, initialize
`final_reconcile_stats` with the exact zero counter, and guard the final scan
with `second_reconcile_stats`, every fusion counter through
`_stats_have_positive_count()`, or a tensor-count decrease. Extend the
structural contract to a third ordered guard. Validate activation fusion,
indexed final convergence, dynamic Reshape, shape reconciliation, core,
architecture, pass-efficiency, and TensorFlow-import-blocked suites
sequentially. Commit and push only; do not create or update a pull request.

## Post-fusion reconciliation implementation checkpoint

The coordinator now records `fusion_tensor_count` immediately before
activation fusion and initializes `final_reconcile_stats` with the exact zero
counter. The final scan runs only when `second_reconcile_stats` or any fusion
counter is positive, or when zero-rewrite pruning reduces the tensor table.

The complete stable path now ends after fusion. A changing second
reconciliation, fusion rewrite, and prune-only mutation each preserve the scan
with the same `ModelIRGraphIndex`. The architecture contract verifies all three
guards in source order, the tensor-count boundary, and index forwarding. The
activation-fusion owner fixture confirms zero-count pruning and LayoutState
synchronization.

Sequential validation across activation fusion, indexed final convergence,
dynamic Reshape, shape reconciliation, graph cleanup, HardSwish/SE layout,
core, pass-efficiency, architecture, and TensorFlow-import-blocked suites is
`414 passed in 26.42s`. Focused final-coordinator, prune-only, structure, and
legacy-equivalence coverage is `13 passed in 2.27s`.

At resume, return to the larger lowerer reconciliation inventory. The three
indexed final-convergence scans now have explicit mutation contracts; do not
merge their guards unless tests can still distinguish each convergence
opportunity. Select the next boundary only when every mutation since the prior
scan has a returned counter or prune accounting. Commit and push only; do not
create or update a pull request.

## Placeholder-MatMul repair reconciliation characterization checkpoint

The absolute-final placeholder-MatMul restoration block currently performs a
first static-shape reconciliation, exact rank-four binary repair, singleton
broadcast repair, and a second unconditional reconciliation whenever the
restoration counter is positive.

The first scan must remain. Its returned mutation counter and both repair
counters completely describe their rewrites, while the exact repair may also
prune unused tensors on a zero rewrite. A strict expected-failure lowerer
fixture requires the second scan only after a changing first reconciliation,
an exact repair, a singleton repair, or prune-only tensor-count reduction.

At implementation, assign the restoration result, retain its existing outer
guard, capture the first reconciliation and both repair results, and record
tensor count before the repairs. Guard only the second reconciliation with the
three dictionaries or a tensor-count decrease. Preserve call order,
LayoutState forwarding, and the following topology sort. Validate placeholder
MatMul restoration, binary adapters, shape reconciliation, core,
architecture, pass-efficiency, and TensorFlow-import-blocked suites
sequentially. Commit and push only; do not create or update a pull request.

## Placeholder-MatMul repair reconciliation implementation checkpoint

The restoration result is now assigned to
`final_placeholder_matmul_stats` before the unchanged positive-result guard.
Inside that block, the first reconciliation remains mandatory and its result is
captured. The lowerer then records tensor count and captures exact and
singleton binary-repair results in the original order.

The second reconciliation now runs only when the first scan or either repair
reports a positive mutation, or when zero-rewrite exact repair pruning reduces
the tensor table. The topology sort remains unconditional inside the
restoration block. A structural contract fixes all six statements and the
complete guard; lowerer wiring independently covers every mutation source and
prune-only behavior.

Sequential validation across placeholder-MatMul/dynamic Reshape, binary
adapters, shape reconciliation, core, pass-efficiency, architecture, and
TensorFlow-import-blocked suites is `403 passed in 26.48s`. Focused wiring and
structure coverage is `2 passed in 2.27s`.

At resume, continue the direct lowerer reconciliation inventory. Avoid the
large phase-barrier scans unless every intervening owner result is preserved;
prefer another local conditional block with a complete result interval.
Characterize before production changes, then commit and push only. Do not
create or update a pull request.

## Late binary-layout recovery runner characterization checkpoint

The conditional block after late binary repair currently runs PReLU
passthrough, dual pre-Add recovery, terminal affine-FC recovery, optional
PReLU-BMM recovery, affine pre/post recovery, optional generic layout cleanup,
and then reconciles unconditionally.

Every owner returns rewrite counters, but several owners prune unused tensors
on zero rewrites. Generic layout cleanup also returns `iterations`, which is an
execution count rather than a mutation. Strict expected-failure architecture
and lowerer-wiring contracts now require one dedicated runner. It must preserve
the outer branch, owner order, optional branches, Session LayoutState and
diagnostics; return only mutation counts plus net pruning; and trigger
reconciliation only for a positive aggregate.

At implementation, add a TensorFlow-independent pass module that imports the
six existing owners directly. Normalize optional-owner counters to zero, filter
the layout `iterations` field, and add `pruned_unused_tensors` from the net
tensor-count reduction. Replace the inline lowerer sequence with one runner
assignment and one aggregate guard. Validate the runner, every affected owner,
core, architecture, pass-efficiency, and TensorFlow-import-blocked suites
sequentially. Commit and push only; do not create or update a pull request.

## Late binary-layout recovery runner implementation checkpoint

The new TensorFlow-independent `run_late_binary_layout_recovery()` pass module
now owns the full late recovery cluster. It preserves the existing PReLU,
dual-pre-Add, terminal affine-FC, optional PReLU-BMM, affine pre/post, and
optional generic-layout order. LayoutState and diagnostics cross the new
boundary explicitly, and optional owners return stable zero counters when they
are disabled.

The runner exposes only mutation evidence: five owner rewrite counters, four
generic-layout mutation counters, and `pruned_unused_tensors` calculated from
net tensor-table reduction. It filters generic layout cleanup's non-mutating
`iterations` count. The lowerer retains the original outer condition but now
contains only the runner call and an aggregate mutation guard, so reconciliation
is skipped on a fully stable sequence and preserved after a rewrite or
prune-only mutation.

Dedicated runner and lowerer-wiring coverage initially passed as `5 passed in
2.16s`. The seven legacy tests that encoded direct-lowerer call ownership were
updated to recognize the new runner boundary and passed as `7 passed in
2.43s`. The complete sequential related suite covering all six owners, layout
cleanup, core, pass efficiency, architecture, and TensorFlow import blocking is
`523 passed in 26.93s`.

At resume, inventory the next local lowerer phase boundary before changing
production. Prefer a bounded sequence whose complete mutation interval is
represented by returned counters or explicit prune accounting; characterize
that boundary first. Commit and push coherent units only. Do not create or
update a pull request.

## Typed Constant lowering characterization checkpoint

The next unconditional reconciliation after terminal layout cleanup spans a
large phase whose owners do not yet expose complete mutation and pruning
evidence, so it was deliberately rejected as the next optimization boundary.
The selected bounded unit is the inline ONNX `Constant` special case in the
node-lowering loop. This also addresses the outstanding Pylance diagnostic on
protobuf `attribute.name` access.

Characterization fixes tensor data, dtype, shape/signature, graph output,
provenance, no-operator behavior, and the exact missing tensor-`value` error. A
strict expected-failure architecture contract requires one
TensorFlow-independent `op_families.constant` owner, a two-argument typed call
from the lowerer, and an explicit `onnx.AttributeProto` cast inside the owner.
The existing tensor-`value` feature scope, collision handling, and placeholder
replacement behavior must remain unchanged.

At implementation, move only the current Constant branch into the typed owner;
do not route it through general registry dispatch or add support for additional
Constant attribute encodings in this mechanical extraction. Validate focused
Constant/core/architecture tests, then the sequential core, architecture,
pass-efficiency, and TensorFlow-import-blocked suites. Commit and push only;
do not create or update a pull request.

## Typed Constant lowering implementation checkpoint

`op_families.constant.lower_constant_node()` now owns the existing
tensor-valued ONNX Constant lowering behavior. The lowerer keeps the same
`Constant` guard and progress/error boundary but delegates through one typed
`node`/`ctx` call. The owner explicitly casts protobuf attributes to
`onnx.AttributeProto`, so Pylance no longer has to infer `name` and `t` from the
ambiguous generated container element type.

The new owner preserves direct tensor creation, in-place placeholder updates,
legacy collision renaming, constants-map synchronization, dtype and
shape/signature normalization, graph-output behavior, provenance, and the exact
missing tensor-`value` exception. No additional Constant encodings were added,
and the path remains TensorFlow-independent.

Focused owner, public behavior, and architecture coverage is `5 passed in
2.27s`. The sequential core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `351 passed in 26.06s`. Ruff, Python bytecode
compilation, and whitespace validation pass. Pyright is not installed in the
existing uv environment; the source contract directly verifies the explicit
`AttributeProto` cast without adding a dependency.

At resume, inventory the remaining special control flow inside the ONNX node
lowering loop, especially the demand-driven unresolved-rank reconciliation
before shape-sensitive ops. Characterize its trigger and no-op behavior before
extracting it. Do not fold Constant into registry dispatch as part of that unit.
Commit and push only; do not create or update a pull request.

## Demand-driven shape readiness characterization checkpoint

The remaining inline special control flow in the ONNX node loop reconciles the
partial ModelIR immediately before six shape-sensitive op types. Five
characterization cases freeze the trigger: an unresolved nonconstant MatMul
input reconciles once; an equally unresolved Add input, fully known MatMul
input, explicit rank-one hint, and constant tensor do not reconcile before
dispatch.

A strict expected-failure architecture contract requires one
TensorFlow-independent `core.shape_readiness` owner to contain the target-op
set, unresolved-input test, and static-shape reconciliation. The central loop
must make one typed `node`/`ctx` call and must no longer define the nested
`_has_unresolved_rank` closure.

At implementation, preserve the exact six-op set and return the existing shape
reconciliation result, or its exact zero dictionary when no scan is requested.
Do not broaden unresolved-rank semantics or combine this lowering-time scan with
post-lowering phase barriers. Validate focused trigger/owner/core/architecture
coverage, then the sequential core, pass-efficiency, architecture, and
TensorFlow-import-blocked suites. Commit and push only; do not create or update
a pull request.

## Demand-driven shape readiness implementation checkpoint

`core.shape_readiness.reconcile_shape_sensitive_inputs_on_demand()` now owns
the six-op target set, unresolved-rank predicate, and lowering-time static-shape
scan. It returns `{"reconciled_static_tensor_shapes": 0}` for non-target ops,
known rank-two-or-higher inputs, explicit rank-one shape hints, and constant
tensors. An unresolved nonconstant input with no raw shape hint invokes the
same static reconciliation owner as before.

The ONNX node loop now contains one typed `node`/`ctx` call between `NodeView`
construction and dispatch. Its inline op set, input-name collection, and nested
closure were removed. This preserves demand-driven behavior while making the
policy independently testable and avoiding graph scans for stable paths.

Focused owner, integration-trigger, and architecture coverage is `16 passed in
2.27s`. The sequential shape-readiness, Constant, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `367 passed in 26.36s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, return to the post-lowering phase inventory. The broad terminal
phase before its unconditional reconciliation is still unsafe to guard until
all participating owners return complete mutation and prune evidence. Select a
smaller orchestration cluster inside that interval and characterize its result
propagation before changing the phase barrier. Commit and push only; do not
create or update a pull request.

## Late layout cluster result propagation characterization checkpoint

The broad terminal phase before its unconditional reconciliation still lacks a
complete mutation aggregate. Its final composite cluster is a bounded first
step: late layout/mean/SPP/gather/constant-fold/cast builds five required
invocations and an optional generic-layout invocation. The shared recovery
utility returns all results in order, but the public orchestrator and private
lowerer helper currently discard them.

Strict expected-failure coverage requires the exact five- or six-result tuple
to pass unchanged through both layers. Result order, shared state scope,
optional policy, diagnostics, exceptions, and the production call site remain
unchanged. No phase reconciliation is guarded in this checkpoint. The optional
generic-layout result includes `iterations`; a later aggregate must filter that
execution count and use only mutation keys.

At implementation, annotate and return the raw ordered tuple from the
orchestrator, then return it from the lowerer helper. Keep the existing call
site as an ignored expression so ModelIR behavior is mechanical and identical.
Validate both optional policies, state reuse, orchestration structure, core,
architecture, pass efficiency, and TensorFlow import blocking sequentially.
Commit and push only; do not create or update a pull request.

## Late layout cluster result propagation implementation checkpoint

`run_late_layout_mean_spp_gather_constant_cast()` now returns the exact raw
tuple produced by `run_recovery_invocations()`. The lowerer's private helper
returns that tuple unchanged. Required-only execution therefore exposes five
ordered dictionaries, while layout-enabled execution exposes six, with the
same invocation order, state scope, diagnostics, and exception semantics.

The terminal production call remains an ignored expression. No result is yet
captured, no mutation aggregate is formed, and the unconditional phase
reconciliation is unchanged. The optional layout dictionary still carries the
non-mutating `iterations` field and must be normalized before any future guard.

Focused orchestration and structural coverage is `15 passed in 2.20s`. The
sequential late-cluster, child constant-fold/cast, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `376 passed in 26.14s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, characterize a normalized mutation aggregate for this one cluster
without guarding the broad phase reconciliation yet. The aggregate must map
the optional layout result to its four mutation keys, retain each required
pass dictionary, and account for any zero-rewrite pruning performed by those
owners. Commit and push only; do not create or update a pull request.

## Late layout cluster mutation summary characterization checkpoint

The raw five- or six-result tuple is now available, but it is not safe to pass
directly to `_stats_have_positive_count()` because the optional layout result
contains the non-mutating `iterations` field. Strict expected-failure coverage
defines a fixed summary: the four layout mutation keys, all required-pass
mutation dictionaries, and `pruned_unused_tensors` from the cluster's net
tensor-count reduction. Layout keys are explicit zeros when the optional pass
is disabled, and `iterations` is absent.

The terminal call site must record tensor count, capture the ordered raw tuple,
and compute this summary in three adjacent statements. The subsequent
Expand/Squeeze owner and unconditional reconciliation remain unchanged and do
not consume the summary yet.

At implementation, add a pure validating summary function to the orchestration
module and capture its result at the current production call. Preserve the raw
runner/helper return contract and all owner order. Add no reconciliation guard.
Validate both policies, malformed result length, net pruning, structure, core,
architecture, pass efficiency, and TensorFlow import blocking sequentially.
Commit and push only; do not create or update a pull request.

## Late layout cluster mutation summary implementation checkpoint

The pure
`summarize_late_layout_mean_spp_gather_constant_cast_mutations()` helper now
validates the expected five- or six-result tuple, emits four fixed layout
mutation keys, drops `iterations`, merges every required-pass dictionary, and
adds a clamped `pruned_unused_tensors` count.

The production call site records the cluster's starting tensor count, captures
the raw result tuple, and derives `_late_layout_cluster_stats` from net tensor
reduction. The underscore is intentional: this evidence is staged but not yet
used by the broad phase barrier. Expand/Squeeze and the unconditional static
reconciliation remain exactly where they were.

Focused summary, malformed-length, pruning, return, boundary, and architecture
coverage is `19 passed in 2.28s`. The sequential late-cluster, child
constant-fold/cast, core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `379 passed in 26.27s`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, propagate mutation/prune evidence from another small cluster in the
same terminal interval. Do not use `_late_layout_cluster_stats` to guard the
phase reconciliation until every preceding owner since the prior mandatory
shape barrier is accounted for. Commit and push only; do not create or update a
pull request.

## Terminal Expand/Squeeze result capture characterization checkpoint

Immediately after the staged late-layout summary, the terminal
Expand/Squeeze-to-Reshape owner returns two complete mutation counters but its
result is discarded. The owner prunes only after a positive rewrite, so those
counters completely describe whether it changed ModelIR.

A strict expected-failure structure test requires the production call to assign
`_terminal_expand_squeeze_stats`, preserve Session LayoutState forwarding, and
leave the following unconditional static reconciliation immediately in place.
No phase barrier is guarded in this checkpoint.

At implementation, change only the expression to a staged underscored
assignment and update the two surrounding boundary contracts. Validate the
complete Expand/Squeeze owner suite, late-layout orchestration, core,
architecture, pass efficiency, and TensorFlow import blocking sequentially.
Commit and push only; do not create or update a pull request.

## Terminal Expand/Squeeze result capture implementation checkpoint

The terminal `_replace_expand_dims_and_squeeze_with_reshape()` call now assigns
its unchanged result dictionary to `_terminal_expand_squeeze_stats`. Its
Session LayoutState, ordering after the late-layout summary, and immediately
following unconditional static reconciliation are unchanged. The staged result
is not consumed by a guard.

Focused owner, boundary, summary, and orchestration coverage is `20 passed in
2.33s`. The sequential Expand/Squeeze, late-layout, child constant-fold/cast,
core, pass-efficiency, architecture, and TensorFlow-import-blocked gate is `382
passed in 27.09s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, continue moving backward through the same terminal interval and
propagate one small orchestration cluster's mutation/prune evidence. The broad
phase barrier must remain unconditional until every intervening owner has
complete accounting. Commit and push only; do not create or update a pull
request.

## Late hard-activation cluster evidence characterization checkpoint

Moving backward through the terminal interval, the late hard-activation/layout
pair is the next bounded cluster. The required hard-activation owner and
optional generic-layout owner both may prune on zero rewrites; the layout result
also contains non-mutating `iterations`. The orchestrator and private helper
currently discard the recovery utility's ordered one- or two-result tuple.

Strict expected-failure coverage requires raw tuple propagation, tuple-length
validation, four fixed layout mutation keys, `iterations` filtering, required
counter preservation, and net `pruned_unused_tensors`. The production call must
capture count/results/summary in order while preserving its Hardswish-SE and
pre-Concat boundaries. No reconciliation guard changes.

At implementation, return the raw tuple through both runner layers, add a pure
summary helper, and stage `_late_hard_activation_stats` at the existing call
site. Validate both policies, malformed lengths, pruning, owner order, shared
state, core, architecture, pass efficiency, and TensorFlow import blocking
sequentially. Commit and push only; do not create or update a pull request.

## Late hard-activation cluster evidence implementation checkpoint

The orchestration runner and lowerer's private delegate now return the recovery
utility's ordered one- or two-result tuple. The pure
`summarize_late_hard_activation_layout_mutations()` helper validates the tuple
length for the active policy, preserves both required hard-activation counters,
emits four fixed zero-default layout mutation keys, excludes the non-mutating
layout `iterations` metric, and adds a clamped `pruned_unused_tensors` count.

At the original terminal call site, the lowerer records the starting tensor
count, captures the raw results, and derives `_late_hard_activation_stats` from
the exact net tensor reduction. The leading underscore marks staged evidence;
the broad phase reconciliation remains unconditional. The existing pass order,
optional-layout policy, shared LayoutState, Hardswish-SE/pre-Concat boundaries,
public behavior, artifacts, and TensorFlow-free direct path are unchanged.

Focused orchestration coverage is `12 passed in 0.64s`. The sequential late
hard-activation, Expand/Squeeze, late-layout, constant-fold/cast, core,
pass-efficiency, architecture, and TensorFlow-import-blocked gate is `394 passed
in 28.01s`. Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, continue moving backward through the same terminal interval and
characterize another small owner or orchestration cluster before using any
staged evidence to guard broad reconciliation. Commit and push only; do not
create or update a pull request.

## Terminal Hardswish-SE evidence characterization checkpoint

The terminal Hardswish-SE owner immediately before the late hard-activation
cluster returns a complete rewrite counter, but it also invokes unused-tensor
pruning unconditionally. A dedicated zero-rewrite fixture confirms that the
owner can remove a tensor while returning zero, so its raw dictionary alone
cannot prove ModelIR stability.

A strict expected-failure structural contract requires two adjacent production
assignments: capture the starting tensor count, then merge the unchanged owner
result with an exact, clamped `pruned_unused_tensors` count. The preceding
split/conv bridge call and following late hard-activation count remain the
fixed boundaries. No pass call, ordering, policy, or reconciliation changes.

At implementation, replace only the discarded terminal call with this
prune-aware staged evidence and update the two affected boundary contracts.
Validate the complete Hardswish-SE owner suite, late hard-activation
orchestration, terminal recovery structure, core, architecture, pass
efficiency, and TensorFlow import blocking sequentially. Commit and push only;
do not create or update a pull request.

## Terminal Hardswish-SE evidence implementation checkpoint

The terminal call now records `terminal_hardswish_se_tensor_count` and stores
the owner's unchanged rewrite dictionary in `_terminal_hardswish_se_stats`
together with the exact, clamped net `pruned_unused_tensors` count. This makes
the owner's unconditional zero-rewrite pruning observable without changing the
owner, its compatibility wrapper, or its output counter.

The staged dictionary is observation-only. The preceding split/conv bridge,
following late hard-activation cluster, pass policy and order, shared
LayoutState, broad phase reconciliation, public behavior, artifacts, and
TensorFlow-free direct path are unchanged.

Focused Hardswish-SE and late hard-activation coverage is `22 passed in 0.74s`.
The sequential Hardswish-SE, late hard-activation, QKV, split/conv bridge,
SINet terminal, Expand/Squeeze, late-layout, constant-fold/cast, core,
pass-efficiency, architecture, and TensorFlow-import-blocked gate is `469 passed
in 28.37s`. Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, continue backward with the adjacent split/conv bridge owner or the
late QKV cluster. Characterize raw results and zero-rewrite pruning before
extending the staged terminal aggregate. Do not guard broad reconciliation
until the full interval is accounted for. Commit and push only; do not create
or update a pull request.

## Terminal split/conv bridge evidence characterization checkpoint

The indexed split/conv/concat bridge owner immediately preceding the terminal
Hardswish-SE capture returns one rewrite counter and prunes unused tensors only
when that counter is positive. Its bounded candidate/idempotence coverage and
24 transactional unsafe-candidate fixtures establish that a zero result leaves
ModelIR unchanged. The raw return is therefore complete mutation evidence.

A strict expected-failure structural contract requires the terminal call to
assign its unchanged result to `_terminal_split_conv_concat_bridge_stats` while
preserving ModelIR and Session LayoutState arguments. The late QKV call and
Hardswish-SE tensor-count assignment remain the exact neighboring boundaries.
No reconciliation guard changes.

At implementation, replace only the discarded expression with the staged
assignment and update boundary-aware QKV and architecture tests. Validate the
complete indexed owner suite and the adjacent QKV, Hardswish-SE,
hard-activation, core, pass-efficiency, architecture, and TensorFlow import
blocker sequentially. Commit and push only; do not create or update a pull
request.

## Terminal split/conv bridge evidence implementation checkpoint

Only the split/conv/concat bridge invocation directly after the late QKV
cluster now assigns its unchanged result to
`_terminal_split_conv_concat_bridge_stats`. The earlier invocations of the same
owner remain untouched. This confines mutation staging to the terminal
interval currently being accounted and prevents an ambiguous first-match
rewrite.

The owner call still receives the same ModelIR and Session LayoutState, and the
late QKV and Hardswish-SE boundaries remain adjacent. Since pruning occurs only
after a positive rewrite, the existing counter remains complete evidence. The
staged result is observation-only and no reconciliation guard changes.

Focused indexed owner, QKV, and Hardswish-SE coverage is `69 passed in 0.80s`.
The sequential indexed owner, QKV, Hardswish-SE, late hard-activation, SINet
terminal, Expand/Squeeze, late-layout, constant-fold/cast, core,
pass-efficiency, architecture, and TensorFlow-import-blocked gate is `470 passed
in 28.41s`. Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, characterize the late QKV cluster's ordered results and prune
behavior. Preserve `include_prefix=False`, the optional generic-layout policy,
and all three production forms. Commit and push only; do not create or update a
pull request.

## Late QKV mutation evidence characterization checkpoint

The QKV orchestration owner serves three production policies: required prefix
plus bridge, bridge only, and optional generic layout plus bridge. The terminal
form uses `include_prefix=False`. Its generic layout and required bridge owners
both may prune on zero rewrites, and generic layout also returns non-mutating
`iterations`, so raw counters alone are insufficient.

Strict expected-failure coverage requires the runner and private delegate to
return the ordered one- or two-result tuple for all three policies. A pure
summary must validate tuple length and emit a stable schema of four layout,
four prefix, two bridge, and one net-prune counter, with inactive-family zeros
and no `iterations`. The terminal production form must capture starting tensor
count, raw results, and normalized `_late_qkv_stats` between shape-extract and
the staged split/conv bridge. No reconciliation guard changes.

At implementation, preserve all invocation defaults and orders, return raw
results through both layers, add the fixed pure summary, and stage only the
terminal call. Validate all policies, malformed lengths, summary filtering,
net pruning, shared pass state, boundary structure, core, pass efficiency,
architecture, and TensorFlow import blocking sequentially. Commit and push
only; do not create or update a pull request.

## Late QKV mutation evidence implementation checkpoint

The QKV orchestration runner and lowerer's private delegate now return the
ordered raw tuple for all three production policies. The pure
`summarize_qkv_attention_mutations()` helper validates the policy-specific
tuple length and always emits four layout, four prefix, two bridge, and one
net-prune counter. Inactive families are explicit zeros, generic-layout
`iterations` is excluded, and pruning is clamped to the exact nonnegative
tensor-count reduction.

Only the terminal `include_prefix=False` invocation stages
`late_qkv_tensor_count`, `late_qkv_results`, and `_late_qkv_stats`. The two
default invocation forms still discard their now-available results. All
defaults, pass IDs and order, optional-layout policy, shared pass state,
shape-extract/split-bridge boundaries, and unconditional broad reconciliation
remain unchanged.

Focused QKV, indexed split/conv bridge, and Hardswish-SE coverage is `73 passed
in 0.91s`. The sequential QKV, split/conv bridge, Hardswish-SE, late
hard-activation, SINet terminal, Expand/Squeeze, late-layout,
constant-fold/cast, shared-context, core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `503 passed in 28.68s`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, move backward to the shape-extract owner immediately before late
QKV. Confirm whether its return counter fully accounts for pruning before
staging it. Commit and push only; do not create or update a pull request.

## Pre-QKV shape-extract evidence characterization checkpoint

The shape-extract owner prunes unused tensors only after its rewrite counter is
positive. Its idempotence coverage and eight unsafe/unsupported snapshot
fixtures confirm that zero means no ModelIR mutation, so the raw result is
complete evidence.

Three production invocations share the same owner. A strict expected-failure
contract selects only the terminal call between the late SPP pair and
`late_qkv_tensor_count` and requires it to assign
`_late_pre_qkv_shape_extract_stats`. It also freezes the total call count at
three to prevent another ambiguous first-match edit. No arguments, order, or
reconciliation behavior change.

At implementation, replace only that exact terminal expression and update the
late-SPP, QKV, and architecture boundary contracts. Validate the complete
shape-extract suite and adjacent terminal orchestration, core, pass efficiency,
architecture, and TensorFlow import blocker sequentially. Commit and push only;
do not create or update a pull request.

## Pre-QKV shape-extract evidence implementation checkpoint

Only the shape-extract invocation between the late SPP pair and QKV starting
count now assigns its unchanged result to
`_late_pre_qkv_shape_extract_stats`. The first production call and the later
absolute-end shape-extract call remain expressions, and the structural contract
continues to require exactly three owner calls.

The owner still receives only ModelIR, its rewrite/prune behavior is unchanged,
and the staged result is not consumed by a reconciliation guard. Late SPP and
QKV remain its exact neighbors, with all other terminal evidence and the broad
reconciliation unchanged.

Focused shape-extract, late-SPP, and QKV coverage is `37 passed in 0.94s`. The
sequential shape-extract, late-SPP, QKV, split/conv bridge, Hardswish-SE, late
hard-activation, SINet terminal, Expand/Squeeze, late-layout,
constant-fold/cast, shared-context, core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `524 passed in 29.02s`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, propagate ordered results and net-prune evidence from the late SPP
pair immediately before this owner. Commit and push only; do not create or
update a pull request.

## Late SPP pair mutation evidence characterization checkpoint

Inspection shows that both late SPP child owners prune only after a positive
rewrite. Their preflight/no-op paths leave ModelIR unchanged, so the SPP and
concat-unary-conv counters already provide complete evidence and no separate
net-prune count is required.

Strict expected-failure coverage requires the orchestration runner and lowerer
delegate to return the ordered two-result tuple. A pure summary validates the
exact length and emits the two declared mutation counters. The production call
must stage `late_spp_results` and `_late_spp_stats` between the preceding raw
layout rewrite and `_late_pre_qkv_shape_extract_stats`. No reconciliation guard
changes.

At implementation, return results through both layers, add the pure fixed-key
summary, and stage both assignments at the existing call. Validate malformed
length, order, shared state, complete SPP and concat-unary-conv owner suites,
adjacent terminal orchestration, core, pass efficiency, architecture, and the
TensorFlow import blocker sequentially. Commit and push only; do not create or
update a pull request.

## Late SPP pair mutation evidence implementation checkpoint

The late SPP orchestration runner and lowerer's private delegate now return the
ordered two-result tuple. The pure
`summarize_late_spp_concat_unary_conv_mutations()` helper validates the exact
length and exposes only the SPP and concat-unary-conv mutation counters. Since
both owners prune only after a positive rewrite, no tensor-count proxy is
needed.

Production captures `late_spp_results` and `_late_spp_stats` at the existing
call site. The preceding raw layout rewrite, following pre-QKV shape-extract
result, owner order, shared pass state, and broad reconciliation remain
unchanged. The staged summary is not consumed by a guard.

Focused late-SPP and shape-extract coverage is `23 passed in 0.71s`. The
sequential complete SPP, concat-unary-conv, late-SPP, shape-extract, QKV,
split/conv bridge, Hardswish-SE, late hard-activation, SINet terminal,
Expand/Squeeze, late-layout, constant-fold/cast, shared-context, core,
pass-efficiency, architecture, and TensorFlow-import-blocked gate is `596 passed
in 29.28s`. Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately preceding terminal strided-slice/pad/concat
raw owner. Confirm its return and prune contract before staging evidence.
Commit and push only; do not create or update a pull request.

## Second terminal slice/pad/concat evidence characterization checkpoint

The strided-slice/pad/concat bridge owner prunes only after a positive rewrite,
and its transactional rejection and idempotence coverage make the single
counter complete mutation evidence. Two direct production calls use this owner,
with the second terminal affine recovery between them.

A strict expected-failure contract requires only the second call to assign
`_terminal_slice_pad_concat_stats`. It freezes the preceding terminal affine
recovery, following `late_spp_results`, direct-call count of two, and first call
as an unchanged expression. No arguments, owner behavior, or reconciliation
guard changes.

At implementation, replace the exact second expression and update late-SPP,
terminal-affine, and architecture boundary tests. Validate the complete bridge
owner suite and adjacent terminal orchestration, core, pass efficiency,
architecture, and TensorFlow import blocker sequentially. Commit and push only;
do not create or update a pull request.

## Second terminal slice/pad/concat evidence implementation checkpoint

Only the second direct strided-slice/pad/concat invocation now assigns its
unchanged counter to `_terminal_slice_pad_concat_stats`; the first direct call
remains an expression. The terminal affine boundary contracts distinguish the
two positions while confirming that both still invoke the same owner with only
ModelIR.

The first expanded gate found one stale architecture expectation that still
required an expression before late SPP (`693 passed, 1 failed`). Updating that
specific boundary produced `694 passed in 27.88s`; a focused three-test check
also confirmed that the unrelated very-late boundary stayed unchanged. The
staged result is observation-only and no reconciliation behavior changes.

Focused bridge, terminal-affine, and late-SPP coverage is `107 passed in
1.04s`. The final sequential bridge-owner, terminal-affine, complete SPP,
concat-unary-conv, late-SPP, shape-extract, QKV, split/conv bridge,
Hardswish-SE, late hard-activation, SINet terminal, Expand/Squeeze,
late-layout, constant-fold/cast, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `694 passed in 27.88s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the second terminal affine recovery's ordered child results.
Its current result is discarded and several child owners may prune, so
characterize complete mutation evidence before extending the terminal
aggregate. Commit and push only; do not create or update a pull request.

## Second terminal affine mutation evidence characterization checkpoint

Runtime schema probing on an empty ModelIR confirms eleven ordered results and
twelve declared mutation keys. The final probable-NHWC sanitizer contributes
two keys; each other owner contributes one. Because child prune conditions
differ, the cluster needs an explicit net tensor-reduction count in addition to
its raw counters.

Strict expected-failure coverage requires ordered tuple propagation through the
orchestrator and lowerer delegate, exact eleven-result validation, fixed-key
summary extraction, and clamped `pruned_unused_tensors`. Only the second of two
production recovery calls must stage starting tensor count, raw results, and
`_terminal_affine_stats`; the first stays an expression. The preceding first
slice/pad/concat call and following staged second call remain exact boundaries.

At implementation, add the pure summary, return raw results through both
layers, and replace only the second call with three adjacent assignments.
Validate malformed length, all twelve keys, net pruning, owner order, both
production calls, complete recovery coverage, core, pass efficiency,
architecture, and TensorFlow import blocking sequentially. Commit and push
only; do not create or update a pull request.

## Second terminal affine mutation evidence implementation checkpoint

The terminal affine orchestration runner and lowerer's delegate now return the
ordered eleven-result tuple. The pure
`summarize_terminal_affine_concat_split_mutations()` helper validates the exact
length, extracts only the twelve declared mutation keys, and appends clamped
net `pruned_unused_tensors` evidence.

The first production recovery remains an expression. The second records
`terminal_affine_tensor_count`, captures `terminal_affine_results`, and derives
`_terminal_affine_stats` in three adjacent assignments. A forward-string return
annotation keeps the straight-line delegate's runtime loaded-data contract
limited to its shared context. The first slice/pad/concat and staged second
slice/pad/concat boundaries are unchanged, and no reconciliation consumes the
summary yet.

Focused recovery, second slice/pad/concat, and architecture coverage is `101
passed in 0.87s`. The sequential bridge-owner, terminal-affine, complete SPP,
concat-unary-conv, late-SPP, shape-extract, QKV, split/conv bridge,
Hardswish-SE, late hard-activation, SINet terminal, Expand/Squeeze,
late-layout, constant-fold/cast, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `696 passed in 28.11s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, move backward to the first strided-slice/pad/concat call immediately
before this cluster. Its positive-only prune contract permits direct result
capture, but keep the earlier terminal affine recovery unchanged. Commit and
push only; do not create or update a pull request.

## First terminal slice/pad/concat evidence characterization checkpoint

The first direct strided-slice/pad/concat call shares the same positive-only
prune contract as the already staged second call, so its raw counter is complete
mutation evidence. A strict expected-failure contract requires the first call
to assign `_pre_terminal_affine_slice_pad_concat_stats` between the raw
transpose/Mul/Add owner and `terminal_affine_tensor_count`.

The contract also requires exactly two direct owner statements and preserves
the second target as `_terminal_slice_pad_concat_stats`. This prevents another
ambiguous occurrence edit. The earlier terminal affine recovery and all owner
arguments, ordering, and reconciliation behavior remain unchanged.

At implementation, replace only the first direct expression and update the
terminal-affine and bridge-owner boundary contracts. Validate both occurrence
targets, the complete bridge suite, terminal affine recovery, core, pass
efficiency, architecture, and TensorFlow import blocking sequentially. Commit
and push only; do not create or update a pull request.

## First terminal slice/pad/concat evidence implementation checkpoint

The first direct strided-slice/pad/concat invocation now assigns its unchanged
counter to `_pre_terminal_affine_slice_pad_concat_stats`. The second direct
invocation retains the distinct `_terminal_slice_pad_concat_stats` target, and
the architecture contracts verify both exact positions and the shared owner.

This is observation-only staging. The owner still receives only ModelIR, its
transactional guards and positive-only pruning contract are unchanged, the
earlier terminal affine recovery remains untouched, and no reconciliation
decision consumes either staged dictionary.

Focused bridge-owner, terminal-affine, and architecture coverage is `102
passed in 0.87s`. The sequential bridge-owner, terminal-affine, complete SPP,
concat-unary-conv, late-SPP, shape-extract, QKV, split/conv bridge,
Hardswish-SE, late hard-activation, SINet terminal, Expand/Squeeze,
late-layout, constant-fold/cast, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `697 passed in 28.25s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the raw transpose/Mul/Add owner immediately before this
first direct call, or the preceding channel-slice cluster if that owner cannot
provide complete mutation evidence. Characterize its return and prune contract
before changing production. Commit and push only; do not create or update a
pull request.

## Pre-terminal affine post-ADD evidence characterization checkpoint

The indexed affine post-ADD owner returns one fixed rewrite counter and calls
unused-tensor pruning only after at least one accepted rewrite. Its existing
transactional and idempotence contracts, plus an explicit prune-hook test, show
that the raw result is complete mutation evidence without a tensor-count proxy.

There are three direct lowerer calls and one orchestrated occurrence. A strict
expected-failure contract selects only the first direct call, immediately after
the channel-slice cluster and before
`_pre_terminal_affine_slice_pad_concat_stats`, and requires it to assign
`_pre_terminal_affine_post_add_stats`. The two later direct calls must remain
expressions, and owner arguments and LayoutState forwarding are frozen.

At implementation, replace only that exact direct expression, update the
channel-slice and first slice/pad/concat boundary contracts, and preserve the
other direct and orchestrated occurrences. Validate the complete indexed owner,
channel-slice, bridge-owner, terminal-affine, core, pass-efficiency,
architecture, and TensorFlow-import-blocked suites sequentially. Commit and
push only; do not create or update a pull request.

## Pre-terminal affine post-ADD evidence implementation checkpoint

Only the first direct indexed affine post-ADD invocation now assigns its
unchanged result to `_pre_terminal_affine_post_add_stats`. It remains between
the channel-slice cluster and
`_pre_terminal_affine_slice_pad_concat_stats`, and continues to forward the
same ModelIR and Session LayoutState.

The two later direct calls and the terminal-slice orchestration occurrence are
unchanged. No tensor-count proxy is needed because the owner prunes only after
a positive rewrite, and no reconciliation branch consumes the staged result.

Focused indexed-owner, channel-slice, bridge-owner, and terminal-affine
coverage is `165 passed in 1.17s`. The sequential indexed-owner,
channel-slice, bridge-owner, terminal-affine, complete SPP,
concat-unary-conv, late-SPP, shape-extract, QKV, split/conv bridge,
Hardswish-SE, late hard-activation, SINet terminal, Expand/Squeeze,
late-layout, constant-fold/cast, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `761 passed in 28.24s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the preceding channel-slice/pad/Mul orchestration cluster.
Determine whether its runner already propagates complete ordered results and
whether any child can prune without a positive counter before adding mutation
evidence. Commit and push only; do not create or update a pull request.

## Pre-terminal channel-slice/pad-Mul evidence characterization checkpoint

The channel-slice/pad-Mul orchestration has two ordered child results. Empty
ModelIR probing confirms three fixed channel-slice counters followed by one
fixed pad-Mul counter. Source and behavioral owner coverage confirm that all
four underlying rewrites prune only after a positive counter, so no net tensor
count is needed.

Strict expected-failure coverage requires the orchestration runner and lowerer
delegate to return the ordered pair, validates an exact four-key summary, and
requires only the direct terminal invocation to stage
`channel_slice_pad_mul_results` and
`_pre_terminal_channel_slice_pad_mul_stats`. The terminal-slice callback,
shared pass-state scope, Session LayoutState, diagnostics, and pass order remain
unchanged.

At implementation, add the pure fixed-schema summary, propagate the raw pair
through the runner and delegate, and replace only the direct expression with
two assignments. Update the pre-add and pre-terminal affine boundary contracts.
Validate malformed length, result order, the complete channel-slice and pad
owners, terminal-slice recovery, core, pass-efficiency, architecture, and
TensorFlow-import blocking sequentially. Commit and push only; do not create or
update a pull request.

## Pre-terminal channel-slice/pad-Mul evidence implementation checkpoint

The channel-slice/pad-Mul runner and lowerer delegate now return the ordered
two-result tuple. The pure `summarize_channel_slice_pad_mul_mutations()` helper
validates the exact length and extracts only the three channel-slice counters
and one pad-Mul counter.

The direct terminal invocation stages `channel_slice_pad_mul_results` followed
by `_pre_terminal_channel_slice_pad_mul_stats`. The terminal-slice callback
continues to ignore the returned pair, shared pass-state and diagnostics remain
unchanged, and no reconciliation decision consumes the summary.

Focused runner, summary, owner, and boundary coverage is `17 passed in 0.66s`;
terminal-slice callback and architecture coverage is `17 passed in 0.71s`.
The first expanded gate exposed one stale boundary expectation (`782 passed, 1
failed`) after the two new assignments. Updating that exact contract produced
`783 passed in 28.91s` across channel-slice, pad, terminal-slice, callback,
affine, bridge, SPP, QKV, hard-activation, SINet, late-layout, shared-context,
core, pass-efficiency, architecture, and TensorFlow-import-blocked coverage.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the preceding `_optimize_transpose_pre_add_nhwc_chains`
owner. Establish its production occurrence count, return schema, and pruning
contract before choosing an unambiguous evidence target. Commit and push only;
do not create or update a pull request.

## Pre-terminal pre-ADD evidence characterization checkpoint

The composite pre-ADD owner has exactly one direct lowerer call. Its indexed
and compatibility rewrites report one aggregate
`optimized_transpose_pre_add_nhwc_chains` counter, but the compatibility owner
unconditionally prunes at exit. A dedicated zero-rewrite test confirms that an
unused tensor can be removed while the returned counter remains zero.

Strict expected-failure coverage therefore requires adjacent
`pre_terminal_pre_add_tensor_count` and `_pre_terminal_pre_add_stats`
assignments. The summary must preserve the raw counter and add clamped net
`pruned_unused_tensors`. Terminal affine recovery and
`channel_slice_pad_mul_results` are fixed outer boundaries; ModelIR and Session
LayoutState forwarding remain unchanged.

At implementation, replace the unique direct expression with count plus merged
stats assignments. Update the channel-slice and downstream bridge boundary
contracts, but leave the three orchestration-owned occurrences unchanged.
Validate zero-rewrite pruning, the complete indexed/compatibility owner,
channel-slice, terminal-affine, core, pass-efficiency, architecture, and
TensorFlow import blocking sequentially. Commit and push only; do not create or
update a pull request.

## Pre-terminal pre-ADD evidence implementation checkpoint

The unique direct pre-ADD invocation now records
`pre_terminal_pre_add_tensor_count` and assigns `_pre_terminal_pre_add_stats`
from the owner's unchanged aggregate counter plus clamped net
`pruned_unused_tensors`. This captures the compatibility owner's unconditional
cleanup when its rewrite counter is zero.

The three orchestration-owned occurrences remain unchanged. The first focused
run exposed two stale terminal-affine boundary expectations (`135 passed, 2
failed`) that resolved the immediately following `len` call instead of the
logical owner inside the merged dictionary. Updating only those contracts
produced `137 passed in 2.77s`.

The sequential indexed/compatibility pre-ADD, channel-slice, pad,
terminal-slice, callback, affine, bridge, SPP, QKV, hard-activation, SINet,
late-layout, shared-context, core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `806 passed in 29.07s`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, inspect the first direct
`_run_terminal_affine_concat_split_recovery_sequence()` immediately before the
pre-ADD count. The second occurrence already has complete ordered-result and
net-prune evidence; characterize a distinct target for the first occurrence
without changing the shared runner or summary. Commit and push only; do not
create or update a pull request.

## First terminal affine mutation evidence characterization checkpoint

The first terminal affine recovery uses the same ordered eleven-result runner
and fixed twelve-key plus net-prune summary already validated for the second
occurrence. No new summary or child-owner contract is required.

Strict expected-failure coverage requires adjacent
`pre_terminal_affine_tensor_count`, `pre_terminal_affine_results`, and
`_pre_terminal_affine_stats` assignments between the final InstanceNorm
dual-statistics rewrite and `pre_terminal_pre_add_tensor_count`. The second
occurrence must retain its independent `terminal_affine_*` targets.

At implementation, replace only the first recovery expression with the three
assignments and reuse `summarize_terminal_affine_concat_split_mutations()` with
a clamped tensor-count delta. Update both terminal-affine boundary contracts.
Validate both occurrence targets, all eleven child results, pre-ADD,
channel-slice, core, pass-efficiency, architecture, and TensorFlow import
blocking sequentially. Commit and push only; do not create or update a pull
request.

## First terminal affine mutation evidence implementation checkpoint

The first terminal affine recovery now stages
`pre_terminal_affine_tensor_count`, `pre_terminal_affine_results`, and
`_pre_terminal_affine_stats`. It reuses the existing exact eleven-result,
twelve-key summary and records its own clamped net tensor reduction.

The independently staged second recovery retains `terminal_affine_tensor_count`,
`terminal_affine_results`, and `_terminal_affine_stats`. Neither summary is
consumed by reconciliation, and child order, ModelIR/LayoutState forwarding,
and recovery semantics are unchanged.

Focused terminal-affine, pre-ADD, channel-slice, and architecture coverage is
`45 passed in 2.55s`. The sequential indexed/compatibility pre-ADD,
channel-slice, pad, terminal-slice, callback, affine, bridge, SPP, QKV,
hard-activation, SINet, late-layout, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `807 passed in 28.59s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately preceding
`_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains`
owner. Confirm its occurrence count, fixed result schema, and whether pruning
can occur with a zero counter before staging evidence. Commit and push only; do
not create or update a pull request.

## Pre-terminal affine InstanceNorm dual-statistics characterization checkpoint

The indexed dual-statistics InstanceNorm residual/add/resize owner has four
production occurrences, three of them direct top-level calls. Its one fixed
counter is complete mutation evidence because unused-tensor pruning occurs only
after a positive rewrite; an explicit prune-hook test freezes that behavior.

Strict expected-failure coverage selects only the last direct call, between the
terminal residual/Mul/Concat/Conv owner and
`pre_terminal_affine_tensor_count`, and requires
`_pre_terminal_affine_instancenorm_dualstats_stats`. The two earlier direct
calls and nested indexed-convergence occurrence remain unchanged.

At implementation, replace only that exact direct expression, update the first
terminal-affine boundary contracts, and preserve ModelIR and Session LayoutState
forwarding. Validate the complete indexed dual-statistics owner,
terminal-affine, pre-ADD, channel-slice, core, pass-efficiency, architecture,
and TensorFlow import blocking sequentially. Commit and push only; do not create
or update a pull request.

## Pre-terminal affine InstanceNorm dual-statistics implementation checkpoint

Only the last direct dual-statistics InstanceNorm residual/add/resize call now
assigns its unchanged result to
`_pre_terminal_affine_instancenorm_dualstats_stats`. The other two direct calls
and the nested indexed-convergence occurrence retain their previous forms.

The staged counter is complete because pruning remains positive-only. It is
not consumed by reconciliation, and ModelIR/LayoutState forwarding, owner
guards, and first terminal-affine ordering are unchanged.

Focused complete indexed-owner, terminal-affine, and architecture coverage is
`218 passed in 1.34s`. The sequential dual-statistics, indexed/compatibility
pre-ADD, channel-slice, pad, terminal-slice, callback, affine, bridge, SPP,
QKV, hard-activation, SINet, late-layout, shared-context, core,
pass-efficiency, architecture, and TensorFlow-import-blocked gate is `1014
passed in 29.74s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, inspect the immediately preceding
`_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains` owner.
Confirm all occurrence forms and its pruning contract before selecting only the
last direct call for evidence. Commit and push only; do not create or update a
pull request.

## Pre-terminal affine InstanceNorm residual/Mul/Concat characterization checkpoint

The indexed InstanceNorm residual/Mul/Concat/Conv owner has four production
occurrences, three as direct top-level calls. It returns one fixed rewrite
counter and prunes only after a positive rewrite; an explicit prune-hook test
freezes the completeness of that counter.

Strict expected-failure coverage selects only the last direct call between the
terminal InstanceNorm post-bias owner and the staged dual-statistics result. It
requires `_pre_terminal_affine_instancenorm_residual_mul_concat_stats`; the two
earlier direct calls and nested occurrence remain unchanged.

At implementation, replace only that exact expression and update the adjacent
dual-statistics and first terminal-affine boundary contracts. Validate the
complete indexed residual/Mul/Concat owner, dual-statistics, terminal-affine,
pre-ADD, core, pass-efficiency, architecture, and TensorFlow import blocking
sequentially. Commit and push only; do not create or update a pull request.

## Pre-terminal affine InstanceNorm residual/Mul/Concat implementation checkpoint

Only the last direct InstanceNorm residual/Mul/Concat/Conv call now assigns its
unchanged result to
`_pre_terminal_affine_instancenorm_residual_mul_concat_stats`. The two earlier
direct calls and nested indexed-convergence occurrence remain unchanged.

The staged counter is complete because pruning remains positive-only. It is
observation-only, and the following staged dual-statistics owner, first
terminal-affine recovery, and all reconciliation decisions retain their order
and behavior.

Focused complete residual/Mul/Concat, dual-statistics, terminal-affine, and
architecture coverage is `321 passed in 1.59s`. The sequential
residual/Mul/Concat, dual-statistics, indexed/compatibility pre-ADD,
channel-slice, pad, terminal-slice, callback, affine, bridge, SPP, QKV,
hard-activation, SINet, late-layout, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `1117 passed in 29.74s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately preceding
`_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains` owner.
Confirm its occurrence forms, fixed result schema, and prune behavior before
selecting the last direct call for evidence. Commit and push only; do not create
or update a pull request.

## Pre-terminal affine InstanceNorm post-bias characterization checkpoint

The indexed InstanceNorm post-transpose bias/add owner has five production
occurrences: one nested call and four direct top-level calls. Its one fixed
rewrite counter is complete because pruning occurs only after a positive
rewrite; a prune-hook test freezes that behavior.

Strict expected-failure coverage selects the third direct call, between late
binary-layout recovery and the staged residual/Mul/Concat result, and requires
`_pre_terminal_affine_instancenorm_post_bias_stats`. The first two direct calls,
nested occurrence, and absolute-final fourth direct call remain unchanged.

At implementation, replace only the selected expression and update the
residual/Mul/Concat boundary contract. Validate the complete indexed post-bias,
residual/Mul/Concat, dual-statistics, terminal-affine, core, pass-efficiency,
architecture, absolute-final normalization, and TensorFlow import-blocking
suites sequentially. Commit and push only; do not create or update a pull
request.

## Pre-terminal affine InstanceNorm post-bias implementation checkpoint

Only the third direct InstanceNorm post-transpose bias/add call now assigns its
unchanged result to `_pre_terminal_affine_instancenorm_post_bias_stats`. The
nested occurrence, first two direct calls, and absolute-final fourth direct call
retain their existing forms.

The counter is complete because pruning remains positive-only. The result is
not consumed by reconciliation, and the following staged residual/Mul/Concat,
dual-statistics, and first terminal-affine owners retain their exact order.

Focused post-bias, residual/Mul/Concat, dual-statistics, terminal-affine,
absolute-final, and architecture coverage is `399 passed in 1.88s`. The
sequential InstanceNorm owner, absolute-final, pre-ADD, channel-slice, pad,
terminal-slice, callback, affine, bridge, SPP, QKV, hard-activation, SINet,
late-layout, shared-context, core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `1195 passed in 29.59s`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, audit the preceding `late_binary_layout_recovery_stats` interval.
Confirm that its existing mutation counter and conditional reconciliation form
a complete boundary for this newly staged terminal chain. If it does, move to
the absolute-final fourth post-bias call rather than duplicating already staged
evidence. Commit and push only; do not create or update a pull request.

## Late-binary to terminal evidence boundary audit checkpoint

The existing late-binary runner already returns nine fixed mutation counters
plus clamped `pruned_unused_tensors`, while excluding the layout owner's
non-mutating `iterations`. Existing zero-counter pruning coverage confirms the
net tensor delta is required and present.

Production stores that aggregate in `late_binary_layout_recovery_stats` and
reconciles whenever `_stats_have_positive_count()` sees any mutation. A new
architecture contract freezes the direct transition from this guarded branch
to `_pre_terminal_affine_instancenorm_post_bias_stats`. No additional summary
or staging is needed at this boundary.

Focused late-binary runner and boundary coverage passes. At resume, move to the
absolute-final fourth InstanceNorm post-bias call after final signature
sanitization. Characterize its distinct surrounding affine and normalization
owners before capturing its result. Commit and push only; do not create or
update a pull request.

## Absolute-final InstanceNorm post-bias characterization checkpoint

The fourth direct InstanceNorm post-transpose bias/add call is a distinct
absolute-final occurrence after boundary-signature sanitization. Its fixed raw
counter remains complete under the already frozen positive-only pruning
contract.

Strict expected-failure coverage requires
`_absolute_final_instancenorm_post_bias_stats` between the absolute-final affine
post-ADD owner and normalization/attention pair. It also freezes four direct
occurrences, with only the third pre-terminal call already staged and the first
two unchanged.

At implementation, replace only the fourth expression and update the
normalization/attention outer-boundary contract. Validate all post-bias owner
tests, occurrence targets, affine post-ADD, absolute-final normalization,
architecture, core, pass-efficiency, and TensorFlow import blocking
sequentially. Commit and push only; do not create or update a pull request.

## Absolute-final InstanceNorm post-bias implementation checkpoint

Only the fourth direct InstanceNorm post-transpose bias/add call now assigns
its unchanged result to `_absolute_final_instancenorm_post_bias_stats`. The
first two direct calls remain expressions and the third retains its distinct
`_pre_terminal_affine_instancenorm_post_bias_stats` target.

The staged raw counter is complete because the owner prunes only after a
positive rewrite. It remains observation-only: no new reconciliation decision,
tensor-count proxy, pass invocation, graph scan, or artifact-producing action
was added. The absolute-final affine post-ADD owner still runs immediately
before it, and the normalization/attention pair still runs immediately after
it with the same ModelIR and Session LayoutState.

The first focused run produced `79 passed, 1 failed`; the failure was a stale
occurrence-target expectation that still required the fourth call to be an
expression. After updating that contract, focused coverage produced `80
passed`. The first expanded sequential gate produced `1199 passed, 1 failed`;
the remaining failure was a stale architecture boundary expectation. After
correcting that exact boundary, focused coverage produced `9 passed`, and the
expanded sequential gate completed with `1200 passed`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, characterize the immediately preceding absolute-final
`_optimize_transpose_mul_posttranspose_add_nhwc_chains` occurrence. Keep it
distinct from the already staged pre-terminal affine post-ADD occurrence, and
confirm its positive-only pruning contract before considering an
`_absolute_final_affine_post_add_stats` target. Commit and push only; do not
create or update a pull request.

## Absolute-final affine post-ADD characterization checkpoint

The indexed affine post-ADD owner has three direct top-level production calls.
The first already stages `_pre_terminal_affine_post_add_stats`, the second
remains a very-late expression after unbound-input repair, and the third is the
absolute-final call after boundary-signature realignment and sanitization.

The owner returns one fixed rewrite counter and invokes unused-tensor pruning
only when that counter is positive. Existing prune-hook coverage freezes this
contract, so the raw result is complete mutation evidence and needs neither a
tensor-count proxy nor a summary adapter.

Strict expected-failure coverage selects only the third call and requires an
`_absolute_final_affine_post_add_stats` target immediately before
`_absolute_final_instancenorm_post_bias_stats`. It freezes the first staged
target and requires the second call to remain an expression. Focused
absolute-final, indexed-owner, and pre-terminal boundary coverage is `64
passed, 92 deselected, 1 xfailed`.

At implementation, replace only the selected third expression and update the
post-bias and architecture boundary contracts. Preserve the second very-late
expression, ModelIR/LayoutState forwarding, and all pass order. Validate the
indexed affine owner, absolute-final normalization, pre-terminal affine
boundary, core, pass-efficiency, architecture, and TensorFlow import blocking
sequentially. Commit and push only; do not create or update a pull request.

## Absolute-final affine post-ADD implementation checkpoint

Only the third direct affine post-ADD call now assigns its unchanged result to
`_absolute_final_affine_post_add_stats`. The first call retains the existing
`_pre_terminal_affine_post_add_stats` target, and the second very-late call
remains an expression.

The staged raw counter is complete under the owner's positive-only pruning
contract. It is observation-only and does not feed reconciliation. Boundary
signature realignment and sanitization still precede it, and
`_absolute_final_instancenorm_post_bias_stats` still follows immediately with
the same ModelIR and Session LayoutState.

Focused absolute-final, indexed-owner, pre-terminal, and architecture coverage
is `71 passed, 344 deselected`. The expanded sequential indexed-owner,
late/terminal orchestration, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `1201 passed in 29.71s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the remaining second direct affine post-ADD call after
unbound-input repair and immediately before the very-late Gather/Constant
normalization cluster. Characterize its exact boundaries independently before
considering a `_very_late_affine_post_add_stats` target. Commit and push only;
do not create or update a pull request.

## Very-late affine post-ADD characterization checkpoint

The remaining second direct affine post-ADD call is isolated between
`_repair_unbound_nonconstant_operator_inputs_with_layout_transpose` and the
very-late Gather/Constant normalization cluster. It handles strict affine
fragments recreated specifically by the unbound-input repair.

The same indexed owner returns one fixed counter and prunes only after a
positive rewrite, so its raw result is complete without a tensor-count proxy.
Strict expected-failure coverage requires
`_very_late_affine_post_add_stats`, preserves the first
`_pre_terminal_affine_post_add_stats` target and the third
`_absolute_final_affine_post_add_stats` target, and freezes both adjacent
owners. Focused coverage is `72 passed, 92 deselected, 1 xfailed`.

At implementation, replace only the second direct expression and update the
very-late orchestration and architecture boundary contracts. Preserve both
other staged targets, ModelIR/LayoutState forwarding, and pass order. Validate
the indexed owner, all three occurrence contracts, very-late and absolute-final
orchestration, core, pass-efficiency, architecture, and TensorFlow import
blocking sequentially. Commit and push only; do not create or update a pull
request.

## Very-late affine post-ADD implementation checkpoint

The second direct affine post-ADD call now assigns its unchanged result to
`_very_late_affine_post_add_stats`. Together with
`_pre_terminal_affine_post_add_stats` and
`_absolute_final_affine_post_add_stats`, every direct occurrence has a distinct
observation point.

The result remains unused by reconciliation. Unbound-input repair still runs
immediately before it, the very-late Gather/Constant normalization cluster
still runs immediately after it, and ModelIR/LayoutState forwarding and pass
order are unchanged.

Focused three-occurrence, very-late, absolute-final, indexed-owner, and
architecture coverage is `80 passed, 343 deselected`. The expanded sequential
gate, now explicitly including the complete very-late orchestration suite, is
`1209 passed in 30.03s`. Ruff, Python bytecode compilation, and whitespace
validation pass.

At resume, audit result propagation from the immediately following very-late
Gather/Constant normalization cluster. Determine whether its child owners
already expose complete mutation evidence and whether any cleanup-only path
requires net tensor-count accounting before adding a new observation point.
Commit and push only; do not create or update a pull request.

## Very-late normalization mutation characterization checkpoint

The very-late Gather/Constant normalization runner builds four ordered
invocations but currently discards the tuple returned by
`run_recovery_invocations()`. The child schemas comprise one Gather-axis key,
three constant-input fold keys, two redundant-Cast keys, and two
normalization-Pad keys.

The flatten global-normalization Pad owner unconditionally prunes unused
tensors even when its rewrite count is zero. A prune-hook contract now freezes
that cleanup-only path. Therefore a complete cluster summary requires all eight
fixed mutation keys plus a clamped net tensor reduction measured around the
whole cluster.

Three strict expected-failure contracts require the runner's ordered tuple, a
fixed-schema
`summarize_very_late_gather_constant_normalization_mutations()` helper, and
lowerer assignments for `very_late_normalization_tensor_count`,
`very_late_normalization_results`, and `_very_late_normalization_stats` before
the final dynamic-Reshape resolution. Focused characterization is `9 passed, 3
xfailed`.

At implementation, return the existing ordered tuple without another pass,
validate exactly four results in the pure summary, and measure only the
cluster-wide tensor delta in the lowerer. Keep the summary observation-only;
do not add reconciliation or alter the shared pass-state scope. Validate
very-late orchestration, constant-fold/Cast, normalization-Pad, shared-context,
core, pass-efficiency, architecture, and TensorFlow import blocking
sequentially. Commit and push only; do not create or update a pull request.

## Very-late normalization mutation implementation checkpoint

`run_very_late_gather_constant_normalization()` now returns the existing
ordered four-result tuple from `run_recovery_invocations()`. It does not rebuild
the invocation list or rerun an owner.

The pure
`summarize_very_late_gather_constant_normalization_mutations()` helper requires
exactly four results, extracts only the eight declared mutation keys, defaults
missing keys to zero, and adds clamped `pruned_unused_tensors`. Wrong result
counts raise `ValueError`.

The lowerer records `very_late_normalization_tensor_count`, stages
`very_late_normalization_results`, and builds `_very_late_normalization_stats`
from the cluster-wide tensor delta. The summary is observation-only. The same
shared `ModelIRPassStateScope`, child order, ModelIR/LayoutState/diagnostics
objects, and following dynamic-Reshape resolution are preserved.

Focused runner, summary, cleanup-only pruning, lowerer, architecture, and
pass-state coverage is `17 passed`. The expanded sequential indexed-owner,
late/terminal orchestration, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `1216 passed in 30.03s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately following
`_resolve_dynamic_reshape_shapes(...,
prefer_runtime_inferable_from_onnx_raw=True)` owner. Confirm its fixed result
schema, occurrence count, and cleanup behavior before selecting this very-late
call for an independent observation point. Commit and push only; do not create
or update a pull request.

## Very-late dynamic-Reshape characterization checkpoint

The dynamic-Reshape resolver has four lowerer occurrences. Two are already
captured inside indexed convergence helpers. Two are direct top-level calls:
an earlier core-cleanup expression and the distinct very-late call with
`prefer_runtime_inferable_from_onnx_raw=True` immediately after
`_very_late_normalization_stats`.

The owner returns one fixed `resolved_dynamic_reshape_shapes` counter. Source
and behavior coverage confirm that it updates RESHAPE options, the optional
shape tensor, and output shape metadata, counts each changed operator, and does
not prune tensors or remove operators. Its raw result is therefore complete
mutation evidence.

Strict expected-failure coverage requires only the very-late direct call to
assign `_very_late_dynamic_reshape_stats`, preserves the earlier direct
expression, and freezes the following indexed Conv-input adapter boundary.
Focused very-late and dynamic-Reshape coverage is `28 passed, 1 xfailed`.

At implementation, replace only that selected expression and update the
very-late and architecture boundary contracts. Do not add a tensor-count
proxy, summary adapter, or reconciliation consumer. Validate dynamic-Reshape,
very-late normalization, indexed convergence, core, pass-efficiency,
architecture, and TensorFlow import blocking sequentially. Commit and push
only; do not create or update a pull request.

## Very-late dynamic-Reshape implementation checkpoint

Only the direct dynamic-Reshape call with
`prefer_runtime_inferable_from_onnx_raw=True` now assigns its unchanged result
to `_very_late_dynamic_reshape_stats`. The earlier direct core-cleanup call
remains an expression, and the two indexed convergence helpers continue to
consume their local `reshape_stats` results.

The staged single counter is complete because the owner performs no pruning or
topology removal. It remains observation-only immediately after
`_very_late_normalization_stats` and immediately before the indexed Conv-input
adapter repair. No summary, tensor-count proxy, or reconciliation branch was
added.

Focused very-late, dynamic-Reshape, indexed-convergence, and architecture
coverage is `44 passed, 256 deselected`. The expanded sequential gate,
explicitly including complete dynamic-Reshape and indexed-final-convergence
coverage, is `1243 passed in 30.55s`. Ruff, Python bytecode compilation, and
whitespace validation pass.

At resume, inspect the immediately following
`_run_indexed_conv_input_adapter_repairs(model_ir)` call. Compare it with the
fallback-path `fallback_conv_input_stats` occurrence, confirm its result schema
and pruning behavior, and characterize only the direct very-late call if its
raw result is complete. Commit and push only; do not create or update a pull
request.

## Very-late indexed Conv-input repair characterization checkpoint

The indexed Conv-input adapter runner returns two fixed counters for singleton
Reshape and stale Transpose repairs. Both child owners unconditionally invoke
unused-tensor pruning, including zero-rewrite calls. A prune-hook contract now
freezes the two cleanup opportunities, so the raw counters alone are not
complete mutation evidence.

Strict expected-failure coverage requires
`very_late_conv_input_tensor_count` followed by
`_very_late_conv_input_stats`, which spreads the unchanged runner result and
adds clamped `pruned_unused_tensors`. It selects only the direct call after
`_very_late_dynamic_reshape_stats`; the fallback-path
`fallback_conv_input_stats` assignment remains unchanged.

Focused indexed-owner and very-late coverage is `25 passed, 1 xfailed`. The
focused run also exposed a stale test assumption that legacy lowerer map-builder
attributes still existed. The sentinel monkeypatch now uses `raising=False`,
which preserves detection if either legacy builder is reintroduced and called.

At implementation, measure the tensor delta only around the direct call and
build the fixed dictionary in place. Do not alter the runner, fallback path,
following stale-channel-shuffle owner, or reconciliation decisions. Validate
indexed Conv-input repairs, very-late orchestration, dynamic-Reshape, core,
pass-efficiency, architecture, and TensorFlow import blocking sequentially.
Commit and push only; do not create or update a pull request.

## Very-late indexed Conv-input repair implementation checkpoint

The direct call now records `very_late_conv_input_tensor_count` and builds
`_very_late_conv_input_stats` by spreading the runner's unchanged two counters
and adding clamped `pruned_unused_tensors`.

This captures cleanup-only pruning from both child owners without rerunning
either repair. The fallback-path `fallback_conv_input_stats` assignment and its
existing reconciliation decision are unchanged. The direct summary remains
observation-only between `_very_late_dynamic_reshape_stats` and the stale NCHW
channel-shuffle repair.

Focused indexed-owner, legacy-compatibility, dynamic-Reshape, very-late, and
architecture coverage is `42 passed, 255 deselected`. The expanded sequential
gate, explicitly including indexed Conv-input and legacy Conv-layout suites, is
`1265 passed in 30.75s`. Ruff, Python bytecode compilation, and whitespace
validation pass.

At resume, inspect the immediately following
`run_stale_nchw_channel_shuffle_repair()` owner. Confirm its result schema,
pruning behavior, and production occurrence count before selecting the
very-late call for an independent observation point. Commit and push only; do
not create or update a pull request.

## Very-late stale channel-shuffle characterization checkpoint

`run_stale_nchw_channel_shuffle_repair()` has one production occurrence. Its
single `repaired_nchw_channel_shuffle_concat_gathers` counter covers all
changes to the Concat axis and related tensor metadata. The owner performs no
pruning and removes no operators, so the raw result is complete mutation
evidence.

Strict expected-failure coverage requires
`_very_late_stale_channel_shuffle_stats` immediately after
`_very_late_conv_input_stats`, preserves ModelIR/LayoutState/diagnostics
forwarding, and freezes the following NCHW Concat/Transpose/Conv axis repair.
Focused shuffle-owner and very-late coverage is `21 passed, 1 xfailed`.

At implementation, replace only the direct expression with the assignment. Do
not add a tensor-count proxy, summary adapter, reconciliation consumer, or
second pass invocation. Validate shuffle layout, very-late orchestration, core,
pass-efficiency, architecture, and TensorFlow import blocking sequentially.
Commit and push only; do not create or update a pull request.

## Very-late stale channel-shuffle implementation checkpoint

The sole production call now assigns its unchanged result to
`_very_late_stale_channel_shuffle_stats`. Its single counter remains complete
because the owner neither prunes tensors nor changes topology.

The result is observation-only immediately after
`_very_late_conv_input_stats`. The following NCHW
Concat/Transpose/Conv-axis repair, argument forwarding, diagnostics, and pass
count are unchanged.

Focused shuffle-owner, very-late, and architecture coverage is `27 passed, 253
deselected`. The expanded sequential gate, explicitly including the complete
shuffle-layout suite, is `1270 passed in 30.20s`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, inspect the immediately following
`_repair_nchw_concat_transpose_conv_axes(model_ir)` owner. Confirm its result
schema, occurrence count, and pruning behavior before selecting the very-late
call for an observation point. Commit and push only; do not create or update a
pull request.

## Very-late Concat/Transpose/Conv-axis characterization checkpoint

The NCHW Concat/Transpose/Conv-axis owner has three production occurrences: a
very-late direct expression, `fallback_concat_axis_stats`, and
`final_concat_axis_stats`. Its single
`repaired_nchw_concat_transpose_conv_axes` counter covers every applied repair
plan. The owner changes only Concat options and tensor metadata and performs no
pruning or topology mutation.

Strict expected-failure coverage selects only the direct occurrence for
`_very_late_concat_transpose_conv_axis_stats`, preserves Session LayoutState
forwarding, and freezes the following Concat/global-pool/Conv-axis owner. It
also requires the fallback and final targets to remain present. Focused indexed
owner, legacy Conv-layout, and very-late coverage is `58 passed, 1 xfailed`.

At implementation, replace only the direct expression with the assignment. Do
not add a proxy, summary, reconciliation consumer, or another owner invocation.
Validate indexed Concat/Transpose/Conv layout, legacy Conv layout, very-late
orchestration, core, pass-efficiency, architecture, and TensorFlow import
blocking sequentially. Commit and push only; do not create or update a pull
request.

## Very-late Concat/Transpose/Conv-axis implementation checkpoint

Only the direct very-late occurrence now assigns its unchanged result to
`_very_late_concat_transpose_conv_axis_stats`. The fallback and final
assignments retain their existing targets and reconciliation decisions.

The single raw counter is complete because the owner neither prunes tensors nor
changes topology. The result remains observation-only between
`_very_late_stale_channel_shuffle_stats` and the NCHW
Concat/global-pool/Conv-axis repair.

Focused indexed-owner, legacy Conv-layout, very-late, and architecture coverage
is `63 passed, 254 deselected`. The expanded sequential gate, explicitly
including the indexed Concat/Transpose/Conv suite, is `1299 passed in 30.62s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately following
`_repair_nchw_concat_global_pool_conv_axes(model_ir)` owner. Confirm its result
schema, production occurrence count, and pruning behavior before selecting the
very-late direct call. Commit and push only; do not create or update a pull
request.

## Very-late Concat/global-pool/Conv-axis characterization checkpoint

The NCHW Concat/global-pool/Conv-axis owner has one production occurrence. Its
single `repaired_nchw_concat_global_pool_conv_axes` counter covers every
applied plan. The owner changes only Concat options and tensor metadata and
performs no pruning or topology mutation.

Strict expected-failure coverage requires
`_very_late_concat_global_pool_conv_axis_stats` immediately after
`_very_late_concat_transpose_conv_axis_stats`, preserves Session LayoutState
forwarding, and freezes the following dynamic rank-one
Unsqueeze/Reshape-shape rewrite. Focused indexed owner, legacy Conv-layout, and
very-late coverage is `64 passed, 1 xfailed`.

At implementation, replace only the direct expression with the assignment. Do
not add a proxy, summary, reconciliation consumer, or another owner invocation.
Validate indexed Concat/global-pool layout, legacy Conv layout, very-late
orchestration, core, pass-efficiency, architecture, and TensorFlow import
blocking sequentially. Commit and push only; do not create or update a pull
request.

## Very-late Concat/global-pool/Conv-axis implementation checkpoint

The sole production call now assigns its unchanged result to
`_very_late_concat_global_pool_conv_axis_stats`. Its single raw counter remains
complete because the owner neither prunes tensors nor changes topology.

The result is observation-only immediately after
`_very_late_concat_transpose_conv_axis_stats` and before the dynamic rank-one
Unsqueeze/Reshape-shape rewrite. ModelIR/LayoutState forwarding and pass count
are unchanged.

Focused indexed-owner, legacy Conv-layout, very-late, and architecture coverage
is `69 passed, 254 deselected`. The expanded sequential gate, explicitly
including indexed Concat/global-pool coverage, is `1333 passed in 31.23s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately following
`_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs()` owner. Confirm its
result schema, occurrence forms, and cleanup behavior before selecting this
very-late call for a distinct observation point. Commit and push only; do not
create or update a pull request.

## Very-late dynamic rank-one Reshape characterization checkpoint

The dynamic rank-one Unsqueeze/Reshape-shape owner has three production
occurrences: the current very-late direct call, a fallback expression, and an
absolute-final expression. Its single
`rewritten_dynamic_rank1_unsqueeze_reshape_shape_inputs` counter covers both
the indexed runtime SHAPE/Concat insertion path and the metadata-only
higher-rank fallback. The owner performs no pruning.

Strict expected-failure coverage selects only the first direct call for
`_very_late_dynamic_rank1_reshape_stats`, preserves Session LayoutState
forwarding, and freezes the following static-shape reconciliation. It also
requires the fallback and absolute-final calls to remain expressions with their
existing target ModelIRs. Focused dynamic-Reshape and very-late coverage is `33
passed, 1 xfailed`.

At implementation, replace only the selected direct expression with the
assignment. Do not add a proxy, summary, reconciliation consumer, or another
owner invocation. Validate dynamic-Reshape, very-late orchestration, fallback
and absolute-final occurrence contracts, core, pass-efficiency, architecture,
and TensorFlow import blocking sequentially. Commit and push only; do not
create or update a pull request.

## Very-late dynamic rank-one Reshape implementation checkpoint

Only the first very-late dynamic rank-one Unsqueeze/Reshape-shape call now
assigns its unchanged result to `_very_late_dynamic_rank1_reshape_stats`. The
fallback and absolute-final calls remain expressions targeting `fallback_ir`
and `model_ir`, respectively.

The raw counter is complete because it covers both operator/tensor insertion
and metadata-only rewrites and the owner performs no pruning. The result remains
observation-only before the unchanged static-shape reconciliation. The
occurrence contract compares the two remaining expression targets without
depending on unspecified `ast.walk()` traversal order.

Focused dynamic-Reshape, very-late, absolute-final, and architecture coverage
is `51 passed, 250 deselected`. The expanded sequential gate is `1334 passed in
30.38s`. Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately following direct
`_reconcile_static_tensor_shapes(model_ir)` occurrence. Distinguish it from the
many conditional and convergence-owned reconciliations, confirm its result
schema and mutation completeness, and characterize only this very-late
boundary. Commit and push only; do not create or update a pull request.

## Very-late static reconciliation characterization checkpoint

The legacy `reconciled_static_tensor_shapes` key counts updates applied through
the output-tensor shape helper. It is not complete mutation evidence: a focused
RESHAPE fixture begins with correct output metadata but a stale `newShape`
option, and reconciliation corrects that option while returning the unchanged
legacy value `0`.

To preserve every existing caller and exact-dictionary contract, strict
expected-failure coverage requires an opt-in `include_mutation_count=True`
parameter. Only that mode adds `reconciled_static_shape_mutations`, covering
output shapes, constant shape parameters, operator options, and direct tensor
metadata updates. Default calls must continue returning only the legacy key.

A second strict expected-failure contract selects only the direct
reconciliation after `_very_late_dynamic_rank1_reshape_stats`, requires
`_very_late_static_shape_stats`, and freezes the following
`split_fallback_stats` assignment. Focused characterization is `24 passed, 2
xfailed`.

At implementation, instrument mutation sites during the existing fixed-point
walk; do not add a pre/post ModelIR fingerprint or another graph traversal.
Return the extra key only when requested, and enable it only at the selected
very-late call. Validate parameter-only mutation, all static-shape
reconciliation suites, dynamic-Reshape, very-late orchestration, core,
pass-efficiency, architecture, and TensorFlow import blocking sequentially.
Commit and push only; do not create or update a pull request.

## Very-late static reconciliation implementation checkpoint

`reconcile_static_tensor_shapes()` now accepts
`include_mutation_count=False`. The default path preserves the exact legacy
dictionary. Opt-in mode additionally returns
`reconciled_static_shape_mutations`.

Mutation accounting is embedded in the existing fixed-point walk. Small local
setters count output shape updates, operator-option changes, constant
shape-parameter writes, direct shape-signature updates, and constant-vector
metadata changes. No ModelIR fingerprint, copy, pre/post graph scan, or new
dependency is introduced.

Only the direct reconciliation after
`_very_late_dynamic_rank1_reshape_stats` enables the option and stages
`_very_late_static_shape_stats`. Other callers, including indexed convergence
and fallback/final paths, retain the legacy one-key schema. The focused
reconciliation/very-late gate is `47 passed, 250 deselected`; the complete
static reconciliation, convergence, core, architecture, and import-boundary
gate is `428 passed in 27.50s`; and the expanded sequential gate is `1339
passed in 30.54s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, audit the immediately following `split_fallback_stats` owner and its
conditional reconciliation. Confirm whether its raw rewrite counter covers any
cleanup and whether the existing positive guard is complete before changing
that boundary. Commit and push only; do not create or update a pull request.

## Post-Split fallback reconciliation characterization checkpoint

`replace_unsupported_split_with_slice()` has one complete mutation predicate:
`replaced_unsupported_split_with_slice`. The owner creates tensors and replaces
operators only when that counter increments, and both tensor pruning and
`LayoutState` synchronization are nested under `rewritten > 0`. A new no-op
fixture proves that a zero result invokes neither maintenance path; a positive
fixture proves that both paths execute exactly once. The lowerer's existing
positive guard is therefore sound and must not be broadened or removed.

The remaining evidence gap is the result of the guarded
`_reconcile_static_tensor_shapes(model_ir)` call. A strict expected-failure
architecture contract requires a stable two-key zero default and, on a
positive Split rewrite, an assigned opt-in complete reconciliation result.
The assignment must use `_post_split_fallback_static_shape_stats`, preserve the
existing guard expression and order, and request `include_mutation_count=True`.

Focused owner/orchestration characterization is `6 passed, 278 deselected, 1
xfailed`. The broader sequential Split, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `432
passed, 1 xfailed in 26.37s`. Ruff and whitespace validation pass.

At implementation, change only that result plumbing. Do not change the Split
matcher, rewrite, raw result schema, pruning, GraphIndex updates, LayoutState
sync, or conditional execution. Run Split fallback, static reconciliation,
very-late orchestration, core, pass-efficiency, architecture, and TensorFlow
import-blocking tests sequentially. Commit and push only; do not create or
update a pull request.

## Post-Split fallback reconciliation implementation checkpoint

The Split owner, its exact one-key result, and the existing positive guard are
unchanged. The lowerer now initializes
`_post_split_fallback_static_shape_stats` with both complete reconciliation
keys set to zero. Only a positive
`replaced_unsupported_split_with_slice` value replaces that default by calling
`_reconcile_static_tensor_shapes(model_ir, include_mutation_count=True)`.

This is observation-only result plumbing. It adds no graph scan, fingerprint,
copy, dependency, rewrite, pruning, layout synchronization, or unconditional
maintenance work. The strict expected-failure contract is now green while the
zero/positive Split owner fixtures continue to prove that the raw rewrite
counter fully covers its maintenance paths.

Focused Split/orchestration/architecture validation is `7 passed, 278
deselected`. The broader sequential Split, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `433
passed in 26.91s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, audit the safety-fallback boundary beginning with
`_find_unbound_nonconstant_operator_inputs(model_ir)`. Characterize the
fallback-only reconciliation results and recursive relowering boundaries
before changing them; do not alter fallback eligibility or broaden inference
validation. Commit and push coherent units only, and do not create or update a
pull request.

## Safety-fallback norm evidence characterization checkpoint

The first fallback-only owner after recursive relowering is the norm-only
`run_pad_layout_cleanup()` call. Its child
`_optimize_transpose_norm_subgraph_pad_prepost_nhwc_chains()` unconditionally
prunes unused tensors after its fixed-point loop. A focused fixture proves that
an empty rewrite can remove an unused constant while returning
`optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains: 0`; the raw result
is therefore incomplete mutation evidence.

A strict expected-failure lowerer contract requires
`fallback_norm_tensor_count` immediately before the existing call and a
`pruned_unused_tensors` clamped net delta merged into `fallback_norm_stats`.
The existing norm-rewrite guard, its three child owners, reconciliation,
topological sort, recursive relowering arguments, and all later fallback work
remain unchanged.

Focused safety-fallback characterization is `1 passed, 1 xfailed`. The broader
sequential fallback-owner, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `445
passed, 1 xfailed in 27.33s`. Ruff and whitespace validation pass.

At implementation, add only this observation point. Do not use the new cleanup
counter to broaden the current reconciliation guard: pruning unused tensors
alone does not require shape reconciliation. Validate the direct norm owner,
Pad orchestration, safety-fallback AST contract, pass efficiency, architecture,
core, and TensorFlow import blocking sequentially. Commit and push only; do not
create or update a pull request.

## Safety-fallback norm evidence implementation checkpoint

The norm-only fallback call now samples `fallback_norm_tensor_count` and
merges a clamped `pruned_unused_tensors` delta into the unchanged owner result.
This makes `fallback_norm_stats` complete for both rewrite and cleanup-only
ModelIR mutations without rerunning the owner or scanning the graph.

The following guard still reads only
`optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains`. Consequently a
prune-only result remains observation-only and does not start binary repairs,
singleton/Reshape cleanup, shape reconciliation, or topological sorting. Pass
order, recursive relowering, diagnostics, and all later fallback work are
unchanged.

Focused safety-fallback and singleton/Reshape orchestration coverage is `13
passed`. The broader sequential fallback-owner, reconciliation, convergence,
core, pass-efficiency, architecture, and TensorFlow import-blocking gate is
`446 passed in 27.35s`. Ruff, Python bytecode compilation, and whitespace
validation pass.

At resume, inspect the immediately following fallback dynamic rank-one
Unsqueeze/Reshape-shape call. Characterize its result and the unconditional
topological/layout refresh separately; do not skip either refresh until the
recursive return-state and all preceding fallback mutation evidence are proven
complete. Commit and push only; do not create or update a pull request.

## Safety-fallback dynamic rank-one characterization checkpoint

`rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs()` returns one complete
counter. It increments for both the folded higher-rank metadata repair and the
indexed runtime SHAPE/CONCAT insertion path, synchronizes layout only after a
positive rewrite, and performs no tensor or operator cleanup on zero.

The fallback call still discards that result. A strict expected-failure AST
contract requires `_fallback_dynamic_rank1_stats` while preserving its sole
`fallback_ir` argument and the immediately following unconditional
`_topologically_sort_operators()` and `infer_model_ir_logical_layouts()` calls.
No refresh is skipped in this unit because recursive relowering and earlier
fallback repairs have not yet been proven to leave terminal layout metadata
equivalent on every zero-owner path.

Focused dynamic-rank-one characterization is `4 passed, 34 deselected, 1
xfailed`. The broader sequential fallback-owner, reconciliation, convergence,
core, pass-efficiency, architecture, and TensorFlow import-blocking gate is
`446 passed, 1 xfailed in 26.72s`. Ruff and whitespace validation pass.

At implementation, assign only this result and update the existing
three-occurrence contract so the absolute-final `model_ir` call remains the
sole expression. Validate dynamic Reshape, safety fallback, very-late
orchestration, core, pass efficiency, architecture, and TensorFlow import
blocking sequentially. Commit and push only; do not create or update a pull
request.

## Safety-fallback dynamic rank-one implementation checkpoint

The fallback-only call now assigns its unchanged one-key result to
`_fallback_dynamic_rank1_stats`. The immediately following topological sort and
logical-layout inference remain unconditional and in their original order.
The very-late main-path result remains `_very_late_dynamic_rank1_reshape_stats`,
and the absolute-final `model_ir` call remains the sole discarded expression.

This is observation-only plumbing: no matcher, metadata write, GraphIndex
operation, LayoutState behavior, recursive relowering, refresh, or return path
changed.

Focused dynamic-rank-one validation is `5 passed, 34 deselected`. The broader
sequential fallback-owner, reconciliation, convergence, core, pass-efficiency,
architecture, and TensorFlow import-blocking gate is `447 passed in 27.34s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, determine whether the recursive fallback return state plus complete
norm and dynamic-rank-one evidence is sufficient to guard either refresh. If
that equivalence cannot be proven locally, leave both unconditional and audit
the following `fallback_broadcast_repair_stats` reconciliation result instead.
Commit and push only; do not create or update a pull request.

## Safety-fallback broadcast reconciliation characterization checkpoint

`repair_rank4_channelwise_broadcast_constants_to_runtime_layout()` mutates
constant data/metadata or creates and rewires a clone only when
`repaired_rank4_channelwise_broadcast_constants` increments. It performs no
pruning or zero-rewrite cleanup, and existing positive, no-op, shared-constant,
and convergence fixtures cover its one-key result. The fallback guard is
therefore a complete mutation predicate.

The guarded static-shape reconciliation result is still discarded. A strict
expected-failure AST contract requires a stable two-key zero default named
`_fallback_broadcast_static_shape_stats` and an opt-in complete result on the
positive path. The existing guard, topological sort, logical-layout inference,
and following SE/FC/Gather recovery order remain unchanged.

Focused safety-fallback/broadcast/convergence characterization is `17 passed,
1 xfailed`. The broader sequential fallback-owner, reconciliation,
convergence, core, pass-efficiency, architecture, and TensorFlow
import-blocking gate is `450 passed, 1 xfailed in 27.02s`. Ruff and whitespace
validation pass.

At implementation, add only this reconciliation result plumbing. Do not alter
the broadcast matcher, clone behavior, raw result schema, convergence owner,
guard, or refresh calls. Validate binary layout, safety fallback, static
reconciliation, convergence, core, pass efficiency, architecture, and
TensorFlow import blocking sequentially. Commit and push only; do not create or
update a pull request.

## Safety-fallback broadcast reconciliation implementation checkpoint

The fallback broadcast owner, its exact one-key result, and its positive guard
are unchanged. The lowerer initializes
`_fallback_broadcast_static_shape_stats` with both complete reconciliation keys
set to zero. A positive broadcast rewrite replaces that default with
`_reconcile_static_tensor_shapes(fallback_ir,
include_mutation_count=True)` before the unchanged topological sort and
logical-layout inference.

This result plumbing adds no graph scan, fingerprint, copy, rewrite, cleanup,
dependency, or unconditional work. The broadcast matcher, shared-constant
clone path, convergence owner, and all later fallback processing retain their
existing behavior.

Focused safety-fallback/broadcast/convergence validation is `18 passed`. The
broader sequential fallback-owner, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `451
passed in 27.03s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, audit the following fallback SINet-shuffle plus SE/FC/Gather
aggregate. Confirm that its existing tensor-count delta covers cleanup-only
paths and characterize the guarded reconciliation result without changing the
combined predicate. Commit and push only; do not create or update a pull
request.

## Safety-fallback SE/FC/Gather reconciliation characterization checkpoint

The fallback boundary samples `fallback_se_fc_gather_tensor_count` before the
SINet shuffle and SE/FC/Gather owners. Its predicate combines all three exact
rewrite counters with `len(fallback_ir.tensors) <
fallback_se_fc_gather_tensor_count`, so zero-rewrite pruning is also covered.
This is complete mutation evidence for the existing reconciliation guard.

The guarded reconciliation return is discarded. A strict expected-failure AST
contract requires `_fallback_se_fc_gather_static_shape_stats`, initialized with
both zero keys and replaced by an opt-in complete result only when the existing
combined predicate is true. It freezes every predicate term and the following
placeholder-MatMul boundary.

Focused safety-fallback and SE/FC/Gather characterization is `18 passed, 1
xfailed`. The broader sequential fallback-owner, reconciliation, convergence,
core, pass-efficiency, architecture, and TensorFlow import-blocking gate is
`462 passed, 1 xfailed in 27.24s`. Ruff and whitespace validation pass.

At implementation, add only this result plumbing. Do not change the SINet,
SE/FC, Gather, pruning, tensor-delta, cluster, or guard contracts. Validate the
complete owner suites, safety fallback, static reconciliation, core, pass
efficiency, architecture, and TensorFlow import blocking sequentially. Commit
and push only; do not create or update a pull request.

## Safety-fallback SE/FC/Gather reconciliation implementation checkpoint

The fallback SINet-shuffle and SE/FC/Gather owners, their result schemas, the
before/after tensor count, and all four combined predicate terms are unchanged.
The lowerer initializes `_fallback_se_fc_gather_static_shape_stats` with both
complete reconciliation keys set to zero and replaces it with the opt-in
complete result only when the established predicate is true.

This adds no scan, fingerprint, copy, rewrite, cleanup, dependency, or
unconditional reconciliation. The following placeholder-MatMul restore
boundary remains immediately adjacent and unchanged.

Focused safety-fallback and SE/FC/Gather validation is `19 passed`. The broader
sequential fallback-owner, reconciliation, convergence, core, pass-efficiency,
architecture, and TensorFlow import-blocking gate is `463 passed in 27.00s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, audit the placeholder-MatMul restore owner and its guarded
reconciliation. Confirm its positive-only pruning contract and characterize
the discarded reconciliation result without changing the guard. Commit and
push only; do not create or update a pull request.

## Safety-fallback placeholder-MatMul characterization checkpoint

`restore_placeholder_matmul_flattened_inputs()` rewires a proven
placeholder-Reshape consumer and removes that Reshape through the differential
GraphIndex. Its unused-tensor pruning runs only after `restored > 0`, so
`restored_placeholder_matmul_flattened_inputs` is complete mutation evidence.

The fallback lowerer currently invokes the owner inline in its guard and
discards both its result and the guarded reconciliation result. A strict
expected-failure AST contract requires one
`fallback_placeholder_matmul_stats` assignment, a stable two-key
`_fallback_placeholder_matmul_static_shape_stats` default, and an opt-in
complete reconciliation assignment under the unchanged positive predicate.
The immediately following topological sort remains fixed.

Focused fallback placeholder-MatMul characterization is `7 passed, 11
deselected, 1 xfailed`. The broader sequential fallback-owner, reconciliation,
convergence, core, pass-efficiency, architecture, and TensorFlow
import-blocking gate is `463 passed, 1 xfailed in 27.32s`. Ruff and whitespace
validation pass.

At implementation, preserve a single owner invocation and change no matcher,
GraphIndex, pruning, guard, or sort behavior. Validate dynamic Reshape,
safety fallback, static reconciliation, core, pass efficiency, architecture,
and TensorFlow import blocking sequentially. Commit and push only; do not
create or update a pull request.

## Safety-fallback placeholder-MatMul implementation checkpoint

The fallback owner now runs exactly once into
`fallback_placeholder_matmul_stats`. The lowerer initializes
`_fallback_placeholder_matmul_static_shape_stats` with both complete zero keys
and replaces it with the opt-in reconciliation result only when
`restored_placeholder_matmul_flattened_inputs` is positive.

No matcher, GraphIndex edit, positive-only pruning, guard, reconciliation
eligibility, or following topological sort changed. The former inline call is
only made explicit and its two results are retained.

Focused fallback placeholder-MatMul validation is `8 passed, 11 deselected`.
The broader sequential fallback-owner, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `464
passed in 26.93s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, audit the following fallback unbound-input repair owner and its
guarded reconciliation. Confirm whether it performs cleanup or layout
maintenance outside its rewrite counter before selecting that boundary. Commit
and push only; do not create or update a pull request.

## Safety-fallback unbound reconciliation characterization checkpoint

`_repair_unbound_nonconstant_operator_inputs_with_layout_transpose()` already
owns static-shape reconciliation after a positive indexed repair and forwards
the repair's current GraphIndex. A new positive fixture proves exactly one
wrapper-owned reconciliation and confirms that the same ModelIR index is
reused.

The fallback caller nevertheless checks the returned counter and immediately
runs a second reconciliation, with no intervening graph or metadata mutation.
A strict expected-failure AST contract requires
`fallback_conv_input_stats` to follow `_fallback_unbound_repair_stats` directly,
removing only this duplicate full-graph scan. The owner invocation, wrapper
reconciliation, return schema, following Conv-input owner, and all guards after
it remain unchanged.

Focused indexed-unbound and safety-fallback characterization is `11 passed, 1
xfailed`. It also exposed a stale test assumption that removed legacy
producer/consumer map-builder attributes still existed; the sentinel
monkeypatches now use `raising=False` and still fail if either builder is
reintroduced and called. The broader sequential fallback-owner,
reconciliation, convergence, core, pass-efficiency, architecture, and
TensorFlow import-blocking gate is `469 passed, 1 xfailed in 27.76s`. Ruff and
whitespace validation pass.

At implementation, delete only the redundant caller-side guard and
reconciliation. Validate positive/no-op unbound repair, safety fallback,
static reconciliation, core, pass efficiency, architecture, and TensorFlow
import blocking sequentially. Commit and push only; do not create or update a
pull request.

## Safety-fallback unbound reconciliation implementation checkpoint

The redundant fallback caller-side guard and reconciliation are removed. A
positive unbound-input repair still performs exactly one static reconciliation
inside `_repair_unbound_nonconstant_operator_inputs_with_layout_transpose()`,
using the repair's live GraphIndex. The fallback then proceeds directly from
`_fallback_unbound_repair_stats` to `fallback_conv_input_stats`.

No indexed matcher, inserted transpose, wrapper result schema, GraphIndex,
no-op behavior, or following Conv-input repair changed. This is a strict
one-scan elimination with no intervening mutation to preserve.

Focused indexed-unbound and safety-fallback validation is `12 passed`. The
broader sequential fallback-owner, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `470
passed in 27.49s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, audit `fallback_conv_input_stats`. Its two child owners can prune
on zero rewrite, so do not treat the existing stale-transpose counter as
complete without tensor-delta evidence. Commit and push only; do not create or
update a pull request.

## Safety-fallback Conv-input evidence characterization checkpoint

`_run_indexed_conv_input_adapter_repairs()` returns exact counters for
singleton-Reshape and stale-Transpose repairs, but each child unconditionally
prunes unused tensors. Existing zero-rewrite coverage proves two cleanup
opportunities, so the raw dictionary is incomplete mutation evidence.

A strict expected-failure safety-fallback contract requires
`fallback_conv_input_tensor_count`, a clamped `pruned_unused_tensors` delta
merged into `fallback_conv_input_stats`, and a stable two-key
`_fallback_conv_input_static_shape_stats`. The existing reconciliation guard
continues to read only
`repaired_stale_nchw_to_nhwc_conv_input_transposes`; its opt-in complete result
is assigned instead of discarded. Singleton repair already writes the Conv
output shape/signature directly, and prune-only cleanup does not broaden this
guard. The following mixed-Concat owner remains fixed.

Focused Conv-input and safety-fallback characterization is `16 passed, 1
xfailed`. The broader sequential indexed-owner, fallback-owner,
reconciliation, convergence, core, pass-efficiency, architecture, and
TensorFlow import-blocking gate is `479 passed, 1 xfailed in 27.90s`. Ruff and
whitespace validation pass.

At implementation, add only the tensor-delta and reconciliation result
plumbing. Do not change either indexed owner, shared GraphIndex, raw counters,
guard, or following fallback order. Validate indexed Conv-input repair, safety
fallback, static reconciliation, core, pass efficiency, architecture, and
TensorFlow import blocking sequentially. Commit and push only; do not create or
update a pull request.

## Safety-fallback Conv-input evidence implementation checkpoint

The fallback runner now samples `fallback_conv_input_tensor_count` and merges a
clamped `pruned_unused_tensors` delta with its unchanged singleton-Reshape and
stale-Transpose counters. This captures both child owners' zero-rewrite cleanup
without rerunning either owner or scanning the graph.

`_fallback_conv_input_static_shape_stats` supplies the stable complete zero
schema and receives the opt-in reconciliation result only under the unchanged
stale-Transpose predicate. Singleton repair continues to update Conv output
metadata directly, and cleanup-only pruning does not broaden the guard. The
following mixed-Concat owner remains adjacent.

Focused Conv-input, safety-fallback, and occurrence validation is `18 passed,
22 deselected`. The broader sequential indexed-owner, fallback-owner,
reconciliation, convergence, core, pass-efficiency, architecture, and
TensorFlow import-blocking gate is `480 passed in 27.81s`. Ruff, Python
bytecode compilation, and whitespace validation pass.

At resume, audit `fallback_concat_layout_stats` and its guarded reconciliation.
Confirm its counter and absence or presence of cleanup-only mutation before
changing that boundary. Commit and push only; do not create or update a pull
request.

## Safety-fallback mixed-Concat reconciliation characterization checkpoint

`_repair_mixed_nhwc_inputs_for_nchw_concat()` inserts local transpose adapters,
rewires the Concat, and updates its output metadata only when
`repaired_mixed_nhwc_inputs_for_nchw_concat` increments. It performs no pruning
or zero-rewrite cleanup, and existing positive/no-op/idempotence/fan-out tests
make the one-key result a complete mutation predicate.

The fallback guard is already correct, but its reconciliation result is
discarded. A strict expected-failure AST contract requires a stable two-key
`_fallback_mixed_concat_static_shape_stats` default and an opt-in complete
result inside the unchanged positive guard. The following Concat-axis repair
remains fixed.

Focused mixed-Concat and safety-fallback characterization is `18 passed, 1
xfailed`. The broader sequential indexed-owner, fallback-owner,
reconciliation, convergence, core, pass-efficiency, architecture, and
TensorFlow import-blocking gate is `490 passed, 1 xfailed in 27.67s`. Ruff and
whitespace validation pass.

At implementation, add only this result plumbing. Do not alter the matcher,
adapter insertion, Concat rewire, output metadata, raw result, guard, or next
owner. Validate mixed-Concat repair, safety fallback, static reconciliation,
core, pass efficiency, architecture, and TensorFlow import blocking
sequentially. Commit and push only; do not create or update a pull request.

## Safety-fallback mixed-Concat reconciliation implementation checkpoint

The fallback mixed-Concat owner, its exact one-key result, and positive guard
are unchanged. `_fallback_mixed_concat_static_shape_stats` supplies both zero
keys and receives the opt-in complete reconciliation result only after a
positive repair.

No matcher, adapter insertion, Concat rewire, output metadata, scan,
dependency, or following Concat-axis owner changed. This is observation-only
result plumbing under the existing conditional reconciliation.

Focused mixed-Concat and safety-fallback validation is `19 passed`. The broader
sequential indexed-owner, fallback-owner, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `491
passed in 29.82s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, audit `fallback_concat_axis_stats` and its guarded reconciliation.
Confirm its complete counter and cleanup behavior before changing that
boundary. Commit and push only; do not create or update a pull request.

## Safety-fallback Concat-axis reconciliation characterization checkpoint

`_repair_nchw_concat_transpose_conv_axes()` updates a proven Concat axis and
related tensor metadata only when
`repaired_nchw_concat_transpose_conv_axes` increments. The indexed owner has no
pruning or zero-rewrite cleanup, and existing positive/no-op/multi-rewrite
coverage makes the raw counter complete.

The fallback positive guard is unchanged, but its reconciliation result is
discarded. A strict expected-failure contract requires a stable
`_fallback_concat_axis_static_shape_stats` zero dictionary and an opt-in
complete result inside the same guard. The following stale binary-layout owner
remains fixed.

Focused indexed Concat-axis and safety-fallback characterization is `37 passed,
1 xfailed`. The broader sequential indexed-owner, fallback-owner,
reconciliation, convergence, core, pass-efficiency, architecture, and
TensorFlow import-blocking gate is `519 passed, 1 xfailed in 27.75s`. Ruff and
whitespace validation pass.

At implementation, add only this result plumbing. Do not change any of the
three production occurrences, indexed matching, axis/metadata writes, result
schema, guard, or following owner. Validate indexed Concat-axis repair, safety
fallback, static reconciliation, core, pass efficiency, architecture, and
TensorFlow import blocking sequentially. Commit and push only; do not create or
update a pull request.

## Safety-fallback Concat-axis reconciliation implementation checkpoint

Only the fallback occurrence now initializes
`_fallback_concat_axis_static_shape_stats` with both zero keys and assigns the
opt-in complete reconciliation result on a positive repair. Its exact owner
counter and guard are unchanged.

The very-late and final occurrences, indexed matching, axis/metadata writes,
and following stale binary-layout owner remain unchanged. No scan, dependency,
or unconditional work is added.

Focused indexed Concat-axis and safety-fallback validation is `38 passed`. The
broader sequential indexed-owner, fallback-owner, reconciliation, convergence,
core, pass-efficiency, architecture, and TensorFlow import-blocking gate is
`520 passed in 27.79s`. Ruff, Python bytecode compilation, and whitespace
validation pass.

At resume, audit `fallback_binary_layout_stats` and its guarded reconciliation.
Confirm the owner's cleanup behavior and counter completeness before changing
that boundary. Commit and push only; do not create or update a pull request.

## Safety-fallback binary-layout evidence characterization checkpoint

`_repair_stale_nchw_to_nhwc_channelwise_binary_transposes()` has one exact
rewrite counter but unconditionally prunes unused tensors. A focused zero-
rewrite fixture removes an unused constant while returning zero, proving that
the raw result is incomplete mutation evidence.

A strict expected-failure fallback contract requires
`fallback_binary_layout_tensor_count`, a clamped `pruned_unused_tensors` delta,
and a stable `_fallback_binary_layout_static_shape_stats` zero dictionary. The
existing stale-binary repair guard remains unchanged and assigns an opt-in
complete reconciliation result instead of discarding it. Cleanup-only evidence
does not broaden the guard, and the immediately following topological sort is
fixed.

At implementation, add only this result plumbing. Do not alter indexed
matching, rewiring, output metadata, pruning, raw counter, guard, or sort.
Validate stale binary repair, safety fallback, static reconciliation,
convergence, core, pass efficiency, architecture, and TensorFlow import
blocking sequentially. Commit and push only; do not create or update a pull
request.

Characterization validation completed sequentially under `uv`:

- focused fallback/binary/convergence selection: `22 passed, 12 deselected, 1 xfailed`
- broad related regression gate: `533 passed, 1 xfailed`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet complete-evidence contract
defined by this checkpoint; there are no unexpected failures.

## Safety-fallback binary-layout evidence implementation checkpoint

Only the recursive safety-fallback occurrence now records the stale-binary
owner's complete local evidence. It captures the tensor count before the owner,
merges a clamped `pruned_unused_tensors` delta into the exact rewrite counter,
and initializes `_fallback_binary_layout_static_shape_stats` with both stable
zero keys. A positive rewrite still controls the same reconciliation guard,
which now stores the opt-in complete result instead of discarding it.

Cleanup-only calls do not trigger shape propagation. Indexed matching,
rewiring, output metadata, pruning behavior, the raw owner schema, and the
following unconditional topological sort are unchanged. The preceding
Concat-axis orchestration contract now identifies the binary-layout tensor
count as the next owner's first statement.

Implementation validation completed sequentially under `uv`:

- focused fallback/binary/convergence selection: `23 passed, 12 deselected`
- broad related regression gate: `534 passed in 27.64s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the fallback layout-validation metadata boundary before the
high-rank BatchMatMul compression owner. Commit and push coherent units only;
do not create or update a pull request.

## ONNX graph-input type-checking checkpoint

The ONNX protobuf repeated-field stub can cause Pylance to infer a graph input
iterator item as the generic class `type[In]`, which hides the concrete
`ValueInfoProto.name` attribute. The lowerer now narrows only that local item
with `typing.cast(onnx.ValueInfoProto, ...)`. This is a static-analysis-only
clarification: graph traversal, input filtering, names, shapes, and runtime
behavior are unchanged.

## Safety-fallback terminal layout-validation characterization checkpoint

`validate_model_ir_layout_annotations()` is a pure read-only query, but its
fallback call currently precedes high-rank BatchMatMul compression and indexed
binary convergence. It therefore describes a non-terminal graph. In addition,
an empty result leaves `logical_layout_validation_errors` inherited from the
recursive lower in metadata.

A passing purity fixture records an invalid annotation without modifying the
ModelIR. A strict expected-failure orchestration contract requires validation
after high-rank compression, its guarded reconciliation/sort, indexed binary
convergence, and the final sort. A non-empty result keeps the existing list
schema; an empty result removes only the stale validation-error key. The
`layout_optimize_fallback` metadata assignment remains before the high-rank
owner, and finalization remains the terminal operation.

At implementation, move only the pure validation/metadata block and add the
empty-result removal. Do not change either mutation owner, its guard, sorting,
fallback reason/samples, finalization, dependencies, or TensorFlow boundary.
Validate sequentially, then commit and push only; do not create or update a
pull request.

The first attempt to add the two existing rank-6 BatchMatMul tests to the broad
gate stopped during collection: `tests/test_tflite_builder_direct.py` imports
`run_elementwise_gate_layout_cleanup` from the lowerer, while commit
`5c4f72ae` removed that compatibility re-export during orchestration
extraction. An AST/runtime audit of all 115 names imported from the lowerer by
that test file found this as the sole missing name. The implementation remains
owned by `passes/elementwise_gate_layout.py`; restoring an explicitly marked
compatibility import is sufficient and does not reconnect it to lowerer
orchestration. Record and repair this independently before using that file as a
gate.

Characterization validation completed sequentially under `uv` after isolating
that collection issue:

- focused layout/binary fallback selection: `3 passed, 10 deselected, 1 xfailed`
- broad related regression gate: `535 passed, 1 xfailed in 27.37s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet terminal-validation contract;
the broad gate has no unexpected failure.

## Elementwise gate compatibility re-export checkpoint

The lowerer again exposes `run_elementwise_gate_layout_cleanup`, importing it
from the extracted gate-layout orchestration module and marking it as a
compatibility re-export. This restores the established Python/test import
contract removed by `5c4f72ae` without adding a direct lowerer call, a second
owner, a new scan, or a dependency.

Validate collection of `tests/test_tflite_builder_direct.py`, the two rank-6
BatchMatMul tests, gate orchestration/architecture, pass efficiency, and the
TensorFlow import boundary sequentially. Commit and push this repair
independently; do not create or update a pull request.

Compatibility validation completed sequentially under `uv`:

- previously blocked rank-6 structure/numeric tests: `2 passed`
- gate orchestration, architecture, pass efficiency, optional TensorFlow
  boundary, and rank-6 tests: `312 passed in 26.08s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

## Safety-fallback terminal layout-validation implementation checkpoint

The pure fallback validation block now runs after high-rank BatchMatMul
compression, its guarded reconciliation/sort, indexed binary convergence, and
the terminal topological sort. A non-empty result writes the same
`logical_layout_validation_errors` list. An empty result removes only that key
when it was inherited from the recursive lower.

The high-rank owner, convergence owner, guards, sorts,
`layout_optimize_fallback` reason/count/samples, finalizer, dependency set, and
TensorFlow-free boundary are unchanged. The strict characterization contract
is now a normal passing test.

Implementation validation completed sequentially under `uv`:

- focused layout/binary fallback selection: `4 passed, 10 deselected`
- broad related gate plus rank-6 structure/numeric parity: `538 passed in 27.90s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit `fallback_high_rank_bmm_stats` for complete mutation evidence
and retain its guard and following indexed binary convergence boundary. Commit
and push only; do not create or update a pull request.

## Safety-fallback high-rank BatchMatMul characterization checkpoint

`_compress_static_high_rank_batch_matmul()` rewrites only fully-static,
same-batch-shape rank-greater-than-five `BATCH_MATMUL` operators. Its unused-
tensor pruning and optional layout-state sync occur only when the exact
`compressed_static_high_rank_batch_matmul` counter is positive. A focused
zero-rewrite fixture retains an unrelated unused constant, proving that the raw
counter is complete mutation evidence without a cleanup-only path. Existing
rank-6 structure and numeric-parity tests cover the positive path.

A strict expected-failure fallback contract requires a stable
`_fallback_high_rank_bmm_static_shape_stats` zero dictionary and assigns the
opt-in complete reconciliation result under the unchanged positive guard. The
guarded topological sort and immediately following indexed binary convergence
remain fixed.

At implementation, add only this result plumbing. Do not change eligibility,
reshape construction, graph-index mutation, pruning, raw result schema, guard,
sort, convergence, terminal validation, dependencies, or TensorFlow boundary.
Validate sequentially, then commit and push only; do not create or update a
pull request.

Characterization validation completed sequentially under `uv`:

- focused high-rank/fallback plus rank-6 parity selection:
  `4 passed, 13 deselected, 1 xfailed`
- broad related gate plus rank-6 structure/numeric parity:
  `539 passed, 1 xfailed in 28.13s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet result-staging contract; there
are no unexpected failures.

## Safety-fallback high-rank BatchMatMul implementation checkpoint

Only the recursive fallback occurrence now initializes
`_fallback_high_rank_bmm_static_shape_stats` with the two stable zero keys and
stores the opt-in complete reconciliation result under the existing positive
compression guard. The owner call, raw result schema, guard, and guarded
topological sort are unchanged.

Compression eligibility, reshape construction, graph-index mutation, pruning,
the final indexed binary convergence, terminal validation, dependency set, and
TensorFlow-free boundary are unchanged. The preceding terminal-validation
contract was updated only for the new result statement at the next owner's
boundary.

Implementation validation completed sequentially under `uv`:

- safety fallback plus rank-6 structure/numeric parity: `18 passed`
- broad related gate plus rank-6 structure/numeric parity: `540 passed in 27.98s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the result returned by
`_run_indexed_binary_layout_convergence(fallback_ir)` and retain its following
terminal sort/validation boundary. Commit and push only; do not create or
update a pull request.

## Safety-fallback indexed binary-convergence characterization checkpoint

`_run_indexed_binary_layout_convergence()` owns up to three rounds of
rank-four broadcast repair, stale binary-Transpose repair, and static-shape
reconciliation with one GraphIndex. It aggregates all three pure mutation
counters and stops after the first all-zero round. Existing immediate-stable,
second-round-stable, three-round-cap, multi-repair, index-reuse, and
legacy-equivalence tests prove its result is complete.

The fallback caller currently discards that complete dictionary. A strict
expected-failure orchestration contract requires only assigning it to
`_fallback_binary_layout_convergence_stats`. The owner call remains single and
the following terminal sort/validation boundary remains fixed; no additional
reconciliation is required.

At implementation, replace only the fallback expression with a result-capture
assignment. Do not change convergence rounds, owner order, GraphIndex reuse,
statistics, sorting, terminal validation, dependencies, or TensorFlow
boundary. Validate sequentially, then commit and push only; do not create or
update a pull request.

Characterization validation completed sequentially under `uv`:

- focused convergence/high-rank/terminal-layout selection:
  `14 passed, 13 deselected, 1 xfailed`
- broad related gate plus rank-6 structure/numeric parity:
  `540 passed, 1 xfailed in 28.63s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet result-capture contract; there
are no unexpected failures.

## Safety-fallback indexed binary-convergence implementation checkpoint

Only the recursive fallback occurrence now assigns the complete convergence
dictionary to `_fallback_binary_layout_convergence_stats`. The owner call
remains single and retains its three-round cap, stable-round exit, owner order,
single GraphIndex, and aggregate three-key schema.

No additional reconciliation, sort, scan, or dependency is added. The
following terminal sort and validation retain their order and now identify the
result-capture assignment as the preceding owner boundary.

Implementation validation completed sequentially under `uv`:

- focused convergence/high-rank/terminal-layout selection:
  `15 passed, 13 deselected`
- broad related gate plus rank-6 structure/numeric parity:
  `541 passed in 27.89s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the corresponding primary-path terminal layout-validation and
indexed binary-convergence boundary. Commit and push only; do not create or
update a pull request.

## Primary terminal layout-validation characterization checkpoint

The primary path currently computes `layout_problems` before indexed binary
convergence, high-rank binary coalescing, dynamic-boundary signature
realignment, and the final topological sort. The comment describes a terminal
graph, but the actual validation result can precede all four later mutations.
The same validator was already proven pure by the fallback checkpoint.

A dedicated strict expected-failure orchestration contract requires validation
immediately after the final sort. A non-empty result preserves the established
`logical_layout_validation_errors` list schema; an empty result removes only a
stale key. Finalization follows immediately. Progress advancement/spinner
closure remain before indexed convergence and are not moved.

At implementation, move only the pure validation/metadata block and add its
empty-result removal. Do not change progress reporting, convergence,
coalescing, signature realignment, sorting, finalization, dependencies, or the
TensorFlow boundary. Validate sequentially, then commit and push only; do not
create or update a pull request.

Characterization validation completed sequentially under `uv`:

- terminal validation, indexed convergence, and high-rank binary owners:
  `13 passed, 1 xfailed`
- broad related gate plus rank-6 structure/numeric parity:
  `543 passed, 1 xfailed in 27.93s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet primary terminal-validation
contract; there are no unexpected failures.

## Primary terminal layout-validation implementation checkpoint

The pure primary validation block now runs after indexed binary convergence,
high-rank binary coalescing, dynamic-boundary signature realignment, and the
final topological sort. A non-empty result writes the same
`logical_layout_validation_errors` list; an empty terminal result removes only
that stale key. Finalization remains immediately after validation.

Progress advancement/spinner closure, all four terminal mutation owners,
sorting, dependencies, and the TensorFlow-free boundary are unchanged. The
misleading pre-convergence validation comment now describes terminal layout
cleanup rather than claiming that the earlier graph was terminal.

Implementation validation completed sequentially under `uv`:

- terminal validation, indexed convergence, and high-rank binary owners:
  `14 passed`
- broad related gate plus rank-6 structure/numeric parity:
  `544 passed in 27.95s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit result capture for the primary terminal indexed convergence,
high-rank binary coalescing, and boundary-signature realignment calls. Commit
and push only; do not create or update a pull request.

## Primary terminal mutation-result characterization checkpoint

The primary terminal indexed binary-convergence owner returns its complete
three-counter aggregate. Static high-rank binary coalescing mutates only when
its exact rewrite counter increments and its zero path is covered as a no-op.
Dynamic-boundary signature realignment returns the exact updated-tensor count,
with changed, no-op, missing-metadata, wrapper-equivalence, and idempotency
coverage. None requires caller-side reconciliation.

The primary caller currently discards all three dictionaries. A strict
expected-failure orchestration contract requires assigning them respectively
to `_final_binary_layout_convergence_stats`,
`_final_high_rank_binary_stats`, and
`_final_dynamic_boundary_signature_stats`. Calls, arguments, layout-state
handoff, order, final sort, terminal validation, and finalization remain fixed.

At implementation, replace only the three expressions with assignments. Do
not change any owner, result schema, GraphIndex use, layout sync, sort,
validation, dependency, or TensorFlow boundary. Validate sequentially, then
commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused terminal/convergence/high-rank/signature selection:
  `22 passed, 11 deselected, 1 xfailed`
- expanded broad related gate plus rank-6 structure/numeric parity:
  `563 passed, 1 xfailed in 28.27s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet three-result capture contract;
there are no unexpected failures.

## Primary terminal mutation-result implementation checkpoint

The primary path now assigns the complete terminal dictionaries to
`_final_binary_layout_convergence_stats`,
`_final_high_rank_binary_stats`, and
`_final_dynamic_boundary_signature_stats`. All three calls, arguments,
layout-state handoff, order, and result schemas are unchanged.

No reconciliation, scan, sort, or dependency is added. The terminal sort,
layout validation, stale-error removal, finalization, and TensorFlow-free
boundary remain unchanged. The strict characterization contract is now a
normal passing test.

Implementation validation completed sequentially under `uv`:

- focused terminal/convergence/high-rank/signature selection:
  `23 passed, 11 deselected`
- expanded broad related gate plus rank-6 structure/numeric parity:
  `564 passed in 28.36s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, continue the primary final-pass reconciliation inventory with
`final_high_rank_bmm_stats`. Confirm counter/cleanup completeness before
retaining its guarded reconciliation result. Commit and push only; do not
create or update a pull request.

## Primary final high-rank BatchMatMul characterization checkpoint

`final_high_rank_bmm_stats` uses the same static high-rank BatchMatMul owner
already characterized in the recursive fallback. Its exact positive counter
contains every reshape/prune/layout-sync mutation, while zero is a true no-op.
Existing rank-6 structure and numeric-parity tests cover the positive path.

A strict expected-failure primary-path contract requires a stable
`_final_high_rank_bmm_static_shape_stats` zero dictionary and assigns the
opt-in complete reconciliation result under the unchanged positive guard. The
guarded topological sort and immediately following `final_pad_layout_stats`
owner remain fixed.

At implementation, add only result plumbing. Do not change compression
eligibility, graph-index mutation, pruning, layout sync, owner schema, guard,
sort, Pad boundary, dependencies, or TensorFlow behavior. Validate
sequentially, then commit and push only; do not create or update a pull
request.

Characterization validation completed sequentially under `uv`:

- terminal orchestration plus rank-6 structure/numeric parity:
  `4 passed, 1 xfailed`
- expanded broad related gate plus rank-6 structure/numeric parity:
  `564 passed, 1 xfailed in 28.33s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final high-rank reconciliation
contract; there are no unexpected failures.

## Primary final high-rank BatchMatMul implementation checkpoint

The primary path now initializes `_final_high_rank_bmm_static_shape_stats`
with both stable zero keys and stores the opt-in complete reconciliation result
under the unchanged positive compression guard. The guarded topological sort
and following Pad owner retain their positions.

Compression eligibility, reshape construction, graph-index mutation, pruning,
layout-state sync, raw result schema, dependencies, and the TensorFlow-free
boundary are unchanged. The strict characterization contract is now a normal
passing test.

Implementation validation completed sequentially under `uv`:

- terminal orchestration plus rank-6 structure/numeric parity: `5 passed`
- expanded broad related gate plus rank-6 structure/numeric parity:
  `565 passed in 28.72s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit `final_pad_layout_stats` for counter/cleanup completeness and
retain its guarded reconciliation result without changing the following Conv-
input owner boundary. Commit and push only; do not create or update a pull
request.

## Primary final Pad reconciliation characterization checkpoint

`repair_channel_last_inputs_for_channel_first_pad()` inserts its adapter,
constant, and Transpose only when
`repaired_channel_last_inputs_for_channel_first_pad` increments. Optional
layout-state sync is under the same positive predicate. The zero path has no
pruning or cleanup, and existing positive/native-NCHW/unproven-mismatch tests
cover both paths plus GraphIndex/layout-state consistency.

A strict expected-failure primary-path contract requires a stable
`_final_pad_layout_static_shape_stats` zero dictionary and assigns the opt-in
complete reconciliation result under the unchanged positive guard. The
guarded topological sort and following `final_conv_input_stats` owner remain
fixed.

At implementation, add only result plumbing. Do not change Pad matching,
adapter construction, lineage, GraphIndex mutation, layout sync, result schema,
guard, sort, Conv boundary, dependencies, or TensorFlow behavior. Validate
sequentially, then commit and push only; do not create or update a pull
request.

Characterization validation completed sequentially under `uv`:

- terminal orchestration and Pad owner: `7 passed, 1 xfailed`
- expanded broad related gate plus Pad/rank-6 coverage:
  `569 passed, 1 xfailed in 28.54s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final Pad reconciliation
contract; there are no unexpected failures.

## Primary final Pad reconciliation implementation checkpoint

The primary path now initializes `_final_pad_layout_static_shape_stats` with
both stable zero keys and stores the opt-in complete reconciliation result
under the unchanged positive Pad-repair guard. The guarded topological sort and
following Conv-input owner retain their positions.

Pad matching, adapter/constant/Transpose construction, lineage, GraphIndex
mutation, layout-state sync, raw result schema, dependencies, and the
TensorFlow-free boundary are unchanged. The strict characterization contract
is now a normal passing test.

Implementation validation completed sequentially under `uv`:

- terminal orchestration and Pad owner: `8 passed`
- expanded broad related gate plus Pad/rank-6 coverage:
  `570 passed in 28.52s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit `final_conv_input_stats` for zero-rewrite cleanup/pruning
before retaining its guarded reconciliation result. Keep the following mixed-
Concat owner boundary fixed. Commit and push only; do not create or update a
pull request.

## Primary final Conv-input evidence characterization checkpoint

The standalone stale NCHW-to-NHWC Conv-input owner unconditionally prunes
unused tensors after its indexed rewrite loop. A focused real-tensor fixture
removes an unrelated unused constant while returning a zero rewrite counter,
proving that `final_conv_input_stats` is incomplete mutation evidence today.

A strict expected-failure primary-path contract requires
`final_conv_input_tensor_count`, a clamped `pruned_unused_tensors` delta, and a
stable `_final_conv_input_static_shape_stats` zero dictionary. The existing
rewrite-only guard assigns the opt-in complete reconciliation result but does
not run for cleanup-only evidence. Its guarded sort and the following mixed-
Concat owner remain fixed.

At implementation, add only this evidence plumbing. Do not change indexed
matching, rewiring, output metadata, pruning, raw owner schema, guard, sort,
mixed-Concat boundary, dependencies, or TensorFlow behavior. Validate
sequentially, then commit and push only; do not create or update a pull
request.

Characterization validation completed sequentially under `uv`:

- focused final/prune/convergence selection:
  `4 passed, 10 deselected, 1 xfailed`
- expanded broad related gate: `571 passed, 1 xfailed in 28.23s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet complete Conv-input evidence
contract; there are no unexpected failures.

## Primary final Conv-input evidence implementation checkpoint

The primary path now snapshots the tensor count, merges a clamped
`pruned_unused_tensors` delta into `final_conv_input_stats`, initializes
`_final_conv_input_static_shape_stats` with both stable zero keys, and stores
the opt-in complete reconciliation result under the unchanged positive rewrite
guard. Cleanup-only evidence does not trigger shape propagation.

Indexed matching, rewiring, output metadata, pruning behavior, the raw owner
schema, guarded sort, dependencies, and the following mixed-Concat boundary
are unchanged. The preceding Pad contract now identifies the Conv-input tensor
snapshot as the next owner start.

Implementation validation completed sequentially under `uv`:

- focused final/prune/convergence selection: `5 passed, 10 deselected`
- expanded broad related gate: `572 passed in 28.31s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit `final_concat_layout_stats` for zero-rewrite cleanup before
retaining its guarded reconciliation result. Keep the following Concat-axis
owner boundary fixed. Commit and push only; do not create or update a pull
request.

## Primary final mixed-Concat reconciliation characterization checkpoint

`final_concat_layout_stats` uses the same mixed-NHWC-input repair for NCHW
Concat already characterized in the recursive fallback. Its exact counter
covers the input adapter/rewire and output metadata mutation; the zero path has
no pruning or cleanup.

A strict expected-failure primary-path contract requires a stable
`_final_mixed_concat_static_shape_stats` zero dictionary and assigns the opt-in
complete reconciliation result under the unchanged positive guard. The
guarded sort and following `final_concat_axis_stats` owner remain fixed.

At implementation, add only result plumbing. Do not change matching, adapter
insertion, rewiring, metadata, result schema, guard, sort, Concat-axis boundary,
dependencies, or TensorFlow behavior. Validate sequentially, then commit and
push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- terminal orchestration and mixed-Concat owner: `15 passed, 1 xfailed`
- expanded broad related gate: `572 passed, 1 xfailed in 28.34s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final mixed-Concat
reconciliation contract; there are no unexpected failures.

## Primary final mixed-Concat reconciliation implementation checkpoint

The primary path now initializes `_final_mixed_concat_static_shape_stats` with
both stable zero keys and stores the opt-in complete reconciliation result
under the unchanged positive mixed-Concat guard. The guarded sort and following
Concat-axis owner retain their positions.

Matching, adapter insertion, rewiring, output metadata, raw result schema,
dependencies, and the TensorFlow-free boundary are unchanged. The strict
characterization contract is now a normal passing test.

Implementation validation completed sequentially under `uv`:

- terminal orchestration and mixed-Concat owner: `16 passed`
- expanded broad related gate: `573 passed in 28.97s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, stage complete reconciliation evidence for
`final_concat_axis_stats`, retaining its guard, sort, and following stale-
binary owner boundary. Commit and push only; do not create or update a pull
request.

## Primary final Concat-axis/binary evidence characterization checkpoint

The final Concat-axis owner is the same counter-complete, cleanup-free repair
already characterized in the fallback. The immediately following stale
channelwise-binary owner is also shared with the fallback and unconditionally
prunes unused tensors, including zero-rewrite calls.

A strict expected-failure primary-path contract requires a stable
`_final_concat_axis_static_shape_stats` result under the existing axis guard,
then `final_binary_layout_tensor_count`, a clamped `pruned_unused_tensors`
delta, and stable `_final_binary_layout_static_shape_stats` under the existing
binary guard. Both guarded sorts and the following progress boundary remain
fixed; cleanup-only evidence does not trigger reconciliation.

At implementation, add only result/prune plumbing. Do not change either owner,
matching, rewiring, metadata, pruning, raw schemas, guards, sorts, progress,
dependencies, or TensorFlow behavior. Validate sequentially, then commit and
push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- terminal/Concat-axis/binary owner gate: `37 passed, 1 xfailed`
- expanded broad related gate: `573 passed, 1 xfailed in 29.12s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet complete final Concat-axis and
binary evidence contract; there are no unexpected failures.

## Primary final Concat-axis/binary evidence implementation checkpoint

The primary path now initializes `_final_concat_axis_static_shape_stats` with
both stable zero keys and stores the opt-in complete result under the unchanged
positive axis guard. It then snapshots the tensor count, merges a clamped
`pruned_unused_tensors` delta into `final_binary_layout_stats`, initializes
`_final_binary_layout_static_shape_stats`, and stores the opt-in complete result
under the unchanged positive binary guard.

Cleanup-only stale-binary evidence does not trigger reconciliation. Both
owners, matching, rewiring, metadata, pruning, raw schemas, guards, guarded
sorts, progress boundary, dependencies, and TensorFlow-free behavior are
unchanged.

Implementation validation completed sequentially under `uv`:

- terminal/Concat-axis/binary owner gate: `38 passed`
- expanded broad related gate: `574 passed in 28.45s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, continue the remaining primary final-pass reconciliation inventory
immediately before `final_high_rank_bmm_stats`, beginning with
`final_sinet_concat_resize_stats`. Confirm each owner's counter and cleanup
completeness before changing its caller. Commit and push only; do not create or
update a pull request.

## Primary final SiNet Concat/Resize characterization checkpoint

`_optimize_sinet_concat_resize_affine_transpose_chains()` mutates only after a
transactional plan revalidation succeeds. Its exact positive counter covers
rewiring, metadata updates, removals, optional legacy adapter insertion,
pruning, and layout-state sync. Preflight, unsafe/stale-plan, rewrite-cap, and
second-run zero results are true no-ops. Existing indexed and numeric-parity
coverage spans dtype, constant form, input order, Resize type, fan-out, and
legacy behavior.

A strict expected-failure primary-path contract requires a stable
`_final_sinet_concat_resize_static_shape_stats` zero dictionary and assigns the
opt-in complete reconciliation result under the unchanged positive guard. No
sort is added, and the following `final_high_rank_bmm_stats` boundary remains
fixed.

At implementation, add only result plumbing. Do not change matching,
transactional plan application, rewiring, metadata, pruning, layout sync, raw
schema, guard, following owner, dependencies, or TensorFlow behavior. Validate
sequentially, then commit and push only; do not create or update a pull
request.

Characterization validation completed sequentially under `uv`:

- terminal orchestration and indexed SiNet Concat/Resize owner:
  `65 passed, 1 xfailed`
- expanded broad related gate: `632 passed, 1 xfailed in 28.51s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final SiNet Concat/Resize
reconciliation contract; there are no unexpected failures.

## Primary final SiNet Concat/Resize implementation checkpoint

The primary path now initializes
`_final_sinet_concat_resize_static_shape_stats` with both stable zero keys and
stores the opt-in complete result under the unchanged positive rewrite guard.
No sort is added, and the following high-rank BatchMatMul owner retains its
position.

Matching, transactional plan application, rewiring, metadata, pruning,
layout-state sync, raw result schema, dependencies, and TensorFlow-free
behavior are unchanged. The first broad run exposed one stale architecture
assumption that all six final SiNet guards immediately follow stats and discard
reconciliation results; that contract now distinguishes only this completed
owner while retaining the legacy shape for the other five.

Implementation validation completed sequentially under `uv`:

- terminal orchestration and indexed owner: `66 passed`
- targeted owner/architecture contract: `68 passed, 256 deselected`
- expanded broad related gate: `633 passed in 28.54s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit `final_sinet_deep_skip_stats` for counter/cleanup completeness
before retaining its guarded reconciliation result. Keep the following SiNet
Concat/Resize owner boundary fixed. Commit and push only; do not create or
update a pull request.

## Remaining primary final SiNet reconciliation characterization checkpoint

The five preceding final SiNet owners—late residual, deep-skip pre-add fan-out,
deep-skip dual Resize, shared-post PReLU fan-out, and deep-skip Concat/Resize
tail—use indexed transactional plans. Each exact counter increments only after
preflight revalidation and mutation. Pruning and optional layout-state sync are
inside the same positive predicate; preflight, unsafe, stale-plan, capped, and
second-run zero results are true no-ops. Their dedicated suites cover numeric
parity, transactionality, idempotency, fan-out, constant cloning, GraphIndex,
and layout-state consistency.

A single strict expected-failure orchestration contract requires stable
two-key results for all five existing positive guards and assigns each opt-in
complete reconciliation result. No sort is added. The order from late residual
through final SiNet Concat/Resize remains exact.

At implementation, add only result plumbing for these five callers. Do not
change any owner, match/plan logic, rewiring, metadata, pruning, layout sync,
raw schema, guard, ordering, dependencies, or TensorFlow behavior. Validate all
five dedicated suites sequentially, then commit and push only; do not create or
update a pull request.

Characterization validation completed sequentially under `uv`:

- terminal orchestration plus all five dedicated SiNet owner suites:
  `464 passed, 1 xfailed in 1.67s`
- expanded broad related gate: `1089 passed, 1 xfailed in 29.28s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet five-owner result contract;
there are no unexpected failures.

## Remaining primary final SiNet reconciliation implementation checkpoint

The primary path now initializes stable two-key static-shape results for late
residual, pre-add fan-out, dual Resize, shared-post fan-out, and deep-skip, and
stores each opt-in complete result under its unchanged positive guard. No sort
or additional scan is added, and the exact owner order remains unchanged.

All five transactional owners, matching/planning, rewiring, metadata, pruning,
layout-state sync, raw schemas, dependencies, and TensorFlow-free behavior are
unchanged. The shared architecture contract now requires complete results for
all six final SiNet owners.

Implementation validation completed sequentially under `uv`:

- all five owner suites plus orchestration/architecture selection:
  `470 passed, 253 deselected in 3.57s`
- expanded broad related gate: `1090 passed in 29.25s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, continue the remaining primary final-pass reconciliation inventory
immediately before this SiNet cluster, beginning with
`final_consecutive_reshape_stats`. Confirm its multi-counter and cleanup
completeness before changing the caller. Commit and push only; do not create or
update a pull request.

## Primary final consecutive-Reshape reconciliation characterization checkpoint

The absolute-final consecutive-Reshape runner has three returned mutation
counters: no-op removal, ordinary consecutive-chain bypass/removal, and the
fan-out bypass subset. Every input/output rewire or operator removal increments
at least one of those counters. Unused-tensor pruning and optional layout-state
sync are inside the same positive `rewritten > 0 or removed_noop > 0` predicate;
preflight, no-candidate, unsafe, dynamic-shape, and preserved-semantic-rank
paths are true no-ops. The existing aggregate guard is therefore complete even
though the fan-out subset is intentionally represented by two positive keys.

A strict expected-failure orchestration contract now requires a stable two-key
`_final_consecutive_reshape_static_shape_stats` value and assigns the opt-in
complete reconciliation result under the unchanged aggregate guard. It also
fixes the immediately following `final_sinet_late_residual_stats` boundary.

At implementation, add only this result plumbing. Do not change the runner,
counter schema, matching, rewiring, operator removal, pruning, layout sync,
aggregate guard, pass order, dependencies, or TensorFlow behavior. Validate the
consecutive-Reshape owner and terminal orchestration sequentially, then commit
and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- consecutive-Reshape owner, terminal orchestration, and architecture:
  `281 passed, 1 xfailed in 17.27s`
- expanded broad related gate: `1093 passed, 1 xfailed in 29.50s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final consecutive-Reshape
reconciliation-result contract; there are no unexpected failures.

## Primary final consecutive-Reshape reconciliation implementation checkpoint

The primary path now initializes
`_final_consecutive_reshape_static_shape_stats` with the stable legacy-plus-
complete two-key schema and replaces it with the opt-in complete reconciliation
result only under the unchanged three-counter aggregate guard. This adds no
scan: reconciliation already ran on the same positive path.

The runner, its returned counter schema, matching, rewiring, operator removal,
positive-only pruning/layout synchronization, guard expression, pass order,
dependencies, and TensorFlow-free behavior are unchanged. The strict
characterization contract is now a normal passing contract, and the existing
architecture guard has been updated only for the retained result assignment.

Implementation validation completed sequentially under `uv`:

- consecutive-Reshape owner, terminal orchestration, and architecture:
  `282 passed in 18.99s`
- expanded broad related gate: `1094 passed in 29.69s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit `final_prelu_stats` and its tensor-count-assisted guard for
counter and cleanup completeness before retaining its guarded reconciliation
result. Keep the following consecutive-Reshape boundary fixed. Commit and push
only; do not create or update a pull request.

## Primary final PReLU reconciliation characterization checkpoint

The absolute-final PReLU owner has one exact rewrite counter, but intentionally
runs unused-tensor pruning on every invocation to preserve its legacy cleanup
contract. A zero-rewrite call can therefore mutate the tensor table. The caller
already samples `final_prelu_tensor_count` and combines the rewrite counter with
a clamped-by-comparison net tensor reduction condition. This guard covers both
rewrite and cleanup-only paths; stale, unsafe, capped, and idempotent calls with
no removable tensor remain true no-ops.

A strict expected-failure orchestration contract now requires a stable two-key
`_final_prelu_static_shape_stats` value and assigns the opt-in complete
reconciliation result under that unchanged counter-or-tensor-delta guard. It
also fixes the immediately following `final_consecutive_reshape_stats`
boundary.

At implementation, add only this result plumbing. Do not change the owner,
counter schema, matching/planning, rewiring, alpha handling, pruning, layout
sync, tensor-count sample, guard, ordering, dependencies, or TensorFlow
behavior. Validate the indexed PReLU owner, terminal orchestration, and
architecture sequentially, then commit and push only; do not create or update a
pull request.

Characterization validation completed sequentially under `uv`:

- indexed PReLU owner, terminal orchestration, and architecture:
  `297 passed, 1 xfailed in 17.16s`
- expanded broad related gate: `1123 passed, 1 xfailed in 29.66s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final PReLU reconciliation-
result contract; there are no unexpected failures.

## Primary final PReLU reconciliation implementation checkpoint

The primary path now initializes `_final_prelu_static_shape_stats` with the
stable two-key schema and replaces it with the opt-in complete reconciliation
dictionary only under the unchanged rewrite-or-tensor-reduction guard. This
adds no scan because reconciliation already ran on that positive path.

The indexed owner, raw counter, matching/planning, alpha copy-on-write and
metadata handling, unconditional prune/layout synchronization, tensor-count
sample, guard expression, pass order, dependencies, and TensorFlow-free
behavior are unchanged. The strict characterization contract is now a normal
passing contract, and the existing architecture guard checks the retained
result assignment.

Implementation validation completed sequentially under `uv`:

- indexed PReLU owner, terminal orchestration, and architecture:
  `298 passed in 19.18s`
- expanded broad related gate: `1124 passed in 30.38s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the preceding `final_sinet_shuffle_stats` plus
`final_se_fc_stats`/`final_gather_stats` aggregate and its tensor-count-assisted
guard for complete rewrite and cleanup evidence before retaining its guarded
reconciliation result. Keep the following final PReLU boundary fixed. Commit
and push only; do not create or update a pull request.

## Primary final SE/FC/Gather reconciliation characterization checkpoint

The absolute-final aggregate combines three exact rewrite counters. The
indexed SiNet shuffle owner prunes and syncs layout only after a positive
transactional rewrite. The SE/FC and Gather callbacks preserve legacy pruning
on every candidate execution and can therefore reduce the tensor table with a
zero rewrite counter. Their orchestration runner returns only the two ordered
child dictionaries and performs no independent cleanup. The existing
`final_se_fc_gather_tensor_count` sample plus the three-counter sum therefore
covers every ModelIR rewrite and cleanup-only deletion.

The recursive fallback already stages a stable complete reconciliation result
under the same contract. A strict expected-failure main-path contract now
requires the symmetric `_final_se_fc_gather_static_shape_stats` value and
assigns the opt-in complete result under the unchanged aggregate guard. It also
fixes the following `final_prelu_tensor_count` boundary.

At implementation, add only main-path result plumbing. Do not change any
owner, counter schema, orchestration order, matching/planning, rewiring,
pruning, layout sync, tensor-count sample, guard, dependencies, fallback path,
or TensorFlow behavior. Validate the three owners, orchestration, terminal
contract, and architecture sequentially, then commit and push only; do not
create or update a pull request.

Characterization validation completed sequentially under `uv`:

- SiNet shuffle, SE/FC, Gather, orchestration, core guard, and architecture:
  `542 passed, 1 xfailed in 18.76s`
- expanded broad related gate: `1135 passed, 1 xfailed in 30.04s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet primary final aggregate
reconciliation-result contract; there are no unexpected failures.

## Primary final SE/FC/Gather reconciliation implementation checkpoint

The main path now initializes `_final_se_fc_gather_static_shape_stats` with the
same stable two-key schema as the recursive fallback and replaces it with the
opt-in complete reconciliation dictionary only under the unchanged three-
counter-or-tensor-reduction guard. No reconciliation, traversal, or invocation
is added.

The three owners, raw result schemas, orchestration order, matching/planning,
rewiring, positive and zero-rewrite pruning behavior, layout synchronization,
tensor-count sample, guard expression, fallback path, dependencies, and
TensorFlow-free behavior are unchanged. The strict main-path contract is now a
normal passing contract, and the shared boundary test requires symmetric main
and fallback result assignments.

Implementation validation completed sequentially under `uv`:

- SiNet shuffle, SE/FC, Gather, orchestration, core guard, and architecture:
  `543 passed in 17.93s`
- expanded broad related gate: `1136 passed in 30.02s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the preceding `final_placeholder_matmul_stats` conditional
block, including its first shape result and nested exact/singleton binary
repairs, before changing any retained evidence. Keep the following final
SE/FC/Gather boundary fixed. Commit and push only; do not create or update a
pull request.

## Primary final placeholder-MatMul reconciliation characterization checkpoint

The placeholder restore owner increments its exact counter for every MatMul
input rewire and Reshape removal, and prunes only after a positive restore. The
first static-shape reconciliation then runs unconditionally inside that
positive outer guard. Exact binary repair preserves legacy unconditional
pruning, while singleton-broadcast repair prunes only after a positive repair;
the existing inner tensor-count sample captures cleanup-only deletion.

The first reconciliation's legacy output-shape counter participates in the
inner second-reconciliation guard. Passing its opt-in complete dictionary
directly to `_stats_have_positive_count()` would also treat parameter-only
mutations as a new reason for the second scan and would therefore change the
current guard. A strict expected-failure contract instead requires stable
`_final_placeholder_matmul_static_shape_stats` and
`_final_placeholder_binary_static_shape_stats` results, while projecting only
the legacy output-shape key into `final_placeholder_reconcile_stats`. The exact
inner guard text and following final SE/FC/Gather boundary remain fixed.

At implementation, add only these two result assignments and the in-memory
legacy projection. Do not add a reconciliation, broaden the inner guard, or
change restore/binary owners, matching, rewiring, pruning, sorting, tensor-count
sampling, pass order, dependencies, fallback behavior, or TensorFlow behavior.
Validate dynamic-Reshape restore, both binary owners, terminal orchestration,
core runtime guards, and architecture sequentially, then commit and push only;
do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- restore, binary adapters, terminal orchestration, core guard, and architecture:
  `381 passed, 1 xfailed in 18.44s`
- expanded broad related gate: `1180 passed, 1 xfailed in 29.75s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet two-result placeholder
reconciliation contract; there are no unexpected failures.

## Primary final placeholder-MatMul reconciliation implementation checkpoint

The primary path now initializes stable complete results for the restore
reconciliation and nested binary-repair reconciliation. After a positive
restore, the first existing reconciliation opts into complete accounting. Its
legacy output-shape count is projected into the original one-key
`final_placeholder_reconcile_stats` dictionary, so the exact existing
`_stats_have_positive_count(...)` guard is preserved. The second existing
reconciliation stores its complete result under that unchanged guard.

No reconciliation, graph traversal, or binary-owner invocation is added. The
restore and binary owners, raw counter schemas, matching, rewiring, pruning,
tensor-count sample, inner condition, topological sort, pass order, fallback
path, dependencies, and TensorFlow-free behavior are unchanged. Runtime tests
confirm the second reconciliation invocation count remains identical for
unchanged, first-shape, exact-binary, singleton-binary, and cleanup-only cases.

Implementation validation completed sequentially under `uv`:

- restore, binary adapters, terminal orchestration, core guard, and architecture:
  `382 passed in 19.28s`
- expanded broad related gate: `1181 passed in 30.44s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the preceding `final_mixed_singleton_concat_stats` owner and
guard before retaining its reconciliation result. Keep the following
placeholder-MatMul block fixed. Commit and push only; do not create or update a
pull request.

## Primary final mixed-singleton Concat characterization checkpoint

The indexed mixed-singleton Concat owner builds all plans before applying them,
then increments its single counter for every successfully inserted Reshape
adapter and Concat rewire plan. Unused-tensor pruning and optional layout-state
sync run only after a positive count. No-Concat, already-NHWC, unsafe/dynamic,
clone-failure, unbound, and second-run zero results are true ModelIR no-ops.

A strict expected-failure orchestration contract now requires stable two-key
`_final_mixed_singleton_concat_static_shape_stats` evidence and assigns the
opt-in complete reconciliation result under the unchanged single-counter guard.
It also fixes the following `final_placeholder_matmul_stats` boundary.

At implementation, add only caller-side result plumbing. Do not change the
owner, counter schema, planning/application, adapter construction, rewiring,
pruning, layout sync, guard, ordering, dependencies, placeholder block, or
TensorFlow behavior. Validate the indexed owner, terminal orchestration, core
runtime guard, and architecture sequentially, then commit and push only; do not
create or update a pull request.

Characterization validation completed sequentially under `uv`:

- indexed owner, terminal orchestration, core guard, and architecture:
  `356 passed, 1 xfailed in 17.83s`
- expanded broad related gate: `1212 passed, 1 xfailed in 29.97s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final mixed-singleton result
contract; there are no unexpected failures.

## Primary final mixed-singleton Concat implementation checkpoint

The primary path now initializes
`_final_mixed_singleton_concat_static_shape_stats` with the stable two-key
schema and replaces it with the opt-in complete reconciliation dictionary only
under the unchanged exact repair-counter guard. This adds no reconciliation or
graph traversal.

The indexed owner, raw counter, plan construction/application, adapter names and
metadata, Concat rewiring, positive-only pruning/layout synchronization, guard,
ordering, placeholder block, dependencies, and TensorFlow-free behavior are
unchanged. The strict characterization contract is now a normal passing
contract, and the existing architecture guard checks the retained assignment.

Implementation validation completed sequentially under `uv`:

- indexed owner, terminal orchestration, core guard, and architecture:
  `357 passed in 19.46s`
- expanded broad related gate: `1213 passed in 29.96s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the preceding `final_broadcast_repair_stats` owner and its
reconciliation/sort/layout-inference block before retaining complete evidence.
Keep the following mixed-singleton Concat boundary fixed. Commit and push only;
do not create or update a pull request.

## Primary final broadcast reconciliation characterization checkpoint

The indexed rank-four channelwise broadcast owner increments its exact counter
for every in-place constant rotation or shared-constant clone plus binary-input
rewire. It performs no pruning or topology removal, and unsupported,
already-broadcastable, unknown-layout, unsafe-shape, and second-run zero results
are ModelIR no-ops. The counter therefore completely guards shape follow-up.

A strict expected-failure orchestration contract now requires stable two-key
`_final_broadcast_static_shape_stats` evidence and assigns the opt-in complete
result at the existing first statement of the positive guard. The following
topological sort, layout inference, and `final_mixed_singleton_concat_stats`
boundary remain exact.

At implementation, add only caller-side result plumbing. Do not change the
owner, counter schema, constant clone/rotation, rewiring, guard, reconcile/sort/
infer order, dependencies, fallback behavior, following owner, or TensorFlow
behavior. Validate the binary-layout owner, terminal orchestration, convergence,
and architecture sequentially, then commit and push only; do not create or
update a pull request.

Characterization validation completed sequentially under `uv`:

- binary owner, convergence, terminal orchestration, and architecture:
  `285 passed, 1 xfailed in 17.22s`
- expanded broad related gate: `1213 passed, 1 xfailed in 30.33s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final broadcast reconciliation-
result contract; there are no unexpected failures.

## Primary final broadcast reconciliation implementation checkpoint

The primary path now initializes `_final_broadcast_static_shape_stats` with the
stable two-key schema and replaces it with the opt-in complete reconciliation
dictionary under the unchanged exact repair-counter guard. The existing
reconciliation call remains the first statement in that guard.

The indexed owner, raw counter, constant rotation and clone behavior, input
rewiring, guard, following topological sort and layout inference, mixed-
singleton boundary, dependencies, fallback path, and TensorFlow-free behavior
are unchanged. No reconciliation or graph traversal is added.

Implementation validation completed sequentially under `uv`:

- binary owner, convergence, terminal orchestration, and architecture:
  `286 passed in 17.42s`
- expanded broad related gate: `1214 passed in 30.53s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the preceding `final_instancenorm_repair_stats` owner and its
reconciliation/sort/layout-inference block before retaining complete evidence.
Keep the following broadcast repair boundary fixed. Commit and push only; do
not create or update a pull request.

## Primary final InstanceNorm reconciliation characterization checkpoint

The indexed decomposed-InstanceNorm owner constructs and validates all axes,
constant, and tensor-shape plans for a candidate before applying them. It
increments its exact counter only when at least one plan changes ModelIR,
performs no pruning or topology mutation, and synchronizes layout only after a
positive count. Missing markers, unsafe fan-out/boundaries, invalid constants,
plan failure, already-correct, and second-run zero results are transactional
ModelIR no-ops.

A strict expected-failure orchestration contract now requires stable two-key
`_final_instancenorm_static_shape_stats` evidence and assigns the opt-in
complete result as the existing first statement of the positive guard. The
following sort, layout inference, and `final_broadcast_repair_stats` boundary
remain exact.

At implementation, add only caller-side result plumbing. Do not change the
owner, plan validation/application, counter schema, constant/tensor metadata,
layout sync, guard, reconcile/sort/infer order, dependencies, following owner,
or TensorFlow behavior. Validate the indexed InstanceNorm owner, terminal
orchestration, and architecture sequentially, then commit and push only; do not
create or update a pull request.

Characterization validation completed sequentially under `uv`:

- indexed InstanceNorm owner, terminal orchestration, and architecture:
  `309 passed, 1 xfailed in 18.50s`
- expanded broad related gate: `1251 passed, 1 xfailed in 29.99s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final InstanceNorm
reconciliation-result contract; there are no unexpected failures.

## Primary final InstanceNorm reconciliation implementation checkpoint

The primary path now initializes `_final_instancenorm_static_shape_stats` with
the stable two-key schema and replaces it with the opt-in complete
reconciliation dictionary under the unchanged exact repair-counter guard. The
existing reconciliation remains the first guarded statement.

The indexed owner, plan validation/application, raw counter, constant and tensor
metadata writes, positive-only layout synchronization, guard, following sort
and layout inference, broadcast boundary, dependencies, and TensorFlow-free
behavior are unchanged. No reconciliation or graph traversal is added.

Implementation validation completed sequentially under `uv`:

- indexed InstanceNorm owner, terminal orchestration, and architecture:
  `310 passed in 18.35s`
- expanded broad related gate: `1252 passed in 30.10s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the preceding `final_convinteger_layout_stats` owner and its
reconciliation/sort/layout-inference block before retaining complete evidence.
Keep the following InstanceNorm boundary fixed. Commit and push only; do not
create or update a pull request.

## Primary final ConvInteger reconciliation characterization checkpoint

The ConvInteger owner returns two semantically distinct counters. Channel-last
hint propagation updates provenance metadata and tensor layout annotations but
is self-contained and does not require shape reconciliation or sorting. The
structural repair counter increments for every Conv input rewire and stale
Transpose removal; its positive path also updates chain shapes/layouts, prunes,
and synchronizes `LayoutState`. The existing repair-only guard is therefore the
correct boundary for reconciliation→sort→layout inference and must not be
broadened to hint-only changes.

A strict expected-failure orchestration contract now requires stable two-key
`_final_convinteger_static_shape_stats` evidence under that unchanged repair-
only guard. It explicitly rejects the propagation counter from the condition
and fixes the following `final_instancenorm_repair_stats` boundary.

At implementation, add only caller-side reconciliation result plumbing. Do not
change hint propagation, metadata/layout writes, structural repair, counter
schema, pruning, layout sync, guard, reconcile/sort/infer order, dependencies,
following owner, or TensorFlow behavior. Validate the quantized-layout owner,
terminal orchestration, and architecture sequentially, then commit and push
only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- quantized-layout owner, terminal orchestration, and architecture:
  `276 passed, 1 xfailed in 17.49s`
- expanded broad related gate: `1255 passed, 1 xfailed in 30.17s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final ConvInteger
reconciliation-result contract; there are no unexpected failures.

## Primary final ConvInteger reconciliation implementation checkpoint

The primary path now initializes `_final_convinteger_static_shape_stats` with
the stable two-key schema and replaces it with the opt-in complete
reconciliation dictionary only under the unchanged structural-repair counter
guard. Channel-last hint propagation remains outside that condition.

The owner, both raw counters, hint metadata/layout writes, structural chain
metadata/rewire/removal, positive-only pruning/layout synchronization, guard,
following sort and layout inference, InstanceNorm boundary, dependencies, and
TensorFlow-free behavior are unchanged. No reconciliation or scan is added.

Implementation validation completed sequentially under `uv`:

- quantized-layout owner, terminal orchestration, and architecture:
  `277 passed in 17.28s`
- expanded broad related gate: `1256 passed in 30.30s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the absolute-final
`_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs()` expression immediately
before the unconditional sort/layout inference and final ConvInteger owner.
Determine whether its exact raw result can be retained without adding a guard
or reconciliation. Commit and push only; do not create or update a pull request.

## Absolute-final dynamic rank-one result characterization checkpoint

The dynamic rank-one Unsqueeze/Reshape-shape owner increments its exact counter
for each shape-parameter metadata/data rewrite or runtime SHAPE/CONCAT pipeline
insertion plus RESHAPE rewire. It performs no pruning and synchronizes layout
only after a positive result. Of its three production occurrences, the very-
late and recursive-fallback calls already retain their raw dictionaries; only
the absolute-final direct call discards its result.

A strict expected-failure orchestration contract now requires that last call to
assign `_absolute_final_dynamic_rank1_stats`. It fixes all three occurrence
counts and preserves the immediately following unconditional topological sort,
layout inference, and `final_convinteger_layout_stats` boundary.

At implementation, change only the last expression to an assignment. Do not
change the owner, counter schema, option/tensor/operator writes, layout sync,
other two occurrences, add a guard or reconciliation, reorder sort/inference,
change dependencies, or affect TensorFlow behavior. Validate dynamic-Reshape,
terminal orchestration, architecture, and occurrence contracts sequentially,
then commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- dynamic-Reshape owner, terminal orchestration, and architecture:
  `287 passed, 1 xfailed in 17.91s`
- expanded broad related gate: `1256 passed, 1 xfailed in 30.47s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet absolute-final raw-result
assignment; there are no unexpected failures.

## Absolute-final dynamic rank-one result implementation checkpoint

The primary path now assigns the unchanged raw result of the absolute-final
dynamic rank-one Unsqueeze/Reshape-shape rewrite to
`_absolute_final_dynamic_rank1_stats`. Together with the existing very-late and
recursive-fallback assignments, all three production occurrences now retain
their exact mutation evidence.

The owner, counter schema, option/tensor/operator mutations, positive-only
layout synchronization, input ModelIR selection, and the following
unconditional topological sort, layout inference, and final ConvInteger owner
are unchanged. No guard, reconciliation, graph traversal, or dependency is
added. The occurrence contract was updated to require the three assignments
and reject any remaining discarded-result expression.

Implementation validation completed sequentially under `uv`:

- dynamic-Reshape owner, terminal orchestration, architecture, and occurrence
  contracts: `311 passed in 18.45s`
- expanded broad related gate: `1257 passed in 30.51s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit result propagation through
`_run_absolute_final_normalization_attention_pass_pair()` and
`run_absolute_final_normalization_attention()`. The shared recovery runner
returns per-pass results, while this boundary currently declares and discards
`None`; characterize the exact caller and ordering contract before changing
it. Keep the following absolute-final dynamic rank-one rewrite fixed. Commit
and push only; do not create or update a pull request.

## Absolute-final normalization/attention result characterization checkpoint

The absolute-final pair invokes flattened-global-normalization Pad propagation
followed by mixed reduction/MirrorPad attention cleanup through one shared
`ModelIRPassStateScope`. Each owner returns a stable mutation dictionary, and
`run_recovery_invocations()` already returns the two callback results in the
declared pass-ID order. The pair-specific runner currently omits that return,
its lowerer helper declares `None`, and the primary path invokes the helper as
a discarded expression.

Strict expected-failure contracts now require ordered result propagation
through all three boundaries: the pair runner returns the two dictionaries,
the helper returns that tuple, and the primary path assigns it to
`_absolute_final_normalization_attention_results`. The adjacent
`_absolute_final_instancenorm_post_bias_stats` and
`_absolute_final_dynamic_rank1_stats` assignments, zero-argument helper call,
pass IDs, callback options, shared state scope, execution order, and
TensorFlow-free behavior remain fixed.

At implementation, add only return/assignment plumbing and accurate tuple type
annotations. Do not add a summarizer, guard, reconciliation, pass execution,
graph traversal, dependency, or metadata write, and do not reorder either pass
or either adjacent boundary. Validate the pair owner, pass-efficiency,
architecture, terminal orchestration, and broad related gates sequentially,
then commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- pair owner, pass-efficiency, architecture, and terminal orchestration:
  `315 passed, 2 xfailed in 18.08s`
- expanded broad related gate: `1266 passed, 2 xfailed in 30.29s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The two strict xfails are the deliberately unmet runner-return and lowerer-
capture contracts; there are no unexpected failures.

## Absolute-final normalization/attention result implementation checkpoint

`run_absolute_final_normalization_attention()` now returns the ordered tuple
already produced by `run_recovery_invocations()`. The lowerer helper returns
that same tuple, and the primary path retains it as
`_absolute_final_normalization_attention_results`. The tuple contains the
flattened-global-normalization Pad result followed by the mixed-attention
MirrorPad result in the existing declared pass-ID order.

The two owners, callback arguments, shared `ModelIRPassStateScope`, diagnostics,
pass IDs, execution order, previous InstanceNorm post-bias assignment, and
following dynamic rank-one assignment are unchanged. No summarizer, guard,
reconciliation, pass invocation, graph traversal, metadata write, or dependency
is added.

Implementation validation completed sequentially under `uv`:

- pair owner, pass-efficiency, architecture, and terminal orchestration:
  `317 passed in 20.12s`
- expanded broad related gate: `1268 passed in 31.17s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the discarded dictionaries from
`_realign_dynamic_boundary_shape_signature_map()` and
`_sanitize_static_shape_signature_consistency()` immediately before the
absolute-final affine/normalization/attention sequence. Characterize their
exact mutation schemas and adjacent ordering before retaining any result. Keep
all existing final owners fixed. Commit and push only; do not create or update
a pull request.

## Absolute-final boundary-signature result characterization checkpoint

Dynamic-boundary signature realignment returns the exact count of metadata-map
entries changed. It has three direct primary-path occurrences: the shared-late
and terminal calls already retain their dictionaries, while the call directly
before the absolute-final sequence discards its result. Static-signature
sanitization returns one repair counter plus three dynamic-preservation
counters. Its earlier late call retains the complete dictionary, while its
absolute-final call discards it.

A strict expected-failure orchestration contract now requires assigning only
those two discarded results to `_absolute_final_boundary_signature_stats` and
`_absolute_final_static_signature_stats`. It fixes all occurrence counts and
targets, the realign→sanitize adjacency, and the immediately following
`_absolute_final_affine_post_add_stats` boundary.

At implementation, replace only the two expressions with assignments. Do not
change either owner or schema, metadata/tensor writes, other occurrences, add a
guard, reconciliation, scan, sort, dependency, or TensorFlow behavior, or
reorder the following absolute-final owners. Validate signature-owner,
terminal-orchestration, architecture, and broad related gates sequentially,
then commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- signature owner, terminal orchestration, and architecture:
  `294 passed, 1 xfailed in 17.57s`
- expanded broad related gate: `1268 passed, 1 xfailed in 30.76s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet two-result capture contract;
there are no unexpected failures.

## Absolute-final boundary-signature result implementation checkpoint

The absolute-final dynamic-boundary realignment call now retains its unchanged
one-counter dictionary as `_absolute_final_boundary_signature_stats`. The
immediately following static-signature sanitizer retains its complete
four-counter repair/preservation dictionary as
`_absolute_final_static_signature_stats`. All direct production occurrences of
both owners now retain their raw mutation evidence.

The owners, result schemas, metadata/tensor writes, other occurrence targets,
realign→sanitize order, and following absolute-final affine,
normalization/attention, dynamic-rank-one, and layout sequence are unchanged.
No guard, reconciliation, scan, sort, metadata write, dependency, or
TensorFlow behavior is added.

Implementation validation completed sequentially under `uv`:

- signature owner, terminal orchestration, absolute-final orchestration, and
  architecture: `306 passed in 17.88s`
- expanded broad related gate: `1269 passed in 30.78s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the two discarded results inside the guarded
`apply_safe_transpose_reduction_lite_on_no_layout_opt` block immediately before
the completed boundary-signature restore. Characterize their schemas, guarded
execution, and surrounding sort/signature boundaries before retaining any
evidence. Commit and push only; do not create or update a pull request.

## Guarded no-layout final cleanup result characterization checkpoint

When safe transpose reduction is enabled without the full layout optimizer,
the final cleanup reruns SE/FC layout propagation and the indexed constant
affine pre/post owner after final topological normalization, then sorts once
more. Each owner returns a stable one-counter dictionary and already handles
its own graph-index, pruning, and layout-state synchronization. This guarded
occurrence discards both results.

A strict expected-failure orchestration contract now requires assigning the
raw dictionaries to `_no_layout_final_se_fc_stats` and
`_no_layout_final_affine_prepost_stats`. It fixes the guard expression, both
calls and keyword contracts, the preceding and guarded topological sorts, and
the following `_absolute_final_boundary_signature_stats` boundary.

At implementation, replace only the two guarded expressions with assignments.
Do not initialize or consume the variables outside the guard, change either
owner/schema, add a pass, guard, reconciliation, scan, sort, dependency, or
metadata write, alter layout-state/diagnostics handoff, or affect the normal
layout-optimized path or TensorFlow boundary. Validate SE/FC, indexed affine,
terminal orchestration, architecture, and broad related gates sequentially,
then commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- SE/FC, indexed affine, terminal orchestration, architecture, and pass
  efficiency: `416 passed, 1 xfailed in 18.02s`
- expanded broad related gate: `1371 passed, 1 xfailed in 30.72s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet guarded two-result capture
contract; there are no unexpected failures.

## Guarded no-layout final cleanup result implementation checkpoint

Inside the unchanged safe-transpose-reduction fallback guard, the final SE/FC
layout cleanup now retains its raw dictionary as
`_no_layout_final_se_fc_stats`, and the following indexed constant affine
pre/post cleanup retains its raw dictionary as
`_no_layout_final_affine_prepost_stats`. Neither variable is initialized or
consumed outside that conditional path.

The owners, schemas, guarded execution, callback arguments, diagnostics and
layout-state handoff, preceding and guarded topological sorts, following
boundary-signature restore, normal layout-optimized path, dependencies, and
TensorFlow-free behavior are unchanged. No pass, guard, reconciliation, scan,
sort, metadata write, or result summarizer is added.

Implementation validation completed sequentially under `uv`:

- SE/FC, indexed affine, terminal orchestration, architecture, and pass
  efficiency: `417 passed in 17.82s`
- expanded broad related gate: `1372 passed in 30.75s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit result propagation through the final precision cleanup
sequence (`_rewrite_constant_divisors_to_multiplicative_reciprocals()`,
`run_consecutive_mul_constants_cleanup()`, and
`_restore_precision_sensitive_reciprocal_divisions()`) immediately before the
post-progress topological sort. Characterize schemas, exact occurrence counts,
and surrounding order before retaining any evidence. Commit and push only; do
not create or update a pull request.

## Primary final precision-cleanup result characterization checkpoint

The final precision sequence first rewrites eligible floating-point constant
DIV operators to reciprocal MUL, then folds guarded consecutive constant MUL
chains, and finally restores DIV where reciprocal multiplication feeds an
integer cast through a precision-sensitive affine lineage. Each owner returns
a stable one-counter dictionary and performs its own indexed mutation,
pruning, diagnostics, and layout synchronization as applicable.

The lowerer contains two divisor-rewrite calls, three consecutive-Mul cleanup
calls, and two precision-restore calls. A strict expected-failure orchestration
contract selects only the adjacent primary final trio and requires raw result
targets `_final_precision_div_rewrite_stats`,
`_final_precision_consecutive_mul_stats`, and
`_final_precision_div_restore_stats`. Earlier core-cleanup and recursive-
fallback occurrences remain unchanged, as does the following post-progress
description and topological-sort boundary.

At implementation, replace only the three primary final expressions with
assignments. Do not change any owner/schema, other occurrence, callback
argument, diagnostics/layout-state handoff, add a guard, reconciliation, scan,
sort, dependency, metadata write, or TensorFlow behavior, or consume the new
variables. Validate precision, graph cleanup, terminal orchestration,
architecture, pass efficiency, and broad related gates sequentially, then
commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- precision, graph cleanup, terminal orchestration, architecture, and pass
  efficiency: `331 passed, 1 xfailed in 18.06s`
- expanded broad related gate: `1395 passed, 1 xfailed in 31.22s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final three-result capture
contract; there are no unexpected failures.

## Primary final precision-cleanup result implementation checkpoint

The primary final precision sequence now retains the unchanged divisor-rewrite,
consecutive-Mul-fold, and precision-sensitive divisor-restore dictionaries as
`_final_precision_div_rewrite_stats`,
`_final_precision_consecutive_mul_stats`, and
`_final_precision_div_restore_stats`, respectively.

All three owners and schemas, indexed mutation, pruning, diagnostics and
layout-state synchronization, callback arguments, earlier core-cleanup and
recursive-fallback occurrences, following post-progress description and sort,
dependencies, and TensorFlow-free behavior are unchanged. No guard,
reconciliation, scan, sort, metadata write, or result consumption is added.

Implementation validation completed sequentially under `uv`:

- precision, graph cleanup, terminal orchestration, architecture, and pass
  efficiency: `332 passed in 17.67s`
- expanded broad related gate: `1396 passed in 30.84s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the corresponding recursive-fallback precision trio. Its
callbacks operate on `fallback_ir`, omit layout-state handoff, and sit between
the fallback topological sort and unbound-input repair. Characterize those
exact differences before retaining results. Commit and push only; do not create
or update a pull request.

## Recursive-fallback precision-cleanup result characterization checkpoint

The safety fallback runs the same divisor rewrite→consecutive-Mul fold→
precision-sensitive divisor restore sequence over `fallback_ir` immediately
after placeholder-MatMul reconciliation and a topological sort. Unlike the
primary final trio, the direct precision owners receive no layout state; only
the transactional consecutive-Mul runner receives `session.diagnostics`. All
three returned one-counter dictionaries are currently discarded.

A strict expected-failure orchestration contract requires assigning the raw
results to `_fallback_precision_div_rewrite_stats`,
`_fallback_precision_consecutive_mul_stats`, and
`_fallback_precision_div_restore_stats`. The fallback input, exact keyword
contracts, preceding sort, and following `_fallback_unbound_repair_stats`
boundary remain fixed.

At implementation, replace only these three fallback expressions with
assignments. Do not change the owners/schemas, primary final or earlier core-
cleanup occurrences, add layout-state handoff, a guard, reconciliation, scan,
sort, dependency, metadata write, or result consumer, or affect TensorFlow
behavior. Validate safety fallback, precision, graph cleanup, terminal
orchestration, architecture, and broad related gates sequentially, then commit
and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- safety fallback, precision, graph cleanup, terminal orchestration, and
  architecture: `318 passed, 1 xfailed in 17.85s`
- expanded broad related gate: `1396 passed, 1 xfailed in 31.03s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet fallback three-result capture
contract; there are no unexpected failures.

## Recursive-fallback precision-cleanup result implementation checkpoint

The safety fallback now retains its unchanged divisor-rewrite,
consecutive-Mul-fold, and precision-sensitive divisor-restore dictionaries as
`_fallback_precision_div_rewrite_stats`,
`_fallback_precision_consecutive_mul_stats`, and
`_fallback_precision_div_restore_stats`, respectively.

The owners/schemas, `fallback_ir` input, deliberate absence of layout-state
handoff, consecutive-Mul diagnostics, preceding sort, following unbound-input
repair, primary final and earlier core-cleanup occurrences, dependencies, and
TensorFlow-free behavior are unchanged. No guard, reconciliation, scan, sort,
metadata write, or result consumption is added.

Implementation validation completed sequentially under `uv`:

- safety fallback, precision, graph cleanup, terminal orchestration, and
  architecture: `319 passed in 18.49s`
- expanded broad related gate: `1397 passed in 31.63s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the earlier core-cleanup
`run_consecutive_mul_constants_cleanup()` expression after the pseudo-LeakyReLU
and YOLO-decode cleanups. It is the only remaining discarded result among these
three precision owners. Fix its surrounding core-cleanup order before retaining
evidence. Commit and push only; do not create or update a pull request.

## Primary core-cleanup consecutive-Mul result characterization checkpoint

The remaining discarded precision-owner result is the earlier primary
`run_consecutive_mul_constants_cleanup()` call in the core-cleanup phase. It
runs after pseudo-LeakyReLU fusion and YOLO decode Mul-square/anchor cleanup,
and immediately before terminal Transpose/Dequantize sanitization. Its ModelIR,
layout-state, diagnostics, and one-counter schema match the already captured
primary-final occurrence.

A strict expected-failure orchestration contract requires assigning this call
to `_core_cleanup_consecutive_mul_stats`. It fixes both direct primary
occurrences and the existing `_final_precision_consecutive_mul_stats` target,
plus the exact pseudo-LeakyReLU→YOLO→consecutive-Mul→terminal-sanitizer order.
The recursive-fallback occurrence remains independently captured.

At implementation, replace only this one expression with an assignment. Do not
change the owner/schema, any other occurrence or target, callback arguments,
diagnostics/layout-state handoff, add a guard, reconciliation, scan, sort,
dependency, metadata write, or result consumer, or affect TensorFlow behavior.
Validate graph cleanup, terminal orchestration, architecture, pass efficiency,
and broad related gates sequentially, then commit and push only; do not create
or update a pull request.

Characterization validation completed sequentially under `uv`:

- graph cleanup, terminal orchestration, architecture, and pass efficiency:
  `329 passed, 1 xfailed in 17.58s`
- expanded broad related gate: `1397 passed, 1 xfailed in 31.02s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet core-cleanup result capture
contract; there are no unexpected failures.

## Primary core-cleanup consecutive-Mul result implementation checkpoint

The earlier primary core-cleanup call now retains the unchanged transactional
consecutive-Mul dictionary as `_core_cleanup_consecutive_mul_stats`. Together
with `_fallback_precision_consecutive_mul_stats` and
`_final_precision_consecutive_mul_stats`, every production occurrence of this
owner now retains its raw mutation evidence.

The owner/schema, ModelIR, layout-state and diagnostics handoff, preceding
pseudo-LeakyReLU/YOLO cleanups, following terminal Transpose/Dequantize
sanitizer, other occurrence targets, dependencies, and TensorFlow-free behavior
are unchanged. No guard, reconciliation, scan, sort, metadata write, or result
consumption is added.

Implementation validation completed sequentially under `uv`:

- graph cleanup, terminal orchestration, architecture, and pass efficiency:
  `330 passed in 18.00s`
- expanded broad related gate: `1398 passed in 31.86s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the immediately preceding
`_optimize_fuse_pseudo_leakyrelu_chains()` and
`_optimize_yolo_decode_mul_square_anchor_chains()` core-cleanup expressions.
Characterize their schemas, occurrence counts, and fixed progress/consecutive-
Mul boundaries before retaining evidence. Commit and push only; do not create
or update a pull request.

## Core-cleanup pseudo-LeakyReLU/YOLO result characterization checkpoint

The primary core-cleanup phase begins with guarded pseudo-LeakyReLU fusion,
followed by YOLO decode `Mul(x,x)` plus anchor-constant folding. Each owner has
one direct lowerer occurrence and returns a stable one-counter dictionary. The
pseudo-LeakyReLU owner uses an indexed topology/fan-out proof; the YOLO owner
prevalidates finite constants and shared-input/consumer boundaries. Both caller
results are currently discarded.

A strict expected-failure orchestration contract requires assigning the raw
results to `_core_cleanup_pseudo_leakyrelu_stats` and
`_core_cleanup_yolo_decode_stats`. It fixes the `core cleanup passes` progress
boundary, pseudo→YOLO order, exact arguments, and the following captured
`_core_cleanup_consecutive_mul_stats` boundary.

At implementation, replace only these two expressions with assignments. Do not
change either owner/schema, add GraphIndex/layout arguments, alter any matcher,
guard, pruning, add a reconciliation, scan, sort, dependency, metadata write,
or result consumer, reorder the core cleanup, or affect TensorFlow behavior.
Validate indexed pseudo-LeakyReLU, indexed YOLO constant folding, graph cleanup,
terminal orchestration, architecture, pass efficiency, and broad related gates
sequentially, then commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- indexed pseudo-LeakyReLU, indexed YOLO folding, graph cleanup, terminal
  orchestration, architecture, and pass efficiency:
  `363 passed, 1 xfailed in 18.51s`
- expanded broad related gate: `1431 passed, 1 xfailed in 31.17s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet two-result capture contract;
there are no unexpected failures.

## Core-cleanup pseudo-LeakyReLU/YOLO result implementation checkpoint

The primary core-cleanup phase now retains the unchanged pseudo-LeakyReLU
fusion and YOLO decode constant-fold dictionaries as
`_core_cleanup_pseudo_leakyrelu_stats` and
`_core_cleanup_yolo_decode_stats`, respectively.

Both owners/schemas, indexed topology and constant safety guards, pruning,
single occurrence counts, `model_ir` arguments, core-progress boundary,
pseudo→YOLO→consecutive-Mul order, dependencies, and TensorFlow-free behavior
are unchanged. No GraphIndex/layout argument, guard, reconciliation, scan,
sort, metadata write, or result consumption is added.

Implementation validation completed sequentially under `uv`:

- indexed pseudo-LeakyReLU, indexed YOLO folding, graph cleanup, terminal
  orchestration, architecture, and pass efficiency: `364 passed in 18.37s`
- expanded broad related gate: `1432 passed in 31.30s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the immediately following
`_sanitize_terminal_transpose_before_dequantize()` and
`run_terminal_quantize_dequantize_cleanup()` expressions. Characterize their
schemas, occurrence counts, exact diagnostics/layout handoff, and surrounding
consecutive-Mul/Conv-affine boundaries before retaining evidence. Commit and
push only; do not create or update a pull request.

## Terminal quantization-cleanup result characterization checkpoint

The primary path contains two identical ordered cleanup pairs: one immediately
after the captured core-cleanup consecutive-Mul result, and one after late
recovery sweeps under the `terminal cleanup passes` progress stage. Each pair
first normalizes or removes terminal Transpose/Dequantize boundaries, returning
two counters, then transactionally removes exact-grid terminal
Quantize/Dequantize pairs, returning one counter. Both pairs precede Conv-affine
folding, and all four caller results are currently discarded.

A strict expected-failure orchestration contract requires targets
`_core_cleanup_terminal_dequant_stats`,
`_core_cleanup_terminal_qdq_stats`,
`_terminal_cleanup_terminal_dequant_stats`, and
`_terminal_cleanup_terminal_qdq_stats`. It fixes both occurrence counts,
pairwise order, ModelIR/layout-state/diagnostics contracts, distinct preceding
boundaries, and both Conv-affine successors.

At implementation, replace only the four expressions with assignments. Do not
change either owner/schema, pass transaction, GraphIndex/pruning behavior,
callback arguments, progress stages, add a guard, reconciliation, scan, sort,
dependency, metadata write, or result consumer, reorder either pair, or affect
TensorFlow behavior. Validate quantization cleanup, terminal orchestration,
architecture, pass efficiency, and broad related gates sequentially, then
commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- quantization cleanup, terminal orchestration, architecture, and pass
  efficiency: `358 passed, 1 xfailed in 17.86s`
- expanded broad related gate: `1479 passed, 1 xfailed in 31.00s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet four-result capture contract;
there are no unexpected failures.

## Terminal quantization-cleanup result implementation checkpoint

The core-cleanup pair now retains its terminal Transpose/Dequantize sanitizer
and exact-grid Q/DQ dictionaries as `_core_cleanup_terminal_dequant_stats` and
`_core_cleanup_terminal_qdq_stats`. The later terminal-cleanup pair retains the
same owner results as `_terminal_cleanup_terminal_dequant_stats` and
`_terminal_cleanup_terminal_qdq_stats`.

Both owners/schemas, pair occurrence counts and order, GraphIndex/pruning and
transaction behavior, ModelIR/layout-state/diagnostics handoff, distinct
preceding progress/consecutive-Mul boundaries, following Conv-affine folds,
dependencies, and TensorFlow-free behavior are unchanged. No guard,
reconciliation, scan, sort, metadata write, or result consumption is added.

Implementation validation completed sequentially under `uv`:

- quantization cleanup, terminal orchestration, architecture, and pass
  efficiency: `359 passed in 17.79s`
- expanded broad related gate: `1480 passed in 31.26s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the two following
`_optimize_fold_conv_mul_add_affine_chains()` and
`_optimize_fuse_conv_activation_chains()` pairs. Characterize both occurrence
schemas, the shared layout-state/Conv-ADD option contracts, and their fixed Q/DQ
predecessors before retaining evidence. Commit and push only; do not create or
update a pull request.

## Quantization-successor Conv result characterization checkpoint

Each captured terminal quantization-cleanup pair is followed by Conv MUL/ADD
affine folding and Conv/binary activation fusion. The affine owner returns a
four-counter total/category dictionary and has three direct primary-path
occurrences. Activation fusion returns seven Conv/binary/total counters and has
two direct occurrences, both in these pairs. The first two affine results and
both activation results are currently discarded.

A strict expected-failure orchestration contract requires phase-specific
targets `_core_cleanup_conv_affine_stats`,
`_core_cleanup_conv_activation_stats`,
`_terminal_cleanup_conv_affine_stats`, and
`_terminal_cleanup_conv_activation_stats`. It fixes the shared Conv-ADD option
and layout-state contracts, captured Q/DQ predecessors, and distinct dynamic-
Reshape/ArgMax successors. The third later affine expression remains explicitly
unchanged for a separate audit.

At implementation, replace only these four expressions with assignments. Do
not change either owner/schema, the third affine call, callback arguments,
indexed mutation/pruning/layout sync, add a guard, reconciliation, scan, sort,
dependency, metadata write, or result consumer, reorder either pair, or affect
TensorFlow behavior. Validate indexed Conv-affine folding, activation fusion,
terminal convergence/orchestration, architecture, and broad related gates
sequentially, then commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused Conv-affine/activation, terminal convergence/orchestration, and
  architecture gate: `321 passed, 1 xfailed in 17.90s`
- expanded broad related gate: `1507 passed, 1 xfailed in 31.15s`

The sole strict expected failure is the intentionally unimplemented four-result
retention contract above.

## Quantization-successor Conv result implementation checkpoint

The two Conv MUL/ADD affine-fold results and the two Conv/binary activation-
fusion results immediately following the captured terminal quantization pairs
are now retained under the four phase-specific targets fixed by
characterization. The pass calls, options, layout-state handoff, pair order,
mutation behavior, Q/DQ predecessors, and dynamic-Reshape/ArgMax successors are
unchanged. No guard, reconciliation, scan, sort, metadata write, or result
consumer was added. The third later Conv-affine expression remains unchanged.

Implementation validation completed sequentially under `uv`:

- focused Conv-affine/activation, terminal convergence/orchestration, and
  architecture gate: `322 passed in 19.48s`
- expanded broad related gate: `1508 passed in 30.41s`

Ruff, Python bytecode compilation, and `git diff --check` passed. At resume,
audit the third direct
`_optimize_fold_conv_mul_add_affine_chains()` occurrence between cost-volume
cleanup and `late_concat_layout_state_scope` before retaining its result.

## Late cost-volume Conv-affine result characterization checkpoint

The third and final direct Conv MUL/ADD affine-fold occurrence uses the same
stable four-counter owner, `enable_conv_add_only_fold=True`, and live
`session.layout_state` contract as the two captured terminal successors. Its
result is currently discarded. It is the raw boundary after the shared
`late_ndhwc_cost_volume_state_scope` pair and before construction of
`late_concat_layout_state_scope`.

A strict expected-failure orchestration contract requires the result under
`_late_cost_volume_conv_affine_stats`. It fixes the two adjacent state-scope
assignments, the NDHWC-gate and cost-volume runner predecessors, and the exact
owner arguments. The two earlier captured affine occurrences remain unchanged.

At implementation, replace only this final expression with an assignment. Do
not change the owner/schema, either state scope, callback arguments, pass order,
indexed/fallback mutation and pruning, layout synchronization, diagnostics,
add a guard, reconciliation, scan, sort, metadata write, result consumer,
dependency, or TensorFlow behavior. Validate the Conv-affine owner, both scope
clusters, pass efficiency, terminal orchestration, architecture, and broad
related gates sequentially, then commit and push only; do not create or update
a pull request.

Characterization validation completed sequentially under `uv`:

- focused Conv-affine, NDHWC/cost-volume, pass-efficiency, terminal-
  orchestration, and architecture gate: `372 passed, 1 xfailed in 18.44s`
- expanded broad related gate: `1556 passed, 1 xfailed in 31.03s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented late
cost-volume result-retention contract above.

## Late cost-volume Conv-affine result implementation checkpoint

The third and final direct Conv MUL/ADD affine-fold result is now retained as
`_late_cost_volume_conv_affine_stats`. This completes raw result capture for all
three production occurrences of the stable four-counter owner. Its exact
options, live layout-state handoff, shared NDHWC/cost-volume predecessor scope,
and following late-Concat scope construction are unchanged.

No owner/schema, callback argument, pass order, indexed/fallback mutation or
pruning, layout synchronization, diagnostic, guard, reconciliation, scan, sort,
metadata write, result consumer, dependency, or TensorFlow behavior changed.

Implementation validation completed sequentially under `uv`:

- focused Conv-affine, NDHWC/cost-volume, pass-efficiency, terminal-
  orchestration, and architecture gate: `373 passed in 18.53s`
- expanded broad related gate: `1557 passed in 32.00s`

At resume, audit the four result schemas and occurrence contracts for
`run_axis3_const_concat_layout_cleanup()`,
`run_dequant_concat_quantize_layout_cleanup()`,
`run_layernorm_statistics_layout_cleanup()`, and
`run_layout_transpose_cleanup()` inside `late_concat_layout_state_scope` before
retaining evidence. Commit and push only; do not create or update a pull
request.

## Late Concat shared-scope result characterization checkpoint

The four runners inside `late_concat_layout_state_scope` return stable mutation
dictionaries: axis-3 constant-Concat and Dequantize/Concat/Quantize each return
one counter, LayerNorm statistics returns two counters, and generic Transpose
cleanup returns five counters including its non-mutating iteration count. All
four results are currently discarded.

A strict expected-failure orchestration contract requires, in order,
`_late_concat_axis3_const_layout_stats`,
`_late_concat_dequant_quantize_layout_stats`,
`_late_concat_layernorm_layout_stats`, and
`_late_concat_transpose_layout_stats`. It fixes the captured late cost-volume
affine predecessor, exact shared-scope/diagnostic/layout arguments, and the
following optimize-layout guard. The other two lowerer Transpose-cleanup
occurrences remain expressions.

At implementation, replace only these four adjacent expressions with
assignments. Do not change any owner/schema, shared scope, callback argument,
pass order, transactional mutation or rollback, GraphIndex/pruning/layout
synchronization, diagnostic, guard, reconciliation, scan, sort, metadata write,
result consumer, dependency, or TensorFlow behavior. Validate all four owners,
the shared-scope efficiency and architecture contracts, terminal orchestration,
and broad related gates sequentially, then commit and push only; do not create
or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused four-owner, shared-scope efficiency, terminal-orchestration, and
  architecture gate: `386 passed, 1 xfailed in 18.03s`
- expanded broad related gate: `1629 passed, 1 xfailed in 31.76s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented four-result
retention contract above.

## Late Concat shared-scope result implementation checkpoint

The four adjacent late Concat runner results are now retained as
`_late_concat_axis3_const_layout_stats`,
`_late_concat_dequant_quantize_layout_stats`,
`_late_concat_layernorm_layout_stats`, and
`_late_concat_transpose_layout_stats`. Their stable one-, one-, two-, and
five-counter dictionaries are preserved without projection or aggregation.

The shared `late_concat_layout_state_scope`, exact callback arguments,
transactional mutation/rollback, GraphIndex/pruning/layout synchronization,
diagnostics, pass order, captured affine predecessor, following optimize-layout
guard, and two other lowerer Transpose-cleanup occurrences are unchanged. No
guard, reconciliation, scan, sort, metadata write, result consumer, dependency,
or TensorFlow behavior changed.

Implementation validation completed sequentially under `uv`:

- focused four-owner, shared-scope efficiency, terminal-orchestration, and
  architecture gate: `387 passed in 17.89s`
- expanded broad related gate: `1630 passed in 32.11s`

At resume, audit both guarded production occurrences of
`_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains()` and their
one-counter schema before retaining their results. Commit and push only; do not
create or update a pull request.

## Guarded elementwise-fanout result characterization checkpoint

The elementwise NHWC→NCHW fanout-roundtrip owner returns the stable one-counter
dictionary
`optimized_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains`. Its two
production occurrences are both guarded by
`optimize_layout_transpose_chains`, and both results are currently discarded.

A strict expected-failure orchestration contract requires
`_late_concat_elementwise_fanout_stats` after the captured late-Concat
Transpose dictionary and `_terminal_elementwise_fanout_stats` before terminal
singleton-MaxPool/Reshape orchestration. It fixes both guards, the model-only
callback contract, and their distinct preceding/following boundaries.

At implementation, replace only the two guarded expressions with assignments.
Do not initialize values outside the guards, consume the results, or change the
owner/schema, guards, callback arguments, pass order, rollback/pruning,
metadata, dependencies, TensorFlow behavior, or any surrounding owner. Validate
the owner, terminal singleton boundary, terminal orchestration, architecture,
and broad related gates sequentially, then commit and push only; do not create
or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused elementwise-fanout owner, terminal-singleton boundary, terminal-
  orchestration, layout-recovery, and architecture gate:
  `298 passed, 1 xfailed in 19.61s`
- expanded broad related gate: `1644 passed, 1 xfailed in 33.76s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented guarded
two-result retention contract above.

## Guarded elementwise-fanout result implementation checkpoint

Both production results are now retained inside their existing guards as
`_late_concat_elementwise_fanout_stats` and
`_terminal_elementwise_fanout_stats`. Their unchanged one-counter dictionaries
remain unavailable when `optimize_layout_transpose_chains` is false; no
guard-external default or result consumer was added.

The owner/schema, model-only callback contract, guards, pass order,
rollback/pruning, metadata, dependencies, and four surrounding boundaries are
unchanged. TensorFlow behavior is unaffected.

Implementation validation completed sequentially under `uv`:

- focused elementwise-fanout owner, terminal-singleton boundary, terminal-
  orchestration, layout-recovery, and architecture gate:
  `299 passed in 19.15s`
- expanded broad related gate: `1645 passed in 32.11s`

At resume, audit the adjacent
`_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains()` and
`_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains()` result
schemas and occurrence boundaries before retaining evidence.

Workflow policy: pull request #955 was closed at the user's direction. Do not
create, reopen, or update any pull request. Future units must use commits and
pushes only.

## Late ExpandDims/flatten-HW result characterization checkpoint

The adjacent ExpandDims and flatten-HW Transpose/Reshape compatibility owners
each return a stable one-counter dictionary, each has exactly one production
occurrence, and both receive `model_ir` plus the live
`session.layout_state`. Their results are currently discarded.

A strict expected-failure orchestration contract requires
`_late_expanddims_reshape_layout_stats` followed by
`_late_flatten_hw_reshape_layout_stats`. It fixes the captured guarded
late-Concat fanout predecessor, exact adjacency and callback contracts, and the
following reshape/transpose-to-NHWC-Reshape owner.

At implementation, replace only these two expressions with assignments. Do not
change either compatibility/indexed owner or schema, fallback behavior,
GraphIndex/pruning/layout synchronization, callback arguments, pass order,
guard, reconciliation, scan, sort, metadata write, result consumer, dependency,
or TensorFlow behavior. Validate both indexed/compatibility owners, terminal
orchestration, architecture, and broad related gates sequentially, then commit
and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused indexed/compatibility owner, terminal-orchestration, and architecture
  gate: `300 passed, 1 xfailed in 18.25s`
- expanded broad related gate: `1660 passed, 1 xfailed in 32.09s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented adjacent
two-result retention contract above.

## Late ExpandDims/flatten-HW result implementation checkpoint

The two adjacent compatibility-owner dictionaries are now retained as
`_late_expanddims_reshape_layout_stats` and
`_late_flatten_hw_reshape_layout_stats`. Their stable one-counter schemas,
indexed-first/fallback behavior, GraphIndex/pruning/layout synchronization,
live Session LayoutState, exact order, and outer boundaries are unchanged.

No guard, reconciliation, scan, sort, metadata write, result consumer,
dependency, or TensorFlow behavior changed.

Implementation validation completed sequentially under `uv`:

- focused indexed/compatibility owner, terminal-orchestration, and architecture
  gate: `301 passed in 18.71s`
- expanded broad related gate: `1661 passed in 32.15s`

At resume, audit the immediately following
`_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains()` result,
its owner schema, and its captured predecessor/successor boundaries. Commit and
push only; do not create, reopen, or update a pull request.

## Late NHWC-Reshape collapse result characterization checkpoint

The private rank-three reshape/transpose layout-shim collapse owner returns the
stable one-counter dictionary
`optimized_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains`. It
accepts only `model_ir`, owns one direct production occurrence, and currently
discards that result.

A strict expected-failure orchestration contract requires
`_late_nhwc_reshape_collapse_stats`. It fixes the captured flatten-HW
predecessor, model-only callback contract, and following channel-shuffle/Gather
cluster invocation with both optional shuffle families disabled.

At implementation, replace only this expression with an assignment. Do not
change the owner/schema, GraphIndex mutation/pruning, callback arguments, pass
order, neighboring targets or cluster options, add a guard, reconciliation,
scan, sort, metadata write, result consumer, dependency, or TensorFlow behavior.
Validate the owner, channel-shuffle boundary, layout recovery, terminal
orchestration, architecture, and broad related gates sequentially, then commit
and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused collapse-owner, channel-shuffle boundary, layout-recovery, terminal-
  orchestration, and architecture gate: `364 passed, 1 xfailed in 18.71s`
- expanded broad related gate: `1733 passed, 1 xfailed in 33.34s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented single-
result retention contract above.

## Late NHWC-Reshape collapse result implementation checkpoint

The single production result is now retained as
`_late_nhwc_reshape_collapse_stats`. Its unchanged one-counter dictionary is
captured without projection or consumption. The owner-internal GraphIndex,
positive-only pruning, model-only callback contract, captured flatten-HW
predecessor, and channel-shuffle/Gather successor options remain fixed.

No guard, reconciliation, scan, sort, metadata write, dependency, or TensorFlow
behavior changed. An initially misapplied AST-test edit was reverted before the
focused gate; no unrelated production or test contract remains changed.

Implementation validation completed sequentially under `uv`:

- focused collapse-owner, channel-shuffle boundary, layout-recovery, terminal-
  orchestration, and architecture gate: `365 passed in 20.07s`
- expanded broad related gate: `1734 passed in 32.13s`

At resume, audit the return schema of `run_channel_shuffle_gather()`, the local
`_run_channel_shuffle_gather_layout_pass_cluster()` helper's current `None`
contract, all helper invocations/policy combinations, and the late base-policy
boundary before propagating and retaining its result. Commit and push only; do
not create, reopen, or update a pull request.

## Channel-shuffle/Gather result propagation characterization checkpoint

`run_channel_shuffle_gather()` selects two through seven transactional child
passes from three independent policy flags. `run_recovery_invocations()`
already produces their ordered dictionaries, but the runner discards that
tuple. The local `_run_channel_shuffle_gather_layout_pass_cluster()` helper is
also annotated `None` and discards the runner result. Its two production calls
therefore expose no mutation evidence.

Two strict expected-failure contracts require the runner to return every
policy's exact ordered tuple and the helper to propagate it as
`Tuple[Dict[str, int], ...]`. The guarded full-post call must retain
`_layout_opt_channel_shuffle_gather_results`; the late base-only call must
retain `_late_channel_shuffle_gather_results`. Existing policy arguments,
shared-scope construction, the captured NHWC-Reshape predecessor, and following
QKV-attention boundary are fixed.

At implementation, return the already-created tuple through both layers and
replace only the two invocation expressions with assignments. Do not aggregate
or consume results, alter policy selection, callback order/arguments,
transactional behavior, shared scope, diagnostics, guards, boundaries,
dependencies, or TensorFlow behavior. Validate all eight policies, helper AST,
shared-state efficiency, boundaries, architecture, and broad related gates
sequentially, then commit and push only; do not create, reopen, or update a pull
request.

Characterization validation completed sequentially under `uv`:

- focused all-policy runner, helper AST, pass-efficiency, layout-recovery,
  terminal-orchestration, and architecture gate:
  `346 passed, 2 xfailed in 17.79s`
- expanded broad related gate: `1734 passed, 2 xfailed in 32.30s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The two strict expected failures are the deliberately unimplemented runner/
helper return propagation and the two production result assignments.

## Channel-shuffle/Gather result propagation implementation checkpoint

`run_channel_shuffle_gather()` now returns the existing ordered child tuple as
`Tuple[Dict[str, int], ...]`, and the local helper propagates the same type and
value. No child invocation, scan, copy, aggregation, or summary traversal was
added. The guarded full-post and unguarded late-base results are retained as
`_layout_opt_channel_shuffle_gather_results` and
`_late_channel_shuffle_gather_results`.

All eight policy combinations preserve exact selected-pass order. Shared state
scope, layout/diagnostic arguments, transaction behavior, guard placement,
captured NHWC-Reshape predecessor, pre-ADD/mean-attention and QKV-attention
successors, dependencies, and TensorFlow behavior are unchanged.

Implementation validation completed sequentially under `uv`:

- focused all-policy runner, helper AST, pass-efficiency, layout-recovery,
  terminal-orchestration, and architecture gate: `348 passed in 19.69s`
- expanded broad related gate: `1736 passed in 31.42s`

At resume, audit the immediately following
`_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains()`
result, owner schema, live LayoutState contract, and channel-shuffle/next-owner
boundaries before retaining evidence. Commit and push only; do not create,
reopen, or update a pull request.

## Late attention-QKV Reshape result characterization checkpoint

The attention-QKV Reshape/Transpose compatibility owner returns the stable
one-counter dictionary
`optimized_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains`.
It uses indexed-first/fallback handling, has one direct production occurrence,
and receives the live Session LayoutState. Its result is currently discarded.

A strict expected-failure orchestration contract requires
`_late_attention_qkv_reshape_stats`. It fixes the captured base-only
channel-shuffle/Gather tuple predecessor, exact ModelIR/LayoutState callback,
and following attention Gather/Transpose/Reshape cleanup owner.

At implementation, replace only this expression with an assignment. Do not
change the compatibility/indexed owner or schema, fallback behavior,
GraphIndex/pruning/layout synchronization, callback arguments, pass order,
neighboring targets, add a guard, reconciliation, scan, sort, metadata write,
result consumer, dependency, or TensorFlow behavior. Validate the indexed/
compatibility owner, channel-shuffle and layout-recovery boundaries, terminal
orchestration, architecture, and broad related gates sequentially, then commit
and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused indexed/compatibility owner, channel-shuffle/layout-recovery
  boundaries, terminal-orchestration, and architecture gate:
  `325 passed, 1 xfailed in 18.76s`
- expanded broad related gate: `1744 passed, 1 xfailed in 33.31s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented single-
result retention contract above.

## Late attention-QKV Reshape result retention implementation checkpoint

The sole production call now retains the existing one-counter dictionary as
`_late_attention_qkv_reshape_stats`. This is an assignment-only orchestration
change: the compatibility/indexed owner, return schema, fallback behavior,
GraphIndex and pruning behavior, live Session LayoutState argument, pass order,
guards, neighboring calls, dependencies, and TensorFlow behavior are unchanged.
The retained value has no consumer and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused indexed/compatibility owner, channel-shuffle/layout-recovery
  boundaries, terminal-orchestration, and architecture gate:
  `326 passed in 19.68s`
- branch-changed broad related suite plus indexed QKV, layout recovery, and
  pass-efficiency coverage: `1379 passed in 23.78s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_optimize_attention_gather_transpose_reshape_cleanup_chains()` result, owner
schema, production occurrences, and QKV/Gather-axis0 boundaries before adding
characterization. Commit and push only; do not create, reopen, or update a pull
request.

## Late attention Gather cleanup result characterization checkpoint

The attention Gather/Transpose/Reshape cleanup owner returns the stable
two-counter dictionary for pattern A and pattern B rewrites. It accepts only the
ModelIR. The owner is selected once by the existing attention-recovery runner
and has one additional direct late production call; only the latter currently
discards its result.

A strict expected-failure orchestration contract requires
`_late_attention_gather_cleanup_stats` for that direct call. It fixes the
captured QKV dictionary predecessor, exact model-only callback, and following
Gather-axis0 compatibility call with its live Session LayoutState.

At implementation, replace only the direct expression with an assignment. Do
not change the owner, two-key schema, recovery-runner selection or captured
results, GraphIndex/pruning behavior, pass order, callback arguments, neighbor
targets, guards, dependencies, or TensorFlow behavior. Validate the cleanup
owner, QKV/Gather-axis0 boundaries, terminal orchestration, architecture, and
broad related gates sequentially, then commit and push only; do not create,
reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused cleanup owner, QKV/Gather-axis0 boundaries, layout recovery,
  terminal-orchestration, and architecture gate:
  `436 passed, 1 xfailed in 18.94s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  layout recovery, and pass-efficiency coverage:
  `1513 passed, 1 xfailed in 23.86s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented direct
late-result retention contract above.

## Late attention Gather cleanup result retention implementation checkpoint

The sole direct late production call now retains the existing two-counter
dictionary as `_late_attention_gather_cleanup_stats`. This is an
assignment-only orchestration change. The owner and schema, separate recovery-
runner selection and captured results, GraphIndex/pruning behavior, callback
arguments, pass order, QKV predecessor, live-LayoutState Gather-axis0
successor, dependencies, and TensorFlow behavior are unchanged. The value has
no consumer and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused cleanup owner, QKV/Gather-axis0 boundaries, layout recovery,
  terminal-orchestration, and architecture gate: `437 passed in 19.04s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  layout recovery, and pass-efficiency coverage: `1514 passed in 24.34s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_optimize_gather_axis0_singleton_to_reshape_input_chains()` result, owner
schema, live LayoutState contract, production occurrences, and cleanup/
preprojection boundaries before adding characterization. Commit and push only;
do not create, reopen, or update a pull request.

## Late Gather-axis0 Reshape result characterization checkpoint

The Gather-axis0 singleton-to-Reshape compatibility owner returns the stable
one-counter dictionary
`optimized_gather_axis0_singleton_to_reshape_input_chains`. It uses the indexed
GraphIndex implementation, receives the live Session LayoutState, is selected
once by the existing attention-recovery runner, and has one additional direct
late production call. Only that direct call currently discards its result.

A strict expected-failure orchestration contract requires
`_late_gather_axis0_reshape_stats` for the direct call. It fixes the captured
attention-cleanup predecessor, exact ModelIR/LayoutState callback, and following
model-only attention-preprojection rank-lift owner.

At implementation, replace only the direct expression with an assignment. Do
not change the owner, one-key schema, recovery-runner selection or captured
results, GraphIndex/layout synchronization, pass order, callback arguments,
neighbor targets, guards, dependencies, or TensorFlow behavior. Validate the
Gather owner, cleanup/preprojection boundaries, layout recovery, terminal
orchestration, architecture, and broad related gates sequentially, then commit
and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused Gather owner, cleanup/preprojection boundaries, layout recovery,
  terminal-orchestration, and architecture gate:
  `495 passed, 1 xfailed in 18.96s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, layout recovery, and pass-efficiency coverage:
  `1580 passed, 1 xfailed in 24.26s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented direct
late-result retention contract above.

## Late Gather-axis0 Reshape result retention implementation checkpoint

The sole direct late production call now retains the existing one-counter
dictionary as `_late_gather_axis0_reshape_stats`. This is an assignment-only
orchestration change. The indexed owner and schema, separate recovery-runner
selection and captured results, GraphIndex/layout synchronization, callback
arguments, pass order, attention-cleanup predecessor, preprojection successor,
dependencies, and TensorFlow behavior are unchanged. The value has no consumer
and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused Gather owner, cleanup/preprojection boundaries, layout recovery,
  terminal-orchestration, and architecture gate: `496 passed in 18.93s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, layout recovery, and pass-efficiency coverage:
  `1581 passed in 24.33s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains()` result,
owner schema, production occurrences, and Gather/window-partition boundaries
before adding characterization. Commit and push only; do not create, reopen, or
update a pull request.

## Late attention preprojection result characterization checkpoint

The attention-preprojection Reshape-to-BatchMatMul rank-lift owner returns the
stable one-counter dictionary
`optimized_attention_preproj_reshape_to_batchmatmul_ranklift_chains`. It accepts
only the ModelIR, is selected once by the existing attention-recovery runner,
and has one additional direct late production call. Only that direct call
currently discards its result.

A strict expected-failure orchestration contract requires
`_late_attention_preproj_ranklift_stats` for the direct call. It fixes the
captured Gather-axis0 predecessor, exact model-only callback, and following
window-partition callback with its live Session LayoutState.

At implementation, replace only the direct expression with an assignment. Do
not change the owner, one-key schema, recovery-runner selection or captured
results, GraphIndex/pruning behavior, pass order, callback arguments, neighbor
targets, guards, dependencies, or TensorFlow behavior. Validate the
preprojection owner, Gather/window-partition boundaries, layout recovery,
terminal orchestration, architecture, and broad related gates sequentially,
then commit and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused preprojection owner, Gather/window-partition boundaries, layout
  recovery, terminal-orchestration, and architecture gate:
  `450 passed, 1 xfailed in 18.84s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, window partition, layout recovery, and pass-efficiency
  coverage: `1632 passed, 1 xfailed in 24.26s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented direct
late-result retention contract above.

## Late attention preprojection result retention implementation checkpoint

The sole direct late production call now retains the existing one-counter
dictionary as `_late_attention_preproj_ranklift_stats`. This is an assignment-
only orchestration change. The owner and schema, separate recovery-runner
selection and captured results, GraphIndex/pruning behavior, callback
arguments, pass order, Gather-axis0 predecessor, live-LayoutState window-
partition successor, dependencies, and TensorFlow behavior are unchanged. The
value has no consumer and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused preprojection owner, Gather/window-partition boundaries, layout
  recovery, terminal-orchestration, and architecture gate:
  `451 passed in 18.88s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, window partition, layout recovery, and pass-efficiency
  coverage: `1633 passed in 24.31s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_optimize_window_partition_reshape_transpose_to_space_to_depth_chains()`
result, owner schema, live LayoutState contract, production occurrences, and
preprojection/window-reverse boundaries before adding characterization. Commit
and push only; do not create, reopen, or update a pull request.

## Late window-partition result characterization checkpoint

The window-partition Reshape/Transpose-to-SpaceToDepth indexed owner returns the
stable one-counter dictionary
`optimized_window_partition_reshape_transpose_to_space_to_depth_chains`. It
receives the live Session LayoutState, is selected once by the existing
attention-recovery runner, and has one additional direct late production call.
Only that direct call currently discards its result.

A strict expected-failure orchestration contract requires
`_late_window_partition_stats` for the direct call. It fixes the captured
attention-preprojection predecessor, exact ModelIR/LayoutState callback, and
following window-reverse callback with the same live Session LayoutState.

At implementation, replace only the direct expression with an assignment. Do
not change the owner, one-key schema, recovery-runner selection or captured
results, GraphIndex/layout synchronization, pass order, callback arguments,
neighbor targets, guards, dependencies, or TensorFlow behavior. Validate the
window owner, preprojection/window-reverse boundaries, layout recovery,
terminal orchestration, architecture, and broad related gates sequentially,
then commit and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused window owner, preprojection/window-reverse boundaries, layout
  recovery, terminal-orchestration, and architecture gate:
  `414 passed, 1 xfailed in 19.47s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, window owners, layout recovery, and pass-efficiency coverage:
  `1633 passed, 1 xfailed in 24.87s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented direct
late-result retention contract above.

## Late window-partition result retention implementation checkpoint

The sole direct late production call now retains the existing one-counter
dictionary as `_late_window_partition_stats`. This is an assignment-only
orchestration change. The indexed owner and schema, separate recovery-runner
selection and captured results, GraphIndex/layout synchronization, callback
arguments, pass order, attention-preprojection predecessor, window-reverse
successor, dependencies, and TensorFlow behavior are unchanged. The value has
no consumer and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused window owner, preprojection/window-reverse boundaries, layout
  recovery, terminal-orchestration, and architecture gate:
  `415 passed in 18.79s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, window owners, layout recovery, and pass-efficiency coverage:
  `1634 passed in 25.07s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_optimize_window_reverse_reshape_transpose_to_depth_to_space_chains()` result,
owner schema, live LayoutState contract, production occurrences, and window-
partition/final-convergence boundaries before adding characterization. Commit
and push only; do not create, reopen, or update a pull request.

## Late window-reverse result characterization checkpoint

The window-reverse Reshape/Transpose-to-DepthToSpace indexed owner returns the
stable one-counter dictionary
`optimized_window_reverse_reshape_transpose_to_depth_to_space_chains`. It
receives the live Session LayoutState, is selected once by the existing
attention-recovery runner, and has one additional direct late production call.
Only that direct call currently discards its result.

A strict expected-failure orchestration contract requires
`_late_window_reverse_stats` for the direct call. It fixes the captured window-
partition predecessor, exact ModelIR/LayoutState callback, and following
indexed final shape/activation convergence call with the same live Session
LayoutState.

At implementation, replace only the direct expression with an assignment. Do
not change the owner, one-key schema, recovery-runner selection or captured
results, GraphIndex/layout synchronization, pass order, callback arguments,
neighbor targets, guards, dependencies, or TensorFlow behavior. Validate the
window owners, partition/final-convergence boundaries, layout recovery,
terminal orchestration, architecture, and broad related gates sequentially,
then commit and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused window owners, partition/final-convergence boundaries, layout
  recovery, terminal-orchestration, and architecture gate:
  `410 passed, 1 xfailed in 19.38s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, both window owners, final convergence, layout recovery, and
  pass-efficiency coverage: `1682 passed, 1 xfailed in 24.62s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented direct
late-result retention contract above.

## Late window-reverse result retention implementation checkpoint

The sole direct late production call now retains the existing one-counter
dictionary as `_late_window_reverse_stats`. This is an assignment-only
orchestration change. The indexed owner and schema, separate recovery-runner
selection and captured results, GraphIndex/layout synchronization, callback
arguments, pass order, window-partition predecessor, indexed final-convergence
successor, dependencies, and TensorFlow behavior are unchanged. The value has
no consumer and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused window owners, partition/final-convergence boundaries, layout
  recovery, terminal-orchestration, and architecture gate:
  `411 passed in 18.65s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, both window owners, final convergence, layout recovery, and
  pass-efficiency coverage: `1683 passed in 24.57s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_run_indexed_final_shape_activation_convergence()` result, return schema, live
LayoutState contract, production occurrences, and window-reverse/boundary-
normalization boundaries before adding characterization. Commit and push only;
do not create, reopen, or update a pull request.

## Late indexed final convergence result characterization checkpoint

`_run_indexed_final_shape_activation_convergence()` returns the existing
aggregate mutation dictionary from indexed shape convergence, HardSwish shape
sanitation, dynamic-Reshape resolution, static reconciliation, and activation
fusion. It has one production call and receives the live Session LayoutState.
Its result is currently discarded.

A strict expected-failure orchestration contract requires
`_late_final_shape_activation_convergence_stats`. It fixes the captured window-
reverse predecessor, exact ModelIR/LayoutState callback, and following final
boundary-input normalization call with the same LayoutState and Session
diagnostics.

At implementation, replace only the expression with an assignment. Do not
change the aggregate schema, one-index convergence internals, conditional
reconciliation, fusion behavior, pass order, callback arguments, neighboring
targets, guards, dependencies, diagnostics, or TensorFlow behavior. Validate
indexed final convergence, window-reverse/boundary-normalization boundaries,
terminal orchestration, architecture, and broad related gates sequentially,
then commit and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused indexed final convergence, window-reverse/boundary-normalization,
  terminal-orchestration, architecture, and pass-efficiency gate:
  `389 passed, 1 xfailed in 18.19s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, both window owners, final convergence, boundary normalization,
  layout recovery, and pass-efficiency coverage:
  `1687 passed, 1 xfailed in 24.58s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented aggregate-
result retention contract above.

## Late indexed final convergence result retention implementation checkpoint

The sole production call now retains the existing aggregate mutation
dictionary as `_late_final_shape_activation_convergence_stats`. This is an
assignment-only orchestration change. The aggregate schema, one-index
convergence internals, conditional reconciliation, fusion behavior, callback
arguments, pass order, window-reverse predecessor, final boundary-normalization
successor, diagnostics, dependencies, and TensorFlow behavior are unchanged.
The value has no consumer and triggers no additional graph work.

The architecture contract now locates the production assignment and verifies
its target while preserving its existing internal convergence assertions.

Implementation validation completed sequentially under `uv`:

- focused indexed final convergence, window-reverse/boundary-normalization,
  terminal-orchestration, architecture, and pass-efficiency gate:
  `390 passed in 20.13s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, both window owners, final convergence, boundary normalization,
  layout recovery, and pass-efficiency coverage: `1688 passed in 24.36s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following final
`run_boundary_input_normalization_cleanup()` result, distinguish it from the
earlier production occurrence, and fix its convergence/internal-channel-slice
boundaries before adding characterization. Commit and push only; do not create,
reopen, or update a pull request.

## Final boundary-input normalization result characterization checkpoint

`run_boundary_input_normalization_cleanup()` returns the stable one-counter
dictionary
`rewritten_boundary_input_transpose_mul_sum_reshape_nhwc_chains` and has two
production occurrences. Both receive the live Session LayoutState and
diagnostics. The final occurrence follows indexed final convergence and
currently discards its result; the earlier occurrence is outside this unit and
remains a raw call.

A strict expected-failure orchestration contract requires
`_final_boundary_input_normalization_stats` only for the final occurrence. It
fixes the captured final-convergence predecessor, exact ModelIR/LayoutState/
diagnostics callback, and following model-only internal Transpose/channel-slice
propagation owner. It also fixes the earlier occurrence as a distinct raw call.

At implementation, replace only the final expression with an assignment. Do
not change the earlier occurrence, owner, one-key schema, transaction or
preflight behavior, GraphIndex/layout synchronization, pass order, callback
arguments, neighbor targets, diagnostics, guards, dependencies, or TensorFlow
behavior. Validate the normalization owner, both occurrence contracts,
convergence/channel-slice boundaries, terminal orchestration, architecture, and
broad related gates sequentially, then commit and push only; do not create,
reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused normalization owner, both occurrence contracts, convergence/channel-
  slice boundaries, terminal-orchestration, architecture, and pass-efficiency
  gate: `342 passed, 1 xfailed in 18.57s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, both window owners, final convergence, boundary normalization,
  layout recovery, and pass-efficiency coverage:
  `1688 passed, 1 xfailed in 24.53s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented final-
occurrence result retention contract above.

## Final boundary-input normalization result retention implementation checkpoint

Only the final of the two production calls now retains the existing one-counter
dictionary as `_final_boundary_input_normalization_stats`. The earlier
occurrence remains a raw call. This is an assignment-only orchestration change:
the owner and schema, transaction/preflight behavior, GraphIndex/layout
synchronization, callback arguments, diagnostics, pass order, final-convergence
predecessor, internal channel-slice successor, dependencies, and TensorFlow
behavior are unchanged. The retained value has no consumer or extra graph work.

Implementation validation completed sequentially under `uv`:

- focused normalization owner, both occurrence contracts, convergence/channel-
  slice boundaries, terminal-orchestration, architecture, and pass-efficiency
  gate: `343 passed in 18.16s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, both window owners, final convergence, boundary normalization,
  layout recovery, and pass-efficiency coverage: `1689 passed in 24.15s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the earlier raw
`run_boundary_input_normalization_cleanup()` result and its terminal-softmax/
boundary-channel-slice boundaries before adding characterization. Keep the
final target fixed. Commit and push only; do not create, reopen, or update a
pull request.

## Terminal boundary-input normalization result characterization checkpoint

The earlier `run_boundary_input_normalization_cleanup()` production occurrence
returns the same stable one-counter dictionary as the already captured final
occurrence and receives the same live Session LayoutState and diagnostics. Its
result is still discarded.

A strict expected-failure orchestration contract requires
`_terminal_boundary_input_normalization_stats` for only the earlier occurrence.
It fixes the terminal Softmax/Transpose predecessor, exact ModelIR/LayoutState/
diagnostics callback, following boundary-input Transpose/channel-slice owner,
and the existing `_final_boundary_input_normalization_stats` target.

At implementation, replace only the earlier expression with an assignment. Do
not change the final occurrence, owner, one-key schema, transaction or
preflight behavior, GraphIndex/layout synchronization, pass order, callback
arguments, neighbor targets, diagnostics, guards, dependencies, or TensorFlow
behavior. Validate the normalization and adjacent indexed owners, both
occurrence contracts, terminal orchestration, architecture, and broad related
gates sequentially, then commit and push only; do not create, reopen, or update
a pull request.

Characterization validation completed sequentially under `uv`:

- focused normalization owner, terminal Softmax and boundary channel-slice
  neighbors, both occurrence contracts, terminal-orchestration, architecture,
  and pass-efficiency gate: `359 passed, 1 xfailed in 18.23s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, window/final convergence, boundary normalization, terminal
  Softmax, layout recovery, and pass-efficiency coverage:
  `1718 passed, 1 xfailed in 24.30s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented terminal-
occurrence result retention contract above.

## Terminal boundary-input normalization result retention implementation checkpoint

The earlier production call now retains the existing one-counter dictionary as
`_terminal_boundary_input_normalization_stats`; the already captured final call
keeps `_final_boundary_input_normalization_stats`. This is an assignment-only
orchestration change. The owner and schema, transaction/preflight behavior,
GraphIndex/layout synchronization, callback arguments, diagnostics, pass order,
terminal Softmax predecessor, boundary channel-slice successor, final target,
dependencies, and TensorFlow behavior are unchanged. Neither value has a
consumer or triggers additional graph work.

Implementation validation completed sequentially under `uv`:

- focused normalization owner, terminal Softmax and boundary channel-slice
  neighbors, both occurrence contracts, terminal-orchestration, architecture,
  and pass-efficiency gate: `360 passed in 18.72s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, window/final convergence, boundary normalization, terminal
  Softmax, layout recovery, and pass-efficiency coverage:
  `1719 passed in 24.61s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately preceding
`_optimize_terminal_softmax_transpose_after_nhwc_propagation()` result, owner
schema, live LayoutState contract, production occurrences, and Gather-fanout/
boundary-normalization boundaries before adding characterization. Commit and
push only; do not create, reopen, or update a pull request.

## Terminal Softmax/Transpose result characterization checkpoint

The terminal Softmax/Transpose-after-NHWC-propagation indexed owner returns the
stable one-counter dictionary
`removed_terminal_softmax_transpose_after_nhwc_propagation`. It receives the
live Session LayoutState and has one production occurrence. Its result is
currently discarded.

A strict expected-failure orchestration contract requires
`_terminal_softmax_transpose_stats`. It fixes the exact ModelIR/LayoutState
callback between the diagnostics-aware Gather-channel-fanout runner and the
captured `_terminal_boundary_input_normalization_stats` successor.

At implementation, replace only the expression with an assignment. Do not
change the owner, one-key schema, GraphIndex/layout synchronization, pass order,
callback arguments, neighbor targets, diagnostics, guards, dependencies, or
TensorFlow behavior. Validate the indexed Softmax owner, Gather-fanout/
normalization boundaries, terminal orchestration, architecture, and broad
related gates sequentially, then commit and push only; do not create, reopen,
or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused indexed Softmax, Gather-fanout, boundary normalization, terminal-
  orchestration, architecture, and pass-efficiency gate:
  `364 passed, 1 xfailed in 18.89s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, window/final convergence, boundary normalization, terminal
  Softmax, Gather fanout, layout recovery, and pass-efficiency coverage:
  `1723 passed, 1 xfailed in 24.50s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented terminal
Softmax result retention contract above.

## Terminal Softmax/Transpose result retention implementation checkpoint

The sole production call now retains the existing one-counter dictionary as
`_terminal_softmax_transpose_stats`. This is an assignment-only orchestration
change. The indexed owner and schema, GraphIndex/layout synchronization,
callback arguments, pass order, diagnostics-aware Gather-channel-fanout
predecessor, captured terminal boundary-normalization successor, dependencies,
and TensorFlow behavior are unchanged. The value has no consumer and triggers
no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused indexed Softmax, Gather-fanout, boundary normalization, terminal-
  orchestration, architecture, and pass-efficiency gate:
  `365 passed in 18.32s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, window/final convergence, boundary normalization, terminal
  Softmax, Gather fanout, layout recovery, and pass-efficiency coverage:
  `1724 passed in 25.28s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately preceding direct
`run_transpose_gather_channel_fanout_cleanup()` result, distinguish it from the
two orchestrated selections, and fix its ArgMax/Softmax boundaries before
adding characterization. Commit and push only; do not create, reopen, or update
a pull request.

## Direct terminal Gather-channel-fanout result characterization checkpoint

`run_transpose_gather_channel_fanout_cleanup()` returns the stable one-counter
dictionary `optimized_transpose_gather_transpose_nhwc_channel_chains`. The same
callback is selected by two existing orchestrators and has one direct terminal
production call. Only that direct result is currently discarded.

A strict expected-failure orchestration contract requires
`_terminal_transpose_gather_channel_fanout_stats` for the direct call. It fixes
the diagnostics-aware ModelIR/LayoutState callback between the terminal ArgMax
owner and captured `_terminal_softmax_transpose_stats` successor.

At implementation, replace only the direct expression with an assignment. Do
not change either orchestrated selection, the runner, one-key schema,
transaction or preflight behavior, shared state, GraphIndex/layout
synchronization, pass order, callback arguments, neighbor targets, diagnostics,
guards, dependencies, or TensorFlow behavior. Validate the fanout and adjacent
indexed owners, direct/orchestrated call accounting, terminal orchestration,
architecture, and broad related gates sequentially, then commit and push only;
do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused ArgMax, direct Gather-channel-fanout, terminal Softmax, direct/
  orchestrated accounting, terminal-orchestration, architecture, and pass-
  efficiency gate: `399 passed, 1 xfailed in 19.18s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, window/final convergence, boundary normalization, terminal
  ArgMax/Softmax, Gather fanout, layout recovery, and pass-efficiency coverage:
  `1762 passed, 1 xfailed in 24.63s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented direct
fanout result retention contract above.

## Direct terminal Gather-channel-fanout result retention implementation checkpoint

The direct production call now retains the existing one-counter dictionary as
`_terminal_transpose_gather_channel_fanout_stats`. This is an assignment-only
orchestration change. Both orchestrated selections, the runner and schema,
transaction/preflight behavior, shared state, GraphIndex/layout synchronization,
callback arguments, diagnostics, pass order, ArgMax predecessor, captured
Softmax successor, dependencies, and TensorFlow behavior are unchanged. The
value has no consumer or additional graph work.

Implementation validation completed sequentially under `uv`:

- focused ArgMax, direct Gather-channel-fanout, terminal Softmax, direct/
  orchestrated accounting, terminal-orchestration, architecture, and pass-
  efficiency gate: `400 passed in 18.38s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, window/final convergence, boundary normalization, terminal
  ArgMax/Softmax, Gather fanout, layout recovery, and pass-efficiency coverage:
  `1763 passed in 24.86s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately preceding
`_optimize_transpose_pre_argmax_nhwc_terminal_chains()` result, owner schema,
live LayoutState contract, production occurrences, and Conv-activation/Gather-
fanout boundaries before adding characterization. Commit and push only; do not
create, reopen, or update a pull request.

## Terminal pre-ArgMax result characterization checkpoint

`_optimize_transpose_pre_argmax_nhwc_terminal_chains()` returns the stable
one-counter dictionary
`optimized_transpose_pre_argmax_nhwc_terminal_chains`. Its wrapper has one
production call, which currently discards that dictionary while passing the
live `session.layout_state`.

A strict expected-failure orchestration contract requires that direct result to
be retained as `_terminal_pre_argmax_stats`. It also fixes the captured
`_terminal_cleanup_conv_activation_stats` predecessor and captured
`_terminal_transpose_gather_channel_fanout_stats` successor.

At implementation, replace only the direct expression with an assignment. Do
not change the wrapper or pass implementation, one-key result schema,
transaction/preflight guards, graph mutation, tensor pruning, shared state,
GraphIndex/layout synchronization, callback arguments, pass order, adjacent
targets, dependencies, or TensorFlow behavior. The retained value must have no
consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- focused Conv-activation, terminal ArgMax, Gather-channel-fanout, terminal
  Softmax, terminal orchestration, architecture, and pass-efficiency gate:
  `416 passed, 1 xfailed in 18.72s`
- branch-changed broad related suite plus activation fusion, attention/Gather,
  preprojection, window/final convergence, boundary normalization, terminal
  ArgMax/Softmax, Gather fanout, layout recovery, and pass-efficiency coverage:
  `1763 passed, 1 xfailed in 24.64s`

The sole strict expected failure is the intentionally unimplemented terminal
pre-ArgMax result retention contract above. Implement that assignment, rerun
the same gates sequentially, then commit and push only; do not create, reopen,
or update a pull request.

## Terminal pre-ArgMax result retention implementation checkpoint

The sole production call now retains its existing one-counter dictionary as
`_terminal_pre_argmax_stats`. This is an assignment-only orchestration change.
The wrapper and pass implementation, one-key result schema, transaction and
preflight guards, graph mutation, tensor pruning, shared state, GraphIndex/
layout synchronization, callback arguments, pass order, captured terminal
Conv-activation predecessor, captured Gather-channel-fanout successor,
dependencies, and TensorFlow behavior are unchanged. The value has no consumer
and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused Conv-activation, terminal ArgMax, Gather-channel-fanout, terminal
  Softmax, terminal orchestration, architecture, and pass-efficiency gate:
  `417 passed in 19.64s`
- branch-changed broad related suite plus activation fusion, attention/Gather,
  preprojection, window/final convergence, boundary normalization, terminal
  ArgMax/Softmax, Gather fanout, layout recovery, and pass-efficiency coverage:
  `1764 passed in 25.32s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_optimize_boundary_input_transpose_channel_slice_blocks()` result, owner
schema, live LayoutState contract, production occurrences, and captured
normalization/internal-channel-slice boundaries before adding characterization.
Commit and push only; do not create, reopen, or update a pull request.

## Terminal boundary-input channel-slice result characterization checkpoint

`_optimize_boundary_input_transpose_channel_slice_blocks()` returns the stable
four-counter dictionary comprising removed boundary input Transposes, rewritten
boundary channel Slices, rewritten boundary axis operations, and inserted local
boundary Transposes. Its wrapper has one production call, which currently
discards the dictionary while passing the live `session.layout_state`.

The zero-mutation schema test now fixes the exact four-key result, while the
existing indexed owner coverage validates the supplied GraphIndex and
LayoutState after a rewrite. A strict expected-failure orchestration contract
requires the production result to be retained as
`_terminal_boundary_input_channel_slice_stats`. It also fixes the captured
`_terminal_boundary_input_normalization_stats` predecessor and the first
internal Transpose/channel-slice propagation successor with the same live
LayoutState.

At implementation, replace only the direct expression with an assignment. Do
not change the wrapper or pass implementation, four-key result schema, rewrite
guards, graph mutation, tensor pruning, shared GraphIndex/LayoutState behavior,
callback arguments, pass order, adjacent targets, dependencies, diagnostics,
or TensorFlow behavior. The retained value must have no consumer or additional
graph work.

Characterization validation completed sequentially under `uv`:

- focused boundary-input channel-slice owner, internal channel-slice successor,
  boundary normalization/layout, terminal orchestration, architecture, and
  pass-efficiency gate: `338 passed, 1 xfailed in 20.53s`
- branch-changed broad suite plus boundary-input, channel-slice/pad-Mul,
  terminal orchestration, architecture, and pass-efficiency coverage:
  `1382 passed, 1 xfailed in 24.29s`

The sole strict expected failure is the intentionally unimplemented terminal
boundary-input channel-slice result retention contract above. Implement that
assignment, rerun the same gates sequentially, then commit and push only; do
not create, reopen, or update a pull request.

## Terminal boundary-input channel-slice result retention implementation checkpoint

The sole production call now retains its existing four-counter dictionary as
`_terminal_boundary_input_channel_slice_stats`. This is an assignment-only
orchestration change. The wrapper and pass implementation, result schema,
rewrite guards, graph mutation, tensor pruning, shared GraphIndex/LayoutState
behavior, callback arguments, pass order, captured boundary-normalization
predecessor, first internal channel-slice successor, dependencies, diagnostics,
and TensorFlow behavior are unchanged. The value has no consumer and triggers
no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused boundary-input channel-slice owner, internal channel-slice successor,
  boundary normalization/layout, terminal orchestration, architecture, and
  pass-efficiency gate: `339 passed in 19.11s`
- branch-changed broad suite plus boundary-input, channel-slice/pad-Mul,
  terminal orchestration, architecture, and pass-efficiency coverage:
  `1383 passed in 24.43s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following first
`_optimize_internal_transpose_channel_slice_nhwc_propagation_chains()` result,
distinguish it from the later raw production occurrence, and fix its captured
boundary-input channel-slice/MulAdd-bridge boundaries before adding
characterization. Commit and push only; do not create, reopen, or update a pull
request.

## First terminal internal channel-slice result characterization checkpoint

`_optimize_internal_transpose_channel_slice_nhwc_propagation_chains()` returns
the stable four-counter dictionary comprising removed internal Transpose/
channel-slice stems, rewritten internal channel Slices, rewritten internal axis
operations, and inserted internal local Transposes. It has two production
calls. The first passes the live `session.layout_state` and currently discards
its result; the later model-only call is also raw.

The zero-mutation schema test fixes the exact four-key result, while existing
indexed owner coverage validates a supplied GraphIndex and LayoutState after a
rewrite. A strict expected-failure orchestration contract requires only the
first result to be retained as `_terminal_internal_channel_slice_stats`. It
fixes the captured `_terminal_boundary_input_channel_slice_stats` predecessor,
the first live-LayoutState MulAdd-bridge successor, and the later raw occurrence
between captured final boundary normalization and its model-only MulAdd bridge.

At implementation, replace only the first expression with an assignment. Do
not change the later occurrence, wrapper or pass implementation, four-key
schema, rewrite guards, graph mutation, tensor pruning, shared GraphIndex/
LayoutState behavior, callback arguments, pass order, adjacent targets,
dependencies, diagnostics, or TensorFlow behavior. The retained value must
have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- focused boundary/internal channel-slice owners, both occurrence boundaries,
  boundary normalization/layout, terminal orchestration, architecture, and
  pass-efficiency gate: `340 passed, 1 xfailed in 19.02s`
- branch-changed broad suite plus boundary-input, channel-slice/pad-Mul,
  terminal orchestration, architecture, and pass-efficiency coverage:
  `1384 passed, 1 xfailed in 24.25s`

The sole strict expected failure is the intentionally unimplemented first
internal channel-slice result retention contract above. Implement that
assignment, rerun the same gates sequentially, then commit and push only; do
not create, reopen, or update a pull request.

## First terminal internal channel-slice result retention implementation checkpoint

Only the first of the two production calls now retains its existing four-
counter dictionary as `_terminal_internal_channel_slice_stats`. The later
model-only occurrence remains raw. This is an assignment-only orchestration
change. The wrapper and pass implementation, result schema, rewrite guards,
graph mutation, tensor pruning, shared GraphIndex/LayoutState behavior,
callback arguments, pass order, captured boundary-input channel-slice
predecessor, first live-LayoutState MulAdd-bridge successor, later occurrence
and boundaries, dependencies, diagnostics, and TensorFlow behavior are
unchanged. The value has no consumer and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused boundary/internal channel-slice owners, both occurrence boundaries,
  boundary normalization/layout, terminal orchestration, architecture, and
  pass-efficiency gate: `341 passed in 18.70s`
- branch-changed broad suite plus boundary-input, channel-slice/pad-Mul,
  terminal orchestration, architecture, and pass-efficiency coverage:
  `1385 passed in 24.19s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the later raw
`_optimize_internal_transpose_channel_slice_nhwc_propagation_chains()` result,
keep `_terminal_internal_channel_slice_stats` fixed for the first occurrence,
and preserve its captured final boundary-normalization/model-only MulAdd-bridge
boundaries before adding characterization. Commit and push only; do not create,
reopen, or update a pull request.

## Final internal channel-slice result characterization checkpoint

The later of the two
`_optimize_internal_transpose_channel_slice_nhwc_propagation_chains()`
production calls returns the same stable four-counter dictionary as the already
captured first occurrence. It is model-only and currently discards its result.

A strict expected-failure orchestration contract requires only that later
result to be retained as `_final_internal_channel_slice_stats`. It keeps the
first `_terminal_internal_channel_slice_stats` target fixed and preserves the
captured `_final_boundary_input_normalization_stats` predecessor plus the later
model-only Transpose/channel-slice MulAdd-bridge successor.

At implementation, replace only the later expression with an assignment. Do
not change the first occurrence or target, wrapper or pass implementation,
four-key schema, rewrite guards, graph mutation, tensor pruning, GraphIndex/
LayoutState behavior, callback arguments, pass order, adjacent targets,
dependencies, diagnostics, or TensorFlow behavior. The retained value must
have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- focused boundary/internal channel-slice owners, both occurrence boundaries,
  boundary normalization/layout, terminal orchestration, architecture, and
  pass-efficiency gate: `341 passed, 1 xfailed in 19.24s`
- branch-changed broad suite plus boundary-input, channel-slice/pad-Mul,
  terminal orchestration, architecture, and pass-efficiency coverage:
  `1385 passed, 1 xfailed in 23.71s`

The sole strict expected failure is the intentionally unimplemented final
internal channel-slice result retention contract above. Implement that
assignment, rerun the same gates sequentially, then commit and push only; do
not create, reopen, or update a pull request.

## Final internal channel-slice result retention implementation checkpoint

The later model-only production call now retains its existing four-counter
dictionary as `_final_internal_channel_slice_stats`. The first live-LayoutState
occurrence keeps `_terminal_internal_channel_slice_stats`. This is an
assignment-only orchestration change. The wrapper and pass implementation,
result schema, rewrite guards, graph mutation, tensor pruning, GraphIndex/
LayoutState behavior, callback arguments, pass order, captured final boundary-
normalization predecessor, later model-only MulAdd-bridge successor, first
occurrence and boundaries, dependencies, diagnostics, and TensorFlow behavior
are unchanged. Neither retained value has a consumer or triggers additional
graph work.

Implementation validation completed sequentially under `uv`:

- focused boundary/internal channel-slice owners, both occurrence boundaries,
  boundary normalization/layout, terminal orchestration, architecture, and
  pass-efficiency gate: `342 passed in 18.84s`
- branch-changed broad suite plus boundary-input, channel-slice/pad-Mul,
  terminal orchestration, architecture, and pass-efficiency coverage:
  `1386 passed in 23.44s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the later model-only
`_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains()` result,
distinguish it from the earlier live-LayoutState occurrence, and preserve its
captured final internal-channel-slice/terminal recovery boundaries before
adding characterization. Commit and push only; do not create, reopen, or
update a pull request.

## Final channel-slice MulAdd-bridge result characterization checkpoint

`_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains()` returns the
stable one-counter dictionary
`optimized_transpose_channel_slice_muladd_nhwc_bridge_chains` and has two
production calls. The first receives the live `session.layout_state` and is
raw; the later model-only call also currently discards its result.

The zero-mutation schema test fixes the exact one-key result, while existing
owner coverage validates a supplied GraphIndex and LayoutState after a rewrite.
A strict expected-failure orchestration contract requires only the later result
to be retained as `_final_channel_slice_muladd_bridge_stats`. It keeps the first
live-LayoutState occurrence raw and fixes both terminal recovery successors,
the captured `_terminal_internal_channel_slice_stats` first predecessor, and
the captured `_final_internal_channel_slice_stats` later predecessor.

At implementation, replace only the later expression with an assignment. Do
not change the first occurrence, wrapper or pass implementation, one-key
schema, rewrite guards, graph mutation, tensor pruning, GraphIndex/LayoutState
behavior, callback arguments, pass order, recovery calls, adjacent targets,
dependencies, diagnostics, or TensorFlow behavior. The retained value must
have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- focused MulAdd-bridge owner, both terminal recovery boundaries, schema,
  terminal orchestration, architecture, and pass-efficiency gate:
  `345 passed, 1 xfailed in 19.04s`
- branch-changed broad suite plus boundary/channel-slice, pad-Mul, both terminal
  recovery sequences, architecture, and pass-efficiency coverage:
  `1390 passed, 1 xfailed in 24.70s`

The sole strict expected failure is the intentionally unimplemented final
channel-slice MulAdd-bridge result retention contract above. Implement that
assignment, rerun the same gates sequentially, then commit and push only; do
not create, reopen, or update a pull request.

## Final channel-slice MulAdd-bridge result retention implementation checkpoint

Only the later model-only production call now retains its existing one-counter
dictionary as `_final_channel_slice_muladd_bridge_stats`. The first live-
LayoutState occurrence remains raw. This is an assignment-only orchestration
change. The wrapper and pass implementation, result schema, rewrite guards,
graph mutation, tensor pruning, GraphIndex/LayoutState behavior, callback
arguments, pass order, captured final internal-channel-slice predecessor, both
terminal recovery calls, first occurrence and boundaries, dependencies,
diagnostics, and TensorFlow behavior are unchanged. The retained value has no
consumer and triggers no additional graph work.

Recovery orchestration and architecture contracts now distinguish the first
raw expression from the later `_final_channel_slice_muladd_bridge_stats`
assignment while preserving both zero-argument recovery invocations and their
successors.

Implementation validation completed sequentially under `uv`:

- focused MulAdd-bridge owner, both terminal recovery boundaries, schema,
  terminal orchestration, architecture, and pass-efficiency gate:
  `346 passed in 20.08s`
- branch-changed broad suite plus boundary/channel-slice, pad-Mul, both terminal
  recovery sequences, architecture, and pass-efficiency coverage:
  `1391 passed in 24.78s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the first raw live-LayoutState
`_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains()` result, keep
`_final_channel_slice_muladd_bridge_stats` fixed for the later occurrence, and
preserve its captured terminal internal-channel-slice/recovery boundaries
before adding characterization. Commit and push only; do not create, reopen,
or update a pull request.

## Terminal channel-slice MulAdd-bridge result characterization checkpoint

The first of the two
`_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains()` production
calls returns the same stable one-counter dictionary as the captured later
occurrence. It receives the live `session.layout_state` and currently discards
its result.

A strict expected-failure orchestration contract requires only the first
result to be retained as `_terminal_channel_slice_muladd_bridge_stats`. It keeps
the later `_final_channel_slice_muladd_bridge_stats` target fixed and preserves
the captured `_terminal_internal_channel_slice_stats` and
`_final_internal_channel_slice_stats` predecessors plus both zero-argument
terminal recovery successors.

At implementation, replace only the first expression with an assignment. Do
not change the later occurrence or target, wrapper or pass implementation,
one-key schema, rewrite guards, graph mutation, tensor pruning, GraphIndex/
LayoutState behavior, callback arguments, pass order, recovery calls, adjacent
targets, dependencies, diagnostics, or TensorFlow behavior. The retained value
must have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- focused MulAdd-bridge owner, both terminal recovery boundaries, schema,
  terminal orchestration, architecture, and pass-efficiency gate:
  `346 passed, 1 xfailed in 19.05s`
- branch-changed broad suite plus boundary/channel-slice, pad-Mul, both terminal
  recovery sequences, architecture, and pass-efficiency coverage:
  `1391 passed, 1 xfailed in 24.18s`

The sole strict expected failure is the intentionally unimplemented terminal
channel-slice MulAdd-bridge result retention contract above. Implement that
assignment, rerun the same gates sequentially, then commit and push only; do
not create, reopen, or update a pull request.

## Terminal channel-slice MulAdd-bridge result retention implementation checkpoint

The first live-LayoutState production call now retains its existing one-counter
dictionary as `_terminal_channel_slice_muladd_bridge_stats`. The later model-
only occurrence keeps `_final_channel_slice_muladd_bridge_stats`. This is an
assignment-only orchestration change. The wrapper and pass implementation,
result schema, rewrite guards, graph mutation, tensor pruning, GraphIndex/
LayoutState behavior, callback arguments, pass order, captured terminal and
final internal-channel-slice predecessors, both terminal recovery calls,
dependencies, diagnostics, and TensorFlow behavior are unchanged. Neither
retained value has a consumer or triggers additional graph work.

Terminal orchestration, recovery orchestration, and architecture contracts now
require the two distinct targets while preserving both zero-argument recovery
invocations and their successors.

Implementation validation completed sequentially under `uv`:

- focused MulAdd-bridge owner, both terminal recovery boundaries, schema,
  terminal orchestration, architecture, and pass-efficiency gate:
  `347 passed in 20.85s`
- branch-changed broad suite plus boundary/channel-slice, pad-Mul, both terminal
  recovery sequences, architecture, and pass-efficiency coverage:
  `1392 passed in 24.26s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the sole
`_optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks()` result,
owner schema, live LayoutState contract, production occurrence, preceding
terminal recovery, and following Swish-residual-closure boundary before adding
characterization. Commit and push only; do not create, reopen, or update a pull
request.

## Terminal boundary StridedSlice/QDQ/Concat result characterization checkpoint

`_optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks()` returns
the stable four-counter dictionary comprising removed boundary input
Transpose/StridedSlice blocks, rewritten boundary StridedSlices, rewritten QDQ
Concat axes, and removed boundary post-Transposes. Its wrapper has one
production call, which currently discards the dictionary while passing the
live `session.layout_state`.

The zero-mutation schema test fixes the exact four-key result, while existing
owner coverage validates the nonzero counters plus supplied GraphIndex and
LayoutState after a rewrite. A strict expected-failure orchestration contract
requires the production result to be retained as
`_terminal_boundary_stridedslice_qdq_concat_stats`. It fixes the preceding
zero-argument terminal recovery call and the following model-only Swish-
residual-closure owner.

At implementation, replace only the direct expression with an assignment and
update the existing recovery outer-boundary contract to recognize that target.
Do not change the wrapper or pass implementation, four-key schema, rewrite
guards, graph mutation, tensor pruning, GraphIndex/LayoutState behavior,
callback arguments, pass order, recovery call, adjacent targets, dependencies,
diagnostics, or TensorFlow behavior. The retained value must have no consumer
or additional graph work.

Characterization validation completed sequentially under `uv`:

- focused owner/schema, terminal recovery, Swish boundary orchestration,
  architecture, and pass-efficiency gate:
  `348 passed, 1 xfailed in 18.48s`
- branch-changed broad suite plus boundary/channel-slice, pad-Mul, terminal
  recovery, architecture, and pass-efficiency coverage:
  `1393 passed, 1 xfailed in 23.94s`

The sole strict expected failure in those selected gates is the intentionally
unimplemented result-retention contract above. The separate indexed quantized-
Swish test module currently has two pre-existing failures because it
monkeypatches `_build_tensor_consumer_map`, which is already absent from the
committed lowerer at `HEAD`; those stale-test failures are not caused by this
characterization and were excluded from the gate.

Implement the assignment, rerun the same gates sequentially, then commit and
push only; do not create, reopen, or update a pull request.

## Terminal boundary StridedSlice/QDQ/Concat result retention implementation checkpoint

The sole production call now retains its existing four-counter dictionary as
`_terminal_boundary_stridedslice_qdq_concat_stats`. This is an assignment-only
orchestration change. The wrapper and pass implementation, result schema,
rewrite guards, graph mutation, tensor pruning, GraphIndex/LayoutState
behavior, callback arguments, pass order, preceding recovery call, following
Swish-residual closure, dependencies, diagnostics, and TensorFlow behavior are
unchanged. The retained value has no consumer and triggers no additional graph
work.

Recovery orchestration and architecture contracts now recognize the assignment
as the first recovery invocation's successor, while the later recovery
successor remains raw.

Implementation validation completed sequentially under `uv`:

- focused owner/schema, terminal recovery, Swish boundary orchestration,
  architecture, and pass-efficiency gate: `349 passed in 20.54s`
- branch-changed broad suite plus boundary/channel-slice, pad-Mul, terminal
  recovery, architecture, and pass-efficiency coverage:
  `1394 passed in 24.34s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run. The two pre-existing stale monkeypatch
failures in `test_flatbuffer_direct_indexed_quantized_swish_layout.py` remain a
separate known issue and were not changed in this unit.

At resume, audit the immediately following model-only
`_optimize_transpose_swish_residual_concat_closure_nhwc_chains()` result,
owner schema, production occurrence, captured boundary-StridedSlice
predecessor, and following dequant-logistic bridge. Keep the stale indexed-
Swish test issue separately classified. Commit and push only; do not create,
reopen, or update a pull request.

## Indexed quantized-Swish stale-test repair checkpoint

The two indexed quantized-Swish tests no longer monkeypatch
`_build_tensor_consumer_map` or `_build_tensor_producer_map` on the lowerer,
where those compatibility helpers no longer exist. They now assert that the
actual `quantized_swish_layout` owner module does not expose either legacy map
helper. Existing runtime assertions still verify the supplied GraphIndex,
single refresh behavior, current producer/consumer indices, rewrite counters,
metadata propagation, and fixed-point results.

This is a test-only repair. Production code, pass behavior, dependencies, and
TensorFlow boundaries are unchanged.

Validation completed sequentially under `uv`:

- complete indexed quantized-Swish owner module: `21 passed in 0.53s`
- indexed quantized-Swish plus terminal recovery/orchestration, architecture,
  and pass-efficiency coverage: `364 passed in 18.52s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The previously recorded stale-test issue is resolved. At resume, audit the
Swish-residual closure result. Commit and push only; do not create, reopen, or
update a pull request.

## Terminal Swish-residual closure result characterization checkpoint

`_optimize_transpose_swish_residual_concat_closure_nhwc_chains()` returns the
stable four-counter dictionary comprising rewritten closure branches, removed
pre-Transposes, rewritten Concat axes, and removed post-Transposes. Its wrapper
fixes `min_spatial_stage=0` and `require_concat_closure=True` and has one model-
only production call whose result is currently discarded.

Existing owner tests fix the exact four-key result, wrapper equivalence, fixed
options, index behavior, metadata propagation, and graph result. A strict
expected-failure orchestration contract requires the direct result to be
retained as `_terminal_swish_residual_concat_closure_stats`. It fixes the
captured `_terminal_boundary_stridedslice_qdq_concat_stats` predecessor and the
following model-only dequant-logistic-Mul-quantize bridge.

At implementation, replace only the direct expression with an assignment. Do
not change the wrapper or owner implementation, fixed options, four-key result
schema, indexed phase order, graph mutation, tensor pruning, callback arguments,
pass order, adjacent targets, dependencies, diagnostics, or TensorFlow
behavior. The retained value must have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- complete indexed quantized-Swish owner plus terminal recovery/orchestration,
  architecture, and pass-efficiency gate:
  `364 passed, 1 xfailed in 18.62s`
- branch-changed broad suite plus the complete indexed quantized-Swish owner,
  terminal recovery/orchestration, architecture, and pass-efficiency coverage:
  `1415 passed, 1 xfailed in 24.39s`

The sole strict expected failure is the intentionally unimplemented terminal
Swish-residual closure result retention contract above. Implement that
assignment, rerun the same gates sequentially, then commit and push only; do
not create, reopen, or update a pull request.

## Terminal Swish-residual closure result retention implementation checkpoint

The sole model-only production call now retains its existing four-counter
dictionary as `_terminal_swish_residual_concat_closure_stats`. This is an
assignment-only orchestration change. The wrapper and owner implementation,
fixed options, result schema, indexed phase order, graph mutation, tensor
pruning, callback arguments, pass order, captured boundary-StridedSlice
predecessor, dequant-logistic bridge successor, dependencies, diagnostics, and
TensorFlow behavior are unchanged. The retained value has no consumer and
triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- complete indexed quantized-Swish owner plus terminal recovery/orchestration,
  architecture, and pass-efficiency gate: `365 passed in 19.09s`
- branch-changed broad suite plus the complete indexed quantized-Swish owner,
  terminal recovery/orchestration, architecture, and pass-efficiency coverage:
  `1416 passed in 24.00s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the sole
`_optimize_transpose_dequant_logistic_mul_quantize_bridges()` result, owner
schema, production occurrence, captured Swish-closure predecessor, and
following Swish-QDQ-island boundary before adding characterization. Commit and
push only; do not create, reopen, or update a pull request.

## Terminal dequant-logistic bridge result characterization checkpoint

`_optimize_transpose_dequant_logistic_mul_quantize_bridges()` returns the
stable one-counter dictionary
`removed_transpose_dequant_logistic_mul_quantize_bridges`. Its indexed owner
accepts an optional GraphIndex, and its wrapper has one model-only production
call whose result is currently discarded.

Existing owner tests fix the exact one-key result, one-refresh behavior,
supplied-index currentness, protected/no-op graphs, and graph equivalence. A
strict expected-failure orchestration contract requires the direct result to be
retained as `_terminal_dequant_logistic_mul_quantize_bridge_stats`. It fixes
the captured `_terminal_swish_residual_concat_closure_stats` predecessor and
the following model-only Swish-QDQ-island owner.

At implementation, replace only the direct expression with an assignment. Do
not change the wrapper or indexed owner, optional GraphIndex contract, one-key
schema, rewrite guards, graph mutation, tensor pruning, callback arguments,
pass order, adjacent targets, dependencies, diagnostics, or TensorFlow
behavior. The retained value must have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- indexed quantized-logistic and Swish owners plus terminal recovery/
  orchestration, architecture, and pass-efficiency gate:
  `381 passed, 1 xfailed in 18.65s`
- branch-changed broad suite plus both indexed owners, terminal recovery/
  orchestration, architecture, and pass-efficiency coverage:
  `1432 passed, 1 xfailed in 24.24s`

The sole strict expected failure is the intentionally unimplemented terminal
dequant-logistic bridge result retention contract above. Implement that
assignment, rerun the same gates sequentially, then commit and push only; do
not create, reopen, or update a pull request.

## Terminal dequant-logistic bridge result retention implementation checkpoint

The sole model-only production call now retains its existing one-counter
dictionary as `_terminal_dequant_logistic_mul_quantize_bridge_stats`. This is an
assignment-only orchestration change. The wrapper and indexed owner, optional
GraphIndex contract, result schema, rewrite guards, graph mutation, tensor
pruning, callback arguments, pass order, captured Swish-closure predecessor,
Swish-QDQ-island successor, dependencies, diagnostics, and TensorFlow behavior
are unchanged. The retained value has no consumer and triggers no additional
graph work.

Implementation validation completed sequentially under `uv`:

- indexed quantized-logistic and Swish owners plus terminal recovery/
  orchestration, architecture, and pass-efficiency gate:
  `382 passed in 19.29s`
- branch-changed broad suite plus both indexed owners, terminal recovery/
  orchestration, architecture, and pass-efficiency coverage:
  `1433 passed in 24.66s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the sole `_optimize_transpose_swish_qdq_nhwc_islands()`
result, owner schema, fixed default options, production occurrence, captured
dequant-logistic predecessor, and following InstanceNorm-bias boundary before
adding characterization. Commit and push only; do not create, reopen, or update
a pull request.

## Terminal Swish-QDQ-island result characterization checkpoint

`_optimize_transpose_swish_qdq_nhwc_islands()` returns the stable five-counter
dictionary comprising rewritten Swish branches, removed pre-Transposes,
propagated NHWC metadata, rewritten Concat axes, and removed post-Transposes.
Its wrapper exposes optional spatial-stage/closure controls and has one model-
only production call using the defaults `min_spatial_stage=160` and
`require_concat_closure=False`; that result is currently discarded.

Existing owner tests fix the exact five-key result, phase order, option
forwarding, wrapper equivalence, indexed mutations, metadata propagation, and
graph result. A strict expected-failure orchestration contract requires the
direct result to be retained as `_terminal_swish_qdq_island_stats`. It fixes
the captured `_terminal_dequant_logistic_mul_quantize_bridge_stats` predecessor
and the following live-LayoutState InstanceNorm post-Transpose bias owner.

At implementation, replace only the direct expression with an assignment. Do
not change the wrapper or owner, default options, five-key schema, indexed phase
order, graph mutation, tensor pruning, callback arguments, pass order, adjacent
targets, dependencies, diagnostics, or TensorFlow behavior. The retained value
must have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- indexed quantized-logistic and Swish owners plus terminal orchestration,
  architecture, and pass-efficiency gate:
  `375 passed, 1 xfailed in 18.56s`
- branch-changed broad suite plus indexed logistic/Swish owners, terminal
  recovery/orchestration, architecture, and pass-efficiency coverage:
  `1433 passed, 1 xfailed in 24.14s`

The sole strict expected failure is the intentionally unimplemented terminal
Swish-QDQ-island result retention contract above. Implement that assignment,
rerun the same gates sequentially, then commit and push only; do not create,
reopen, or update a pull request.

## Terminal Swish-QDQ-island result retention implementation checkpoint

The sole model-only production call now retains its existing five-counter
dictionary as `_terminal_swish_qdq_island_stats`. This is an assignment-only
orchestration change. The wrapper and owner, default options, result schema,
indexed phase order, graph mutation, tensor pruning, callback arguments, pass
order, captured dequant-logistic predecessor, live-LayoutState InstanceNorm-
bias successor, dependencies, diagnostics, and TensorFlow behavior are
unchanged. The retained value has no consumer and triggers no additional graph
work.

Implementation validation completed sequentially under `uv`:

- indexed quantized-logistic and Swish owners plus terminal orchestration,
  architecture, and pass-efficiency gate: `376 passed in 18.74s`
- branch-changed broad suite plus indexed logistic/Swish owners, terminal
  recovery/orchestration, architecture, and pass-efficiency coverage:
  `1434 passed in 24.48s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit every production occurrence of
`_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains()`, then
isolate the terminal call immediately following `_terminal_swish_qdq_island_stats`
and its normalization-pad successor. Do not conflate nested convergence, late,
or final calls. Commit and push only; do not create, reopen, or update a pull
request.

## Terminal InstanceNorm post-bias result characterization checkpoint

`_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains()` returns
the stable one-counter dictionary
`optimized_transpose_instancenorm_posttranspose_bias_add_nhwc_chains`. It has
four direct production calls plus one nested convergence call. The third and
fourth direct results are already retained as
`_pre_terminal_affine_instancenorm_post_bias_stats` and
`_absolute_final_instancenorm_post_bias_stats`; the first terminal and second
very-late direct calls are raw.

A strict expected-failure orchestration contract requires only the first
direct result to be retained as `_terminal_instancenorm_post_bias_stats`. It
fixes the captured `_terminal_swish_qdq_island_stats` predecessor and following
diagnostics-aware normalization-pad cleanup, while preserving the second raw
call, the two later targets, and the nested call.

At implementation, replace only the first direct expression with an assignment
and update the existing four-direct occurrence-shape assertions accordingly.
Do not change the second direct or nested call, later targets, wrapper or
indexed owner, one-key schema, rewrite guards, positive-only pruning, optional
GraphIndex/candidate/max-rewrite controls, callback arguments, pass order,
adjacent targets, dependencies, diagnostics, or TensorFlow behavior. The
retained value must have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- owner rewrite, indexed Swish predecessor, terminal/final occurrence
  accounting, normalization/attention boundaries, architecture, and pass-
  efficiency gate: `385 passed, 1 xfailed in 19.20s`
- branch-changed broad suite plus indexed Swish, terminal/final occurrence
  accounting, normalization/attention boundaries, architecture, and pass-
  efficiency coverage: `1418 passed, 1 xfailed in 23.93s`

The sole strict expected failure is the intentionally unimplemented terminal
InstanceNorm post-bias result retention contract above. Implement that
assignment, rerun the same gates sequentially, then commit and push only; do
not create, reopen, or update a pull request.

## Terminal InstanceNorm post-bias result retention implementation checkpoint

The first of the four direct production calls now retains its existing one-
counter dictionary as `_terminal_instancenorm_post_bias_stats`. The second
very-late direct call remains raw; the third and fourth keep
`_pre_terminal_affine_instancenorm_post_bias_stats` and
`_absolute_final_instancenorm_post_bias_stats`; the nested convergence call is
unchanged. This is an assignment-only orchestration change. The wrapper and
indexed owner, schema, rewrite guards, positive-only pruning, GraphIndex/
LayoutState/candidate/max-rewrite controls, callback arguments, pass order,
captured Swish-QDQ predecessor, normalization-pad successor, dependencies,
diagnostics, and TensorFlow behavior are unchanged. The retained value has no
consumer and triggers no additional graph work.

The existing terminal-affine and absolute-final occurrence-shape contracts now
require the first target and second raw expression while preserving both later
targets.

Implementation validation completed sequentially under `uv`:

- owner rewrite, indexed Swish predecessor, terminal/final occurrence
  accounting, normalization/attention boundaries, architecture, and pass-
  efficiency gate: `386 passed in 19.95s`
- branch-changed broad suite plus indexed Swish, terminal/final occurrence
  accounting, normalization/attention boundaries, architecture, and pass-
  efficiency coverage: `1419 passed in 23.89s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the remaining raw second direct
`_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains()` result
in the very-late block. Keep `_terminal_instancenorm_post_bias_stats`,
`_pre_terminal_affine_instancenorm_post_bias_stats`,
`_absolute_final_instancenorm_post_bias_stats`, and the nested convergence call
fixed. Commit and push only; do not create, reopen, or update a pull request.

## Very-late InstanceNorm post-bias result characterization checkpoint

The second of the four direct
`_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains()` calls
returns the same stable one-counter dictionary as the three already captured
direct occurrences. It is the only remaining raw direct result; the nested
convergence call separately consumes its counter.

A strict expected-failure orchestration contract requires that second direct
result to be retained as `_very_late_instancenorm_post_bias_stats`. It fixes the
preceding diagnostics-aware `run_pad_layout_cleanup()` call and following live-
LayoutState InstanceNorm residual/Mul/Concat owner, while preserving the
terminal, pre-terminal, absolute-final, and nested occurrence contracts.

At implementation, replace only the second direct expression with an
assignment and update the existing four-direct occurrence-shape assertions.
Do not change the other direct or nested calls, existing targets, wrapper or
indexed owner, one-key schema, rewrite guards, positive-only pruning,
GraphIndex/LayoutState/candidate/max-rewrite controls, callback arguments, pass
order, adjacent targets, dependencies, diagnostics, or TensorFlow behavior.
The retained value must have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- owner rewrite, indexed Swish, all direct/nested occurrence accounting,
  terminal/final boundaries, architecture, and pass-efficiency gate:
  `386 passed, 1 xfailed in 19.35s`
- branch-changed broad suite plus indexed Swish, all occurrence accounting,
  terminal/final boundaries, architecture, and pass-efficiency coverage:
  `1419 passed, 1 xfailed in 24.80s`

The sole strict expected failure is the intentionally unimplemented very-late
InstanceNorm post-bias result retention contract above. Implement that
assignment, rerun the same gates sequentially, then commit and push only; do
not create, reopen, or update a pull request.

## Very-late InstanceNorm post-bias result retention implementation checkpoint

The second of the four direct production calls now retains its unchanged one-
counter dictionary as `_very_late_instancenorm_post_bias_stats`. All direct
occurrences therefore have distinct observation points:
`_terminal_instancenorm_post_bias_stats`,
`_very_late_instancenorm_post_bias_stats`,
`_pre_terminal_affine_instancenorm_post_bias_stats`, and
`_absolute_final_instancenorm_post_bias_stats`. The nested convergence call
continues to consume its counter internally.

This is an assignment-only orchestration change. It preserves the wrapper and
indexed owner, one-key result schema, rewrite guards, positive-only pruning,
GraphIndex/LayoutState/candidate/max-rewrite controls, callback arguments,
pass order, diagnostics-aware pad-layout predecessor, live-LayoutState
residual/Mul/Concat successor, existing direct targets, nested call,
dependencies, diagnostics, and TensorFlow behavior. The new retained value has
no consumer and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- owner rewrite, indexed Swish, all direct/nested occurrence accounting,
  terminal/final boundaries, architecture, and pass-efficiency gate:
  `387 passed in 20.17s`
- branch-changed broad suite plus indexed Swish, all occurrence accounting,
  terminal/final boundaries, architecture, and pass-efficiency coverage:
  `1420 passed in 23.78s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit all production occurrences of
`_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains()`.
There are three direct calls plus one nested convergence call: the first
terminal and second very-late direct results are raw, the third is retained as
`_pre_terminal_affine_instancenorm_residual_mul_concat_stats`, and the nested
call consumes its counter. Characterize the very-late direct call adjacent to
`_very_late_instancenorm_post_bias_stats` without conflating those other
occurrences. Commit and push only; do not create, reopen, or update a pull
request.

## Very-late InstanceNorm residual/Mul/Concat result characterization checkpoint

`_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains()`
returns the stable one-counter dictionary
`optimized_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains`. The
counter is complete mutation evidence because unused-tensor pruning and
LayoutState synchronization occur only after a positive rewrite.

The owner has three direct production calls plus one nested convergence call.
The first terminal and second very-late direct results are raw, the third is
retained as
`_pre_terminal_affine_instancenorm_residual_mul_concat_stats`, and the nested
call consumes its counter from the shared `residual_graph_index` invocation.

A strict expected-failure orchestration contract requires only the second
direct result to be retained as
`_very_late_instancenorm_residual_mul_concat_stats`. It fixes the captured
`_very_late_instancenorm_post_bias_stats` predecessor and following live-
LayoutState dual-statistics InstanceNorm owner. It also fixes the other two
direct forms and proves that exactly one nested occurrence continues to receive
`residual_graph_index` and the live Session LayoutState.

At implementation, replace only the second direct expression with an
assignment and update the existing three-direct occurrence-shape assertion.
Do not change the first direct or nested calls, existing staged target, wrapper
or indexed owner, one-key schema, rewrite guards, positive-only pruning,
GraphIndex/LayoutState/candidate/max-rewrite controls, callback arguments,
pass order, adjacent targets, dependencies, diagnostics, or TensorFlow
behavior. The retained value must have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- indexed owner, concrete owner rewrite, direct/nested occurrence accounting,
  terminal/final boundaries, architecture, and pass-efficiency gate:
  `468 passed, 1 xfailed in 19.92s`
- branch-changed broad suite plus indexed owner, occurrence accounting,
  terminal/final boundaries, architecture, and pass-efficiency coverage:
  `1420 passed, 1 xfailed in 24.22s`

The sole strict expected failure is the intentionally unimplemented very-late
InstanceNorm residual/Mul/Concat result retention contract above. Implement
that assignment, rerun the same gates sequentially, then commit and push only;
do not create, reopen, or update a pull request.

## Very-late InstanceNorm residual/Mul/Concat result retention implementation checkpoint

The second of the three direct production calls now retains its unchanged one-
counter dictionary as `_very_late_instancenorm_residual_mul_concat_stats`.
The first terminal direct call remains raw, the third retains
`_pre_terminal_affine_instancenorm_residual_mul_concat_stats`, and the nested
convergence call continues to consume its counter from the shared graph index.

This is an assignment-only orchestration change. It preserves the wrapper and
indexed owner, one-key result schema, rewrite guards, positive-only pruning,
GraphIndex/LayoutState/candidate/max-rewrite controls, callback arguments,
pass order, captured very-late post-bias predecessor, live-LayoutState dual-
statistics successor, other occurrence forms, dependencies, diagnostics, and
TensorFlow behavior. The retained value has no consumer and triggers no
additional graph work.

Implementation validation completed sequentially under `uv`:

- indexed owner, concrete owner rewrite, direct/nested occurrence accounting,
  terminal/final boundaries, architecture, and pass-efficiency gate:
  `469 passed in 19.78s`
- branch-changed broad suite plus indexed owner, occurrence accounting,
  terminal/final boundaries, architecture, and pass-efficiency coverage:
  `1421 passed in 24.65s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit all production occurrences of
`_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains()`.
There are three direct calls plus one nested convergence call: the first
terminal and second very-late direct results are raw, the third is retained as
`_pre_terminal_affine_instancenorm_dualstats_stats`, and the nested call
consumes its counter. Characterize the very-late call adjacent to
`_very_late_instancenorm_residual_mul_concat_stats` without conflating those
other occurrences. Commit and push only; do not create, reopen, or update a
pull request.

## Very-late dual-statistics InstanceNorm result characterization checkpoint

`_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains()`
returns the stable one-counter dictionary
`optimized_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains`.
The counter is complete mutation evidence because unused-tensor pruning and
LayoutState synchronization occur only after a positive rewrite.

The owner has three direct production calls plus one nested convergence call.
The first terminal and second very-late direct results are raw, the third is
retained as `_pre_terminal_affine_instancenorm_dualstats_stats`, and the nested
call consumes its counter from the shared `residual_graph_index` invocation.

A strict expected-failure orchestration contract requires only the second
direct result to be retained as `_very_late_instancenorm_dualstats_stats`. It
fixes the captured `_very_late_instancenorm_residual_mul_concat_stats`
predecessor and following singleton consecutive-Reshape cluster. It also fixes
the other two direct forms and proves that exactly one nested occurrence keeps
the shared graph index and live Session LayoutState.

At implementation, replace only the second direct expression with an
assignment and update the existing three-direct occurrence-shape assertion.
Do not change the first direct or nested calls, existing staged target, wrapper
or indexed owner, one-key schema, rewrite guards, positive-only pruning,
GraphIndex/LayoutState/candidate/max-rewrite controls, callback arguments,
pass order, adjacent targets, dependencies, diagnostics, or TensorFlow
behavior. The retained value must have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- dual-statistics and residual indexed owners, direct/nested occurrence
  accounting, terminal/final boundaries, architecture, and pass-efficiency
  gate: `674 passed, 1 xfailed in 20.13s`
- branch-changed broad suite plus both indexed owners, occurrence accounting,
  terminal/final boundaries, architecture, and pass-efficiency coverage:
  `1421 passed, 1 xfailed in 24.84s`

The sole strict expected failure is the intentionally unimplemented very-late
dual-statistics InstanceNorm result retention contract above. Implement that
assignment, rerun the same gates sequentially, then commit and push only; do
not create, reopen, or update a pull request.

## Very-late dual-statistics InstanceNorm result retention implementation checkpoint

The second of the three direct production calls now retains its unchanged one-
counter dictionary as `_very_late_instancenorm_dualstats_stats`. The first
terminal direct call remains raw, the third retains
`_pre_terminal_affine_instancenorm_dualstats_stats`, and the nested convergence
call continues to consume its counter from the shared graph index.

This is an assignment-only orchestration change. It preserves the wrapper and
indexed owner, one-key result schema, rewrite guards, positive-only pruning,
GraphIndex/LayoutState/candidate/max-rewrite controls, callback arguments,
pass order, captured very-late residual predecessor, following singleton
consecutive-Reshape cluster, other occurrence forms, dependencies,
diagnostics, and TensorFlow behavior. The retained value has no consumer and
triggers no additional graph work.

The first broad implementation run found one stale structural contract:
`test_singleton_consecutive_preserves_both_main_boundaries` required the
cluster predecessor to be `ast.Expr`. The production assignment was correct;
the test now requires `ast.Assign`, the exact
`_very_late_instancenorm_dualstats_stats` target, and the unchanged owner call.
The singleton runner and all other production boundaries remain unchanged.

Implementation validation completed sequentially under `uv`:

- stale boundary contract alone: `1 passed in 0.52s`
- dual-statistics/residual indexed owners, direct/nested occurrence accounting,
  singleton orchestration, terminal/final boundaries, architecture, and pass-
  efficiency gate: `686 passed in 20.88s`
- branch-changed broad suite plus both indexed owners, singleton orchestration,
  occurrence accounting, terminal/final boundaries, architecture, and pass-
  efficiency coverage: `1422 passed in 24.93s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the three production occurrences of
`_run_singleton_consecutive_reshape_pass_cluster()`: the first model-level call
after `_very_late_instancenorm_dualstats_stats` is raw, the second model-level
call destructures its three ordered results, and the conditional fallback call
is raw. Establish whether all three child dictionaries form complete mutation
evidence before selecting any new observation point. Commit and push only; do
not create, reopen, or update a pull request.

## Very-late singleton/consecutive-Reshape result characterization checkpoint

`_run_singleton_consecutive_reshape_pass_cluster()` forwards the three ordered
result dictionaries from singleton-channel Transpose cleanup, duplicate
Reshape fan-out cleanup, and consecutive Reshape cleanup. The empty-model
schema is fixed at one, one, and three mutation counters respectively. Each
child owner prunes only after a positive mutation count, so the tuple contains
pure mutation evidence without an additional tensor-count proxy.

The private runner has three production occurrences. The first model-level
call after `_very_late_instancenorm_dualstats_stats` is raw; the second model-
level call destructures all three dictionaries into the shared late
reconciliation guard; and the conditional fallback call remains a raw
expression with `fallback_ir` and no LayoutState.

A strict expected-failure orchestration contract requires only the first
model-level result to be retained as
`_very_late_singleton_consecutive_reshape_results`. It fixes the exact model
and live LayoutState arguments, captured dual-statistics predecessor,
following optional layout-transpose branch, later three-target destructuring,
and fallback expression.

At implementation, replace only the first model-level expression with an
assignment. Do not change the helper or child owners, three-result order or
schemas, shared state scope, callback arguments, diagnostics, later
destructuring, fallback call, pass order, reconciliation behavior,
dependencies, or TensorFlow behavior. The retained tuple must have no consumer
or additional graph work.

Characterization validation completed sequentially under `uv`:

- singleton/consecutive orchestration and all three child-owner families plus
  terminal occurrence, architecture, and pass-efficiency coverage:
  `379 passed, 1 xfailed in 19.25s`
- branch-changed broad suite plus the same singleton/consecutive owner and
  orchestration coverage: `1444 passed, 1 xfailed in 24.31s`

The sole strict expected failure is the intentionally unimplemented very-late
singleton/consecutive-Reshape result retention contract above. Implement that
assignment, rerun the same gates sequentially, then commit and push only; do
not create, reopen, or update a pull request.

## Very-late singleton/consecutive-Reshape result retention implementation checkpoint

The first model-level production call now retains its unchanged ordered three-
dictionary tuple as `_very_late_singleton_consecutive_reshape_results`. The
second model-level call continues to destructure
`shared_singleton_channel_stats`, `shared_duplicate_fanout_stats`, and
`shared_consecutive_reshape_stats` for its existing reconciliation guard. The
conditional fallback call remains a raw expression.

This is an assignment-only orchestration change. It preserves the helper and
three child owners, result order and schemas, shared state scope, callback
arguments, diagnostics, captured very-late dual-statistics predecessor,
following optional layout-transpose branch, later destructuring, fallback
call, pass order, reconciliation behavior, dependencies, and TensorFlow
behavior. The retained tuple has no consumer and triggers no additional graph
work.

Implementation validation completed sequentially under `uv`:

- singleton/consecutive orchestration and all three child-owner families plus
  terminal occurrence, architecture, and pass-efficiency coverage:
  `380 passed in 18.95s`
- branch-changed broad suite plus the same singleton/consecutive owner and
  orchestration coverage: `1445 passed in 24.34s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit every production occurrence and the complete result schema of
`run_layout_transpose_cleanup()`. Isolate the guarded very-late call immediately
after `_very_late_singleton_consecutive_reshape_results` without conflating the
other helper/late occurrences or changing the
`optimize_layout_transpose_chains` guard. Commit and push only; do not create,
reopen, or update a pull request.

## Very-late layout-Transpose cleanup result characterization checkpoint

`run_layout_transpose_cleanup()` returns a fixed five-key dictionary:
`iterations`, identity removals, inverse-pair removals, inverse-fanout branch
removals, and consecutive-pair compositions. The four rewrite counters cover
operator rewrites, but the underlying owner calls unused-tensor pruning
unconditionally. A zero-counter result can therefore omit a prune-only ModelIR
change. The dictionary must not drive a stability or scan-elision guard without
an independently sampled tensor-count delta.

The lowerer has three direct occurrences. The earlier primary layout block and
the very-late block are guarded raw expressions; the late-Concat occurrence is
already retained as `_late_concat_transpose_layout_stats` with its shared pass
state scope. A strict expected-failure contract selects only the very-late
guarded call for `_very_late_layout_transpose_cleanup_stats`.

The contract fixes the `optimize_layout_transpose_chains` guard, exact live
LayoutState and diagnostics arguments, captured
`_very_late_singleton_consecutive_reshape_results` predecessor, following
rank-four broadcast-constant repair, and all three occurrence forms. The new
target is observation-only and must have no consumer or additional graph work.

At implementation, replace only the expression inside that guard with an
assignment and update the existing three-occurrence accounting assertion. Do
not change owner pruning, result schema, arguments, state-scope behavior, guard
condition, adjacent calls, pass order, reconciliation, dependencies,
diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- layout-Transpose owner, late-binary integration, terminal occurrence,
  singleton boundary, architecture, and pass-efficiency coverage:
  `363 passed, 1 xfailed in 19.37s`
- branch-changed broad suite plus the same owner and orchestration coverage:
  `1427 passed, 1 xfailed in 24.34s`

The sole strict expected failure is the intentionally unimplemented very-late
layout-Transpose cleanup result retention contract above. Implement only the
observation assignment, rerun the same gates sequentially, then commit and push
only; do not create, reopen, or update a pull request.

## Very-late layout-Transpose cleanup result retention implementation checkpoint

The guarded very-late call now retains its unchanged five-key dictionary as
`_very_late_layout_transpose_cleanup_stats`. The earlier guarded primary-layout
occurrence remains raw, and the late-Concat occurrence keeps
`_late_concat_transpose_layout_stats` and its shared state scope.

This is an assignment-only orchestration change. The retained dictionary is
explicitly observation-only because its rewrite counters omit possible zero-
rewrite unused-tensor pruning. No stability guard, reconciliation decision, or
scan elision consumes it. Owner behavior and pruning, result schema, live
LayoutState and diagnostics arguments, option guard, captured singleton tuple
predecessor, broadcast-repair successor, other occurrences, pass order,
dependencies, and TensorFlow behavior are unchanged.

The first focused implementation run exposed a bug in the newly added
characterization selector: it assumed cleanup was the first statement in each
guard, while the earlier primary-layout guard has setup statements before the
call. The selector now inspects every direct guard statement and still uses the
captured predecessor to isolate the very-late guard. Production was unchanged
by this test-only correction.

Implementation validation completed sequentially under `uv`:

- corrected strict contract alone: `1 passed in 0.55s`
- layout-Transpose owner, late-binary integration, terminal occurrence,
  singleton boundary, architecture, and pass-efficiency coverage:
  `364 passed in 18.87s`
- branch-changed broad suite plus the same owner and orchestration coverage:
  `1428 passed in 24.87s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit every production occurrence and the complete result schema of
`_repair_rank4_channelwise_broadcast_constants_to_runtime_layout()`. Isolate
the very-late raw call immediately after the guarded layout-Transpose result
without conflating the nested recovery, fallback, or final retained calls.
Commit and push only; do not create, reopen, or update a pull request.

## Very-late broadcast-constant repair result characterization checkpoint

`_repair_rank4_channelwise_broadcast_constants_to_runtime_layout()` returns
the fixed one-key dictionary
`repaired_rank4_channelwise_broadcast_constants`. Every counted mutation is a
constant data/shape update or a shared-constant clone plus indexed operator-
input rewire. The owner has no cleanup-only path, so the counter is complete
mutation evidence.

There are four production occurrences. Indexed binary convergence consumes its
result with a shared GraphIndex; the very-late model-level direct call is raw;
and the fallback and final calls retain their results for existing positive-
count static-shape reconciliation guards.

A strict expected-failure contract requires only the very-late direct result
to be retained as `_very_late_broadcast_repair_stats`. It fixes the preceding
guard that retains `_very_late_layout_transpose_cleanup_stats`, immediate raw
static-shape reconciliation successor, final target, one fallback occurrence,
and exactly one graph-indexed convergence occurrence.

At implementation, replace only the very-late expression with an assignment.
Do not change the wrapper or indexed owner, one-key schema, GraphIndex behavior,
constant clone policy, callback arguments, existing fallback/final guards,
adjacent reconciliation, pass order, dependencies, diagnostics, or TensorFlow
behavior. The retained value must have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- broadcast owner, indexed convergence, safety fallback, terminal occurrence,
  architecture, and pass-efficiency coverage:
  `376 passed, 1 xfailed in 19.84s`
- branch-changed broad suite plus the same owner and orchestration coverage:
  `1424 passed, 1 xfailed in 24.94s`

The sole strict expected failure is the intentionally unimplemented very-late
broadcast-constant repair result retention contract above. Implement only that
assignment, rerun the same gates sequentially, then commit and push only; do
not create, reopen, or update a pull request.

## Very-late broadcast-constant repair result retention implementation checkpoint

The very-late model-level call now retains its unchanged one-counter dictionary
as `_very_late_broadcast_repair_stats`. The fallback and final calls retain
their existing targets and positive static-shape reconciliation guards. The
indexed convergence helper continues to consume the fourth module-wide
occurrence with its shared GraphIndex.

This is an assignment-only orchestration change. It preserves the wrapper and
indexed owner, one-key schema, GraphIndex maintenance, constant clone policy,
callback arguments, existing fallback/final guards, guarded layout-Transpose
predecessor, immediate unconditional static-shape reconciliation, pass order,
dependencies, diagnostics, and TensorFlow behavior. The new target has no
consumer and triggers no additional graph work.

The first implementation run exposed an error in the newly added occurrence
inventory. There are four module-wide calls, but only three are inside
`lower_onnx_to_ir`; the graph-indexed convergence call is a separate module-
level helper. The corrected contract counts three lowerer calls, one fallback
argument form, and four module calls with exactly one `graph_index` argument.
Production was unchanged by this test-only correction.

Implementation validation completed sequentially under `uv`:

- corrected strict contract alone: `1 passed in 0.60s`
- broadcast owner, indexed convergence, safety fallback, terminal occurrence,
  architecture, and pass-efficiency coverage: `377 passed in 18.74s`
- branch-changed broad suite plus the same owner and orchestration coverage:
  `1425 passed in 24.42s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediate very-late `_reconcile_static_tensor_shapes()`
call and every preceding mutation source before considering a guard. The
captured broadcast counter is complete, but the adjacent layout-Transpose
dictionary omits zero-rewrite unused-tensor pruning, so it cannot by itself
prove the graph stable. Commit and push only; do not create, reopen, or update
a pull request.

## Very-late post-broadcast static-shape result characterization checkpoint

The static-shape reconciliation immediately after
`_very_late_broadcast_repair_stats` is currently unconditional and discards its
result. It cannot safely be guarded by that broadcast counter alone: the
preceding guarded layout-Transpose cleanup can prune unused tensors while
reporting zero rewrites.

The reconciler's default `reconciled_static_tensor_shapes` key counts only
tensor shape writes. Its established opt-in
`reconciled_static_shape_mutations` key also counts constant shape-parameter,
operator-option, and direct tensor-metadata writes during the same fixed-point
walk. Requesting it adds no ModelIR copy, fingerprint, or graph traversal.

A strict expected-failure contract therefore keeps the call unconditional but
requires `include_mutation_count=True` and retains the two-key result as
`_very_late_broadcast_static_shape_stats`. It fixes the captured broadcast-
repair predecessor and following `shared_late_tensor_count` boundary. The new
target is observation-only and must not control a guard or add graph work.

At implementation, replace only the raw call with that assignment and opt-in
keyword. Do not change reconciler behavior, fixed-point order, preceding
passes, unconditional execution, adjacent tensor count, result consumers,
dependencies, diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- complete reconciler contract, layout-Transpose/broadcast owners, terminal
  occurrence, architecture, and pass-efficiency coverage:
  `356 passed, 1 xfailed in 18.98s`
- branch-changed broad suite plus the same reconciliation and owner coverage:
  `1429 passed, 1 xfailed in 24.26s`

The sole strict expected failure is the intentionally unimplemented very-late
post-broadcast static-shape result retention contract above. Implement only the
unconditional assignment and opt-in counter, rerun the same gates sequentially,
then commit and push only; do not create, reopen, or update a pull request.

## Very-late post-broadcast static-shape result retention implementation checkpoint

The unconditional reconciliation now requests
`include_mutation_count=True` and retains the two-key result as
`_very_late_broadcast_static_shape_stats`. The additional key counts constant
shape-parameter, operator-option, direct tensor-metadata, and ordinary output-
shape writes during the existing fixed-point walk. It adds no ModelIR copy,
fingerprint, or traversal.

This change preserves unconditional execution because the preceding layout-
Transpose result does not expose prune-only mutation. Reconciler behavior and
fixed-point order, preceding owners, captured broadcast target, following
tensor-count boundary, pass order, dependencies, diagnostics, and TensorFlow
behavior remain unchanged. The new result has no consumer and controls no
guard.

The first implementation run exposed one stale adjacent boundary assertion
that required the successor to be a raw default-schema call. It now requires
the exact `_very_late_broadcast_static_shape_stats` assignment and
`include_mutation_count=True`. Production was unchanged by this test-only
correction.

Implementation validation completed sequentially under `uv`:

- broadcast and reconciliation boundary contracts: `2 passed in 0.64s`
- complete reconciler contract, layout-Transpose/broadcast owners, terminal
  occurrence, architecture, and pass-efficiency coverage:
  `357 passed in 19.30s`
- branch-changed broad suite plus the same reconciliation and owner coverage:
  `1430 passed in 25.27s`

These are unit, contract, and orchestration checks; this observation-only
change does not claim a new model-corpus run.

At resume, audit the guarded shared-late `_reconcile_static_tensor_shapes()`
call after the boundary-signature, HardSwish, Squeeze, Conv-Transpose, two
binary-repair, and three singleton/consecutive dictionaries plus tensor-count
delta. Preserve its existing predicate and characterize only a complete opt-in
result target. Commit and push only; do not create, reopen, or update a pull
request.

## Guarded shared-late static-shape result characterization checkpoint

The shared-late static-shape reconciliation is already guarded by nine pure
mutation-result dictionaries: boundary-signature, HardSwish, Squeeze, wrong-
way Conv-Transpose, two binary repairs, and the three singleton/consecutive
cluster results. A `len(model_ir.tensors) < shared_late_tensor_count` clause
also covers cleanup-only pruning.

Existing runtime fixtures independently force each dictionary positive and
the tensor-count delta, proving that every changed outcome adds exactly one
reconciliation over the all-zero/no-prune path. The predicate is complete and
must not change.

A strict expected-failure structural contract requires only the guarded call
to retain `_shared_late_static_shape_stats` with
`include_mutation_count=True`. It fixes all nine evidence names, the prune
clause, one-statement guard body, and following
`late_binary_repair_tensor_count` boundary. The result is observation-only and
must have no consumer.

At implementation, replace only the guarded expression with the assignment
and opt-in keyword. Do not alter the predicate, execution count, reconciler
fixed point, preceding owners, following tensor count, pass order,
dependencies, diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- positive-count helper, runtime nine-result/prune guard, complete reconciler,
  terminal occurrence, singleton boundary, architecture, and pass-efficiency
  coverage: `364 passed, 1 xfailed in 19.99s`
- branch-changed broad suite plus the same runtime and structural coverage:
  `1426 passed, 1 xfailed in 24.78s`

The sole strict expected failure is the intentionally unimplemented guarded
shared-late result retention contract above. Implement only that assignment
and opt-in counter, rerun the same gates sequentially, then commit and push
only; do not create, reopen, or update a pull request.

## Guarded shared-late static-shape result retention implementation checkpoint

The existing shared-late predicate remains unchanged over nine pure mutation-
result dictionaries plus the tensor-count decrease that covers prune-only
cleanup. When the guard fires, its reconciliation now requests
`include_mutation_count=True` and retains
`_shared_late_static_shape_stats`. The complete result has no consumer.

This change preserves execution count, predicate order and names, reconciler
fixed point, preceding owners, following `late_binary_repair_tensor_count`,
pass order, dependencies, diagnostics, and TensorFlow behavior. Runtime tests
continue to force each positive dictionary and the prune delta independently.

The first implementation run found the expected stale architecture assertion
that required the guard body to be `ast.Expr`. During its correction, a
structurally similar late-binary assertion was initially selected; that hunk
was restored immediately, and the update was reapplied with the shared-late
test function as context. The final targeted gate explicitly verifies the
unchanged late-binary expression and the new shared-late assignment.

Implementation validation completed sequentially under `uv`:

- late-binary raw contract, shared-late assignment contract, and terminal
  result contract: `3 passed in 2.22s`
- positive-count helper, runtime nine-result/prune guard, complete reconciler,
  terminal occurrence, singleton boundary, architecture, and pass-efficiency
  coverage: `365 passed in 18.89s`
- branch-changed broad suite plus the same runtime and structural coverage:
  `1427 passed in 24.69s`

These are unit, contract, runtime-orchestration, and architecture checks; this
result-retention change does not claim a new model-corpus run.

At resume, audit the guarded late-binary `_reconcile_static_tensor_shapes()`
call after `late_signature_stats`, `late_binary_adapter_stats`, and
`late_singleton_adapter_stats` plus `late_binary_repair_tensor_count`.
Preserve its existing predicate and characterize only a complete opt-in result
target. Commit and push only; do not create, reopen, or update a pull request.

## Guarded late-binary static-shape result characterization checkpoint

The late-binary static-shape reconciliation is already guarded by three
mutation counters: static shape-signature sanitization, rank-four binary layout
adapter insertion, and singleton broadcast repair. A
`len(model_ir.tensors) < late_binary_repair_tensor_count` clause covers
cleanup-only pruning.

The existing runtime fixture forces each positive counter and the tensor-count
delta independently, proving that every changed outcome adds one reconciliation
over the stable path. The predicate and execution count must remain unchanged.

A strict expected-failure contract requires only the guard body to retain
`_late_binary_repair_static_shape_stats` with
`include_mutation_count=True`. It fixes all result and counter names, the prune
clause, one-statement body, and following optional late-binary layout-recovery
guard. The complete result must have no consumer.

At implementation, replace only the guarded expression with that assignment
and opt-in keyword. Do not alter the predicate, reconciler fixed point,
preceding repairs, following option guard, pass order, dependencies,
diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- runtime counter/prune guard, complete reconciler, terminal occurrence,
  architecture, and pass-efficiency coverage:
  `352 passed, 1 xfailed in 20.24s`
- branch-changed broad suite plus the same runtime and structural coverage:
  `1427 passed, 1 xfailed in 24.79s`

The sole strict expected failure is the intentionally unimplemented guarded
late-binary result retention contract above. Implement only that assignment
and opt-in counter, rerun the same gates sequentially, then commit and push
only; do not create, reopen, or update a pull request.

## Guarded late-binary static-shape result retention implementation checkpoint

The existing three-counter plus tensor-count predicate remains unchanged. When
it fires, static-shape reconciliation now requests
`include_mutation_count=True` and retains
`_late_binary_repair_static_shape_stats`. The complete result has no consumer.

This change preserves every evidence and counter name, prune-only detection,
execution count, reconciler fixed point, preceding repairs, following optional
late-binary layout-recovery guard, pass order, dependencies, diagnostics, and
TensorFlow behavior. The runtime fixture continues to force each positive
counter and tensor-count delta independently.

Implementation validation completed sequentially under `uv`:

- runtime counter/prune guard, complete reconciler, terminal occurrence,
  architecture, and pass-efficiency coverage: `353 passed in 21.19s`
- branch-changed broad suite plus the same runtime and structural coverage:
  `1428 passed in 24.96s`

These are unit, contract, runtime-orchestration, and architecture checks; this
result-retention change does not claim a new model-corpus run.

At resume, audit the guarded reconciliation immediately after
`late_binary_layout_recovery_stats`. Its aggregate already excludes iteration
metrics and includes clamped net tensor reduction, so preserve the existing
positive-count predicate and characterize only a complete opt-in result target.
Commit and push only; do not create, reopen, or update a pull request.

## Guarded late-binary layout-recovery shape result characterization checkpoint

`late_binary_layout_recovery_stats` is a complete mutation aggregate. The
runner excludes non-mutating iteration metrics, includes fixed rewrite
counters, and adds clamped net tensor reduction for cleanup-only pruning. Its
nested `_stats_have_positive_count()` guard is already covered by runtime
rewrite, prune, and stable outcomes.

A strict expected-failure contract requires only the reconciliation inside
that nested guard to retain
`_late_binary_layout_recovery_static_shape_stats` with
`include_mutation_count=True`. It fixes the outer layout-option predicate,
runner arguments, recovery target, inner positive predicate, one-statement
body, and following pre-terminal InstanceNorm evidence boundary.

At implementation, replace only the inner raw reconciliation with that
assignment and opt-in keyword. Do not alter either guard, aggregate schema,
runner behavior, reconciler fixed point, following terminal pass, pass order,
dependencies, diagnostics, or TensorFlow behavior. The complete result must
have no consumer.

Characterization validation completed sequentially under `uv`:

- recovery aggregate/orchestration, runtime rewrite/prune guard, complete
  reconciler, architecture, and pass-efficiency coverage:
  `298 passed, 1 xfailed in 17.48s`
- branch-changed broad suite plus the same runtime and structural coverage:
  `1428 passed, 1 xfailed in 24.32s`

The sole strict expected failure is the intentionally unimplemented guarded
late-binary layout-recovery shape result contract above. Implement only that
assignment and opt-in counter, rerun the same gates sequentially, then commit
and push only; do not create, reopen, or update a pull request.

## Guarded late-binary layout-recovery shape result retention implementation checkpoint

The reconciliation inside the existing nested positive-count guard now
requests `include_mutation_count=True` and retains
`_late_binary_layout_recovery_static_shape_stats`. The complete result has no
consumer.

This change preserves the recovery aggregate schema, outer layout-option
predicate, runner arguments, inner positive-count guard, runtime execution
count, reconciler fixed point, following pre-terminal InstanceNorm evidence,
pass order, dependencies, diagnostics, and TensorFlow behavior. Runtime tests
continue to distinguish rewrite, prune-only, and stable outcomes.

Implementation validation completed sequentially under `uv`:

- recovery aggregate/orchestration, runtime rewrite/prune guard, complete
  reconciler, architecture, and pass-efficiency coverage:
  `299 passed in 19.68s`
- branch-changed broad suite plus the same runtime and structural coverage:
  `1429 passed in 24.81s`

These are unit, contract, runtime-orchestration, and architecture checks; this
result-retention change does not claim a new model-corpus run.

At resume, audit all four production occurrences of
`_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains()`:
three direct calls plus one nested convergence call. The first terminal direct
result remains raw, while the very-late and pre-terminal results are retained.
Characterize only that first direct call and its residual-adapter predecessor
and dual-statistics successor. Commit and push only; do not create, reopen, or
update a pull request.

## Terminal InstanceNorm residual/Mul/Concat result characterization checkpoint

`_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains()` has
three direct production calls plus one nested convergence call. Its fixed one-
counter result is complete mutation evidence because unused-tensor pruning and
LayoutState synchronization occur only after a positive rewrite.

The first terminal direct result remains raw. The second and third are retained
as `_very_late_instancenorm_residual_mul_concat_stats` and
`_pre_terminal_affine_instancenorm_residual_mul_concat_stats`; the nested call
consumes its counter with `residual_graph_index`.

A strict expected-failure contract selects only the first direct result for
`_terminal_instancenorm_residual_mul_concat_stats`. It fixes the preceding
live-LayoutState residual/add-to-single-adapter owner, following live-
LayoutState dual-statistics owner, both retained later targets, and exactly one
graph-indexed nested occurrence.

At implementation, replace only the first direct expression with an
assignment and update the existing three-direct occurrence accounting. Do not
change the wrapper or owner, one-key schema, positive-only pruning, GraphIndex/
LayoutState behavior, other occurrence forms, adjacent calls, pass order,
dependencies, diagnostics, or TensorFlow behavior. The retained value must
have no consumer or additional graph work.

Characterization validation completed sequentially under `uv`:

- indexed owner, concrete rewrite, terminal/direct/nested occurrence,
  pre-terminal boundary, architecture, and pass-efficiency coverage:
  `464 passed, 1 xfailed in 20.07s`
- branch-changed broad suite plus the same owner and orchestration coverage:
  `1429 passed, 1 xfailed in 24.83s`

The sole strict expected failure is the intentionally unimplemented terminal
InstanceNorm residual/Mul/Concat result retention contract above. Implement
only that assignment, rerun the same gates sequentially, then commit and push
only; do not create, reopen, or update a pull request.

## Terminal InstanceNorm residual/Mul/Concat result retention implementation checkpoint

The first direct production call now retains its unchanged one-counter
dictionary as `_terminal_instancenorm_residual_mul_concat_stats`. The second
and third retain `_very_late_instancenorm_residual_mul_concat_stats` and
`_pre_terminal_affine_instancenorm_residual_mul_concat_stats`; the nested
convergence call continues to consume its counter with the shared graph index.

This is an assignment-only orchestration change. It preserves the wrapper and
indexed owner, one-key schema, positive-only pruning, GraphIndex/LayoutState
behavior, residual-adapter predecessor, dual-statistics successor, all other
occurrences, pass order, dependencies, diagnostics, and TensorFlow behavior.
The new result has no consumer and triggers no graph work.

The first implementation run found the expected stale very-late occurrence
assertion that still required the terminal call to be `ast.Expr`. It now
requires the exact terminal assignment. Production was unchanged by this test-
only correction.

Implementation validation completed sequentially under `uv`:

- terminal and very-late cross-occurrence contracts: `2 passed in 0.62s`
- indexed owner, concrete rewrite, terminal/direct/nested occurrence,
  pre-terminal boundary, architecture, and pass-efficiency coverage:
  `465 passed in 19.59s`
- branch-changed broad suite plus the same owner and orchestration coverage:
  `1430 passed in 25.07s`

These are unit, contract, and orchestration checks; this result-retention change
does not claim a new model-corpus run.

At resume, audit all four production occurrences of
`_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains()`:
three direct calls plus one nested convergence call. The first terminal direct
result remains raw, while the very-late and pre-terminal results are retained.
Characterize only that first call between the terminal residual/Mul/Concat
target and the terminal boundary cluster. Commit and push only; do not create,
reopen, or update a pull request.

## Terminal dual-statistics InstanceNorm result characterization checkpoint

`_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains()`
has three direct production calls plus one nested convergence call. Its fixed
one-counter result is complete mutation evidence because unused-tensor pruning
and LayoutState synchronization occur only after a positive rewrite.

The first terminal direct result remains raw. The second and third are retained
as `_very_late_instancenorm_dualstats_stats` and
`_pre_terminal_affine_instancenorm_dualstats_stats`; the nested call consumes
its counter with `residual_graph_index`.

A strict expected-failure contract selects only the first direct result for
`_terminal_instancenorm_dualstats_stats`. It fixes the preceding
`_terminal_instancenorm_residual_mul_concat_stats` assignment, following
terminal boundary cluster, both retained later targets, and exactly one graph-
indexed nested occurrence.

At implementation, replace only the first direct expression with an assignment
and update the existing three-direct occurrence accounting. Do not change the
wrapper or owner, one-key schema, positive-only pruning, GraphIndex/LayoutState
behavior, other occurrence forms, adjacent calls, pass order, dependencies,
diagnostics, or TensorFlow behavior. The retained value must have no consumer
or additional graph work.

Characterization validation completed sequentially under `uv`:

- dual-statistics indexed owner, terminal/direct/nested occurrence, pre-
  terminal boundary, architecture, and pass-efficiency coverage:
  `568 passed, 1 xfailed in 20.04s`
- branch-changed broad suite plus the same owner and orchestration coverage:
  `1430 passed, 1 xfailed in 25.79s`

The sole strict expected failure is the intentionally unimplemented terminal
dual-statistics InstanceNorm result retention contract above. Implement only
that assignment, rerun the same gates sequentially, then commit and push only;
do not create, reopen, or update a pull request.

## Terminal dual-statistics InstanceNorm result retention implementation checkpoint

The first direct production call now retains its unchanged one-counter
dictionary as `_terminal_instancenorm_dualstats_stats`. The second and third
retain `_very_late_instancenorm_dualstats_stats` and
`_pre_terminal_affine_instancenorm_dualstats_stats`; the nested convergence
call continues to consume its counter with the shared graph index.

This is an assignment-only orchestration change. It preserves the wrapper and
indexed owner, one-key schema, positive-only pruning, GraphIndex/LayoutState
behavior, terminal residual predecessor, terminal boundary-cluster successor,
all other occurrences, pass order, dependencies, diagnostics, and TensorFlow
behavior. The new result has no consumer and triggers no graph work.

The first implementation run found the expected stale terminal-boundary
architecture assertion that required its predecessor to be `ast.Expr`. It now
requires the exact terminal dual-statistics assignment. The very-late and pre-
terminal cross-occurrence contracts were updated in the same bounded unit.

Implementation validation completed sequentially under `uv`:

- boundary and terminal/very-late occurrence contracts: `3 passed in 2.27s`
- dual-statistics indexed owner, terminal/direct/nested occurrence, pre-
  terminal boundary, architecture, and pass-efficiency coverage:
  `569 passed in 20.57s`
- branch-changed broad suite plus the same owner and orchestration coverage:
  `1431 passed in 25.37s`

These are unit, contract, and orchestration checks; this result-retention change
does not claim a new model-corpus run.

At resume, audit all production occurrences and the complete result schema of
`_optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains()`.
Isolate the terminal raw call immediately before
`_terminal_instancenorm_residual_mul_concat_stats` without conflating any
nested convergence occurrence. Commit and push only; do not create, reopen, or
update a pull request.

## Terminal InstanceNorm residual-add result characterization checkpoint

`_optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains()`
has exactly two production occurrences. The nested convergence occurrence
consumes its result with `residual_graph_index`; the terminal direct occurrence
after diagnostics-aware normalization/pad cleanup remains raw.

The indexed owner returns the fixed one-counter dictionary
`optimized_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains`.
Unused-tensor pruning and LayoutState synchronization run only after a positive
rewrite, so the counter covers all owner mutation paths.

A strict expected-failure contract selects only the terminal direct result for
`_terminal_instancenorm_residual_add_stats`. It fixes the live Session
LayoutState argument, diagnostics-aware normalization/pad predecessor, retained
`_terminal_instancenorm_residual_mul_concat_stats` successor, total occurrence
count, and exactly one graph-indexed nested occurrence.

At implementation, replace only the terminal direct expression with an
assignment. Do not change the wrapper or owner, one-key schema, positive-only
pruning, GraphIndex/LayoutState behavior, nested counter consumption, adjacent
calls, pass order, dependencies, diagnostics, or TensorFlow behavior. The
retained value must have no consumer and trigger no graph work.

Characterization validation completed sequentially under `uv`:

- indexed residual-add owner, terminal/direct/nested occurrence, adjacent
  terminal InstanceNorm results, architecture, and pass-efficiency coverage:
  `446 passed, 1 xfailed in 19.82s`

The sole strict expected failure is the intentionally unimplemented terminal
result-retention contract above. Implement only that assignment, rerun focused
and branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## Terminal InstanceNorm residual-add result retention implementation checkpoint

The terminal direct occurrence now retains its unchanged one-counter
dictionary as `_terminal_instancenorm_residual_add_stats`. The nested
convergence occurrence continues to consume the same counter with
`residual_graph_index`.

This is an assignment-only orchestration change. It preserves the wrapper and
indexed owner, one-key schema, positive-only pruning, GraphIndex/LayoutState
behavior, diagnostics-aware normalization/pad predecessor, retained terminal
residual/Mul/Concat successor, pass order, dependencies, diagnostics, and
TensorFlow behavior. The new result has no consumer and triggers no graph work.

Implementation validation completed sequentially under `uv`:

- indexed residual-add owner, terminal/direct/nested occurrence, adjacent
  terminal InstanceNorm results, architecture, and pass-efficiency coverage:
  `447 passed in 20.01s`
- branch-changed broad suite plus the same owner and orchestration coverage:
  `1515 passed in 25.58s`

These are unit, contract, and orchestration checks; this result-retention change
does not claim a new model-corpus run.

At resume, audit the result schema and every production occurrence of the
diagnostics-aware `run_normalization_pad_layout_cleanup()` immediately before
`_terminal_instancenorm_residual_add_stats`. Determine whether its aggregate
fully represents all graph mutation, including cleanup-only paths, before
retaining or consuming it. Commit and push only; do not create, reopen, or
update a pull request.

## Terminal normalization/pad result characterization checkpoint

`run_normalization_pad_layout_cleanup()` returns a fixed two-key dictionary for
the InstanceNorm/Pad and flattened global-norm/Pad child owners. Both children
unconditionally prune unused tensors after candidate processing but count only
successful rewrites. The aggregate can therefore be retained for observation,
but its counters are incomplete evidence for cleanup-only mutation and must not
be used alone in a guard.

There are two direct lowerer occurrences and two orchestrated callback
selections. The loop-local direct result remains consumed as
`normalization_pad_stats`; the terminal direct result after
`_terminal_instancenorm_post_bias_stats` remains raw. The very-late and
absolute-final orchestrators continue to select flatten-only invocations with
shared pass-state scopes.

A strict expected-failure contract selects only the terminal direct result for
`_terminal_normalization_pad_stats`. It fixes the live Session LayoutState,
diagnostics sink, captured post-bias predecessor, captured residual-add
successor, total direct-call count, and loop-local result consumer.

At implementation, replace only the terminal direct expression with an
assignment. Do not add a guard or consumer, and do not change either child
owner, the two-key schema, unconditional pruning, loop convergence,
orchestrated selections, state scopes, adjacent calls, pass order,
dependencies, diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- terminal normalization/pad boundary, architecture, pass efficiency, and
  very-late orchestration coverage: `374 passed, 1 xfailed in 19.77s`
- concrete InstanceNorm/Pad and flattened global-norm/Pad fixtures:
  `2 passed, 739 deselected in 0.59s`

The sole strict expected failure is the intentionally unimplemented terminal
result-retention contract. Implement only that observation-only assignment,
rerun focused and branch-changed broad gates sequentially, then commit and push
only; do not create, reopen, or update a pull request.

## Terminal normalization/pad result retention implementation checkpoint

The terminal direct call now retains its unchanged fixed two-key dictionary as
`_terminal_normalization_pad_stats`. The loop-local call remains consumed as
`normalization_pad_stats`, and the very-late and absolute-final orchestrators
retain their flatten-only callback selections and shared state scopes.

This is an assignment-only observation change. Because both child owners can
prune unused tensors while reporting zero rewrites, the new result has no
consumer and is not used as guard evidence. The child owners, schema,
unconditional pruning, live LayoutState, diagnostics sink, adjacent captured
InstanceNorm results, pass order, dependencies, and TensorFlow behavior remain
unchanged.

Implementation validation completed sequentially under `uv`:

- terminal normalization/pad boundary, architecture, pass efficiency, and
  very-late orchestration coverage: `375 passed in 19.73s`
- concrete InstanceNorm/Pad and flattened global-norm/Pad fixtures:
  `2 passed, 739 deselected in 0.60s`
- branch-changed broad suite plus the same orchestration coverage:
  `1433 passed in 24.60s`

These are unit, contract, and orchestration checks; this result-retention change
does not claim a new model-corpus run.

At resume, audit the next raw terminal result boundary rather than consuming
`_terminal_normalization_pad_stats` as complete mutation evidence. Preserve
the explicit observation-only limitation, commit and push only, and do not
create, reopen, or update a pull request.

## Terminal boundary-layout result propagation characterization checkpoint

`run_terminal_boundary_layout()` executes five ordered owners through
`run_recovery_invocations()` and one shared `ModelIRPassStateScope`. The generic
runner already returns the ordered five-result tuple, but the terminal phase
runner currently drops it and the local
`_run_terminal_boundary_layout_pass_cluster()` helper is annotated and
implemented as returning `None`. Its sole primary call is consequently raw.

A strict expected-failure contract requires the phase runner to return the
existing tuple, the local helper to transparently return it, and the primary
call to retain it as `_terminal_boundary_layout_results`. It fixes the captured
`_terminal_instancenorm_dualstats_stats` predecessor and the following
`optimize_layout_transpose_chains` guard.

The first characterization run also exposed a stale dedicated boundary test
that still required the already-retained dual-statistics predecessor to be an
`ast.Expr`. It now requires the exact assignment and owner call; no production
code changed for that correction.

At implementation, propagate only the existing tuple. Do not summarize or
consume it, do not run a child twice, and do not change child order, arguments,
shared scope, diagnostics, adjacent calls, pass order, dependencies, or
TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- terminal boundary owner/orchestration, adjacent mean/attention boundary,
  terminal result contracts, architecture, and pass-efficiency coverage:
  `374 passed, 1 xfailed in 19.12s`

The sole strict expected failure is the intentionally unimplemented result
propagation contract. Implement only the transparent returns and primary
assignment, rerun focused and branch-changed broad gates sequentially, then
commit and push only; do not create, reopen, or update a pull request.

## Terminal boundary-layout result propagation implementation checkpoint

`run_terminal_boundary_layout()` now returns the existing ordered five-result
tuple from `run_recovery_invocations()`. The local
`_run_terminal_boundary_layout_pass_cluster()` helper transparently returns it,
and the sole primary call retains it as
`_terminal_boundary_layout_results`.

The result is not summarized or consumed. Each child still runs exactly once
in the same order and with the same shared `ModelIRPassStateScope`, arguments,
live LayoutState, and diagnostics sink. Adjacent terminal InstanceNorm and
optional mean/attention boundaries, dependencies, and TensorFlow behavior are
unchanged.

The first implementation gate found four expected stale AST contracts that
required raw expression statements at the helper, primary invocation, or
adjacent mean/attention boundary. They now require the transparent return and
exact retained assignment; no extra production behavior was introduced.

Implementation validation completed sequentially under `uv`:

- terminal boundary owner/orchestration, adjacent mean/attention boundary,
  terminal result contracts, architecture, and pass-efficiency coverage:
  `375 passed in 21.39s`
- branch-changed broad suite plus the same boundary coverage:
  `1456 passed in 25.15s`

These are unit, contract, and orchestration checks; this result propagation
does not claim a new model-corpus run.

At resume, audit the guarded terminal mean/attention helper result immediately
after `_terminal_boundary_layout_results`. Preserve the existing option guard
and child policy before deciding whether ordered results can be retained.
Commit and push only; do not create, reopen, or update a pull request.

## Mean/attention result propagation characterization checkpoint

`run_mean_attention()` selects an ordered five-to-seven-child sequence from
independent `include_layernorm` and `include_conv_attention` policies. The
generic recovery runner already returns the corresponding tuple, but the phase
runner and local `_run_mean_attention_layout_pass_cluster()` helper currently
drop it.

The helper has two direct primary calls and two existing callback references.
The first direct call enables LayerNorm and leaves Conv-attention at its default
enabled state; the guarded terminal call disables Conv-attention and leaves
LayerNorm disabled. The recovery contexts type the callback result as `Any` and
do not branch on it.

A strict expected-failure contract requires policy-correct tuples to propagate
through the phase runner and helper. It fixes direct targets
`_layout_pass_set_1_mean_attention_results` and
`_terminal_mean_attention_results`, all four policy combinations, both callback
references, and existing option keywords.

At implementation, return only the tuple already produced by
`run_recovery_invocations()` and retain both direct results. Do not summarize
or consume them, do not run a child twice, and do not change policy selection,
callback references, shared scopes, option guards, adjacent calls, pass order,
dependencies, diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- mean/attention policy and callback contexts, attention recovery, quantized
  suffix, terminal boundary, architecture, and pass-efficiency coverage:
  `329 passed, 1 xfailed in 17.89s`

The sole strict expected failure is the intentionally unimplemented tuple
propagation contract. Implement only the transparent returns and two direct
assignments, rerun focused and branch-changed broad gates sequentially, then
commit and push only; do not create, reopen, or update a pull request.

## Mean/attention result propagation implementation checkpoint

`run_mean_attention()` now returns the policy-selected ordered tuple from
`run_recovery_invocations()`, and the local helper transparently returns it.
The first direct call retains
`_layout_pass_set_1_mean_attention_results`; the guarded terminal call retains
`_terminal_mean_attention_results`.

Both existing recovery contexts still receive the helper as an argument-free
callback and do not branch on its result. Every selected child still runs once
in the same policy-specific order and shared scope. Option guards, keywords,
adjacent calls, diagnostics, dependencies, graph mutation, and TensorFlow
behavior are unchanged; neither retained tuple has a new consumer.

The first implementation gate found four expected stale AST assertions for the
former raw helper/direct expressions. They now require the exact helper Return
and both named assignments.

Implementation validation completed sequentially under `uv`:

- mean/attention policy and callback contexts, attention recovery, quantized
  suffix, terminal boundary, architecture, and pass-efficiency coverage:
  `330 passed in 20.05s`
- branch-changed broad suite plus the same policy/callback coverage:
  `1464 passed in 26.61s`

These are unit, contract, and orchestration checks; this result propagation
does not claim a new model-corpus run.

At resume, audit the next raw terminal helper result after
`_terminal_mean_attention_results`. Keep both mean/attention tuples
observation-only and preserve their policy guards. Commit and push only; do not
create, reopen, or update a pull request.

## BatchMatMul affine-input result characterization checkpoint

`_optimize_batchmatmul_affine_transpose_input_chains()` dispatches one dedicated
owner and returns the fixed one-counter dictionary
`optimized_batchmatmul_affine_transpose_input_chains`. The owner unconditionally
prunes unused tensors after candidate processing, so a zero counter can still
coexist with cleanup-only mutation. Its result must remain observation-only and
must not be used as complete guard evidence.

There are exactly two raw direct production calls. The guarded terminal call
follows `_terminal_mean_attention_results`; the later post-SiNet call follows
SA/PA MirrorPad propagation. Both immediately precede
`_optimize_batchmatmul_reshape_se_nhwc_chains()`.

A strict expected-failure contract selects distinct
`_terminal_batchmatmul_affine_input_stats` and
`_post_sinet_batchmatmul_affine_input_stats` targets. It fixes the two-call
count, model argument, empty keyword sets, policy guard, predecessors, and
shared successor.

At implementation, replace only the two raw expressions with assignments. Do
not add a guard or consumer and do not change the wrapper, owner, one-key
schema, unconditional pruning, adjacent owners, option guard, pass order,
dependencies, diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- affine-input owner fixture, both production boundaries, terminal
  mean/attention, architecture, terminal result contracts, and pass-efficiency
  coverage: `368 passed, 1 xfailed in 18.86s`

The sole strict expected failure is the intentionally unimplemented two-result
retention contract. Implement only those assignments, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## Layout-recovery prefix result characterization checkpoint

`run_layout_recovery_prefix()` selects nineteen ordered children. Its boundary
BatchMatMul/unary, pre-ConCat cleanup, and channel-shuffle/Gather callbacks can
contribute nested heterogeneous results, while the remaining children return
their existing owner statistics. `run_recovery_invocations()` already creates
the complete ordered tuple, but the phase runner and zero-argument lowerer
helper currently discard it.

There is one direct lowerer call in layout pass-set 2. It lies between
`_run_qlinear_mean_concat_recovery_sequence()` and the retained
`_layout_pass_set_2_preadd_mean_attention_results`. The same phase runner is
also selected independently as the first callback of the fifteen-slot
attention prefix. The attention parent still discards its own result in this
unit.

A strict expected-failure contract instruments all nineteen result identities,
including the three callback results. It selects observation-only target
`_layout_pass_set_2_layout_recovery_prefix_results`, freezes the runner/helper
return boundary, shared context, sole zero-argument direct call, QLinear/
pre-add boundaries, nested attention selection, and absence of a consumer.
The tuple cannot safely drive a mutation guard because child counters need not
account for cleanup-only graph changes.

Characterization validation completed sequentially under `uv`:

- layout and attention orchestration, all callback boundaries, direct
  elementwise/Concat/Conv, SPP, pre-Concat, NDHWC Concat, StridedSlice,
  split-mixed, Concat-input, Slice/Logistic/Concat, architecture, and
  pass-efficiency coverage: `414 passed, 1 xfailed in 18.88s`
- branch-changed broad suite: `1597 passed, 1 xfailed in 28.73s`

The sole strict expected failure is the intentionally unimplemented ordered
result propagation contract. At implementation, return the existing tuple
from the phase runner and helper and replace only the direct raw expression
with the selected assignment. Do not consume the result or change a child,
callback, context, pass order, surrounding boundary, attention-parent policy,
dependency, public API, or TensorFlow behavior. Validate sequentially, commit,
and push only; do not create, reopen, or update a pull request.

## Layout-recovery prefix result propagation implementation checkpoint

`run_layout_recovery_prefix()` and its zero-argument lowerer helper now return
the existing nineteen-slot tuple produced by `run_recovery_invocations()`.
The sole direct pass-set-2 call retains it as
`_layout_pass_set_2_layout_recovery_prefix_results` between the unchanged
QLinear recovery and retained pre-add/mean/attention result.

The direct tuple is unconsumed and observation-only. The independent first
callback of the attention prefix now contributes the same nested tuple to its
parent invocation; `run_layout_reshape_attention_recovery_prefix()` still
discards the complete attention result. No new consumer, guard, summary, graph
scan, fingerprint, child invocation, context, order, cleanup timing,
dependency, public API, or TensorFlow import path was added.

Implementation validation completed sequentially under `uv`:

- layout and attention orchestration, QLinear boundary, all layout callback
  boundaries and direct child-owner contracts, architecture, and
  pass-efficiency coverage: `422 passed in 21.02s`
- branch-changed broad suite: `1605 passed in 30.11s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
propagation does not claim a new model-corpus run.

At resume, audit all three direct
`_run_layout_reshape_attention_recovery_prefix()` calls and the complete
fifteen-slot parent result now that its first slot contains the nested
nineteen-slot layout tuple. Preserve each distinct production boundary and
keep incomplete child summaries observation-only. Commit and push only; do not
create, reopen, or update a pull request.

## Direct SA/PA MirrorPad result retention implementation checkpoint

The layout-option call now retains `_layout_opt_sa_pa_mirrorpad_stats`, and the
post-cleanup call retains `_post_cleanup_sa_pa_mirrorpad_stats`. Both preserve
the indexed owner's unchanged complete one-counter mutation dictionary and
remain unconsumed in this unit.

These are assignment-only changes. No consumer or guard was added. The lowerer
wrapper, owner, one-key schema, positive-only unused-tensor pruning and
LayoutState synchronization, live Session LayoutState, layout option guard,
reduced gate-layout boundary, captured CSP and BatchMatMul boundaries,
attention-gate/QDQ owner selection, dependencies, diagnostics, and TensorFlow
behavior remain unchanged. A gate-layout AST helper now accepts an assigned
predecessor while the test verifies its exact target.

Implementation validation completed sequentially under `uv`:

- SA/PA result/schema semantics and indexed owner fixtures, gate-layout and
  attention-recovery orchestration, CSP and BatchMatMul boundaries,
  architecture, and pass-efficiency coverage: `357 passed in 17.70s`
- branch-changed broad suite plus the same SA/PA coverage:
  `1584 passed in 26.18s`

These are unit, contract, owner-fixture, and orchestration checks; this result
retention does not claim a new model-corpus run.

At resume, audit the post-SiNet
`_optimize_transpose_relu_split_all_outputs_to_nhwc_chains()` result immediately
after `_post_sinet_qkv_attention_results`. Preserve its other production forms,
live LayoutState, following ReLU/Split/Conv/Concat recovery, and the already
captured Split/Conv/Concat bridge result. Commit and push only; do not create,
reopen, or update a pull request.

## ReLU/Split all-outputs result characterization checkpoint

The ReLU/Split all-outputs wrapper dispatches one indexed owner and returns the
fixed one-counter dictionary
`optimized_transpose_relu_split_all_outputs_to_nhwc_chains`. The owner counts
only successful complete plan applications and calls unused-tensor pruning only
when at least one rewrite succeeded. The counter therefore covers all owner
mutation paths.

There are exactly two direct calls. The post-SiNet call follows
`_post_sinet_qkv_attention_results`; the terminal call follows late pre-Concat
NHWC recovery. Both receive the live Session LayoutState and immediately
precede the same ReLU/Split/Conv/Concat recovery owner.

A strict expected-failure contract selects
`_post_sinet_relu_split_all_outputs_stats` and
`_terminal_relu_split_all_outputs_stats`. It fixes the two-call count, model
and LayoutState arguments, lowerer wrapper, one-key schema, positive-only
pruning, both predecessors, shared successor, QKV boundary, and related indexed
owner behavior.

At implementation, replace only the two raw direct expressions with
assignments. Do not add a consumer or guard, and do not change the wrapper,
owner, schema, pruning, graph-index behavior, live LayoutState, following
Split/Conv/Concat calls, surrounding sequences, dependencies, diagnostics, or
TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- ReLU/Split result/schema semantics and indexed owner fixtures, QKV and
  Split/Conv/Concat boundaries, architecture, and pass-efficiency coverage:
  `386 passed, 1 xfailed in 17.76s`

The sole strict expected failure is the intentionally unimplemented two-result
retention contract. Implement only those assignments, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## ReLU/Split all-outputs result retention implementation checkpoint

The post-SiNet call now retains
`_post_sinet_relu_split_all_outputs_stats`, and the terminal call retains
`_terminal_relu_split_all_outputs_stats`. Both preserve the indexed owner's
unchanged complete one-counter mutation dictionary and remain unconsumed in
this unit.

These are assignment-only changes. No consumer or guard was added. The lowerer
wrapper, owner, one-key schema, successful-plan counting, positive-only
unused-tensor pruning, graph-index behavior, live Session LayoutState,
post-SiNet QKV and terminal pre-Concat predecessors, shared
ReLU/Split/Conv/Concat successor, dependencies, diagnostics, and TensorFlow
behavior remain unchanged.

Implementation validation completed sequentially under `uv`:

- ReLU/Split result/schema semantics and indexed owner fixtures, QKV and
  Split/Conv/Concat boundaries, architecture, and pass-efficiency coverage:
  `387 passed in 18.81s`
- branch-changed broad suite plus the same ReLU/Split coverage:
  `1575 passed in 27.30s`

These are unit, contract, owner-fixture, and orchestration checks; this result
retention does not claim a new model-corpus run.

At resume, audit both adjacent raw
`_optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains()`
results. Preserve the two newly retained ReLU/Split all-output dictionaries,
the following captured Split/Conv/Concat bridge result, live LayoutState,
surrounding pass order, and observation-only evidence rules. Commit and push
only; do not create, reopen, or update a pull request.

## ReLU/Split/Conv/Concat result characterization checkpoint

The adjacent ReLU/Split/Conv/ReLU/Concat post-transpose wrapper dispatches the
second indexed owner in `split_all_outputs_layout.py` and returns the fixed
one-counter dictionary
`optimized_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains`.
The owner counts only successful complete plan applications and calls
unused-tensor pruning only when at least one rewrite succeeded. Its counter
therefore covers all owner mutation paths.

There are exactly two raw direct calls. Each follows one of the newly retained
ReLU/Split all-output dictionaries and receives the live Session LayoutState.
The post-SiNet call precedes the retained Split/Conv/Concat bridge result; the
terminal call precedes mixed pre-Concat adapter recovery.

A strict expected-failure contract selects
`_post_sinet_relu_split_conv_concat_stats` and
`_terminal_relu_split_conv_concat_stats`. It fixes the two-call count, model
and LayoutState arguments, lowerer wrapper, one-key schema, positive-only
pruning, both captured predecessors, both successors, and absence of result
consumers.

At implementation, replace only the two raw direct expressions with
assignments. Do not add a consumer or guard, and do not change the wrapper,
owner, schema, pruning, graph-index behavior, live LayoutState, surrounding
calls, dependencies, diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- ReLU/Split/Conv/Concat result/schema semantics, the preceding ReLU/Split
  owner fixtures, QKV and Split/Conv/Concat boundaries, architecture, and
  pass-efficiency coverage: `388 passed, 1 xfailed in 18.66s`
- concrete ReLU/Split/Conv/Concat rewrite fixture:
  `1 passed, 740 deselected in 0.61s`

The sole strict expected failure is the intentionally unimplemented two-result
retention contract. Implement only those assignments, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## ReLU/Split/Conv/Concat result retention implementation checkpoint

The post-SiNet call now retains
`_post_sinet_relu_split_conv_concat_stats`, and the terminal call retains
`_terminal_relu_split_conv_concat_stats`. Both preserve the indexed owner's
unchanged complete one-counter mutation dictionary and remain unconsumed in
this unit.

These are assignment-only changes. No consumer or guard was added. The lowerer
wrapper, owner, one-key schema, successful-plan counting, positive-only
unused-tensor pruning, graph-index behavior, live Session LayoutState,
captured ReLU/Split all-output predecessors, Split/Conv/Concat bridge and mixed
pre-Concat successors, dependencies, diagnostics, and TensorFlow behavior
remain unchanged.

Implementation validation completed sequentially under `uv`:

- ReLU/Split/Conv/Concat result/schema semantics, preceding ReLU/Split owner
  fixtures, QKV and Split/Conv/Concat boundaries, architecture, and
  pass-efficiency coverage: `389 passed in 18.39s`
- concrete ReLU/Split/Conv/Concat rewrite fixture:
  `1 passed, 740 deselected in 0.60s`
- branch-changed broad suite plus the same ReLU/Split/Conv/Concat coverage:
  `1577 passed in 26.79s`

These are unit, contract, owner-fixture, and orchestration checks; this result
retention does not claim a new model-corpus run.

At resume, audit the terminal
`_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains()`
result and every other production form of that owner. Preserve both newly
retained ReLU/Split/Conv/Concat dictionaries, the post-SiNet bridge sequence,
live LayoutState, pass order, and observation-only evidence rules. Commit and
push only; do not create, reopen, or update a pull request.

## Split/mixed pre-Concat result characterization checkpoint

The Split/mixed pre-Concat adapter wrapper dispatches the indexed owner in
`split_mixed_concat_layout.py` and returns the fixed one-counter dictionary
`optimized_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains`.
The owner counts only successful complete plan applications and calls
unused-tensor pruning only after a positive rewrite. Its counter therefore
covers all owner mutation paths.

There are exactly two direct wrapper calls. The first is inside layout recovery
pass-set 2 between StridedSlice pre-Concat recovery and Concat input-adapter
recovery. The terminal call follows `_terminal_relu_split_conv_concat_stats`
and precedes the same input-adapter owner. Both receive the live Session
LayoutState. Layout-recovery orchestration directly selects the public owner as
a third, distinct form.

A strict expected-failure contract selects
`_layout_opt_split_mixed_pre_concat_stats` and
`_terminal_split_mixed_pre_concat_stats` for only the direct wrapper calls. It
fixes the wrapper, one-key schema, positive-only pruning, exact two-call count,
model and LayoutState arguments, option guard, four boundaries, independent
orchestration selection, and absence of result consumers.

At implementation, replace only the two raw direct expressions with
assignments. Do not add a consumer or guard, and do not change the wrapper,
owner, schema, pruning, graph-index behavior, live LayoutState, orchestration
selection, surrounding calls, dependencies, diagnostics, or TensorFlow
behavior.

Characterization validation completed sequentially under `uv`:

- Split/mixed pre-Concat schema and indexed-owner fixtures, layout-recovery
  orchestration, both direct boundaries, adjacent StridedSlice and Concat
  input-adapter owners, architecture, and pass-efficiency coverage:
  `407 passed, 1 xfailed in 18.76s`

The sole strict expected failure is the intentionally unimplemented two-result
retention contract. Implement only those assignments, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## Split/mixed pre-Concat result retention implementation checkpoint

The layout-option call now retains
`_layout_opt_split_mixed_pre_concat_stats`, and the terminal call retains
`_terminal_split_mixed_pre_concat_stats`. Both preserve the indexed owner's
unchanged complete one-counter mutation dictionary and remain unconsumed in
this unit.

These are assignment-only changes. No consumer or guard was added. The lowerer
wrapper, owner, one-key schema, successful-plan counting, positive-only
unused-tensor pruning, graph-index behavior, live Session LayoutState,
layout-option guard, StridedSlice and retained ReLU/Split/Conv/Concat
predecessors, shared Concat input-adapter successor, independent orchestration
selection, dependencies, diagnostics, and TensorFlow behavior remain
unchanged.

Implementation validation completed sequentially under `uv`:

- Split/mixed pre-Concat schema and indexed-owner fixtures, layout-recovery
  orchestration, both direct boundaries, adjacent StridedSlice and Concat
  input-adapter owners, architecture, and pass-efficiency coverage:
  `408 passed in 18.71s`
- branch-changed broad suite plus the same Split/mixed pre-Concat coverage:
  `1663 passed in 28.60s`

These are unit, contract, owner-fixture, and orchestration checks; this result
retention does not claim a new model-corpus run.

At resume, audit both direct
`_optimize_transpose_input_chains_pre_concat_to_single_post_adapter()` results
and its independent layout-recovery orchestration selection. Preserve both
newly retained Split/mixed pre-Concat dictionaries, option policy, live
LayoutState, following layout owners, and observation-only evidence rules.
Commit and push only; do not create, reopen, or update a pull request.

## Concat input-adapter result characterization checkpoint

The Concat input-adapter wrapper dispatches the indexed owner in
`concat_input_adapter_layout.py` and returns the fixed one-counter dictionary
`optimized_transpose_input_chains_pre_concat_to_single_post_adapter`. The owner
unconditionally prunes unused tensors after candidate processing, so its
rewrite counter is incomplete evidence for cleanup-only mutation and must
remain observation-only.

There are exactly two direct wrapper calls. The first is inside layout recovery
pass-set 2 after `_layout_opt_split_mixed_pre_concat_stats` and before the
Slice/Logistic/Concat tail owner. The terminal call follows
`_terminal_split_mixed_pre_concat_stats` and precedes Concat/unary/Conv cleanup.
Both receive the live Session LayoutState. Layout-recovery orchestration
directly selects the public owner, while safe-transpose-reduction selects the
private wrapper as two additional, distinct forms.

A strict expected-failure contract selects
`_layout_opt_concat_input_adapter_stats` and
`_terminal_concat_input_adapter_stats` for only the direct wrapper calls. It
fixes the wrapper, one-key schema, unconditional pruning, exact two-call count,
model and LayoutState arguments, option guard, four boundaries, both independent
selections, and absence of result consumers.

At implementation, replace only the two raw direct expressions with
assignments. Do not add a consumer or guard, and do not change the wrapper,
owner, schema, unconditional pruning, graph-index behavior, live LayoutState,
orchestration or safe-reduction selections, surrounding calls, dependencies,
diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- Concat input-adapter schema and indexed-owner fixtures, unconditional cleanup,
  layout-recovery and safe-reduction selections, both direct boundaries,
  adjacent Split/mixed and Slice/Logistic/Concat owners, terminal
  Concat/unary/Conv cleanup, architecture, and pass-efficiency coverage:
  `405 passed, 1 xfailed in 18.68s`

The sole strict expected failure is the intentionally unimplemented two-result
retention contract. Implement only those assignments, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## Concat input-adapter result retention implementation checkpoint

The layout-option call now retains
`_layout_opt_concat_input_adapter_stats`, and the terminal call retains
`_terminal_concat_input_adapter_stats`. Both preserve the indexed owner's
unchanged one-counter dictionary and remain unconsumed because a zero counter
does not exclude cleanup-only pruning.

These are assignment-only changes. No consumer or guard was added. The lowerer
wrapper, owner, one-key schema, unconditional unused-tensor pruning,
graph-index behavior, live Session LayoutState, layout-option guard, retained
Split/mixed pre-Concat predecessors, Slice/Logistic/Concat tail and
Concat/unary/Conv successors, layout-recovery and safe-reduction selections,
dependencies, diagnostics, and TensorFlow behavior remain unchanged.

Implementation validation completed sequentially under `uv`:

- Concat input-adapter schema and indexed-owner fixtures, unconditional cleanup,
  layout-recovery and safe-reduction selections, both direct boundaries,
  adjacent Split/mixed and Slice/Logistic/Concat owners, terminal
  Concat/unary/Conv cleanup, architecture, and pass-efficiency coverage:
  `406 passed in 18.66s`
- branch-changed broad suite plus the same Concat input-adapter coverage:
  `1663 passed in 27.31s`

These are unit, contract, owner-fixture, and orchestration checks; this result
retention does not claim a new model-corpus run.

At resume, audit the guarded
`_optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains()` result
and every other production form of that owner. Preserve the newly retained
Concat input-adapter dictionaries, layout option policy, live LayoutState,
channel-shuffle successor, independent orchestration selection, and
observation-only evidence rules. Commit and push only; do not create, reopen,
or update a pull request.

## Slice/Logistic/Concat tail result characterization checkpoint

The Slice/Logistic/Concat/Reshape-tail wrapper dispatches the indexed owner in
`slice_logistic_concat_reshape_tail_layout.py` and returns the fixed one-counter
dictionary
`optimized_transpose_slice_logistic_concat_reshape_tail_nhwc_chains`. The owner
unconditionally prunes unused tensors after candidate processing, so its
rewrite counter is incomplete evidence for cleanup-only mutation and must
remain observation-only.

There is exactly one direct wrapper call. It is inside layout recovery pass-set
2 after `_layout_opt_concat_input_adapter_stats` and before
`_layout_opt_channel_shuffle_gather_results`, whose policy keeps post-gather
cleanup enabled. The call receives the live Session LayoutState.
Layout-recovery orchestration directly selects the public owner as a second,
distinct form.

A strict expected-failure contract selects
`_layout_opt_slice_logistic_concat_tail_stats`. It fixes the wrapper, one-key
schema, unconditional pruning, exact one-call count, model and LayoutState
arguments, option guard, both captured boundaries, channel-shuffle policy,
independent orchestration selection, and absence of result consumers.

At implementation, replace only the raw direct expression with an assignment.
Do not add a consumer or guard, and do not change the wrapper, owner, schema,
unconditional pruning, graph-index behavior, live LayoutState, orchestration
selection, surrounding calls, dependencies, diagnostics, or TensorFlow
behavior.

Characterization validation completed sequentially under `uv`:

- Slice/Logistic/Concat tail schema and indexed-owner fixtures, unconditional
  cleanup, layout-recovery selection, guarded direct boundary,
  channel-shuffle/gather policy, architecture, and pass-efficiency coverage:
  `362 passed, 1 xfailed in 19.11s`

The sole strict expected failure is the intentionally unimplemented
result-retention contract. Implement only that assignment, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## Slice/Logistic/Concat tail result retention implementation checkpoint

The guarded direct call now retains
`_layout_opt_slice_logistic_concat_tail_stats`. It preserves the indexed
owner's unchanged one-counter dictionary and remains unconsumed because a zero
counter does not exclude cleanup-only pruning.

This is an assignment-only change. No consumer or guard was added. The lowerer
wrapper, owner, one-key schema, unconditional unused-tensor pruning,
graph-index behavior, live Session LayoutState, layout-option guard, retained
Concat input-adapter predecessor, channel-shuffle/gather successor and policy,
independent orchestration selection, dependencies, diagnostics, and TensorFlow
behavior remain unchanged.

Implementation validation completed sequentially under `uv`:

- Slice/Logistic/Concat tail schema and indexed-owner fixtures, unconditional
  cleanup, layout-recovery selection, guarded direct boundary,
  channel-shuffle/gather policy, architecture, and pass-efficiency coverage:
  `363 passed in 18.58s`
- branch-changed broad suite plus the same Slice/Logistic/Concat coverage:
  `1598 passed in 27.99s`

These are unit, contract, owner-fixture, and orchestration checks; this result
retention does not claim a new model-corpus run.

At resume, audit the terminal `run_concat_unary_conv_layout_cleanup()` result.
Preserve `_terminal_concat_input_adapter_stats`, the following shape-extract
owner, live LayoutState and diagnostics, pass order, and observation-only
evidence rules. Commit and push only; do not create, reopen, or update a pull
request.

## Terminal Concat/unary/Conv result characterization checkpoint

`run_concat_unary_conv_layout_cleanup()` executes one transactional layout
PassSpec and returns the fixed one-counter dictionary
`optimized_transpose_concat_unary_fanout_conv_nhwc_chains`. A positive owner
rewrite prunes unused tensors and synchronizes LayoutState; precondition misses
avoid graph-state construction and return zero. Diagnostics record either
outcome, so the result remains observation-only.

There is exactly one production call. It follows
`_terminal_concat_input_adapter_stats`, precedes the raw shape-extract owner,
and receives the live Session LayoutState and diagnostics collection.

A strict expected-failure contract selects
`_terminal_concat_unary_conv_stats`. It fixes the one-key result schema,
transactional PassSpec, preflight/default details, positive-only prune and
layout sync, exact one-call count, model/LayoutState/diagnostics arguments,
both boundaries, and absence of result consumers.

At implementation, replace only the raw expression with an assignment. Do not
add a consumer or guard, and do not change the runner, owner, schema,
transactional/preflight behavior, cleanup, layout sync, diagnostics,
surrounding calls, dependencies, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- Concat/unary/Conv runner and owner fixtures, transactional/preflight schema,
  diagnostics, positive-only cleanup and layout sync, terminal result boundary,
  Concat input-adapter and shape-extract neighbors, architecture,
  pass-efficiency, and terminal-layout coverage:
  `400 passed, 1 xfailed in 18.79s`

The sole strict expected failure is the intentionally unimplemented
result-retention contract. Implement only that assignment, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## Terminal Concat/unary/Conv result retention implementation checkpoint

The production call now retains `_terminal_concat_unary_conv_stats`. It
preserves the runner's unchanged one-counter dictionary and remains unconsumed;
diagnostics continue to record changed and skipped outcomes independently of
the retained value.

This is an assignment-only change. No consumer or guard was added. The runner,
owner, one-key schema, transactional PassSpec, preflight/default details,
positive-only unused-tensor pruning and LayoutState synchronization, live
Session LayoutState and diagnostics, retained Concat input-adapter predecessor,
shape-extract successor, dependencies, and TensorFlow behavior remain
unchanged.

Implementation validation completed sequentially under `uv`:

- Concat/unary/Conv runner and owner fixtures, transactional/preflight schema,
  diagnostics, positive-only cleanup and layout sync, terminal result boundary,
  Concat input-adapter and shape-extract neighbors, architecture,
  pass-efficiency, and terminal-layout coverage: `401 passed in 20.06s`
- branch-changed broad suite plus the same Concat/unary/Conv coverage:
  `1585 passed in 28.42s`

These are unit, contract, owner-fixture, and orchestration checks; this result
retention does not claim a new model-corpus run.

At resume, audit the raw terminal
`_optimize_transpose_shape_extract_nhwc_to_nchw_chains()` result and every other
production form of that owner. Preserve `_terminal_concat_unary_conv_stats`,
the following terminal layout-option guard, existing retained shape-extract
results, pass order, and observation-only evidence rules. Commit and push only;
do not create, reopen, or update a pull request.

## Remaining shape-extract result characterization checkpoint

The shape-extract compatibility wrapper returns the dedicated owner's fixed
one-counter dictionary
`optimized_transpose_shape_extract_nhwc_to_nchw_chains`. The owner prunes
unused tensors only after a positive rewrite, and its idempotence plus unsafe
boundary fixtures establish the counter as complete mutation evidence.

There are three production calls. The middle pre-QKV form already retains
`_late_pre_qkv_shape_extract_stats`. The terminal form after
`_terminal_concat_unary_conv_stats` and the absolute-end form after pre-Concat
cleanup remain raw.

A strict expected-failure contract selects `_terminal_shape_extract_stats` and
`_late_pre_layout_cluster_shape_extract_stats` for those two calls while
preserving the existing target. It fixes the exact three-call count, one-key
schema, positive-only pruning, model-only arguments, terminal option-guard
boundary, late-SPP/QKV boundary, pre-Concat/late-layout-cluster boundary, and
absence of result consumers.

At implementation, replace only the first and third raw expressions with
assignments. Do not rename or consume the existing pre-QKV target, and do not
change the wrapper, owner, schema, pruning, surrounding calls, dependencies,
diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- shape-extract owner/schema and unsafe-boundary fixtures, exact three-form
  retention contract, terminal Concat/unary/Conv and fanout guard, late-SPP and
  QKV, absolute-end late-layout cluster, architecture, pass-efficiency, and
  terminal-layout coverage: `411 passed, 1 xfailed in 20.23s`

The sole strict expected failure is the intentionally unimplemented two-result
retention contract. Implement only those assignments, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## Remaining shape-extract result retention implementation checkpoint

The terminal call now retains `_terminal_shape_extract_stats`, the existing
pre-QKV call keeps `_late_pre_qkv_shape_extract_stats`, and the absolute-end
call now retains `_late_pre_layout_cluster_shape_extract_stats`. All three
complete one-counter dictionaries remain unconsumed.

These are assignment-only changes. No consumer or guard was added. The
compatibility wrapper, owner, one-key schema, positive-only unused-tensor
pruning, model-only arguments, terminal Concat/unary/Conv and fanout-guard
boundaries, late-SPP/QKV boundaries, pre-Concat and late-layout-cluster
boundaries, dependencies, diagnostics, and TensorFlow behavior remain
unchanged. Three stale AST contracts now verify the exact assigned targets
instead of requiring raw expressions.

Implementation validation completed sequentially under `uv`:

- shape-extract owner/schema and unsafe-boundary fixtures, exact three-form
  retention contract, terminal Concat/unary/Conv and fanout guard, late-SPP and
  QKV, absolute-end late-layout cluster, architecture, pass-efficiency, and
  terminal-layout coverage: `412 passed in 23.37s`
- branch-changed broad suite plus the same three-form shape-extract coverage:
  `1556 passed in 27.91s`

These are unit, contract, owner-fixture, and orchestration checks; this result
retention does not claim a new model-corpus run.

At resume, audit the guarded
`_optimize_convpool_output_transpose_nhwc_passthrough_chains()` result and every
other production form of that owner. Preserve the retained terminal
shape-extract/fanout results, terminal Singleton/MaxPool sequence, no-layout
fallback branch, pass order, and observation-only evidence rules. Commit and
push only; do not create, reopen, or update a pull request.

## Guarded Conv/Pool output result characterization checkpoint

The Conv/Pool output-transpose compatibility wrapper returns the fixed
one-counter dictionary
`optimized_convpool_output_transpose_nhwc_passthrough_chains`. Its owner prunes
unused tensors unconditionally after candidate processing, so a zero rewrite
counter does not exclude cleanup-only mutation and the result must remain
observation-only.

There is exactly one production call. It is the body of the terminal
`optimize_layout_transpose_chains` guard after the Singleton/MaxPool/Reshape
sequence. The `elif` branch preserves safe-transpose-reduction and strict
Mul/Add constant bridge cleanup; both paths rejoin before the dequantized
HardSigmoid bridge owner.

A strict expected-failure contract selects
`_terminal_convpool_output_passthrough_stats`. It fixes the wrapper, one-key
schema, unconditional pruning, exact one-call count, model-only argument,
option guard, predecessor and successor, no-layout fallback order, and absence
of result consumers.

At implementation, replace only the guarded raw expression with an assignment.
Do not add a consumer or guard, and do not change the wrapper, owner, schema,
unconditional pruning, no-layout fallback, surrounding calls, dependencies,
diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- Conv/Pool output wrapper/schema and owner fixtures, unconditional cleanup,
  terminal Singleton/MaxPool and retained shape/fanout boundaries, no-layout
  fallback, quantized HardSigmoid successor, architecture, and pass-efficiency
  coverage: `397 passed, 1 xfailed in 18.54s`

The sole strict expected failure is the intentionally unimplemented
result-retention contract. Implement only that assignment, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## Guarded Conv/Pool output result retention implementation checkpoint

The guarded call now retains
`_terminal_convpool_output_passthrough_stats`. It preserves the owner's
unchanged one-counter dictionary and remains unconsumed because a zero counter
does not exclude cleanup-only pruning.

This is an assignment-only change. No consumer or guard was added. The wrapper,
owner, one-key schema, unconditional unused-tensor pruning, model-only
argument, layout option guard, terminal Singleton/MaxPool predecessor,
no-layout safe-reduction/Mul-Add fallback, HardSigmoid successor, dependencies,
diagnostics, and TensorFlow behavior remain unchanged. Two stale boundary
tests now require the exact assigned target instead of a raw expression.

Implementation validation completed sequentially under `uv`:

- Conv/Pool output wrapper/schema and owner fixtures, unconditional cleanup,
  terminal Singleton/MaxPool and retained shape/fanout boundaries, no-layout
  fallback, quantized HardSigmoid successor, architecture, and pass-efficiency
  coverage: `398 passed in 20.60s`
- branch-changed broad suite plus the same guarded Conv/Pool coverage:
  `1656 passed in 28.18s`

These are unit, contract, owner-fixture, and orchestration checks; this result
retention does not claim a new model-corpus run.

At resume, audit the raw
`_optimize_transpose_dequant_hardsigmoid_quantize_bridges()` result and every
other production form of that owner. Preserve the guarded Conv/Pool/no-layout
branch, following late dequant/unary/fanout cluster, pass order, and
observation-only evidence rules. Commit and push only; do not create, reopen,
or update a pull request.

## Dequantized HardSigmoid result characterization checkpoint

The dequantized HardSigmoid bridge wrapper returns the indexed owner's fixed
one-counter dictionary
`removed_transpose_dequant_hardsigmoid_quantize_bridges`. With no Transpose the
owner returns zero early; otherwise it prunes unused tensors after scanning
even when no bridge was removed. The counter is therefore incomplete evidence
for cleanup-only mutation and must remain observation-only.

There are exactly three direct wrapper calls: before terminal SiNet
pre-add/resize recovery, after post-SiNet mixed-attention cleanup, and after the
terminal Conv/Pool/no-layout branch. Attention-gate/QDQ orchestration directly
selects the public owner as a fourth, distinct form.

A strict expected-failure contract selects
`_terminal_dequant_hardsigmoid_bridge_stats`,
`_post_sinet_dequant_hardsigmoid_bridge_stats`, and
`_late_dequant_hardsigmoid_bridge_stats`. It fixes the wrapper, one-key schema,
early return and cleanup semantics, exact three-call count, model-only
arguments, six direct boundaries, independent orchestration selection, and
absence of result consumers.

At implementation, replace only the three direct raw expressions with
assignments. Do not add a consumer or guard, and do not change the wrapper,
owner, schema, cleanup, graph-index behavior, orchestration selection,
surrounding calls, dependencies, diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- dequantized HardSigmoid wrapper/schema and indexed-owner fixtures, early
  return and cleanup semantics, attention-recovery selection, three direct
  boundaries, adjacent SiNet/ConvPool/late-dequant orchestration, architecture,
  and pass-efficiency coverage: `404 passed, 1 xfailed in 18.99s`

The sole strict expected failure is the intentionally unimplemented
three-result retention contract. Implement only those assignments, rerun
focused and branch-changed broad gates sequentially, then commit and push only;
do not create, reopen, or update a pull request.

## Dequantized HardSigmoid result retention implementation checkpoint

The three direct calls now retain their unchanged one-counter dictionaries as
`_terminal_dequant_hardsigmoid_bridge_stats`,
`_post_sinet_dequant_hardsigmoid_bridge_stats`, and
`_late_dequant_hardsigmoid_bridge_stats`. All remain unconsumed because a zero
counter does not exclude cleanup-only pruning when Transpose operators exist.

These are assignment-only changes. No consumer or guard was added. The wrapper,
indexed owner, one-key schema, no-Transpose early return, graph-index policy,
cleanup behavior, attention-recovery selection, terminal SiNet, post-SiNet
cost-volume, and ConvPool/late-dequant boundaries, dependencies, diagnostics,
and TensorFlow behavior remain unchanged. Four stale AST boundary contracts
now require the new assigned targets.

Implementation validation completed sequentially under `uv`:

- dequantized HardSigmoid wrapper/schema and indexed-owner fixtures, early
  return and cleanup semantics, attention-recovery selection, three direct
  boundaries, adjacent SiNet/ConvPool/late-dequant orchestration, architecture,
  and pass-efficiency coverage: `405 passed in 21.05s`
- branch-changed broad suite plus the same three-form HardSigmoid coverage:
  `1654 passed in 29.28s`

These are unit, contract, owner-fixture, and orchestration checks; this result
retention does not claim a new model-corpus run.

At resume, audit the raw terminal
`_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains()`
result and every other production form of that owner. Preserve the new terminal
dequant-HardSigmoid result, existing late HardSwish-SE evidence, SiNet recovery,
pass order, and observation-only evidence rules. Commit and push only; do not
create, reopen, or update a pull request.

## Terminal-SiNet HardSwish-SE result characterization checkpoint

The HardSwish-SE layout owner has exactly two production forms. The later
absolute-terminal form already captures a starting tensor count and merges the
owner's unchanged one-counter dictionary with an exact net
`pruned_unused_tensors` count. The earlier call immediately after terminal
SiNet recovery still discards its raw result. Because the owner invokes
unused-tensor pruning unconditionally, zero rewrites can coexist with cleanup
mutation.

A strict expected-failure contract now freezes the wrapper and one-key schema,
unconditional cleanup, exact two-call count, existing prune-aware late form,
model-only arguments, and terminal-SiNet/dequant-HardSigmoid boundaries. It
selects `_terminal_sinet_hardswish_se_stats` for assignment-only retention of
the earlier raw dictionary and requires the result to remain unconsumed.

The focused wrapper/owner, both production forms, adjacent SiNet/dequant and
late-hard-activation orchestration, architecture, and pass-efficiency gate is
`323 passed, 1 xfailed in 17.83s`. Ruff, Python bytecode compilation, and
whitespace validation pass. The sole strict xfail is the selected earlier
result assignment; production is unchanged at this checkpoint.

At implementation, replace only the discarded earlier call with the selected
assignment. Update only the boundary contracts made stale by that assignment,
then run focused and branch-changed broad gates sequentially. Do not use the
raw zero counter as a stability guard. Commit and push only; do not create,
reopen, or update a pull request.

## Terminal-SiNet HardSwish-SE result retention implementation checkpoint

The earlier production call now assigns its unchanged one-counter dictionary
to `_terminal_sinet_hardswish_se_stats`. The value remains unconsumed and is
not a stability guard: the owner can still prune unused tensors while reporting
zero rewrites. The later `_terminal_hardswish_se_stats` form continues to carry
its complete starting-count and `pruned_unused_tensors` evidence.

This is an assignment-only production change. The compatibility wrapper,
owner implementation and schema, exact two-call count, model-only arguments,
terminal SiNet/dequant-HardSigmoid boundaries, pass order, dependencies,
diagnostics, public behavior, and TensorFlow-free direct path are unchanged.
Two stale outer-boundary contracts now require the new assigned target.

Implementation validation completed sequentially under `uv`:

- wrapper/owner, both production forms, HardSwish-SE fixtures, terminal SiNet,
  dequant-HardSigmoid, late-hard-activation, architecture, and pass-efficiency
  coverage: `324 passed in 20.03s`
- branch-changed broad suite: `1539 passed in 27.43s`

These are unit, owner-fixture, structural, orchestration, and broad contract
checks; this observation-only retention does not claim a new model-corpus run.

At resume, characterize result propagation for the immediately preceding
`_run_terminal_clamp_unary_relu_pass_cluster()`. Its three-owner orchestration
currently returns `None`; freeze raw child results, cleanup semantics, shared
state scope, and the guarded singleton/terminal-SiNet boundaries before
changing either runner layer. Commit and push only; do not create, reopen, or
update a pull request.

## Terminal clamp/unary/ReLU result characterization checkpoint

The cluster runs three child owners in fixed order through one shared
`ModelIRPassStateScope`: clamp canonicalization, unary Transpose passthrough,
and Maximum-zero ReLU canonicalization. The shared recovery executor already
returns their ordered dictionaries, but the orchestration runner and lowerer
delegate both return `None`, and the sole production call discards the result.

Empty-model fixtures freeze the three one-key zero schemas. Structural checks
also freeze cleanup semantics: the clamp and unary internal owners invoke
unused-tensor pruning unconditionally when their transactional callbacks run,
while Maximum-zero ReLU prunes only when its rewrite counter is positive. The
raw result tuple is therefore observation-only and cannot safely serve as a
general no-mutation guard.

A strict expected-failure contract requires both runner layers to propagate
`Tuple[Dict[str, int], ...]`, retains the sole production result as
`_terminal_clamp_unary_relu_results`, and leaves it unconsumed between the
guarded singleton/reshape and terminal-SiNet boundaries. Shared scope, pass
IDs, ordering, arguments, diagnostics, and dependencies remain fixed.

The focused child-owner, schema, cleanup, propagation, shared-scope, adjacent
orchestration, architecture, and pass-efficiency gate is
`370 passed, 1 xfailed in 17.98s`. Ruff, Python bytecode compilation, and
whitespace validation pass. The sole strict xfail covers the two missing
returns and production assignment; production is unchanged at this checkpoint.

At implementation, add only the two returns and production assignment, then
update the structural expectations made stale by expression-to-return or
expression-to-assignment changes. Validate sequentially, commit, and push only;
do not create, reopen, or update a pull request.

## Terminal clamp/unary/ReLU result propagation implementation checkpoint

The orchestration runner and lowerer delegate now return the shared executor's
unchanged ordered three-dictionary tuple. The sole production call retains it
as `_terminal_clamp_unary_relu_results`. The tuple remains unconsumed and is not
a mutation guard because clamp and unary callbacks can perform cleanup not
fully represented by a zero rewrite counter.

These are return- and assignment-only changes. Child selection, stable IDs,
order, the single shared `ModelIRPassStateScope`, ModelIR/layout/diagnostics
routing, transactional behavior, invocation count, guarded singleton/reshape
predecessor, terminal-SiNet successor, dependencies, public behavior, and the
TensorFlow-free direct path remain unchanged.

Implementation validation completed sequentially under `uv`:

- child schemas/cleanup, ordered propagation, shared scope, adjacent
  singleton/SiNet/HardSwish-SE orchestration, architecture, and pass-efficiency
  coverage: `371 passed in 19.72s`
- branch-changed broad suite: `1541 passed in 27.26s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The first focused run exposed four stale AST expectations, including two edits
that had matched repeated test fragments; those edits were corrected with
function-specific contexts before the clean focused and broad results above.
No production failure was observed. These checks do not claim a new
model-corpus run.

At resume, audit the two production calls to
`_run_terminal_slice_concat_layout_recovery_sequence()`. Its orchestration
currently returns `None` at both runner layers. Freeze every child result and
cleanup semantic, both distinct boundary pairs, shared state scope, and exact
two-call multiplicity before choosing result targets. Commit and push only; do
not create, reopen, or update a pull request.

## Terminal slice/concat result characterization checkpoint

This recovery has fourteen ordered slots and exactly two production calls. The
first slot is itself the two-dictionary channel-slice/pad-Mul result tuple. The
remaining slots return fixed rewrite dictionaries, a two-key probable-NHWC
sanitizer dictionary, and a final layout dictionary with four mutation
counters plus non-mutating `iterations`. The shared executor already returns
the complete nested tuple, but the orchestration runner and lowerer delegate
return `None`, and both production calls discard it.

Empty-model fixtures now freeze every child schema. Structural checks also
freeze unconditional unused-tensor cleanup in the pre-Add and layout owners,
so zero counters cannot be used as a general no-mutation guard. A strict
expected-failure contract requires both runner layers to propagate
`Tuple[Any, ...]`, retains the two production results as
`_terminal_slice_concat_recovery_results` and
`_final_slice_concat_recovery_results`, and requires both to remain unconsumed.

The focused fourteen-slot schema/cleanup, channel-slice child, boundary,
terminal-layout, architecture, and pass-efficiency gate is
`465 passed, 1 xfailed in 20.37s`. Ruff, Python bytecode compilation, and
whitespace validation pass. The sole strict xfail is the missing two-layer
propagation and two production assignments; production is unchanged here.

The two distinct predecessor/successor pairs, fourteen stable IDs, callback
identity, ModelIR/layout/diagnostics routing, invocation multiplicity, and pass
order remain fixed. At implementation, add only two returns and two production
assignments, update stale structural expectations, validate sequentially,
commit, and push only. Do not create, reopen, or update a pull request.

## Terminal slice/concat result propagation implementation checkpoint

The orchestration runner and lowerer delegate now return the shared executor's
unchanged fourteen-slot nested tuple. The two production calls retain it as
`_terminal_slice_concat_recovery_results` and
`_final_slice_concat_recovery_results`. Both remain unconsumed: cleanup-only
mutation and the final non-mutating `iterations` metric are observation, not a
control condition.

These are return- and assignment-only changes. Every child schema, stable ID,
callback identity, ModelIR/layout/diagnostics argument, pass order, exact
two-call multiplicity, and both predecessor/successor pairs are unchanged. No
summary, mutation guard, dependency, public behavior change, or TensorFlow
import path was added.

Implementation validation completed sequentially under `uv`:

- fourteen-slot schemas/cleanup, ordered propagation, channel-slice child,
  both boundary pairs, terminal-layout, architecture, and pass-efficiency
  coverage: `466 passed in 20.96s`
- branch-changed broad suite: `1543 passed in 27.94s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, audit the raw
`_optimize_transpose_slice_prepost_nhwc_passthrough_chains()` call immediately
after `_final_slice_concat_recovery_results`. It has one production form;
freeze its owner schema, cleanup behavior, final-slice/pre-Concat boundaries,
and result consumption policy before changing production. Commit and push
only; do not create, reopen, or update a pull request.

## Final Slice pre/post result characterization checkpoint

The Slice pre/post NHWC passthrough owner has one production call immediately
after `_final_slice_concat_recovery_results`. It returns one rewrite counter
and prunes unused tensors only when that counter is positive, so the dictionary
fully describes mutation performed by this owner.

A strict expected-failure contract freezes the compatibility wrapper, one-key
schema, positive-only cleanup, exact one-call count, model-only argument, and
final slice/concat to pre-Concat boundaries. It selects
`_final_slice_prepost_passthrough_stats` for assignment-only retention and
requires the value to remain unconsumed.

The focused wrapper/owner/schema/cleanup, Slice fixture, final-slice boundary,
terminal-layout, architecture, and pass-efficiency gate is
`371 passed, 1 xfailed in 19.73s`. Ruff, Python bytecode compilation, and
whitespace validation pass. The sole strict xfail is the selected result
assignment; production is unchanged at this checkpoint.

At implementation, replace only the discarded expression with the selected
assignment and update stale boundary targets. Do not add a reconciliation
guard in the same unit. Validate sequentially, commit, and push only; do not
create, reopen, or update a pull request.

## Final Slice pre/post result retention implementation checkpoint

The sole production call now retains its unchanged one-counter dictionary as
`_final_slice_prepost_passthrough_stats`. The value remains unconsumed in this
unit even though its positive-only cleanup makes it complete owner mutation
evidence.

This is an assignment-only production change. The compatibility wrapper,
owner implementation and schema, cleanup guard, exact call count, model-only
argument, final slice/concat predecessor, pre-Concat successor, dependencies,
public behavior, and TensorFlow-free direct path are unchanged. Two stale
boundary target expectations now require the assigned result.

Implementation validation completed sequentially under `uv`:

- wrapper/owner/schema/cleanup, Slice fixtures, final-slice boundary,
  terminal-layout, architecture, and pass-efficiency coverage:
  `372 passed in 22.41s`
- branch-changed broad suite: `1545 passed in 28.53s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, audit all three
production forms of `_optimize_transpose_pre_concat_nhwc_chains()`, beginning
with the call immediately after `_final_slice_prepost_passthrough_stats`.
Freeze each distinct boundary, shared layout/diagnostics arguments, owner
schema and cleanup behavior, and existing result policies before changing
production. Commit and push only; do not create, reopen, or update a pull
request.

## Direct pre-Concat result characterization checkpoint

The pre-Concat composite has three direct production calls plus one independent
layout-recovery callback execution. It runs indexed, quantized-indexed, and
legacy families in fixed order and returns their summed rewrite count as one
dictionary. The legacy owner prunes unused tensors unconditionally, so a zero
composite counter can coexist with cleanup-only mutation.

A strict expected-failure contract selects `_layout_opt_pre_concat_stats`,
`_final_pre_concat_stats`, and `_absolute_final_pre_concat_stats` for the three
direct calls. It freezes identical ModelIR/layout/diagnostics routing, exact
three-call count, three boundary pairs, one-key schema, dispatch order,
unconditional legacy cleanup, and the independent layout-recovery callback
selection. All three direct results must remain unconsumed.

The focused composite/schema/cleanup, float and quantized Concat fixtures,
layout-recovery callback, Slice/late-hard-activation/shape-extract/ReLU-split
boundaries, architecture, and pass-efficiency gate is
`430 passed, 1 xfailed in 19.00s`. Ruff, Python bytecode compilation, and
whitespace validation pass. The sole strict xfail covers the three selected
direct assignments; production is unchanged at this checkpoint.

At implementation, replace only the three discarded direct expressions with
the selected assignments and update stale boundary targets. Do not change or
consume the layout-recovery callback result in this unit. Validate
sequentially, commit, and push only; do not create, reopen, or update a pull
request.

## Direct pre-Concat result retention implementation checkpoint

The three direct calls now retain their unchanged one-counter dictionaries as
`_layout_opt_pre_concat_stats`, `_final_pre_concat_stats`, and
`_absolute_final_pre_concat_stats`. All remain unconsumed because the legacy
family can prune unused tensors while the summed rewrite counter is zero.

These are assignment-only changes. Indexed/quantized-indexed/legacy dispatch,
the independent layout-recovery callback execution, identical
ModelIR/layout/diagnostics routing, exact three-call multiplicity, pass order,
dependencies, public behavior, and the TensorFlow-free direct path remain
unchanged. Adjacent Slice/ReLU-split and late-hard-activation/shape-extract
contracts now require the assigned targets.

Implementation validation completed sequentially under `uv`:

- composite/schema/cleanup, float and quantized Concat fixtures,
  layout-recovery callback, all three boundary pairs, architecture, and
  pass-efficiency coverage: `431 passed in 19.60s`
- branch-changed broad suite: `1548 passed in 29.18s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The first focused implementation run found two predecessor call names that the
strict xfail had not reached; the characterization was corrected to the exact
Slice owner and late-hard-activation summarizer before the clean results above.
No production failure was observed. These checks do not claim a model-corpus
run.

At resume, audit the sole direct `run_ndhwc_concat_layout_cleanup()` call
immediately after `_layout_opt_pre_concat_stats`, plus its independent
layout-recovery orchestration occurrence. Freeze its schema, cleanup behavior,
layout/diagnostics arguments, and pre-Concat/strided-Slice boundaries before
changing the direct call. Commit and push only; do not create, reopen, or
update a pull request.

## Direct NDHWC Concat result characterization checkpoint

The NDHWC Concat transactional cleanup has one direct production call after
`_layout_opt_pre_concat_stats` plus one independent execution selected by
layout recovery. Its runner returns one counter, and the inner owner prunes
unused tensors only after a positive rewrite, so the dictionary completely
describes owner mutation.

A strict expected-failure contract selects `_layout_opt_ndhwc_concat_stats`
for the direct call and requires it to remain unconsumed. It freezes the
one-key schema, positive-only cleanup, ModelIR/layout/diagnostics routing,
pre-Concat/strided-Slice boundaries, exact sole direct call, and independent
layout-recovery selection.

The focused schema/cleanup, NDHWC fixture, layout-recovery selection,
pre-Concat/strided-Slice boundary, architecture, and pass-efficiency gate is
`375 passed, 1 xfailed in 18.42s`. Ruff, Python bytecode compilation, and
whitespace validation pass. The sole strict xfail is the selected direct
assignment; production is unchanged at this checkpoint.

At implementation, replace only the discarded direct expression with the
selected assignment and update stale boundary targets. Do not change the
layout-recovery occurrence in the same unit. Validate sequentially, commit,
and push only; do not create, reopen, or update a pull request.

## Direct NDHWC Concat result retention implementation checkpoint

The sole direct call now retains its unchanged one-counter dictionary as
`_layout_opt_ndhwc_concat_stats`. It remains unconsumed in this unit even
though positive-only cleanup makes it complete owner mutation evidence.

This is an assignment-only production change. The transactional runner,
inner owner, schema, cleanup guard, ModelIR/layout/diagnostics routing,
independent layout-recovery occurrence, pass order, dependencies, public
behavior, and TensorFlow-free direct path are unchanged.

Implementation validation completed sequentially under `uv`:

- schema/cleanup, NDHWC fixture, layout-recovery selection,
  pre-Concat/strided-Slice boundary, architecture, and pass-efficiency
  coverage: `376 passed in 18.50s`
- branch-changed broad suite: `1551 passed in 28.84s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a model-corpus run. At resume, audit the sole direct
`_optimize_transpose_stridedslice_pre_concat_nhwc_chains()` call immediately
after `_layout_opt_ndhwc_concat_stats`, plus its independent layout-recovery
occurrence. Freeze the indexed schema, cleanup semantics, graph-index/layout
arguments, NDHWC/split-mixed boundaries, and current result policy before
changing production. Commit and push only; do not create, reopen, or update a
pull request.

## Direct indexed strided-Slice pre-Concat result characterization checkpoint

The indexed owner has one direct production call after
`_layout_opt_ndhwc_concat_stats` plus one independent public-owner execution
inside layout recovery. It returns one rewrite counter but invokes
unused-tensor pruning unconditionally. A dedicated zero-rewrite fixture proves
that the raw zero counter can coexist with cleanup-only mutation.

A strict expected-failure contract selects
`_layout_opt_stridedslice_pre_concat_stats` for the direct result and requires
it to remain unconsumed. It freezes optional graph-index/layout/bound/candidate
forwarding, one-key schema, unconditional cleanup, model/layout arguments,
exact sole direct call, NDHWC/split-mixed boundaries, and independent
orchestration selection.

The focused wrapper/schema/cleanup and zero-prune fixture, indexed owner suite,
NDHWC/split-mixed boundaries, layout-recovery selection, architecture, and
pass-efficiency gate is `360 passed, 1 xfailed in 18.98s`. Ruff, Python
bytecode compilation, and whitespace validation pass. The sole strict xfail is
the selected direct assignment; production is unchanged at this checkpoint.

At implementation, replace only the discarded direct expression with the
selected assignment and update stale boundary targets. Do not change the
layout-recovery occurrence or use the zero counter as a guard. Validate
sequentially, commit, and push only; do not create, reopen, or update a pull
request.

## Direct indexed strided-Slice pre-Concat result retention checkpoint

The sole direct call now retains its unchanged one-counter dictionary as
`_layout_opt_stridedslice_pre_concat_stats`. It remains unconsumed because the
owner's unconditional unused-tensor pruning permits cleanup-only mutation with
zero rewrites. The independent public-owner layout-recovery occurrence remains
unchanged.

This is an assignment-only production change. Optional graph-index/layout/
bound/candidate forwarding, one-key schema, direct ModelIR/layout arguments,
exact call multiplicity, NDHWC/split-mixed boundaries, pass order,
dependencies, public behavior, and TensorFlow-free direct path remain fixed.

Implementation validation completed sequentially under `uv`:

- wrapper/schema/zero-prune fixture, indexed owner suite, NDHWC/split-mixed
  boundaries, layout-recovery selection, architecture, and pass-efficiency
  coverage: `361 passed in 18.38s`
- branch-changed broad suite: `1554 passed in 29.49s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a model-corpus run. At resume, audit the sole direct
`run_spp_layout_cleanup()` call immediately before
`_layout_opt_pre_concat_stats`, plus its independent layout-recovery
occurrence. Freeze its transactional schema, positive-only cleanup semantics,
layout/diagnostics arguments, elementwise-Concat/pre-Concat boundaries, and
current result policy before changing production. Commit and push only; do not
create, reopen, or update a pull request.

## Direct SPP cleanup result characterization checkpoint

The SPP transactional cleanup has one direct call immediately before
`_layout_opt_pre_concat_stats` plus three independent selections in layout
recovery, the late-layout cluster, and the late-SPP pair. Its inner owner prunes
unused tensors only after a positive rewrite, so the one-counter dictionary
completely describes owner mutation.

A strict expected-failure contract selects `_layout_opt_spp_stats` for the
direct result and requires it to remain unconsumed. It freezes the one-key
schema, positive-only cleanup, ModelIR/layout/diagnostics arguments, exact sole
direct call, elementwise-Concat/pre-Concat boundaries, and all three
independent orchestration selections.

The focused schema/cleanup, SPP owner fixtures, three orchestration selections,
elementwise-Concat/pre-Concat boundaries, architecture, and pass-efficiency
gate is `363 passed, 1 xfailed in 18.83s`. Ruff, Python bytecode compilation,
and whitespace validation pass. The sole strict xfail is the selected direct
assignment; production is unchanged at this checkpoint.

At implementation, replace only the discarded direct expression with the
selected assignment and update stale boundary targets. Do not change any
orchestration result policy in the same unit. Validate sequentially, commit,
and push only; do not create, reopen, or update a pull request.

## Direct SPP cleanup result retention implementation checkpoint

The sole direct call now retains its unchanged one-counter dictionary as
`_layout_opt_spp_stats`. It remains unconsumed in this unit even though
positive-only cleanup makes it complete owner mutation evidence.

This is an assignment-only production change. The transactional runner,
inner owner and schema, cleanup guard, ModelIR/layout/diagnostics routing,
three independent orchestration selections, pass order, dependencies, public
behavior, and TensorFlow-free direct path remain unchanged.

Implementation validation completed sequentially under `uv`:

- schema/cleanup, SPP fixtures, three orchestration selections,
  elementwise-Concat/pre-Concat boundaries, architecture, and pass-efficiency
  coverage: `364 passed in 18.10s`
- branch-changed broad suite: `1556 passed in 28.08s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a model-corpus run. At resume, audit the sole direct
`_optimize_transpose_elementwise_concat_conv_nhwc_groups()` call immediately
before `_layout_opt_spp_stats`, plus its independent layout-recovery owner
execution. Freeze optional graph-index/layout/bound/candidate forwarding,
schema, cleanup semantics, binary-bridge/SPP boundaries, and current result
policy before changing production. Commit and push only; do not create,
reopen, or update a pull request.

## Direct elementwise-Concat/Conv result characterization checkpoint

The indexed elementwise-Concat/Conv owner has one direct private-wrapper call
between the quantized-activation binary-bridge recovery sequence and
`_layout_opt_spp_stats`. Layout recovery independently selects the public owner
once. The compatibility wrapper forwards optional graph index, layout state,
rewrite bound, and candidate identity unchanged.

The owner returns the fixed one-key rewrite dictionary but invokes
unused-tensor pruning unconditionally. A zero-rewrite fixture proves that the
owner can remove an unrelated unused tensor while returning zero, so the
direct result is observation-only and cannot be used as complete mutation or
guard evidence.

A strict expected-failure contract selects
`_layout_opt_elementwise_concat_conv_stats` for the direct result and requires
it to remain unconsumed. It freezes wrapper forwarding, the one-key schema,
unconditional cleanup, exact ModelIR/layout arguments, sole direct call,
binary-bridge/SPP boundaries, and the independent public-owner orchestration
selection.

The focused wrapper/owner schema, zero-rewrite prune, elementwise-Concat/Conv
fixtures, layout-recovery selection, binary-bridge/SPP boundaries,
architecture, and pass-efficiency gate is
`358 passed, 1 xfailed in 19.01s`. The sole strict xfail is the selected direct
assignment; production is unchanged at this checkpoint.

At implementation, replace only the discarded direct expression with the
selected assignment and update the stale SPP predecessor target. Do not add a
consumer or guard and do not change the public-owner layout-recovery
occurrence. Validate sequentially, commit, and push only; do not create,
reopen, or update a pull request.

## Direct elementwise-Concat/Conv result retention implementation checkpoint

The sole direct private-wrapper call now retains its unchanged one-counter
dictionary as `_layout_opt_elementwise_concat_conv_stats`. It remains
observation-only and unconsumed because the owner's unconditional prune can
mutate ModelIR while the rewrite counter is zero. The independent public-owner
layout-recovery occurrence is unchanged.

This is an assignment-only production change. The compatibility wrapper,
owner, one-key schema, optional graph-index/layout/bound/candidate forwarding,
ModelIR/layout arguments, pass order, binary-bridge/SPP boundaries,
dependencies, diagnostics behavior, public API, and TensorFlow-free direct
path remain unchanged.

The first implementation gate exposed one stale AST boundary that required a
raw expression after the second quantized-activation binary recovery call
(`358 passed, 1 failed`). It was updated to require the new assignment target
and the same unchanged RHS owner call; this was a characterization adjustment,
not a production regression.

Implementation validation completed sequentially under `uv`:

- wrapper/owner schema, zero-rewrite prune, elementwise-Concat/Conv fixtures,
  layout-recovery selection, binary-bridge/SPP boundaries, architecture, and
  pass-efficiency coverage: `359 passed in 19.54s`
- branch-changed broad suite: `1559 passed in 27.59s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a model-corpus run. At resume, audit both calls to
`_run_quantized_activation_binary_bridge_recovery_sequence()` and the runner
result contract they currently discard. Preserve the conditional binary-bridge
policy following the first call and the retained elementwise-Concat/Conv result
following the second call. Commit and push only; do not create, reopen, or
update a pull request.

## Quantized-activation binary result characterization checkpoint

`run_quantized_activation_binary_recovery()` selects six ordered children:
dequantized HardSigmoid, MaxPool, Softmax, and Logistic folding;
Softmax/Transpose canonicalization; and nested safe-binary recovery. The nested
`run_safe_binary_recovery()` selects one owner returning the five fixed safe
binary mode counters. Both runners currently discard the tuples produced by
`run_recovery_invocations()`, so the outer runner's sixth slot is currently
`None`.

The child schemas are frozen as five one-key dictionaries followed by one
nested one-slot tuple containing the safe owner's five-key dictionary. Every
child owner performs unused-tensor pruning independently of its rewrite
counter. A dedicated fixture proves the first child can remove an unused
tensor while returning zero, so neither production result is complete mutation
evidence.

A strict expected-failure contract requires `Tuple[Any, ...]` propagation from
both phase runners and the lowerer helper. It selects observation-only targets
`_layout_pass_set_1_quantized_activation_binary_results` and
`_layout_pass_set_2_quantized_activation_binary_results`. It freezes the exact
order, shared context arguments, nested result shape, two zero-argument calls,
unconsumed policy, and their distinct reshape/conditional-binary and
dequant-TransposeConv/elementwise-Concat boundaries.

The focused orchestration, child schema, cleanup-only mutation, activation
fold, canonicalization, safe-binary, both production boundaries, architecture,
and pass-efficiency gate is `665 passed, 1 xfailed in 18.31s`. Ruff, Python
bytecode compilation, and whitespace validation pass. The sole strict xfail is
the ordered nested-result propagation contract; production is unchanged at
this checkpoint.

At implementation, return the existing tuples from both phase runners and the
lowerer helper, then replace only the two discarded helper expressions with
the selected assignments. Do not summarize or consume either result and do not
change child owners, call order, contexts, the conditional binary-bridge
policy, or the retained elementwise-Concat/Conv successor. Validate
sequentially, commit, and push only; do not create, reopen, or update a pull
request.

## Quantized-activation binary result propagation implementation checkpoint

`run_safe_binary_recovery()` now returns its existing one-slot ordered tuple,
and `run_quantized_activation_binary_recovery()` returns its existing six-slot
tuple. The outer tuple's sixth slot is therefore the nested safe-binary tuple
instead of `None`. The lowerer helper transparently returns that outer tuple.

The first production call retains
`_layout_pass_set_1_quantized_activation_binary_results`; the second retains
`_layout_pass_set_2_quantized_activation_binary_results`. Both remain
unconsumed and observation-only because zero counters do not exclude
cleanup-only pruning. No mutation summary or guard was added.

The first implementation gate exposed only two stale structural contracts:
one walked the helper's new return annotation as runtime loaded data, and one
required the helper and both production calls to remain raw expressions
(`664 passed, 2 failed`). The former now scopes its data-flow scan to the
helper body, and the latter requires the return plus both explicit assignment
targets. These were characterization adjustments, not production regressions.

Implementation validation completed sequentially under `uv`:

- quantized recovery, all activation-fold children, canonicalization,
  safe-binary, both result boundaries, elementwise-Concat/Conv, architecture,
  and pass-efficiency coverage: `666 passed in 20.24s`
- branch-changed broad suite: `1570 passed in 28.87s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a model-corpus run. At resume, audit the two direct
calls to `_run_safe_binary_bridge_recovery_sequence()` outside the nested
quantized-activation runner. The phase runner now exposes its one-slot tuple,
but the lowerer helper still discards it. Preserve the layout-attention/
dequantized-Mean and unary-fanout/progress boundaries. Commit and push only; do
not create, reopen, or update a pull request.

## Direct safe-binary results characterization checkpoint

The one-slot tuple returned by `run_safe_binary_recovery()` is also reached by
two independent zero-argument lowerer-helper calls outside the nested
quantized-activation recovery. The first follows the layout-attention/
quantized suffix and precedes the dequantized-Mean/Quantize bridge. The second
follows the Transpose/unary fanout cluster and immediately precedes pass-set
progress advancement. The lowerer helper currently discards both results.

A strict expected-failure contract selects
`_layout_pass_set_1_safe_binary_results` and
`_layout_pass_set_1_final_safe_binary_results`. It requires transparent helper
return, exact two-call assignment, unchanged zero arguments, the existing
one-slot/five-key schema, both boundary pairs, and an unconsumed
observation-only policy. The five-mode owner prunes unconditionally, so zero
counters cannot serve as complete mutation evidence.

The first focused run also detected that the previous implementation
checkpoint's final multiline formatting changed the quantized helper's fixed
line count from four to five after its last pytest run
(`409 passed, 1 failed, 1 xfailed`). The stale line-count expectation is now
five; this corrects structural coverage and does not change production.

The corrected focused safe-binary owner, nested quantized recovery,
layout-attention suffix, unary-fanout, both direct boundaries, architecture,
and pass-efficiency gate is `410 passed, 1 xfailed in 18.93s`. The sole strict
xfail is the two direct-result assignments. Production is otherwise unchanged
at this checkpoint. The branch-changed broad suite also restores a clean
characterization state at `1570 passed, 1 xfailed in 29.04s`; the same selected
assignment contract is its only xfail. Targeted Ruff, Python bytecode
compilation, and whitespace validation pass.

At implementation, return the existing safe-binary tuple from the lowerer
helper and replace only the two discarded direct expressions with the selected
assignments. Do not add a consumer or guard and do not alter the already
propagated nested call, owner, schema, order, contexts, or adjacent policies.
Validate sequentially, commit, and push only; do not create, reopen, or update
a pull request.

## Direct safe-binary result retention implementation checkpoint

The lowerer safe-binary helper now transparently returns the one-slot tuple
already exposed by `run_safe_binary_recovery()`. Its first direct call retains
`_layout_pass_set_1_safe_binary_results`; its final pass-set call retains
`_layout_pass_set_1_final_safe_binary_results`. Both results remain unconsumed
and observation-only because all-zero mode counters do not exclude the owner's
unconditional prune.

The first implementation gate exposed four stale structural boundaries that
required raw safe-binary expressions after the layout-attention suffix and
Transpose/unary-fanout cluster (`407 passed, 4 failed`). Each now requires the
appropriate assignment target and the same unchanged RHS helper call. These
were characterization adjustments, not production regressions.

Implementation validation completed sequentially under `uv`:

- safe-binary owner and helper, nested quantized recovery, layout-attention
  suffix, Transpose/unary-fanout, both result boundaries, architecture, and
  pass-efficiency coverage: `411 passed in 20.67s`
- branch-changed broad suite: `1571 passed in 29.20s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The implementation changes only helper return and two assignments. It adds no
summary, guard, child invocation, pass-order change, dependency, public API
change, or TensorFlow import path. These checks do not claim a model-corpus
run.

At resume, audit the two direct
`_run_layout_attention_quantized_recovery_suffix()` calls immediately before
the new first safe-binary result and the Transpose/unary-fanout cluster. Freeze
the phase runner's ordered result contract, policy argument, and both distinct
boundaries before changing production. Commit and push only; do not create,
reopen, or update a pull request.

## Layout-attention/quantized suffix result characterization checkpoint

`run_layout_attention_quantized_suffix()` selects thirteen ordered children:
three layout owners; nested mean-attention, attention-gate, and duplicate-PReLU
clusters; dequantized TransposeConv; quantized Reshape cleanup; four quantized
activation folds; and Softmax/Transpose canonicalization. The runner and its
zero-argument lowerer helper currently discard the tuple returned by
`run_recovery_invocations()`.

An instrumented contract freezes the exact thirteen pass IDs and result slots,
including the three nested callback results. Both production calls pass the
same `enable_duplicate_transpose_fanout_optimizations` policy. The first lies
between affine folding and `_layout_pass_set_1_safe_binary_results`; the second
lies between Squeeze/Reshape cleanup and the Transpose/unary-fanout cluster.

A strict expected-failure contract selects observation-only targets
`_layout_pass_set_1_attention_quantized_suffix_results` and
`_layout_pass_set_1_final_attention_quantized_suffix_results`. It requires
tuple propagation through the runner and helper, exact policy arguments, both
boundaries, and no consumers. Quantized suffix children perform cleanup even
when their rewrite counters are zero, so neither tuple is complete mutation
evidence.

The focused suffix, nested cluster, activation-fold, TransposeConv, quantized
Reshape, canonicalization, both production boundaries, architecture, and
pass-efficiency gate is `726 passed, 1 xfailed in 19.28s`. Ruff, Python
bytecode compilation, and whitespace validation pass. The sole strict xfail is
the selected two-result propagation contract; production is unchanged at this
checkpoint.

At implementation, return the existing ordered tuple from the phase runner and
helper, then replace only the two discarded expressions with the selected
assignments. Do not normalize, summarize, consume, or guard on the results and
do not change child callbacks, policy routing, order, shared context, safe-
binary successors, or unary-fanout behavior. Validate sequentially, commit,
and push only; do not create, reopen, or update a pull request.

## Layout-attention/quantized suffix result propagation implementation checkpoint

`run_layout_attention_quantized_suffix()` and its lowerer helper now return the
existing thirteen-slot tuple without normalization. The first production call
retains `_layout_pass_set_1_attention_quantized_suffix_results`; the second
retains `_layout_pass_set_1_final_attention_quantized_suffix_results`. Both
remain unconsumed and observation-only because suffix cleanup is not fully
represented by rewrite counters.

The first implementation gate exposed six stale structural contracts that
either counted `Tuple[Any, ...]` annotation names as runtime data or required
raw suffix expressions at the helper, safe-binary, and Transpose/unary-fanout
boundaries (`730 passed, 6 failed`). Each now scopes data-flow inspection to
the helper body or requires the correct assignment target plus unchanged RHS
and policy. These were characterization adjustments, not production
regressions.

Implementation validation completed sequentially under `uv`:

- suffix runner/helper, all direct and nested children, both result boundaries,
  safe-binary and Transpose/unary-fanout adjacency, architecture, and
  pass-efficiency coverage: `736 passed in 20.88s`
- branch-changed broad suite: `1588 passed in 28.15s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The implementation adds only two return statements and two assignments. It
adds no summary, guard, child invocation, order or policy change, dependency,
public API change, or TensorFlow import path. These checks do not claim a
model-corpus run.

At resume, audit the post-QDQ direct
`_run_transpose_unary_fanout_layout_pass_cluster()` result immediately after
`_layout_pass_set_1_final_attention_quantized_suffix_results`, and inventory
all other helper occurrences before changing any call. Preserve its final
safe-binary successor. Commit and push only; do not create, reopen, or update a
pull request.

## Transpose/unary-fanout result characterization checkpoint

`run_transpose_unary_fanout()` supports two three-slot policies on one shared
`ModelIRPassStateScope`. The default attention-gate/QDQ callback policy selects
unary-passthrough, unary-fanout, and unary/binary-fanout cleanup. The sole
post-QDQ direct call selects layout-Transpose, unary-fanout, and unary/binary-
fanout cleanup. The phase runner and lowerer helper currently discard both
ordered tuples.

The helper has exactly one direct invocation and one callback identity in
`AttentionRecoveryContext`. Changing it to return results will make the
callback's existing parent slot contain a three-dictionary tuple rather than
`None`; the attention-gate/QDQ parent currently discards its complete result
tuple and neither summarizes nor guards on that slot.

A strict expected-failure contract instruments and freezes both active pass-ID
orders and result identities. It selects observation-only target
`_layout_pass_set_1_transpose_unary_fanout_results` for the direct call and
freezes its exact `include_layout_transpose=True` /
`include_unary_passthrough=False` policy, final-suffix/final-safe-binary
boundaries, callback identity, helper return contract, and absence of a
consumer.

The focused Transpose/unary-fanout variants, attention recovery, suffix,
safe-binary, direct/callback boundaries, architecture, and pass-efficiency gate
is `370 passed, 1 xfailed in 18.07s`. Ruff, Python bytecode compilation, and
whitespace validation pass. The sole strict xfail is runner/helper/direct
result propagation; production is unchanged at this checkpoint.

At implementation, return the existing ordered tuple from the runner and
helper and assign only the direct post-QDQ call. Preserve both policy variants,
shared scope, callback identity, attention parent result policy, pass order,
and both adjacent retained results. Do not add a summary, guard, or consumer.
Validate sequentially, commit, and push only; do not create, reopen, or update
a pull request.

## Transpose/unary-fanout result propagation implementation checkpoint

`run_transpose_unary_fanout()` and its lowerer helper now return the active
three-dictionary tuple for both policy variants. The sole post-QDQ direct call
retains `_layout_pass_set_1_transpose_unary_fanout_results` between
`_layout_pass_set_1_final_attention_quantized_suffix_results` and
`_layout_pass_set_1_final_safe_binary_results`. It remains unconsumed and
observation-only.

The default-policy callback registered in `AttentionRecoveryContext` now
returns its three-dictionary tuple into the parent's existing callback slot
instead of `None`. The attention-gate/QDQ parent continues to discard its
enclosing result and has no summary, guard, or consumer, so runtime pass
behavior is unchanged.

The first implementation gate exposed three stale structural contracts that
required the sole direct helper call to remain a raw expression or the final
suffix successor to remain uncaptured (`368 passed, 3 failed`). They now
require the new target and unchanged RHS/policy. These were characterization
adjustments, not production regressions.

Implementation validation completed sequentially under `uv`:

- both Transpose/unary-fanout policy variants, attention callback, suffix and
  safe-binary boundaries, architecture, and pass-efficiency coverage:
  `371 passed in 20.83s`
- branch-changed broad suite: `1589 passed in 28.35s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The implementation adds only transparent returns and one direct assignment.
It adds no pass invocation, summary, guard, order/policy change, dependency,
public API change, or TensorFlow import path. These checks do not claim a
model-corpus run.

At resume, audit the parent
`_run_attention_gate_qdq_recovery_sequence()` result contract now that its
Transpose/unary-fanout callback slot is populated. Inventory every direct
parent invocation and preserve their distinct boundaries before changing the
parent runner or helper. Commit and push only; do not create, reopen, or update
a pull request.

## Attention-gate/QDQ parent result characterization checkpoint

`run_attention_gate_qdq_recovery()` selects ten ordered children covering
SA/PA MirrorPad, SINet mix-attention, the gate cluster, two TransposeConv output
owners, the Transpose/unary-fanout callback, two quantized activation bridges,
trailing-output cleanup, and a quantized PReLU bridge. The newly propagated
unary-fanout callback contributes its three-dictionary default-policy tuple at
slot five. The parent runner and lowerer helper currently discard the complete
tuple.

There are two zero-argument direct calls plus one structural callback selection
inside the thirteen-slot layout-attention/quantized suffix. Because that suffix
runs twice, parent propagation will populate both retained suffix tuples as
well as the two distinct direct observations; no current consumer summarizes
or guards on any of these results.

A strict expected-failure contract instruments all ten slots, including the
nested unary-fanout tuple. It selects observation-only direct targets
`_layout_pass_set_1_attention_gate_qdq_results` and
`_layout_pass_set_2_attention_gate_qdq_results`. It freezes helper/runner
return, zero arguments, direct-call count, mean-attention/quantized-PReLU and
pre-add/dequant-TransposeConv boundaries, suffix nesting order, and absence of
consumers.

The focused attention parent, suffix, unary-fanout, gate, SA/PA, SINet,
TransposeConv, quantized activation, trailing-output, both direct boundaries,
architecture, and pass-efficiency gate is
`571 passed, 1 xfailed in 18.81s`. Ruff, Python bytecode compilation, and
whitespace validation pass. The sole strict xfail is parent runner/helper/two-
call propagation; production is unchanged at this checkpoint.

At implementation, return the existing ten-slot tuple from the parent runner
and helper and assign only the two direct calls. Allow the existing suffix
callback slots to receive the tuple without adding a summary, guard, or
consumer. Preserve every child, nested result identity, shared context, order,
and boundary. Validate sequentially, commit, and push only; do not create,
reopen, or update a pull request.

## Attention-gate/QDQ parent result propagation implementation checkpoint

`run_attention_gate_qdq_recovery()` and its lowerer helper now return the
existing ten-slot tuple. The two direct calls retain
`_layout_pass_set_1_attention_gate_qdq_results` and
`_layout_pass_set_2_attention_gate_qdq_results`. Both remain unconsumed and
observation-only.

The same helper remains the child callback selected by the thirteen-slot
layout-attention/quantized suffix. Each retained suffix result now contains the
nested attention-parent tuple instead of `None`. Neither suffix result has a
consumer, summary, or guard, so this evidence propagation does not alter pass
execution.

The first implementation gate exposed three stale structural contracts that
counted return annotation names as body data or required the attention helper
and two direct calls to remain raw expressions (`569 passed, 3 failed`). They
now inspect body-only data and require the return plus both assignment targets.
These were characterization adjustments, not production regressions.

Implementation validation completed sequentially under `uv`:

- attention parent and all children, nested unary-fanout and suffix results,
  both direct boundaries, architecture, and pass-efficiency coverage:
  `572 passed in 20.79s`
- branch-changed broad suite: `1590 passed in 28.69s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The implementation adds only transparent return and assignment boundaries. It
adds no child, summary, guard, execution-order or context change, dependency,
public API change, or TensorFlow import path. These checks do not claim a
model-corpus run.

At resume, audit both direct
`_run_preadd_mean_attention_recovery_sequence()` results and their distinct
boundaries. Its parent runner still discards its ordered child tuple. Preserve
the newly retained pass-set-2 attention result that follows the second call.
Commit and push only; do not create, reopen, or update a pull request.

## Pre-add/mean/attention parent result characterization checkpoint

`run_preadd_mean_attention_recovery()` selects seven ordered children: pre-add,
two residual-affine variants, direct affine, pre-unary affine, mean-affine, and
the nested mean-attention cluster. The runner and its zero-argument lowerer
helper currently discard both pass-set-2 result tuples.

The first direct call lies between the layout-recovery prefix and the newly
retained `_layout_pass_set_2_attention_gate_qdq_results`. The second lies
between `_layout_opt_channel_shuffle_gather_results` and
`_layout_opt_sa_pa_mirrorpad_stats`. Neither call has a current consumer,
summary, or guard.

A strict expected-failure contract instruments all seven result identities,
including a nested mean-attention tuple. It selects observation-only targets
`_layout_pass_set_2_preadd_mean_attention_results` and
`_layout_opt_preadd_mean_attention_results`. It freezes runner/helper return,
shared context, exact zero-argument call count, both boundary pairs, and the
unconsumed policy.

The focused pre-add parent, nested mean-attention, layout recovery,
channel-shuffle, SA/PA, retained attention boundary, architecture, and
pass-efficiency gate is `355 passed, 1 xfailed in 18.12s`. Ruff, Python
bytecode compilation, and whitespace validation pass. The sole strict xfail is
runner/helper/two-call propagation; production is unchanged at this
checkpoint.

At implementation, return the existing seven-slot tuple from the runner and
helper and replace only the two discarded expressions with the selected
assignments. Do not normalize or consume the nested mean-attention result and
do not change any child, context, order, boundary, summary, or guard. Validate
sequentially, commit, and push only; do not create, reopen, or update a pull
request.

## Pre-add/mean/attention parent result propagation implementation checkpoint

`run_preadd_mean_attention_recovery()` and its lowerer helper now return the
existing seven-slot tuple. The first direct call retains
`_layout_pass_set_2_preadd_mean_attention_results`; the later call retains
`_layout_opt_preadd_mean_attention_results`. Both remain unconsumed and
observation-only, and the nested mean-attention tuple remains intact.

The first implementation gate exposed one stale architecture contract that
required the helper and both direct calls to remain raw expressions
(`355 passed, 1 failed`). It now requires the helper return plus the two
explicit targets. This was a characterization adjustment, not a production
regression.

Implementation validation completed sequentially under `uv`:

- pre-add parent and nested mean-attention, layout recovery, channel-shuffle,
  SA/PA, both retained boundary pairs, architecture, and pass-efficiency
  coverage: `356 passed in 20.18s`
- branch-changed broad suite: `1591 passed in 29.45s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The implementation adds only transparent returns and two assignments. It adds
no child, summary, guard, context/order change, dependency, public API change,
or TensorFlow import path. These checks do not claim a model-corpus run.

At resume, audit `_run_layout_recovery_prefix_pass_sequence()` immediately
before `_layout_pass_set_2_preadd_mean_attention_results` and inventory every
other direct or nested occurrence before changing its runner/helper result
policy. Commit and push only; do not create, reopen, or update a pull request.

## Singleton/Reshape result characterization checkpoint

`run_singleton_reshape()` selects seven to ten ordered child runners from the
layout-transpose, duplicate-fanout, multi-branch-gate, and spatial-Concat
post-transpose policy flags. All selected runners already share one
`ModelIRPassStateScope`, and `run_recovery_invocations()` already returns their
results in the selected pass-ID order. The phase runner and its local helper
currently discard that tuple.

There are exactly two direct primary calls. The call after
`_terminal_qkv_split_conv_concat_bridge_stats` is the last statement in the
`optimize_layout_transpose_chains` guard and enables layout-transpose plus
multi-branch-gate cleanup. The later call follows terminal SiNet pre-add/resize
recovery, enables duplicate-fanout cleanup, disables spatial-Concat
post-transpose cleanup, and precedes indexed shape convergence.

A strict expected-failure contract selects
`_terminal_singleton_reshape_results` and
`_post_terminal_singleton_reshape_results` as the two retention targets. It
fixes all sixteen policy-specific result tuples, exact child order, helper
return boundary, direct-call count, policy keywords, shared scope, and both
production boundaries.

At implementation, transparently return the existing tuple through the phase
runner and local helper, then replace only the two raw direct expressions with
assignments. Do not add a consumer or guard, and do not change policy defaults,
child order, shared-scope construction, surrounding sequences, dependencies,
diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- Singleton/Reshape policies and boundaries, indexed Split/Conv/Concat bridge,
  QKV orchestration, architecture, and pass-efficiency coverage:
  `391 passed, 1 xfailed in 18.54s`

The sole strict expected failure is the intentionally unimplemented ordered
result propagation contract. Implement only that contract, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## Singleton/Reshape result propagation implementation checkpoint

`run_singleton_reshape()` now returns the ordered dictionary tuple produced by
`run_recovery_invocations()`, and the local helper transparently forwards it.
The guarded terminal call retains `_terminal_singleton_reshape_results`; the
later top-level call retains `_post_terminal_singleton_reshape_results`.

Both tuples are observation-only and have no consumer or guard. All sixteen
policy combinations still select the same child pass IDs, every selected child
executes exactly once through the original shared scope, and the option guard,
two production boundaries, pass order, dependencies, diagnostics, and
TensorFlow behavior remain unchanged. Two architecture tests were updated to
recognize and verify the newly assigned Singleton boundary instead of assuming
that it remained a raw expression.

Implementation validation completed sequentially under `uv`:

- Singleton/Reshape policies and boundaries, indexed Split/Conv/Concat bridge,
  QKV orchestration, architecture, and pass-efficiency coverage:
  `392 passed in 19.36s`
- branch-changed broad suite plus the same Singleton/Reshape coverage:
  `1503 passed in 25.34s`

These are unit, contract, and orchestration checks; this result propagation
does not claim a new model-corpus run.

At resume, audit the raw top-level
`_run_indexed_shape_convergence_cleanup()` result immediately after
`_post_terminal_singleton_reshape_results`. Preserve the already consumed
nested convergence form, the terminal call's live LayoutState, and all
following very-late recovery boundaries. Commit and push only; do not create,
reopen, or update a pull request.

## Top-level indexed shape-convergence result characterization checkpoint

`_run_indexed_shape_convergence_cleanup()` builds or accepts one
`ModelIRGraphIndex` and uses it for dead-operator pruning, the first static
shape reconciliation, dynamic-Reshape resolution, and a conditional final
reconciliation. Its fixed complete mutation dictionary contains
`removed_dead_operators`, `resolved_dynamic_reshape_shapes`, and
`reconciled_static_tensor_shapes`.

There are exactly two production forms. The nested form in
`_run_indexed_final_shape_activation_convergence()` is already retained as
`convergence_stats`, supplies its shared graph index, and is consumed by later
guards and the final result. The raw top-level form follows
`_post_terminal_singleton_reshape_results`, supplies the live Session
LayoutState, creates its own graph index, and precedes very-late SiNet terminal
recovery.

A strict expected-failure contract selects
`_post_terminal_indexed_shape_convergence_stats` for only the top-level form.
It fixes the three-key result schema, exact two-form count, arguments and
keywords, existing nested consumer, captured Singleton predecessor, and SiNet
successor.

At implementation, replace only the raw top-level expression with an
assignment. Do not change the helper, schema, graph-index reuse, conditional
final reconciliation, nested consumer, live LayoutState, surrounding recovery
sequence, dependencies, diagnostics, or TensorFlow behavior. Do not add a new
consumer or guard.

Characterization validation completed sequentially under `uv`:

- indexed shape-convergence schema and both forms, dynamic-Reshape and indexed
  final convergence, Singleton and SiNet boundaries, architecture, and
  pass-efficiency coverage: `361 passed, 1 xfailed in 18.46s`

The sole strict expected failure is the intentionally unimplemented top-level
result retention contract. Implement only that assignment, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## Top-level indexed shape-convergence result retention implementation checkpoint

The raw top-level call now retains its unchanged three-key mutation dictionary
as `_post_terminal_indexed_shape_convergence_stats`. The nested call remains
retained as `convergence_stats`, continues to receive the enclosing shared
graph index, and remains consumed by the final convergence helper.

This is an assignment-only change. No result consumer or guard was added. The
indexed helper, fixed schema, one-index reuse, conditional final reconciliation,
live Session LayoutState, captured Singleton predecessor, very-late SiNet
successor, dependencies, diagnostics, and TensorFlow behavior remain
unchanged. Two stale boundary tests were updated to verify the exact assigned
target instead of requiring a raw expression.

Implementation validation completed sequentially under `uv`:

- indexed shape-convergence schema and both forms, dynamic-Reshape and indexed
  final convergence, Singleton and SiNet boundaries, architecture, and
  pass-efficiency coverage: `362 passed in 20.57s`
- branch-changed broad suite plus the same indexed-convergence coverage:
  `1512 passed in 26.73s`

These are unit, contract, and orchestration checks; this result retention does
not claim a new model-corpus run.

At resume, audit the first very-late
`_run_sinet_terminal_layout_recovery_sequence()` result immediately after
`_post_terminal_indexed_shape_convergence_stats`. Preserve its other production
occurrence, nested pre-add/resize callback, exact child order, and the following
top-level pre-add/resize call. Commit and push only; do not create, reopen, or
update a pull request.

## SiNet terminal-layout result characterization checkpoint

`run_sinet_terminal_layout_recovery()` executes three fixed invocations in
order: shuffle-residual recovery, the injected pre-add/resize callback, and
terminal affine/PRELU recovery. The callback intentionally accepts no
arguments and can return any value. The generic recovery runner already
returns all three child results, while the phase runner and local helper
currently discard the tuple.

There are exactly two zero-argument direct calls. The terminal call follows
terminal clamp/unary/ReLU cleanup and precedes HardSwish-SE recovery. The
very-late call follows `_post_terminal_indexed_shape_convergence_stats` and
precedes a separate top-level pre-add/resize recovery call.

A strict expected-failure contract selects
`_terminal_sinet_layout_recovery_results` and
`_very_late_sinet_layout_recovery_results`. It fixes the three-result order,
arbitrary callback result, `Tuple[Any, ...]` phase/helper boundary, direct-call
count and policies, context wiring, and both surrounding sequences.

At implementation, transparently return the existing tuple through the phase
runner and local helper, then replace only the two raw direct expressions with
assignments. Do not change any child, the injected callback, child order,
context, surrounding recovery calls, dependencies, diagnostics, or TensorFlow
behavior. Do not add a result consumer or guard.

Characterization validation completed sequentially under `uv`:

- SiNet terminal and pre-add/resize orchestration, indexed-convergence and
  Singleton boundaries, architecture, and pass-efficiency coverage:
  `343 passed, 1 xfailed in 18.47s`

The sole strict expected failure is the intentionally unimplemented ordered
result propagation contract. One stale pre-add/resize boundary assertion was
also updated to recognize the already captured Singleton result explicitly.
Implement only the propagation contract, rerun focused and branch-changed broad
gates sequentially, then commit and push only; do not create, reopen, or update
a pull request.

## SiNet terminal-layout result propagation implementation checkpoint

`run_sinet_terminal_layout_recovery()` now returns the existing ordered
`Tuple[Any, ...]`, and the local helper transparently forwards it. The first
direct call retains `_terminal_sinet_layout_recovery_results`; the very-late
call retains `_very_late_sinet_layout_recovery_results`.

Both tuples are observation-only and have no consumer or guard. All three
children still execute exactly once in their fixed order, including the
injected pre-add/resize callback, whose return value remains arbitrary and is
preserved as the middle element. The context, zero-argument calls, terminal
clamp and indexed-convergence predecessors, HardSwish-SE and top-level
pre-add/resize successors, dependencies, diagnostics, and TensorFlow behavior
remain unchanged.

Implementation validation completed sequentially under `uv`:

- SiNet terminal and pre-add/resize orchestration, terminal-clamp,
  indexed-convergence and Singleton boundaries, remaining context composition,
  architecture, and pass-efficiency coverage: `357 passed in 19.62s`
- branch-changed broad suite plus the same SiNet terminal-layout coverage:
  `1533 passed in 27.45s`

These are unit, contract, and orchestration checks; this result propagation
does not claim a new model-corpus run. Stale terminal-clamp and pre-add/resize
boundary assertions were tightened to the exact retained targets.

At resume, audit `run_sinet_preadd_resize_recovery()` and its local helper. The
helper is both the middle callback of the two retained terminal-layout tuples
and a direct top-level call at three boundaries. Preserve all six child pass
IDs, callback wiring, direct-call order, and surrounding recovery sequences.
Commit and push only; do not create, reopen, or update a pull request.

## SiNet pre-add/resize result characterization checkpoint

`run_sinet_preadd_resize_recovery()` executes six fixed dictionary-returning
children in order: two residual affine repairs, Concat/Resize affine repair,
dual-Resize repair, tail-Concat repair, and Softmax-mask residual repair. The
last four receive the live LayoutState. `run_recovery_invocations()` already
returns all six dictionaries, while the phase runner and local helper discard
the tuple.

The helper is the middle callback of both retained SiNet terminal-layout tuples
and also has exactly three zero-argument direct calls. Those direct calls occur
after the terminal dequant bridge, after
`_very_late_sinet_layout_recovery_results`, and after final static-shape
reconciliation.

A strict expected-failure contract selects
`_terminal_sinet_preadd_resize_results`,
`_very_late_sinet_preadd_resize_results`, and
`_post_cleanup_sinet_preadd_resize_results` as the three direct retention
targets. It fixes the ordered `Tuple[Dict[str, int], ...]`, six pass IDs,
arguments and LayoutState keywords, callback identity, direct-call count, and
all three boundary pairs.

At implementation, transparently return the existing tuple through the phase
runner and local helper, then replace only the three raw direct expressions
with assignments. The two terminal-layout tuples will naturally retain this
tuple as their middle callback result; they remain unconsumed. Do not change a
child, callback identity, pass order, LayoutState wiring, surrounding recovery
calls, dependencies, diagnostics, or TensorFlow behavior. Do not add a result
consumer or guard.

Characterization validation completed sequentially under `uv`:

- SiNet pre-add/resize and terminal-layout orchestration, remaining context
  composition, indexed-convergence and Singleton boundaries, architecture, and
  pass-efficiency coverage: `350 passed, 1 xfailed in 17.83s`

The sole strict expected failure is the intentionally unimplemented six-result
propagation contract. Implement only that contract, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## SiNet pre-add/resize result propagation implementation checkpoint

`run_sinet_preadd_resize_recovery()` now returns its existing ordered
six-dictionary tuple, and the local helper transparently forwards it. The three
direct calls retain `_terminal_sinet_preadd_resize_results`,
`_very_late_sinet_preadd_resize_results`, and
`_post_cleanup_sinet_preadd_resize_results`.

All direct tuples are observation-only and have no consumer or guard. The same
helper remains the middle callback of both retained terminal-layout tuples;
their middle element is now the six-dictionary tuple instead of discarded
`None`, but both outer tuples remain unconsumed. All six children still execute
exactly once with unchanged pass IDs, arguments, and live LayoutState wiring.
Callback identity, direct-call order, surrounding recovery sequences,
dependencies, diagnostics, and TensorFlow behavior remain fixed.

Implementation validation completed sequentially under `uv`:

- SiNet pre-add/resize and terminal-layout orchestration, remaining context
  composition, indexed-convergence and Singleton boundaries, architecture, and
  pass-efficiency coverage: `351 passed in 18.37s`
- branch-changed broad suite plus the same SiNet pre-add/resize coverage:
  `1534 passed in 26.35s`

These are unit, contract, and orchestration checks; this result propagation
does not claim a new model-corpus run.

At resume, audit the post-cleanup
`_optimize_transpose_csp_attention_nhwc_chains()` result immediately after
`_post_cleanup_sinet_preadd_resize_results`. Preserve its other production
occurrence, live LayoutState wiring, and adjacent SA/PA MirrorPad propagation.
Commit and push only; do not create, reopen, or update a pull request.

## Post-cleanup CSP-attention result characterization checkpoint

The audit corrects the preceding resume note: CSP-attention has exactly one
production call, not multiple occurrences. Its lowerer wrapper dispatches the
single indexed owner and returns the fixed one-counter dictionary
`optimized_transpose_csp_attention_nhwc_chains`.

The owner performs indexed transactional-style rewrites, then calls
unused-tensor pruning unconditionally. LayoutState synchronization is guarded
by a positive rewrite count. The counter therefore remains stable observation
data but is incomplete evidence for cleanup-only pruning and must not control
later work.

The sole call follows `_post_cleanup_sinet_preadd_resize_results`, receives the
live Session LayoutState, and immediately precedes SA/PA MirrorPad propagation.
A strict expected-failure contract selects
`_post_cleanup_csp_attention_stats`. It fixes the one-call count, model and
LayoutState arguments, one-key schema, unconditional prune, positive-only
layout sync, captured predecessor, and SA/PA successor.

At implementation, replace only the raw production expression with an
assignment. Do not add a consumer or guard, and do not change the wrapper,
owner, schema, graph-index behavior, pruning, layout sync, surrounding
sequence, dependencies, diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- CSP result/schema semantics, SiNet pre-add/resize and terminal boundaries,
  architecture, and pass-efficiency coverage:
  `306 passed, 1 xfailed in 17.78s`
- concrete CSP-attention owner fixtures: `2 passed, 739 deselected in 0.61s`

The sole strict expected failure is the intentionally unimplemented retention
assignment. Implement only that assignment, rerun focused and branch-changed
broad gates sequentially, then commit and push only; do not create, reopen, or
update a pull request.

## Post-cleanup CSP-attention result retention implementation checkpoint

The sole production call now retains its unchanged one-counter dictionary as
`_post_cleanup_csp_attention_stats`. The dictionary remains observation-only
and has no consumer or guard because the owner prunes unused tensors
unconditionally while its counter reports rewrites only.

This is an assignment-only change. The lowerer wrapper, indexed owner, one-key
schema, unconditional prune, positive-only LayoutState synchronization, live
Session LayoutState, captured `_post_cleanup_sinet_preadd_resize_results`
predecessor, adjacent SA/PA MirrorPad successor, dependencies, diagnostics, and
TensorFlow behavior remain unchanged.

Implementation validation completed sequentially under `uv`:

- CSP result/schema semantics, SiNet pre-add/resize and terminal boundaries,
  architecture, and pass-efficiency coverage: `307 passed in 19.19s`
- concrete CSP-attention owner fixtures: `2 passed, 739 deselected in 0.59s`
- branch-changed broad suite plus the same CSP-attention coverage:
  `1530 passed in 26.74s`

These are unit, contract, owner-fixture, and orchestration checks; this result
retention does not claim a new model-corpus run.

At resume, audit the adjacent direct
`_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains()` result after
`_post_cleanup_csp_attention_stats`. Preserve the same owner selections inside
attention-recovery orchestration, the separate gate-layout boundary, live
LayoutState wiring, and the captured BatchMatMul affine-input successor. Commit
and push only; do not create, reopen, or update a pull request.

## Direct SA/PA MirrorPad result characterization checkpoint

The SA/PA MirrorPad wrapper dispatches one indexed owner and returns the fixed
one-counter dictionary
`optimized_transpose_sa_pa_mirrorpad_nhwc_propagation_chains`. The owner prunes
unused tensors and synchronizes LayoutState only when its rewrite count is
positive, so the counter covers every owner mutation path.

There are exactly two direct wrapper calls. The first is inside the
`optimize_layout_transpose_chains` guard between pre-add/mean/attention recovery
and reduced gate-layout recovery. The second follows
`_post_cleanup_csp_attention_stats` and precedes
`_post_sinet_batchmatmul_affine_input_stats`. Attention-gate/QDQ orchestration
selects the owner module directly as a third, distinct form.

A strict expected-failure contract selects
`_layout_opt_sa_pa_mirrorpad_stats` and
`_post_cleanup_sa_pa_mirrorpad_stats` for only the direct calls. It fixes the
two-call count, model and live LayoutState arguments, one-key schema,
positive-only pruning and layout sync, option guard, four boundaries, captured
neighbors, and independent orchestration selection.

At implementation, replace only the two raw direct expressions with
assignments. Do not add a consumer or guard and do not change the wrapper,
owner, schema, pruning, layout sync, orchestration selection, gate policy,
surrounding calls, dependencies, diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- SA/PA result/schema semantics and indexed owner fixtures, gate-layout and
  attention-recovery orchestration, CSP and BatchMatMul boundaries,
  architecture, and pass-efficiency coverage:
  `356 passed, 1 xfailed in 18.11s`

The sole strict expected failure is the intentionally unimplemented two-result
retention contract. Implement only those assignments, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## Earlier Split/Conv/Concat bridge result retention implementation checkpoint

The terminal-QKV call now retains
`_terminal_qkv_split_conv_concat_bridge_stats`, and the post-SiNet call retains
`_post_sinet_split_conv_concat_bridge_stats`. The late call remains retained as
`_terminal_split_conv_concat_bridge_stats`.

These are assignment-only observation changes. No consumer or guard was added.
The wrapper, indexed owner, one-key schema, positive-only pruning,
transactional rewrite behavior, live LayoutState, QKV/Singleton/SiNet
boundaries, existing late result, pass order, dependencies, diagnostics, and
TensorFlow behavior remain unchanged. The Singleton boundary call extractor
now accepts assigned predecessors.

Implementation validation completed sequentially under `uv`:

- indexed bridge owner, all three production occurrences, QKV and Singleton
  orchestration, HardSwish-SE late boundary, architecture, and pass-efficiency
  coverage: `401 passed in 18.59s`
- branch-changed broad suite plus the same three-occurrence coverage:
  `1502 passed in 25.50s`

These are unit, contract, and orchestration checks; this result retention does
not claim a new model-corpus run.

At resume, audit the guarded terminal Singleton/Reshape cluster result after
`_terminal_qkv_split_conv_concat_bridge_stats`. Preserve its exact policy and
all other production occurrences before propagating or retaining a result.
Commit and push only; do not create, reopen, or update a pull request.

## Terminal QKV result retention implementation checkpoint

The default-policy calls now retain `_terminal_qkv_attention_results` and
`_post_sinet_qkv_attention_results`. The existing late call remains
`late_qkv_results` and continues to feed the policy-aware summary with an
explicit net tensor-pruning delta.

These are assignment-only observation changes. The helper, QKV owner
selection, shared scope, policies, captured adj-flags predecessors, distinct
successors, late summary, option guard, pass order, dependencies, diagnostics,
and TensorFlow behavior remain unchanged. Neither new tuple has a consumer.

The first implementation gate reached a previously hidden test-helper
assumption after the former xfail passed: lowerer-wide statement scanning did
not tolerate `AnnAssign`. The call extractor now returns `None` for non-call
statements and supports both raw and assigned calls; production code was not
affected by that correction.

Implementation validation completed sequentially under `uv`:

- QKV policies and late summary, adj-flags owner fixture, both default-policy
  boundaries, architecture, terminal result contracts, and pass-efficiency
  coverage: `371 passed in 19.64s`
- branch-changed broad suite plus the same QKV/late-summary coverage:
  `1464 passed in 25.62s`

These are unit, contract, and orchestration checks; this result retention does
not claim a new model-corpus run.

At resume, audit the raw split/conv/concat bridge result after
`_terminal_qkv_attention_results` before moving further through the terminal
sequence. Preserve all three QKV result policies and the late summary. Commit
and push only; do not create, reopen, or update a pull request.

## Earlier Split/Conv/Concat bridge result characterization checkpoint

The indexed
`_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw()` owner
returns one fixed rewrite counter. Each count follows a successful
transactional rewrite, while unused-tensor pruning occurs only after at least
one rewrite. The dictionary is complete owner mutation evidence.

Three direct production calls exist. The late call is already retained as
`_terminal_split_conv_concat_bridge_stats`. The terminal call after
`_terminal_qkv_attention_results` and the post-SiNet call after the ReLU/Split/
Conv/Concat propagation remain raw.

A strict expected-failure contract selects
`_terminal_qkv_split_conv_concat_bridge_stats` and
`_post_sinet_split_conv_concat_bridge_stats` for only those raw calls. It fixes
the live Session LayoutState keyword, both predecessors and successors, total
three-call count, terminal option guard, and existing late retained target.

At implementation, replace only the two raw expressions with assignments. Do
not add a consumer or guard and do not change the wrapper, indexed owner,
one-key schema, positive-only pruning, transactional behavior, adjacent QKV/
Singleton/SiNet calls, existing late target, pass order, dependencies,
diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- indexed bridge owner, all three production occurrences, QKV and Singleton
  orchestration, HardSwish-SE late boundary, architecture, and pass-efficiency
  coverage: `400 passed, 1 xfailed in 18.27s`

The sole strict expected failure is the intentionally unimplemented two-result
retention contract. Implement only those assignments, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## BatchMatMul adj-flags result retention implementation checkpoint

The guarded terminal call now retains
`_terminal_batchmatmul_adj_flags_stats`, and the post-SiNet call retains
`_post_sinet_batchmatmul_adj_flags_stats`. Both preserve the dedicated owner's
unchanged complete one-counter mutation dictionary.

These are assignment-only observation changes. No consumer or guard was added.
The wrapper, owner, schema, positive-only pruning, captured reshape/SE
predecessors, QKV successors, option guard, pass order, dependencies,
diagnostics, and TensorFlow behavior remain unchanged. The QKV boundary test
helper now accepts either a raw call or an assigned call as its predecessor.

Implementation validation completed sequentially under `uv`:

- adj-flags and reshape/SE owner fixtures, both production boundaries, QKV
  orchestration, architecture, terminal result contracts, and pass-efficiency
  coverage: `372 passed in 19.41s`
- branch-changed broad suite plus the same two-boundary/QKV coverage:
  `1463 passed in 25.65s`

These are unit, contract, and orchestration checks; this result retention does
not claim a new model-corpus run.

At resume, audit the raw QKV attention cluster results immediately after the
newly retained adj-flags dictionaries. Preserve all QKV production policies and
callback ordering before propagating or capturing any result. Commit and push
only; do not create, reopen, or update a pull request.

## Terminal QKV result characterization checkpoint

`_run_qkv_attention_layout_pass_cluster()` already returns the ordered tuple
selected from optional layout-transpose, optional prefix, and required bridge
owners. There are three direct production calls. The late call is retained as
`late_qkv_results` and normalized with a separate net tensor-pruning delta;
the terminal and post-SiNet default-policy calls remain raw.

A strict expected-failure contract selects
`_terminal_qkv_attention_results` and
`_post_sinet_qkv_attention_results` for only those two raw calls. It fixes empty
arguments and keywords, captured terminal/post-SiNet adj-flags predecessors,
the split/concat and ReLU/split successors, total three-call count, and the
existing late `include_layout_transpose`/`include_prefix=False` policy.

At implementation, replace only the two raw default-policy expressions with
assignments. Do not summarize or consume the new tuples and do not alter the
helper, QKV owner selection, shared scope, late result/summary, adjacent calls,
option guard, pass order, dependencies, diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- QKV policies and late summary, adj-flags owner fixture, both raw production
  boundaries, architecture, terminal result contracts, and pass-efficiency
  coverage: `370 passed, 1 xfailed in 19.69s`

The sole strict expected failure is the intentionally unimplemented two-result
retention contract. Implement only those assignments, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## BatchMatMul reshape/SE result retention implementation checkpoint

The guarded terminal call now retains
`_terminal_batchmatmul_reshape_se_stats`, and the post-SiNet call retains
`_post_sinet_batchmatmul_reshape_se_stats`. Both preserve the dedicated owner's
unchanged one-counter dictionary.

These are assignment-only observation changes. Because the owner can prune
unused tensors while reporting zero rewrites, neither result is consumed or
used as guard evidence. The wrapper, owner, schema, unconditional pruning,
captured affine-input predecessors, shared adj-flags successor, option guard,
pass order, dependencies, diagnostics, and TensorFlow behavior remain
unchanged.

Implementation validation completed sequentially under `uv`:

- reshape/SE and affine-input owner fixtures, both production boundaries,
  architecture, terminal result contracts, and pass-efficiency coverage:
  `356 passed in 19.80s`
- branch-changed broad suite plus the same two-boundary coverage:
  `1461 passed in 25.43s`

These are unit, contract, and orchestration checks; this result retention does
not claim a new model-corpus run.

At resume, audit both raw BatchMatMul transpose-input-to-adj-flags owner calls
immediately after the newly retained reshape/SE results. Preserve the two
surrounding sequences and observation-only evidence rules. Commit and push
only; do not create, reopen, or update a pull request.

## BatchMatMul adj-flags result characterization checkpoint

`_optimize_batchmatmul_transpose_input_to_adj_flags()` dispatches one dedicated
owner and returns the fixed one-counter dictionary
`optimized_batchmatmul_transpose_input_to_adj_flags`. Every count follows the
input bypass or singleton-preserving RESHAPE conversion and matching
`adjX`/`adjY` toggle. Unused-tensor pruning runs only after at least one rewrite,
so the counter covers all owner mutation paths.

There are exactly two raw direct production calls. They follow
`_terminal_batchmatmul_reshape_se_stats` and
`_post_sinet_batchmatmul_reshape_se_stats`, respectively, and both immediately
precede `_run_qkv_attention_layout_pass_cluster()`.

A strict expected-failure contract selects distinct
`_terminal_batchmatmul_adj_flags_stats` and
`_post_sinet_batchmatmul_adj_flags_stats` targets. It fixes the two-call count,
model argument, empty keywords, both captured predecessors, both QKV
successors, and terminal option guard.

At implementation, replace only the two raw expressions with assignments. Do
not add a consumer or guard in this unit and do not change the wrapper, owner,
one-key schema, positive-only pruning, surrounding sequences, QKV policies,
pass order, dependencies, diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- adj-flags and reshape/SE owner fixtures, both production boundaries, QKV
  orchestration, architecture, terminal result contracts, and pass-efficiency
  coverage: `371 passed, 1 xfailed in 19.44s`

The sole strict expected failure is the intentionally unimplemented two-result
retention contract. Implement only those assignments, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## BatchMatMul affine-input result retention implementation checkpoint

The guarded terminal call now retains
`_terminal_batchmatmul_affine_input_stats`, and the later post-SiNet call
retains `_post_sinet_batchmatmul_affine_input_stats`. Both preserve the
unchanged one-counter dictionary from the dedicated owner.

These are assignment-only observation changes. Because the owner can prune
unused tensors while reporting zero rewrites, neither result has a consumer or
is used as guard evidence. The wrapper, owner, schema, unconditional pruning,
policy guard, predecessors, shared BatchMatMul reshape/SE successor, pass order,
dependencies, diagnostics, and TensorFlow behavior remain unchanged.

Implementation validation completed sequentially under `uv`:

- affine-input owner fixture, both production boundaries, terminal
  mean/attention, architecture, terminal result contracts, and pass-efficiency
  coverage: `369 passed in 18.78s`
- branch-changed broad suite plus the same two-boundary coverage:
  `1459 passed in 25.79s`

These are unit, contract, and orchestration checks; this result retention does
not claim a new model-corpus run.

At resume, audit both raw BatchMatMul reshape/SE owner calls immediately after
the newly retained affine-input results. Preserve both surrounding sequences
and observation-only constraints. Commit and push only; do not create, reopen,
or update a pull request.

## BatchMatMul reshape/SE result characterization checkpoint

`_optimize_batchmatmul_reshape_se_nhwc_chains()` dispatches one dedicated owner
and returns the fixed one-counter dictionary
`optimized_batchmatmul_reshape_se_nhwc_chains`. The owner unconditionally
prunes unused tensors after candidate processing, so its counter is incomplete
evidence for cleanup-only mutation and must remain observation-only.

There are exactly two raw direct production calls. They follow
`_terminal_batchmatmul_affine_input_stats` and
`_post_sinet_batchmatmul_affine_input_stats`, respectively, and both
immediately precede `_optimize_batchmatmul_transpose_input_to_adj_flags()`.

A strict expected-failure contract selects distinct
`_terminal_batchmatmul_reshape_se_stats` and
`_post_sinet_batchmatmul_reshape_se_stats` targets. It fixes the two-call count,
model argument, empty keywords, both captured predecessors, the terminal
option guard, and shared adj-flags successor.

At implementation, replace only the two raw expressions with assignments. Do
not add a guard or consumer and do not change the wrapper, owner, one-key
schema, unconditional pruning, surrounding sequences, option guard, pass
order, dependencies, diagnostics, or TensorFlow behavior.

Characterization validation completed sequentially under `uv`:

- reshape/SE and affine-input owner fixtures, both production boundaries,
  architecture, terminal result contracts, and pass-efficiency coverage:
  `355 passed, 1 xfailed in 19.62s`

The sole strict expected failure is the intentionally unimplemented two-result
retention contract. Implement only those assignments, rerun focused and
branch-changed broad gates sequentially, then commit and push only; do not
create, reopen, or update a pull request.

## Layout/reshape/attention prefix result characterization checkpoint

`run_layout_reshape_attention_recovery_prefix()` selects fifteen ordered
children. Its first slot now contains the complete nested nineteen-slot
layout-recovery result; the remaining slots preserve the pre-add, reshape,
attention, window-partition/reverse, unary-squeeze, and final squeeze-cleanup
owner results. The parent runner and zero-argument lowerer helper currently
discard the aggregate tuple.

There are exactly three direct calls in layout pass-set 1. Their boundaries are
layout-transpose cleanup/affine fold, duplicate-fanout cleanup/affine fold, and
QLinear recovery/InstanceNorm cleanup. None currently has a consumer, summary,
or guard.

A strict expected-failure contract instruments the nested layout result and
all fourteen remaining identities. It selects observation-only targets
`_layout_pass_set_1_initial_attention_recovery_results`,
`_layout_pass_set_1_post_binary_attention_recovery_results`, and
`_layout_pass_set_1_final_attention_recovery_results`. It freezes the shared
context, helper/runner return, exact three-call count, all boundary pairs, and
absence of consumers. Incomplete child counters and cleanup-only changes make
the aggregate unsafe as guard evidence.

Characterization validation completed sequentially under `uv`:

- attention-prefix result identity and production boundaries, QLinear,
  architecture, and pass-efficiency coverage: `303 passed, 1 xfailed in 18.24s`
- branch-changed broad suite: `1605 passed, 1 xfailed in 29.09s`

The sole strict expected failure is the intentionally unimplemented parent
result propagation. At implementation, return the existing tuple through the
runner/helper and replace only the three raw direct expressions with the
selected assignments. Do not consume a result or change a child, nested tuple,
context, order, boundary, summary, guard, dependency, public API, or TensorFlow
behavior. Validate sequentially, commit, and push only; do not create, reopen,
or update a pull request.

## Layout/reshape/attention prefix result propagation implementation checkpoint

`run_layout_reshape_attention_recovery_prefix()` and its zero-argument lowerer
helper now return the unchanged fifteen-slot tuple. The three direct calls
retain `_layout_pass_set_1_initial_attention_recovery_results`,
`_layout_pass_set_1_post_binary_attention_recovery_results`, and
`_layout_pass_set_1_final_attention_recovery_results` at their original
layout-cleanup/affine, duplicate-fanout/affine, and QLinear/InstanceNorm
boundaries.

All three tuples are unconsumed and observation-only. Slot zero preserves the
complete nested nineteen-slot layout-recovery result. The implementation adds
no consumer, guard, summary, graph scan, tuple copy, child invocation, context,
order, cleanup timing, dependency, public API, or TensorFlow import path.

Implementation validation completed sequentially under `uv`:

- attention-prefix result identity and all direct boundaries, QLinear,
  architecture, and pass-efficiency coverage: `304 passed in 19.89s`
- branch-changed broad suite: `1606 passed in 28.89s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
propagation does not claim a new model-corpus run.

At resume, audit both direct
`_run_qlinear_mean_concat_recovery_sequence()` results and its five-slot
runner/helper contract. Preserve the newly retained final attention result
that follows its second call, both distinct outer boundaries, and the
observation-only policy. Commit and push only; do not create, reopen, or update
a pull request.

## QLinear/mean/Concat parent result characterization checkpoint

`run_qlinear_mean_concat_recovery()` selects five ordered children:
mean/HardSigmoid/MulAdd, QLinear SiLU prefix, QLinear Concat/Conv, pre-quantized
Concat cleanup, and mean/MaxPool/Concat/Conv. All five return their existing
dictionaries. The runner and zero-argument lowerer helper currently discard
the aggregate tuple at both direct calls.

The pass-set-1 call lies between dequant-mean bridge recovery and the newly
retained `_layout_pass_set_1_final_attention_recovery_results`. The pass-set-2
call lies between the progress-description update and the newly retained
`_layout_pass_set_2_layout_recovery_prefix_results`.

A strict expected-failure contract instruments all five result identities and
selects observation-only targets
`_layout_pass_set_1_qlinear_mean_concat_results` and
`_layout_pass_set_2_qlinear_mean_concat_results`. It freezes the shared
`ModelIRPassContext`, runner/helper return, exact two-call count, both boundary
pairs, and absence of consumers. Child pruning means zero counters are not
complete evidence that no cleanup mutation occurred.

Characterization validation completed sequentially under `uv`:

- QLinear parent and child-owner contracts, both retained outer boundaries,
  layout recovery, architecture, and pass-efficiency coverage:
  `424 passed, 1 xfailed in 18.09s`
- branch-changed broad suite: `1606 passed, 1 xfailed in 28.82s`

The sole strict expected failure is the intentionally unimplemented ordered
result propagation. At implementation, return the existing five-slot tuple
through the runner/helper and replace only the two raw direct expressions with
the selected assignments. Do not consume either result or change a child,
context, pass order, surrounding retained target, summary, guard, dependency,
public API, or TensorFlow behavior. Validate sequentially, commit, and push
only; do not create, reopen, or update a pull request.

## QLinear/mean/Concat parent result propagation implementation checkpoint

`run_qlinear_mean_concat_recovery()` and its zero-argument lowerer helper now
return the existing five-slot tuple. The pass-set-1 call retains
`_layout_pass_set_1_qlinear_mean_concat_results`; the pass-set-2 call retains
`_layout_pass_set_2_qlinear_mean_concat_results`.

Both tuples are unconsumed and observation-only. The first remains between
dequant-mean recovery and `_layout_pass_set_1_final_attention_recovery_results`;
the second remains between the progress update and
`_layout_pass_set_2_layout_recovery_prefix_results`. No consumer, guard,
summary, graph scan, tuple copy, child invocation, context, pass order, cleanup
timing, dependency, public API, or TensorFlow import path was added.

Implementation validation completed sequentially under `uv`:

- QLinear parent and child-owner contracts, both retained outer boundaries,
  layout recovery, architecture, and pass-efficiency coverage:
  `425 passed in 19.97s`
- branch-changed broad suite: `1607 passed in 29.54s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
propagation does not claim a new model-corpus run.

At resume, audit the raw primary `run_layout_transpose_cleanup()` result
immediately before `_layout_pass_set_1_initial_attention_recovery_results`.
Inventory its other already retained and nested occurrences, preserve the live
LayoutState/diagnostics arguments and five-key cleanup semantics, and keep any
new result observation-only. Commit and push only; do not create, reopen, or
update a pull request.

## Primary layout-transpose cleanup result characterization checkpoint

`run_layout_transpose_cleanup()` returns the fixed five-key integer dictionary
containing `iterations`, identity removal, inverse-pair removal, inverse-fanout
removal, and consecutive-pair composition. Its owner prunes unused tensors and
synchronizes a supplied LayoutState after candidate execution.

There are three direct lowerer occurrences. The late-ConCat and very-late calls
already retain `_late_concat_transpose_layout_stats` and
`_very_late_layout_transpose_cleanup_stats`; only the primary pass-set-1 call
is a raw expression. The late-ConCat form alone receives
`late_concat_layout_state_scope`. Late-binary recovery has one independent
nested call and consumes only its four mutation fields.

A passing structural contract freezes the schema, all three direct forms, the
nested selection, live LayoutState/diagnostics routing, and shared-scope
difference. A strict expected-failure contract selects observation-only target
`_layout_pass_set_1_layout_transpose_cleanup_stats` between
`enable_duplicate_transpose_fanout_optimizations` and
`_layout_pass_set_1_initial_attention_recovery_results`, with no consumer.

Characterization validation completed sequentially under `uv`:

- primary, late-ConCat, very-late, and nested layout-transpose contracts,
  terminal layout orchestration, attention boundary, architecture, and
  pass-efficiency coverage: `370 passed, 1 xfailed in 19.20s`
- branch-changed broad suite including the new result contract:
  `1608 passed, 1 xfailed in 29.60s`

The sole strict expected failure is the intentionally unimplemented primary
assignment. Replace only that raw expression with the selected target. Do not
change the owner, schema, preflight, transaction, nested/retained occurrences,
arguments, shared scope, pass order, successor, guard, dependency, public API,
or TensorFlow behavior. Validate sequentially, commit, and push only; do not
create, reopen, or update a pull request.

## Primary layout-transpose cleanup result retention implementation checkpoint

The primary pass-set-1 call now retains the unchanged five-key dictionary as
`_layout_pass_set_1_layout_transpose_cleanup_stats`. It remains between the
duplicate-fanout policy assignment and
`_layout_pass_set_1_initial_attention_recovery_results`.

All three direct lowerer occurrences are now explicit assignments. The
late-ConCat call still receives its shared state scope, the very-late call keeps
its option guard, and the late-binary nested call still consumes its selected
mutation fields. The new primary target is unconsumed and observation-only.
No owner, schema, preflight, transaction, cleanup, LayoutState sync, argument,
scope, pass order, successor, guard, dependency, public API, or TensorFlow
import path changed.

Implementation validation completed sequentially under `uv`:

- primary, late-ConCat, very-late, and nested layout-transpose contracts,
  terminal layout orchestration, attention boundary, architecture, and
  pass-efficiency coverage: `371 passed in 20.03s`
- branch-changed broad suite: `1609 passed in 29.42s`
- post-cleanup focused result contract: `2 passed in 0.15s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run.

At resume, audit the sole raw `run_duplicate_fanout_cleanup()` result
immediately before `_layout_pass_set_1_post_binary_attention_recovery_results`.
Preserve its QDQ-dependent `include_transpose` policy, live LayoutState and
diagnostics, cleanup semantics, and following attention boundary. Commit and
push only; do not create, reopen, or update a pull request.

## Primary duplicate-fanout cleanup result characterization checkpoint

`run_duplicate_fanout_cleanup()` returns a policy-dependent dictionary. With
`include_transpose=False` it contains only
`removed_duplicate_reshape_fanout`; enabling Transpose adds
`removed_duplicate_transpose_fanout`. Both selected passes remain separately
transactional and share the runner's pass state.

The lowerer has one direct call. It forwards
`enable_duplicate_transpose_fanout_optimizations`, which is fixed as
`not has_qdq_ops`, plus the live LayoutState and diagnostics. The same owner is
selected independently by duplicate/PReLU, singleton/consecutive-Reshape, and
Singleton/Reshape orchestration.

A passing contract freezes both schemas, transactional pass IDs, and all three
nested selections. A strict expected-failure contract selects observation-only
target `_layout_pass_set_1_duplicate_fanout_stats`, exact arguments, the
conditional binary-bridge predecessor, the retained post-binary attention
successor, sole direct-call count, and absence of a consumer.

Characterization validation completed sequentially under `uv`:

- duplicate-fanout owner and policy schemas, three nested selections, direct
  boundary, graph cleanup, Singleton orchestration, architecture, and
  pass-efficiency coverage: `378 passed, 1 xfailed in 18.47s`
- branch-changed broad suite including the new result contract:
  `1610 passed, 1 xfailed in 28.74s`

The sole strict expected failure is the intentionally unimplemented direct
assignment. Replace only the raw expression with the selected target. Do not
change either schema, pass selection, preflight, transaction, QDQ policy,
nested occurrence, live context, pass order, surrounding boundary, guard,
dependency, public API, or TensorFlow behavior. Keep the result unconsumed,
validate sequentially, commit, and push only; do not create, reopen, or update
a pull request.

## Primary duplicate-fanout cleanup result retention implementation checkpoint

The sole direct call now retains its unchanged policy-dependent dictionary as
`_layout_pass_set_1_duplicate_fanout_stats`. It remains after the conditional
binary-bridge block and before
`_layout_pass_set_1_post_binary_attention_recovery_results`.

The target is unconsumed and observation-only. `not has_qdq_ops` still controls
whether the Transpose pass and its second result field are present; the live
LayoutState and diagnostics are unchanged. The three independent orchestration
selections retain their own policy and shared-state routing. No schema,
preflight, transaction, pass selection, nested occurrence, context, order,
guard, dependency, public API, or TensorFlow import path changed.

Implementation validation completed sequentially under `uv`:

- duplicate-fanout owner and policy schemas, three nested selections, direct
  boundary, graph cleanup, Singleton orchestration, architecture, and
  pass-efficiency coverage: `379 passed in 18.50s`
- branch-changed broad suite: `1611 passed in 29.45s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run.

At resume, audit the raw direct
`_optimize_transpose_dequantize_mean_quantize_bridges()` result immediately
before `_layout_pass_set_1_qlinear_mean_concat_results`. Preserve its owner
schema, cleanup semantics, sole production occurrence, and QLinear boundary.
Commit and push only; do not create, reopen, or update a pull request.

## Dequantize/mean/quantize bridge result characterization checkpoint

`_optimize_transpose_dequantize_mean_quantize_bridges()` returns the fixed
one-key dictionary `moved_transpose_dequantize_mean_quantize_bridges`. The owner
prunes unused tensors in both missing-required-type early exits and after
candidate processing. Its lowerer wrapper transparently forwards only ModelIR.

There is one production call. It is a raw expression between the retained
`_layout_pass_set_1_safe_binary_results` and
`_layout_pass_set_1_qlinear_mean_concat_results`. A zero rewrite count does not
exclude cleanup-only pruning, so the result cannot safely drive a guard.

A passing contract freezes wrapper forwarding, the one-key schema, both early
zero-return paths, and all three prune sites. A strict expected-failure
contract selects observation-only target
`_layout_pass_set_1_dequant_mean_quantize_stats`, exact ModelIR-only call,
sole-call count, both retained boundaries, and absence of a consumer.

Characterization validation completed sequentially under `uv`:

- bridge owner fixtures, wrapper/schema/prune contract, safe-binary and QLinear
  boundaries, architecture, and pass-efficiency coverage:
  `357 passed, 1 xfailed in 17.88s`
- branch-changed broad suite including the new result contract:
  `1612 passed, 1 xfailed in 29.37s`

The sole strict expected failure is the intentionally unimplemented direct
assignment. Replace only the raw expression with the selected target. Do not
change the wrapper, owner, schema, graph-index allocation, cleanup, pass order,
surrounding results, guard, dependency, public API, or TensorFlow behavior.
Keep the value unconsumed, validate sequentially, commit, and push only; do not
create, reopen, or update a pull request.

## Dequantize/mean/quantize bridge result retention implementation checkpoint

The sole production call now retains its unchanged one-key dictionary as
`_layout_pass_set_1_dequant_mean_quantize_stats`. It remains between
`_layout_pass_set_1_safe_binary_results` and
`_layout_pass_set_1_qlinear_mean_concat_results`.

The target is unconsumed and observation-only because a zero rewrite counter
does not exclude the owner's cleanup-only pruning. No wrapper, owner, schema,
graph-index allocation, early/final prune path, pass order, surrounding
retained result, guard, dependency, public API, or TensorFlow import path
changed.

Implementation validation completed sequentially under `uv`:

- bridge owner fixtures, wrapper/schema/prune contract, safe-binary and QLinear
  boundaries, architecture, and pass-efficiency coverage:
  `358 passed in 18.10s`
- branch-changed broad suite: `1613 passed in 29.32s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run.

At resume, audit both production occurrences of
`_optimize_transpose_instancenorm_prepost_nhwc_chains()`, including the direct
call after `_layout_pass_set_1_final_attention_recovery_results` and the later
conditional form. Preserve their distinct arguments, guards, and surrounding
boundaries. Commit and push only; do not create, reopen, or update a pull
request.

## InstanceNorm pre/post result characterization checkpoint

`_optimize_transpose_instancenorm_prepost_nhwc_chains()` dispatches four
indexed decomposed-InstanceNorm tail owners in graph order, caps total rewrites
at 32, and returns the one-key dictionary
`optimized_transpose_instancenorm_prepost_nhwc_chains`.

There are two production forms. The raw pass-set-1 call receives ModelIR and
the live LayoutState. The later form runs inside `for _ in range(2)`, receives
`normalization_graph_index`, immediately extracts the same counter, and feeds
the existing multi-owner convergence break. That consumed form must remain
unchanged.

A passing contract freezes the schema, all four owner dispatches, rewrite cap,
both argument forms, exact call count, loop placement, and consumed `.get()`
expression. A strict expected-failure contract selects observation-only target
`_layout_pass_set_1_instancenorm_prepost_stats` between
`_layout_pass_set_1_final_attention_recovery_results` and direct
`run_squeeze_reshape_identity_cleanup()`, with no consumer.

Characterization validation completed sequentially under `uv`:

- four InstanceNorm owner families, direct and consumed-loop forms, final
  attention boundary, architecture, and pass-efficiency coverage:
  `536 passed, 1 xfailed in 18.21s`
- branch-changed broad suite including the new result contract:
  `1614 passed, 1 xfailed in 29.38s`

The sole strict expected failure is the intentionally unimplemented direct
assignment. Replace only that raw expression with the selected target. Do not
change the dispatcher, schema, owner order, rewrite cap, GraphIndex/LayoutState
routing, later consumed form, convergence guard, surrounding calls,
dependency, public API, or TensorFlow behavior. Keep the direct result
unconsumed, validate sequentially, commit, and push only; do not create,
reopen, or update a pull request.

## InstanceNorm pre/post result retention implementation checkpoint

The direct pass-set-1 call now retains its unchanged one-key dictionary as
`_layout_pass_set_1_instancenorm_prepost_stats`. It remains between
`_layout_pass_set_1_final_attention_recovery_results` and direct
`run_squeeze_reshape_identity_cleanup()`.

The target is unconsumed and observation-only. The later two-iteration form is
unchanged: it still receives `normalization_graph_index`, extracts the counter,
and contributes to the existing convergence break. No dispatcher, owner order,
rewrite cap, schema, GraphIndex/LayoutState routing, convergence guard, pass
order, dependency, public API, or TensorFlow import path changed.

The first focused implementation run exposed one stale architecture assertion
that required the third attention-prefix successor to remain a raw expression
(`536 passed, 1 failed`). It now requires the same call identity on the new
InstanceNorm target. This was a structural contract update, not a production
regression.

Implementation validation completed sequentially under `uv`:

- four InstanceNorm owner families, direct and consumed-loop forms, final
  attention boundary, architecture, and pass-efficiency coverage:
  `537 passed in 20.50s`
- branch-changed broad suite: `1615 passed in 28.89s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run.

At resume, audit the direct `run_squeeze_reshape_identity_cleanup()` result
immediately after `_layout_pass_set_1_instancenorm_prepost_stats` and inventory
all other direct or orchestration-selected policy forms before changing it.
Preserve `include_unary_passthrough=True`, live LayoutState/diagnostics, and the
following final suffix boundary. Commit and push only; do not create, reopen,
or update a pull request.

## Squeeze/Reshape identity result characterization checkpoint

`run_squeeze_reshape_identity_cleanup()` returns
`optimized_squeeze_reshape_identity_chains` for the identity-only policy and
adds `optimized_squeeze_unary_reshape_passthrough_chains` when
`include_unary_passthrough=True`. Both pass specs remain transactional and
ordered unary-first when both are enabled.

The lowerer has three raw direct calls, all using the two-key policy with the
live LayoutState and diagnostics: pass-set 1 after retained InstanceNorm,
core cleanup after dynamic-Reshape resolution, and pass-set 2 after the
two-iteration normalization convergence loop. The attention prefix selects the
same two-key policy; Singleton/Reshape independently selects identity-only.

A passing contract freezes both schemas, all three exact direct calls, and both
nested selections. A strict expected-failure contract selects observation-only
targets `_layout_pass_set_1_squeeze_reshape_identity_stats`,
`_core_cleanup_squeeze_reshape_identity_stats`, and
`_layout_pass_set_2_squeeze_reshape_identity_stats`. It fixes the InstanceNorm/
final-suffix, dynamic-Reshape/prune, and normalization-loop/prune boundaries
and absence of consumers.

Characterization validation completed sequentially under `uv`:

- both cleanup policies, three direct boundaries, attention/Singleton nested
  selections, InstanceNorm and suffix orchestration, architecture, and
  pass-efficiency coverage: `366 passed, 1 xfailed in 18.27s`
- branch-changed broad suite including the new result contract:
  `1616 passed, 1 xfailed in 29.63s`

The sole strict expected failure is the intentionally unimplemented three-call
retention contract. Replace only the three raw expressions with the selected
targets. Do not change either schema, pass selection/order, preflight,
transaction, policy arguments, nested occurrences, live context, normalization
loop, surrounding calls, guard, dependency, public API, or TensorFlow behavior.
Keep all results unconsumed, validate sequentially, commit, and push only; do
not create, reopen, or update a pull request.

## Squeeze/Reshape identity result retention implementation checkpoint

The three raw lowerer expressions now retain their unchanged two-key results
as `_layout_pass_set_1_squeeze_reshape_identity_stats`,
`_core_cleanup_squeeze_reshape_identity_stats`, and
`_layout_pass_set_2_squeeze_reshape_identity_stats`. All three targets remain
unconsumed and observation-only. The identity-only Singleton selection and the
unary-enabled attention selection are unchanged.

No result schema, pass selection or order, preflight, transaction boundary,
rewrite cap, argument, live LayoutState/diagnostics route, normalization loop,
surrounding production call, guard, dependency, public API, or TensorFlow
boundary changed.

The initial focused implementation gate reported `366 passed, 1 failed`: the
final-attention suffix test still required the cleanup predecessor to be an
`ast.Expr`. It now accepts either direct-call statement form and continues to
assert the exact cleanup call and suffix boundary. This was a stale structural
contract, not a production regression.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `2 passed in 0.54s`
- both cleanup policies, three direct boundaries, attention/Singleton nested
  selections, InstanceNorm and suffix orchestration, architecture, and
  pass-efficiency coverage: `367 passed in 18.11s`
- branch-changed broad suite: `1617 passed in 29.27s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run. At resume, inventory the next
discarded direct pass result and all of its nested or consumed forms before
selecting another boundary. Commit and push only; do not create, reopen, or
update a pull request.

## Quantized-PReLU direct result characterization checkpoint

`run_quantized_prelu_cleanup()` returns a fixed four-key dictionary for its
transpose/quantize bridge, transpose bridge, quantize fusion, and depthwise
fusion passes. The default direct call enables all four transactional passes in
priorities 10 through 40.

The lowerer has one raw direct call with the live LayoutState and diagnostics,
after `_layout_pass_set_1_attention_gate_qdq_results` and before
`_optimize_dequant_transposeconv_quantize_chains`. The duplicate-fanout/
quantized-PReLU orchestration separately selects the same callback with its
shared `ModelIRPassStateScope`; that nested invocation remains unchanged.

A passing contract freezes the four-key schema, exact direct arguments, sole
direct occurrence, nested invocation arguments, and both boundary calls. A
strict expected-failure contract selects the unconsumed observation-only target
`_layout_pass_set_1_quantized_prelu_stats`.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `1 passed, 1 xfailed in 0.57s`
- quantized-PReLU owners, duplicate orchestration, attention recovery and
  quantized suffix boundaries, architecture, and pass-efficiency coverage:
  `325 passed, 1 xfailed in 17.98s`
- branch-changed broad suite including the new result contract:
  `1618 passed, 1 xfailed in 29.14s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented direct-result
retention contract. Replace only the raw direct expression with
`_layout_pass_set_1_quantized_prelu_stats`. Do not change the result schema,
pass selection/order, preflight, transaction, state scope, direct arguments,
nested invocation, surrounding calls, guard, dependency, public API, or
TensorFlow behavior. Keep the result unconsumed, validate sequentially, commit,
and push only; do not create, reopen, or update a pull request.

## Quantized-PReLU direct result retention implementation checkpoint

The sole raw lowerer call now retains its unchanged four-key dictionary as
`_layout_pass_set_1_quantized_prelu_stats`. The target remains unconsumed and
observation-only. The duplicate-fanout/quantized-PReLU orchestration still
selects the callback with the same shared `ModelIRPassStateScope`.

No result schema, pass implementation, default selection or priority,
preflight, transaction boundary, state scope, direct argument, nested
invocation, surrounding production call, guard, dependency, public API, or
TensorFlow boundary changed.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `2 passed in 0.53s`
- quantized-PReLU owners, duplicate orchestration, attention recovery and
  quantized suffix boundaries, architecture, and pass-efficiency coverage:
  `326 passed in 17.98s`
- branch-changed broad suite: `1619 passed in 29.96s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run. At resume, audit the direct
`run_quantized_reshape_cleanup()` result together with its nested quantized-
suffix selection before changing that boundary. Commit and push only; do not
create, reopen, or update a pull request.

## Quantized-Reshape direct result characterization checkpoint

`run_quantized_reshape_cleanup()` returns the fixed one-key dictionary
`folded_dequant_reshape_quantize_chains` from one transactional layout pass.

The lowerer has one raw direct call with the live LayoutState and diagnostics,
after `_optimize_dequant_transposeconv_quantize_chains` and before
`_layout_pass_set_1_quantized_activation_binary_results`. The layout/attention
quantized suffix separately selects the same callback at its fixed index with
the same live LayoutState and diagnostics.

A passing contract freezes the one-key schema, exact direct arguments, sole
direct occurrence, nested suffix index/context flags, and both boundary calls.
A strict expected-failure contract selects the unconsumed observation-only
target `_layout_pass_set_1_quantized_reshape_stats`.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `1 passed, 1 xfailed in 0.56s`
- quantized-Reshape owner, quantized suffix and activation boundaries,
  quantized-PReLU predecessor, architecture, and pass-efficiency coverage:
  `314 passed, 1 xfailed in 18.59s`
- branch-changed broad suite including the new result contract:
  `1620 passed, 1 xfailed in 29.58s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented direct-result
retention contract. Replace only the raw direct expression with
`_layout_pass_set_1_quantized_reshape_stats`. Do not change the result schema,
pass selection/order, preflight, transaction, direct arguments, nested suffix,
surrounding calls, guard, dependency, public API, or TensorFlow behavior. Keep
the result unconsumed, validate sequentially, commit, and push only; do not
create, reopen, or update a pull request.

## Quantized-Reshape direct result retention implementation checkpoint

The sole raw lowerer call now retains its unchanged one-key dictionary as
`_layout_pass_set_1_quantized_reshape_stats`. The target remains unconsumed and
observation-only. The layout/attention quantized suffix still selects the same
callback with the same live LayoutState and diagnostics.

No result schema, pass implementation, selection or order, preflight,
transaction boundary, direct argument, nested suffix invocation, surrounding
production call, guard, dependency, public API, or TensorFlow boundary changed.

The initial focused implementation gate reported `314 passed, 1 failed`: the
quantized-activation architecture test still required its cleanup predecessor
to be an `ast.Expr`. It now accepts either direct-call statement form and
continues to assert the exact predecessor call. This was a stale structural
contract, not a production regression.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `2 passed in 0.55s`
- quantized-Reshape owner, quantized suffix and activation boundaries,
  quantized-PReLU predecessor, architecture, and pass-efficiency coverage:
  `315 passed in 20.48s`
- branch-changed broad suite: `1621 passed in 30.14s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run. At resume, inventory all
direct and nested `_optimize_dequant_transposeconv_quantize_chains()` results
before changing that adjacent boundary. Commit and push only; do not create,
reopen, or update a pull request.

## Dequant-TransposeConv direct result characterization checkpoint

`_optimize_dequant_transposeconv_quantize_chains()` returns the fixed one-key
dictionary `folded_dequant_transposeconv_quantize_chains`. Candidate-missing
early exits can prune unused tensors and still return zero, so this counter is
not complete mutation evidence.

The lowerer has two raw direct calls with the live LayoutState. Pass-set 1 lies
between retained quantized-PReLU and quantized-Reshape results; pass-set 2 lies
between retained attention-gate/QDQ and quantized-activation results. The
layout/attention quantized suffix independently selects the same callback at
its fixed index with the live LayoutState.

A passing contract freezes the one-key schema, both exact direct calls, nested
suffix index/context flag, and all four retained-neighbor boundaries. A strict
expected-failure contract selects the unconsumed observation-only targets
`_layout_pass_set_1_dequant_transposeconv_quantize_stats` and
`_layout_pass_set_2_dequant_transposeconv_quantize_stats`.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `1 passed, 1 xfailed in 0.58s`
- indexed owner behavior, both direct boundaries, attention/quantized suffix,
  quantized recovery, adjacent result contracts, architecture, and pass-
  efficiency coverage: `386 passed, 1 xfailed in 18.24s`
- branch-changed broad suite including the new result contract:
  `1622 passed, 1 xfailed in 30.56s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented two-result
retention contract. Replace only the two raw direct expressions with the
selected targets. Do not change the result schema, GraphIndex construction or
mutation, pruning, call order, direct arguments, nested suffix, surrounding
calls, guard, dependency, public API, or TensorFlow behavior. Keep both results
unconsumed, validate sequentially, commit, and push only; do not create,
reopen, or update a pull request.

## Dequant-TransposeConv direct result retention implementation checkpoint

The two raw lowerer expressions now retain their unchanged one-key dictionaries
as `_layout_pass_set_1_dequant_transposeconv_quantize_stats` and
`_layout_pass_set_2_dequant_transposeconv_quantize_stats`. Both targets remain
unconsumed and observation-only. The layout/attention quantized suffix still
selects the same callback with the same live LayoutState.

No result schema, GraphIndex construction, mutation, pruning, call order,
direct argument, nested suffix invocation, surrounding production call, guard,
dependency, public API, or TensorFlow boundary changed. Candidate-missing
pruning can still accompany a zero counter, so these retained values are not
complete mutation evidence.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `2 passed in 0.53s`
- indexed owner behavior, both direct boundaries, attention/quantized suffix,
  quantized recovery, adjacent result contracts, architecture, and pass-
  efficiency coverage: `387 passed in 18.36s`
- branch-changed broad suite: `1623 passed in 29.53s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run. At resume, re-inventory the
remaining raw pass expressions and select the next complete owner family before
changing another boundary. Commit and push only; do not create, reopen, or
update a pull request.

## Affine-chain fold direct result characterization checkpoint

`optimize_fold_mul_add_mul_affine_chains()` returns the fixed one-key dictionary
`optimized_fold_mul_add_mul_affine_chains` from an indexed owner with a default
32-rewrite cap.

The lowerer wrapper has two raw direct calls with the live LayoutState. The
first follows retained initial-attention recovery and precedes affine pre/post
layout cleanup; the second follows retained post-binary attention recovery and
precedes retained attention/quantized suffix recovery. Terminal affine/Concat/
Split recovery separately selects the public callback with the live
LayoutState; that nested result remains consumed by the existing terminal
mutation summary.

A passing contract freezes the one-key schema, both exact direct calls, the
nested terminal index/layout flag, and all four boundaries. A strict expected-
failure contract selects the unconsumed observation-only direct targets
`_layout_pass_set_1_initial_affine_chain_fold_stats` and
`_layout_pass_set_1_post_binary_affine_chain_fold_stats` without changing the
consumed nested form.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `1 passed, 1 xfailed in 0.56s`
- indexed owner behavior, terminal summary/orchestration, both attention
  boundaries, architecture, and pass-efficiency coverage:
  `409 passed, 1 xfailed in 19.22s`
- branch-changed broad suite including the new result contract:
  `1624 passed, 1 xfailed in 31.32s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented two-result
retention contract. Replace only the two raw wrapper-call expressions with the
selected targets. Do not change the result schema, GraphIndex ownership,
rewrite cap, candidate handling, layout synchronization, direct arguments,
terminal selection or summary, surrounding calls, dependency, public API, or
TensorFlow behavior. Keep both direct results unconsumed, validate
sequentially, commit, and push only; do not create, reopen, or update a pull
request.

## Affine-chain fold direct result retention implementation checkpoint

The two raw lowerer expressions now retain their unchanged one-key dictionaries
as `_layout_pass_set_1_initial_affine_chain_fold_stats` and
`_layout_pass_set_1_post_binary_affine_chain_fold_stats`. Both targets remain
unconsumed and observation-only. Terminal affine/Concat/Split recovery still
selects the public callback and consumes its nested result in the same mutation
summary.

No result schema, GraphIndex ownership, rewrite cap, candidate handling, layout
synchronization, direct argument, terminal selection or summary, surrounding
production call, dependency, public API, or TensorFlow boundary changed.

The initial focused implementation gate reported `409 passed, 1 failed`: the
attention-prefix architecture test still expected both affine successors to
have no assignment targets. It now requires the two selected targets while
continuing to assert the same successor calls and order. This was a stale
structural contract, not a production regression.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `2 passed in 0.55s`
- indexed owner behavior, terminal summary/orchestration, both attention
  boundaries, architecture, and pass-efficiency coverage:
  `410 passed in 20.86s`
- branch-changed broad suite: `1625 passed in 30.08s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run. At resume, inventory all
direct and nested `_optimize_transpose_mul_add_const_prepost_nhwc_chains()`
forms before changing the adjacent affine-pre/post boundary. Commit and push
only; do not create, reopen, or update a pull request.

## Affine pre/post direct result characterization checkpoint

`optimize_transpose_mul_add_const_prepost_nhwc_chains()` returns the fixed
one-key dictionary `optimized_transpose_mul_add_const_prepost_nhwc_chains` from
an indexed owner with a default 32-rewrite cap. Candidate-missing and normal
exits both prune unused tensors, so a zero counter is not complete mutation
evidence.

The lowerer wrapper has three calls with the live LayoutState. The initial and
no-layout fallback calls are raw; the final no-layout call already retains
`_no_layout_final_affine_prepost_stats`. Terminal affine/Concat/Split,
attention/quantized suffix, and pre-add attention recovery each select the
public callback with LayoutState. Late-binary recovery separately retains and
consumes the same callback result inside its composite statistics.

A passing contract freezes the one-key schema, all three exact lowerer calls,
all initial/fallback/final boundaries, three declarative selection indices and
layout flags, and the consumed late-binary form. A strict expected-failure
contract selects `_layout_pass_set_1_affine_prepost_stats` and
`_no_layout_fallback_affine_prepost_stats` for the two raw calls while
preserving `_no_layout_final_affine_prepost_stats`.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `1 passed, 1 xfailed in 0.57s`
- indexed owner, terminal/attention/suffix/late-binary routes, initial and
  no-layout boundaries, final validation, architecture, and pass-efficiency
  coverage: `495 passed, 1 xfailed in 19.89s`
- branch-changed broad suite including the new result contract:
  `1626 passed, 1 xfailed in 30.38s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented two-result
retention contract. Replace only the initial and fallback raw expressions with
the selected targets. Do not change the result schema, GraphIndex ownership,
rewrite cap, candidate handling, pruning, layout synchronization, direct
arguments or conditions, existing final target, nested selections, late-binary
consumer, surrounding calls, dependency, public API, or TensorFlow behavior.
Keep both new results unconsumed, validate sequentially, commit, and push only;
do not create, reopen, or update a pull request.

## Affine pre/post direct result retention implementation checkpoint

The initial and no-layout fallback raw expressions now retain their unchanged
one-key dictionaries as `_layout_pass_set_1_affine_prepost_stats` and
`_no_layout_fallback_affine_prepost_stats`. Both targets remain unconsumed and
observation-only. The final lowerer call still retains
`_no_layout_final_affine_prepost_stats`; all three declarative orchestration
selections and the consumed late-binary form remain unchanged.

No result schema, GraphIndex ownership, rewrite cap, candidate handling,
pruning, layout synchronization, direct argument or condition, existing final
target, nested selection, late-binary consumer, surrounding production call,
dependency, public API, or TensorFlow boundary changed. A zero counter remains
insufficient to infer absence of mutation.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `2 passed in 0.56s`
- indexed owner, terminal/attention/suffix/late-binary routes, initial and
  no-layout boundaries, final validation, architecture, and pass-efficiency
  coverage: `496 passed in 20.88s`
- branch-changed broad suite: `1627 passed in 30.99s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run. At resume, re-inventory the
remaining raw pass expressions and select the next complete owner family before
changing another boundary. Commit and push only; do not create, reopen, or
update a pull request.

## Pre-unary affine fan-out direct result characterization checkpoint

`optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains()` returns
the fixed one-key dictionary
`optimized_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains`. The owner
always prunes unused tensors on exit, so a zero counter is not complete mutation
evidence.

The lowerer wrapper has one raw model-only call between retained affine
pre/post statistics and direct mean-affine cleanup. The attention/quantized
suffix and pre-add attention recovery independently select the public callback
with the same model-only contract.

A passing contract freezes the one-key schema, exact direct argument, sole
direct occurrence, both declarative indices and empty keyword contracts, and
the two boundary calls. A strict expected-failure contract selects the
unconsumed observation-only target
`_layout_pass_set_1_pre_unary_affine_fanout_stats`.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `1 passed, 1 xfailed in 0.58s`
- owner behavior, both attention orchestration routes, adjacent affine owners,
  architecture, and pass-efficiency coverage:
  `323 passed, 1 xfailed in 18.08s`
- branch-changed broad suite including the new result contract:
  `1628 passed, 1 xfailed in 30.84s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented direct-result
retention contract. Replace only the raw wrapper-call expression with the
selected target. Do not change the result schema, producer/consumer maps,
constant handling, pruning, direct argument, nested selections, surrounding
calls, dependency, public API, or TensorFlow behavior. Keep the result
unconsumed, validate sequentially, commit, and push only; do not create,
reopen, or update a pull request.

## Pre-unary affine fan-out direct result retention implementation checkpoint

The sole raw lowerer expression now retains its unchanged one-key dictionary as
`_layout_pass_set_1_pre_unary_affine_fanout_stats`. The target remains
unconsumed and observation-only. The attention/quantized suffix and pre-add
attention recovery still select the public callback with their model-only
contracts.

No result schema, producer/consumer-map construction, constant handling,
pruning, direct argument, nested selection, surrounding production call,
dependency, public API, or TensorFlow boundary changed. A zero counter remains
insufficient to infer absence of mutation.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `2 passed in 0.57s`
- owner behavior, both attention orchestration routes, adjacent affine owners,
  architecture, and pass-efficiency coverage: `324 passed in 18.41s`
- branch-changed broad suite: `1629 passed in 30.91s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run. At resume, inventory all
direct and nested `_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains()`
forms before changing the adjacent mean-affine boundary. Commit and push only;
do not create, reopen, or update a pull request.

## Mean-affine pre/post direct result characterization checkpoint

`optimize_transpose_mean_mul_add_const_prepost_nhwc_chains()` returns the fixed
one-key dictionary
`optimized_transpose_mean_mul_add_const_prepost_nhwc_chains`. The owner always
prunes unused tensors on exit, so a zero counter is not complete mutation
evidence.

The lowerer wrapper has one raw model-only call between retained pre-unary
affine fan-out statistics and retained mean-attention results. The attention/
quantized suffix and pre-add attention recovery independently select the public
callback with the same model-only contract.

A passing contract freezes the one-key schema, exact direct argument, sole
direct occurrence, both declarative indices and empty keyword contracts, and
the two retained-target boundaries. A strict expected-failure contract selects
the unconsumed observation-only target
`_layout_pass_set_1_mean_affine_prepost_stats`.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `1 passed, 1 xfailed in 0.61s`
- owner behavior, both attention orchestration routes, adjacent affine and
  mean-attention owners, architecture, and pass-efficiency coverage:
  `328 passed, 1 xfailed in 18.26s`
- branch-changed broad suite including the new result contract:
  `1630 passed, 1 xfailed in 30.30s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented direct-result
retention contract. Replace only the raw wrapper-call expression with the
selected target. Do not change the result schema, producer/consumer maps,
constant/axis handling, pruning, direct argument, nested selections,
surrounding calls, dependency, public API, or TensorFlow behavior. Keep the
result unconsumed, validate sequentially, commit, and push only; do not create,
reopen, or update a pull request.

## Mean-affine pre/post direct result retention implementation checkpoint

The sole raw lowerer expression now retains its unchanged one-key dictionary as
`_layout_pass_set_1_mean_affine_prepost_stats`. The target remains unconsumed
and observation-only. The attention/quantized suffix and pre-add attention
recovery still select the public callback with their model-only contracts.

No result schema, producer/consumer-map construction, constant or axis
handling, pruning, direct argument, nested selection, surrounding production
call, dependency, public API, or TensorFlow boundary changed. A zero counter
remains insufficient to infer absence of mutation.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `2 passed in 0.54s`
- owner behavior, both attention orchestration routes, adjacent affine and
  mean-attention owners, architecture, and pass-efficiency coverage:
  `329 passed in 18.61s`
- branch-changed broad suite: `1631 passed in 30.12s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run. At resume, re-inventory the
remaining raw pass expressions and select the next complete owner family before
changing another boundary. Commit and push only; do not create, reopen, or
update a pull request.

## Transpose-binary bridge direct result characterization checkpoint

`optimize_transpose_binary_bridges()` returns the fixed two-key dictionary
`removed_transpose_binary_bridges` and
`removed_transpose_binary_asymmetric_bridges` from an indexed owner with a
default 32-rewrite cap. Zero-cap and normal exits prune unused tensors, so zero
counters are not complete mutation evidence.

The lowerer wrapper has one raw call under
`enable_transpose_binary_bridge_optimizations`, with the live LayoutState and
no else path. The guard lies between retained quantized-activation recovery and
duplicate-fanout statistics. No nested selection of this owner exists.

A passing contract freezes the two-key schema, exact direct call, feature
guard, no-else policy, sole owner occurrence, and both retained-target
boundaries. A strict expected-failure contract selects the unconsumed
observation-only target `_layout_pass_set_1_transpose_binary_bridge_stats`.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `1 passed, 1 xfailed in 0.60s`
- indexed owner and recovery behavior, quantized-activation and duplicate-
  fanout boundaries, architecture, and pass-efficiency coverage:
  `346 passed, 1 xfailed in 17.89s`
- branch-changed broad suite including the new result contract:
  `1632 passed, 1 xfailed in 30.34s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented guarded-result
retention contract. Replace only the raw guarded expression with the selected
target. Do not change the result schema, GraphIndex ownership, rewrite cap,
candidate handling, pruning, layout synchronization, feature guard, direct
arguments, surrounding calls, dependency, public API, or TensorFlow behavior.
Keep the result unconsumed, validate sequentially, commit, and push only; do not
create, reopen, or update a pull request.

## Transpose-binary bridge direct result retention implementation checkpoint

The sole raw guarded expression now retains its unchanged two-key dictionary as
`_layout_pass_set_1_transpose_binary_bridge_stats`. The target remains
unconsumed and observation-only. The
`enable_transpose_binary_bridge_optimizations` guard and its no-else policy are
unchanged.

No result schema, GraphIndex ownership, rewrite cap, candidate handling,
pruning, layout synchronization, feature guard, direct argument, surrounding
production call, dependency, public API, or TensorFlow boundary changed. Zero
counters remain insufficient to infer absence of mutation.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `2 passed in 0.58s`
- indexed owner and recovery behavior, quantized-activation and duplicate-
  fanout boundaries, architecture, and pass-efficiency coverage:
  `347 passed in 18.42s`
- branch-changed broad suite: `1633 passed in 30.06s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, and orchestration checks; this observation-only
assignment does not claim a new model-corpus run. At resume, inventory all
direct and nested `_run_gate_layout_pass_cluster()` forms before changing the
next raw boundary. Commit and push only; do not create, reopen, or update a pull
request.

## Gate-layout result propagation characterization checkpoint

Gate-layout orchestration declares eight full-policy children and seven
required-policy children. Every policy builds one shared
`ModelIRPassStateScope`, and `run_recovery_invocations()` already produces the
ordered child-result tuple. `run_gate_layout()` and the lowerer helper currently
discard that tuple, as does the reduced direct call.

The direct call sets `include_mixed_attention=False` between retained SA/PA
MirrorPad statistics and the two-iteration normalization loop. The helper
defaults to the full policy and remains the argument-free callback selected by
attention recovery.

A passing contract freezes both pass-ID sequences, the helper policy signature,
exact reduced direct call, single direct occurrence, and both boundaries. A
strict expected-failure contract requires runner and helper return annotations/
statements, verifies synthetic full and reduced result tuples, and selects the
unconsumed observation-only direct target `_layout_opt_gate_layout_results`.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `1 passed, 1 xfailed in 0.56s`
- gate orchestration, full attention callback, reduced direct boundary,
  SA/PA-MirrorPad owner, callback composition, architecture, and pass-
  efficiency coverage: `337 passed, 1 xfailed in 17.88s`
- branch-changed broad suite including the new result contract:
  `1634 passed, 1 xfailed in 30.47s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented three-boundary
propagation contract. Return the existing recovery tuple from
`run_gate_layout()`, return it from the helper, and retain the reduced direct
tuple as `_layout_opt_gate_layout_results`. Do not change child callbacks,
policy order or selection, shared state scope, direct arguments, full callback,
normalization loop, surrounding calls, dependency, public API, or TensorFlow
behavior. Keep the direct result unconsumed, validate sequentially, commit, and
push only; do not create, reopen, or update a pull request.

## Gate-layout result propagation implementation checkpoint

`run_gate_layout()` now returns the ordered tuple already produced by
`run_recovery_invocations()`. The lowerer helper returns that tuple with the
same `Tuple[Dict[str, int], ...]` contract, and the reduced direct invocation
retains it as `_layout_opt_gate_layout_results`.

The retained direct tuple remains unconsumed and observation-only. Full and
reduced child selection, child order, the shared `ModelIRPassStateScope`, the
argument-free full-policy attention callback, reduced direct arguments,
SA/PA-MirrorPad predecessor, normalization-loop successor, dependencies,
public API, and TensorFlow behavior are unchanged.

The first focused implementation run reported `335 passed, 3 failed in
18.27s`. All three failures were stale structural assertions that still
required the helper/direct calls to discard results as `Expr` statements; no
production behavior failure was found. Those contracts now require the typed
helper `Return` and the selected direct `Assign`.

Corrected implementation validation completed sequentially under `uv`:

- dedicated result contract: `2 passed in 0.57s`
- gate orchestration, full attention callback, reduced direct boundary,
  SA/PA-MirrorPad owner, callback composition, architecture, and pass-
  efficiency coverage: `338 passed in 20.10s`
- 95 branch-changed test files: `1635 passed in 30.60s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These are unit, contract, orchestration, and changed-suite checks; this
observation-only propagation does not claim a new model-corpus run. At resume,
inventory the next raw pass-result boundary before selecting a new target.
Commit and push only; do not create, reopen, or update a pull request.

## Terminal singleton-MaxPool/Reshape result characterization checkpoint

The terminal singleton-MaxPool/Reshape runner owns two ordered children:
singleton-MaxPool layout cleanup and consecutive-Reshape cleanup. Both use one
shared `ModelIRPassStateScope` and the same ModelIR, LayoutState, diagnostics,
and keyword contract. On an empty graph their fixed schemas contain the two
singleton-MaxPool counters and three Reshape counters respectively.

The recovery engine already creates the ordered child-result tuple, but the
runner and lowerer helper return `None`, and the sole zero-argument direct call
is an expression. That call remains between the guarded retained terminal
elementwise-fanout result and the guarded retained convpool-output result.

A passing contract freezes both child IDs and schemas, current runner/helper
discard form, exact direct boundary, sole occurrence, and shared orchestration
contract. A strict expected-failure contract verifies synthetic ordered child
results and selects the unconsumed observation target
`_terminal_singleton_maxpool_reshape_results`.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `1 passed, 1 xfailed in 0.60s`
- child owners, orchestration, direct neighbors, terminal validation, shared
  context, architecture, and pass-efficiency coverage:
  `397 passed, 1 xfailed in 19.66s`
- 96 branch-changed test files: `1636 passed, 1 xfailed in 31.81s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented three-boundary
propagation contract. Return the existing tuple from the runner and helper,
retain it at the direct call, and update only stale structural contracts. Do
not change child callbacks or schemas, pass order, shared state scope, direct
arguments, guards, surrounding calls, dependency, public API, or TensorFlow
behavior. Keep the target unconsumed, validate sequentially, commit, and push
only; do not create, reopen, or update a pull request.

## Terminal singleton-MaxPool/Reshape result implementation checkpoint

`run_terminal_singleton_maxpool_reshape()` now returns the ordered pair already
created by `run_recovery_invocations()`. The lowerer helper returns that pair
with the same `Tuple[Dict[str, int], ...]` contract, and the sole direct call
retains it as `_terminal_singleton_maxpool_reshape_results`.

The target remains unconsumed and observation-only. Child callbacks and fixed
schemas, pass order, one shared `ModelIRPassStateScope`, zero-argument helper
use, the guarded terminal elementwise-fanout predecessor, the guarded
convpool-output successor, feature guards, dependencies, public API, and
TensorFlow behavior are unchanged.

The first dedicated implementation run reported `1 passed, 1 failed`; its
failure was the passing characterization assertion that still froze the
discard form. After that baseline was updated, the first focused run reported
`394 passed, 4 failed in 19.91s`. The four failures required the old helper
line count/`Expr` form or searched only for a direct `Expr`. A subsequent run
reported `397 passed, 1 failed in 21.32s`; that last check counted return-type
annotation names as captured runtime data. The contracts now inspect the typed
`Return`, selected `Assign`, unchanged surrounding boundaries, and only the
helper execution body for closure data. No production behavior failure was
found.

Corrected implementation validation completed sequentially under `uv`:

- dedicated result contract: `2 passed in 0.58s`
- child owners, orchestration, direct neighbors, terminal validation, shared
  context, architecture, and pass-efficiency coverage:
  `398 passed in 19.13s`
- 96 branch-changed test files: `1637 passed in 30.37s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, inventory the next
small discarded-result boundary before modifying production code. Commit and
push only; do not create, reopen, or update a pull request.

## Late dequant/unary fan-out result characterization checkpoint

The late dequant/unary fan-out runner owns three ordered one-key results:
dequant-Concat-Quantize layout cleanup, transpose-unary passthrough cleanup,
and transpose-unary fan-out bridge cleanup. Every child receives the same
ModelIR, LayoutState, diagnostics, and one shared `ModelIRPassStateScope`.

`run_recovery_invocations()` already creates their ordered tuple, but the
runner and lowerer helper return `None`, and the sole zero-argument direct call
is an expression. It remains between the retained
`_late_dequant_hardsigmoid_bridge_stats` result and the raw swish-transpose
passthrough call. No nested use was found.

A passing contract freezes all child IDs and empty-model schemas, current
runner/helper discard form, exact direct boundary, sole occurrence, and shared
orchestration contract. A strict expected-failure contract verifies synthetic
ordered results and selects the unconsumed observation target
`_late_dequant_unary_fanout_results`.

Characterization validation completed sequentially under `uv`:

- child owners, orchestration, direct neighbors, shared context, architecture,
  and pass-efficiency coverage: `376 passed, 1 xfailed in 17.77s`
- 97 branch-changed test files: `1638 passed, 1 xfailed in 31.21s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented runner/helper/
direct propagation contract. Return only the existing tuple and retain it at
the direct call. Do not change child callbacks or schemas, pass order, shared
state scope, direct arguments, adjacent calls, dependencies, public API, or
TensorFlow behavior. Keep the target unconsumed, validate sequentially,
commit, and push only; do not create, reopen, or update a pull request.

## Late dequant/unary fan-out result implementation checkpoint

`run_late_dequant_unary_fanout()` now returns the ordered three-child tuple
already created by `run_recovery_invocations()`. The lowerer helper returns the
same `Tuple[Dict[str, int], ...]`, and the sole direct call retains it as
`_late_dequant_unary_fanout_results`.

The target remains unconsumed and observation-only. Child callbacks and fixed
schemas, child order, one shared `ModelIRPassStateScope`, exact shared context,
zero-argument helper use, retained dequant-HardSigmoid predecessor, raw swish-
transpose successor, dependencies, public API, and TensorFlow behavior are
unchanged. Existing structural contracts were updated in the same change, and
no intermediate test failure occurred.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `2 passed in 0.62s`
- child owners, orchestration, direct neighbors, shared context, architecture,
  and pass-efficiency coverage: `377 passed in 21.25s`
- 97 branch-changed test files: `1639 passed in 31.71s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, inventory the next
small discarded-result boundary before modifying production code. Commit and
push only; do not create, reopen, or update a pull request.

## Residual affine/PReLU direct result characterization checkpoint

The extracted residual affine/PReLU owner returns the fixed one-key dictionary
`optimized_transpose_pre_add_mul_add_prelu_nhwc_chains`. It unconditionally
prunes unused tensors before returning, so its zero counter cannot prove that
the graph was not mutated.

The lowerer has one raw direct wrapper call. Pre-Add/mean-attention selects the
public owner at index 1, and SINet pre-Add/resize recovery selects it at index
0; both declarative forms are model-only and remain consumed by their parent
result tuples. The raw direct call is between the retained
`_very_late_sinet_preadd_resize_results` tuple and the residual-affine-fan-out
owner.

Passing contracts freeze the owner schema and unconditional cleanup, wrapper,
exact direct call and neighbors, sole direct occurrence, and both nested
selection indices and empty keyword contracts. A strict expected-failure
contract selects the unconsumed observation target
`_very_late_residual_affine_prelu_stats` only for the raw direct result.

Characterization validation completed sequentially under `uv`:

- owner behavior, direct wrapper, adjacent fan-out owner, both nested routes,
  architecture, and focused direct rewrite coverage:
  `284 passed, 1 xfailed in 18.07s`
- 98 branch-changed test files: `1641 passed, 1 xfailed in 30.68s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented direct assignment.
Retain only the existing dictionary. Do not change owner cleanup or schema,
nested selections, call order, direct arguments, adjacent calls, dependencies,
public API, or TensorFlow behavior. Keep the target unconsumed, validate
sequentially, commit, and push only; do not create, reopen, or update a pull
request.

## Residual affine/PReLU direct result implementation checkpoint

The very-late raw call now retains its unchanged one-key dictionary as
`_very_late_residual_affine_prelu_stats`. The target remains unconsumed and
observation-only. The two model-only declarative selections and their parent
result tuples are unchanged.

No owner behavior, unconditional pruning, schema, wrapper, direct argument,
call order, adjacent owner, nested route, dependency, public API, or TensorFlow
behavior changed. A zero counter remains insufficient to prove no tensor
cleanup mutation.

The first focused implementation run reported `283 passed, 2 failed in
18.80s`. Both failures already recognized the unchanged neighboring call names
but omitted the newly assigned observation target from their explicit boundary-
target lists. Those lists now include the target without weakening call-order
or neighbor assertions. No production behavior failure was found.

Corrected implementation validation completed sequentially under `uv`:

- dedicated result contract: `3 passed in 0.56s`
- owner behavior, direct wrapper, adjacent fan-out owner, both nested routes,
  architecture, and focused direct rewrite coverage: `285 passed in 20.05s`
- 98 branch-changed test files: `1642 passed in 31.33s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, inventory the next
raw result boundary before modifying production code. Commit and push only; do
not create, reopen, or update a pull request.

## Residual affine fan-out direct result characterization checkpoint

The extracted residual affine fan-out owner returns the fixed one-key
dictionary
`optimized_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains`. It
unconditionally prunes unused tensors before returning, so zero cannot prove
absence of graph mutation.

The lowerer has one raw direct wrapper call. Pre-Add/mean-attention selects the
public owner at index 2, and SINet pre-Add/resize recovery selects it at index
1; both declarative forms are model-only and remain part of their existing
parent result tuples. The raw call is between retained
`_very_late_residual_affine_prelu_stats` and dead-operator pruning.

Passing contracts freeze the owner schema and cleanup, wrapper, exact direct
call and neighbors, sole direct occurrence, and both nested selection indices
and empty keyword contracts. A strict expected-failure contract selects the
unconsumed observation target `_very_late_residual_affine_fanout_stats` only
for the raw direct result.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `2 passed, 1 xfailed in 0.58s`
- both residual owners, both nested routes, architecture, and focused owner
  behavior: `286 passed, 1 xfailed in 17.51s`
- 99 branch-changed test files: `1644 passed, 1 xfailed in 31.59s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented direct assignment.
Retain only the existing dictionary. Do not change owner cleanup or schema,
nested selections, call order, direct arguments, adjacent pruning,
dependencies, public API, or TensorFlow behavior. Keep the target unconsumed,
validate sequentially, commit, and push only; do not create, reopen, or update
a pull request.

## Residual affine fan-out direct result implementation checkpoint

The very-late raw call now retains its unchanged one-key dictionary as
`_very_late_residual_affine_fanout_stats`. The target remains unconsumed and
observation-only. Both model-only declarative selections and their parent
result tuples are unchanged.

No owner behavior, unconditional pruning, schema, wrapper, direct argument,
adjacent retained residual-affine/PReLU result, dead-operator pruning, nested
route, dependency, public API, or TensorFlow behavior changed. No intermediate
test failure occurred.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `3 passed in 0.57s`
- both residual owners, both nested routes, architecture, and focused owner
  behavior: `287 passed in 17.97s`
- 99 branch-changed test files: `1645 passed in 30.93s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, inventory the next
raw result boundary before modifying production code. Commit and push only; do
not create, reopen, or update a pull request.

## SINet mixed-attention direct result characterization checkpoint

The indexed SINet mixed-attention owner returns the fixed one-key dictionary
`optimized_sinet_mix_attention_double_logistic_nhwc_chains`, accepts an
optional GraphIndex/candidate, and uses a default 32-rewrite cap. It prunes and
synchronizes layout only after a successful rewrite.

The lowerer has one raw direct wrapper call. Attention gate/QDQ recovery also
selects the public owner at index 1 with ModelIR and LayoutState only. The raw
call remains between retained
`_post_sinet_split_conv_concat_bridge_stats` and the distinct mixed-attention
layout cleanup owner.

Passing contracts freeze the owner defaults, result schema, guarded cleanup,
wrapper forwarding, exact direct call and neighbors, sole direct occurrence,
and nested layout-aware selection. A strict expected-failure contract selects
the unconsumed observation target `_post_sinet_mix_attention_stats` only for
the raw direct result.

The first dedicated characterization run reported `1 passed, 1 failed, 1
xfailed in 0.60s`; the new test incorrectly expected the wrapper to rename its
`graph_index` parameter to `active_index`. The expectation now freezes the
actual unchanged forwarding contract.

Corrected characterization validation completed sequentially under `uv`:

- dedicated result contract: `2 passed, 1 xfailed in 0.59s`
- indexed owner, attention and gate orchestration, direct neighbors, and
  architecture coverage: `375 passed, 1 xfailed in 18.42s`
- 100 branch-changed test files: `1647 passed, 1 xfailed in 32.58s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented direct assignment.
Retain only the existing dictionary. Do not change owner logic or schema,
rewrite cap, GraphIndex/candidate handling, guarded pruning, nested selection,
call order, direct arguments, dependencies, public API, or TensorFlow behavior.
Keep the target unconsumed, validate sequentially, commit, and push only; do
not create, reopen, or update a pull request.

## SINet mixed-attention direct result implementation checkpoint

The post-SINet raw call now retains its unchanged one-key dictionary as
`_post_sinet_mix_attention_stats`. The target remains unconsumed and
observation-only. The layout-aware attention gate/QDQ selection and its parent
result tuple remain unchanged, as does the following distinct mixed-attention
layout cleanup owner.

No owner matching or application logic, GraphIndex/candidate handling, default
rewrite cap, guarded pruning, LayoutState synchronization, result schema,
wrapper forwarding, direct arguments, call order, dependency, public API, or
TensorFlow behavior changed. No intermediate test failure occurred.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `3 passed in 0.60s`
- indexed owner, attention and gate orchestration, direct neighbors, and
  architecture coverage: `376 passed in 18.76s`
- 100 branch-changed test files: `1648 passed in 31.83s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, inventory the next
raw result boundary before modifying production code. Commit and push only; do
not create, reopen, or update a pull request.

## Mixed-attention layout direct result characterization checkpoint

The mixed-attention layout runner returns the fixed one-key dictionary
`optimized_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains` from a
transactional layout pass. Its underlying optimizer prunes unused tensors on
exit, so a zero rewrite counter is not complete mutation evidence.

Three routes are frozen: the raw post-SINet direct call, gate-layout full-policy
index 0, and absolute-final normalization/attention index 1. The reduced gate
policy intentionally excludes this child. Both nested routes receive their
parent's shared state scope. The raw call remains between retained
`_post_sinet_mix_attention_stats` and
`_post_sinet_dequant_hardsigmoid_bridge_stats` and does not pass a scope.

Passing contracts freeze the fixed schema, transactional pass ID, all direct
arguments and neighbors, nested indices and shared-scope contracts, reduced-
policy exclusion, and sole direct occurrence. A strict expected-failure
contract selects the unconsumed observation target
`_post_sinet_mixed_attention_layout_stats` only for the raw call.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `2 passed, 1 xfailed in 0.68s`
- gate and absolute-final orchestration, indexed SINet and dequant neighbors,
  architecture, pass efficiency, and focused owner behavior:
  `320 passed, 1 xfailed in 21.01s`
- 101 branch-changed test files: `1650 passed, 1 xfailed in 35.95s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented direct assignment.
Retain only the existing dictionary. Do not change transactional owner logic,
schema, underlying pruning, state-scope ownership, full/reduced selection,
direct arguments, call order, dependencies, public API, or TensorFlow behavior.
Keep the target unconsumed, validate sequentially, commit, and push only; do
not create, reopen, or update a pull request.

## Mixed-attention layout direct result implementation checkpoint

The post-SINet raw call now retains its unchanged one-key dictionary as
`_post_sinet_mixed_attention_layout_stats`. The target remains unconsumed and
observation-only. Gate-layout full/reduced selection, absolute-final
normalization/attention selection, and both nested parent tuples are unchanged.

No transactional owner logic, underlying unused-tensor pruning, result schema,
state-scope ownership, direct ModelIR/LayoutState/diagnostics arguments,
retained indexed-SINet predecessor, retained dequant-HardSigmoid successor,
call order, dependency, public API, or TensorFlow behavior changed. No
intermediate test failure occurred.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `3 passed in 0.57s`
- gate and absolute-final orchestration, indexed SINet and dequant neighbors,
  architecture, pass efficiency, and focused owner behavior:
  `321 passed in 18.00s`
- 101 branch-changed test files: `1651 passed in 31.09s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, inventory the next
raw result boundary before modifying production code. Commit and push only; do
not create, reopen, or update a pull request.

## Late NDHWC/cost-volume direct results characterization checkpoint

The late NDHWC gate runner returns two fixed counters from two transactional
layout pass specs. The adjacent cost-volume scatter runner returns one fixed
counter from one transactional pass spec. Their lower-level optimizers prune
unused tensors on exit, so zero counters are not complete mutation evidence.

The raw lowerer pair shares one explicit `late_ndhwc_cost_volume_state_scope`.
It remains between retained post-SINet dequant-HardSigmoid statistics and
retained cost-volume Conv-affine statistics. Gate-layout required policy selects
the same owners at indices 3/4, and full policy at indices 4/5; each nested pair
shares its parent scope.

Passing contracts freeze both schemas, all transactional pass IDs, lower-level
cleanup, exact direct arguments and common scope, neighbors, required/full
nested indices and scope identity, and sole direct occurrences. A strict
expected-failure contract selects the unconsumed observation targets
`_late_ndhwc_gate_layout_stats` and
`_late_cost_volume_scatter_layout_stats` together.

The first dedicated characterization run reported `1 passed, 1 failed, 1
xfailed in 0.63s`; the new test treated the NDHWC loop-generated `pass_id` AST
name as a literal. It now reads the two fixed IDs from the callbacks declaration
while separately requiring the transactional PassSpec factory.

Corrected characterization validation completed sequentially under `uv`:

- dedicated pair contract: `2 passed, 1 xfailed in 0.59s`
- both owners, gate orchestration, direct neighbors, terminal validation,
  architecture, and pass-efficiency coverage:
  `416 passed, 1 xfailed in 19.41s`
- 102 branch-changed test files: `1653 passed, 1 xfailed in 31.98s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The sole expected failure is the intentionally unimplemented pair of direct
assignments. Retain only the existing dictionaries. Do not change transactional
owners or schemas, underlying pruning, shared scope, nested selection, direct
arguments, call order, dependencies, public API, or TensorFlow behavior. Keep
both targets unconsumed, validate sequentially, commit, and push only; do not
create, reopen, or update a pull request.

## Late NDHWC/cost-volume direct results implementation checkpoint

The adjacent late calls now retain their unchanged dictionaries as
`_late_ndhwc_gate_layout_stats` and
`_late_cost_volume_scatter_layout_stats`. Both targets remain unconsumed and
observation-only. The calls remain adjacent and continue to share the existing
`late_ndhwc_cost_volume_state_scope`.

No transactional owner behavior, fixed schema, lower-level pruning, shared
scope, direct ModelIR/LayoutState/diagnostics arguments, gate required/full
selection or scope identity, retained predecessor/successor, call order,
dependency, public API, or TensorFlow behavior changed. No intermediate test
failure occurred.

Implementation validation completed sequentially under `uv`:

- dedicated pair contract: `3 passed in 0.57s`
- both owners, gate orchestration, direct neighbors, terminal validation,
  architecture, and pass-efficiency coverage: `417 passed in 19.19s`
- 102 branch-changed test files: `1654 passed in 30.84s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, inventory the next
raw result boundary before modifying production code. Commit and push only; do
not create, reopen, or update a pull request.

## Very-late pad-layout direct result characterization checkpoint

The very-late pad-layout runner returns three fixed counters from transactional
layout passes with IDs `layout.pad_prepost_nhwc`,
`layout.unary_pad_prepost_nhwc`, and
`layout.norm_subgraph_pad_prepost_nhwc`. Each lower-level optimizer prunes
unused tensors on exit, so a zero counter is not complete mutation evidence.

Four route classes are frozen: the raw very-late direct call, gate-layout
required/full selections at indices 1/2, terminal-boundary selection at index
2, and the consumed norm-only safety fallback. Nested routes continue to own
or receive their existing shared state scopes. The fallback continues to pass
`include_pad=False`, `include_unary=False`, and `include_norm=True`, and its
result remains consumed as `fallback_norm_stats`.

Passing contracts freeze the three-key schema, all transactional pass IDs,
lower-level cleanup, exact direct arguments, retained terminal
Squeeze/Mean/Squeeze predecessor, InstanceNorm post-bias successor, nested
indices and scope contracts, fallback feature flags and consumer, and the two
raw lowerer occurrences. A strict expected-failure contract selects the
unconsumed observation target `_very_late_pad_layout_stats` only for the raw
`model_ir` call.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `2 passed, 1 xfailed in 0.60s`
- direct, gate, terminal-boundary, fallback, architecture, pass-efficiency,
  and focused Pad owner coverage: `394 passed, 1 xfailed in 20.62s`
- 103 branch-changed test files: `1656 passed, 1 xfailed in 30.91s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

The first broad invocation lost its output handle near completion and did not
provide an authoritative exit status; the identical suite was rerun to the
clean result above. No test assertion failed. The sole expected failure is the
intentionally unimplemented raw direct assignment.

Retain only the raw call's existing dictionary. Do not change transactional
owner behavior or schemas, lower-level pruning, nested route selection or
scope ownership, fallback flags or consumption, direct arguments, call order,
dependencies, public API, or TensorFlow behavior. Keep the target unconsumed,
validate sequentially, commit, and push only; do not create, reopen, or update
a pull request.

## Very-late pad-layout direct result implementation checkpoint

The raw `model_ir` call now retains its unchanged three-key dictionary as
`_very_late_pad_layout_stats`. The target remains unconsumed and
observation-only. The separate norm-only safety fallback remains consumed as
`fallback_norm_stats` with `include_pad=False`, `include_unary=False`, and
`include_norm=True`.

No transactional owner behavior, fixed schema, lower-level unused-tensor
pruning, gate required/full or terminal-boundary selection, nested state-scope
ownership, fallback flags or consumption, direct ModelIR/LayoutState/diagnostic
arguments, retained predecessor/successor, call order, dependency, public API,
or TensorFlow behavior changed. No intermediate test failure occurred.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `3 passed in 0.59s`
- direct, gate, terminal-boundary, fallback, architecture, pass-efficiency,
  and focused Pad owner coverage: `395 passed in 19.98s`
- 103 branch-changed test files: `1657 passed in 32.02s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, inventory the next
raw result boundary before modifying production code. Commit and push only; do
not create, reopen, or update a pull request.

## Fallback singleton/consecutive-reshape result characterization checkpoint

The singleton/consecutive-reshape runner returns a fixed three-dictionary tuple
from `run_singleton_channel_transpose_cleanup`,
`run_duplicate_fanout_cleanup`, and `run_consecutive_reshape_cleanup`. The
three invocations share one `ModelIRPassStateScope`; duplicate-fanout keeps
`include_transpose=False`. The fallback route intentionally constructs its
context with `layout_state=None` while forwarding the lowerer's diagnostics.

All three lowerer routes are frozen. The earlier very-late `model_ir` tuple is
already retained for observation. The later shared-late `model_ir` tuple is
unpacked and consumed by the reconciliation guard. The remaining `fallback_ir`
call is raw and runs only when the norm-subgraph Pad cleanup counter is
positive. It remains between singleton-broadcast adapter repair and static
shape reconciliation.

Passing contracts freeze the pass IDs and tuple schema, shared state scope,
duplicate-fanout flag, all three target forms, exact fallback guard and
arguments, repair/reconcile neighbors, and sole fallback occurrence. A strict
expected-failure contract selects the unconsumed observation target
`_fallback_singleton_consecutive_reshape_results` only for the guarded
fallback call.

Characterization validation completed sequentially under `uv`:

- dedicated fallback result contract: `2 passed, 1 xfailed in 0.57s`
- fallback, singleton/consecutive orchestration, terminal validation,
  architecture, pass-efficiency, and representative rewrite coverage:
  `391 passed, 1 xfailed in 19.16s`
- 104 branch-changed test files: `1659 passed, 1 xfailed in 31.12s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

No test assertion failed. The sole expected failure is the intentionally
unimplemented fallback assignment. Retain only the existing tuple. Do not
change pass order or schemas, shared state-scope ownership, duplicate-fanout
flags, main-route result handling, fallback guard, direct arguments, adjacent
repair/reconciliation, dependencies, public API, or TensorFlow behavior. Keep
the target unconsumed, validate sequentially, commit, and push only; do not
create, reopen, or update a pull request.

## Fallback singleton/consecutive-reshape result implementation checkpoint

The guarded `fallback_ir` call now retains its unchanged three-dictionary tuple
as `_fallback_singleton_consecutive_reshape_results`. The target remains
unconsumed and observation-only. The existing very-late observation tuple and
shared-late reconciliation tuple remain unchanged.

No pass order or schema, shared `ModelIRPassStateScope`, duplicate-fanout
`include_transpose=False` contract, fallback `layout_state=None` choice,
diagnostic forwarding, positive norm-cleanup guard, direct arguments,
repair/reconciliation neighbors, main-route result handling, dependency,
public API, or TensorFlow behavior changed. No intermediate test failure
occurred.

Implementation validation completed sequentially under `uv`:

- dedicated fallback result contract: `3 passed in 0.54s`
- fallback, singleton/consecutive orchestration, terminal validation,
  architecture, pass-efficiency, and representative rewrite coverage:
  `392 passed in 19.82s`
- 104 branch-changed test files: `1660 passed in 32.15s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, inventory the next
raw result boundary before modifying production code. Commit and push only; do
not create, reopen, or update a pull request.

## Late Swish passthrough direct result characterization checkpoint

The indexed Swish transpose-passthrough owner returns the fixed one-key
dictionary `rewritten_swish_transpose_passthrough_chains`. It accepts optional
GraphIndex, LayoutState, rewrite-limit, and candidate inputs. Successful plans
perform indexed graph edits, layout synchronization, and unused-tensor pruning;
an empty or rejected candidate set returns zero.

The lowerer has one raw direct wrapper call with the live LayoutState. It
remains between retained late dequant/unary-fanout results and the distinct
Squeeze/Unary/ExpandDims passthrough owner. Layout-recovery also selects the
public owner at index 5 with ModelIR and LayoutState only.

Passing contracts freeze the fixed schema, owner and wrapper defaults, wrapper
forwarding, successful-plan cleanup, exact direct arguments and neighbors,
sole direct wrapper occurrence, and nested selection index and keyword
contract. A strict expected-failure contract selects the unconsumed observation
target `_late_swish_transpose_passthrough_stats` only for the raw direct call.

Characterization validation completed sequentially under `uv`:

- dedicated result contract: `2 passed, 1 xfailed in 0.58s`
- indexed owner, layout-recovery, dequant/unary-fanout boundary, architecture,
  pass-efficiency, and direct Swish rewrite coverage:
  `366 passed, 1 xfailed in 18.67s`
- 105 branch-changed test files: `1662 passed, 1 xfailed in 31.40s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

No test assertion failed. The sole expected failure is the intentionally
unimplemented direct assignment. Retain only the existing dictionary. Do not
change matching or application logic, GraphIndex/candidate handling, rewrite-
limit semantics, guarded pruning, result schema, wrapper forwarding,
layout-recovery selection, direct arguments, call order, dependencies, public
API, or TensorFlow behavior. Keep the target unconsumed, validate sequentially,
commit, and push only; do not create, reopen, or update a pull request.

## Late Swish passthrough direct result implementation checkpoint

The raw late wrapper call now retains its unchanged one-key dictionary as
`_late_swish_transpose_passthrough_stats`. The target remains unconsumed and
observation-only. The layout-recovery public-owner selection remains unchanged,
as do the retained dequant/unary-fanout predecessor and following distinct
Squeeze/Unary/ExpandDims passthrough owner.

No owner matching or indexed application logic, GraphIndex/candidate handling,
rewrite-limit semantics, successful-plan pruning, result schema, wrapper
forwarding, LayoutState updates, layout-recovery selection, direct arguments,
call order, dependency, public API, or TensorFlow behavior changed. Two
pre-existing boundary tests now recognize the assigned observation target
without weakening their call-identity or ordering assertions. No intermediate
test failure occurred.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `3 passed in 0.59s`
- indexed owner, layout-recovery, dequant/unary-fanout boundary, architecture,
  pass-efficiency, and direct Swish rewrite coverage: `367 passed in 18.59s`
- 105 branch-changed test files: `1663 passed in 31.71s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, inventory the next
raw result boundary before modifying production code. Commit and push only; do
not create, reopen, or update a pull request.

## Late Conv1D unary direct results characterization checkpoint

Three adjacent indexed owners cover the Conv1D
Squeeze/Unary/ExpandDims passthrough, the rank-4
Unary/Transpose/Reshape/ExpandDims variant, and the fan-out bypass variant.
Each returns a fixed one-key dictionary and accepts optional GraphIndex and
LayoutState through a forwarding lowerer wrapper. Each owner prunes unused
tensors on both the preflight-zero and normal exit paths, so zero counters are
not complete mutation evidence.

All three wrappers have exactly one raw direct call with the live LayoutState
and no nested orchestration selection. The adjacent calls remain between the
retained late Swish passthrough result and the distinct InstanceNorm unary
passthrough owner.

Passing contracts freeze all three fixed schemas, wrapper defaults and
forwarding, unconditional cleanup, exact direct arguments and adjacency, and
sole occurrences. A strict expected-failure contract selects the unconsumed
observation targets `_late_conv1d_squeeze_unary_stats`,
`_late_conv1d_rank4_unary_stats`, and `_late_conv1d_unary_fanout_stats`
together.

Characterization validation completed sequentially under `uv`:

- dedicated family result contract: `2 passed, 1 xfailed in 0.59s`
- all three indexed owners, architecture, pass-efficiency, retained Swish
  boundary, and representative direct rewrites:
  `487 passed, 1 xfailed in 18.25s`
- 106 branch-changed test files: `1665 passed, 1 xfailed in 32.70s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

No test assertion failed. The sole expected failure is the intentionally
unimplemented group of direct assignments. Retain only the existing
dictionaries. Do not change matching or indexed application, GraphIndex or
LayoutState handling, unconditional pruning, schemas, wrapper forwarding,
direct arguments, adjacency, dependencies, public API, or TensorFlow behavior.
Keep all targets unconsumed, validate sequentially, commit, and push only; do
not create, reopen, or update a pull request.

## Late Conv1D unary direct results implementation checkpoint

The three adjacent raw calls now retain their unchanged dictionaries as
`_late_conv1d_squeeze_unary_stats`, `_late_conv1d_rank4_unary_stats`, and
`_late_conv1d_unary_fanout_stats`. All targets remain unconsumed and
observation-only. The calls remain adjacent between the retained Swish result
and the distinct InstanceNorm unary passthrough owner.

No matching or indexed application logic, GraphIndex or LayoutState handling,
unconditional unused-tensor pruning, result schema, wrapper forwarding, direct
arguments, adjacency, dependency, public API, or TensorFlow behavior changed.
No pre-existing test required adjustment and no intermediate test failure
occurred.

Implementation validation completed sequentially under `uv`:

- dedicated family result contract: `3 passed in 0.59s`
- all three indexed owners, architecture, pass-efficiency, retained Swish
  boundary, and representative direct rewrites: `488 passed in 19.05s`
- 106 branch-changed test files: `1666 passed in 32.03s`
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed

These checks do not claim a new model-corpus run. At resume, inventory the next
raw result boundary before modifying production code. Commit and push only; do
not create, reopen, or update a pull request.
