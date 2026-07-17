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
