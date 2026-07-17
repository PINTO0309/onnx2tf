# `flatbuffer_direct` refactor continuation checkpoint — 2026-07-14

## Status

The active branch is `fb-refactor5`, created from `main` after pull request
`#949` merged the complete `fb-refactor4` checkpoint. Pull request `#950` is
closed, and no open pull request tracks this branch. This checkpoint is ready
for the next Goal continuation. Work continues through coherent commits and
pushes without opening a pull request.

The latest implementation unit moves Slice/optional-Logistic/Concat/Reshape
detection-tail recovery into the TensorFlow-free indexed
`passes/slice_logistic_concat_reshape_tail_layout.py` owner. Its former
480-line full-map, unbounded fixed-point helper is now a thin compatibility
dispatcher at both unchanged source positions. The owner resolves an immutable
whole-group plan, re-resolves the complete contract before apply, and maintains
one differential `ModelIRGraphIndex` plus the Session `LayoutState` through
every rewrite.

The audited fast-precanonicalize orchestrator remains 294 lines, down from 482
lines at Goal resumption, 1,025 lines at the beginning of the previous
continuation, and 1,608 lines before the broader extraction.

### Split/mixed-Concat continuation — 2026-07-16

The extracted owner starts from graph-ordered channel-axis Concat candidates
instead of rebuilding complete producer and consumer maps. Every Concat input
must be classified as either a private typed NHWC-to-NCHW adapter or a direct
output of a channel-axis Split. At least one Split root is mandatory. The
resolver proves rank-four static and dynamic views, dtype, per-tensor
quantization, explicit or unknown layout, producer uniqueness, graph order,
exact consumer ownership, typed INT32/INT64 permutation and Split-axis
constants, the derived NHWC Concat result, and the private terminal boundary
before a plan exists.

All outputs of each accepted Split are classified together. An output may feed
the target Concat directly or feed one or more private NCHW-to-NHWC adapters;
any unrelated or public consumer rejects the complete group. Adapter aliases
are rewired to the converted Split output only after revalidation. Shared
Split-axis constants receive one deterministic dtype-preserving clone, while
an axis with exactly one owned consumer slot changes in place. Produced,
variable, public, malformed, or per-axis-quantized constants and tensors are
not rewritten.

The immutable plan records all axis ownership, new Split-input adapters,
converted output metadata, direct-input adapter removals, alias rewrites,
Concat axis/output changes, the terminal NCHW compatibility adapter, and full
tensor/operator and graph-boundary contracts. The same Concat is resolved
again immediately before mutation. Candidate-only operation and an explicit
rewrite limit use the production entry point; inserted and removed operators
update one differential graph index. Converted tensors are recorded in both
TensorIR and Session `LayoutState`, and success-only pruning preserves the
legacy zero-match side effect.

Before extraction, seven short Tier 0-4 models containing both ONNX Split and
Concat were converted sequentially under an instrumented helper:
`alike_t_opset11_192x320`, `yolov9_n_wholebody_with_wheelchair_post`,
`sgscsh`, `best`, `mobileformer`, `parseq-tiny`, and `LINEA`. The mixed helper
remained zero-match in every runtime invocation. The immediately following
input-chain helper remained active where expected, including three first-sweep
groups in `yolov9` and two later-sweep groups in `sgscsh`; extraction therefore
did not absorb or pre-empt that separate family.

Sixteen dedicated tests cover the active mixed and all-Split forms, dynamic
signatures, negative axes, shared-axis cloning, explicit layouts, public
boundaries, fan-out, per-axis quantization, produced and duplicate producers,
stale-plan rejection, candidate limits, idempotence, GraphIndex/LayoutState
integrity, the compatibility dispatcher, and repeated production sweeps. The
new owner, adjacent Split/Concat owners, boundary-input checks, complete
architecture suite, and TensorFlow import blocker pass together with `524
passed in 51.68s`. Scoped Ruff, syntax compilation, and `git diff --check`
pass.

Two post-extraction `-cotof` checks ran sequentially. `yolov9` remains a pass
with maximum absolute error `3.0517578125e-05`; `sgscsh` remains a pass with
maximum absolute error `2.5331974029541016e-07`. Both exactly reproduce the
managed quick-profile accuracy values and remain well below `1e-1`. No
TensorFlow import, new dependency, parallel inference, SWAP, timeout, or Tier
corpus run was introduced. Temporary characterization and conversion outputs
were removed.

### General Concat input-adapter continuation — 2026-07-16

The extracted owner starts from graph-ordered rank-four channel-axis Concat
candidates instead of rebuilding complete producer and consumer maps. Every
input must be either the private output of a typed NHWC-to-NCHW Transpose or
one of the historical thirteen unary operations immediately following such a
Transpose or a singleton-channel memory-equivalent Reshape. Direct and unary
forms may be mixed. Repeated Concat input slots share one resolved branch and
one metadata update rather than applying the old mutation twice.

The resolver proves static and dynamic rank-four views, dtype, per-tensor
quantization, logical and physical layout, producer uniqueness and graph
order, exact consumer ownership, typed permutation and Reshape-shape
constants, unary input/output equivalence, the derived NHWC Concat result, and
terminal consumer order. Public Concat outputs and existing NCHW consumers
remain on their original tensor name through one post-Concat compatibility
Transpose. Unsupported fan-out, malformed constants, per-axis quantization,
duplicate producers, stale plans, and unresolved sources reject the complete
candidate without mutation.

The immutable plan owns all adapter removals, unary rewires and metadata,
Concat inputs/axis/output, optional deterministic permutation creation, the
terminal adapter, complete tensor/operator contracts, and graph boundaries.
The same Concat is resolved again immediately before apply. Candidate-only
operation and an explicit rewrite limit use the production entry point;
removed and inserted operators update one differential graph index. TensorIR
and Session `LayoutState` both record resolved NHWC and retained NCHW
boundaries. Pruning runs once at owner exit even when no candidate matches,
preserving the legacy zero-match side effect and safe-bundle contract.

Pre-extraction characterization observed six invocations on each of `yolov9`
and `sgscsh`. `yolov9` retained three direct two-input groups in its first
invocation at spatial sizes 48x84, 24x42, and 12x21. `sgscsh` retained two
direct three-input groups in its second invocation; every other invocation was
zero-match. The preceding indexed Split/mixed-Concat owner remained zero-match
for these groups, confirming the ordered ownership boundary.

Thirty-six dedicated tests plus the two existing active fixtures cover direct,
dynamic, negative-axis, public-output, repeated-slot, existing-consumer,
existing/new permutation, all thirteen unary, singleton-Reshape, dispatcher,
candidate limit, stale-plan, zero-match pruning, safe-bundle, determinism, copy
isolation, and transactional rejection behavior. The new and adjacent indexed
owners, boundary-input checks, TensorFlow import blocker, and complete
architecture suite pass together with `561 passed in 51.11s`. Scoped Ruff,
format verification, syntax compilation, and `git diff --check` pass.

Baseline/current conversion-only comparisons used identical Python API calls
from detached source checkpoint `f5505052`. For both `yolov9` and `sgscsh`,
float32 TFLite, float16 TFLite, tensor correspondence report, `schema.fbs`,
and `schema_generated.py` are byte-identical. Separate sequential `-cotof`
checks pass at maximum absolute errors `3.0517578125e-05` and
`2.5331974029541016e-07`, respectively. An initial apparent `yolov9` report
difference was traced before any source change to unlike invocation modes:
the `-cotof` CLI path normalizes leading `/` names to `wa/`, whereas the
conversion-only Python comparison did not. Matching invocation modes removed
the difference. No implementation fix was necessary.

### Slice/Logistic/Concat/Reshape tail continuation — 2026-07-16

The extracted owner starts from graph-ordered rank-three channel/spatial tail
Concat candidates instead of rebuilding complete producer and consumer maps.
Every unique tail input must be a private rank-three Reshape fed by a private
two-input channel-axis Concat. Each branch input must be a private rank-four
Slice, optionally followed by Logistic, and both Slices must be the complete
fan-out of one typed NHWC-to-NCHW Transpose. Multiple branches may share the
same NHWC source, but every removable adapter and every downstream branch is
owned independently.

The resolver proves static and dynamic rank-three/rank-four views, dtype,
per-tensor quantization, explicit or unknown layout, unique producers, strict
graph order, exact consumer slots, typed INT32/INT64 permutation, Slice begin,
Slice size, and Reshape-shape constants, exact Slice bounds, both Concat
results, Reshape element counts, and the retained terminal NCHW boundary.
`newShape` and `onnxRawNewShape` options are validated and remapped together.
Exclusive constants change in place; a safe shared Reshape shape receives one
deterministic dtype-preserving clone. Slice parameters must remain exclusive,
so unrelated use rejects the complete group.

The immutable plan owns all Slice input rewrites and constants, optional
Logistic metadata, branch Concat axes, Reshape constants/options/metadata, tail
Concat axis/output, optional typed 3D permutation creation, the terminal
compatibility Transpose, adapter removals, complete tensor/operator contracts,
and graph boundaries. The same tail Concat is fully resolved again immediately
before apply. Candidate-only operation and an explicit rewrite limit use the
production entry point; inserted and removed operators update one differential
graph index. TensorIR and Session `LayoutState` are updated together. Pruning
runs once at owner exit, including zero-match calls.

Characterization selected the 4.5 MiB Tier-2
`nanodet-plus-m_416.onnx` without broad conversion. The helper is invoked five
times: the first call retains one four-branch group and the remaining four are
zero-match. Branch spatial sizes are 52x52, 26x26, 13x13, and 7x7; each splits
37 channels into a Logistic-wrapped 5-channel path and a direct 32-channel
path before Reshape. The preceding Split/mixed owner remains zero-match. The
general input-adapter owner rewrites seven separate groups on its first call
but deliberately leaves this tail group for its ordered owner. Five yolov9
calls remain zero-match because its superficially similar rank-three tail is
fed by Transposes rather than the required Slice/Concat branches.

Forty dedicated tests plus the existing active fixture cover all four optional
Logistic forms, numerical equivalence, dynamic full-extent Slice signatures,
INT32/INT64 constants, shared-shape copy-on-write, existing/new 3D
permutations, shared NHWC sources, compatibility dispatch, candidate limits,
stale-plan rejection, zero-match pruning, determinism, GraphIndex/LayoutState
integrity, and twenty-four unsafe transactional no-op cases. The new and
adjacent indexed owners, boundary-input checks, TensorFlow import blocker, and
complete architecture suite pass together with `603 passed in 49.68s`.
Scoped Ruff, format verification, syntax compilation, and `git diff --check`
pass.

Baseline/current conversion-only comparisons use the same Python API and raw
source checkpoint `762dcdef`. Nanodet float32 TFLite, float16 TFLite, tensor
correspondence report, `schema.fbs`, and `schema_generated.py` are all
byte-identical. The indexed helper retains the exact production sequence of
one rewrite followed by four zero-match calls. A separate sequential `-cotof`
check passes with maximum absolute error `6.19888e-06`, below both the managed
profile value `0.03541278839111328` and the required `1e-1` ceiling. The
conversion process reported `VmSwap: 0 kB`; no timeout or parallel inference
was introduced.

## Continuation snapshot — 2026-07-16

The indexed elementwise/Concat owner preserves the complete historical
capability set: eleven unary operations, all six binary operations without
reordering operands, connected Concat groups, rank-four broadcast constants,
direct graph-input boundaries, explicit NHWC-to-NCHW adapters, reusable
inverse-adapter aliases, multiple post aliases, safe legacy consumers, and
pre-adapter fan-out. Typed INT32/INT64 permutations, static and dynamic
rank-four views, dtype, per-tensor quantization, graph order, public
boundaries, producer uniqueness, exact consumer slots, constant ownership,
and broadcast results are proven before a plan is accepted.

The candidate set is fixed in graph order, reverse closure traversal is
bounded by the current input-edge count, and an optional rewrite limit bounds
accepted groups. Every input/output rewrite, axis and metadata update,
constant transpose, adapter insertion/removal, tensor contract, and operator
contract belongs to immutable plan state. The same seed is fully resolved a
second time immediately before mutation. Pruning runs once after all groups,
preserving the legacy zero-match side effect and correspondence-event order.

Sequential pre/post characterization reached the helper five times on each
of YuNet, FastestDet, HumanSeg, OSNet, and SiNet. FastestDet retained two
connected groups and HumanSeg retained one group in their first invocation;
the other 22 invocations were zero-match. The pre/post semantic ModelIR digest
matched the preceding checkpoint at all ten FastestDet/HumanSeg invocation
boundaries. The only full-digest difference is intentional: converted tensors
now carry explicit `NHWC` in both TensorIR and Session `LayoutState` instead of
leaving stale `UNKNOWN` layout provenance.

The dedicated 56-test suite, adjacent indexed layout owners, both existing
active fixtures, SPP owner, and complete architecture suite pass together:

```text
546 passed in 42.49s
```

TensorFlow-import-blocked explicit direct, default direct, and direct `-cotof`
conversion pass sequentially with `3 passed in 4.07s`. Sequential YuNet,
FastestDet, and HumanSeg conversions reproduce all fifteen fixed float32,
float16, correspondence, and schema artifact hashes. Scoped Ruff, syntax
compilation, and `git diff --check` pass. Temporary comparison worktrees and
outputs were removed. No Tier corpus was run.

### StridedSlice/Concat continuation

The indexed owner accepts only one private rank-four NHWC-to-NCHW adapter whose
entire fan-out consists of at least two supported `STRIDED_SLICE` operators.
Every slice must use four distinct, immutable typed INT32/INT64 begin/end/
stride constants, a nonzero stride vector, zero masks, and no offset. Every
slice output must feed exactly one common channel-axis Concat, and at least one
private NCHW-to-NHWC post adapter must establish the removable round trip.

Static and dynamic views, dtype, per-tensor quantization, layout, producer
uniqueness, graph order, exact consumer slots, Concat shape/signature, every
post alias, and every legacy boundary are fixed before a plan exists. All
constants, Slice input rewrites, output metadata, Concat axis/output changes,
alias rewrites, adapter reuse/removal, tensor contracts, and operator contracts
belong to immutable plan state and are fully resolved again immediately before
apply. Graph-ordered Transpose candidates and an optional rewrite limit replace
the raw fixed-point loop; pruning still runs once on every invocation to retain
the historical zero-match contract.

The ordinary and multi-post synthetic active forms produce the exact same
non-layout ModelIR digest as source checkpoint `44f3b2ca`, including tensor
lineage metadata and event ordering. Converted Slice and Concat tensors now
carry explicit NHWC in TensorIR and Session LayoutState. A retained NCHW
boundary reuses the already-proven typed pre-permutation rather than overwriting
the first post-permutation buffer as INT32; shared post-permutation users remain
unchanged. Conflicting constant roles and a legacy consumer ordered before the
retained adapter reject the complete transaction.

Pre- and post-extraction characterization reached the helper five times on
each of YuNet, FastestDet, HumanSeg, OSNet, and SiNet. All 25 invocations remain
zero-match with unchanged operator and tensor counts. Fifty-eight focused tests
cover typed constants and permutations, dynamic signatures, negative axes,
multiple post aliases, repeated input slots, public and legacy boundaries,
shared post permutations, numerical equivalence, candidate limits,
idempotence, GraphIndex/LayoutState integrity, lineage compatibility, zero-
match pruning, and forty-two unsafe transactional no-op cases.

The new owner, adjacent pre-Concat Slice/Split and Split/Conv/Concat owners,
elementwise/Concat owner, and complete architecture suite pass together with
`542 passed in 40.54s`. TensorFlow-import-blocked explicit direct, default
direct, and direct `-cotof` conversion pass sequentially with `3 passed in
4.01s`. One sequential YuNet conversion reproduces all five fixed artifacts.
Scoped Ruff, syntax compilation, and `git diff --check` pass. Temporary
comparison worktrees and outputs were removed. No Tier corpus was run.

## Continuation snapshot — 2026-07-15

The characterized
`_optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw` helper
is now extracted. Its dedicated owner performs pure match/plan/revalidation
before mutation, uses one differential graph index, and replaces both the
outer fixed-point loop and closure-wide repeated scans with graph-ordered
candidates plus a bounded consumer `deque`. The existing two production
sequence positions and stats key are unchanged.

The plan proves typed permutation constants, exact rank-four shape and dynamic
signature relations, all six plain binary operations, operand order, resolved
sources, unique producers, graph order, per-tensor quantization, NHWC
broadcasts, equal channel Split contracts, channel Concat axes, and a closed
unary/binary/Concat/Split tail. Shared Split axes are cloned deterministically
with their original INT32 or INT64 dtype. Unsupported consumers, dead converted
branches, public intermediates, multiple outputs, fused activation, per-axis
quantization, explicit NCHW external inputs, stale order, missing tensors, and
duplicate producers reject the complete candidate without mutation.

The terminal adapter now preserves the original public NCHW tensor metadata
and moves only its private producer output to NHWC. It reuses the proven input
permutation. This fixes the raw helper's public-metadata mismatch and prevents
the input Transpose from being removed when any converted closure edge remains
unhandled.

Pre-extraction characterization remains authoritative: four zero-match runtime
invocations each on YuNet, FastestDet, HumanSeg, and OSNet, and eight zero-match
invocations on SiNet. The indexed active synthetic contract is covered by 33
focused tests. The two preceding binary owners, the new owner, and the complete
architecture suite passed together with `268 passed in 44.46s`. The existing
direct-builder active-path test and the new architecture ownership check passed
together with `2 passed in 1.67s`. TensorFlow-import-blocked explicit direct,
default direct, and `-cotof` conversion passed with
`3 passed, 8 deselected in 3.77s`.

One sequential YuNet conversion was compared with source checkpoint
`5a186acd`. Every emitted file was byte-identical. Float32 remains 236,564 bytes
with SHA-256
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`;
float16 remains 131,120 bytes with SHA-256
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`;
and the 105,578-byte correspondence report remains
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`.
Both schema files also match. The detached worktree and all temporary outputs
were removed. No Tier corpus run was performed.

Scoped Ruff, syntax compilation, and `git diff --check` pass. No dependency or
TensorFlow import was added. There is no source-file 2,000-line Goal
requirement; the 2,000 threshold applies only to ONNX operation-count Tier
classification.

### Direct Split root continuation

The adjacent
`_optimize_transpose_split_channelwise_tail_to_single_post_nchw` helper is now
also extracted. Its two production positions and stats key are unchanged. A
separate immutable direct-root plan proves the private NHWC-to-NCHW Transpose,
the channel Split, and the shared closed-tail contract before the first write.
Only Split outputs seed closure traversal, so unrelated consumers of the raw
NHWC source are no longer accidentally classified as converted tensors.

The shared closure now optionally plans rank-four Slice operations. Typed
INT32/INT64 begin and size constants, bounds, original NCHW result metadata,
converted NHWC metadata, and dynamic signatures are all validated. A constant
changes in place only when every actual consumer slot is one of the planned
Slice uses with the same replacement. Otherwise every planned use shares one
deterministic clone, preserving both declared and NumPy dtype. Invalid or
conflicting constants reject the whole candidate without mutation.

The root Split axis uses the existing copy-on-write contract, the public output
keeps its original NCHW shape/layout, and one private NHWC producer output feeds
the terminal adapter. All metadata, axis, Concat, Slice-constant, graph-index,
and LayoutState changes are applied only after full re-resolution and
preflight. The configurable 32-rewrite ceiling and edge-bounded consumer queue
replace the old unbounded fixed-point scans.

The 20-test direct-root suite and 33-test binary-root suite pass together. The
two preceding binary owners, both Split-root suites, and the complete
architecture suite pass with `288 passed in 44.74s`. TensorFlow-import-blocked
explicit direct, default direct, and `-cotof` conversion pass with `3 passed in
3.85s`. A sequential YuNet comparison against `5f8e662a` emits five
byte-identical files with the same float32, float16, and correspondence hashes
recorded above. The detached worktree and temporary outputs were removed; no
Tier corpus run was performed.

### Unary/Split/Concat root continuation

The preceding `_optimize_transpose_unary_split_concat_single_post_nchw` helper
is now extracted into the same semantic owner under a separate immutable root
plan. Its two production positions, order relative to the direct and binary
Split roots, and stats key are unchanged. Both calls now supply Session
LayoutState.

The plan proves an exact typed private NHWC-to-NCHW Transpose, one allowed
pre-Split unary, one channel Split, every Split output exactly once through
either a direct edge or one allowed unary, exactly one external branch, and a
channel Concat. All converted branch fan-out is closed. Public intermediates,
duplicate or missing Split branches, multiple external branches, duplicate
producers, stale order, per-axis quantization, and malformed unary operators
reject the candidate without mutation.

An external singleton Reshape is bypassed only when its source/output shapes
and dynamic signatures have the exact NHWC/NCHW relationship and both channel
dimensions are one. Dtype, quantization, provenance, graph order, and physical
layout are also validated. Already-proven direct NHWC external inputs remain
supported. Shared external Reshape consumers are untouched.

The Concat output may be public or an intermediate legacy NCHW tensor. One
private NHWC producer output and the proven input permutation preserve that
local contract. Split-axis copy-on-write, metadata, Concat inputs/options,
adapter insertion, graph-index updates, and LayoutState updates occur only
after full re-resolution and preflight. This removes the raw path that could
change the Split axis, metadata, and Concat before discovering an invalid
output tensor.

Pre-extraction characterization recorded zero matches in all 24 invocations:
four each on YuNet, FastestDet, HumanSeg, and OSNet, and eight on SiNet. The
29-test owner suite, existing active-path test, adjacent Split owners, binary
bridge owners, and architecture suite pass together with `318 passed in
43.96s`. TensorFlow-import-blocked explicit direct, default direct, and
`-cotof` conversion pass with `3 passed in 3.83s`. A sequential YuNet
comparison against `0b4a0001` emits the same five byte-identical files. The
detached worktree and all temporary outputs were removed; no Tier corpus run
was performed.

### RELU/Split all-output continuation

The three adjacent raw Split/Conv/Concat helpers were audited together before
implementation. They are semantically exclusive: the all-output root requires
`Transpose -> RELU -> Split` with every Split result consumed by one inverse
Transpose; the Conv/Concat root adds an exact two-branch Conv/RELU/Concat tail;
and the bridge root begins with a direct Transpose-produced Split and retains a
local NCHW compatibility adapter after its Concat. The audit recorded 41
sequential runtime invocations on YuNet, FastestDet, HumanSeg, OSNet, and SiNet
and every invocation returned zero without changing operator or tensor counts.
The all-output root accounted for 12 of those invocations.

The new all-output owner resolves graph-ordered Split candidates against one
`ModelIRGraphIndex`. It accepts an immutable typed INT32/INT64 axis-one
constant, a private typed NHWC-to-NCHW adapter, RELU, an equal channel Split,
exactly one typed inverse adapter for every output, and all later consumers of
those adapters. Static shapes must be fully known and positive; dynamic shape
signatures are preserved and permuted independently. Source provenance,
unique producers, consumer multiplicity, graph order, dtype, per-tensor
quantization, physical layout, public boundaries, and `numSplits` are proven
before a plan exists.

Downstream rewrites are grouped by operator, so a Concat or other consumer that
uses several former post-Transpose outputs is updated once with every input
slot preserved. A shared Split axis receives one deterministic clone with the
original TensorIR and NumPy dtype; an exclusive axis changes in place. The
complete tensor/operator contract is resolved again immediately before apply,
then input rewrites, NHWC metadata, differential index compaction, LayoutState
updates, and success-only pruning occur as one bounded operation. Candidate
count and an optional explicit rewrite limit replace the raw unbounded
fixed-point loop.

Thirty-two focused tests cover two- and three-way Split, INT32 and INT64 axes,
negative axes, static and dynamic signatures, exact numerical equivalence,
one consumer using every branch, multiple consumers of one post output,
copy-on-write shared axes, candidate limits, idempotence, GraphIndex and
LayoutState integrity, and nineteen transactional unsafe no-op cases. The new
owner, adjacent indexed owners, active direct-builder fixture, and complete
architecture suite pass together with `475 passed in 46.51s`.
TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 4.01s`.

One sequential YuNet conversion emits the five artifacts already fixed by the
preceding checkpoint. Float32 remains 236,564 bytes with SHA-256
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`.
Float16 remains 131,120 bytes with SHA-256
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`.
The 105,578-byte correspondence report remains
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`;
`schema.fbs` remains
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `schema_generated.py` remains
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
No Tier corpus run was performed.

### RELU/Split/Conv/Concat continuation

The exact two-branch helper is now the second plan in the related Split-layout
owner. It requires a private NHWC-to-NCHW Transpose and RELU, a two-way equal
channel Split, one Split branch returning to NHWC for Conv2D, that result
returning to NCHW for RELU, the untouched branch joining it through channel
Concat, and one final inverse Transpose. Its two production positions and
order immediately after the all-output helper are unchanged.

The Conv branch may be either Split output and may appear in either Concat
slot. Neither order is normalized or commuted. Split input channels must divide
exactly by two; both output shapes/signatures and optional `numSplits` must
describe the equal result. The old synthetic fixture's invalid six-channel
2/4 split is corrected to a valid eight-channel 4/4 split. Conv output channels
may differ from the branch width, so old NCHW and new NHWC Concat results are
derived independently and required to agree through the final adapter.

Every typed permutation, source and Conv-side input provenance, unique
producer, exact consumer, graph-order edge, public boundary, tensor presence,
static/dynamic shape, dtype, per-tensor activation quantization, and physical
layout is resolved before a plan exists. Adapter endpoints must have identical
dtype and quantization. Fan-out from an owned intermediate, an additional or
unequal Split output, stale order, missing filter/bias, duplicate producer,
public intermediate, or contradictory layout rejects the candidate without
mutation.

The immutable plan records the pre-RELU, Split-axis, Conv, post-Conv RELU, and
all final-consumer input changes; five NHWC metadata updates; the Concat-axis
update; the four adapter removals; and every tensor/operator contract. Shared
Split axes use deterministic INT32/INT64 copy-on-write. A complete second
resolution and preflight precede all writes, after which one differential
graph index performs the input changes and one four-operator compaction.
Candidate count bounds execution, and pruning occurs only after success.

Forty-nine dedicated tests cover both Split branch positions, both Concat
orders, static and dynamic signatures, INT32/INT64 and negative axes, Conv
channel changes, exact numerical equivalence, multiple final consumers,
shared-axis cloning, candidate limits, idempotence, GraphIndex, LayoutState,
and twenty-nine transactional unsafe no-op cases. With the active fixture,
adjacent indexed owners, and complete architecture suite, `525 passed in
47.51s`. TensorFlow-import-blocked explicit direct, default direct, and
`-cotof` conversion pass sequentially with `3 passed in 4.33s`.

One sequential YuNet conversion reproduces the same five artifacts and hashes
recorded above: 236,564-byte float32, 131,120-byte float16, 105,578-byte
correspondence report, `schema.fbs`, and `schema_generated.py`. No Tier corpus
run was performed.

### Direct Split/Conv/Concat bridge continuation

The last raw helper in the audited three-member Split/Conv/Concat group is now
extracted to `passes/split_conv_concat_bridge_layout.py`. Its three production
positions and order relative to the preceding all-output and exact
Conv/Concat plans and the late QKV recovery remain unchanged. All three calls
now supply Session LayoutState.

The new resolver requires one private NHWC-to-NCHW pre adapter feeding an
equal channel Split exclusively. One Split result owns exactly one inverse
adapter; every other Split result must be a direct input of the same channel
Concat. Every remaining Concat input must be the exclusive output of an
NHWC-to-NCHW post adapter, and at least one post-adapter input must be reachable
from the converted branch before the Concat. The bounded reachability walk
allows an unchanged generic NHWC interior rather than relying on a model name,
a Conv opcode, or a hard-coded chain.

All typed permutations, INT32/INT64 axes, source provenance, producers,
consumers and exact input slots, public boundaries, graph order, equal Split
shapes, dynamic signatures, dtype, per-tensor quantization, physical layout,
Concat classification, and old/new Concat shapes are resolved before a plan
exists. Branch-side NHWC consumers are preserved through grouped exact-slot
rewrites. Unclassified inputs, fan-out on an owned NCHW edge, an unreachable
branch, stale order, duplicate producer, missing tensor, per-axis
quantization, or contradictory layout rejects the complete candidate without
mutation.

The immutable plan records input rewrites, Split metadata, axis copy-on-write,
the Concat axis and output change, the private NHWC Concat tensor, every
adapter removal, and all tensor/operator contracts. The original NCHW Concat
tensor name and metadata are retained behind one post adapter reusing the
proven input permutation. A full second resolution and preflight precedes all
writes; one differential graph index performs the transaction, LayoutState is
updated differentially, and pruning is success-only.

The focused suite passes with `46 passed`. It covers two/three Split
outputs, either branch position, one/two post paths, either Concat order,
static/dynamic signatures, INT32/INT64 and negative axes, exact numerical
equivalence, branch-side consumers, shared-axis cloning, candidate limits,
idempotence, GraphIndex, LayoutState, and twenty-four unsafe transactional
no-op cases. The new owner, adjacent indexed owners, two active fixtures, and
complete architecture suite pass together with `534 passed in 44.94s`.
TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 4.04s`.

One sequential YuNet conversion reproduced the fixed five hashes: float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
`schema.fbs`
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `schema_generated.py`
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
Temporary outputs were removed. No Tier corpus run was performed.

### Unquantized pseudo-Swish passthrough continuation

The raw `_optimize_swish_transpose_passthrough_chains` implementation is now
owned by `passes/activation_passthrough_layout.py`. Its position after registered
hard-activation cleanup in the ordered recovery prefix and its late
no-layout-compatible recovery position are unchanged. Both calls supply
Session LayoutState, and the legacy stats key remains unchanged.

The resolver requires a typed immutable rank-two-or-higher INT32/INT64
permutation, exact source/transposed shapes and independently dynamic
signatures, one Logistic consumer, and one residual Mul consumer in either
operand order. At least one typed inverse post adapter must close the
source-layout path. All producer, consumer-slot, graph-order, public-boundary,
dtype, per-tensor quantization, layout, and alias contracts are resolved before
a plan exists. Immutable operator-produced and constant sources are supported;
source buffers remain unchanged because elementwise Swish commutes with the
permutation.

All inverse-post aliases are grouped by exact downstream input slot and fold
to one representative source-layout Mul result. One public post alias may be
the representative. When the old transposed Mul tensor still has a legacy
consumer or is itself public, the plan inserts one adapter immediately after
Mul and reuses the proven pre-permutation. The old helper mutated the selected
post-permutation tensor to the opposite direction; the new structure leaves a
shared post constant and every unrelated consumer untouched.

Input rewrites, output changes, metadata, public lists, removals, and complete
tensor/operator contracts are immutable plan state and are fully re-resolved
before apply. One differential graph index performs all rewrites, one adapter
compaction, and the optional legacy-adapter insertion. LayoutState changes only
with the accepted transaction, candidate count bounds execution, and pruning
is success-only.

Pre-extraction and post-extraction characterization ran sequentially on YuNet,
FastestDet, HumanSeg, OSNet, and SiNet. The helper was invoked five times on
each of the first four models and six times on SiNet. All 26 invocations were
zero-match and preserved operator/tensor counts before and after extraction.

The focused suite passes with `56 passed`. It covers rank-three/rank-four,
static/dynamic signatures, INT32/INT64 permutations, both Mul operand orders,
one/multiple posts, repeated alias slots, public post selection, immutable
constant source, shared post permutation, legacy/public transposed boundaries,
exact numerical equivalence, candidate limits, idempotence, GraphIndex,
LayoutState, and twenty-seven unsafe transactional no-op cases. The new owner,
adjacent input and quantized-Swish suites, four active Swish fixtures, and full
architecture suite pass together with `304 passed in 44.59s`.

TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 4.19s`. One sequential YuNet
conversion reproduced float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
`schema.fbs`
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `schema_generated.py`
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
Temporary outputs were removed and no Tier corpus was run.

### Tanh-expanded GELU passthrough continuation

The raw `_optimize_gelu_tanh_transpose_passthrough_chains` implementation is
now an indexed compatibility dispatcher in the common activation-passthrough
owner. Its sole production position remains immediately after pseudo-Swish and
before center/size-offset recovery, its stats key is unchanged, and its call
receives Session LayoutState.

The resolver proves the exact nine-operation topology from square and cubic
multiplication through cubic-scale, outer-scale, offset, and final-scale
singleton constant roles. All five uses of the original transposed source, every
linear intermediate, unique producer, graph order, constant dtype/data/shape/
provenance, and per-tensor quantization are validated before a plan exists.
Both valid operand orders are retained for commutative binary operations.

Typed rank-two-or-higher INT32/INT64 permutations, static or independently
dynamic views, dtype, quantization, layout transitions, source provenance, and
inverse post aliases are resolved before mutation. Immutable runtime and
constant sources are supported. Repeated downstream input slots remain exact,
one public post alias may be the representative, and a public or consumed old
transposed final tensor receives one local adapter immediately after the final
Mul. The adapter reuses the proven pre-permutation; shared post constants and
unrelated consumers remain untouched.

The immutable plan records every input/output slot, tensor and operator
contract, public list, metadata update, removal, and optional insertion. It is
fully re-resolved before apply. One differential graph index performs the
transaction, LayoutState changes only after acceptance, candidate enumeration
is graph ordered and bounded, and pruning is success-only. This removes the
raw full-map rebuilds, unbounded fixed-point loop, and partial mutation path.

Pre- and post-extraction characterization ran sequentially on YuNet,
FastestDet, HumanSeg, OSNet, and SiNet. Each run reached the helper four times;
all 20 invocations were zero-match and preserved operator and tensor counts.

The dedicated suite passes with `56 passed`. It covers rank-three/rank-four
static and dynamic views, INT32/INT64 permutations, both operand orders,
one/multiple and repeated post aliases, public and legacy boundaries,
immutable constant input, shared permutations, exact numerical equivalence,
candidate limits, idempotence, GraphIndex, LayoutState, and thirty unsafe
transactional no-op cases. The Swish and GELU owners, adjacent input and
quantized-Swish suites, active fixtures, and complete architecture suite pass
together with `362 passed in 42.95s`.

TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 3.95s`. One sequential YuNet
conversion reproduced float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
`schema.fbs`
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `schema_generated.py`
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
Temporary outputs were removed and no Tier corpus was run.

### Center/size/offset terminal-head continuation

The raw `_optimize_center_size_offset_terminal_transpose_chains` implementation
is now a thin indexed compatibility dispatcher. Its sole production position,
public name, and stats key are unchanged, and its call receives Session
LayoutState. The dedicated `passes/center_size_offset_layout.py` owner has no
model-name or fixed operator-distance condition.

Three typed NHWC-to-NCHW roots are classified by complete semantic branches.
The singleton-channel center branch closes through
Logistic/Maximum/Minimum/Reshape to `[N,H*W]`; size uses the same activation
prefix plus `[N,C,H*W]` Reshape/GatherND; offset uses the matching direct
Reshape/GatherND tail. Static shape, dynamic signature, dtype, per-tensor
quantization, source provenance, layout, unique production, exact consumers,
graph order, and every private/public boundary are validated. Shape-compatible
multiple triples reject transactionally instead of being paired by proximity.

Gather coordinates are classified as immutable batch grid, dynamic axis
Reshape, and immutable channel grid using tensor shape, dtype, provenance, and
range evidence. The owner supports INT32/INT64 coordinates, arbitrary original
Concat order, a shared coordinate Concat, and the equal-grid case where either
constant name is numerically equivalent. Each coordinate output must be closed
over the planned GatherND slots before the order changes to
`[batch, axis, channel]`.

Size and offset reshape literals rotate to `[N,HW,C]`, including valid inferred
`-1` dimensions and both option fields. Shape constants are grouped by tensor
and consumer slot. Exclusive constants change in place; shared constants use
one deterministic typed clone, so unrelated Reshape consumers retain the old
`[N,C,HW]` value. Activation intermediates receive their proven NHWC metadata,
rank-three data views become NWC, and LayoutState is updated only after the
plan has been accepted.

Every input rewrite, option, constant value/use, metadata update, adapter
removal, tensor/operator contract, and public list is immutable plan state. A
full second resolution and preflight precede all writes. One differential
ModelIRGraphIndex performs the rewires and three-adapter compaction;
graph-ordered candidates plus a rewrite limit replace the raw full-map
fixed-point loop, and pruning is success-only.

Pre- and post-extraction characterization ran sequentially on YuNet,
FastestDet, HumanSeg, OSNet, and SiNet. The helper was reached four times per
model; all 20 invocations were zero-match and preserved operator/tensor counts.

The dedicated suite passes with `37 passed`. It covers static/dynamic shape,
INT32/INT64 buffers, operand order, per-tensor quantization, inferred Reshape
dimensions, shared constant and coordinate closure, exact numerical
equivalence, candidate limits, idempotence, GraphIndex, LayoutState, and
twenty-two unsafe transactional no-op variants. The new owner, adjacent
activation/layout suites, active fixtures, and complete architecture suite pass
together with `399 passed in 42.94s`.

TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 3.99s`. One sequential YuNet
conversion reproduced float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
`schema.fbs`
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `schema_generated.py`
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
Temporary outputs were removed and no Tier corpus was run.

### Pseudo-expanded LeakyReLU passthrough continuation

The raw `_optimize_leakyrelu_transpose_passthrough_chains` implementation is
now a thin indexed compatibility dispatcher. Its production position, public
name, two returned statistics, and historical passthrough-then-fusion order are
unchanged. The call receives Session LayoutState, and the existing indexed
pseudo-LeakyReLU fusion reuses the passthrough owner's differential graph
index instead of rebuilding it.

The resolver accepts only the exact two-branch expansion: `Neg(x)` followed by
negative Relu and singleton-alpha Mul, direct positive Relu, then ordered
`Sub(positive, scaled-negative)`. Both source uses, every intermediate, alpha
dtype/data/shape/provenance, unique producer, graph order, exact consumer slot,
shape and independently dynamic signature, dtype, per-tensor quantization,
layout transition, and public boundary are proven before a plan exists. Typed
rank-three/rank-four INT32 or INT64 permutations, either alpha operand slot,
and immutable constant sources are supported.

Every inverse post alias folds to one source-layout representative, preferring
one public alias when present. Downstream rewrites are grouped per operator and
slot, including repeated aliases. A legacy consumer or public use of the old
transposed Sub result receives one local adapter immediately after the join.
It uses the pre-permutation tensor. The raw helper instead rewrote a selected
post-permutation buffer to an INT32 opposite permutation, potentially changing
its dtype and every unrelated shared use.

The plan captures all rewrites, metadata, public lists, removals, optional
insertion, and complete tensor/operator contracts. It is fully re-resolved
before apply. One differential ModelIRGraphIndex removes the outer adapters,
changes the join output, and inserts the local boundary; LayoutState changes
only after acceptance. The existing fusion then converts the retained Sub in
place to native `LEAKY_RELU` and removes its four private producers using the
same index. Pruning remains success-aware and occurs at the composite end.

Pre- and post-extraction characterization ran sequentially on YuNet,
FastestDet, HumanSeg, OSNet, and SiNet. Each model reached the composite helper
four times; all 20 invocations reported zero passthrough rewrites, zero
fusions, and unchanged operator/tensor counts.

The dedicated suite passes with `51 passed`. It covers static/dynamic rank
three and four, INT32/INT64 permutations, both alpha positions, per-tensor
quantization, immutable constant source, one/multiple/public aliases, repeated
slots, legacy and public transposed boundaries, shared post constants, exact
numerical equivalence, candidate limits, idempotence, GraphIndex, LayoutState,
and twenty-seven unsafe transactional passthrough no-op variants. The new
owner, indexed fusion owner, adjacent activation/layout suites, active fixtures,
and complete architecture suite pass together with `467 passed in 42.33s`.

TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 4.01s`. One sequential YuNet
conversion reproduced float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
`schema.fbs`
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `schema_generated.py`
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
Temporary outputs were removed and no Tier corpus was run.

### PReLU passthrough continuation

The raw `_optimize_prelu_transpose_passthrough_chains` implementation is now a
thin compatibility dispatcher over
`passes/prelu_passthrough_layout.py`. Its three ordered production positions,
public name, statistic key, and zero-match tensor pruning remain unchanged.
Every call now receives Session LayoutState.

The resolver proves one typed rank-three-or-higher input Transpose, exactly one
PReLU data consumer, at least one typed inverse post-Transpose, unique
producers, graph order, exact consumer slots, static shape, independently
dynamic signature, dtype, per-tensor quantization, provenance, public
boundaries, and known logical/physical layout transitions. Other consumers of
the pre-Transpose result are permitted and retain that adapter. All post aliases
are grouped by exact downstream slot, including repeated inputs, and one public
alias is preferred as the source-layout representative.

Alpha selection retains the historical priority: rank-equal inverse-layout
transpose first, original data second, and the rank-three NCHW channel form
third. Every selected value must broadcast to both the concrete source view
and its dynamic signature. An exclusively owned alpha changes in place; a
shared alpha receives one deterministic `_nhwc` copy that preserves declared
and NumPy dtype, quantization, layout metadata, and ONNX provenance. Scalars,
rank-three/rank-four channel parameters, and ambiguous equal static shapes are
covered.

A legacy consumer or public use of the old transposed PReLU result retains one
local adapter. When its permutation buffer is exclusively owned, that existing
adapter and buffer are reused so the historical correspondence lineage is
unchanged. If the post permutation is shared, the adapter instead uses the
immutable pre-permutation and leaves every unrelated use untouched. INT32 and
INT64 buffers retain their dtype. The raw helper could overwrite a shared post
buffer with an INT32 opposite permutation.

The complete immutable plan is resolved a second time before apply. One
differential ModelIRGraphIndex rewires PReLU and alias slots, compacts only
proven adapters, and retains any required local boundary. LayoutState changes
only after acceptance. Graph-ordered Transpose candidates plus an optional
rewrite limit replace the raw unbounded fixed-point loop.

Pre- and post-extraction characterization ran sequentially on YuNet,
FastestDet, HumanSeg, OSNet, and SiNet. Each model reached the helper six times.
The first four models remained zero-match; FastestDet also retained the former
zero-match 519-to-518 tensor prune. SiNet retained two active invocations of 23
rewrites each. Its float32, float16, correspondence, and both schema artifacts
are byte-identical to `5762128f`, including float32
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and correspondence
`24c423ea51b26b178d3764be027855e797bbf9b5ba1930810d2e1dbe281d8e25`.

The dedicated suite passes with `28 passed`. It covers static/dynamic rank
three and four, INT32/INT64 permutations, in-place and copy-on-write alpha,
scalar and ambiguous alpha forms, pre fan-out, multiple/public/legacy post
boundaries, repeated alias slots, shared post permutations, numerical
equivalence, candidate limits, idempotence, GraphIndex, LayoutState, and
fifteen unsafe transactional no-op variants. The PReLU owner, adjacent
activation/layout owners, active fixtures, and complete architecture suite
pass together with `454 passed in 50.22s`.

TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 4.38s`. One sequential YuNet
conversion reproduced all five fixed hashes. Temporary worktrees and outputs
were removed and no Tier corpus was run.

## Completed work

The merged `fb-refactor4` checkpoints included:

- `062ddc4` — centralized DepthToSpace/Gather repair and the permuted-Conv
  statement decoder;
- `a522f1b` — centralized static NHWC Pool layout selection;
- `5eb86e1` — centralized CF Pool-neighbor repair with an explicit
  short-circuit contract;
- `d16265b` — centralized dynamic Pool layout repair and added the exact
  aligned-rank4 decoder;
- `e0bc280` — centralized simple-alias layout repair and moved all
  permuted-Conv decoder consumers into the policy owner;
- `afb5bb5` — centralizes aligned scalar-binary
  shape reconciliation and removes the now-unused aligned-rank4 and Softmax
  parser imports from the exporter.

The current `fb-refactor5` work contains 136 coherent implementation
continuations:

- `3ac19b40` centralizes the ordered fallback that repairs aligned binary
  shapes only when general binary repair made no change and the immediate next
  statement supplies matching BN, direct-return, or channel-first Resize
  evidence;
- `008e4ad0` centralizes the following Resize fallback, including
  exact direct and reshaped BN-constant parsing, preferred-channel selection,
  input-layout guards, and CF evidence propagation;
- `80d1d6a5` centralizes the aligned BatchNorm-constant rewrite itself while
  preserving the different direct and already-reshaped guards;
- `91a0a52d` centralizes LRN output evidence propagation without changing or
  broadening the generated source grammar;
- `b00774a7` centralizes literal static-shape recording while retaining each
  update at its original point in the ordered scan;
- `95fbd0cb` makes the NHWC AveragePool bridge own the CF/NHWC and static-shape
  state resulting from its rewrite;
- `907c91fa` routes all direct-export option reads through the normalized
  request and adds a structural boundary test;
- `5848cc28` adds request-aware optional exporter controls and removes eager
  parsing of unrequested PyTorch settings;
- `e3c03e3d` makes quant type and input/output quant dtype part of the guarded
  immutable quantization controls;
- `4f3d20b0` restores Session-owned consumer counts at the `LoweringContext`
  boundary;
- `6e8a8486` synchronizes lowering-time logical and physical layout mutations
  with the Session before the first post-lowering pass;
- `e7c1457d` centralizes lowering-time operator removal and protects synthetic
  inverse-Transpose fan-out with differential consumer counts;
- `2a2968cc` lazily shares `ModelIRPassState` across the repeated
  Mean/attention registered-pass cluster;
- `514fc683` applies the same bounded reuse contract to the repeated mixed-
  attention through dual-Mul/Concat registered-pass cluster;
- `9b32c680` shares state only across the separately audited late NDHWC-gate/
  cost-volume-scatter pair;
- `1aa636e3` shares state across the following four registered Concat,
  LayerNorm, and transpose-cleanup runners;
- `969d5e26` shares state across the repeated channel-shuffle/Gather-axis and
  unary fan-out runner clusters;
- `251edc58` shares state across four repeated boundary-input BatchMatMul/input-
  unary runner pairs;
- `543d7cc3` shares state across three repeated channel-slice-merge/Pad-Mul
  pairs;
- `93a0295a` shares state across the repeated long singleton/
  Reshape sequences and all three terminal singleton-channel/duplicate-fan-
  out/consecutive-Reshape triplets;
- `9a75c43d` shares state across two repeated QKV attention prefix/bridge
  pairs;
- `417ee06e` shares state across two repeated duplicate-fan-out/quantized-PReLU
  pairs;
- `0c76774e` shares state across two repeated constant-input-fold/redundant-
  Cast pairs;
- `f0dac050` shares state across the fallback and primary absolute-final SE-FC/
  Gather-channel-fan-out pairs;
- `e177face` shares state across the five-runner terminal boundary/layout
  sequence;
- `ce11c27f` shares state across the late Dequantize/Concat/Quantize and unary-
  fan-out sequence;
- `d8b2b58c` shares state across the terminal singleton-MaxPool/consecutive-
  Reshape pair;
- `fcae7233` shares state across the terminal scalar-clamp, unary-passthrough,
  and maximum-zero-to-ReLU sequence;
- `36f73b18` shares state across the late Mean/Mul/Add/Conv, generic SPP, and
  Gather-axis sequence;
- `887db85d` shares state across the late generic SPP and Concat/unary/Conv
  pair;
- `9680fa33` shares state across the absolute-final normalization-Pad and mixed-
  attention pair;
- `a8baec98` shares state across the post-QDQ layout-transpose and unary fan-out
  sequence;
- `916419d9` shares state across the late NCHW channel-shuffle and Gather-axis
  pair;
- `8eaab05b` shares state across the conditional late generic-transpose and
  QKV-bridge pair;
- `e6be5539` shares state across the very-late Gather-axis, constant-fold/Cast,
  and normalization-Pad sequence;
- `fa33fd67` shares state across the terminal hard-activation and optional
  generic-Transpose pair;
- `57d79e3b` shares state across the conditional generic-Transpose, late
  Mean/SPP/Gather, and constant-fold/Cast sequence;
- `a353580b` makes `ArtifactPlan` the only request input to artifact controls
  and progress planning;
- `5e4e14d5` centralizes TFLite evaluation-path selection from returned direct
  artifacts;
- `d7fb5969` centralizes direct report and quantized-artifact
  validation and completion logging without changing messages or skip
  behavior;
- `603d6557` makes the direct fast path the sole direct conversion
  owner and removes the unreachable TensorFlow-failure fallback and duplicate
  post-SavedModel direct serialization path;
- `69a4eccd` centralizes the four identical 19-call layout-recovery prefixes
  without sharing pass state across raw mutation boundaries;
- `c89e94b5` centralizes two identical 22-call attention and quantized-
  recovery suffixes while preserving the separate LayerNorm variant;
- `329306d3` centralizes three identical 16-call layout/reshape/attention
  recovery prefixes while preserving their distinct successors;
- `40dcd142` centralizes two identical 14-call terminal slice/Concat layout-
  recovery sequences while retaining their boundary variants;
- `daab0828` centralizes two identical 11-call terminal affine/Concat/split
  recovery sequences between their distinct raw boundaries;
- `04c7dc03` centralizes three identical 10-call attention/gate/QDQ recovery
  sequences while retaining their distinct successors;
- `1713d089` centralizes two identical 10-call quantized-activation/binary-
  bridge sequences while preserving their conditions;
- `501e616f` centralizes two identical 8-call SiNet terminal recovery sequences
  without crossing shape reconciliation;
- `9bd57ac2` centralizes two identical 7-call pre-Add/Mean attention-recovery
  sequences while retaining their distinct boundaries;
- `ef61c03c` centralizes four identical 6-call SiNet pre-Add/Resize recovery
  sequences while retaining all external boundaries;
- `48dd2324` centralizes the remaining safe-binary and QLinear/Mean/Concat
  5-call families while preserving their conditions;
- `4a9bde4c` shares one differential graph index through each repeated
  prune/reconcile/Reshape-resolution convergence block;
- `20290fce` extends the final indexed convergence boundary
  through HARD_SWISH sanitation and activation fusion without rebuilding
  consumers or the graph index;
- `864af4c9` indexes repeated rank-four channelwise broadcast-constant repair
  while preserving its start-of-pass shared-constant policy;
- `0027ccfa` indexes stale channelwise-binary Transpose repair and shares one
  index across both terminal three-round convergence loops;
- `902cab42` indexes the singleton-Reshape and stale-Transpose Conv-input
  repair pair and shares one index across its primary and fallback invocations;
- `79d30ae1` indexes the wrong-way NCHW-to-NHWC Transpose-before-Conv
  sanitizer and removes its per-match consumer-map rebuild;
- `1ad30cbc` centralizes direct and PyTorch recurrent orphan-step alias repair
  in one Torch-free differential-index owner;
- `2574ae1f` extracts all unbound-input layout repair families to one
  differential-index owner and removes their repeated graph rescans;
- `16bba4ea` extracts quantized RELU/RELU6 Transpose bridge cleanup to one
  differential-index activation owner;
- `bbc9d345` extracts both expanded HardSigmoid QDQ Transpose
  bridge forms to that owner, adds transactional constant preflight, and
  protects every clamp intermediate at the public boundary;
- `515bc99b` extracts expanded MUL/ADD/PRELU QDQ Transpose bridge cleanup and
  shares only the identical constant plan/apply mechanism;
- `49f53b1a` extracts quantized logistic-gated MUL bridge cleanup to a
  dedicated indexed owner with differential alias consolidation;
- `30d00239` gives the wrong-way Transpose-before-Conv sanitizer one dedicated
  owner and removes its duplicate Swish-local implementation;
- `bad1a806` extracts the primary Swish-QDQ NHWC branch rewrite into one
  differential-index owner with an explicit phase result contract;
- `406136b5` extracts its four-family metadata fixed point and shares one index
  across both ordered primary phases;
- `a91d7bad` gives the two identical inverse post-Transpose sweeps one
  differential-index owner while preserving their separate call sites;
- `02462462` extracts transactional late mixed-input Concat
  normalization and shares one maintained index with its following post-
  Transpose cleanup;
- `03742a6a` moves Concat pre-Q/DQ exact-grid bypass to its
  quantization owner and replaces repeated whole-graph maps with one
  differential index;
- `55ad5c88` moves both terminal Transpose/Dequantize sanitation
  subphases to the same owner and maintains one index through edge rewrites,
  operator movement, rename, and removal;
- `f6b62363` moves the Transpose-DQ-Mean-Q bridge to one indexed, fully planned
  quantization-cleanup transaction;
- `3042329e` moves pseudo-op LeakyReLU fusion to one indexed graph-cleanup
  owner with batch producer compaction;
- `9a513d4c` moves the former YOLO MUL-square fold to a generic indexed
  constant-fold owner and protects public intermediates;
- `616a6a6b` moves leading-singleton Gather-to-Reshape cleanup to one indexed
  shape/indexing owner and makes every unsafe metadata or topology case
  transactional;
- `f3da692f` moves marker-gated terminal Softmax/Transpose cleanup to one
  indexed terminal-layout owner and centralizes the propagation marker;
- `e1e8ab39` moves pre-ArgMax channel-layout cleanup to one indexed owner with
  transactional shape and constant-ownership guards;
- `0cfc1ef9` moves exact-grid quantized MaxPool cleanup to one indexed owner
  with transactional topology, grid, and metadata guards;
- `e3dde5a4` moves canonical quantized Logistic cleanup to one indexed owner
  with transactional topology, grid, and metadata guards;
- `2a6178f6` moves canonical last-axis quantized Softmax cleanup to one indexed
  owner with transactional option, grid, and metadata guards;
- `163f9875` moves expanded HardSigmoid QDQ cleanup to one indexed owner with
  a complete four-constant transaction;
- `cc699155` moves quantized TransposeConv QDQ cleanup to one indexed owner
  with a complete filter/output transaction;
- `2b181ed7` moves decomposed InstanceNormalization layout repair to one
  indexed owner with a complete tensor-metadata transaction;
- `558973fd` moves NCHW Concat/global-pool/Conv axis repair to one indexed
  owner with a complete options/metadata/buffer transaction;
- `78ba42ae` moves NCHW Concat/Transpose/(Transpose)Conv axis repair to one
  indexed owner with a complete metadata transaction;
- `bee33d8e` moves mixed singleton NCHW-input repair for NHWC Concat to one
  indexed owner with complete adapter transactions;
- `84ac0fae` moves Swin-style window-partition canonicalization to one indexed
  owner with complete topology/constant/metadata transactions;
- `b134767b` moves the paired Swin-style window-reverse
  canonicalization to that owner with deterministic shared-shape cloning and
  complete topology/constant/metadata transactions;
- `b86ce908` moves the Conv1D-shim Squeeze/Unary/ExpandDims
  canonicalization to one indexed owner with complete shape, topology,
  constant, dtype, and quantization transactions;
- `54d37f35` moves the adjacent rank-four
  Unary/Reshape/ExpandDims variant to that indexed owner, makes shared
  constant changes transactional, and keeps fan-out compatibility bridges
  topological;
- `0837ee1b` moves the unary fan-out bypass to the same indexed
  owner, shares the common Transpose/Squeeze/Unary prefix contract, preserves
  only genuine NCHW side branches, fixes operator ordering, and repairs CAST
  metadata;
- `54384df0` moves flattened InstanceNormalization Conv1D layout
  canonicalization to a dedicated indexed owner with complete decomposition
  and shared-constant transactions;
- `6be406ec` shares that exact normalization prefix with a
  dedicated indexed tencoder residual-gate owner, makes the complete dual-
  branch rewrite transactional, and keeps compatibility bridges topological;
- `2202bd0d` moves the adjacent Squeeze/unary/BatchMatMul layout rewrite to an
  indexed owner with explicit axis-to-adjoint semantics;
- `0ef2050c` moves the decoder deconvolution-input adapter to a complete
  indexed matrix/layout/constant transaction;
- `481098db` moves the terminal Squeeze/Mean decoder adapter to a complete
  indexed axis/layout/output transaction;
- `fd3d1d32` moves the direct decomposed-InstanceNormalization pre/post adapter
  to a complete indexed topology/layout/constant transaction;
- `c7496639` moves its dual-consumer post-Transpose plus side-Squeeze tail to
  the same indexed owner with a transactional local compatibility adapter;
- `50278afa` moves the Squeeze/unary/Reshape tail to the same indexed owner
  with a transactional second-Reshape constant and output-name rewrite;
- the current checkpoint extracts the common SiNet Shuffle residual prefix,
  moves the paired post-MUL Transpose variant to the same indexed module,
  preserves the already-NHWC ADD/PReLU tail, and removes its second legacy raw
  mutator from the compatibility dispatcher.
- the latest checkpoint moves the late residual affine/PReLU fan-out to that
  indexed module, preserves the conv and legacy branches with one inverse
  adapter, and replaces the spatial-size heuristic with semantic shape and
  broadcast validation.
- the current checkpoint moves the deep-skip Resize/dual-Concat/affine tail to
  a staged indexed owner, shares constant and metadata application across
  SiNet owners, and removes four adapters as one preflighted transaction.
- the latest checkpoint moves the adjacent pre-ADD Concat/PReLU fan-out to a
  dedicated indexed owner, shares the terminal affine/PReLU resolver with the
  late-residual owner, removes only the redundant Concat adapter, and retains
  both the direct sibling adapter and legacy NCHW branch.
- the current checkpoint unifies both dual-Resize residual variants in one
  indexed owner, makes direct versus sibling residual-adapter ownership
  explicit, and removes two more independent full-map raw mutators.
- the latest checkpoint moves the shared-post affine/PReLU fan-out to a
  dedicated indexed owner, validates exactly one channel-last Concat-backed
  input and the complete Conv/ADD consumer set, and removes three adapters as
  one preflighted transaction without a fixed spatial-size heuristic.
- the current checkpoint moves the mid-stage Concat/Resize affine residual
  island to a dedicated indexed owner, validates Resize provenance and both
  original/target Concat contracts, and preserves post aliases plus the legacy
  NCHW residual branch in one preflighted transaction.
- the latest checkpoint moves the two-Concat affine tail to a dedicated
  indexed owner, reuses the adapter/Resize branch contracts, validates both
  residual stages and eight constants, and preserves post aliases plus final
  legacy NCHW consumers transactionally.
- the current checkpoint moves the active Softmax-mask residual tail to a
  dedicated indexed owner, validates the channel Softmax and mask-shape axis
  transforms plus six grouped constants, and preserves post aliases and final
  legacy NCHW consumers in one preflighted transaction.
- the latest checkpoint moves the dormant double-Logistic mix-attention path
  to a dedicated indexed owner, validates the complete CA/SA/PA topology and
  both residual-source variants, replaces eight or nine removable adapters as
  one transaction, and corrects the legacy branch rewire that could leave an
  unbound NCHW tensor when this compatibility topology did match.
- the current checkpoint moves the preceding SA/PA MirrorPad compatibility
  path to a dedicated indexed owner, validates the singleton-channel condition
  required for numerical equivalence, preserves both direct-NHWC and legacy-
  NCHW Mul outputs, and removes five or six adapters as one transaction.
- the latest checkpoint moves the general transpose/binary bridge path to a
  dedicated indexed owner, preserves symmetric single-post, mixed-fan-out,
  synthesized legacy-adapter, and asymmetric forms, and applies every accepted
  rewrite only after permutation, source, shape, quantization, public-boundary,
  and graph-order contracts have been resolved and revalidated.
- the current checkpoint folds the five later safe binary recovery helpers
  into the same indexed owner, preserves their legacy-only, single-post,
  mixed multi-post, asymmetric fan-out, and full-post phase order, and shares
  one differential index per sequence invocation rather than rebuilding five
  producer/consumer maps after every rewrite.
- the latest checkpoint moves the binary/Split channelwise tail to a dedicated
  indexed owner, replaces repeated full-graph closure scans with a bounded
  consumer worklist, preserves all six binary operations and the public NCHW
  output, and applies Split axes, Concat options, metadata, and the terminal
  adapter as one revalidated transaction.
- the current checkpoint moves the direct Split channelwise tail to the same
  indexed owner under a separate root plan, adds fully transactional Slice
  begin/size remapping with copy-on-write for shared constants, stops unrelated
  source consumers from entering the closure, and preserves the original
  public NCHW contract through the common terminal adapter.
- the latest checkpoint moves the unary/Split/Concat compatibility island to
  the same indexed owner under a separate root plan, proves every Split branch
  and the one external branch, restricts the layout-only Reshape bypass to a
  true singleton channel, and preserves either a public or intermediate NCHW
  Concat contract as one revalidated transaction.
- the current checkpoint moves the singleton gate/Conv/Concat compatibility
  island to a dedicated indexed owner, proves both auxiliary variants and the
  optional RGB bridge, preserves view-equivalent side consumers, and applies
  all rewrites, metadata updates, constant reshapes, and adapter removals only
  after full plan revalidation.
- the latest checkpoint moves the active Conv-family output passthrough chain
  to a dedicated indexed owner, preserves every producer/unary/binary variant
  and operand slot, groups shared side-constant updates, and removes both
  layout adapters only after a complete revalidated transaction.
- the current checkpoint moves the separate channel-one TransposeConv/Squeeze
  terminal family into the same op-family module under an independent plan,
  validates semantic Squeeze-axis order and the exact graph output, and removes
  its leading adapter only after complete revalidation.
- the latest checkpoint moves the Transpose/RELU/Split all-output island to a
  dedicated indexed owner, proves every inverse-adapter branch and downstream
  input slot, and applies equal-Split metadata plus copy-on-write axis changes
  only after complete revalidation.
- the current checkpoint moves the adjacent exact two-branch
  Split/Conv/RELU/Concat island into a second plan in that owner, preserves
  both Split and Concat orders plus Conv channel changes, and removes four
  adapters only after complete contract revalidation.
- the latest checkpoint moves the direct Split/Conv/Concat bridge to its own
  indexed owner, classifies every direct and post-adapter Concat input, permits
  only a bounded proven NHWC branch interior, and preserves the original local
  NCHW Concat contract through one revalidated transaction.
- the current checkpoint moves unquantized pseudo-Swish passthrough to an
  indexed owner, proves all Logistic/Mul/post-alias slots, preserves an
  optional legacy transposed boundary with the pre-permutation, and eliminates
  mutation of shared post-permutation constants.
- the latest checkpoint moves tanh-expanded GELU passthrough to the same
  activation-family owner under an independent immutable plan, proves all nine
  operations, five source slots, four constants, and every post alias, and
  preserves public or consumed transposed boundaries with one local adapter.
- the current checkpoint moves center/size/offset terminal-head recovery to a
  dedicated indexed owner, replaces operator-distance pairing with complete
  branch/shape/coordinate contracts, and updates shared Reshape constants with
  deterministic copy-on-write before removing three layout adapters.
- the latest checkpoint moves pseudo-expanded LeakyReLU passthrough to a
  dedicated indexed owner, preserves public and legacy boundaries with the
  pre-permutation instead of mutating shared post constants, and hands the
  maintained graph index directly to the existing native-activation fusion.
- the current checkpoint moves PReLU passthrough to a dedicated indexed owner,
  preserves alpha remapping and zero-match pruning, uses copy-on-write for
  shared alpha, retains exact SiNet correspondence lineage, and never mutates
  a shared post-permutation buffer.
- the latest checkpoint moves connected elementwise/Concat NHWC recovery to a
  dedicated indexed owner, replaces the unbounded full-map loop with
  edge-bounded immutable group plans, preserves every unary/binary and legacy
  boundary form, and records converted layout consistently in TensorIR and
  Session LayoutState.
- the current checkpoint moves StridedSlice/Concat fan-in recovery to a
  dedicated indexed owner, proves every typed Slice parameter and whole-group
  boundary before mutation, preserves ordinary/alias lineage exactly, and
  replaces unsafe post-permutation mutation with typed pre-permutation reuse.

The extraction preserves the ordered source-rewrite behavior. Layout evidence
continues to mutate only the per-run CF/NHWC sets; repair context maps remain
shared. Rules that formerly used `continue` return an explicit short-circuit
result to the exporter. Exact generated-statement grammars remain rule-local or
use the shared Torch-free parser owner.

No dependency was added and no TensorFlow path was introduced. The latest
checkpoint includes one sequential direct-backend integration smoke; no Tier
corpus run was performed.

## Current branch and changed files

Branch: `fb-refactor5`, tracking `origin/fb-refactor5`.

The latest implementation checkpoint changes:

- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`;
- `onnx2tf/tflite_builder/passes/stridedslice_concat_layout.py`;
- `tests/test_flatbuffer_direct_indexed_stridedslice_concat_layout.py`;
- `tests/test_flatbuffer_direct_architecture.py`;
- `docs/flatbuffer_direct_architecture.md`;
- this handoff document.

The expected handoff state after committing and pushing is an empty `git
status --short` with local `fb-refactor5` equal to `origin/fb-refactor5`.

## Important design decisions

- The binary/Split tail has its own `split_channelwise_layout.py` owner rather
  than extending the binary bridge owner. Its semantic boundary is the closed
  axis-sensitive Split/Concat closure, not a source-line threshold.
- The consumer worklist is bounded by the current operator and input-edge
  count. A candidate is accepted only when every converted tensor terminates
  at an accepted closure operator or the sole public output; the pass does not
  silently expose NHWC data to an unsupported legacy consumer.
- The public tensor retains its original NCHW shape and receives the terminal
  adapter output. A deterministic private NHWC tensor receives the producer
  output. The proven input permutation constant is shared instead of creating
  another equal buffer.
- Split axis constants are immutable plan inputs. Private axes change in place;
  shared axes receive deterministic clones that preserve both TensorIR and
  NumPy INT32/INT64 dtype. Every plan is re-resolved before these writes.
- The direct Split root seeds closure discovery from Split outputs, not from
  the original NHWC Transpose input. Unrelated raw-source consumers remain
  untouched and do not affect the accepted tail.
- Slice begin/size tensors are grouped by identity across all planned uses.
  They change in place only if every real consumer slot participates with the
  same value; otherwise planned uses share one deterministic typed clone.
  Invalid bounds, conflicting roles, mutable constants, and per-axis
  quantization reject the transaction.
- The unary-root island requires the pre-Transpose and pre-unary edges plus
  every Split branch to have exact closed ownership through the Concat. Every
  Split output appears exactly once, with at most one allowed branch unary,
  and there is exactly one external Concat input.
- The external Reshape bypass is valid only for `[N,H,W,1]` to `[N,1,H,W]`
  with matching static/dynamic shape, dtype, per-tensor quantization,
  provenance, and physical-layout evidence. Other consumers of the Reshape
  output continue using the original NCHW view.
- The unary-root adapter is a local boundary rather than a sole-public-output
  boundary. It preserves the original Concat tensor name and NCHW metadata for
  either graph outputs or existing downstream consumers while moving only the
  producer output to a private NHWC tensor.
- The singleton-gate owner accepts only the exact gate topology and true
  `[N,H,W,1]`/`[N,1,H,W]` view adapters. Both the direct auxiliary and Logistic
  auxiliary variants are explicit plan forms; no model-name or chain-name
  special case is used.
- Optional RGB propagation removes an input Transpose only when its typed
  permutation, ownership, order, shape/signature, dtype, and quantization are
  proven. A direct stale-NCHW input is accepted only for a private constant
  with physical NHWC evidence, and its data buffer is reshaped together with
  its metadata.
- Every singleton-gate input rewrite, tensor/operator contract, metadata
  update, and removal is immutable plan state and is re-resolved before apply.
  Shared view-equivalent adapter consumers are rewired, unrelated source
  consumers are preserved, and an unsupported fan-out rejects the whole
  candidate without mutation.
- Conv-output passthrough and channel-one terminal Squeeze chains remain
  separate plans because the former preserves an intermediate NHWC boundary,
  while the latter terminates at a graph output and remaps Squeeze axes. They
  share only typed adapter, contract, constant-planning, and apply primitives.
- The general Conv-output owner preserves all three Conv producer types, all
  historical unary/binary operations, and binary operand slots. It validates
  static/dynamic broadcast results and never commutes SUB or DIV.
- Rank-four binary constants are grouped by tensor identity. All planned slots
  share one NHWC value; an unrelated consumer forces one deterministic clone,
  while an exclusive constant changes in place. Scalar constants remain
  unchanged.
- The inverse-adapter output metadata is derived from the converted final chain
  output rather than blindly permuting the existing output metadata. The plan
  captures and revalidates every tensor/operator, constant, input/output slot,
  metadata update, and removal before its first write.
- Terminal Squeeze axes are remapped by semantic label. Each squeezed static
  and dynamic dimension must equal one, and the surviving semantic axis order
  must match between the original NCHW and converted NHWC paths. A spatial-
  only squeeze that would reorder a public rank-three output is rejected.
- Non-scalar terminal-chain constants are allowed only before Squeeze and use
  the common grouped rank-four planner. Post-Squeeze binary operations must use
  scalar constants. Public intermediates, a nonterminal graph output, a second
  Squeeze, or any output consumer rejects the complete plan.
- The RELU/Split all-output island remains separate from the exact
  Conv/Concat root and the direct Split/Conv/Concat bridge. Its semantic
  boundary is that every Split result immediately returns to NHWC through one
  owned inverse Transpose; it does not absorb either adjacent Conv topology.
- TFLite `SPLIT` is an equal-partition operation. The owner verifies that the
  channel count divides the output count and that every static/dynamic output
  shape matches the equal result. The former synthetic fixture's unequal 2/4
  metadata was corrected to the valid 3/3 contract.
- A downstream operator may consume several former post-Transpose aliases or
  one alias more than once. Replacements are grouped per operator and input
  slot before apply, preventing one branch rewrite from overwriting another.
- Axis constants use copy-on-write based on exact consumer slots. Private
  INT32/INT64 buffers change from axis 1 to 3 in place; shared buffers retain
  their original value while the Split receives a deterministic typed clone.
- The exact Conv/Concat root is independent from the all-output root even
  though both share one op-family owner. It owns exactly four adapters and one
  Conv branch. The direct Split/Conv/Concat bridge is a separate owner because
  it accepts a generic bounded NHWC interior and retains a local legacy NCHW
  boundary rather than owning a fixed Conv/RELU chain.
- Split-output position and Concat-input position are semantic order. The plan
  evaluates both positions but never reorders them. Conv may change channel
  width, and both original NCHW and target NHWC Concat shapes must prove the
  same final adapter contract.
- Changing the Conv and post-RELU inputs is safe only when the removed adapter
  endpoints have equal dtype and quantization. Conv filter/bias inputs are not
  rewritten, but each must have resolved provenance before the Conv.
- The pre-Transpose, branch-to-Conv adapter, Conv-to-RELU adapter, and final
  Concat adapter are an all-or-nothing removal group. Any fan-out or public
  boundary inside that group rejects the plan transactionally.
- The direct bridge classifies every Split and Concat edge before mutation.
  All non-branch Split outputs feed one Concat directly, every remaining
  Concat input is an exclusive post-adapter output, and a bounded graph walk
  must connect the inverse-adapter branch to at least one post-adapter input.
- The bridge does not validate a Conv opcode or a model-specific chain. The
  existing NHWC interior is left unchanged; only proven boundary adapters,
  Split/Concat axes, and exact consumer slots are part of the transaction.
- The original Concat tensor remains the local NCHW contract. A deterministic
  private NHWC tensor becomes the Concat producer output and one new post
  adapter reuses the proven pre-adapter permutation. Split axes preserve their
  INT32/INT64 dtype and use copy-on-write when shared.
- Unquantized pseudo-Swish is independent from Swish-QDQ. Its semantic root is
  one `Logistic(x) * x` residual island with inverse post aliases, not a
  Dequantize/Quantize closure. Both Mul operand positions are preserved.
- Typed permutations may be rank two or higher and INT32 or INT64. Static
  shape and dynamic signature views must match in both directions. Known
  logical/physical layouts must describe the same transition; unknown layout
  remains valid when tensor views prove the permutation directly.
- Post aliases are grouped by exact consumer slot. A public alias becomes the
  representative when present. Multiple public aliases reject the candidate
  because removing them would otherwise change the output-name contract.
- A legacy consumer of the old transposed Mul tensor is preserved by inserting
  one adapter immediately after Mul. The adapter uses the immutable pre-
  permutation instead of reversing a post-permutation buffer in place, so
  shared post constants and unrelated consumers cannot be corrupted.
- Immutable constant sources are supported without data mutation. Moving
  Logistic and Mul to the source view is exact because Swish is elementwise;
  runtime intermediates and adapter endpoints remain fully typed and
  per-tensor quantized.
- Tanh-GELU uses its own immutable plan within the activation owner. It accepts
  only the complete nine-operation approximation and proves all five source
  slots, linear intermediate ownership, unique production, order, and four
  immutable singleton constants before any layout rewrite.
- GELU post aliases follow the same exact-slot and public-representative
  contract as Swish. If the old final transposed tensor remains observable, a
  local adapter reuses the pre-permutation; neither activation pass mutates a
  shared post-permutation constant.
- Center/size/offset association is semantic rather than positional. One
  singleton-channel center branch, one activated size branch, and one direct
  offset branch must agree on batch/spatial shape and the size/offset data view;
  more than one compatible triple rejects instead of choosing the nearest
  Transpose.
- GatherND coordinate roles are proven from the dynamic Reshape and bounded
  immutable integer grids. Every coordinate Concat consumer slot must belong
  to the planned size/offset gathers before its inputs are reordered.
- Reshape shape constants are grouped by identity and exact slot. Exclusive
  INT32/INT64 constants rotate in place; a constant with any unrelated use is
  cloned once for the planned size/offset Reshapes. Inferred `-1` values and
  option metadata stay symbolic.
- Pseudo-LeakyReLU layout recovery and native activation fusion remain one
  ordered composite contract. The passthrough owner exposes a focused
  transaction entry point for rejection tests, while production always runs
  the indexed fusion afterward on the same ModelIRGraphIndex.
- The Sub join is order-sensitive: only
  `positive_relu - alpha * relu(-x)` is accepted. Every inverse post alias is
  grouped by exact slot, one public alias may be representative, and observable
  legacy NCHW use gets a local adapter immediately after the join.
- The local adapter reuses the typed immutable pre-permutation. No post-
  permutation constant is reversed or recast, so an INT64 or shared constant
  retains both its bytes and every unrelated consumer contract.
- The exporter remains the ordered orchestration owner; match/guard/rewrite
  decisions move to `pytorch_fast_precanonicalize_policy.py` one coherent
  family at a time.
- Indexed helpers receive the current line index, shared source lines, mutable
  layout evidence, and the shared repair context. They do not rescan the full
  generated source unless the preserved rule already required a bounded scan.
- Former loop `continue` behavior is represented explicitly in helper results;
  extraction must not silently allow later rules to run.
- General binary repair remains first. The downstream-evidence fallback is
  called only from its unchanged no-rewrite branch, and its returned CF
  evidence is visible to the following Resize repair in the same scan.
- The fallback deliberately retains its narrower positional grammar and legacy
  `_in` naming evidence. It additionally requires an immediate matching BN,
  direct return, or channel-first Resize; mismatched channels and mixed-layout
  names remain no-ops.
- General Resize repair also remains first. The input/BN-evidence fallback runs
  only afterward, uses an immediate matching direct or reshaped BN constant as
  the preferred channel hint when available, and otherwise retains the legacy
  input/source channel fallback. Its returned CF evidence remains visible to
  Pool and later aligned-constant decisions in the same ordered scan.
- Explicit NHWC Resize inputs and already-channel-first target shapes remain
  no-ops. BN evidence refines the preferred channel count but is not a
  prerequisite for the legacy CF-input repair.
- Direct aligned BatchNorm constants require a registered channel count that
  matches the generated target channel before a reshape is introduced.
  Already-reshaped constants intentionally retain the older, narrower rule:
  their explicit reshape channel drives normalization without requiring a
  registered-buffer channel lookup. Both forms still require CF input and a
  BatchNorm-derived attribute name.
- LRN output propagation is state-only: exact CF input evidence adds the output
  to the CF set, removes stale NHWC evidence, and copies only a known rank-four
  static input shape. It does not mark the source file changed or rewrite the
  LRN statement.
- Rewritten-shape caching accepts only a literal `target_shape=[...]` or the
  exact trailing aligned shape. Dynamic and unparseable expressions do not
  replace existing cache entries, and binary/Resize/Pool callers still update
  the shared context immediately after their successful rewrite.
- The NHWC AveragePool bridge keeps its returned-name contract, but successful
  calls now update the layout sets and all four affected static-shape entries
  internally. To preserve behavior, the cached state shape is still recomputed
  from the pre-rewrite Pool shape after the layout sets change; it is not
  replaced by the rendered rewrite target.
- `ConversionRequest.from_kwargs` is the direct exporter's only raw-kwargs
  boundary. Quantization validation receives `request.options`, normal option
  reads use `request.get`, and typed artifact decisions remain on
  `request.artifacts`. Checkpoint `907c91fa` mechanically converted all 36
  former raw reads without changing keys, defaults, coercions, public return
  values, or downstream arguments.
- `resolve_requested_exporter_controls` now owns seven artifact-specific
  settings. It performs no option reads when SavedModel, PyTorch, and integer
  calibration are all unrequested. Requested output paths, persistence,
  timeout conversion, shape/test data, and custom-input values preserve their
  existing defaults and dependencies.
- Requested quantization controls now also own `quant_type`,
  `input_quant_dtype`, and `output_quant_dtype`. The builder reads these values
  only from the resolved immutable mapping; when quantization is unrequested it
  uses the legacy local defaults without touching the corresponding options.
- Artifact execution controls, exporter controls, and export-progress labels
  now accept `ArtifactPlan` directly. They do not receive independently
  reconstructed split, quantization, SavedModel, PyTorch, or calibration
  booleans. Derived PyTorch artifacts are normalized once by
  `ArtifactPlan.from_options`, and all downstream policy sees the same
  immutable dependency decision.
- TFLite evaluation consumes the direct builder's returned artifact mapping
  through one fixed seven-key selector. The three compatibility-layer exit
  paths no longer own parallel key-copy chains, cannot diverge in variant
  order, and do not infer a path for an artifact the builder did not return.
- `LoweringContext.tensor_consumer_count` is populated from
  `ConversionSession.tensor_consumer_count`, not an empty compatibility
  dictionary and not a new ONNX scan. This restores the original fan-out guard
  used by inverse-transpose elision and preserves duplicate input occurrences.
- `LoweringContext.set_tensor_layout()` is the lowering-time layout mutation
  boundary. It normalizes and writes `TensorIR` metadata and immediately
  records the same logical/physical values in the Session-owned `LayoutState`.
  Shape-family edge-Pad passthroughs, integer-linear Resize casts, and rank-three
  Resize adapters no longer assign layout fields directly. This fixes observed
  pre-pass staleness without adding an eager ModelIR-wide synchronization.
- `LoweringContext.add_operator()` increments a differential consumer count for
  every emitted input occurrence. `remove_operator()` decrements those counts
  and removes only producer entries owned by the removed object. The inverse
  Transpose helper uses the authoritative ONNX count when present; otherwise it
  adds the pending inverse use to the current synthetic IR count. An exclusive
  pair is still elided, while a synthetic side consumer keeps its producer.
  This replaces the only direct op-builder deletion and adds no partial-graph
  scan.
- `ModelIRPassStateScope` is lazy and identity-bound. A group with a successful
  model-only preflight acquires the state; subsequent adjacent groups reuse it
  and report `state_built: false`, so diagnostic `state_build_count` reflects
  actual construction rather than pass invocation count. The scope is never
  carried across legacy helpers that mutate ModelIR outside the differential
  index. All six production occurrences retain the exact order
  transpose-Mean, Mean/Mul/Add/Conv, optional LayerNorm, terminal Mean, SE conv,
  SE FC, and optional Conv attention.
- Each repeated shape-convergence block uses one `ModelIRGraphIndex`.
  Dead-operator pruning updates that index through `remove_operators`; the
  following reconciliation and dynamic-Reshape steps change only metadata,
  options, and constant data. The first block builds its own index. The final
  convergence owner supplies one index to the second block and retains it
  through HARD_SWISH sanitation, a second Reshape/reconcile cycle, activation
  fusion, and final reconciliation. Standalone callers retain compatibility
  fallbacks, and an index for another ModelIR is ignored safely.
- Indexed activation fusion preserves the former graph-order and case-
  normalized op matching. It queries producer/activation fan-out from the
  index, updates producer outputs through `_set_operator_outputs`, and removes
  fused activation operators through `remove_operator`. It no longer rebuilds
  the full consumer map for every successful match. Differential single-
  operator removal drops empty type buckets so its type dispatch exactly
  matches a fresh index.
- Rank-four channelwise broadcast-constant repair takes an optional matching
  graph index and otherwise builds exactly one. It enumerates only the exact
  binary-op family, queries producer layout evidence through the index, and
  routes cloned-constant input changes through the differential setter. Its
  consumer fan-out map is intentionally snapshotted once from that index:
  clone-versus-in-place decisions therefore retain the former start-of-pass
  behavior even after earlier candidates update live consumers.
- Stale channelwise-binary Transpose repair also accepts an optional matching
  graph index. It retains exact graph-order binary matching, resolves adapter
  and NHWC peer producers through the index, and requires the indexed adapter
  consumers to equal the current binary index. Successful rewrites use the
  differential input setter and operator removal; a fan-out adapter remains
  untouched. `_run_indexed_binary_layout_convergence` owns the existing three
  broadcast-repair, Transpose-repair, and reconciliation rounds and supplies
  the same index to all nine calls in both primary and fallback finalization.
- The same scope contract covers only the five repeated gate-layout sequences
  that were audited as contiguous registered runners. Four keep the exact
  mixed-attention, elementwise-gate, Pad, dual-postconv-gate, NDHWC-gate,
  cost-volume-scatter, Add/Concat-suffix, and dual-Mul/Concat order. The fifth
  starts at elementwise-gate exactly as before. All eight runners retain
  standalone behavior through an optional `state_scope` argument, and the
  later isolated mixed-attention, Pad, NDHWC, cost-volume, and dual-Mul calls
  intentionally do not share this scope.
- The late mixed-attention/NDHWC/cost-volume candidate is not one valid scope:
  the raw dequantize/HardSigmoid/quantize optimizer between mixed attention and
  NDHWC is a hard boundary. A new scope is therefore constructed only for the
  immediately adjacent NDHWC and cost-volume runners, and ends before the raw
  convolution-affine optimizer. Both runners preserve standalone behavior.
- After that convolution-affine boundary, axis-3 constant-Concat,
  Dequantize/Concat/Quantize, LayerNorm-statistics, and generic transpose
  cleanup form one independently audited four-runner scope. Each runner either
  already accepted a scope or now exposes the same optional
  standalone-compatible argument; the scope ends before the conditional raw
  elementwise-roundtrip optimizer.
- Two repeated cluster families now have explicit helper-owned scopes. The
  channel-shuffle helper preserves two-way, NHWC, NCHW, and Gather-axis order
  at all five call sites; only its final invocation enables the already-
  contiguous generic transpose and unary/binary fan-out suffix. The separate
  unary helper preserves passthrough, unary fan-out, and unary/binary fan-out
  order at four call sites. Every runner remains callable standalone through
  an optional scope argument.
- Four repeated boundary-input BatchMatMul/input-unary pairs now use a small
  helper-owned scope. The boundary BatchMatMul runner and the three-spec input-
  unary runner expose the same optional standalone-compatible scope argument.
  No scope crosses the legacy transformations surrounding any occurrence.
  Their two stale `_build_tensor_consumer_map` imports are removed; neither
  module constructs an ad hoc consumer map for these runners.
- Three repeated channel-slice-merge/Pad-Mul pairs now use a two-group helper-
  owned scope. Both runners retain optional standalone-compatible scope
  arguments, and the scope ends before the legacy optimizer following each
  pair.
- Two long singleton/Reshape sequences now use one flag-controlled helper-
  owned scope per occurrence. The first retains generic transpose cleanup and
  terminal multi-branch gate cleanup; the second retains reshape-only
  duplicate fan-out cleanup and disables only the former spatial post-Concat
  variant. Four singleton-Reshape-family runners, three graph-cleanup runners,
  singleton MaxPool, and multi-branch gate retain standalone behavior through
  optional scope arguments.
- The three later singleton-channel/reshape-only-duplicate/consecutive-Reshape
  triplets use a target-parameterized helper. Two invocations use the primary
  ModelIR and Session layout state; fallback relowering passes `fallback_ir`
  and no LayoutState, preventing state identity from crossing conversion
  instances. The terminal singleton-MaxPool/consecutive-Reshape pair remains
  outside this target-parameterized helper and owns a separate bounded scope.
- Two repeated QKV attention prefix/bridge pairs use a two-runner helper-owned
  scope. The four prefix specs and two bridge specs retain exact order and
  diagnostic grouping. Both runners expose optional standalone-compatible
  scope arguments; the separate later bridge-only call remains independent.
- Two repeated duplicate-fan-out/quantized-PReLU pairs use a helper-owned
  scope. The helper forwards
  `enable_duplicate_transpose_fanout_optimizations` unchanged, then runs all
  four PReLU specs. The quantized-PReLU runner retains standalone behavior
  through an optional scope argument, and no scope crosses the following raw
  quantized TransposeConv cleanup.
- Two repeated constant-input-fold/redundant-Cast pairs use a helper-owned
  scope. The constant Pad, Pool, and Cast specs retain order before the
  redundant widening-alias and narrowing-chain specs. Both runners expose
  optional standalone-compatible scopes, and neither production scope crosses
  the immediately following legacy mutator.
- The fallback and primary absolute-final SE-FC/Gather-channel-fan-out pairs
  use a target-parameterized helper. Fallback receives `fallback_ir` and no
  LayoutState; primary receives the main ModelIR and Session state. Gather
  channel fan-out now exposes an optional standalone-compatible scope, and
  neither target's scope crosses shape reconciliation.
- The five-runner terminal dual-Mul/Concat, boundary-input, Pad, generic
  transpose, and Gather-channel-fan-out sequence uses one helper-owned scope.
  Boundary-input cleanup now exposes an optional standalone-compatible scope.
  Architecture checks fix the raw InstanceNorm predecessor and conditional
  Mean/attention successor as hard boundaries.
- The late Dequantize/Concat/Quantize, unary-passthrough, and unary-fan-out
  sequence uses one helper-owned scope. All three runners already expose
  optional standalone-compatible scopes. Architecture checks fix the raw
  Dequantize/HardSigmoid/Quantize predecessor and independently indexed Swish
  successor as hard boundaries. The Swish owner maintains its own differential
  index rather than reusing pass state across that semantic phase boundary.
- The terminal singleton-MaxPool/consecutive-Reshape pair uses one helper-
  owned scope. The two singleton-MaxPool specs retain their order before the
  general Reshape cleanup spec. Architecture checks fix the conditional
  elementwise-roundtrip predecessor and conditional Conv/Pool-output successor
  as hard boundaries.
- The terminal scalar-clamp, unary-passthrough, and maximum-zero-to-ReLU
  sequence uses one helper-owned scope. Clamp and maximum-zero runners now
  expose optional standalone-compatible scopes. Their op-type rewrites use
  `ModelIRGraphIndex.replace_operator_type()` instead of direct assignment, so
  later passes see current type dispatch without a full refresh. Architecture
  checks fix the conditional terminal layout-recovery predecessor and raw
  SiNet successor as hard boundaries.
- The conditional generic-Transpose, late Mean/Mul/Add/Conv, generic SPP,
  Gather-axis, and constant-fold/Cast sequence uses one helper-owned scope.
  Disabling layout optimization skips only the first runner. Architecture
  checks fix the complete order, runtime flag, shared scope keywords, and raw
  shape-extract/ExpandDims boundaries.
- The late generic SPP and Concat/unary/Conv pair uses one helper-owned scope.
  Concat/unary/Conv cleanup now exposes an optional standalone-compatible
  scope and retains its differential index mutations. Architecture checks fix
  the raw StridedSlice/Pad/Concat predecessor and raw shape-extract successor
  as hard boundaries.
- The absolute-final flattened-normalization Pad and mixed-attention pair uses
  one helper-owned scope. Normalization-Pad cleanup now exposes an optional
  standalone-compatible scope. The helper preserves `include_instance=False`
  and `include_flatten=True`; architecture checks fix those flags and the raw
  InstanceNorm/dynamic-rank shape-rewrite boundaries.
- The post-QDQ layout-transpose, unary-fan-out, and unary/binary-fan-out
  sequence reuses the existing helper with compatible mode flags. Four prior
  invocations keep their default unary-passthrough path; one new invocation
  enables layout-transpose and disables unary-passthrough. Architecture checks
  fix the unique flag combination and the raw Softmax/transpose-binary
  boundaries.
- The late NCHW channel-shuffle/Gather-axis pair reuses the existing helper
  with two-way and NHWC shuffle disabled. Five prior invocations retain both
  modes by default. NHWC and NCHW shuffle op-type changes now use
  `replace_operator_type()`; real rewrite tests assert the reused type index.
  Architecture checks fix the unique flag combination and raw Reshape/QKV
  boundaries.
- The conditional late generic-transpose/QKV-bridge pair reuses the QKV helper
  with prefix disabled. Two prior invocations retain the default prefix-plus-
  bridge path. The new invocation forwards `optimize_layout_transpose_chains`
  to the layout mode while always running bridge cleanup. Architecture checks
  fix the runtime flag and raw shape-extract/split-Conv boundaries.
- The very-late Gather-axis, constant-fold/Cast, and normalization-Pad sequence
  uses one wrapper-owned scope. The constant-fold/Cast helper accepts an
  optional external scope; both production invocations now receive their
  enclosing wrapper scope. The normalization include flags remain false/true.
  Architecture checks fix both external scopes, runner order, flags, and raw
  repair/Reshape boundaries.
- The terminal hard-activation and conditional generic-Transpose pair uses one
  helper-owned scope. Hard activation now exposes an optional standalone-
  compatible scope. Its late false/true/true/reversed flags are unchanged,
  generic Transpose still depends on the runtime layout switch, and the scope
  cannot cross either neighboring raw rewrite.
- The singleton-Reshape and stale NCHW-to-NHWC Transpose Conv-input repairs
  accept an optional matching `ModelIRGraphIndex` while retaining standalone
  compatibility. Their primary and fallback pair owner constructs one index,
  and successful Conv input rewrites and adapter removals update that index
  differentially. Exact filter input-channel, one-consumer, graph-output,
  tensor-shape, and Transpose-permutation guards are unchanged. The later
  standalone stale-Transpose cleanup remains a separate compatibility call
  because intervening raw mutations form an ownership boundary.
- The wrong-way NCHW-to-NHWC Transpose-before-Conv sanitizer has one Torch/
  TensorFlow-free semantic owner in `passes/conv_input_layout.py`. It owns one
  `ModelIRGraphIndex` for an invocation with Transpose candidates, enumerates
  indexed roots, validates every indexed consumer before changing a shared
  adapter, rewrites all accepted Conv inputs through the indexed global-input
  replacement helper, and removes the adapter differentially. A graph without
  any Transpose skips index construction while preserving the former tensor-
  pruning side effect. The lowerer compatibility wrapper and the independent
  safety-valve phase inside the Swish-QDQ optimizer both delegate to this same
  owner; the latter preserves its historical execution point and removal
  statistics. Exact permutation, rank-four metadata, all-consumers-are-Conv,
  filter-channel, graph-output, and nonempty-consumer guards are unchanged.
- The primary Swish-QDQ branch phase has one Torch/TensorFlow-free owner in
  `passes/quantized_swish_layout.py`. Its result carries the exact branch and
  pre-Transpose counts plus an immutable rewritten-tensor set into the
  existing later phases. A matching supplied index is reused; otherwise one
  index is built only when a Transpose exists. Graph-order candidates, all
  source/gate/data/tail guards, both DQ rewrites, and unused pre-Transpose
  removal use the maintained index. Only source edges and an unused root can
  change, so downstream consumers are read directly without full-map copying.
  Shared-input ordering, quantized and float tails, peer-Swish acceptance,
  spatial and concat-closure modes, public boundaries, fan-out, metadata
  permutation, and ordered restart retain the former implementation exactly.
- The Swish-QDQ metadata phase keeps unary quantization, binary broadcast,
  Pool/Resize channel propagation, and strict Concat-tail normalization in one
  fixed-point owner because every family mutates the same rewritten-tensor
  state. It iterates only the graph-ordered relevant type buckets and reuses
  stable indexed consumers; it performs no topology mutation. Empty seeds
  allocate no index. A module-level runner gives the branch and metadata phases
  the same index, reducing the complete primary sequence to one construction.
  Public outputs, shape/signature copy, broadcast fallback, channel guards,
  normalized axis, tail fan-out, Concat/quantized metadata, and fixed-point
  restart semantics remain unchanged. The shape/signature copier is also the
  explicit owner used by the later Dequantize-input repair, preserving the
  previously hidden cross-phase dependency without a lowerer-local closure.
- The two Swish-QDQ inverse post-Transpose sweeps share one semantic owner.
  The first remains before late Concat normalization; the second is called by
  the shared late-phase runner after normalization. Empty rewritten state and
  Transpose-free graphs return without an index. Otherwise indexed graph-order
  candidates, global alias replacement, and differential removal preserve
  ordered restart without per-removal full scans. Public aliases, wrong
  permutations, and inputs outside the rewritten set remain protected; alias
  chains and arbitrary consumer fan-out are rewired before removal.
- Late Swish-QDQ Concat normalization validates one complete transaction before
  mutation. Direct and Dequantize-wrapped pre-Transposes, rank-four normalized
  shapes, a private Concat output, and the strict Quantize/all-inverse-
  Transpose tail must all agree. Accepted Concat/DQ edge rewrites update one
  maintained index; axis and shape/signature metadata commit together; and
  only newly unused input adapters are removed. The owner restarts after each
  accepted transaction to avoid stale indices after compaction. Its runner
  gives the immediately following inverse-post owner that same index while
  retaining the original statistics and phase order.
- Concat pre-Q/DQ bypass remains intentionally narrower than generic redundant-
  quantization cleanup. The Quantize input must be the direct output of a
  Dequantize, source and destination quantized tensors must share the complete
  exact grid, and no arithmetic intermediate is accepted. The owner rewires
  only the Concat edge; existing Q/DQ operators remain available to later
  cleanup, preserving the historical ordered pipeline. It restarts after each
  indexed edge change and performs the former pruning side effect even when no
  Concat exists, without allocating an index in that no-candidate case.
- Terminal Transpose/Dequantize sanitation retains two explicit counters and
  subphases even though a first-subphase match becomes eligible for the second
  after its indexed reorder. This preserves the established stats and phase
  semantics. Operator order changes use `remove_operator()` followed by
  `insert_operator()` on the same index; public output rename occurs before
  differential Transpose removal so the Dequantize remains the sole producer.
  Graphs missing either required operator type still receive historical tensor
  pruning without paying for an index.
- The Transpose-DQ-Mean-Q bridge commits only after the mapped axes, reduced
  metadata, bridge permutation, and both unique tensor names are valid. The new
  preserving Transpose is inserted immediately before the current Quantize,
  then the old pre-Transpose is resolved by object identity and removed from
  the same index. This preserves valid ordering while making invalid-
  permutation rejection a complete no-op instead of retaining the former
  partial DQ/Mean metadata mutation.
- Pseudo-LeakyReLU fusion remains an exact ordered grammar rather than a
  commutative algebraic matcher: positive RELU is SUB input zero and the scaled
  negative branch is input one. Only MUL's singleton alpha may swap sides. The
  retained SUB is converted through indexed type/input updates and its options,
  axis semantics, version, and ONNX provenance are reset to the same defaults
  as a fresh `OperatorIR`; all four private producers are then removed in one
  batch compaction.
- The MUL-square fold's semantic owner is model-neutral; only the compatibility
  wrapper and historical stats key retain `yolo` naming. The self-square must
  use the identical tensor name in both MUL slots, while constant side inputs
  at the pre/anchor/scale MULs remain commutative. Fused data is calculated in
  float32, checked finite, cast back to anchor dtype, and receives cloned anchor
  quantization. All three removable intermediates are now explicitly rejected
  when public before any tensor or edge mutation.
- Leading-singleton Gather-to-Reshape cleanup accepts one signed integer zero
  in either a scalar or TFLite-legalized singleton buffer, because both select
  the only leading slice without changing element order or count. It requires
  axis zero after negative-axis normalization, batch dimensions zero, a
  statically fixed leading-one signature, exact rank-reduced tail shape and
  signature, matching dtype and quantization, one topologically later Reshape
  consumer at data input zero, and no public or duplicate-produced Gather
  output. All guards finish before the indexed Reshape edge is changed and the
  Gather is removed. Missing Gather/Reshape families still receive historical
  unused-tensor pruning without allocating an index, and the active
  `LayoutState` receives the same pruning at both production call sites.
- Terminal Softmax/Transpose cleanup consumes only the shared
  `_SOFTMAX_NHWC_PROPAGATED_MARKER` produced by the preceding canonicalizer;
  the string literal no longer has two owners. Public outputs remain the
  deterministic candidate order. The maintained index requires the output to
  have no internal consumer and one Transpose producer, the private Softmax
  intermediate to have one Softmax producer and exactly that Transpose
  consumer, and Softmax to precede Transpose. Duplicate producers and a
  Softmax intermediate exposed at either public boundary are rejected; a
  terminal output cannot also be an input. Rank-four source
  shape/signature, destination existence, and cloned quantization are planned
  before mutation. Commit removes only the marker, uses the lineage-aware
  indexed output setter, copies the former source metadata onto the existing
  public tensor object, and removes the Transpose differentially. Missing
  Softmax/Transpose families retain historical pruning without index
  construction, including optional `LayoutState` pruning at the production
  call site.
- Pre-ArgMax terminal layout cleanup accepts only an exact rank-four
  `[0,3,1,2]` Transpose whose private output has one topologically later
  `ARG_MAX` consumer at data input zero. The signed INT32/INT64 singleton axis
  must normalize to NCHW channel axis one and is remapped to NHWC axis three.
  Source, transposed, and output shape/signature metadata must prove the same
  permutation and rank-reduced output, and source/adapter dtypes must agree.
  A private axis constant is updated in place; a shared or public-input/output
  axis constant receives a uniquely named clone with its NumPy dtype and
  cloned quantization. This preserves the public constant value that the
  former rule could silently change. All clone data and topology guards finish
  before either constant or edge mutation. Indexed ArgMax input replacement
  and Transpose removal preserve the fixed point, historical stats, and
  lineage; post-prune `LayoutState` synchronization registers any clone.
  Missing required families retain historical pruning without index
  construction.
- Quantized MaxPool cleanup accepts only an exact linear
  `DEQUANTIZE -> MAX_POOL_2D -> QUANTIZE` chain. Input and output grids must
  use the same INT8 or UINT8 dtype and exactly equal positive finite scale and
  in-range zero point; approximate equality is not sufficient because a
  quantized MaxPool builtin preserves integer samples and therefore requires
  identical grids. All four tensors must exist, float bridge dtypes must
  agree, and rank-four shapes and signatures must match across each Q/DQ
  boundary. Both bridge tensors are private, uniquely produced, exclusively
  consumed, and topologically ordered. The output cannot also be a graph
  input. All topology, metadata, and cloned-quantization planning completes
  before indexed Pool edge mutation and differential wrapper removal. This
  intentionally fixes former rewrites that accepted near-equal grids, absent
  float metadata, or a float bridge exposed as a public input. Missing
  required operator families retain historical pruning without allocating an
  index, and both production call sites pass the Session `LayoutState`.
- Quantized Logistic cleanup accepts only an exact linear
  `DEQUANTIZE -> LOGISTIC -> QUANTIZE` chain. Input and output tensors must use
  the same INT8 or UINT8 dtype. The input grid requires a positive finite
  scale and in-range zero point; the output grid is exactly scale `1/256` with
  zero point `-128` for INT8 or `0` for UINT8. Tolerant scale equality is not
  sufficient for the builtin's canonical output contract. All four tensors
  must exist, float dtypes must agree, and elementwise shape/signature metadata
  must be identical across the complete chain without imposing a fixed rank.
  Both float bridges are private, uniquely produced, exclusively consumed,
  and topologically ordered; the quantized output cannot also be an input.
  Every guard completes before indexed Logistic edge mutation, version
  selection, and differential wrapper removal. This intentionally fixes
  former rewrites that accepted a near-canonical output scale, missing or
  invalid input quantization, absent float metadata, or a public-input bridge.
  Missing required families retain historical pruning without allocating an
  index, and both production call sites pass the Session `LayoutState`.
- Quantized Softmax cleanup accepts only an exact linear
  `DEQUANTIZE -> SOFTMAX -> QUANTIZE` chain. Its grid requirements match the
  canonical quantized Logistic contract: the input uses a positive finite
  per-tensor INT8/UINT8 grid, while output scale is exactly `1/256` with zero
  point `-128` or `0`. The existing beta tolerance of `1e-6` is preserved, but
  beta must be finite and parseable. Rank must be positive, and an explicit or
  default axis must normalize to the final dimension because the serialized
  TFLite builtin has no independent axis field. All four tensors must exist,
  float dtypes must agree, and shape/signature metadata must be identical
  across the elementwise chain. Both float bridges are private, uniquely
  produced, exclusively consumed, and topologically ordered; the quantized
  output cannot also be an input. Every guard completes before indexed
  Softmax edge mutation, version selection, and differential wrapper removal.
  This intentionally fixes former rewrites that accepted near-canonical
  output scales, missing/invalid input grids, absent float metadata, public-
  input bridges, malformed options, or non-last axes. Missing required
  families retain historical pruning without index allocation, and both
  production call sites pass the Session `LayoutState`.
- Expanded HardSigmoid QDQ cleanup matches the exact linear
  `DEQUANTIZE -> MUL(alpha) -> ADD(beta) -> MAXIMUM(low) -> MINIMUM(high) ->
  QUANTIZE` grammar with either scalar-input position at each binary operator.
  Input and output use exactly the same finite positive per-tensor INT8/UINT8
  grid with an in-range zero point. All seven data tensor records must exist;
  the five float tensors have one dtype and every data tensor has identical
  elementwise shape/signature metadata. Every bridge is private, uniquely
  produced, exclusively consumed, and topologically ordered; the quantized
  output cannot also be an input. Each scalar must be finite, singleton,
  producer-free, and representable within the preserved quarter-scale/`1e-3`
  tolerance.
- The four constant retargets are immutable plans containing the quantized
  value, cloned quantization, ownership choice, metadata, and reserved clone
  name. Private exclusive constants retain in-place behavior. Shared or public
  constants receive deterministic `_q` clones, so a public float scalar no
  longer silently changes dtype/value. Only after all four plans and four
  intermediate quantization clones succeed are constant and data edges
  rewritten and DQ/Q wrappers removed. Fault injection proves that a clone
  failure on the second scalar leaves complete ModelIR unchanged, whereas the
  former helper changed the first scalar data/dtype before raising. Missing
  required families retain historical pruning without index allocation; both
  production call sites pass LayoutState, which is synchronized after clones
  and pruning.
- Quantized TransposeConv cleanup accepts only the exact linear
  `DEQUANTIZE -> TRANSPOSE_CONV -> QUANTIZE` grammar with TFLite input roles
  `[output_shape, filter, data]`. Input and output activations independently
  require finite positive per-tensor INT8 grids and in-range zero points.
  Their quantized/float shape and signature metadata must match at each
  boundary, the bridges must share a floating dtype, and both bridges are
  private, uniquely produced, exclusively consumed, and topologically ordered.
  The quantized output cannot also be a graph input.
- Filter conversion is a pre-mutation plan. A producer-free rank-four INT8
  filter with matching buffer metadata and a valid grid remains unchanged. A
  finite FLOAT16/FLOAT32/FLOAT64 filter is quantized in place only when it is
  private and exclusive; shared or public filters receive a deterministic
  `_q` clone. The filter data/grid/name and output quantization clone all
  complete before indexed input/output mutation. Fault injection proves that
  output-grid clone failure leaves the complete ModelIR unchanged, whereas the
  former helper had already converted a private float filter before raising.
  Missing families retain historical pruning without index allocation; all
  three production call sites pass the Session LayoutState.
- Decomposed InstanceNormalization repair requires the exact marked chain from
  the first Mean through bias Add, including correct Sub and reciprocal-Div
  operand roles, keep-dim reductions, graph order, unique producers, exclusive
  internal consumers, a finite epsilon, and a producer-free scalar one. The
  input logical layout chooses channel axis one or the final axis for ranks
  three through five; an optional post-Transpose must be a complete
  permutation before the bias broadcast axis is derived.
- Both Mean axes, nine intermediate shape/signature records, and scale/bias
  data plus metadata are planned before mutation. Changing a constant requires
  integer axes or a channel-count-sized buffer, no producer/public boundary,
  and exactly the expected Mean/Mul/Add consumers. This preserves the shared
  two-Mean axes tensor while rejecting external sharing. A malformed final
  bias shape is now a complete no-op instead of raising after earlier axes,
  shapes, and scale data were already changed. The final production call passes
  the Session LayoutState; graphs without the marked first Mean allocate no
  index.
- NCHW Concat/global-pool/Conv repair requires the exact ordered
  `CONCATENATION -> MEAN -> RESHAPE -> CONV_2D` chain. Each internal tensor is
  uniquely produced, exclusively consumed by the next operator, and private.
  The keep-dim Mean reduces rank-four axes two and three; negative axes are
  normalized. Fully positive NCHW Concat inputs share batch/spatial dimensions,
  and their axis-one channel sum must equal the producer-free OHWI Conv filter
  input channel.
- Concat and Reshape options, Concat/Mean/Reshape metadata, and the four-value
  integer Reshape buffer are a complete pre-mutation plan. The shape constant
  is producer-free, private, and exclusively consumed by that Reshape. This
  prevents a late buffer-read exception from leaving the Concat axis and three
  tensor records changed, and rejects the former non-global, fan-out/public,
  duplicate-producer, runtime-filter, and malformed/shared shape-buffer cases.
  The sole production call passes the Session LayoutState; incomplete operator
  families allocate no index.
- NCHW Concat/Transpose/(Transpose)Conv repair traces optional shape-preserving
  RELU/RELU6/QUANTIZE/DEQUANTIZE/CAST before the exact `[0,2,3,1]` Transpose
  and optional PAD/CAST/SUB afterward. Every internal output is uniquely
  produced, exclusively consumed by the next topologically ordered operator,
  and private. The permutation and OHWI filter are producer-free constants,
  and the filter buffer exactly matches its rank-four metadata.
- Fully positive NCHW Concat inputs share batch/spatial dimensions, and their
  axis-one channel sum equals the filter input channel. Concat options and all
  Concat/pre-passthrough/Transpose shape records are planned together. Direct
  Conv without a post-prefix also plans its output shape; prefixed Conv and
  TransposeConv intentionally preserve output metadata. This rejects the
  former public/fan-out/duplicate adapter and runtime-filter cases without
  broadening the four existing positive families. The production call passes
  the Session LayoutState; missing Concat or Conv families allocate no index.
- Mixed singleton Concat repair accepts an exact axis-three NHWC Concat only
  when its output channel equals the input count and every same-dtype input is
  either `[N,H,W,1]` or its singleton-channel NCHW projection. Input/output
  shape signatures must express the same contract. One dynamic dimension is
  retained as the sole Reshape `-1`; multiple dynamic dimensions cannot be
  represented by this local adapter and therefore remain untouched.
- A runtime input must be public or uniquely produced before the Concat; a
  producer-free constant is also valid. Duplicate, later, and unresolved
  producers are rejected. Names are reserved across all existing tensors,
  operator edges, and boundaries. Repeated source inputs share one adapter.
  Every shape tensor, adapter tensor, operator, and quantization clone is
  prepared before indexed insertion and lineage-aware rewiring. This prevents
  a late clone exception from leaving the first adapter behind. The production
  call passes the Session LayoutState; graphs without Concat allocate no index.
- Window-partition canonicalization requires exact NHWC input, six-dimensional
  partition Reshape, `[0,1,3,2,4,5]` Transpose, and three-dimensional window
  output Reshape equations. All internal edges are private, uniquely produced,
  exclusively consumed, and ordered. The input is public, constant, or
  uniquely produced earlier. Shape/permutation vectors are producer-free,
  non-input INT32/INT64 constants with exact vector metadata.
- All four data tensors share dtype and either no quantization or one exact
  per-tensor grid. The existing Transpose object becomes SPACE_TO_DEPTH so its
  version, axis semantics, and provenance survive. Static metadata remains
  exact. A consistent dynamic batch/spatial/channel signature is propagated;
  when the retained Reshape needs one `-1`, its private vector and both shape
  options are changed together and marked dynamic. Two inferred dimensions,
  shared/public shape mutation, or any incomplete contract is a complete
  no-op. Both production calls pass the Session LayoutState; incomplete
  operator families prune historically unused tensors without allocating an
  index.
- Recurrent orphan-step alias repair has one Torch-free semantic owner in
  `passes/recurrent_alias.py`. Candidate discovery occurs before index
  construction, so graphs without the exact step-name grammar allocate no
  index. A supplied matching index is reused; otherwise exactly one index is
  built. Producer rejection, shape-tensor consumer order, Reshape arity,
  public input/output, consumer rewrites, and non-public orphan tensor removal
  retain the direct implementation's behavior. The direct and PyTorch modules
  are compatibility wrappers only and do not carry parallel match/rewrite
  rules.
- Unbound nonconstant-input discovery and layout repair have one Torch/
  TensorFlow-free owner in `passes/unbound_input_layout.py`. Standalone issue
  reporting retains its lightweight producer-name scan. Repair first snapshots
  issue consumers by object identity; it constructs one `ModelIRGraphIndex`
  only when an issue exists, resolves current positions after each insertion,
  and skips later issues once an earlier bridge produces the same tensor.
  DEQUANTIZE exact/fallback source policy, nearest source ordering, SPLIT data
  slot, MUL all-consumers guard, dtype/shape checks, quantization metadata,
  unique perm naming, and insertion-before-consumer order remain unchanged.
  The lowerer wrapper reconciles shapes with the maintained index and preserves
  the existing stats key for both primary and fallback callers.
- Quantized RELU/RELU6 layout-bridge cleanup has one Torch/TensorFlow-free
  owner in `passes/quantized_activation.py`. It skips index allocation when no
  Transpose exists, otherwise uses one current index for exact chain traversal,
  DQ input and Q output rewrites, and batch removal of the inverse Transposes.
  The restart loop remains intentional: removing a later bridge may make an
  earlier graph-order candidate linear, while current indexed candidates avoid
  every compatibility-map rebuild. Public intermediate/source guards, exact
  inverse permutations, per-tensor-only quantization, source-shape propagation,
  destination dtype/quantization cloning, pruning, and stats are unchanged.
- Expanded HardSigmoid QDQ layout-bridge cleanup now shares that owner and
  recognizes both RELU_0_TO_1 and MAXIMUM/MINIMUM clamp forms. One current
  index supplies all linear-consumer and shared-constant decisions, indexed
  setters maintain cloned-constant and DQ/Q edges, and both Transposes are
  removed by one differential compaction. All required constants are
  validated before any mutation, making rejection transactional. Private
  rank-matched constants retain in-place remapping; shared constants retain
  clone-and-rewire behavior. Public guards now include the clamp form's
  MAXIMUM intermediate as well as every former boundary, while inverse-perm,
  per-tensor quantization, shape/signature, destination metadata, lineage,
  pruning, and stats contracts remain intact.
- Expanded MUL/ADD/PRELU QDQ bridge cleanup now has the same single owner.
  MUL/ADD input ordering and PRELU data-slot semantics are unchanged; one
  index supplies all topology and shared-constant decisions, both DQ/Q edge
  updates, and removal of the two wrapper Transposes. The common constant
  planner is mutation-free and the common applier owns private updates and
  shared clone rewires, but PReLU still requires all three buffers to be NumPy
  arrays while HardSigmoid accepts any non-`None` buffer. This prevents a
  convenience helper from broadening rule eligibility.
- Quantized logistic-gated MUL recovery has a separate
  `passes/quantized_gate.py` owner because its shared input, dual DQ/Q
  branches, and multi-post aliases are not a linear activation contract. One
  index supplies producer and consumer topology, both DQ rewires, canonical
  output selection, indexed alias replacement, and one compaction for the
  pre-Transpose plus every post-Transpose. The first graph-order post remains
  canonical and receives the permuted MUL-Q metadata. All internal data/gate
  tensors are now protected when publicly observable; fixed permutations,
  per-tensor quantization, pruning, lineage, and stats remain unchanged. A
  bounded `_match_logistic_gate_branch` helper isolates backward gate matching
  while preserving incomplete-chain fallback and duplicate-branch rejection.
- General transpose/binary bridge folding now has one TensorFlow-free owner in
  `passes/binary_bridge_layout.py`. Symmetric candidates are processed before
  asymmetric candidates to preserve the former phase priority. A supplied
  index is reused, candidates are fixed in graph order, and at most 32
  accepted rewrites are applied. Typed immutable permutations, unique source
  provenance, exact private pre-adapter use, producer-before-consumer order,
  per-tensor quantization, dtype, static broadcast, dynamic signature, public
  boundaries, output fan-out, and every insertion/removal index are resolved
  before mutation and re-resolved immediately before apply. The mixed-fan-out
  form reuses an existing pre-permutation constant instead of mutating the
  possibly shared post-permutation buffer. The legacy-only form creates and
  inserts its adapter only after the complete plan passes. The asymmetric form
  retains operand order for SUB and DIV and rejects a plain source produced
  after the Transpose it would reuse. The disabled Pattern C implementation is
  intentionally absent rather than carried as unreachable production code.
- The later safe recovery sequence is not five independent owners. One
  `run_safe_binary_bridge_recovery` call retains the exact legacy-only,
  single-post, mixed fan-out, asymmetric fan-out, and full-post order on one
  differential index. The 32-rewrite bound applies per phase. The first two
  phases reuse the strict symmetric plan and add the historical preserved-
  boundary marker only at this late recovery point. Mixed and full-post modes
  share one multi-post plan and canonical-first-post alias contract.
  Asymmetric fan-out requires an already-earlier inverse Transpose of the plain
  operand and preserves SUB/DIV order. Retained adapters reference the typed
  pre-permutation tensor rather than changing a shared inverse constant. All
  modes re-resolve their complete tensor/operator contracts before mutation,
  and pruning/LayoutState synchronization run once after the sequence.
- Elementwise/Concat layout recovery is grouped by connected Concat closure,
  not rewritten one node at a time. This prevents a shared elementwise
  producer from being converted under one Concat while another dependent
  Concat remains in the old layout.
- Existing inverse adapters are accepted as NHWC boundary aliases only after
  their typed permutation, shape/signature, dtype, quantization, graph order,
  and non-public output are proven. Explicit pre-adapters are removed only
  when every remaining consumer is handled; otherwise the adapter is retained
  for its untouched fan-out.
- Rank-four binary constants are transposed only when the entire consumer-slot
  set belongs to the accepted closure. A shared or mutable constant rejects
  the candidate, preventing a layout rewrite from changing unrelated binary
  behavior. SUB and DIV keep their original operand positions.
- Closure outputs used by supported legacy unary/binary consumers receive one
  deterministic local NHWC-to-NCHW adapter. Unsupported fan-out or a public
  closure intermediate rejects the complete group without mutation.
- The explicit NHWC layout annotation is an intentional internal contract
  improvement. Operator/tensor topology, metadata other than layout,
  constants, and emitted artifacts remain identical to the preceding
  checkpoint, while TensorIR and Session LayoutState no longer disagree about
  the converted physical view.
- StridedSlice fan-in is one whole-group transaction rooted at the shared
  pre-Transpose. The pass does not rewrite an individual Slice unless every
  pre-adapter consumer is a supported Slice, every Slice output joins the same
  Concat exactly once, and every Concat boundary has been classified.
- Begin, end, and stride vectors must be distinct immutable rank-four INT32 or
  INT64 constants whose TensorIR dtype, NumPy dtype, shape/signature, producer,
  public ownership, and exact consumer slots agree. Zero strides or conflicting
  parameter roles reject the whole candidate before any constant changes.
- Multiple private post adapters collapse to the first output while preserving
  the legacy `replace_operator_input_at`, output-rename, alias-replacement, and
  pruning lineage-event order. This keeps active-path correspondence behavior
  unchanged rather than treating report stability as incidental.
- A required NCHW compatibility boundary reuses the typed pre-permutation. The
  raw helper changed the first post-permutation buffer to an INT32 opposite
  permutation even when that tensor was INT64 or shared. The indexed owner
  leaves every post-permutation and unrelated consumer untouched.
- A retained inverse adapter must already precede every legacy consumer.
  Stale order is rejected transactionally instead of emitting a consumer before
  its new producer. Public Concat outputs and later public post paths remain
  supported through the same local boundary.
- Shared parsers preserve the exact old generated syntax when broadening would
  change rule eligibility. Parser ownership tests prevent duplicate exporter
  implementations and unused compatibility imports.
- No real-model conversion gate is required for these mechanical checkpoints
  under the current instruction to prioritize implementation and minimize
  conversion tests. This does not prove broad corpus regression safety.

## Tests executed

The resumed downstream-binary, Resize-evidence, aligned-BatchNorm, LRN,
static-shape-cache, and NHWC-bridge checkpoints passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pytorch_fast_precanonicalize_policy.py \
  tests/test_flatbuffer_direct_architecture.py

124 passed
```

Seven pre/post-extraction characterization cases also preserve the exact
orchestrator output for matching BN, direct return, channel-first Resize,
channel mismatch, channel-last Resize, mixed operands, and an already-CF shape.
Five additional pre/post-extraction cases preserve direct BN, reshaped BN,
no-BN fallback, NHWC-input no-op, and already-CF Resize behavior.
Seven aligned-BatchNorm cases preserve direct rewrite, non-BN no-op, channel
mismatch no-op, NHWC-input no-op, reshaped rewrite, already-CF behavior, and
reshaped non-BN no-op.
The LRN checkpoint additionally passed a four-test selection covering CF/NHWC
and static-shape state, Pool/LRN interaction, architecture ownership, and the
existing generated-source integration case.
The cache checkpoint passed four focused cases covering aligned binary,
Resize/Pool, literal recording, parse-failure no-op, and architecture ownership.
The bridge checkpoint adds positive state-set/cache assertions and a no-op
state-preservation case to the existing whole-chain normalization test.
The request-boundary checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_core.py \
  tests/test_flatbuffer_direct_architecture.py

126 passed
```

Its AST gate proves zero `kwargs.get` calls and exactly one raw `kwargs` read,
as the argument to `ConversionRequest.from_kwargs`.
The requested-exporter checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_artifact_preparation.py \
  tests/test_flatbuffer_direct_core.py \
  tests/test_flatbuffer_direct_architecture.py

145 passed
```

Dedicated resolver tests use a mapping that raises on every `get` to prove
unrequested settings are untouched, then verify requested and calibration-only
values and timeout coercion.
The same 145-test selection passed after adding requested-only quant type/dtype
resolution. Focused assertions cover explicit values, the three legacy
defaults, immutable mapping behavior, and absence of direct `request.get`
calls for those keys.
The Session consumer-count checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_core.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_quantize_transpose_optimization \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_quantize_transpose_fanout_optimization \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_quantize_transpose_preserves_dynamic_batch_signature

32 passed
```

The core spy fixture uses one input in both Add and Identity and verifies the
context receives counts `{"x": 2, "y": 1}` from the Session index.
The lowering-time layout checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_core.py \
  tests/test_pad_edge_lowering.py \
  tests/test_flatbuffer_direct_resize_integer_linear.py \
  tests/test_flatbuffer_direct_architecture.py::test_op_builders_mutate_layout_only_through_lowering_context

33 passed
```

The rank-three Resize regression hook inspects `LayoutState` before the first
post-lowering pass and verifies that the NWC/NHWC adapter tensors have no
ModelIR mismatch. The focused edge-Pad and integer-linear Resize tests also
serialize and execute their TFLite artifacts sequentially. The architecture
gate rejects future direct logical/physical layout assignments anywhere below
`op_builders`.
The synthetic-consumer checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_core.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_elides_inverse_transpose_chain_at_generation \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_no_dead_operator_outputs_after_prune \
  tests/test_flatbuffer_direct_architecture.py::test_op_builders_mutate_operator_list_only_through_lowering_context

35 passed
```

The two direct context cases prove both sides of the contract: exclusive
inverse Transposes remove their operator and differential indexes, while a
synthetic bridge already consumed by an Identity retains its producer. The
architecture gate rejects direct operator-list writes and mutating method calls
from op builders.
The adjacent pass-state reuse checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_mean_layout.py \
  tests/test_flatbuffer_direct_layernorm_layout.py \
  tests/test_flatbuffer_direct_terminal_mean_layout.py \
  tests/test_flatbuffer_direct_se_layout.py \
  tests/test_flatbuffer_direct_core.py \
  tests/test_flatbuffer_direct_pass_efficiency.py \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_mean_attention_cluster_reuses_one_pass_state_scope

65 passed
```

The focused reuse case observes one `ModelIRGraphIndex.refresh()` across two
candidate Mean runners and a diagnostic build sequence of `[true, false]`.
The all-preflight-miss case observes zero index refreshes and `[false, false]`.
The architecture gate fixes all seven runner calls, their order, their shared
scope keyword, and the six production cluster invocations.
The adjacent gate-pass reuse checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_elementwise_gate_layout.py \
  tests/test_flatbuffer_direct_pad_layout.py \
  tests/test_flatbuffer_direct_dual_postconv_gate_layout.py \
  tests/test_flatbuffer_direct_3d_gate_layout.py \
  tests/test_flatbuffer_direct_conv3d_gate_layout.py \
  tests/test_flatbuffer_direct_cost_volume_scatter_layout.py \
  tests/test_flatbuffer_direct_add_concat_suffix_layout.py \
  tests/test_flatbuffer_direct_dual_mul_concat_layout.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_mixed_mean_reducemax_concat_mirrorpad_nhwc_chain \
  tests/test_flatbuffer_direct_pass_efficiency.py \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_gate_cluster_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

116 passed
```

The synthetic shared-scope fixture makes every runner's model-only preflight
match but contains no deep rewrite candidate. It records one
`ModelIRGraphIndex.refresh()` across all eight calls and 15 diagnostic events:
the first reports `state_built: true`, and every later event reports `false`.
The architecture checks fix the eight-runner order, five production helper
invocations, and the single invocation that omits mixed attention. They also
bring the global direct-runner count characterization up to date with both
bounded helper extractions.
A focused late-pair checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_ndhwc_cost_volume_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_ndhwc_cost_volume_pair_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_3d_gate_layout.py \
  tests/test_flatbuffer_direct_conv3d_gate_layout.py \
  tests/test_flatbuffer_direct_cost_volume_scatter_layout.py

50 passed
```

The runtime characterization observes one graph-index refresh and diagnostic
build flags `[true, true, false]`: both events in the first two-spec runner
belong to the one state-building group, and the second runner reuses that
state. The architecture test fixes both raw boundaries and proves that mixed
attention receives no shared scope.
A focused late-Concat cluster checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_concat_layout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_concat_layout_cluster_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_axis3_const_concat_layout.py \
  tests/test_flatbuffer_direct_dequant_concat_quantize_layout.py \
  tests/test_flatbuffer_direct_layernorm_layout.py \
  tests/test_flatbuffer_direct_layout_transpose.py

74 passed
```

The synthetic runtime fixture records one graph-index refresh and build flags
`[true, false, false, false, false]`. The architecture test fixes the four-
runner order and both raw boundaries. The core layout-handoff monkeypatch now
accepts and forwards the runner's optional scope without changing its original
pre-pass layout assertions.
A focused shuffle/unary cluster checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_shuffle_gather_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_unary_fanout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_shuffle_and_unary_clusters_reuse_pass_state_scopes \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics \
  tests/test_flatbuffer_direct_nhwc_channel_shuffle.py \
  tests/test_flatbuffer_direct_nchw_channel_shuffle.py \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_transpose_unary.py \
  tests/test_flatbuffer_direct_transpose_unary_fanout.py \
  tests/test_flatbuffer_direct_transpose_unary_binary_fanout.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_shufflenet_transpose_shuffle_chain_optimized

29 passed
```

Both synthetic fixtures observe exactly one graph-index refresh. The seven-
runner fixture records one `state_built: true` followed by six `false` events;
the three-runner fixture records one `true` followed by two `false` events.
The architecture gate fixes the two helper orders, five and four invocations,
and the single extended channel-shuffle invocation.
A focused boundary-input pair checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_boundary_batchmatmul_unary_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_boundary_batchmatmul_unary_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics \
  tests/test_flatbuffer_direct_boundary_input_chains.py \
  tests/test_flatbuffer_direct_input_passthrough_layout.py

17 passed
```

The runtime fixture records one graph-index refresh and build flags
`[true, false, false, false]`; the latter three events belong to the reused
three-spec input-unary group. The architecture gate fixes the two-runner order
and all four helper invocations.
A focused channel-slice/Pad-Mul checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_channel_slice_pad_mul_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_channel_slice_pad_mul_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics \
  tests/test_flatbuffer_direct_pad_layout.py \
  tests/test_tflite_builder_direct.py \
  -k 'channel_slice or transpose_pad_mul_posttranspose or channel_slice_pad_mul_pair or ordered_model_ir_runner or lowerer_channel_slice'

8 passed, 754 deselected
```

The runtime fixture records one graph-index refresh and build flags
`[true, true, true, false]`: the first three events belong to the one state-
building channel-slice group, and Pad-Mul reuses that state. The architecture
gate fixes the two-group order and all three helper invocations.
A focused singleton/Reshape checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_singleton_channel_transpose.py \
  tests/test_flatbuffer_direct_singleton_reshape.py \
  tests/test_flatbuffer_direct_singleton_maxpool.py \
  tests/test_flatbuffer_direct_flatten_concat_reshape.py \
  tests/test_flatbuffer_direct_consecutive_reshape.py \
  tests/test_flatbuffer_direct_singleton_spatial_reshape.py \
  tests/test_flatbuffer_direct_graph_cleanup.py \
  tests/test_flatbuffer_direct_osnet_gate_layout.py \
  <two new efficiency tests and two architecture checks>

41 passed
```

The long synthetic fixture makes all ten runner preflights match and records
13 diagnostic events with one graph-index refresh and build flags
`[true, false, ...]`. The short fixture records one refresh and flags
`[true, false, false]`. Architecture checks fix both long variants, all three
short-helper target/layout combinations, the shared scope keyword, and the
147-call global runner characterization.
A focused QKV attention checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  <seven focused QKV rewrite tests from test_tflite_builder_direct.py> \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_qkv_attention_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_qkv_attention_pair_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

10 passed
```

The preflight-only fixture records one graph-index refresh and build flags
`[true, true, true, true, false, false]`: the four prefix events belong to the
state-building group, and both bridge events reuse it. The seven existing
functional cases cover Gather/Reshape/Transpose hoisting, Gather-to-Slice,
Slice-to-Split, Split/Reshape collapse, shared pre-Transpose, weighted-sum
bridging, and the KV pipeline.
A focused duplicate/PReLU checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_graph_cleanup.py \
  tests/test_flatbuffer_direct_quantized_prelu.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_duplicate_quantized_prelu_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_duplicate_quantized_prelu_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

28 passed
```

The QDQ preflight-only fixture disables duplicate-Transpose cleanup exactly as
production does, then records one graph-index refresh and build flags
`[true, false, false, false, false]` across the reshape-duplicate group and
four PReLU specs. Architecture checks fix the helper's flag forwarding, exact
runner order, two invocations, and the 143-call global characterization.
A focused constant-fold/Cast checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_constant_fold.py \
  tests/test_flatbuffer_direct_cast_cleanup.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_constant_fold_cast_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_constant_fold_cast_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

16 passed
```

The preflight-only widening-Cast fixture records one graph-index refresh and
build flags `[true, true, true, false, false]`: the three constant-fold events
belong to the state-building group, and both Cast-cleanup events reuse it.
Architecture checks fix runner order, both helper invocations, and the 141-call
global characterization.
A focused SE-FC/Gather checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_se_layout.py \
  tests/test_flatbuffer_direct_transpose_gather_channel_fanout.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_se_fc_gather_channel_fanout_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_se_fc_gather_fanout_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

14 passed
```

The preflight-only fixture records one graph-index refresh and build flags
`[true, false]`. Architecture checks fix both target/LayoutState combinations,
runner order, and shared scope keywords. The global characterization was 139
calls at that checkpoint, 137 after the later post-QDQ consolidation, and 135
after the subsequent NCHW channel-shuffle consolidation, then 134 after the
conditional QKV-bridge consolidation.
A focused terminal-boundary checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_boundary_input_layout.py \
  tests/test_flatbuffer_direct_dual_mul_concat_layout.py \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_pad_layout.py \
  tests/test_flatbuffer_direct_transpose_gather_channel_fanout.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_terminal_boundary_layout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_terminal_boundary_layout_cluster_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

38 passed
```

The preflight-only fixture records one graph-index refresh across seven events;
only the first reports `state_built: true`. Architecture checks fix all five
runner calls, their shared scope, the preceding raw InstanceNorm rewrite, and
the following conditional stage.
A focused late-Dequantize/unary checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_dequant_concat_quantize_layout.py \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_dequant_unary_fanout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_dequant_unary_fanout_cluster_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

41 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the first reports `state_built: true`. Architecture checks fix all three
runner calls, their shared scope, the preceding raw QDQ bridge, and the
following independently indexed Swish dispatcher.
A focused terminal singleton-MaxPool/Reshape checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_singleton_maxpool.py \
  tests/test_flatbuffer_direct_consecutive_reshape.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_terminal_singleton_maxpool_reshape_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_terminal_singleton_maxpool_reshape_pair_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

8 passed
```

The preflight-only fixture records one graph-index refresh across three events;
the two singleton-MaxPool events report `state_built: true` because they share
the first registered group, and the Reshape event reports `false`.
Architecture checks fix both runner calls, their shared scope, and the exact
conditional legacy-rewrite boundaries.
A focused terminal clamp/unary/ReLU checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_graph_cleanup.py \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_terminal_clamp_unary_relu_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_terminal_clamp_unary_relu_cluster_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

27 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the first reports `state_built: true`. The two real rewrite tests also
assert that the reused graph index removes `MAXIMUM`/`MINIMUM` type entries and
adds the correct `RELU_0_TO_1` or `RELU` entry without refreshing.
A focused late Mean/SPP/Gather checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_mean_layout.py \
  tests/test_flatbuffer_direct_spp_layout.py \
  tests/test_flatbuffer_direct_transpose_gather_axis.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_mean_spp_gather_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_mean_spp_gather_cluster_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

51 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the first reports `state_built: true`. Architecture checks fix the runner
order, shared scope keywords, conditional generic-transpose predecessor, and
constant-fold/Cast helper successor.
A focused late SPP/Concat-unary-Conv checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_spp_layout.py \
  tests/test_flatbuffer_direct_concat_unary_conv_layout.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_spp_concat_unary_conv_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_spp_concat_unary_conv_pair_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

73 passed
```

The preflight-only fixture records one graph-index refresh across two events;
the first reports `state_built: true` and the second `false`. Architecture
checks fix both runner calls, shared scope keywords, and both raw rewrite
boundaries.
A focused absolute-final normalization-Pad/attention checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_instancenorm_mirror_pad_prepost_nhwc_chain_optimized \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_flatten_globalnorm_pad_prepost_nhwc_chain_optimized \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_mixed_mean_reducemax_concat_mirrorpad_nhwc_chain \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_absolute_final_normalization_attention_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_absolute_final_normalization_attention_pair_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

6 passed
```

The preflight-only fixture records one graph-index refresh across two events;
the first reports `state_built: true` and the second `false`. The focused
selection also executes real InstanceNorm, flattened normalization-Pad, and
mixed-attention rewrites. Architecture checks preserve both include flags and
the exact raw boundaries.
A focused post-QDQ unary-fan-out checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_unary_fanout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_post_qdq_layout_unary_fanout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_shuffle_and_unary_clusters_reuse_pass_state_scopes \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_post_qdq_unary_fanout_cluster_stays_between_raw_rewrites \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

9 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the first reports `state_built: true`. Architecture checks preserve four
default helper invocations, require exactly one alternate-mode invocation,
fix both raw boundaries, and characterize 137 registered runner calls.
A focused late NCHW channel-shuffle/Gather checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_nhwc_channel_shuffle.py \
  tests/test_flatbuffer_direct_nchw_channel_shuffle.py \
  tests/test_flatbuffer_direct_transpose_gather_axis.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_shuffle_gather_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_nchw_shuffle_gather_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_shuffle_and_unary_clusters_reuse_pass_state_scopes \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_nchw_shuffle_gather_pair_stays_between_raw_rewrites \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

15 passed
```

The preflight-only fixture records one graph-index refresh across two events;
the first reports `state_built: true` and the second `false`. Real NHWC and
NCHW rewrite tests assert that `RESHAPE` type entries disappear and the
surviving `GATHER` entry has its correct shifted index without refreshing.
Architecture checks preserve five default helper invocations, require one
NCHW-only invocation, fix both raw boundaries, and characterize 135 runner
calls.
A focused conditional layout/QKV-bridge checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_attention_qkv_shared_pretranspose_slice_nchw \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_attention_qkv_weighted_sum_bridge_to_nhwc \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_qkv_attention_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_layout_qkv_bridge_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_qkv_attention_pair_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_layout_qkv_bridge_pair_stays_between_raw_rewrites \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

11 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the generic-transpose event reports `state_built: true`. Architecture
checks preserve two default helper invocations, require one bridge-only
invocation with the runtime layout flag, fix both raw boundaries, and
characterize 134 runner calls.
A focused very-late Gather/constant/normalization checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_transpose_gather_axis.py \
  tests/test_flatbuffer_direct_constant_fold.py \
  tests/test_flatbuffer_direct_cast_cleanup.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_flatten_globalnorm_pad_prepost_nhwc_chain_optimized \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_constant_fold_cast_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_very_late_gather_constant_normalization_cluster_reuses_one_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_constant_fold_cast_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_very_late_gather_constant_normalization_cluster_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

21 passed
```

The preflight-only fixture records one graph-index refresh across seven events;
only the Gather-axis event reports `state_built: true`. Architecture checks
require both constant-fold/Cast invocations to receive an external scope,
preserve the normalization flags, and fix both raw boundaries.
A focused terminal hard-activation/layout scope checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_hard_activation_layout_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_hard_activation_layout_pair_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

3 passed
```

The preflight-only fixture records one graph-index refresh across three events;
both HardSigmoid events report `state_built: true` for their shared group and
the following generic-Transpose event reports `false`. Architecture checks
preserve all four late hard-activation flags, the conditional layout switch,
and both raw rewrite boundaries. The related real hard-activation and generic-
Transpose selection passed separately with `7 passed`.
A focused late layout/Mean/SPP/Gather/constant/Cast scope checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_layout_mean_spp_gather_constant_cast_cluster_reuses_one_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_layout_mean_spp_gather_constant_cast_cluster_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

3 passed
```

The preflight-only fixture records one graph-index refresh across nine events;
only the conditional generic-Transpose group reports `state_built: true`.
Architecture checks preserve the full runner/helper order, the runtime layout
flag, both external constant-fold/Cast scopes, and the raw boundaries. The six
related real pass modules passed separately with `65 passed`.
A broader single-process selection of
`test_flatbuffer_direct_core.py`, `test_flatbuffer_direct_pass_efficiency.py`,
and the complete `test_flatbuffer_direct_architecture.py` passed with
`186 passed` after adding the late combined-scope checks.

The typed artifact-plan checkpoint passed its policy selection:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q tests/test_flatbuffer_direct_artifact_preparation.py

20 passed
```

Default, dynamic-range/integer-quantized, split, SavedModel, PyTorch-derived,
and report progress labels are characterized from `ArtifactPlan`. Rejecting
option mappings prove that an unrequested artifact reads no related option or
environment value. Sequential dynamic-range, strict-integer, and split-
manifest direct-backend smokes passed with `3 passed`. The combined artifact-
policy, core, pass-efficiency, and architecture selection passed with
`206 passed`.

The evaluation-artifact selection checkpoint passed its focused unit and
ownership selection with `4 passed`. Sequential dynamic-range, strict-integer,
and split-manifest direct-backend smokes again passed with `3 passed`. The
combined artifact-metadata, artifact-policy, core, pass-efficiency, and
architecture selection passed with `210 passed`.

The report/quantized-artifact finalization checkpoint passed its focused
single-owner structure test with `1 passed`. The sequential integration
quantization/evaluation/coverage, strict integer/int16, and split-manifest
direct-backend smokes passed with `3 passed`. The combined artifact-metadata,
artifact-policy, core, pass-efficiency, and architecture selection passed with
`211 passed`.

The terminal direct-boundary checkpoint passed the three focused evaluation-
selection, report/quantized-finalization, and fast-path control-flow contracts
with `3 passed`. TensorFlow-import-blocked direct and `-cotof`, followed by the
sequential integration quantization/evaluation/coverage, strict integer/int16,
and split-manifest smokes, passed with `5 passed`. The combined artifact-
metadata, artifact-policy, core, pass-efficiency, and architecture selection
passed with `212 passed`.

The ordered layout-recovery-prefix checkpoint passed its focused ordering,
runner-ownership, SPP, NDHWC Concat, and NHWC/NCHW channel-shuffle selection
with `68 passed`. A sequential quantization/evaluation/coverage integration
smoke passed with `1 passed`. The complete architecture selection passed with
`128 passed`; the combined artifact-metadata, artifact-policy, core, pass-
efficiency, and architecture selection passed with `213 passed`.

The attention/quantized-recovery-suffix checkpoint passed focused ordering,
scope-boundary, quantized PReLU, quantized Reshape, and trailing-output-
Transpose tests with `19 passed`. Its single sequential quantization,
evaluation, and coverage integration smoke passed with `1 passed`. The
combined artifact-metadata, artifact-policy, core, pass-efficiency, and
architecture selection passed with `214 passed`.

The layout/reshape/attention-recovery-prefix checkpoint passed its focused
owner, exact-order, successor-boundary, and runner-diagnostics selection with
`4 passed`. The complete architecture file passed with `130 passed`; artifact-
metadata, artifact-policy, core, and pass-efficiency passed separately with
`85 passed`, for the same combined selection total of `215 passed`. Its single
sequential quantization, evaluation, and coverage integration smoke passed
with `1 passed`.

The terminal slice/Concat layout-recovery checkpoint passed focused QKV,
channel-slice, exact-order, variant-boundary, runner-diagnostics, and layout-
Transpose ownership checks with `6 passed`. The complete architecture file
passed with `131 passed`; artifact-metadata, artifact-policy, core, and pass-
efficiency passed separately with `85 passed`, for a combined selection total
of `216 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The terminal affine/Concat/split recovery checkpoint passed focused exact-
order, raw-boundary, terminal slice/Concat, QKV-boundary, and runner-
diagnostics checks with `4 passed`. The complete architecture file passed with
`132 passed`; artifact-metadata, artifact-policy, core, and pass-efficiency
passed separately with `85 passed`, for a combined selection total of `217 passed`.
Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The attention/gate/QDQ recovery checkpoint passed focused exact-order, three-
boundary, outer-suffix, gate/unary-scope, and runner-diagnostics checks with
`5 passed`. The complete architecture file passed with `133 passed`; artifact-
metadata, artifact-policy, core, and pass-efficiency passed separately with
`85 passed`, for a combined selection total of `218 passed`. Its single
sequential quantization, evaluation, and coverage integration smoke passed
with `1 passed`.

The quantized-activation/binary-bridge recovery checkpoint passed its focused
exact-order and two-boundary selection with `4 passed`. The adapted post-QDQ
boundary selector and new owner passed together with `2 passed`. The complete
architecture file passed with `134 passed`; artifact-metadata, artifact-policy,
core, and pass-efficiency passed separately with `85 passed`, for a combined
selection total of `219 passed`. Its single sequential quantization,
evaluation, and coverage integration smoke passed with `1 passed`.

The SiNet terminal layout-recovery checkpoint passed focused exact-order,
shape-boundary, terminal affine/slice, and runner-diagnostics checks with
`4 passed`. Its adapted terminal-clamp boundary and new owner passed together
with `2 passed`. The complete architecture file passed with `135 passed`; artifact-
metadata, artifact-policy, core, and pass-efficiency passed separately with
`85 passed`, for a combined selection total of `220 passed`. Its single
sequential quantization, evaluation, and coverage integration smoke passed
with `1 passed`.

The pre-Add/Mean attention-recovery checkpoint passed focused exact-order,
two-boundary, attention/QDQ composition, Mean-cluster scope, layout-prefix, and
runner-diagnostics checks with `5 passed`. The complete architecture file
passed with `136 passed`; artifact-metadata, artifact-policy, core, and pass-
efficiency passed separately with `85 passed`, for a combined selection total
of `221 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The SiNet pre-Add/Resize recovery checkpoint passed focused exact-order, four-
boundary, terminal-helper composition, terminal-clamp, and runner-diagnostics
checks with `4 passed`. Recursive helper expansion matches the preceding
lowerer AST exactly. The complete architecture file passed with `137 passed`;
artifact-metadata, artifact-policy, core, and pass-efficiency passed separately
with `85 passed`, for a combined selection total of `222 passed`. Its single
sequential quantization, evaluation, and coverage integration smoke passed
with `1 passed`.

The safe-binary and QLinear/Mean/Concat recovery checkpoint passed focused
exact-order, nested-helper composition, condition/progress/layout boundaries,
post-QDQ ownership, and runner-diagnostics checks with `6 passed`. Recursive
helper expansion matches the preceding lowerer AST exactly. The complete
architecture file passed with `139 passed`; artifact-metadata, artifact-policy,
core, and pass-efficiency passed separately with `85 passed`, for a combined
selection total of `224 passed`. Its single sequential quantization,
evaluation, and coverage integration smoke passed with `1 passed`.

The indexed shape-convergence checkpoint passed its focused dynamic-Reshape,
shape-reconciliation, legacy-equivalence, single-index-build, and ownership
selection with `13 passed`. The complete architecture file passed with
`140 passed`; artifact-metadata, artifact-policy, core, and pass-efficiency
passed separately with `85 passed`, for a combined selection total of
`225 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The final indexed shape/activation convergence checkpoint passed its focused
legacy-equivalence, single-index-build, no-consumer-rescan, differential-index,
shape, and ownership selection with `18 passed`. The complete architecture
file passed with `141 passed`; artifact-metadata, artifact-policy, core, pass-
efficiency, and the two new convergence cases passed separately with
`87 passed`, for a combined selection total of `228 passed`. Existing Conv,
DepthwiseConv, Add, Sub, Mul, and Div activation-fusion coverage passed with
`20 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The indexed broadcast-constant repair checkpoint passed four focused
shared-constant, no-op, inverse-rotation, and ownership cases plus four existing
rank-three/rank-four repair characterizations (`8 passed`). The complete
architecture file passed with `142 passed`; artifact-metadata, artifact-policy,
core, pass-efficiency, indexed final-convergence, and binary-layout coverage
passed separately with `90 passed`, for a combined selection total of
`232 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The indexed binary-layout convergence checkpoint passed six focused multi-
match, reversed-input, peer-producer, channelwise-constant, fan-out, exact-
legacy, one-build, and ownership checks. The complete related Conv/layout and
indexed-convergence selection passed with `15 passed`. The complete
architecture file passed with `143 passed`; artifact-metadata, artifact-policy,
core, pass-efficiency, indexed final-convergence, binary-layout, Conv-layout,
and indexed binary-convergence coverage passed separately with `105 passed`,
for a combined selection total of `248 passed`. Its single sequential
quantization, evaluation, and coverage integration smoke passed with
`1 passed`.

The indexed Conv-input adapter checkpoint passed its exact former-pair
equivalence, two-match stale-Transpose, one-index-build, no-map-rebuild,
fan-out, graph-output, filter-channel, and ownership coverage. The complete
related Conv/layout and indexed-convergence selection passed with `18 passed`.
The complete architecture file passed with `144 passed`; artifact-metadata,
artifact-policy, core, pass-efficiency, indexed final-convergence, binary-
layout, Conv-layout, indexed binary-convergence, and indexed Conv-input repair
coverage passed with `108 passed`. Its single sequential quantization,
evaluation, and coverage integration smoke passed with `1 passed`.

The indexed wrong-way Conv-Transpose sanitizer checkpoint passed its exact
former-implementation equivalence, two-match removal, multi-Conv consumer,
one-index-build, no-consumer-rescan, non-Conv fan-out, filter-channel, graph-
output, maintained-index, and ownership coverage with `3 passed`. The complete
related Conv/layout and indexed-convergence selection passed with `20 passed`.
The complete architecture file passed with `145 passed`; artifact-metadata,
artifact-policy, core, pass-efficiency, indexed final-convergence, binary-
layout, Conv-layout, indexed binary-convergence, indexed Conv-input repair, and
wrong-way sanitizer coverage passed with `110 passed`. Its single sequential
quantization, evaluation, and coverage integration smoke passed with
`1 passed`.

The shared indexed recurrent-alias checkpoint passed direct legacy-equivalence,
three-alias repair, first-Reshape ordering, public input/output, produced,
missing-shape, invalid-grammar, no-consumer, no-candidate/no-index, maintained-
index, direct/PyTorch wrapper equality, and ownership coverage with `8 passed`.
The complete recurrent, PyTorch normalization, recurrent-codegen-policy, and
new shared-owner selection passed with `18 passed`. The complete architecture
file passed with `146 passed`; the lightweight core/indexed selection passed
with `113 passed`. The real PyTorch exporter normalization regression passed
with `1 passed`, and the single sequential direct quantization, evaluation, and
coverage integration smoke passed with `1 passed`.

The indexed unbound-input layout checkpoint passed exact issue-report and
former-implementation equivalence across DEQUANTIZE, SHAPE, RESHAPE, SPLIT,
and two-consumer MUL-alias repair, plus nearest-source, quantization/signature,
mixed-fan-out guard, one-index-build, no-issue/no-index, maintained-index, and
ownership coverage plus nearest DEQUANTIZE fallback and strict exact-source
preference with `8 passed`. Its complete related QLinear/layout selection
passed with `9 passed`. The complete architecture file passed with
`147 passed`; the lightweight core/indexed selection passed with `117 passed`.
An actual GRU lowering/unbound-input check passed with `1 passed`, and the
single sequential direct quantization, evaluation, and coverage integration
smoke passed with `1 passed`.

The indexed quantized-activation checkpoint passed complete former-mutation
equivalence for two RELU/RELU6 chains, one-index-build, no-consumer-rescan,
maintained-index, public intermediate/source, fan-out, per-channel
quantization, non-inverse permutation, no-Transpose/no-index, ownership, and
real ONNX lowering coverage with `10 passed`. The complete architecture file
passed with `148 passed`; the lightweight core/indexed selection passed with
`125 passed`. Its single sequential direct quantization, evaluation, and
coverage integration smoke passed with `1 passed`.

The indexed expanded-HardSigmoid checkpoint passed exact valid-result
equivalence for RELU_0_TO_1 and MAXIMUM/MINIMUM forms in one graph, one-index
construction, no legacy consumer-map rebuild, maintained-index equivalence,
private and shared rank-four constant remapping, scalar constant preservation,
public add/clamp intermediates and source, fan-out, per-channel quantization,
non-inverse permutation, transactional missing-late-constant rejection, and
no-Transpose/no-index coverage. The complete architecture plus both indexed
quantized-activation files passed with `167 passed`; the lightweight core/
indexed selection passed with `112 passed`. Three real ONNX lowering checks,
TensorFlow-import-blocked direct and `-cotof`, and the sequential direct
quantization/evaluation/coverage integration smoke passed together with
`6 passed`.

The indexed expanded-PReLU checkpoint passed exact former-result equivalence
for two chains, one-index construction, no legacy consumer-map rebuild,
maintained-index equivalence, reversed MUL/ADD input order, private rank-four
MUL/alpha remapping, shared rank-four ADD cloning with quantization metadata,
scalar preservation, public intermediate/PRELU/source boundaries, fan-out,
per-channel quantization, non-inverse permutation, non-array and missing-alpha
rejection, and no-Transpose/no-index coverage. Complete architecture plus all
three indexed quantized-activation files passed with `178 passed`; the
lightweight core/indexed and related quantized-PReLU selection passed with
`128 passed`. Related real ONNX lowering, TensorFlow-import-blocked direct and
`-cotof`, and the sequential quantization/evaluation/coverage integration
smoke passed together with `7 passed`. An attempted exact public-output ONNX
fixture was not retained because the existing ordered trailing-output cleanup
correctly removes its post-Transpose before the specialized owner boundary.

The indexed quantized-logistic-gate checkpoint passed complete former-result
equivalence for simultaneous single- and multi-post chains, one-index
construction, no producer/consumer-map rebuild, maintained-index equivalence,
MUL input reversal, graph-order canonical output selection, alias consumer
consolidation, dtype/shape/signature/quantization propagation, public internal
data/gate/source/alias boundaries, pre/data/gate fan-out, non-Transpose post
users, per-channel quantization, wrong post permutation, and
no-Transpose/no-index coverage. Complete architecture plus all four indexed
quantized suites passed with `195 passed`; the lightweight core/indexed,
related quantized-PReLU, legacy logistic-gate characterization selection
passed with `145 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential direct integration selection passed with `7 passed`.

The single-owner wrong-way Conv-input safety-valve checkpoint preserves the
pre-extraction Swish-only ModelIR digest
`9b47e7f2e879895af600f66c6ac6929acc25580cfea8d5620fca9a6319ee4343` and
its two-removal Swish statistics. Focused exact legacy, multi-Conv, maintained-
index, public-output, mixed-fan-out, filter-channel, no-Transpose/no-index,
Swish-delegation, both existing Swish variants, and ownership coverage passed
with `7 passed`. The complete architecture, lightweight core/pass-efficiency,
focused sanitizer, and both Swish characterizations passed together with
`218 passed`. TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage integration smoke passed with `3 passed`.
No Tier corpus conversion was run.

The indexed primary Swish-QDQ branch checkpoint preserves complete former-
phase ModelIR and result equality across shared multi-branch, explicit concat-
closure, spatial guard, public intermediate, and data-fan-out fixtures. The
comprehensive existing fixture retains phase digest
`529b9889fafe9982ebb37ca63687b9329fa11a837562c154480c1856bbc05760`,
three rewritten branches, two removed pre-Transposes, and twenty rewritten
tensors. Focused shared quantized/float tails, one-index, maintained-index,
public/post-output/fan-out guards, small-spatial closure, no-Transpose/no-index,
both legacy Swish variants, and ownership coverage passed with `8 passed`.
Complete architecture, lightweight core/pass-efficiency, both indexed Swish/
Conv-safety suites, and the two legacy Swish characterizations passed together
with `224 passed`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage integration smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed Swish-QDQ metadata checkpoint preserves complete prior-phase
ModelIR/result equality for reverse-ordered fixed-point, public-output,
Pool-channel-mismatch, and wrong-tail fixtures. The comprehensive fixture
retains post-metadata digest
`bab34e6351ec24bc564b9f95b4550bbfaca867f15906f9d77b92f7e8adf1d804`,
one rewritten Concat axis, and twenty-four rewritten tensors. Focused unary,
binary broadcast/signature, Pool/Resize, strict Concat tail, fixed-point,
family guards, empty-seed/no-index, shared primary-index, shared late-shape
owner, both legacy Swish variants, and ownership coverage passed with
`13 passed`. Complete architecture, lightweight core/pass-efficiency, both
indexed Swish/Conv-safety suites, and legacy Swish characterizations passed
together with `229 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential quantization/
evaluation/coverage integration smoke passed with `3 passed`. No Tier corpus
conversion was run.

The indexed Swish-QDQ inverse-post checkpoint first proves the two former
lowerer loops have identical ASTs, then preserves complete former-loop ModelIR
and the three-removal result for chained aliases, multi-consumer fan-out,
public alias output, wrong permutation, and untracked input. Focused exact-
legacy, maintained-index, one-index, empty-seed/no-index, two-call ownership,
all existing indexed Swish cases, and both legacy Swish variants passed with
`16 passed`. Complete architecture, lightweight core/pass-efficiency, both
indexed Swish/Conv-safety suites, and legacy Swish characterizations passed
together with `232 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential quantization/evaluation/coverage integration smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed Swish-QDQ late-Concat checkpoint compiles the exact prior committed
late-loop AST and preserves its complete ModelIR, rewritten tensor set, one
axis rewrite, and two input-adapter removals on the mixed direct/DQ fixture.
Focused characterization covers maintained-index equivalence, mixed-input
rewiring, one shared late index, complete post-tail removal, retained direct
fan-out, public source and Concat outputs, missing tensors, mismatched shapes,
wrong tail permutation, transactional no-op behavior, and the missing-required-
type/no-index preflight. The complete indexed Swish and architecture selection
passed with `169 passed`. Architecture, core, pass-efficiency, indexed Swish,
wrong-way Conv safety, and both legacy Swish characterizations passed together
with `237 passed`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage integration smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed Concat pre-Q/DQ checkpoint compiles the complete prior committed
function AST and preserves exact ModelIR and statistics for both one and two
simultaneous matches. Focused characterization covers two-match fixed-point
rewriting with one index construction, maintained-index equivalence, scale and
dtype mismatch, quantized fan-out, public quantized/dequantized boundaries,
shape mismatch, non-Dequantize provenance, rounding-preserving arithmetic
rejection, exact-grid acceptance, and no-Concat/no-index pruning. The focused
owner and legacy selection passed with `26 passed`. Architecture, core, pass-
efficiency, quantization cleanup, and the two established Concat-Q/DQ tests
passed together with `238 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage integration smoke
passed with `3 passed`. No Tier corpus conversion was run.

The indexed terminal Transpose/Dequantize checkpoint compiles the complete
prior committed function AST and preserves exact ModelIR and both statistics
for Transpose-to-Dequantize and Dequantize-to-Transpose forms with one and two
simultaneous matches. Focused characterization covers a single index build
across both subphases, maintained-index equivalence, output-name and metadata
preservation, terminal/public/consumer boundaries, shared Transpose output,
per-channel quantization, invalid permutation, missing tensor, required-type/
no-index pruning, ownership, and the established real ONNX lowering case. The
focused selection passed with `35 passed`. Architecture, core, pass-efficiency,
quantization cleanup, real terminal sanitation, and both Concat-Q/DQ
characterizations passed together with `249 passed`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed`. No Tier corpus conversion was run.

The indexed Transpose-DQ-Mean-Q checkpoint compiles the complete prior
committed function AST and preserves exact valid ModelIR and statistics for one
and two simultaneous matches. A separate differential check proves that an
invalid permutation, which formerly left partially rewritten DQ/Mean metadata
despite returning zero, is now a complete ModelIR no-op. Focused coverage
includes one-index multi-match execution, maintained-index equivalence,
negative-axis remapping, edge and operator order, shape/signature propagation,
public and fan-out boundaries at every intermediate, `keepDims`, shared axes,
invalid axes/permutation, missing tensors, missing-required-type/no-index
pruning, and ownership with `48 passed`. Architecture, core, pass-efficiency,
and the complete quantization-cleanup suite passed with `260 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential quantization/
evaluation/coverage smoke passed with `3 passed`. No Tier corpus conversion was
run.

The indexed pseudo-LeakyReLU checkpoint compiles the complete prior committed
function AST and preserves exact ModelIR and statistics for one and two
simultaneous matches, including alpha constants on both MUL sides and nondefault
legacy SUB fields. Focused coverage verifies one index build, maintained-index
and LayoutState equivalence, batch producer removal, alpha values, reversed SUB
rejection, missing constant, every public intermediate, fan-out at every edge,
negative-source mismatch, integer boundaries, missing-family/no-index pruning,
and ownership with `17 passed`. Architecture, core, pass-efficiency, complete
graph cleanup, and the indexed fusion suite passed with `249 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential quantization/
evaluation/coverage smoke passed with `3 passed`. No Tier corpus conversion was
run.

The generic indexed MUL-square constant-fold checkpoint compiles the complete
prior committed YOLO-named function AST and preserves exact valid ModelIR and
statistics for one and two simultaneous matches. A separate differential check
proves that a public square intermediate, formerly rewritten despite losing its
producer contract, is now a complete no-op. Focused coverage verifies one-index
multi-match execution, all pre/anchor/scale constant-side combinations,
maintained-index and LayoutState equivalence, float16 fused values, normalized
metadata, quantization cloning, batch compaction, public/fan-out guards for all
three intermediates, singleton and finite pre-scale, floating anchor/scale,
finite result, exact self-square, missing constants, no-MUL/no-index/no-prune,
and generic ownership with `18 passed`. Architecture, core, pass-efficiency,
constant-fold, and indexed fold coverage passed with `234 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential quantization/
evaluation/coverage smoke passed with `3 passed`. No Tier corpus conversion was
run.

The indexed leading-singleton Gather-to-Reshape checkpoint compiles the
complete prior committed function AST and preserves exact valid ModelIR,
lineage metadata, and statistics for one and two simultaneous matches. A
separate differential check proves that multiple zero indices and a dynamic
leading signature, both formerly rewritten, are now complete no-ops. Focused
coverage verifies one-index multi-match execution, negative-axis
normalization, nested fixed-point exposure, maintained-index and LayoutState
equivalence, matching quantization, all public/fan-out/duplicate/order/input-
position boundaries, axis and batch-dimension options, static and dynamic
metadata consistency, constant buffer dtype/value/cardinality,
dtype/quantization equality, missing tensors, transactional rejection,
missing-family/no-index pruning, and unique semantic ownership with `38 passed`.
Architecture, core, pass-efficiency, ModelIR utilities, dynamic Reshape, and
indexed Gather/Reshape coverage passed with `269 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage smoke passed with `3 passed`. No
Tier corpus conversion was run.

The indexed terminal Softmax/Transpose checkpoint compiles the complete prior
committed function AST and preserves exact valid ModelIR, lineage metadata,
and statistics for one and two simultaneous terminal outputs. Separate
differential checks prove that a separately public Softmax intermediate and a
missing Softmax-output tensor, both formerly rewritten, are now complete
no-ops. Focused coverage verifies one-index multi-output execution,
maintained-index and LayoutState equivalence, marker removal with axis/options
preservation, public output identity, all public input/output boundaries,
dtype/shape/signature propagation, quantization cloning, terminal and Softmax
fan-out, duplicate producers, operator order, marker truth, exact permutation,
operator arity/type, missing permutation and runtime buffer, missing tensors,
rank/signature validation, non-output exclusion, missing-family/no-index
pruning, shared marker ownership, and unique semantic ownership with
`30 passed`. Architecture, core, pass-efficiency, terminal Mean layout, layout
Transpose, and indexed terminal
Softmax coverage passed with `255 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed pre-ArgMax terminal-layout checkpoint compiles the complete prior
committed function AST and preserves exact valid ModelIR, lineage metadata,
and statistics for private singleton axes, two matches sharing one axis, and a
negative channel axis. Differential checks prove that a public axis remains
one while a private clone becomes three, instead of mutating the public value
as before, and that a Transpose intermediate exposed as a public input is now a
complete no-op. Focused coverage verifies one-index multi-match execution,
shared-axis ownership changes across differential removal, maintained-index
and LayoutState equivalence, private and public input/output axis constants,
negative-axis normalization, NumPy dtype and quantization cloning, exact
operator options/provenance, every public/fan-out/duplicate/order boundary,
permutation and operator arity/type, signed singleton axis validation, all
required tensors, rank-four permuted shape/signature and reduced-output
metadata, dtype agreement, transactional rejection, missing-family/no-index
pruning, and unique semantic ownership with `39 passed`. Architecture, core,
pass-efficiency, layout Transpose, indexed terminal ArgMax, and indexed terminal
Softmax coverage passed with `288 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed quantized-MaxPool checkpoint compiles the complete prior committed
function and preserves exact ModelIR and statistics for valid one- and two-
chain fixtures across INT8 and UINT8. Differential checks separately prove
that the former tolerant comparison folded a near-but-different scale, that
missing float bridge tensors were accepted, and that a public-input bridge
could lose its producer; all three are now transactional no-ops. Focused
coverage verifies one-index multi-match execution, maintained-index and
LayoutState equivalence, quantized input/output fan-out, dictionary grids,
Pool options/version/provenance, cloned quantization, public boundaries,
duplicate producers, operator order and arity, exact grid/dtype/range, float
bridge dtype, exact rank-four shape/signature metadata, missing tensors,
missing-family/no-index pruning, and unique semantic ownership with
`52 passed`. Architecture, core, pass-efficiency, established quantized-Pool and
quantization-cleanup coverage, and the new indexed suite passed together with
`318 passed`. TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff on the new
owner/test and architecture test, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed quantized-Logistic checkpoint compiles the complete prior
committed function and preserves exact ModelIR and statistics for valid one-
and two-chain fixtures across INT8 and UINT8. Differential checks separately
prove that the former tolerant comparison folded a near-canonical output
scale, missing input quantization was accepted, missing float bridge tensors
were accepted, and a public-input bridge could lose its producer; all four are
now transactional no-ops. Focused coverage verifies one-index multi-match
execution, maintained-index and LayoutState equivalence, quantized input/output
fan-out, dictionary grids, rank-independent elementwise metadata, Logistic
options/version/provenance, public boundaries, duplicate producers, operator
order and arity, input grid validity, exact canonical output grid, float dtype,
shape/signature equality, missing tensors, missing-family/no-index pruning,
the two established direct tests, and unique semantic ownership with
`55 passed`. Architecture, core, pass-efficiency, established quantized-Pool
and quantization-cleanup coverage, both indexed quantized suites, and the two
direct Logistic tests passed together with `373 passed`. TensorFlow-import-
blocked direct and `-cotof` plus the sequential quantization/evaluation/
coverage smoke passed with `3 passed`. Ruff on the new owner/test and
architecture test, syntax compilation, and `git diff --check` passed. No Tier
corpus conversion was run.

The indexed quantized-Softmax checkpoint compiles the complete prior committed
function and preserves exact ModelIR and statistics for valid one- and two-
chain fixtures across INT8 and UINT8, including the former beta tolerance.
Differential checks separately prove that the former tolerant grid comparison
folded a near-canonical output scale, missing input quantization and float
bridge tensors were accepted, a public-input bridge could lose its producer,
and an explicit non-last axis was ignored; all five are now transactional no-
ops. Focused coverage verifies one-index multi-match execution, maintained-
index and LayoutState equivalence, quantized input/output fan-out, dictionary
grids, rank-two negative-axis handling, beta/default/axis option preservation,
Softmax version/provenance, public boundaries, duplicate producers, operator
order and arity, input grid validity, exact canonical output grid, malformed
options, positive rank, float dtype, shape/signature equality, missing tensors,
missing-family/no-index pruning, and unique semantic ownership with
`62 passed`. The real QLinearSoftmax wrap conversion and sequential inference
matched ONNX exactly with `1 passed`. Architecture, core, pass-efficiency,
established quantized-Pool and quantization-cleanup coverage, and all three
indexed quantized suites passed together with `433 passed`. TensorFlow-import-
blocked direct and `-cotof` plus the sequential quantization/evaluation/
coverage smoke passed with `3 passed`. Ruff on the new owner/test and
architecture test, syntax compilation, and `git diff --check` passed. No Tier
corpus conversion was run.

The indexed expanded-HardSigmoid fold checkpoint compiles the complete prior
committed function and preserves exact ModelIR/statistics for private one- and
two-chain INT8/UINT8 fixtures and a shared-four-constant fixture. Differential
checks prove that a near-equal output grid and missing float tensors formerly
folded, a public-input bridge could lose its producer, and public scalar
outputs were mutated in place. A fault-injected second quantization clone also
proves the former helper changed the first scalar data/dtype before raising,
while the new four-plan transaction returns a complete no-op. Focused coverage
verifies one-index multi-match execution, maintained-index and LayoutState
equivalence, all scalar input positions, shared/public clone ownership and
names, operator options/version/provenance, every public/fan-out/duplicate/
order/arity boundary, exact grid validity, all data metadata, finite singleton
constants, producer rejection, representability, clone-failure transaction,
missing-family/no-index pruning, and unique semantic ownership with
`78 passed`. Architecture, core, pass-efficiency, the established quantized
Pool and quantization-cleanup suites, all prior indexed quantized folds,
quantized activation and
expanded-HardSigmoid bridge suites, and the new fold suite passed together
with `529 passed`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test and architecture test, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed quantized-TransposeConv checkpoint compiles the complete prior
committed function and preserves exact ModelIR/statistics for valid private
one- and two-chain fixtures and a shared-filter fixture. Differential checks
prove that missing float input/output tensor records and a public-input bridge
formerly folded, a public float filter was mutated in place, an invalid
negative input scale was accepted, and output-grid clone failure changed a
private float filter before raising. The new owner rejects or clones each case
transactionally. Focused coverage verifies one-index multi-match execution,
maintained-index and LayoutState equivalence, private/shared/public and
already-INT8 filter ownership, exact output-shape/filter/data roles, operator
options/version/provenance, every public/fan-out/duplicate/order/arity
boundary, independent activation-grid validity, bridge dtype and metadata,
filter rank/buffer/dtype/grid/producer constraints, clone-failure transaction,
missing-family/no-index pruning, and unique semantic ownership with
`61 passed`. Architecture, core, pass-efficiency, the established quantized
Pool and quantization-cleanup suites, all prior indexed quantized folds,
quantized activation and expanded-HardSigmoid bridge/fold suites, and the new
TransposeConv suite passed together with `590 passed`. TensorFlow-import-
blocked direct and `-cotof` plus the sequential quantization/evaluation/
coverage smoke passed with `3 passed`. Ruff on the new owner/test, scoped
architecture/lowerer checks, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed decomposed-InstanceNormalization checkpoint compiles the complete
prior committed function and preserves exact ModelIR/statistics for valid
NHWC, NCHW, post-Transpose, and rank-five fixtures. Differential checks prove
that reversed Sub operands, a non-Add epsilon node, a public Mean intermediate,
a shared scale, a wrong-sized scale, and floating axes formerly mutated graph
state. A malformed late bias shape additionally proves that the former helper
changed axes, intermediate shapes, and scale data before raising, while the
new owner returns a complete no-op. Focused coverage verifies one-index
multi-layout execution, maintained-index and LayoutState equivalence, ranks
three/four/five, separate/shared Mean axes, optional post-Transpose bias-axis
mapping, already-correct idempotence, plan-failure transaction, all operator
types/roles/arity/order, public/fan-out/duplicate boundaries, finite epsilon
and reciprocal-one constants, axes dtype/ownership, complete intermediate
metadata, scale/bias cardinality and ownership, missing-marker/no-index
behavior, and unique semantic ownership with `38 passed`. Four existing real
ONNX builder/serialization characterizations passed. Architecture, core,
pass-efficiency, the new indexed suite, and those real characterizations passed
together with `264 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test, scoped architecture/lowerer checks, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed NCHW Concat/global-pool/Conv checkpoint compiles the complete prior
committed function and preserves exact ModelIR/statistics for valid one- and
two-chain fixtures, negative spatial axes, and INT64 Reshape shape buffers.
Differential checks prove that non-global Mean axes, a fan-out or public Concat
intermediate, a three-value or floating shape buffer, a duplicate Reshape
producer, and a missing runtime filter buffer formerly changed graph state. A
faulting late shape-buffer read additionally proves that the former helper
changed the Concat axis and three tensor records before raising, while the new
owner returns a complete no-op. Focused coverage verifies one-index multi-
match execution, maintained-index and LayoutState equivalence, exact operator
roles/order/arity, normalized global axes, public/fan-out/duplicate boundaries,
positive compatible NCHW inputs, the filter/channel equation and buffer,
private producer-free integer shape-buffer ownership, option/metadata/data
updates, incomplete-family/no-index behavior, the existing characterization,
and unique semantic ownership with `34 passed`. Architecture, core, pass-
efficiency, all existing Conv-layout tests, and the new indexed suite passed
together with `268 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test, scoped architecture/lowerer checks, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed NCHW Concat/Transpose/(Transpose)Conv checkpoint compiles the
complete prior committed function and preserves exact ModelIR/statistics for a
combined direct-Conv, pre/post-prefix Conv, and TransposeConv fixture.
Differential checks prove that public or fan-out Transpose/pre-passthrough
outputs, a duplicate Transpose-output producer, a nonpositive input channel,
a produced permutation, and a missing runtime filter buffer formerly changed
graph state; the new owner rejects all eight cases transactionally. Focused
coverage verifies one-index multi-family execution, maintained-index and
LayoutState equivalence, the four established positive characterizations,
exact Transpose and data/filter roles, pre/post passthrough traversal,
public/fan-out/duplicate/order boundaries, positive compatible NCHW inputs,
the filter/channel/buffer equation, direct-Conv-only output refresh, already-
correct exclusion, incomplete-family/no-index behavior, and unique semantic
ownership with `30 passed`. Architecture, core, pass-efficiency, all existing
Conv-layout tests, and both new indexed Concat repair suites passed together
with `294 passed`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test, scoped architecture/lowerer checks, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed mixed-singleton NCHW-input/NHWC-Concat checkpoint compiles the
complete prior committed function and preserves exact ModelIR, lineage,
operator order, deterministic names, quantization metadata, and statistics for
valid multi-candidate, multi-adapter, and name-collision fixtures. Differential
checks prove that an output-channel mismatch, duplicate or later source
producer, inconsistent dynamic signature, and mixed dtype formerly changed
graph state; all five are now complete no-ops. A late fault in the second
quantization clone additionally proves that the former helper inserted the
first Reshape before raising, while the new owner leaves ModelIR unchanged.

Focused coverage verifies one-index multi-match execution, supplied-index and
LayoutState equivalence, public/produced/constant sources, same-source adapter
reuse, global name reservation, deep quantization cloning, exact Concat axis,
arity, output-channel and dtype equations, static and one-dynamic-dimension
shape/signature contracts, final shape-reconciliation stability, duplicate/
later/unresolved producer rejection, clone-failure transaction, no-repair and
missing-family behavior, the established characterization, and unique
semantic ownership with `33 passed`. Architecture, core, pass-efficiency,
singleton-Reshape coverage, the two preceding indexed Concat owners, flatten-
Concat and NDHWC-Concat coverage, the new indexed suite, and the existing
characterization passed together with `336 passed`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed`. Ruff on the new owner/test and architecture test,
scoped lowerer checks, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed window-partition checkpoint compiles the complete prior committed
function and preserves exact ModelIR, lineage, operator order, Transpose
options replacement, version/axis semantics/provenance, quantization metadata,
unused-tensor pruning, and statistics for valid public-input, produced-input,
and quantized multi-chain fixtures. Differential checks prove that duplicate
first-Reshape producers, floating shape vectors, missing final-output metadata,
mixed data dtypes, and a graph-input/producer conflict formerly changed graph
state; all five are now complete no-ops.

Focused coverage verifies one-index multi-match execution, supplied-index and
LayoutState equivalence, exact block/spatial and three-stage shape equations,
static and one-dynamic-dimension contracts, final dynamic-Reshape convergence,
per-tensor quantization, in-place provenance, topology/order/arity, public and
fan-out boundaries, duplicate/later/unresolved producers, typed producer-free
shape/permutation vectors, dynamic shape-vector ownership, two-dynamic-output
rejection, historical prune/no-index behavior, and unique semantic ownership
with `52 passed`. Architecture, core, pass-efficiency, dynamic-Reshape, the new
indexed suite, and two real ONNX SpaceToDepth chain characterizations passed
together with `289 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test and architecture test, scoped lowerer checks, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed window-reverse checkpoint compiles the complete prior committed
function and preserves exact ModelIR, lineage, operator order, deterministic
shared-shape clone names, Reshape and DEPTH_TO_SPACE options, version/axis
semantics/provenance, quantization metadata, unused-tensor pruning, and
statistics for a five-chain public-input, produced-input, quantized, and
shared-vector fixture. Differential checks prove that an extra first-Reshape
input, a floating shape vector, a public first shape vector, mixed data dtypes,
and an inconsistent shape signature formerly changed graph state; all five are
now complete no-ops.

Focused coverage verifies one-index multi-match execution, supplied-index and
LayoutState equivalence, legacy sequential shared-vector clone/update behavior,
exact reverse block/flatten/spatial and three-stage shape equations, static and
one-dynamic-dimension contracts, final shape-convergence stability, per-tensor
quantization, in-place provenance, topology/order/arity, public and fan-out
boundaries, duplicate/later/unresolved producers, typed producer-free vectors,
two-dynamic-target rejection, clone-failure transaction, historical
prune/no-index behavior, unique semantic ownership, and a production real-ONNX
characterization. The reverse and partition focused suites passed together
with `99 passed`. Architecture, core, pass-efficiency, dynamic-Reshape,
ModelIR-writer, strict-integer-quantization, both indexed window suites, and an
established DepthToSpace characterization passed together with `381 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff on the
owner/test and architecture test, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed Conv1D-shim unary checkpoint compiles the complete prior committed
function and preserves exact ModelIR, lineage, operator order, unary identity,
options/version/axis semantics/provenance, output dtype and quantization repair,
unused-tensor pruning, and statistics for a five-chain public-input,
produced-input, quantized, CAST, inferred-axis, and alternate-axis fixture.
Differential checks prove that a floating permutation, produced ExpandDims
axis, inconsistent internal shape metadata, mixed input dtype, and duplicate
final producer formerly changed graph state; all five are now complete no-ops.
A faulting quantization clone additionally proves that the former helper
rewired the unary before raising, while the indexed owner leaves ModelIR
unchanged.

Focused coverage verifies one-index multi-match execution, supplied-index and
LayoutState equivalence, all sixteen supported unary types, explicit/negative
and uniquely inferred axes, CAST dtype transition, output metadata repair,
consistent multi-dynamic signatures, exact NHWC/NCHW permutation and
drop/insert equations, every operator role/order/arity, public and fan-out
boundaries, duplicate/later/unresolved producers, typed producer-free
permutation and axis vectors, per-tensor quantization, per-axis rejection,
clone-failure transaction, historical prune/no-index behavior, and unique
semantic ownership with `75 passed`. Architecture, core, pass-efficiency,
three established transpose-unary suites, five adjacent Conv1D-shim
characterizations, and the new indexed suite passed together with `321
passed`. TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff on the
owner/test and architecture test, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed rank-four Conv1D-shim unary checkpoint also compiles the complete
prior committed function. A static public-input, produced-input, quantized,
and CAST four-chain fixture preserves exact operators, tensors, inputs,
outputs, metadata, and statistics. Differential checks prove that floating
permutation constants, a produced ExpandDims axis, mixed data dtypes, and a
duplicate final producer formerly rewrote graph state; all four are now
complete no-ops. Focused coverage verifies one-index multi-match execution,
supplied-index and LayoutState equivalence, deterministic shared shape/axis
constant cloning, fan-out bridge topology, one dynamic height/width/channel
dimension, 53 unsafe contracts, quantization-clone failure, historical
prune/no-index behavior, and both established direct-builder
characterizations with `63 passed`. Both indexed Conv1D suites, architecture,
core, pass-efficiency, three established Transpose-Unary suites, and five
adjacent direct-builder characterizations passed together with `383 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed Conv1D unary fan-out checkpoint compiles the complete prior
committed helper. For four static public-input, produced-input, quantized, and
alternate-axis branches, the new ModelIR and statistics are exactly equal to
the legacy result after topologically sorting that result. The legacy result
required twelve operator moves; the new owner emits the final order directly.
CAST additionally repairs the retained Transpose output from `FLOAT32` to the
unary's `INT32` output contract. Differential checks prove that a chain without
fan-out, floating permutation, produced ExpandDims axis, duplicate final
producer, and mixed input dtype formerly rewrote graph state; all five are now
complete no-ops. A faulting quantization clone formerly raised after changing
the final dtype, while the indexed owner leaves ModelIR unchanged.

Focused coverage verifies one-index five-chain execution, supplied-index and
LayoutState equivalence, all sixteen unary types, public NCHW output, three
dynamic-signature forms, 27 unsafe contracts, clone-failure transaction,
equivalent negative axes, historical prune/no-index behavior, and the
established direct-builder characterization with `58 passed`. All three
indexed Conv1D suites, architecture, core, pass-efficiency, three established
Transpose-Unary suites,
and five adjacent direct-builder characterizations passed together with `437
passed`. TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff with only
the documented legacy unused-import exclusions, syntax compilation, and `git
diff --check` passed. No Tier corpus conversion was run.

The indexed flattened InstanceNormalization checkpoint compiles the complete
prior committed 491-line helper. Four static public-input, produced-input,
CAST, and alternate-unary branches retain exact operators, tensors, inputs,
outputs, metadata, and statistics. Differential checks prove that a floating
permutation, produced second-Reshape shape, negative epsilon, reversed
reciprocal DIV, mixed intermediate dtype, and duplicate final producer
formerly rewrote graph state; all six are now complete no-ops.

Focused coverage verifies one-index five-chain execution, supplied-index and
LayoutState equivalence, deterministic shared Reshape/axis cloning, all
sixteen unary types, one dynamic batch/width/channel dimension, 37 unsafe
contracts, clone-failure transaction, historical prune/no-index behavior, and
the established direct-builder characterization with `61 passed`. All four
indexed Conv1D suites, architecture, core, pass-efficiency, three established
Transpose-Unary suites, and five adjacent direct-builder characterizations
passed together with `498 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed`. Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed tencoder checkpoint removes the former 723-line raw helper from
the lowerer. Its dedicated owner matches the exact flattened
InstanceNormalization prefix rather than searching an arbitrary upstream
producer path, validates the simple rank-four or legacy rank-three residual
branch, and proves the complete two-Slice/Logistic/Mul/scale gate, residual
ADD, ExpandDims, post-Transpose, and Conv consumer topology before mutation.
The InstanceNormalization owner now exposes that common prefix as a
side-effect-free plan while retaining its prior complete-chain behavior.

The rewrite converts both residual inputs from NCW to NWC, adjusts the second
Reshape, Slice begin/size, floating channel-scale, and ExpandDims constants,
repairs every changed tensor including Logistic and gate intermediates, and
removes the three boundary Transposes in one indexed compaction. Private
constants update in place; shared constants receive unique planned clones even
when two changed operator inputs originally share the same tensor. A side
consumer receives one `[0,2,1]` bridge immediately before its earliest use.

Focused coverage includes exact numeric equivalence for simple/legacy left
branches with and without fan-out, supplied-index and LayoutState equivalence,
topological bridge placement, deterministic shared integer/float cloning,
one dynamic batch or width dimension, eight unsafe transactional no-ops, the
three established characterizations, and both semantic-ownership checks with
`84 passed`. The four indexed Conv1D
suites, architecture, core, pass-efficiency, three established
Transpose-Unary suites, and eight adjacent direct-builder characterizations
passed together with `520 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed`. Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed Conv1D BatchMatMul checkpoint removes the former 223-line helper
and its repeated whole-graph consumer-map rebuild. Its dedicated owner accepts
only exact typed `[0,3,1,2]` Transpose roots, one explicit singleton Squeeze,
an optional strict supported-unary chain, and the left input of one
BatchMatMul. Source and intermediate shapes/signatures, producer order,
consumer multiplicity, public boundaries, dtype transitions, per-tensor
quantization, right operand, contracted dimensions, batch broadcasting, and
output shape are all validated before mutation.

Squeezing the transposed channel axis maps to the original NHWC channel axis
without changing rank-three order or `adjX`. Squeezing either supported
spatial axis maps back to the corresponding NHWC spatial axis, swaps the last
two dimensions of Squeeze and unary metadata, and toggles `adjX`; the effective
matrix and BatchMatMul output remain identical. The original unary and
BatchMatMul objects retain all unrelated options, provenance, version, and
axis semantics, and the single boundary Transpose is removed through the
maintained index.

Focused coverage includes twelve exact NumPy equivalence variants across all
three axes, both `adjX` values, and public/produced sources; zero-length unary
chains, rank-two right operands, `adjY`; all sixteen unary types; one dynamic
batch signature; fifteen unsafe transactional no-ops; the preflight/no-index
path; and semantic ownership with `49 passed`. The five
indexed Conv1D suites, architecture, core, pass-efficiency, three established
Transpose-Unary suites, and eight adjacent direct-builder characterizations
passed together with `569 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed`. Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed decoder deconvolution-input checkpoint removes the former
193-line helper and its repeated producer/consumer-map rebuilds. Its dedicated
owner validates the complete BatchMatMul, commutative bias ADD, axis-two
ExpandDims, `[0,2,3,1]` Transpose, and input-two TransposeConv path. Both matrix
operands, every producer and consumer, public boundaries, concrete and dynamic
shape/signature equations, dtype, per-tensor quantization, contracted
dimension, batch broadcasting, bias broadcast, and operator order are proven
before any mutation.

The rewrite applies `(A·B)^T = B^T·A^T`: it swaps the original BatchMatMul
inputs, maps `adjX` to `not old_adjY` and `adjY` to `not old_adjX`, changes
`[N,C,L]` metadata to `[N,L,C]`, reshapes the length bias to `[1,L,1]`, moves
ExpandDims from axis two to axis one, and connects its retained
`[N,1,L,C]` output directly to TransposeConv. Unrelated BatchMatMul options and
provenance remain intact. Private constants update in place; shared bias and
axis constants are cloned before edges change. Clone failure and every unsafe
contract are complete no-ops.

Focused coverage includes sixteen exact NumPy equivalence variants across all
`adjX/adjY` values, both ADD input orders, and public/produced operands; rank-
two RHS, rank-two/three bias, negative axis, shared bias/axis cloning, one
dynamic batch signature, twenty-six unsafe transactional no-ops, clone
failure, preflight/no-index behavior, and semantic ownership with `51 passed`.
The six indexed Conv1D/decoder suites, architecture, core, pass-efficiency,
three established Transpose-Unary suites, and eight adjacent direct-builder
characterizations passed together with `620 passed`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed`. Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed terminal Squeeze/Mean checkpoint removes the former 152-line raw
helper and its repeated whole-graph consumer-map rebuild. Its dedicated owner
validates the exact `[0,3,1,2]` Transpose, axis-two Squeeze, axis-one kept-
dimension Mean, and axis-one terminal Squeeze topology. Every producer,
consumer, public boundary, typed constant, operator order, singleton,
rank-four/rank-three/rank-two shape and signature equation, dtype, and per-
tensor quantization contract is proven before mutation.

The rewrite removes the Transpose, moves the first Squeeze to the NHWC source
axis one, changes `[N,C,W]` metadata to `[N,W,C]`, moves the Mean to axis two,
changes `[N,1,W]` metadata to `[N,W,1]`, and moves the terminal Squeeze to axis
two. The final `[N,W]` tensor name, values, metadata, graph-output position,
and downstream edges remain unchanged. Private Mean axes update in place;
shared axes receive a deterministic clone. Negative equivalent axes and one
dynamic batch, width, or reduced-channel signature are preserved. Clone
failure and all unsafe contracts are complete no-ops.

Focused coverage includes sixteen exact NumPy equivalence variants across
public/produced sources and all positive/negative axis representations,
shared-axis cloning, dynamic batch/width/reduced-channel signatures, two-chain
execution, twenty-seven unsafe transactional no-ops, clone failure,
apply-preflight collision, and preflight/no-index behavior with `51 passed`;
the semantic-ownership test also passed. The seven indexed
Conv1D/decoder/terminal suites, architecture, core, pass-efficiency,
Mean/terminal-Mean, and three Transpose-Unary suites passed
together with `677 passed in 56.59s`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed in 6.81s`. Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed direct InstanceNormalization checkpoint makes the single private
post-Transpose tail the first bounded owner within the former four-mode raw
helper. It proves the pre-Transpose/Squeeze/Reshape boundary, both kept-
dimension Means, ordered SUB and reciprocal DIV, square/normalize/affine
branches, and the sole post-Transpose before changing ModelIR. Exact
shape/signature equations, all producers and consumers, graph order, public
boundaries, FLOAT16/FLOAT32 dtype, unquantized tensors, typed reshape/axis
constants, nonnegative finite epsilon, unit numerator, and finite scale/bias
are validated in one `ModelIRGraphIndex`.

The rewrite moves the normalization to NHWC axes `[1,2]`, converts rank-three
CHW and every full/reduced rank-four metadata contract, changes scale and bias
from `[1,C,1,1]` to `[1,1,1,C]`, reuses the post-Transpose output name on the
bias ADD, and removes the two boundary Transposes. Reshape, shared Mean-axis,
scale, and bias constants are planned together; unrelated consumers receive
deterministic clones. Five representative static legacy variants are exactly
ModelIR-identical to the committed helper, including lineage-event order.
Separate equivalent Mean axes are now handled transactionally instead of
being skipped by the legacy shared-axis guard.

The side-Squeeze, Squeeze/unary/Reshape, and
Squeeze/residual-ADD/Reshape modes remain in the compatibility path. Each is
an indexed-owner no-op followed by a numerically equivalent legacy rewrite;
supplied GraphIndex and LayoutState remain current. Unsafe direct candidates
such as reversed SUB, wrong axes, negative epsilon, or quantized intermediates
are explicitly short-circuited and cannot fall through to the legacy mutator.
Per-candidate dispatch preserves graph order across mixed direct and legacy
tails; a 33-chain characterization proves the original shared 32-rewrite
limit selects the same prefix rather than favoring all direct tails first.

Focused coverage includes thirty-two NumPy equivalence variants across
FLOAT16/FLOAT32, public/produced source, shared/separate axes, positive/negative
axes, and affine input order; shared cloning for all four changed constant
roles; dynamic height/width/channel signatures; two-chain execution; forty-
four unsafe transactional no-ops; clone failure; apply-preflight collision;
four legacy-fallback blockers; all three retained legacy modes; compatibility
statistics; mixed-mode order/limit; and preflight/no-index behavior with
`93 passed`. The related
InstanceNormalization, Pad, Mean, architecture, core, and pass-efficiency
suites passed with `495 passed in 54.58s`; twelve selected direct-builder
characterizations passed in `0.94s`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed in 6.70s`. Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed side-Squeeze InstanceNormalization checkpoint reuses that common
decomposition matcher for the exact dual-consumer tail: one NCHW-to-NHWC
post-Transpose and one axis-zero Squeeze. It moves the common normalization to
NHWC, removes the two boundary Transposes, and inserts one local
NHWC-to-NCHW adapter immediately before the side Squeeze. The side tensor name,
CHW shape/signature, dtype, quantization contract, public-output behavior, and
downstream edges remain unchanged. A compatible shared INT32 or INT64 adapter
permutation is reused; every absent, conflicting, produced, public, floating,
or quantized constant case is resolved before mutation. Unsafe side candidates
cannot fall through to the legacy mutator. Direct and side candidates retain
their original graph-order position and shared 32-rewrite ceiling; the two
remaining unary/Reshape and residual-ADD/Reshape modes are untouched.

Focused coverage includes sixteen NumPy equivalence variants across side
operator order, public/downstream-consumed side outputs, public/produced
sources, and FLOAT16/FLOAT32; dynamic height/width/channel signatures;
multi-chain constant reuse; five invalid existing-adapter cases; nine unsafe
side contracts; adapter-allocation collision; three compatibility-fallback
blockers; and the existing direct and mixed-mode cases with `131 passed in
0.89s`. Four representative side-tail variants were exactly ModelIR-identical
to the committed legacy helper. The related InstanceNormalization, Pad, Mean,
architecture, core, and pass-efficiency suites passed with `533 passed in
53.58s`; twelve selected direct-builder characterizations passed with `12
passed in 0.94s`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage smoke passed with `3 passed in
6.67s`. Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed unary/Reshape InstanceNormalization checkpoint reuses the same
core matcher for the exact `Squeeze -> unary -> Reshape -> post-Transpose`
tail. The tail Squeeze and unary metadata move from CHW to HWC, the second
Reshape moves from NCHW to NHWC, its typed shape constant and `newShape` are
rewritten together, and that Reshape receives the former post-Transpose output
name. Shared shape constants receive deterministic clones. All thirteen
legacy unary operators remain accepted; CAST alone may change dtype, while
every other unary retains the core FLOAT16/FLOAT32 dtype. Quantized or unsafe
tail contracts are complete no-ops and cannot fall through to the legacy
mutator. The residual-ADD/Reshape tail is unchanged and remains the sole
legacy mode.

Focused coverage includes eight exact NumPy equivalence variants across
FLOAT16/FLOAT32, public/produced sources, and shared/separate Mean axes; all
thirteen unary operators; dynamic height/width/channel signatures; shared
tail-shape cloning; multi-chain execution; thirteen unsafe transactional
no-ops; three compatibility-fallback blockers; mixed direct/side/unary graph
order under the shared 32-rewrite ceiling; and preflight/no-index behavior
with `175 passed in 0.73s`. Static, produced-source, FLOAT16, negative-axis,
commuted-affine, and CAST variants were exactly ModelIR-identical to the
committed legacy helper. Separate equivalent Mean axes are an intentional
improvement over the legacy shared-axis restriction. The related
InstanceNormalization, Pad, Mean, architecture, core, and pass-efficiency
suites passed with `577 passed in 54.16s`; twelve selected direct-builder
characterizations passed with `12 passed in 0.94s`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed in 6.63s`. Ruff, syntax compilation, and `git diff
--check` passed. No Tier corpus conversion was run.

The indexed residual/Reshape InstanceNormalization checkpoint completes the
four-mode migration. It validates the main `Squeeze -> residual ADD -> Reshape
-> post-Transpose` tail together with either a rank-three HWC-to-CHW residual
bridge or an NHWC-to-NCHW bridge followed by Squeeze and an optional retained
unary. The residual bridge is removed only after every shape/signature, dtype,
quantization, public-boundary, fan-out, producer, consumer, and graph-order
contract succeeds. Residual unary CAST may change FLOAT16/FLOAT32 dtype; the
other twelve operators must preserve it.

ADD output fan-out is planned in the same transaction. Its Reshape path moves
to HWC/NHWC, while all other consumer slots share one deterministic
HWC-to-CHW adapter. A compatible INT32/INT64 fixed permutation is reused;
invalid, produced, public, quantized, or collision cases are complete no-ops.
After this migration the former 975-line compatibility helper contains no raw
ModelIR mutation: it is a 60-line dispatcher that tries the four indexed modes
at each pre-Transpose's original graph position and preserves their shared
32-rewrite ceiling.

Focused coverage includes twelve NumPy equivalence variants across all three
residual source forms, FLOAT16/FLOAT32, and public/produced main sources; all
thirteen residual unary operators; nine dynamic-signature combinations; four
fan-out position/repeated-slot variants; multi-chain execution; commuted ADD;
existing adapter reuse; fifteen unsafe transactional no-ops; four
compatibility-fallback blockers; adapter-allocation collision; all four mixed
tail modes under the shared cap; and preflight/no-index behavior with `238
passed in 0.92s`. Ten representative static, produced-source, FLOAT16,
negative-axis, commuted-affine, residual-source, and fan-out variants were
exactly ModelIR-identical to the committed legacy helper. The related
InstanceNormalization, Pad, Mean, architecture, core, and pass-efficiency
suites passed with `640 passed in 54.91s`; twelve selected direct-builder
characterizations passed with `12 passed in 1.00s`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed in 6.92s`. Ruff, syntax compilation, and `git diff
--check` passed. No Tier corpus conversion was run.

Residual-specific fixtures and cases are isolated in
`test_flatbuffer_direct_indexed_instance_norm_residual_layout.py`; the common
ModelIR builder/evaluator remains in the direct-tail module. This keeps the
common characterization file at 1,825 lines and the residual module at 642
lines without duplicating fixture construction.

The focused indexed and architecture tests pass Ruff normally. The changed
legacy characterization file passes with its pre-existing `F401` findings
scoped out, and the lowerer passes with its pre-existing `F401` and `F841`
findings scoped out. Every changed Python file passes `python -m py_compile`,
and `git diff --check` passes. The
immediately preceding DepthToSpace, Pool, dynamic-Pool, simple-alias, and
aligned-scalar checkpoints passed their focused synthetic and ownership
selections.

The indexed post-Transpose-bias InstanceNormalization checkpoint extracts the
rank-four decomposition matcher and generic constant transaction into
`passes/decomposed_instance_norm.py`. Both the established pre/post-tail owner
and the new `passes/instance_norm_post_bias_layout.py` owner now validate the
same exact Mean/SUB/square/variance/epsilon/SQRT/reciprocal/normalize/scale
contract. Epsilon must be finite and nonnegative, the DIV numerator must be
exactly one, all retained tensors must share one unquantized FLOAT16/FLOAT32
contract, and every producer, consumer multiplicity, public boundary,
shape/signature, and graph-order relation is proven before mutation.

The new owner accepts shared or separate positive, negative, or reversed Mean
axes; commuted SUB and affine operands; scalar, NCHW, or already-NHWC scale and
bias constants; public or produced NHWC sources; and dynamic height, width, or
channel signatures. Axes and coefficients are planned by use. Private values
update in place, unrelated consumers receive deterministic clones, and one
constant shared by scale and bias is updated once for both slots. Rejected
candidates do not prune orphan tensors or partially rewrite constants. On
success the Means and SUB consume the original NHWC source, axes become
`[1,2]`, retained core metadata becomes NHWC, the bias ADD consumes the scaled
tensor directly, and only the two boundary Transposes are removed.

The lowerer helper is now a 19-line compatibility dispatcher. In the repeated
normalization recovery loop, the four-tail owner and post-bias owner share one
live `ModelIRGraphIndex`; every late production call supplies the Session
`LayoutState`. The new owner has one bounded graph-order candidate scan, a
32-rewrite ceiling, differential index updates, and no repeated full consumer
map or unbounded fixed-point loop. The existing four-tail characterization
remains unchanged after adopting the common matcher.

Focused coverage includes twenty-four NumPy equivalence variants across
FLOAT16/FLOAT32, public/produced source, shared/separate and positive/negative/
reversed Mean axes, commuted SUB/affine operands, and valid scalar/NCHW-scale/
NHWC-bias forms; three dynamic-signature cases; a public bias-ADD output;
shared changed-constant cloning; one shared scale/bias tensor; legacy
coefficient-layout acceptance; two-chain capped execution; thirty-six unsafe
transactional no-ops; clone-allocation collision; and preflight/no-index
behavior. The post-bias, existing four-tail, compatibility direct-builder, and
ownership selections passed with `310 passed in 1.93s`. The full architecture
suite passed with `176 passed in 52.04s`; thirteen selected InstanceNorm direct-
builder tests passed with `13 passed in 1.42s`; TensorFlow-import-blocked import,
direct conversion, and `-cotof` passed sequentially with `3 passed in 3.97s`.
Scoped Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed InstanceNormalization residual-ADD checkpoint moves the adjacent
dual-branch layout rewrite to
`passes/instance_norm_residual_add_layout.py`. The owner accepts an exact
`NHWC -> NCHW` main branch, the common decomposed InstanceNorm core through
scale and bias, a second `NHWC -> NCHW` residual branch, and a tail ADD with
downstream NCHW consumers. It lifts both branches and the tail ADD to NHWC,
removes both pre-Transposes, and inserts one post-ADD `[0,3,1,2]` adapter that
preserves the original tensor name, NCHW metadata, downstream fan-out, and
repeated consumer slots.

The common helper now plans Mean axes, scale, and bias together before any
mutation. Private constants update in place, shared values receive deterministic
clones, scalar/NCHW/already-NHWC affine forms are handled explicitly, and
existing adapter constants are reused only after their dtype, value, ownership,
and quantization contracts are proven. The owner rejects public boundaries,
duplicate or backward producers, mixed dtypes, quantized normalization tensors,
invalid dynamic signatures, unsafe constant sharing, and all allocation
collisions without changing ModelIR. It uses one differential
`ModelIRGraphIndex`, a graph-order candidate scan, a deterministic 32-rewrite
ceiling, success-only pruning, and Session `LayoutState` synchronization. The
former 475-line lowerer mutator is now a 19-line dispatcher, and the repeated
normalization loop passes its live `residual_graph_index` into the owner.

Focused coverage includes thirty-two FLOAT16/FLOAT32 NumPy-equivalence
variants across public/produced main and residual sources, positive/negative/
reversed Mean axes, commuted affine operands, and alternate downstream
topologies; dynamic height, width, and channel signatures; fan-out and repeated
input slots; shared-constant cloning; valid adapter reuse; multi-chain rewrite
limits; thirty-nine transactional unsafe cases; three allocation-collision
cases; and preflight/no-index behavior. The new owner, adjacent post-bias and
four-tail owners, direct-builder characterizations, and ownership checks passed
with `395 passed in 2.45s`. The full architecture suite passed with
`177 passed in 56.80s`; thirteen selected InstanceNorm direct-builder tests
passed with `13 passed, 742 deselected in 1.47s`; TensorFlow-import-blocked
import, direct conversion, and `-cotof` passed sequentially with
`3 passed in 4.21s`. Scoped Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed residual-MUL/CONCAT InstanceNormalization checkpoint moves the
next three-Transpose tail to
`passes/instance_norm_residual_mul_concat_layout.py`. It validates the direct
decomposed core and bias, the first NCHW-to-NHWC bridge, one same-contract
residual ADD, exactly two tail MUL branches, one multiplicity-preserving
channel-axis CONCAT, and the final NCHW-to-NHWC bridge. The historical helper
name mentions Conv, but the indexed owner intentionally preserves the actual
legacy boundary: the former final-Transpose output may be a public output or
feed any later graph-ordered consumers, with no required Conv operator.

On success, all three Transposes are removed, the normalization and residual
tail run in NHWC, both tail-MUL outputs and CONCAT metadata are permuted, the
CONCAT axis changes from 1 to 3, and CONCAT directly produces the preserved
final output name. Dynamic height, width, and channel signatures, downstream
fan-out, repeated input slots, public outputs, and existing CONCAT option fields
are retained. The common constant planner now accepts additional coefficient
uses and plans both Mean axes, scale, bias, and both tail coefficients as one
transaction. Shared constants update once, unrelated users receive a
deterministic clone, and a late invalid tail coefficient can no longer leave
earlier axes or affine constants partially modified.

The owner requires exact producer, consumer multiplicity, graph-order,
shape/signature, dtype, quantization, typed-permutation, output-renaming, and
public-boundary contracts. It compares CONCAT inputs with `Counter`, uses one
differential `ModelIRGraphIndex`, scans candidates in graph order with a
32-rewrite ceiling, prunes only after success, and synchronizes the Session
`LayoutState`. Its former 501-line lowerer mutator is a 19-line dispatcher. All
four production calls pass LayoutState, and the repeated normalization loop
shares `residual_graph_index` with the preceding residual-ADD owner.

Focused coverage includes thirty-two FLOAT16/FLOAT32 logical-equivalence
variants across direct/produced main and residual sources, shared/separate and
positive/negative/reversed Mean axes, commuted operands, scalar/NCHW/NHWC
coefficients, and both CONCAT orders; three dynamic-signature cases; fan-out
and repeated input slots; public final output; shared-constant cloning; one
coefficient shared by all four affine uses; two-chain capped execution;
fifty-seven transactional unsafe cases; clone collision; and preflight/no-index
behavior. The new owner, adjacent five InstanceNorm owner groups,
direct-builder characterizations, and ownership checks passed with
`497 passed in 2.99s`. The full architecture suite passed with
`178 passed in 54.41s`; thirteen selected InstanceNorm direct-builder tests
passed with `13 passed, 742 deselected in 1.44s`; TensorFlow-import-blocked
import, direct conversion, and `-cotof` passed sequentially with
`3 passed in 4.22s`. Scoped Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed dual-statistics checkpoint moves the next normalization family to
`passes/instance_norm_dual_stats_layout.py`. Audit proved that neither branch
is the standard decomposed InstanceNorm core: one reduces spatial axes and one
reduces all non-batch axes, and both use
`SUB -> square -> Mean -> variance factor -> epsilon -> SQRT ->
DIV(centered,std) -> scale`. The owner therefore keeps a dedicated path matcher
and shares only typed constant, tensor-contract, metadata, and graph-index
utilities. This prevents a superficially similar topology from being assigned
the standard reciprocal/MUL normalization semantics.

The two branches feed blend ADD, gamma MUL, and beta ADD. Spatial axes are
planned together as `[1,2]`; global `[1,2,3]` axes are validated unchanged.
Branch scales and direct gamma/beta constants use one grouped coefficient
transaction, so shared constants update once and unrelated consumers receive a
deterministic clone. Exact `[1,C,1,1]` gamma/beta Reshapes from rank-one or
rank-two vectors are validated through their producer, shape constant, source,
consumer, dtype, quantization, and graph order, then bypassed and removed.
Variance factors and epsilon values must be finite, nonnegative scalar
FLOAT16/FLOAT32 constants.

Direct mode removes the input/output Transposes and gives beta ADD the former
NHWC output name. Residual mode additionally validates and removes an
independent NHWC-to-NCHW residual bridge, lifts the residual ADD to NHWC, and
uses that ADD as the preserved output producer. Its old NCHW output contract is
validated and permuted before rename, preventing dynamic-axis contamination.
The historical function name mentions Resize, but no Resize is required by the
legacy boundary; later ordered consumers, fan-out, and repeated input slots are
preserved.

The owner validates complete producer/consumer multiplicity, dependency order,
public boundaries, shape/signature, dtype, quantization, typed constants,
coefficient ownership, optional Reshape removal, and output rename before any
mutation. It uses a graph-order candidate scan, one differential
`ModelIRGraphIndex`, a 32-rewrite ceiling, success-only pruning, and Session
`LayoutState` synchronization. The former 712-line lowerer mutator is now a
19-line dispatcher; all four production calls pass LayoutState and the repeated
normalization loop shares `residual_graph_index` with the preceding owners.

Focused coverage includes forty-eight FLOAT16/FLOAT32 numerical-equivalence
variants across direct, residual-input, and produced-residual tails; direct and
produced main sources; shared/separate positive, negative, and arbitrarily
permuted axes; commuted operands; scalar/NCHW scales; direct and vector-Reshape
gamma/beta; six dynamic-signature modes; downstream fan-out and repeated slots
without
Resize; already-NHWC coefficients; shared-axis cloning; one coefficient shared
by all four affine sites; capped multi-chain execution; 143 transactional
unsafe contracts; clone collision; and preflight/no-index behavior. The new
owner and all indexed InstanceNorm owner groups passed with
`792 passed in 2.40s`. The full architecture suite passed with
`179 passed in 53.94s`; thirteen selected InstanceNorm direct-builder tests
passed with `13 passed, 742 deselected in 1.51s`; TensorFlow-import-blocked
import, direct conversion, and `-cotof` passed sequentially with
`3 passed in 4.45s`. Scoped Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed affine-chain checkpoint moves
`_optimize_fold_mul_add_mul_affine_chains` to
`passes/affine_chain_fold.py`. The legacy helper distinguished constants only
by the presence of tensor data, rebuilt full producer/consumer maps inside an
unbounded loop, compared intermediate users as sets, mutated shared constants
before all checks were known, and copied the removed intermediate ADD tensor's
metadata onto the preserved final output. It did not validate dtype,
quantization, fused activation, public boundaries, producer order, constant
provenance, or the original and folded broadcast contracts.

The new owner accepts finite, non-variable FLOAT16, FLOAT32, and FLOAT64
constants with matching array dtype and exact static tensor metadata. All data
and output tensors must share that dtype and be unquantized, and all three
binary operators must have `NONE` fused activation. Original static broadcast
shapes and dynamic signatures are checked across all three operators; the two
folded broadcasts must independently reproduce the final output contract.
When the removed final MUL introduced a broadcast expansion, the surviving
first-MUL intermediate receives the correctly expanded shape and signature.
The final output tensor itself remains untouched, retaining its dtype,
quantization, shape, signature, layouts, and ONNX provenance.

The first-MUL and ADD constant roles are grouped before apply. Constants shared
inside those roles are updated once; sharing with the removed final MUL is an
allowed internal use. Any unrelated consumer receives one deterministic
`_folded` clone while the original constant is preserved. Produced, public,
variable, quantized, non-finite, mismatched, colliding, or incompatible
constants reject the complete plan. Exact producer uniqueness, consumer
multiplicity, graph order, intermediate privacy, source resolution, downstream
fan-out, repeated final-output slots, and output rename are proven with one
`ModelIRGraphIndex`. The plan is resolved again immediately before apply, the
candidate scan is graph ordered and capped at 32 rewrites, pruning is
success-only, and LayoutState is synchronized. The former 219-line lowerer
helper is now a 17-line dispatcher; all three production calls pass
LayoutState.

Focused coverage contains forty-eight FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across all operand orders and scalar/channelwise
broadcasts; dynamic signatures; final broadcast expansion; constants shared
within the chain or with unrelated consumers; downstream fan-out and repeated
slots; candidate-only and capped execution; thirty-three transactional unsafe
contracts; clone collision; and no-index preflight. The focused owner plus the
pre-existing direct-builder characterization passed with `91 passed in 0.57s`.
The full architecture suite passed with `180 passed in 54.55s`; eighteen
selected affine direct-builder tests passed with
`18 passed, 737 deselected in 1.45s`; TensorFlow-import-blocked import, direct
conversion, and `-cotof` passed sequentially with `3 passed in 4.24s`. Scoped
Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed affine pre/post checkpoint moves
`_optimize_transpose_mul_add_const_prepost_nhwc_chains` to
`passes/affine_prepost_layout.py`. The legacy helper rebuilt full maps in an
unbounded loop, contained a permanently disabled PRELU branch and unused
`valid_posts`, and rotated rank-four constants heuristically up to three times.
It preflighted constants separately but then mutated them sequentially, so a
late failure or clone-name interaction could leave a partial rewrite. It also
permuted the old ADD output metadata, copied that metadata to the canonical
post output, and permuted the canonical tensor again, producing a double-
transpose metadata path.

The new owner matches from a MUL candidate and proves one exact private
NHWC-to-NCHW pre adapter, the MUL/ADD chain, and every private inverse post
adapter. The pre Transpose is removed only when that MUL is its last consumer;
other pre fan-out remains intact. The ADD output may have multiple post
adapters but no legacy consumers. Their downstream uses are redirected to the
first graph-ordered post output with exact multiplicity, preserving fan-out and
repeated slots before all post adapters are removed. The canonical post tensor
is not overwritten. Its shape, signature, logical layout, physical layout,
dtype, quantization, and provenance remain authoritative; the surviving MUL
intermediate adopts that contract.

Finite FLOAT16/FLOAT32/FLOAT64 scalar and rank-four constants are supported.
Raw NCHW channel, spatial, and full constants rotate once to NHWC, while
already-NHWC constants remain stable for idempotent recovery. Known layout
annotations disambiguate orientation. If direct and rotated non-invariant data
are both compatible because axes have equal lengths, the candidate is rejected
instead of guessing. MUL and ADD roles are grouped into one plan, so a shared
constant updates once and unrelated consumers receive deterministic `_nhwc`
clones. Produced, public, variable, quantized, non-finite, wrongly typed or
shaped constants reject the complete transaction.

Typed private permutation vectors, rank-four shape/signature permutations,
same floating dtype, no quantization, no fused activation, unique producers,
exact consumer multiplicity, dependency order, private intermediates, source
resolution, alias layouts, and final renaming are all checked with one
`ModelIRGraphIndex`. The plan is resolved again immediately before apply. The
candidate scan is graph ordered and capped at 32 rewrites; pruning and
LayoutState maintenance are explicit. The former 409-line helper is now a
17-line dispatcher, and all seven production calls pass LayoutState.

Focused coverage contains forty-eight FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across all operand orders and scalar, channel,
spatial, and full constants; dynamic signatures; already-NHWC idempotence;
canonical layout propagation; retained pre fan-out; multi-post alias merging
and repeated slots;
constants shared inside and outside the chain; candidate-only and capped
execution; forty-four transactional unsafe contracts; equal-axis ambiguity;
clone collision; and no-index preflight. The focused owner plus two existing
direct-builder characterizations passed with `104 passed in 0.53s`. The full
architecture suite passed with `181 passed in 50.32s`; three selected related
direct-builder tests passed with `3 passed, 752 deselected in 0.45s`;
TensorFlow-import-blocked import, direct conversion, and `-cotof` passed
sequentially with `3 passed in 4.01s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed affine post-ADD checkpoint moves
`_optimize_transpose_mul_posttranspose_add_nhwc_chains` to
`passes/affine_post_add_layout.py`. The legacy helper rebuilt full producer and
consumer maps inside an unbounded loop, collapsed sole-consumer multiplicity
through a set, mutated or cloned the MUL constant during matching, and
permuted the surviving MUL output metadata heuristically. It did not validate
dtype, quantization, fused activation, public boundaries, producer uniqueness,
constant provenance, exact graph order, or the complete post-ADD fan-out
before mutation.

The new owner resolves from a graph-ordered MUL candidate. It proves a typed
private NHWC-to-NCHW pre adapter, the MUL output's one exact typed inverse post
adapter, and every consumer of the private post output as a plain ADD with a
finite same-dtype scalar or exact `[1,1,1,C]` side constant. Multiple ADD tails
and their downstream repeated slots remain intact. The pre Transpose is kept
when another NCHW branch uses it. Otherwise both adapters are removed, all ADD
tails consume the surviving MUL output, and that output adopts the canonical
post tensor's shape, dynamic signature, logical layout, and physical layout.

MUL constants share the affine pre/post owner's finite FLOAT16/FLOAT32/FLOAT64
orientation contract. Scalar, raw NCHW channel/spatial/full, already-NHWC, and
legacy direct non-rank-four forms are retained. Ambiguous equal-axis
non-invariant rank-four tensors are rejected. A changed constant with an
unrelated consumer receives a deterministic `_nhwc` clone. The complete plan
is resolved again before apply, and clone names, input slots, and operator
removals are preflighted before any mutation. One differential index, a
32-rewrite ceiling, success-only pruning, and LayoutState synchronization
replace the legacy loop. The former 278-line helper is a 17-line dispatcher;
all four production calls pass LayoutState. The Pad compatibility wrapper
continues to dispatch only to its independent `passes/pad_layout.py` owner.

Focused coverage contains twenty-four FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across both operand orders and scalar, channel,
spatial, and full MUL constants; dynamic signatures; two ADD tails; canonical
layout propagation; legacy already-NHWC/vector constant modes; retained pre
fan-out; unrelated constant cloning; candidate-only and capped execution;
twenty-three transactional unsafe contracts; equal-axis ambiguity; clone
collision; and no-index preflight. The focused owner plus two existing direct-
builder characterizations passed with `56 passed`; the full architecture suite
passed with `182 passed in 51.86s`; the selected direct-builder tests passed
with `2 passed, 753 deselected in 0.46s`; and TensorFlow-import-blocked direct,
default, and `-cotof` conversion passed sequentially with
`3 passed, 8 deselected in 3.57s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed SiNet Shuffle residual checkpoint moves
`_optimize_sinet_shuffle_residual_transpose_chains` to
`passes/sinet_shuffle_residual_layout.py`. The legacy helper rebuilt full
producer and consumer maps in an unbounded loop and matched forward and
backward from the second post-Transpose with incomplete ownership checks. It
collapsed several consumer checks through sets, did not validate graph order,
dtype, quantization, fused activation, public intermediates, constant
provenance, concrete/dynamic Concat contracts, or downstream order, and
rotated/cloned six constants sequentially before the complete candidate was
known. Shared external constants could receive multiple redundant clones. It
also permuted intermediate and canonical post metadata heuristically.

The new owner roots at the terminal post-Transpose and resolves all thirteen
operators: three NHWC-to-NCHW input adapters, the first residual ADD/MUL/ADD/
PReLU, its side post adapter, the channel Concat, the second MUL/ADD/PReLU, and
the final post adapter. Every private edge has one unique producer, exact
consumer multiplicity, and valid dependency order. The first PReLU has exactly
the intended post and Concat branches; the second PReLU has exactly its final
post. Later consumers of the two post outputs, including fan-out and repeated
input slots, remain unchanged. The Concat input order and commuted ADD/MUL
operands are preserved while its axis moves from NCHW channel 1 to NHWC
channel 3.

Rank-four concrete shapes and dynamic signatures are proven through both
residual branches and the Concat. Unknown non-channel dimensions propagate
conservatively; channel signatures sum only when known. All data tensors share
one unquantized FLOAT16/FLOAT32/FLOAT64 dtype. Five typed private permutation
constants and NONE fused activations are required. Six finite scalar or exact
NCHW/NHWC channel constants cover both affine/PReLU stages. Roles sharing one
constant are grouped, so one update or one deterministic `_nhwc` clone serves
all uses, including a scalar shared across both stages. Produced, public,
variable, quantized, non-finite, wrongly typed, mismatched, or colliding
constants reject the complete plan.

Both post-Transpose tensors remain unchanged and authoritative. The first
stage intermediates adopt the first post tensor's exact shape, signature, and
layouts; the Concat and second-stage intermediates adopt the second post
contract. Both PReLUs produce the canonical names directly, after which all
five adapters are removed differentially. The plan is re-resolved before
apply, and clone names, mutation indices, output tensors, and removal indices
are preflighted before the first write. Candidate traversal is graph ordered
and capped at 32 rewrites; pruning and LayoutState synchronization occur only
after success. The former 482-line helper is a 17-line dispatcher and its one
production call supplies LayoutState.

Focused coverage contains twenty-four FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across scalar/raw constants, commuted operands and
Concat inputs, and both legal post/Concat orders; dynamic signatures;
canonical layout propagation; two legacy already-NHWC modes; constants shared
within one stage or as one scalar across both stages; one-clone external
sharing; repeated downstream slots; candidate-only and capped execution;
fifty-eight transactional unsafe contracts; clone collision; no-index
preflight; differential index validation; and LayoutState validation. The
focused suite passed with `90 passed in 0.51s`; the full architecture suite
passed with `183 passed in 53.22s`; the selected adjacent SiNet direct-builder
characterization passed with `1 passed, 754 deselected in 0.48s`; and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed, 8 deselected in 3.65s`. Scoped Ruff, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The paired post-MUL SiNet checkpoint moves
`_optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains` to the same
`passes/sinet_shuffle_residual_layout.py` owner. Its first nine operators are
identical to the full-tail island, so both variants now use one
`_resolve_prefix` implementation for the three input adapters, first
ADD/MUL/ADD/PReLU stage, side post adapter, channel Concat, shape/signature
proof, three constant roles, canonical first-post metadata, public boundaries,
fan-out, and graph order. This removes the risk that fixes to the formerly
duplicated residual prefix diverge between the two legacy helpers.

The variant-specific tail proves
`Concat(NCHW) -> MUL -> post-MUL Transpose(NHWC) -> ADD -> PReLU`. The MUL and
post output are private and have exact producer/consumer multiplicity; the ADD
has one exact PReLU consumer. The final PReLU output may remain a public output
or keep later graph-ordered fan-out. Concrete and dynamic contracts require
the MUL output to equal the NCHW Concat, and the post, ADD, and PReLU outputs to
share the exact NHWC permutation. All tail tensors share the prefix floating
dtype and are unquantized. Fused activations, duplicate producers, invalid
order, partial fan-out, public intermediates, and stale metadata reject the
complete plan.

The shared six-role constant transaction rotates or retains the first-stage
constants, the second MUL constant, and the already-NHWC ADD/PReLU constants
together. Sharing, external clones, constant provenance, dtype, finiteness,
shape, signature, quantization, collision, and variable state use the same
contract as the full-tail owner. The MUL produces the existing post-Transpose
name directly. The post tensor, ADD output, final PReLU output, and their
provenance remain untouched; only the Concat intermediate adopts the canonical
post contract. Both plans are re-resolved before apply, and all mutations and
five removals are preflighted before the first write.

Variant-focused coverage adds twenty-four FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across scalar/mixed raw constants, commuted operands
and Concat inputs, and both legal post/Concat orders; two legacy raw tail
constant cases; external MUL-constant cloning; candidate-only and capped
execution; twenty-seven transactional unsafe tail contracts; clone collision;
and no-index preflight. The combined two-owner focused suite plus the existing
direct-builder characterization passed with `148 passed in 0.88s`; the full
architecture suite passed with `183 passed in 50.63s`; the selected SiNet
direct-builder characterization passed with
`1 passed, 754 deselected in 0.47s`; and TensorFlow-import-blocked direct,
default, and `-cotof` conversion passed sequentially with
`3 passed, 8 deselected in 3.54s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed late-residual checkpoint moves
`_optimize_sinet_late_residual_pre_add_mul_add_prelu_chains` into the same
SiNet residual owner. The semantic candidate is two private
NHWC-to-NCHW adapters feeding `ADD -> MUL -> ADD -> PReLU`. Exactly one NHWC
source has a channel-last Concat producer. The PReLU output feeds one private
NCHW-to-NHWC adapter for a Conv2D or DepthwiseConv2D branch and one or more
later legacy NCHW consumers. The old fixed 40-by-40 predicate is gone; rank-
four concrete shapes and dynamic signatures must prove each permutation and
all affine intermediates must share one unquantized FLOAT16/FLOAT32/FLOAT64
contract.

The rewrite removes the two input adapters and lifts the affine/PReLU island
to NHWC. PReLU produces the existing canonical post tensor directly. The post
adapter is inverted in place and now produces the former PReLU tensor name, so
legacy consumers and repeated slots remain unchanged. Legacy consumers must
be graph ordered after that retained adapter; an earlier independent branch is
rejected instead of creating a producer-after-consumer edge. The canonical
post tensor remains authoritative and is never double-permuted. Only the
ADD/MUL/ADD intermediates adopt its exact shape, signature, and layouts; the
legacy output tensor retains its original NCHW metadata.

The three floating constants are grouped by tensor identity. Each must be a
finite, same-dtype, private constant that broadcasts in the original NCHW
graph; non-scalars are rotated and must also broadcast in the target NHWC
graph. This safely handles scalar and raw channel constants and retains the
legacy already-oriented rank-four case only when its actual axes make both
graphs valid. Unrelated consumers receive one deterministic clone. The
retained permutation constant participates in the same transaction and is
cloned when another Transpose still needs the original permutation.

The plan is re-resolved immediately before apply. Constant clones, mutation
indices, metadata targets, output-name swaps, and both adapter removals are
preflighted before the first write. One differential index, deterministic
candidate order, a 32-rewrite ceiling, success-only pruning, and LayoutState
synchronization replace the legacy full-map `while True` loop. The lowerer now
contains a 17-line dispatcher and the production call supplies the Session
LayoutState.

Focused coverage now passes with `207 passed in 0.69s`. It includes thirty-six
FLOAT16/FLOAT32/FLOAT64 numerical-equivalence combinations across both operand
orders, both downstream convolution families, scalar/raw constants, and the
formerly size-specific rank-four constant case on a non-40 spatial contract;
dynamic signatures; canonical, public, and repeated-slot legacy output
preservation; shared-role and external-use constant cloning; shared
permutation cloning; ambiguous oriented-constant rejection; candidate-only and
capped execution; fifteen transactional unsafe contracts; earlier legacy
consumer rejection; stale-plan revalidation; clone collision; no-index
preflight; differential-index validation; and LayoutState validation. The full
architecture suite passed with `183 passed in 50.11s`; the one sequential real
SiNet direct-builder characterization passed with `1 passed in 2.62s`; and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed with
`3 passed, 8 deselected in 3.58s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed deep-skip checkpoint moves
`_optimize_sinet_deep_skip_concat_resize_affine_tail_chains` to the dedicated
`passes/sinet_deep_skip_layout.py` owner. Matching is split into three bounded,
side-effect-free resolvers: the terminal Concat/MUL/post-Transpose/ADD/PReLU
tail, the central ADD/MUL/ADD/PReLU residual stage, and the Resize/affine dual
branch feeding the first Concat. The combined plan proves all sixteen semantic
operators and four removable adapters before any mutation; a partially
matched stage is never applied.

The Resize MUL/ADD, both channel Concats, the central residual stage, and the
terminal MUL are lifted to NHWC. The terminal ADD/PReLU retains its canonical
NHWC tensors. The central residual source may be an already annotated NHWC
tensor, as in the current SiNet graph, or an NCHW tensor with one explicit
earlier NCHW-to-NHWC adapter. Relational concrete shape, dynamic signature,
logical/physical layout, and typed permutation contracts replace the former
40-by-40 heuristic. The four removed adapters are the Resize input adapter,
the independent branch adapter, the deep-skip adapter, and the terminal post
adapter; the explicit central adapter, when present, remains available to its
existing branch and becomes the residual input.

Concat contracts are derived independently on original NCHW axis 1 and target
NHWC axis 3. Producer uniqueness, exact consumer multiplicity, dependency
order, private intermediates, floating dtype, quantization, fused activation,
Resize provenance, and canonical output metadata are validated across the
whole island. Six NCHW-side affine/PReLU constants must broadcast before and
after an explicit axis rotation. The already-NHWC terminal ADD/PReLU constants
use the canonical post contract. All eight roles are grouped by tensor
identity; unrelated consumers receive one deterministic clone and conflicting
shared orientations reject the plan.

Constant cloning/rewiring and metadata updates are now shared helpers used by
the full-tail, post-MUL, late-residual, and deep-skip SiNet owners. The deep-
skip plan is re-resolved immediately before apply, then clone names, four
removals, five data-input changes, both Concat axes, and every metadata target
are preflighted. One differential index, graph-ordered candidates, a 32-
rewrite ceiling, success-only pruning, and Session LayoutState synchronization
replace the full-map `while True` loop. The lowerer helper is now a 17-line
dispatcher and its sole production call supplies LayoutState.

Deep-skip focused coverage passed with `54 passed in 0.50s`; the combined
SiNet indexed suites passed with `261 passed in 0.79s`. Coverage includes
twenty-four FLOAT16/FLOAT32/FLOAT64 numerical-equivalence combinations across
both operand/Concat orders, both Resize families, scalar/raw constants, and
the explicit central adapter; four direct-NHWC correction/reference cases;
dynamic signatures; shared external constant cloning; candidate-only and
capped execution; twenty transactional unsafe contracts; explicit channel-
last selection; stale-plan revalidation; clone collision; no-index preflight;
differential-index validation; and LayoutState validation.

The sequential real SiNet integration passed with `1 passed in 2.51s`.
Legacy and indexed owners produced identical 449,824-byte float32 artifacts
with SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and identical 253,452-byte float16 artifacts with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The sequential real-model `-cotof` run reported
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
The full architecture suite passed with `184 passed in 49.18s`, and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed with
`3 passed, 8 deselected in 3.66s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed pre-ADD fan-out checkpoint moves
`_optimize_sinet_deep_skip_pre_add_concat_prelu_fanout_chains` to
`passes/sinet_preadd_fanout_layout.py`. Its distinct prefix is one
channel-last Concat followed by a private NHWC-to-NCHW adapter and one direct
NCHW input with exactly one earlier NCHW-to-NHWC sibling adapter. Those two
NCHW values feed the shared strict
`ADD -> MUL -> ADD -> PReLU -> Transpose -> Conv` tail. The PReLU also has one
or more later legacy NCHW consumers. The common terminal resolver now lives in
the existing SiNet residual owner, so downstream Conv ownership, affine
producer/consumer order, fused activation, and legacy fan-out semantics cannot
drift between the late-residual and pre-ADD variants.

The rewrite removes only the Concat-side adapter. The ADD consumes the
canonical NHWC Concat output and the already-existing sibling adapter output.
The affine/PReLU tail is lifted to NHWC and PReLU produces the canonical post
tensor directly. The terminal post adapter is inverted and produces the old
PReLU NCHW name, preserving all later legacy consumers and repeated input
slots. The direct sibling adapter is retained for its original branch. No
model name or fixed spatial dimension is used: relational rank-four concrete
shapes, dynamic signatures, logical/physical layout, typed permutations, and
the channel-last Concat axis prove the topology.

All activation tensors must share one unquantized FLOAT16/FLOAT32/FLOAT64
contract. Public intermediates, duplicate producers, unexpected direct-source
fan-out, a missing or late sibling adapter, invalid dependency order, wrong
Concat axis, non-Conv downstream ownership, or a missing/early legacy branch
reject the complete plan. The MUL, ADD, and PReLU constants must broadcast in
the original NCHW graph and after an explicit NHWC rotation. They are grouped
with the inverted terminal permutation; unrelated consumers receive one
deterministic clone. Candidate resolution is repeated before apply, followed
by collision, mutation-index, metadata-target, output-swap, and removal
preflight before the first write.

The focused pre-ADD suite passed with `47 passed in 0.47s`; combined with the
existing indexed SiNet residual and deep-skip suites it passed with
`308 passed in 0.92s`. Coverage includes twenty-four numerical-equivalence
combinations across FLOAT16/FLOAT32/FLOAT64, scalar/raw constants, both operand
and Concat orders, and both convolution families; idempotence; differential
index and LayoutState validation; shared permutation and external-constant
cloning; public repeated legacy slots; candidate-only and capped execution;
sixteen transactional unsafe contracts; earlier legacy rejection; stale-plan
revalidation; clone collision; and no-index preflight.

The sequential real SiNet integration passed with `1 passed in 2.55s`. The
current owner produced the same 449,824-byte float32 artifact with SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and the same 253,452-byte float16 artifact with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`
as the preceding checkpoint. Therefore the already recorded sequential SiNet
`-cotof` evidence remains applicable: `max_abs=2.57205e-09`,
`rmse=9.15391e-11`, `cosine=1`, and `pass=True`. The full architecture suite
passed with `185 passed in 49.37s`; TensorFlow-import-blocked direct, default,
and `-cotof` conversion passed sequentially with
`3 passed, 8 deselected in 3.57s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed dual-Resize checkpoint moves both
`_optimize_sinet_dual_resize_affine_transpose_chains` and
`_optimize_sinet_deep_skip_dual_resize_affine_transpose_chains` to the shared
`passes/sinet_dual_resize_layout.py` owner. Both former raw mutators prove the
same two `Resize -> Transpose -> MUL -> ADD` branches, NCHW channel Concat,
residual ADD, terminal MUL/ADD/PReLU, and post-Transpose tail. Their only
semantic difference is now an explicit residual mode. The direct mode owns
and removes one private NHWC-to-NCHW adapter. The sibling mode requires one
earlier NCHW-to-NHWC adapter with exact residual fan-out and retains it for its
existing branch.

Each Resize output is the authoritative branch-local NHWC tensor. Its MUL and
ADD outputs adopt that exact concrete shape, dynamic signature, logical
layout, and physical layout. The two branches independently derive the
original axis-1 NCHW Concat and target axis-3 NHWC Concat. The first
graph-ordered post output is the authoritative merged contract and remains
unchanged; the Concat and terminal ADD/MUL/ADD intermediates adopt it. This
replaces the duplicated metadata permutation/copy sequences and removes the
former fixed 40-by-40 shape guard.

Four branch affine constants and three terminal affine/PReLU constants are
validated as finite same-dtype broadcasts in both the original NCHW graph and
the explicitly rotated NHWC graph. Identity-grouped roles are updated once,
and unrelated consumers receive deterministic clones. Both Resize families,
FLOAT16/FLOAT32/FLOAT64, scalar/raw constants, operand and Concat order,
typed permutations, producer uniqueness, exact consumer multiplicity,
dependency order, public boundaries, quantization, fused activation, and
dynamic signatures are part of the plan contract.

Equivalent post aliases are merged into the first canonical output while
downstream repeated slots are retained. Direct-mode legacy NCHW consumers
receive one topologically inserted inverse adapter and therefore preserve
local numerical semantics. Sibling mode retains the established ordered
pipeline boundary and rewires later compatibility consumers to the canonical
NHWC tensor for following SiNet recovery passes. The complete plan is
re-resolved before apply; clone names, mutation/removal indices, input slots,
metadata targets, alias rewrites, and optional adapter insertion are
preflighted before the first write. One differential index and a bounded
32-candidate rewrite ceiling replace both full-map `while True` loops. The
former 505-line and 503-line helpers are now 17-line dispatchers and both
production calls receive Session LayoutState.

The focused shared-owner suite passed with `102 passed in 0.60s`; combined
with the preceding indexed SiNet suites it passed with
`410 passed in 1.24s`. Coverage includes forty-eight numerical-equivalence
combinations across both residual modes, all three floating dtypes, both
Resize families, scalar/raw constants, and commuted operands/Concat inputs;
direct legacy inverse-adapter equivalence; sibling repeated-slot compatibility
rewiring; post-alias merging; external constant cloning; mode isolation;
candidate-only and capped execution; thirty-six shared transactional unsafe
contracts plus sibling fan-out/order guards; earlier legacy rejection;
stale-plan revalidation; clone collision; no-index preflight; differential
index validation; and LayoutState validation.

The existing direct dual-Resize characterization and the sequential real
SiNet integration passed together with `2 passed in 2.59s`. During the real
conversion, the direct owner retained the established two one-candidate
applications and the later sibling owner retained its two-candidate
application. The indexed result is the same 449,824-byte float32 artifact with
SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and the same 253,452-byte float16 artifact with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The byte identity preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
The full architecture suite passed with `186 passed in 50.01s`;
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed, 8 deselected in 3.61s`. Scoped Ruff, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed shared-post checkpoint moves
`_optimize_sinet_shared_post_prelu_transpose_fanout_chains` to
`passes/sinet_shared_post_layout.py`. The complete candidate is two private
NHWC-to-NCHW adapters feeding an ADD/MUL/ADD/PReLU tail and one terminal
NCHW-to-NHWC adapter. Exactly one source must be produced by a plain
channel-last Concat. The canonical post tensor must be consumed as the data
input of at least one Conv or DepthwiseConv and by at least one plain ADD;
unsupported consumer roles reject the plan.

The canonical post tensor supplies the exact concrete NHWC shape, dynamic
signature, logical layout, and physical layout. Both input sources must match
that contract, and every internal NCHW tensor must match its exact
permutation. The rewrite removes the two input adapters and terminal adapter,
connects the first ADD directly to both NHWC sources, and makes PReLU produce
the existing post tensor. Consumer names and repeated ADD slots are preserved,
and only the three affine intermediates adopt canonical metadata. This removes
the former 40-by-40 heuristic without broadening the semantic topology.

Typed permutation constants, unique producers, exact fan-out and dependency
order, public boundaries, fused activation, dtype, quantization, layout, and
Concat/Conv input roles are proven before mutation. FLOAT16/FLOAT32/FLOAT64
MUL, ADD, and PReLU constants must broadcast in NCHW and after explicit NHWC
rotation. Shared roles are grouped by tensor identity, and unrelated consumers
receive deterministic clones. Candidate resolution is repeated immediately
before apply, followed by collision, removal-index, consumer-index, input,
output, and metadata-target preflight. One differential graph index, ordered
candidates, a 32-rewrite ceiling, success-only pruning, and LayoutState
synchronization replace the full-map fixed-point loop.

Focused shared-post coverage passed with `46 passed in 0.48s`; combined with
the preceding indexed SiNet suites it passed with `456 passed in 1.27s`.
Coverage includes twenty-four numerical-equivalence combinations across all
three floating dtypes, scalar/raw constants, both operand and Concat orders,
and both convolution families; non-40 spatial shapes and dynamic signatures;
idempotence; repeated post-ADD slots; external constant cloning; candidate-only
and capped execution; seventeen transactional unsafe contracts; stale-plan
revalidation; no-index preflight; differential-index validation; and
LayoutState validation.

The sequential real SiNet integration passed. The indexed result remains the
same 449,824-byte float32 artifact with SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and the same 253,452-byte float16 artifact with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The byte identity preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
The full architecture suite passed with `187 passed in 48.16s`, and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed in 3.64s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed Concat/Resize affine checkpoint moves
`_optimize_sinet_concat_resize_affine_transpose_chains` to
`passes/sinet_concat_resize_layout.py`. Its three input adapters carry the
merged residual source, an independent Concat branch, and a bilinear or
nearest-neighbor Resize result into NCHW. The Resize branch has a private
MUL/ADD affine tail. Both branch results feed a plain axis-1 Concat followed by
the residual ADD/MUL/ADD/PReLU tail and one or more NCHW-to-NHWC post adapters.

The Resize output is authoritative for the branch-local NHWC shape, dynamic
signature, and layout. The first graph-ordered post output is authoritative
for the merged NHWC contract. Independent and affine branch contracts derive
both the original NCHW axis-1 Concat and target NHWC axis-3 Concat; the
residual source and tail must match those merged contracts exactly. The
rewrite removes the three input adapters and every equivalent post adapter,
changes the Concat axis to 3, reconnects the two canonical branch sources,
and makes PReLU produce the canonical post tensor directly.

Additional post outputs are merged into the first canonical tensor while all
repeated downstream slots remain intact. Later consumers of the former PReLU
NCHW tensor receive one inverse adapter inserted before their first use.
FLOAT16/FLOAT32/FLOAT64 activation tensors must be unquantized and share the
same dtype. Typed permutations, Resize provenance, unique producers, exact
fan-out, dependency order, public boundaries, fused activation, shape,
signature, and layout are validated before mutation. Two branch constants and
three merged-tail constants must broadcast in the original NCHW graph and
after explicit NHWC rotation; unrelated consumers receive deterministic
clones.

The complete plan is resolved again immediately before apply, followed by
clone-name, mutation/removal-index, alias-slot, metadata-target, and legacy-
adapter preflight. One differential graph index, graph-ordered candidates, a
32-rewrite ceiling, success-only pruning, and LayoutState synchronization
replace the full-map fixed-point loop. Both existing production call sites
retain their exact order and now supply Session LayoutState.

Focused Concat/Resize coverage passed with `58 passed in 0.50s`; combined with
the preceding indexed SiNet suites it passed with `514 passed in 1.30s`.
Coverage includes twenty-four numerical-equivalence combinations across all
three floating dtypes, scalar/raw constants, both operand and Concat orders,
and both Resize families; non-40 spatial shapes and dynamic signatures;
idempotence; post-alias merging; repeated alias and legacy slots; no-legacy
operation; external constant cloning; candidate-only and capped execution;
twenty-seven transactional unsafe contracts; stale-plan revalidation; clone
collision; no-index preflight; differential-index validation; and LayoutState
validation.

The sequential real SiNet integration passed. The indexed result remains the
same 449,824-byte float32 artifact with SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and the same 253,452-byte float16 artifact with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The byte identity preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
The full architecture suite passed with `188 passed in 48.25s`, and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed in 3.68s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed two-Concat tail checkpoint moves
`_optimize_sinet_concat_resize_affine_tail_concat_transpose_chains` to
`passes/sinet_tail_concat_layout.py`. It reuses the exact adapter and
Resize-affine branch resolvers from the preceding owner. The first stage
combines an independent branch and one Resize/MUL/ADD branch with a same-width
residual, then applies MUL/ADD/PReLU. A second channel Concat combines that
result with an independent skip adapter before the final MUL/ADD/PReLU and
post adapter.

The first residual source supplies the authoritative NHWC contract for the
first stage, while the first graph-ordered post output supplies the merged
second-stage contract. Both original NCHW axis-1 Concats and target NHWC
axis-3 Concats are derived from the concrete branch shapes and dynamic
signatures. The rewrite removes four input adapters and every equivalent post
adapter, reconnects the canonical sources, changes both axes to 3, and makes
the final PReLU produce the canonical post tensor directly.

Post aliases retain every repeated downstream slot. Later consumers of the
former final NCHW PReLU output receive one inverse adapter inserted before
their first use. Eight constants are grouped across three orientation domains:
two Resize-branch constants, three first-stage MUL/ADD/PReLU constants, and
three final-stage constants. Each must be a finite same-dtype broadcast before
and after explicit NHWC rotation; unrelated consumers receive deterministic
clones. Typed permutations, exact Resize provenance, producer uniqueness,
consumer multiplicity, dependency order, public boundaries, fused activation,
FLOAT16/FLOAT32/FLOAT64 dtype, quantization, layout, rank-four shape, and
signature are validated before mutation.

The complete plan is re-resolved before apply, followed by clone-name,
mutation/removal-index, alias-slot, metadata-target, and legacy-adapter
preflight. One differential graph index, graph-ordered candidates, a
32-rewrite ceiling, success-only pruning, and LayoutState synchronization
replace the full-map fixed-point loop. The sole production call retains its
ordered recovery position and now supplies Session LayoutState.

Focused two-Concat coverage passed with `52 passed in 0.50s`; combined with
the preceding indexed SiNet suites it passed with `566 passed in 1.42s`.
Coverage includes twenty-four numerical-equivalence combinations across all
three floating dtypes, scalar/raw constants, both operand/Concat orders, and
both Resize families; non-40 spatial shapes and dynamic signatures;
idempotence; post aliases; repeated alias and legacy slots; no-legacy
operation; external constant cloning; candidate-only and capped execution;
twenty-three transactional unsafe contracts; stale-plan revalidation;
no-index preflight; differential-index validation; and LayoutState validation.

The old and indexed owners both match zero candidates in the current real
SiNet ordered pipeline; this dormant-path fact was measured before extraction.
The sequential real integration nevertheless remained byte-identical: the
449,824-byte float32 artifact has SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`,
and the 253,452-byte float16 artifact has SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The byte identity preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
The full architecture suite passed with `189 passed in 48.07s`, and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed in 3.67s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed Softmax-mask checkpoint moves
`_optimize_sinet_softmax_mask_residual_nhwc_tail_chains` to
`passes/sinet_softmax_mask_layout.py`. The active real-model island has two
private NHWC-to-NCHW input adapters. Its main MUL/ADD result enters a
`[0,3,2,1] -> Softmax(axis=3) -> [0,3,2,1]` wrapper, then
ReduceMax/SUB/Reshape/MUL forms the mask applied to the side PReLU branch
before the final residual ADD and post adapter.

The first graph-ordered post output supplies the authoritative NHWC contract.
All NCHW, NWHC, reduced-rank, and singleton-channel mask shapes and dynamic
signatures are derived from it. The rewrite removes both input adapters, both
Softmax wrapper adapters, and every equivalent terminal adapter. Softmax
directly consumes the main NHWC affine result and produces the former
soft-back tensor, ReduceMax axis `[1]` becomes `[3]`, and Reshape target
`[N,1,H,W]` becomes `[N,H,W,1]` without a fixed spatial size.

The main MUL/ADD constants, side PReLU alpha, and mask expansion constant must
be finite same-dtype broadcasts both before and after explicit rotation. The
ReduceMax axis and Reshape target constants preserve their original INT32 or
INT64 dtype. All six transformed roles are grouped by identity, so unrelated
consumers receive deterministic clones and conflicting shared roles reject
the plan. The SUB singleton is strictly validated but left unchanged.

Equivalent post aliases preserve every repeated downstream input slot. Later
consumers of the former NCHW residual receive one inverse adapter inserted
before their first use. Typed permutations, producer uniqueness, exact fan-
out, dependency order, public boundaries, fused activation, Softmax beta,
ReduceMax keep-dims, FLOAT16/FLOAT32/FLOAT64 dtype, quantization, layout,
shape, and signature are all checked before mutation. The complete plan is
re-resolved before apply, and clone names, mutation/removal indices, reshape
options, alias slots, metadata targets, and legacy-adapter insertion are
preflighted before the first write.

One differential graph index, graph-ordered candidates, a 32-rewrite ceiling,
success-only pruning, and LayoutState synchronization replace the full-map
fixed-point loop. The sole production call retains its ordered recovery
position and now supplies Session LayoutState. Focused coverage passed with
`50 passed in 0.53s`; combined with all preceding indexed SiNet suites it
passed with `616 passed in 1.53s`. Coverage includes eighteen numerical-
equivalence combinations over all three floating dtypes,
scalar/channel/full constants, both operand orders, dynamic signatures,
idempotence, post aliases, repeated
alias and legacy slots, no-legacy operation, external float and integer
constant cloning, candidate-only and capped execution, twenty-five unsafe
contracts, stale-plan and clone-collision revalidation, no-index preflight,
differential-index validation, and LayoutState validation.

The sequential production integration applied this owner once in each of the
float32 and float16 ModelIR builds. The output remains byte-identical: the
449,824-byte float32 artifact has SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`,
and the 253,452-byte float16 artifact has SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
This preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
The full architecture suite passed with `190 passed in 48.85s`, and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed in 3.80s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed mix-attention checkpoint moves
`_optimize_sinet_mix_attention_double_logistic_nhwc_chains` to
`passes/sinet_mix_attention_layout.py`. A pre-extraction production audit ran
all six ordered invocations against `sinet_320_op.onnx`; the legacy owner
matched zero candidates in every invocation. The path is therefore retained
as a compatibility topology rather than used to tune the active model.

The new resolver starts at the terminal NCHW-to-NHWC adapter and proves its
sole post-Conv owner. It then resolves the two Logistic gates and the
`gate * branch + (1 - gate) * residual` tail back to the shared source ADD.
The branch is one private NHWC-to-NCHW adapter. The residual may be either a
private ADD of two such adapters or a single private adapter, preserving both
forms supported by the former helper. Binary and Concat operand order is
semantic, not positional.

The channel-attention branch must contain Mean, NCHW-to-NHWC, two Conv
operators, and NHWC-to-NCHW. The spatial-attention branch must contain Mean
and ReduceMax, a channel Concat, MirrorPad, NCHW-to-NHWC, Conv, and Reshape.
The position-attention merge must contain the two rank-five Reshapes, Concat,
rank-four Reshape, MirrorPad, NCHW-to-NHWC, Conv, and NHWC-to-NCHW. Exact
producer uniqueness, consumer multiplicity, graph order, public boundaries,
plain fused activations, floating dtype, quantization, layout, shape, and
dynamic signature contracts are checked for the whole island.

Reduce axes, both MirrorPad pair tensors, the two rank-five targets, the
rank-four target, and the rank-five Concat axis preserve INT32 or INT64 dtype
while being remapped to their NHWC contracts. Shared constants are grouped by
identity. An unrelated consumer receives a deterministic clone; incompatible
shared roles reject the plan. Metadata for rank-four tensors is derived from
the canonical terminal NHWC output, while transformed rank-five tensors are
explicitly layout-unknown rather than assigned a rank-four label.

The complete plan is resolved again immediately before apply. Constant clone
names, operator indices, input slots, option changes, metadata targets, and
all eight adapters in direct-residual mode or nine adapters in ADD-residual
mode are preflighted before the first write. The branch and residual tail are
rewired to their canonical NHWC sources. This deliberately fixes the former
helper's unreachable-topology defect that removed the branch adapter while
rewiring its consumers back to the removed adapter output, which could have
left an unbound tensor.

One differential `ModelIRGraphIndex`, graph-order candidates, a configurable
32-rewrite ceiling, success-only pruning, and Session `LayoutState`
synchronization replace the full-map `while True` loop. Both production calls
retain their ordered positions and now supply the Session LayoutState. No
model name or fixed spatial size participates in matching.

Focused coverage passed with `45 passed in 0.49s`; all indexed SiNet suites
passed with `661 passed in 1.67s`. Coverage spans FLOAT16/FLOAT32/FLOAT64,
both residual forms, reversed binary and Concat inputs, dynamic signatures,
shared and externally consumed constants, candidate-only and capped
execution, twenty-eight unsafe mutations, stale-plan revalidation, clone-name
collision, no-index preflight, differential-index validation, and LayoutState
validation. The full architecture suite, run together with the focused owner,
passed with `236 passed in 46.85s`, including `191` architecture tests.
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed in 3.66s`.

The sequential real SiNet integration smoke passed with `1 passed in 2.79s`.
A separate sequential CLI conversion produced the unchanged 449,824-byte
float32 artifact with SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and the unchanged 253,452-byte float16 artifact with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The byte identity preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
Scoped Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed SA/PA MirrorPad checkpoint moves
`_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains` to
`passes/sinet_sa_pa_mirrorpad_layout.py`. A pre-extraction production audit
ran all seven ordered invocations against `sinet_320_op.onnx`; the legacy
owner matched zero candidates in every invocation. The replacement is
therefore a compatibility owner and does not use the current SiNet artifact to
broaden its guards.

The resolver starts at one private NHWC-to-NCHW source adapter. Its exact
fan-out is Mean, ReduceMax, and one source Reshape, plus the terminal Mul only
for the legacy NCHW-output form. Mean and ReduceMax converge at a channel
Concat followed by MirrorPad, NCHW-to-NHWC, Conv, and the removable singleton-
channel Reshape. The second ADD input must be one external NHWC channel-
attention value behind its own private adapter. The ADD result and original
source are expanded to rank five, concatenated, reshaped to rank four, padded,
and passed through NCHW-to-NHWC and Conv before Logistic and Mul.

The terminal Mul supports two historical forms. A direct NHWC source requires
an already-NHWC Logistic gate and retains its output name. A legacy NCHW source
requires the matching gate adapter; the owner rewires Mul to NHWC, gives it a
deterministic private NHWC output, and inserts one inverse adapter immediately
after Mul to preserve the original NCHW output name, public output, and all
downstream consumers.

The rewrite explicitly requires a concrete singleton source channel. Removing
the SA Reshape and channel-attention adapter is generally value-preserving only
under that condition; the former raw helper did not state this invariant. A
non-singleton candidate is now a transactional no-op instead of a potentially
incorrect layout rewrite. Concrete shape and dynamic signature relationships
are derived without a fixed spatial size.

Both reduction axes, both MirrorPad tensors, both rank-five expansion targets,
and the final rank-four target preserve INT32 or INT64 dtype while being
remapped to NHWC. Shared constants are grouped by identity. Unrelated users
receive deterministic clones; conflicting roles reject the plan. Rank-four
metadata follows the proven NHWC source contract, and rank-five intermediates
are explicitly layout-unknown.

The complete plan is resolved again immediately before apply. Constant clone
names, source provenance, unique producers, exact consumers, dependency order,
public boundaries, input slots, option changes, metadata targets, legacy
output names, and all five or six removals are preflighted before the first
write. One differential `ModelIRGraphIndex`, graph-ordered candidates, a
configurable 32-rewrite ceiling, success-only pruning, and Session
`LayoutState` synchronization replace the full-map `while True` loop. All
three production calls retain their ordered positions and now supply the
Session LayoutState.

Focused coverage passed with `42 passed in 0.47s`; all indexed SiNet suites
passed with `703 passed in 1.76s`. Coverage spans FLOAT16/FLOAT32/FLOAT64,
direct-NHWC and legacy-NCHW Mul forms, reversed binary and Concat inputs,
dynamic signatures, external constant cloning, candidate-only and capped
execution, twenty-five unsafe topology/provenance/layout/public-boundary
contracts, stale-plan revalidation, legacy-name collision, differential-index
validation, and LayoutState validation. The full architecture suite, run with
the focused owner, passed with `234 passed in 47.46s`, including `192`
architecture tests. TensorFlow-import-blocked direct, default, and `-cotof`
conversion passed sequentially with `3 passed in 3.70s`.

The sequential production conversion confirmed seven zero-match indexed
invocations and produced the unchanged 449,824-byte float32 artifact with
SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and the unchanged 253,452-byte float16 artifact with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The byte identity preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
Scoped Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed transpose/binary bridge checkpoint moves
`_optimize_transpose_binary_bridges` to
`passes/binary_bridge_layout.py`. A pre-extraction sequential audit ran the
legacy helper once on each of `face_detection_yunet_2023mar.onnx`,
`FastestDet.onnx`, `human_segmentation_pphumanseg_2021oct_org.onnx`,
`osnet025_Nx3x256x128.onnx`, and `sinet_320_op.onnx`. Both the symmetric and
asymmetric counters were zero for all five models. QDQ graphs continue to skip
this production call exactly as before.

The new owner retains the active semantic families only. Symmetric mode
supports one inverse post with no extra fan-out, one inverse post plus later
legacy-layout consumers, and no inverse post plus a synthesized adapter before
the first legacy consumer. Asymmetric mode moves the inverse permutation to
the plain operand and preserves the original operand order. The permanently
disabled Pattern C body was removed.

Every candidate now requires a plain binary operation, typed INT32/INT64
permutation constants, exact inverses, unique producers, resolved sources,
private intermediate tensors, dependency order, public-output preservation,
same dtype, per-tensor quantization, static broadcast consistency, and dynamic
signature consistency. Mixed fan-out no longer mutates the post permutation
buffer; it references the already-proven pre permutation. A complete plan is
resolved again immediately before apply. This removes the former no-post
partial-mutation bug and rejects asymmetric rewrites whose replacement plain
source is not available before the reused Transpose.

Focused owner and existing direct-lowering coverage passed together with
`42 passed, 735 deselected in 1.12s`. The new 22-test owner suite covers all
four binary operations in all three symmetric modes, exact numerical
equivalence, asymmetric SUB/DIV on both operand sides, idempotence,
candidate-only and zero-limit operation, per-axis quantization, fused
activation, public boundary, duplicate producer, late-source rejection,
differential-index freshness, invariant validation, and LayoutState
synchronization. The complete architecture suite passed with
`193 passed in 44.80s`. TensorFlow-import-blocked direct, default, and `-cotof`
conversion passed sequentially with `3 passed in 3.84s`.

One sequential YuNet conversion was run both from preceding commit `c09b8b88`
and from the current implementation. Every emitted file was byte-identical.
The 236,564-byte float32 artifact has SHA-256
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
the 131,120-byte float16 artifact has SHA-256
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
and the 105,578-byte correspondence report has SHA-256
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`.
The temporary detached worktree and both output directories were removed.
Scoped Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed safe binary recovery checkpoint replaces the five helpers called
by `_run_safe_binary_bridge_recovery_sequence`. Before extraction, each helper
was observed four times per model in the ordered pipelines for
`face_detection_yunet_2023mar.onnx`, `FastestDet.onnx`,
`human_segmentation_pphumanseg_2021oct_org.onnx`,
`osnet025_Nx3x256x128.onnx`, and `sinet_320_op.onnx`. Asymmetric fan-out,
full-post fan-out, single-post, and mixed fan-out returned zero in all twenty
calls each. Legacy-only returned zero except for the first SiNet invocation,
where it rewrote exactly `Add_52` and `Add_109`.

The five modes now run through one indexed owner in their unchanged order.
Legacy-only and single-post reuse the strict symmetric plan while retaining
the late `__preserve_layout_boundary__` contract. Mixed and full-post fan-out
share a multi-post plan: the first post output is canonical, later aliases are
rewired through indexed consumers, and mixed mode keeps one verified adapter
before all legacy users. Its permutation input changes to the proven pre-
permutation tensor rather than overwriting the potentially shared inverse
constant. Asymmetric fan-out retains its separate existing-plain-Transpose
contract and rejects that Transpose when it would occur after the binary.

Typed permutations, source provenance, unique producers, exact consumer
multiplicity, graph order, public boundaries, fused activation, dtype,
per-tensor quantization, static broadcast, dynamic signatures, options,
metadata, aliases, and removal/insertion indices are captured in immutable
plans and re-resolved before apply. Each phase has a configurable 32-rewrite
ceiling. One `ModelIRGraphIndex` is reused across all selected phases, and
pruning plus Session LayoutState synchronization run once afterward. Five
lowerer compatibility helpers remain as 17-line dispatchers; production uses
the single ordered entry point.

The two focused binary owners and the complete architecture suite passed
together with `234 passed in 44.37s`, comprising 22 general bridge tests, 19
safe recovery tests, and 193 architecture tests. The safe recovery suite
covers all four binary operations for mixed/full multi-post fan-out,
order-sensitive asymmetric SUB/DIV on both sides, phase stats, candidate and
rewrite limits, retained-boundary markers, shared inverse constants,
differential-index freshness, LayoutState, public/order rejection, per-axis
quantization, idempotence, and exact numerical equivalence. The selected 23
adjacent direct-builder tests passed with `23 passed, 732 deselected in 1.21s`,
including the four no-post boundary regressions. TensorFlow-import-blocked
direct, default, and `-cotof` conversion
passed sequentially with `3 passed in 3.75s`.

The indexed production sequence still reports two legacy-only rewrites in the
first SiNet invocation and zero for every other phase/invocation. Sequential
conversion from preceding commit `53381952` and the current implementation
emitted byte-identical files: float32 remains 449,824 bytes with SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`,
float16 remains 253,452 bytes with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`,
and the 182,687-byte correspondence report remains
`24c423ea51b26b178d3764be027855e797bbf9b5ba1930810d2e1dbe281d8e25`.
The schema outputs also match. The detached worktree and temporary output
directories were removed. No Tier corpus conversion was run.

The indexed direct Split-root checkpoint replaces the former mixed
match/mutate/fixed-point helper with a separate direct-root plan and the common
closed-tail resolver in `passes/split_channelwise_layout.py`. Pre-extraction
characterization recorded zero matches in every invocation: four each on
YuNet, FastestDet, HumanSeg, and OSNet, and eight on SiNet. The helper remains
in both unchanged ordered production positions as a thin dispatcher and now
receives Session LayoutState.

The direct-root plan validates a typed private `[0,3,1,2]` Transpose, a sole
channel Split consumer, Split-axis copy-on-write, unique producers, graph
order, rank-four shape and dynamic-signature relationships, dtype,
per-tensor quantization, and the complete closed tail. Only root Split outputs
seed the closure. Unary, binary, Concat, downstream Split, and exact rank-four
Slice operations may propagate NHWC; unsupported consumers and dead branches
reject the candidate.

Slice begin/size constants are planned without mutation, validated against the
old and new layout, and grouped by tensor identity. Exclusive same-value uses
may update in place. Any unrelated consumer causes one deterministic shared
clone for all planned slots, retaining actual INT32 or INT64 NumPy dtype.
Conflicting roles, variables, producer-backed/public/missing constants, bounds
errors, dtype mismatches, public intermediates, duplicate producers, and stale
consumer order are transactional no-ops.

The new direct suite passed with `20 passed`; it and the binary-root suite
passed together with `53 passed in 0.51s`. The two indexed binary bridge
suites, both Split-root suites, and all architecture tests passed with
`288 passed in 44.74s`. Coverage includes exact semantics for INT32 and INT64
constants with static and dynamic signatures, shared Split and Slice constant
cloning, an unrelated source consumer, candidate and rewrite limits,
idempotence, differential graph-index freshness, LayoutState validation, and
thirteen unsafe no-op cases. TensorFlow-import-blocked explicit direct,
default direct, and `-cotof` conversion passed sequentially with `3 passed in
3.85s`.

One sequential YuNet conversion was run from source checkpoint `5f8e662a` and
from the current implementation. All five emitted files were byte-identical.
Float32 remains
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16 remains
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
and the correspondence report remains
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`.
Both schema hashes also match. The detached worktree and temporary outputs were
removed. Scoped Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed unary/Split/Concat checkpoint replaces the former raw helper with
a separate unary-root plan in `passes/split_channelwise_layout.py`.
Pre-extraction characterization observed zero matches in all 24 sequential
runtime invocations: four each on YuNet, FastestDet, HumanSeg, and OSNet, and
eight on SiNet. The two existing production positions and their ordering before
the direct and binary Split roots are unchanged, and both now receive Session
LayoutState.

The owner validates the private typed Transpose, allowed pre-unary, channel
Split, exact one-to-one coverage of every Split output, optional branch
unaries, exactly one external Concat branch, channel Concat, unique producers,
resolved provenance, graph order, public boundaries, dtype, per-tensor
quantization, static shapes, and dynamic signatures. Every converted branch
has exact consumers through the Concat. The local terminal adapter preserves
the original NCHW Concat tensor for either a public output or later consumers.

The external layout-only Reshape now requires a real singleton channel and an
exact NHWC-to-NCHW shape/signature relation. A proven direct NHWC external
branch is also accepted. Shared Reshape or raw-source consumers remain
untouched. The full plan is re-resolved and preflighted before changing the
pre-unary input, Split axis, branch metadata, Concat inputs/options, or output
adapter. This specifically eliminates the raw partial-mutation failure when
the Concat output tensor is absent or invalid.

The focused suite passed with `29 passed in 0.44s`. It covers INT32/INT64
axes, static and dynamic signatures, public and intermediate NCHW boundaries,
exact numerical equivalence, a direct NHWC external branch, shared axis and
side-consumer preservation, candidate-only and capped execution, idempotence,
differential-index freshness, LayoutState validation, and eighteen unsafe
transactional no-op cases. The active legacy fixture was corrected to declare
its previously producerless external tensor as a graph input.

The new owner, existing active fixture, both preceding Split-root owners, both
binary bridge owners, and the complete architecture suite passed with
`318 passed in 43.96s`. TensorFlow-import-blocked explicit direct, default
direct, and `-cotof` conversion passed sequentially with `3 passed in 3.83s`.
One sequential YuNet conversion from source checkpoint `0b4a0001` and the
current implementation emitted five byte-identical files. Float32 remains
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16 remains
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
and correspondence remains
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`.
Both schema files also match. The detached worktree and temporary outputs were
removed. Scoped Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed singleton-gate checkpoint replaces the former raw gate/Conv/
Concat helper with `passes/singleton_gate_layout.py`. Pre-extraction
characterization observed zero matches in all 24 sequential runtime
invocations: four each on YuNet, FastestDet, HumanSeg, and OSNet, and eight on
SiNet. The helper remains at both unchanged production sequence positions as a
thin dispatcher and now receives Session LayoutState.

The owner proves the exact clip adapter, gate multiply, scalar subtraction,
direct or Logistic auxiliary branch, signal multiply, Add, allowed terminal
unary, output adapter, and channel-last Concat before mutation. Every removed
adapter is a true singleton-channel NHWC/NCHW view with compatible static and
dynamic shapes, dtype, per-tensor quantization, typed immutable Reshape shape
input when present, unique producer, resolved provenance, and valid graph
order. Public intermediates, fused activations, duplicate producers, stale
order, unsupported fan-out, non-singleton channels, per-axis quantization, and
invalid broadcast contracts are transactional no-ops.

The optional RGB branch accepts either a fully owned typed `[0,3,1,2]`
Transpose or a private constant whose physical NHWC evidence proves that its
NCHW metadata is stale. The latter constant's data buffer is reshaped together
with its metadata. View-equivalent split and terminal adapter consumers are
rewired as a group, while unrelated source and side consumers remain intact.
All input rewrites, tensor/operator contracts, metadata updates, and removals
are captured in an immutable plan and fully re-resolved immediately before
apply through one differential graph index. Graph-ordered candidates, a
configurable 32-rewrite ceiling, success-only pruning, and LayoutState updates
replace the unbounded full-map loop.

The focused suite passes with `29 passed in 0.47s` and covers both auxiliary
forms across static/dynamic and RGB/no-RGB modes, direct stale-layout RGB,
shared side consumers, exact numerical equivalence, candidate-only and capped
execution, idempotence, differential-index freshness, LayoutState validation,
and nineteen unsafe no-op cases. The dedicated owner, two corrected active
fixtures, the unary/direct/binary Split roots, both binary bridge owners, and
the complete architecture suite pass together with `349 passed, 753
deselected in 43.88s`. TensorFlow-import-blocked explicit direct, default
direct, and `-cotof` conversion pass sequentially with `3 passed in 3.80s`.

One sequential YuNet conversion was run from source checkpoint `12680f75` and
from the current implementation. All five emitted files are byte-identical.
Float32 remains
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16 remains
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
the correspondence report remains
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
and both schema hashes also match. The detached worktree and all temporary
outputs were removed. Scoped Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed Conv-output passthrough checkpoint replaces the former 316-line
raw `_optimize_transposeconv_output_nhwc_passthrough_chains` implementation
with `passes/conv_output_passthrough_layout.py`. Its one production position
and phase order before the channel-one terminal helper are unchanged, and the
dispatcher now receives Session LayoutState.

Pre-extraction characterization ran the two adjacent helpers sequentially on
the five short representatives. Each helper was invoked four times per model.
Only the first general-passthrough invocation was active: YuNet rewrote 10,
FastestDet 23, HumanSeg 27, OSNet 63, and SiNet zero chains. All 123 active
chains were Conv2D or DepthwiseConv2D followed by RELU. The distinct
channel-one TransposeConv/Squeeze terminal helper returned zero in all 20
invocations and remains a separate semantic family.

The new owner proves a typed private NHWC-to-NCHW Transpose produced by
Conv2D, DepthwiseConv2D, or TransposeConv, a strictly linear nonempty chain,
and one typed inverse Transpose with an intermediate NHWC output. It preserves
all seven historical unary operations and all six binary operations without
changing operand slots. Every tensor, producer, consumer, graph-order,
static/dynamic shape, dtype, quantization, public-boundary, and post-output
contract is resolved before mutation.

Rank-four binary constants require matching immutable data and TensorIR
metadata plus valid NCHW and NHWC broadcasts. Exclusive constants change in
place. Shared constants receive one deterministic NHWC clone reused by every
planned chain slot, while unrelated legacy consumers retain the original.
Scalar constants remain unchanged. The surviving post-adapter output metadata
is derived from the converted final chain output rather than blindly permuting
possibly stale existing metadata.

All input/output rewrites, constant changes, metadata, tensor/operator
contracts, and adapter removals are immutable plan state and are re-resolved
immediately before apply through one differential graph index. A current-
candidate bound and optional explicit rewrite limit replace the full-map
unbounded fixed-point loop. LayoutState is synchronized differentially and
pruning runs only after success.

The new focused suite passes with `56 passed in 0.51s`. It covers all three
producer types, all unary and binary operations, both operand positions, exact
SUB/DIV and other binary numerical behavior, dynamic signatures, grouped
shared constants, candidate/limit/idempotence behavior, differential index and
LayoutState, and twenty unsafe transactional no-op cases. The new owner,
singleton-gate and three Split-root owners, both binary bridge owners, existing
active fixtures, and complete architecture suite pass together with `408
passed, 751 deselected in 44.32s`. TensorFlow-import-blocked explicit direct,
default direct, and `-cotof` conversion pass sequentially with `3 passed in
3.91s`.

Sequential conversions from source checkpoint `410b86e6` and the current
implementation produced byte-identical float32, float16, correspondence, and
both schema files for every active representative: YuNet, FastestDet,
HumanSeg, and OSNet. The temporary detached worktree and outputs were removed.
Scoped Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed channel-one terminal checkpoint replaces the remaining 390-line
raw `_optimize_transposeconv_output_channel1_terminal_transpose_chains`
implementation with a second immutable plan in
`passes/conv_output_passthrough_layout.py`. The helper remains immediately
after the general passthrough owner in the unchanged attention/gate recovery
sequence and now receives Session LayoutState.

The terminal resolver requires a TransposeConv output whose static and dynamic
NHWC channel is exactly one, one private typed NHWC-to-NCHW adapter, a strictly
linear unary/binary chain containing exactly one Squeeze, and a final graph
output with no consumer. It preserves all fourteen historical unary operations
and all six binary operations without reordering operands. Public
intermediates, a second Squeeze, stale order, fan-out, invalid quantization,
missing tensors, or a nonterminal output reject the complete candidate.

Explicit, negative, and implicit Squeeze axes are normalized in NCHW and
remapped to NHWC by semantic label. Every squeezed static and dynamic
dimension must equal one. The surviving labels must occur in the same order in
both paths, preventing a spatial-only Squeeze from exposing a reordered public
rank-three tensor. Metadata and LayoutState use NHWC/NWC only when the
surviving labels support that annotation; channel-free outputs remain
explicitly unknown rather than receiving a misleading layout.

Non-scalar binary constants are accepted only before Squeeze and reuse the
general owner's grouped rank-four constant planner. A post-Squeeze binary must
use a scalar constant. All chain inputs, Squeeze options, tensor/operator
contracts, constant updates, output metadata, and the one adapter removal are
re-resolved before apply through the differential graph index. Candidate count
and an optional limit bound execution; pruning occurs only after success.

The terminal suite passes with `57 passed in 0.69s`, covering explicit,
negative, and implicit axes, static/dynamic signatures, all unary and binary
operations, both operand positions, numerical equivalence, post-Squeeze scalar
binary, shared constants, candidate limits, idempotence, GraphIndex,
LayoutState, and twenty-two unsafe no-op cases. It and the preceding indexed
owners, active fixtures, and complete architecture suite pass together with
`465 passed, 751 deselected in 43.74s`. TensorFlow-import-blocked explicit
direct, default direct, and `-cotof` conversion pass sequentially with `3
passed in 3.95s`.

YuNet retained four zero-match terminal invocations. A sequential comparison
against source checkpoint `ac8c4cf7` emitted the same five byte-identical
files: float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
and both unchanged schema artifacts. The first comparison-harness attempt
omitted its output-directory environment variable and exited before current-
branch conversion; the corrected run passed. The temporary worktree and all
outputs were removed. No Tier corpus conversion was run.

The RELU/Split all-output checkpoint passed the dedicated owner and active
compatibility fixture first:

```text
uv run pytest -q \
  tests/test_flatbuffer_direct_indexed_split_all_outputs_layout.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_relu_split_all_outputs_to_nhwc_chains_optimized

32 passed in 2.92s
```

The new owner, adjacent binary/Split, direct Split, unary/Split/Concat,
singleton-gate, Conv-output, terminal TransposeConv, and binary-bridge owners,
the complete architecture suite, and the active fixture then passed together:

```text
475 passed in 46.51s
```

TensorFlow import blocking was exercised separately and sequentially for the
explicit direct backend, default direct backend, and direct `-cotof` path:

```text
3 passed in 4.01s
```

Scoped Ruff passed for the new owner, dedicated tests, and architecture test.
All changed Python files passed `py_compile`; the lowerer and legacy direct
fixture passed Ruff with only their documented inherited F841/F401 categories
ignored. `git diff --check` passed. One sequential YuNet conversion reproduced
the five hashes recorded in the continuation snapshot. No Tier corpus was run.

The exact Split/Conv/Concat checkpoint passed its 49 dedicated cases and the
corrected active compatibility fixture with `50 passed in 0.61s`. The new
plan, all-output plan, adjacent binary/Split, direct Split,
unary/Split/Concat, singleton-gate, Conv-output, terminal TransposeConv, and
binary-bridge owners, both active fixtures, and the complete architecture
suite then passed together:

```text
525 passed in 47.51s
```

TensorFlow import blocking was again exercised sequentially for explicit
direct, default direct, and direct `-cotof` conversion:

```text
3 passed in 4.33s
```

One sequential YuNet conversion reproduced all five fixed hashes. Scoped Ruff,
`py_compile`, and `git diff --check` passed. No Tier corpus was run.

The tanh-expanded GELU checkpoint passed its 56 dedicated cases. The activation
owner suites, adjacent input/quantized-Swish suites, active Swish and GELU
fixtures, and the complete architecture suite passed together:

```text
362 passed in 42.95s
```

TensorFlow import blocking was exercised sequentially for explicit direct,
default direct, and direct `-cotof` conversion:

```text
3 passed in 3.95s
```

Scoped Ruff and syntax compilation passed before the checkpoint. The lowerer
uses only its documented inherited F841 exclusion. `git diff --check` passed.
Pre- and post-extraction characterization produced 20 zero-match invocations
with unchanged operator/tensor counts, and one sequential YuNet conversion
reproduced all five fixed hashes. Temporary outputs were removed. No Tier
corpus was run.

The center/size/offset checkpoint passed its 37 dedicated cases. The new
owner, adjacent activation/input/quantized-Swish owners, active center, Swish,
and GELU fixtures, and the complete architecture suite passed together:

```text
399 passed in 42.94s
```

TensorFlow import blocking was exercised sequentially for explicit direct,
default direct, and direct `-cotof` conversion:

```text
3 passed in 3.99s
```

Scoped Ruff and syntax compilation pass for the owner, dedicated test,
architecture test, and lowerer with only its inherited F841 exclusion.
`git diff --check` passes. Pre/post characterization produced the same 20
zero-match invocations and unchanged counts. One sequential YuNet conversion
reproduced all five fixed hashes; temporary outputs were removed. No Tier
corpus was run.

The pseudo-expanded LeakyReLU checkpoint passed its 51 dedicated cases. The
new passthrough owner, existing indexed fusion owner, adjacent activation and
layout owners, active fixtures, and the complete architecture suite passed
together:

```text
467 passed in 42.33s
```

TensorFlow import blocking was exercised sequentially for explicit direct,
default direct, and direct `-cotof` conversion:

```text
3 passed in 4.01s
```

Scoped Ruff and syntax compilation pass for the owner, dedicated test,
architecture test, and lowerer with its inherited F841 exclusion.
`git diff --check` passes. Pre/post characterization produced 20
zero-passthrough/zero-fusion invocations with unchanged counts. One sequential
YuNet conversion reproduced all five fixed hashes; temporary outputs were
removed. No Tier corpus was run.

The PReLU passthrough checkpoint passed its 28 dedicated cases. The new owner,
adjacent Swish/GELU/center/LeakyReLU owners, quantized-Swish owner, active
fixtures, and the complete architecture suite passed together:

```text
454 passed in 50.22s
```

TensorFlow import blocking was exercised sequentially for explicit direct,
default direct, and direct `-cotof` conversion:

```text
3 passed in 4.38s
```

Scoped Ruff, syntax compilation, and `git diff --check` pass. Sequential
pre/post characterization preserved six invocations on each of the five short
representatives: four models remained zero-match and SiNet retained 23+23
rewrites. FastestDet retained its zero-match 519-to-518 tensor prune. One
sequential YuNet conversion reproduced all five fixed hashes. SiNet also
reproduced its fixed float32, float16, correspondence, and schema hashes.
Temporary worktrees and outputs were removed. No Tier corpus was run.

The elementwise/Concat layout checkpoint passed all 56 dedicated cases. The
new owner, adjacent PReLU/LeakyReLU/center/GELU/Swish and quantized-Swish
owners, SPP owner, both active compatibility fixtures, and the complete
architecture suite passed together:

```text
546 passed in 42.49s
```

TensorFlow import blocking was exercised sequentially for explicit direct,
default direct, and direct `-cotof` conversion:

```text
3 passed in 4.07s
```

Pre/post characterization preserved five invocations on each of YuNet,
FastestDet, HumanSeg, OSNet, and SiNet. FastestDet retained two active groups,
HumanSeg retained one, and the other 22 invocations remained zero-match.
Non-layout ModelIR digests match the preceding checkpoint at every active-model
invocation; full digests differ only through the intentional explicit NHWC
layout provenance.

Sequential YuNet, FastestDet, and HumanSeg conversions reproduced all fifteen
fixed artifacts. The fixed float32/float16/correspondence hashes are:

```text
YuNet:      43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380
            13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433
            7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d
FastestDet: 3bdbec5d7ad81f98cf7890fbf1a98570ebeb1a4a5c19883aca23733b31e1573b
            a14bad05eba99dc211a09aa820eb38396329b98168a2d4b20e463eb64deab617
            2bd03e9e775b4dede0310813cf36e2efc3ad9d0635ce0c5797895fe18d7fb074
HumanSeg:   b69eb57a7628668d73fbf4e06ffa23403d02ebab33b5661f0c60a81395610bb9
            e79c4081989b5a69b65dc220b6e24ad0cff5633363698feebc23b59efe1139df
            87fd06dbd120aac7f0229b02484f7929d6e752c60c19555d6795221cb0a21e46
```

All three conversions also reproduced schema hashes
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`
and
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
Scoped Ruff, syntax compilation, and `git diff --check` pass. Temporary
worktrees and outputs were removed. No Tier corpus was run.

The StridedSlice/Concat checkpoint passed all 58 dedicated cases. The new
owner, adjacent pre-Concat Slice/Split and Split/Conv/Concat owners,
elementwise/Concat owner, and the complete architecture suite passed together:

```text
542 passed in 40.54s
```

TensorFlow import blocking was exercised sequentially for explicit direct,
default direct, and direct `-cotof` conversion:

```text
3 passed in 4.01s
```

Pre/post characterization preserved five zero-match invocations on each of
YuNet, FastestDet, HumanSeg, OSNet, and SiNet, with unchanged operator and
tensor counts in all 25 calls. The ordinary and multi-post synthetic active
forms reproduce the exact source-checkpoint non-layout ModelIR digest including
lineage metadata:

```text
basic:   1b3187150bd2819fe9e2324568dad0582c7227e12d0509e1820929fe5e438ca6
aliases: 02a2db7154cc74250a8c8f2f7a14edf2293cd4c80bc14f48f7b86b41c26ad801
```

One sequential YuNet conversion reproduced float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
and the two fixed schema hashes. Scoped Ruff, syntax compilation, and
`git diff --check` pass. Temporary comparison worktrees and outputs were
removed. No Tier corpus was run.

## Latest checkpoint file set

The Slice/Logistic/Concat/Reshape tail checkpoint changes only the following
tracked source, test, and documentation files relative to `762dcdef`:

- `onnx2tf/tflite_builder/passes/slice_logistic_concat_reshape_tail_layout.py`:
  new indexed semantic owner and immutable whole-tail transaction plan.
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`: thin compatibility wrapper,
  owner import, and Session `LayoutState` at both direct call sites.
- `tests/test_flatbuffer_direct_indexed_slice_logistic_concat_reshape_tail_layout.py`:
  new focused capability, numerical, rejection, bounded-dispatch, and
  determinism coverage.
- `tests/test_flatbuffer_direct_architecture.py`: ownership, transaction, and
  unchanged direct call-boundary contract.
- `docs/flatbuffer_direct_architecture.md`: indexed owner responsibilities and
  validation boundary.
- `docs/flatbuffer_direct_handoff_2026-07-14.md`: checkpoint evidence,
  remaining work, known issues, and restart instruction.

No generated model artifact, root ONNX model, dependency file, bulk-regression
profile, or exclusion policy is changed by this checkpoint.

## Tier 0–4 measured-quick regression checkpoint after `f5a40947`

The complete 49-model measured-short, zero-SWAP Tier 0–4 manifest was rerun
sequentially after the Slice/Logistic/Concat/Reshape-tail checkpoint. The run
completed in 390.255 seconds with 43 passes and six preserved known
non-passes. There were zero timeouts, zero converter nonzero exits, and zero
model-process SWAP detections. All 49 classifications and all 47 available
numeric maximum-error results match the corrected recorded expectations, so
no new correctness regression was found and no production fix was applied.

The pass-metric comparison is also stable. Of 48 comparable models, only
`sinet_320_op.onnx` changed aggregate work: preflight operator visits fell from
33,303 to 31,906 while events, statuses, snapshots, state builds, accuracy, and
classification stayed identical. This is the intended indexed-traversal
benefit.

DEIM completed successfully with the already approved TopK-index acceptance,
but its one measured duration increased from 24.043 to 37.630 seconds. It used
zero SWAP and preserved the exact raw maximum error. Treat this as an
unconfirmed runtime observation, not a semantic failure: remeasure it once
before the next quick-profile refresh, and exclude it from that quick profile
only if the over-30-second duration repeats. Do not change source code in
response to this single sample.

Detailed analysis is in
`docs/flatbuffer_direct_quick_regression_2026-07-16.md`; compact per-model
evidence is in
`docs/baselines/flatbuffer_direct_quick_tier0_4_f5a40947_result.json`.

## Failing tests and known issues

- No newly failing focused test is known at this checkpoint.
- A whole-file Ruff run on `pytorch_exporter.py` reports 282 pre-existing
  compatibility re-export, unused scaffold, and undefined-name findings. It is
  not used as the scoped checkpoint gate; changed owners/tests pass Ruff and
  the exporter passes syntax compilation.
- A whole-file Ruff run on `onnx2tf.py` reports pre-existing import-order,
  star-import, bare-except, undefined-name, and placeholder-f-string findings.
  This checkpoint uses the existing scoped exclusions for those categories;
  the changed helper owner/tests pass Ruff normally and all changed Python
  files pass syntax compilation.
- A whole-file Ruff run on `lower_from_onnx2tf.py` reports 12 pre-existing
  unused-local (`F841`) findings outside this extracted helper. The changed
  lowerer passes Ruff with that inherited category ignored; the new owner and
  dedicated owner/architecture tests pass Ruff without exclusions.
- A whole-file Ruff run on `test_tflite_builder_direct.py` reports ten
  pre-existing unused compatibility imports (`F401`). The corrected
  equal-Split fixture passes with that inherited category ignored; the
  dedicated new owner test and architecture test pass Ruff without exclusions.
- The optional PyTorch exporter suite runs when the host's Python 3.10
  `LD_LIBRARY_PATH` and `PYTHONPATH` are removed from the command environment.
  The focused results, restored native-codegen bindings, real-model artifact
  gate, and remaining inherited failures are recorded in
  `docs/flatbuffer_direct_pytorch_regression_2026-07-14.md`.
- The optional TensorFlow suite was not synchronized or run.
- Recent PyTorch source-policy checkpoints have not been followed by a Tier
  corpus conversion run. This is intentional under the current minimal-
  conversion instruction, but broad model-level regression remains unproven.

## Unfinished work

The full Goal is not complete. The fast-precanonicalize orchestrator still has
294 lines. Its remaining body is primarily the intended ordered helper
orchestration, source-line replacement, changed-flag handling, and the explicit
short-circuit boundaries required by the extracted policy decisions.

The adjacent Split/Conv/Concat trio has now been fully extracted. The
all-output and exact two-branch Conv/Concat roots share the Split-layout owner;
the generic direct bridge has its own indexed owner and preserves all three
production positions. The five short representatives remain zero-match for
the bridge, and YuNet retains byte-identical artifacts.

The unquantized pseudo-Swish, tanh-GELU, center/size/offset, pseudo-LeakyReLU,
PReLU, connected elementwise/Concat, StridedSlice/Concat, Split/mixed-Concat,
general Concat input-adapter, Slice/Logistic/Concat/Reshape tail, and strict
direct/unary pre-Add recovery helpers are now indexed at their unchanged
production positions. The broader pre-Add compatibility helper remains in
place for its Swish, Gather, affine, PReLU, broadcast, nested-Add, and direct-
fallback families, including the conservative safe-transpose entry. The next
standalone raw recovery helper in the ordered prefix is
`_optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains`.
The composite pre-Concat dispatcher also retains its previously documented
legacy-family fallback after the indexed families.

The broader fixed-pipeline, remaining artifact-plan coverage, artifact-matrix,
optional TensorFlow, PyTorch/TorchScript/Dynamo/ExportedProgram, and full Tier
regression work also remains subject to the original refactor plan and its
verification gates.

## Next work

1. Confirm `git status --short --branch` is clean and local `fb-refactor5`
   matches `origin/fb-refactor5`.
2. Characterize
   `_optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains`
   at its ordered production boundary before modifying it. Fix its runtime
   match set, shape/layout contracts, constant ownership, public boundaries,
   and interaction with the new pre-Add indexed owner, then select only a
   bounded semantic family with real non-zero coverage.
3. Treat `_optimize_transpose_swish_qdq_nhwc_islands` as a thin 69-line
   compatibility orchestrator unless a bounded phase-contract simplification
   is identified; all of its former raw top-level mutation loops now have
   indexed semantic owners.
4. Keep the terminal direct backend boundary explicit; do not reintroduce
   fallback into the legacy TensorFlow pipeline or broaden optional artifact
   execution.
5. Keep the audited 294-line PyTorch source orchestrator as explicit sequencing
   unless a new bounded decision is found.
6. Run only the focused synthetic/ownership/static checks unless the user asks
   for broader conversion validation. Use `uv`, run inference sequentially if
   any is explicitly requested, commit and push coherent units, and do not
   create a pull request.

## Pre-Add direct/unary extraction: pre-fix observation

The first implementation attempt for the bounded
`pre_add_direct_unary_layout` owner passed 232 focused, interaction, QLinear,
and architecture tests. A sequential three-model `-cotof` check then found an
important coverage problem before any follow-up source adjustment: the new
owner reported zero rewrites at every one of its production invocations for
`sinet_320_op.onnx`, `FastestDet.onnx`, and
`human_segmentation_pphumanseg_2021oct_org.onnx`.

This is not a conversion regression. All three models passed, used zero
model-process SWAP, reproduced their exact recorded maximum errors, and
reproduced their fixed float32, float16, and correspondence-report hashes.
The observed values were:

- SiNet: max abs `2.572051016613841e-09`, owner events eleven zeroes;
- FastestDet: max abs `1.3113021850585938e-05`, owner events eight zeroes;
- HumanSeg: max abs `2.384185791015625e-07`, owner events eight zeroes.

The preserved legacy helper therefore still performed every applicable
rewrite. Characterization before the implementation had already shown why a
strict direct-only output contract is insufficient: FastestDet uses an
`ADD -> unary -> post-transpose` suffix and a shared input requiring the
historical direct fallback, HumanSeg contains both optional-unary and shared
branch forms, and SiNet is primarily affine/PReLU/fallback rather than the
bounded direct/unary family. No production fix was made in response to this
observation. The next investigation step is to check the previously
characterized OSNet strict residuals and then either add a transactionally
validated optional-output-unary contract or narrow the checkpoint claim to a
real, non-zero production family. The broad affine, PReLU, Gather, nested-Add,
and direct-fallback behavior must remain on the compatibility path.

After adding the bounded optional-output-unary contract, the first post-change
focused run exposed one public-report compatibility difference before cleanup
was adjusted. OSNet moved six residual Adds and HumanSeg moved one; SiNet and
FastestDet correctly remained on the compatibility path. All four models
passed with zero SWAP, exact recorded accuracy, and byte-identical float32 and
float16 TFLite files. OSNet, FastestDet, and SiNet also retained their exact
correspondence hashes. HumanSeg alone changed its correspondence hash from
`87fd06dbd120aac7f0229b02484f7929d6e752c60c19555d6795221cb0a21e46`
to
`3d0848515e026eb3f1b4c6f69b936da77584eaa2e5793f3276ae3cb447783f34`.

The JSON diff is fully classified: record content and all 370 historical
lineage events are unchanged, but the indexed owner pruned eight now-unused
adapter tensors immediately after its first rewrite. The compatibility helper
historically pruned the same eight names later as the first portion of a
larger event. This split one `prune_unused_tensors` event into two, increased
`lineage_event_count` from 370 to 371, and shifted later event indexes by one.
The TFLite graph and numeric behavior did not change. This report-only issue
was recorded before changing cleanup ownership; the safe correction is to
leave pruning at the existing compatibility-wrapper boundary instead of
performing an extra early prune in the indexed sub-owner.

## Pre-Add direct/unary extraction: final checkpoint

Cleanup ownership now remains at the historical compatibility-wrapper exit.
The indexed sub-owner no longer emits an early prune event. Final sequential
conversion-only checks reproduce HumanSeg correspondence
`87fd06dbd120aac7f0229b02484f7929d6e752c60c19555d6795221cb0a21e46`
and OSNet correspondence
`35a42832e43b2076b00399ba7b22a1ff5aff83795cd333474d6bf61bf7221677`.
Their float32/float16 hashes also remain fixed. The optional-output-unary
contract remains active: the first production call indexes one HumanSeg
residual and six OSNet residuals. FastestDet and SiNet remain zero-match in the
bounded owner and continue through the characterized compatibility families.

The final focused model gate ran SiNet, OSNet, FastestDet, and HumanSeg in
that fixed order, with one converter/inference subprocess at a time. All four
passed, recorded zero model-process SWAP, and retained exact maximum absolute
errors `2.572051016613841e-09`, `2.193450927734375e-05`,
`1.3113021850585938e-05`, and `2.384185791015625e-07`, respectively. Every
float32 and float16 TFLite file was byte-identical to its fixed checkpoint.

The final synthetic/architecture gate passes 235 tests in 47.81 seconds. The
TensorFlow import-blocked explicit direct, default-direct, and direct `-cotof`
gate passes three tests in 4.24 seconds. Scoped Ruff, syntax compilation, and
`git diff --check` pass; the lowerer uses only its existing F841 exclusion.

The checkpoint changes these tracked files:

- `onnx2tf/tflite_builder/passes/pre_add_direct_unary_layout.py`: immutable
  direct/unary residual plan, indexed resolution, differential apply, optional
  output unary, post aliases, and retained legacy boundary;
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`: indexed-first compatibility
  wrapper and Session `LayoutState` at all four direct production calls;
- `tests/test_flatbuffer_direct_indexed_pre_add_direct_unary_layout.py`:
  focused capability, ownership, rejection, bounded-dispatch, stale-plan,
  cleanup-boundary, and determinism coverage;
- `tests/test_flatbuffer_direct_architecture.py`: indexed ownership and
  unchanged call-boundary contract;
- `docs/flatbuffer_direct_architecture.md` and this handoff: design,
  pre-fix finding, report-diff attribution, final evidence, remaining work,
  and restart instruction.

No ONNX corpus model, generated TFLite artifact, dependency file, managed
quick profile, exclusion policy, or public API is changed. No broad Tier run
was performed for this bounded checkpoint.

## Pre-Add Mul-const reshape suffix extraction: pre-fix observation

The next raw helper was characterized at its unchanged ordered production
boundary before source modification. Five short, zero-SWAP representatives
(`face_detection_yunet_2023mar_int8.onnx`, `FastestDet.onnx`,
`human_segmentation_pphumanseg_2021oct_org.onnx`,
`osnet025_Nx3x256x128.onnx`, and `sinet_320_op.onnx`) recorded zero rewrites
at all three invocations. A static corpus query then identified the bounded
Add/Reshape/Transpose topology in `iat_llie_180x320.onnx`, a Tier 2 model that
previously passed in 17.339 seconds with zero model-process SWAP.

Sequential conversion-only characterization of IAT-LLIE found 5, 4, and 4
rewrites at the three prefix invocations. The 13 rewrites comprise seven
direct/direct pre-Add branches and six direct/Mul-constant branches. Some
branches become eligible only after the preceding phase has rewritten their
input layout, so all three production positions are semantically relevant.
The complete temporary trace fixed every changed Add, Mul, and Reshape input
and output plus every removed Transpose before implementation work began.

The pre-fix artifacts are fixed as follows:

- float32 TFLite:
  `75e355ba8fc01f32b9e4cf2d3c630a4c5c18e6091615a07f30851be6c6eb2881`;
- float16 TFLite:
  `4e6f95610870597b74995f441c5cc755cdd2a555a5322504a919aef85f102c43`;
- tensor correspondence report:
  `a52ffab6c473547076538d993dbadd39305304521232b85dda74fae492f77322`.

The characterization also exposes an existing ownership mismatch rather than
a new regression: despite its Mul-const name, the raw helper claims valid
direct/direct chains before the adjacent general direct helper can see them.
The extraction must preserve this ordering and artifact behavior. It must use
one graph index per invocation, immutable plans with stale-contract rejection,
copy-on-write constants, explicit layout updates, bounded dispatch, and the
existing compatibility wrapper's final prune/report boundary. No production
fix has been applied at this point.

## Pre-Add Mul-const reshape suffix extraction: final checkpoint

The bounded owner is now implemented and connected before the unchanged raw
compatibility fallback. It indexes only Add candidates, shares one
`ModelIRGraphIndex` for the complete invocation, resolves immutable plans, and
revalidates every operator/tensor contract before apply. Typed permutations,
rank-four and rank-three views, reshape semantics, layout, dtype,
quantization, graph boundaries, producer order, and constant ownership are
validated before mutation. Shared channel constants and reshape-shape
constants use copy-on-write; exclusive constants retain their historical names
and update in place. Legacy NCHW consumers retain one dedicated adapter.
`LayoutState` is updated with every physical layout change, while pruning and
lineage-report grouping remain at the historical wrapper exit.

IAT-LLIE now records all 13 production rewrites in the indexed owner itself:
5, 4, and 4 across the three ordered prefix invocations. The fallback adds no
rewrite for those accepted candidates. Conversion-only output retains the
exact pre-fix hashes:

- float32 TFLite:
  `75e355ba8fc01f32b9e4cf2d3c630a4c5c18e6091615a07f30851be6c6eb2881`;
- float16 TFLite:
  `4e6f95610870597b74995f441c5cc755cdd2a555a5322504a919aef85f102c43`;
- tensor correspondence report:
  `a52ffab6c473547076538d993dbadd39305304521232b85dda74fae492f77322`.

The single sequential managed `-cotof` gate completed in 18.084 seconds with
`classification=pass`, maximum absolute error
`4.470348358154297e-07`, and `peak_swap_kib=0`. No timeout, converter failure,
accuracy regression, artifact difference, or SWAP exclusion was found.

### Changed files and design decisions

- `onnx2tf/tflite_builder/passes/pre_add_mulconst_reshape_suffix_layout.py`
  owns indexed candidate resolution, immutable transactional plans,
  differential graph updates, constant copy-on-write, layout reconciliation,
  bounded dispatch, and retained legacy adapters.
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py` invokes that owner first in
  the existing compatibility helper and supplies the Session `LayoutState` at
  the unchanged ordered prefix boundary.
- `tests/test_flatbuffer_direct_indexed_pre_add_mulconst_reshape_suffix_layout.py`
  covers direct/direct and direct/Mul-constant capability, closed and legacy
  graphs, shared constants, rejection atomicity, stale plans, bounded
  candidate dispatch, idempotence, index/layout consistency, and cleanup
  ownership.
- `tests/test_flatbuffer_direct_architecture.py` fixes the indexed-first
  ownership, differential-index policy, no-early-prune contract, and Session
  layout boundary.
- `docs/flatbuffer_direct_architecture.md` and this handoff record the
  semantic contract, pre-fix observation, evidence, and restart instructions.

The main compatibility decision is intentional: the new owner accepts
direct/direct as well as direct/Mul-constant inputs because the historical
helper already claims both families before the adjacent general direct helper.
Changing that ownership would alter pass timing and could alter artifacts.
Strict cases outside the new contract still use the raw fallback. No public
API, CLI default, artifact name, dependency, ONNX corpus model, managed
profile, exclusion rule, or optional TensorFlow boundary changed.

### Verification completed

- focused indexed owners, QLinear interactions, active suffix fixtures, and
  complete architecture suite: `257 passed in 44.99s`;
- TensorFlow import-blocked explicit direct, default direct, and direct
  `-cotof`: `3 passed in 4.53s`;
- dedicated owner during its focused run: `12 passed in 0.50s`;
- ordered owner/fixture integration during its focused run:
  `17 passed in 2.33s`;
- sequential IAT-LLIE managed `-cotof`: one pass, zero failure, zero SWAP;
- scoped Ruff, Python syntax compilation, and `git diff --check`: pass.

### Known issues and unfinished work

No new failing test or model regression is known at this checkpoint. The
whole-file inherited Ruff findings and optional exporter limitations recorded
earlier in this handoff remain unchanged. The optional TensorFlow suite was
not synchronized or executed; only its import-blocked direct boundary was
tested. No broad Tier run was performed because this checkpoint deliberately
used the only short real model with non-zero owner coverage. A later
post-commit 49-model measured-quick gate is recorded below and supersedes that
model-level limitation without changing source.

The original Goal remains incomplete: the fixed pipeline contract, remaining
raw layout/reshape helpers, op-family lowering consolidation, quantization and
split/crop refresh, requested-artifact matrix, optional TensorFlow exporters,
shared PyTorch/TorchScript/Dynamo/ExportedProgram package, and final Tier
0-through-5 performance/regression gates still require later checkpoints.

### Restart instruction

Confirm `fb-refactor5` is clean and synchronized with `origin/fb-refactor5`.
Then characterize the adjacent raw
`_optimize_transpose_pre_add_reshape_transpose_suffix_nhwc_chains` helper at
its unchanged production boundary. First establish its non-zero real-model
families and overlap with the newly indexed direct/direct owner; record any
problem before changing source. Extract only a bounded family with real
coverage, retain strict rejects on the compatibility path, and validate with
focused fixtures plus the smallest sequential zero-SWAP model set. Continue
to commit and push coherent units only; do not create a pull request.

## Post-checkpoint Tier 0-4 measured-quick regression gate

Commit `5b387bc6` was subsequently checked against the fixed 49-model Tier 0-4
measured-quick profile. The runner used `uv`, one converter/inference
subprocess at a time, a fixed model order, a 45-second per-model ceiling, and
subprocess-tree `VmSwap` monitoring. The broad run completed in 452.114
seconds. Every model recorded zero SWAP and no converter returned a nonzero
exit status.

The initial result contained 42 passes, the six preserved known non-passes,
and one new unconfirmed LINEA timeout. This issue was recorded in detail,
including all generated artifact hashes and the exact difference from the
prior 23.331-second pass, before any follow-up action. No source was changed.
One isolated 90-second-headroom run then restored LINEA to pass in 25.213
seconds with exact prior maximum absolute error `0.002297189086675644`, zero
SWAP, identical TFLite/report hashes, and identical pass metrics. The timeout
is therefore runtime variance rather than a semantic regression.

The final classification is zero newly confirmed regressions among the 49
selected models. IAT-LLIE, the only non-zero production model for the newly
indexed owner, passed in 17.404 seconds with maximum absolute error
`4.470348358154297e-07` and zero SWAP. DEIM retained its user-approved success
classification in 37.006 seconds. No source fix, exclusion, timeout-policy
change, or managed-profile update was necessary. Detailed and compact evidence
is stored in:

- `docs/flatbuffer_direct_quick_regression_2026-07-16.md`;
- `docs/baselines/flatbuffer_direct_quick_tier0_4_5b387bc6_result.json`;
- `docs/baselines/flatbuffer_direct_quick_tier0_4_5b387bc6_linea_followup.json`.

## Adjacent direct/direct reshape suffix helper: pre-change observation

The adjacent raw
`_optimize_transpose_pre_add_reshape_transpose_suffix_nhwc_chains` helper was
characterized at its unchanged production boundary before any source change.
The instrumentation counted both that helper and its immediately preceding
Mul-const compatibility owner at each of the three ordered prefix
invocations. All conversions used `uv`, ran sequentially, and generated only
the requested direct TFLite artifacts.

IAT-LLIE recorded 5, 4, and 4 rewrites in the preceding indexed/compatibility
owner, but 0, 0, and 0 in the adjacent direct/direct helper. Five additional
short zero-SWAP representatives
(`face_detection_yunet_2023mar_int8.onnx`, `FastestDet.onnx`,
`human_segmentation_pphumanseg_2021oct_org.onnx`,
`osnet025_Nx3x256x128.onnx`, and `sinet_320_op.onnx`) recorded zero rewrites in
both helpers at all invocations. A static topology query over the fixed
49-model Tier 0-4 measured-quick profile found only IAT-LLIE and LINEA with an
ONNX Add -> Reshape -> Transpose `[0,2,1]` suffix. LINEA then also recorded
0, 0, and 0 in the adjacent helper during an isolated successful conversion.

The observed issue is redundant work, not an accuracy regression. The
preceding compatibility helper deliberately accepts direct/direct as well as
direct/Mul-constant inputs. For a direct input its producer, permutation,
graph-output, and consumer guards are the same as the adjacent helper's
guards. Its suffix selection, legacy-user handling, shape-constant
copy-on-write, metadata permutation, adapter insertion, operator removal, and
final prune are also identical for that family. Because it runs first at the
same three boundaries, every candidate the adjacent helper could accept has
already been accepted and removed. The adjacent helper therefore performs
three repeated full producer/consumer-map builds and Add scans without owning
a reachable semantic family.

The safe change to evaluate is removal of only the unreachable private helper
and its three production invocations. The preceding indexed owner plus its raw
compatibility fallback must remain unchanged. The active direct/direct legacy
fixture, indexed owner tests, deterministic artifact hashes, TensorFlow import
blocker, and smallest real-model inference gate must pass before the removal
can be retained. No production fix has been applied at this point.

## Adjacent direct/direct reshape suffix helper: final checkpoint

The pre-change finding was confirmed and the unreachable private helper plus
its sole production call site were removed. This deletes 290 raw lowerer lines
and prevents the recovery prefix, which executes three times, from rebuilding
producer/consumer maps and scanning every Add for an already-owned family.
The preceding indexed owner and its compatibility fallback are unchanged, so
strict indexed rejects and all direct/direct or direct/Mul-constant behavior
remain at the historical ordered boundary. An architecture assertion now
prevents either the redundant definition or a call to it from returning.

The direct/direct legacy fixture and Mul-constant legacy fixture pass through
the remaining owner. The complete indexed owner plus architecture gate passes
223 tests in 45.73 seconds. TensorFlow-import-blocked explicit direct, default
direct, and direct `-cotof` pass 3 tests in 4.71 seconds. IAT-LLIE conversion
retains the exact fixed artifacts:

- float32 TFLite:
  `75e355ba8fc01f32b9e4cf2d3c630a4c5c18e6091615a07f30851be6c6eb2881`;
- float16 TFLite:
  `4e6f95610870597b74995f441c5cc755cdd2a555a5322504a919aef85f102c43`;
- tensor correspondence report:
  `a52ffab6c473547076538d993dbadd39305304521232b85dda74fae492f77322`.

The single sequential managed IAT-LLIE accuracy gate passes in 20.335 seconds
with maximum absolute error `4.470348358154297e-07`, converter exit code 0,
and `peak_swap_kib=0`. No model exclusion, timeout-policy change, dependency,
public API, CLI behavior, generated corpus artifact, optional TensorFlow
boundary, or managed profile changed. No failure was found and therefore no
additional corrective source change was required. Scoped architecture Ruff,
Python syntax compilation, and `git diff --check` pass. Whole-file Ruff on the
lowerer continues to report the same ten inherited F841 findings as the parent
commit; this deletion introduces no new lint finding.

### Restart instruction after this checkpoint

After confirming the branch is clean and synchronized, characterize the next
raw `_optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains`
helper at its unchanged three production boundaries. Use the fixed short,
zero-SWAP corpus first and locate a real non-zero family before extracting
anything. Record any ownership conflict, artifact change, accuracy failure, or
SWAP before changing source. Keep conversion checks minimal, inference
strictly sequential, and continue with coherent commit/push units only; do not
create a pull request.

## Pre-unary reshape suffix extraction: pre-change observation

The next raw helper was instrumented at its unchanged production boundary
before source modification. Seven conversion-only checks ran sequentially
under `uv`. IAT-LLIE, Face Detection YuNet int8, FastestDet, HumanSeg, OSNet,
and SiNet recorded 0, 0, and 0 rewrites. LINEA recorded one rewrite at the
first invocation and zero at the next two. Its exact changed subgraph is:

```text
node_conv2d_98_output_nhwc
  -> TRANSPOSE [0,3,1,2] -> conv2d_98
  -> LOGISTIC -> val_855
  -> MUL(conv2d_98, val_855) -> silu_49
  -> RESHAPE(view_16_reshape_shape) -> view_16
  -> TRANSPOSE [0,2,1] -> permute_7
```

The helper rewrites this generic Swish suffix to operate directly on
`node_conv2d_98_output_nhwc`, changes the Reshape output to `permute_7`, and
removes both Transposes. The first invocation changes the operator count from
1,839 to 1,837. The current raw implementation rebuilds complete producer and
consumer maps for each fixed-point iteration, scans every Reshape, searches
operator identity to rediscover removal indices, mutates ModelIR directly,
and does not update the Session `LayoutState`. The issue is architectural and
performance-related; no LINEA accuracy regression is currently known.

The exact pre-change artifacts are fixed as follows:

- float32 TFLite:
  `fd91ea915b600b3581e8e0e68925fefd5302cd1bfb373ebca8b9b9410138c611`;
- float16 TFLite:
  `c8e44a48221eeead187869d93dfef1f7775420335aae5c63873118738d39f9a8`;
- tensor correspondence report:
  `ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`.

The fixed sequential LINEA follow-up baseline passes with maximum absolute
error `0.002297189086675644` and zero process-tree SWAP. The safe extraction
scope is only the non-zero generic Swish family. It must use one differential
`ModelIRGraphIndex`, immutable plans with full revalidation before apply,
typed constant and shape/layout guards, explicit `LayoutState` updates,
bounded dispatch, and no internal prune. Plain unary cases and all strict
rejects must remain on the raw compatibility fallback. No production fix has
been applied at this point.

### First indexed implementation report-contract finding

The initial strict owner accepted exactly the intended LINEA Swish candidate
at counts 1, 0, and 0. Both TFLite files remained byte-identical, but the
correspondence report changed from
`ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`
to `20f9736c4118fcfad42e35fe786783d2d9dec587012a4b55ddd62747cc7ed6e5`.
This problem was isolated and recorded before corrective source work.

The report summaries, all 920 records, all 579 lineage-event positions, and
578 lineage events are identical. Only event 269 differs. The historical raw
helper records the Mul data-edge rewrite with source
`replace_operator_input_at`; the first indexed implementation used
`set_operator_inputs`. The semantic edge is identical in both reports:
`conv2d_98 -> node_conv2d_98_output_nhwc`. This is a report-source-label
compatibility regression, not a graph or accuracy difference. The bounded
correction is to use the graph-index-aware single-slot mutation helper for
that one Mul data edge, matching the historical lineage contract without
altering any other mutation.

## Pre-unary reshape suffix extraction: final checkpoint

The bounded indexed owner is implemented in
`pre_unary_reshape_suffix_layout.py` and connected at the unchanged wrapper
boundary. It dispatches only indexed Mul candidates and reuses one
`ModelIRGraphIndex` per invocation. Candidate resolution requires the generic
Swish topology, typed `[0,3,1,2]` and `[0,2,1]` constants, exact positive
rank-four NHWC/NCHW views, exact rank-three NCW/NWC reshape semantics,
compatible dtype and per-tensor quantization, graph ordering and boundaries,
exclusive data edges, and an exclusive typed reshape constant. Accepted
candidates become immutable plans with complete tensor/operator contracts and
are fully revalidated before apply. Mutation is differential, bounded, and
updates the Session `LayoutState`; the raw wrapper remains the only prune and
lineage cleanup boundary. Plain unary paths and every strict reject remain on
the original fallback.

The recorded lineage-source incompatibility was corrected with the
graph-index-aware single-slot input helper. LINEA now records indexed owner
counts 1, 0, and 0, while the compatibility fallback adds no rewrite for the
accepted candidate. All fixed artifacts are restored exactly:

- float32 TFLite:
  `fd91ea915b600b3581e8e0e68925fefd5302cd1bfb373ebca8b9b9410138c611`;
- float16 TFLite:
  `c8e44a48221eeead187869d93dfef1f7775420335aae5c63873118738d39f9a8`;
- tensor correspondence report:
  `ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`.

The single sequential managed LINEA `-cotof` gate passes in 23.336 seconds
with maximum absolute error `0.002297189086675644`, converter exit code 0,
and `peak_swap_kib=0`. The indexed owner, preceding suffix owner, active
compatibility fixtures, and complete architecture suite pass 230 tests in
45.57 seconds. The dedicated owner and wrapper cleanup suite passes 6 tests;
TensorFlow-import-blocked explicit direct, default direct, and direct `-cotof`
pass 3 tests in 4.53 seconds. Scoped Ruff, Python syntax compilation, and
`git diff --check` pass. The same ten inherited lowerer F841 findings remain
unchanged.

### Changed files and restart instruction

- `onnx2tf/tflite_builder/passes/pre_unary_reshape_suffix_layout.py` owns the
  strict indexed Swish family, immutable plans, differential mutation, and
  rank-aware layout reconciliation.
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py` invokes the owner before the
  unchanged raw fallback, supplies Session layout state, and preserves one
  cleanup/report boundary.
- `tests/test_flatbuffer_direct_indexed_pre_unary_reshape_suffix_layout.py`
  covers capability, atomic rejection, bounded candidate dispatch, stale
  plans, determinism, graph-index/layout consistency, exact lineage source,
  and wrapper cleanup.
- `tests/test_flatbuffer_direct_architecture.py` fixes indexed-first ownership,
  bounded dispatch, no internal prune/full-map build, and the production
  layout-state boundary.
- `docs/flatbuffer_direct_architecture.md` and this handoff record the design,
  pre-change evidence, report-contract problem before correction, final
  evidence, and remaining work.

No public API, CLI behavior, dependency, managed profile/exclusion, timeout
policy, ONNX corpus model, or optional TensorFlow exporter changed. No broad
Tier run was performed because LINEA is the only established non-zero model
for this bounded family and the preceding fixed 49-model checkpoint already
confirmed zero new regressions.

After confirming the branch is clean and synchronized, characterize the next
raw `_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains` helper at
its unchanged production boundary. Establish a real non-zero family before
source work, record any problem before correction, and retain strict rejects
on the compatibility path. Continue with minimal sequential zero-SWAP model
checks and coherent commit/push units only; do not create a pull request.

## YOLO factorized expand-dims extraction: pre-change observation

The raw `_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains`
helper was instrumented at all four unchanged production invocations before
source modification. Fourteen conversion-only models ran sequentially under
`uv` with subprocess-tree `VmSwap` monitoring. Every conversion exited zero
and every model recorded `peak_swap_kib=0`. Twelve models recorded zero
rewrites throughout. `yolov7-tiny.onnx` and `yolo_test.onnx` each recorded
3, 0, 0, and 0 rewrites.

All six non-zero chains are the generic factorized YOLO Case B:

```text
NHWC -> TRANSPOSE [0,3,1,2] -> NCHW
     -> RESHAPE [N,A,B,H,W]
     -> TRANSPOSE [0,1,3,4,2] -> [N,A,H,W,B]

NHWC -> RESHAPE [N,H,W,A,B]
     -> TRANSPOSE [0,3,1,2,4] -> [N,A,H,W,B]
```

Both models use `A=3` at spatial sizes 80, 40, and 20. YOLOv7-tiny uses
`B=85`; yolo_test uses `B=8`. The first invocation removes three pre-layout
Transposes, rewrites three Reshape inputs, shapes, and options, and remaps the
three retained post-Transpose permutations. The next three invocations own no
additional chain.

The current raw Case B implementation lacks an explicit transactional
contract. It checks the post permutation value early enough that its normal
single-threaded path does not leave the Reshape half-written, but it mutates
both the Reshape shape and post-permutation constants in place without
checking exclusive ownership. A shared constant can therefore change an
unrelated consumer. The helper also rebuilds full consumer maps and scans all
Transposes seven times for a three-chain model: four fixed-point scans in the
first invocation plus one no-op scan at each later invocation. Constant type
and ownership, layouts, dtype, quantization, operator ordering, and graph
boundaries are not validated as one immutable contract.

The exact pre-change artifacts are:

- yolo_test float32:
  `439d9a8b893bf6bfbd92aa0155bd15a4185b5fcdb6e65ddb48f718a41b75bdfc`;
- yolo_test float16:
  `7b1ef8b13de65068b3fe8166d5481553e2e41194c0cfe9ee48f4be5ad3417eff`;
- yolo_test correspondence:
  `36d728e9294f1d4f1319c45306a088bced6b54ad393f71f4925f3178f0d9c1ca`;
- YOLOv7-tiny float32:
  `4738ec36f18f3ccfcf3d53e7b43a59091ea75b88a9620460e95c724bf363326e`;
- YOLOv7-tiny float16:
  `1675ba6a669e22f0e7cf24941c421c9d3241c837c681d7a97c465a5665a536bd`;
- YOLOv7-tiny correspondence:
  `c6667fcef416fafe7eed113ac43a6df4cab85c6c475006bbd86a0ad01de4ffb8`.

The pre-change sequential yolo_test `-cotof` gate passes in 10.489 seconds
with maximum absolute error `2.4437904357910156e-06`, converter exit code 0,
and zero SWAP. The safe extraction scope is only Case B. It must validate both
typed constants and all tensor/operator/layout contracts before any mutation,
use an immutable revalidated plan and one differential graph index, update
Session layout state, preserve the single wrapper prune/report boundary, and
leave singleton Case A plus every strict reject on the raw compatibility
fallback. No production fix has been applied at this point.

### First indexed Case B implementation guard finding

The first focused owner run rejected four valid synthetic Case B expectations;
the two historical Case A fallback fixtures and shared-constant atomicity still
passed. This was recorded before correction. The cause is bounded and does not
affect the raw production result: the initial owner reused a typed permutation
utility whose contract is deliberately rank four, so the valid rank-five post
permutation `[0,1,3,4,2]` could never pass its guard. The indexed owner
therefore returned zero and the unchanged fallback continued to own real
models. The correction is a local typed permutation reader parameterized by
the exact expected vector length, while retaining the stricter exclusive
mutable-vector guard for the post constant before apply.

## YOLO factorized expand-dims extraction: final checkpoint

The strict generic Case B owner is implemented in
`expanddims_reshape_layout.py` and connected at both historical call sites,
which execute at four production boundaries. It builds one
`ModelIRGraphIndex` per invocation and dispatches each indexed Transpose
candidate at most once. Resolution validates the exact factorized topology,
typed rank-four and rank-five permutations, positive static rank-four and
rank-five views/signatures, `A > 1`, `C=A*B`, dtype and per-tensor
quantization, graph boundaries and order, exclusive edges, and exclusive
typed mutable constants. The immutable plan records all tensor/operator
contracts and is fully re-resolved before differential apply. Session layout
state is updated, while pruning and report grouping remain at the single raw
wrapper boundary.

The raw compatibility path still owns singleton Case A and all relaxed
rejects. Its Case B mutation now validates both constant values and rejects a
shared Reshape-shape or post-permutation constant before any write, preventing
an unrelated consumer from observing a layout-specific in-place update. The
two historical Case A fixtures remain unchanged.

Both production models record all accepted rewrites in the indexed owner:
3, 0, 0, and 0 for `yolo_test.onnx`, and 3, 0, 0, and 0 for
`yolov7-tiny.onnx`. The raw fallback adds zero for those accepted candidates.
All six fixed artifacts remain exact:

- yolo_test float32:
  `439d9a8b893bf6bfbd92aa0155bd15a4185b5fcdb6e65ddb48f718a41b75bdfc`;
- yolo_test float16:
  `7b1ef8b13de65068b3fe8166d5481553e2e41194c0cfe9ee48f4be5ad3417eff`;
- yolo_test correspondence:
  `36d728e9294f1d4f1319c45306a088bced6b54ad393f71f4925f3178f0d9c1ca`;
- YOLOv7-tiny float32:
  `4738ec36f18f3ccfcf3d53e7b43a59091ea75b88a9620460e95c724bf363326e`;
- YOLOv7-tiny float16:
  `1675ba6a669e22f0e7cf24941c421c9d3241c837c681d7a97c465a5665a536bd`;
- YOLOv7-tiny correspondence:
  `c6667fcef416fafe7eed113ac43a6df4cab85c6c475006bbd86a0ad01de4ffb8`.

The final single sequential yolo_test `-cotof` gate passes in 11.264 seconds
with maximum absolute error `2.4437904357910156e-06`, converter exit code 0,
and `peak_swap_kib=0`. The three indexed suffix owners, active compatibility
fixtures, and complete architecture suite pass 241 tests in 44.41 seconds.
The final dedicated Case B plus Case A compatibility suite passes 9 tests in
0.74 seconds. TensorFlow-import-blocked explicit direct, default direct, and
direct `-cotof` pass 3 tests in 4.48 seconds. Scoped Ruff, Python syntax
compilation, and `git diff --check` pass; the same ten inherited lowerer F841
findings remain unchanged.

### Changed files and restart instruction

- `onnx2tf/tflite_builder/passes/expanddims_reshape_layout.py` owns strict
  factorized Case B resolution, immutable revalidation, differential mutation,
  constant ownership, bounded dispatch, and layout reconciliation.
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py` invokes the owner at the two
  existing call sites, passes Session layout state, retains singleton Case A,
  and protects shared constants before the compatibility mutation.
- `tests/test_flatbuffer_direct_indexed_expanddims_reshape_layout.py` covers
  capability, Case A rejection/fallback, shared constants, atomic stale-plan
  rejection, candidate bounds, determinism, exact lineage, graph-index/layout
  consistency, and wrapper cleanup.
- `tests/test_flatbuffer_direct_architecture.py` fixes indexed-first ownership,
  single-pass bounded dispatch, no internal prune/full-map build, and both
  production layout-state boundaries.
- `docs/flatbuffer_direct_architecture.md` and this handoff record the
  characterization, pre-change ownership issue, first implementation guard
  finding before correction, final design, evidence, and restart order.

No public API, CLI behavior, dependency, managed profile/exclusion, timeout
policy, ONNX corpus model, or optional TensorFlow exporter changed. No broad
Tier run was repeated because the fixed 49-model checkpoint already reports
zero new regressions and this extraction used the only two established
non-zero, short, zero-SWAP models.

After confirming the branch is clean and synchronized, characterize the next
raw `_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains` helper at
all unchanged production boundaries. Record real non-zero ownership and any
problem before source work, keep strict rejects on the compatibility path,
and use only the smallest sequential zero-SWAP model gate. Continue with
coherent commit/push units only; do not create a pull request.

## Flatten-HW reshape suffix checkpoint: pre-change record

Before changing source, the raw
`_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains` helper was
instrumented at all four unchanged production invocations. Fourteen previously
measured short Tier 0-4 representatives were converted strictly sequentially:
IAT-LLIE, LINEA, YuNet INT8, FastestDet, PPHumanSeg, OSNet, SiNet,
Tiny-YOLOv2, YOLOv7-tiny, DAMO-YOLO, NanoDet, YOLOX INT8, YOLO-Free, and
yolo_test. All fourteen conversions exited 0, none timed out, and every
process-tree monitor recorded `peak_swap_kib=0`. Individual conversion times
were 1.793-8.535 seconds.

Only LINEA establishes real ownership. Its four invocation counts are
`2, 0, 0, 0`: the first prefix invocation collapses two exact
`NHWC -> NCHW -> [N,C,H*W] -> [N,H*W,C]` chains, with final shapes
`[1,6400,128]` and `[1,1600,128]`. The other thirteen models record zero at
every invocation. The unchanged pre-extraction LINEA artifacts are fixed as:

- float32: `fd91ea915b600b3581e8e0e68925fefd5302cd1bfb373ebca8b9b9410138c611`;
- float16: `c8e44a48221eeead187869d93dfef1f7775420335aae5c63873118738d39f9a8`;
- correspondence: `ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`.

No observed conversion or accuracy regression prompted this checkpoint.
However, the two proven rewrites are currently owned by an unbounded
`while True` full-operator scan that rebuilds the whole consumer map after
each rewrite. The helper does not use the shared graph index or Session
`LayoutState`, accepts reshape/permutation constants without typed ownership
contracts, and mutates the reshape constant in place even when another
operator or a graph boundary can observe it. It also does not revalidate an
immutable plan immediately before mutation. These are recorded latent
correctness, interaction, and efficiency problems before implementation.
The safe scope is therefore only LINEA's exact static family; relaxed or
shared-constant variants must remain unchanged on a compatibility fallback.

### Implemented result and verification

`flatten_hw_reshape_layout.py` now owns that exact static family. It dispatches
the indexed Transpose set once, validates typed `[0,3,1,2]` and `[0,2,1]`
constants, exact positive shape/signature views, dtype and per-tensor
quantization, operator order, graph boundaries, exclusive data edges, an
exclusive typed reshape constant, and Session layout compatibility. Complete
operator/tensor contracts are stored in an immutable plan and resolved again
before apply. Apply uses graph-index-aware input/output mutation, removes both
Transposes differentially, and reconciles `LayoutState`; the owner neither
prunes nor builds whole-graph maps. The wrapper keeps the sole compatibility
prune and raw fallback. Shared, boundary-visible, produced, or variable shape
constants are rejected before the fallback can mutate them.

All four LINEA boundaries now report indexed counts `2, 0, 0, 0`; the wrapper
totals are the same, proving no residual raw rewrite for accepted candidates.
The post-extraction artifacts are byte-for-byte identical to the pre-change
files and retain the three hashes recorded above. The single sequential
managed LINEA `-cotof` gate exits 0 in 21.914 seconds, reports maximum absolute
error `0.002297189086675644` with `pass=True`, and records
`peak_swap_kib=0`. No new regression or exclusion was found.

A mechanical context patch made while hardening the fallback initially placed
the local `model_inputs` snapshot in an earlier legacy loop rather than this
wrapper. Immediate targeted diff inspection found the misplaced line and the
undefined target reference before conversion execution; both were corrected,
then the dedicated suite was rerun. This did not reach a committed or tested
checkpoint and produced no artifact difference.

Verification completed under `uv`:

- 14 sequential short Tier 0-4 characterization conversions: all exit 0,
  1.793-8.535 seconds each, no timeout, and SWAP 0;
- focused owner plus architecture selectors: 10 passed;
- the four adjacent reshape-suffix owners, architecture suite, active
  compatibility selectors, and TensorFlow-import-blocked explicit direct,
  default direct, and direct `-cotof`: 238 passed, 773 deselected in 47.59
  seconds;
- scoped Ruff, syntax compilation, and `git diff --check`: passed;
- whole-lowerer Ruff retains exactly the same ten inherited F841 findings and
  reports no new finding.

Changed files in this checkpoint are
`onnx2tf/tflite_builder/passes/flatten_hw_reshape_layout.py`,
`onnx2tf/tflite_builder/lower_from_onnx2tf.py`,
`tests/test_flatbuffer_direct_indexed_flatten_hw_reshape_layout.py`,
`tests/test_flatbuffer_direct_architecture.py`,
`docs/flatbuffer_direct_architecture.md`, and this handoff. No public API,
CLI behavior, dependency, corpus profile/exclusion, timeout policy, or
optional TensorFlow boundary changed.

After confirming this checkpoint is clean and synchronized, characterize the
next raw
`_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains`
helper at each unchanged production boundary. Record any non-zero model and
problem before source work, retain relaxed cases on fallback, and use only the
smallest sequential zero-SWAP accuracy gate. Continue with coherent
commit/push units only; do not create a pull request.

## Rank-3 to NHWC reshape shim: measured no-change decision

The raw
`_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains` helper
was instrumented at all four unchanged production invocations before any
source change. The same fourteen short Tier 0-4 representatives used for the
preceding checkpoint all converted successfully and sequentially in
1.788-8.917 seconds, with no timeout and `peak_swap_kib=0`. Every invocation
count was zero. A read-only ONNX topology scan then found zero exact
`Reshape -> Transpose[0,3,2,1] -> Reshape -> Transpose[0,2,3,1]` candidates in
both the fixed 49-model measured-quick set and all 420 active Tier 0-4 models.

The helper still contains an unbounded rewrite loop, repeated full consumer
map construction, raw constant copy-on-write, and no graph-index or Session
layout-state contract. These are architectural liabilities, but there is no
real current-corpus ownership evidence from which to freeze artifact and
accuracy behavior. Removing it would also discard an existing feature.
Accordingly, no source or test behavior was changed and no synthetic-only
replacement was introduced. Revisit this helper only when a real non-zero
model or a public compatibility fixture is available. The next evidence-first
candidate is
`_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains`.

## QKV reshape/transpose recovery checkpoint: pre-change record

Before changing source, the raw
`_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains`
helper was instrumented at all four production invocations. The fourteen short
zero-SWAP representatives and four additional Tier 0/2 static-topology
candidates all converted sequentially, exited 0, and recorded zero rewrites.
The additional candidates were ViT-B/16 partial, both CRNN English recognition
models, and monodepth; their conversions took 2.118-3.681 seconds with SWAP 0.

A shape-aware read-only scan of all 420 active Tier 0-4 ONNX graphs narrowed
the exact fully static family to htdemucs and three RF-DETR variants. htdemucs
was not converted because it is an explicit managed exclusion. The smallest
active RF-DETR representative, Tier 3 `rf-detr-nano.onnx` (770 ONNX nodes),
then converted in 10.890 seconds with exit code 0 and `peak_swap_kib=0`. Its
four raw invocation counts are `5, 0, 0, 0`; all five accepted chains use
`[300,1,256] -> [300,8,32] -> [8,300,32] -> [1,8,300,32]` with permutation
`[1,0,2]`. No `[1,2,0]` production owner has been established. The fixed
pre-extraction artifacts are:

- float32: `fda7d97eaad2b19ee2ac31411099067e78b747515952b7c65ba52a0f1454f1fb`;
- float16: `a80051b2d6bb871ee871f0d1528e1ea7c8d4e7f6ecbfc16daec4fa78d696fd1f`;
- correspondence: `262235cec5a8df73ff2afd7f1eb28678cc7312f4a19dd09d278fd8db77cbdec4`.

No observed regression prompted the change. The proven raw owner nevertheless
has a correctness hazard that is recorded before implementation: it updates
or clones the first Reshape shape constant before attempting the Transpose
permutation constant. If the second operation fails, the first mutation and
possibly its lineage event remain, so the rewrite is not transactional. The
helper also repeats a full consumer-map build after every rewrite, uses an
unbounded whole-RESHAPE scan, accepts untyped/produced/variable/boundary
constants, and does not update the shared graph index or Session
`LayoutState`. The safe implementation scope is the exact RF-DETR `[1,0,2]`
family with fully typed, exclusively mutable constants and immutable
revalidation. The unproven `[1,2,0]`, shared-constant copy-on-write, dynamic,
and otherwise relaxed cases must retain compatibility fallback behavior.

## QKV reshape/transpose recovery checkpoint: completed

The strict production-proven family is now owned by
`onnx2tf/tflite_builder/passes/attention_qkv_reshape_layout.py`. It dispatches
only indexed Reshape candidates, validates the exact positive static
`[A,1,C] -> [A,H,D] -> [H,A,D] -> [1,H,A,D]` contract with `C=H*D` and
permutation `[1,0,2]`, and requires typed exclusive mutable constants, matching
dtype/per-tensor quantization, resolved sources, exclusive edges, operator
order, graph boundaries, and UNKNOWN Session layout. Its immutable plan
captures tensor and operator contracts and is re-resolved before any mutation.
Apply changes the shared graph index differentially, preserves the historical
output lineage event, removes only the tail Reshape, and explicitly reconciles
`LayoutState`. Candidate and rewrite bounds are deterministic; the owner has
no internal prune, repeated producer/consumer-map construction, or unbounded
loop.

The lowerer wrapper invokes this indexed owner first and retains the unchanged
raw implementation for `[1,2,0]`, shared-constant copy-on-write, dynamic, and
otherwise relaxed compatibility cases. It remains the only prune boundary and
removes the corresponding stale layout entries. Both production boundaries
now supply the Session `LayoutState`. Focused fixtures prove indexed
differential mutation and lineage, wrapper cleanup, strict candidate/bound
handling, stale-plan atomic rejection, determinism, and the three named raw
fallback classes.

Post-change `rf-detr-nano.onnx` instrumentation records indexed invocation
counts `5, 0, 0, 0` and identical wrapper totals, so the raw residual is zero.
The conversion exited 0 in 12.121 seconds with `peak_swap_kib=0`. All three
outputs are byte-identical to the fixed pre-change baseline:

- float32: `fda7d97eaad2b19ee2ac31411099067e78b747515952b7c65ba52a0f1454f1fb`;
- float16: `a80051b2d6bb871ee871f0d1528e1ea7c8d4e7f6ecbfc16daec4fa78d696fd1f`;
- correspondence: `262235cec5a8df73ff2afd7f1eb28678cc7312f4a19dd09d278fd8db77cbdec4`.

Its sequential managed `-cotof` run exited 0 in 41.259 seconds, used no SWAP,
and passed all output checks with maximum absolute error
`0.000102996826171875`, RMSE approximately `1.01567e-05`, and cosine
similarity 1. The unchanged artifacts produced during that accuracy run have
the same three hashes.

Validation completed for this checkpoint:

- the three default/direct/`-cotof` TensorFlow import-blocker tests: `3 passed`;
- the selected adjacent indexed-owner, direct-builder, optional-TensorFlow,
  and architecture suite: `247 passed, 773 deselected`;
- focused QKV/architecture validation before the combined gate:
  `11 passed, 210 deselected`;
- scoped Ruff for the new owner and its tests: clean;
- `py_compile` for the new owner, lowerer, and related tests: clean;
- `git diff --check`: clean.

Whole-file Ruff for the legacy lowerer still reports the same ten inherited
`F841` unused-local findings at lines outside this change. No new finding was
introduced. No conversion failure, timeout, SWAP event, artifact drift,
TensorFlow import, or accuracy regression is known for this checkpoint.

Changed files in this checkpoint are:

- `onnx2tf/tflite_builder/passes/attention_qkv_reshape_layout.py` (new strict
  indexed owner);
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py` (owner dispatch, Session
  layout propagation, and compatibility/prune boundary);
- `tests/test_flatbuffer_direct_indexed_attention_qkv_reshape_layout.py` (new
  unit and compatibility tests);
- `tests/test_flatbuffer_direct_architecture.py` (bounded owner/wiring
  constraints);
- `docs/flatbuffer_direct_architecture.md` and this handoff (design and measured
  evidence).

The preceding rank-3-to-NHWC reshape helper intentionally remains unchanged
because neither the 49-model measured-quick set nor all 420 active Tier 0-4
graphs established a real owner. The broader Goal also remains incomplete;
quantization/split/exporter extraction and the remaining raw layout helpers
have not been completed. At resume, first inspect the next adjacent raw helper,
`_optimize_attention_gather_transpose_reshape_cleanup_chains`, characterize
all unchanged production invocations with short zero-SWAP models, and record
any non-zero owner or problem before source changes. Continue with sequential
inference only, minimal conversion gates, coherent commit/push units, and no
pull request.

## Current-HEAD 49-model measured-quick gate: pre-exclusion record

Commit `0e666234` was subsequently exercised with the fixed 49-model Tier 0-4
measured-quick profile. This was a strictly sequential `uv` run with a
45-second per-model ceiling and subprocess-tree SWAP monitoring. It completed
all entries in 503.563 seconds with 42 passes, six preserved known non-passes,
and one timeout. Every model recorded `peak_swap_kib=0`; no converter returned
a nonzero exit status. All classifications and available numeric errors match
the stable `f5a40947` result except HybridNets. LINEA, IAT-LLIE, and DEIM all
passed with their established results.

Before any follow-up change, the HybridNets issue is fixed in detail in
`docs/baselines/flatbuffer_direct_quick_tier0_4_0e666234_result.json` and the
quick-regression report. It timed out at 45.115 seconds after writing all three
normal conversion artifacts. The float32 artifact is byte-identical to the
prior successful diagnostic, whose maximum absolute error was
`0.0002593994140625`, and SWAP remained zero. This is the model's second
45-second quick-ceiling timeout, so it is runtime variance rather than a
semantic regression but is no longer reliably short. The next action is a
profile-only exclusion with `repeated_quick_ceiling_timeout`, as required by
the user timeout-exclusion policy. Do not modify converter source or rerun the
model before applying that policy update.

The pre-change evidence above was committed and pushed as `c3cab627`. Only
after that checkpoint, HybridNets was changed to `excluded` with reason
`repeated_quick_ceiling_timeout` in both the measured-quick and managed Tier
0-4 profiles. Active counts are now 48/49 and 381/420 respectively. Exact
profile contract tests cover the entry and aggregate counts. No converter
source, timeout, artifact, or accuracy policy changed, and no additional model
conversion was run. The fixed active quick set therefore has zero newly
confirmed semantic regressions: 42 passes and six preserved known non-passes,
all with zero SWAP.

## Attention Gather/Transpose/Reshape cleanup: measured no-change decision

The raw `_optimize_attention_gather_transpose_reshape_cleanup_chains` helper
was instrumented at every unchanged production invocation before any source
change. The same fourteen short Tier 0-4 representatives used by the adjacent
reshape checkpoints all converted sequentially, exited 0 in 1.870-8.709
seconds, and recorded `peak_swap_kib=0`. Every Pattern A and Pattern B rewrite
count was zero.

A read-only ONNX scan then inspected all 381 active managed Tier 0-4 models.
It found eleven graphs with the loose `Gather -> Transpose/Gather -> Reshape`
topology, but zero candidates satisfying the helper's complete axis, zero
index, permutation, static shape, and reshape-target contract at the ONNX
boundary. The scan itself used no SWAP. The already bounded RF-DETR
representative was converted because preceding layout passes could still have
created the target contract; it exited 0 in 11.473 seconds with zero rewrites
and SWAP 0. The closest attention-specific topology candidate,
`mod_dn_dab_detr.onnx`, likewise exited 0 in 20.116 seconds with zero rewrites
and SWAP 0. The known approximately 267-second `new_encoder.onnx` path was not
run, consistent with the timeout/short-validation policy.

The raw helper still contains an unbounded rewrite loop, repeated whole-graph
consumer-map construction, raw constant mutation/copy-on-write, direct list
deletion, and no shared graph-index or Session layout-state contract. Those
are real architectural liabilities, but the current corpus provides no real
owner from which artifact, lineage, and accuracy behavior can be frozen.
Removing the helper would discard an existing compatibility feature, while a
synthetic-only indexed replacement would not establish production ownership.
Accordingly no source or test behavior is changed. Revisit it only when a
non-zero production model or public compatibility fixture exists. The next
evidence-first raw helper is
`_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains`; the
intervening axis-0 singleton Gather rewrite is already delegated to its pass
module.

## Attention pre-projection rank lift: measured no-change decision

The raw
`_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains` helper was
then instrumented at every unchanged production invocation. All fourteen short
Tier 0-4 representatives again exited 0 sequentially in 1.916-8.677 seconds,
with zero rewrites and `peak_swap_kib=0`. Several models retained loose
Reshape-to-BatchMatMul candidates, confirming that the instrumentation observed
the intended boundary rather than an empty operator class, but none satisfied
the complete fan-out, binary, and tail-reshape contract.

A read-only scan of all 381 active Tier 0-4 ONNX graphs found 69 models with a
loose Reshape-to-MatMul fan-out and zero exact static contracts. It recorded no
SWAP. Two attention-specific DAB-DETR candidates were nevertheless converted
because preceding layout passes could create the ModelIR-only rank contract.
`mod_dn_dab_detr.onnx` exited 0 in 26.006 seconds and
`dn_dab_detr_480x480_div.onnx` exited 0 in 11.822 seconds; both recorded zero
rewrites at all four invocations and zero SWAP. The larger Tier 4 sibling was
not run because no smaller variant established ownership.

This helper also retains an unbounded rewrite loop, repeated whole-graph map
construction, direct list deletion, in-place metadata changes across all
fan-out branches, and no GraphIndex/LayoutState transaction. With no real
non-zero owner, there is no defensible production digest or accuracy baseline
for an indexed extraction. The compatibility implementation therefore remains
unchanged and is not replaced by synthetic-only logic. The next raw helper in
the ordered recovery prefix is
`_optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains`; the two
window transforms between them are already indexed pass-module delegates.

## Pre-unary Squeeze suffix checkpoint: pre-change record

Before changing source, the raw
`_optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains` helper was
instrumented at every unchanged production invocation. The fourteen short
Tier 0-4 representatives all exited 0 sequentially in 1.946-8.866 seconds,
with zero rewrites, no timeout, and SWAP 0. A read-only scan of all 381 active
Tier 0-4 graphs found ten loose
`Transpose -> unary/Swish -> Squeeze -> Transpose` topologies and zero exact
contracts at the ONNX boundary; the scan used no SWAP.

The smallest loose candidate, Tier 1 `inference_ops15.onnx` (191 ONNX nodes),
was then converted because preceding layout passes can establish the physical
shape contract. It exited 0 in 2.526 seconds with invocation counts `1, 0, 0`
and `peak_swap_kib=0`. The sole rewrite is the exact Swish family:

`[1,1,40,64] -> Transpose[0,3,1,2] -> [1,64,1,40] ->
Logistic/Mul -> Squeeze(axis=2) -> [1,64,40] ->
Transpose[0,2,1] -> [1,40,64]`.

Both permutation constants are typed INT32 vectors, all data tensors are
FLOAT32 with fixed equal signatures, per-tensor quantization is absent, every
observed layout is UNKNOWN, and the data edges have the expected exclusive
Swish/Squeeze/post consumers. The fixed pre-extraction artifacts are:

- float32: `3ce4af63727dd927666f09bb51555ccfd60e1cf01b4ba7fc674170e8277b9a96`;
- float16: `ee97304641e2b1330bbbe1f1472fc32a4a4d41d4bdb08a3e660da64b5204ce47`;
- correspondence: `a50f21319df0380165e8fee2c47f679ccb1682eee965fbd3b0f05ad02cc3d276`.

The managed baseline maximum absolute error is
`3.0994415283203125e-06`. No regression prompted this extraction. The proven
raw owner nevertheless uses an unbounded full-Squeeze scan, rebuilds both
producer and consumer maps after each rewrite, mutates operator inputs,
options, outputs, and tensor metadata without an immutable revalidation
transaction, deletes operators directly, and does not reconcile the shared
GraphIndex or Session `LayoutState`. The safe extraction scope is only the
exact static Swish/axis-2 family above with typed permutations, complete
shape/signature/dtype/quantization/layout contracts, graph boundaries,
exclusive slots, deterministic bounds, and stale-plan rejection. Plain unary,
axis-3, dynamic, shared/relaxed constants, and every strict reject must remain
on the existing compatibility fallback.

## Pre-unary Squeeze suffix checkpoint: completed extraction

The strict production family above is now owned by the bounded indexed pass
`pre_unary_squeeze_suffix_layout.py`. It dispatches indexed Mul candidates and
requires the exact typed permutations, static views and signatures,
dtype/per-tensor quantization, graph boundaries, operator order, exclusive
consumer slots, and Session layout contract measured before the change. Each
candidate is captured as an immutable tensor/operator plan and fully resolved
again immediately before apply. Input/output mutations and pre/post Transpose
removal update one `ModelIRGraphIndex` differentially, and layout metadata is
updated together with `LayoutState`. Malformed Squeeze axes are a strict no-op
rather than an exception. The pass has an explicit rewrite bound and contains
no pruning, whole-graph producer/consumer-map rebuild, or unbounded loop.

The existing wrapper still runs the unchanged raw compatibility fallback and
performs the sole historical prune. Plain unary, axis-3 Squeeze, dynamic
signature, and every relaxed candidate therefore retain the old behavior.
The production call now passes the Session layout state, and stale layout
entries removed by the one prune are reconciled at that boundary.

Post-change `inference_ops15.onnx` conversion exited 0 in 2.510 seconds. The
indexed counts were `1, 0, 0`, the combined wrapper counts remained `1, 0, 0`,
and process-tree SWAP remained zero. All three artifacts are byte-identical to
the pre-change baseline and retain the hashes recorded above. A separate
strictly sequential `-cotof` run exited 0 in 5.147 seconds with
`evaluation_pass=true`, maximum absolute error
`1.9073486328125e-06`, and process-tree SWAP zero.

Focused and related verification completed as follows:

- the new indexed/fallback test module plus the adjacent indexed reshape
  module and the complete architecture suite: 228 passed;
- TensorFlow-blocked direct, default-direct, and direct `-cotof`: 3 passed;
- scoped Ruff for the new owner and both touched test modules: passed;
- full lowerer Ruff: the same ten inherited F841 findings remain and no new
  finding was introduced.

No new runtime, artifact, accuracy, timeout, or SWAP failure was found in this
checkpoint. The broader Goal remains incomplete: many raw layout helpers,
op-family extraction, quantization/split/exporter coverage, and final Tier
performance work remain. The first implementation task after resume is to
inspect and characterize the next raw helper in production order,
`_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains`. The two
helpers immediately before it in the first layout pass-set are already indexed
delegates. As always, record a real non-zero owner and the pre-change problem
before editing source, use only short sequential zero-SWAP conversions, and do
not issue a pull request.

Checkpoint branch and tracked change inventory before commit:

- branch: `fb-refactor5`;
- `onnx2tf/tflite_builder/passes/pre_unary_squeeze_suffix_layout.py` (new
  indexed semantic owner);
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py` (owner dispatch, Session
  layout propagation, compatibility/prune boundary);
- `tests/test_flatbuffer_direct_indexed_pre_unary_squeeze_suffix_layout.py`
  (strict-owner, atomicity, determinism, fallback, and layout tests);
- `tests/test_flatbuffer_direct_architecture.py` (bounded owner and production
  wiring constraints);
- `docs/flatbuffer_direct_architecture.md` and this handoff (design, evidence,
  test results, known issues, and resume point).

## Pre-unary Mul/Add fanout: measured no-change decision

After the preceding checkpoint was committed and synchronized, the raw
`_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains` helper
was instrumented at every unchanged production invocation. The same fourteen
short Tier 0-4 representatives used by the adjacent layout checkpoints were
converted strictly sequentially: IAT-LLIE, LINEA, YuNet INT8, FastestDet,
PPHumanSeg, OSNet, SiNet, Tiny-YOLOv2, YOLOv7-tiny, DAMO-YOLO, NanoDet,
YOLOX INT8, YOLO-Free, and yolo_test. All exited 0 in 1.867-8.504 seconds,
none timed out, and every process-tree monitor recorded `peak_swap_kib=0`.
The helper was invoked five times per model and every rewrite count was zero.

A read-only shape-independent ONNX topology scan then inspected all 381
active managed Tier 0-4 models sequentially. All graphs loaded successfully
in 8.663 seconds, with no SWAP. It found zero loose
`Transpose[0,3,1,2] -> unary -> Mul` starts and therefore zero complete
`Mul(const) -> Add(const) -> Transpose[0,2,3,1]` fanout contracts at the ONNX
boundary. No larger or excluded model was converted merely to search for an
owner.

The raw compatibility helper still has an unbounded whole-graph loop and
rebuilds complete producer and consumer maps after every accepted rewrite. It
mutates or clones constants without typed producer/variable/boundary
ownership contracts, changes tensor shapes and operator outputs without a
shared `GraphIndex` or Session `LayoutState`, and has no immutable full-plan
revalidation immediately before its multi-branch mutation. These remain real
interaction and maintenance risks. However, neither the measured production
boundaries nor the complete active-corpus ONNX scan establishes a real
non-zero owner from which artifact, lineage, layout, or accuracy behavior can
be frozen. Removing the helper would discard an existing compatibility
feature, while a synthetic-only indexed replacement would not prove current
production ownership. No converter source or test behavior is therefore
changed in this checkpoint.

The next evidence-first raw helper in the same ordered sequence is
`_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains`. Characterize its
unchanged production invocations and scan the active corpus before source
work. Continue to use only short sequential zero-SWAP conversion gates,
record problems before fixes, commit and push coherent units, and do not
create a pull request.

## Mean/Mul/Add pre/post recovery: measured no-change decision

The raw `_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains` helper
was next inspected and measured without changing source. A read-only scan of
all 381 active managed Tier 0-4 ONNX graphs completed in 8.818 seconds. Every
graph loaded, no process-tree SWAP occurred, and the scan found zero loose
`Transpose[0,3,1,2] -> ReduceMean -> Mul` starts and zero complete
`Mul(const) -> Add(const) -> Transpose[0,2,3,1]` families at the ONNX
boundary.

Because layout lowering can create ModelIR-only adapters, six short models
covering the most likely reduction/layout families were then instrumented at
all five unchanged production invocations: IAT-LLIE, LINEA, FastestDet,
PPHumanSeg, OSNet, and SiNet. They converted strictly sequentially, all exited
0 in 2.065-8.028 seconds, and every process-tree monitor reported SWAP 0.
Every helper invocation returned zero rewrites. The other eight previously
measured representatives were not repeated because the complete active-corpus
scan found no ONNX start and the six ModelIR boundaries established no owner.

The helper retains an unbounded whole-graph loop and rebuilds a complete
consumer map after every accepted rewrite. It accepts reduction axes and
binary constants without typed producer/variable/graph-boundary ownership,
can mutate a shared axes tensor in place, permits relaxed or unknown target
shapes, changes multiple operator edges and tensor views without immutable
full-plan revalidation, and neither updates a shared graph index nor the
Session layout state. Alias rewiring also scans the whole graph. These are
latent correctness and interaction hazards, but no current real model owns
the path. The compatibility feature is therefore neither removed nor replaced
by an unproven synthetic-only pass, and no converter source or test behavior
changes in this checkpoint.

The ordered mean/attention cluster immediately after this helper is already
delegated to bounded pass modules. At resume, inspect the next raw helper in
actual production order before selecting a source target; continue to record
non-zero ownership and problems before implementation, use minimal sequential
zero-SWAP validation, commit and push coherent units, and do not create a pull
request.

## QLinear recovery group: measured no-change decision

The next production interval contains four remaining raw QLinear recovery
helpers:

- `_optimize_transpose_mean_hardsigmoid_muladd_chains`;
- `_optimize_nhwc_prefix_qlinear_silu_chains`;
- `_optimize_nhwc_propagation_qlinear_concat_conv`;
- `_optimize_transpose_mean_maxpool_concat_conv_chains`.

To avoid repeating conversions helper by helper, all four were instrumented
simultaneously at both unchanged production invocations. Six short INT8/QDQ
representatives first covered YuNet INT8, PPHumanSeg INT8, LPD-YuNet INT8,
Version-RFB INT8, NanoDet INT8, and YOLOX INT8. All exited 0 in 1.580-3.899
seconds with process-tree SWAP zero, and every helper count was zero.

A read-only op-set scan of all 381 active Tier 0-4 graphs then completed in
9.115 seconds with no load failure and SWAP zero. No active graph even
contained the complete source-op set for either Mean-based helper. Six graphs
contained the loose prefix-SiLU op set and four contained the loose
Concat/Conv op set. The smallest distinct uncovered candidates,
`ssd_mobilenet_v1_12-int8.onnx` and `dequantize_linear.onnx`, were converted
once with all four helpers still instrumented. Both exited 0 in 3.109 and
4.043 seconds respectively, with SWAP zero and all counts zero. Version-RFB
already represented the smallest prefix candidate; the remaining candidates
were duplicate graph variants or larger known non-passes and were not run.

These raw helpers retain large rule-specific scans and mutation surfaces, but
there is no current real non-zero owner from which artifact and accuracy
behavior can be frozen. No source or test behavior is changed for this group;
the compatibility paths remain available until a real model establishes
ownership.

## Conv/Mul affine fold checkpoint: pre-change record

The core and terminal cleanup boundaries were then instrumented together for
`_optimize_fold_conv_mul_add_affine_chains` and the already indexed
`_optimize_fuse_conv_activation_chains`. Six short models converted
sequentially in 1.820-3.319 seconds with exit code 0 and process-tree SWAP
zero. FastestDet, PPHumanSeg, OSNet, and IAT exercised the activation fusion
owner. Only Tier 2 `iat_llie_180x320.onnx` established ownership for the raw
affine fold: its three invocation counts were `12, 0, 0`, all classified as
Mul-only folds. YuNet INT8, FastestDet, PPHumanSeg, OSNet, and yolo_test
recorded zero affine folds.

All twelve IAT chains have the exact static production form
`CONV_2D(fused=NONE) -> MUL(fused=NONE, [1,1,1,16] FLOAT32 constant)`.
Every filter is exclusive, unquantized FLOAT32 `[16,1,1,I]`, where `I` is 16
or 64; every bias is exclusive unquantized FLOAT32 `[16]`; every side constant
is an unproduced, non-variable, exclusive FLOAT32 tensor; and the Conv/Mul
views are identical fixed `[1,180,320,16]` FLOAT32 tensors with UNKNOWN or
NHWC layout. The Mul outputs continue into dynamic Add branches, so the safe
fold removes only Mul and renames the Conv output. One later
`CONV_2D(fused=RELU) -> MUL([1,180,320,3] dynamic side)` candidate remains
correctly rejected at both later invocations.

The fixed pre-extraction IAT artifacts are:

- float32: `75e355ba8fc01f32b9e4cf2d3c630a4c5c18e6091615a07f30851be6c6eb2881`;
- float16: `4e6f95610870597b74995f441c5cc755cdd2a555a5322504a919aef85f102c43`;
- correspondence: `a52ffab6c473547076538d993dbadd39305304521232b85dda74fae492f77322`.

No observed regression prompted this extraction. The raw owner nevertheless
uses an unbounded full-Conv scan and rebuilds the whole consumer map after
each of twelve rewrites. It mutates filter and bias buffers before graph-edge
replacement and operator removal without an immutable full-plan
revalidation or rollback. In its no-bias path it can create a bias tensor and
rewire the Conv before later coefficient validation rejects the candidate.
Filter, bias, and side constants lack complete typed producer and public-
boundary ownership checks, and mutations do not update a shared graph index
or Session layout state. The safe indexed scope is therefore only the exact
static IAT Mul-only family above. Add-only, Mul/Add, fused-RELU, missing-bias,
scalar/relaxed coefficients, dynamic signatures, quantized/shared/public
constants, and every strict rejection must remain on the existing raw
fallback.

The first post-extraction conversion was deliberately compared before any
corrective edit. Indexed and wrapper invocation counts were both `12, 0, 0`,
the correspondence report remained byte-identical at
`a52ffab6c473547076538d993dbadd39305304521232b85dda74fae492f77322`,
and conversion exited 0 in 2.181 seconds with SWAP zero. However, the float32
and float16 artifacts were not byte-identical: their first observed float32
difference was at byte 4928, and the new hashes were
`44480f7707ff371f3acdcea19f4e1f9b75ffc91586e204ba52af563119a937bc`
and `c5562e790c9e1e21ff7d5d5ae5ea57a388a5367414686678e9074e173ed95bfd`.
This is recorded as an artifact regression before correction. Do not accept
the extraction until tensor-buffer comparison identifies the cause, exact
artifacts are restored, and the sequential accuracy gate passes.

## Conv/Mul affine fold checkpoint: completed state

The recorded artifact regression was diagnosed before any corrective edit.
Name-based FlatBuffer inspection proved that tensor order, operator codes,
operator metadata, all filter buffers, and every finite non-zero bias value
were unchanged. Exactly twelve `[16]` bias buffers differed, and every byte
difference represented only IEEE-754 negative zero versus positive zero. The
raw compatibility implementation calculates Mul-only bias as
`bias * scale + float32(+0)`, whereas the first indexed implementation omitted
the numerically redundant addition. Adding positive zero canonicalizes each
negative-zero product to positive zero, so the omission violated byte-level
artifact compatibility even though inference values were equal.

The indexed owner now preserves that exact float32 operation order, and a
dedicated bit-pattern test prevents its removal as an apparent simplification.
The corrected IAT conversion exited 0 in 2.180 seconds with process-tree SWAP
zero. The indexed and wrapper counts are both `12, 0, 0`, and all three fixed
artifacts are byte-identical again:

- float32:
  `75e355ba8fc01f32b9e4cf2d3c630a4c5c18e6091615a07f30851be6c6eb2881`;
- float16:
  `4e6f95610870597b74995f441c5cc755cdd2a555a5322504a919aef85f102c43`;
- correspondence:
  `a52ffab6c473547076538d993dbadd39305304521232b85dda74fae492f77322`.

### Completed work and design decisions

The strict static IAT Mul-only family is now owned by
`conv_mul_affine_fold.py`. It uses indexed Conv dispatch, a deterministic
rewrite limit, immutable tensor/operator contracts, full candidate
re-resolution immediately before mutation, differential graph-index updates,
Session layout reconciliation, and no internal whole-graph prune or repeated
producer/consumer-map build. The compatibility wrapper dispatches this owner
first, seeds all four historical counters from its result, retains the raw
fallback for every relaxed or unproven family, and remains the only historical
prune/report boundary. All three production calls pass
`session.layout_state`.

The strict contract requires SAME/unit-stride/unit-dilation
`CONV_2D(fused=NONE)`, exactly three inputs, exclusive unquantized FLOAT32
filter `[O,1,1,I]`, bias `[O]`, and scale `[1,1,1,O]`, identical positive
static rank-four Conv/Mul output views, UNKNOWN or NHWC-compatible layout,
resolved producers, exclusive mutable edges, and graph-boundary safety. A
constant Add suffix is deliberately excluded so the historical Mul/Add owner
retains it. Add-only, fused-ReLU, missing-bias, scalar/relaxed coefficient,
dynamic, quantized, shared, public, and malformed variants remain on the raw
fallback. No public API, CLI default, artifact name, dependency, optional
TensorFlow boundary, managed profile, corpus exclusion, or ONNX file changed.

The preceding QLinear recovery characterization is also complete in this
working interval. All four raw helpers retained zero ownership across the
short INT8 representatives and the additional smallest op-set candidates, so
their compatibility implementations remain unchanged.

### Verification completed

- strict owner, fallback interaction, legacy affine fixtures, signed-zero
  bit compatibility, determinism, stale-plan atomicity, graph index, layout,
  and cleanup boundary: `21 passed, 745 deselected in 0.61s`;
- complete flatbuffer-direct architecture contract:
  `215 passed in 44.24s`;
- TensorFlow-import-blocked explicit direct, default direct, and direct
  `-cotof`: `3 passed in 4.28s`;
- corrected sequential IAT conversion: exit 0 in 2.180 seconds, all fixed
  artifacts byte-identical, indexed counts `12, 0, 0`, and
  `peak_swap_kib=0`;
- corrected sequential IAT `-cotof`: exit 0 in 16.676 seconds,
  `evaluation_pass=true`, maximum absolute error
  `4.470348358154297e-07`, and `peak_swap_kib=0`;
- scoped Ruff, Python syntax compilation, and `git diff --check`: pass after
  final documentation synchronization. Whole-file lowerer Ruff reports the
  same ten inherited F841 findings as the parent and no new finding.

### Branch, checkpoint files, and known issues

The branch is `fb-refactor5`. This checkpoint changes:

- `onnx2tf/tflite_builder/passes/conv_mul_affine_fold.py` (new strict indexed
  owner);
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py` (indexed-first wrapper,
  historical counters, Session layout propagation, and cleanup reconciliation);
- `tests/test_flatbuffer_direct_indexed_conv_mul_affine_fold.py` (owner,
  atomicity, determinism, bit-level compatibility, fallback, and layout tests);
- `tests/test_flatbuffer_direct_architecture.py` (bounded owner and production
  wiring contract);
- `docs/flatbuffer_direct_architecture.md` and this handoff (design, evidence,
  validation, remaining work, and resume point).

There is no known failing test, artifact regression, accuracy regression, or
new model issue in this checkpoint. The whole-file inherited lowerer Ruff
findings and optional exporter limitations recorded earlier remain unchanged.
The optional TensorFlow exporter suite was not synchronized or executed; only
the direct TensorFlow-import boundary was tested. No broad Tier run was
performed because the user requested minimal conversion work and IAT is the
only measured non-zero owner for this extraction. The Goal as a whole remains
incomplete: remaining raw semantic helpers, lowering consolidation,
quantization/split/crop and artifact-matrix coverage, optional TensorFlow
exporter confirmation, shared PyTorch-family canonicalization/emission, and
final Tier/efficiency regression work are still outstanding.

### First work after resume

After verifying that `fb-refactor5` is clean and synchronized with
`origin/fb-refactor5`, inspect the next raw helper in actual production order.
Characterize all of its unchanged production invocations on the smallest
short zero-SWAP representatives before changing source. If no real non-zero
owner exists, record a no-change decision; otherwise freeze its exact
artifact, lineage, layout, and accuracy contract before extracting a bounded
semantic owner. Continue strictly sequential validation, exclude any model
whose converter process tree generates SWAP, commit and push coherent units,
and do not create a pull request.

## Activation fusion extraction: pre-change record

The next unmodularized production helper is
`_optimize_fuse_conv_activation_chains`. Its matching already uses
`ModelIRGraphIndex`, so the next safe step is a mechanical owner extraction
rather than another semantic narrowing. Before changing source, four short
real owners were converted strictly sequentially with the helper instrumented
at all three production invocations. All exited 0 in 2.126-2.876 seconds and
all process-tree monitors recorded SWAP zero.

The ordered type-specific counts were:

- FastestDet: `35, 0, 0` total, with 34 Conv and 1 Add fusion at the first
  invocation;
- HumanSeg: `60, 0, 0` total, with 32 Conv and 28 Add fusion at the first
  invocation;
- OSNet: `77, 0, 24` total, with 70 Conv and 7 Add fusion first, then 24 Conv
  fusions at the terminal convergence boundary;
- IAT-LLIE: `1, 0, 0`, consisting of one Conv fusion.

The fixed pre-extraction float32, float16, and correspondence hashes are:

```text
FastestDet: 3bdbec5d7ad81f98cf7890fbf1a98570ebeb1a4a5c19883aca23733b31e1573b
            a14bad05eba99dc211a09aa820eb38396329b98168a2d4b20e463eb64deab617
            2bd03e9e775b4dede0310813cf36e2efc3ad9d0635ce0c5797895fe18d7fb074
HumanSeg:   b69eb57a7628668d73fbf4e06ffa23403d02ebab33b5661f0c60a81395610bb9
            e79c4081989b5a69b65dc220b6e24ad0cff5633363698feebc23b59efe1139df
            87fd06dbd120aac7f0229b02484f7929d6e752c60c19555d6795221cb0a21e46
OSNet:      ed63ef56007979e0f13d1e8e63cbfb590e58af1adee1d214595d03e57412c282
            f22a25ab094217ea1ebc0844da8752c6b95b38b3d98be6cb58314b39e2029a7d
            35a42832e43b2076b00399ba7b22a1ff5aff83795cd333474d6bf61bf7221677
IAT-LLIE:   75e355ba8fc01f32b9e4cf2d3c630a4c5c18e6091615a07f30851be6c6eb2881
            4e6f95610870597b74995f441c5cc755cdd2a555a5322504a919aef85f102c43
            a52ffab6c473547076538d993dbadd39305304521232b85dda74fae492f77322
```

No conversion problem prompted this extraction. The implementation is still
a large op-family rule embedded in the central lowerer, however, and therefore
expands the context needed to maintain unrelated lowering. It restarts its
indexed producer scan after every accepted fusion, owns its prune internally,
does not accept the Session `LayoutState` when deleting now-unused tensors,
and has no dedicated module/API boundary on which focused ownership and wiring
tests can depend. These are the recorded problems before source work. Preserve
the exact producer/activation eligibility, boundary guards, lineage event
order, counters, cleanup timing, public wrapper, and artifacts while moving
the implementation into a small pass module; strict semantic changes are out
of scope for this mechanical checkpoint.

## Activation fusion extraction: completed state

The implementation now lives in `passes/activation_fusion.py`; the historical
`_optimize_fuse_conv_activation_chains` name remains a thin lowerer wrapper.
The extraction preserves the original indexed scan/restart algorithm,
supported producer/activation table, option mutation, protected boundaries,
graph-output bridge guard, dtype guard, output lineage order, type-specific
counters, and cleanup timing. The two direct production calls and the shared
final convergence now forward `session.layout_state`; the final convergence
continues to share its existing `ModelIRGraphIndex`. Pruning therefore removes
stale layout entries without adding an extra cleanup or report event.

Focused tests cover all six producer families, all supported ReLU variants,
per-family counters, Conv/Add/binary lineage types, marker cleanup, supplied
and foreign graph indexes, Session layout reconciliation, idempotence,
determinism, compatibility-wrapper equivalence, fan-out, existing fusion,
dtype mismatch, both protected boundaries, public-output/internal-bridge
ownership, and unsupported activations. The existing lowering fixtures retain
their Add, Conv, depthwise Conv, binary fusion, negative, and final convergence
coverage.

The same four real owners were then converted once after extraction in the
same fixed order. All exited 0 in 2.079-2.726 seconds, all process-tree monitors
recorded SWAP zero, and the exact pre-change type-specific counts were
reproduced. All twelve float32, float16, and correspondence artifacts are
byte-identical to the hashes in the pre-change section. Because the executable
TFLite artifacts themselves are identical, no redundant real-model inference
run was added; the direct `-cotof` TensorFlow-blocker fixture still exercised
the accuracy path.

Verification for this checkpoint is:

- focused new owner, existing activation fixtures, final convergence, and
  architecture selection: `37 passed, 950 deselected in 2.56s`;
- complete flatbuffer-direct architecture contract:
  `215 passed in 42.34s`;
- TensorFlow-import-blocked explicit direct, default direct, and direct
  `-cotof`: `3 passed in 4.34s`;
- scoped Ruff, syntax compilation, and `git diff --check`: pass after final
  documentation synchronization. Whole-file lowerer Ruff retains the same ten
  inherited F841 findings as the parent and introduces no new finding.

This checkpoint changes `passes/activation_fusion.py`, its focused test,
`lower_from_onnx2tf.py`, the architecture test, the architecture document,
and this handoff. It introduces no public API, CLI, artifact, dependency,
profile, exclusion, optional TensorFlow, or ONNX corpus change. No failure or
regression is known. The indexed scan/restart loop remains intentionally
unchanged and is a future performance candidate only after its ordering and
lineage contract can be preserved independently; it is no longer central
lowerer context.

After this checkpoint is synchronized, inspect the next unmodularized helper
at the actual production boundary. Continue to freeze real ownership and exact
artifacts before source changes, use only short sequential zero-SWAP models,
record any problem before correction, commit and push coherent units, and do
not create a pull request.

## Dynamic Reshape resolution extraction: pre-change record

The next central shape-family helper was `_resolve_dynamic_reshape_shapes`
together with `_resolve_reshape_new_shape_from_static_input`. The implementation
occupied roughly 430 lowerer lines and combined selection, ONNX template
precedence, zero-copy semantics, static element-count inference, final-stage
runtime `-1` preference, shape-constant mutation, and output metadata updates.
It already supported optional indexed Reshape dispatch and made no topology
changes, so the safe checkpoint scope was a mechanical module extraction with
no semantic rewrite.

IAT-LLIE and OSNet were first instrumented at all five production invocations.
Both exited 0 in 2.187 and 2.771 seconds with SWAP zero, but all ten counts were
zero. A shortest-likely-owner search then started with Tier 3
`rf-detr-nano.onnx` and stopped immediately: the model exited 0 in 11.573
seconds with SWAP zero and recorded resolver counts `0, 0, 0, 0, 3`. Thus only
the absolute-final call with runtime-inferable ONNX raw-shape preference owns
three current production rewrites.

The fixed RF-DETR pre-extraction artifacts are:

- float32:
  `fda7d97eaad2b19ee2ac31411099067e78b747515952b7c65ba52a0f1454f1fb`;
- float16:
  `a80051b2d6bb871ee871f0d1528e1ea7c8d4e7f6ecbfc16daec4fa78d696fd1f`;
- correspondence:
  `262235cec5a8df73ff2afd7f1eb28678cc7312f4a19dd09d278fd8db77cbdec4`.

No conversion problem prompted this extraction. The recorded maintenance
problem was that the entire decision tree remained central lowerer context,
including its nested sanitizer and static helper, despite being a cohesive
metadata-only pass. Mutation is intentionally direct and non-transactional,
but each Reshape is independent, the traversal is single-pass, and existing
fixtures already cover malformed and unresolvable no-ops. Preserve every
branch, mutation order, counter, optional-index behavior, and private wrapper
while moving the owner; do not narrow or reinterpret shape semantics in this
checkpoint.

## Dynamic Reshape resolution extraction: completed state

`passes/dynamic_reshape_resolution.py` now owns the complete static helper and
resolver. `lower_from_onnx2tf.py` keeps thin wrappers for both historical
private names, so existing imports, positional arguments, optional
`graph_index`, and all four call sites remain compatible. Shared convergence
paths still reuse their existing index; standalone early and absolute-final
calls retain the original operator-list traversal. No graph, layout, lineage,
cleanup, counter, or artifact ordering was changed.

The focused gate covers runtime high-rank `-1`, indexed traversal without a
full operator iteration, zero-copy dimensions, empty `newShape`, static and
dynamic ONNX raw templates, `allowZero`, stale shape constants, window
partition/reverse interactions, shape convergence, final convergence, and
direct module/wrapper equivalence. It passes:

- dynamic Reshape, window, convergence, existing direct, and architecture
  selection: `130 passed, 951 deselected in 3.00s`;
- complete flatbuffer-direct architecture contract:
  `216 passed in 41.98s`;
- TensorFlow-import-blocked explicit direct, default direct, and direct
  `-cotof`: `3 passed in 4.37s`;
- scoped Ruff, syntax compilation, and `git diff --check`: pass. Whole-file
  lowerer Ruff retains the same ten inherited F841 findings as the parent and
  introduces no new finding.

The post-extraction RF-DETR conversion exited 0 in 11.034 seconds with SWAP
zero, reproduced counts `0, 0, 0, 0, 3`, and reproduced all three fixed
artifacts byte-for-byte. Because the executable TFLite files are identical,
no duplicate real-model inference run was added. IAT-LLIE and OSNet were not
repeated after establishing zero ownership; their wrapper behavior is covered
by the direct equivalence and existing fixtures.

This checkpoint changes the new resolver module, the lowerer wrappers, dynamic
Reshape and architecture tests, the architecture document, and this handoff.
It changes no public API, CLI, dependency, artifact, optional TensorFlow
boundary, managed profile, exclusion, or ONNX model. No failure or regression
is known. After synchronization, inspect the next unmodularized production
helper and repeat the evidence-first process; commit and push only, with no
pull request.

## Static shape reconciliation extraction: pre-change record

The next unmodularized production owner was
`_reconcile_static_tensor_shapes`. The fixed-point implementation occupied
1,291 lines and depended on nine adjacent pure shape helpers for Slice,
BatchMatMul, rank-four signature, reduce, Squeeze, and Conv/pool inference.
Together, this cohesive shape-family interval contributed more than 1,500
lines to the central lowerer and forced unrelated pass maintenance to load its
full multi-op decision tree.

Ownership was measured before source work. IAT-LLIE exited 0 in 2.087 seconds
with SWAP zero and invoked reconciliation 29 times, all with zero updates.
Tier 3 `rf-detr-nano.onnx` then established the real owner: it exited 0 in
12.101 seconds with SWAP zero, invoked the helper 29 times, and recorded
non-zero `(invocation, updates)` pairs `(1,141)`, `(3,16)`, `(11,16)`,
`(13,138)`, `(15,16)`, and `(16,6)`, for 333 total updates.

The fixed RF-DETR artifacts are:

- float32:
  `fda7d97eaad2b19ee2ac31411099067e78b747515952b7c65ba52a0f1454f1fb`;
- float16:
  `a80051b2d6bb871ee871f0d1528e1ea7c8d4e7f6ecbfc16daec4fa78d696fd1f`;
- correspondence:
  `262235cec5a8df73ff2afd7f1eb28678cc7312f4a19dd09d278fd8db77cbdec4`.

A symbol-table dependency audit was completed before movement. The interval
closes over ten functions and existing core ModelIR utilities only; it needs
no lowerer import and introduces no cycle. The recorded maintenance problem
was central context size and op-family coupling, not an observed conversion
failure. Preserve the 32-pass fixed-point ceiling, operator order, direct
metadata mutation, dynamic-signature rules, optional producer index, counters,
and every helper signature. Algorithmic optimization or semantic narrowing is
out of scope for this mechanical checkpoint.

## Static shape reconciliation extraction: completed state

`passes/static_shape_reconciliation.py` now owns reconciliation and all nine
adjacent pure inference helpers. Ten thin lowerer wrappers retain every
historical private name and signature. The move removes 1,542 implementation
lines from the central lowerer while keeping the full decision tree in one
cohesive module. Shared convergence continues to supply its existing
`ModelIRGraphIndex`; standalone and fallback callers retain their original
behavior. There is no topology, layout, lineage, cleanup, counter, pass-order,
public API, or artifact change.

The focused gate covers direct module/wrapper equivalence, indexed producer
reuse, all existing shape-reconcile op families, dynamic Reshape, high-rank
runtime shapes, window partition/reverse, binary layout convergence, final
convergence, BatchMatMul, and rank-four signature inference. Results are:

- focused shape-family and architecture selection:
  `133 passed, 954 deselected in 3.15s`;
- complete flatbuffer-direct architecture contract:
  `217 passed in 40.62s`;
- TensorFlow-import-blocked explicit direct, default direct, and direct
  `-cotof`: `3 passed in 4.27s`;
- scoped Ruff, syntax compilation, and `git diff --check`: pass. Whole-file
  lowerer Ruff retains exactly the parent's one inherited F401 and ten F841
  findings, with no new finding.

The post-extraction RF-DETR run exited 0 in 10.811 seconds with SWAP zero and
reproduced all 29 invocation counts, the six non-zero positions, 333 total
updates, and all three artifacts byte-for-byte. Because the executable TFLite
artifacts are identical, no duplicate real-model inference was run. No failure
or regression is known. After this checkpoint is synchronized, continue with
the next unmodularized production helper using the same evidence-first,
short-model, sequential, zero-SWAP policy; commit and push only, and do not
create a pull request.

## HARD_SWISH shape sanitization extraction: pre-change record

The next small metadata owner was `_sanitize_hardswish_tensor_shapes`. It
occupied about 80 central-lowerer lines, accepted an optional shared
`ModelIRGraphIndex`, and enforced the invariant that HARD_SWISH output shape
and signature match its input. This was a mechanical ownership extraction;
no conversion failure or semantic gap prompted the work.

Before source changes, every root model that directly contains ONNX
`HardSwish` was tested in ascending node-count order. The strictly sequential
set was `text_detection_en_ppocrv3_2023may.onnx`, `inference_ops15.onnx`,
`object_tracking_vittrack_2023sep.onnx`, `vttrack.onnx`, the INT8BQ tracking
variant, and `ssdlite320_mobilenet_v3_large.onnx`. The six model files comprise
twelve unchanged production invocations, two per conversion. All invocation
counts were zero. Every conversion exited 0 in 1.757-3.775 seconds and every
process-tree monitor recorded SWAP zero. The absence of a real non-zero owner
is retained as a zero-owner control; the existing synthetic stale-metadata
fixture remains the authoritative positive contract.

The fixed `inference_ops15.onnx` pre-extraction artifacts were:

- float32: `3ce4af63727dd927666f09bb51555ccfd60e1cf01b4ba7fc674170e8277b9a96`;
- float16: `ee97304641e2b1330bbbe1f1472fc32a4a4d41d4bdb08a3e660da64b5204ce47`;
- correspondence: `a50f21319df0380165e8fee2c47f679ccb1682eee965fbd3b0f05ad02cc3d276`.

The safe scope was therefore an exact code move only. Preserve the indexed
HARD_SWISH selection, malformed/missing-tensor no-ops, positive-static-shape
signature canonicalization, dynamic signature fallback, direct metadata
mutation, update count, production scheduling, and compatibility name. Do not
add broader inference or topology changes without first establishing a real
owner and a separate semantic contract.

## HARD_SWISH shape sanitization extraction: completed state

`passes/hardswish_shape_sanitization.py` now owns the implementation and
`lower_from_onnx2tf.py` retains the historical private wrapper. The pass uses
the supplied `ModelIRGraphIndex` only when it belongs to the current ModelIR;
standalone callers retain the original single operator-list traversal. It
performs no topology, layout-state, lineage, constant-buffer, cleanup, or pass-
order mutation. The two production calls and their shared-index behavior are
unchanged.

A direct module/wrapper equivalence test fixes the non-zero stale-metadata
behavior, including the input signature canonicalization that occurs for a
fully static input. The architecture contract fixes the single module owner,
indexed dispatch, absence of producer/consumer-map rebuilds, and absence of a
lowerer import cycle.

Post-extraction `inference_ops15.onnx` conversion exited 0 in 2.358 seconds,
recorded counts `[0,0]`, and process-tree SWAP zero. Its float32, float16, and
correspondence files are byte-identical to the fixed hashes above. Because the
executable artifacts are identical and the helper has no production rewrite
in the measured corpus, no redundant real-model inference was added.

Verification completed as follows:

- targeted owner/final-convergence selection: `3 passed, 218 deselected in
  2.11s`;
- complete final-convergence module: `3 passed in 0.49s`;
- complete flatbuffer-direct architecture contract: `218 passed in 41.77s`;
- TensorFlow-import-blocked explicit direct, default direct, and direct
  `-cotof`: `3 passed, 8 deselected in 4.18s`;
- scoped Ruff, syntax compilation, and `git diff --check`: pass. Whole-file
  lowerer Ruff retains exactly the parent's one inherited F401 and ten F841
  findings, with no new finding.

This checkpoint changes the new sanitizer module, its lowerer wrapper, the
final-convergence and architecture tests, the architecture document, and this
handoff. It changes no public API, CLI, artifact, dependency, optional
TensorFlow boundary, corpus profile, exclusion, or ONNX model. No failure or
regression is known. After synchronization, inspect the next unmodularized
production helper in actual source order. Establish real ownership before any
semantic change, keep model checks short, sequential, and zero-SWAP, commit and
push coherent units, and do not create a pull request.

## Static SQUEEZE-axis sanitization extraction: pre-change record

Work resumed on `fb-refactor6` from merge checkpoint `e41037f1`, with a clean
tree based on the merged `fb-refactor5` result. The earlier handoff's named
pre-Add/Mul-constant/Reshape suffix candidate was re-audited before source
work and found to be stale: that implementation is already completely owned
by `passes/pre_add_mulconst_reshape_suffix_layout.py`. The next small
unmodularized helper in actual lowerer source order was therefore
`_sanitize_squeeze_axes_with_static_input_shapes`, a 135-line terminal
metadata guard.

The helper normalizes explicit SQUEEZE axes after all late layout and shape
rewrites. It removes invalid and duplicate axes, may repair a non-constant
input dimension to singleton, treats a same-rank constant payload as
authoritative evidence against an invalid squeeze, and reconciles output shape
and signature metadata. It has one production invocation, performs no
topology, layout-state, lineage, quantization, constant-buffer, or artifact
mutation, and shares its pure inference helpers with static shape
reconciliation. No conversion failure prompted this checkpoint; the recorded
problem was that the cohesive SQUEEZE policy still lived in the central
lowerer.

Real ownership was measured before movement with temporary, uncommitted stats
instrumentation. The eight shortest active Tier 0 models containing ONNX
Squeeze were run strictly sequentially through `-tb flatbuffer_direct -cotof`
with a 60-second process-group ceiling:

- `gru_14_b1.onnx` and `gru_14_b2.onnx`;
- `torch_lstm.onnx`, `GRU.onnx`, and `GRU_org.onnx`;
- `CNN_AUTOENCODER.onnx`, `ts_ad_model.onnx`, and `UM_best_model.onnx`.

All eight passed in 2.41-3.12 seconds, every process-tree SWAP peak was zero,
and all eight recorded zero sanitizer updates in all three counters. This is
retained as a production zero-owner control; the existing positive synthetic
fixtures remain the semantic owner. `GRU.onnx` fixed the pre-change artifacts:

- float32:
  `f597fff552422165dcf034aa5628618c8349ae4dc776c750fbdc13de5f086b8f`;
- float16:
  `6bc8c810326efe563dd9730f0875b628b0cc8fa1bfa51337f39b6e4c462ef830`;
- tensor correspondence:
  `00f802d0ba18acc81ba8bb32e6915b374e33f0cef6ff2b3d4cb965915e2be3f7`.

The safe scope was an exact owner move only. Preserve the operator order,
axis normalization, constant-data guard, metadata mutation order, stats keys,
terminal scheduling, and private compatibility name. Do not add topology or
layout inference to this mechanical checkpoint.

## Static SQUEEZE-axis sanitization extraction: completed state

`passes/squeeze_shape_sanitization.py` now owns the complete implementation;
`lower_from_onnx2tf.py` retains a thin wrapper with the historical name and
the same signature. The owner imports the two SQUEEZE shape helpers from the
already extracted static-shape module and has no dependency on the lowerer.
It retains the original one-pass operator traversal because the terminal
production boundary has no reusable live graph index and the policy needs no
producer or consumer query. Constructing a new index solely for this scan
would add work without reducing repeated traversal.

Focused tests cover non-constant singleton repair, negative/duplicate/invalid
axis normalization, constant-payload rejection of a non-singleton axis,
output metadata reconciliation, exact counter values, idempotence, and direct
module/compatibility-wrapper equivalence. The architecture contract fixes the
single owner, shared inference-helper calls, absence of producer/consumer map
builders, and absence of a lowerer import cycle.

Post-extraction `GRU.onnx` completed in 2.572 seconds with
`evaluation_pass=true`, maximum absolute error `8.940696716308594e-08`, and
process-tree SWAP zero. Its float32, float16, and correspondence SHA-256 values
are byte-identical to all three pre-change values above.

Verification for this checkpoint is:

- the new owner, the two existing SQUEEZE sanitizer fixtures, the complete
  shape-reconciliation suite, and the complete architecture contract:
  `226 passed in 39.81s`;
- scoped Ruff, Python syntax compilation, and `git diff --check`: pass before
  documentation synchronization; the same final checks are required before
  commit.

This checkpoint changes the new SQUEEZE owner, its lowerer wrapper, a focused
test module, the architecture contract, the architecture document, and this
handoff. The first `uv run` after the merged 2.6.4 release bump also corrected
the editable root-package version recorded in `uv.lock` from 2.6.3 to 2.6.4;
the dependency set and every resolved third-party version are unchanged. The
checkpoint changes no public API, CLI, artifact, dependency, TensorFlow
boundary, corpus profile, exclusion, or ONNX model. No failure or regression
is known.

After this checkpoint is synchronized, inspect
`_replace_expand_dims_and_squeeze_with_reshape`, the next unmodularized helper
in source order. It performs topology and LayoutState mutation, so first freeze
its real invocation counts, exact artifacts, dynamic-shape branches,
pre-operator insertion order, pruning behavior, and wrapper contract before
moving any code. Continue with short sequential zero-SWAP validation, commit
and push coherent units, and do not create a pull request.

## ExpandDims/Squeeze-to-Reshape extraction: pre-change record

The next source-order helper is
`_replace_expand_dims_and_squeeze_with_reshape`. Its 344-line implementation
has one terminal production invocation and owns five coupled behaviors:

- static ExpandDims/Squeeze replacement with deterministic shape constants;
- speculative inactive-If Squeeze safety through a single `-1` target;
- dynamic Squeeze shape plumbing through inserted SHAPE and GATHER operators;
- semantic-rank and original-axis metadata retained for later reconciliation;
- dead-tensor pruning and LayoutState synchronization after successful work.

The helper already uses `ModelIRGraphIndex.insert_operator()` for its dynamic
pre-operators, but the policy, nested target-selection helpers, tensor
creation, mutation order, and pruning boundary remain in the central lowerer.
No conversion failure prompted this checkpoint. The maintenance problem is
the central ownership of a cohesive topology-and-shape pass, not an observed
semantic defect.

Real ownership was measured before source work with temporary, uncommitted
stats instrumentation at the single production call. The shortest active
positive representative, `GRU.onnx`, rewrote 13 ExpandDims/Squeeze operators
and created 13 shape tensors. Its sequential `-tb flatbuffer_direct -cotof`
run passed in 2.711 seconds with maximum absolute error
`8.940696716308594e-08` and process-tree SWAP zero. The fixed artifacts are:

- float32:
  `f597fff552422165dcf034aa5628618c8349ae4dc776c750fbdc13de5f086b8f`;
- float16:
  `6bc8c810326efe563dd9730f0875b628b0cc8fa1bfa51337f39b6e4c462ef830`;
- tensor correspondence:
  `00f802d0ba18acc81ba8bb32e6915b374e33f0cef6ff2b3d4cb965915e2be3f7`.

Existing synthetic tests already cover the positive static and dynamic
Squeeze branches, speculative inactive-If execution, all-ones and dynamic
ExpandDims no-op guards, later shape reconciliation, differential pre-operator
insertion, and LayoutState-aware pruning. The safe scope is therefore a
mechanical owner move only. Preserve traversal order, unique-name selection,
branch precedence, operator/tensor construction order, option keys, graph
index insertion order, pruning, counters, production scheduling, and the
private compatibility signature. Algorithmic or semantic changes require a
separate checkpoint.

## ExpandDims/Squeeze-to-Reshape extraction: completed state

`passes/expand_squeeze_reshape.py` now owns the complete implementation and
`lower_from_onnx2tf.py` retains a thin private wrapper with the same positional
and keyword contract. The single production invocation remains immediately
after late layout/Mean/SPP/Gather/constant/Cast cleanup and before static shape
reconciliation. No branch, traversal, mutation, counter, pruning, or scheduling
behavior was intentionally changed.

The owner still creates a `ModelIRGraphIndex` only when dynamic Squeeze
pre-operators were collected. It inserts original operator groups in reverse
index order and each local SHAPE/GATHER pair in reverse insertion order, which
preserves final SHAPE, GATHER, RESHAPE execution order without direct operator-
list replacement. Successful work retains the original unused-tensor prune and
LayoutState synchronization boundary. Static rewrites do not pay for a graph
index they do not need.

A dedicated owner test now proves dynamic kept-axis tensor data, exact SHAPE/
GATHER/RESHAPE order, LayoutState validity, old-wrapper equivalence, full
ModelIR equality, and idempotence. Together with the existing dynamic-Reshape,
shape-reconciliation, speculative-branch, all-ones, and dynamic-ExpandDims
fixtures, the focused gate completes with `19 passed in 0.63s`. The architecture
contract now reads the implementation from its pass module, fixes the one-call
wrapper, differential insertion, prune/sync boundary, and absence of a lowerer
import cycle. The complete focused set plus the complete flatbuffer-direct
architecture contract passes `238 passed in 37.63s`.

Post-extraction `GRU.onnx` completed in 2.608 seconds with
`evaluation_pass=true`, maximum absolute error `8.940696716308594e-08`, and
process-tree SWAP zero. Its float32, float16, and correspondence artifacts are
byte-identical to the three pre-change hashes, proving compatibility across
the 13 real production rewrites.

This checkpoint changes the new ExpandDims/Squeeze owner, the lowerer wrapper,
one dedicated test module, the architecture contract, the architecture
document, and this handoff. It changes no public API, CLI, artifact, dependency,
TensorFlow boundary, corpus profile, exclusion, or ONNX model. No failure or
regression is known.

After synchronization, inspect
`_repair_rank4_binary_layout_mismatch_with_transpose_adapter`, the next raw
source-order helper. First determine whether its behavior is already covered
by an indexed binary-layout owner and measure a shortest non-zero real owner;
do not create a duplicate pass. Continue with short sequential zero-SWAP
validation, commit and push coherent units, and do not create a pull request.

## Rank-four binary layout-adapter extraction: pre-change record

`_repair_rank4_binary_layout_mismatch_with_transpose_adapter` is the next raw
source-order helper. Its 91-line fixed-point implementation recognizes exact
rank-four NHWC/NCHW shape permutations on ADD, MUL, SUB, DIV, MAXIMUM, and
MINIMUM, inserts one Transpose on input 1, clones per-tensor quantization onto
the adapted tensor, rewires only the matched operand, restarts discovery, and
prunes unused tensors after convergence.

The existing indexed binary-layout owners were audited before source work.
They remove stale channelwise adapters or reduce established transpose bridge
patterns; they do not insert a missing adapter for two already-materialized,
full rank-four tensor shapes. Reusing those owners would conflate inverse
contracts. This helper therefore remains a distinct semantic owner, but it is
placed in a binary adapter module that can later receive the adjacent
singleton-broadcast variant without creating duplicate central dispatch.

Temporary, uncommitted instrumentation measured both unconditional production
invocations on six short representatives: `GRU.onnx`, `iat_llie_180x320.onnx`,
`FastestDet.onnx`, `face_detection_yunet_2023mar.onnx`,
`human_segmentation_pphumanseg_2021oct.onnx`, and
`osnet025_Nx3x256x128.onnx`. All twelve invocation results were zero. Every
model passed, durations were 2.695-15.949 seconds, maximum absolute errors
were below `2.2e-05`, and every process-tree SWAP peak was zero. The fallback
and placeholder-restoration calls were not entered. This is retained as a
production zero-owner control; a dedicated synthetic permutation fixture is
the positive contract.

`FastestDet.onnx` fixes the zero-owner artifacts:

- float32:
  `3bdbec5d7ad81f98cf7890fbf1a98570ebeb1a4a5c19883aca23733b31e1573b`;
- float16:
  `a14bad05eba99dc211a09aa820eb38396329b98168a2d4b20e463eb64deab617`;
- tensor correspondence:
  `2bd03e9e775b4dede0310813cf36e2efc3ad9d0635ce0c5797895fe18d7fb074`.

The safe scope is an exact code move only. Preserve binary op eligibility,
input-1-only adaptation, permutation selection precedence, unique names,
tensor metadata and quantization, insertion/rewrite order, fixed-point restart,
unconditional prune, stats key, four production positions, and the private
compatibility signature. Differential indexing or expanded semantic guards
require separate real ownership and regression evidence.

## Rank-four binary layout-adapter extraction: completed state

`passes/binary_layout_adapter.py` now owns the exact full-rank permutation
adapter and `lower_from_onnx2tf.py` retains a thin private wrapper. All four
production positions are unchanged: two unconditional terminal passes, one
fallback normalization path, and one placeholder-MatMul restoration path. The
adjacent singleton-broadcast repair is deliberately not moved in this
checkpoint; it has different output-shape and operand-selection semantics and
must be characterized separately.

The implementation retains its original direct operator insertion and
compatibility input setter. This checkpoint does not claim a differential-
index performance change: it establishes module ownership first, while the
six measured real models prove only zero-owner compatibility. A later indexed
rewrite must first prove exact candidate ordering, restart behavior, and
pruning equivalence with a non-zero real owner or an independently accepted
synthetic contract.

Seventeen dedicated tests cover all six binary types in both NHWC/NCHW
directions, exact permutation constants, input-1-only rewiring, adapted tensor
shape/signature, independent quantization cloning, equal/dynamic/rank/non-
permutation no-ops, idempotence, and old-wrapper equivalence. The architecture
contract fixes one module owner, four production calls, insertion/rewrite/
prune ownership, and the no-import-cycle boundary. The complete focused owner
suite plus the complete flatbuffer-direct architecture contract passes
`237 passed in 38.80s`.

Post-extraction `FastestDet.onnx` passed in 3.718 seconds with maximum absolute
error `1.3113021850585938e-05` and process-tree SWAP zero. Its float32,
float16, and correspondence artifacts are byte-identical to the three fixed
pre-change hashes.

This checkpoint changes the binary adapter owner, the lowerer wrapper, one
focused test module, the architecture contract, the architecture document,
and this handoff. It changes no public API, CLI, artifact, dependency,
TensorFlow boundary, corpus profile, exclusion, or ONNX model. No failure or
regression is known.

After synchronization, characterize
`_repair_rank4_binary_singleton_broadcast_layout_mismatch`, the adjacent raw
helper. It may join `binary_layout_adapter.py` only after its two operand
directions, output-shape guard, insertion order, and real invocation counts are
fixed independently. Continue with short sequential zero-SWAP validation,
commit and push coherent units, and do not create a pull request.

## Singleton-broadcast binary adapter integration: pre-change record

The adjacent `_repair_rank4_binary_singleton_broadcast_layout_mismatch` helper
is a separate 172-line compatibility policy for an NCHW singleton-channel
tensor paired with an NHWC rank-four tensor. It supports the same six binary
operator types but has four output-driven branches:

- input 0 is singleton NCHW and input 1 is NHWC, with an NCHW output: insert
  NHWC-to-NCHW Transpose on input 1;
- input 1 is singleton NCHW and input 0 is NHWC, with an NCHW output: insert
  NHWC-to-NCHW Transpose on input 0;
- input 0 is singleton NCHW with an NHWC output: reshape input 0 to
  `[N,H,W,1]`;
- input 1 is singleton NCHW with an NHWC output: reshape input 1 to
  `[N,H,W,1]`.

Every branch first rejects a pair that already broadcasts to the declared
output. Accepted branches clone quantization to the adapted tensor, insert the
adapter immediately before the binary operator, rewire only the selected
operand, restart fixed-point discovery, and prune only when at least one repair
occurred. This contract is related to, but not interchangeable with, the exact
full-rank permutation adapter already in `binary_layout_adapter.py`; the two
belong in the same op-family module under separate public owners and stats.

Temporary, uncommitted instrumentation measured both unconditional production
invocations on `GRU.onnx`, `iat_llie_180x320.onnx`, `FastestDet.onnx`,
`face_detection_yunet_2023mar.onnx`,
`human_segmentation_pphumanseg_2021oct.onnx`, and
`osnet025_Nx3x256x128.onnx`. All twelve results were zero; the fallback and
placeholder-restoration paths were not entered. All six models passed in
2.607-16.015 seconds, their maximum absolute errors remained below `2.2e-05`,
and all process-tree SWAP peaks were zero. This is another production zero-
owner control, so the complete four-branch synthetic contract is required
before integration.

The fixed `FastestDet.onnx` hashes remain:

- float32:
  `3bdbec5d7ad81f98cf7890fbf1a98570ebeb1a4a5c19883aca23733b31e1573b`;
- float16:
  `a14bad05eba99dc211a09aa820eb38396329b98168a2d4b20e463eb64deab617`;
- tensor correspondence:
  `2bd03e9e775b4dede0310813cf36e2efc3ad9d0635ce0c5797895fe18d7fb074`.

The safe scope is an exact move into the existing binary adapter module. Keep
all four branch predicates and their precedence, NumPy broadcast guard,
operand selection, adapter type, permutation/shape constants, metadata and
quantization, unique-name behavior, insertion/rewrite order, fixed-point
restart, conditional prune, stats key, four production positions, and private
wrapper. Differential indexing or semantic generalization remains separate.

## Singleton-broadcast binary adapter integration: completed state

`passes/binary_layout_adapter.py` now owns the singleton-broadcast policy under
`repair_rank4_binary_singleton_broadcast_layout_mismatch`. The historical
lowerer symbol remains a thin compatibility wrapper and its four production
positions are unchanged. The exact full-rank and singleton-broadcast policies
remain separate functions with separate stats, because the latter chooses an
operand and either Transpose or Reshape from the declared output layout.

Twenty-four positive synthetic cases cover all four output-driven branches for
all six supported binary operators. They additionally prove selected-operand
rewiring, exact permutation or target-shape constants, tensor shape/signature,
independent quantization cloning, and idempotence. A dedicated NumPy-broadcast
no-op and direct-owner/private-wrapper equivalence complete the new focused
contract. Together with the previously extracted exact adapter and the two
legacy singleton tests, the focused binary module run passes `45 passed in
0.58s`. The complete flatbuffer-direct architecture suite passes `221 passed
in 39.12s` and fixes one module owner plus four production calls for each
binary policy. The final combined branch gate (all four extracted owner test
modules, shape reconciliation, active legacy compatibility selectors, and the
complete architecture suite) passes `280 passed in 39.86s`.

Post-integration `FastestDet.onnx` passed sequential isolated conversion in
3.739 seconds with maximum absolute error
`1.3113021850585938e-05`, evaluation pass true, and process-tree SWAP zero.
Its float32, float16, and tensor-correspondence artifacts are byte-identical to
the three pre-change hashes. No dependency, public API, CLI, artifact format,
TensorFlow boundary, corpus profile, exclusion, or ONNX model changed, and no
regression is known.

The next adjacent raw lowerer helper is
`_sanitize_static_shape_signature_consistency`. At restart, first characterize
its production call count, positive ownership, no-op guards, metadata and
stats contract before considering a separate extraction. Keep validation
small and sequential, reject any model that triggers SWAP, commit and push one
coherent checkpoint at a time, and do not create a pull request.

## Static shape-signature sanitizer extraction: completed state

The former 206-line `_sanitize_static_shape_signature_consistency` lowerer
implementation is now owned by
`passes/static_shape_signature_sanitization.py`. The old private symbol is a
thin compatibility wrapper and both production calls remain at their original
late-pipeline positions. This checkpoint is a mechanical ownership move: it
retains the single producer-map build, boundary metadata handling, special
WHERE/RANGE/RESHAPE/TOPK_V2 roots, recursive memoization, cycle stop, constant
lineage stop, graph-output leading-axis policy, mutation order, and all four
stats keys. It changes tensor shape signatures only; topology and LayoutState
are untouched.

Temporary environment-gated instrumentation was removed before implementation.
On six sequential short controls, all twelve calls made zero static repairs.
The ownership counters were nevertheless non-zero: the final `FastestDet`
call preserved 13 boundary signatures and 26 multi-axis lineage signatures;
both `osnet025_Nx3x256x128` calls preserved three boundary signatures and 295
leading-axis signatures. The other measured calls were zero. All six controls
passed in 2.609-16.515 seconds, maximum absolute error stayed below `2.2e-05`,
and process-tree SWAP was zero.

Eleven focused owner tests cover scalar completion, missing/rank-mismatched/
stale signatures, dynamic runtime no-op, boundary-map normalization, every
runtime-dependent root family, options and constant-shape RESHAPE targets,
recursive leading and multi-axis lineage, constant termination, graph-output
preservation, cycle termination, idempotence, and wrapper equality. With the
four active legacy fixtures and the ownership selector the focused gate passes
`16 passed in 1.89s`; the complete architecture suite passes `222 passed in
38.25s`. The final combined branch gate across all five extracted owner test
modules, active legacy selectors, shape reconciliation, and architecture
passes `296 passed in 39.30s`. A fixed-seed differential harness extracted the
old helper from the pre-change commit and compared stats plus every tensor
signature with the new owner across 250 generated ModelIR cases; all matched.

Post-extraction sequential validation used the two real positive owners only.
`FastestDet.onnx` passed in 3.755 seconds with maximum absolute error
`1.3113021850585938e-05`; `osnet025_Nx3x256x128.onnx` passed in 4.053 seconds
with maximum absolute error `2.193450927734375e-05`. Both had process-tree
SWAP zero. Their six float32, float16, and tensor-correspondence artifacts are
byte-identical to the pre-change controls. No regression is known.

The next adjacent raw lowerer helper is
`_realign_dynamic_boundary_shape_signature_map`. Before changing it, determine
whether it belongs with this sanitizer or ONNX boundary analysis, fix the
alignment-helper contract and three production positions, and measure whether it
has a non-zero short real owner. Continue with sequential zero-SWAP validation,
commit and push coherent checkpoints, and do not create a pull request.

## Dynamic boundary-signature map realignment: completed state

`passes/static_shape_signature_sanitization.py` now also owns
`realign_dynamic_boundary_shape_signature_map`. The core axis-alignment
primitive remains in `core/onnx_analysis.py` because lowerer boundary-map
construction uses it independently. The old private lowerer symbol is a thin
wrapper and all three late production positions are unchanged. The owner
retains exact map object mutation, list/type/rank guards, deterministic static-
extent placement, update counter, and final metadata assignment. It reads no
operator and changes no tensor or topology.

Temporary instrumentation measured all three calls on `FastestDet.onnx` and
`osnet025_Nx3x256x128.onnx`; all six results were zero. Both models passed,
their maximum absolute errors were `1.3113021850585938e-05` and
`2.193450927734375e-05`, and process-tree SWAP was zero. These are explicit
zero-owner controls, while the synthetic layout-change fixtures provide the
positive contract.

The expanded focused module plus the active legacy realignment fixture and
ownership selector passes `21 passed in 1.88s`. It covers unchanged axes,
layout movement, repeated static extents, insufficient matches, empty/rank-
mismatched signatures, malformed maps and entries, missing tensors/shapes,
idempotence, and private-wrapper equality. The complete architecture suite
passes `223 passed in 39.80s`. A fixed-seed differential harness compared the
old and new owners over 250 generated maps; stats and final metadata matched
in every case. The final combined branch gate across all extracted owner test
modules, active legacy selectors, shape reconciliation, and architecture
passes `306 passed in 39.69s`.

Post-extraction sequential validation passed `FastestDet.onnx` in 3.725
seconds and `osnet025_Nx3x256x128.onnx` in 4.080 seconds with the same two
maximum absolute errors and SWAP zero. Their six float32, float16, and tensor-
correspondence artifacts are byte-identical to the pre-change controls. No
regression is known.

The intervening split, static-shape inference, indexed convergence, and
placeholder-MatMul functions are already compatibility wrappers or established
indexed owners. The next raw source-order implementation is the large
`_optimize_transpose_quant_dequant_bridges` helper. Before modifying it, split
its pattern families and stats contract, identify any existing QDQ pass owners,
measure each production call, and select the smallest real positive models.
Continue with sequential zero-SWAP validation, commit and push coherent
checkpoints, and do not create a pull request.

## Transpose QDQ bridge ownership extraction: completed state

The former 929-line `_optimize_transpose_quant_dequant_bridges` implementation
is now owned by `passes/transpose_qdq_bridge_layout.py`; the lowerer retains a
two-line private compatibility wrapper. This owner is distinct from the
registered terminal Q/DQ, Concat-input, Mean, activation, PReLU, Reshape, and
TransposeConv quantization cleanup modules. Partial family extraction was
rejected because the existing behavior depends on a shared fixed-point loop
with strict A→B→C→D precedence and restart after every accepted rewrite.

The four retained families are the complete Transpose/Q/DQ/Transpose round
trip, a single Q-or-DQ bridge with single/multiple post fan-out, the two-branch
QDQ/Add/QDQ residual closure, and the mixed float/QDQ Add residual closure.
Branch linearity, public outputs, exact inverse permutations, per-tensor grids,
legacy fan-out, representative output selection, metadata permutation,
quantization cloning, mutation order, pruning, and all three stats keys are
unchanged. The ordered layout-recovery prefix contains one wrapper call and is
expanded at the same four runtime positions.

Temporary instrumentation was removed before extraction. Five existing
positive fixtures showed first-sweep stats of one removed bridge for Pattern B,
one for Pattern A, two removals plus one residual rewrite for Pattern C, and
one removal plus one mixed-residual rewrite for both Pattern D variants. Their
following three sweeps were no-ops. Six direct owner tests add exact Pattern A
and B topology/metadata assertions, independent quantization cloning, per-
channel/public/non-inverse no-ops, idempotence, and wrapper equality. The
focused owner, five end-to-end families, and architecture selector pass `12
passed in 2.18s`; the complete architecture suite passes `224 passed in
36.96s`. The final combined branch gate across all extracted owner modules,
active legacy selectors, shape reconciliation, and architecture passes `318
passed in 38.66s`. The entire old and new owner ASTs are identical after
normalizing the public function name.

`face_detection_yunet_2023mar_int8.onnx` is a short positive real owner: its
first sweep removes nine bridges and the next three sweeps are no-ops. Before
extraction it passed in 5.049 seconds with `max_abs=0` and SWAP zero. After
extraction it passed in 5.204 seconds with the same error and SWAP result. Its
float32, float16, and tensor-correspondence artifacts are byte-identical to the
pre-change controls. No regression is known.

The next raw source-order orchestration is
`_optimize_transpose_swish_qdq_nhwc_islands` and its residual-Concat closure.
Before moving it into the existing `quantized_swish_layout.py` family, fix the
primary/late/safety-valve ownership boundaries, both option combinations,
stats aggregation, production call expansion, and a short positive real model.
Continue with sequential zero-SWAP validation, commit and push coherent
checkpoints, and do not create a pull request.

## Swish-QDQ orchestration ownership extraction: completed state

The complete `_optimize_transpose_swish_qdq_nhwc_islands` orchestration and
its residual-Concat closure are now owned by
`passes/quantized_swish_layout.py`. The central lowerer retains both historical
private symbols as thin wrappers, and the closure and normal production
positions remain one call each. This is a mechanical ownership move over the
already indexed phase owners: primary branch/metadata processing, first
inverse-post cleanup, late Concat/post cleanup, the independently owned
wrong-way Conv-input safety valve, and final unused-tensor pruning retain their
exact order and statistics.

Temporary instrumentation was removed before implementation. The established
160-spatial fixture and the 80-spatial closure fixture both recorded primary
results of three rewritten branches, two removed pre-Transposes, one rewritten
Concat axis, and twenty-four propagated tensors. Their first post cleanup
removed two Transposes; late cleanup and the safety valve were zero. The
independent safety fixture recorded zero primary/late work and two safety-valve
removals. Both public orchestration ASTs are identical to their prior committed
lowerer functions after normalizing only the moved function and direct safety-
owner names.

Three new focused tests fix exact phase order, both forwarded options,
statistics aggregation across disjoint rewritten-tensor sets, closure option
fixing and result remapping, and public-owner/private-wrapper equality. The
focused indexed Swish module, existing safety delegation, both legacy Swish
variants, and the two ownership selectors pass `26 passed in 2.26s`. The
complete architecture suite passes `224 passed in 38.42s`. The final branch
selection across all extracted owners and active legacy selectors passes
`321 passed in 38.50s`. TensorFlow-import-blocked import, explicit direct
conversion, and direct `-cotof` pass `3 passed in 4.85s`. Scoped Ruff, syntax
compilation, and whitespace checks all pass.

A read-only scan of active passing Tier 0-4 models up to 100 MiB found no graph
with the complete ONNX Transpose/Q/DQ/Sigmoid/Mul source family. The bounded
`dequantize_linear.onnx` candidate recorded zero closure and normal rewrites,
so positive semantics remain fixed by the existing synthetic fixtures rather
than a claimed production owner. Its single post-extraction sequential run
exited zero in 12.789 seconds with process-tree SWAP zero. The established
known `tflite_fail`, `max_abs=58.7506103515625`, and normalized error-signature
hash were unchanged. Float32, float16, correspondence, `schema.fbs`, and
`schema_generated.py` hashes were byte-identical to the pre-extraction
control. This checkpoint does not reclassify or improve that known failure.

Changed files are the quantized-Swish pass module, the lowerer wrappers, the
focused indexed-Swish tests, architecture ownership checks, and the three
branch design/handoff documents. No package, public API, CLI, artifact name,
corpus profile, exclusion policy, or TensorFlow boundary changes. Temporary
instrumentation and all temporary conversion outputs are removed before the
checkpoint. No new pull request is created; future work remains commit/push
only.

The next raw source-order implementation is
`_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains`.
At restart, first characterize all production positions, exact family
boundaries, statistics, and short zero-SWAP model ownership before changing
source. Keep inference sequential and minimal, preserve any compatibility
fallback that lacks a real owner, commit and push one coherent unit, and do not
create a pull request.

## HardSwish SE gating layout extraction: completed state

The former 463-line
`_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains`
implementation is now owned by `passes/hardswish_se_layout.py`. The lowerer
retains a thin private wrapper at both unchanged production positions. This is
an exact mechanical move: after normalizing the function name, the complete
old and new ASTs are identical. The shared consumer-map rebuild, ordered
restart, direct/decomposed root matching, expanded/fused gate matching,
Mean-axis rewrite, four-Transpose removal, metadata/quantization propagation,
pruning, and statistic all remain unchanged.

Eight focused owner tests cover all four direct/decomposed root and
expanded/fused gate combinations, idempotence, public pre-output, invalid
reduction axes, activation fan-out, and direct-owner/private-wrapper equality.
Together with the architecture selector they pass `9 passed in 1.77s`. The
complete architecture suite passes `225 passed in 37.10s`; the expanded branch
gate passes `330 passed in 38.57s`; and TensorFlow-import-blocked import,
direct, and direct `-cotof` pass `3 passed in 4.82s`.

Three sequential short models were sufficient to characterize all six
production invocations. SSDLite MobileNetV3, `inference_ops15`, and MobileNetV3
PyTorch each recorded `[0,0]`, so no production rewrite is claimed. SSDLite is
the fixed artifact control because it contains HardSwish, HardSigmoid, global
pooling, Conv, Mul, Add, and Transpose families in one 384-node graph. Its
post-extraction run passed in 7.892 seconds with
`max_abs=3.0517578125e-05` and process-tree SWAP zero. Float32, float16,
correspondence, `schema.fbs`, and `schema_generated.py` hashes are identical to
the 8.338-second pre-extraction control.

Changed source is limited to the new pass module, the lowerer wrapper, its
focused tests, architecture ownership coverage, and branch documentation. No
dependency, public API, CLI, artifact, TensorFlow boundary, corpus profile, or
exclusion changes. Temporary instrumentation and conversion outputs are
removed before commit. No new pull request is created.

The next raw source-order boundary is
`_optimize_transpose_pre_concat_nhwc_chains_legacy` behind the 61-line
indexed-first `_optimize_transpose_pre_concat_nhwc_chains` wrapper. At restart,
first inventory its existing indexed owner, legacy family boundaries,
fallback ownership, call positions, and focused fixtures. Do not split or
semantically narrow the 2,452-line fallback without evidence; keep conversion
validation minimal and sequential, then commit and push only.

## Generic NHWC pre-Concat legacy ownership extraction: completed state

The complete 2,452-line
`_optimize_transpose_pre_concat_nhwc_chains_legacy` implementation is now owned
by `passes/nhwc_concat_legacy_layout.py`. The lowerer retains the historical
private symbol as a one-call compatibility wrapper. This was an exact
mechanical ownership move, not a semantic split or a source-line-limit change:
after normalizing only the function name, the old and new full ASTs are
identical. All nested planners and appliers, float/quantized indexed-family
exclusion contracts, action precedence, fixed-point restart, constant
copy-on-write, shape/layout/quantization propagation, canonical output
selection, operator removal, pruning, and the historical statistic are
unchanged.

The 61-line composite wrapper is unchanged. It still dispatches the eleven
float indexed families, then the thirteen quantized indexed families, then the
legacy compatibility owner, and aggregates them into the single public
`optimized_transpose_pre_concat_nhwc_chains` counter. All four explicit
production positions are preserved. The exact pseudo-LeakyRelu plus Pad-
companion fixture establishes a positive legacy rewrite, idempotence, and
direct-owner/private-wrapper ModelIR equality. The pre-change nine-file Concat
corpus passed 284 tests; after adding wrapper ownership coverage it passes 285.
The complete architecture suite passes `226 passed in 34.16s`, the Concat plus
architecture gate passes `511 passed in 34.86s`, and the final branch-wide
selection passes `616 passed in 34.88s`. The complete optional-TensorFlow
import-blocked suite, including direct conversion and direct `-cotof`, passes
`11 passed in 9.30s`. Scoped Ruff, syntax compilation, and whitespace checks
pass.

Temporary instrumentation measured seven wrapper invocations each on
`FastestDet.onnx` and `osnet025_Nx3x256x128.onnx`. Every float indexed,
quantized indexed, and legacy count was zero, so this checkpoint does not claim
a non-zero production legacy owner. FastestDet passed before extraction in
4.541 seconds and after extraction in 4.474 seconds with identical
`max_abs=1.3113021850585938e-05` and process-tree SWAP zero. Its core artifacts
are byte-identical:

- float32 TFLite:
  `3bdbec5d7ad81f98cf7890fbf1a98570ebeb1a4a5c19883aca23733b31e1573b`;
- float16 TFLite:
  `a14bad05eba99dc211a09aa820eb38396329b98168a2d4b20e463eb64deab617`;
- tensor correspondence:
  `2bd03e9e775b4dede0310813cf36e2efc3ad9d0635ce0c5797895fe18d7fb074`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

OSNet supplied the second seven zero-owner invocations and passed in 4.810
seconds with `max_abs=2.193450927734375e-05` and SWAP zero. Instrumentation and
all temporary conversion output are removed before commit. Changed files are
the new legacy pass module, lowerer compatibility wrapper/import, one focused
legacy-owner test, the architecture ownership test, and the three branch
documents. No dependency, public API, CLI, artifact, TensorFlow boundary,
corpus profile, exclusion, or 2,000-ONNX-operation tier policy changes. No pull
request is created; work remains commit/push only.

The next raw source-order implementation is the 148-line
`_optimize_transpose_slice_prepost_nhwc_passthrough_chains`. At restart, first
inventory its production positions, exact Slice parameter/permutation and
fan-out/public-output guards, statistics, dependencies, existing fixtures, and
short zero-SWAP model ownership before changing source. Keep inference
strictly sequential and minimal, then commit and push one coherent unit without
creating a pull request.

## Slice pre/post NHWC passthrough extraction: completed state

The complete 148-line
`_optimize_transpose_slice_prepost_nhwc_passthrough_chains` implementation is
now owned by `passes/slice_prepost_layout.py`; the lowerer retains a one-call
private compatibility wrapper at the unchanged single production position.
After normalizing only the function name, the prior lowerer function and new
owner ASTs are identical. The owner directly reuses the established Slice
shape resolver in `static_shape_reconciliation.py`, preserving the existing
dependency without copying it.

The exact rank-four NHWC→NCHW Transpose, constant Slice, and inverse Transpose
contract is unchanged. Both begin/size tensors must be constant and exclusive,
the pre and Slice intermediates must be exclusive and non-public, permutations
must be exact inverses, and input/output metadata must be rank four. The matcher
continues to test the constants as-is before their NCHW→NHWC remap and accepts
only the first representation reproducing the known post-Transpose shape.
Constant rewrites, Slice input/output aliasing, two-Transpose removal,
conditional pruning, fixed-point restart, and the historical statistic retain
their exact order.

The new focused corpus covers remap-required and already-NHWC constants,
idempotence, public pre and Slice outputs, shared begin constants, pre-adapter
fan-out, output-shape mismatch, wrong permutation, and direct-owner/private-
wrapper equality. It passes with the architecture selector as `10 passed in
1.77s`. The complete architecture suite passes `227 passed in 34.56s`; the
branch-wide selection passes `626 passed in 36.18s`; and the complete optional-
TensorFlow import-blocked suite, including direct conversion and direct
`-cotof`, passes `11 passed in 9.32s`. Scoped Ruff, syntax compilation, and
whitespace checks pass.

Temporary instrumentation measured the single production call on Tier 0
`UM_best_model.onnx` and Tier 2 `alike_t_opset11_192x320.onnx`; both were zero,
so this checkpoint does not claim a non-zero production owner. ALike passed in
6.744 seconds with `max_abs=2.345442771911621e-05` and process-tree SWAP zero.
UM Best Model is the fixed artifact control. Before extraction it passed in
3.335 seconds and after extraction in 3.228 seconds with identical
`max_abs=2.384185791015625e-07` and process-tree SWAP zero. Its core artifacts
are byte-identical:

- float32 TFLite:
  `fd4f73f2d267b7f300c164d749b2d0abf6d529cf093d32913642c7ad7c81bdbd`;
- float16 TFLite:
  `f003bfc1253a2dca9a81e6051bb2a9c9951f9bddbdd19b6a8111edfd11b9f2cb`;
- tensor correspondence:
  `87478284278c9d7e796c1e85a310f02f7f0b2f6cf87e0e74884c5465ed2857e2`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

Instrumentation and temporary outputs are removed before commit. Changed files
are the new pass module, lowerer wrapper/import, focused corpus, architecture
ownership check, and three branch documents. No dependency, public API, CLI,
artifact, TensorFlow boundary, corpus profile, exclusion, or ONNX tier policy
changes. No pull request is created; work remains commit/push only.

The next raw source-order implementation is the 285-line
`_optimize_transpose_shape_extract_nhwc_to_nchw_chains`. At restart, inventory
its Gather/Slice remapping families, shared/public constant behavior,
non-contiguous Slice-to-Gather conversion, production positions, statistic,
existing fixtures, and short zero-SWAP model ownership before changing source.
Keep inference strictly sequential and minimal, then commit and push one
coherent unit without creating a pull request.

## Shape-extraction layout ownership extraction: completed state

The complete 285-line
`_optimize_transpose_shape_extract_nhwc_to_nchw_chains` implementation is now
owned by `passes/shape_extract_layout.py`; the lowerer retains a one-call
private compatibility wrapper at all three unchanged production positions.
This is an exact mechanical move: after normalizing only the function name,
the prior lowerer function and the new owner ASTs are identical.

The owner preserves Gather-index remapping from logical NCHW to physical NHWC,
contiguous Slice remapping, and non-contiguous Slice-to-Gather conversion.
Shared constants remain clone-on-write, with dtype and quantization metadata
retained. All Shape consumers must be supported before mutation; public
Transpose or Shape outputs, adapter fan-out, unsupported users or axes,
non-constant and invalid indices, and empty selections remain strict no-ops.
Fixed-point restart, conditional pruning, and the historical statistic are
unchanged.

The focused owner corpus plus architecture selector passes `14 passed in
1.79s`. The complete architecture suite passes `228 passed in 34.70s`; the
branch-wide selection passes `640 passed in 35.35s`; and the complete optional-
TensorFlow import-blocked suite, including direct conversion and direct
`-cotof`, passes `11 passed in 9.75s`.

Tier 2 `retinaface_onnx_dynamic.onnx` establishes positive real ownership with
production counts `1,0,0`. Before extraction it passed in 3.996 seconds and
after extraction in 4.030 seconds with identical
`max_abs=4.4405460357666016e-06` and process-tree SWAP zero. Its core artifacts
are byte-identical:

- float32 TFLite:
  `ff0cbf4697ab2f7f0304d49eff912babaf2b3c31060374023b175ad01342a8ff`;
- float16 TFLite:
  `91b40c03d9c9628d0acb5d3c87c7a2e8cd23f8df089be6d00e52ee4db87d1c42`;
- tensor correspondence:
  `6edbb4f0d8436d9c064c1415840807a00e004c96b8a1f7df40954cb35d5bdc43`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

Tier 2 `alike_t_opset11_192x320.onnx` supplies three zero-owner production
calls. It passes in 6.658 seconds with
`max_abs=2.345442771911621e-05` and zero process-tree SWAP.

Changed files are the new Shape-extraction pass, lowerer import and wrapper,
the focused owner corpus, architecture ownership checks, and the three branch
documents. No package, public API, CLI, artifact name, TensorFlow boundary,
corpus profile, exclusion policy, or ONNX tier policy changes. Temporary
instrumentation and conversion outputs are removed before commit. PR #952
remains closed; work is commit/push only and no pull request is created.

The next raw source-order implementation is the 1,593-line
`_optimize_transpose_pre_add_nhwc_chains` composite. At restart, first
inventory its indexed owner, compatibility fallback, production positions,
statistics, and focused fixtures. Do not split or semantically narrow this
boundary without characterized ownership and artifact evidence. Keep inference
strictly sequential and minimal, then commit and push one coherent unit without
creating a pull request.

## Indexed-first pre-Add compatibility ownership extraction: completed state

The complete 1,593-line `_optimize_transpose_pre_add_nhwc_chains`
implementation is now owned by `passes/pre_add_layout.py`. The lowerer retains
a one-call private compatibility wrapper at all four unchanged production
positions and in the conservative safe-transpose bundle. This is an exact
mechanical move: after normalizing only the function name, the prior lowerer
function and new owner ASTs are identical.

The bounded `pre_add_direct_unary_layout` owner still runs first. The composite
fallback retains its complete Swish, unary, Mul-constant, Mul/Sub-constant,
Gather, constant-Add, nested-Add, PReLU, direct-NCHW bridge, post-alias, and
legacy-consumer behavior. Full producer/consumer map construction, fixed-point
restart, clone-on-write constants, shape/layout/dtype/quantization propagation,
operator and tensor mutation order, marker behavior, pruning, and the single
historical statistic are unchanged. This is not a source-line limit and does
not claim that the fallback has been modernized.

Before the move, the focused indexed, QLinear, and four active compatibility
fixtures passed 27 tests. After adding forced-fallback module-owner/private-
wrapper equality, those fixtures plus the complete architecture suite pass
`256 passed in 35.28s`. The branch-wide extracted-owner selection passes
`668 passed in 34.12s`; the complete optional-TensorFlow import-blocked suite,
including direct conversion and direct `-cotof`, passes `11 passed in 9.53s`.
Scoped Ruff, whole-lowerer Ruff with its established exclusions, syntax
compilation, and whitespace checks pass.

The first architecture selector run after the move failed because it still
looked for the indexed-first assignment inside the lowerer function. This was
an expected ownership-contract mismatch, not a conversion failure. The test
now proves that `pre_add_layout.py` invokes the indexed owner first, contains
the compatibility map/prune loop, never imports the lowerer, and that the
lowerer retains exactly one dispatch call plus four Session-layout production
positions and the safe-bundle position.

`FastestDet.onnx` establishes positive real fallback ownership. Before and
after extraction its eight composite results are `1,0,0,0,0,0,0,0`, while all
eight bounded indexed-owner results are zero. The pre-extraction sequential
`-cotof` run completed in 9.168 seconds and the post-extraction run in 9.394
seconds. Both pass with `max_abs=1.3113021850585938e-05` and process-tree SWAP
zero. The five core artifacts are byte-identical:

- float32 TFLite:
  `3bdbec5d7ad81f98cf7890fbf1a98570ebeb1a4a5c19883aca23733b31e1573b`;
- float16 TFLite:
  `a14bad05eba99dc211a09aa820eb38396329b98168a2d4b20e463eb64deab617`;
- tensor correspondence:
  `2bd03e9e775b4dede0310813cf36e2efc3ad9d0635ce0c5797895fe18d7fb074`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

Changed files are the new composite pass module, lowerer import and wrapper,
the focused forced-fallback equivalence test, architecture ownership checks,
and the three branch documents. No dependency, public API, CLI, artifact name,
TensorFlow boundary, corpus profile, exclusion policy, or ONNX tier policy
changes. Temporary tracing and conversion outputs are removed before commit.
PR #952 remains closed; work is commit/push only and no pull request is
created.

The next raw source-order implementation is the 166-line
`_optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains`. At
restart, inventory its strict dual-adapter and single-post contract, constant
ownership, legacy-consumer and public-boundary guards, single production
position, statistic, existing fixtures, and short zero-SWAP ownership before
changing source. Keep inference strictly sequential and minimal, then commit
and push one coherent unit without creating a pull request.

## Dual-pre-Add single-post ownership extraction: completed state

The complete 166-line
`_optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains`
implementation is now owned by `passes/dual_pre_add_layout.py`. The lowerer
retains a one-call private compatibility wrapper at the unchanged single late
production position. After normalizing only the function name, the prior
lowerer function and new owner ASTs are identical.

The rule still accepts two exclusive rank-four NHWC-to-NCHW Transpose outputs
feeding one non-public Add only when no existing NCHW-to-NHWC post adapter
consumes the Add result. It rewires the Add to the original NHWC inputs,
creates an NHWC result with cloned dtype/shape/signature/quantization, removes
both input adapters, and inserts one NHWC-to-NCHW compatibility adapter after
the Add. Unique-name behavior, graph-order restart, unconditional pruning, and
the historical statistic are unchanged.

The new focused corpus fixes the positive rewrite, independent quantization
clone, idempotence, public Add output, public pre-adapter output, wrong
permutation, rank mismatch, input fan-out, existing inverse-post rejection,
and direct-owner/private-wrapper equality. It passes with the architecture
selector as `10 passed, 228 deselected in 1.96s`. The complete focused plus
architecture gate passes `238 passed in 34.68s`; the branch-wide selection
passes `678 passed in 36.60s`; and the complete optional-TensorFlow import-
blocked suite passes `11 passed in 10.53s`.

FastestDet, OSNet, and HumanSeg were characterized strictly sequentially before
the move. Each reached the helper once, recorded zero rewrites, exited zero,
and used zero process-tree SWAP. No non-zero production owner is claimed; the
positive contract is synthetic. FastestDet is the fixed artifact control. Its
pre-extraction conversion completed in 1.614 seconds and post-extraction in
1.593 seconds, with byte-identical core artifacts:

- float32 TFLite:
  `3bdbec5d7ad81f98cf7890fbf1a98570ebeb1a4a5c19883aca23733b31e1573b`;
- float16 TFLite:
  `a14bad05eba99dc211a09aa820eb38396329b98168a2d4b20e463eb64deab617`;
- tensor correspondence:
  `2bd03e9e775b4dede0310813cf36e2efc3ad9d0635ce0c5797895fe18d7fb074`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

FastestDet had passed the immediately preceding sequential `-cotof` checkpoint
at `max_abs=1.3113021850585938e-05`. Because this move reproduced the exact
executed TFLite artifact, no redundant inference run was added. All actual
model runs in this checkpoint remained sequential.

One latent compatibility risk was recorded without semantic change. The
historical helper creates a fixed `__nhwc_to_nchw_perm_rank4__` tensor only
when that name is absent, but does not validate an existing tensor's dtype,
payload, producer, variable state, consumers, or graph visibility before reuse.
Hardening this collision requires its own focused negative/positive contract
and artifact gate; it must not be mixed into this exact ownership move.

Changed files are the new dual-pre-Add pass module, lowerer import and wrapper,
the focused test corpus, architecture ownership test, and the three branch
documents. No dependency, public API, CLI, artifact name, TensorFlow boundary,
corpus profile, exclusion policy, or ONNX tier policy changes. Temporary
tracing and conversion outputs are removed before commit. PR #952 remains
closed; work is commit/push only and no pull request is created.

The next raw source-order implementation is the 293-line
`_optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains`. At restart,
inventory its terminal Transpose/Mul/Add/Reshape/FullyConnected topology,
constant ownership and mutation, graph-output contract, single production
position, statistic, existing fixtures, and short zero-SWAP ownership before
changing source. Keep inference strictly sequential and minimal, then commit
and push one coherent unit without creating a pull request.

## Terminal affine/Reshape/FullyConnected ownership extraction: completed state

The complete 293-line
`_optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains` implementation is
now owned by `passes/terminal_affine_fc_layout.py`. The lowerer retains a
one-call private compatibility wrapper at the unchanged single late production
position. After normalizing only the function name, the prior lowerer function
and new owner ASTs are identical.

The owner preserves the exact exclusive
Transpose→Mul→Add→Reshape→FullyConnected chain, all public intermediate guards,
static positive NHWC/NCHW view relation, channel-constant rotation, shared
constant copy-on-write, both `[O,I]` and `[I,O]` FullyConnected weight layouts,
NHWC flatten-order permutation, metadata and quantization cloning, fixed-point
restart, pruning, and statistic. No new dependency, index path, or semantic
guard was introduced.

The focused corpus covers both weight orientations, exclusive and shared Mul
and FC constants, independent quantization clones, idempotence, public
Transpose/Mul/Add/Reshape intermediates, wrong permutation, dynamic channel,
weight-width mismatch, input fan-out, and module-owner/private-wrapper
equality. The selector passes `14 passed, 229 deselected in 2.07s`; focused plus
complete architecture passes `243 passed in 32.69s`; the branch-wide selection
passes `692 passed in 34.75s`; and the complete optional-TensorFlow import-
blocked suite passes `11 passed in 9.28s`.

OSNet was characterized strictly sequentially and reached the helper once with
zero rewrites before and after extraction. Its pre conversion completed in
2.171 seconds and post conversion in 2.195 seconds; both recorded process-tree
SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `ed63ef56007979e0f13d1e8e63cbfb590e58af1adee1d214595d03e57412c282`;
- float16 TFLite:
  `f22a25ab094217ea1ebc0844da8752c6b95b38b3d98be6cb58314b39e2029a7d`;
- tensor correspondence:
  `35a42832e43b2076b00399ba7b22a1ff5aff83795cd333474d6bf61bf7221677`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The immediately preceding OSNet accuracy checkpoint remains
`max_abs=2.193450927734375e-05`; no redundant inference run was added because
the executed TFLite artifact is identical. A read-only scan of every root ONNX
file up to 50 MiB found no exact raw Transpose→Mul→Add→Reshape→Gemm/MatMul
source chain. No non-zero production owner is claimed; positive behavior is
fixed by the synthetic corpus.

One existing transactional defect is recorded, not corrected. The helper can
rotate an exclusive Mul side constant in place before attempting to materialize
the Add side constant. If the latter is invalid, it returns a zero statistic
after leaving the Mul constant changed. Fixing this requires immutable planning,
full prevalidation, and an atomic negative fixture in a separate semantic
checkpoint; mixing it into this mechanical move would obscure compatibility.

Changed files are the new terminal affine/FC pass module, lowerer import and
wrapper, focused corpus, architecture ownership test, and three branch
documents. No dependency, public API, CLI, artifact name, TensorFlow boundary,
corpus profile, exclusion policy, or ONNX tier policy changes. Temporary
tracing and conversion outputs are removed before commit. PR #952 remains
closed; work is commit/push only and no pull request is created.

The next raw source-order implementation is the 263-line
`_optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains`. At
restart, inventory its PReLU-alpha remap, Reshape/BatchMatMul flatten-order
permutation, shared-constant behavior, graph-output guards, conditional
production position, statistic, fixtures, and short zero-SWAP ownership before
changing source. Keep inference strictly sequential and minimal, then commit
and push one coherent unit without creating a pull request.

## Terminal PReLU/Reshape/BatchMatMul ownership extraction: completed state

The complete 263-line
`_optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains`
implementation is now owned by `passes/terminal_prelu_bmm_layout.py`. The
lowerer retains a one-call private compatibility wrapper at the unchanged
single conditional late production position. After normalizing only the
function name, the prior lowerer function and new owner ASTs are identical.

The owner preserves the exclusive Transpose→PReLU→Reshape→BatchMatMul chain,
all public intermediate guards, positive NHWC/NCHW view relation, scalar,
rank-three CHW, rank-four NCHW, and already-NHWC alpha forms, shared alpha and
RHS copy-on-write, RHS flatten-order permutation, adjX/adjY rejection, metadata
and quantization cloning, fixed-point restart, pruning, and statistic. The
legacy unused producer-map build remains in place with a local compatibility
annotation so scan cost and order are not changed by the move.

The dedicated corpus plus the existing positive fixture cover every supported
alpha form, exclusive and shared alpha/RHS constants, independent quantization
clones, idempotence, public Transpose/PReLU/Reshape intermediates, wrong
permutation, dynamic channel, RHS width, adjX, adjY, input fan-out,
one-dimensional alpha rejection, and module-owner/private-wrapper equality.
The focused selector passes `18 passed, 231 deselected in 2.06s`; the existing
positive fixture passes independently; focused plus complete architecture
passes `249 passed in 33.09s`; the branch-wide selection passes `711 passed in
34.52s`; and the optional-TensorFlow import-blocked suite passes `11 passed in
9.54s`.

Tier 1 `inference_ops15.onnx` was characterized strictly sequentially. Its
conditional helper call remained zero before and after extraction. The pre
conversion completed in 1.823 seconds and post conversion in 1.860 seconds;
both recorded process-tree SWAP zero. Its core artifacts are byte-identical:

- float32 TFLite:
  `3ce4af63727dd927666f09bb51555ccfd60e1cf01b4ba7fc674170e8277b9a96`;
- float16 TFLite:
  `ee97304641e2b1330bbbe1f1472fc32a4a4d41d4bdb08a3e660da64b5204ce47`;
- tensor correspondence:
  `a50f21319df0380165e8fee2c47f679ccb1682eee965fbd3b0f05ad02cc3d276`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The immediately preceding `inference_ops15` sequential accuracy baseline is
`max_abs=1.9073486328125e-06`. No duplicate inference run was added because
the executed TFLite artifact is identical. A read-only scan of all root ONNX
files up to 50 MiB found no exact raw Transpose→PRelu→Reshape→MatMul/Gemm
source chain. No non-zero production owner is claimed; the positive contract
is synthetic.

The existing alpha/RHS ownership weakness remains recorded. The helper checks
data and consumer sharing but does not fully reject a producer, variable state,
or graph-visible constant before mutation or cloning. Adding those contracts
requires focused compatibility evidence and an immutable planning checkpoint;
they are not mixed into the mechanical ownership move.

Changed files are the new terminal PReLU/BMM pass module, lowerer import and
wrapper, focused corpus, architecture ownership test, and three branch
documents. No dependency, public API, CLI, artifact name, TensorFlow boundary,
corpus profile, exclusion policy, or ONNX tier policy changes. Temporary
tracing and conversion outputs are removed before commit. PR #952 remains
closed; work is commit/push only and no pull request is created.

The next raw source-order implementation is the 415-line
`_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains`. At restart, inventory
its dual pre-Add residual prefix, Mul/Add affine constants, PReLU alpha, post
adapters and legacy consumers, production positions, statistic, fixtures, and
short zero-SWAP ownership before changing source. Keep inference strictly
sequential and minimal, then commit and push one coherent unit without creating
a pull request.

## Residual Add/Mul/Add/PReLU ownership extraction: completed state

The complete 415-line `_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains`
implementation is now owned by `passes/residual_affine_prelu_layout.py`. The
lowerer retains a one-call private compatibility wrapper at all three unchanged
source positions. After normalizing only the function name, the prior lowerer
function and new owner ASTs are identical.

The owner preserves the dual NHWC-to-NCHW pre-Add planning, shared pre-adapter
handling, Mul/Add/PReLU constant prevalidation, scalar and rank-four broadcast
forms, rotation and copy-on-write, PReLU post aliases, legacy NCHW consumer
adapter retention, tensor metadata and quantization propagation, exact mutation
and removal order, fixed-point restart, pruning, and statistic. The separate
indexed SiNet late-residual owner remains unchanged and in its existing phase
order.

The existing positive fixture now creates deep-copy models, invokes the module
owner and lowerer private wrapper independently, and compares statistics,
every operator, tensor names, dtype, shape/signature, and NumPy payload. It
passes with the new architecture selector as `2 passed, 231 deselected in
2.05s`. The complete indexed SiNet shuffle/residual suite passes `207 passed in
0.83s`. Direct owner plus complete architecture passes `233 passed in 32.61s`;
the branch-wide selection passes `713 passed in 33.97s`; and the optional-
TensorFlow import-blocked suite passes `11 passed in 9.44s`.

SiNet establishes positive real ownership. Its fourteen runtime invocation
counts are `0,0,0,1,1,0,0,0,0,0,0,0,0,0` before and after extraction. The
pre conversion completed in 2.048 seconds and post conversion in 1.959 seconds;
both recorded process-tree SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`;
- float16 TFLite:
  `180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`;
- tensor correspondence:
  `24c423ea51b26b178d3764be027855e797bbf9b5ba1930810d2e1dbe281d8e25`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The immediately preceding SiNet sequential accuracy baseline remains
`max_abs=2.572051016613841e-09`. No duplicate inference was added because the
executed TFLite artifact is identical. All actual model runs remained
sequential.

The compatibility owner still validates constant data and consumers less
strictly than newer indexed owners: producer, variable-state, and graph-
visibility ownership is not a complete immutable contract. Hardening requires
independent negative fixtures, full prevalidation, and atomic revalidation; it
is not mixed into this exact ownership move.

Changed files are the new residual affine/PReLU pass module, lowerer import and
wrapper, direct owner/wrapper equality coverage in the existing fixture,
architecture ownership test, and three branch documents. No dependency, public
API, CLI, artifact name, TensorFlow boundary, corpus profile, exclusion policy,
or ONNX tier policy changes. Temporary tracing and conversion outputs are
removed before commit. PR #952 remains closed; work is commit/push only and no
pull request is created.

The next raw source-order implementation is the 477-line
`_optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains`. At restart,
inventory its shared residual Add prefix, each Mul/Add/post-Transpose branch,
legacy NCHW consumers, constant copy-on-write, production positions, statistic,
fixtures, and short zero-SWAP ownership before changing source. Keep inference
strictly sequential and minimal, then commit and push one coherent unit without
creating a pull request.

## Residual affine/post-Transpose fan-out ownership extraction: completed state

The complete 477-line
`_optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains`
implementation is now owned by `passes/residual_affine_fanout_layout.py`. The
lowerer retains a one-call private compatibility wrapper at all three unchanged
source positions. After normalizing only the function name, the prior lowerer
function and new owner ASTs are identical.

The owner preserves the shared dual NHWC-to-NCHW pre-Add prefix, each
Mul-constant→Add-constant→post-Transpose branch, optional legacy NCHW
consumers, the exact profitability guard, broadcast-aware constant
prevalidation and rotation, shared-constant copy-on-write, one retained legacy
adapter, tensor metadata and quantization propagation, exact mutation/removal
order, fixed-point restart, pruning, and statistic. No dependency, graph-index
path, guard, or semantic policy was introduced.

The focused positive fixture has two affine/post-Transpose branches, one legacy
NCHW consumer, and a Mul constant shared with an unrelated consumer. It invokes
the module owner and lowerer wrapper on deep copies, compares every operator,
tensor name, shape/signature, dtype, quantization, and NumPy payload, proves
copy-on-write, and checks idempotence. A second fixture fixes the public
post-output no-op guard. The direct owner plus complete architecture gate passes
`235 passed in 31.39s`; the complete indexed SiNet residual suite passes
`207 passed in 0.77s`; the branch-wide selection passes
`716 passed in 33.19s`; and the optional-TensorFlow import-blocked suite passes
`11 passed in 10.03s`.

SiNet is the strictly sequential artifact control. Its fourteen runtime helper
results are all zero before and after extraction. The pre conversion completed
in 2.438 seconds and the post conversion in 2.587 seconds; both recorded
process-tree SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`;
- float16 TFLite:
  `180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`;
- tensor correspondence:
  `24c423ea51b26b178d3764be027855e797bbf9b5ba1930810d2e1dbe281d8e25`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The immediately preceding SiNet accuracy checkpoint remains
`max_abs=2.572051016613841e-09`. No duplicate inference was added because the
executed TFLite artifact is identical. FastestDet supplied eight additional
pre-move zero-owner invocations in 2.120 seconds with zero SWAP, but is not used
as the post-move artifact control. No non-zero production owner is claimed;
the positive contract is synthetic.

One compatibility weakness remains recorded rather than changed. Side
constants are validated for data, rank, broadcast compatibility, and consumer
sharing, but not completely for producer ownership, variable state, or graph
visibility. Adding immutable ownership and transactional revalidation requires
separate negative fixtures and a semantic checkpoint; it is not mixed into
this exact mechanical move.

Changed files are the new residual affine fan-out pass module, lowerer import
and wrapper, focused corpus, architecture ownership test, and three branch
documents. No public API, CLI, artifact name, TensorFlow boundary, dependency,
corpus profile, exclusion policy, or ONNX operation-tier policy changed.
Temporary tracing and conversion outputs are removed before commit. PR #952
remains closed; work is commit/push only and no pull request is created.

The next raw source-order implementation is the 401-line
`_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains`. At
restart, inventory its pre-Transpose and unary-family guards, every
Mul/Add/post-Transpose branch, constant sharing and mutation, graph-output
boundaries, all three production positions, statistic, existing fixtures, and
short zero-SWAP ownership before changing source. Keep inference strictly
sequential and minimal, then commit and push one coherent unit without creating
a pull request.

## Pre-unary affine/post-Transpose fan-out ownership extraction: completed state

The complete 401-line
`_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains`
implementation is now owned by `passes/pre_unary_affine_fanout_layout.py`. The
lowerer retains a one-call private compatibility wrapper at all three unchanged
source positions. After normalizing only the function name, the prior lowerer
function and new owner ASTs are identical.

The owner preserves the exclusive NHWC-to-NCHW pre-Transpose, the exact
RELU/RELU6/LOGISTIC/TANH/HARD_SWISH/LEAKY_RELU/GELU allowlist, every
Mul-constant→Add-constant→post-Transpose branch, broadcast-aware constant
prevalidation and rotation, shared-constant copy-on-write, tensor metadata and
quantization propagation, exact mutation/removal order, fixed-point restart,
pruning, and statistic. No dependency, graph-index path, layout-state behavior,
guard, or semantic policy was introduced.

Ten focused cases fix all seven accepted unary forms, a two-branch positive
graph, an externally shared Mul constant, module-owner/private-wrapper full
ModelIR equality, idempotence, unsupported unary rejection, and a public
post-output no-op boundary. The adjacent focused architecture selector passes
`12 passed, 232 deselected in 2.06s`; the complete indexed SiNet residual suite
passes `207 passed in 0.79s`; the branch-wide selection passes
`727 passed in 33.48s`; and the optional-TensorFlow import-blocked suite passes
`11 passed in 9.65s`.

SiNet is the minimal strictly sequential artifact control. Its five runtime
helper results are all zero before and after extraction. The pre conversion
completed in 2.430 seconds and the post conversion in 2.503 seconds; both
recorded process-tree SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`;
- float16 TFLite:
  `180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`;
- tensor correspondence:
  `24c423ea51b26b178d3764be027855e797bbf9b5ba1930810d2e1dbe281d8e25`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The immediately preceding SiNet accuracy checkpoint remains
`max_abs=2.572051016613841e-09`; no duplicate inference was added because the
executed TFLite artifact is identical. This zero-owner result agrees with the
earlier fourteen-model, five-boundary conversion characterization and the
read-only 381-model active Tier 0-4 ONNX topology scan. No non-zero production
owner is claimed; positive behavior is fixed synthetically.

The recorded compatibility risk is unchanged. The helper rebuilds whole-graph
producer/consumer maps in an unbounded loop, does not share GraphIndex or
LayoutState, and does not completely validate constant producer ownership,
variable state, or graph visibility. Replacing it with an immutable indexed
plan requires independent semantic fixtures and is not mixed into this exact
mechanical ownership move.

Changed files are the new pre-unary affine fan-out pass module, lowerer import
and wrapper, focused corpus, architecture ownership test, and three branch
documents. No public API, CLI, artifact name, TensorFlow boundary, dependency,
corpus profile, exclusion policy, or ONNX operation-tier policy changed.
Temporary tracing and conversion outputs are removed before commit. PR #952
remains closed; work is commit/push only and no pull request is created.

The next raw source-order implementation is the 509-line indexed-first
compatibility composite
`_optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains`.
At restart, inventory its indexed owner and raw fallback boundary, per-call
GraphIndex construction and LayoutState forwarding, Mul and reshape constants,
legacy consumers, combined statistic, production positions, existing positive
IAT-LLIE ownership, and fallback fixtures before changing source. Keep
inference strictly sequential and minimal, then commit and push one coherent
unit without creating a pull request.

## Pre-Add/Mul/Reshape-suffix compatibility composite extraction: completed state

The complete 509-line indexed-first
`_optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains`
composite is now owned by
`passes/pre_add_mulconst_reshape_suffix_compat_layout.py`. The lowerer retains
one private compatibility wrapper at the unchanged production position and
forwards the Session `LayoutState`. After normalizing only the function name,
the prior lowerer composite and new owner ASTs are identical.

The compatibility module preserves the existing indexed semantic owner as its
first dispatch, one per-call `ModelIRGraphIndex`, caller LayoutState forwarding,
combined statistic accumulation, direct/direct and direct/Mul-constant raw
fallback, Mul and reshape constant mutation/copy-on-write, legacy NCHW adapter,
fixed-point restart, exact mutation/removal order, and the sole historical
prune/report boundary. The indexed immutable plan and its module are unchanged.

The complete thirteen-case family suite covers direct and Mul-constant indexed
paths, typed and shared constants, immutable-plan guards, graph/layout state,
bounded dispatch, the compatibility module owner versus lowerer wrapper on deep
copies, the single prune event, and a forced-zero indexed dispatch that proves
the raw fallback still rewrites the accepted family. It passes
`13 passed in 0.52s`; the branch-wide selection including the full family
passes `740 passed in 31.18s`; and the optional-TensorFlow import-blocked suite
passes `11 passed in 9.43s`.

Tier 2 IAT-LLIE is the strictly sequential positive artifact control. Its three
combined and indexed results remain `5,4,4`, while raw fallback results remain
`0,0,0`, before and after extraction. The pre conversion completed in 2.162
seconds and the post conversion in 2.074 seconds; both recorded process-tree
SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `75e355ba8fc01f32b9e4cf2d3c630a4c5c18e6091615a07f30851be6c6eb2881`;
- float16 TFLite:
  `4e6f95610870597b74995f441c5cc755cdd2a555a5322504a919aef85f102c43`;
- tensor correspondence:
  `a52ffab6c473547076538d993dbadd39305304521232b85dda74fae492f77322`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The preceding IAT-LLIE sequential accuracy checkpoint remains
`max_abs=4.470348358154297e-07`; no duplicate inference was added because the
executed TFLite artifact is identical. All actual model runs remained strictly
sequential.

The raw fallback's known compatibility risk is unchanged. It rebuilds complete
producer and consumer maps within a fixed-point loop and has looser producer,
variable-state, and graph-visibility ownership for mutable constants than the
indexed plan. Replacing it with an immutable transaction requires independent
semantic fixtures and is not mixed into this exact mechanical move.

Changed files are the new pre-Add/Mul/Reshape-suffix compatibility module,
lowerer import and wrapper, expanded indexed/compatibility family corpus,
updated architecture ownership test, and three branch documents. No public
API, CLI, artifact name, TensorFlow boundary, dependency, corpus profile,
exclusion policy, or ONNX operation-tier policy changed. Temporary tracing and
conversion outputs are removed before commit. PR #952 remains closed; work is
commit/push only and no pull request is created.

The next raw source-order implementation is the 302-line indexed-first
compatibility composite
`_optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains`. At
restart, inventory its indexed Swish owner, plain-unary and relaxed raw
fallback, GraphIndex construction and LayoutState forwarding, reshape constant
ownership, combined statistic, single prune/report boundary, production
position, existing positive LINEA ownership, and fallback fixtures before
changing source. Keep inference strictly sequential and minimal, then commit
and push one coherent unit without creating a pull request.

## Pre-unary/Reshape-suffix compatibility composite extraction: completed state

The complete 302-line indexed-first
`_optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains` composite
is now owned by `passes/pre_unary_reshape_suffix_compat_layout.py`. The lowerer
retains one private compatibility wrapper at the unchanged production position
and forwards Session `LayoutState`. After normalizing only the function name,
the prior lowerer composite and new owner ASTs are identical.

The compatibility module preserves the existing indexed Swish semantic owner,
one per-call `ModelIRGraphIndex`, caller LayoutState, combined statistic,
thirteen-operation plain-unary and relaxed Swish raw fallback, reshape constant
and option updates, tensor metadata propagation, fixed-point restart, exact
mutation/removal order, sole prune/report boundary, and removal of pruned names
from LayoutState. The indexed immutable plan and module are unchanged.

The indexed family plus the existing direct fixture now cover indexed Swish,
shared-shape atomic rejection, bounded dispatch, stale-plan rejection,
determinism, complete compatibility-owner/lowerer-wrapper equality, LayoutState
cleanup, and a plain LEAKY_RELU raw fallback. The focused owner/fallback/
architecture selection passes `9 passed, 233 deselected in 2.19s`; the
branch-wide selection passes `748 passed in 29.98s`; and the optional-
TensorFlow import-blocked suite passes `11 passed in 9.25s`.

Tier 2 LINEA is the strictly sequential positive artifact control. Its three
combined and indexed results remain `1,0,0`, while raw fallback results remain
`0,0,0`, before and after extraction. The pre conversion completed in 7.958
seconds and the post conversion in 7.751 seconds; both recorded process-tree
SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `fd91ea915b600b3581e8e0e68925fefd5302cd1bfb373ebca8b9b9410138c611`;
- float16 TFLite:
  `c8e44a48221eeead187869d93dfef1f7775420335aae5c63873118738d39f9a8`;
- tensor correspondence:
  `ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The preceding LINEA sequential accuracy checkpoint remains
`max_abs=0.002297189086675644`; no duplicate inference was added because the
executed TFLite artifact is identical. All actual model runs remained strictly
sequential.

The raw fallback's known compatibility risk is unchanged. It rebuilds complete
producer and consumer maps in a fixed-point loop, mutates relaxed reshape
constants/options in place, and does not use the indexed owner's immutable
differential transaction. Hardening requires separate semantic fixtures and is
not mixed into this exact mechanical move.

Changed files are the new pre-unary/Reshape-suffix compatibility module,
lowerer import and wrapper, expanded indexed/compatibility tests, updated
architecture ownership test, and three branch documents. No public API, CLI,
artifact name, TensorFlow boundary, dependency, corpus profile, exclusion
policy, or ONNX operation-tier policy changed. Temporary tracing and conversion
outputs are removed before commit. PR #952 remains closed; work is commit/push
only and no pull request is created.

The next raw source-order implementation is the 297-line indexed-first
compatibility composite
`_optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains`. At
restart, inventory its indexed static Swish owner, plain-unary/dynamic/relaxed
raw fallback, GraphIndex construction and LayoutState forwarding, Squeeze axes
constants/options, combined statistic, single prune/report boundary, production
position, existing positive `inference_ops15` ownership, and fallback fixtures
before changing source. Keep inference strictly sequential and minimal, then
commit and push one coherent unit without creating a pull request.

## Pre-unary/Squeeze-suffix compatibility composite extraction: completed state

The complete 297-line indexed-first
`_optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains` composite
is now owned by `passes/pre_unary_squeeze_suffix_compat_layout.py`. The lowerer
retains one private compatibility wrapper at the unchanged production position
and forwards Session `LayoutState`. After normalizing only the function name,
the prior lowerer composite and new owner ASTs are identical.

The compatibility module preserves the existing indexed static Swish semantic
owner, one per-call `ModelIRGraphIndex`, caller LayoutState, combined statistic,
plain-unary, axis-3, dynamic-signature, and relaxed Swish raw fallback behavior,
Squeeze axis option remapping, tensor metadata propagation, fixed-point restart,
exact mutation/removal order, sole prune/report boundary, and removal of pruned
names from LayoutState. The indexed immutable plan and module are unchanged.

The indexed family plus the existing direct fixtures now cover indexed Swish,
plain unary, axis-3 Swish, dynamic-signature fallback, complete compatibility-
owner/lowerer-wrapper equality, LayoutState cleanup, bounded dispatch, stale-
plan rejection, and determinism. The focused owner/fallback/architecture
selection passes `9 passed, 233 deselected in 1.93s`; the branch-wide selection
passes `756 passed in 30.79s`; and the optional-TensorFlow import-blocked suite
passes `11 passed in 9.34s`.

Tier 1 `inference_ops15.onnx` is the strictly sequential positive artifact
control. Its three combined and indexed results remain `1,0,0`, while raw
fallback results remain `0,0,0`, before and after extraction. The pre conversion
completed in 2.280 seconds and the post conversion in 2.266 seconds; both
recorded process-tree SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `3ce4af63727dd927666f09bb51555ccfd60e1cf01b4ba7fc674170e8277b9a96`;
- float16 TFLite:
  `ee97304641e2b1330bbbe1f1472fc32a4a4d41d4bdb08a3e660da64b5204ce47`;
- tensor correspondence:
  `a50f21319df0380165e8fee2c47f679ccb1682eee965fbd3b0f05ad02cc3d276`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The preceding sequential accuracy checkpoint remains
`max_abs=1.9073486328125e-06`; no duplicate inference was added because the
executed TFLite artifact is identical. All actual model runs remained strictly
sequential.

The raw fallback's known compatibility risk is unchanged. It rebuilds complete
producer and consumer maps in a fixed-point loop and performs relaxed in-place
Squeeze axis updates instead of using the indexed owner's immutable
differential transaction. Hardening requires separate semantic fixtures and is
not mixed into this exact mechanical move.

Changed files are the new pre-unary/Squeeze-suffix compatibility module,
lowerer import and wrapper, expanded indexed/compatibility tests, updated
architecture ownership test, and three branch documents. No public API, CLI,
artifact name, TensorFlow boundary, dependency, corpus profile, exclusion
policy, or ONNX operation-tier policy changed. Temporary tracing and conversion
outputs are removed before commit. PR #952 remains closed; work is commit/push
only and no pull request is created.

The next raw source-order implementation is the 271-line indexed-first
compatibility composite
`_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains`. At restart,
inventory its indexed factorized rank-five owner, singleton and relaxed raw
fallback, GraphIndex construction and LayoutState forwarding, reshape and
permutation constant ownership, combined statistic, single prune/report
boundary, production position, existing positive YOLO ownership, and fallback
fixtures before changing source. Keep inference strictly sequential and
minimal, then commit and push one coherent unit without creating a pull request.

## Factorized/singleton ExpandDims compatibility composite extraction: completed state

The complete 271-line indexed-first
`_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains` composite is
now owned by `passes/expanddims_reshape_compat_layout.py`. The lowerer retains
one private compatibility wrapper at both unchanged production call positions
and forwards Session `LayoutState`. After normalizing only the function name,
the prior lowerer composite and new owner ASTs are identical.

The compatibility module preserves the strict indexed factorized rank-four to
rank-five Case B semantic owner, one per-call `ModelIRGraphIndex`, caller
LayoutState, combined statistic, singleton Case A and relaxed raw fallback,
Reshape shape and post-permutation constant/option updates, fixed-point restart,
exact mutation/removal order, sole prune/report boundary, and removal of pruned
names from LayoutState. The indexed immutable plan and module are unchanged.

The focused indexed/compatibility suite plus both historical direct singleton
fixtures and the architecture ownership selector passes `10 passed in 1.91s`.
It covers indexed Case B, singleton Case A fallback, shared-constant atomic
rejection, bounded dispatch, stale-plan rejection, determinism, complete
compatibility-owner/lowerer-wrapper equality, and LayoutState cleanup. The
changed-file branch regression collection including both historical direct
fixtures passes `494 passed in 29.63s`; the optional-TensorFlow import-blocked
suite passes `11 passed in 9.36s`.

`yolo_test.onnx` is the strictly sequential positive artifact control. The
previously established indexed invocation counts remain `3,0,0,0`, with zero
raw-fallback rewrites for those accepted candidates. The pre conversion
completed in 3.347 seconds and the post conversion in 3.378 seconds; both
recorded process-tree SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `439d9a8b893bf6bfbd92aa0155bd15a4185b5fcdb6e65ddb48f718a41b75bdfc`;
- float16 TFLite:
  `7b1ef8b13de65068b3fe8166d5481553e2e41194c0cfe9ee48f4be5ad3417eff`;
- tensor correspondence:
  `36d728e9294f1d4f1319c45306a088bced6b54ad393f71f4925f3178f0d9c1ca`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The preceding sequential accuracy checkpoint remains
`max_abs=2.4437904357910156e-06`; no duplicate inference was added because the
executed TFLite artifact is identical. All actual model runs remained strictly
sequential.

The raw fallback's known compatibility risk is unchanged. It rebuilds a
complete consumer map in a fixed-point loop and uses relaxed in-place Reshape
and post-permutation constant updates instead of the indexed owner's immutable
differential transaction. The existing shared-constant guard remains intact;
broader hardening requires separate semantic fixtures and is not mixed into
this exact mechanical move.

Changed files are the new factorized/singleton ExpandDims compatibility module,
lowerer import and wrapper, expanded indexed/compatibility tests, updated
architecture ownership test, and three branch documents. No public API, CLI,
artifact name, TensorFlow boundary, dependency, corpus profile, exclusion
policy, or ONNX operation-tier policy changed. Temporary conversion outputs
are removed before commit. PR #952 remains closed; work is commit/push only and
no pull request is created.

The next raw source-order implementation is the 175-line indexed-first
compatibility composite
`_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains`. At restart,
inventory its indexed static flatten-HW owner, dynamic and relaxed raw fallback,
GraphIndex construction and LayoutState forwarding, Reshape constant ownership,
combined statistic, single prune/report boundary, both production positions,
existing positive LINEA ownership, and fallback fixtures before changing
source. Keep inference strictly sequential and minimal, then commit and push
one coherent unit without creating a pull request.

## Static/dynamic flatten-HW compatibility composite extraction: completed state

The complete 175-line indexed-first
`_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains` composite is
now owned by `passes/flatten_hw_reshape_compat_layout.py`. The lowerer retains
one private compatibility wrapper at both unchanged production call positions
and forwards Session `LayoutState`. After normalizing only the function name,
the prior lowerer composite and new owner ASTs are identical.

The compatibility module preserves the strict indexed static flatten-HW
semantic owner, one per-call `ModelIRGraphIndex`, caller LayoutState, combined
statistic, dynamic-signature and relaxed raw fallback, Reshape shape constant
and option updates, fixed-point restart, exact mutation/removal order, sole
prune/report boundary, and removal of pruned names from LayoutState. The indexed
immutable plan and module are unchanged.

The eight-case indexed/compatibility suite plus the architecture ownership
selector passes `9 passed in 1.93s`. It covers the indexed static path,
dynamic-signature fallback, shared/boundary/produced/variable shape-constant
atomic rejection, bounded dispatch, stale-plan rejection, determinism,
complete compatibility-owner/lowerer-wrapper equality, and LayoutState cleanup.
The changed-file branch regression collection passes `500 passed in 29.01s`;
the optional-TensorFlow import-blocked suite passes `11 passed in 9.22s`.

Tier 2 `LINEA.onnx` is the strictly sequential positive artifact control. The
previously established indexed invocation counts remain `2,0,0,0`, with zero
raw-fallback rewrites for those accepted candidates. The pre conversion
completed in 7.758 seconds and the post conversion in 7.975 seconds; both
recorded process-tree SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `fd91ea915b600b3581e8e0e68925fefd5302cd1bfb373ebca8b9b9410138c611`;
- float16 TFLite:
  `c8e44a48221eeead187869d93dfef1f7775420335aae5c63873118738d39f9a8`;
- tensor correspondence:
  `ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The preceding sequential accuracy checkpoint remains
`max_abs=0.002297189086675644`; no duplicate inference was added because the
executed TFLite artifact is identical. All actual model runs remained strictly
sequential.

The raw fallback's known compatibility risk is unchanged. It rebuilds a
complete consumer map in a fixed-point loop and uses relaxed in-place Reshape
constant updates instead of the indexed owner's immutable differential
transaction. Existing graph-boundary, producer, variable-state, and exclusive-
consumer guards remain intact; broader hardening requires separate semantic
fixtures and is not mixed into this exact mechanical move.

Changed files are the new static/dynamic flatten-HW compatibility module,
lowerer import and wrapper, expanded indexed/compatibility tests, updated
architecture ownership test, and three branch documents. No public API, CLI,
artifact name, TensorFlow boundary, dependency, corpus profile, exclusion
policy, or ONNX operation-tier policy changed. Temporary conversion outputs
are removed before commit. PR #952 remains closed; work is commit/push only and
no pull request is created.

The next raw source-order implementation is the 218-line compatibility rewrite
`_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains`. At
restart, inventory both rank-adapter families, public-boundary and fan-out
guards, shape and permutation constant behavior, fixed-point restart and prune/
report boundaries, both production positions, existing fixtures, and short
zero-SWAP real-model ownership before changing source. Keep inference strictly
sequential and minimal, then commit and push one coherent unit without creating
a pull request.

## Static/relaxed attention-QKV compatibility composite extraction: completed state

The intervening 218-line rank-3-to-NHWC reshape helper was re-audited before
this unit and remains intentionally unchanged. Its previous all-active Tier 0-4
scan found zero complete production owners and no public compatibility fixture;
the recorded no-change decision therefore still applies. No synthetic-only
replacement was introduced.

The complete 245-line indexed-first
`_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains`
composite is now owned by `passes/attention_qkv_reshape_compat_layout.py`. The
lowerer retains one private compatibility wrapper at both unchanged production
call positions and forwards Session `LayoutState`. After normalizing only the
function name, the prior lowerer composite and new owner ASTs are identical.

The compatibility module preserves the strict indexed static HAD semantic
owner, one per-call `ModelIRGraphIndex`, caller LayoutState, combined statistic,
HDA `[1,2,0]`, shared-constant copy-on-write, dynamic-signature, and relaxed
raw fallback, shape/permutation constant cloning and updates, fixed-point
restart, exact mutation/removal order, sole prune/report boundary, and removal
of pruned names from LayoutState. The indexed immutable plan and module are
unchanged.

The eight-case indexed/compatibility suite plus the architecture ownership
selector passes `9 passed in 1.93s`. It covers the indexed static HAD path, HDA
fallback, shared-constant copy-on-write, dynamic fallback, bounded dispatch,
stale-plan rejection, determinism, complete compatibility-owner/lowerer-wrapper
equality, and LayoutState cleanup. The changed-file branch regression
collection passes `508 passed in 28.47s`; the optional-TensorFlow import-blocked
suite passes `11 passed in 9.28s`.

Tier 3 `rf-detr-nano.onnx` is the strictly sequential positive artifact
control. The previously established indexed invocation counts remain
`5,0,0,0`, with zero raw-fallback rewrites for those accepted candidates. The
pre conversion completed in 10.928 seconds and the post conversion in 11.288
seconds; both recorded process-tree SWAP zero. The core artifacts are byte-
identical:

- float32 TFLite:
  `fda7d97eaad2b19ee2ac31411099067e78b747515952b7c65ba52a0f1454f1fb`;
- float16 TFLite:
  `a80051b2d6bb871ee871f0d1528e1ea7c8d4e7f6ecbfc16daec4fa78d696fd1f`;
- tensor correspondence:
  `262235cec5a8df73ff2afd7f1eb28678cc7312f4a19dd09d278fd8db77cbdec4`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The preceding sequential accuracy checkpoint remains
`max_abs=0.000102996826171875`; no duplicate inference was added because the
executed TFLite artifact is identical. All actual model runs remained strictly
sequential.

The raw fallback's known compatibility risk is unchanged. It rebuilds a
complete consumer map in a fixed-point loop and performs relaxed clone-on-write
shape/permutation mutations instead of the indexed owner's immutable
differential transaction. Broader hardening requires separate semantic
fixtures and is not mixed into this exact mechanical move.

Changed files are the new static/relaxed attention-QKV compatibility module,
lowerer import and wrapper, expanded indexed/compatibility tests, updated
architecture ownership test, and three branch documents. No public API, CLI,
artifact name, TensorFlow boundary, dependency, corpus profile, exclusion
policy, or ONNX operation-tier policy changed. Temporary conversion outputs
are removed before commit. PR #952 remains closed; work is commit/push only and
no pull request is created.

The adjacent 293-line attention Gather cleanup and 190-line attention pre-
projection rank-lift helpers retain their previously measured no-change
decisions: active-corpus scans and selected ModelIR conversions found zero
complete production owners. At restart, preserve those decisions and inventory
the next raw source-order implementation with an existing public fixture,
`_optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains`, before
changing source. Confirm real ownership, mutation dependencies, production
position, and a short zero-SWAP artifact control first. Then commit and push one
coherent unit without creating a pull request.

## Terminal Transpose/Mul/Add/PReLU compatibility extraction: completed state

The complete 295-line
`_optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains` helper is
now owned by `passes/terminal_affine_prelu_layout.py`. The lowerer retains one
private one-call wrapper at the unchanged ordered production statement, which
is reached through four runtime recovery invocations. After normalizing only
the function name, the prior lowerer owner and new pass owner ASTs are
identical. The moved inherited unused producer-map snapshot is explicitly
marked `noqa` without changing the AST or runtime behavior.

The owner preserves commutative affine inputs, NCHW-to-NHWC channel-constant
rotation, shared-constant copy-on-write, multiple post-Transpose aliases,
retained legacy NCHW consumers through one reverse adapter, tensor metadata and
quantization propagation, exact mutation/removal order, fixed-point restart,
sole prune/report boundary, and the historical statistic.

The former direct-builder fixture is removed from the giant test module and
now lives in `test_flatbuffer_direct_terminal_affine_prelu_layout.py`. It runs
the module owner and lowerer wrapper on deep copies, compares the complete
ModelIR, and fixes the positive terminal rewrite plus retained legacy adapter.
Together with the architecture owner and ordered-production selectors it passes
`3 passed, 754 deselected in 4.45s`. The changed-file branch regression
collection passes `510 passed in 28.41s`; the optional-TensorFlow import-blocked
suite passes `11 passed in 9.28s`.

Tier 2 `sinet_320_op.onnx` is the strictly sequential zero-owner artifact
control. Its four runtime counts remain `0,0,0,0` before and after extraction.
Both conversions completed in 2.413 seconds and recorded process-tree SWAP
zero. The core artifacts are byte-identical:

- float32 TFLite:
  `40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`;
- float16 TFLite:
  `180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`;
- tensor correspondence:
  `24c423ea51b26b178d3764be027855e797bbf9b5ba1930810d2e1dbe281d8e25`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The preceding sequential SiNet accuracy checkpoint remains
`max_abs=2.572051016613841e-09`; no duplicate inference was added because the
executed TFLite artifact is identical. Positive production ownership is not
claimed; the module contract is fixed by the relocated public fixture. All
actual model runs remained strictly sequential.

The raw owner's known compatibility risk is unchanged. It rebuilds complete
producer and consumer maps in an unbounded fixed-point loop, and sequentially
rotates three constants before the whole candidate is known to be valid. A
later constant rejection can therefore leave an earlier exclusive constant
mutated despite a zero statistic. Correcting that requires an immutable
transaction and independent semantic fixtures and is not mixed into this exact
mechanical move.

Changed files are the new terminal affine/PReLU pass, lowerer import and
wrapper, relocated focused fixture, giant-test import/removal, updated
architecture ownership test, and three branch documents. No public API, CLI,
artifact name, TensorFlow boundary, dependency, corpus profile, exclusion
policy, or ONNX operation-tier policy changed. Temporary tracing and conversion
outputs are removed before commit. PR #952 remains closed; work is commit/push
only and no pull request is created.

The next raw source-order implementation with an existing public fixture is the
359-line `_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains`. At
restart, inventory its axis remapping, constant ownership, partial-mutation
risks, statistics, five production invocations, the existing direct fixture,
and the earlier measured zero-owner decision before changing source. Use the
smallest sequential zero-SWAP artifact control, then commit and push one
coherent unit without creating a pull request.

## Transpose/Mean/Mul/Add compatibility extraction: completed state

The complete 359-line
`_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains` helper is now owned
by `passes/mean_affine_prepost_layout.py`. The lowerer retains one private one-
call wrapper at all three unchanged ordered source call positions, reached
through five runtime invocations. After normalizing only the function name, the
prior lowerer owner and new pass owner ASTs are identical.

The owner preserves NCHW-to-NHWC reduction-axis remapping, commutative affine
inputs, static broadcast validation, channel-constant rotation and copy-on-
write, post-Transpose alias collapse, tensor metadata and quantization
propagation, exact mutation/removal order, fixed-point restart, sole prune/
report boundary, and the historical statistic.

The former direct-builder axis-remap fixture is removed from the giant test
module and now lives in `test_flatbuffer_direct_mean_affine_prepost_layout.py`.
It runs the module owner and lowerer wrapper on deep copies, compares the
complete ModelIR, and fixes exact `[2,3]` NCHW reduction axes remapping to
`[1,2]` NHWC axes. Together with the architecture owner/production selector it
passes `2 passed in 1.90s`. The changed-file branch regression collection
passes `512 passed in 27.96s`; the optional-TensorFlow import-blocked suite
passes `11 passed in 9.29s`.

Tier 2 `LINEA.onnx` is the strictly sequential zero-owner artifact control. Its
five runtime counts remain `0,0,0,0,0` before and after extraction. The pre
conversion completed in 7.869 seconds and the post conversion in 7.868 seconds;
both recorded process-tree SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `fd91ea915b600b3581e8e0e68925fefd5302cd1bfb373ebca8b9b9410138c611`;
- float16 TFLite:
  `c8e44a48221eeead187869d93dfef1f7775420335aae5c63873118738d39f9a8`;
- tensor correspondence:
  `ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The preceding sequential LINEA accuracy checkpoint remains
`max_abs=0.002297189086675644`; no duplicate inference was added because the
executed TFLite artifact is identical. Positive production ownership is not
claimed; the module contract is fixed by the relocated public fixture. All
actual model runs remained strictly sequential.

The raw owner's known compatibility risk is unchanged. It rebuilds the
complete consumer map in an unbounded fixed-point loop, and can update the axes
tensor and affine constants before the whole candidate is represented by an
immutable plan. Correcting that requires an all-or-nothing transaction and
independent semantic fixtures and is not mixed into this exact mechanical move.

Changed files are the new Mean/Mul/Add pass, lowerer import and wrapper,
relocated focused fixture, giant-test import/removal, updated architecture
ownership test, and three branch documents. No public API, CLI, artifact name,
TensorFlow boundary, dependency, corpus profile, exclusion policy, or ONNX
operation-tier policy changed. Temporary tracing and conversion outputs are
removed before commit. PR #952 remains closed; work is commit/push only and no
pull request is created.

The next raw source-order implementation with an existing public fixture is the
317-line `_optimize_batchmatmul_affine_transpose_input_chains`. At restart,
inventory both input-side transpose/affine families, constant ownership,
statistics, production positions, the existing direct fixture, and the
smallest sequential zero-SWAP real-model control before changing source. Then
commit and push one coherent unit without creating a pull request.

## Dual affine-input BatchMatMul compatibility extraction: completed state

The complete 317-line `_optimize_batchmatmul_affine_transpose_input_chains`
helper is now owned by `passes/batchmatmul_affine_input_layout.py`. The lowerer
retains one private one-call wrapper at both unchanged ordered production
positions. After normalizing only the function name, the prior lowerer owner
and new pass owner ASTs are identical.

The owner preserves commutative Mul/Add inputs, exact exclusive branch
matching, NCHW-to-NHWC channel-constant rotation, rank-three Reshape shape
reversal for both branches, left post-Transpose removal, `adjY=True`
conversion, tensor metadata propagation, exact mutation/removal order, fixed-
point restart, sole prune/report boundary, and the historical statistic.

The former direct-builder dual-branch fixture is removed from the giant test
module and now lives in
`test_flatbuffer_direct_batchmatmul_affine_input_layout.py`. It runs the module
owner and lowerer wrapper on deep copies, compares the complete ModelIR, and
fixes both affine branches, shape vectors, removed Transposes, and the adjoint
flag. Together with the architecture owner/production selector it passes
`2 passed in 1.94s`. The changed-file branch regression collection passes
`514 passed in 27.20s`; the optional-TensorFlow import-blocked suite passes
`11 passed in 9.27s`.

Tier 2 `LINEA.onnx` is the strictly sequential zero-owner artifact control. Its
two runtime counts remain `0,0` before and after extraction. The pre conversion
completed in 7.827 seconds and the post conversion in 7.917 seconds; both
recorded process-tree SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `fd91ea915b600b3581e8e0e68925fefd5302cd1bfb373ebca8b9b9410138c611`;
- float16 TFLite:
  `c8e44a48221eeead187869d93dfef1f7775420335aae5c63873118738d39f9a8`;
- tensor correspondence:
  `ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The preceding sequential LINEA accuracy checkpoint remains
`max_abs=0.002297189086675644`; no duplicate inference was added because the
executed TFLite artifact is identical. Positive production ownership is not
claimed; the module contract is fixed by the relocated public fixture. All
actual model runs remained strictly sequential.

The raw owner's known compatibility risk is unchanged. It mutates both affine
branches sequentially before every Reshape shape constant is known valid. A
late failure can therefore leave input rewires, tensor metadata, and rotated
constants from an earlier branch despite a zero statistic. Correcting that
requires an immutable all-or-nothing transaction and independent semantic
fixtures and is not mixed into this exact mechanical move.

Changed files are the new BatchMatMul affine-input pass, lowerer import and
wrapper, relocated focused fixture, giant-test import/removal, updated
architecture ownership test, and three branch documents. No public API, CLI,
artifact name, TensorFlow boundary, dependency, corpus profile, exclusion
policy, or ONNX operation-tier policy changed. Temporary tracing and conversion
outputs are removed before commit. PR #952 remains closed; work is commit/push
only and no pull request is created.

The next raw source-order implementation with an existing public fixture is the
363-line `_optimize_batchmatmul_reshape_se_nhwc_chains`. At restart, inventory
its Mean/Conv/gate branch, affine constants, statistics, production positions,
the existing direct fixture, and the smallest sequential zero-SWAP real-model
control before changing source. Then commit and push one coherent unit without
creating a pull request.

## BatchMatMul-to-SE layout compatibility extraction: completed state

The complete 363-line `_optimize_batchmatmul_reshape_se_nhwc_chains` helper is
now owned by `passes/batchmatmul_se_layout.py`. The lowerer retains one private
one-call wrapper at both unchanged ordered production positions. After
normalizing only the function name, the prior lowerer owner and new pass owner
ASTs are identical.

The owner preserves the BatchMatMul/Reshape source, NCHW Mean and axis remap,
NHWC Conv gate branch, reverse gate adapter, Logistic and residual Mul merge,
constant updates, alias rewiring, tensor metadata and quantization propagation,
exact mutation/removal order, fixed-point restart, sole prune/report boundary,
and the historical statistic.

The former direct-builder SE fixture is removed from the giant test module and
now lives in `test_flatbuffer_direct_batchmatmul_se_layout.py`. It runs the
module owner and lowerer wrapper on deep copies, compares the complete ModelIR,
and fixes the Mean axes, Conv gate branch, affine merge, removed Transposes, and
output aliases. Together with the architecture owner/production selector it
passes `2 passed in 1.92s`. The changed-file branch regression collection
passes `516 passed in 27.02s`; the optional-TensorFlow import-blocked suite
passes `11 passed in 9.29s`.

Tier 2 `LINEA.onnx` is the strictly sequential zero-owner artifact control. Its
two runtime counts remain `0,0` before and after extraction. The pre conversion
completed in 7.903 seconds and the post conversion in 7.939 seconds; both
recorded process-tree SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `fd91ea915b600b3581e8e0e68925fefd5302cd1bfb373ebca8b9b9410138c611`;
- float16 TFLite:
  `c8e44a48221eeead187869d93dfef1f7775420335aae5c63873118738d39f9a8`;
- tensor correspondence:
  `ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The preceding sequential LINEA accuracy checkpoint remains
`max_abs=0.002297189086675644`; no duplicate inference was added because the
executed TFLite artifact is identical. Positive production ownership is not
claimed; the module contract is fixed by the relocated public fixture. All
actual model runs remained strictly sequential.

The raw owner's known compatibility risk is unchanged. It performs a long
sequence of axes, constant, option, edge, metadata, output, and alias mutations
without first representing the complete candidate as an immutable plan. A late
failure can therefore leave partial state despite a zero statistic. Correcting
that requires an all-or-nothing transaction and independent semantic fixtures
and is not mixed into this exact mechanical move.

Changed files are the new BatchMatMul SE pass, lowerer import and wrapper,
relocated focused fixture, giant-test import/removal, updated architecture
ownership test, and three branch documents. No public API, CLI, artifact name,
TensorFlow boundary, dependency, corpus profile, exclusion policy, or ONNX
operation-tier policy changed. Temporary tracing and conversion outputs are
removed before commit. PR #952 remains closed; work is commit/push only and no
pull request is created.

The adjacent 145-line `_optimize_batchmatmul_transpose_input_to_adj_flags`
helper has no dedicated public fixture. At restart, inspect its indirect tests,
two production positions, transpose/adjoint guards, and short zero-SWAP real-
model ownership before deciding whether evidence is sufficient for a mechanical
move. Do not introduce a synthetic-only replacement merely to advance source
order; commit and push only a coherent verified unit and do not create a pull
request.

## Rank-three BatchMatMul adjoint-input extraction: completed state

The complete 145-line `_optimize_batchmatmul_transpose_input_to_adj_flags`
helper is now owned by `passes/batchmatmul_adjoint_layout.py`. The lowerer
retains one private one-call wrapper at both unchanged ordered production
positions. After normalizing only the function name, the prior lowerer owner
and new pass owner ASTs are identical; the sole Ruff suppression is a comment
on the historical unused `removed_transpose` assignment and does not alter the
AST.

The owner preserves exclusive Transpose-output ownership, graph-output
protection, fully known positive shape checks, exact permutation/shape
validation, direct `[0,2,1]` Transpose removal, singleton-preserving
Transpose-to-Reshape conversion, deterministic INT32 shape-tensor creation,
the corresponding `adjX` or `adjY` toggle, exact mutation/removal order, fixed-
point restart, conditional pruning, and the historical statistic.

The new focused fixture runs the module owner and lowerer wrapper on deep
copies and compares the complete ModelIR. It covers both BatchMatMul input
positions, direct Transpose removal, singleton-preserving Reshape conversion,
shape-tensor payload, both adjoint toggles, pruning, and idempotence. Together
with the architecture owner/production selector it passes `2 passed in 1.88s`.
The changed-file branch regression collection passes `518 passed in 27.32s`;
the optional-TensorFlow import-blocked suite passes `11 passed in 9.24s`.
Ruff, Python compilation, AST equivalence, and whitespace checks pass.

Tier 0 `speech_command_classifier_trained.onnx` is the strictly sequential
positive artifact control. Its two runtime counts remain `1,0` before and
after extraction. The conversion-only pre run completed in 0.240 seconds and
the post run in 0.239 seconds; both recorded process-tree SWAP zero. The core
artifacts are byte-identical:

- float32 TFLite:
  `2cb2ff30c92901f802c32c483ae201ef45b1ea35520fb958cdbedc35e7b11cbf`;
- float16 TFLite:
  `eb11dd7d3120e06da1227d7f7f7c66482b5c0c56a4d5edf3ea18064f85778f44`;
- tensor correspondence:
  `b7abdafa2cd2f8bec4bc3e060c9006913f924eafb7ccc2f0eca9f4304c8a86da`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The single sequential pre-move accuracy checkpoint passed with
`evaluation_pass=true`, no skipped output, and
`max_abs=2.86102294921875e-06`. No duplicate post-move inference was added
because the executed float32 TFLite artifact is identical. Existing isolated
accuracy workers ran one at a time; all model conversions remained strictly
sequential.

The raw owner's known compatibility risk is unchanged. It rebuilds complete
producer and consumer maps after each accepted adapter and directly deletes or
mutates an operator without a transaction or shared GraphIndex/LayoutState.
The accepted candidate has no late rejection after mutation begins, but an
indexed transactional migration must separately prove candidate order,
fixed-point restart, pruning, and exact artifact equivalence and is not mixed
into this mechanical move.

Changed files are the new BatchMatMul adjoint-input pass, lowerer import and
wrapper, new focused fixture, updated architecture ownership test, and three
branch documents. No public API, CLI, artifact name, TensorFlow boundary,
dependency, corpus profile, exclusion policy, or ONNX operation-tier policy
changed. Temporary tracing and conversion outputs are removed before commit.
PR #952 remains closed; work is commit/push only and no pull request is created.

The next raw source-order implementation is the 245-line
`_sanitize_probable_nhwc_axis_sensitive_ops`. Its only direct fixture currently
fixes the explicit-NCHW no-op. At restart, inventory its positive
SPLIT/CONCATENATION/PACK/UNPACK axis repairs, terminal output-adapter branch,
constant copy-on-write, both production positions, and the smallest sequential
zero-SWAP real-model owner before changing source. Do not extract an
insufficiently characterized positive owner merely to advance source order;
commit and push only a coherent verified unit and do not create a pull request.

## Probable-NHWC axis sanitizer characterization: completed state

The 245-line `_sanitize_probable_nhwc_axis_sensitive_ops` implementation
remains in `lower_from_onnx2tf.py`; this checkpoint deliberately does not move
or change production code. Its former sole direct fixture, the explicit-NCHW
Concat no-op, is removed from the giant direct-builder test module and placed
in `test_flatbuffer_direct_probable_nhwc_sanitizer.py` with three new positive
characterization cases.

The four-case focused contract now fixes:

- SPLIT axis 1 to axis 3 repair on a probable-NHWC input, shared-axis copy-on-
  write, output metadata repair, unary propagation, and terminal NHWC-to-NCHW
  graph-output adapter insertion;
- CONCATENATION axis 1 to axis 3 repair and SLICE begin/size rotation with
  exact output shapes;
- metadata-only unary and binary broadcast propagation, including its zero
  rewrite statistic;
- preservation of an explicit NCHW Concat axis and public NCHW output.

The focused suite passes `4 passed in 0.48s`. The changed-file branch
regression collection passes `522 passed in 26.91s`; Ruff checks pass. No
optional TensorFlow rerun is needed for this test-only checkpoint because no
runtime or import boundary changed.

Four strictly sequential real-model traces establish current production
ownership evidence without a broad sweep. FastestDet has four zero results in
0.795 seconds, SiNet has eight zero results in 1.121 seconds,
inference_ops15 has four zero results in 0.785 seconds, and LINEA has four zero
results in 5.650 seconds. Every conversion succeeded and every process-tree
monitor recorded SWAP zero. Positive production ownership is therefore not
claimed; the positive semantic branches are fixed by the dedicated synthetic
contract.

The previous restart note incorrectly named PACK/UNPACK branches. The actual
owner handles SPLIT, CONCATENATION, SLICE, unary metadata, binary broadcast
metadata, and conditional terminal output adapters; this checkpoint corrects
the record.

Changed files are the new focused sanitizer characterization module, removal
of its relocated import/test from the giant direct-builder test, and two branch
documents. No production source, public API, CLI, artifact, TensorFlow
boundary, dependency, corpus profile, exclusion policy, or ONNX operation-tier
policy changed. Temporary trace and conversion outputs are removed before
commit. PR #952 remains closed; work is commit/push only and no pull request is
created.

At restart, compare the complete old helper AST with a proposed pass owner,
preserve both ordered production positions, and extend the four cases to run
the owner and compatibility wrapper on deep copies. Use one of the fixed short
zero-owner models for sequential pre/post byte-equivalence and SWAP control.
Do not claim positive production ownership, and do not mix a semantic or
indexed rewrite into the exact mechanical ownership move.

## Probable-NHWC axis sanitizer extraction: completed state

The complete 245-line `_sanitize_probable_nhwc_axis_sensitive_ops` helper is
now owned by `passes/probable_nhwc_axis_sanitizer.py`. The lowerer retains one
private one-call wrapper at both unchanged ordered production positions. After
normalizing only the function name, the prior lowerer owner and new pass owner
ASTs are identical.

The owner preserves the probable-NHWC shape heuristic, SPLIT axis repair and
shared-axis copy-on-write, CONCATENATION axis repair, SLICE begin/size
rotation, unary and binary output-metadata propagation, explicit/public NCHW
guards, conditional terminal NHWC-to-NCHW graph-output adapters, fixed-point
restart, exact mutation/insertion order, and both historical statistics.

All four characterization cases now run the module owner and lowerer wrapper
on deep copies and compare the complete ModelIR. Together with the architecture
owner/production selector the focused gate passes `5 passed in 1.99s`. The
changed-file branch regression collection passes `523 passed in 28.09s`; the
optional-TensorFlow import-blocked suite passes `11 passed in 9.42s`. Ruff,
Python compilation, full old/new AST equivalence, and whitespace checks pass.

FastestDet is the strictly sequential zero-owner artifact control. Its four
runtime results remain zero before and after extraction. The conversion-only
pre run completed in 0.802 seconds and the post run in 0.783 seconds; both
recorded process-tree SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `3bdbec5d7ad81f98cf7890fbf1a98570ebeb1a4a5c19883aca23733b31e1573b`;
- float16 TFLite:
  `a14bad05eba99dc211a09aa820eb38396329b98168a2d4b20e463eb64deab617`;
- tensor correspondence:
  `2bd03e9e775b4dede0310813cf36e2efc3ad9d0635ce0c5797895fe18d7fb074`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The preceding sequential FastestDet accuracy checkpoint remains
`max_abs=1.3113021850585938e-05`; no duplicate inference was added because the
executed TFLite artifact is identical. Positive production ownership is not
claimed; all positive semantic branches are fixed by the dedicated four-case
contract. All actual model runs remained strictly sequential.

The raw owner's known compatibility risks are unchanged. It rebuilds complete
producer/consumer maps in its fixed-point loop, mutates SLICE constants without
copy-on-write, treats metadata-only unary/binary changes outside the rewrite
statistic, and inserts terminal operators directly without a transaction or
shared GraphIndex/LayoutState. Semantic hardening and indexed migration require
independent fixtures and are not mixed into this exact ownership move.

Changed files are the new probable-NHWC sanitizer pass, lowerer import and
wrapper, owner/wrapper-focused fixture, architecture ownership test, and three
branch documents. No public API, CLI, artifact name, TensorFlow boundary,
dependency, corpus profile, exclusion policy, or ONNX operation-tier policy
changed. Temporary tracing and conversion outputs are removed before commit.
PR #952 remains closed; work is commit/push only and no pull request is created.

The next raw source-order implementation is the 207-line
`_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains`. It has no
dedicated direct fixture. At restart, inventory multi-input pre-Transpose
ownership, the allowed elementwise closure, constant rotation, fan-out and
public-boundary guards, post-Transpose alias handling, its single production
position, and the smallest sequential zero-SWAP real-model owner before
changing source. Do not introduce a synthetic-only extraction merely to
advance source order; commit and push only a coherent verified unit and do not
create a pull request.

## NCHW/NHWC elementwise roundtrip root-metadata correction: completed state

The 207-line `_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains`
implementation remains in `lower_from_onnx2tf.py`; this checkpoint deliberately
does not extract it. A new focused module first characterized one positive
multi-input elementwise closure plus pre-Transpose fan-out and public post-
output rejection.

The positive characterization exposed a pre-existing metadata defect before
any correction was applied. The owner permuted every elementwise subgraph
output from NHWC to NCHW, including the private root, then copied that already-
permuted root metadata to the canonical post-Transpose output and permuted it a
second time. For a `[1,8,8,3]` root this produced `[1,8,3,8]` rather than the
required `[1,3,8,8]`. The issue was recorded as a strict xfail before the
implementation changed.

The safe correction excludes only `root_nhwc_name` from the intermediate-
output metadata loop. Existing guards prove that this private root is consumed
only by the matched post-Transpose and is not public. Intermediate tensors are
still permuted once; the root metadata is copied to the canonical output and
permuted exactly once. Rewiring, constants, alias replacement, removal order,
fixed-point behavior, pruning, and the historical statistic are unchanged.

The focused positive, fan-out rejection, public-output rejection, pruning, and
idempotence suite passes `3 passed in 0.53s`. The changed-file branch regression
collection passes `526 passed in 26.93s`; the optional-TensorFlow import-blocked
suite passes `11 passed in 9.34s`. Ruff and whitespace checks pass. There are no
remaining xfails in this focused module.

Tier 1 `gaze_estimation_adas_0002.onnx` is the strictly sequential zero-owner
artifact control. A read-only ONNX topology scan found seven structurally
similar Transpose/elementwise/inverse-Transpose regions, but earlier lowering
passes eliminate or alter them before this helper. Its four runtime results
remain zero before and after the correction. The pre conversion completed in
0.398 seconds and the post conversion in 0.395 seconds; both recorded process-
tree SWAP zero. The core artifacts are byte-identical:

- float32 TFLite:
  `fe026fa4d996ab526e2c65506c83c0f3b709f381fc9247a5453c8731abdf70c5`;
- float16 TFLite:
  `392c1312bde822bd4f824d9fdca19612cf07018ac8cdbac3303407530d4a2b55`;
- tensor correspondence:
  `002b47c50efda861d76b941f97992a7dae1a6d8758d627f182340bf954dca272`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The active Tier 0-4 accuracy baseline remains
`max_abs=1.2665987014770508e-07`; no duplicate inference was added because the
executed TFLite artifact is identical. Positive production ownership is not
claimed. All model runs remained strictly sequential.

Changed files are the one-line guarded lowerer correction, the new focused
elementwise-roundtrip characterization module, and three branch documents. No
public API, CLI, artifact name, TensorFlow boundary, dependency, corpus
profile, exclusion policy, or ONNX operation-tier policy changed. Temporary
tracing and conversion outputs are removed before commit. PR #952 remains
closed; work is commit/push only and no pull request is created.

Do not mechanically extract this corrected helper until positive production
ownership is observed or a later checkpoint explicitly accepts the fixed
zero-owner evidence. At restart, inventory the adjacent 555-line
`_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains` helper's
existing fixtures, fan-out/constant contracts, production positions, and short
zero-SWAP real ownership before choosing the next evidence-backed unit.

## Opposite-direction elementwise fan-out extraction: completed state

The complete 555-line
`_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains` helper is
now owned by `passes/elementwise_fanout_layout.py`. The lowerer retains one
private one-call wrapper at all three unchanged ordered production positions.
After normalizing only the function name, the prior lowerer owner and new pass
owner ASTs are identical.

The owner preserves forward elementwise-DAG discovery, the conservative
external-runtime-input rejection, local/shared NCHW per-channel constant
rotation, inverse boundary-Transpose collapse, canonical aliases, legacy NCHW
adapter retention, metadata and quantization propagation, candidate deep-copy
snapshots, unbound-input rollback, exact mutation/removal order, fixed-point
restart, pruning, and the historical statistic. The public unbound-input
validator is imported through a same-name compatibility alias, so the pass has
no reverse dependency on `lower_from_onnx2tf.py`.

The former giant direct-builder fan-out fixture now lives in
`test_flatbuffer_direct_elementwise_fanout_layout.py`. It runs the module owner
and lowerer wrapper on deep copies, compares the complete ModelIR, and fixes
the ERF/SIGN fan-out, three per-channel constants, two inverse boundaries,
canonical aliases, and removed Transposes. Together with the architecture
owner/production selector it passes `2 passed in 1.96s`. The changed-file
branch regression collection passes `528 passed in 26.01s`; the optional-
TensorFlow import-blocked suite passes `11 passed in 9.46s`. Ruff, Python
compilation, full old/new AST equivalence, and whitespace checks pass.

Tier 0 `shadowformer_istd_160x240_split.onnx` is the strictly sequential zero-
owner artifact control. Its six runtime results remain zero before and after
extraction. The pre conversion completed in 0.259 seconds and the post
conversion in 0.261 seconds; both recorded process-tree SWAP zero. The core
artifacts are byte-identical:

- float32 TFLite:
  `b9e9f67c5cd06f9c0bc74f3227257662f5aa4c310d2be4f51d2bb2f7f62e4e94`;
- float16 TFLite:
  `108d88c2ee1cb1af4a6426f545a40aa511f2a9b79405f79498384a6433ba5992`;
- tensor correspondence:
  `0fc970b0eeb31cbc0a62055e1313db32931e91c9792cf19eb795e362915e8114`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The active Tier 0-4 accuracy baseline remains
`max_abs=4.0531158447265625e-06`; no duplicate inference was added because the
executed TFLite artifact is identical. Positive production ownership is not
claimed; the positive contract is the relocated focused fixture. A second
sequential structural candidate, `convnext-det.onnx`, also converted with six
zero results in 1.006 seconds and process-tree SWAP zero.

The raw owner's known compatibility and efficiency risks are unchanged. It
rebuilds complete producer/consumer maps and deep-copies the whole ModelIR for
each candidate, directly mutates operators/tensors, and retains an external-
runtime-input adapter implementation behind an earlier conservative rejection
guard. Replacing these behaviors requires an independently proven indexed
transaction and is not mixed into this exact ownership move.

Changed files are the new elementwise fan-out pass, lowerer import and wrapper,
relocated owner/wrapper-focused fixture, giant-test import/removal, architecture
ownership test, and three branch documents. No public API, CLI, artifact name,
TensorFlow boundary, dependency, corpus profile, exclusion policy, or ONNX
operation-tier policy changed. Temporary tracing and conversion outputs are
removed before commit. PR #952 remains closed; work is commit/push only and no
pull request is created.

The next raw source-order implementation is the 199-line
`_repair_rank4_channelwise_broadcast_constants_to_runtime_layout`. At restart,
inventory its GraphIndex contract, constant ownership/copy-on-write, statistic
aggregation, dedicated binary-layout and indexed-convergence fixtures, all
production positions, and the smallest sequential zero-SWAP real-model owner
before changing source. Keep the corrected 207-line opposite elementwise
helper central until positive production ownership is observed or a later
checkpoint explicitly accepts zero-owner evidence.

## Channelwise broadcast-constant repair extraction: completed state

The complete former 199-line
`_repair_rank4_channelwise_broadcast_constants_to_runtime_layout`
implementation is now owned by `passes/binary_layout_adapter.py`. The lowerer
retains one signature-compatible private wrapper. Its three direct
`lower_onnx_to_ir` positions and the one position inside three-round indexed
binary-layout convergence are unchanged. After normalizing only the function
name, the prior lowerer implementation and new module owner have identical
ASTs.

The owner preserves the existing `ModelIRGraphIndex` contract. It reuses a
valid caller index, otherwise creates one index, snapshots consumers once,
iterates only indexed ADD/SUB/MUL/DIV/MAXIMUM/MINIMUM/POW operators, and uses
the indexed producer lookup for ambiguous runtime-layout hints. Standard
NCHW-to-NHWC constant rotation, exact inverse recovery for stale NHWC rank-four
constants, rank-three `[C,1,1]` handling, dtype/shape/signature updates, and
the historical statistic are unchanged. Exclusive constants mutate in place;
shared constants retain deterministic copy-on-write, quantization cloning, and
differential consumer updates through `_set_operator_inputs` with the shared
index.

The positive owner fixture now runs both the module owner and lowerer wrapper
on deep copies and compares complete `ModelIRPassState` fingerprints. The
existing no-op, shared-constant/clone-policy, no-rescan, differential-index,
three-round convergence, and four historical rank-three/rank-four direct cases
remain active. Architecture tests fix one implementation owner, one wrapper
dispatch with GraphIndex forwarding, exactly three direct lowerer calls, and
one convergence call.

Validation completed as follows:

- binary-layout plus indexed-convergence focused modules: `6 passed`;
- ownership, GraphIndex, and convergence architecture selector: `3 passed`;
- historical direct-builder broadcast cases: `4 passed, 745 deselected`;
- changed-file branch regression collection: `532 passed in 25.64s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.30s`;
- old/new owner AST equivalence and whitespace checks: passed.

FastestDet is the strictly sequential zero-owner artifact control. All five
runtime results remain zero before and after extraction. The pre conversion
completed in 0.790 seconds and the post conversion in 0.815 seconds; both
recorded process-tree SWAP zero. The five core artifacts are byte-identical:

- float32 TFLite:
  `3bdbec5d7ad81f98cf7890fbf1a98570ebeb1a4a5c19883aca23733b31e1573b`;
- float16 TFLite:
  `a14bad05eba99dc211a09aa820eb38396329b98168a2d4b20e463eb64deab617`;
- tensor correspondence:
  `2bd03e9e775b4dede0310813cf36e2efc3ad9d0635ce0c5797895fe18d7fb074`;
- `schema.fbs`:
  `0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`;
- `schema_generated.py`:
  `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.

The preceding sequential FastestDet accuracy checkpoint remains
`max_abs=1.3113021850585938e-05`; duplicate inference was not run because the
executed TFLite artifact is identical. Positive production ownership is not
claimed; the focused synthetic cases remain the semantic authority. All model
runs were strictly sequential, no dependency or TensorFlow boundary changed,
and no broad Tier conversion sweep was performed.

The current branch is `fb-refactor6`. Changed files in this checkpoint are the
binary layout owner module, lowerer import/wrapper, focused binary-layout and
architecture tests, and the architecture, branch-description, and handoff
documents. No public API, CLI, artifact name, corpus profile, exclusion policy,
or ONNX operation-tier policy changed. PR #952 remains closed; this Goal uses
commit and push only and must not create a pull request.

The known limitation is deliberate: this is an exact ownership move, not a
new transactional or semantic rewrite. No measured root model exercised a
non-zero production rewrite; positive behavior is therefore proven by the
focused ModelIR contracts rather than claimed from the corpus. The corrected
207-line opposite elementwise helper remains central under its earlier no-owner
decision.

At restart, first characterize the next raw source-order implementation, the
535-line `_optimize_convpool_output_transpose_nhwc_passthrough_chains`. Inventory
its match/guard/rewrite phases, passthrough closure, legacy adapter retention,
fan-out and public-boundary guards, metadata/constant handling, all production
positions, and the smallest sequential zero-SWAP owner before changing source.
Do not mix characterization and extraction unless the evidence proves the
boundary, and do not create a pull request.

## Conv/Pool output passthrough characterization: completed state

The 535-line
`_optimize_convpool_output_transpose_nhwc_passthrough_chains` remains unchanged
in `lower_from_onnx2tf.py` at its single production position. This checkpoint
does not extract or semantically generalize it. The former 99-line giant
direct-builder success fixture and its private-helper import have moved to the
new compact
`tests/test_flatbuffer_direct_convpool_output_passthrough_layout.py` module.

The focused contract fixes the leading Conv/Pool NHWC-to-NCHW adapter match,
elementwise forward closure, root bypass, private NHWC metadata hints, retained
NHWC-to-NCHW legacy boundary adapter, valid rank-four external-runtime
NCHW-to-NHWC adapter, and keepdims Mean-axis absorption. Seven complete
ModelIR no-op cases cover the wrong permutation, public leading output,
non-Conv/Pool producer, absent elementwise region, non-elementwise root fanout,
public elementwise output, and multi-output elementwise operator. The
architecture selector records one raw owner/call and its current whole-graph
producer/consumer scans, direct append/delete topology mutation, setter calls,
and prune boundary.

Characterization exposes one pre-existing unsafe rejection path. The helper
rewires `pre_output` to `pre_input` before checking that every external runtime
input is static rank four. A rank-three external input therefore returns
`optimized_convpool_output_transpose_nhwc_passthrough_chains: 0` while leaving
the root input changed. A strict xfail records the required atomic no-op; no
production correction is mixed into this test-only checkpoint.

Validation completed as follows:

- focused characterization plus raw-owner selector:
  `11 passed, 242 deselected, 1 xfailed in 0.64s`;
- changed-file branch regression collection:
  `543 passed, 1 xfailed in 26.27s`;
- whitespace checks: passed.

Four small real-model ownership traces ran strictly sequentially with no
inference or broad corpus sweep. FastestDet, HumanSeg
(`human_segmentation_pphumanseg_2021oct_org.onnx`), OSNet, and inference_ops15
each produced one zero rewrite result, completed in 0.789, 0.513, 1.239, and
0.764 seconds, and recorded process-tree SWAP zero. All conversions succeeded.
Positive production ownership is not claimed; the focused synthetic success
case is the current semantic authority.

Changed files are the new focused characterization module, removal of the
relocated giant fixture/import, one architecture selector, and the three
branch documents. Production code, public API, CLI, artifacts, TensorFlow
boundary, dependencies, corpus profiles, exclusion policy, and ONNX operation-
tier policy are unchanged. PR #952 remains closed; work is commit/push only.

At restart, make the smallest atomicity correction: validate every external
runtime tensor and compute its NHWC shape before rewiring any operator or
creating any adapter. Turn the strict xfail into an ordinary passing test, keep
all positive fingerprints and statistics unchanged, and use one short zero-
owner model for pre/post byte-equivalence and SWAP control. Do not extract the
helper in the same commit, and do not create a pull request.

## Conv/Pool external-runtime atomicity correction: completed state

The recorded unsafe rejection path is corrected without extracting or
otherwise generalizing the raw helper. Before the first metadata or graph
mutation, `_optimize_convpool_output_transpose_nhwc_passthrough_chains` now
validates every discovered external runtime tensor, requires a rank-four shape,
computes its NCHW-to-NHWC projected shape, and stores those shapes in a local
plan. Any invalid tensor rejects the candidate before channel-last hints,
rewiring, adapter creation, output renaming, or topology mutation.

The former strict xfail is now an ordinary passing test. It supplies two
external runtime inputs in deterministic name order: the first is valid rank
four and the second is invalid rank three. The complete ModelIR fingerprint and
zero statistic remain unchanged, proving that partial plan construction cannot
leak a mutation. The valid external-input success case consumes the precomputed
shape and retains the same adapter names, tensor metadata, permutation, and
operator order. Architecture assertions fix prevalidation before both the first
channel-last hint and first `_set_operator_inputs` call.

Validation completed as follows:

- focused Conv/Pool contract plus architecture selector:
  `12 passed, 242 deselected in 1.87s`;
- changed-file branch regression collection: `544 passed in 25.66s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.30s`;
- targeted Ruff, Python compilation, and whitespace checks: passed.

FastestDet is the strictly sequential zero-owner artifact control. Its single
runtime result remains zero. The pre/post conversion-only runs completed in
0.783 and 0.791 seconds, both with process-tree SWAP zero. Float32, float16,
tensor-correspondence, schema, and generated-schema artifacts remain byte-
identical with the same five hashes recorded in the preceding channelwise
broadcast checkpoint. Its established accuracy remains
`max_abs=1.3113021850585938e-05`; duplicate inference was not run because the
executed TFLite artifact is identical.

Changed files are the central helper, focused atomicity fixture, architecture
ordering assertion, and three branch documents. No public API, CLI, artifact
name, pass position, statistic, TensorFlow boundary, dependency, corpus profile,
exclusion policy, or ONNX operation-tier policy changed. PR #952 remains
closed; work is commit/push only.

At restart, compare the complete corrected helper AST with a focused pass owner
and explicitly decide whether four measured zero-owner models plus the positive
synthetic contract justify an exact mechanical extraction. If extracted,
preserve the single production call and private compatibility wrapper and do
not mix in GraphIndex, LayoutState, or transactional redesign. Otherwise leave
the helper central and move to the next evidence-backed family. Do not create a
pull request.

## Conv/Pool output passthrough extraction: completed state

The complete corrected 556-line
`_optimize_convpool_output_transpose_nhwc_passthrough_chains` implementation is
now owned by `passes/convpool_output_passthrough_compat.py`. The lowerer retains
one signature-compatible private wrapper at the unchanged single production
position. After normalizing only the function name, the corrected pre-move
lowerer owner and new module owner have identical ASTs.

The module preserves the exact forward elementwise closure, external runtime
dependency discovery and prevalidation, channel-last hint accumulation, root
rewiring, valid external NCHW-to-NHWC adapters, metadata permutation, legacy
NHWC-to-NCHW boundary adapters, keepdims Mean-axis absorption, follow-up
Reshape/Transpose/Squeeze adjustments, direct append/delete order, fixed-point
restart, prune boundary, and historical statistic. It deliberately retains
whole-graph producer/consumer maps and raw topology mutation; GraphIndex,
LayoutState, and transactional redesign are separate future semantic work.

All focused success and rejection cases now run the module owner and lowerer
wrapper on deep copies and compare complete ModelIR fingerprints. This includes
the two-op elementwise closure, seven unsafe no-ops, valid rank-four external
runtime adaptation, keepdims Mean absorption, and the corrected multi-external-
input atomic rejection. The architecture gate fixes the module as the only
implementation owner, one wrapper dispatch, one production call, no reverse
lowerer import, and the prevalidation-before-mutation ordering.

Validation completed as follows:

- full old/new owner AST comparison after function-name normalization: exact;
- focused owner/wrapper plus architecture selector:
  `12 passed, 242 deselected in 1.90s`;
- changed-file branch regression collection: `544 passed in 25.25s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.36s`;
- targeted Ruff, Python compilation, and whitespace checks: passed.

FastestDet remains the strictly sequential zero-owner artifact control. Its
single runtime result remains zero. The pre/post conversion-only runs completed
in 0.788 and 0.807 seconds, both with process-tree SWAP zero. Float32, float16,
tensor-correspondence, schema, and generated-schema outputs are byte-identical
and retain the five established FastestDet hashes. The preceding accuracy
checkpoint remains `max_abs=1.3113021850585938e-05`; duplicate inference was
not run because the executed TFLite artifact is identical. Positive production
ownership is not claimed.

Changed files are the new compatibility owner, lowerer import/wrapper, focused
owner/wrapper corpus, architecture ownership selector, and three branch
documents. No public API, CLI, artifact name, pass order, statistic,
TensorFlow boundary, dependency, corpus profile, exclusion policy, or ONNX
operation-tier policy changed. PR #952 remains closed; work is commit/push
only.

At restart, inventory the next raw 381-line
`_optimize_fold_conv_mul_add_affine_chains` compatibility fallback. Its indexed
owner and focused corpus already exist, while the raw fixed-point fallback and
three production positions remain central. Separate indexed and raw ownership,
statistics, and cleanup boundaries; prove raw candidate order, constant
copy-on-write, activation guards, and the smallest non-zero sequential owner
before changing source. Do not create a pull request.

## Conv/Mul affine compatibility orchestration extraction: completed state

The complete 381-line indexed-first
`_optimize_fold_conv_mul_add_affine_chains` orchestration is now owned by
`passes/conv_mul_affine_fold_compat.py`. The existing bounded indexed owner
remains in `conv_mul_affine_fold.py`; the compatibility owner invokes it first
with one `ModelIRGraphIndex` and caller `LayoutState`, then executes the exact
historical raw fallback. The lowerer keeps one signature-compatible private
wrapper at all three unchanged production positions.

After normalizing only the function name, the pre-move lowerer owner and new
compatibility owner have identical ASTs. Two `noqa` comments merely document
legacy unused locals and do not alter that AST. Add-only, Mul/Add, fused-ReLU,
missing-bias, scalar/dynamic coefficients, shared constants, missing-bias
creation, signed-zero behavior, direct removal order, single prune, four
statistics, and LayoutState removal of pruned tensor names are unchanged.

The focused wrapper boundary now runs the compatibility owner and lowerer
wrapper on deep copies, compares complete ModelIR fingerprints and statistics,
and compares logical/physical LayoutState maps. Existing indexed planning,
stale-plan atomicity, determinism, rewrite bounds, legacy signed-zero bits, and
raw fallback cases remain active. The architecture gate fixes the two-module
ownership, indexed-before-raw order, one compatibility prune, one lowerer
dispatch, and three production calls with Session LayoutState.

Validation completed as follows:

- full old/new orchestration AST comparison after name normalization: exact;
- focused indexed/compatibility owner plus architecture selector:
  `12 passed, 242 deselected in 1.89s`;
- changed-file branch regression collection: `544 passed in 24.55s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.29s`;
- targeted Ruff, Python compilation, and whitespace checks: passed.

Tier 2 `iat_llie_180x320.onnx` is the strictly sequential positive artifact
control. Its three ordered results remain `12,0,0`; all twelve first-pass
rewrites are indexed Mul-only folds and no raw fallback rewrite remains. The
pre/post conversion-only runs completed in 0.808 and 0.791 seconds, recorded
process-tree SWAP zero, and produced byte-identical float32, float16, tensor-
correspondence, schema, and generated-schema artifacts. The established
accuracy remains `max_abs=4.470348358154297e-07`; duplicate inference was not
run because the executed TFLite artifact is identical.

Changed files are the new compatibility orchestration owner, lowerer import and
wrapper, indexed/fallback owner-wrapper fixture, architecture ownership gate,
and three branch documents. No public API, CLI, artifact name, pass order,
statistic, TensorFlow boundary, dependency, corpus profile, exclusion policy,
or ONNX operation-tier policy changed. PR #952 remains closed; work is commit/
push only.

At restart, characterize the next raw 487-line
`_optimize_transpose_mean_hardsigmoid_muladd_chains` helper. Fix its Mean axes,
fused and decomposed HardSigmoid roots, affine constant shapes, fan-out/public
guards, metadata permutations, retry/mutation order, all production positions,
and the smallest sequential non-SWAP real owner before changing source. Do not
create a pull request.

## Mean/HardSigmoid/MulAdd characterization: completed state

The raw 487-line
`_optimize_transpose_mean_hardsigmoid_muladd_chains` helper remains unchanged in
`lower_from_onnx2tf.py`. Its one syntactic call belongs to the ordered QLinear/
Mean/Concat recovery sequence, which runs at two production boundaries. This
checkpoint does not extract or otherwise generalize the helper.

The new focused ModelIR graph fixes both NHWC-to-NCHW input adapters, the
Dequantize/keepdims-Mean/Quantize/post-Transpose branch, the decomposed
Mul/Add/Maximum/Minimum HardSigmoid branch, residual Mul/Add wiring, downstream
Mean consumer, legacy residual consumer, three bridge removals, local
NHWC-to-NCHW adapter insertion, metadata propagation, and idempotence. Eight
complete no-op cases cover a wrong q0 permutation, public q0 bridge, q0 fanout,
non-keepdims Mean, wrong post permutation, shared sigmoid output, wrong q1
permutation, and per-axis quantization. The architecture selector records one
raw owner/call plus the current full producer/consumer scans, constant reads and
writes, direct insertion/deletion, and prune boundary.

Characterization exposes two pre-existing unsafe paths as strict xfails:

- invalid Mean axes are normalized only after the q0 Dequantize input and
  output metadata have already changed, so a zero-statistic rejection is not
  atomic;
- a public residual `add0_out` is converted from NCHW metadata and semantics to
  NHWC without retaining a public NCHW adapter or rejecting the candidate.

Validation completed as follows:

- focused characterization alone: `9 passed, 2 xfailed in 0.51s`;
- focused characterization plus raw-owner selector:
  `10 passed, 243 deselected, 2 xfailed in 1.95s`;
- changed-file branch regression collection:
  `565 passed, 2 xfailed in 25.51s`;
- targeted Ruff, Python compilation, and whitespace checks: passed.

Three current short INT8 representatives were traced strictly sequentially.
YuNet INT8, PPHumanSeg INT8, and SSD MobileNet INT8 each reached both ordered
runtime boundaries, returned `0,0`, completed in 1.007, 1.715, and 2.107
seconds, and recorded process-tree SWAP zero. All conversions succeeded. This
confirms the earlier broader QLinear recovery survey; positive production
ownership is not claimed and no broad corpus sweep was repeated.

Changed files are the new focused characterization module, one architecture
selector, and the three branch documents. Production source, public API, CLI,
artifacts, TensorFlow boundary, dependencies, corpus profiles, exclusion
policy, and ONNX operation-tier policy are unchanged. PR #952 remains closed;
work is commit/push only.

At restart, make two separate semantic corrections before considering
extraction. First, validate and plan normalized Mean axes before the first
input or metadata mutation and turn only the atomicity xfail green. Second,
reject or preserve public `add0_out` NCHW semantics and turn only the public-
boundary xfail green. Each commit must retain the positive fingerprint,
statistics, ordered call, sequential zero-SWAP artifact control, and must not
create a pull request.

## Mean/HardSigmoid/MulAdd Mean-axis atomicity: completed state

The first semantic correction is complete without changing the successful
rewrite. After all topology, tensor, and per-tensor quantization guards, the
helper now normalizes Mean axes and commits the axes constant before changing
the q0 Dequantize input or any dependent tensor metadata. An out-of-range axis
therefore rejects with zero statistic and a byte-identical ModelIR fingerprint.
The same atomic contract covers the historical no-change writer rejection when
an axis maps to itself. The architecture selector fixes validation/write-before-
rewiring source order.

The focused graph and raw-owner selector pass `12 passed, 243 deselected, 1
xfailed in 1.93s`; the only remaining strict xfail is the public residual
output boundary. The changed-file branch regression passes `567 passed, 1
xfailed in 24.78s`. The TensorFlow-import-blocked optional-boundary suite passes
`11 passed in 9.29s`. Targeted test Ruff, Python compilation, and whitespace
checks pass. Whole-file Ruff for the central lowerer still reports its eight
pre-existing unused import/local findings; they are unrelated and were not
mixed into this checkpoint.

YuNet INT8 was executed once at the characterization commit and once with the
correction, strictly sequentially through the managed process-tree SWAP
monitor. Both runs passed `-cotof` with `max_abs=0`, zero SWAP, and durations of
5.280 and 5.320 seconds. Internal pass metrics are identical. Float32 and
float16 TFLite, tensor-correspondence, op-error CSV, schema, and generated-
schema artifacts are byte-identical. Accuracy and op-error JSON content differs
only in output-directory and temporary-file paths.

Changed files are the central helper, its focused fixture, the architecture
source-order gate, and the three branch documents. Public API, CLI, artifact
names, successful rewrite order/statistic, TensorFlow boundary, dependencies,
corpus profiles, exclusions, and ONNX operation tiers are unchanged. PR #952
remains closed; work is commit/push only.

At restart, correct only the public `add0_out` boundary. The safest current
contract is an early complete no-op when `add0_out` is a declared ModelIR
output; place that guard before the first axes write or other mutation, turn the
remaining strict xfail green, retain the positive rewrite fingerprint and
statistics, and repeat one strictly sequential zero-SWAP artifact control. Do
not create a pull request.

## Mean/HardSigmoid/MulAdd public-output safety: completed state

The second semantic correction is complete. Once the residual Add output name
is known, the helper rejects a candidate whose `add0_out` is a declared ModelIR
output. The guard runs before Mean-axis normalization/write, graph rewiring, or
dependent metadata changes. The public NCHW boundary therefore remains a
complete zero-statistic no-op instead of silently becoming NHWC. Successful
private-output rewrites, legacy-consumer adapters, statistics, and idempotence
are unchanged.

The focused graph and raw-owner selector pass `13 passed, 243 deselected in
0.61s`; no strict xfail remains. The changed-file branch regression passes `568
passed in 25.10s`. The TensorFlow-import-blocked optional-boundary suite passes
`11 passed in 9.35s`. Targeted test Ruff, Python compilation, and whitespace
checks pass.

YuNet INT8 was again executed strictly sequentially through the process-tree
SWAP monitor at commit `0eced940` and with the public-output correction. Both
runs passed `-cotof` with `max_abs=0`, zero SWAP, and durations of 6.323 and
5.148 seconds. Internal pass metrics are identical. Float32/float16 TFLite,
tensor-correspondence, op-error CSV, schema, and generated-schema artifacts are
byte-identical; the three JSON files differ only in output-directory and
temporary-file paths.

Changed files are the central helper, focused fixture, architecture source-
order gate, and three branch documents. Public API, CLI, artifacts, successful
rewrite order/statistic, TensorFlow boundary, dependencies, corpus profiles,
exclusions, and ONNX operation tiers are unchanged. PR #952 remains closed;
work is commit/push only.

At restart, the 496-line helper has a complete positive/rejection contract and
both known unsafe rejection paths are fixed. If it is moved, perform an exact
mechanical extraction into a focused pass-family module, retain a thin private
lowerer wrapper and its single syntactic call/two runtime boundaries, prove the
old/new body AST after function-name normalization, and repeat the existing
focused, branch, optional-boundary, and one sequential zero-SWAP artifact
control. Do not create a pull request.

## Mean/HardSigmoid/MulAdd ownership extraction: completed state

The corrected 496-line owner is now in
`passes/mean_hardsigmoid_muladd_layout.py`. The lowerer imports it under the
historical private pass alias and retains a two-line private wrapper. Its single
syntactic call inside the QLinear/Mean/Concat recovery sequence and the two
ordered runtime boundaries are unchanged. No dispatch, match, mutation,
statistic, cleanup, or retry logic was generalized.

After normalizing only the function name, the complete old lowerer owner and
new module owner have identical ASTs; both function bodies are 496 lines. The
focused contract now also executes the owner and lowerer wrapper on deep copies
and compares complete ModelIR fingerprints and statistics. Architecture tests
fix the module owner, wrapper dispatch, producer/consumer scans, constant read/
write, public-output and Mean-axis source order, direct insertion/deletion,
prune, and one production call. Whole-lowerer Ruff retains exactly the same
eight pre-existing findings and gained no unused import from the move.

Validation completed as follows:

- normalized old/new owner AST: exact;
- focused owner/wrapper and architecture selector:
  `14 passed, 243 deselected in 0.63s`;
- changed-file branch regression: `569 passed in 23.93s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.29s`;
- targeted Ruff, Python compilation, and whitespace checks: passed.

YuNet INT8 was executed strictly sequentially at commit `7ce7c658` and after
the ownership move. Both runs passed `-cotof` with `max_abs=0`, zero process-
tree SWAP, and durations of 6.290 and 5.215 seconds. Internal pass metrics are
identical. Float32/float16 TFLite, tensor-correspondence, op-error CSV, schema,
and generated-schema artifacts are byte-identical; the three JSON files differ
only in output-directory and temporary-file paths.

Changed files are the new owner module, lowerer import/wrapper, focused owner-
wrapper fixture, architecture ownership gate, and three branch documents.
Public API, CLI, artifacts, successful rewrite order/statistic, TensorFlow
boundary, dependencies, corpus profiles, exclusions, and ONNX operation tiers
are unchanged. PR #952 remains closed; work is commit/push only.

At restart, inventory the next raw source-order helper before editing it. Fix
its match/guard/rewrite contract, mutation order, all production positions, and
the shortest sequential non-SWAP control first. Do not assume positive owner
evidence, do not start a broad corpus sweep, and do not create a pull request.

## QLinear Concat/Conv propagation characterization: completed state

The next raw source-order owner is the 608-line
`_optimize_nhwc_propagation_qlinear_concat_conv` helper. It remains unchanged
in the lowerer, with one syntactic call inside the QLinear/Mean/Concat recovery
sequence and two ordered runtime boundaries. This checkpoint does not extract
or generalize it.

The focused qlinear suite now fixes all four accepted input forms: quantized
NHWC-to-NCHW Transpose before Dequantize, float Transpose before Quantize,
singleton Reshape before Quantize, and singleton-spatial metadata
reinterpretation. It also covers one or multiple post-Quantize adapters, an
additional direct Concat adapter, axis remap, dynamic batch signatures, per-
axis quantization-dimension remap, complete tensor shapes/signatures,
idempotence, and nine atomic no-op guards. The former 119-line giant ModelIR
fixture moved into this focused module with an identical AST, and the giant
test no longer imports the private helper.

Characterization exposes three pre-existing unsafe paths as strict xfails:

- a missing Concat output tensor is detected after DQ/Quantize rewiring,
  metadata updates, and axis mutation have begun;
- a missing Quantize output tensor is detected even later, after the Concat
  output metadata has also changed;
- a public Dequantize output is permitted to change from NCHW to NHWC rather
  than rejecting or preserving its public layout boundary.

Validation completed as follows:

- focused qlinear file: `17 passed, 3 xfailed in 0.53s`;
- focused owner contract plus architecture selector:
  `16 passed, 246 deselected, 3 xfailed in 0.64s`;
- changed-file branch regression including the focused file:
  `587 passed, 3 xfailed in 24.81s`;
- moved giant fixture AST: exact;
- targeted Ruff, Python compilation, and whitespace checks: passed.

Production source is unchanged, so the established QLinear recovery survey was
not repeated. YuNet INT8, PPHumanSeg INT8, LPD-YuNet INT8, Version-RFB INT8,
NanoDet INT8, YOLOX INT8, SSD MobileNet INT8, and dequantize_linear previously
reached both runtime boundaries sequentially with zero rewrites and zero SWAP.
The active Tier 0-4 op-set scan likewise found no positive owner. This remains
zero-owner evidence, not a claim of production match coverage.

Changed files are the focused qlinear suite, the mechanically reduced giant
test, one architecture raw-owner gate, and three branch documents. Public API,
CLI, production behavior, artifacts, TensorFlow boundary, dependencies, corpus
profiles, exclusions, and ONNX operation tiers are unchanged. PR #952 remains
closed; work is commit/push only.

At restart, first prevalidate the required Concat and Quantize output tensors
before any candidate mutation and turn the two parameterized atomicity xfails
green. Then separately reject or preserve a public Dequantize output and turn
the final xfail green. Keep successful fingerprints, statistics, sequence
position, and the existing zero-owner evidence unchanged. Do not create a pull
request.

## QLinear Concat/Conv required-output atomicity: completed state

The first QLinear semantic correction is complete. After all candidate input
forms and prospective Concat shapes are validated, the helper now resolves both
the Concat output tensor and Quantize output tensor before applying any pending
DQ/Quantize input rewrite, metadata update, qdim remap, or axis change. A
missing required output therefore returns zero statistic with a complete
unchanged ModelIR fingerprint. The successful mutation sequence is unchanged;
moving and reusing the two tensor references reduced the raw owner from 608 to
607 lines.

The focused owner contract and architecture selector pass `18 passed, 246
deselected, 1 xfailed in 0.64s`; only the public Dequantize-output boundary
remains xfailed. The changed-file branch regression passes `589 passed, 1
xfailed in 24.76s`. The TensorFlow-import-blocked optional-boundary suite passes
`11 passed in 9.25s`. Targeted test Ruff, Python compilation, and whitespace
checks pass.

YuNet INT8 was run strictly sequentially at commit `526bd6fa` and with the
atomicity correction. Both runs passed `-cotof` with `max_abs=0`, zero process-
tree SWAP, and durations of 6.329 and 5.291 seconds. Internal pass metrics are
identical. Float32/float16 TFLite, tensor-correspondence, op-error CSV, schema,
and generated-schema artifacts are byte-identical; the three JSON files differ
only in output-directory and temporary-file paths.

Changed files are the central raw helper, focused atomicity fixture,
architecture source-order gate, and three branch documents. Public API, CLI,
successful behavior/statistics, artifacts, TensorFlow boundary, dependencies,
corpus profiles, exclusions, and ONNX operation tiers are unchanged. PR #952
remains closed; work is commit/push only.

At restart, reject or preserve only a public Dequantize output before any
pending rewrite is committed. Turn the final strict xfail green, retain all
private-output success fingerprints/statistics and the existing zero-owner
evidence, and repeat one strictly sequential zero-SWAP artifact control. Do not
create a pull request.

## QLinear Concat/Conv public-output safety: completed state

The second QLinear semantic correction is complete. After planning all input
rewrites and metadata updates but before axis validation or mutation, the helper
now rejects any pending tensor-shape update whose tensor is a declared ModelIR
output. Pattern 1 public Dequantize outputs therefore remain complete no-ops
instead of changing from NCHW to NHWC. The guard is intentionally plan-based:
an already physical NHWC public Dequantize output with no pending update still
allows the safe singleton-Reshape optimization to proceed. Successful private-
output behavior and statistics are unchanged. The explicit guard expands the
raw owner from 607 to 612 lines.

The focused owner contract and architecture selector pass `19 passed, 246
deselected in 0.60s`; no strict xfail remains. The changed-file branch
regression passes `590 passed in 24.47s`. The TensorFlow-import-blocked optional-
boundary suite passes `11 passed in 9.27s`. Targeted test Ruff, Python
compilation, and whitespace checks pass.

YuNet INT8 was run strictly sequentially at commit `a4e4bff9` and with the
public-output guard. Both runs passed `-cotof` with `max_abs=0`, zero process-
tree SWAP, and durations of 6.426 and 5.259 seconds. Internal pass metrics are
identical. Float32/float16 TFLite, tensor-correspondence, op-error CSV, schema,
and generated-schema artifacts are byte-identical; the three JSON files differ
only in output-directory and temporary-file paths.

Changed files are the central raw helper, focused public-boundary fixtures,
architecture source-order gate, and three branch documents. Public API, CLI,
successful behavior/statistics, artifacts, TensorFlow boundary, dependencies,
corpus profiles, exclusions, and ONNX operation tiers are unchanged. PR #952
remains closed; work is commit/push only.

At restart, the 612-line helper has a complete four-pattern positive/rejection
contract and no known strict xfail. If moved, perform an exact mechanical
extraction into a focused pass-family module, keep a thin private lowerer
wrapper and the single syntactic call/two runtime boundaries, prove old/new AST
identity after function-name normalization, and repeat focused, branch,
optional-boundary, and one strictly sequential zero-SWAP artifact control. Do
not create a pull request.

## QLinear Concat/Conv ownership extraction: completed state

The corrected 612-line QLinear Concat/Conv propagation owner now resides in
`onnx2tf/tflite_builder/passes/qlinear_concat_conv_compat.py`. The lowerer
imports it under a private pass alias and retains a two-line compatibility
wrapper. Its single syntactic call in the ordered QLinear/Mean/Concat recovery
sequence and both runtime boundaries are unchanged. This move is mechanical;
it does not generalize the semantics or claim positive production ownership.

The old and new owner bodies are both 612 lines and have identical normalized
ASTs after changing only the function name. The focused suite executes the
module owner and lowerer wrapper on deep copies and confirms identical return
statistics and complete ModelIR fingerprints. Architecture tests require the
wrapper to dispatch exactly once, keep the one production call, preserve the
public-output guard before required-output validation and the first mutation,
and prevent the owner module from importing the lowerer. Removing the lowerer's
now-unused `_invert_perm` import leaves its whole-file Ruff result at exactly
the same eight pre-existing findings.

Validation completed as follows:

- normalized old/new owner AST comparison: exact;
- focused owner/wrapper and architecture selector:
  `20 passed, 246 deselected in 0.61s`;
- changed-file branch regression: `591 passed in 23.37s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.27s`;
- targeted Ruff, Python compilation, whitespace, and diff checks: passed.

YuNet INT8 was run strictly sequentially at corrected checkpoint `e2ccb4ac` and
after the extraction. Both runs passed `-cotof` with `max_abs=0`, zero process-
tree SWAP, and durations of 6.389 and 5.279 seconds. Internal pass metrics are
exact. Float32/float16 TFLite, tensor-correspondence, op-error CSV, schema, and
generated-schema artifacts are byte-identical; the three JSON differences
contain output-directory or temporary-file paths only.

Changed files are the new focused owner, the lowerer import/wrapper and unused-
import cleanup, the focused owner/wrapper comparison, the architecture
ownership/source-order checks, and three branch documents. Public API, CLI,
successful behavior/statistics, artifacts, TensorFlow boundary, dependencies,
corpus profiles, exclusions, and ONNX operation tiers are unchanged. PR #952
remains closed; work is commit/push only.

At restart, inventory the next raw helper in actual production source order and
characterize its positive, rejection, ownership, and call-boundary behavior
before editing it. Keep validation minimal and strictly sequential, do not run
a broad corpus sweep, and do not create a pull request.

## Indexed Conv-input adapter repair characterization: completed state

The next raw ownership boundary is the existing Conv-input adapter repair
group in `lower_from_onnx2tf.py`: singleton-Reshape repair, stale NCHW-to-NHWC
Transpose repair, and `_run_indexed_conv_input_adapter_repairs`. The runner
already builds one `ModelIRGraphIndex` for the pair. Its primary and fallback
production calls, the later standalone stale-Transpose compatibility call,
multiple successful rewrites, complete legacy-pair equality, index equality,
fan-out protection, and graph-output protection were already characterized.

The extraction audit found one shared pre-existing atomicity defect. Each raw
repair rewrites the Conv input through `_set_operator_inputs` before reading
the source `shape_signature` needed to update Conv output metadata. If that
signature is present but shorter than rank four, indexing it raises
`IndexError` after the graph edge has changed. The new parameterized strict
xfail requires a zero statistic and complete unchanged ModelIR fingerprint for
both singleton-Reshape and stale-Transpose candidates. It fails at the expected
post-mutation exception in both cases.

Validation completed as follows:

- focused Conv-input adapter selector:
  `3 passed, 257 deselected, 2 xfailed in 0.82s`;
- changed-file focused branch regression, excluding the giant legacy module:
  `594 passed, 2 xfailed in 23.69s`;
- targeted Ruff: passed.

One exploratory broad command accidentally included
`tests/test_tflite_builder_direct.py` and is not the branch gate. It completed
with `1335 passed, 2 xfailed, 6 failed`. Four failures require the intentionally
absent TensorFlow optional extra, one reaches an incompatible user-site Torch
binary, and the standalone giant-test SiNet fixture already fails at checkpoint
`e7ec3a4b` with a zero rewrite. None is caused by this test-only
characterization; no production source changed in this checkpoint.

Changed files are the focused indexed Conv-input repair test and three branch
documents. PR #952 remains closed; work is commit/push only.

At restart, materialize and validate the source shape/signature before the
first indexed Conv-input rewrite in both raw repairs. Turn both strict xfails
green while preserving every successful fingerprint, statistic, shared-index
behavior, and production call boundary. Do not extract the group until that
atomicity correction is separately committed and verified. Do not create a
pull request.

## Conv-input adapter source-signature atomicity: completed state

The shared Conv-input repair defect is fixed without changing successful
rewrite behavior. Both raw repairs now materialize the source shape signature
after all existing tensor, rank, permutation, locality, filter-channel, and
element-count guards but before `_set_operator_inputs`. A signature that is
present with a rank other than four rejects that candidate before any graph,
index, tensor metadata, or topology mutation. A valid signature is the same
value formerly computed after the edge rewrite.

The stale-Transpose atomicity fixture protects its independent second valid
adapter as a declared output, isolating the malformed first candidate while
preserving the function's expected continue-to-later-candidates behavior. Both
former strict xfails now return zero statistics and retain complete ModelIR
fingerprints. The existing successful multi-rewrite comparison still matches
the explicit repair pair exactly and retains one shared index build.

Validation completed as follows:

- focused Conv-input adapter and architecture selector:
  `5 passed, 257 deselected in 0.64s`;
- changed-file focused branch regression: `596 passed in 23.51s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.34s`;
- targeted Ruff and Python compilation: passed.

Changed files are the two raw repairs, the focused atomicity fixture, the
architecture source-order guard, and three branch documents. Public API, CLI,
valid-candidate behavior/statistics, TensorFlow boundary, dependencies, corpus
profiles, exclusions, and ONNX operation tiers are unchanged. No real-model
conversion was added because this correction only rejects malformed metadata
before the former first mutation and the requested validation policy favors
minimal conversions. PR #952 remains closed; work is commit/push only.

At restart, mechanically extract the two corrected repair owners and their
shared-index runner into a focused pass-family module. Keep lowerer private
compatibility wrappers for all three names, preserve the primary and fallback
runner calls plus the later standalone stale-Transpose call, prove normalized
old/new body identity, and compare direct owners with wrappers on complete
ModelIR fingerprints and statistics. Do not create a pull request.

## Conv-input adapter repair ownership extraction: completed state

The corrected singleton-Reshape repair, stale NCHW-to-NHWC Transpose repair,
and shared-index runner now reside in
`onnx2tf/tflite_builder/passes/conv_input_adapter_repair.py`. The lowerer
imports all three under private pass aliases and keeps private compatibility
wrappers with the historical signatures. The primary and fallback runner calls
and the later standalone stale-Transpose compatibility call are unchanged.

The three old/new function bodies are individually AST-identical: 104 lines for
singleton Reshape, 122 lines for stale Transpose, and 23 lines for the runner.
Focused tests execute each owner and wrapper on deep copies and compare return
statistics and complete ModelIR fingerprints. Architecture tests require one
wrapper dispatch per API, module ownership of indexed candidate lookup,
producer/consumer access, differential setter/removal, source-signature
prevalidation, shared-index repair order, two runner production calls, and the
standalone stale-Transpose call. The owner module does not import the lowerer.
The whole lowerer retains exactly its eight pre-existing Ruff findings.

Validation completed as follows:

- old/new AST comparisons: exact for all three owners;
- focused owner/wrapper and architecture selector:
  `8 passed, 257 deselected in 0.63s`;
- changed-file focused branch regression: `599 passed in 23.36s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.35s`;
- targeted Ruff, Python compilation, and whitespace checks: passed.

Tier 1 `face_blendshapes.onnx` is the historical positive model for the
singleton-Reshape repair. It was run strictly sequentially at corrected
checkpoint `a76ad6ff` and after extraction. Both runs passed `-cotof` with
`max_abs=1.3709068298339844e-06`, zero process-tree SWAP, and durations of
4.933 and 3.867 seconds. Pass metrics are exact. Float32/float16 TFLite, tensor-
correspondence, op-error CSV, schema, and generated-schema artifacts are byte-
identical. The three JSON differences contain only output-directory or
temporary-file paths. The other historical positive model, Tier 3
`sgscsh.onnx`, was not rerun because the requested policy favors one minimal,
short positive control.

Changed files are the new owner module, lowerer imports/wrappers, focused owner-
wrapper comparison, architecture ownership checks, and three branch documents.
Public API, CLI, valid-candidate behavior/statistics, TensorFlow boundary,
dependencies, corpus profiles, exclusions, and ONNX operation tiers are
unchanged. PR #952 remains closed; work is commit/push only.

At restart, characterize the next 181-line raw source-order owner,
`_repair_mixed_nhwc_inputs_for_nchw_concat`, before editing it. Fix any unsafe
path in a separate checkpoint before a mechanical extraction. Keep validation
minimal and strictly sequential; do not create a pull request.

## Mixed NHWC-input/NCHW-Concat characterization: completed state

The next 181-line raw owner is
`_repair_mixed_nhwc_inputs_for_nchw_concat` in `lower_from_onnx2tf.py`. It has
two production calls: fallback ModelIR and final ModelIR. The new focused module
fixes the three-input canonical-spatial success path, the two-input output-
contract fallback, inserted adapter shape, output channel reconciliation,
idempotence, and four complete no-op guards. The raw-owner architecture gate
records direct operator insertion, quantization cloning, input replacement,
and both production positions. Existing positive tests in the broader Conv
layout module remain active.

The audit exposes three pre-existing unsafe paths as strict xfails:

- with multiple NHWC candidates, the owner inserts the first adapter before
  reading the later source signature; a short later signature raises
  `IndexError` and leaves a partial graph mutation;
- when at least two NCHW inputs establish the canonical spatial contract, the
  required Concat output tensor is not resolved until after adapters and input
  rewiring, so a missing output tensor still reports one repair;
- a per-axis source quantization on NHWC dimension `3` is cloned onto the NCHW
  adapter without remapping its quantized dimension to `1`.

Validation completed as follows:

- focused mixed-Concat owner and architecture selector:
  `10 passed, 254 deselected, 3 xfailed in 0.77s`;
- changed-file focused branch regression, including the new untracked test in
  discovery: `606 passed, 3 xfailed in 24.92s`;
- targeted Ruff and Python compilation: passed.

Production source is unchanged. Public API, CLI, successful behavior,
TensorFlow boundary, dependencies, corpus profiles, exclusions, and ONNX
operation tiers are unchanged. PR #952 remains closed; work is commit/push
only.

At restart, build a complete immutable adapter plan before mutation: resolve
the required output tensor, materialize every source shape/signature, choose
collision-free tensor names across the whole plan, clone and remap per-axis
quantization, and compute the final output metadata. Then commit insertions and
input/output updates in the existing order. Turn all three strict xfails green
before extracting the owner. Do not create a pull request.

## Mixed NHWC-input/NCHW-Concat transactional repair: completed state

The raw mixed-Concat owner now separates candidate planning from graph commit.
Before mutation it resolves the required Concat output tensor, validates every
input shape and every prospective source signature, reserves collision-free
adapter and permutation names across the whole candidate, computes all adapter
and final output shapes, and clones quantization metadata. Per-axis
quantization remaps an original NHWC channel dimension `3` to NCHW dimension
`1`; per-tensor quantization remains unchanged.

Only a complete plan enters the commit phase. It inserts permutation tensors,
adapter tensors, and Transpose operators in the historical input order, then
rewrites Concat inputs and output metadata. A malformed later signature can no
longer leave an earlier adapter behind, and an unresolved output cannot enter
the plan. All three former strict xfails now pass. Architecture checks require
output/signature resolution and plan append before the first ModelIR tensor,
operator, or input mutation.

Validation completed as follows:

- focused mixed-Concat owner and architecture selector:
  `13 passed, 254 deselected in 0.67s`;
- changed-file focused branch regression: `609 passed in 23.55s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.36s`;
- targeted Ruff and Python compilation: passed.

Tier 3 `sgscsh.onnx` was run strictly sequentially at characterization
checkpoint `ec9f6bf0` and after the correction. Both runs passed `-cotof` with
`max_abs=2.5331974029541016e-07`, zero process-tree SWAP, and durations of
15.216 and 14.375 seconds. Pass metrics are exact. Float32/float16 TFLite,
tensor-correspondence, op-error CSV, schema, and generated-schema artifacts are
byte-identical. The three JSON differences contain only output-directory or
temporary-file paths.

Changed files are the raw repair owner, focused transactional/quantization
fixtures, architecture mutation-order guard, and three branch documents.
Public API, CLI, valid float/per-tensor behavior/statistics, TensorFlow
boundary, dependencies, corpus profiles, exclusions, and ONNX operation tiers
are unchanged. PR #952 remains closed; work is commit/push only.

At restart, mechanically extract the corrected owner into a focused pass-family
module. Keep the lowerer private compatibility wrapper and both fallback/final
production calls, prove normalized old/new body identity, and compare direct
owner/wrapper complete ModelIR fingerprints and statistics. Do not create a
pull request.

## Mixed NHWC-input/NCHW-Concat ownership extraction: completed state

The corrected mixed-Concat owner now resides in
`onnx2tf/tflite_builder/passes/mixed_concat_input_repair.py`. The lowerer
imports it under a private pass alias and retains a two-line private
compatibility wrapper. Both fallback and final production calls still target
the historical lowerer name in the same order.

The old and new owner functions are each 223 lines and have identical ASTs.
The focused suite executes the module owner and lowerer wrapper on deep copies
of a multi-adapter ModelIR and confirms identical return statistics and complete
fingerprints. Architecture tests keep the corrected prevalidation plan before
the first mutation, require one wrapper dispatch, preserve both production
calls, and prevent the owner module from importing the lowerer. The whole
lowerer retains exactly its eight pre-existing Ruff findings.

Validation completed as follows:

- old/new owner AST comparison: exact, 223 lines each;
- focused owner/wrapper and architecture selector:
  `14 passed, 254 deselected in 0.69s`;
- changed-file focused branch regression: `610 passed in 23.28s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.30s`;
- targeted Ruff, Python compilation, and whitespace checks: passed.

Tier 3 `sgscsh.onnx` was run strictly sequentially at corrected checkpoint
`55f1a541` and after extraction. Both runs passed `-cotof` with
`max_abs=2.5331974029541016e-07`, zero process-tree SWAP, and durations of
15.582 and 14.434 seconds. Pass metrics are exact. Float32/float16 TFLite,
tensor-correspondence, op-error CSV, schema, and generated-schema artifacts are
byte-identical; the three JSON differences contain only output-directory or
temporary-file paths.

Changed files are the new owner module, lowerer import/wrapper, focused owner-
wrapper comparison, architecture ownership checks, and three branch documents.
Public API, CLI, behavior/statistics, TensorFlow boundary, dependencies, corpus
profiles, exclusions, and ONNX operation tiers are unchanged. PR #952 remains
closed; work is commit/push only.

At restart, inventory the next raw helper in actual production source order
before editing it. Characterize positive, rejection, ownership, and call-
boundary behavior first. Keep validation minimal and strictly sequential; do
not create a pull request.

## Stale channelwise-binary Transpose characterization: completed state

The next raw source-order owner is the 129-line
`_repair_stale_nchw_to_nhwc_channelwise_binary_transposes`. It already accepts
an optional matching `ModelIRGraphIndex`, enumerates exact indexed binary
candidates, handles both data-input positions and channelwise-constant/Conv-
peer evidence, removes multiple adapters differentially, preserves fan-out and
public adapters, and participates in the lowerer's three-round shared-index
binary convergence runner. There are also two standalone production calls.

The extraction audit adds two invalid-metadata cases. A short source shape is
already a complete zero-statistic no-op. A rank-four source with a short
`shape_signature` exposes one pre-existing unsafe path: the binary input is
rewritten first, the short signature is then assigned to the output, and the
adapter is removed. The strict xfail requires a zero statistic and complete
unchanged ModelIR fingerprint.

Validation completed as follows:

- focused stale-binary/convergence selector:
  `5 passed, 257 deselected, 1 xfailed in 0.65s`;
- changed-file focused branch regression:
  `614 passed, 1 xfailed in 23.43s`;
- targeted Ruff: passed.

Production source is unchanged. Public API, CLI, successful behavior,
TensorFlow boundary, dependencies, corpus profiles, exclusions, and ONNX
operation tiers are unchanged. PR #952 remains closed; work is commit/push
only.

At restart, materialize and require a rank-four source signature before
`_set_operator_inputs`, turn the strict xfail green, and preserve all indexed
candidate order, statistics, differential index updates, convergence-runner
behavior, and direct production calls. Extract the repair owner only after the
correction checkpoint; keep the convergence runner central because it also
coordinates broadcast repair and shape reconciliation. Do not create a pull
request.

## Stale binary source-signature atomicity: completed state

The stale channelwise-binary Transpose repair now materializes the source
signature after all existing tensor, shape, permutation, locality, and peer-
evidence guards but before `_set_operator_inputs`. A present signature whose
rank is not four rejects the candidate before graph, index, output metadata, or
topology mutation. A valid signature is the same value formerly computed after
the edge rewrite. Both short-shape and short-signature cases are ordinary
complete no-ops; no strict xfail remains.

Validation completed as follows:

- focused stale-binary/convergence selector:
  `6 passed, 257 deselected in 2.07s`;
- changed-file focused branch regression: `615 passed in 23.40s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.44s`;
- targeted Ruff and Python compilation: passed.

No additional real-model conversion was run. The correction only moves valid
signature materialization before the former first mutation, and the immediately
preceding sequential `sgscsh.onnx` controls already proved exact pass metrics,
zero SWAP, and byte-identical major artifacts across both semantic correction
and ownership checkpoints. This follows the requested minimal-conversion
policy.

Changed files are the raw repair, focused invalid-metadata fixture,
architecture source-order guard, and three branch documents. Public API, CLI,
valid behavior/statistics, TensorFlow boundary, dependencies, corpus profiles,
exclusions, and ONNX operation tiers are unchanged. PR #952 remains closed;
work is commit/push only.

At restart, mechanically extract only the corrected 132-line repair owner into
a focused pass-family module. Keep its lowerer private wrapper, the two
standalone calls, and the three-round convergence runner call. Leave the runner
in the lowerer because it coordinates separate broadcast and shape owners.
Prove exact old/new AST and direct owner/wrapper fingerprint equality. Do not
create a pull request.

## Channelwise-constant stale-binary rank characterization: completed state

The final pre-extraction audit distinguishes the two evidence branches in the
stale channelwise-binary Transpose repair. Conv-peer matching short-circuits
safely on a short source shape. Channelwise-constant matching first accepts the
`[1,1,1,C]` constant prefix, then reads `source_shape[3]` and
`adapter_shape[3]` before the common rank guard. A short source or adapter shape
therefore raises `IndexError` before the helper can return its expected no-op.

Two new parameterized strict xfails isolate those source and adapter cases with
the independent second adapter protected as a graph output. Each requires a
zero statistic and complete unchanged ModelIR fingerprint. The earlier source-
signature correction remains green.

Validation completed as follows:

- focused stale-binary/convergence selector:
  `6 passed, 257 deselected, 2 xfailed in 0.80s`;
- changed-file focused branch regression:
  `615 passed, 2 xfailed in 23.62s`;
- targeted Ruff: passed.

Production source is unchanged from checkpoint `b84b9d13`. The uncommitted
ownership move was deliberately rolled back before this characterization, so
the branch remains at a complete raw-owner checkpoint. Public API, CLI,
successful behavior, TensorFlow boundary, dependencies, corpus profiles,
exclusions, and ONNX operation tiers are unchanged. PR #952 remains closed;
work is commit/push only.

At restart, move `len(source_shape) == 4` and `len(adapter_shape) == 4` guards
before channelwise-constant and Conv-peer evidence evaluation. Turn both strict
xfails green while preserving candidate order, statistics, and GraphIndex
updates. Only then redo the mechanical repair-owner extraction; keep the
convergence runner central. Do not create a pull request.

## Channelwise-constant stale-binary rank safety: completed state

The stale binary repair now rejects a source or adapter whose shape is not rank
four before evaluating either channelwise-constant or Conv-peer evidence. The
existing evidence expressions, candidate order, input-position order, indexed
setter/removal sequence, output metadata update, and statistics are unchanged.
Both former strict xfails are complete zero-statistic ModelIR no-ops. The
earlier malformed-signature guard remains before the first mutation.

Validation completed as follows:

- focused stale-binary/convergence selector:
  `8 passed, 257 deselected in 2.07s`;
- changed-file focused branch regression: `617 passed in 23.23s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.30s`;
- targeted Ruff and Python compilation: passed.

No real-model conversion was added because the change only prevents invalid-
rank metadata from reaching channel evidence and the requested validation
policy favors minimal conversions. Production success behavior is covered by
the existing multiple-match, both-input-position, channelwise-constant, Conv-
peer, fan-out, index-equivalence, and convergence-runner fixtures.

Changed files are the raw repair, focused rank fixtures, architecture guard-
order check, and three branch documents. Public API, CLI, valid behavior/
statistics, TensorFlow boundary, dependencies, corpus profiles, exclusions,
and ONNX operation tiers are unchanged. PR #952 remains closed; work is
commit/push only.

At restart, mechanically extract only the corrected repair owner into a focused
pass-family module. Keep the lowerer private wrapper, both standalone calls,
and the three-round convergence runner call. Leave the runner central, prove
exact old/new AST and direct owner/wrapper fingerprint equality, and do not
create a pull request.

## Stale channelwise-binary adapter ownership extraction: completed state

The corrected stale NCHW-to-NHWC channelwise-binary Transpose repair now
resides in
`onnx2tf/tflite_builder/passes/stale_binary_adapter_repair.py`. The 132-line
module owner is AST-identical to the corrected lowerer implementation at
checkpoint `c869c410`. The lowerer retains the historical private function as
a thin compatibility wrapper and forwards its optional `graph_index` to the
module owner.

The two standalone fallback/final production calls still target the private
lowerer name in their original order. The three-round
`_run_indexed_binary_layout_convergence` coordinator also still calls that
wrapper with one shared index and intentionally remains in the lowerer because
it coordinates three separate owners: broadcast-constant repair, stale
Transpose repair, and static shape reconciliation. A focused deep-copy test
confirms identical statistics and complete ModelIR fingerprints from direct
owner and wrapper execution. Architecture checks keep the module independent
of the lowerer, preserve pre-mutation rank/signature guards, require wrapper
index forwarding, and freeze the runner and standalone call boundaries.

Validation completed as follows:

- old/new owner AST comparison: exact, 132 lines each;
- focused stale-binary owner/wrapper and architecture selector:
  `9 passed, 257 deselected in 2.11s`;
- changed-file focused branch regression: `618 passed in 24.62s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.30s`;
- targeted Ruff, Python compilation, and whitespace checks: passed;
- the whole lowerer retains exactly its eight pre-existing Ruff findings.

No additional real-model conversion was run. This checkpoint is an exact
mechanical ownership move of the already corrected function, with direct
owner/wrapper fingerprint coverage, and follows the requested minimal-
conversion policy. Public API, CLI, successful behavior/statistics,
TensorFlow boundary, dependencies, corpus profiles, exclusions, and ONNX
operation tiers are unchanged. PR #952 remains closed; future work is
commit/push only and must not create a pull request.

At restart, inventory and characterize the next raw source-order owner before
editing it: `_optimize_nhwc_prefix_qlinear_silu_chains` (419 lines). Preserve
its positive/rejection behavior, call boundaries, statistics, and artifact
contract before considering ownership or semantic changes. Keep validation
minimal and strictly sequential, and do not create a pull request.

## QLinear SiLU prefix characterization: completed state

The raw 419-line `_optimize_nhwc_prefix_qlinear_silu_chains` owner remains
unchanged in `lower_from_onnx2tf.py`. New synthetic fixtures freeze both
supported activation forms: direct DEQUANTIZE/LOGISTIC/QUANTIZE and decomposed
DEQUANTIZE/MUL/ADD/MAXIMUM/MINIMUM/QUANTIZE. They verify exact statistics,
NHWC input rewiring, redundant pre/post Transpose removal, rank-four metadata
permutation, and downstream alias rewiring. A combined two-chain fixture
freezes fixed-point restart and order, while a legacy RELU consumer freezes
the inserted NHWC-to-NCHW adapter, tensor shapes, and permutation payload.

Eight ordinary rejection cases cover wrong pre-permutation, public and fan-out
pre-adapters, per-axis quantization, shared sigmoid-quantize output, a blocked
layout-sensitive consumer, public post-adapter output, and a non-singleton
HardSigmoid constant. The raw owner returns zero and preserves operator and
shape metadata for each. Architecture coverage records its 419-line central
ownership, consumer-map scan, mutation APIs, fixed-point loop, and existing
ordered QLinear recovery boundary.

Four strict xfails isolate pre-existing defects before any source correction:

- rejected calls eagerly create and prune
  `__nhwc_to_nchw_perm_rank4__`, leaving a tensor-lineage metadata event rather
  than a complete ModelIR no-op;
- a second zero-rewrite call after success repeats that metadata mutation, so
  the helper is not fully idempotent;
- an unrelated public tensor occupying the reserved name is reused as the
  adapter permutation without validating `[0,3,1,2]`;
- a rank-two `mul_out` signature is accepted, after which the helper rewires
  the graph and creates a rank-four Transpose adapter carrying malformed
  signature metadata.

Validation completed as follows:

- focused QLinear SiLU prefix plus ordered-owner architecture selector:
  `14 passed, 245 deselected, 4 xfailed in 0.68s`;
- changed-file focused branch regression:
  `631 passed, 4 xfailed in 23.56s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.35s`;
- targeted Ruff, Python compilation, and whitespace checks: passed.

No model conversion was repeated. The earlier QLinear recovery-group audit
already ran the candidate set strictly sequentially, observed zero rewrites in
every instrumented invocation, and recorded zero process-tree SWAP. Production
source, public API, CLI, artifacts, TensorFlow boundary, dependencies, corpus
profiles, exclusions, and ONNX operation tiers are unchanged. PR #952 remains
closed; future work is commit/push only and must not create a pull request.

At restart, fix these four xfails before extracting the owner. Prevalidate the
Mul output shape and effective signature as rank four and build the complete
legacy-adapter plan before changing DEQUANTIZE/MUL/user inputs or metadata.
Create the permutation constant only if a committed plan needs an adapter.
Reuse the reserved name only when dtype, shape, signature, and payload are
exact; otherwise allocate a collision-safe name without mutating the existing
tensor. Preserve valid candidate order, fixed-point count, statistics, both
production sequence boundaries, and all ordinary characterization. Keep tests
and any necessary model validation strictly sequential, then commit and push;
do not create a pull request.

## QLinear SiLU prefix transactional correction: completed state

The four strict xfails from the preceding characterization are now green. The
509-line raw lowerer owner prevalidates every metadata target shape and
effective signature as rank four before changing any tensor or operator. It
builds adapter tensors, Transpose operators, collision-free names, and
cumulative legacy-consumer input updates as an immutable local plan before
commit. A malformed DEQUANTIZE, activation, Quantize, or final Mul signature
therefore returns zero with an unchanged graph and metadata.

`__nhwc_to_nchw_perm_rank4__` is no longer created at function entry. A
candidate with no legacy consumer never allocates it. A legacy-adapter plan
reuses an existing tensor only when its INT32 dtype, `[4]` shape/signature,
exact `[0,3,1,2]` payload, non-variable state, and absent quantization all
match. Any other occupied name remains unchanged and a collision-safe name is
planned. Tensor pruning now runs only after at least one successful rewrite,
so rejected and second zero-rewrite calls preserve tensor-lineage metadata and
are fully idempotent.

An additional same-consumer/two-input-slot fixture exposed a separate existing
issue while validating cumulative plans. `_build_tensor_consumer_map` returns
the same operator index once per matching input. The raw owner then enumerates
both matching slots for each repeated index, plans four adapters, and leaves
two redundant adapters. This is recorded as the sole strict xfail and was not
fixed in the transactional checkpoint, satisfying the record-before-fix
boundary.

Validation completed as follows:

- focused QLinear SiLU correction plus ordered-owner architecture selector:
  `22 passed, 245 deselected, 1 xfailed in 2.04s`;
- changed-file focused branch regression:
  `639 passed, 1 xfailed in 23.83s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.25s`;
- targeted Ruff, Python compilation, and whitespace checks: passed;
- the whole lowerer retains exactly its eight pre-existing Ruff findings.

No real-model conversion was added because the prior sequential QLinear group
audit already established zero production rewrites and zero process-tree SWAP
for every measured candidate. Public API, CLI, production call order and
statistics, TensorFlow boundary, dependencies, corpus profiles, exclusions,
and ONNX operation tiers are unchanged. PR #952 remains closed; future work is
commit/push only and must not create a pull request.

At restart, deduplicate `mul_users` by operator index in first-observed order
before classifying Transpose and legacy consumers. Turn the two-slot strict
xfail green and prove that distinct consumer order, one adapter per input slot,
unique naming, exact permutation reuse/collision behavior, fixed-point count,
and all ordinary rejections remain unchanged. Only after that correction
should the 509-line owner be mechanically extracted into a focused pass module
with an exact corrected AST and a lowerer compatibility wrapper. Keep tests
strictly sequential, commit and push coherent units, and do not create a pull
request.

## QLinear SiLU legacy-consumer deduplication: completed state

The sole remaining strict xfail is green. The raw owner now converts the final
Mul consumer edge list into first-observed unique operator indices before
classifying Transpose and legacy consumers. It does not reorder distinct
operators. Each unique legacy operator is visited once and its matching input
slots are enumerated once, so a two-slot ADD receives two adapters instead of
four. A second fixture with two independent RELU consumers proves stable
consumer order, deterministic `_adapter`/`_adapter_1` naming, and one adapter
per edge. Existing single-slot, exact-permutation reuse, collision-safe
allocation, fixed-point, rejection, idempotence, and malformed-metadata tests
remain green.

Validation completed as follows:

- focused QLinear SiLU plus ordered-owner architecture selector:
  `24 passed, 245 deselected in 0.66s`;
- changed-file focused branch regression: `641 passed in 23.41s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.38s`;
- targeted Ruff, Python compilation, and whitespace checks: passed;
- the whole lowerer retains exactly its eight pre-existing Ruff findings.

No additional model conversion was run because the earlier sequential QLinear
group audit established zero production rewrites and zero process-tree SWAP
for every measured candidate. Public API, CLI, valid statistics, ordered
production boundaries, TensorFlow isolation, dependencies, corpus profiles,
exclusions, and ONNX operation tiers remain unchanged. PR #952 remains closed;
future work is commit/push only and must not create a pull request.

At restart, mechanically extract the corrected 513-line
`_optimize_nhwc_prefix_qlinear_silu_chains` owner into a focused QLinear SiLU
pass module. Keep the historical lowerer private function as a thin wrapper and
preserve its position inside both `_run_qlinear_mean_concat_recovery_sequence`
invocations. Prove exact corrected old/new AST identity and direct
owner/wrapper statistics plus complete ModelIR equality across LOGISTIC,
decomposed HardSigmoid, legacy-adapter, multiple-match, and collision cases.
Keep validation minimal and strictly sequential, then commit and push; do not
create a pull request.

## QLinear SiLU prefix ownership extraction: completed state

The corrected owner now resides in
`onnx2tf/tflite_builder/passes/qlinear_silu_prefix_layout.py`. Its function and
the corrected predecessor at checkpoint `0cf699fd` are each 513 lines and have
identical ASTs. The lowerer imports the module owner under a private pass alias
and retains `_optimize_nhwc_prefix_qlinear_silu_chains` as a one-return
compatibility wrapper. `_run_qlinear_mean_concat_recovery_sequence` still
calls that name in its original position, and both production sequence
boundaries are unchanged.

The focused suite executes direct module owner and lowerer wrapper on deep
copies of four contracts: LOGISTIC, decomposed HardSigmoid, legacy consumer
adapter insertion, and invalid reserved-name collision. Statistics, complete
ModelIR fingerprints/layout state, and metadata are identical. Architecture
tests parse transactional planning, conditional prune, and ordered consumer
deduplication from the module owner, prevent a lowerer import cycle, require a
single wrapper dispatch, and preserve the existing ordered recovery sequence.
Extraction made `_is_singleton_constant_tensor` unused in the lowerer, so that
import alone was removed; the lowerer retains exactly its eight earlier Ruff
findings.

Validation completed as follows:

- corrected old/new owner AST comparison: exact, 513 lines each;
- focused owner/wrapper and architecture selector:
  `28 passed, 245 deselected in 1.96s`;
- changed-file focused branch regression: `645 passed in 23.53s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.40s`;
- targeted Ruff, Python compilation, and whitespace checks: passed.

No real-model conversion was repeated. The owner move is mechanically exact,
direct owner/wrapper equality covers all non-zero synthetic families, and the
prior sequential QLinear group audit established zero production rewrites and
zero process-tree SWAP for every measured candidate. Public API, CLI, valid
behavior/statistics, ordered production boundaries, TensorFlow isolation,
dependencies, corpus profiles, exclusions, and ONNX operation tiers are
unchanged. PR #952 remains closed; future work is commit/push only and must not
create a pull request.

At restart, inventory and characterize the next raw source-order owner before
editing it: `_optimize_transpose_mean_maxpool_concat_conv_chains` (310 lines).
Freeze its positive, rejection, fixed-point, metadata, quantization, pruning,
statistics, and ordered production boundaries before deciding on correctness
changes or mechanical ownership. Reuse the earlier zero-owner model evidence,
keep any additional validation minimal and strictly sequential, then commit
and push; do not create a pull request.

## Mean/MaxPool/Concat/Conv characterization: completed state

The raw 310-line
`_optimize_transpose_mean_maxpool_concat_conv_chains` owner remains unchanged
in `lower_from_onnx2tf.py`. Its new synthetic positive contract verifies:

- direct NHWC input rewiring for the Mean DEQUANTIZE branch;
- Mean axes `[2,3]` to `[1,2]` and keepDims metadata;
- removal of the pool NCHW adapter and every post-Quantize NHWC adapter;
- Concat input replacement and axis `1` to `3`;
- static and dynamic batch shape/signature propagation;
- per-axis QDIM `1` to `3`;
- deterministic multiple-post and multiple-chain fixed-point behavior;
- exact statistics, pruning, and second-call idempotence.

Ten ordinary rejection cases cover wrong pre/pool permutation, public/fan-out
pre-adapter, keepDims and Mean axes, Concat axis, incompatible per-axis QDIM,
public post output, and non-Transpose post consumers. They return zero and
preserve the complete ModelIR. Architecture coverage records the central
310-line owner, consumer/producer map rebuilds, mutation utilities, unbounded
fixed-point loop, and existing ordered recovery boundary.

Nine strict xfails isolate pre-existing problems before any source correction:

- short `q_raw_nhwc` and `pool_nhwc` signatures are consumed after input,
  axes, and metadata mutation;
- a missing, rank-three, or short-signature additional Concat input is checked
  only after both branches and the Concat have been rewritten;
- the Mean axes constant is modified when shared with another operator, listed
  as a graph output, listed as a graph input, or marked variable.

Each strict xfail requires a zero statistic and unchanged graph/tensor/options/
constant/metadata state. Current behavior raises or leaves partial/nonlocal
mutation, proving the correction boundary rather than merely documenting a
theoretical risk.

Validation completed as follows:

- focused raw-owner plus ordered-boundary architecture selector:
  `17 passed, 246 deselected, 9 xfailed in 0.93s`;
- changed-file focused branch regression:
  `661 passed, 9 xfailed in 23.74s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.30s`;
- targeted Ruff, Python compilation, and whitespace checks: passed.

No model conversion was repeated. The earlier sequential QLinear recovery
audit already established zero rewrites and zero process-tree SWAP for every
measured candidate. Production source, public API, CLI, behavior, ordered
boundaries, TensorFlow isolation, dependencies, corpus profiles, exclusions,
and ONNX operation tiers are unchanged. PR #952 remains closed; future work is
commit/push only and must not create a pull request.

At restart, require `mean_axes` to be an immutable local INT32 constant: not a
graph input/output, not variable, and consumed only by the matched Mean. Before
mutation, resolve and rank-validate the source signature, every planned Concat
input after pool substitution, every target tensor/signature, the new Mean and
Concat shapes/signatures, the Concat axis, and optional per-axis QDIM update.
Only a complete plan may commit setters, constant data, options, metadata,
alias rewiring, and removals. Turn all nine strict xfails green while preserving
valid candidate order/statistics and production boundaries. Do not extract the
owner until the correction is committed; keep tests strictly sequential,
commit and push, and do not create a pull request.

## Mean/MaxPool/Concat/Conv transactional correction: completed state

All nine strict xfails from the preceding characterization are green. The raw
owner now requires `mean_axes` to satisfy every local-ownership invariant:

- TensorIR dtype and backing NumPy dtype are both INT32;
- the tensor is non-variable and unquantized;
- it is absent from graph inputs and graph outputs;
- its exact consumer list contains only the matched Mean.

Before the first ModelIR mutation, the owner resolves rank-four shape and
effective signature for the NHWC source, removed NCHW intermediates, both
DEQUANTIZE branches, Mean and pool outputs, every pool-substituted Concat input,
Concat/Quantize outputs, and every removable post-adapter output. It computes
the new Mean axes/shape/signature, complete Concat shape/signature, optional
per-axis QDIM remap, post aliases, and removal indices as one local plan. The
commit phase has no rejection branch after its first setter. Missing/rank-three/
short-signature Concat inputs and short source/pool signatures now return zero
without graph, constant, option, tensor, or metadata mutation.

Pruning is conditional on a non-zero rewrite. The owner no longer constructs
the producer map that was never read, removing one whole-graph scan from every
fixed-point round. Three additional ordinary guards verify axes TensorIR dtype,
buffer dtype, and quantization state. Valid static/dynamic, multiple-post,
multiple-chain, per-axis quantization, rejection, and idempotence contracts all
remain green.

Validation completed as follows:

- focused corrected-owner plus ordered-boundary architecture selector:
  `29 passed, 246 deselected in 1.97s`;
- changed-file focused branch regression: `673 passed in 22.86s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 10.27s`;
- targeted Ruff, Python compilation, and whitespace checks: passed;
- central lowerer Ruff findings decreased from eight to seven after removal of
  the unused producer-map assignment.

No real-model conversion was repeated because the prior sequential QLinear
recovery audit established zero production rewrites and zero process-tree SWAP
for every measured candidate. Public API, CLI, valid behavior/statistics,
ordered boundaries, TensorFlow isolation, dependencies, corpus profiles,
exclusions, and ONNX operation tiers are unchanged. PR #952 remains closed;
future work is commit/push only and must not create a pull request.

At restart, mechanically extract the corrected 382-line
`_optimize_transpose_mean_maxpool_concat_conv_chains` owner into a focused
Mean/MaxPool/Concat pass module. Keep the historical private lowerer name as a
one-return wrapper and preserve its position in
`_run_qlinear_mean_concat_recovery_sequence` plus both production sequence
boundaries. Prove exact corrected old/new AST identity and direct owner/wrapper
statistics plus complete ModelIR/metadata equality for static, dynamic,
multiple-post, multiple-chain, per-axis, and rejection cases. Keep validation
minimal and strictly sequential, commit and push, and do not create a pull
request.

## Mean/MaxPool/Concat/Conv ownership extraction: completed state

The corrected owner now resides in
`onnx2tf/tflite_builder/passes/mean_maxpool_concat_layout.py`. Its function and
the corrected predecessor at checkpoint `7b0f08a9` are each 382 lines and have
identical ASTs. The lowerer imports the module owner under a private pass alias
and retains `_optimize_transpose_mean_maxpool_concat_conv_chains` as a
one-return compatibility wrapper. Its position in
`_run_qlinear_mean_concat_recovery_sequence` and both production sequence
boundaries are unchanged.

Direct owner/wrapper characterization covers static, dynamic batch,
multiple-post, multiple-chain, and rejection graphs. Deep-copied executions
produce identical statistics and complete normalized ModelIR state, including
constant payloads, options, per-axis quantization, tensor metadata, topology,
and diagnostics. Architecture tests parse axes ownership, rank-four planning,
commit ordering, and conditional pruning from the module owner, prevent a
lowerer import cycle, require one wrapper dispatch, and preserve the ordered
recovery sequence. Extraction made `_quant_scale_count` unused in the lowerer,
so that import was removed; seven prior Ruff findings remain.

Validation completed as follows:

- corrected old/new owner AST comparison: exact, 382 lines each;
- focused owner/wrapper and architecture selector:
  `34 passed, 246 deselected in 2.00s`;
- changed-file focused branch regression: `678 passed in 22.62s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.31s`;
- targeted Ruff, Python compilation, and whitespace checks: passed.

No model conversion was repeated. The ownership move is mechanically exact,
direct tests cover every synthetic non-zero family, and the previous sequential
QLinear recovery audit established zero production rewrites and zero process-
tree SWAP for every measured candidate. Public API, CLI, valid behavior/
statistics, ordered boundaries, TensorFlow isolation, dependencies, corpus
profiles, exclusions, and ONNX operation tiers are unchanged. PR #952 remains
closed; future work is commit/push only and must not create a pull request.

At restart, inventory and characterize the next raw source-order owner before
editing it: `_canonicalize_softmax_transpose_chains` (190 lines). Freeze its
positive and rejection topology, axis option changes, aliasing, fixed-point/
pruning behavior, statistics, and ordered production boundaries before any
correction or extraction. Reuse existing model evidence, keep additional
validation minimal and strictly sequential, commit and push, and do not create
a pull request.

## Softmax/Transpose canonicalizer characterization: completed state

The raw 190-line `_canonicalize_softmax_transpose_chains` owner and its two
ordered production boundaries are unchanged. A focused synthetic contract now
freezes the successful nonterminal and terminal chain, independent graph-order
matches, fixed-point idempotence, shared-permutation cloning and collision-safe
names, Softmax option/axis/provenance preservation, historical unmatched-graph
pruning, ten existing topology/arity/fan-out/public-output/per-axis rejection
guards, and both nested recovery-sequence boundaries.

Twenty-four concrete unsafe cases are strict xfails:

- the Softmax input is replanned as NHWC but the Softmax output remains NWHC,
  causing the post-Transpose shape and signature to swap H/W;
- six non-last, out-of-range, or malformed Softmax axes are rewritten even
  though only normalized rank-four axis three is semantically invariant;
- seven missing or incomplete rank-four tensor/signature cases still mutate;
- five public-input, variable, dtype-invalid, buffer-dtype-invalid, or
  quantized permutation constants are modified in place;
- one public permutation output is modified instead of preserved through a
  private constant clone;
- duplicate Softmax/post producers, reverse Softmax/post order, and a public
  internal input are not rejected.

Each unsafe case requires a zero statistic and byte-for-byte normalized
ModelIR equality. Production code, public API, CLI, artifacts, dependencies,
corpus profiles, exclusions, operation tiers, and TensorFlow boundaries are
unchanged. No additional model conversion was run; this checkpoint is
synthetic characterization only. PR #952 remains closed, and no pull request
may be created or reopened.

Validation completed as follows:

- focused characterization: `16 passed, 24 xfailed in 0.72s`;
- characterization, terminal Softmax owner, and architecture integration:
  `293 passed, 24 xfailed in 20.76s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.32s`;
- targeted Ruff, Python compilation, and whitespace checks: passed.

At restart, correct the raw owner transactionally before extracting it.
Require a normalized last-axis Softmax, unique topologically ordered producers,
private intermediate tensors, complete rank-four shape/signature metadata,
and immutable local INT32 permutation buffers. Public constant outputs may be
preserved through private clones, but public inputs and variable permutations
must reject. Precompute both permutation actions, every metadata value, marker
options, clone names, and pruning consequences before the first mutation. Turn
all 24 strict xfails green while preserving the successful graph-order count,
fixed point, shared-buffer cloning, terminal-output behavior, marker contract,
and both ordered boundaries. Validate sequentially, commit and push, and do not
create a pull request.

## Softmax/Transpose transactional correction: completed state

All 24 strict xfails are green. The corrected raw owner is 343 lines and builds
one `ModelIRGraphIndex` instead of rebuilding complete producer and consumer
maps during each fixed-point round. It requires unique producers and strict
`pre-previous < pre < Softmax < post` order, rejects produced tensors also
declared as public inputs, and preserves the former private fan-out and
terminal-output guards. Softmax options must describe normalized rank-four
last axis (`3` or `-1`); a missing axis retains TFLite last-axis semantics.

All four required activation tensors now supply complete rank-four shape and
effective signature metadata before mutation. The new NHWC shape/signature is
applied to both Softmax input and output, and the final NCHW metadata is derived
from that planned Softmax output. This fixes the former H/W swap without
depending on a later reconcile pass. Per-axis activation quantization remains
conservatively ineligible.

Both permutation changes are immutable plans. Each source must be a
non-variable, unquantized INT32 TensorIR with an INT32 NumPy buffer. Public
inputs reject because their runtime value is overridable. Shared and public-
output constants receive deterministic private clones, while a uniquely owned
local constant is updated in place. Clone names are reserved in a candidate-
local set and become globally reserved only after both permutation plans,
metadata, and marker options succeed. A dedicated post-permutation failure
case proves that the pre-permutation, lineage, tensors, options, and metadata
remain unchanged.

Validation completed as follows:

- focused corrected owner: `42 passed in 0.55s`;
- corrected owner plus all focused Softmax/layout and architecture suites:
  `443 passed in 20.90s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 10.06s`;
- targeted Python compilation and whitespace checks: passed;
- targeted Ruff reports the same seven pre-existing lowerer findings and no
  new test finding.

No real-model conversion was added. The correction rejects unsafe
optimizations while retaining the original graph when evidence is incomplete;
all known valid synthetic families and the broader Softmax pass set are green.
Public API, CLI, artifacts, dependencies, corpus profiles, exclusions,
operation tiers, and TensorFlow boundaries are unchanged. PR #952 remains
closed; no pull request was created or reopened.

At restart, mechanically extract the corrected 343-line
`_canonicalize_softmax_transpose_chains` owner to a focused pass module. Retain
the historical lowerer private name as a one-return wrapper, import the shared
terminal marker from its existing owner, and preserve both nested recovery-
sequence positions. Prove corrected old/new AST identity and direct owner/
wrapper equality for static/dynamic metadata, multiple branches, shared and
public-output clones, `axis=-1`, terminal output, pruning, rejection, and
atomicity cases. Validate sequentially, commit and push, and do not create a
pull request.

## Softmax/Transpose ownership extraction: completed state

The corrected owner now resides in
`onnx2tf/tflite_builder/passes/softmax_transpose_canonicalization.py`. Its
function and the corrected raw owner at checkpoint `9a9898e3` are each 343
lines and have identical ASTs. The central lowerer imports it under the private
`_canonicalize_softmax_transpose_chains_pass` alias and retains the historical
private name as a one-return wrapper. The two positions inside
`_run_quantized_activation_binary_bridge_recovery_sequence` and
`_run_layout_attention_quantized_recovery_suffix` are unchanged. The module
imports the shared NHWC-propagation marker from `terminal_softmax_layout`
without importing the lowerer.

Ten direct owner/wrapper comparisons cover static shapes with dynamic batch
signatures, two independent branches, shared permutation clones, a public-
output clone, normalized axis `-1`, a terminal output, historical zero-match
pruning, unsafe axis rejection, incomplete metadata rejection, and post-plan
atomic rejection. Deep-copied executions produce identical statistics and
complete normalized ModelIR state, including tensor buffers, topology,
options, metadata, provenance, quantization, and lineage diagnostics.

Validation completed as follows:

- corrected checkpoint/module AST comparison: exact, 343 lines each;
- focused owner/wrapper and architecture selector: `300 passed in 20.88s`;
- all focused Softmax/layout plus architecture suites:
  `453 passed in 21.13s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.42s`;
- targeted Ruff for the new module and tests, Python compilation, and
  whitespace checks: passed;
- the central lowerer retains exactly seven pre-existing Ruff findings.

No real-model conversion was repeated. The move is mechanically identical to
the corrected checkpoint, and direct equality covers every synthetic non-zero
family plus the critical rejection boundaries. Public API, CLI, artifacts,
dependencies, corpus profiles, exclusions, operation tiers, ordered runtime
behavior, and TensorFlow isolation are unchanged. PR #952 remains closed; no
pull request was created, reopened, or updated.

At restart, inventory and characterize the next raw source-order owner before
editing it: the 394-line
`_optimize_concat_mul_add_transpose_nhwc_bridge_chains`. Reuse its two existing
public fixtures (ordinary and legacy-consumer variants), then freeze multiple-
match, constant ownership/rotation, adapter naming, rank/signature,
quantization, public-boundary, pruning, fixed-point, statistics, and ordered-
production-boundary behavior. Record unsafe behavior as strict xfails before
correction. Keep conversion validation minimal and strictly sequential,
commit and push coherent checkpoints, and do not create a pull request.

## Concat/Mul/Transpose/Add bridge characterization: completed state

The raw 394-line `_optimize_concat_mul_add_transpose_nhwc_bridge_chains` owner
and both ordered production positions are unchanged. Its ordinary and legacy-
consumer fixtures were moved from `test_tflite_builder_direct.py` into the
focused `test_flatbuffer_direct_concat_mul_add_bridge_layout.py`, reducing the
giant direct test by 211 lines while preserving and expanding the same public
behavior.

The focused contract covers ordinary static and dynamic-batch signatures,
legacy-consumer and public-Concat-output compatibility adapters, two
independent graph-order matches, second-call fixed point, four-dimensional and
scalar Mul constants, shared-constant collision-safe cloning, zero-match no-
prune behavior, Concat options/axis/provenance, nine existing arity/layout/
fan-out/public-output/constant rejection guards, statistics, and both nested
recovery-sequence boundaries.

Sixteen concrete problems are strict xfails:

- the appended legacy adapter is later than its existing consumer, violating
  topological order;
- five missing/rank-three/short-signature retained metadata cases still
  rewrite;
- public-input and variable Mul constants rotate in place, and a public
  constant output is not cloned;
- ordinary and legacy per-axis quantized tensors retain QDIM 1 after moving to
  NHWC instead of remapping to QDIM 3;
- the fixed adapter-permutation name overwrites a public input rather than
  allocating a private collision-safe constant;
- malformed legacy metadata raises only after Mul-constant mutation;
- a duplicate post producer, reverse post/Add order, and a produced internal
  tensor also declared as a public input are not rejected.

Validation completed as follows:

- focused characterization: `18 passed, 16 xfailed in 0.75s`;
- focused characterization plus ordered architecture suite:
  `266 passed, 16 xfailed in 20.88s`;
- broad direct/architecture collection: `1005 passed, 16 xfailed, 6 failed in
  172.21s`; four failures require the absent TensorFlow extra, one requires a
  compatible PyTorch binary, and one is an independently failing unchanged
  SiNet helper expectation that also fails alone;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.42s`;
- targeted Ruff for the new focused module, Python compilation, and whitespace
  checks: passed. The giant direct test retains ten unrelated pre-existing
  unused-import findings after removal of this owner's import.

No model conversion was run. Production source, public API, CLI, artifacts,
dependencies, corpus profiles, exclusions, operation tiers, and TensorFlow
isolation are unchanged. PR #952 remains closed; no pull request was created,
reopened, or updated.

At restart, correct the raw owner transactionally before extracting it. Build
one indexed, graph-order candidate plan that validates unique producers,
strict operator order, public boundaries, complete rank-four shape/signature
metadata, constant ownership, and quantization. Precompute Mul-constant update
or clone, QDIM remaps, canonical Concat metadata/name, adapter-permutation
reuse or private clone, all setters/removals, and the adapter insertion index
before mutation. Insert the adapter before its first legacy consumer, turn all
16 strict xfails green, preserve the 18 existing cases and both production
boundaries, validate sequentially, commit and push, and do not create a pull
request.

## Concat/Mul/Transpose/Add transactional correction: completed state

All 16 strict xfails are green. The corrected raw owner is 652 lines and uses
one `ModelIRGraphIndex` for candidate enumeration, unique-producer checks,
consumer ownership, setters, batched removals, and adapter insertion. Two
independent matches allocate the index once; complete maps are no longer
rebuilt on each fixed-point round.

Before the first mutation, each candidate now proves strict
pre-Transpose/Concat/Mul/post-Transpose/Add order, unique retained producers,
private internal edges, complete rank-four source/Concat/Mul shapes and
effective signatures, valid Concat axis/options, NHWC add-bias broadcast, and
all public-boundary rules. Missing tensors, short signatures, duplicate post
producers, reverse post/Add order, and public internal aliases return zero with
complete ModelIR equality.

Mul constants have a complete immutable plan. Scalars and already compatible
lower-rank broadcasts remain untouched. Rank-three or rank-four constants are
rotated only after ownership and broadcast validation; shared and public-
output constants receive deterministic private clones, while public inputs
and variables reject. Per-axis quantization is cloned and remapped with the
same permutation for the Mul constant, canonical Concat tensor, and Mul output.
Both ordinary and legacy variants now use QDIM 3 for NHWC while the legacy
adapter output retains its original NCHW metadata.

Canonical tensor names and adapter-permutation names use candidate-local
reservations that are published only on commit. A reserved, public, variable,
wrong-dtype, quantized, or wrong-value adapter constant is preserved and a
private INT32 constant is allocated. Every setter, removal, metadata update,
option update, and adapter insertion is planned before commit. Compatibility
adapters are inserted immediately after Concat, ahead of the main Mul and all
legacy consumers, so both legacy and public-Concat-output graphs validate as
topological ModelIR.

Validation completed as follows:

- focused corrected owner, including one-index and invariant checks:
  `35 passed in 0.59s`;
- focused corrected owner plus ordered architecture suite:
  `283 passed in 20.49s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.35s`;
- targeted Python compilation, focused-test Ruff, and whitespace checks:
  passed;
- central lowerer Ruff findings decreased from seven to six because the
  corrected owner removed an unused local assignment.

No real-model conversion or broad direct-suite repeat was added. The rewrite
is restricted to proven candidates; incomplete evidence now leaves the
original graph intact. Public API, CLI, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, both ordered runtime boundaries, and
TensorFlow isolation are unchanged. PR #952 remains closed; no pull request
was created, reopened, or updated.

At restart, mechanically extract the corrected 652-line
`_optimize_concat_mul_add_transpose_nhwc_bridge_chains` owner into a focused
pass module. Keep the historical lowerer private name as a one-return wrapper
and preserve its position in both terminal recovery sequences. Prove corrected
checkpoint/module AST identity plus direct owner/wrapper equality for ordinary
static/dynamic, multiple, scalar, shared/public constants, legacy/public
Concat adapters, per-axis quantization, collision, pruning, rejection, and
atomicity cases. Validate sequentially, commit and push, and do not create a
pull request.

## Concat/Mul/Transpose/Add ownership extraction: completed state

The corrected owner now resides in
`onnx2tf/tflite_builder/passes/concat_mul_add_bridge_layout.py`. Its function
and the corrected raw owner at checkpoint `5193fc11` are each 652 lines and
have identical ASTs. The central lowerer imports it under the private
`_optimize_concat_mul_add_transpose_nhwc_bridge_chains_pass` alias and keeps
the historical private name as a one-return wrapper. Its third position in
both terminal Concat recovery sequences and the immediate neighboring calls
are unchanged. The pass module does not import the lowerer.

Fifteen direct owner/wrapper comparisons cover ordinary static and dynamic
metadata, two independent matches, scalar constants, shared and public-output
constant clones, legacy and public-Concat-output adapters, ordinary and legacy
per-axis quantization, adapter-name collision, unmatched pruning behavior,
missing metadata, reverse topology, and a public internal boundary. Deep-
copied executions produce identical statistics and complete normalized
ModelIR state, including tensor buffers, quantization, options, provenance,
topology, metadata, and diagnostics.

Validation completed as follows:

- corrected checkpoint/module AST comparison: exact, 652 lines each;
- focused owner/wrapper and safety contract: `50 passed in 0.61s`;
- focused contract plus ordered architecture suite:
  `298 passed in 20.20s`;
- targeted Python compilation and whitespace checks: passed;
- targeted Ruff reports only the six pre-existing lowerer findings after the
  extraction-specific `_quant_scale_count` import was removed.

No real-model conversion or broad direct-suite repeat was added. The move is
mechanically identical to the corrected checkpoint. Public API, CLI,
artifacts, dependencies, corpus profiles, exclusions, operation tiers,
ordered runtime behavior, and TensorFlow isolation are unchanged. PR #952
remains closed; no pull request was created, reopened, or updated.

At restart, inventory and characterize the next raw source-order owner before
editing it: the 452-line
`_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains`. Its existing
legacy-consumer fixture remains in `tests/test_tflite_builder_direct.py` and
its two ordered production positions are already asserted by architecture
tests. Move relevant fixtures into a focused module, freeze positive,
multiple-match, constant ownership, metadata, quantization, adapter naming,
topology, public-boundary, pruning, fixed-point, statistics, and ordered-
boundary behavior, and record unsafe behavior as strict xfails before any
correction. Keep conversion validation minimal and strictly sequential,
commit and push coherent checkpoints, and do not create a pull request.

## Concat/Mul/Add/Transpose/Add characterization: completed state

The 452-line raw
`_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains` owner and its
fourth position in both terminal Concat recovery sequences are unchanged. The
existing legacy-consumer fixture moved from `test_tflite_builder_direct.py`
into the focused
`test_flatbuffer_direct_concat_mul_add_transpose_add_bridge_layout.py`,
reducing the giant direct test by 102 lines.

The focused contract freezes ordinary static and dynamic signatures, legacy
compatibility output, two independent matches, fixed point, scalar constants,
collision-safe shared Mul/Add constant clones, no-match pruning behavior,
Concat options/axis semantics/version/provenance, nine existing rejection
guards, statistics, the 452-line/two-While raw-owner shape, and both ordered
production boundaries.

Twenty-seven reproduced safety problems are strict xfails:

- one non-topological legacy adapter placement;
- seven incomplete source/Concat/Mul/Add metadata cases;
- six public-input, variable, or public-output affine-constant ownership
  violations;
- two ordinary/legacy per-axis QDIM failures;
- five unsafe reserved adapter-permutation ownership/value cases;
- two late partial-mutation cases for the second affine constant and legacy
  metadata;
- one malformed Concat-axis exception;
- three duplicate-producer, reverse-order, or public-internal-alias cases.

Validation completed as follows:

- focused characterization: `18 passed, 27 xfailed in 0.96s`;
- focused characterization plus ordered architecture suite:
  `266 passed, 27 xfailed in 21.04s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.60s`;
- focused-test Ruff, Python compilation, and whitespace checks: passed.

No production source or real-model conversion was changed or run. Public API,
CLI, artifacts, dependencies, corpus profiles, exclusions, operation tiers,
and TensorFlow isolation are unchanged. PR #952 remains closed; no pull
request was created, reopened, or updated.

At restart, correct the raw owner transactionally before extracting it. Use
one `ModelIRGraphIndex` to enumerate candidates and prove unique producers,
strict pre-Transpose/Concat/Mul/Add/post-Transpose/tail-Add order, private
internal edges, complete rank-four effective metadata, immutable affine-
constant ownership, valid broadcasts, per-axis QDIM remaps, and safe adapter
ownership/names. Precompute both constant update-or-clone actions, canonical
Concat metadata/name, every setter/removal, and the producer-before-consumer
adapter insertion index before mutation. Turn all 27 strict xfails green,
preserve the 18 existing cases and both production boundaries, validate
sequentially, commit and push, and do not create a pull request.

## Concat/Mul/Add/Transpose/Add transactional correction: completed state

All 27 former strict xfails are green. The corrected raw owner is 866 lines
and constructs one `ModelIRGraphIndex`; a two-branch fixed-point rewrite
proves the index is refreshed exactly once. Producer/consumer maps are no
longer rebuilt on every candidate round.

Before mutation, each candidate proves strict pre-Transpose/Concat/Mul/Add/
post-Transpose/tail-Add order, unique producers, exact internal consumers,
private intermediate boundaries, complete rank-four source/Concat/Mul/Add
shape and effective-signature metadata, safe Concat axis/options, and a valid
NHWC tail broadcast. Missing tensors, rank-three sources, short signatures,
malformed axes, duplicate producers, reverse order, and public internal aliases
now return zero without changing tensors, operators, metadata, lineage, or
diagnostics.

Mul and pre-Transpose Add constants share one immutable planning policy.
Scalars and already NHWC-compatible broadcasts remain unchanged. NCHW rank-
three/four values rotate only after ownership and target-broadcast validation;
shared and public-output constants receive deterministic private clones,
while public inputs and variables reject. Per-axis quantization is cloned and
remapped for both constants, the canonical Concat tensor, Mul output, and Add
output. Ordinary and legacy paths therefore use QDIM 3 for all NHWC tensors,
while the compatibility output keeps its original NCHW contract.

Candidate-local name reservations are published only at commit. The reserved
adapter permutation is reused only when it is a private immutable unquantized
INT32 tensor with exact `[4]` metadata, an INT32 backing buffer, and the exact
permutation. Public, variable, wrong-dtype, quantized, or wrong-value tensors
are preserved and receive a private collision-safe replacement. The adapter
is inserted immediately after Concat and before all existing consumers. Both
constant plans, every metadata/QDIM result, canonical tensor, adapter, setter,
removal, and insertion are complete before the first mutation.

Validation completed as follows:

- corrected focused contract, including all former xfails and one-index
  construction: `46 passed in 0.59s`;
- corrected focused contract, adjacent extracted bridge contract, and ordered
  architecture suite: `344 passed in 20.58s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.40s`;
- targeted Python compilation and whitespace checks: passed;
- central lowerer Ruff remains at the same six pre-existing findings.

No real-model conversion or broad direct-suite repeat was added. The rewrite
is limited to fully proven candidates; incomplete evidence leaves the graph
unchanged. Public API, CLI, artifacts, dependencies, corpus profiles,
exclusions, operation tiers, both ordered runtime boundaries, and TensorFlow
isolation are unchanged. PR #952 remains closed; no pull request was created,
reopened, or updated.

At restart, mechanically extract the corrected 866-line
`_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains` owner into a
focused pass module. Keep the historical lowerer private name as a one-return
wrapper and preserve its position in both terminal recovery sequences. Prove
corrected checkpoint/module AST identity and direct owner/wrapper equality for
ordinary static/dynamic, multiple, scalar, shared/public constants, legacy
adapters, per-axis quantization, adapter collisions, pruning, rejection, and
atomicity cases. Validate sequentially, commit and push, and do not create a
pull request.

## Concat/Mul/Add/Transpose/Add ownership extraction: completed state

The corrected owner now resides in
`onnx2tf/tflite_builder/passes/concat_mul_add_transpose_add_bridge_layout.py`.
Its function and the corrected raw owner at checkpoint `4a5f0394` are each 866
lines and have identical ASTs. The central lowerer imports it under the private
`_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains_pass` alias and
keeps the historical private name as a one-return wrapper. Its fourth position
in both terminal Concat recovery sequences and both immediate neighboring
calls are unchanged. The pass module does not import the lowerer.

Nineteen direct owner/wrapper comparisons cover ordinary static and dynamic
metadata, two independent matches, scalar constants, separate shared Mul/Add
constant collisions, separate public-output clones, legacy adapters, ordinary
and legacy per-axis quantization, adapter-name collision, unmatched pruning,
missing metadata, late constant and metadata evidence, malformed axis,
reverse topology, and a public internal boundary. Deep-copied executions
produce identical statistics and complete normalized ModelIR state, including
buffers, quantization, options, provenance, topology, metadata, lineage, and
diagnostics.

Validation completed as follows:

- corrected checkpoint/module AST comparison: exact, 866 lines each;
- focused safety plus owner/wrapper contract: `65 passed in 0.63s`;
- focused contract, adjacent extracted bridge contract, and ordered
  architecture suite: `363 passed in 19.78s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.34s`;
- targeted Ruff for the new module and focused test, Python compilation, and
  whitespace checks: passed;
- the central lowerer retains exactly six pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. The move is
mechanically identical to the corrected checkpoint. Public API, CLI,
artifacts, dependencies, corpus profiles, exclusions, operation tiers,
ordered runtime behavior, and TensorFlow isolation are unchanged. PR #952
remains closed; no pull request was created, reopened, or updated.

At restart, inventory and characterize the next raw source-order owner before
editing it: the 461-line
`_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains`. Its
existing public fixture remains in `tests/test_tflite_builder_direct.py`, and
both ordered positions are already asserted by architecture tests. Move the
fixture into a focused module, freeze positive, multiple-match, constants,
metadata, quantization, adapter naming, topology, public-boundary, pruning,
fixed-point, statistics, and ordered-boundary behavior, and record unsafe
behavior as strict xfails before correction. Keep validation minimal and
strictly sequential, commit and push coherent checkpoints, and do not create a
pull request.

## Concat/Mul/Add/Add/Mean/Reshape characterization: completed state

The 461-line raw
`_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains` owner and
its fifth position in both terminal Concat recovery sequences are unchanged.
The existing positive fixture moved from `test_tflite_builder_direct.py` into
the focused `test_flatbuffer_direct_concat_mul_add_add_mean_reshape_layout.py`,
reducing the giant direct test by 94 lines.

The focused contract freezes static and dynamic signatures, two independent
matches, fixed point, scalar constants, collision-safe shared clones for Mul
and both Adds, shared Mean-axes cloning, exact old-Mean-shape rewriting, no-
match pruning behavior, Concat/Mean options/version/provenance, ten existing
rejection guards, statistics, the 461-line/two-While owner shape, and both
ordered production boundaries.

Forty-two reproduced safety problems are strict xfails:

- eleven incomplete source/Concat/Mul/Add/Mean metadata cases;
- nine affine-constant public/variable/output ownership violations;
- one complete per-axis QDIM case;
- six unsafe or public-output Mean-axes cases;
- six unsafe, shared, or public-output Reshape-shape cases;
- four late partial-mutation cases;
- one malformed Concat-axis exception;
- three duplicate-producer, reverse-order, or public-alias cases;
- one identity Mean-axis mapping false negative with partial mutation.

Validation completed as follows:

- focused characterization: `21 passed, 42 xfailed in 1.21s`;
- focused characterization plus ordered architecture suite:
  `269 passed, 42 xfailed in 20.55s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.49s`;
- focused-test Ruff, Python compilation, and whitespace checks: passed.

No production source or real-model conversion was changed or run. Public API,
CLI, artifacts, dependencies, corpus profiles, exclusions, operation tiers,
and TensorFlow isolation are unchanged. PR #952 remains closed; no pull
request was created, reopened, or updated.

At restart, correct the raw owner transactionally before extracting it. Use
one `ModelIRGraphIndex` to prove unique producers, strict pre-Transpose/Concat/
Mul/Add/Add/Mean/Reshape order, private intermediates, exact consumers, and
complete rank-four effective metadata. Precompute immutable update-or-clone
plans for all three affine constants, an ownership/type-safe Mean-axes plan
that accepts identity mappings, and a conditional ownership/type-safe Reshape-
shape plan. Remap every affected per-axis QDIM and complete all names, tensors,
setters, metadata changes, and removals before the first mutation. Turn all 42
strict xfails green, preserve the 21 existing cases and both production
boundaries, validate sequentially, commit and push, and do not create a pull
request.

## Concat/Mul/Add/Add/Mean/Reshape transactional correction: completed state

All 42 former strict xfails are green. The corrected raw owner is 869 lines
and constructs one `ModelIRGraphIndex`; a two-branch fixed-point rewrite proves
the index is refreshed exactly once. Indexed setters and one batched removal
replace repeated whole-graph producer/consumer reconstruction and direct
operator deletion.

Before mutation, each candidate proves unique producers, strict pre-Transpose/
Concat/Mul/Add/Add/Mean/Reshape order, exact private internal consumers,
complete rank-four source/Concat/Mul/Add/Mean effective metadata, valid Concat
axis/options, Mean keep-dims semantics, and no unsupported Concat fan-out.
Missing tensors, rank-three sources, short signatures, malformed axes,
duplicate Mean producers, reverse order, and public internal aliases now leave
the complete ModelIR unchanged.

Mul and both Add constants share one immutable action planner. Scalars and
already compatible NHWC broadcasts remain unchanged. Rotated rank-three/four
constants require valid target broadcasting and non-public-input, non-variable
ownership; shared and public-output values receive deterministic private
clones. Rank-specific per-axis QDIM remaps apply to all three constants, while
Concat, Mul, both Add outputs, and Mean output use the full rank-four remap.

Mean axes now have an explicit immutable unquantized INT32 contract covering
TensorIR dtype, backing-buffer dtype, shape/signature, ownership, range, and
negative-axis normalization. Identity remaps are successful no-change plans;
changed shared/public-output axes clone. The Reshape shape is rewritten only
when it exactly equals the old Mean shape, and then uses the same ownership,
type, metadata, and cloning policy. Both plans, every affine action, all QDIM
results, names, tensors, setters, metadata writes, and removals are complete
before commit, so late constant or Mean evidence cannot leave partial state.

Validation completed as follows:

- corrected focused contract, including all former xfails, one-index, and
  Concat-fan-out checks: `65 passed in 0.64s`;
- corrected focused contract, both adjacent extracted bridge contracts, and
  ordered architecture suite: `428 passed in 20.13s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.39s`;
- targeted Ruff for the focused test, Python compilation, and whitespace
  checks: passed;
- central lowerer Ruff findings decreased from six to five because the new
  owner removed the obsolete `reshape_out_name` local.

No real-model conversion or broad direct-suite repeat was added. The rewrite
is limited to fully proven candidates; incomplete evidence leaves the graph
unchanged. Public API, CLI, artifacts, dependencies, corpus profiles,
exclusions, operation tiers, both ordered runtime boundaries, and TensorFlow
isolation are unchanged. PR #952 remains closed; no pull request was created,
reopened, or updated.

At restart, mechanically extract the corrected 869-line
`_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains` owner into
a focused pass module. Keep the historical lowerer private name as a one-
return wrapper and preserve its position in both terminal recovery sequences.
Prove corrected checkpoint/module AST identity and direct owner/wrapper
equality for static/dynamic, multiple, scalar, shared/public affine constants,
Mean axes, Reshape shape, quantization, pruning, rejection, and atomicity
cases. Validate sequentially, commit and push, and do not create a pull
request.

## Concat/Mul/Add/Add/Mean/Reshape ownership extraction: completed state

The corrected owner now resides in
`onnx2tf/tflite_builder/passes/concat_mul_add_add_mean_reshape_layout.py`. Its
function and the corrected raw owner at checkpoint `3c3579fd` are each 869
lines and have identical ASTs. The central lowerer imports it under the private
`_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains_pass` alias
and keeps the historical private name as a one-return wrapper. Its fifth
position in both terminal Concat recovery sequences and immediate neighboring
calls are unchanged. The pass module does not import the lowerer.

Twenty-three direct owner/wrapper comparisons cover ordinary static and
dynamic metadata, two independent matches, scalar constants, separate shared
and public-output actions for all three affine constants, shared and public
Mean axes, exact and public Reshape shapes, per-axis quantization, identity
axes, unmatched pruning, missing metadata, late affine and Mean evidence,
malformed axis, reverse topology, and a public internal boundary. Deep-copied
executions produce identical statistics and complete normalized ModelIR state,
including buffers, quantization, options, provenance, topology, metadata,
lineage, and diagnostics.

Validation completed as follows:

- corrected checkpoint/module AST comparison: exact, 869 lines each;
- focused safety plus owner/wrapper contract: `88 passed in 0.66s`;
- focused contract, both adjacent extracted bridge contracts, and ordered
  architecture suite: `451 passed in 20.05s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.44s`;
- targeted Ruff for the new module and focused test, Python compilation, and
  whitespace checks: passed;
- the central lowerer retains exactly five pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. The move is
mechanically identical to the corrected checkpoint. Public API, CLI,
artifacts, dependencies, corpus profiles, exclusions, operation tiers,
ordered runtime behavior, and TensorFlow isolation are unchanged. PR #952
remains closed; no pull request was created, reopened, or updated.

At restart, inventory and characterize the next raw source-order owner before
editing it: the 356-line
`_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains`. Its existing
public fixture remains in `tests/test_tflite_builder_direct.py`, and both
ordered positions are asserted by architecture tests. Move the fixture into a
focused module, freeze positive, multiple-match, nested-Concat, constants,
metadata, quantization, topology, public-boundary, pruning, fixed-point,
statistics, and ordered-boundary behavior, and record unsafe behavior as
strict xfails before correction. Keep validation minimal and strictly
sequential, commit and push coherent checkpoints, and do not create a pull
request.

## Nested-Concat/Mul/Transpose characterization: completed state

The 356-line raw
`_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains` owner and its
historical position in both terminal Concat recovery sequences are unchanged.
The existing mixed-axis fixture moved from `test_tflite_builder_direct.py` into
the focused `test_flatbuffer_direct_concat_tree_mul_add_bridge_layout.py`, and
the giant direct test no longer imports this private owner.

The twenty green focused cases freeze static and dynamic-batch mixed-axis
Concat trees, two independent graph-order matches, fixed point, scalar
constants, collision-safe shared constant cloning, negative-axis
normalization, zero-match no-prune behavior, twelve existing rejection guards,
statistics, the current 356-line/three-While owner shape, and both ordered
production boundaries.

Nineteen reproduced safety problems are strict xfails:

- eight incomplete source/inner-Concat/root-Concat/Mul metadata cases;
- three unsafe public-input, variable, or public-output Mul-constant ownership
  paths;
- one complete per-axis QDIM case;
- two malformed inner/root Concat-axis cases;
- one late metadata case that leaves a rotated constant behind;
- four duplicate-producer, reverse-order, or public-alias cases.

Validation completed sequentially as follows:

- focused characterization: `20 passed, 19 xfailed in 0.90s`;
- focused characterization plus ordered architecture suite:
  `268 passed, 19 xfailed in 19.22s`;
- focused characterization, the three adjacent Concat bridge contracts, and
  ordered architecture suite: `471 passed, 19 xfailed in 20.10s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.66s`;
- focused-test Ruff, Python compilation, and whitespace checks: passed.

No production source or real-model conversion changed or ran. Public API,
CLI, artifacts, dependencies, corpus profiles, exclusions, operation tiers,
ordered runtime behavior, and TensorFlow isolation are unchanged. The 356-line
count is descriptive only; 2,000 remains the ONNX operation-count threshold
for tiering. PR #952 remains closed; no pull request was created, reopened, or
updated.

At restart, correct the raw owner transactionally before extracting it. Use
one `ModelIRGraphIndex` to enumerate the recursive Concat tree in deterministic
graph order and prove unique producers, strict pre-Transpose/Concat/Mul/post-
Transpose/Add order, private intermediates, exact consumers, and complete
rank-four effective metadata. Precompute an ownership-aware Mul-constant
update-or-clone plan, QDIM remaps, validated normalized axes, all indexed
setters, metadata writes, rewires, and removals before the first mutation.
Turn all 19 strict xfails green while preserving the twenty existing cases,
statistics, fixed point, pruning behavior, and both production boundaries.
Validate sequentially, commit and push, and do not create a pull request.

## Nested-Concat/Mul/Transpose transactional correction: completed state

All nineteen former strict xfails are green. The corrected raw owner is 675
lines and constructs one `ModelIRGraphIndex`; a two-branch fixed-point rewrite
refreshes it exactly once. Indexed setters and one batched removal replace
repeated complete producer/consumer reconstruction and direct operator
deletion.

Before mutation, each candidate proves unique producers, strict leaf pre-
Transpose/nested-Concat/Mul/post-Transpose/Add graph order, exact internal
consumers, private intermediate boundaries, valid normalized Concat axes, and
complete rank-four source, every Concat-output, and Mul-output shape/effective-
signature metadata. Missing or short metadata, rank-three sources, malformed
axes, duplicate post producers, reverse topology, and public aliases now leave
the complete ModelIR unchanged.

Every recursive Concat input replacement, remapped axis/options dictionary,
shape/signature permutation, quantization action, Add rewire, and removal is
planned before commit. All per-axis QDIM values follow the NCHW-to-NHWC
permutation for every retained Concat output and the Mul output. The Add-side
NHWC channel broadcast is validated against the planned Mul shape.

The Mul constant uses one immutable ownership plan. Scalars and already
compatible NHWC broadcasts remain unchanged. Required rank-four rotations
reject public inputs and variables; shared and public-output values receive a
deterministic collision-safe private clone; private single-use values update in
place. The constant QDIM follows its data permutation, and clones preserve
dtype, quantization, logical/physical layout, and ONNX provenance.

Validation completed sequentially as follows:

- corrected focused safety and one-index contract: `42 passed in 0.56s`;
- corrected focused contract, the three adjacent extracted Concat bridge
  contracts, and ordered architecture suite: `493 passed in 19.70s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.68s`;
- focused-test Ruff, Python compilation, and whitespace checks: passed;
- the central lowerer retains exactly five pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. The rewrite
is limited to fully proven candidates; incomplete evidence leaves the graph
unchanged. Public API, CLI, artifacts, dependencies, corpus profiles,
exclusions, operation tiers, both ordered runtime boundaries, and TensorFlow
isolation are unchanged. The 675-line count is descriptive only; 2,000 remains
the ONNX operation-count tier threshold. PR #952 remains closed; no pull
request was created, reopened, or updated.

At restart, mechanically extract the corrected 675-line
`_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains` owner into a
focused pass module. Keep the historical lowerer private name as a one-return
wrapper and preserve both ordered positions. Prove corrected checkpoint/module
AST identity and direct owner/wrapper equality for static/dynamic, multiple,
scalar, shared/public constants, quantization, pruning, rejection, and atomicity
cases. Validate sequentially, commit and push, and do not create a pull request.

## Nested-Concat/Mul/Transpose ownership extraction: completed state

The corrected owner now resides in
`onnx2tf/tflite_builder/passes/concat_tree_mul_add_bridge_layout.py`. Its
function and the corrected raw owner at checkpoint `4111187c` are each 675
lines and have identical ASTs. The central lowerer imports it under the private
`_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains_pass` alias and
keeps the historical private name as a one-return wrapper. Its position after
the extracted Concat/Mul/Add/Add/Mean/Reshape owner and before singleton-gate
recovery in both terminal sequences is unchanged. The pass module does not
import the lowerer.

Seventeen direct owner/wrapper comparisons cover ordinary static and dynamic
metadata, two matches, scalar constants, shared collision-safe and public-
output clones, public-input and variable rejection, per-axis quantization,
unmatched pruning, missing and late metadata, malformed axes, reverse nested
topology, a public internal boundary, and reverse or duplicate source
producers. Deep-copied executions produce identical statistics and complete
normalized ModelIR state, including buffers, quantization, options,
provenance, topology, metadata, lineage, and diagnostics.

Validation completed sequentially as follows:

- corrected checkpoint/module AST comparison: exact, 675 lines each;
- focused safety plus owner/wrapper contract: `59 passed in 0.58s`;
- focused contract, the three adjacent extracted Concat bridge contracts, and
  ordered architecture suite: `510 passed in 19.29s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.54s`;
- targeted Ruff for the new module and focused test, Python compilation, and
  whitespace checks: passed;
- the central lowerer retains exactly five pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. The move is
mechanically identical to the corrected checkpoint. Public API, CLI,
artifacts, dependencies, corpus profiles, exclusions, operation tiers,
ordered runtime behavior, and TensorFlow isolation are unchanged. PR #952
remains closed; no pull request was created, reopened, or updated.

At restart, inventory and characterize the next raw source-order owner before
editing it: the 543-line
`_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains`.
Its existing public fixture remains in `tests/test_tflite_builder_direct.py`,
and all three production calls are already asserted by architecture tests.
Move the fixture into a focused module; freeze positive, multiple-match,
StridedSlice/Pad constants, Concat/Mul/Add constants, metadata, quantization,
topology, public-boundary, pruning, fixed-point, statistics, and ordered-call
behavior; and record unsafe behavior as strict xfails before correction. Keep
validation minimal and strictly sequential, commit and push coherent
checkpoints, and do not create a pull request.

## StridedSlice/Pad/Concat bridge characterization: completed state

The 543-line raw
`_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains`
owner and all three production calls are unchanged. The existing public fixture
moved from `test_tflite_builder_direct.py` into the focused
`test_flatbuffer_direct_stridedslice_pad_concat_bridge_layout.py`, reducing the
giant direct test by 117 lines and removing its private owner import.

The twenty-six green cases freeze static and dynamic-batch signatures, two
independent graph-order matches and fixed point, multiple Add users, Pad and
MirrorPad options/provenance, scalar Mul constants, collision-safe external-
user clones for Slice-end, Pad, and Mul constants, zero-match no-prune behavior,
seventeen existing rejection guards, statistics, the current 543-line/two-
While owner shape, and all three ordered production calls.

Forty-two reproduced safety problems are strict xfails:

- ten incomplete source/Slice/Pad/Concat/Mul metadata cases;
- sixteen unsafe public-input, variable, wrong-dtype, or quantized Slice-vector
  and Pad-matrix constant paths;
- two changed public Slice/Pad constant outputs that mutate instead of clone;
- three unsafe public-input, variable, or public-output Mul-constant paths;
- one complete per-axis QDIM case;
- two public Slice/Pad intermediate boundaries;
- six duplicate-producer, reverse-order, or public-alias cases;
- two malformed Concat/Slice option cases that raise.

Validation completed sequentially as follows:

- focused characterization: `26 passed, 42 xfailed in 1.23s`;
- focused characterization plus ordered architecture suite:
  `274 passed, 42 xfailed in 18.98s`;
- focused characterization, the four adjacent extracted Concat bridge
  contracts, and ordered architecture suite:
  `536 passed, 42 xfailed in 20.06s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.31s`;
- focused-test Ruff, Python compilation, and whitespace checks: passed.

No production source or real-model conversion changed or ran. Public API, CLI,
artifacts, dependencies, corpus profiles, exclusions, operation tiers, ordered
runtime behavior, and TensorFlow isolation are unchanged. The 543-line count
is descriptive only; 2,000 remains the ONNX operation-count tier threshold. PR
#952 remains closed; no pull request was created, reopened, or updated.

At restart, correct the raw owner transactionally before extracting it. Use one
`ModelIRGraphIndex` to prove unique producers, strict pre-Transpose/
StridedSlice/Pad/Concat/Mul/post-Transpose/Add order, exact private internal
consumers, complete rank-four metadata, and the supported ordered multi-Add
tail. Precompute grouped ownership/type-safe unquantized INT32 actions for all
Slice vectors and Pad matrices, an ownership-aware Mul-constant action, every
QDIM remap, normalized mask/axis option, clone name, indexed setter, Mul-output
rename, and batched removal before the first mutation. Turn all 42 strict
xfails green while preserving the twenty-six existing cases, statistics,
fixed point, pruning, Pad/MirrorPad behavior, and all three production calls.
Validate sequentially, commit and push, and do not create a pull request.

## StridedSlice/Pad/Concat transactional correction: completed state

All forty-two former strict xfails are green. The corrected raw owner is 1,100
lines and constructs one `ModelIRGraphIndex`; a two-branch fixed-point rewrite
refreshes it exactly once. Indexed input/output setters and one batched removal
replace repeated complete producer/consumer scans and direct operator deletion.

Before mutation, each candidate proves unique producers, strict source/pre-
Transpose/StridedSlice/Pad/Concat/Mul/post-Transpose/ordered-Add graph order,
exact private internal consumers, complete rank-four source/Slice/Pad/Concat/
Mul metadata, a present post-output tensor, valid normalized Concat and Slice
options, and the supported one-or-more Add tail. Missing or short metadata,
missing post output, malformed axes/masks, duplicate producers, reverse order,
and public aliases now return zero with complete ModelIR equality.

All Slice, Pad, Concat, and renamed Mul-output shapes/signatures and per-axis
QDIM values are planned before commit. Add channel-last constants are checked
against the planned Mul shape. The Mul output is renamed through the indexed
producer setter before the post-Transpose is removed, so every existing valid
Add user remains connected without a transient stale index.

Slice begin/end/stride vectors and Pad matrices share one grouped immutable
constant transaction. Every value must have exact unquantized INT32 TensorIR,
buffer, shape, and signature metadata, no public-input or variable ownership,
and no runtime producer. Requirements sharing a tensor identity must agree on
one target. Unchanged values remain shared, private changed values update once,
and changed public outputs or values with any unrelated consumer edge receive
one deterministic collision-safe clone reused at all planned sites. Clone
metadata preserves layout and ONNX provenance. The Mul constant uses the same
ownership-aware update/clone policy and remaps its QDIM with its data.

Validation completed sequentially as follows:

- corrected focused safety, grouped-clone, missing-post, and one-index
  contract: `70 passed in 0.62s`;
- corrected focused contract, the four adjacent extracted Concat bridge
  contracts, and ordered architecture suite: `580 passed in 19.32s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.38s`;
- focused-test Ruff, Python compilation, and whitespace checks: passed;
- the central lowerer retains exactly five pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. The rewrite
is restricted to fully proven candidates; incomplete evidence leaves the
graph unchanged. Public API, CLI, artifacts, dependencies, corpus profiles,
exclusions, operation tiers, Pad/MirrorPad behavior, multiple Add users, all
three ordered runtime calls, and TensorFlow isolation are unchanged. The
1,100-line count is descriptive only; 2,000 remains the ONNX operation-count
tier threshold. PR #952 remains closed; no pull request was created, reopened,
or updated.

At restart, mechanically extract the corrected 1,100-line
`_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains`
owner into a focused pass module. Keep the historical lowerer private name as
a one-return wrapper and preserve all three production calls. Prove corrected
checkpoint/module AST identity and direct owner/wrapper equality for static/
dynamic, multiple, multi-Add, Pad/MirrorPad, scalar, grouped shared/public
constants, quantization, pruning, rejection, and atomicity cases. Validate
sequentially, commit and push, and do not create a pull request.

## StridedSlice/Pad/Concat ownership extraction: completed state

The corrected owner now resides in
`onnx2tf/tflite_builder/passes/stridedslice_pad_concat_bridge_layout.py`. Its
function and the corrected raw owner at checkpoint `95a5555b` are each 1,100
lines and have identical ASTs. The central lowerer imports it under the private
`_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains_pass`
alias and keeps the historical private name as a one-return wrapper. All three
production calls remain unchanged. The pass module does not import the lowerer.

Twenty direct owner/wrapper comparisons cover ordinary static and dynamic
metadata, two matches, multiple Add users, Pad and MirrorPad, scalar Mul,
grouped shared constants, public index outputs, public-input and wrong-dtype
index rejection, public and variable Mul ownership, per-axis quantization,
unmatched pruning, missing retained and post-output metadata, malformed
options, reverse topology, a public intermediate, and duplicate source
producers. Deep-copied executions produce identical statistics and complete
normalized ModelIR state, including buffers, quantization, options,
provenance, topology, metadata, lineage, and diagnostics.

Validation completed sequentially as follows:

- corrected checkpoint/module AST comparison: exact, 1,100 lines each;
- focused safety plus owner/wrapper contract: `90 passed in 0.65s`;
- focused contract, the four adjacent extracted Concat bridge contracts, and
  ordered architecture suite: `600 passed in 18.48s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.41s`;
- targeted Ruff for the new module and focused test, Python compilation, and
  whitespace checks: passed;
- the central lowerer retains exactly five pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. The move is
mechanically identical to the corrected checkpoint. Public API, CLI,
artifacts, dependencies, corpus profiles, exclusions, operation tiers,
Pad/MirrorPad and multiple-Add behavior, all three ordered runtime calls, and
TensorFlow isolation are unchanged. PR #952 remains closed; no pull request
was created, reopened, or updated.

At restart, inventory and characterize the next substantive raw source-order
owner before editing it: the 218-line
`_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains`.
No focused public fixture currently owns this pass; only architecture coverage
references it. Build focused synthetic ModelIR cases for positive, dynamic,
multiple-match, shape-constant ownership, metadata, quantization, topology,
public boundaries, pruning, fixed point, statistics, and every production-call
boundary. Record unsafe behavior as strict xfails before correction. Keep
validation minimal and strictly sequential, commit and push coherent
checkpoints, and do not create a pull request.

## Reshape/Transpose collapse characterization: completed state

The 218-line raw
`_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains` owner and
both production calls are unchanged. It had no focused public fixture; the new
`test_flatbuffer_direct_reshape_transpose_collapse_layout.py` now owns its
synthetic ModelIR contract.

The fourteen green cases freeze ordinary static collapse, two independent
graph-order matches and fixed point, collision-safe shared shape cloning, ten
existing wrong-permutation/public-intermediate/fan-out/incompatible-shape/
missing-output rejection guards, statistics, the current 218-line/two-While
owner shape, and both production calls.

Nineteen reproduced safety problems are strict xfails:

- one dynamic-batch signature case that writes concrete batch one;
- one zero-match case that prunes an unrelated tensor;
- six unsafe public-input, variable, TensorIR/buffer dtype, quantized, or
  data-less shape-constant cases;
- one changed public shape output that mutates instead of cloning;
- five short input/intermediate/output signature cases;
- five duplicate-producer, reverse-order, or public-alias cases.

Validation completed sequentially as follows:

- focused characterization: `14 passed, 19 xfailed in 0.67s`;
- focused characterization plus ordered architecture suite:
  `262 passed, 19 xfailed in 18.17s`;
- focused characterization, the five adjacent extracted bridge contracts, and
  ordered architecture suite: `614 passed, 19 xfailed in 18.63s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.23s`;
- focused-test Ruff, Python compilation, and whitespace checks: passed.

No production source or real-model conversion changed or ran. Public API, CLI,
artifacts, dependencies, corpus profiles, exclusions, operation tiers, both
ordered runtime calls, and TensorFlow isolation are unchanged. The 218-line
count is descriptive only; 2,000 remains the ONNX operation-count tier
threshold. PR #952 remains closed; no pull request was created, reopened, or
updated.

At restart, correct the raw owner transactionally before extracting it. Use one
`ModelIRGraphIndex` to prove unique producers, strict Reshape/Transpose/
Reshape/Transpose order, exact private internal consumers, complete rank-three/
rank-four shapes and signatures, and a valid output boundary. Derive the target
shape/options from compatible output signatures so a dynamic batch remains
`-1`. Precompute an ownership/type-safe unquantized INT32 shape update-or-clone,
collision-safe name, indexed output setter, option changes, removals, and prune
decision before mutation. Turn all 19 strict xfails green while preserving the
fourteen existing cases, statistics, fixed point, provenance/options, and both
production calls. Validate sequentially, commit and push, and do not create a
pull request.

## Reshape/Transpose collapse transactional correction: completed state

All nineteen former strict xfails are green. The corrected raw owner is 399
lines and constructs one `ModelIRGraphIndex`; a two-branch fixed-point rewrite
refreshes it exactly once. Indexed input/output setters and one batched removal
replace the repeated consumer scan and direct operator deletion.

Before mutation, each candidate proves unique source/intermediate/output
producers, strict Reshape/Transpose/Reshape/Transpose graph order, exact private
internal consumers, complete rank-three/rank-four positive physical shapes,
and compatible full-length signatures. The proven physical N/S/C/H/W
relationship remains unchanged. Every non-batch signature dimension must equal
its physical dimension, while each batch signature may be the physical batch
or `-1`; any dynamic boundary makes the planned target batch `-1`. Shape-buffer,
`newShape`, and `onnxRawNewShape` targets therefore remain consistent.

The first Reshape shape input has an immutable unquantized INT32 contract:
TensorIR and NumPy buffer dtype, `[4]` shape/signature, data presence, ownership,
runtime-producer absence, and equality with the proven first Reshape shape are
validated. Public inputs, variables, wrong dtypes, quantization, missing data,
and produced constants reject. Private values update once; public outputs and
values with unrelated consumer edges receive one deterministic collision-safe
clone preserving layout and ONNX provenance.

Shape action, clone name, dynamic options, indexed output rename, removals, and
pruning are complete before commit. Missing/short metadata, duplicate or
reverse topology, and public aliases leave the complete ModelIR unchanged. A
zero-match invocation no longer prunes unrelated tensors.

Validation completed sequentially as follows:

- corrected focused dynamic, no-prune, ownership, topology, and one-index
  contract: `34 passed in 0.54s`;
- corrected focused contract, the five adjacent extracted bridge contracts,
  and ordered architecture suite: `634 passed in 19.12s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.77s`;
- focused-test Ruff, Python compilation, and whitespace checks: passed;
- the central lowerer retains exactly five pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. The rewrite
is restricted to fully proven candidates. Public API, CLI, artifacts,
dependencies, corpus profiles, exclusions, operation tiers, both ordered
runtime calls, and TensorFlow isolation are unchanged. The 399-line count is
descriptive only; 2,000 remains the ONNX operation-count tier threshold. PR
#952 remains closed; no pull request was created, reopened, or updated.

At restart, mechanically extract the corrected 399-line
`_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains` owner
into a focused pass module. Keep the historical lowerer private name as a one-
return wrapper and preserve both production calls. Prove corrected checkpoint/
module AST identity and direct owner/wrapper equality for static/dynamic,
multiple, shared/public/invalid shape constants, no-prune, signatures,
topology, and atomicity cases. Validate sequentially, commit and push, and do
not create a pull request.

## Reshape/Transpose collapse ownership extraction: completed state

The corrected owner now resides in
`onnx2tf/tflite_builder/passes/reshape_transpose_collapse_layout.py`. Its
function and the corrected raw owner at checkpoint `48aae4b0` are each 399
lines and have identical ASTs. The central lowerer imports it under the private
`_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains_pass`
alias and keeps the historical private name as a one-return wrapper. Both
production calls remain unchanged. The pass module does not import the lowerer.

Sixteen direct owner/wrapper comparisons cover ordinary static and dynamic
metadata, two matches, shared collision-safe and public-output shape clones,
public-input, variable, wrong-dtype, quantized, and missing-data rejection,
zero-match no-prune, short signatures, reverse topology, a public internal
alias, duplicate source producers, and missing output metadata. Deep-copied
executions produce identical statistics and complete normalized ModelIR state,
including buffers, quantization, options, provenance, topology, metadata,
lineage, and diagnostics.

Validation completed sequentially as follows:

- corrected checkpoint/module AST comparison: exact, 399 lines each;
- focused safety plus owner/wrapper contract: `50 passed in 0.55s`;
- focused contract, the five adjacent extracted bridge contracts, and ordered
  architecture suite: `650 passed in 18.26s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.59s`;
- targeted Ruff for the new module and focused test, Python compilation, and
  whitespace checks: passed;
- the central lowerer retains exactly five pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. The move is
mechanically identical to the corrected checkpoint. Public API, CLI,
artifacts, dependencies, corpus profiles, exclusions, operation tiers, both
ordered runtime calls, and TensorFlow isolation are unchanged. PR #952 remains
closed; no pull request was created, reopened, or updated.

At restart, inventory and characterize the next substantive raw source-order
owner before editing it: the 293-line
`_optimize_attention_gather_transpose_reshape_cleanup_chains`. It currently has
only architecture references and no focused public fixture. Build focused
synthetic ModelIR cases for positive and multiple matches, Gather/Transpose/
Reshape shape and axis constants, metadata, quantization, topology, public
boundaries, pruning, fixed point, statistics, and every production-call
boundary. Record unsafe behavior as strict xfails before correction. Keep
validation minimal and strictly sequential, commit and push coherent
checkpoints, and do not create a pull request.

## Attention Gather cleanup characterization: completed state

The 293-line raw
`_optimize_attention_gather_transpose_reshape_cleanup_chains` owner and both
production calls remain unchanged. It had no focused public fixture; the new
`test_flatbuffer_direct_attention_gather_cleanup_layout.py` now owns the
synthetic ModelIR contract.

Thirty-three green cases freeze Pattern A and Pattern B, exact NumPy equality,
negative-axis normalization, two matches of each pattern and fixed point,
collision-safe shared-permutation naming, the supported public Pattern-A
Reshape output, existing public-intermediate/fan-out/axis/shape/constant
rejections, statistics, the current 293-line/two-While owner shape, and both
production calls.

Forty-six reproduced safety problems are strict xfails:

- one zero-match case prunes an unrelated tensor;
- thirty unsafe public-input, variable, TensorIR/buffer dtype, or quantized
  index/permutation/shape constant cases;
- two unsafe public-output/shared-clone permutation ownership/provenance cases;
- two multi-element zero-index cases;
- one dynamic-signature and one retained per-axis-QDIM case in Pattern A;
- one inconsistent intermediate-shape and one quantization-bypass case in
  Pattern B;
- three incomplete-metadata and four invalid-topology cases.

Validation completed sequentially as follows:

- focused characterization: `33 passed, 46 xfailed in 1.05s`;
- focused characterization, the six adjacent extracted bridge/collapse
  contracts, and ordered architecture suite:
  `683 passed, 46 xfailed in 19.42s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.51s`;
- focused-test Ruff/format checks, Python compilation, and whitespace checks:
  passed.

No production source or real-model conversion changed or ran. Public API, CLI,
artifacts, dependencies, corpus profiles, exclusions, operation tiers, both
ordered runtime calls, and TensorFlow isolation are unchanged. PR #952 remains
closed; no pull request was created, reopened, or updated.

At restart, correct the raw owner transactionally before extracting it. Build
one `ModelIRGraphIndex` and complete Pattern-A/Pattern-B plans proving unique
ordered topology, exact private intermediates, scalar zero indices, complete
shape/signature/dtype/layout/quantization compatibility, and public boundaries.
Give all match constants an immutable unquantized INT32 ownership/type
contract; clone changed shared/public permutation values deterministically with
full provenance. Preserve dynamic signatures, remap Pattern-A QDIM, require
exact Pattern-B metadata equivalence, and precompute every setter, removal, and
prune decision before mutation. Turn all 46 strict xfails green while
preserving valid behavior, statistics, fixed point, and both production calls.
Validate sequentially, commit and push, and do not create a pull request.

## Attention Gather cleanup transactional correction: completed state

All forty-six former strict xfails are green. The corrected raw owner is 740
lines and constructs one `ModelIRGraphIndex`; mixed two-Pattern-A/two-Pattern-B
fixed-point execution refreshes it exactly once. Indexed replacements and
batched/single removals replace every repeated full consumer scan.

Before mutation, each candidate proves unique producers, strict graph order,
exact private intermediate consumers, complete positive physical shapes and
full compatible signatures, data-preserving dtypes, scalar zero-index Gather
semantics, and valid public boundaries. Pattern B proves exact singleton-axis
shape reduction and source/Reshape layout plus quantization equivalence before
bypassing the chain. Pattern A proves the complete rank-lift algebra, target
Reshape, permutation options, and retained output metadata. Its dynamic axes
remain dynamic, NCW/NWC layout annotations rank-lift to NCHW/NHWC, and retained
per-axis QDIM advances by one.

Index, permutation, and target-shape constants now require immutable,
unquantized INT32 TensorIR and NumPy buffers with exact values and shapes.
Public inputs, variables, runtime producers, wrong TensorIR/buffer dtypes, and
quantized values reject. Scalar `[]` and normalized `[1]` index representations
remain supported, while multi-element zeros reject. A private permutation
updates once; unrelated consumers and public outputs receive deterministic
collision-safe clones preserving layout and ONNX provenance. All actions are
planned before commit, and zero-match execution no longer prunes.

Validation completed sequentially as follows:

- corrected focused contract: `81 passed in 0.59s`;
- corrected focused contract plus ordered architecture suite:
  `329 passed in 18.32s`;
- corrected focused contract, six adjacent extracted bridge/collapse
  contracts, and ordered architecture suite: `731 passed in 18.68s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.42s`;
- focused-test Ruff, Python compilation, and whitespace checks: passed;
- the central lowerer retains exactly six pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. The rewrite
is limited to fully proven candidates. Public API, CLI, artifacts, dependencies,
corpus profiles, exclusions, operation tiers, both ordered runtime calls, and
TensorFlow isolation are unchanged. The 740-line count is descriptive only;
2,000 remains the ONNX operation-count tier threshold. PR #952 remains closed;
no pull request was created, reopened, or updated.

At restart, mechanically extract the corrected 740-line
`_optimize_attention_gather_transpose_reshape_cleanup_chains` owner into a
focused pass module. Keep the historical lowerer private name as a one-return
wrapper and preserve both production calls. Prove corrected checkpoint/module
AST identity and direct owner/wrapper equality for both patterns, dynamic and
scalar metadata, multiple matches, constant ownership/cloning, quantization,
pruning, rejection, topology, and atomicity cases. Validate sequentially,
commit and push, and do not create a pull request.

## Attention Gather cleanup ownership extraction: completed state

The corrected owner now resides in
`onnx2tf/tflite_builder/passes/attention_gather_cleanup_layout.py`. Its function
and the corrected raw owner at checkpoint `a48ee607` are each 740 lines and
have identical ASTs. The central lowerer imports it under the private
`_optimize_attention_gather_transpose_reshape_cleanup_chains_pass` alias and
keeps the historical private name as a one-return wrapper. Both production
calls remain unchanged. The pass module does not import the lowerer.

Sixteen direct owner/wrapper comparisons cover ordinary Pattern A and Pattern
B, two matches of each, negative axes, scalar indices, dynamic signatures,
shared and public permutation cloning, variable-index rejection, per-axis
QDIM, zero-match no-prune, Pattern-B quantization mismatch, missing metadata,
reverse topology, a public internal alias, and duplicate source producers.
Deep-copied executions produce identical statistics and complete normalized
ModelIR state, including buffers, quantization, options, provenance, topology,
metadata, lineage, and diagnostics.

Validation completed sequentially as follows:

- corrected checkpoint/module AST comparison: exact, 740 lines each;
- focused safety plus owner/wrapper contract: `97 passed in 0.63s`;
- focused contract plus ordered architecture suite: `345 passed in 17.51s`;
- focused contract, six adjacent extracted bridge/collapse contracts, and
  ordered architecture suite: `747 passed in 18.22s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.61s`;
- targeted Ruff for the new module and focused test, Python compilation, and
  whitespace checks: passed;
- the central lowerer retains exactly six pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. The move is
mechanically identical to the corrected checkpoint. Public API, CLI, artifacts,
dependencies, corpus profiles, exclusions, operation tiers, both ordered
runtime calls, and TensorFlow isolation are unchanged. PR #952 remains closed;
no pull request was created, reopened, or updated.

At restart, inventory and characterize the next substantive raw source-order
owner before editing it: the 190-line
`_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains`. It
currently has only architecture references and no focused public fixture.
Build focused synthetic ModelIR cases for valid and multiple matches, dynamic
shape/signature metadata, Reshape constants/options, BatchMatMul flags,
quantization, topology, public boundaries, pruning, fixed point, statistics,
and both production-call boundaries. Record unsafe behavior as strict xfails
before correction. Keep validation minimal and strictly sequential, commit and
push coherent checkpoints, and do not create a pull request.

## Attention pre-projection rank-lift characterization: completed state

The 190-line raw
`_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains` owner and
both production calls remain unchanged. It had no focused public fixture; the
new `test_flatbuffer_direct_attention_preproj_ranklift_layout.py` now owns the
synthetic ModelIR contract.

Twenty-one green cases freeze one- and two-branch rank lift, ADD/SUB/MUL/DIV
including reversed SUB/DIV inputs, exact NumPy equality, fixed point, twelve
existing public-boundary/shape/fan-out/operator rejections, statistics, the
current 190-line/one-While owner shape, and both production calls.

Twenty-seven reproduced safety problems are strict xfails:

- one zero-match case prunes an unrelated tensor;
- ten unsafe public-input, variable, TensorIR/buffer dtype, or quantized
  leading/tail shape-constant cases;
- one dynamic-signature and one per-axis-QDIM case;
- one rank-sensitive bias broadcast and two BatchMatMul flag cases;
- one nonpositive tail-shape case;
- five incomplete metadata/dtype cases and five invalid-topology cases.

Validation completed sequentially as follows:

- focused characterization: `21 passed, 27 xfailed in 0.77s`;
- focused characterization, attention Gather cleanup, six adjacent extracted
  bridge/collapse contracts, and ordered architecture suite:
  `768 passed, 27 xfailed in 18.92s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.66s`;
- focused-test Ruff/format checks, Python compilation, and whitespace checks:
  passed.

No production source or real-model conversion changed or ran. Public API, CLI,
artifacts, dependencies, corpus profiles, exclusions, operation tiers, both
ordered runtime calls, and TensorFlow isolation are unchanged. PR #952 remains
closed; no pull request was created, reopened, or updated.

At restart, correct the raw owner transactionally before extracting it. Use one
`ModelIRGraphIndex` to prove the complete leading-Reshape and all-branch
topology, metadata, untransposed BatchMatMul flags, positive tail shape, and
old/new binary broadcast equivalence. Give leading and tail shape inputs an
immutable unquantized INT32 ownership/type contract. Preserve dynamic
signatures, shift retained per-axis QDIM with the rank lift, precompute every
branch setter/metadata action and the single removal, and leave zero-match
execution untouched. Turn all 27 strict xfails green while preserving exact
valid outputs, statistics, fixed point, and both production calls. Validate
sequentially, commit and push, and do not create a pull request.

## Attention pre-projection rank-lift transactional correction: completed state

All twenty-seven former strict xfails are green. The corrected raw owner is 563
lines and constructs one `ModelIRGraphIndex`; the two-branch fixed-point rewrite
refreshes it exactly once. Indexed BatchMatMul input setters and one indexed
leading-Reshape removal replace every repeated full consumer scan.

Before mutation, the leading source/Reshape and every projection branch prove
unique producers, strict operator order, exact private intermediate consumers,
complete positive shapes and compatible signatures, data-preserving dtypes,
valid boundaries, and concrete tail outputs. All BatchMatMul transpose flags
must be false. The binary's other input must exist and broadcast to the exact
old and new result shapes, retaining scalar and `[K]` bias behavior while
rejecting rank-sensitive forms. Tail dimensions are all positive and multiply
to the projection width.

Leading and tail shape constants now require immutable, unquantized INT32
TensorIR and NumPy buffers with exact values, shapes, signatures, ownership,
and no runtime producer. Dynamic sequence signatures, NCW/NWC layout metadata,
and per-axis QDIM rank-lift by one. The complete all-branch plan is built before
the first setter; invalid later branches cannot leave partial updates, and
zero-match execution no longer prunes.

Validation completed sequentially as follows:

- corrected focused contract: `50 passed in 0.58s`;
- corrected focused contract, attention Gather cleanup, six adjacent extracted
  bridge/collapse contracts, and ordered architecture suite:
  `797 passed in 19.42s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.52s`;
- focused-test Ruff, Python compilation, and whitespace checks: passed;
- the central lowerer retains exactly five pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. The rewrite
is restricted to fully proven candidates. Public API, CLI, artifacts,
dependencies, corpus profiles, exclusions, operation tiers, both ordered
runtime calls, and TensorFlow isolation are unchanged. The 563-line count is
descriptive only; 2,000 remains the ONNX operation-count tier threshold. PR
#952 remains closed; no pull request was created, reopened, or updated.

At restart, mechanically extract the corrected 563-line
`_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains` owner into
a focused pass module. Keep the historical lowerer private name as a one-return
wrapper and preserve both production calls. Prove corrected checkpoint/module
AST identity and direct owner/wrapper equality for valid multiple/binary/scalar/
dynamic/quantized cases plus constant, broadcast, flag, metadata, topology, and
atomic rejection cases. Validate sequentially, commit and push, and do not
create a pull request.

## Attention pre-projection rank-lift ownership extraction: completed state

The corrected owner now resides in
`onnx2tf/tflite_builder/passes/attention_preproj_ranklift_layout.py`. Its
function and the corrected raw owner at checkpoint `727c19c6` are each 563
lines and have identical ASTs. The central lowerer imports it under the private
`_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains_pass` alias
and keeps the historical private name as a one-return wrapper. Both production
calls remain unchanged. The pass module does not import the lowerer.

Sixteen direct owner/wrapper comparisons cover ordinary and two-branch
rewrites, reversed SUB, scalar bias, dynamic signatures, per-axis QDIM,
variable leading and public-input tail shape rejection, zero-match no-prune,
rank-sensitive bias, `adjX`, missing bias/output metadata, reverse topology, a
public internal alias, and duplicate source producers. Deep-copied executions
produce identical statistics and complete normalized ModelIR state, including
buffers, quantization, options, provenance, topology, metadata, lineage, and
diagnostics.

Validation completed sequentially as follows:

- corrected checkpoint/module AST comparison: exact, 563 lines each;
- focused safety plus owner/wrapper contract: `66 passed in 0.59s`;
- focused contract, attention Gather cleanup, six adjacent extracted bridge/
  collapse contracts, and ordered architecture suite:
  `813 passed in 18.20s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.28s`;
- targeted Ruff for the new module and focused test, Python compilation, and
  whitespace checks: passed.

No real-model conversion or broad direct-suite repeat was added. The move is
mechanically identical to the corrected checkpoint. Public API, CLI, artifacts,
dependencies, corpus profiles, exclusions, operation tiers, both ordered
runtime calls, and TensorFlow isolation are unchanged. PR #952 remains closed;
no pull request was created, reopened, or updated.

At restart, inventory the next substantive unextracted raw owner before editing
it: the 209-line
`_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains`. Unlike the two
attention owners, it already has the focused
`test_flatbuffer_direct_elementwise_roundtrip_nchw_nhwc.py` fixture. First
measure the existing positive/rejection/topology/metadata/quantization/pruning/
fixed-point and production-call coverage, move or extend only missing focused
contracts, and record unsafe behavior as strict xfails before correction. Keep
validation minimal and strictly sequential, commit and push coherent
checkpoints, and do not create a pull request.

## NCHW-to-NHWC elementwise roundtrip characterization: completed state

The 209-line raw
`_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains` owner and its one
ordered production call remain unchanged. Its existing focused fixture had
only one valid rewrite and two boundary rejections. The expanded fixture now
owns the complete synthetic ModelIR characterization.

Thirty-two green cases freeze structural and NumPy-exact rewrites, all fifteen
allowed unary/binary op types, two independent matches and fixed point,
dynamic signatures, provenance/options/version retention, nine existing
permutation/operator/public-boundary/fan-out/runtime-input rejections, the
existing duplicate-producer rejection, statistics, the current 209-line/two-
While owner shape, and the single production call.

Twenty-eight reproduced safety problems are strict xfails:

- one zero-match invocation prunes an unrelated tensor;
- twelve public-input, variable, TensorIR/buffer dtype, quantized, or
  runtime-produced pre/post permutation tensors are accepted as compile-time
  constants;
- three local channel/full-rank NHWC constants and one shared NHWC constant
  are not remapped or cloned for the rewritten NCHW subgraph;
- one layout-metadata and one per-axis-QDIM case retain NHWC coordinates after
  the shape has changed to NCHW;
- nine incomplete tensor/dtype/shape/signature/public-boundary/topology/output
  candidates mutate instead of being rejected transactionally.

Validation completed sequentially as follows:

- focused characterization: `32 passed, 28 xfailed in 0.90s`;
- focused characterization, the two preceding attention contracts, six
  adjacent extracted bridge/collapse contracts, and ordered architecture
  suite: `845 passed, 28 xfailed in 19.54s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.48s`;
- focused-test Ruff formatting/lint checks: passed.

No production source or real-model conversion changed or ran. Public API, CLI,
artifacts, dependencies, corpus profiles, exclusions, operation tiers, the
ordered runtime call, and TensorFlow isolation are unchanged. The 209-line
count is descriptive only; 2,000 remains the ONNX operation-count tier
threshold. PR #952 remains closed; no pull request was created, reopened, or
updated.

At restart, correct the raw owner transactionally before extracting it. Build
one `ModelIRGraphIndex` and a complete candidate plan before mutation. Prove
unique ordered producers/consumers, exact arity, private boundaries, complete
rank-four shape/signature/dtype/layout metadata, transpose equivalence, and
elementwise broadcast compatibility. Give both transpose permutations an
immutable unquantized INT32 ownership/type contract. Plan local constant
rotation and shared/public constant cloning without changing nonlocal users;
remap dynamic metadata, logical/physical layout, and per-axis QDIM from NHWC
to NCHW. Apply indexed setters/removals only after the full preflight and leave
zero-match execution untouched. Turn all 28 strict xfails green while
preserving NumPy-exact valid outputs, all allowed ops, statistics, fixed point,
and the production call. Validate sequentially, commit and push, and do not
create a pull request.

## NCHW-to-NHWC elementwise roundtrip correction: completed state

All twenty-eight former strict xfails are green. The corrected raw owner is 705
lines and constructs one `ModelIRGraphIndex`; it performs no legacy producer or
consumer-map rebuilds. Indexed input/output setters and one batched indexed
removal keep the same index current across multiple matches.

Before mutation, each candidate proves exact unary/binary and Transpose arity,
unique ordered producers, private intermediate consumers, complete positive
rank-four physical shapes, compatible dynamic signatures, matching dtypes,
valid logical/physical layout annotations, and exact old/new elementwise
broadcast results. Both pre and post permutation inputs now require immutable,
private, unquantized INT32 TensorIR and NumPy buffers with exact values, shape,
signature, and no runtime producer.

Non-scalar NHWC constants are rank-expanded and transposed into NCHW only after
their metadata and original broadcast have been proven. Private constants are
updated locally; constants with public or nonlocal consumers are cloned with
their dtype, quantization, layout, and ONNX provenance preserved. Dynamic
output signatures, logical/physical layout, and per-axis QDIM are remapped from
NHWC to NCHW for intermediate, canonical output, and transformed constant
tensors. A zero-match invocation is now a complete no-op and does not prune.

Validation completed sequentially as follows:

- corrected focused contract: `64 passed in 0.62s`;
- corrected focused contract, the two preceding attention contracts, six
  adjacent extracted bridge/collapse contracts, and ordered architecture
  suite: `877 passed in 19.11s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.52s`;
- focused-test Ruff and Python compilation/whitespace checks: passed;
- the central lowerer retains exactly two pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. Public API,
CLI, artifacts, dependencies, corpus profiles, exclusions, operation tiers,
the ordered production call, and TensorFlow isolation are unchanged. The
705-line count is descriptive only; 2,000 remains the ONNX operation-count
tier threshold. PR #952 remains closed; no pull request was created, reopened,
or updated.

At restart, mechanically extract the corrected 705-line
`_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains` owner into a
focused pass module. Keep the historical lowerer private name as a one-return
wrapper and preserve the ordered production call. Prove corrected checkpoint/
module AST identity and direct owner/wrapper equality for valid, multiple,
dynamic, constant-remap/clone, layout/QDIM, permutation-ownership, missing-
metadata, public-boundary, reverse-topology, duplicate-producer, and zero-match
cases. Validate sequentially, commit and push, and do not create a pull request.

## NCHW-to-NHWC elementwise roundtrip ownership extraction: completed state

The corrected owner now resides in
`onnx2tf/tflite_builder/passes/elementwise_roundtrip_nchw_nhwc_layout.py`.
Its function and the corrected raw owner at checkpoint `79862309` are each 705
lines and have identical ASTs. The central lowerer imports it under the private
`_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains_pass` alias, keeps
the historical private name as a one-return wrapper, and preserves the single
ordered production call. The pass module does not import the lowerer.

Sixteen direct owner/wrapper comparisons cover ordinary and two-chain rewrites,
dynamic signatures, local and shared constant remapping, output and constant
per-axis QDIM, variable/public permutation rejection, zero-match no-prune,
missing output metadata, a public internal alias, reverse topology, duplicate
root/pre producers, and variable feature constants. Deep-copied executions
produce identical statistics and complete normalized ModelIR state, including
buffers, quantization, layouts, provenance, topology, metadata, lineage, and
diagnostics. A runtime contract also proves that two matches construct and
refresh exactly one `ModelIRGraphIndex`.

Validation completed sequentially as follows:

- corrected checkpoint/module AST comparison: exact, 705 lines each;
- focused safety plus owner/wrapper contract: `81 passed in 0.62s`;
- focused contract, the two preceding attention contracts, six adjacent
  extracted bridge/collapse contracts, and ordered architecture suite:
  `894 passed in 18.16s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.37s`;
- targeted Ruff for the new module and focused test, Python compilation, and
  whitespace checks: passed;
- the central lowerer retains exactly two pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. The move is
mechanically identical to the corrected checkpoint. Public API, CLI, artifacts,
dependencies, corpus profiles, exclusions, operation tiers, ordered runtime
behavior, and TensorFlow isolation are unchanged. PR #952 remains closed; no
pull request was created, reopened, or updated.

No substantive top-level raw owner remains outside `lower_onnx_to_ir`; the
remaining central body is now mostly ordered orchestration and small
compatibility wrappers. At restart, inventory the nested layout orchestration
before editing it, beginning with the 66-line
`_run_layout_recovery_prefix_pass_sequence` and adjacent 51-line
`_run_layout_reshape_attention_recovery_prefix`. Freeze their exact order,
repetition, session/layout-state/diagnostic dependencies, and call boundaries;
then design the smallest explicit phase-runner contract that can move them out
of the 2,634-line lowerer without changing pass order or behavior. Record unsafe
or implicit coupling before correction, validate sequentially, commit and push,
and do not create a pull request.

## Nested layout-recovery orchestration characterization: completed state

The 66-line `_run_layout_recovery_prefix_pass_sequence`, the adjacent 51-line
`_run_layout_reshape_attention_recovery_prefix`, and all production call sites
remain unchanged. Existing architecture coverage already fixed the nineteen-
call and fifteen-call order, the three attention-prefix repetitions and their
immediate boundaries, the nested layout-prefix invocation, and the direct final
layout-prefix recovery call.

The new focused `test_flatbuffer_direct_layout_recovery_orchestration.py`
fixture fills the remaining extraction-contract gaps without extending the
large architecture test. It freezes every positional and keyword argument,
including which calls receive only `model_ir`, which receive
`session.layout_state`, which also receive `session.diagnostics`, and the final
`include_unary_passthrough=True`. Both helpers are proven to have no parameters,
branch/loop/try/context-manager control flow, or local
`ModelIRPassStateScope`; after excluding their call targets, their only captured
data names are `model_ir` and `session`.

Validation completed sequentially as follows:

- focused orchestration contract: `4 passed in 0.22s`;
- focused orchestration plus ordered architecture suite:
  `252 passed in 16.81s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.23s`;
- focused-test Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, pass order, real-model conversion, or broad direct suite
changed or ran. Public API, CLI, artifacts, dependencies, corpus profiles,
exclusions, operation tiers, and TensorFlow isolation are unchanged. PR #952
remains closed; no pull request was created, reopened, or updated.

At restart, map each of the thirty-four call targets to its extracted module
owner before moving either helper. Three dependencies are still nested lowerer
clusters (`_run_boundary_batchmatmul_unary_layout_pass_cluster`,
`_run_channel_shuffle_gather_layout_pass_cluster`, and the layout-prefix call
from the attention prefix); preserve those as explicit injected callbacks while
direct module owners receive an explicit `model_ir`/layout-state/diagnostics
context. Design a declarative ordered phase specification with stable pass IDs
and no lambda closure over mutable lowerer locals, then add direct old/new
execution-order and argument-equivalence tests before changing production call
sites. Keep the change mechanical, validate sequentially, commit and push, and
do not create a pull request.

## Explicit layout-recovery orchestration: completed state

The two characterized nested sequences now delegate to
`passes/layout_recovery_orchestration.py`. A frozen `LayoutRecoveryContext`
owns the explicit ModelIR, layout-state, diagnostics, and three unavoidable
lowerer-local callback dependencies. The phase module imports the other thirty
pass owners directly; it does not import the central lowerer and uses no lambda
closure over mutable lowerer locals.

The module exposes nineteen stable layout-recovery IDs and fifteen stable
layout/reshape/attention-recovery IDs. Immutable `RecoveryInvocation` values
preserve every positional and keyword argument and execute in the characterized
order. The attention sequence invokes the complete layout sequence as its
first explicit step. Runtime assertions reject accidental drift between the
declared IDs and constructed invocation order.

The historical nested helper names and all outer call sites remain in place.
Their bodies shrink from 66 to 2 lines and from 51 to 4 lines respectively,
each capturing only the single explicit context. The three injected callbacks
remain lowerer-local because they compose nested pass clusters; all other work
is owned by the new phase module. Existing pass return values continue to be
ignored exactly as before.

Focused tests prove the stable ID lists, all thirty-four argument contracts,
wrapper/context wiring, and identical flattened execution order under
instrumented callbacks. The ordered architecture suite now treats stable phase
IDs as first-class execution boundaries while retaining module-owner,
compatibility-wrapper, repetition, and total-call-count checks. Four adjacent
owner fixtures were adjusted to count one moved stable phase boundary in
addition to the remaining direct lowerer calls; their owner and call-argument
checks remain intact.

Validation completed sequentially as follows:

- focused orchestration contract: `6 passed in 0.58s`;
- central lowerer synthetic smoke: `32 passed in 0.59s`;
- ordered architecture suite: `248 passed in 18.11s`;
- focused orchestration, nine adjacent extracted pass contracts, and ordered
  architecture suite: `900 passed in 18.10s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.49s`.

No real-model conversion or broad direct-suite repeat was added. This is a
mechanical orchestration extraction: public API, CLI, artifacts, dependencies,
corpus profiles, exclusions, operation tiers, pass order, repetition, and
TensorFlow isolation are unchanged. PR #952 remains closed; no pull request
was created, reopened, or updated.

At restart, inventory and characterize the next self-contained orchestration
cluster before moving it. Prefer a small focused fixture and stable phase IDs,
retain lowerer-local composites as explicitly injected callbacks, and prove
old/new order and argument equality before changing production wiring. Keep
real-model conversions minimal and sequential, commit and push coherent
checkpoints, and do not create a pull request.

## Attention-recovery orchestration characterization: completed state

The next extraction boundary is the adjacent 14-line
`_run_preadd_mean_attention_recovery_sequence` and 27-line
`_run_attention_gate_qdq_recovery_sequence`. They remain unchanged in
production. The first has seven ordered steps and two zero-argument production
invocations. The second has ten ordered steps and three zero-argument
invocations, one of which is nested between mean-attention and duplicate
quantized-PReLU clusters in the layout/attention/quantized suffix.

The new focused
`test_flatbuffer_direct_attention_recovery_orchestration.py` fixture freezes
all seventeen call slots and every positional and keyword argument. It proves
which calls receive only `model_ir`, which also receive
`session.layout_state`, and that trailing-output cleanup alone additionally
receives `session.diagnostics`. Both helpers have no parameters, local
`ModelIRPassStateScope`, or branch/loop/try/context-manager control flow; after
excluding call targets, their only captured data is `model_ir` and `session`.
The fixture also fixes all zero-argument outer invocations and the nested
quantized-suffix boundary.

Validation completed sequentially as follows:

- focused attention-recovery characterization: `4 passed in 0.19s`;
- both orchestration fixtures plus ordered architecture suite:
  `258 passed in 17.08s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.38s`;
- focused-test Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, pass order, real-model conversion, or broad direct suite
changed or ran. Public API, CLI, artifacts, dependencies, corpus profiles,
exclusions, operation tiers, and TensorFlow isolation are unchanged. PR #952
remains closed; no pull request was created, reopened, or updated.

At restart, map the seventeen call slots to their extracted module owners.
Preserve the three lowerer-local composite dependencies
(`_run_mean_attention_layout_pass_cluster`, `_run_gate_layout_pass_cluster`,
and `_run_transpose_unary_fanout_layout_pass_cluster`) as explicit callbacks.
Give the remaining owners an immutable explicit ModelIR/layout/diagnostics
context, declare stable IDs for both ordered sequences, and prove old/new
flattened order and argument equality before switching the two historical
helpers. Keep validation sequential, commit and push coherent checkpoints, and
do not create a pull request.

## Explicit attention-recovery orchestration: completed state

The characterized sequences now delegate to
`passes/attention_recovery_orchestration.py`. A frozen
`AttentionRecoveryContext` carries ModelIR, layout state, diagnostics, and the
three lowerer-local mean-attention, gate-layout, and transpose-unary-fanout
composite callbacks. The other fourteen targets are imported from their
existing pass module owners; the orchestration module does not import the
lowerer.

The module declares seven stable preadd/mean/attention IDs and ten stable
attention/gate/QDQ IDs. Both builders emit immutable invocations with the
characterized positional and keyword arguments. The shared
`passes/recovery_orchestration.py` primitive now owns immutable invocation
execution and verifies the complete stable-ID sequence before running any
callback. The preceding layout runner uses the same primitive, eliminating its
duplicate execution/drift-check logic without changing its specifications.

The historical lowerer helper names and every outer invocation remain in
place. Each helper now has a two-line body that captures only the explicit
attention context. Existing direct call-count architecture tests now add the
stable phase multiplicity; this is intentionally a sequence count rather than
set membership because pre-add appears in both the earlier layout prefix and
the new preadd/mean/attention phase.

Focused tests prove context construction, wrapper wiring, all seventeen IDs
and argument contracts, instrumented execution order, zero-argument outer
boundaries, lowerer-import isolation, and rejection of ID drift before the
first callback. Three pre-existing quantized expected-builder tests were also
corrected to import graph mutation helpers from their real owner,
`core.model_ir_utils`, instead of relying on private lowerer re-exports that
were already absent at checkpoint `9c210cf2`.

Validation completed sequentially as follows:

- both orchestration fixtures: `15 passed in 0.75s`;
- central lowerer synthetic smoke: `32 passed in 0.59s`;
- ordered architecture suite: `248 passed in 17.61s`;
- three corrected quantized expected-builder fixtures: `29 passed in 0.59s`;
- attention target-focused plus architecture suite: `433 passed in 17.39s`;
- both orchestration fixtures, adjacent extracted pass contracts, and ordered
  architecture suite: `909 passed in 18.70s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.44s`;
- targeted Ruff, Python compilation, and whitespace checks: passed;
- the central lowerer retains exactly its two pre-existing Ruff findings.

No real-model conversion or broad direct suite was added. This is a mechanical
orchestration extraction: public API, CLI, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, pass order, invocation multiplicity,
and TensorFlow isolation are unchanged. PR #952 remains closed; no pull
request was created, reopened, or updated.

At restart, inventory the next adjacent recovery sequence rather than widening
this checkpoint. The one-step safe-binary wrapper and the six-step quantized
activation/binary sequence form a likely next nested boundary, but first freeze
their exact shared invocation multiplicity and module-owner/callback routing.
Keep validation sequential, commit and push coherent checkpoints, and do not
create a pull request.

## Quantized activation/binary orchestration characterization: completed state

The next extraction boundary is the five-line one-step
`_run_safe_binary_bridge_recovery_sequence` and the nineteen-line six-step
`_run_quantized_activation_binary_bridge_recovery_sequence`. Production is
unchanged. Safe-binary has three zero-argument outer invocations, including the
nested final step of quantized activation/binary; the latter has two
zero-argument outer invocations.

The focused `test_flatbuffer_direct_quantized_recovery_orchestration.py`
fixture freezes both helpers as parameterless straight-line closures over only
`model_ir` and `session`, with no local pass-state scope or control flow. It
fixes the safe-binary ModelIR/layout arguments; the hard-sigmoid, max-pool,
softmax, and logistic ModelIR/layout calls; the model-only softmax
canonicalization; the nested safe-binary final step; and every zero-argument
outer boundary.

Validation completed sequentially as follows:

- focused quantized-recovery characterization: `4 passed in 0.21s`;
- focused characterization plus ordered architecture suite:
  `252 passed in 16.85s`;
- TensorFlow-import-blocked optional-boundary suite: `11 passed in 9.51s`;
- focused-test Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, pass order, real-model conversion, or broad direct suite
changed or ran. Public API, CLI, artifacts, dependencies, corpus profiles,
exclusions, operation tiers, and TensorFlow isolation are unchanged. PR #952
remains closed; no pull request was created, reopened, or updated.

At restart, map the five quantized targets and safe-binary target directly to
their existing module owners. This boundary requires no lowerer-local composite
callback: an immutable ModelIR/layout context and a nested stable-ID runner are
sufficient. Preserve the historical helper names, both outer invocation
counts, and the three total safe-binary invocations. Prove instrumented order
and arguments before switching production wiring, validate sequentially,
commit and push, and do not create a pull request.

## Explicit quantized activation/binary recovery orchestration: completed state

The characterized nested boundary now delegates to
`passes/quantized_recovery_orchestration.py`. A frozen
`QuantizedRecoveryContext` carries the shared ModelIR and `LayoutState`; this
boundary needs no injected lowerer-local callback. The new module imports the
safe-binary, quantized hard-sigmoid, max-pool, softmax, logistic, and softmax-
transpose canonicalization owners directly and does not import the central
lowerer.

`SAFE_BINARY_RECOVERY_PASS_IDS` declares the one-step safe-binary phase, while
`QUANTIZED_ACTIVATION_BINARY_PASS_IDS` declares the six-step quantized phase.
The latter invokes the complete safe-binary runner as its final stable step.
Both use the shared immutable `RecoveryInvocation` executor, which verifies
the exact ID order before executing a callback. The characterized positional
and keyword arguments, nested structure, and ignored return values are
unchanged.

The historical lowerer helper names and every zero-argument outer call remain
in place. Their bodies shrink from five to two lines and from nineteen to four
lines and capture only the explicit context. Architecture accounting includes
the ordered stable-ID multiplicity, preserving three total safe-binary helper
invocations and two total quantized helper invocations even though one nested
call is no longer visible in the lowerer AST. The obsolete lowerer import of
the safe-binary owner was removed.

Focused contracts verify both stable ID sequences, every argument, explicit
context and wrapper wiring, instrumented order, outer invocation counts, and
lowerer-import isolation. The softmax canonicalization architecture fixture
now checks its moved boundary through the stable quantized sequence while
retaining the remaining direct lowerer boundary. The indexed logistic-gate
expected builder now imports graph mutation helpers from their actual owner,
`core.model_ir_utils`, rather than an already-absent private lowerer re-export.

Sequential validation completed as follows:

- focused quantized orchestration: `8 passed in 0.57s`;
- focused orchestration plus ordered architecture: `256 passed in 17.91s`;
- affected logistic/softmax/orchestration/architecture set:
  `324 passed in 17.41s`;
- related safe-binary and quantized pass-family set:
  `703 passed in 17.36s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 9.92s` (`32` plus `11`);
- targeted Ruff, Python compilation, and whitespace checks: passed; the
  central lowerer retains exactly its two pre-existing Ruff findings.

One diagnostic broad direct-suite run before the two fixture-owner updates
reported `1436 passed, 8 failed`. Two failures were the stale logistic-gate and
softmax AST contracts corrected above. Four require the intentionally absent
TensorFlow optional extra:
`test_tflite_backend_matrix_add`,
`test_tflite_backend_matrix_hardswish_rewrite_on_off`,
`test_tf_converter_resize_cubic_avoids_flex_resize_bicubic`, and
`test_tf_converter_resize_cubic_honors_cubic_coeff_a`. One is the documented
incompatible local Torch binary failure in
`test_flatbuffer_direct_group_norm_alias_builtin_conversion`. The remaining
unrelated existing failure is
`test_flatbuffer_direct_sinet_concat_resize_affine_transpose_chain_optimized`,
whose direct pass statistic is zero instead of one; this extraction neither
calls nor changes that owner. No real-model conversion was added.

Public APIs, CLI behavior, artifacts, dependencies, corpus profiles,
exclusions, operation tiers, runtime pass order, invocation multiplicity, and
TensorFlow isolation are unchanged. PR #952 remains closed, no branch PR is
open, and no pull request was created, reopened, or updated.

At restart, inventory and characterize the adjacent five-step
`_run_qlinear_mean_concat_recovery_sequence` before moving production code.
Freeze every argument and all outer repetition/boundary contracts first, then
introduce another explicit stable-ID phase only if direct module ownership can
preserve the exact flattened sequence. Keep verification sequential and
minimal, commit and push a coherent checkpoint, and do not create a pull
request.

## QLinear mean/concat recovery orchestration characterization: completed state

The adjacent six-line `_run_qlinear_mean_concat_recovery_sequence` remains
unchanged in production. It is a parameterless straight-line closure over only
`model_ir`, contains no pass-state scope or control flow, and performs five
model-only calls in this exact order: mean/HardSigmoid/MulAdd, QLinear SiLU
prefix, QLinear concat/conv propagation, concat pre-QDQ cleanup, and
mean/max-pool/concat/conv recovery. Its return values continue to be ignored.

All five targets already have extracted module owners in
`mean_hardsigmoid_muladd_layout.py`, `qlinear_silu_prefix_layout.py`,
`qlinear_concat_conv_compat.py`, `quantization_cleanup.py`, and
`mean_maxpool_concat_layout.py`. No lowerer-local composite callback, layout
state, diagnostics, or option flag is required for a future phase context.

The focused `test_flatbuffer_direct_qlinear_recovery_orchestration.py` freezes
the helper shape and capture set, all five positional/keyword argument
contracts, both zero-argument outer invocations, and their exact neighboring
boundaries. Validation completed sequentially as follows:

- focused characterization: `4 passed in 0.18s`;
- focused characterization plus ordered architecture:
  `252 passed in 16.98s`;
- TensorFlow-import-blocked optional boundary: `11 passed in 10.06s`;
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran in this checkpoint. Public APIs, CLI behavior, artifacts,
dependencies, corpus profiles, exclusions, operation tiers, and TensorFlow
isolation are unchanged. PR #952 remains closed, no branch PR is open, and no
pull request was created, reopened, or updated.

At restart, introduce a frozen ModelIR-only context and a five-ID immutable
invocation specification that imports these owners directly. Preserve the
historical helper name, its two outer calls and boundaries, and the exact
argument/order contract. Prove direct builder arguments and instrumented order
before changing the lowerer delegate, validate sequentially, commit and push,
and do not create a pull request.

## Explicit QLinear mean/concat recovery orchestration: completed state

The characterized sequence now delegates to
`passes/qlinear_recovery_orchestration.py`. A frozen
`QLinearRecoveryContext` contains only the shared ModelIR. The phase module
imports all five existing owners directly and does not import or capture the
central lowerer.

`QLINEAR_MEAN_CONCAT_PASS_IDS` declares the exact five-step order. Immutable
`RecoveryInvocation` values pass the same single ModelIR positional argument
with no keyword arguments, and the shared executor validates the complete ID
sequence before executing any callback. Existing pass return values remain
ignored exactly as before.

The historical lowerer helper and both zero-argument outer calls remain in
place. Its body shrinks from six to two lines and captures only the explicit
context. The two neighboring boundaries are unchanged. Architecture tests
account for moved direct calls through ordered stable-ID multiplicity while
retaining each module owner and compatibility-wrapper contract.

Focused tests prove all five IDs and argument contracts, context construction,
wrapper wiring, both outer boundaries and invocation counts, instrumented
execution order, and lowerer-import isolation. Sequential validation completed
as follows:

- focused qlinear orchestration: `7 passed in 0.60s`;
- focused orchestration plus ordered architecture: `255 passed in 18.05s`;
- five owner-focused suites plus orchestration and architecture:
  `394 passed in 17.69s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 10.81s` (`32` plus `11`);
- targeted Ruff, Python compilation, and whitespace checks: passed; the
  central lowerer retains exactly its two pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. Public APIs,
CLI behavior, artifacts, dependencies, corpus profiles, exclusions, operation
tiers, runtime pass order, invocation multiplicity, and TensorFlow isolation
are unchanged. PR #952 remains closed, no branch PR is open, and no pull
request was created, reopened, or updated.

At restart, inventory and characterize the adjacent
`_run_layout_attention_quantized_recovery_suffix` before changing production
code. It combines direct owners, already-extracted phase runners, lowerer-local
clusters, layout/diagnostic routing, and the
`include_duplicate_transpose` option, so freeze its exact flattened order,
arguments, repetition, and conditional boundary first. Keep verification
sequential and minimal, commit and push coherent checkpoints, and do not
create a pull request.

## Layout/attention/quantized recovery suffix characterization: completed state

The 41-line `_run_layout_attention_quantized_recovery_suffix` remains unchanged
in production. It is a straight-line helper with one required keyword-only
boolean, no branch/loop/try/context-manager control flow, and no local
`ModelIRPassStateScope`. Its data captures are ModelIR, session, and
`include_duplicate_transpose`.

The focused
`test_flatbuffer_direct_layout_attention_quantized_suffix_orchestration.py`
freezes all thirteen call slots and every positional and keyword argument. It
distinguishes model-only calls from layout-aware calls and the one
layout/diagnostics cleanup. It also proves that the option reaches only the
duplicate quantized-PReLU cluster as `include_transpose`, both outer calls pass
the global duplicate-transpose option by keyword, and both exact neighboring
boundaries remain fixed.

Ten slots resolve to existing module owners or already-extracted phase
runners. Three dependencies remain lowerer-local/nested boundaries:
`_run_mean_attention_layout_pass_cluster`,
`_run_attention_gate_qdq_recovery_sequence`, and
`_run_duplicate_quantized_prelu_pass_cluster`. A future context must therefore
carry ModelIR, layout state, diagnostics, and these three explicit callbacks;
the duplicate callback must retain its per-invocation boolean argument.

Validation completed sequentially as follows:

- focused suffix characterization: `4 passed in 0.16s`;
- focused characterization plus ordered architecture:
  `252 passed in 16.93s`;
- TensorFlow-import-blocked optional boundary: `11 passed in 9.58s`;
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran. Public APIs, CLI behavior, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, and TensorFlow isolation are unchanged.
PR #952 remains closed, no branch PR is open, and no pull request was created,
reopened, or updated.

At restart, map the ten direct slots to their current module callables and
introduce a frozen explicit context plus thirteen stable IDs. Keep the three
nested dependencies as injected callbacks, pass the duplicate-transpose flag
through the immutable invocation specification, and prove flattened
instrumented order and exact argument equivalence before switching the
historical helper. Validate sequentially, commit and push, and do not create a
pull request.

## Explicit layout/attention/quantized recovery suffix: completed state

The characterized suffix now delegates to
`passes/layout_attention_quantized_suffix_orchestration.py`. A frozen
`LayoutAttentionQuantizedSuffixContext` carries ModelIR, layout state,
diagnostics, and exactly three explicit lowerer-local callbacks: mean-
attention, attention-gate/QDQ recovery, and duplicate quantized-PReLU. The
other ten targets are imported directly from their existing pass modules, and
the new phase module does not import the central lowerer.

`LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS` declares the exact thirteen-step
order. Immutable invocations preserve all positional and keyword arguments.
The per-call `include_duplicate_transpose` value is forwarded without coercion
only to the duplicate callback's `include_transpose` keyword. Shared execution
validates the complete ID sequence before running any callback.

The lowerer constructs the context after all three callback dependencies are
defined. The historical keyword-only helper and both outer call sites remain;
its body shrinks from 41 to eight lines and delegates through the explicit
context. Both neighboring boundaries and the global option routing are
unchanged. Ordered stable-ID accounting preserves three total attention-
gate/QDQ helper calls, one duplicate quantized-PReLU cluster call, two total
quantized-reshape cleanups, and the existing total of 118 registered runner
calls.

Focused tests prove all thirteen IDs and argument contracts, callback identity,
option identity, context and wrapper wiring, outer invocation counts and
boundaries, instrumented execution order, and lowerer-import isolation.
Adjacent attention and softmax fixtures now follow the stable suffix IDs rather
than assuming every nested call remains visible in the lowerer AST.

Sequential validation completed as follows:

- focused suffix orchestration: `7 passed in 0.58s`;
- focused orchestration plus ordered architecture: `255 passed in 18.25s`;
- adjacent attention/softmax/orchestration/architecture set:
  `316 passed in 16.92s`;
- related thirteen-slot owner and phase-family set:
  `820 passed in 19.23s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 10.68s` (`32` plus `11`);
- targeted Ruff, Python compilation, and whitespace checks: passed; the
  central lowerer retains exactly its two pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. Public APIs,
CLI behavior, artifacts, dependencies, corpus profiles, exclusions, operation
tiers, runtime pass order, invocation multiplicity, and TensorFlow isolation
are unchanged. PR #952 remains closed, no branch PR is open, and no pull
request was created, reopened, or updated.

At restart, inventory and characterize the adjacent
`_run_terminal_slice_concat_layout_recovery_sequence` before changing
production code. It is substantially larger and mixes several lowerer-local
clusters with direct owners and layout/diagnostic routing, so freeze its exact
call slots, arguments, captures, repetition, and outer boundaries first. Keep
verification sequential and minimal, commit and push coherent checkpoints,
and do not create a pull request.

## Terminal slice/concat recovery orchestration characterization: completed state

The 37-line `_run_terminal_slice_concat_layout_recovery_sequence` remains
unchanged in production. It is a parameterless straight-line closure over
ModelIR and session, contains no control flow or local pass-state scope, and
has fourteen ordered call slots. Thirteen slots resolve to existing module
owners; only `_run_channel_slice_pad_mul_layout_pass_cluster` remains a
lowerer-local composite dependency.

The focused
`test_flatbuffer_direct_terminal_slice_concat_recovery_orchestration.py`
freezes every call and positional/keyword argument, including six layout-aware
owners and the final layout/diagnostics cleanup. It also proves both outer
invocations remain zero-argument top-level boundaries. Both share the same
preceding channel-slice/MulAdd call, but only the earlier predecessor receives
layout state, and their following calls remain the boundary QDQ/concat and
slice-pre/post owners respectively.

Validation completed sequentially as follows:

- focused terminal slice/concat characterization: `4 passed in 0.16s`;
- focused characterization plus ordered architecture:
  `252 passed in 17.20s`;
- TensorFlow-import-blocked optional boundary: `11 passed in 9.72s`;
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran. Public APIs, CLI behavior, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, and TensorFlow isolation are unchanged.
PR #952 remains closed, no branch PR is open, and no pull request was created,
reopened, or updated.

At restart, map the thirteen direct slots to their current module callables and
introduce a frozen ModelIR/layout/diagnostics context with the single channel-
slice/pad/mul callback plus fourteen stable IDs. Prove every builder argument
and instrumented order before switching the historical helper, preserve both
outer boundaries, validate sequentially, commit and push, and do not create a
pull request.

## Explicit terminal slice/concat recovery orchestration: completed state

The characterized sequence now delegates to
`passes/terminal_slice_concat_recovery_orchestration.py`. A frozen
`TerminalSliceConcatRecoveryContext` carries ModelIR, layout state,
diagnostics, and the one lowerer-local channel-slice/pad/mul composite
callback. The remaining thirteen targets are imported directly from their
existing pass modules; the new phase module does not import the central
lowerer.

`TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS` declares the exact fourteen-step
order. Immutable invocations preserve each model-only, layout-aware, and final
layout/diagnostics argument contract. Shared execution validates the complete
ID sequence before running the first callback.

The historical zero-argument helper and both top-level outer calls remain. Its
body shrinks from 37 to four lines and captures only the explicit context. Both
distinct outer neighbors and the earlier predecessor's layout keyword remain
unchanged. Multiplicity-aware architecture accounting preserves the prior
sanitizer, affine-post-add, channel-cluster, split-family, singleton-gate,
stride-slice bridge, layout-cleanup, and overall registered-runner totals.

Focused tests prove all fourteen IDs and arguments, callback identity, context
and wrapper wiring, both outer invocation contracts and neighbors,
instrumented order, and lowerer-import isolation. Four concat-family fixtures
now check the moved terminal adjacency through stable IDs while continuing to
check the separate terminal affine/concat/split sequence directly. The stride-
slice fixture similarly combines its moved stable occurrence with remaining
direct calls.

Sequential validation completed as follows:

- focused terminal slice/concat orchestration: `7 passed in 0.58s`;
- focused orchestration plus ordered architecture: `255 passed in 18.62s`;
- five adjacent concat/stride-slice fixtures plus orchestration and
  architecture: `607 passed in 18.43s`;
- related fourteen-slot owner and phase-family set:
  `867 passed in 18.64s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 10.76s` (`32` plus `11`);
- targeted Ruff, Python compilation, and whitespace checks: passed; the
  central lowerer retains exactly its two pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. Public APIs,
CLI behavior, artifacts, dependencies, corpus profiles, exclusions, operation
tiers, runtime pass order, invocation multiplicity, and TensorFlow isolation
are unchanged. PR #952 remains closed, no branch PR is open, and no pull
request was created, reopened, or updated.

At restart, inventory and characterize the neighboring
`_run_terminal_affine_concat_split_recovery_sequence` before changing
production code. Several owner fixtures already expose its relationship to the
new stable terminal slice/concat phase; freeze its complete call/argument list,
captures, repetition, and outer boundaries before choosing the next explicit
context. Validate sequentially, commit and push, and do not create a pull
request.

## Terminal affine/concat/split recovery characterization: completed state

The 30-line `_run_terminal_affine_concat_split_recovery_sequence` remains
unchanged in production. It is a parameterless straight-line closure over
ModelIR and session, has no control flow or local pass-state scope, and contains
eleven ordered calls. Every target already has an extracted module owner; no
lowerer-local callback, diagnostics, or conversion option is required.

The focused
`test_flatbuffer_direct_terminal_affine_concat_split_recovery_orchestration.py`
freezes all eleven positional/keyword argument contracts, distinguishing the
six layout-aware slots from five model-only slots. It also proves both outer
invocations remain zero-argument top-level boundaries and retains their two
distinct predecessor/follower pairs at the absolute end of the optimization
pipeline.

Validation completed sequentially as follows:

- focused terminal affine/concat/split characterization:
  `4 passed in 0.16s`;
- focused characterization plus ordered architecture:
  `252 passed in 17.63s`;
- TensorFlow-import-blocked optional boundary: `11 passed in 10.21s`;
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran. Public APIs, CLI behavior, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, and TensorFlow isolation are unchanged.
PR #952 remains closed, no branch PR is open, and no pull request was created,
reopened, or updated.

At restart, introduce a frozen ModelIR/layout context and eleven stable IDs
with direct imports of every existing owner. Prove builder argument equality
and instrumented order before switching the historical helper, preserve both
top-level outer boundaries, validate sequentially, commit and push, and do not
create a pull request.

## Explicit terminal affine/concat/split recovery orchestration: completed state

The characterized sequence now delegates to
`passes/terminal_affine_concat_split_recovery_orchestration.py`. A frozen
`TerminalAffineConcatSplitRecoveryContext` contains only ModelIR and layout
state. All eleven existing owners are imported directly; no lowerer callback,
diagnostics, option, or central-lowerer import is required.

`TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS` declares the exact eleven-step
order. Immutable invocations preserve the six layout-aware and five model-only
argument contracts, and the shared executor validates the complete sequence
before running any owner.

The historical helper and both zero-argument top-level invocations remain as
ordering boundaries. Its body shrinks from 30 to four lines and captures only
the context. Both distinct predecessor/follower pairs remain unchanged.
Several IDs intentionally also occur in the terminal slice/concat phase;
ordered multiplicity rather than unique-ID membership preserves every former
owner execution total. The only newly unique moved owner, affine-chain fold,
retains its total of three invocations.

Focused tests prove all stable IDs and arguments, explicit context/wrapper
wiring, both outer boundaries, instrumented order, and lowerer-import
isolation. Four concat-family fixtures now verify both terminal sequences via
their independent stable adjacency lists.

Sequential validation completed as follows:

- focused terminal affine/concat/split orchestration: `7 passed in 0.57s`;
- focused orchestration plus ordered architecture: `255 passed in 17.92s`;
- four adjacent concat fixtures plus orchestration and architecture:
  `517 passed in 17.55s`;
- related eleven-slot owner and phase-family set:
  `806 passed in 18.08s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 10.08s` (`32` plus `11`);
- targeted Ruff, Python compilation, and whitespace checks: passed; the
  central lowerer retains exactly its two pre-existing Ruff findings.

No real-model conversion or broad direct-suite repeat was added. Public APIs,
CLI behavior, artifacts, dependencies, corpus profiles, exclusions, operation
tiers, runtime pass order, invocation multiplicity, and TensorFlow isolation
are unchanged. PR #952 remains closed, no branch PR is open, and no pull
request was created, reopened, or updated.

At restart, inventory and characterize the adjacent
`_run_sinet_preadd_resize_recovery_sequence` before changing production code.
Freeze its exact call/argument list, lowerer-local dependencies, captures,
repetition, and outer boundaries before defining another phase context.
Validate sequentially, commit and push, and do not create a pull request.

## SINet pre-add/resize recovery orchestration characterization: completed state

The 20-line `_run_sinet_preadd_resize_recovery_sequence` remains unchanged in
production. It is a parameterless straight-line closure over ModelIR and
session, has no control flow or local pass-state scope, and contains six
ordered calls. The first two calls are ModelIR-only and the final four route
the session layout state explicitly. Every target already has an extracted
module owner; no lowerer-local callback, diagnostics, or conversion option is
required.

The focused
`test_flatbuffer_direct_sinet_preadd_resize_recovery_orchestration.py` freezes
all six positional/keyword argument contracts and their exact order. It also
proves all four zero-argument invocations remain present: three top-level
pipeline boundaries and one nested boundary inside
`_run_sinet_terminal_layout_recovery_sequence`. The nested predecessor and
follower and all three distinct top-level predecessor/follower pairs are
recorded explicitly.

Validation completed sequentially as follows:

- focused SINet pre-add/resize characterization: `4 passed in 0.16s`;
- focused characterization plus ordered architecture:
  `252 passed in 16.74s`;
- TensorFlow-import-blocked optional boundary: `11 passed in 9.86s`;
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran. Public APIs, CLI behavior, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, and TensorFlow isolation are unchanged.
PR #952 remains closed, no branch PR is open, and no pull request was created,
reopened, or updated.

At restart, introduce a frozen ModelIR/layout context, six stable IDs, and
direct imports of every existing owner. Preserve the historical helper, all
three top-level invocations, the one nested invocation, and every neighboring
boundary. Prove builder argument equality and instrumented order before
switching the helper to a delegate, validate sequentially, commit and push,
and do not create a pull request.

## Explicit SINet pre-add/resize recovery orchestration: completed state

The characterized sequence now delegates to
`passes/sinet_preadd_resize_recovery_orchestration.py`. A frozen
`SINetPreaddResizeRecoveryContext` contains only ModelIR and layout state. All
six existing owners are imported directly; no lowerer callback, diagnostics,
conversion option, or central-lowerer import is required.

`SINET_PREADD_RESIZE_RECOVERY_PASS_IDS` declares the exact six-step order.
Immutable invocations preserve the two ModelIR-only and four layout-aware
argument contracts, and the shared executor validates the complete sequence
before running an owner. The historical helper remains a four-line delegate.
Its three top-level calls, one nested terminal-layout call, and all four
neighboring boundaries are unchanged.

Architecture ownership accounting now combines the stable phase occurrence
with remaining direct lowerer calls. This preserves the existing module-owner
contracts for the two residual-affine passes and all four SINet passes without
mistaking a mechanically moved call for a removed invocation.

Sequential validation completed as follows:

- focused SINet pre-add/resize orchestration: `7 passed in 0.58s`;
- ordered architecture: `248 passed in 17.80s`;
- focused orchestration plus ordered architecture:
  `255 passed in 17.13s`;
- related residual-affine/SINet owner set, excluding one documented stale
  assertion: `282 passed in 1.42s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 10.09s` (`32` plus `11`);
- targeted Ruff, Python compilation, formatting, and whitespace checks:
  passed; the central lowerer retains exactly its two pre-existing F401
  findings.

The diagnostic related-owner run including
`test_flatbuffer_direct_sinet_concat_resize_affine_transpose_chain_optimized`
reported `282 passed, 1 failed`. Its stale `optimized == 1` expectation also
fails unchanged at parent checkpoint `71d1814e`, where the optimizer returns
zero, proving that it is not a regression from this extraction. The temporary
detached comparison worktree was removed immediately after verification.

No real-model conversion or broad direct-suite repeat was added. Public APIs,
CLI behavior, artifacts, dependencies, corpus profiles, exclusions, operation
tiers, runtime pass order, invocation multiplicity, and TensorFlow isolation
are unchanged. PR #952 remains closed, no branch PR is open, and no pull
request was created, reopened, or updated.

At restart, inventory and characterize the adjacent
`_run_sinet_terminal_layout_recovery_sequence` before changing production
code. Freeze its three-slot order, nested pre-add/resize phase boundary,
ModelIR/layout routing, top-level repetitions, and outer neighbors before
choosing its explicit context. Validate sequentially, commit and push, and do
not create a pull request.

## SINet terminal-layout recovery orchestration characterization: completed state

The seven-line `_run_sinet_terminal_layout_recovery_sequence` remains
unchanged in production. It is a parameterless straight-line closure over
ModelIR and session, has no control flow or local pass-state scope, and contains
three ordered calls. The first call routes ModelIR and layout state to the
shuffle-residual owner, the second invokes the existing zero-argument SINet
pre-add/resize helper, and the third routes ModelIR alone to the terminal
affine/PReLU owner.

The focused
`test_flatbuffer_direct_sinet_terminal_layout_recovery_orchestration.py`
freezes all three positional/keyword argument contracts and their exact order.
It proves both outer invocations remain zero-argument top-level boundaries and
records their two distinct predecessor/follower pairs. Since the middle slot
must retain the historical nested helper boundary, a future extracted phase
requires one explicit callback in addition to ModelIR and layout state; its two
outer owners can be imported directly.

Validation completed sequentially as follows:

- focused SINet terminal-layout characterization: `4 passed in 0.16s`;
- focused characterization plus ordered architecture:
  `252 passed in 16.36s`;
- TensorFlow-import-blocked optional boundary: `11 passed in 9.47s`;
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran. Public APIs, CLI behavior, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, and TensorFlow isolation are unchanged.
PR #952 remains closed, no branch PR is open, and no pull request was created,
reopened, or updated.

At restart, introduce a frozen ModelIR/layout context, one explicit pre-add/
resize callback, three stable IDs, and direct imports of the two outer owners.
Preserve the historical helper, both zero-argument top-level invocations, the
nested pre-add/resize helper call, and both neighboring boundary pairs. Prove
builder argument equality and instrumented order before switching the helper
to a delegate, validate sequentially, commit and push, and do not create a pull
request.

## Explicit SINet terminal-layout recovery orchestration: completed state

The characterized sequence now delegates to
`passes/sinet_terminal_layout_recovery_orchestration.py`. A frozen
`SINetTerminalLayoutRecoveryContext` contains ModelIR, layout state, and one
explicit zero-argument pre-add/resize recovery callback. The shuffle-residual
and terminal affine/PReLU owners are imported directly; the module does not
import the central lowerer.

`SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS` declares the exact three-step order.
Immutable invocations preserve the first ModelIR/layout contract, the middle
zero-argument callback, and the final ModelIR-only contract. The shared
executor validates all IDs before running an owner. The historical terminal
helper is now a four-line delegate, while both zero-argument top-level calls
and their two distinct predecessor/follower pairs remain unchanged.

The pre-add/resize helper remains the identical callback object stored in the
new context. Its three direct top-level calls plus the stable nested callback
retain the former total of four executions. Architecture accounting likewise
combines stable phase occurrences with remaining direct calls for the two
outer module owners.

Sequential validation completed as follows:

- both adjacent SINet orchestration fixtures: `14 passed in 0.70s`;
- ordered architecture: `248 passed in 17.77s`;
- both orchestration fixtures, both outer owner suites, and ordered
  architecture: `470 passed in 17.38s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 10.01s` (`32` plus `11`);
- targeted Ruff, Python compilation, formatting, and whitespace checks:
  passed; the central lowerer retains exactly its two pre-existing F401
  findings.

No real-model conversion or broad direct-suite repeat was added. Public APIs,
CLI behavior, artifacts, dependencies, corpus profiles, exclusions, operation
tiers, runtime pass order, invocation multiplicity, and TensorFlow isolation
are unchanged. PR #952 remains closed, no branch PR is open, and no pull
request was created, reopened, or updated.

At restart, inventory and characterize
`_run_terminal_clamp_unary_relu_pass_cluster` before changing production code.
Freeze its three cleanup calls, shared per-invocation `ModelIRPassStateScope`,
ModelIR/layout/diagnostics routing, repetition count, and outer boundaries.
Preserve the existing scope-efficiency invariant, validate sequentially,
commit and push, and do not create a pull request.

## Terminal clamp/unary/ReLU orchestration characterization: completed state

The 26-line `_run_terminal_clamp_unary_relu_pass_cluster` remains unchanged in
production. It is a parameterless straight-line closure over ModelIR and
session, has no control flow, and is invoked exactly once. It creates one
`ModelIRPassStateScope` from ModelIR and layout state, then routes that same
scope, ModelIR, layout state, and diagnostics through three ordered cleanup
runners: clamp canonicalization, transpose-unary passthrough, and maximum-zero
ReLU canonicalization.

The focused
`test_flatbuffer_direct_terminal_clamp_unary_relu_orchestration.py` freezes the
single scope-construction contract, the complete three-runner argument order,
the shared `state_scope` identity by name, the zero-argument outer invocation,
and its location after the layout-gated singleton-reshape boundary and before
the SINet terminal-layout boundary. The existing runtime efficiency fixture
continues to prove that these runners build the graph index once and reuse it
for both later calls.

Validation completed sequentially as follows:

- focused terminal clamp/unary/ReLU characterization: `4 passed in 0.16s`;
- focused characterization plus ordered architecture:
  `252 passed in 16.48s`;
- pass-efficiency plus TensorFlow-import-blocked optional boundary:
  `41 passed in 9.86s` (`30` plus `11`);
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran. Public APIs, CLI behavior, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, and TensorFlow isolation are unchanged.
PR #952 remains closed, no branch PR is open, and no pull request was created,
reopened, or updated.

At restart, introduce a frozen ModelIR/layout/diagnostics context and three
stable IDs with direct imports from `graph_cleanup` and `layout_transpose`.
Construct exactly one fresh `ModelIRPassStateScope` for each phase invocation
and attach that same object to all three immutable invocations. Preserve the
historical zero-argument helper and its sole outer boundary, prove builder
arguments, scope identity, and instrumented order before switching it to a
delegate, validate sequentially, commit and push, and do not create a pull
request.

## Explicit terminal clamp/unary/ReLU orchestration: completed state

The characterized cluster now delegates to
`passes/terminal_clamp_unary_relu_orchestration.py`. A frozen
`TerminalClampUnaryReLUContext` contains only ModelIR, layout state, and
diagnostics. The three cleanup owners are imported directly from
`graph_cleanup` and `layout_transpose`; the phase module does not import the
central lowerer.

`TERMINAL_CLAMP_UNARY_RELU_PASS_IDS` declares the exact three-step order. Each
builder call creates one fresh `ModelIRPassStateScope` from the context's
ModelIR/layout pair and attaches the identical scope object, ModelIR, layout,
and diagnostics to all three immutable invocations. The shared executor
validates every ID before running an owner. The canonicalization-order comment
was moved with the second invocation rather than discarded.

The historical helper is now a four-line delegate, and its sole zero-argument
top-level call remains between the same layout-gated singleton-reshape and
SINet terminal-layout boundaries. Architecture accounting combines the three
stable IDs with remaining direct runner calls. The runtime efficiency fixture
now executes the explicit phase runner itself and still proves one graph-index
build followed by two scope reuses.

Sequential validation completed as follows:

- focused terminal clamp/unary/ReLU orchestration: `7 passed in 0.59s`;
- focused orchestration, ordered architecture, and pass-efficiency:
  `285 passed in 19.09s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 10.71s` (`32` plus `11`);
- targeted Ruff, Python compilation, formatting, and whitespace checks:
  passed; two moved cleanup imports were removed and the central lowerer now
  retains exactly its two pre-existing F401 findings.

No real-model conversion or broad direct-suite repeat was added. Public APIs,
CLI behavior, artifacts, dependencies, corpus profiles, exclusions, operation
tiers, runtime pass order, invocation multiplicity, shared-scope efficiency,
and TensorFlow isolation are unchanged. PR #952 remains closed, no branch PR
is open, and no pull request was created, reopened, or updated.

At restart, inventory and characterize the neighboring
`_run_terminal_singleton_maxpool_reshape_pass_pair` before changing production
code. Freeze both cleanup calls, their shared per-invocation
`ModelIRPassStateScope`, all ModelIR/layout/diagnostics arguments, repetition,
and outer boundaries. Validate sequentially, commit and push, and do not create
a pull request.

## Terminal singleton-maxpool/reshape orchestration characterization: completed state

The 19-line `_run_terminal_singleton_maxpool_reshape_pass_pair` remains
unchanged in production. It is a parameterless straight-line closure over
ModelIR and session, has no control flow, and is invoked exactly once. It
creates one `ModelIRPassStateScope` from ModelIR and layout state, then routes
that same scope, ModelIR, layout state, and diagnostics first to singleton-
maxpool layout cleanup and then to consecutive-reshape cleanup. The retained
comment documents why the second cleanup must follow boundary cleanup.

The focused
`test_flatbuffer_direct_terminal_singleton_maxpool_reshape_orchestration.py`
freezes the single scope-construction contract, both complete runner argument
contracts, their exact order and shared `state_scope` name, the sole zero-
argument invocation, and its location between two layout-gated blocks. The
existing pass-efficiency fixture continues to prove that the pair builds the
graph index once and reuses it for the second runner.

Validation completed sequentially as follows:

- focused terminal singleton-maxpool/reshape characterization:
  `4 passed in 0.16s`;
- focused characterization plus ordered architecture:
  `252 passed in 16.69s`;
- pass-efficiency plus TensorFlow-import-blocked optional boundary:
  `41 passed in 10.02s` (`30` plus `11`);
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran. Public APIs, CLI behavior, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, and TensorFlow isolation are unchanged.
PR #952 remains closed, no branch PR is open, and no pull request was created,
reopened, or updated.

At restart, introduce a frozen ModelIR/layout/diagnostics context and two
stable IDs with direct imports from `singleton_maxpool_layout` and
`graph_cleanup`. Construct exactly one fresh `ModelIRPassStateScope` per phase
invocation and attach the same object to both immutable invocations. Preserve
the historical helper, its sole zero-argument call, both layout-gated outer
boundaries, and the canonicalization-order comment. Prove builder arguments,
scope identity, and instrumented order before switching to a delegate,
validate sequentially, commit and push, and do not create a pull request.

## Explicit terminal singleton-maxpool/reshape orchestration: completed state

The characterized pair now delegates to
`passes/terminal_singleton_maxpool_reshape_orchestration.py`. A frozen
`TerminalSingletonMaxPoolReshapeContext` contains only ModelIR, layout state,
and diagnostics. The two cleanup owners are imported directly from
`singleton_maxpool_layout` and `graph_cleanup`; the phase module does not
import the central lowerer.

`TERMINAL_SINGLETON_MAXPOOL_RESHAPE_PASS_IDS` declares the exact two-step
order. Every builder call constructs one fresh `ModelIRPassStateScope` from the
context's ModelIR/layout pair and attaches the identical scope, ModelIR,
layout, and diagnostics values to both immutable invocations. The shared
executor validates both IDs before running an owner, and the boundary-cleanup
comment moved with the second invocation.

The historical helper is now a four-line delegate. Its sole zero-argument call
remains between the same two layout-gated blocks. Architecture accounting
combines both stable IDs with remaining direct runner calls. The runtime
efficiency fixture now executes this explicit runner and still proves one
graph-index build with reuse by consecutive-reshape cleanup.

Sequential validation completed as follows:

- focused terminal singleton-maxpool/reshape orchestration:
  `7 passed in 0.60s`;
- focused orchestration, ordered architecture, and pass-efficiency:
  `285 passed in 18.86s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 10.94s` (`32` plus `11`);
- targeted Ruff, Python compilation, formatting, and whitespace checks:
  passed; the central lowerer retains exactly its two pre-existing F401
  findings.

No real-model conversion or broad direct-suite repeat was added. Public APIs,
CLI behavior, artifacts, dependencies, corpus profiles, exclusions, operation
tiers, runtime pass order, invocation multiplicity, shared-scope efficiency,
and TensorFlow isolation are unchanged. PR #952 remains closed, no branch PR
is open, and no pull request was created, reopened, or updated.

At restart, inventory and characterize the neighboring
`_run_late_dequant_unary_fanout_pass_cluster` before changing production code.
Freeze its three cleanup calls, shared per-invocation pass-state scope,
ModelIR/layout/diagnostics arguments, sole outer invocation, and both
neighbors. Validate sequentially, commit and push, and do not create a pull
request.

## Late dequant/unary/fanout orchestration characterization: completed state

The 26-line `_run_late_dequant_unary_fanout_pass_cluster` remains unchanged in
production. It is a parameterless straight-line closure over ModelIR and
session, has no control flow, and is invoked exactly once. It creates one
`ModelIRPassStateScope` from ModelIR and layout state, then routes the same
scope, ModelIR, layout state, and diagnostics through dequant/concat/quantize
cleanup, transpose-unary passthrough cleanup, and transpose-unary fanout bridge
cleanup in that exact order.

The focused
`test_flatbuffer_direct_late_dequant_unary_fanout_orchestration.py` freezes the
single scope construction, all three complete runner argument contracts, the
shared `state_scope` name, the sole zero-argument invocation, and its exact
placement between the quantized HardSigmoid bridge and swish passthrough. The
existing pass-efficiency fixture continues to prove one graph-index build and
two later scope reuses.

Validation completed sequentially as follows:

- focused late dequant/unary/fanout characterization: `4 passed in 0.17s`;
- focused characterization plus ordered architecture:
  `252 passed in 17.65s`;
- pass-efficiency plus TensorFlow-import-blocked optional boundary:
  `41 passed in 10.84s` (`30` plus `11`);
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran. Public APIs, CLI behavior, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, and TensorFlow isolation are unchanged.
PR #952 remains closed, no branch PR is open, and no pull request was created,
reopened, or updated.

At restart, introduce a frozen ModelIR/layout/diagnostics context and three
stable IDs with direct imports from `dequant_concat_quantize_layout` and
`layout_transpose`. Construct exactly one fresh `ModelIRPassStateScope` per
phase invocation and attach the identical object to all three immutable
invocations. Preserve the historical helper, sole zero-argument call, both
outer neighbors, and unary-fold rationale comment. Prove builder arguments,
scope identity, and instrumented order before switching to a delegate,
validate sequentially, commit and push, and do not create a pull request.

## Explicit late dequant/unary/fanout orchestration: completed state

The characterized cluster now delegates to
`passes/late_dequant_unary_fanout_orchestration.py`. A frozen
`LateDequantUnaryFanoutContext` contains only ModelIR, layout state, and
diagnostics. All three cleanup owners are imported directly from
`dequant_concat_quantize_layout` and `layout_transpose`; the phase module does
not import the central lowerer.

`LATE_DEQUANT_UNARY_FANOUT_PASS_IDS` declares the exact three-step order. Each
builder call constructs one fresh `ModelIRPassStateScope` from the context's
ModelIR/layout pair and attaches the same scope, ModelIR, layout, and
diagnostics to all three immutable invocations. The shared executor validates
every ID before execution, and the final unary-fold rationale comment moved
with its invocation.

The historical helper is a four-line delegate at the same sole zero-argument
boundary between the quantized HardSigmoid bridge and swish passthrough.
Architecture ownership uses stable-ID multiplicity for the moved dequant,
unary passthrough, and unary fanout occurrences while retaining all other
direct calls. The efficiency fixture now drives the explicit runner and still
observes one graph-index build followed by two scope reuses.

Sequential validation completed as follows:

- focused late dequant/unary/fanout orchestration: `7 passed in 0.64s`;
- focused orchestration, ordered architecture, and pass-efficiency:
  `285 passed in 19.25s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 11.02s` (`32` plus `11`);
- targeted Ruff, Python compilation, formatting, and whitespace checks:
  passed; the central lowerer retains exactly its two pre-existing F401
  findings.

No real-model conversion or broad direct-suite repeat was added. Public APIs,
CLI behavior, artifacts, dependencies, corpus profiles, exclusions, operation
tiers, runtime pass order, invocation multiplicity, shared-scope efficiency,
and TensorFlow isolation are unchanged. PR #952 remains closed, no branch PR
is open, and no pull request was created, reopened, or updated.

At restart, inventory and characterize the neighboring
`_run_transpose_unary_fanout_layout_pass_cluster` before changing production
code. Freeze its option-dependent calls, default values, shared pass-state
scope, all ModelIR/layout/diagnostics arguments, invocation variants, and outer
boundaries. Validate sequentially, commit and push, and do not create a pull
request.

## Transpose/unary-fanout orchestration characterization: completed state

The 35-line `_run_transpose_unary_fanout_layout_pass_cluster` remains
unchanged in production. It exposes two keyword-only options with the exact
defaults `include_layout_transpose=False` and
`include_unary_passthrough=True`. Every invocation creates one fresh
`ModelIRPassStateScope` from ModelIR and layout state and shares it, together
with the same ModelIR, layout, and diagnostics values, across all active
cleanup runners.

The focused
`test_flatbuffer_direct_transpose_unary_fanout_orchestration.py` freezes the
four ordered runner slots and their complete arguments. Layout-transpose and
unary-passthrough cleanup are independently conditional; unary-fanout and
unary-binary-fanout cleanup are unconditional. It also freezes both runtime
variants: the attention-recovery callback uses the defaults, whereas the sole
direct post-QDQ invocation requests layout-transpose and disables unary
passthrough. The direct call remains between the layout-attention suffix and
safe-binary recovery sequence, and the callback remains between the terminal
transpose-convolution cleanup and dequant/ReLU/quantize bridge in
`ATTENTION_GATE_QDQ_PASS_IDS`. The two existing efficiency fixtures continue
to prove one graph-index build for each variant.

Sequential validation completed as follows:

- focused transpose/unary-fanout characterization: `4 passed in 0.60s`;
- focused characterization plus ordered architecture:
  `252 passed in 18.72s`;
- pass-efficiency plus TensorFlow-import-blocked optional boundary:
  `41 passed in 11.18s` (`30` plus `11`);
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran. Public APIs, CLI behavior, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, and TensorFlow isolation are unchanged.
PR #952 remains closed, no branch PR is open, and no pull request was created,
reopened, or updated.

At restart, introduce a frozen ModelIR/layout/diagnostics context and four
stable IDs with direct imports from `layout_transpose`. Keep the two options as
runner arguments rather than context state. Build the active expected-ID tuple
per variant, create exactly one fresh `ModelIRPassStateScope` per phase
invocation, and attach it to every active immutable invocation. Preserve the
historical keyword-only helper, both option defaults, the attention callback
identity, the explicit post-QDQ call, and every surrounding boundary. Prove
both variants, fresh/shared scope identity, and instrumented order before
switching to a delegate; validate sequentially, commit and push, and do not
create a pull request.

## Explicit transpose/unary-fanout orchestration: completed state

The characterized cluster now delegates to
`passes/transpose_unary_fanout_orchestration.py`. A frozen
`TransposeUnaryFanoutContext` contains only ModelIR, layout state, and
diagnostics. All four cleanup owners are imported directly from
`layout_transpose`; the phase module does not import the central lowerer.

`TRANSPOSE_UNARY_FANOUT_PASS_IDS` is the canonical four-owner sequence, while
`active_transpose_unary_fanout_pass_ids` derives the exact expected sequence
for each option combination. The default attention callback executes unary-
passthrough, unary-fanout, and unary-binary-fanout cleanup. The explicit post-
QDQ call executes layout-transpose, unary-fanout, and unary-binary-fanout
cleanup. Options remain per-invocation arguments and are not frozen into the
context. Each builder call creates one fresh `ModelIRPassStateScope` and
attaches the identical scope, ModelIR, layout, and diagnostics values to all
active immutable invocations. The shared executor validates the variant-
specific IDs before executing an owner.

The historical helper is now a ten-line keyword-only delegate with the same
`False`/`True` defaults. Its attention-recovery callback identity, explicit
post-QDQ `True`/`False` call, and all adjacent boundaries are unchanged.
Architecture accounting moves one syntactic occurrence of every owner to its
stable ID while retaining the callback's separate phase multiplicity. Both
runtime efficiency fixtures now execute the explicit runner and still prove
one graph-index build per three-step variant. The obsolete direct unary-
passthrough import was removed from the central lowerer.

Sequential validation completed as follows:

- focused transpose/unary-fanout orchestration: `9 passed in 0.66s`;
- focused orchestration, ordered architecture, and pass-efficiency:
  `287 passed in 19.73s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 11.00s` (`32` plus `11`);
- targeted Ruff, Python compilation, formatting, and whitespace checks:
  passed; the central lowerer retains exactly its two pre-existing F401
  findings.

No real-model conversion or broad direct-suite repeat was added. Public APIs,
CLI behavior, artifacts, dependencies, corpus profiles, exclusions, operation
tiers, runtime pass order, invocation multiplicity, option semantics, shared-
scope efficiency, and TensorFlow isolation are unchanged. PR #952 remains
closed, no branch PR is open, and no pull request was created, reopened, or
updated.

At restart, inventory the next still-central post-lowering cluster before any
production edit. Prefer a small cluster with existing module owners, freeze
its complete option/callback/state-scope contract and every runtime boundary,
then create characterization and implementation as separate sequentially
validated commits. Continue to minimize real-model conversion, commit and push
only, and do not create a pull request.

## Late SPP/concat-unary-conv orchestration characterization: completed state

The 17-line `_run_late_spp_concat_unary_conv_pass_pair` remains unchanged in
production. It is a parameterless straight-line closure over ModelIR and
session with no option, callback, or control-flow dependency. Every invocation
creates one `ModelIRPassStateScope` from ModelIR and layout state, then routes
the identical scope, ModelIR, layout state, and diagnostics first to SPP layout
cleanup and then to concat/unary/conv layout cleanup.

The focused
`test_flatbuffer_direct_late_spp_concat_unary_conv_orchestration.py` freezes
the one scope construction, both complete cleanup argument contracts, their
exact order, the sole zero-argument invocation, and its terminal placement
between strided-slice/pad/concat bridge recovery and shape-extract layout
recovery. The existing efficiency fixture continues to prove one graph-index
build across both runners.

Sequential validation completed as follows:

- focused late SPP/concat-unary-conv characterization:
  `4 passed in 0.17s`;
- focused characterization plus ordered architecture:
  `252 passed in 18.18s`;
- pass-efficiency plus TensorFlow-import-blocked optional boundary:
  `41 passed in 10.81s` (`30` plus `11`);
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran. Public APIs, CLI behavior, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, and TensorFlow isolation are unchanged.
PR #952 remains closed, no branch PR is open, and no pull request was created,
reopened, or updated.

At restart, introduce a frozen ModelIR/layout/diagnostics context and two
stable IDs with direct imports from `spp_layout` and
`concat_unary_conv_layout`. Construct exactly one fresh
`ModelIRPassStateScope` per phase invocation and attach the identical object to
both immutable invocations. Preserve the historical helper, its sole zero-
argument call, both outer neighbors, and runtime efficiency. Prove builder
arguments, fresh/shared scope identity, and instrumented order before switching
to a delegate; validate sequentially, commit and push, and do not create a pull
request.

## Explicit late SPP/concat-unary-conv orchestration: completed state

The characterized pair now delegates to
`passes/late_spp_concat_unary_conv_orchestration.py`. A frozen
`LateSPPConcatUnaryConvContext` contains only ModelIR, layout state, and
diagnostics. The SPP and concat/unary/conv cleanup owners are imported directly
from their existing pass modules; the phase module does not import the central
lowerer.

`LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS` declares the exact two-step order. Every
builder call constructs one fresh `ModelIRPassStateScope` from the context's
ModelIR/layout pair and attaches the identical scope, ModelIR, layout, and
diagnostics values to both immutable invocations. The shared executor validates
both IDs before running an owner.

The historical helper is now a four-line zero-argument delegate at the same
sole terminal boundary between strided-slice/pad/concat bridge recovery and
shape-extract layout recovery. Architecture accounting combines both stable
IDs with all remaining direct owner calls. The runtime efficiency fixture now
executes the explicit phase runner and still proves one graph-index build with
reuse by the second cleanup.

Sequential validation completed as follows:

- focused late SPP/concat-unary-conv orchestration: `7 passed in 0.61s`;
- focused orchestration plus ordered architecture:
  `255 passed in 19.72s`;
- pass-efficiency: `30 passed in 0.55s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 11.06s` (`32` plus `11`);
- targeted Ruff, Python compilation, formatting, and whitespace checks:
  passed; the central lowerer retains exactly its two pre-existing F401
  findings.

No real-model conversion or broad direct-suite repeat was added. Public APIs,
CLI behavior, artifacts, dependencies, corpus profiles, exclusions, operation
tiers, runtime pass order, invocation multiplicity, shared-scope efficiency,
and TensorFlow isolation are unchanged. PR #952 remains closed, no branch PR
is open, and no pull request was created, reopened, or updated.

At restart, inventory and characterize another small still-central shared-
scope cluster before changing production. The 17-line boundary-batchmatmul/
input-unary pair is a likely candidate, but first freeze all of its actual
callback and outer phase boundaries. Validate sequentially, keep real-model
conversion minimal, commit and push only, and do not create a pull request.

## Boundary-batchmatmul/input-unary orchestration characterization: completed state

The 17-line `_run_boundary_batchmatmul_unary_layout_pass_cluster` remains
unchanged in production. It is a parameterless straight-line closure over
ModelIR and session with no option or control-flow dependency. It creates one
`ModelIRPassStateScope` from ModelIR and layout state, then passes the identical
scope, ModelIR, layout state, and diagnostics first to boundary-input
batch-matmul cleanup and then to input-unary passthrough cleanup.

This helper has no direct invocation. It remains the exact
`boundary_batchmatmul_unary_cluster` callback stored in
`LayoutRecoveryContext`, and its stable execution slot remains between
transpose quant/dequant bridge recovery and the NCHW/NHWC elementwise
roundtrip owner in `LAYOUT_RECOVERY_PASS_IDS`.

The focused
`test_flatbuffer_direct_boundary_batchmatmul_unary_orchestration.py` freezes
the scope construction, both complete cleanup argument contracts, their exact
order, callback-only ownership, callback identity, and both stable phase
neighbors. The existing efficiency fixture continues to prove one graph-index
build across both runners.

Sequential validation completed as follows:

- focused boundary-batchmatmul/input-unary characterization:
  `4 passed in 0.57s`;
- focused characterization plus ordered architecture:
  `252 passed in 17.92s`;
- pass-efficiency plus TensorFlow-import-blocked optional boundary:
  `41 passed in 11.03s` (`30` plus `11`);
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran. Public APIs, CLI behavior, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, and TensorFlow isolation are unchanged.
PR #952 remains closed, no branch PR is open, and no pull request was created,
reopened, or updated.

At restart, introduce a frozen ModelIR/layout/diagnostics context and two
stable IDs with direct imports from `boundary_input_chains` and
`input_passthrough_layout`. Build one fresh `ModelIRPassStateScope` per phase
invocation and attach the identical object to both immutable invocations.
Preserve the historical zero-argument helper as the same
`LayoutRecoveryContext` callback and retain both stable-list neighbors. Prove
builder arguments, fresh/shared scope identity, instrumented order, and
callback identity before switching to a delegate; validate sequentially,
commit and push, and do not create a pull request.

## Explicit boundary-batchmatmul/input-unary orchestration: completed state

The characterized pair now delegates to
`passes/boundary_batchmatmul_unary_orchestration.py`. A frozen
`BoundaryBatchMatMulUnaryContext` contains only ModelIR, layout state, and
diagnostics. Both cleanup owners are imported directly from
`boundary_input_chains` and `input_passthrough_layout`; the phase module does
not import the central lowerer.

`BOUNDARY_BATCHMATMUL_UNARY_PASS_IDS` declares the exact two-step order. Every
builder call creates one fresh `ModelIRPassStateScope` from the context's
ModelIR/layout pair and attaches the identical scope, ModelIR, layout, and
diagnostics values to both immutable invocations. The shared executor validates
both IDs before execution.

The historical helper is now a four-line zero-argument delegate and remains
the same callback object supplied to `LayoutRecoveryContext`. Its stable slot
between quant/dequant bridge recovery and elementwise roundtrip recovery is
unchanged. The two direct runner imports were removed from the central lowerer,
and architecture checks now require their ownership to reside in stable phase
IDs. The efficiency fixture executes the explicit runner and still observes
one graph-index build across the pair.

Sequential validation completed as follows:

- focused boundary-batchmatmul/input-unary orchestration:
  `7 passed in 0.58s`;
- focused orchestration plus ordered architecture:
  `255 passed in 19.25s`;
- pass-efficiency: `30 passed in 0.57s`;
- central lowerer synthetic smoke plus TensorFlow-import-blocked optional
  boundary: `43 passed in 11.00s` (`32` plus `11`);
- targeted Ruff, Python compilation, formatting, and whitespace checks:
  passed; the central lowerer retains exactly its two pre-existing F401
  findings.

No real-model conversion or broad direct-suite repeat was added. Public APIs,
CLI behavior, artifacts, dependencies, corpus profiles, exclusions, operation
tiers, callback identity, runtime pass order, invocation multiplicity, shared-
scope efficiency, and TensorFlow isolation are unchanged. PR #952 remains
closed, no branch PR is open, and no pull request was created, reopened, or
updated.

At restart, inventory and characterize the adjacent channel-slice/pad-mul
shared-scope pair before changing production. Freeze its two potentially
different keyword contracts, callback-only ownership in the terminal
slice/concat phase, stable neighbors, and existing efficiency behavior.
Validate sequentially, keep real-model conversion minimal, commit and push
only, and do not create a pull request.

## Channel-slice/pad-mul orchestration characterization: completed state

The 17-line `_run_channel_slice_pad_mul_layout_pass_cluster` remains unchanged
in production. It is a parameterless straight-line closure over ModelIR and
session. Every invocation creates one `ModelIRPassStateScope` from ModelIR and
layout state, then passes the identical scope, ModelIR, layout state, and
diagnostics first to channel-slice merge cleanup and then to pad/mul cleanup.

The helper has two runtime boundaries. It is the first callback in
`TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS`, immediately before affine post-add
recovery, and it also has one zero-argument late-terminal direct call between
pre-add recovery and the same affine post-add owner. The focused
`test_flatbuffer_direct_channel_slice_pad_mul_orchestration.py` freezes both
complete cleanup contracts, the shared scope, direct invocation, callback
identity, stable-list position, and direct neighbors. The existing efficiency
fixture continues to prove one graph-index build per pair invocation.

Sequential validation completed as follows:

- focused channel-slice/pad-mul characterization: `5 passed in 0.61s`;
- focused characterization plus ordered architecture:
  `253 passed in 17.96s`;
- pass-efficiency plus TensorFlow-import-blocked optional boundary:
  `41 passed in 10.62s` (`30` plus `11`);
- focused Ruff formatting/lint, Python compilation, and whitespace checks:
  passed.

No production source, runtime sequence, real-model conversion, or broad suite
changed or ran. Public APIs, CLI behavior, artifacts, dependencies, corpus
profiles, exclusions, operation tiers, and TensorFlow isolation are unchanged.
PR #952 remains closed, no branch PR is open, and no pull request was created,
reopened, or updated.

At restart, introduce a frozen ModelIR/layout/diagnostics context and two
stable IDs with direct imports from `channel_slice_layout` and `pad_layout`.
Build one fresh `ModelIRPassStateScope` per phase invocation and attach the same
object to both immutable invocations. Preserve the historical zero-argument
helper as the terminal-slice/concat callback, its additional direct call, and
both kinds of boundaries. Prove builder arguments, fresh/shared scope identity,
instrumented order, callback identity, and two total executions before
switching to a delegate; validate sequentially, commit and push, and do not
create a pull request.
