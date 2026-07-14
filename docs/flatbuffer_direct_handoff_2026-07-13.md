# flatbuffer_direct refactor handoff — 2026-07-13

## `fb-refactor4` PyTorch WHILE stream checkpoint

The two native-PyTorch WHILE expansion paths no longer deep-copy the complete
root operator list and then replace it with a second list. A Torch-free helper
in `passes/pytorch_control_flow.py` deep-clones tensors, metadata, and complete
subgraphs while starting the root graph with an empty operator stream. Static
trip-count and counter-bounded WHILE expansion then reads the original root
operators in stable order, deep-copies each retained operator exactly once,
and appends retained or expanded operators directly to the new stream.

This preserves the prior independent-copy contract, complete WHILE subgraphs,
operator ordering, and generated-name order while removing an eagerly cloned
operator list and the peak lifetime in which both root streams were retained.
Copy-on-write preflight now returns the borrowed input unchanged when no static
WHILE, counter-bounded WHILE, or unsupported recurrent-sequence rewrite is
required. The mandatory channel-first normalizer remains the single owner copy
in that common path, instead of receiving the fourth consecutive deep copy. A
matched rewrite still produces a fully independent ModelIR before mutation.
The complete subgraph lookup, constant/alias guards, static and
counter-bounded matchers, shape-literal creation, and both rewrite entry points
now have this module as their single implementation owner. The 42k-line
exporter imports only the two ordered entry points; it no longer contains this
approximately 470-line control-flow implementation.
Each matcher now constructs one `ModelIRGraphIndex` for its WHILE body and
reuses it for iterator, alias, condition, comparison, and Cast producer
lookups. The former repeated linear body scans are removed. Any duplicate
producer on a required edge rejects the optimization before cloning or
mutation, rather than selecting an order-dependent producer.
An AST architecture gate now rejects assignments to any object's `operators`
attribute in the PyTorch exporter instead of checking only the literal
`model_ir.operators =` spelling. Focused Torch-free compatibility and
architecture validation, including direct two-iteration static and
counter-bounded expansion fixtures and a duplicate-producer rejection, passed
63 tests. No ONNX conversion, inference,
dependency change, TensorFlow import, or parallel process was involved.

Static and counter-bounded WHILE entry points now match every root operator
once and retain successful matches as rewrite plans. Their copy-on-write emit
loops consume those plans rather than rerunning the complete body-subgraph
matcher against the cloned graph, eliminating a second body index and guard
evaluation for every expanded WHILE. Both canonical fixtures match checkpoint
`cfa47da` byte for byte at the ModelIR fingerprint level. A seven-run synthetic
64-WHILE median improved from 0.010825s to 0.010082s (1.07x). Focused
control-flow and architecture gates pass 86 tests, including exact
one-match-per-root instrumentation. No model conversion or inference was run.

The recurrent-sequence companion is now isolated in the Torch-free
`passes/pytorch_recurrent.py` module. It is the single owner of legacy
unidirectional/bidirectional LSTM input-index contracts, constant/optional
input validation, direct native RNN/LSTM capability selection, and delegation
to the shared split-planner unroller. The exporter imports the capability
functions required by its generated-code environment plus the one preparation
entry point; their function ASTs are unchanged from checkpoint `249f2d5`.

Focused tests cover all supported LSTM arities, complete constant-backed RNN
and LSTM direct paths, dynamic-weight and non-time-major rejection, borrowed
no-op results, and normalized unroll failures. Together with control-flow,
compatibility, and architecture tests, 67 tests pass. This extraction did not
run a model conversion or inference process.

PyTorch Softmax layout validation is now owned by Torch-free
`passes/pytorch_layout_validation.py`. Attention-consumer detection and
Transpose→Softmax→inverse-Transpose recognition use a caller-supplied
`ModelIRGraphIndex` instead of scanning the complete operator list for every
Softmax tensor. `validate_channel_first_exportability()` constructs this index
lazily at the first unknown-layout Softmax and shares it across every remaining
candidate. Models without such a Softmax pay no index-construction cost.
Required sandwich inputs with duplicate producers are rejected deterministically.

Focused index-reuse, attention, sandwich, duplicate-producer, recurrent,
control-flow, compatibility, and architecture validation passes 71 tests. No
model conversion or inference was run for this checkpoint.

Recurrent orphan-step repair moved beside the recurrent capability policy. It
first identifies names matching the legacy `_h_step_`/`_c_step_` contract; an
irrelevant graph returns before index construction. Candidate graphs build one
`ModelIRGraphIndex`, resolve the shape-driven Reshape through indexed consumers,
rewire orphan consumers through differential input replacement, and preserve
public-output tensors. This replaces the former candidate-by-candidate complete
operator scan and keeps the index coherent during multiple repairs. The focused
suite now passes 73 tests, including one-index and zero-index fast-path checks.

Feature-last region propagation no longer rescans the complete operator list
until a fixed point. `_collect_feature_last_sequence_tensor_names()` constructs
one `ModelIRGraphIndex` for its existing seed discovery and passes it to the
Torch-free layout-validation module. A deterministic tensor worklist visits
only indexed producer/consumer edges, applies the same bidirectional
passthrough rules, and retains standard channel-layout Transposes and
factorized rank-three Reshapes as propagation barriers. The monotonic result is
the same fixed point without graph-size-times-region-depth scanning. Focused
bidirectional closure and Transpose-barrier cases bring the suite to 75 passing
tests; no model conversion or inference was run.

The forward channel-last annotation loop in
`_apply_feature_last_sequence_layouts()` now uses the same indexed approach.
It seeds a worklist from tensors already carrying a channel-last layout, visits
only their recorded consumers, and enqueues newly annotated rank-three through
rank-five outputs. The established safe-op allowlist remains unchanged;
unsupported operators stop propagation. This handles non-topological operator
order without rescanning the graph. Reverse-order-chain and unsafe-boundary
fixtures bring the focused suite to 77 passing tests.

`_apply_feature_last_sequence_layouts()` itself is now owned by the Torch-free
layout-validation module. Its established Transpose/Reshape layout decisions,
raw ONNX Reshape contract restoration, and shape-constant update order are
unchanged; only the fallback producer/consumer construction now comes from one
`ModelIRGraphIndex`. After normalizing that intentional substitution, the moved
function AST is identical to checkpoint `51b49a6`. Direct Reshape restoration
and empty-preserve-set fast-path cases bring the focused suite to 79 passing
tests. The exporter retains only the ordered calls.

Feature-last seed collection, rank-four channel-last island capability,
preserved-region shrinking, and non-preserved channel-first restoration now
share the same Torch-free owner. The collector constructs one
`ModelIRGraphIndex`; shrink uses caller-provided maps or one indexed fallback
instead of its former ad hoc map builder. Complete-island, partial safe-region,
public-boundary, and public-layout-bridge decisions remain unchanged. After
normalizing the indexed fallback substitution, all six moved function ASTs are
identical to checkpoint `db18f97`. Focused validation now passes 82 tests.

Public input/output layout bridge insertion has also moved to
`passes/pytorch_layout_validation.py`. It validates shape/layout compatibility
before constructing graph state, then shares one lazy `ModelIRGraphIndex`
across all boundary rewrites. Input consumers, output producers, and output
consumers are updated differentially; bridge operators retain their established
front/append ordering and metadata contract. An already matching public
contract remains a zero-index no-op. The focused Torch-free suite now passes 84
tests, with no model conversion or inference.

The general PyTorch-friendly layout fixed point has moved into the same module
and now uses one indexed worklist. Unary, binary, Concat, Pack/Unpack, Split,
resize, and pool operators preserve the prior propagation rules, including
Concat peer inference, but only the producer and consumers adjacent to a
changed tensor are rescheduled. Reverse-order fixtures pass, and 64 fixed-seed
randomized ModelIR graphs produce exactly the same tensor layouts as the former
whole-graph fixed-point loop. The focused suite now passes 86 tests; no model
conversion or inference was run.

TransposeConv2D and TransposeConv3D emission have been split into distinct
Torch-free functions. Both preserve module parameter expressions, output-shape
constant/fallback behavior, target layout, and post-call fused activation. The
2D fixture also fixes the legacy `_nhwc`-named output override when metadata is
channel-first; the 3D fixture fixes metadata fallback and NDHWC output. Emitter
and architecture validation passes 78 tests; syntax, Ruff, and diff checks
pass. No model conversion or inference was run.

Regular Conv3D source emission now has its own Torch-free owner. Direct module
calls preserve NCDHW aliases and NDHWC output materialization; unsupported
direct calls retain `_apply_module_conv3d`, target-shape/layout, alias cleanup,
and activation behavior. Focused tests cover both paths. The dispatcher keeps
only one Conv3D mention for fused-module raw-layout classification. Emitter and
architecture validation passes 80 tests; syntax, Ruff, and diff checks pass. No
model conversion or inference was run.

The direct-module dispatch table and ordered orchestration have now moved out
of the exporter too. The Torch-free emitter module owns all direct-module
family logic and routing, while the exporter only imports the table/function
names used by capability selection and the stored generated pipeline. A direct
dispatcher fixture proves FullyConnected routing and unsupported-op early
rejection before attribute lookup. Emitter and architecture validation passes
85 tests; syntax, Ruff, and diff checks pass. No model conversion or inference
was run.

The binary emitter compatibility wrapper is gone as well. The stored codegen
function source now adds exactly one `binary_output_target_shape_literal_fn`
keyword at its binary call, passing the existing exporter shape policy while
calling the imported Torch-free emitter directly. Architecture validation
checks the single insertion and parses the transformed source AST. Emitter and
architecture validation remains 85 passing tests; syntax, Ruff, and diff checks
pass. The exporter now defines none of the extracted native unary, binary,
transpose, shape-transform, Concat, or direct-module emitters. No model
conversion or inference was run.

Native encoder-stage composition now lives in the Torch-free
`pytorch_codegen_stages.py` module. The production composite builder retains
the same imported name used by the stored codegen pipeline and preserves BERT
layer grouping, attention/FFN splitting, liveness-derived signatures,
initialization lines, and forward calls. The unused older non-composite builder
was removed from the exporter. Three direct fixtures cover inline, grouped, and
attention/FFN paths; 300 deterministic generated stage specifications return
exactly the same values as the pre-extraction implementation. Stage and
architecture validation passes 61 tests; syntax, Ruff, and diff checks pass.
No model conversion or inference was run.

Forward-stage partitioning and single-use static Reshape-chain folding now
share the same Torch-free owner. The 18/28/36-line liveness-based partition
policy, stage signatures, calls, specs, reshape-boundary protection, and inline
fallback are unchanged; an unused local tensor-name reverse map exposed by
Ruff after extraction was removed. Four reshape-chain cases and 250
deterministic forward bodies match the pre-extraction implementation exactly.
Direct stage and architecture validation passes 65 tests; syntax, Ruff, and
diff checks pass. No model conversion or inference was run. Resume by
separating the remaining artifact exporters only where metadata and child-
process dependencies can be made explicit; keep the large raw-source
canonicalizer as a later, separately characterized boundary.

TorchScript export now lives in `pytorch_artifact_exporters.py`, while shared
artifact metadata, dynamic-input policy, native/torch.export skip policy, and
the one-at-a-time child runner live in `pytorch_export_support.py`. The exporter
retains imported public/internal names, so call sites and API shape are
unchanged. The five shared helpers are AST-identical to the old implementations;
the TorchScript body is identical after its local Torch availability guard.
Support-module import no longer loads Torch eagerly, and its image resize path
imports Torch only when requested. Direct tests prove legacy runtime-wrapper
detection, dynamic shape-signature handling, and non-native TorchScript skip
metadata without spawning a child. Artifact and architecture validation passes
62 tests; syntax, Ruff, and diff checks pass. No model conversion or inference
was run.

Dynamo ONNX export is now implemented in the same artifact module. The exporter
retains only a public-signature wrapper that supplies the existing temporary
model-source rewrite and final-repair hooks. External-data inspection, missing
output-shape restoration, and ONNX cleanup/sanitization moved to the Torch-free
`pytorch_onnx_artifact_support.py`. The four ONNX helpers are AST-identical to
their old implementations; the Dynamo artifact body is AST-identical after
normalizing its two explicit callback names. A direct non-native-package test
proves skip metadata and verifies that neither callback nor a child process is
invoked. Artifact and architecture validation passes 64 tests; syntax, Ruff,
and diff checks pass. No model conversion or inference was run. Resume by
characterizing the remaining ExportedProgram host and archive boundaries
separately.

The 1,813-line ExportedProgram subprocess source literal now lives as the inert
`_EXPORTED_PROGRAM_CHILD_SCRIPT` constant in
`pytorch_exported_program_child.py`. Its 71,054 bytes exactly match the previous
embedded value; a fixed SHA-256 and `ast.parse` gate prevent accidental payload
drift. The exporter host now contains only `child_script =` the imported
constant, exposing its approximately 140 lines of metadata, rewrite-context,
child-runner, archive-cleanup, and repair orchestration. Artifact and
architecture validation passes 66 tests; syntax, Ruff, and diff checks pass. No
model conversion or inference was run.

The exposed ExportedProgram host now lives in
`pytorch_artifact_exporters.py`. Its public-signature wrapper supplies explicit
temporary source-rewrite and final-repair callbacks; the artifact owner retains
metadata, skip, input, single-child, timeout, cleanup, and error ordering. After normalizing those two
callback names, the host AST exactly matches the previous implementation. A
non-native-package fixture proves skip metadata and that no callback or child is
invoked. Artifact and architecture validation passes 68 tests; syntax, Ruff,
and diff checks pass. No model conversion or inference was run. Resume by
isolating the archive algorithms without changing their behavior.

The 46-line stack-trace archive cleanup now has a Torch-free owner in
`pytorch_exported_program_archive.py`, and the artifact host calls it directly.
Its AST exactly matches the former exporter implementation. A real zip fixture
proves recursive removal from `models/model.json` while retaining unrelated JSON
fields and a binary archive entry. The host remains AST-equivalent after
normalizing the removed callback to the direct helper. Artifact and architecture
validation passes 69 tests; syntax, Ruff, and diff checks pass. No model
conversion or inference was run.

The 2,015-line inverse-permute/FX archive optimizer now shares the archive owner.
It is AST-identical to the former exporter implementation after excluding one
local delayed `import torch`; the artifact host is likewise identical after
normalizing the removed callback to its direct helper. Importing the module does
not load Torch, and a missing archive fails before the optional dependency is
requested. The exporter retains only an imported compatibility alias. Artifact
and architecture validation passes 70 tests; syntax, Ruff, py_compile, and diff
checks pass. The full optimizer execution tests remain uncollectable because
Python 3.12 resolves the incompatible Python 3.10 libtorch. No model conversion
or inference was run. Resume with generated-source rewrite/canonicalization
separation; do not alter archive matching logic until a compatible Torch runtime
is available.

Twelve reusable generated-source parsers now live in the Torch-free
`pytorch_source_parser.py`. The move covers nested CSV, outer parentheses,
binary/alignment arguments, cached assignment lines, rank-four shapes, runtime
Concat/`torch.cat`, integer lists, and permutation dimensions. All twelve ASTs,
including the assignment parser's `functools.lru_cache(maxsize=131072)`
decorator, exactly match the prior exporter definitions. Direct fixtures cover
nested expressions, positional/keyword forms, annotations, shapes, and balanced
syntax; parser and architecture validation passes 67 tests. Syntax, Ruff,
pycompile, and diff checks pass. No model conversion or inference was run.
Resume by extracting only cohesive pure helpers used by the raw-export
canonicalizer; do not move or edit the 6,677-line canonicalizer as one block.

Eight additional generated-call parsers and their shared dynamic-batch pattern
now use the same owner. They cover channel-last Gather slices, rank-four shape
expressions, resize, pool argument/assignment forms, tensor split assignments,
softmax, and NHWC-to-NCHW bridge sources. Every moved AST and the ShadowFormer
batch-pattern value exactly matches the exporter checkpoint. Direct fixtures
cover dynamic batch expressions, pool keyword filtering, normalized split axes,
and accepted/rejected bridge permutations. Parser and architecture validation
passes 70 tests; syntax, Ruff, pycompile, and diff checks pass. No model
conversion or inference was run. Resume with the remaining pure reduction and
generated-call decoders, leaving graph-aware canonicalization local until their
ModelIR dependencies can be explicit.

The final 12 pure top-level source decoders now share the parser owner. They
cover `copy_`, aligned assignment, cached permute assignment, local response
normalization, compact pool/resize/softmax input forms, constant Pad, and three
binary-alignment forms. Their ASTs—including the permute parser's
`lru_cache(maxsize=131072)` decorator—exactly match the exporter checkpoint.
Direct fixtures cover positional and keyword calls, copied-buffer kwargs,
constant padding, dynamic/static target shapes, and anchor alignment. Parser and
architecture validation passes 73 tests; syntax, Ruff, pycompile, and diff
checks pass. No model conversion or inference was run. Resume with graph-aware
source canonicalization helpers only after defining explicit ModelIR/query
callbacks; the pure parser extraction is complete.

Four final source-scanning utilities moved with the parser boundary: line
splitting, regex presence/count queries, and balanced prefixed-call extraction.
Their ASTs exactly match the exporter checkpoint, and a nested-call fixture
proves parenthesis-depth handling. Parser and architecture validation passes 74
tests; syntax, Ruff, pycompile, and diff checks pass. No model conversion or
inference was run. The common generated-source parsing/scanning boundary now has
36 functions; resume with a separately characterized source-rewrite family.

The first generated-source rewrite family now has a Torch-free owner in
`pytorch_source_rewrites.py`. It contains channel-first GAP-to-Conv folding,
explicit channel-last GAP output rewriting, and SE scale/binary bridge cleanup.
All three ASTs exactly match the exporter checkpoint. Two direct success
fixtures prove redundant bridge removal and channel-last mean emission, while a
parameterized no-op fixture covers every rewrite. Rewrite and architecture
validation passes 69 tests; syntax, Ruff, pycompile, and diff checks pass. No
model conversion or inference was run. Resume by moving another cohesive source
rewrite family only after adding its direct success/no-op characterization.

The next affine/GAP rewrite family is now owned by the same Torch-free module.
Channel-last Mul/Add affine chains feeding Conv and channel-last GAP means for
rank-3/rank-4 permute forms moved without semantic changes; both function ASTs
exactly match the previous exporter checkpoint. Focused success fixtures cover
compact affine assignments and helper/functional permutes, and both paths retain
explicit unmatched-source no-op coverage. Rewrite and architecture validation
passes 73 tests; syntax, Ruff, pycompile, diff, and AST-equivalence checks pass.
No model conversion or inference was run. Resume with another cohesive pure
source-rewrite family; do not start broad canonicalizer extraction as one block.

Five more graph-independent layout rewrites moved to
`pytorch_source_rewrites.py`: boundary transpose/Conv folding, duplicate
permute-chain collapse, public bridge-alias inlining, channel-last PReLU bridge
folding, and rank-4 reshape/permute/Conv folding. All five ASTs match the prior
exporter checkpoint exactly. One direct success fixture and one no-op path per
rewrite keep the extraction locally testable. Rewrite and architecture
validation passes 83 tests; syntax, Ruff, pycompile, diff, and AST-equivalence
checks pass. Graph-aware GatherND boundary repair remains in the exporter by
design. No model conversion or inference was run. Resume with the remaining
pure GAP/hardsigmoid/binary rewrite helpers before changing graph-aware repair
policy.

The channel-first hard-sigmoid gate/Conv rewrite also moved to the Torch-free
source-rewrite owner. Its 441-line implementation is AST-identical to the prior
exporter definition, including safety checks for later consumers and function
boundaries. A direct classifier-gate fixture fixes the successful rewrite and
the shared no-op matrix covers unmatched source. Rewrite and architecture
validation passes 85 tests; syntax, Ruff, pycompile, diff, and AST-equivalence
checks pass. No model conversion or inference was run. Resume with the remaining
channel-last binary bridge rewrite; defer GAP/Conv input repair until its shared
shape-normalization dependency has a clear owner.

The callback-driven channel-last binary bridge-chain rewrite now lives in
`pytorch_source_rewrites.py` too. Its 399-line AST is identical to the exporter
checkpoint, while the existing callbacks keep local-name allocation and
constant-layout materialization outside the pure rewrite owner. A direct
Conv-input-chain fixture and explicit unmatched-source no-op test fix the
boundary. Rewrite and architecture validation passes 87 tests; syntax, Ruff,
pycompile, diff, and AST-equivalence checks pass. No model conversion or
inference was run. The remaining nearby rewrites are graph-aware or depend on
shared shape-normalization policy and should not be moved mechanically.

Rank-4 layout hinting and CF/NHWC shape normalization now have a Torch-free
shared owner in `pytorch_shape_policy.py`. All three ASTs match the prior
exporter checkpoint exactly. Fourteen direct cases characterize rank rejection,
preferred-channel and singleton-channel inference, ambiguous layouts, CF/NHWC
conversion, and `out_hw` preservation. Shape-policy and architecture validation
passes 79 tests; syntax, Ruff, pycompile, diff, and AST-equivalence checks pass.
No model conversion or inference was run. This removes the ownership blocker
for moving channel-last GAP/Conv input repair without introducing an exporter
import cycle.

Channel-last GAP/Conv input repair now uses the shared rank-4 shape policy and
lives in `pytorch_source_rewrites.py`. Its 236-line AST matches the prior
exporter definition exactly. Direct tests cover bridge insertion and scalar-axis
mean rejection, while the common no-op matrix covers unrelated source. Shape
policy, rewrite, and architecture validation passes 105 tests; syntax, Ruff,
pycompile, diff, and AST-equivalence checks pass. No model conversion or
inference was run. At this checkpoint, graph-aware GatherND repair remained
exporter-owned pending an explicit ModelIR/query boundary.

The GatherND boundary rewrite and its 19-line ModelIR-backed shape query now
have that explicit boundary in `pytorch_source_graph_rewrites.py`. The module is
Torch-free but intentionally graph-aware; it is separate from the pure string
rewrite owner. Both ASTs match the prior exporter checkpoint exactly. Direct
tests cover index-depth shape inference, required boundary permutation,
duplicate-permute collapse, and the already-correct no-op path. Graph-rewrite
and architecture validation passes 70 tests; syntax, Ruff, pycompile, diff, and
AST-equivalence checks pass. No model conversion or inference was run. The
exporter now imports the graph query and ordered rewrite instead of owning them.

Generated forward-line pruning now lives with the pure source rewrites. The
44-line backward-liveness implementation is AST-identical to the exporter
checkpoint and takes explicit input/output variable names. Direct cases cover
unreachable assignment removal and multi-assignment dependency preservation.
Rewrite and architecture validation passes 94 tests; syntax, Ruff, pycompile,
diff, and AST-equivalence checks pass. No model conversion or inference was
run.

Generated-package metadata serialization now lives in
`pytorch_export_support.py`. Tensor metadata, recursive NumPy value conversion,
and the complete payload builder are AST-identical to the prior exporter
checkpoint. Direct tests fix nested ndarray/scalar conversion, constant-buffer
flags, operator options and axis semantics, and restoration of public ONNX
shape signatures and layouts. Artifact-support and architecture validation
passes 76 tests; syntax, Ruff, pycompile, diff, and AST-equivalence checks pass.
No model conversion or inference was run.

PyTorch capability selection now has a single Torch-free owner in
`pytorch_capabilities.py`. The direct-codegen registry composes the existing
emitter module/unary/binary declarations, while runtime kernels, explicit CUSTOM
rejection, and normalized unsupported-op diagnostics retain their established
contracts. The registry AST and all three function ASTs match the prior exporter
checkpoint. Direct tests cover defensive query copies, a declared direct op,
unknown-op diagnostics, and CUSTOM rejection. Capability and architecture
validation passes 71 tests; syntax, Ruff, pycompile, diff, and AST-equivalence
checks pass. No model conversion or inference was run.

Generated PyTorch naming policy now lives in the Torch-free
`pytorch_naming.py`. Tensor variables, buffer attributes, storage names,
keyword/digit sanitization, semantic suffixes, bounded long-name hashing, and
deterministic collision handling share one owner. All nine function ASTs and
four policy-constant ASTs match the prior exporter checkpoint. Direct tests
cover storage and variable collisions, NCHW/NHWC suffix handling, long sibling
names, excluded buffers, keywords, digits, and empty names. Naming and
architecture validation passes 73 tests; syntax, Ruff, pycompile, diff, and
AST-equivalence checks pass. No model conversion or inference was run.

Native PyTorch codegen value policy now lives in the Torch-free
`pytorch_codegen_values.py`. Small inline-constant eligibility, nested and
non-finite Python literals, scalar literals, reversed/permuted Torch padding,
dtype spelling, and Conv-block fused activations share one owner. All seven
function ASTs match the prior exporter checkpoint. Thirteen direct tests fix
size/rank/dtype boundaries, padding order, unsupported dtype diagnostics, every
supported fused activation, and LeakyReLU alpha handling. Value-policy and
architecture validation passes 82 tests; Ruff, diff, and AST-equivalence checks
pass. No model conversion or inference was run.

Special Reshape layout planning now shares the Torch-free
`pytorch_shape_policy.py` owner with rank-four layout hinting and CF/NHWC shape
normalization. Its function AST matches the prior exporter checkpoint. Seven
new direct cases fix 4D-to-3D, 3D-to-4D, singleton rank-four permutations,
NCHW-to-NCDHW, high-rank channel expansion, unmatched input, and missing-shape
behavior. Shape-policy and architecture validation passes 90 tests; Ruff,
diff, and AST-equivalence checks pass. No model conversion or inference was
run.

Native indexing expression generation now lives in the Torch-free
`pytorch_indexing_codegen.py`. Slice, static and symbolic StridedSlice, static
and dynamic Gather, fused Gather-plus-Reshape, suffix-flatten recognition,
singleton-axis-drop recognition, and guarded CRD-to-DCR Gather elision share
one owner. All nine function ASTs match the prior exporter checkpoint. Nine
direct tests fix masks, bounds, dynamic shape selection, scalar/multidimensional
indices, invalid configurations, dynamic batch preservation, and exclusive
DepthToSpace consumer guards. Indexing and architecture validation passes 79
tests; Ruff, diff, and AST-equivalence checks pass. No model conversion or
inference was run.

Native model-file generation now constructs one shared `ModelIRGraphIndex` and
passes its producer/consumer maps into codegen. The exporter-local full graph
scan was removed; no codegen path mutates either map. Architecture validation
passes 71 tests and fixes the single constructor plus read-only contract. No
model conversion or inference was run.

Generated-package import and native state-dict reconciliation now live in
`pytorch_state_dict_support.py`. The module owns sanitized generated-module
loading, stale-module eviction, dtype/shape preparation, key reconciliation,
and ModelIR constant mapping while retaining a function-local lazy Torch import.
All three function ASTs match the prior exporter checkpoint. Six direct tests
use a Torch-free tensor double to fix exact-shape, reshape, error, package-load,
key-mismatch, and missing-data behavior. State-dict and architecture validation
passes 78 tests; Ruff, diff, and AST-equivalence checks pass. No model conversion
or inference was run.

Generated package scaffolding now lives in the Torch-free
`pytorch_package_sources.py`. Common initializer/runtime files, wrapper model
source, native runtime assembly, and the idempotent Pool2D channel-last recovery
patch share one owner across native, TFLite-backed, and SavedModel-backed
packages. All four function ASTs match the prior exporter checkpoint. Six
direct filesystem/source tests fix default and explicit runtime content, public
wrapper methods, ordered/idempotent patching, no-op boundaries, and annotation
normalization. Package-source and architecture validation passes 79 tests;
Ruff, diff, and AST-equivalence checks pass. No model conversion or inference
was run.

Runtime-wrapper capability selection and its explicit `ONNX_SLICE` custom-code
allowance now live in `pytorch_capabilities.py`; direct module-attribute base
names now live in `pytorch_naming.py`. Both function ASTs and the custom-code
constant AST match the prior exporter checkpoint. Three new direct cases fix
runtime/custom rejection and established/future attribute spelling. Capability,
naming, and architecture validation passes 85 tests; Ruff, diff, and
AST-equivalence checks pass. No model conversion or inference was run.

Runtime-wrapper artifact generation now lives in
`pytorch_runtime_wrapper_exporter.py`. It uses the shared capability, metadata,
naming, and package-source owners and retains a function-local Torch import for
state serialization. Its function AST matches the prior exporter checkpoint.
Two direct tests use a fake Torch module to fix package files, metadata, storage
names, state serialization, and rejection before output creation. Wrapper and
architecture validation passes 76 tests; Ruff, diff, and AST-equivalence checks
pass. No model conversion or inference was run.

TFLite-backed and SavedModel-backed PyTorch package exports now live in
`pytorch_artifact_exporters.py`; their public-boundary-only metadata builders
live in `pytorch_export_support.py`. The paths copy only the requested backing
artifact and reject missing inputs before creating output. All four function
ASTs match the prior exporter checkpoint. Five new direct cases fix dynamic
boundary metadata, constant exclusion, TFLite-only and SavedModel-only output,
stale SavedModel replacement, and missing-source no-op behavior. Artifact and
architecture validation passes 90 tests; Ruff, diff, and AST-equivalence checks
pass. No model conversion or inference was run.

Fallback package preference now lives in the Torch-free
`pytorch_package_selection.py`. Recurrent/control and length-input guards,
transpose-convolution and channel-first Softmax signals, rank-three detection
counts, and all large NHWC thresholds have one owner shared by TFLite and
SavedModel fallback. Both function ASTs match the prior exporter checkpoint.
Fourteen direct cases characterize every guard and threshold family. Selection
and architecture validation passes 89 tests; Ruff, diff, and AST-equivalence
checks pass. No model conversion or inference was run.

Reference ONNX public-boundary inference now lives in
`pytorch_onnx_artifact_support.py`. Transpose permutation decoding,
layout-preserving passthrough walks, public input/output layout inference, and
batchless rank-three Squeeze/Unsqueeze detection share one owner. All four
function ASTs match the prior exporter checkpoint. Six direct ONNX-helper cases
fix input/output chains, fan-out rejection, batchless boundaries, unsupported
nodes, and missing graphs. Boundary inference and architecture validation
passes 83 tests; Ruff, diff, and AST-equivalence checks pass. No model conversion
or inference was run.

Reference public-boundary metadata merge now uses that same ONNX artifact
owner. It restores public names and shape signatures, delegates layout bridge
materialization to the shared validation pass, preserves batchless boundary
metadata, and forces recurrent rank-three boundaries to NWC. Its function AST
matches the prior exporter checkpoint. Two new direct ModelIR cases fix the
basic public contract and recurrent feature-last override. Boundary and
architecture validation passes 85 tests; Ruff, diff, and AST-equivalence checks
pass. No model conversion or inference was run.

Single-op ONNX StringNormalizer package fallback now lives in the dedicated
Torch- and TensorFlow-free `pytorch_string_normalizer_exporter.py`. Attribute
decoding and wrapper-package generation reuse the shared metadata and package
scaffolding contracts; invalid graphs are rejected before output creation.
Both function ASTs match checkpoint `12535c1` exactly. Five direct ONNX-helper
and filesystem cases plus architecture validation pass 83 tests; Ruff, diff,
and AST-equivalence checks pass. No model conversion or inference was run.

ExportedProgram's pure direct-Conv channel-first Add-target repair now lives
with the other Torch-free transforms in `pytorch_source_rewrites.py`. It still
requires a declared Conv block, recorded input channels, the exact Add/ReLU
chain, and a nearby direct Conv consumer before changing a static target. Its
function AST matches checkpoint `14dfd85` exactly. A positive rewrite and three
unsafe no-op cases plus source-rewrite and architecture validation pass 110
tests; Ruff, diff, and AST-equivalence checks pass. No model conversion or
inference was run.

Native direct-codegen validation and its fallback-error classifier now live
with the Torch-free registry in `pytorch_capabilities.py`. Runtime-supported
ops remain distinct from the smaller direct emitter set; for example `WHILE`
continues to enter fallback instead of native source generation. Both function
ASTs match checkpoint `0515e64` exactly. Capability and architecture validation
pass 85 tests; Ruff, diff, and AST-equivalence checks pass. No model conversion
or inference was run.

Two unreferenced exporter shims were removed: the old full direct-module
attribute-name helper and a Torch ONNX warning helper. Generated naming already
uses `pytorch_naming.py`, while the warning suppression actually used by Dynamo
export remains inside the isolated artifact child. Architecture validation
passes 78 tests and fixes both ownership boundaries. No model conversion or
inference was run.

Dependency-safe split-point discovery no longer rescans every operator edge
for every candidate boundary. One producer/consumer pass builds backward-edge
invalid ranges and forward crossing-set start/end events, preserving the exact
legacy report. Two hundred fixed-seed random graphs, including non-topological
edges and duplicate output names, match the former algorithm exactly; a
256-op instrumentation fixture proves each input/output edge list is read once.
All eight focused split-planner tests pass. A local synthetic 2,000-op chain
microbenchmark improved from 0.405299s to 0.010556s (38.4x) on this host. No
model conversion or inference was run.

The split planner now keeps that topology in one shared `ModelIRGraphIndex`
through candidate discovery, range validation, and manifest-edge construction
instead of rebuilding producer maps three times. Standalone helper calls still
construct their own index unless a caller supplies one. A focused constructor
instrumentation test proves one index build for the complete planner; all nine
split-planner tests pass. No model conversion or inference was run.

Partition input/output tensor collection now uses insertion-ordered `seen`
sets instead of repeated list membership. First-seen ordering, empty-name
filtering, and duplicate suppression are unchanged. All ten split-planner tests
pass. On a local synthetic 2,000-op partition, five-run median collection time
changed from 0.013159s to 0.000176s for outputs (74.7x) and from 0.013017s to
0.000175s for inputs (74.3x). No model conversion or inference was run.

Partition candidate construction now resolves tensors consumed after the
candidate range through the shared consumer index instead of scanning the
complete graph suffix. The final split artifact writer likewise shares one
index across all partitions. Planner and stubbed-writer instrumentation prove
one index construction in each complete flow; all eleven focused split-planner
tests pass. No model conversion or inference was run.

Boundary crop preflight now builds the original runtime-input set once instead
of once per requested input, and removes an unused pass over every retained
operator output. A direct intermediate-input/intermediate-output crop fixes the
unchanged two-op result; crop ownership plus architecture validation pass 90
tests. No model conversion or inference was run.

Custom-op artifact result metadata now comes from one pass in the new
TensorFlow- and Torch-free `artifact_metadata.py`. The legacy raw code list and
trimmed/deduplicated node details retain their distinct normalization and sort
contracts. A direct normalization fixture and 256-op access instrumentation,
plus architecture validation, pass 81 tests. No model conversion or inference
was run.

Split size-estimation candidates now borrow source NumPy constant buffers
read-only instead of copying every weight at every binary-search probe. The
public partition-builder default and final split writer still create
independent buffers. Identity and independence fixtures bring the focused
split-planner suite to 13 passing tests. No model conversion or inference was
run.

All three op-coverage write call sites—preprocess failure, lowering failure,
and successful export—now sit under the explicit request guard. An unrequested
report no longer calls even the no-op wrapper. An AST ownership/gating test
checks every call ancestor; architecture validation passes 80 tests. No model
conversion or inference was run.

SavedModel export progress previously advanced once even when the artifact was
not requested, while its progress label was correctly omitted. The advance now
lives inside the SavedModel request guard. An AST gate fixes that ownership and
architecture validation passes 81 tests. No model conversion or inference was
run.

TFLite precision artifact preparation no longer clones its terminal ModelIR a
second time unconditionally. Float16 writes directly from its only precision
clone. Float32 does the same for the normal direct path and split artifacts,
but retains an isolated write clone when a non-split SavedModel or PyTorch
exporter will later consume the pre-serialization float32 IR. An eight-case
artifact/split matrix verifies the exact isolation policy, independent-buffer
checks prove the retained clone boundary, normalized fingerprints prove that
single and legacy double precision clones have identical content, and an AST
gate fixes the terminal float16 reuse. Focused artifact-preparation, writer,
and architecture validation passes. On a local synthetic 2,000-op ModelIR,
seven-run medians changed from 0.071958s to 0.034205s for float32 (2.10x) and
from 0.068364s to 0.035449s for float16 (1.93x); traced clone-stage peak memory
dropped 50% in both cases. No model conversion or inference was run.

Precision and quantization artifact lifetimes are now bounded at their actual
last consumers. Float32/float16 write indices and serialization IRs, the
pre-export float32 source, dynamic-range IR, all four strict-integer variant
IRs/results, and calibration samples are explicitly released before the next
large artifact is built. Calibration ranges and report state are retained only
until the strict-integer JSON report is written. An AST lifecycle gate covers
all sixteen large local objects and rejects any release before a later load.
Artifact-preparation, writer, and architecture validation passes 94 tests. A
seven-run synthetic 2,000-op/six-artifact lifetime benchmark reduced traced
peak memory from 21.21 MiB to 3.72 MiB (82.4%) when applying the new sequential
release boundary. No model conversion or inference was run.

Precision and quantization cloning now share the `OperatorIR`/`TensorIR`
element-copy contract in `ir.py`. The helpers centrally preserve shape
signatures, constant-buffer independence, variable state, both quantization
representations, logical/physical layout, axis semantics, versions, and ONNX
provenance. Float16/float32 wrappers keep layout normalization, metadata, and
recursive subgraphs; the quantization root clone intentionally keeps its
characterized raw-layout and empty metadata/subgraph behavior. Four focused
contract tests plus quantization and precision selections pass 65 tests. A
fixed-seed 100-graph differential check executed all three pre-checkpoint
functions from Git and matched every new normalized ModelIR fingerprint and
metadata payload exactly. No model conversion or inference was run.

Strict-integer report payload generation is now owned by
`_StrictQuantizationReporter` and follows `return_report`. The two reported
INT8 variants retain the exact calibration JSON schema and insertion order;
model-only public calls and both INT16-activation variants skip tensor-range,
qparam, and operator report serialization entirely. Two gating/equivalence
tests plus the strict quantization suites pass 49 tests. An in-memory execution
of the pre-checkpoint module matched ModelIR fingerprints and requested report
payloads exactly for float/full IO crossed with INT8/INT16. On a synthetic
2,000-op INT16 full-integer graph, seven-run median build time changed from
0.171745s to 0.142100s (1.21x), and traced peak memory from 5.42 MiB to
2.85 MiB (47.5%). No model conversion or inference was run.

Strict full-integer activation dtype selection now constructs input/output name
sets once per variant and reuses them for every tensor decision. The former
standalone helper rebuilt both sets at each call. The focused ownership gate
fixes one construction per boundary and the strict quantization selection
passes 50 tests. Against the immediately preceding checkpoint, an 11-run
synthetic 2,000-op INT16 full-integer median improved from 0.013129s to
0.011785s (1.11x). No model conversion or inference was run.

Strict-model validation now collects used tensor names while performing its
existing supported-op loop rather than scanning every operator immediately
before that loop. Raw tensor-name handling and sorted tensor validation remain
unchanged. The focused ownership gate and strict quantization selection pass
51 tests. Against checkpoint `57c61a5`, a 15-run synthetic 2,000-op INT16
full-integer median improved from 0.012662s to 0.012108s (1.05x). No model
conversion or inference was run.

Strict-integer `ModelIRGraphIndex` construction is now lazy. Float-IO and mixed
boundary-dtype variants still build and differentially update exactly one
index; the common full-integer and full-integer INT16 variants skip it when no
boundary rewiring or pre/post insertion is needed. A focused zero-index fixture
and a mixed-output one-index fixture join the existing one-index float-IO
fixture; the strict quantization selection passes 53 tests. Four float/full ×
INT8/INT16 configurations plus two mixed boundary configurations match the
eager-index checkpoint's ModelIR fingerprints and requested reports exactly.
Against checkpoint `28763e6`, a 15-run synthetic 2,000-op full-INT16 median
improved from 0.012144s to 0.010351s (1.17x). No model conversion or inference
was run.

Dynamic-range graph indexing is now lazy as well. Conv, DepthwiseConv, and
FullyConnected kernel-only weight quantization performs no topology mutation
and constructs no index; quantized elementwise constants still create one
index at the first `DEQUANTIZE` insertion and share it for later rewires. The
zero-index kernel fixture and existing one-index shared-constant fixture bring
the strict/dynamic quantization selection to 54 tests. Kernel-only and
constant-Dequantize ModelIR fingerprints match checkpoint `dca871b` exactly.
On a synthetic 2,000-FullyConnected graph, a 15-run median changed from
0.053744s to 0.050637s (1.06x). No model conversion or inference was run.

Split planning now materializes the shared index's first-producer map once and
passes it through split-point discovery, range validation, and manifest-edge
construction. Partition candidate dead-branch liveness still preserves the
same pruning contract, but a range whose complete operator stream is required
reuses its first input/output/boundary collections; only an actually pruned
range repeats them. One-scan instrumentation and the complete focused split
selection pass 22 tests. The full 2,000-op/20-partition plan report matches
checkpoint `03d0177` exactly; a 15-run median changed from 0.120941s to
0.113134s (1.07x). No model conversion or inference was run.

Fully required split ranges now identify the complete sorted-unique index set
by cardinality instead of comparing every index against `0..N-1`.
`ModelIRGraphIndex.has_consumer_at_or_after()` also exposes its maintained
sorted-consumer invariant, allowing partition boundary-output discovery to
answer each suffix query from the final consumer index without an `any()`
generator scan. Tail-query behavior, complete-range scan reuse, and the split
selection pass 23 tests; the related split/core selection passes 26 tests. The
2,000-op/20-partition report remains identical to checkpoint `4943bcb`; a
15-run median changed from 0.119641s to 0.114021s (1.05x). No model conversion
or inference was run.

Split partition construction now delegates both operator and tensor element
copies to the common `ir.py` clone contract. It explicitly selects copied or
borrowed NumPy buffers according to the existing `copy_tensor_data` policy,
preserves raw layouts, and retains the established shared immutable
quantization object. This removes another duplicated field list without
changing split planning or artifact ownership. One hundred fixed-seed random
partition graphs under both buffer policies match checkpoint `61b19be` byte
for byte at the ModelIR fingerprint level. The focused split, clone, artifact
preparation, and architecture selection passes 119 tests; Ruff, syntax, and
diff checks pass. No model conversion or inference was run.

Boundary crop preflight now defers recursive nested-subgraph tensor-name
collection until a requested boundary is absent from the top-level graph.
Producer discovery is fused with forward reachability, and kept-operator
validation, required-tensor collection, and materialization share one retained
range traversal. Cropped operators delegate the structural field copy to the
common clone contract while retaining deep isolation for nested options and
axis semantics. Two hundred fixed-seed crop graphs match checkpoint `433a2c6`
byte for byte at the ModelIR fingerprint level. Focused crop/split/clone tests
pass 32 tests. An 11-run synthetic 2,000-op median improved from 0.039883s to
0.037330s (1.07x). No model conversion or inference was run.

The final ModelIR invariant gate now accepts an optional current
`ModelIRGraphIndex`. Float32 and float16 artifact preparation reuses the index
already built and differentially maintained by terminal ScatterND and binary
constant folding instead of rebuilding the same topology immediately before
serialization. Callers that do not own an index keep the previous behavior.
Two hundred fixed-seed graphs produce identical validation problem lists with
eager and shared indexes, and the focused core/artifact selection passes 41
tests. A 21-run synthetic 2,000-op validation median improved from 0.002606s to
0.001233s (2.11x). No model conversion or inference was run.

Requested split artifacts now create one caller-owned source
`ModelIRGraphIndex` and reuse it through final source validation, contiguous
partition planning, and partition-file writing. The default non-split path
does not create this index, while standalone planner/writer calls keep their
self-contained fallback. One hundred fixed-seed graphs produce identical
eager and shared-index split reports. Focused split/artifact tests pass 36
tests; instrumentation proves one source index across planning and writing.
An 11-run synthetic 2,000-op validation+planning median changed from 0.120197s
to 0.118535s (1.01x). No model conversion or inference was run.

Split thresholds and quantization calibration controls now have one
TensorFlow- and Torch-free owner in `artifact_preparation.py`. Resolution is
guarded by the normalized artifact plan: default direct conversion reads no
split or quantization option/environment value, while requested artifacts keep
the previous defaults and numeric conversions. The quantization mapping is
immutable, and the central exporter no longer owns a duplicate resolver or
environment-key list. Focused artifact/core/architecture validation passes
126 tests, including rejecting option access on the unrequested path. No model
conversion or inference was run.

The last large direct-module block, fused-module emission, has moved to the
Torch-free emitter. It preserves folded input adapters, legacy NHWC Conv input/
output fallback, raw NCHW/NCDHW aliases, public output correction, omitted
channel-last aliases, materialized layout bridges, and generic output handling.
Direct fixtures cover both planned folded input and implicit NHWC fallback.
Emitter and architecture validation passes 84 tests; syntax, Ruff, and diff
checks pass. The remaining direct-module function is about 171 lines and is a
statement-free ordered dispatcher. No model conversion or inference was run.

Conv2D and DepthwiseConv2D direct-module emission now live in the focused
emitter. The move preserves folded channel-first inputs, shape-based no-permute
guards, degenerate 1x1 alias reuse, planned pre-permutations, explicit F.pad,
direct/runtime helper selection, NCHW aliases, public-output correction, and
activation order. Direct tests cover a folded NHWC bridge and padded runtime
fallback. Emitter and architecture validation passes 82 tests; syntax, Ruff,
and diff checks pass. The dispatcher shrank from 374 to about 255 lines. No
model conversion or inference was run.

Channel-first normalization now owns one graph index for its complete mutable
layout phase. Feature-last collection, both friendly-layout worklists,
Transpose cleanup, ATAN2-to-ATAN canonicalization, recurrent orphan repair,
Softmax validation, and the final residual-Transpose check share that index.
ATAN2 input/type mutation and recurrent input repair update it differentially,
so the former initial and final ad hoc producer/consumer maps and intermediate
index rebuilds are gone. The focused Torch-free suite passes 89 tests. The full
Torch-dependent exporter test remains uncollectable in this environment
because Python 3.12 resolves the incompatible Python 3.10 libtorch; no model
conversion or inference was run.

Successful Torch-free native preparation now carries the channel-first
normalizer's current graph index into public-boundary bridge insertion and
shape alignment. This removes the second index build over the same prepared
graph while the public normalizer still returns only ModelIR; the
layout-agnostic fallback retains its independent index because it constructs a
different graph. One hundred fixed-seed preparation graphs match checkpoint
`1429e10` byte for byte at the ModelIR fingerprint level. The focused
normalization/architecture selection passes 85 tests, including the distinct
layout-agnostic fallback index path, and a seven-run
synthetic 2,000-op median improved from 0.045557s to 0.043769s (1.04x). The
known Python 3.12/Python 3.10 libtorch ABI mismatch still prevents collection
of the full Torch-dependent exporter suite. No model conversion or inference
was run.

Native compatibility canonicalization now classifies root op families once and
invokes static-WHILE, counter-WHILE, and recurrent rewrites only when their
families are present. WHILE expansion is followed by one reclassification so
recurrent operators introduced from a body remain visible. Direct recurrent
capability selection likewise replaces its former `any` plus `all` scans with
one short-circuiting traversal. One hundred ordinary graphs plus the canonical
static and counter-bounded fixtures match checkpoint `0a9ce0f` byte for byte at
the prepared ModelIR fingerprint level. The focused normalization, recurrent,
control-flow, and architecture selection passes 99 tests. On a synthetic
2,000-op irrelevant graph, the compatibility-only preflight median improved
from 0.000315s to 0.000044s (7.19x); end-to-end preparation remains dominated
by deep copy and layout work. No model conversion or inference was run.

Native package capability validation now combines explicit root-CUSTOM
rejection and recursive supported-op validation. Production and debug exports
therefore scan the prepared root operator list once instead of calling two
validators, while the individual validators remain compatibility imports. The
combined path preserves root-CUSTOM error precedence and the established
generic unsupported-op diagnostic for CUSTOM operators found only in a
subgraph. Five hundred fixed-seed root/subgraph capability combinations match
the former two-validator outcome, exception type, and message exactly. The
focused capability and architecture selection passes 90 tests, and a 101-run
synthetic 2,000-op median improved from 0.000070605s to 0.000035143s (2.01x).
No model conversion or inference was run.

Fallback package selection now collects root op types, structural counts, and
Softmax candidates in one operator-list traversal instead of repeatedly
rescanning the same list for each guard and threshold. Recurrent/control,
length-input, transpose-convolution, channel-first Softmax, and large-graph
decisions retain their established evaluation order. Five hundred fixed-seed
selection graphs match checkpoint `064d927` exactly, and the focused selection
and architecture suites pass 95 tests. A 101-run synthetic 2,000-op median
improved from 0.000256567s to 0.000105230s (2.44x). The final unconditional
TFLite fallback also no longer evaluates the same preference policy before two
identical artifact writes. No model conversion or inference was run.

Feature-last layout application now enumerates only the indexed producers of
preserved tensors and reuses that graph-ordered candidate list for both the
initial layout rewrite and final Reshape contract restoration. Duplicate
producers retain their complete graph-order behavior, and all three normalizer
invocations reuse the existing mutable `ModelIRGraphIndex`. Five hundred
fixed-seed mutation graphs match checkpoint `41ae336` exactly, and the focused
layout, normalization, and architecture suites pass 111 tests. On a
sparse-preserve synthetic 2,000-op graph, the shared-index application median
improved from 0.001540002s to 0.000013905s (110.75x). No model conversion or
inference was run.

Kernel-weight identification and filter physicalization now share one
Conv2D/depthwise/transpose-Conv2D/Conv3D op-family declaration. The normalizer
passes its existing graph index to both stages, so the pre-permutation
exclusion set enumerates only indexed family members instead of scanning every
operator. Five hundred fixed-seed kernel-weight sets match checkpoint
`5475221` exactly, and the focused layout, normalization, and architecture
suites pass 116 tests. On a synthetic 2,000-op irrelevant graph, the
shared-index collection median improved from 0.000041247s to 0.000001075s
(38.37x). No model conversion or inference was run.

Conv2D/depthwise/transpose-Conv2D and Conv3D filter physicalization now lives
in the Torch-free layout owner and enumerates only those op families through
the normalizer's shared graph index. Shared weight buffers retain the one-
rewrite contract, and shape/signature updates remain unchanged. A focused
shared-filter fixture brings the Torch-free suite to 90 tests; no model
conversion or inference was run.

Residual layout-Transpose validation and the Reshape-only helper now live in
`passes/pytorch_compat.py`. The normalizer enumerates only the shared index's
Transpose family and reuses its consumer table; no final whole-graph scan is
required. Focused failure and Reshape-only exception fixtures bring the
Torch-free suite to 92 tests, with the established error text unchanged. No
model conversion or inference was run.

The 181-line layout-sensitive op rewrite has moved from the exporter to the
Torch-free layout owner and now enumerates only its affected op families from
the shared index. Axis options/constants, Slice vectors, pad matrices,
Transpose permutations, transpose-convolution output shapes, and Reshape
targets retain their established ordering and shared-constant one-rewrite
contract. Reshape target/name policy moved to `pytorch_layout_utils.py` for
reuse by codegen. The 79-test focused selection passes, and nine synthetic
op-family graphs match the implementation at checkpoint `09ed6b6` exactly. No
model conversion or inference was run.

Reshape target synchronization now follows the shared target policy in the
Torch-free layout owner and queries only the normalizer's indexed Reshape
family. A dynamic-signature INT64 fixture verifies target permutation, option
updates, constant dtype preservation, and single-index reuse. The same focused
selection now passes 80 tests; no model conversion or inference was run.

The 125-line channel-first exportability validator has moved beside its
Softmax and layout-island helpers. It enumerates only indexed layout-sensitive
families and shares the normalizer's index for attention/sandwich edge checks.
Focused unknown-layout rejection and attention-Softmax exception fixtures pass;
the layout/architecture selection passes 78 tests. No model conversion or
inference was run.

Public boundary shape/layout reconciliation and recurrent-context detection
have moved to the same layout owner. The normalizer reuses its graph index for
recurrent op-family detection; the fallback preparation path preserves direct
detection. A dynamic recurrent rank-three fixture verifies the NWC default and
boundary signature concretization. The layout/architecture selection passes 79
tests; no model conversion or inference was run.

The complete 185-line channel-first orchestration has moved from the exporter
to the new Torch-free `passes/pytorch_normalization.py`. It remains the owner of
the single deep copy and single `ModelIRGraphIndex`, and calls the extracted
layout/compatibility/recurrent steps in the unchanged order. Public-output
Transpose inspection now uses the same family index. A direct normalization
fixture proves source/result independence, operator and ONNX provenance,
channel-first metadata, and exactly one index refresh without importing Torch.
The six-file focused selection passes 98 tests; no model conversion or
inference was run.

`prepare_model_ir_for_native_pytorch()` now lives in the Torch-free
normalization module as well. Static and counter-bounded WHILE rewrites,
recurrent canonicalization, channel-first normalization, public bridge
insertion, and boundary alignment retain their established order. Bridge and
alignment steps share one preparation-boundary graph index; recursive subgraph
op-type collection and the layout-agnostic fallback policy moved with the
orchestration. Direct preparation tests verify two total index refreshes,
dynamic boundary signatures, public layout metadata, source immutability, and
subgraph op discovery. The normalization/architecture selection passes 59
tests; no model conversion or inference was run.

The native unary code emitter and its complete expression table now have one
Torch-free owner in `pytorch_emitters.py`. The exporter supplies its existing
tensor-expression, alias, shape, alignment, and local-name policies as
callbacks, so generated statements and runtime-helper imports retain the
established contract while the central exporter loses 123 implementation
lines. Direct tests fix channel-first output, materialized NHWC bridge,
fallback alignment, LeakyReLU option, and unsupported-op no-mutation behavior.
An architecture gate fixes both the function and table ownership and includes
the new module in the TensorFlow-import boundary. Emitter and architecture
validation passes 61 tests. Syntax, Ruff, and diff checks pass; no model
conversion, inference, dependency change, or parallel process was involved.

ReverseV2, ExpandDims, Squeeze, Pack, Unpack, and Split code emission has now
moved to the same Torch-free emitter owner. The function signature and imported
global name remain unchanged, so the stored native-codegen pipeline requires no
template rewrite. Constant integer-vector decoding moved from the exporter to
`pytorch_codegen_utils.py` and remains available to all existing generated-code
helpers through the exporter's imported alias. Direct tests cover constant
negative-axis normalization, runtime axes, ordered multi-axis Squeeze, Pack,
Unpack, and Split output construction. Emitter and architecture validation
passes 63 tests; syntax, Ruff, and diff checks pass. No model conversion or
inference was run.

The native binary expression table and all statement-emission branches now
live in `pytorch_emitters.py`. This includes truncating integer division, bool
scalar dtype/device coercion, channel-first aliases, fused activation order,
NHWC materialization, runtime-shape passthrough, and both anchored and symmetric
broadcast alignment. The generated pipeline keeps its original call signature:
the exporter retains only a delegation adapter which injects the existing
broadcast target-shape resolver, and an AST gate rejects statement emission in
that adapter. Direct binary fixtures cover the highest-risk branches. Emitter
and architecture validation passes 66 tests; syntax, Ruff, and diff checks
pass. No model conversion or inference was run.

Native Transpose emission now has the same direct Torch-free owner. It imports
the existing permutation, inconsistent-layout, and residual-Reshape policies
from their focused layout/compatibility modules rather than retaining exporter
copies. Stale bridge elision, folded channel-first expressions, alias-only
channel-last bridges, and explicit runtime permutation paths have direct
fixtures. The exporter keeps only the imported emitter name expected by the
stored generated pipeline. Emitter and architecture validation passes 68 tests;
syntax, Ruff, and diff checks pass. No model conversion or inference was run.

The approximately 496-line direct-module dispatcher is now being split by real
operator families. Unidirectional RNN, 15/24-input unidirectional LSTM, and
29/48-input bidirectional LSTM emission moved first. The focused emitter uses
the shared recurrent state-index policy and preserves aligned output source and
runtime imports. Direct fixtures cover RNN, 15-input LSTM, and 29-input
bidirectional LSTM state arguments plus unrelated-op no-mutation behavior.
Emitter and architecture validation passes 73 tests; syntax, Ruff, and diff
checks pass. No model conversion or inference was run.

FullyConnected and PReLU have also left the direct-module dispatcher.
FullyConnected preserves direct-call then fused-activation order. PReLU retains
constant/shape-derived parameter counts, channel-last-to-channel-first parameter
axis permutation and restoration, the shape-preserving fast path, and the
existing alignment callback. Direct tests cover fused FullyConnected, NHWC
multi-parameter PReLU, and alignment fallback. Emitter and architecture
validation passes 76 tests; syntax, Ruff, and diff checks pass. No model
conversion or inference was run.

Native Concat emission and its channel-last axis-sensitive consumer guard now
live in the Torch-free emitter module. GatherElements coordinate construction,
channel-first/fused emission, materialized NHWC bridges, alias omission, and
exact-shape `_apply_concat` fallback remain in one implementation. Direct tests
fix the NHWC bridge, Gather channel-axis safety fallback, and exact-shape
runtime reshape behavior. Emitter and architecture validation passes 71 tests;
syntax, Ruff, and diff checks pass. No model conversion or inference was run.

## `fb-refactor4` rank-four bounded-family checkpoint

The first sixteen bounded families of the rank-four generic NHWC
pre-Concat matcher are now separated. `passes/nhwc_concat_layout.py` owns the
strict all-direct float path and the one-or-more-unary float path, with or
without direct inputs. The unary allowlist is RELU, RELU6, LOGISTIC, TANH, and
GELU. It also owns the one-or-more-Pad-plus-direct path and the one-or-more
Dequantize path and the one-or-more PReLU path, each with or without direct
inputs. It also owns exactly one Softmax plus at least one direct input, and
one or more expanded-Swish diamonds with direct or unary companion inputs.
The bounded Slice family additionally owns one or more direct-source Slice
inputs, optionally with direct inputs, while retaining shared/public source
adapters. The bounded Split family owns one
or more outputs from a direct-source Split, again optionally with direct
inputs, while retaining shared/public source adapters. The bounded Add family
owns bounded recursive Add trees with direct, unary, expanded-Swish, or
bounded-Split operands.
The exact pseudo-LeakyRelu diamond is the eleventh family. All
eleven float-path families share one
`ModelIRGraphIndex`/`LayoutState` pass group and run transactionally under
stable IDs `layout.nhwc_pre_concat_direct` and
`layout.nhwc_pre_concat_unary`, `layout.nhwc_pre_concat_pad`, and
`layout.nhwc_pre_concat_dequantize`, `layout.nhwc_pre_concat_prelu`, and
`layout.nhwc_pre_concat_softmax`, `layout.nhwc_pre_concat_swish`, and
`layout.nhwc_pre_concat_slice`, plus `layout.nhwc_pre_concat_split` at all
seven production positions, followed by `layout.nhwc_pre_concat_add`.
The pseudo-LeakyRelu family runs last under
`layout.nhwc_pre_concat_leaky`.
The twelfth through twenty-fourth families are the separate direct, unary,
Pad-plus-direct, unary-plus-Pad, all-Pad, expanded-Swish, Dequantize, PReLU, and
Softmax, followed by exact pseudo-LeakyRelu, bounded Slice/Split, and Add,
quantized-post passes
`layout.nhwc_pre_concat_quantized_direct`,
`layout.nhwc_pre_concat_quantized_unary`, and
`layout.nhwc_pre_concat_quantized_pad`, followed by
`layout.nhwc_pre_concat_quantized_unary_pad` and
`layout.nhwc_pre_concat_quantized_all_pad`, followed by
`layout.nhwc_pre_concat_quantized_swish` and
`layout.nhwc_pre_concat_quantized_dequantize`, followed by
`layout.nhwc_pre_concat_quantized_prelu` and
`layout.nhwc_pre_concat_quantized_softmax`, followed by
`layout.nhwc_pre_concat_quantized_leaky` and
`layout.nhwc_pre_concat_quantized_slice`, followed by
`layout.nhwc_pre_concat_quantized_split` and
`layout.nhwc_pre_concat_quantized_add`, in
`passes/nhwc_concat_quantized_layout.py`.

The direct pass removes only exclusive, non-public leading adapters. Shared or
public direct adapters remain for their other consumers while the Concat is
rewired to the NHWC source. Public Concat/post tensors, invalid permutations
or ranks,
non-Transpose Concat fan-out, and wrong axes are rejected without mutation.
Stale spatial metadata remains accepted for this algebraically strict family,
matching the previous intentional behavior. Canonical per-axis quantization
now remaps NCHW dimension 1 to NHWC dimension 3. The unary family additionally
requires exclusive, non-public unary adapters and output, plus compatible
NHWC batch/spatial metadata, and remaps unary output quantization metadata.
The Pad family preserves optional Pad inputs, retains a shared leading
adapter, and remaps Pad output metadata. Exclusive pads constants are updated
in place; pads constants shared with any other operator or public boundary are
cloned and only the selected Pad input is rewired, preserving other consumers.
The Dequantize family does not rewrite scale or zero-point data: it bypasses
only the layout adapter, preserves source quantization provenance, and remaps
rank-four Dequantize output metadata and any per-axis dimension to NHWC.
PReLU preserves the legacy alpha candidate order for rank-4, unchanged, and
rank-3 constants. An exclusive transformed alpha is updated in place; a
shared or public alpha is cloned with dtype, variable flag, quantization,
layout, and ONNX provenance retained. Alpha and PReLU-output per-axis
dimensions move with their actual permutations.
Softmax retains its original NCHW last-axis semantics with two local,
self-inverse NHWC↔NHCW Transposes around the unchanged Softmax operator. The
old NCHW adapter and Concat post adapter are removed, so the eligible family
still reduces total Transpose count. New intermediate and final per-axis
quantization dimensions follow NHWC→NHCW, NCHW→NHCW, and NCHW→NHWC
permutations respectively.
The expanded-Swish family proves the complete `Logistic(x) * x` diamond and
accepts either Mul input order. Both Logistic and Mul are moved to the NHWC
source, with output shapes and per-axis quantization metadata remapped from
NCHW dimension 1 to NHWC dimension 3. The family rejects unsupported
operators, mismatched Mul data, invalid ranks or spatial metadata, raw
residual inputs, public adapter/Logistic/Mul outputs, and fan-out from any
internal edge before mutation. Rejecting a public Logistic output is an
intentional correctness improvement over the legacy matcher because rewriting
that public tensor would otherwise silently change its layout contract.
The bounded Slice family requires a rank-four NHWC→NCHW source adapter and an
exclusive rank-four Slice output. It remaps begin/size vectors, Slice output
shape, and per-axis quantization into NHWC. Exclusive parameter tensors are
updated once in place; shared or public parameters are cloned with dtype,
variable state, quantization, layout, and ONNX provenance preserved. Shared or
public source adapters remain intact for their existing consumers while only
the Slice is rewired to NHWC. This closes two legacy correctness gaps:
shared/public Slice parameters are no longer silently mutated, and
Slice-output quantization dimension 1 is remapped to dimension 3. Valid
non-public inverse output adapters are bypassed and their consumers are
rewired to the NHWC Slice output; public post aliases reject before mutation.
The bounded Split family validates all outputs as rank four and requires each
to be unused or consumed only by the selected Concat. One Split may therefore
supply multiple Concat inputs while its source, axis, output metadata, and
quantization are rewritten exactly once. Negative channel axis `-3` and
positive axis `1` both canonicalize to NHWC axis `3`. Shared/public axis
tensors use the same provenance-preserving copy-on-write policy. Shared/public
source adapters remain intact for their existing consumers while Split alone
is rewired to NHWC. This also fixes the legacy omission of per-axis
quantization remapping. Valid non-public inverse output adapters are bypassed
and their consumers are rewired to the corresponding NHWC Split output;
public post aliases reject before mutation. Add interactions remain available
through the legacy fallback.
The bounded Add family accepts an acyclic graph of two-input Add operators
whose leaves come through rank-four
NHWC→NCHW adapters, optionally followed by a supported unary operation or
complete expanded-Swish diamond, Dequantize, PReLU, semantics-preserving
Softmax, exact pseudo-LeakyRelu diamond, exact Pad, bounded direct-source
Slice, or bounded Split. Dequantize, PReLU, Softmax, pseudo-LeakyRelu, Pad, and
Slice plans may also accompany the Add graph at the root Concat. Every Add
output consumer must belong to the selected Add graph/root Concat or be an
exact inverse adapter. Operands and bounded branches are rewired in their
original order, each Split is applied
once across all outputs, exclusive leading adapters are removed, shared/public
adapters remain for external consumers, operator options are retained, and
Add/branch output shapes and per-axis quantization are remapped once. Valid
non-public inverse output adapters are bypassed and their consumers are rewired
to the NHWC Add output;
public post aliases and invalid ranks reject before mutation. Source-adapter
removal uses the post-rewrite GraphIndex rather than coupled precomputed flags:
external/public consumers retain the adapter, while an adapter shared with the
root Concat is removed after both Add and Concat are rewired. An Add operand
may itself be another bounded Add plan. Planning carries the visited output
names through the tree, rejects cycles, and stops at depth 64. Application
shares the materialized-integer, applied-Split, and applied-Add sets so a
nested branch is rewritten exactly once. It also shares one Pad
materialization map across nested Add operands and root-Concat companions, so
the existing provenance-preserving Pad copy-on-write behavior is reused
without a second rule. Slice begin/size parameters reuse the shared integer
materialization map already used for Split axes, preserving copy-on-write and
one-time materialization. The former top-level Dequantize mutation is now one
shared apply helper used by both root companions and nested Add operands; it
preserves source quantization provenance while remapping output metadata. Each
nested inverse post adapter is
rewired to its own Add output, while cleanup recursively collects adapters
from the whole successful plan. A bounded pre-scan now collects every Add
operator in the selected graph and combines those indices with the root Concat.
Multiple outputs of one Split may feed different Add nodes and a separate
input of that same root Concat. The candidate-wide set makes resolution
independent of Concat input order, while any consumer outside the set that is
not an exact inverse adapter rejects the complete candidate. Add outputs may
therefore fan out to sibling selected Add branches or to a parent Add and the
root Concat; shared application state rewrites each Add only once. Pad output
and Slice output fan-out reject before mutation. Dequantize output fan-out is
rejected by the same ownership rule. PReLU output fan-out rejects as well. A
candidate-wide alpha cache keyed by source tensor, permutation, and selected
shape lets multiple PReLUs reuse one provenance-preserving transformed alpha
clone. Softmax reuses the existing pair of local NHWC↔NHCW transposes so the
original NCHW last-axis semantics and beta option remain unchanged. Softmax
output fan-out rejects before mutation. The pseudo-LeakyRelu diamond reuses
the existing exact topology, internal-edge ownership, singleton-alpha, and Sub
order guards; its five internal/output tensors are remapped together. Diamond
output or internal fan-out rejects. Other mixed operand families deliberately
remain in legacy.
The pseudo-LeakyRelu family proves the exact
`ReLU(x) - alpha * ReLU(-x)` topology. It accepts either Mul operand order,
requires scalar alpha, preserves Sub order, and supports direct or unary
Concat companions. Neg and positive Relu are rewired to the NHWC source; Neg,
both Relu outputs, Mul output, and Sub output shapes and quantization axes all
move to NHWC exactly once. This adds the alpha-first form that the legacy
matcher attempted but could not select. All public/fan-out internal edges,
rank errors, and partial diamonds reject before snapshot. Pad companions
deliberately remain in legacy.
The direct/unary/Pad/unary-plus-Pad/all-Pad/expanded-Swish/Dequantize/PReLU/Softmax/Leaky/Slice/Split/Add
quantized-post families validate
`adapters → optional bounded branch → Concat → Quantize → inverse Transpose(s)`
independently of the float group. They move Concat and supported branches to
NHWC, retain shared/public direct adapters, make the first post output
canonical, and rewire later aliases to it. Pad constants are reordered and
cloned with layout, quantization, variable-state, and ONNX provenance when
shared or public. The float and quantized paths use the same resolver and
materializer from `passes/nhwc_concat_pad.py`, preventing duplicate Pad rules
from drifting apart. Concat, branch, and quantized-output shapes and per-axis
metadata are remapped to NHWC; this fixes the legacy quantization-dimension
omission. The mixed pass requires both a unary and a Pad input and permits
additional direct inputs. The all-Pad pass requires at least two Pad branches;
when they share the same pads tensor, one provenance-preserving NHWC clone is
materialized and reused. The Swish family accepts either Mul operand order and
direct companions. It reuses the float path's exact expanded-Swish
resolver/apply pair, so topology, public-boundary, fan-out, metadata, and
quantization-axis behavior cannot drift between the float and quantized-post
paths. Public boundaries, invalid ranks/spatial metadata, and non-Transpose
fan-out reject transactionally. Other broader mixed quantized inputs continue
through legacy.

The Dequantize family also reuses the float path's resolver/apply pair. It
rewires Dequantize to the original NHWC quantized source, remaps its float
output and per-axis metadata, and removes the leading adapter only when that
adapter output is exclusive. Public Dequantize output boundaries and invalid
source ranks reject before mutation.

PReLU uses the same broadcast-safe alpha selection and apply implementation as
the float path. Alpha data and quantization axes move to the selected NHWC
broadcast form in place when exclusive; shared or public alpha constants use
provenance-preserving copy-on-write. PReLU output shapes and per-axis metadata
move to NHWC, while public PReLU output boundaries reject before mutation.

Softmax uses the already characterized axis-preserving float plan. The outer
NHWC→NCHW adapter becomes a local NHWC→NHCW adapter, Softmax stays last-axis,
and a local NHCW→NHWC adapter feeds the NHWC Concat. Beta and tensor provenance
are retained. The family requires exactly one Softmax plus at least one direct
companion; public Softmax outputs reject before mutation. The legacy matcher's
existing quantized-Softmax safety gate already rejects this post-Quantize
topology, so no duplicate legacy rewrite remains reachable.

The exact pseudo-LeakyRelu family reuses the float resolver/apply pair for
`ReLU(x) - alpha * ReLU(-x)` with a direct companion. It preserves Sub operand
order, requires singleton alpha, rewires both source arms to NHWC, and remaps
all five internal/output tensors. Wider unary/Pad/other mixed companions remain
in legacy. Public or fan-out internal boundaries reject before mutation.

Expanded-Swish, exact pseudo-LeakyRelu, and bounded Add now accept supported
unary root-Concat companions as well as direct companions. The shared unary
resolver/apply owns rewiring and metadata; the family guard still requires at
least one Swish, Leaky, or Add branch respectively.

The bounded Slice family accepts direct companions and rank-four channel Slice
with constant begin/size. It shares the float resolver and integer-parameter
materializer, including public/shared copy-on-write. This first quantized Slice
step requires the Slice output to feed only the root Concat; Slice outputs with
secondary inverse adapters remain in legacy. Output metadata and per-axis
quantization move to NHWC, while public Slice outputs reject before mutation.

The bounded Split family similarly requires a constant channel axis and no
secondary inverse output adapters. Multiple outputs from one Split may feed the
same root Concat; a shared applied-operator set rewrites the Split input, axis,
and every output tensor exactly once. The axis materializer shares Slice's
copy-on-write cache. Public Split outputs reject, while broader post-adapter
and mixed-input forms remain in legacy.

The bounded Add family accepts a depth-guarded Add tree whose leaves are direct,
supported unary, expanded-Swish, Dequantize, PReLU, Softmax, exact
pseudo-LeakyRelu, Pad, Slice, or Split plans without secondary output adapters,
plus direct root-Concat companions. Every Add input/output
and per-axis metadata moves to NHWC exactly once. Shared-plan cleanup walks
nested operand plans, uses differential consumer state to remove every dead
leading adapter, and retains adapters that remain public or live. Other
operands remain in legacy. Public Add outputs reject before mutation.

The lowerer compatibility helper still returns the original aggregate statistic
and runs the legacy matcher after the direct pass. The legacy matcher now
skips the twenty-four indexed families, but continues to own broader
Split/Slice/Add/Leaky interactions and remaining mixed quantized-post paths.

The quantized-post module no longer repeats one precondition, callback, and
`PassSpec` block per family. A single ordered table now owns family name,
statistics key, and priority; stable pass IDs, callbacks, preconditions, and
default metrics are derived from it. Shared float-plan resolution also uses
two maps: one for the common resolver signature and one for resolvers requiring
public tensor names. A single adapter applies the explicit Dequantize rank and
Slice/Split post-adapter guards before mapping the result into
`_QuantizedInputPlan`. Add remains separate because it recursively validates
bounded leaf plans. The staged refactor keeps the module materially below its
original 1,549 lines while preserving all thirteen quantized family IDs and
their order.

Candidate input acceptance now has a parallel immutable contract per family:
allowed/required kinds, minimum arity, exact counts, and shape-validation
policy. A common validator replaces the thirteen count branches, and an
import-time invariant prevents the pass and input-contract family sets from
drifting. Resolver selection and whole-Concat input acceptance remain separate
contracts, preventing family-specific graph guards from leaking back into the
combination validator. Unary and Pad candidate resolution is now derived from
each contract's allowed kinds rather than duplicate family-name lists. The four
common-signature apply operations use an applier map, with an import-time
resolver/apply coverage invariant; contextual PReLU/Slice/Split and recursive
Add application remain explicit. The thirteen frozen specs, callbacks,
preconditions, default statistics, and preflight function are now constructed
once at module import. Adapter-liveness and recursive shared-plan helpers are
also top-level rather than recreated per candidate. The module is currently
materially smaller than its original state; these checkpoints prioritize
explicit helper contracts and lower runtime allocation, not a source-line
threshold.

`ModelIRGraphIndex` now maintains graph-order operator-type indices through
refresh, insertion, and removal. Quantized-post candidate planning asks the
shared index only for `CONCATENATION` positions, eliminating thirteen repeated
full operator-list scans while preserving graph order and differential
mutation semantics.

`ModelIRPassState` now has one-shot session-local prepared pass data. A
quantized family precondition stores its fully validated candidate and the
callback consumes that same object instead of repeating the complete search.
Prepared data is isolated per conversion session and cleared during rollback,
so restored ModelIR objects cannot retain stale operator references. The next
candidate search after a successful rewrite remains unchanged.

Six additional bounded Concat-root passes now enumerate graph-order
`CONCATENATION` indices directly from `ModelIRGraphIndex`: axis-3 constant
Concat, Add/Concat suffix, rank-five NDHWC Concat, Concat/unary/Conv, SPP, and
Dequantize/Concat/Quantize. Their semantic guards and transactional behavior
are unchanged; unrelated operators are no longer visited by root discovery.

Quantized Reshape fusion and the four quantized PReLU bridge/fusion matchers now
use indexed `DEQUANTIZE` or `TRANSPOSE` roots as well. Each matcher still breaks
after mutation and restarts from the differentially updated index, preserving
its original rewrite order.

`ModelIRGraphIndex.operator_indices_for_types()` now returns the sorted,
deduplicated graph-order union for multi-type matchers. The two Cast cleanup
families and constant-input Cast, Pool, and Pad folds use indexed roots and
restart after mutation as before. ScatterND and binary constant folds now also
accept a shared index/LayoutState and remove operators differentially. Float32
and float16 artifact clones build one index and share it across both folds, so
`constant_fold.py` contains no full operator scan or direct operator-list
deletion.

The final ConvInteger channel-last bridge repair now uses one differential
`ModelIRGraphIndex` instead of rebuilding producer/consumer maps per iteration.
It enumerates indexed Transpose roots, updates the Conv input through the index,
removes the stale adapter differentially, prunes through the shared helper, and
synchronizes LayoutState after changing tensor layout metadata.

The final channel-last-input/channel-first-Pad repair now follows the same
contract. Its original static shape-equation guard is unchanged, but root
discovery is restricted to indexed `PAD`, `PADV2`, and `MIRROR_PAD` operators.
Pad input replacement and NHWC-to-NCHW adapter insertion update one
`ModelIRGraphIndex` differentially, then restart against the current graph.
The late lowerer call passes the session `LayoutState`, which is synchronized
after successful repair. Focused validation passed all four Pad layout tests;
instrumentation proves one initial index refresh and a consistent final layout
state. `py_compile`, targeted Ruff, and `git diff --check` also pass. No ONNX
conversion or inference test was run for this checkpoint, following the active
implementation-first validation policy.

The synthetic-boundary channel-Slice rewrite now builds or accepts one
`ModelIRGraphIndex` and accepts the session `LayoutState`. Indexed Transpose
root discovery replaces the full root scan; all consumer queries reuse the
index; and local NHWC propagation enumerates only its declared unary, binary,
Concat, Slice, and Conv-family types. Slice input replacement, external bridge
rewiring, bridge insertion, remaining-NCHW localization, and boundary-adapter
removal are differential index operations. The original candidate guards,
axis-vector conversion, graph-order insertion, and retry order are unchanged.
The lowerer compatibility wrapper forwards optional state and its production
call supplies the session layout state. The focused bridge and no-bridge
success cases pass with one initial index refresh and a consistent final layout
state; no model conversion or inference was run.

The internal Transpose/channel-Slice NHWC propagation family is indexed as
well. It discovers candidate Transposes and all tensor consumers through one
`ModelIRGraphIndex`, restricts fixed-point propagation to the supported
operator-type union, updates Slice and cloned-constant inputs differentially,
inserts legacy NCHW bridges through the index, and removes the original stem
without direct operator-list mutation. Its lazy initial-consumer cache retains
the legacy copy-on-write behavior for constants shared by multiple binary
operators even as indexed rewires proceed. The lowerer forwards the session
layout state. Its focused multi-branch success case passes with one initial
index refresh, consistent operator-type indices, and a synchronized final
layout state. No model conversion or inference was run.

The channel-Slice/Mul/post-Transpose bridge rewrite now uses one differential
index as well. It enumerates only Transpose roots, passes that index through
Slice, Mul-constant, and global post-alias rewires, removes all selected
adapters through `ModelIRGraphIndex.remove_operator`, and synchronizes the
session layout state after pruning. Each retry snapshots only the existing
consumer-index mapping so all branches retain the legacy pre-rewrite view;
this avoids a semantic change in shared-constant and fan-out decisions while
eliminating graph-map reconstruction and direct operator-list deletion. Its
focused direct-plus-Mul success graph passes with one initial index refresh,
no residual Transpose indices, and a consistent final layout state. No model
conversion or inference was run.

The boundary StridedSlice/QDQ/Concat round-trip family now completes the
channel-slice module's direct-mutation cleanup. Candidate Transpose roots and
every Slice→Quantize→Dequantize→Concat→Quantize→Transpose edge are read from
one index before mutation. StridedSlice input replacement, quantized Concat
output canonicalization, secondary post-alias replacement, and removal of the
boundary/post Transposes all update that same index. The production wrapper
forwards the session layout state, and layout-aware pruning reconciles the
canonical output tensor. The four-branch focused success graph passes with one
initial index refresh, no remaining Transpose indices, and a consistent layout
state. `channel_slice_layout.py` now contains neither whole-graph consumer-map
construction nor direct operator-list insertion/deletion. No model conversion
or inference was run.

CSP attention propagation now builds or accepts one `ModelIRGraphIndex` and
the session `LayoutState`. Its already-strict full-topology guard reads
producers, consumers, and Transpose roots from that index before mutation.
Short/point sigmoid-self-Mul branches, the optional residual Add, reduction
Conv input, gate head, canonical terminal output, and secondary aliases are
all rewired through index-aware helpers. Every selected bridge Transpose is
removed differentially, and pruning synchronizes layout state. Both residual
and no-main-Add focused success variants pass; the instrumented residual case
uses one initial index refresh and finishes with no Transpose indices or layout
state mismatch. No model conversion or inference was run.

Conv-attention propagation no longer batches raw graph mutation followed by a
full `GraphIndex.refresh()`. Both the standard reduction/gate path and the
self-HardSwish Mean fallback enumerate indexed Transpose roots, pass the shared
index to every edge mutation helper, and remove bridge operators
differentially. Legacy NCHW consumer slots retain their `OperatorIR` objects
across removals; current indices are recovered from the live index before one
local adapter is inserted. The four existing gate/activation variants pass.
The instrumented standard case additionally covers a public legacy NCHW
consumer, proves one initial index build with no refresh, inserts exactly one
local adapter, and finishes with a consistent layout state. An architecture
gate now prevents map builders, direct insert/delete, and routine refresh from
returning to `attention_layout.py`. No model conversion or inference was run.

Late constant-DIV precision handling now uses a differential graph index.
Forward rewriting captures the initial indexed DIV objects, evaluates direct
integer-Cast consumers from that index, and replaces each eligible operator in
place with the same ordered optional-Cast/Mul/optional-Cast sequence. It no
longer constructs an operator-object consumer map or replaces the complete
operator list. Precision-sensitive restoration enumerates indexed MUL roots,
traverses indexed affine/shape consumers, rewires the reciprocal input, and
changes MUL to DIV through the new
`ModelIRGraphIndex.replace_operator_type()` primitive. The main conversion path
passes the session layout state; fallback clones retain the compatible
single-argument calls. Three focused precision cases cover normal rewrite,
integer-Cast preservation, and affine-chain restoration. Core and precision
focused validation passes 29 tests with one initial index refresh in each
instrumented case. No model conversion or inference was run.

Static high-rank binary coalescing now replaces each indexed candidate in
place rather than constructing and assigning a complete rebuilt operator list.
The original supported binary objects are fixed in graph order, so generated
rank-four binary operators are not reconsidered. Each candidate retains the
same two input Reshapes, coalesced binary, and output-restoring Reshape at the
original position, while differential remove/insert calls keep producer,
consumer, object, and type indices current. The production call supplies the
session layout state. Static-success and dynamic-signature no-op tests pass;
the success case proves one initial index refresh and exact final operator-type
indices. No model conversion or inference was run.

Static high-rank BatchMatMul compression now follows the differential contract
too. It captures initial indexed `BATCH_MATMUL` objects, preserves the original
operator/options, rewrites its two inputs and one output through the index, and
inserts the two rank-five input Reshapes plus restoring output Reshape around
its current graph position. The pass no longer reconstructs the complete
operator list or assigns operator inputs/outputs directly. The lowerer wrapper
keeps its one-argument compatibility while accepting optional index/layout
state, and the final production call supplies session layout state. The focused
rank-six parity fixture passes with one initial refresh, exact operator-type
indices, and consistent final layout state. No model conversion or inference
was run.

PyTorch native-runtime SAME AveragePool correction is now differential as
well. The compatibility pass captures only initial indexed
`AVERAGE_POOL_2D` objects, updates the live pool output through the shared
`ModelIRGraphIndex`, and inserts the correction `MUL` at the current graph
position. It no longer accumulates and replaces a complete operator list or
assigns operator outputs directly. Its existing single-argument exporter call
remains compatible, while optional graph-index and layout-state parameters
allow composition with shared state. A Torch-free focused fixture verifies the
exact reciprocal correction, one initial index refresh, final operator-type
indices, and layout consistency. The corresponding architecture gate prevents
full-list and direct-output mutation from returning. The pre-existing large
PyTorch exporter test could not be collected in the current `uv` environment
because a Python 3.10 user-site `libtorch_python.so` is being resolved under
Python 3.12; this is an environment mismatch rather than a test assertion
failure. No model conversion or inference was run.

Dynamic rank-one Unsqueeze shape materialization is now owned by
`passes/dynamic_reshape.py`. The lowerer keeps a signature-compatible wrapper,
but the implementation enumerates only indexed `RESHAPE` candidates and no
longer builds or assigns a complete replacement operator list. For a true
rank-one dynamic input it updates the existing Reshape input through the
index, then inserts `SHAPE` and `CONCATENATION` at the live graph position. The
folded higher-rank fallback retains the established shape-constant `-1`
repair. Main-path calls pass the session layout state; fallback relowering
retains the compatible self-indexing call. Focused dynamic-shape and
architecture validation passes 54 tests, including exact operator order,
one initial index refresh, live type/object indices, and final layout
consistency. No model conversion or inference was run.

That focused module now also owns placeholder MatMul-flatten restoration. It
indexes initial placeholder `RESHAPE` objects, obtains the exclusive
`BATCH_MATMUL` consumer from the same index, rewires the recovered high-rank
source through `replace_operator_inputs()`, and removes the obsolete Reshape
differentially. Layout-aware tensor pruning replaces the former whole-graph
consumer map and operator-list filter. The focused rank-recovery fixture proves
one initial refresh, the surviving MatMul indices and inputs, removal of the
flatten tensor, and final layout consistency. The main production call passes
the session layout state; fallback relowering retains the self-indexing
compatibility call. No model conversion or inference was run.

Graph-wide dead-code pruning now uses a dedicated batch index operation.
`ModelIRGraphIndex.remove_operators()` validates and deduplicates selected
positions, detaches their edges against the original graph, filters operators
once, and compacts every producer, consumer, duplicate-producer, object, and
operator-type reference through one old-to-new mapping. It performs no index
refresh and avoids the quadratic cost of repeated single removals.
`passes/graph_cleanup.py` owns the unchanged reverse-liveness policy, including
live variable-state mutation retention, and applies the dead set through that
batch primitive. Tensor pruning receives the active layout state at all main
lowering call sites. Focused core, cleanup, and architecture fixtures cover
non-contiguous/deduplicated removal, refreshed-index equivalence, interleaved
live/dead chains, one initial refresh, tensor pruning, and layout consistency.
No model conversion or inference was run.

Unsupported-dtype Split fallback now lives in
`passes/split_fallback.py`. Initial `SPLIT` candidates come from one type index;
the existing supported-dtype, constant-axis, rank, output-count, metadata, and
equal-partition guards are unchanged. Each accepted Split is removed at its
current position and replaced differentially by an optional dtype-alignment
`CAST` followed by ordered `SLICE` operators. Generated begin/size tensors are
pruned/reconciled with the session layout state. Focused cast and no-cast
fixtures cover axis normalization, slice offsets/sizes, exact operator order,
one initial refresh, final type indices, and layout consistency. The lowerer
keeps its compatibility wrapper and production order. No model conversion or
inference was run.

Dynamic Squeeze runtime-shape prefixes now use differential insertion. The
existing rewrite still produces the same `SHAPE` and `GATHER` pair and converts
the original Squeeze object to Reshape. Instead of rebuilding the complete
operator list, it builds one index after all direct metadata changes and
inserts recorded prefixes in reverse original-index order, preserving graph
order for multiple candidates. The main call supplies session layout state,
and focused validation proves one initial refresh, exact
`SHAPE/GATHER/RESHAPE` order, original operator identity, and layout
consistency. Central lowerer `model_ir.operators` assignments are now limited
to explicit snapshot rollback. No model conversion or inference was run.

Dynamic-range quantization now mutates the already isolated ModelIR clone
through one graph index instead of cloning every operator a second time and
assigning a rebuilt list. Kernel/constant quantization decisions remain
unchanged. The first elementwise consumer of a quantized constant receives one
`DEQUANTIZE` at its live position, subsequent consumers share that tensor, and
all input rewires update the index differentially. This also retains cloned
axis semantics and ONNX operator provenance that the former second copy did
not carry forward. A focused shared-constant fixture proves one initial
refresh, exact Dequantize placement, both consumer rewires, INT8 qparams, and
immutability of the source ModelIR. No model conversion or inference was run.

Full-integer Identity elision is indexed as well. It retains the existing
forward replacement order, transitive Identity-chain resolution, and graph-
output producer promotion semantics. Retained inputs and outputs are rewired
through index primitives, and all eliminated Identity objects are compacted in
one batch. Graphs without Identity operators return before index construction.
Focused producer-promotion and boundary-chain fixtures prove exact final
outputs, surviving operator identity, and one initial refresh. The
quantization module no longer rebuilds its complete operator list for Identity
cleanup. No model conversion or inference was run.

Strict integer boundary construction now continues with one graph index after
Identity elision. Each graph input rewires only its indexed consumers; output
dtype bridges rename the indexed producer instead of scanning every operator.
The core quantization/report loop still sees exactly the original core order.
Afterward, input `QUANTIZE` operators are inserted in declared input order and
output `QUANTIZE`/`DEQUANTIZE` operators are appended in declared output order.
This removes the complete `pre + core + post` list assignment. A focused float-
IO strict-integer fixture proves one initial refresh, exact boundary/core
operator order, connected boundary names, preserved public output, and source
ModelIR immutability. No model conversion or inference was run.

Model serialization now reuses shared indexed dead-code pruning. The serializer
still creates only a shallow graph-container clone, preserving weight-buffer
sharing and source ModelIR reuse. It requests operator-only pruning from
`passes/graph_cleanup.py`, then performs embedded-constant input stripping and
unused-tensor removal in the original order. The duplicate reverse-liveness
implementation and full operator-list filter were removed from
`model_writer.py`. A focused sanitizer fixture proves one initial refresh,
live graph order, dead tensor removal, constant-input stripping, and complete
source-container immutability. No FlatBuffer serialization, model conversion,
or inference was run.

The split/export rewrite builder is now append-only. It clones tensor,
subgraph, boundary, and metadata state into a ModelIR with an initially empty
operator stream instead of deep-copying all source operators and discarding
them after a second list is generated. Grouped Conv expansion, BatchMatMul
unfolding, and recurrent unrolling emit directly into that stream; their final
whole-list assignments are removed. The common unchanged-operator copy now
also preserves axis semantics and ONNX node/op provenance, which the previous
copy helper silently dropped. Focused no-op coverage exercises all three
rewriters, and synthetic grouped-Conv and batched-MatMul fixtures verify each
nontrivial expansion occurs exactly once without mutating the source graph. No
SavedModel export, model conversion, or inference was run.

Boundary-based partition cropping now preserves the complete operator contract
too. Cropped operators retain axis semantics and ONNX node/op provenance in
addition to their existing inputs, outputs, options, and version. The focused
dead-branch crop fixture verifies those fields on the surviving operator while
retaining boundary/tensor pruning behavior. No partition artifact was written.

PyTorch redundant layout-Transpose cleanup now lives beside the AveragePool
compatibility rewrite in the Torch-free `passes/pytorch_compat.py` module. It
enumerates indexed Transpose candidates and consumers, preserves all existing
rank/layout/shape-sensitive guards, rewires internal consumers differentially,
and removes each adapter through the index. A graph-output adapter is replaced
at the same position by Identity with the established cloned output-tensor
metadata. The PyTorch exporter no longer builds a separate consumer map or
filters its complete operator list. Focused internal and public-output fixtures
prove one initial refresh, final type indices, consumer rewiring, tensor
cleanup, output metadata, and layout consistency without importing Torch. No
PyTorch package generation, model conversion, or inference was run.

The float NHWC Concat runner now uses the same declarative structure. One table
owns its eleven family names, statistics keys, and priorities; frozen specs,
callbacks, preconditions, defaults, and preflight are constructed once. Its
candidate search enumerates only indexed `CONCATENATION` positions, and a
successful precondition passes the exact candidate to the common family
optimizer through session-local prepared data. All existing family-specific
optimizer wrappers remain available. Common-signature, public-name, and
unary-companion resolver maps now own every non-direct/non-Add family, with an
import-time coverage invariant. Recursive Add stays explicit because it needs
candidate-wide consumer ownership. A separate immutable input-contract table
now declares allowed/required kinds, exact counts, and shape validation for all
eleven families. One validator replaces the repeated count branches; Slice
retains its unique-operator guard. The module remains below its original 3,120
lines while preserving stable pass IDs, order, statistics, transactions, and
legacy fallback ownership.

Add root-companion fallback resolution now uses one ordered tuple for
Dequantize, PReLU, Softmax, pseudo-LeakyRelu, Pad, and Slice. This preserves the
original precedence and public-name signatures. Split remains the final
explicit resolver because it alone consumes the candidate-wide allowed
consumer set.

Float apply dispatch now maps the four common-signature Unary, Swish, Softmax,
and Dequantize operations. An import-time invariant matches simple and
contextual applier kinds against the union of input-contract kinds. Leaky,
Split, and Add keep their one-time application sets. Adapter-liveness and
recursive input-plan walking moved out of the candidate loop into top-level
helpers, avoiding per-candidate closure creation.

The adjacent legacy ownership gate now builds action-kind counts once. Seven
simple quantized family contracts declare only allowed and required kinds;
all-Pad keeps its minimum-arity guard, and Slice/Split/Add keep their
plan-aware predicates. This removed a further net 79 lines from the central
lowerer without broadening which unsafe candidates bypass legacy fallback.
Seven simple float families now use a parallel allowed/required-kind contract
over the same action multiset, removing another net 74 lowerer lines. Their
compact regression exposed one stale Softmax expectation from before the
quantized-post Softmax pass existed; `quantized_post` is now characterized as
success under `layout.nhwc_pre_concat_quantized_softmax` instead of a no-op.

The compatibility wrapper's aggregate statistic no longer contains twenty-four
hand-written `int(stats.get(...))` additions. Explicit float and quantized key
tuples feed two sums before the legacy count is added, preserving the original
single return key and removing a further net 110 lowerer lines.

Changed files for this checkpoint:

- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`
- `onnx2tf/tflite_builder/passes/nhwc_concat_layout.py`
- `onnx2tf/tflite_builder/passes/nhwc_concat_pad.py`
- `onnx2tf/tflite_builder/passes/nhwc_concat_quantized_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_swish_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_slice_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_split_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_add_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_leaky_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_quantized_layout.py`
- `docs/flatbuffer_direct_architecture.md`
- `docs/flatbuffer_direct_handoff_2026-07-13.md`

Focused verification, all in the existing `uv` environment:

- Direct, unary, Pad, Dequantize, PReLU, Softmax, expanded-Swish,
  pseudo-LeakyRelu, and bounded Slice/Split/Add ModelIR characterization:
  the preceding combined float-path run passed 176 tests across eight compact
  modules; authoritative collection now contains 212. Including the bounded
  direct and unary/Pad/Swish/Dequantize/PReLU/Softmax/Leaky/Slice/Split/Add
  quantized-post suites, the compact inventory contains 282 tests across nine
  modules. The preceding combined run passed 208 tests; the expanded quantized module passes 70 tests, and the focused quantized/Pad
  selection after extracting the shared Pad plan passes 52 tests.
  The Softmax suite includes an exact NumPy equivalence check for the original
  and rewritten layouts. The Swish suite covers both Mul operand orders,
  all-Swish inputs, and fourteen whole-ModelIR unsafe/partial-match no-op
  boundaries. The Slice
  suite covers mixed and all-Slice success, shared/public parameter
  copy-on-write, shared/public source-adapter retention, output-post-adapter
  bypass, and fifteen complete no-op boundaries. The Split suite covers both axis
  signs, multi-output single-application behavior, shared/public axis
  copy-on-write, shared/public source-adapter retention, output-post-adapter
  bypass, and fifteen no-op boundaries. The
  Add suite covers mixed/all-Add success, shared/public source-adapter
  retention in both operand positions, root-Concat adapter sharing,
  output-post-adapter bypass, twenty-three complete no-op boundaries, and one
  indexed supported-unary operand case plus one indexed exact expanded-Swish
  operand case plus one indexed bounded-Split operand case. It also covers a
  bounded recursive Add operand, correct ownership of an inner Add's inverse
  post adapter, one Split feeding two different nodes of a recursive Add tree,
  one Split feeding an Add and the root Concat in either input order, external
  Split-consumer rejection, shared Add output fan-out across sibling branches
  and the root Concat, external Add-consumer rejection, and a whole-ModelIR
  recursive-cycle no-op. It also covers exact Pad as an Add operand and as a
  root-Concat companion, bounded Slice in both positions, and Pad/Slice-output
  fan-out rejection. Dequantize is covered in both positions with source
  quantization provenance and output-fan-out rejection. PReLU is covered in
  both positions, including shared-alpha clone reuse and output-fan-out
  rejection. Softmax is covered in both positions with exact local-axis
  adapters, beta retention, metadata remapping, and output-fan-out rejection.
  Exact pseudo-LeakyRelu is also covered in both positions with internal
  metadata remapping and output-fan-out rejection. The
  pseudo-LeakyRelu suite
  covers both alpha operand orders, direct/unary/all-Leaky success, twenty
  complete no-op boundaries, and one Pad-mixed legacy fallback. The quantized
  suite covers canonical and multiple post outputs, shared/public adapter
  retention, all five supported unary operations, Pad-constant copy-on-write,
  mixed unary-plus-Pad success, all-Pad shared-constant reuse, and thirty-four
  no-op boundaries. It also covers both expanded-Swish Mul operand orders and
  a public logistic-intermediate no-op boundary, plus Dequantize success with
  per-axis metadata remapping and public-output/invalid-rank no-op boundaries.
  PReLU coverage fixes alpha permutation, output metadata remapping, and a
  public-output no-op boundary. Softmax coverage fixes the two local NHCW
  adapters, beta retention, output metadata, and a public-output no-op boundary.
  Exact pseudo-LeakyRelu coverage fixes Sub order, both NHWC source arms, all
  internal quantization axes, and a public-output no-op boundary. Slice
  coverage fixes channel begin/size remapping, output metadata, and a
  public-output no-op boundary. Split coverage fixes multi-output
  single-application behavior, axis remapping, output metadata, and a public
  output no-op boundary. Add coverage fixes both direct operand rewrites,
  complete adapter cleanup, two-level recursive application, a supported unary
  leaf, representative expanded-Swish leaf reuse, output metadata, and a
  public-output no-op boundary. Dequantize/PReLU/Softmax/Leaky/Pad/Slice/Split
  leaves use the same already characterized shared apply paths.
  Root-companion coverage includes expanded-Swish plus supported unary.
- After consolidating the seven shared quantized input resolvers, the focused
  quantized-post module passed `70 tests in 0.38s`; Python compilation, targeted
  Ruff, and `git diff --check` also passed.
- After adding the differential operator-type index and Concat-only candidate
  enumeration, the core-index plus quantized-post selection passed
  `95 tests in 0.47s`; targeted compilation and Ruff also passed.
- After passing prepared candidates from precondition to callback, session
  isolation, rollback clearing, exact resolver-call accounting, and the full
  focused selection passed `97 tests in 0.44s`; targeted compilation and Ruff
  also passed.
- After converting the float runner to declarative frozen specs, indexed Concat
  enumeration, and prepared-candidate reuse, all nine compact float/quantized
  NHWC Concat modules passed `284 tests in 0.77s`; the focused float module
  passed `44 tests in 0.36s` and targeted compilation/Ruff passed.
- After replacing non-Add float family dispatch with three resolver maps and a
  declared-family coverage invariant, the same compact selection remained
  `284 passed in 0.77s`; targeted compilation, Ruff, and diff checks passed.
- After replacing float family count/shape branches with immutable input
  contracts and one validator, the same nine-module selection remained
  `284 passed in 0.77s`; targeted compilation, Ruff, and diff checks passed.
- After consolidating the ordered Add root-companion fallback resolvers, the
  same selection remained `284 passed in 0.77s`; targeted compilation, Ruff,
  and diff checks passed.
- After consolidating simple float appliers and lifting cleanup helpers, the
  same selection passed `284 tests in 0.76s`; targeted compilation, Ruff, and
  diff checks passed.
- After converting six additional bounded Concat-root scans to the shared
  operator-type index, their focused suites passed `169 tests in 0.60s`;
  targeted compilation, Ruff, and diff checks passed.
- After indexing the five quantized Reshape/PReLU root scans, their focused
  suites passed `7 tests in 0.32s`; targeted compilation, Ruff, and diff checks
  passed.
- After adding multi-type indexed enumeration and migrating Cast plus
  constant-input Cast/Pool/Pad roots, core and focused suites passed
  `38 tests in 0.42s`; targeted compilation, Ruff, and diff checks passed.
- After migrating ScatterND/binary folding and sharing one artifact index, the
  focused constant-fold suite passed `3 tests in 0.30s`, including one-refresh
  differential removal and LayoutState validation; targeted compilation, Ruff,
  and diff checks passed. The existing ScatterND constant-update lowering
  regression also passed (`1 passed`, `754 deselected`).
- After migrating the ConvInteger channel-last repair, its focused suite passed
  `3 tests in 0.28s`, including one-refresh accounting and LayoutState
  validation; targeted compilation, Ruff, and diff checks passed.
- Existing mixed-family NHWC matcher characterization: `5 passed`, `750`
  deselected.
- TensorFlow boundary and flatbuffer-direct architecture suite: `43 passed`.
- Ruff on the new pass and its compact test module: passed. A repository-wide
  Ruff gate is not configured; checking the pre-existing central lowerer also
  reports its known unused-import/local baseline.
- No ONNX corpus or large-model conversion was run for this checkpoint, per
  the instruction to minimize conversion testing and prioritize improvement.

Next work should continue reducing duplicated quantized candidate/apply
dispatch while preserving the characterized contracts. Keep uncharacterized
interactions in legacy until independently fixed. Do not begin with a Tier 0–4
corpus run, and do not create a pull request.

The section below records the preceding rank-five checkpoint and remains as
historical context.

## `fb-refactor4` pause checkpoint — `1a343c5`

Work is paused at a clean implementation checkpoint. No new pull request must
be created on resume; use appropriately scoped commits and pushes to
`fb-refactor4` only. The local branch and `origin/fb-refactor4` are synchronized
at `1a343c5` (`index ndhwc pre concat layout pass`) before this handoff-only
commit.

### Completed work in `fb-refactor4`

- The managed Tier 0–4 profile runs in authoritative tier/model order and
  contains 420 recorded models: 382 active and 38 excluded from execution.
  The excluded set consists of 27 expected timeouts and 11 explicit user
  exclusions. The active baseline classifications are 356 pass, 20
  `tflite_fail`, and 6 `missing_tflite_report`.
- User-approved DEIM TopK index instability is accepted without discarding the
  raw maximum absolute errors or the unaccepted classification. The bulk
  result records both the managed acceptance and the underlying numeric
  result.
- `encoder.onnx` is classified as an expected 600-second timeout.
- The explicit future-validation exclusions are:
  `fast_acvnet_generalization_opset16_192x320.onnx`,
  `htdemucs_ft_onnx_1sec.onnx`, `maskrcnn_resnet50_fpn.onnx`, `model1.onnx`,
  `paddlepaddle_26_ocr.onnx`, `bread_180x320.onnx`,
  `bread_nonfm_180x320.onnx`, `double_gru.onnx`, `gtcrn_simple.onnx`,
  `conv_tasnet_dnn_ins.onnx`, and `spkrec-resnet-voxceleb.onnx`.
- The sequential bulk runner now samples Linux `VmSwap` for the active
  converter subprocess and all descendants. Any nonzero process-tree SWAP
  stops that model, records `swap_detected`, peak tree KiB, and per-process
  peaks, and leaves unrelated host-wide SWAP out of the decision. Future
  detected models must be added to the managed profile as `excluded` with
  reason `swap_detected_during_managed_validation` before the next managed
  run.
- The 258-line rank-five NDHWC pre-Concat matcher was mechanically moved to
  `passes/ndhwc_concat_layout.py` at `8908a90`. Its extracted function AST
  exactly matched the characterized central implementation with SHA-256
  `0b0c625290f2ed31351ca204b0bbc5f2a463fa09ffe1bf1eccb8ff15de6aee17`.
- Checkpoint `1a343c5` replaced its repeated producer/consumer maps and direct
  operator deletion with pure `ModelIRGraphIndex` candidate planning,
  differential mutation, `LayoutState` reconciliation, and transactional pass
  ID `layout.ndhwc_pre_concat`. All five raw production calls now use the
  stable runner. Per-axis quantization metadata is cloned and remapped from
  NCDHW dimension 1 to NDHWC dimension 4 for unary and canonical Concat
  tensors.
- The 2,000 threshold remains only the Tier 5 ONNX node-count boundary. It is
  not a source-file line limit.

### Unfinished work

- Continue staged characterization and indexed migration of the remaining
  central layout families. The adjacent rank-four generic
  `_optimize_transpose_pre_concat_nhwc_chains` matcher is much larger and must
  first be audited and divided into semantic characterization units; do not
  treat it as one blind monolithic extraction.
- Complete remaining lowerer/registry decomposition and consolidate op-family
  validation, capability selection, and lowering.
- Complete fixed-ModelIR coverage for quantization modes, split/crop,
  custom/pseudo ops, weights, reports, and requested-artifact-only execution.
- Complete shared PyTorch, TorchScript, Dynamo ONNX, and ExportedProgram
  canonicalization/emitter decomposition.
- Complete the public CLI/Python and artifact matrix audits, optional
  TensorFlow exporter compatibility, TensorFlow-free direct/`-cotof` boundary,
  remaining tier gates, normalized failure comparison, and three-run median
  timing/peak-RSS measurements.
- A full current Tier 0–4 run was intentionally not completed. The latest user
  direction is to keep conversion tests minimal and prioritize implementation
  improvements. Do not restart a whole-corpus run as the first resumed task.
- The previously noted DPT producer-rank investigation and any other
  corpus-only candidates remain unproven until a focused reproducer justifies
  work; do not infer broad non-regression from the partial run.

### Branch and changed files

- Branch: `fb-refactor4`
- Implementation checkpoint: `1a343c5`
- Local/remote divergence before this documentation commit: `0 0`
- Worktree before this documentation commit: clean
- Pull requests: do not create one on resume

Files changed by the `fb-refactor4` checkpoints covered here:

- `docs/baselines/flatbuffer_direct_active_tier0_4.json`
- `docs/flatbuffer_direct_architecture.md`
- `docs/flatbuffer_direct_handoff_2026-07-13.md`
- `onnx2tf/utils/flatbuffer_direct_bulk_runner.py`
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`
- `onnx2tf/tflite_builder/passes/ndhwc_concat_layout.py`
- `tests/test_flatbuffer_direct_bulk_runner.py`
- `tests/test_flatbuffer_direct_architecture.py`
- `tests/test_flatbuffer_direct_ndhwc_concat_layout.py`

### Tests executed and results

All inference remained sequential with one model process at a time and all
commands used the existing `uv` environment.

- Managed-profile/bulk-runner focused suite after the two latest explicit
  exclusions: `34 passed`.
- SWAP monitoring, bulk-runner, and architecture focus: `78 passed`.
- Actual `Acos_11.onnx` SWAP-monitor smoke: pass,
  `swap_detected=false`, peak SWAP `0 KiB`, maximum absolute error
  `7.972121238708496e-07`.
- NDHWC mechanical extraction characterization plus architecture:
  `59 passed`.
- Indexed NDHWC characterization, transaction metrics, ownership, and
  architecture: `60 passed`.
- Differential comparison against checkpoint `8908a90`: all 16 cases (one
  success and fifteen unsafe boundaries) produced identical non-quantized
  ModelIR and statistics.
- Sequential `superpoint.onnx` direct/`-cotof` smoke:
  `evaluation_pass=true`, maximum absolute error
  `1.6666017472743988e-06`, RMSE `1.6207873294228388e-07`, cosine similarity
  `1.0`. The new rank-five pass skipped all five positions with zero snapshots
  and fingerprints on this unrelated rank-four graph.
- Ruff check of the touched NDHWC pass and tests: passed.
- The intentionally interrupted current-profile Tier 0 run completed 37 of
  382 active models: all 37 passed and none reported process-tree SWAP. This is
  diagnostic evidence only, not a complete tier gate.

### Failing tests and known issues

- No focused test is failing at this checkpoint.
- No process-tree SWAP was detected in the 37-model partial run. Existing
  host-wide SWAP was deliberately ignored and is not evidence against a model.
- The managed profile still intentionally records 20 numeric failures and 6
  missing reports among active models, plus 27 expected timeouts. These are
  known baseline classifications, not new failures from the NDHWC migration.
- Current broad corpus non-regression and performance targets are not proven by
  the deliberately minimal validation scope.
- Temporary copied ONNX, generated TFLite, and schema artifacts from the
  interrupted run were deleted. Its JSON/log evidence remains under
  `/tmp/onnx2tf_tier0_4_fb4_ccf5277` (about 7.5 MiB at pause time).

### First work on resume

1. Confirm `git status --short --branch` is clean and local/remote divergence
   is `0 0` after the handoff commit.
2. Do not start a whole Tier 0–4 conversion run. Re-run only the 60-test NDHWC
   focused gate if the environment or base commit changed.
3. Audit the rank-four `_optimize_transpose_pre_concat_nhwc_chains` matcher and
   its existing tests. Select one bounded semantic subfamily and add compact
   success/no-op characterization before moving implementation.
4. Preserve the staged sequence: characterization → exact mechanical
   extraction → indexed transactional runner. Use one representative model at
   most when the selected pass has a known exercising model.
5. If any future sequential model run reports `swap_detected`, immediately add
   that model to the managed profile exclusion set, update exact count tests,
   and start any later authoritative run with a clean output directory.
6. Commit and push each safe checkpoint to `fb-refactor4`; do not open a pull
   request.

## Pause checkpoint

- Branch: `fb-refactor3`
- Latest implementation checkpoint: `9a09553`
  (`characterize ndhwc pre concat layout`)
- Previous pause checkpoint: `3df2903`
  (`document flatbuffer direct pause checkpoint`)
- Remote: after this resumed documentation checkpoint is pushed, local and
  `origin/fb-refactor3` must again report `0 0` divergence.
- Pull request: none; do not create one on resume
- The axis-3 constant-Concat bridge matcher uses pure indexed planning,
  differential graph/layout mutation, and one stable transactional runner.
  Its lowerer wrapper remains; the raw production call is removed.
- The Dequantize/Concat/Quantize matcher uses pure indexed planning,
  differential graph/layout mutation, and one transactional runner at both
  production positions.
- The Concat/optional-unary/post-adapter/Conv matcher uses pure indexed
  planning, differential graph/layout mutation, and one transactional runner
  at both production positions.
- The seven-call SPP family has a generic characterization corpus and its
  complete matcher is owned by `passes/spp_layout.py`. Pure indexed planning,
  shared-constant copy-on-write, differential graph/layout mutation, and
  stable runner `layout.generic_spp_nhwc` replace all seven raw calls.
- The five-call NDHWC pre-Concat matcher now has a dedicated 16-case compact
  characterization corpus. Production remains central and unchanged.

## Completed work

This resumed interval completed sixteen adjacent semantic layout families
using the staged characterization → mechanical extraction → indexed runner
process and completed the same three-stage migration for the seventeenth
family. Characterization is complete for the eighteenth family.

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
    - Added shared indexed candidate/mutation state, suffix-constant
      copy-on-write, corrected post-tensor metadata, stable transactional runner
      `layout.add_concat_const_suffix_nhwc`, and five runner calls in `1b8c307`.
13. Dual-Mul/Concat layout
    - Moved the 131-line central success fixture to a dedicated compact corpus
      in `82d8777`, retaining shared-constant copy-on-write coverage.
    - Added ten whole-ModelIR no-op boundaries for adapter/Mul/Concat fan-out,
      public tensors, permutations, axis, missing constant, and non-shared data
      branches.
    - Preserved production behavior and all six raw call positions;
      mechanical extraction remains the next checkpoint.
    - Moved the complete matcher to `passes/dual_mul_concat_layout.py` with AST
      equivalence in `af26412`; the compatibility wrapper and all six raw
      positions remain.
    - Added pure indexed topology/broadcast planning, differential
      copy-on-write and graph/layout mutation, corrected post metadata, stable
      runner `layout.dual_mul_concat_nhwc`, and six runner calls in `64702b2`.
14. Axis-3 constant-Concat bridge characterization
    - Moved the sole 132-line central success fixture to
      `tests/test_flatbuffer_direct_axis3_const_concat_layout.py` in `019d3c6`.
    - Added compact success variants for multiple inverse post branches and a
      safely shared leading adapter, while retaining the legacy NCHW-consumer
      bridge case.
    - Added nine complete no-op boundaries for public Concat/post tensors,
      invalid pre/post permutations, invalid axis, constant rank/shape/data,
      and a constant shared outside the Concat.
    - Preserved the central production matcher and its single call exactly;
      extraction is intentionally deferred to the next checkpoint.
    - Moved the complete matcher to
      `passes/axis3_const_concat_layout.py` with exact AST equivalence in
      `5228444`; the compatibility wrapper and single raw production call
      remain.
    - Added pure indexed constant/topology/bridge planning, protected public
      adapter and constant boundaries, differential removal/insertion,
      `LayoutState` reconciliation, stable runner
      `layout.axis3_const_concat_bridge_nhwc`, and the single production runner
      call in `a261462`.
15. Dequantize/Concat/Quantize layout characterization
    - Added the first dedicated corpus for the previously untested central
      matcher in `ea74ffd`.
    - Fixed ordinary rewrite, multiple post-adapter canonicalization, shared
      pre-adapter retention, and quantization metadata preservation.
    - Added twelve complete no-op boundaries covering intermediate fan-out,
      public tensors, invalid permutations/axis, and a non-Dequantize branch.
    - Preserved the central production matcher and both raw calls exactly;
      extraction is the next checkpoint.
    - Moved the complete matcher to
      `passes/dequant_concat_quantize_layout.py` with exact AST equivalence in
      `35a4cb1`; the compatibility wrapper and both raw calls remain.
    - Added pure indexed topology and quantization-metadata planning,
      differential graph/layout mutation, stable runner
      `layout.dequant_concat_quantize_nhwc`, and both production runner calls
      in `3be0c3e`.
16. Concat/unary/Conv layout characterization
    - Added the first dedicated compact corpus for the central matcher in
      `f624388`.
    - Fixed unary-free and two-unary/two-post success variants, including
      Conv2D and DepthwiseConv2D consumers.
    - Added thirteen complete no-op boundaries for fan-out, graph outputs,
      permutations, axis, input/unary type, and non-Conv consumers.
    - Preserved production code and both raw calls exactly; extraction remains
      the next checkpoint.
    - Moved the complete matcher to `passes/concat_unary_conv_layout.py` with
      exact AST equivalence in `11e76bd`; the wrapper and both calls remain.
    - Added pure indexed adapter/unary/post/Conv planning, rank-four guards,
      differential graph/layout mutation, stable runner
      `layout.concat_unary_conv_nhwc`, and both production runner calls in
      `b86b31a`.
17. Generic two-island SPP layout characterization
    - Added the first dedicated corpus for the 371-line, seven-call matcher in
      `0804e37`.
    - Replaced implicit model-specific coverage with a compact four-branch,
      two-Concat, two-affine, two-Conv semantic graph.
    - Added sixteen complete no-op boundaries for fan-out, public tensors,
      permutation/axis, Resize producer, and missing Mul constants.
    - Preserved production code and all seven raw calls exactly at the
      characterization checkpoint.
    - Moved the complete matcher mechanically to `passes/spp_layout.py` in
      `c531b54`. Its function AST exactly matches characterization checkpoint
      `0804e37`; the lowerer retains a compatibility wrapper and all seven raw
      production calls.
    - Added full indexed topology/rank/constant planning, shared-constant
      copy-on-write, quantized-dimension remapping, differential graph/layout
      mutation, stable runner `layout.generic_spp_nhwc`, and all seven runner
      calls in `8edf5c2`.
18. NDHWC pre-Concat layout characterization
    - Moved the only 96-line central success fixture to
      `tests/test_flatbuffer_direct_ndhwc_concat_layout.py` in `9a09553`.
    - Extended success coverage to mixed direct/unary inputs with two inverse
      post adapters and canonical alias replacement.
    - Added fifteen complete no-op boundaries covering input/unary/Concat
      fan-out, public tensors, invalid permutations/axis, unsupported unary,
      invalid rank, and incompatible spatial shape.
    - Preserved the complete 258-line production matcher and all five raw
      production calls; mechanical extraction is the next checkpoint.

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
  The immediate next unit is exact-AST mechanical extraction of the 258-line,
  five-call `_optimize_transpose_pre_concat_ndhwc_chains` matcher. Keep the
  much larger generic NHWC pre-Concat matcher as a separately planned family;
  source length is not a Goal gate.
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

Current branch is `fb-refactor3`. Before this resumed documentation update, the
working tree is clean at NDHWC characterization checkpoint `9a09553`. The
indexed SPP work and the dedicated NDHWC pre-Concat corpus are committed;
NDHWC mechanical extraction has not begun.
After the documentation commit is pushed, local/remote divergence must be
`0 0`. The implementation checkpoints since the previous pause changed:

- `docs/flatbuffer_direct_architecture.md`
- `docs/flatbuffer_direct_handoff_2026-07-12.md`
- `docs/flatbuffer_direct_handoff_2026-07-13.md`
- `onnx2tf/tflite_builder/core/model_ir_utils.py`
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`
- `onnx2tf/tflite_builder/passes/elementwise_gate_layout.py`
- `onnx2tf/tflite_builder/passes/axis3_const_concat_layout.py`
- `onnx2tf/tflite_builder/passes/dequant_concat_quantize_layout.py`
- `onnx2tf/tflite_builder/passes/concat_unary_conv_layout.py`
- `onnx2tf/tflite_builder/passes/spp_layout.py`
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
- `tests/test_flatbuffer_direct_axis3_const_concat_layout.py`
- `tests/test_flatbuffer_direct_concat_unary_conv_layout.py`
- `tests/test_flatbuffer_direct_dequant_concat_quantize_layout.py`
- `tests/test_flatbuffer_direct_ndhwc_concat_layout.py`
- `tests/test_flatbuffer_direct_spp_layout.py`
- `tests/test_tflite_builder_direct.py`

The resumed handoff checkpoint updates documentation only. No implementation
or test file remains uncommitted after it.

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
- Indexed Add/Concat suffix runner, shared-constant, architecture, and
  efficiency focus: `63 passed in 19.43s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed Add/Concat suffix migration:
  `1275 passed, 5 deselected, 2 warnings in 158.36s`.
- Dedicated dual-Mul/Concat characterization: `11 passed in 0.29s`.
- Focused dual-Mul selection including residual central-name coverage:
  `12 passed, 756 deselected in 2.71s`.
- Full direct selection after moving the fixture and adding ten boundaries:
  `1285 passed, 5 deselected, 2 warnings in 165.06s`.
- Dual-Mul/Concat extraction, characterization, and ownership focus:
  `49 passed in 20.24s`; the extracted function AST exactly matched `82d8777`.
- Full direct selection after mechanical extraction:
  `1286 passed, 5 deselected, 2 warnings in 163.50s`.
- Indexed dual-Mul/Concat runner, broadcast-plan, architecture, and efficiency
  focus: `64 passed in 20.85s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed dual-Mul/Concat migration:
  `1297 passed, 5 deselected, 2 warnings in 165.59s`.
- Dedicated axis-3 constant-Concat bridge characterization:
  `12 passed in 0.29s`.
- Residual central selection after moving the fixture:
  `756 deselected in 0.37s`; no duplicate central test remains.
- Full direct selection after characterization:
  `1308 passed, 5 deselected, 2 warnings in 166.51s`.
- Axis-3 constant-Concat extraction, characterization, and architecture focus:
  `51 passed in 20.60s`; the extracted function AST exactly matched `019d3c6`.
- Full direct selection after mechanical extraction:
  `1309 passed, 5 deselected, 2 warnings in 165.14s`.
- Indexed axis-3 constant-Concat runner, public-boundary, architecture, and
  irrelevant-graph efficiency focus: `69 passed in 21.95s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed migration:
  `1323 passed, 5 deselected, 2 warnings in 166.52s`.
- Dedicated Dequantize/Concat/Quantize characterization:
  `15 passed in 0.34s`.
- Full direct selection after characterization:
  `1338 passed, 5 deselected, 2 warnings in 166.21s`.
- Dequantize/Concat/Quantize extraction, characterization, and architecture
  focus: `55 passed in 20.93s`; the extracted function AST exactly matched
  `ea74ffd`.
- Full direct selection after mechanical extraction:
  `1339 passed, 5 deselected, 2 warnings in 176.09s`.
- Indexed Dequantize/Concat/Quantize runner, metadata/rank boundary,
  architecture, and irrelevant-graph efficiency focus:
  `78 passed in 21.88s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed migration:
  `1358 passed, 5 deselected, 2 warnings in 173.55s`.
- Dedicated Concat/unary/Conv characterization: `15 passed in 0.31s`.
- Full direct selection after characterization:
  `1373 passed, 5 deselected, 2 warnings in 172.58s`.
- Concat/unary/Conv extraction, characterization, and architecture focus:
  `56 passed in 23.44s`; the extracted function AST exactly matched `f624388`.
- Full direct selection after mechanical extraction:
  `1374 passed, 5 deselected, 2 warnings in 175.23s`.
- Indexed Concat/unary/Conv runner, rank boundary, architecture, and
  irrelevant-graph efficiency focus: `76 passed in 24.25s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed migration:
  `1390 passed, 5 deselected, 2 warnings in 173.99s`.
- Dedicated generic SPP characterization: `17 passed in 0.33s`.
- Full direct selection after characterization:
  `1407 passed, 5 deselected, 2 warnings in 176.30s`.
- SPP characterization plus architecture ownership after mechanical
  extraction: `59 passed in 24.76s`; the extracted function AST exactly
  matched `0804e37`.
- Full direct selection after SPP mechanical extraction:
  `1408 passed, 5 deselected, 2 warnings in 174.80s`.
- Indexed SPP success, shared-constant copy-on-write, quantized-dimension,
  no-op, runner, architecture, and irrelevant-graph efficiency focus:
  `85 passed in 23.15s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed SPP migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed SPP migration:
  `1430 passed, 5 deselected, 2 warnings in 161.91s`.
- Dedicated NDHWC pre-ConCat characterization: `16 passed in 0.30s`.
- Full direct selection after moving the central fixture and adding fifteen
  no-op boundaries:
  `1445 passed, 5 deselected, 2 warnings in 170.91s`.
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
2. Move the complete
   `_optimize_transpose_pre_concat_ndhwc_chains` implementation mechanically
   to a focused pass module while retaining its wrapper and all five raw calls.
3. Confirm exact function-AST equivalence against `9a09553`, add a single-owner
   architecture gate, and run focused plus full direct gates.
4. Commit and push extraction before beginning indexed candidate planning.

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

### Indexed Add/Concat/constant-suffix checkpoint

Checkpoint `1b8c307` replaced repeated producer/consumer map construction with
one indexed candidate plan and differential mutation state. The plan proves
all branch/base adapters, exclusive Add outputs, channel Concat, strict
MUL(const)→ADD(const) suffix, inverse output adapter, constants, fan-out, and
public boundaries before snapshotting. Add inputs, suffix output aliasing, and
operator removal now update `ModelIRGraphIndex`; pruning and metadata reconcile
the shared `LayoutState`.

Both suffix constants now use copy-on-write when another operator consumes the
same buffer. The optimized island receives an NHWC clone while unrelated
consumers retain the original NCHW data and metadata. The canonical post tensor
also retains the once-permuted NHWC shape instead of applying the legacy
metadata permutation twice. Dedicated tests cover both shared constants and
the corrected `[N,H,W,C]` output metadata.

`run_add_concat_suffix_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.add_concat_const_suffix_nhwc`; all five production positions call it
with session layout state and diagnostics. Focused success, shared-constant,
nine no-op boundaries, runner, ownership, architecture, and irrelevant-graph
efficiency validation passed 63 tests. The success graph uses one initial index
refresh and one snapshot; all unsafe boundaries reject before snapshotting.
Tier 1 `superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1275 passed, 5 deselected, 2 warnings in 158.36s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_add_concat_suffix_superpoint` artifacts were removed after
metrics inspection.

### Dual-Mul/Concat characterization checkpoint

Checkpoint `82d8777` moved the 131-line embedded success fixture to
`tests/test_flatbuffer_direct_dual_mul_concat_layout.py`. The compact graph
retains the shared data adapter, two Mul branches, channel Concat, inverse
output adapter, downstream Relu, and an external NCHW consumer of one Mul
constant. It proves direct NHWC data inputs, axis-3 Concat, canonical output
aliasing, in-place conversion of an exclusive constant, and an NHWC clone for
the externally shared constant while its original buffer remains NCHW.

Ten parameterized boundaries prove a complete ModelIR no-op for pre-adapter
fan-out, Mul-output fan-out, Concat fan-out, public Concat/post tensors, invalid
pre/post permutations, invalid Concat axis, missing constant data, and Mul
branches that do not share one adapted data input. Snapshots compare every
operator, option, tensor dtype, shape, signature, and constant value.
Production code and all six raw call positions remain unchanged.

Focused characterization passed 11 tests. The complete sequential direct
selection passed:

```text
1285 passed, 5 deselected, 2 warnings in 165.06s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Mechanical extraction with AST-equivalence and ownership gates
is the next separate checkpoint.

### Dual-Mul/Concat mechanical extraction checkpoint

Checkpoint `af26412` moved the complete 297-line matcher mechanically to
`passes/dual_mul_concat_layout.py`. Its function AST, including docstring and
nested copy-on-write helper, exactly matches checkpoint `82d8777`. The lowerer
keeps a signature-compatible wrapper, and all six raw production positions
remain unchanged to preserve ordering and retry behavior.

The architecture gate fixes the focused module as the single implementation
owner, its lowerer alias and wrapper, and exactly six production calls. Focused
characterization and ownership validation passed 49 tests. The complete
sequential direct selection passed:

```text
1286 passed, 5 deselected, 2 warnings in 163.50s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Indexed candidate planning, differential copy-on-write, layout
state reconciliation, runner integration, and raw-call replacement remain the
next checkpoint.

### Indexed dual-Mul/Concat checkpoint

Checkpoint `64702b2` introduced a pure indexed plan that proves the shared data
adapter, two exclusive Mul branches, Concat/post topology, public boundaries,
constant presence, rank, target broadcast compatibility, and whether each
constant requires cloning before any mutation. This prevents a later invalid
constant from leaving an earlier branch partially converted.

Constant copy-on-write and Mul input replacement now update one
`ModelIRGraphIndex`; Concat output aliasing, pre/post adapter removal, pruning,
and metadata reconciliation share the same `LayoutState`. The canonical post
tensor keeps its once-permuted NHWC shape instead of receiving the legacy
second metadata permutation. `run_dual_mul_concat_layout_cleanup` registers
stable `LAYOUT_PLAN` ID `layout.dual_mul_concat_nhwc`; all six production
positions now call it with session state and diagnostics.

Focused success, ten complete no-op boundaries, runner instrumentation,
ownership, architecture, and irrelevant-graph efficiency validation passed 64
tests. The success graph uses one initial index refresh and one snapshot; all
unsafe boundaries reject before snapshotting. Tier 1 `superpoint.onnx` passed
sequential `-tb flatbuffer_direct -cotof` with `evaluation_pass=true`,
`max_abs=1.6666017472743988e-06`, `rmse=1.6207873294228388e-07`, and cosine
similarity `1.0`.

The complete sequential direct selection passed:

```text
1297 passed, 5 deselected, 2 warnings in 165.59s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_dual_mul_concat_superpoint` artifacts were removed after metrics
inspection.

### Axis-3 constant-Concat bridge characterization checkpoint

Checkpoint `019d3c6` moved the only 132-line embedded success fixture to
`tests/test_flatbuffer_direct_axis3_const_concat_layout.py`. The compact base
graph retains one NHWC-to-NCHW input adapter, a rank-four NCHW constant,
axis-3 Concat, an inverse post adapter, and a legacy NCHW consumer. It proves
constant NCHW-to-NHWC conversion, axis remapping to 2, post-adapter bypass,
and insertion of exactly one NHWC-to-NCHW bridge for legacy consumers.

Two additional success variants prove that every inverse post branch is
bypassed and that a leading adapter shared with an unrelated consumer is
retained. Nine parameterized rejection cases prove a complete ModelIR no-op
for public Concat/post tensors, invalid pre/post permutations, invalid Concat
axis, invalid constant rank or incompatible shape, missing constant data, and
a constant shared outside the Concat. Snapshots compare every operator,
option, tensor dtype, shape, signature, and constant value.

Focused characterization passed 12 tests. The complete sequential direct
selection passed:

```text
1308 passed, 5 deselected, 2 warnings in 166.51s
```

Production code and the single raw call remain unchanged. No dependency or
TensorFlow path was added, and no inference process was run concurrently.
Mechanical extraction with exact AST-equivalence and single-owner gates is the
next separate checkpoint.

### Concat/unary/Conv mechanical extraction checkpoint

Checkpoint `11e76bd` moved the complete matcher mechanically to
`passes/concat_unary_conv_layout.py`. Its function AST, including the docstring
and optional-unary traversal, exactly matches characterization checkpoint
`f624388`. The lowerer keeps a signature-compatible wrapper and both raw
production calls.

The architecture gate fixes the focused module as the single implementation
owner, its lowerer alias and wrapper, and exactly two production calls. Focused
characterization and architecture validation passed 56 tests. The complete
sequential direct selection passed:

```text
1374 passed, 5 deselected, 2 warnings in 175.23s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Indexed candidate planning, differential graph/layout mutation,
transactional runner integration, and raw-call replacement remain the next
checkpoint.

### Indexed Concat/unary/Conv checkpoint

Checkpoint `b86b31a` introduced a pure indexed plan for every exclusive input
adapter, optional accepted-unary chain, complete inverse-post fan-out, and
Conv2D/DepthwiseConv2D consumer set. Rank-four source, Concat, unary, and post
metadata are validated before mutation; a new invalid-rank boundary rejects
before snapshot in addition to the thirteen characterized cases.

Concat input and axis mutation, Concat/unary metadata permutation, post alias
replacement, adapter/post removal, pruning, and layout reconciliation use one
shared `ModelIRGraphIndex` and `LayoutState`. The implementation contains no
whole-graph producer/consumer map construction and no direct operator-list
deletion. `run_concat_unary_conv_layout_cleanup` registers stable
`LAYOUT_PLAN` ID `layout.concat_unary_conv_nhwc`; both raw production calls are
replaced with the runner while the lowerer wrapper remains.

Focused success, fourteen complete no-op boundaries, runner instrumentation,
ownership, architecture, and irrelevant-graph efficiency validation passed 76
tests. The two-unary/two-post success graph uses one initial index refresh and
one snapshot; all unsafe boundaries reject before snapshotting. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1390 passed, 5 deselected, 2 warnings in 173.99s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_concat_unary_conv_superpoint` artifacts were removed after
metrics inspection.

### Generic two-island SPP characterization checkpoint

Checkpoint `0804e37` added `tests/test_flatbuffer_direct_spp_layout.py`, the
first dedicated coverage for the 371-line central matcher. The compact graph
uses four ResizeBilinear branches sharing one base adapter, four Add outputs,
the first channel Concat/Mul and inverse adapter, an NHWC affine/Conv, a return
adapter into a second base/Conv Concat/Mul, and a final inverse adapter,
affine, and Conv. It proves NHWC propagation through both islands, axis-3
Concat, channelwise constant conversion, and removal of eight adapters.

Sixteen parameterized boundaries prove a complete ModelIR no-op for branch,
Concat, Mul, inverse-post, and intermediate-Conv fan-out across both islands;
public base/first-Concat tensors; invalid leading permutation or either Concat
axis; a non-Resize branch producer; and missing first/second Mul constants.
Snapshots compare every operator, option, tensor shape/signature, and constant
value.

Focused characterization passed 17 tests. The complete sequential direct
selection passed:

```text
1407 passed, 5 deselected, 2 warnings in 176.30s
```

Checkpoint `c531b54` then moved the complete matcher mechanically to
`passes/spp_layout.py`. The function AST, including its docstring and legacy
selection/mutation order, exactly matches `0804e37`. The lowerer retains a
signature-compatible wrapper and all seven raw production calls. The
single-owner architecture gate fixes this boundary; focused SPP and
architecture validation passed 59 tests, and the complete sequential direct
selection passed 1,408 tests with the same five deselections and two known
warnings. No dependency or TensorFlow path was added, and no inference process
was run concurrently. Indexed planning and runner integration are the next
separate checkpoint.

### Indexed generic two-island SPP checkpoint

Checkpoint `8edf5c2` replaced the legacy map-rebuilding implementation with a
pure `_SppLayoutCandidate` that validates all four Resize/Add branches, both
Concat/Mul/adapter islands, the intervening and terminal Conv paths, every
fan-out/public boundary, rank-four metadata, and both constant payloads before
mutation. Shared constants are cloned only for the rewritten Mul inputs so
outside consumers retain the original NCHW payload. Per-axis quantization
metadata moves from NCHW dimension 1 to NHWC dimension 3 together with the
constant data.

All input rewrites, operator removals, pruning, and layout synchronization use
one shared `ModelIRGraphIndex` and `LayoutState`; the implementation contains
no whole-graph producer/consumer-map construction and no direct operator-list
deletion. `run_spp_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.generic_spp_nhwc`; all seven former raw production positions now call
it and the lowerer compatibility wrapper remains.

Focused validation passed 85 tests. A candidate uses one index refresh and one
transaction snapshot; every unsafe boundary rejects before snapshotting, and
an irrelevant 256-op graph builds no index, snapshot, or fingerprint. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with the
unchanged metrics `evaluation_pass=true`,
`max_abs=1.6666017472743988e-06`, `rmse=1.6207873294228388e-07`, and cosine
similarity `1.0`. The complete sequential direct selection passed:

```text
1430 passed, 5 deselected, 2 warnings in 161.91s
```

No dependency or TensorFlow import path was added. Temporary
`/tmp/onnx2tf_spp_superpoint` artifacts were removed after metrics inspection.
The next staged family is the adjacent 258-line, five-call NDHWC pre-Concat
matcher; the larger generic NHWC pre-Concat matcher remains a separate future
unit.

### NDHWC pre-Concat characterization checkpoint

Checkpoint `9a09553` moved the only 96-line central success fixture into
`tests/test_flatbuffer_direct_ndhwc_concat_layout.py` and expanded it into a
compact 16-case semantic corpus. The success graph combines one direct NDHWC
input adapter with one adapter/unary input, a channel-axis NCDHW Concat, and two
inverse post adapters. It fixes unary propagation in NDHWC, axis 4, canonical
post-output selection, alias replacement for the second post branch, and
removal of all three adapters.

Fifteen parameterized cases prove a complete ModelIR no-op for direct-adapter,
unary-adapter, unary-output, and Concat fan-out; public direct/unary/Concat/post
tensors; invalid pre/post permutations or Concat axis; an unsupported unary;
invalid direct-input rank; and incompatible projected spatial shapes. The
production matcher and all five raw call positions remain unchanged.

Focused characterization passed 16 tests. The complete sequential direct
selection passed:

```text
1445 passed, 5 deselected, 2 warnings in 170.91s
```

No dependency or TensorFlow path was added, and no inference process ran
concurrently. Exact-AST mechanical extraction against `9a09553` is the first
work on resume. The five-call count is authoritative; the earlier six-call
handoff text accidentally counted the function definition and is superseded.

### Dequantize/Concat/Quantize mechanical extraction checkpoint

Checkpoint `35a4cb1` moved the complete matcher mechanically to
`passes/dequant_concat_quantize_layout.py`. Its function AST, including the
docstring and all legacy selection/mutation order, exactly matches
characterization checkpoint `ea74ffd`. The lowerer keeps a
signature-compatible wrapper and both raw production calls.

The architecture gate fixes the focused module as the single implementation
owner, its lowerer alias and wrapper, and exactly two production calls. Focused
characterization and architecture validation passed 55 tests. The complete
sequential direct selection passed:

```text
1339 passed, 5 deselected, 2 warnings in 176.09s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Indexed candidate planning, differential graph/layout mutation,
transactional runner integration, and raw-call replacement remain the next
checkpoint.

### Indexed Dequantize/Concat/Quantize checkpoint

Checkpoint `3be0c3e` introduced pure indexed plans for every leading adapter,
exclusive Dequantize branch, rank-four Concat→Quantize edge, inverse post
branch, canonical quantized output, and adapter-removal decision. Quantize and
source quantization metadata and rank-four Concat metadata are validated before
mutation; three new malformed-metadata/rank cases reject before snapshot in
addition to the twelve characterized boundaries.

Dequantize input rewrites, metadata permutations, Concat axis mutation,
Quantize output canonicalization, post-alias replacement, adapter/post removal,
pruning, and layout reconciliation use one shared `ModelIRGraphIndex` and
`LayoutState`. The implementation contains no whole-graph producer/consumer
map construction and no direct operator-list deletion. The stable
`LAYOUT_PLAN` ID is `layout.dequant_concat_quantize_nhwc`; both raw production
calls are replaced with `run_dequant_concat_quantize_layout_cleanup`, while the
lowerer compatibility wrapper remains.

Focused success, fifteen complete no-op boundaries, runner instrumentation,
ownership, architecture, and irrelevant-graph efficiency validation passed 78
tests. The two-post success graph uses one initial index refresh and one
snapshot; all unsafe boundaries reject before snapshotting. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1358 passed, 5 deselected, 2 warnings in 173.55s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_dequant_concat_quantize_superpoint` artifacts were removed after
metrics inspection.

### Concat/unary/Conv characterization checkpoint

Checkpoint `f624388` added
`tests/test_flatbuffer_direct_concat_unary_conv_layout.py`, the first dedicated
coverage for this central matcher. One compact success graph proves the
unary-free Concat path; a second proves a RELU/Tanh chain with two inverse post
adapters ending in Conv2D and DepthwiseConv2D. Both remove every adapter,
rewrite Concat inputs and axis to NHWC, permute Concat/unary metadata once, and
feed every Conv-family consumer directly from the NHWC tail.

Thirteen parameterized boundaries prove a complete ModelIR no-op for leading
adapter, Concat, or unary fan-out; public adapter, Concat, unary, or post
tensors; invalid pre/post permutations; invalid Concat axis; a non-Transpose
input; an unsupported unary; and a non-Conv post consumer. Snapshots compare
operator options and all tensor metadata and constant values.

Focused characterization passed 15 tests. The complete sequential direct
selection passed:

```text
1373 passed, 5 deselected, 2 warnings in 172.58s
```

Production code and both raw calls remain unchanged. No dependency or
TensorFlow path was added, and no inference process was run concurrently.
Mechanical extraction with exact AST-equivalence and single-owner gates is the
next separate checkpoint.

### Axis-3 constant-Concat bridge mechanical extraction checkpoint

Checkpoint `5228444` moved the complete matcher mechanically to
`passes/axis3_const_concat_layout.py`. Its function AST, including the
docstring and nested helpers, exactly matches characterization checkpoint
`019d3c6`. The lowerer keeps a signature-compatible wrapper and the single raw
production call, so pass order and retry behavior remain unchanged.

The architecture gate fixes the focused module as the single implementation
owner, its lowerer alias and wrapper, and exactly one production call. Focused
characterization and architecture validation passed 51 tests. The complete
sequential direct selection passed:

```text
1309 passed, 5 deselected, 2 warnings in 165.14s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Indexed candidate planning, differential graph/layout mutation,
transactional runner integration, and raw-call replacement remain the next
checkpoint.

### Indexed axis-3 constant-Concat bridge checkpoint

Checkpoint `a261462` introduced pure indexed planning for the unique leading
adapter, every exclusive rank-four constant conversion, every inverse post
branch, the retained-adapter decision, and the optional legacy NCHW bridge.
All bridge metadata and shape compatibility are validated before mutation.
Public adapter and constant tensors now reject before snapshot in addition to
the nine characterized boundaries, preventing graph-output producer loss or
silent constant-layout changes.

Constant buffers, Concat inputs/axis, post aliases, legacy inputs, adapter/post
removal, bridge insertion, pruning, and layout reconciliation use one shared
`ModelIRGraphIndex` and `LayoutState`. The implementation contains no
whole-graph producer/consumer map construction and no direct operator-list
insertion or deletion. `run_axis3_const_concat_layout_cleanup` registers stable
`LAYOUT_PLAN` ID `layout.axis3_const_concat_bridge_nhwc`; the single raw
production call is replaced with the runner while the lowerer compatibility
wrapper remains.

Focused success, eleven complete no-op boundaries, runner instrumentation,
ownership, architecture, and irrelevant-graph efficiency validation passed 69
tests. The success graph uses one initial index refresh and one snapshot; all
unsafe boundaries reject before snapshotting. Tier 1 `superpoint.onnx` passed
sequential `-tb flatbuffer_direct -cotof` with `evaluation_pass=true`,
`max_abs=1.6666017472743988e-06`, `rmse=1.6207873294228388e-07`, and cosine
similarity `1.0`.

The complete sequential direct selection passed:

```text
1323 passed, 5 deselected, 2 warnings in 166.52s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_axis3_const_concat_superpoint` artifacts were removed after
metrics inspection.

### Dequantize/Concat/Quantize characterization checkpoint

Checkpoint `ea74ffd` added
`tests/test_flatbuffer_direct_dequant_concat_quantize_layout.py`, the first
dedicated coverage for this central matcher. The compact quantized graph uses
two NHWC INT8 inputs, two leading adapters and Dequantize branches, axis-1
Concat, Quantize, inverse post adapters, and downstream Dequantize consumers.
It proves direct NHWC Dequantize inputs, axis-3 Concat, removal of exclusive
adapters, and preservation of the quantized dtype, shape, and full
`QuantParamIR` on the canonical output.

Additional success variants prove that multiple post adapters merge into one
canonical quantized tensor and that a leading adapter shared outside the
island remains available. Twelve parameterized boundaries prove a complete
ModelIR no-op for Dequantize/Concat/quantized fan-out; public pre,
Dequantize, Concat, Quantize, and post tensors; invalid pre/post permutations;
invalid Concat axis; and a non-Dequantize branch. Snapshots include operator
options, tensor metadata, quantization, and constant values.

Focused characterization passed 15 tests. The complete sequential direct
selection passed:

```text
1338 passed, 5 deselected, 2 warnings in 166.21s
```

Production code and both raw calls remain unchanged. No dependency or
TensorFlow path was added, and no inference process was run concurrently.
Mechanical extraction with exact AST-equivalence and single-owner gates is the
next separate checkpoint.
