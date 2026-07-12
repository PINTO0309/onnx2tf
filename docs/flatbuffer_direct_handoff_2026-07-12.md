# flatbuffer_direct refactor handoff — 2026-07-12

## Current resumed checkpoint — `fb-refactor3`

Coverage and tensor-correspondence reporting have been extracted from
`lower_from_onnx2tf.py` into `tflite_builder/reporting.py`.

- Schema policy construction, node dispatch diagnostics, report assembly,
  lineage tracing, downstream correspondence inference, and both JSON writers
  now have one reporting owner.
- `_build_tensor_consumer_map` also moved to the reporting module and is
  imported back by the legacy lowerer for its existing layout passes.
- `lower_from_onnx2tf.py` retains the four public report functions as thin
  signature-preserving wrappers. Existing imports from both
  `lower_from_onnx2tf` and `tflite_builder` remain compatible.
- Imported ONNX-analysis and constant-fold helper symbols previously visible
  through `lower_from_onnx2tf.py` remain explicit compatibility re-exports.
- An architecture test verifies reporting implementation ownership, wrapper
  delegation, and the TensorFlow-free import boundary without imposing any
  source-line threshold.

Sequential verification in the core `uv` environment completed with:

- `33 passed, 749 deselected` for focused coverage/report integration;
- `35 passed` for reporting coverage plus architecture checks;
- `986 passed, 5 deselected, 2 warnings in 122.89s` for the full direct suite.

The five deselections and two FLOAT16 warnings are the same optional-environment
and expected-warning set documented below. No report schema, output path,
public signature, or direct conversion behavior changed.

Static high-rank BatchMatMul compression has also moved to
`passes/high_rank_matmul.py`. The legacy
`_compress_static_high_rank_batch_matmul` symbol remains a thin wrapper, so
existing tests and downstream imports continue to work. Its two dependencies,
`_is_fully_known_positive_shape` and `_prune_unused_tensors`, now have one
canonical implementation in `core/model_ir_utils.py`; the duplicate lowerer,
precision-pass, and constant-fold definitions were removed. A focused utility
test fixes deterministic pruning and lineage-event behavior, and an
architecture test fixes pass/helper ownership without a source-line gate.

Verification after this extraction completed with:

- `15 passed` for architecture and common ModelIR utility tests;
- `989 passed, 5 deselected, 2 warnings in 122.64s` for the full sequential
  direct suite.

The boundary input layout transpose pass now lives in
`passes/boundary_input_layout.py`, with the legacy lowerer symbol delegating to
it. `_build_tensor_consumer_map`, `_read_transpose_perm`, and
`_replace_tensor_inputs` are canonicalized in `core/model_ir_utils.py`; the
reporting and precision copies of the consumer map were removed. Focused tests
preserve the metadata-equality guard, the GatherND safety guard, deterministic
consumer indices, transpose permutation decoding, and lineage-aware input
replacement.

Verification completed with:

- `5 passed, 794 deselected` for focused boundary/report compatibility;
- `991 passed, 5 deselected, 2 warnings in 124.96s` for the full sequential
  direct suite.

The channel-slice and StridedSlice/QDQ/Concat boundary family now lives in
`passes/channel_slice_layout.py`. Five legacy lowerer entry points delegate to
that module: boundary channel-slice elision, internal channel-slice NHWC
propagation, Mul/Add bridge propagation, strict dual-Add bridge propagation,
and boundary StridedSlice/QDQ/Concat cleanup.

Their generic dependencies are canonicalized in `core/model_ir_utils.py`:
operator input/output mutation, indexed input replacement, constant-vector
read/write, static broadcasting, tensor metadata permutation, consumer maps,
transpose permutations, quantization cloning, pruning, and lineage events.
The family preserves all existing semantic guards and does not introduce a
model-name rule.

Verification completed with:

- `6 passed, 771 deselected` for focused channel-slice/StridedSlice and utility
  cases;
- `18 passed` for architecture and common ModelIR utility ownership;
- `992 passed, 5 deselected, 2 warnings in 125.00s` for the full sequential
  direct suite.

The boundary input Mul/Sum/Reshape and BatchMatMul rewrites now live in
`passes/boundary_input_chains.py`. The two legacy lowerer entry points are thin
delegating wrappers. The move retains the original fan-out, model-output,
permutation, constant-shape, axis, and metadata guards and reuses canonical
graph mutation and constant-vector helpers from `core/model_ir_utils.py`.
Focused ModelIR fixtures cover the positive rewrite paths, including NHWC
constant rotation, reduction-axis remapping, intermediate metadata updates,
and BatchMatMul input rewiring.

Verification completed with:

- `18 passed` for architecture and boundary-input-chain tests, including
  fan-out and shared-input no-op guards;
- `996 passed, 5 deselected, 2 warnings in 124.71s` for the full sequential
  direct suite.

The next extraction candidate should be selected from the remaining layout
rewrite families in `lower_from_onnx2tf.py`, favoring a cohesive group whose
generic graph helpers already have canonical core owners. Preserve the legacy
symbols as wrappers and add both semantic-guard fixtures and the existing full
direct-suite gate before committing.

The generic leading-input passthrough rewrite has subsequently moved to
`passes/input_passthrough_layout.py`. It folds a strictly linear sequence of
layout-agnostic unary and constant-side binary operators across a synthetic
input transpose and its inverse output transpose. The implementation is an
exact mechanical move apart from removing one unused local assignment. The
legacy lowerer symbol delegates to the pass module.

`_invert_perm` now has one canonical implementation in
`core/model_ir_utils.py`. Focused tests preserve the positive NHWC rewrite,
constant rotation and metadata behavior, the main-path fan-out no-op guard,
and invalid-permutation rejection. Architecture tests fix both pass and helper
ownership.

Verification completed with:

- `21 passed` for architecture, common ModelIR utilities, and leading-input
  passthrough behavior;
- `999 passed, 5 deselected, 2 warnings in 125.65s` for the full sequential
  direct suite.

The next cohesive extraction can extend `input_passthrough_layout.py` with the
adjacent ASIN/ERF/HardSwish/HardSigmoid semantic passthrough families. Move one
guarded family at a time, keep its legacy entry point, and gate each increment
with focused ModelIR fixtures before the sequential full suite.

The ASIN/ACOS decomposition passthrough is now the second implementation in
`passes/input_passthrough_layout.py`. It preserves the strict
`Mul(x,x) → Sub → Sqrt → Atan2` topology, singleton subtraction constant,
consumer, boundary, inverse-permutation, and output guards. Its legacy lowerer
entry point delegates to the pass module. `_is_singleton_constant_tensor` now
has one canonical implementation in `core/model_ir_utils.py` while remaining
available through the legacy lowerer import surface.

Verification completed with:

- `24 passed` for architecture, common ModelIR utilities, generic input
  passthrough, and ASIN positive/no-op behavior;
- `1002 passed, 5 deselected, 2 warnings in 125.00s` for the full sequential
  direct suite.

The standalone HardSigmoid passthrough has also moved to
`passes/input_passthrough_layout.py`. It retains the strict singleton-side
`Mul → Add → Relu0To1` or `Mul → Add → Maximum → Minimum` decomposition,
single-consumer, inverse-permutation, and metadata guards. The legacy lowerer
entry point remains a thin wrapper. Positive and noninverse-permutation no-op
fixtures cover the Relu0To1 path.

Verification completed with:

- `26 passed` for the focused architecture, utility, and input-passthrough
  suite;
- `1004 passed, 5 deselected, 2 warnings in 125.56s` for the full sequential
  direct suite.

The ERF polynomial decomposition passthrough now lives in
`passes/input_passthrough_layout.py`. Its legacy symbol delegates to the exact
moved implementation. The pass preserves the ABS/SIGN branch split,
reciprocal prelude, square/exponential branch, four-stage Horner polynomial,
final sign merge, singleton constants, exact consumer counts, output boundary,
and inverse-permutation guards. A generated ModelIR fixture exercises the full
21-operator topology, and a non-singleton coefficient fixture fixes the no-op
path.

Verification completed with:

- `28 passed` for the focused architecture, utility, and input-passthrough
  suite;
- `1006 passed, 5 deselected, 2 warnings in 125.19s` for the full sequential
  direct suite.

The pseudo-expanded HardSwish passthrough now also lives in
`passes/input_passthrough_layout.py`. It preserves the residual
`Add → optional Relu6 → Div-or-Mul → Mul(original, branch)` topology,
singleton constants, strict consumers, inverse terminal permutation, metadata,
and output-name guards. Focused fixtures cover the Relu6/Div positive path and
a non-singleton divisor no-op path.

Verification completed with:

- `30 passed` for the focused architecture, utility, and input-passthrough
  suite;
- `1008 passed, 5 deselected, 2 warnings in 125.48s` for the full sequential
  direct suite.

The larger HardSigmoid-plus-residual-Mul passthrough now completes the current
`passes/input_passthrough_layout.py` family. The implementation is moved
mechanically and the legacy symbol delegates to it. Existing characterization
coverage preserves the expanded clamp, residual Mul, legacy fan-out adapter,
optional Mean output, reduction-axis remapping, metadata, and output-name
behavior. Architecture coverage fixes module ownership.

Verification completed with:

- `31 passed, 758 deselected` for focused architecture, utility,
  input-passthrough, and legacy fan-out characterization;
- `1008 passed, 5 deselected, 2 warnings in 125.54s` for the full sequential
  direct suite.

Pad layout is now the next cohesive family. The direct inverse-transpose Pad
rewrite and unary-to-Pad tail rewrite moved from the lowerer into the existing
`passes/pad_layout.py`, alongside the channel-last-input repair. Legacy lowerer
entry points delegate to the module. Existing characterization fixtures retain
padding-axis rotation, inverse permutations, dynamic metadata, quantization,
legacy fan-out slots, the optional local NCHW adapter, and output naming.

Verification completed with:

- `18 passed, 759 deselected` for focused architecture, Pad layout repair, and
  Pad pre/post characterization;
- `1009 passed, 5 deselected, 2 warnings in 125.61s` for the full sequential
  direct suite.

The guarded `Transpose → Pad → Mul → Transpose → Add` rewrite has now joined
`passes/pad_layout.py`. The exact implementation moved behind a legacy wrapper.
Its existing characterization preserves broadcast proof, Pad-axis and Mul
constant rotation, shared-constant cloning, inverse permutations, metadata,
and output rewiring.

Verification completed with:

- `19 passed, 758 deselected` for focused architecture, Pad repair, and the
  Pad/Mul/Add characterization;
- `1009 passed, 5 deselected, 2 warnings in 125.75s` for the full sequential
  direct suite.

The next Pad increment can evaluate the adjacent normalization-subgraph Pad
rewrite. Because that topology is substantially larger, first map its helper
dependencies and existing characterization coverage before moving it.

## Previous pause checkpoint — `fb-refactor2` after `19cb989`

### Completed work

- Completed the quantized op-family split. DynamicQuantizeLinear,
  QLinearMatMul/QGemm, QLinearAveragePool/GlobalAveragePool,
  QLinearAdd/QLinearMul, QLinearSigmoid/LeakyRelu/Softmax, QLinearConcat,
  QuantizeLinear/DequantizeLinear, QLinearConv, and ConvInteger now have
  dedicated family modules.
- Replaced the old combined `op_builders/quantized.py` with
  `op_builders/quantized_common.py`. The common module contains shared
  quantization, shape/signature, padding, and requantization primitives and no
  `build_*` entry point.
- Fixed pre-extraction normalized ModelIR fingerprints for each extracted
  family and consolidated duplicate fingerprint serialization in
  `tests/flatbuffer_direct_fingerprint.py`.
- Preserved the public builder imports, registry dispatch, TensorFlow-free
  boundary, and existing runtime behavior. The latest implementation commit is
  `19cb989` (`complete quantized op family split`) and is pushed to
  `origin/fb-refactor2`.
- Measured `lower_from_onnx2tf.py` with the Python AST. It contains 280
  top-level definitions, including 204 `_optimize_*` functions. The largest
  functions include `_optimize_transpose_pre_concat_nhwc_chains` (2,117
  lines), `lower_onnx_to_ir` (1,711 lines), and
  `_optimize_transpose_pre_add_nhwc_chains` (1,580 lines).
- Selected coverage/correspondence reporting as the next independent
  extraction boundary. The contiguous implementation is currently
  `_collect_schema_ops_for_range` through
  `write_tensor_correspondence_report`; `_build_tensor_consumer_map` at the top
  of the legacy module must move with it and be re-imported for layout passes.

### Incomplete work

- The reporting extraction has not been applied. An attempted generated patch
  failed context verification before changing any file; no partial
  `reporting.py` exists and all reporting functions remain in
  `lower_from_onnx2tf.py`.
- `lower_from_onnx2tf.py` remains approximately 78,000 lines and still owns the
  large layout-rule collection and the main lowering orchestration.
- The broader Goal remains incomplete: ordered pass ownership, transactional
  rewrite coverage, further layout-rule generalization, exporter cleanup, and
  final Tier 0–4/Tier 5 phase gates still require work.
- The managed Tier 0–4 baseline remains 368 passes, 6
  `missing_tflite_report`, 20 `tflite_fail`, and 26 excluded historical
  timeouts. The 26 active non-passes have explicit normalized causes; the two
  DEIM entries are accepted successes by user direction.

### Branch and working tree

- Branch: `fb-refactor2`, synchronized with `origin/fb-refactor2` at
  `19cb989` before this handoff-only checkpoint.
- There are no unfinished code changes or generated temporary files from the
  reporting attempt. This handoff document is the only intended checkpoint
  change before commit.

### Tests run

- Latest full sequential direct regression after the complete quantized split:
  `985 passed, 5 deselected, 2 warnings in 121.63s`.
- Quantized family fingerprint/architecture set: `26 passed`.
- Focused QLinearConv/ConvInteger extraction set: `12 passed, 760 deselected`.
- Reporting characterization command, run without the standard optional-test
  exclusions: `777 passed, 5 failed, 2 warnings in 117.09s`. All five failures
  are the already-known optional environment cases listed below; no reporting
  test failed.

### Failing tests and known issues

- Four TensorFlow converter tests fail because the core `uv` environment does
  not install the optional `tensorflow`/`tf_keras` extra:
  `test_tflite_backend_matrix_add`,
  `test_tflite_backend_matrix_hardswish_rewrite_on_off`,
  `test_tf_converter_resize_cubic_avoids_flex_resize_bicubic`, and
  `test_tf_converter_resize_cubic_honors_cubic_coeff_a`.
- `test_flatbuffer_direct_group_norm_alias_builtin_conversion` fails because a
  system Python 3.10 Torch binary is incompatible with the active Python 3.12
  `uv` environment. These five tests are the standard broad-suite exclusions.
- Two expected FLOAT16 cast overflow warnings remain in the ArgMax/ReduceMax
  and negative-infinity Where tests.
- The first reporting extraction patch failed only because its deletion hunk
  did not preserve the blank-line context before `_read_transpose_perm`; it
  made no filesystem change and is not a product defect.

### First action on resume

1. Reconfirm a clean `fb-refactor2` worktree and the current reporting function
   boundaries with `rg`/AST.
2. Create `onnx2tf/tflite_builder/reporting.py` with
   `_build_tensor_consumer_map`, coverage schema/policy/report writers, rewrite
   tracing, downstream correspondence inference, and correspondence writers.
3. Import/re-export `_build_tensor_consumer_map`, `build_op_coverage_report`,
   `write_op_coverage_report`, `build_tensor_correspondence_report`, and
   `write_tensor_correspondence_report` from `lower_from_onnx2tf.py` so existing
   Python imports remain compatible.
4. Run `tests/test_tflite_builder_op_coverage.py` plus the correspondence cases
   selected from `tests/test_tflite_builder_direct.py`, using the standard five
   optional-test exclusions, then run the full direct suite sequentially.
5. Only after identical reports and a green full suite, update this document,
   commit the reporting extraction, and batch the next push. Do not create a
   pull request.

## Current checkpoint — `fb-refactor2`

The opset-aware Resize lowering and numerically stable Inverse lowering recover
`onnx_dense_optimized.onnx` and its byte-identical `_org` counterpart without a
model-name rule or additional dependency.

- ONNX Resize in opset 10 now defaults to asymmetric coordinates, as required
  by that schema, instead of inheriting the opset 11+ `half_pixel` default.
- The generic 4×4 through 16×16 Inverse lowering now uses per-batch partial
  pivoting. At each Gauss–Jordan iteration it selects the largest absolute
  remaining pivot, swaps the selected row in both the matrix and identity
  state, and only substitutes a signed epsilon when the chosen pivot is
  genuinely near zero. Normal pivots are no longer shifted unconditionally.
- A synthetic opset-10 nearest-neighbor Resize test fixes the coordinate
  contract, and a batched 8×8 Inverse test requires an actual row swap and
  checks ONNX Runtime against the generated TFLite artifact.

Both dense models were evaluated sequentially with all seven outputs compared
and no skip. Their identical fixed-seed result is `evaluation_pass=true`,
`max_abs=0.00015753507614135742`, `mean_abs=2.2844531542128204e-06`,
`rmse=5.3903736593740145e-06`, and cosine similarity
`0.9999999998639824`. The recorded pre-fix baseline maximum was
`0.8238084316253662`; correcting Resize alone reduced it to
`0.16955818608403206`, and removing unconditional pivot perturbation reduced
it to `0.157419` before partial pivoting eliminated the amplified GridSample
error.

Legacy linear Upsample now follows the half-pixel coordinate semantics produced
by ONNX's v9-to-v11 version converter, while legacy nearest Upsample retains its
asymmetric behavior. This general rule recovers `modnet_old.onnx`, whose eight
linear downsample branches previously diverged before the first concatenation.
Its fixed-seed single output has no skip and reports `evaluation_pass=true`,
`max_abs=2.3931264877319336e-05`, `mean_abs=2.478012597297248e-07`,
`rmse=9.7739101243166e-07`, and cosine similarity `0.9999999999988499`.

`LINEA.onnx` also passes on the current static-input runtime path without an
additional lowering rule. Both outputs were compared with no skip:
`evaluation_pass=true`, `max_abs=0.002297189086675644`,
`mean_abs=3.4909796139056033e-06`, `rmse=7.162934073986925e-05`, and cosine
similarity `0.9994290428305682`.

The managed Tier 0–4 profile now records 368 passes, 6
`missing_tflite_report`, 20 `tflite_fail`, and 26 excluded historical timeouts.
There are 26 active non-passes, and every one now has an explicit normalized
cause or the expected invalid/custom-op runtime classification.

`rf-detr-nano.onnx` is promoted from its historical conversion failure to a
normal accuracy pass without a model-specific workaround. Its sequential
fixed-seed run compares both `pred_boxes` and `pred_logits` with no skip:
`evaluation_pass=true`, `max_abs=0.000102996826171875`,
`mean_abs=5.465599675581121e-06`, `rmse=1.015673578580791e-05`, and cosine
similarity `0.9999999999986942`. The source graph has 770 nodes and the lowered
graph has 722 nodes, keeping this recovery inside the Tier 3 active gate.

`LibreRFDETRn.onnx` is likewise promoted from its historical accuracy failure
to a normal pass without a model-specific workaround. Its sequential
fixed-seed run compares both `dets` and `labels` with no skip:
`evaluation_pass=true`, `max_abs=0.0001087188720703125`,
`mean_abs=5.6163524849373e-06`, `rmse=1.0355151438507286e-05`, and cosine
similarity `0.9999999999986449`. The model uses the existing explicit input
shape `input:1,3,384,384`; its source graph has 770 nodes and the lowered graph
has 722 nodes.

`bertsquad-12-int8.onnx` remains an active failure with normalized reason
`onnxruntime_u8s8_matmulinteger_cpu_saturation`. The direct implementation's
first U8×S8 MatMulInteger result matches an explicit INT32 NumPy product and
ONNX `ReferenceEvaluator` exactly. ONNX Runtime's CPUExecutionProvider differs
from that same product by `max_abs=11772`, mean absolute error
`326.5231577555339`, over 24,453 elements, identically at every graph
optimization level from disabled through all. This divergence starts at the
first encoder MatMulInteger even though the then-current preceding
DynamicQuantizeLinear differed at only two of 196,608 UINT8 elements by one.
That pre-correction final report had `max_abs=1.8257164359092712`. Emulating a
host-specific saturating CPU kernel would violate portable ONNX integer-matmul
semantics, so the exact lowering is retained and the previous failure-signature
hash remains fixed.

DynamicQuantizeLinear now uses nearest-even `ROUND` for both zero-point and
data quantization, and rounds `x / scale` before adding the integer zero point.
The previous `+0.5` then CAST path was half-up, and adding a large zero point
before rounding could erase a just-below-half fraction in FLOAT32. Synthetic
tests cover exact half values with both zero and odd nonzero zero points. Of
the active Tier 0–4 corpus, only `afhq_generator.v11.quant.onnx` and
`bertsquad-12-int8.onnx` contain this op; the only other occurrence is the
excluded Tier 4 timeout `vision_encoder_uint8.onnx`.

For `afhq_generator.v11.quant.onnx`, input DynamicQuantizeLinear now agrees at
every element and the first residual mismatch moves after an
InstanceNormalization difference of `4.76837158203125e-07`. Later quantization
boundaries still amplify sparse one-quantum differences through the decoder.
The final no-skip result improves from baseline `max_abs=0.22717905044555664`
to `0.21375656127929688`, with RMSE `0.03692099507463561` and cosine similarity
`0.999052579433886`; it remains a normal threshold failure with reason
`instance_normalization_drift_amplified_by_dynamic_quantization_decoder`.
The corrected BERT path remains dominated by ONNX Runtime's saturating CPU
MatMulInteger behavior; its no-skip fixed-seed maximum is now
`2.001576066017151`, with RMSE `1.2972128029177183` and cosine similarity
`0.9616353624777596`.

The now-fixed DynamicQuantizeLinear implementation was mechanically moved to
`op_builders/dynamic_quantize.py`, a dedicated 391-line op-family module. The
legacy combined `op_builders/quantized.py` shrinks from 3,235 to 2,850 lines.
The public builder import and registry dispatch remain unchanged. A normalized
ModelIR fingerprint covering all operators, tensor metadata, constants,
options, and quantization fields is identical at `d97cba6` and after the move:
`a83d642e4aa7903f9b34495fec2c1edb5ff8779ba6735bedde382578152657f5`
(22 operators, 27 tensors). The architecture regression verifies that the
implementation remains in its family module and is absent from the legacy
file. It does not impose a source-line limit.

QLinearMatMul and QGemm were then moved mechanically into the dedicated
238-line `op_builders/qlinear_fc.py` family module, reducing the remaining
legacy quantized builder from 2,850 to 2,634 lines. Pre-extraction ModelIR
fingerprints are now executable regression tests:
`633d083445fcf765023a948c038c0956c7a0b7646b73bdac0bb65cf4c14173c8`
for QLinearMatMul and
`bf71085f2cc3a5981b209b6d5b02cc65ea55a41251465229a5ef1636a319f70f`
for QGemm, each with 9 operators and 16 tensors. The architecture test keeps
both builders out of the legacy module and includes the new module in the
TensorFlow-import boundary. Sequential
one-sample CRNN verification through the new registry path is unchanged at
`max_abs=0.14842605590820312`, RMSE `0.0011565753987944503`, and cosine
similarity `0.999999996846642`.

QLinearAveragePool and QLinearGlobalAveragePool were subsequently moved
mechanically to `op_builders/qlinear_pool.py`. Public imports and registry
dispatch remain unchanged, and the legacy combined quantized module no longer
defines either builder. Focused ModelIR fingerprints are fixed at
`0bb8b9064ae208810addbcebb27846b05873d817e947a5af212f3fd8ee4a6b7c` and
`1b066e8245cb45f79df76dbc052ecf7485f07d7910fb789cff38b47c298b7f19`.
The full sequential direct regression completed with `970 passed, 5
deselected, 2 warnings in 121.97s`. The architecture checks enforce op-family
ownership and the TensorFlow-free import boundary only; they intentionally do
not enforce a source-line count.

The Goal's `2,000` threshold applies exclusively to ONNX graph operation/node
count: Tier 4 ends at 1,999 nodes and Tier 5 begins at 2,000 nodes. It is not a
limit on production or test source-file length.

QLinearAdd and QLinearMul were then moved mechanically to
`op_builders/qlinear_binary.py`. The dispatch and ModelIR contracts remain
unchanged. Their pre-extraction fingerprints are
`d2f0714a44b2dc376827b845269a217c1df894986f3957128994a2913d611c24`
(9 operators, 15 tensors) and
`b4d9d1a39202474faf52ab43fbde4938fe892a0a38c5739a87b6da2d9b882b34`
(4 operators, 6 tensors), respectively. The fingerprint implementation is now
shared by the FC, pooling, and binary family tests, removing duplicated
normalization and ModelIR serialization code. Existing QLinearAdd rounding,
QLinearConv chain, and QLinear FC chain runtime checks pass through the new
import path. The full sequential direct regression completed with `973 passed,
5 deselected, 2 warnings in 122.70s`.

QLinearSigmoid, QLinearLeakyRelu, and QLinearSoftmax were subsequently moved
mechanically to `op_builders/qlinear_activation.py`. Their pre-extraction
ModelIR fingerprints are
`67e5b3d23cf2cfe03ae8ef1a006ac5fecf221f328553d3c1904ceebad9a7d902`
(1 operator, 2 tensors),
`f1d0b1b74e6f0f056ca595912efcceb2827da416b059dc12992fd06ed137ab09`
(1 operator, 3 tensors), and
`56aef3cabbed33cabcaba95d36058a37b6a12428102f7e83b0aef334eadbb4ec`
(12 operators, 17 tensors). Focused Sigmoid/Softmax runtime and LeakyRelu rank
checks pass through the new import path. The full sequential direct regression
completed with `977 passed, 5 deselected, 2 warnings in 123.56s`.

QLinearConcat was then moved mechanically to
`op_builders/qlinear_concat.py`. Its pre-extraction fingerprint is
`924e1470c62f93ba44dde277144d84bf796f40c5123839b59b44e4cd89c5b927`
(6 operators, 7 tensors). The focused lowering and both concat-to-conv layout
propagation checks pass through the new import path.

QuantizeLinear and DequantizeLinear were moved mechanically to
`op_builders/quantize_linear.py`. Their shared two-node Q/DQ fixture retains
the pre-extraction fingerprint
`333343018c7bb32db3138cefdf4007353140b044472017ae6c3b4cce762e8f91`
(2 operators, 3 tensors). Focused Q/DQ rounding, per-axis quantization, layout,
and QLinearConcat tests completed with `12 passed, 761 deselected`. The full
sequential direct regression completed with `981 passed, 5 deselected, 2
warnings in 122.78s`.

QLinearConv was moved mechanically to `op_builders/qlinear_conv.py`. The
mixed UINT8 activation / INT8 filter fixture retains its pre-extraction
fingerprint
`c752a5b1e31744e65d483733f55a688f2189d6bf11436cabd498cfc6a2ef5019`
(17 operators, 29 tensors). Focused mixed-dtype runtime, filter layout,
explicit/symbolic padding, dynamic-batch, and unknown-rank checks pass through
the new import path.

ConvInteger was moved mechanically to `op_builders/conv_integer.py`. Its
pre-extraction fingerprint remains
`587f53091ce42815e43946d7b73324fe31ec7d5aeb1c3d2d749097351106dfb5`
(7 operators, 13 tensors), and its focused builtin lowering check remains
unchanged. With all builders extracted, `op_builders/quantized.py` was renamed
to `op_builders/quantized_common.py`; it now exposes only shared quantization,
shape, padding, and requantization primitives. All family imports and the
TensorFlow-free architecture boundary use the new common-module name. The
complete quantized family fingerprint/architecture set completed with `26
passed`, and the full sequential direct regression completed with `985 passed,
5 deselected, 2 warnings in 121.63s`.

`campp_vin.onnx` is promoted from an historical accuracy failure to a normal
pass. Its concretized dynamic-time artifact fails during XNNPACK reshape
preparation, so isolated evaluation now retries once, sequentially, with the
builtin interpreter after a default-delegate worker failure. The builtin run
compares the single `output` tensor with no skip and reports
`evaluation_pass=true`, `max_abs=3.3020973205566406e-05`,
`mean_abs=8.416682248935103e-06`, `rmse=1.0447499868661927e-05`, and cosine
similarity `0.9999999999694269`. Successful default-delegate evaluation is
unchanged, and builtin failures are not retried.

`best.onnx` and `best_org.onnx` remain failures rather than receiving a relaxed
tolerance. Both simplify to the same 516-node Q/DQ graph and produce identical
fixed-seed metrics with no output skip: `max_abs=58.7506103515625`,
`mean_abs=0.12212618568213444`, `rmse=0.9568672461041465`, and cosine similarity
`0.9998485974152098`. Their first material mismatch is a sparse QuantizeLinear
rounding outlier (`max_abs=0.13601922988891602` over a
`[1,16,128,160]` tensor), which is repeatedly amplified through the Q/DQ–Conv
backbone and detector decode. The managed reason is now
`qdq_rounding_outliers_amplified_by_detector_decode`; each model retains its
previous normalized failure-signature hash.

The two DEIM variants are treated as accepted successes by explicit user
direction despite the normal metric-threshold judgement remaining false.
Before the first decoder TopK, the small fp16 variant differs by normal fp16
backbone increments while the larger variant's score head is within
`max_abs=0.000110626220703125`. Near-tied score ordering then changes TopK
indices by as much as `1909` and `4205`, respectively. Query gathering and the
final postprocessor TopK amplify this discontinuity to final label maxima of
`27.0` and `20.0`. Both baseline entries record
`user_approved_topk_index_instability_from_near_tied_scores`; no model-name
lowering rule, global tolerance relaxation, or forced index ordering was added.
No cause-unclassified active Tier 0–4 model remains at this checkpoint.

`text_recognition_CRNN_CN_2021nov_int8.onnx` retains its failure but now has
the more precise reason
`lstm_float_drift_crosses_quantization_boundary_before_qlinear_matmul`. The
second fused LSTM is within `max_abs=4.181463737040758e-06`; exactly one value
at `[23,266]` crosses the next QuantizeLinear boundary by one quantum. Six
QLinearMatMul outputs consequently differ by one quantum. For both the ONNX
and direct tensors, an explicit INT32 NumPy matmul plus declared requantization
matches every QLinearMatMul element when fed that runtime's own quantized
input. The final `max_abs=0.14842605590820312` therefore does not justify a
matmul rewrite or a semantics-changing rounding bias.

Those GridSample siblings remain normal threshold failures. Their upstream
feature tensors agree to roughly `1e-4`, and the generated grid agrees except
for sparse coordinates (`max_abs=0.010283231735229492`). With
`align_corners=1` and zero padding, those coordinates cross the discontinuous
inside/outside boundary and the three GridSample outputs amplify the difference
before decoding. The final no-skip metrics are `max_abs=0.296916950494051`,
`rmse=0.007787096662049418`, cosine `0.9998945091127481` for `model_70`, and
`max_abs=0.2830471396446228`, `rmse=0.0074818409992842005`, cosine
`0.9999115783578202` for `model_grid_sample`. Both now use the normalized reason
`grid_coordinate_rounding_amplified_at_zero_padding_boundary` while retaining
their previous failure-signature hashes.

Validation completed in the core `uv` environment, with one pytest process and
no parallel workers:

- `967 passed, 5 deselected, 2 warnings` after the QLinear FC family
  extraction, its two pre-extraction fingerprint tests, and architecture test;
- `964 passed, 5 deselected, 2 warnings` after the DynamicQuantizeLinear
  op-family extraction and architecture boundary test;
- `963 passed, 5 deselected, 2 warnings` after the DynamicQuantizeLinear
  nearest-even and round-before-zero-point correction;
- `5 passed, 782 deselected` for the focused DynamicQuantizeLinear runtime and
  managed-profile checks;
- `961 passed, 5 deselected, 2 warnings` across the direct builder, op
  coverage, all `flatbuffer_direct` regression modules, and all accuracy
  evaluator modules after the delegate fallback change;
- `73 passed` for the focused accuracy evaluator and managed baseline set;
- `rf-detr-nano.onnx`, `LibreRFDETRn.onnx`, `bertsquad-12-int8.onnx`, and
  `campp_vin.onnx` were each run sequentially end to end with `-cotof`;
- `793 passed, 7 deselected, 2 warnings` across the direct builder, managed
  profile, architecture/import boundary, and the two new regression files;
- `28 passed, 772 deselected` for the focused Inverse, Resize, managed profile,
  and TensorFlow-free checks;
- `23 passed, 750 deselected` for the subsequent legacy Upsample, Resize,
  managed profile, and TensorFlow-free regression set;
- `1 passed, 756 deselected` for Compress after removing a pre-existing unused
  local from the touched Resize module;
- both dense corpus models passed sequential end-to-end `-cotof` runs.

The five current broad-suite deselections are four optional TensorFlow backend
tests and the optional Torch GroupNorm integration test. The core environment
does not install TensorFlow and exposes an incompatible system Python 3.10
Torch binary to Python 3.12. No optional dependency was installed for this
checkpoint.

## Previous checkpoint — static delegate family after `53bba37`

The static-input delegate capability introduced by `53bba37` also recovers a
four-model AnimeGAN/face-paint family that previously shared the same unresolved
accuracy signature:

- `anime-gan-v2.onnx`;
- `anime-gan-v2_org.onnx`;
- `face_paint_512_v2_0.onnx`;
- `model_paint_v2_test.onnx`.

All four fixed-seed sequential runs compared their only output with no skip and
produced identical metrics: `evaluation_pass=true`,
`max_abs=0.0017458945512771606`, `mean_abs=0.0003080219994663925`,
`rmse=0.0003674835052452457`, and cosine similarity
`0.9999997946255107`. Their previous delegate-free baseline maximum was
`0.037707426119595766`. No model-specific lowering or tolerance was added.

The managed Tier 0–4 profile now records 359 passes, 6
`missing_tflite_report`, 29 `tflite_fail`, and 26 excluded historical timeouts.
There are 35 active non-passes. The next unresolved generic accuracy group in
managed order is `onnx_dense_optimized.onnx` and its `_org` counterpart; the
earlier remaining failures already carry explicit normalized reasons.

## Previous checkpoint — `fb-refactor2` at `53bba37`

The current checkpoint recovers `vit_b_encoder.onnx` and removes a general
large-model evaluation bottleneck without changing conversion semantics.

- The direct branch releases its legacy GraphSurgeon graph before ModelIR
  lowering. That graph duplicated hundreds of megabytes of initializers and is
  never consumed by the direct pipeline.
- After export, unreachable ModelIR and serialization clones are collected.
  On glibc systems, unused allocator arenas are returned to the OS; the trim is
  optional and failure-tolerant on other libc/platform combinations.
- Isolated evaluation no longer pickles a complete ONNX protobuf through the
  multiprocessing pipe. It writes one managed evaluation model, passes only
  its path, and lets ONNX Runtime open it directly. The ONNX worker is still
  fully reaped before the TFLite worker starts.
- The evaluation graph and temporary worker model are released at their phase
  boundaries. Managed temporary files are removed after comparison.
- LiteRT's default delegate is enabled only when every requested input shape is
  statically positive. Dynamic-shape models retain the existing
  delegate-disabled safety path. This is a capability rule, not a model-name
  or model-size workaround.

Before the fix, each backend was healthy in isolation (ONNX Runtime inference
`2.68s`, LiteRT with XNNPACK `10.94s`), but conversion-plus-evaluation exceeded
300 seconds because the parent retained several graph/protobuf copies while a
delegate-free worker ran the ViT. The final sequential end-to-end run completed
in `40.55s`, with peak RSS `4,632,968 KiB`. Its only output was compared with no
skip:

- `evaluation_pass=true`;
- `max_abs=2.6226043701171875e-06`;
- `mean_abs=1.260113801429541e-07`;
- `rmse=1.9330447647107274e-07`;
- cosine similarity `0.9999999999992181`.

The expanded affected suite completed with `884 passed, 5 deselected,
2 warnings in 113.33s`. Focused evaluator, subprocess, import-boundary, managed
profile, and memory tests also passed. Every worker and model ran sequentially;
no process pool or parallel pytest worker was used.

The managed Tier 0–4 profile now records 355 passes, 6
`missing_tflite_report`, 33 `tflite_fail`, and 26 excluded historical timeouts.
There are 39 active non-passes. Every remaining missing-report entry has a
documented unsupported semantic or invalid-source reason. The next unresolved
accuracy group in managed order starts with `anime-gan-v2.onnx`.

During this checkpoint, obsolete Goal-generated temporary conversions,
diagnostic models, and historical bulk-run directories were removed from
`/tmp`. Available filesystem space increased from approximately `63 GiB` to
`157 GiB`; repository models and tracked files were not removed.

## Earlier checkpoint — `fb-refactor2` at `d278bcf`

The next checkpoint recovers `tiny_decoder_11.onnx` by making dynamic
ScatterND negative-index normalization safe for index tensors above rank 4.
The implementation remains TensorFlow-free and introduces no dependency.

- LiteRT's elementwise comparison broadcast path aborts in native code when
  the result rank exceeds four. The decoder exposed this with eight `LESS`
  operations over dynamic `[1,1,1,1,4]` ScatterND indices.
- The generic ScatterND helper now temporarily coalesces the leading index
  dimensions to `[-1,K]`, normalizes each negative coordinate against the
  indexed data-shape prefix, and reshapes the normalized coordinates back to
  their original runtime shape before both ScatterND operations.
- Only the comparison/normalization representation is flattened. The public
  output, updates, index-vector dimension, dynamic leading dimensions, and
  ScatterND semantics are unchanged.
- The implementation is isolated in `op_builders/scatter_utils.py`; the large
  central `index.py` loses duplicated normalization construction rather than
  gaining another rule.
- A dedicated synthetic regression varies the dynamic leading dimension,
  exercises negative coordinates in a rank-5 index tensor, verifies the safe
  rank-2 `LESS` contract in ModelIR, and requires exact ONNX Runtime/TFLite
  output equality.

Sequential `tiny_decoder_11.onnx` verification used all four managed shape
hints and `keep_shape_absolutely_input_names` values. All three outputs were
compared with no skip:

- `evaluation_pass=true`;
- `max_abs=5.048513412475586e-05`;
- `mean_abs=1.637148860798228e-05`;
- `rmse=2.167510270660002e-05`;
- cosine similarity `0.9999999999902514`.

The affected sequential suite completed with `864 passed, 5 deselected,
2 warnings in 89.20s`. The optional TensorFlow/import-boundary suite separately
completed with `13 passed in 5.79s`. The five deselections and two float16
overflow warnings are the existing environment-specific cases documented
below. No parallel pytest worker or concurrent inference process was used.

The managed Tier 0–4 profile now records 354 passes, 7
`missing_tflite_report`, 33 `tflite_fail`, and 26 excluded historical timeouts.
There are 40 active non-passes remaining. Six missing-report entries already
have explicit unsupported or invalid-source reasons. The next unresolved
generic missing-report model in managed order is `vit_b_encoder.onnx` (Tier 3).

## Earlier checkpoint — `fb-refactor2` at `5b0a098`

Commit `5b0a098` recovers `encoder.onnx` by replacing dynamic rank-4
`GridSample` custom fallback with a TensorFlow-free builtin lowering.

- Runtime image N/C/H/W are read with `SHAPE`; no static spatial dimensions
  are fabricated.
- The image is transposed and flattened once. Runtime batch/spatial offsets and
  global indices gather only the required samples.
- Bilinear/linear and nearest interpolation are supported for zeros and border
  padding with both align-corners modes. Zeros padding uses per-neighbor masks,
  avoiding a dynamic padded-image allocation.
- NaN coordinates retain the existing ONNX Runtime-compatible `-1`
  normalization.
- The implementation is isolated in
  `op_builders/grid_sample_utils.py`; no new package was introduced.

Sequential `encoder.onnx` verification compared its only output with no skip:

- `evaluation_pass=true`;
- `max_abs=1.9293278455734253e-05`;
- `rmse=2.1648828175950605e-07`;
- cosine similarity `0.9999999999964916`;
- all 24 GridSample nodes use builtin operators; no unresolved custom op
  remains.

The expanded affected suite completed with `879 passed, 5 deselected,
2 warnings in 90.32s`. The three new dynamic numerical cases cover
bilinear/zeros/align-corners, bilinear/border/half-pixel, and
nearest/zeros/half-pixel. Static GridSample, validation, managed profile, and
the prior Mask R-CNN tests are included in the same suite. The five deselected
optional-environment tests and two expected float16 warnings are unchanged.

The managed Tier 0–4 profile now records 353 passes, 8
`missing_tflite_report`, 33 `tflite_fail`, and 26 excluded historical timeouts.
There are 41 active non-passes remaining. The next actionable missing-report
model in managed order is `tiny_decoder_11.onnx`; its four recorded shape hints
and `keep_shape_absolutely_input_names` options must be retained during
reproduction.

## Previous checkpoint — `fb-refactor2` at `c3c5ff7`

This section supersedes the older `fb-refactor` checkpoint retained below for
historical context. The active implementation branch is `fb-refactor2`, and
commit `c3c5ff7` recovers the managed
`maskrcnn_resnet50_fpn.onnx` conversion without adding dependencies or using
TensorFlow in the direct path.

### Completed since the older checkpoint

- Repaired the malformed TorchVision paste-masks Loop capture using a guarded
  semantic pattern. The repair reconstructs `expand_boxes` from the padded and
  source mask dimensions and is shared by direct lowering and ONNX Runtime
  evaluation. Ambiguous patterns remain untouched.
- Added zero-batch-safe floating-point Pad lowering. It temporarily supplies a
  safe batch to LiteRT Pad and restores the true runtime output shape, including
  batch zero, without changing non-empty results.
- Replaced RoiAlign's per-ROI full feature-map duplication with a single NHWC
  flatten and four masked neighbor gathers. This removes the Mask R-CNN path
  that attempted to allocate roughly 25 GB for a 612-ROI feature level.
- Preserved dynamic axes through output retargeting, singleton layout
  transpose-to-reshape conversion, late reshape recovery, ConvTranspose
  intermediates, and consecutive dynamic-batch layout reshapes.
- Merged explicit control-flow-body metadata into Loop lowering and recovered
  missing Gather ranks from ONNX semantics. In particular, rank-1 data gathered
  by a scalar index remains a logical scalar before Unsqueeze.
- Split the new control, Pad, and RoiAlign helpers into focused modules by
  responsibility. There is no source-line acceptance limit.
- Promoted the managed Tier 0–4 profile from 351 to 352 expected passes. The
  active profile now contains 352 passes, 9 `missing_tflite_report` records,
  33 `tflite_fail` records, and 26 excluded historical timeouts.

### Verification at this checkpoint

- Final sequential Mask R-CNN run:
  - classification: `pass`;
  - duration: `17.98s`;
  - compared outputs: 4 of 4;
  - skipped outputs: 0;
  - `evaluation_pass=true`;
  - `max_abs=0.0`.
- Main affected suite: `868 passed, 5 deselected, 2 warnings in 88.50s`.
  The five deselections are the previously documented optional TensorFlow and
  incompatible external Python 3.10 Torch environment tests. The warnings are
  the existing expected float16 overflow warnings.
- Additional focused run after repairing the dynamic reshape interaction:
  `6 passed, 751 deselected`.
- `git diff --check`, undefined-name checks, import checks for the new modules,
  Python compilation, and managed baseline JSON parsing all passed.
- Every inference run used the `uv` environment and one active process at a
  time. No ProcessPool or parallel pytest worker was used.

### Remaining work after `c3c5ff7`

- Continue improving the remaining 42 active Tier 0–4 non-passes (9 missing
  reports and 33 accuracy failures), one model at a time in tier order.
- Run the complete sequential Tier 0–4 corpus before the final audit; the
  focused Mask R-CNN run and affected suites do not replace that corpus gate.
- Complete the original artifact-matrix, optional TensorFlow boundary,
  PyTorch-family exporter, performance/RSS, public-contract, and
  requirement-by-requirement audits.
- Tier 5 remains intentionally excluded until the Tier 0–4 core contract is
  stable.

### First action on the next resume

1. Confirm `fb-refactor2` is clean and synchronized with
   `origin/fb-refactor2`.
2. Select the next `missing_tflite_report` entry from
   `docs/baselines/flatbuffer_direct_active_tier0_4.json`, preserving tier and
   model order.
3. Reproduce it with a one-model temporary regression profile,
   `ONNX2TF_EVAL_IN_PROCESS=1`, fixed seed, and inference concurrency one.
4. Fix only a general semantic boundary with a synthetic unit test, then rerun
   the model and the affected suite before promoting its baseline.

No pull request should be created. Future checkpoints end at commit and push to
`fb-refactor2`.

This is the checkpoint for pausing work on `fb-refactor`. The worktree was
clean at the start of this handoff, and all implementation changes described
below were already pushed to `origin/fb-refactor` as commit `5944292`.

## Completed work

- Recovered `superpoint_lightglue_end2end_fused_cpu.onnx` by reconciling static
  tensor ranks on demand after attention/control-flow boundaries. This is in
  commit `0dbba12`; the recorded maximum absolute error is
  `1.946091651916504e-05`.
- Recovered the fixed-shape `silero_vad.onnx` with
  `-kat input state sr`:
  - runtime-state forward LSTM is lowered to ordinary TFLite primitives instead
    of an invalid mutable builtin-LSTM state connection;
  - flattened inactive `If` branches keep speculative Squeeze operations
    executable;
  - rank-1 singleton conditions use `SELECT_V2` when prefix-style `SELECT`
    broadcasting is invalid;
  - sample-rate control inputs use the deterministic value `16000` during
    accuracy evaluation;
  - ONNX Runtime's nested-LSTM rank-inference failure has a narrowly scoped
    ONNX ReferenceEvaluator fallback with complete `Y`, `Y_h`, and `Y_c`
    outputs.
- Verified fixed-shape Silero through the normal isolated `-cotof` path:
  `evaluation_pass=true`, `max_abs=1.375097781419754e-06`,
  `rmse=1.222495367934982e-07`.
- Promoted the managed Tier 0–4 profile to 343 expected passes and 51 expected
  non-passes. There are 394 active models and 26 recorded timeouts excluded
  from future validation. Tier 5 remains excluded; Tier 4 remains in scope.
- Confirmed that `silero_vad (1).onnx` is not an accuracy-comparable dynamic
  variant: the serialized source references 14 nonexistent lexical captures,
  including all four LSTM weight/bias captures. Both ONNX Runtime and ONNX
  ReferenceEvaluator reject it. It remains active as a documented non-pass
  input-model defect rather than being promoted with fabricated weights.
- Committed and pushed the Silero/control-flow work as:
  `5944292 recover silero recurrent control flow`.

## Incomplete work

- Continue improving every non-timeout Tier 0–4 model. The managed checkpoint
  currently has 19 `missing_tflite_report` and 32 `tflite_fail` entries.
- Remaining Tier 0 candidates are:
  - `inverse_11.onnx`;
  - `string_normalizer_11.onnx`;
  - `silero_vad (1).onnx`, whose source-model defect is described above.
- Tier 1–4 non-pass models remain improvement candidates after Tier 0.
- The original plan still needs a final requirement-by-requirement audit,
  including the complete artifact matrix, optional TensorFlow exporter
  boundary, PyTorch-family exporters, full sequential Tier 0–4 regression,
  and conversion-time/peak-RSS measurements. Existing partial tests are not
  evidence that this final audit is complete.

## Current branch and changed files

- Branch: `fb-refactor`
- Remote: `origin/fb-refactor`
- Implementation checkpoint: `5944292`
- The worktree was clean before adding this handoff document.
- Files changed by `5944292`:
  - `docs/baselines/flatbuffer_direct_active_tier0_4.json`
  - `docs/flatbuffer_direct_architecture.md`
  - `onnx2tf/tflite_builder/accuracy_evaluator.py`
  - `onnx2tf/tflite_builder/lower_from_onnx2tf.py`
  - `onnx2tf/tflite_builder/op_builders/control.py`
  - `onnx2tf/tflite_builder/op_builders/elementwise.py`
  - `onnx2tf/tflite_builder/op_builders/recurrent.py`
  - `onnx2tf/tflite_builder/op_registry.py`
  - `onnx2tf/utils/onnx_reference_compat.py`
  - `tests/test_accuracy_evaluator_seeded_input.py`
  - `tests/test_flatbuffer_direct_bulk_runner.py`
  - `tests/test_onnx_reference_compat.py`
  - `tests/test_tflite_builder_direct.py`

## Tests already run

- Affected control-flow/LSTM/Squeeze/Where tests:
  `61 passed, 740 deselected, 1 warning`.
- ONNX reference compatibility and seeded evaluator tests:
  `30 passed`.
- Full relevant suite with the five unavailable optional-environment tests
  deselected:

  ```text
  uv run pytest -q \
    tests/test_tflite_builder_direct.py \
    tests/test_accuracy_evaluator_seeded_input.py \
    tests/test_onnx_reference_compat.py \
    tests/test_flatbuffer_direct_bulk_runner.py \
    -k 'not test_tflite_backend_matrix_add and not test_tflite_backend_matrix_hardswish_rewrite_on_off and not test_tf_converter_resize_cubic_avoids_flex_resize_bicubic and not test_tf_converter_resize_cubic_honors_cubic_coeff_a and not test_flatbuffer_direct_group_norm_alias_builtin_conversion'

  796 passed, 5 deselected, 2 warnings in 93.90s
  ```

- Fixed-shape Silero final verification:

  ```text
  uv run onnx2tf -i silero_vad.onnx \
    -o /tmp/silero_final_verify \
    -tb flatbuffer_direct -kat input state sr -cotof -v error
  ```

All inference checks were executed sequentially with one process active at a
time. No new package was introduced, and all commands used the `uv` environment.

## Failing tests and known issues

- Running the same full suite without deselection produces 796 passes and five
  environment-only failures:
  - four `tf_converter` tests require the optional TensorFlow/tf-keras extra,
    which is intentionally absent from the core environment;
  - one GroupNorm alias test imports an external Python 3.10 Torch binary from
    Python 3.12 and fails with `_PyCode_GetExtra`.
- Two existing float16 conversion tests emit an expected NumPy overflow warning
  while casting extreme values.
- `silero_vad (1).onnx` has the missing-capture defect described above.
- `inverse_11.onnx` is the next diagnosed non-pass. It contains `Resize` followed
  by a 224x224 `Inverse`. The direct builtin lowering intentionally supports
  matrices only up to 16x16, so conversion falls back to the unresolved custom
  op `ONNX_INVERSE` and LiteRT cannot allocate it. The evaluation compatibility
  layer maps the legacy empty-domain `Inverse` to `com.microsoft::Inverse`.
  ONNX Runtime produces finite but extremely large results (observed absolute
  values around `8.3e7`) because the resized matrices are nearly singular.
  A low-order approximate inverse is therefore unlikely to satisfy the `1e-1`
  accuracy requirement. No code was changed during this diagnosis.

## First work on resume

1. Confirm `git status --short --branch` is clean and still on `fb-refactor`.
2. Reproduce the `inverse_11.onnx` explicit evaluator failure:

   ```text
   ONNX2TF_EVAL_IN_PROCESS=1 uv run onnx2tf \
     -i inverse_11.onnx -o /tmp/inverse_explicit \
     -tb flatbuffer_direct --eval_with_onnx --eval_num_samples 1 -v error
   ```

3. Before implementing anything, determine whether an exact/stable 224x224
   inverse can be expressed with the existing TFLite primitive set and current
   dependencies while meeting `max_abs <= 1e-1` for the nearly singular
   reference. Do not introduce a silent approximation or a new dependency.
4. If no accuracy-preserving lowering is viable, record a precise normalized
   unsupported-capability reason, keep the model active as a non-pass, and move
   to `string_normalizer_11.onnx`. If a viable lowering exists, add a small
   well-conditioned numeric unit test first, then a singular/ill-conditioned
   guard, and finally rerun the root model sequentially.

The persistent project goal is paused at this checkpoint; it is not complete.
