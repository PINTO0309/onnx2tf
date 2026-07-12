# flatbuffer_direct architecture

`flatbuffer_direct` keeps its public CLI and Python API in `onnx2tf.onnx2tf`.
New implementation work belongs behind the internal contracts in
`onnx2tf.tflite_builder.core`; those contracts must not be exported as public
API.

## Pipeline

The required stage order is:

1. normalize the public arguments into `ConversionRequest` and `ArtifactPlan`;
2. preprocess ONNX;
3. create one `ConversionSession`, `GraphIndex`, and `LayoutState`;
4. execute registered passes in `PassPhase` order;
5. lower operators into ModelIR;
6. validate ModelIR invariants;
7. invoke only exporters selected by `ArtifactPlan`;
8. adapt `ConversionResult` back to the legacy return dictionary.

A pass has a stable ID, phase, priority, maximum iteration count, and explicit
`changed` result. Repeating passes must use a graph fingerprint so a cycle
terminates deterministically. Risky rewrites use the transactional mode and
must leave the graph unchanged when invariant validation fails.

Operator additions keep capability validation and lowering in the same
op-family module. Do not add model-name checks. A model-specific failure should
be reduced to a semantic graph pattern with explicit guards and a focused ONNX
fixture generated with `onnx.helper`.

## Dependency boundaries

Default TFLite conversion and ONNX/TFLite accuracy checking must import neither
TensorFlow nor tf-keras. TensorFlow is allowed only after the user explicitly
requests a TensorFlow-family artifact. PyTorch remains an optional dependency
and is loaded only for a requested PyTorch-family artifact.

No new third-party dependency may be added for this refactor. Use `uv sync` for
the core profile, `uv sync --extra torch` for PyTorch exports, and
`uv sync --extra tensorflow` for optional TensorFlow exports.

PyTorch-family example-input construction and export metadata persistence live
in `pytorch_export_support.py`; shared export exceptions live in
`pytorch_export_errors.py`. These modules are bounded internal contracts and
must not import TensorFlow. Image trace data is resized with the existing
PyTorch dependency so TorchScript, Dynamo ONNX, and ExportedProgram generation
does not require the TensorFlow optional extra.

## Runtime-check memory boundary

The direct backend releases the legacy GraphSurgeon graph before ModelIR
lowering because no direct stage consumes it. After artifact export,
unreachable ModelIR/serialization clones are collected and allocator caches
are returned to the operating system when the platform exposes a safe
heap-trim operation. Heap trimming is optional and must never affect conversion
correctness.

Isolated ONNX/TFLite checking remains strictly sequential: the ONNX worker
finishes and is reaped before the TFLite worker starts. Large prepared ONNX
graphs are materialized once in a managed temporary file and opened directly
by ONNX Runtime; do not send serialized model bytes through a multiprocessing
pipe. Evaluation permits the standard LiteRT delegate only when every runtime
input dimension is statically positive. Dynamic-input models retain the
delegate-free interpreter path to preserve its crash-isolation behavior.

## Regression workflow

Create the local corpus inventory without copying model files into Git:

```bash
uv run python -m onnx2tf.utils.flatbuffer_direct_corpus_manifest \
  --root_dir . -o flatbuffer_direct_corpus.json
```

Run inference sequentially by node tier. For example, Tier 1 is:

```bash
uv run python -m onnx2tf.utils.flatbuffer_direct_bulk_runner \
  --root_dir . -o flatbuffer_direct_bulk_report_tier1 \
  --min_nodes 50 --max_nodes 199 --tflite_only --root_only
```

The number `2,000` in the validation policy refers to the number of ONNX graph
nodes (operations) in a model. It is not a source-file line limit. Source and
test modules may be split when that improves ownership or reviewability, but
there is no 2,000-line structural gate.

The active improvement and regression scope is every root model in Tier 0
through Tier 4, including models whose baseline classification is a conversion
error, accuracy failure, or missing report. Recorded timeout models are kept
for provenance but excluded from subsequent runs. Tier 5 is a historical
baseline only and must not be run as part of ongoing refactoring.
Do not introduce a process pool or parallel model runner. Every successful
baseline model must remain successful; known Tier 0-4 failures are improvement
targets and must retain a normalized signature until they improve. Accuracy
reports retain their existing stricter judgement and every comparable float
output must remain below a maximum absolute error of `1e-1`.

Use the managed Tier 0-4 profile for a full active regression:

```bash
uv run python -m onnx2tf.utils.flatbuffer_direct_bulk_runner \
  --root_dir . -o flatbuffer_direct_active_regression \
  --regression_profile docs/baselines/flatbuffer_direct_active_tier0_4.json \
  --tflite_only
```

The profile fixes root-only discovery, the 1–1,999 node range, all 420 managed
historical model records, and inference concurrency of one. Models classified
as `timeout` in the current managed baseline remain recorded for provenance but
are automatically excluded from subsequent runs. The active run therefore
contains 394 models: 368 expected passes and 26 expected non-passes, excluding
26 recorded timeouts. The active non-passes are 20 accuracy failures and 6
missing reports. Tier 5 models cannot be added because the profile loader
rejects tiers above 4 and node ranges above 1,999.

`silero_vad.onnx` is validated with `-kat input state sr` and has a recorded
float32 maximum absolute error of `1.3750978e-6`. The similarly named dynamic
`silero_vad (1).onnx` remains a non-pass input-model defect: it references 14
lexically captured tensors that do not exist in the serialized ONNX, including
all four LSTM weight/bias captures. It cannot provide an ONNX reference result
for an accuracy-preserving promotion.

`inverse_11.onnx` also remains an active non-pass, with the normalized reason
`unsupported_exact_inverse_matrix_size_224`. Stock TFLite has matrix-diagonal
and batch-matmul operators but no matrix solve/inverse operator. The model asks
for three 224x224 inverses after bilinear upsampling. With the fixed evaluation
seed, the matrices have condition numbers from approximately `8.0e9` to
`9.2e10`, and ONNX Runtime produces magnitudes up to approximately `8.3e7`.
Even a float64 NumPy inverse rounded to float32 differs from the ONNX Runtime
reference by `1.7e7` to `1.2e8`, so substituting a different approximate
algorithm cannot satisfy the required `1e-1` absolute-error limit. The existing
custom-op fallback is retained; no inaccurate builtin approximation is emitted.

`string_normalizer_11.onnx` remains an active non-pass with the normalized
reason `unsupported_stock_tflite_string_normalizer`. It requires runtime string
case conversion, locale-aware comparison, and stopword filtering. Stock TFLite
supports string tensors and hash tables but has no string normalization,
case-folding, tokenization, or filtering builtin, so the existing
`ONNX_STRINGNORMALIZER` custom fallback is the only semantics-preserving
artifact. The reference model additionally requests the `en_US` locale, which
is not installed in the core validation environment. No locale package or
string-processing dependency is added by this project.

### Recorded Tier 0 baseline

The recursive Tier 0 corpus at commit `c4f3b7a` contains 190 models. Its
managed result is `docs/baselines/flatbuffer_direct_tier0_c4f3b7a.json`:
137 passed, 10 conversion errors, 1 timeout, 6 accuracy failures, and 36
missing accuracy reports. The median per-model duration was 2.165 seconds.
Failures are fixed by normalized signature; local run artifacts remain outside Git.

The formal root-only Tier 0 gate at commit `32c3277` contains 120 readable
models; the planned 121st entry has no readable node count. The managed result
is `docs/baselines/flatbuffer_direct_tier0_root_32c3277.json`: 96 passed, 9
conversion errors, 4 accuracy failures, 11 missing reports, and no timeouts.
Its median duration was 2.281 seconds. All 120 classifications and normalized
signatures match the same models in the recursive baseline.

The root-only Tier 1 gate at commit `a1fd301` contains 86 models. The managed
result is `docs/baselines/flatbuffer_direct_tier1_root_a1fd301.json`: 57
passed, 5 conversion errors, 13 accuracy failures, 11 missing reports, and
no timeouts. Median and maximum durations were 4.046 and 79.296 seconds.

`ssd_mobilenet_v1_12-int8.onnx` remains an active Tier 1 non-pass with the
normalized reason `invalid_onnx_missing_loop_captures_186`. Conversion succeeds
when its dynamic NHWC input is fixed to `inputs:1,300,300,3` and preserved with
`-kat inputs`, but the source model itself fails both the ONNX checker and ONNX
Runtime validation. Its two `Loop` bodies reference 186 unique tensors that
are neither defined locally nor available from the parent graph. These include
the preprocessor loop increment and numerous postprocessor slice bounds,
thresholds, and other constants. Supplying only the evident `int32(1)` loop
increment exposes the next missing capture immediately. Reconstructing all
missing values heuristically would not provide a trustworthy ONNX reference,
so the generated TFLite artifact is not promoted without the required
accuracy comparison.

`efficientnet-lite4-11-int8.onnx` remains an active Tier 1 accuracy non-pass
with the normalized reason
`onnxruntime_u8s8_saturating_pair_accumulation`. Its input Transpose and
QuantizeLinear outputs match exactly, while the first QLinearConv is the first
divergent tensor. The model combines UINT8 activations with INT8 weights. On
the validation host, ONNX Runtime's CPU kernel performs saturating signed
16-bit pair accumulation for this U8S8 path, whereas the direct TFLite
compatibility lowering computes the mathematical integer convolution before
requantization. A minimal 520-channel reproduction using activation `255` and
weight `-127` yields mathematical accumulator output `129`, but ONNX Runtime
returns `212`, consistent with clipping each two-product sum to `-32768` before
the final accumulation. Emulating that host-specific SIMD behavior with a
large expanded graph would degrade portability and pipeline efficiency, so no
such approximation is emitted. The fixed-seed final output remains below the
required `1e-1` maximum absolute-error ceiling, but it stays a non-pass because
the stricter cosine-similarity gate is not satisfied.

`version-RFB-320-int8.onnx` exhibits the same host-runtime U8S8 limitation and
uses the same normalized reason. The input QuantizeLinear result matches
exactly. For the immediately following QLinearConv, the direct TFLite result
and ONNX ReferenceEvaluator differ at only one of 307,200 elements by one
quantum, while ONNX Runtime CPU and the reference differ at 97,431 elements by
as much as 32 quanta. The current fixed-seed final maximum absolute error is
`0.14972958900034428`, improved from `0.15208351612091064` but still above the
required `1e-1` ceiling. A full ReferenceEvaluator run is not a practical
fallback: ONNX lacks built-in implementations for this model's QLinearConcat
and opset-12 DequantizeLinear, and even with local diagnostic implementations
the sequential run remained inside the pure-NumPy QLinearConv loop after more
than 90 seconds. No host-SIMD emulation or slow evaluator substitution is added
to the production pipeline.

`text_recognition_CRNN_CN_2021nov_int8.onnx` now executes through both
bidirectional LSTMs. Its second LSTM is preceded by a runtime Reshape with the
ONNX target `[0, 0, -1]`. Stale metadata previously described the result as
`[1,1,1]`, although the upstream tensor is `[25,1,2,256]` and the LSTM weights
require an input width of 512. This produced a 6,400-to-256 element Reshape
failure after the LSTM. The centralized shape reconciliation pass now resolves
that contract to `[25,1,512]` from the source tensor and the LSTM input-weight
shape, then propagates the sequence length through the bidirectional output.
The normal sequential `-cotof` path consequently completes and emits its
reports. The model remains an active accuracy non-pass with normalized reason
`lstm_float_drift_crosses_quantization_boundary_before_qlinear_matmul`. The
second fused LSTM differs by at most `4.181463737040758e-06`; one value at
`[23,266]` then crosses the following QuantizeLinear boundary by one quantum.
That changes six QLinearMatMul outputs by one quantum. ONNX Runtime and direct
TFLite QLinearMatMul each match an explicit INT32 NumPy calculation exactly
when given their respective quantized input, proving that the matmul lowering
itself is not the source. The difference propagates to final maximum absolute
error `0.14842605590820312`; mean absolute error is
`1.3509051097190739e-5`, RMSE is `0.0011565761101850747`, and cosine similarity
is `0.9999999968466379`. A rounding bias would change valid quantization
semantics, so none is added.

`fcn-resnet50-12-int8.onnx` is the third Tier 1 model with the normalized
`onnxruntime_u8s8_saturating_pair_accumulation` reason. Its input quantization
and shape operations match, and the first difference appears at the first
UINT8-activation/INT8-weight QLinearConv. With a fixed 224x224 diagnostic
shape, ONNX Runtime CPU and ONNX ReferenceEvaluator differ at 46,743 of
802,816 elements by up to 36 quanta, while the direct TFLite tensor matches
the reference at all 802,816 elements. Under the unchanged managed dynamic
input conditions, the fixed-seed final maximum absolute error has improved
from `1.5115007311105728` to `0.5471203327178955`, but it remains above the
required ceiling and therefore stays an active non-pass.

The root-only Tier 2 gate at commit `ad1d508` contains 113 models. The managed
result is `docs/baselines/flatbuffer_direct_tier2_root_ad1d508.json`: 80
passed, 4 conversion errors, 3 timeouts, 6 accuracy failures, and 20 missing
reports. Median and maximum durations were 7.124 and 120.360 seconds.

`afhq_generator.v11.quant.onnx` now executes and produces a comparison report.
The layout planner records channel-last provenance through elementwise and
decomposed DynamicQuantizeLinear regions, and a bounded quantized-layout pass
removes stale ConvInteger NCHW-to-NHWC input bridges only when that provenance
proves the input is already NHWC. This eliminates the double transpose and
improves the fixed-seed maximum error from `2.7819780111312866` to
`0.22717905044555664`, while cosine similarity improves from
`0.7283386855698387` to `0.999042264918951`. DynamicQuantizeLinear now rounds
`x / scale` to nearest-even before adding the integer zero point, matching ONNX
Runtime even when adding a large zero point would erase a just-below-half
fraction in FLOAT32. The model's input quantization consequently matches
exactly, and the first remaining mismatch moves to a later quantizer after an
InstanceNormalization difference of at most `4.76837158203125e-07`. Sparse
one-quantum decisions are still amplified through the decoder; the final
maximum error is `0.21375656127929688` with cosine similarity
`0.999052579433886`. No model-specific normalization or GAN rewrite is applied.
The normalized reason is now
`instance_normalization_drift_amplified_by_dynamic_quantization_decoder`.

The DynamicQuantizeLinear builder now lives in the dedicated 391-line
`op_builders/dynamic_quantize.py` family module instead of the legacy combined
quantized builder. The move reduces `op_builders/quantized.py` from 3,235 to
2,850 lines without changing dispatch or its ModelIR contract. A normalized
fingerprint over every operator, tensor, constant buffer, option, shape,
signature, dtype, and quantization field is identical before and after the
move: `a83d642e4aa7903f9b34495fec2c1edb5ff8779ba6735bedde382578152657f5`
for 22 operators and 27 tensors. An architecture test preserves the op-family
ownership and prevents the builder from returning to the legacy file; source
line count is not an acceptance criterion.

QLinearMatMul and QGemm now share the dedicated 238-line
`op_builders/qlinear_fc.py` family module. This second mechanical extraction
reduces `op_builders/quantized.py` from 2,850 to 2,634 lines while preserving
the public builder imports and registry dispatch. Pre-extraction normalized
ModelIR fingerprints are fixed as executable tests:
`633d083445fcf765023a948c038c0956c7a0b7646b73bdac0bb65cf4c14173c8`
for QLinearMatMul and
`bf71085f2cc3a5981b209b6d5b02cc65ea55a41251465229a5ef1636a319f70f`
for QGemm; each contains 9 operators and 16 tensors. The CRNN corpus model
also retains its exact fixed-seed metrics through the new dispatch path.

QLinearAveragePool and QLinearGlobalAveragePool are isolated in
`op_builders/qlinear_pool.py` without changing their registry names or public
builder imports. Their normalized ModelIR fingerprints are locked by focused
tests at `0bb8b9064ae208810addbcebb27846b05873d817e947a5af212f3fd8ee4a6b7c`
and `1b066e8245cb45f79df76dbc052ecf7485f07d7910fb789cff38b47c298b7f19`,
respectively. The module is part of the TensorFlow-import boundary. This split
is an ownership improvement, not a source-line requirement.

QLinearAdd and QLinearMul are isolated in
`op_builders/qlinear_binary.py`. Their shared validation, quantization
metadata, builtin/float-path selection, and ONNX-compatible requantization now
have one op-family owner while continuing to use the common quantization
primitives. Pre-extraction ModelIR fingerprints are fixed at
`d2f0714a44b2dc376827b845269a217c1df894986f3957128994a2913d611c24`
for QLinearAdd and
`b4d9d1a39202474faf52ab43fbde4938fe892a0a38c5739a87b6da2d9b882b34`
for QLinearMul. Fingerprint normalization and serialization are shared by the
op-family tests instead of being copied into each test module.

`dynamics_rife_sim.onnx` remains an active non-pass with the normalized reason
`invalid_onnx_concat_spatial_mismatch_64_128`. The source passes the structural
ONNX checker but ONNX Runtime rejects it during shape inference at
`Concat_378`: six inputs declare 128x128 spatial dimensions while
`onnx::Concat_193` declares 64x64, and the node concatenates only along the
channel axis. The direct TFLite artifact preserves the same mismatch and
LiteRT rejects allocation at the corresponding CONCATENATION. Removing
intermediate value-info and clearing graph-output shapes does not make ONNX
Runtime accept the structurally inconsistent node. The converter does not
guess which branch should be resized because no valid ONNX reference exists
to prove the required `1e-1` accuracy ceiling.

`yolov3-12-int8.onnx` now produces and executes a direct TFLite artifact under
the fixed `input_1:1,3,416,416` and `image_shape:1,2` contract. Six tf2onnx
arange Loop bodies referenced one optimizer-pruned default `delta=1` capture;
the ONNX Runtime compatibility graph restores only that canonical capture.
The same canonical Loop pattern is lowered to a dynamic TFLite RANGE using
`limit = start + trip_count * delta`, preserving its scan output without
unrolling or a custom `ONNX_LOOP`. Dynamic rank-5 elementwise inputs also no
longer use static high-rank coalescing based on all-one placeholder shapes.
The previous missing report is therefore upgraded to an active accuracy
non-pass with normalized reason `u8s8_detector_strict_metric_mismatch`. Its
fixed-seed overall maximum absolute error is `0.09563881158828735`, below the
independent `1e-1` ceiling, with mean error `7.735504565813028e-5`, RMSE
`0.0009708107564596241`, and cosine similarity `0.9999678197697215`. It remains
a non-pass because the existing stricter per-output thresholds reject a
`0.0956388` box-output outlier and the low-energy score output has cosine
similarity `0.8350063294036085`.

`arcfaceresnet100-11-int8.onnx` also uses the normalized reason
`onnxruntime_u8s8_saturating_pair_accumulation`. Its preprocessing and the
first two QLinearConv boundaries are exact. The first divergence occurs at
`stage1_unit3_conv1_quant`, whose UINT8 activation input still matches before
the INT8-weight convolution. When this node is isolated with that exact input,
the direct TFLite output matches ONNX ReferenceEvaluator at all 200,704
elements, while ONNX Runtime CPU differs at 29 elements by up to two quanta.
Those small host-kernel differences accumulate through 103 QLinearConv nodes;
the current fixed-seed final maximum absolute error is
`0.3681950643658638`, improved from `0.8229759186506271` but still above the
required ceiling. As with the other U8S8 cases, the converter does not emulate
host-specific saturating SIMD pair accumulation.

`object_detection_nanodet_2022nov_int8.onnx` is promoted from an active
accuracy failure to a pass. Its first QLinearConv output shape was absent from
ONNX value-info and initially materialized as `[1,1,1,1]`. The following
quantized LeakyRelu and DequantizeLinear retained that placeholder while the
MaxPool lowering selected its padding strategy. For the actual `[1,24,208,208]`
input, ONNX `kernel=3`, `stride=2`, `pads=[1,1,1,1]` starts its windows one
element before the input, whereas TFLite SAME needs only one total padding
element and places it at the end. This shifted every pooling window. For a
quantized pass-through chain feeding a stride-greater-than-one MaxPool,
QLinearConv now derives static output geometry eagerly and pass-through ops
replace same-rank all-one placeholders from their source. The scope is kept at
that semantic planning boundary so unrelated quantized layout chains are not
prematurely materialized. MaxPool therefore emits explicit negative-infinity
padding followed by VALID pooling. All six
outputs pass sequential comparison; maximum absolute error improved from
`2.570713758468628` to `4.76837158203125e-7`.

`object_detection_yolox_2022nov_int8.onnx` is also promoted to a pass. A late
quantized layout rewrite left one two-input QLinearConcat branch in NCHW
`[1,32,160,160]` and the other in NHWC `[1,160,160,32]`, while the Concat still
used NCHW axis 1. The existing mixed-layout repair required three inputs so it
could elect a spatial majority and therefore skipped this unambiguous
two-input case. For a channel-axis Concat with a fully known NCHW output
contract, the repair now uses that output's spatial dimensions as the
canonical contract and inserts a local NHWC-to-NCHW adapter only for the
permuted branch. LiteRT allocation and inference then complete; maximum
absolute error improved from `3.234160177409649` to
`2.384185791015625e-7`, with all metric gates passing.

`text_detection_en_ppocrv3_2023may_int8.onnx` remains active with the
normalized reason
`int8_requantization_outliers_amplified_by_transpose_conv`. Under its recorded
`x:1,3,480,640` input contract, the first INT8-activation/INT8-weight
QLinearConv differs from ONNX Runtime at only two of 614,400 elements by one
quantum. The final output differs at 16 of 307,200 pixels, concentrated in one
small spatial cluster, but the model's two final stride-2 ConvTranspose stages
amplify six of those pixels above `1e-1` and produce maximum error
`0.7411765307188034`. Mean absolute error remains
`9.280535896323273e-6` and RMSE `0.0022202128069130776`. An isolated first-Conv
comparison shows both LiteRT and ONNX Runtime one-quantum tie differences from
ONNX ReferenceEvaluator. Using LiteRT's native fixed-point quantized Conv for
all QLinearConv nodes reduced the maximum only to `0.7137255659326911` while
worsening mean error and cosine similarity, so the explicit ONNX-style
requantization remains the less degrading general implementation.

`yolov5s.onnx` remains active with the normalized reason
`float16_decode_rounding_boundary`. Its public input, intermediate graph, and
decoded `[1,25200,85]` output use FLOAT16 in ONNX, while the compatibility
float32 TFLite artifact intentionally exposes FLOAT32 inputs and outputs. The
confidence and class fields remain close, but the decoded coordinate fields
combine small convolution differences with large coordinate magnitudes. The
fixed-seed comparison has maximum absolute error `0.32965087890625`, mean
absolute error `0.0015324083874539186`, RMSE `0.012606690502504028`, and cosine
similarity `0.9999999759961475`. Adding only a FLOAT16 round-trip at the public
output reduced mean error to `6.284908615017916e-5` but quantized the remaining
outliers to an adjacent half-precision value, increasing maximum error to
`0.5`. Reproducing FLOAT16 storage after every node increased maximum error to
`1.5` because LiteRT and ONNX Runtime convolution accumulation differences
then crossed additional half-precision boundaries. Both experiments were
rejected: they cannot satisfy the independent `1e-1` ceiling and would impose
extra CAST operators on every FLOAT16 model.

`yolox_nano.onnx` remains an active Tier 2 non-pass with the normalized reason
`float_conv_accumulation_amplified_by_exp_stride`. The normal sequential
comparison satisfies the aggregate metric gate with mean absolute error
`1.3679266313265127e-5`, RMSE `0.0003010161768765817`, and cosine similarity
`0.9999999999699016`. However, ordinary FLOAT32 convolution accumulation-order
differences reach approximately `0.0021` in the regression head, then the
decoded size path amplifies them through Exp (approximately `0.0085`) and its
stride multiplication to a final maximum absolute error of
`0.1362457275390625`. This exceeds the project's independent `1e-1` ceiling,
so the model is not promoted despite the aggregate evaluator result.

`alike_l_opset11_192x320_post.onnx` remains active with the normalized reason
`topk_index_instability_from_float_ties`. Both previous-frame matching outputs
are exact. In the new-keypoint path, the TopK values differ by at most
`1.8298625946044922e-5`, but near-tied scores select or order different indices.
That turns into an index difference up to 59,908 and a decoded coordinate
difference up to `290.0000057220459`. Of the 5,000 returned keypoints, 4,955
coordinates are common as sets and many raw-order differences are adjacent
pair swaps; the associated descriptors move with those indices. Rounding the
scores before TopK would alter the source model's selection semantics, so the
converter does not add such a model-specific stabilization rule.

The related `tmp_alike_debug3.onnx` and `tmp_alike_debug4.onnx` probes now run
to completion after preserving the dynamic `[-1, 1]` output contract when a
late Squeeze/Unsqueeze fold leaves stale higher-rank input metadata. Before the
repair, LiteRT attempted to reshape 1,868 runtime elements to `[1, 1]`. The
serialized shape tensor is now `[-1, 1]`, independent of the stale input rank.
Their ordinary FLOAT32 outputs remain close: `scores_map` differs by at most
`2.4020671844482422e-5`, `descriptors` by `3.732740879058838e-6`, and `scores`
by `7.748603820800781e-7`; `keypoints` is exact. The debug-only exact-equality
and derived boolean masks can nevertheless flip from those accumulation-order
differences, producing boolean maximum error `1.0`. Both probes therefore
remain active non-passes with the normalized reason
`exact_equality_mask_instability_from_float_accumulation`. Introducing a
tolerance into ONNX `Equal` would change model semantics, so no such rewrite is
applied.

The root-only Tier 3 gate at commit `c838b42` contains 71 models. The managed
result is `docs/baselines/flatbuffer_direct_tier3_root_c838b42.json`: 22
passed, 15 conversion errors, 17 timeouts, 1 accuracy failure, and 16 missing
reports. Median and maximum durations were 17.248 and 120.662 seconds.

`new_encoder.onnx` is promoted from a missing report to a pass. Its static
rank-6 attention DIV was unresolved when op lowering ran, then became rank-6
only after final shape reconciliation; LiteRT aborted in `BroadcastDivSlow<5>`.
A bounded final ModelIR pass now coalesces fully-static broadcast-equivalent
axes to rank 4 while leaving negative/dynamic signatures unchanged. Evaluation
also derives deterministic multi-scale controls from the graph: Split sizes
`[8352,2088,522,135]` imply spatial shapes
`[[72,116],[36,58],[18,29],[9,15]]`, valid ratios are one, and float inputs
cast directly to boolean masks are zero. The fixed-seed one-sample result has
maximum error `0.0008774623274803162` and cosine similarity
`0.9999999999724198`. Its managed profile uses `eval_num_samples: 1` because
the ten-sample sequential run exceeds 180 seconds, and `accuracy_only: true`
selects the existing `--eval_with_onnx` path so the much heavier intermediate
operator report is not generated for this model. Inference remains strictly
single-process.

`encoder.onnx` is also promoted from a missing report to a pass. Unlike
`new_encoder.onnx`, its 24 GridSample image tensors retain runtime spatial
dimensions derived from `spatial_shapes`. The TensorFlow-free dynamic rank-4
GridSample lowering reads N/C/H/W with `SHAPE`, flattens the NHWC image once,
and builds runtime batch/spatial gather indices for bilinear or nearest
interpolation. Zeros and border padding and both align-corners modes are
supported without fabricating static dimensions or retaining an
`ONNX_GRIDSAMPLE` custom op. Sequential verification compares its only output
with no skip at `max_abs=1.9293278455734253e-05`, RMSE
`2.1648828175950605e-07`, and cosine similarity `0.9999999999964916`.

`fasterrcnn_resnet50_fpn.onnx` is promoted from a missing report to a pass.
`LoweringContext.ensure_tensor()` previously expanded a producer-confirmed
rank-2 tensor `[1,12543]` back to a stale shape hint `[-1,1,12543]`. Slice was
then lowered with rank-3 vectors, and a downstream TopK inserted a rank-3
transpose whose runtime input was rank 2. The context now indexes ModelIR
producers as operators are added and retains an all-static producer rank when
the higher-rank hint can only add singleton or dynamic axes. Known
non-singleton mismatches and tensors without a producer retain the previous
behavior. The resulting Slice is rank 2 (`[1,12543]` to `[1,9408]`) and no
TopK axis transpose is needed. A fixed-seed one-sample sequential run compares
all three outputs without skips, with maximum absolute error `0` and cosine
similarity `1`. Its managed profile uses `eval_num_samples: 1` and
`accuracy_only: true`; ten sequential samples exceed 180 seconds.

`fasterrcnn_test4_new.onnx` is also promoted from a missing report to a pass.
For its fixed-seed input the two ROI-level ScatterElements nodes operate on
empty `(0,256,7,7)` data, indices, and updates. Empty ScatterElements is valid
and must return an empty tensor, but LiteRT's reference SCATTER_ND kernel raises
`SIGFPE` when its shape contains zero. Dynamic ScatterElements lowering now
clamps only the internal SCATTER_ND shape vector to a minimum of one. The final
combination with the original data restores every zero-sized dimension; for
non-empty inputs the clamp is an identity. A local runtime test covers both a
non-empty value case and a zero-sized input, and the real one-sample managed
run compares all three outputs without skips, with maximum absolute error `0`
and cosine similarity `1`. Its profile uses `eval_num_samples: 1` and
`accuracy_only: true`.

`conv_tasnet.onnx` remains an active missing-report model with the normalized
reason `invalid_onnx_scatterelements_rank_mismatch_4_6`. Its terminal
`/decoder/ScatterElements` receives FLOAT data and updates shaped
`[256,2,2,10]`, but the constant indices are shaped
`[2,1,256,2,2,10]`. ONNX requires data, indices, and updates to have the same
rank, so ONNX Runtime rejects the original model at execution even though the
structural checker accepts it. The direct artifact retains the incompatible
scatter and LiteRT aborts during invocation. Squeezing or selecting either
leading indices dimension would choose unrecorded source semantics, and no
valid ONNX result exists to prove the `1e-1` accuracy ceiling, so the converter
does not fabricate a rank repair.

`dequantize_linear.onnx` remains an active non-pass with the normalized reason
`onnxruntime_qdq_fusion_and_float_conv_decode_amplification`. Its input
QuantizeLinear uses an equivalent signed internal representation and the
following DequantizeLinear is exact. The first and second float Conv boundaries
differ by only `1.1444091796875e-5` and `5.340576171875e-5`, respectively, so
the failure is not an input layout or per-channel scale-axis mismatch. Small
float Conv and requantization differences accumulate through 63 Conv stages and
are amplified by the terminal detector decode; the current managed-seed final
maximum absolute error is `81.25048828125`. ONNX Runtime's Basic graph retains
550 nodes including 63 Conv, 264 DequantizeLinear, and 110 QuantizeLinear,
whereas Extended optimization reduces it to 146 nodes by forming 63
QLinearConv, 16 QLinearConcat, and 6 QLinearAdd nodes. For the same manual input,
the direct artifact differs from the Extended graph by maximum `58.7506103515625`
and from the optimization-disabled graph by `22.74102783203125`; the two ONNX
Runtime paths differ by `59.61541748046875`. Emulating one host optimizer's
quantized fusion would therefore neither reproduce the unfused ONNX path nor
meet the `1e-1` ceiling, so the portable explicit-Q/DQ lowering is retained.

`rtdetrv4_s.onnx` now executes after wide-integer Sign lowering routes
INT64/UINT64 values through FLOAT32 SIGN and casts the `{-1,0,1}` result back to
the ONNX integer dtype. Casting through FLOAT32 is exact for sign semantics even
at integer extrema, unlike narrowing to INT32. The model moves from a missing
report to an active non-pass with normalized reason
`builtin_conv_accumulation_amplified_by_topk`. Under the evaluator's required
builtin-only LiteRT path, 278 of 300 labels differ after TopK and the score
maximum error is `0.37020038068294525`; the aggregate maximum is `79.0` from
integer labels. With the same artifact and inputs under LiteRT's default
XNNPACK delegate, all labels and zero-scaled boxes match exactly and score
maximum error is only `4.954636096954346e-6`. The evaluator remains
builtin-only for portability and dynamic-shape crash isolation; the converter
does not make XNNPACK a hidden accuracy dependency or alter TopK tie semantics.

`yolov9_n_wholebody15_Nx3HxW.onnx` and
`yolov9_t_wholebody28_Nx3HxW.onnx` are promoted from missing reports to passes
with the explicit input shape `images:1,3,640,640`. Their symbolic height and
width previously remained unresolved placeholders; no converter rewrite was
required. Sequential fixed-seed evaluation reports maximum absolute errors of
`0.0008544921875` and `0.0081634521484375`, respectively, with cosine
similarity above `0.9999999999995` for both models.

`yolo11x-obb.onnx` is likewise promoted from a missing report to a pass by
fixing its symbolic batch dimension with `input_image:1,3,1024,1024`. The
sequential fixed-seed comparison reports maximum absolute error
`6.103515625e-05`, RMSE `2.986316236369173e-06`, and cosine similarity
`0.9999999999999999`. No converter rewrite was required.

The root-only Tier 4 gate at commit `0a8ee88` contains 30 models. The managed
result is `docs/baselines/flatbuffer_direct_tier4_root_0a8ee88.json`: 12
passed, 7 conversion errors, 4 timeouts, 2 accuracy failures, and 5 missing
reports. Median and maximum durations were 28.100 and 121.826 seconds. All 12
passing models remained below the required `1e-1` maximum absolute error.

`campp_vin.onnx` is promoted to a pass after isolated accuracy evaluation adds
a sequential builtin-interpreter retry when a default delegate invocation
fails. XNNPACK cannot reshape this model's concretized dynamic-time graph, but
the same artifact runs with the builtin interpreter and matches its single
output at maximum absolute error `3.3020973205566406e-05`. A successful default
delegate remains the first choice, and an already-builtin failure is never
retried.

`bertsquad-12-int8.onnx` remains a managed accuracy failure because ONNX
Runtime's CPU U8×S8 MatMulInteger kernel does not match the exact ONNX integer
product. At the first encoder MatMulInteger, direct TFLite, an explicit INT32
NumPy product, and ONNX `ReferenceEvaluator` agree exactly, while ONNX Runtime
differs by as much as 11,772 regardless of graph optimization level. The
converter retains portable exact semantics rather than emulating this
host-specific saturation behavior. After correcting DynamicQuantizeLinear's
round-before-zero-point order, its fixed-seed final maximum error is
`2.001576066017151`; the failure classification and signature remain unchanged.

The root-only Tier 5 gate at commit `95aa61b` contains 34 models. The managed
result is `docs/baselines/flatbuffer_direct_tier5_root_95aa61b.json`: 6
passed, 4 conversion errors, 10 timeouts, 2 accuracy failures, and 12 missing
reports. Median and maximum durations were 51.621 and 123.486 seconds. All 6
passing models remained below the required `1e-1` maximum absolute error; the
largest passing error was `2.3543834686279297e-4`.

The Tier 4 result is part of the active improvement gate. The Tier 5 result is
retained only as historical evidence; do not rerun Tier 5 or use its failed
models as current improvement gates.

## Remaining refactoring order

1. Improve Tier 0-4 layout, transpose, broadcast, shape reconciliation, and
   fusion failures using semantic passes and focused ONNX fixtures.
2. Continue moving validation, capability selection, and lowering into
   op-family modules while preserving the current public API and artifacts.
3. Complete quantization, split/crop, custom/pseudo op, report, and requested-
   artifact-only regression coverage on the validated ModelIR contract.
4. Complete the shared PyTorch/TorchScript/Dynamo ONNX/ExportedProgram
   canonicalization and emitter separation.
5. Measure warm-run conversion time and peak RSS on the active Tier 0-4 set,
   then document improvements and remaining normalized failures.
