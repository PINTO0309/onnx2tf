# flatbuffer_direct 最適化移植計画（TF 経路資産の移植）

## 目的
`tf_converter` 経路で蓄積されている最適化・置換ロジック（擬似OP化、複合パターン吸収、属性正規化、shape補正）を、`flatbuffer_direct` 経路に段階的に移植し、Custom-op 依存を削減する。

## 背景
1. 現状の `flatbuffer_direct` は ONNX ノードを比較的ストレートに dispatch しており、TF 経路の前段最適化資産を十分に再利用できていない。
2. そのため、TF 経路では Builtin 化できるケースでも、direct では `unsupported_onnx_op` または custom-op candidate へ落ちやすい。
3. 目標は「Flex 非依存での高変換率」を direct 側で実現すること。

## 方針
1. ONNX グラフ前処理を direct 専用に新設せず、可能な限り共通プリパスとして実装する。
2. 変換成功率への寄与が大きい順（頻出モデル順）で段階移植する。
3. 置換は常に可逆性・診断可能性を担保し、`*_op_coverage_report.json` で可視化する。
4. 既存 `tf_converter` の挙動回帰を避けるため、backend 別テストを必須にする。

## 非目標
1. 一気に全 ONNX OP を direct Builtin 化すること。
2. 単一ステップで TF 経路と 100% 同等の内部グラフを再現すること。
3. 巨大モデルでの長時間統合テストを常時実行すること。

## 成果物
1. direct 向け前処理パイプライン（共通プリパス）
2. 置換・吸収ルール群（段階追加）
3. 変換診断レポート拡張（どのルールが適用されたか）
4. README と移行ガイドの更新

## マイルストーン
1. M1: 共通プリパス基盤
2. M2: 擬似OP置換の direct 移植
3. M3: 複合パターン吸収（GELU系など）
4. M4: 形状・属性補正の高度化
5. M5: custom-op 候補の縮小と回帰固定

## 作業ステップ

### Step A1: 現状差分の棚卸し（TF経路資産の可視化）
1. `onnx2tf/ops/*.py` 内の最適化・置換ロジックをカテゴリ化する。
2. direct 側で未移植の項目を `high / medium / low` 優先度で整理する。
3. 「移植可能」「設計変更が必要」「direct 非適合」に分類する。

対象ファイル:
1. `onnx2tf/ops/*.py`
2. `onnx2tf/onnx2tf.py`
3. `onnx2tf/tflite_builder/op_registry.py`
4. `onnx2tf/tflite_builder/lower_from_onnx2tf.py`

完了条件:
1. 差分マップ（ルール一覧）が文書化されている。
2. Step A2 以降の対象が優先度付きで確定している。

#### Step A1 実施結果（2026-02-11）
棚卸しサマリ:
1. 擬似OP置換（`replace_to_pseudo_operators`）関連: 14演算キーワード / 14ファイルで使用確認
2. 連鎖最適化トラッキング辞書: `gelu_replace_op_names` / `space_to_depth_replace_op_names` / `relu_relu6_merge_op_names` / `mul_div_replace_op_names`
3. グラフ前処理系: `onnxsim`、`fuse_expanded_qdq_to_qdq`、`shape_inference` 補正、`op_name_auto_generate`
4. TF実行時ヒューリスティクス: `disable_unnecessary_transpose`、`shape_unmatched_special_avoidance_workaround`、`correction_process_for_accuracy_errors`

差分マップ（優先度・移植可否分類）:

|ID|資産/ルール|主な参照元|優先度|分類|判断理由|
|:-|:-|:-|:-:|:-|:-|
|A1-01|擬似OP置換フラグ群（`-rtpo`）|`onnx2tf/onnx2tf.py:4937`|High|移植可能|direct 前処理ルールへ分離しやすい|
|A1-02|Unary擬似実装（Abs/Acos/Asin/Atan/Neg/HardSwish/LeakyRelu/PRelu/Pow/Erf）|`onnx2tf/ops/*.py`|High|移植可能|ONNXパターン正規化として実装可能|
|A1-03|`Relu + Clip -> Relu6` 連鎖吸収|`onnx2tf/ops/Relu.py:78`, `onnx2tf/ops/Clip.py:87`|High|移植可能|局所2ノード置換で適用容易|
|A1-04|GeLU連鎖吸収（Div->Erf->Add->Mul->Mul）|`onnx2tf/ops/Div.py:291`, `onnx2tf/ops/Erf.py:83`, `onnx2tf/ops/Mul.py:258`, `onnx2tf/ops/Add.py:101`|High|設計変更が必要|複数ノード探索 + スキップ制御の再設計が必要|
|A1-05|SpaceToDepth連鎖吸収（Reshape->Transpose->Reshape）|`onnx2tf/ops/Reshape.py:269`, `onnx2tf/ops/Transpose.py:63`|High|設計変更が必要|グラフ連鎖照合とshape検証のプリパス化が必要|
|A1-06|QDQ展開パターンの融合|`onnx2tf/onnx2tf.py:475`, `onnx2tf/onnx2tf.py:1732`|High|移植可能|既存関数を direct 前段へ共通適用可能|
|A1-07|GatherND擬似化（batch_dims=0制約）|`onnx2tf/ops/GatherND.py:91`, `onnx2tf/ops/GatherND.py:169`|Medium|設計変更が必要|direct IRでの分解規則設計が必要|
|A1-08|MatMulInteger擬似化（float演算経路）|`onnx2tf/ops/MatMulInteger.py:87`|Medium|設計変更が必要|量子化意味論とdtype整合の再定義が必要|
|A1-09|Mul/Div連鎖の定数前計算置換|`onnx2tf/ops/Mul.py:292`, `onnx2tf/ops/Div.py:344`|Medium|設計変更が必要|誤差管理付きの安全な畳み込み条件が必要|
|A1-10|Einsum/OneHot shape補正（ORT推論補助）|`onnx2tf/onnx2tf.py:2653`, `onnx2tf/onnx2tf.py:2678`|Medium|設計変更が必要|direct 前処理へ依存切り出しが必要|
|A1-11|onnxsim 3回最適化|`onnx2tf/onnx2tf.py:1616`, `onnx2tf/onnx2tf.py:1659`|Low|設計変更が必要|外部CLI依存で再現性・速度制御の設計が必要|
|A1-12|OP名自動生成（sng4onnx）|`onnx2tf/onnx2tf.py:1683`|Low|direct 非適合|変換成功率より運用利便性寄り|
|A1-13|TF実行時誤差補正/転置回避ヒューリスティクス|`onnx2tf/ops/Div.py:130`, `onnx2tf/ops/Div.py:233`, `onnx2tf/ops/Mul.py:134`, `onnx2tf/ops/Mul.py:245`|Low|direct 非適合|TFテンソル実行を前提とし、direct IRでは前提が異なる|

分類集計:
1. 移植可能: 4
2. 設計変更が必要: 7
3. direct 非適合: 2

Step A2 以降の優先着手対象（確定）:
1. High / 移植可能: `A1-01`, `A1-02`, `A1-03`, `A1-06`
2. High / 設計変更: `A1-04`, `A1-05`
3. Medium / 設計変更: `A1-07`, `A1-08`, `A1-09`, `A1-10`

### Step A2: direct 共通プリパス基盤の導入
1. ONNX Graph への前処理フックを direct 前段に追加する。
2. ルール適用履歴（rule_id、対象ノード数）を保持するデータ構造を導入する。
3. 既存 `lower_onnx_to_ir` 前にプリパスを実行する。

対象ファイル:
1. `onnx2tf/tflite_builder/preprocess/__init__.py`（新規）
2. `onnx2tf/tflite_builder/preprocess/pipeline.py`（新規）
3. `onnx2tf/tflite_builder/lower_from_onnx2tf.py`

完了条件:
1. ルール未登録でも no-op で従来通り動作する。
2. レポートにプリパス適用履歴を出力できる。

#### Step A2 実施結果（2026-02-11）
実装内容:
1. `onnx2tf/tflite_builder/preprocess/` を新設し、direct 前処理パイプラインの共通基盤を追加。
2. `export_tflite_model_flatbuffer_direct` で `lower_onnx_to_ir` 前に `run_preprocess_pipeline` を実行。
3. `report_op_coverage` 出力に `preprocess_report` を追加し、`rule_id` 単位の適用履歴を出力可能化。
4. ルール未登録時は `registered_rule_count=0` / `executed_rule_count=0` の no-op 動作を確認。
5. テスト: `tests/test_tflite_builder_preprocess.py`（新規）と `tests/test_tflite_builder_direct.py` の coverage report 検証を更新。

カバレッジ差分:
1. 前回値 `15.34%` -> 今回値 `15.34%`（差分 `+0.00%`）
2. 変化なし理由: Step A2 は前処理基盤と計測導線の追加のみで、Builtin 化ルール自体は未追加。

### Step A3: 擬似OP置換ルール（Wave1）移植
1. TF 経路で頻用の `replace_to_pseudo_operators` 相当を direct 前処理へ移植する。
2. 対象候補: `Erf`, `GeLU`, `HardSwish`, `LeakyRelu`, `PReLU`, `Power/Pow`, `MatMulInteger`。
3. 置換後は既存 registry の Builtin へ落ちる形に正規化する。

対象ファイル:
1. `onnx2tf/tflite_builder/preprocess/rules/pseudo_ops.py`（新規）
2. `onnx2tf/tflite_builder/op_registry.py`
3. `onnx2tf/tflite_builder/op_builders/*.py`

完了条件:
1. 置換前に失敗していた小規模ケースが Builtin 変換可能になる。
2. `report_op_coverage` で `unsupported_onnx_op` 件数が減少する。

#### Step A3 実施結果（2026-02-11）
実装内容:
1. `onnx2tf/tflite_builder/preprocess/rules/pseudo_ops.py` を新規追加し、`pseudo_ops_wave1` を実装。
2. 置換ルールを追加:
3. `HardSwish -> Add + Clip(0..6) + Div + Mul`
4. `LeakyRelu -> Neg + Relu + Relu + Mul + Sub`
5. `PRelu -> Neg + Relu + Relu + Mul + Sub`
6. `Gelu(approximate=none/tanh) -> Mul + Mul + Mul + Add + Mul + Tanh + Add + Mul + Mul`
7. `Pow`（指数が定数で `1.0` / `2.0` / `0.5`）を `Identity` / `Mul` / `Sqrt` へ正規化
8. `register_default_preprocess_rules()` を通じて、`flatbuffer_direct` 実行時に Wave1 ルールを自動登録
9. テストを追加・更新:
10. `tests/test_tflite_builder_preprocess.py` に置換ユニットテスト（HardSwish/Gelu/Pow）
11. `tests/test_tflite_builder_direct.py` の operator smoke に `HardSwish/LeakyRelu/PRelu/Gelu/Pow` を追加

未対応（次段階送り）:
1. `MatMulInteger` は dtype/zero-point の意味論維持が必要なため、Wave1 では自動置換対象から除外（Step A4/A5 で再設計）。

カバレッジ差分:
1. 前回値 `15.34%` -> 今回値 `15.34%`（差分 `+0.00%`）
2. 変化なし理由: Step A3 は前処理置換による成功率改善が主目的で、direct Builtin 種別の追加は行っていない。

### Step A4: 複合パターン吸収（Wave2）移植
1. GELU 系など複数ノード連鎖を単純表現へ吸収するパターンを移植する。
2. ReLU/ReLU6 merge、SpaceToDepth 周辺の連鎖最適化を段階移植する。
3. 置換時に shape/dtype 整合チェックを追加する。

対象ファイル:
1. `onnx2tf/tflite_builder/preprocess/rules/pattern_fusion.py`（新規）
2. `onnx2tf/tflite_builder/preprocess/pipeline.py`
3. `onnx2tf/tflite_builder/lower_from_onnx2tf.py`

完了条件:
1. 代表パターンで direct 変換成功率が改善する。
2. 不正置換時は reason_code 付きで明示失敗する。

#### Step A4 実施結果（2026-02-11）
実装内容:
1. `onnx2tf/tflite_builder/preprocess/rules/pattern_fusion.py` を新規追加し、`pattern_fusion_wave2` を実装。
2. 複合パターン融合を追加:
3. `Div -> Erf -> Add -> Mul -> Mul`（GELU連鎖）を `Gelu(approximate="none")` へ融合
4. `Relu -> Clip(min=0,max=6)` を単一 `Clip` へ融合（必要時は Relu ノード残置）
5. `Reshape -> Transpose(perm=[0,1,3,5,2,4]) -> Reshape` を `SpaceToDepth(blocksize)` へ融合
6. 融合結果の direct 受理のため `SpaceToDepth` Builtin dispatch を追加:
7. `onnx2tf/tflite_builder/op_registry.py` に `SpaceToDepth` バリデータ/dispatcher を追加
8. `onnx2tf/tflite_builder/op_builders/shape.py` に `build_space_to_depth_op` を追加
9. `onnx2tf/tflite_builder/model_writer.py` に `SpaceToDepthOptions` の FlatBuffer 出力を追加
10. デフォルト前処理順を `pattern_fusion_wave2 -> pseudo_ops_wave1` に変更

reason_code 付き明示失敗:
1. `pattern_fusion_invalid_rewrite reason_code=gelu_chain_fanout_conflict`
2. `pattern_fusion_invalid_rewrite reason_code=space_to_depth_*`（fanout/blocksize/channel/spatial不整合）

テスト:
1. `tests/test_tflite_builder_preprocess.py` に A4 ルール単体テストを追加
2. `tests/test_tflite_builder_direct.py` に `SpaceToDepth` と chain-fusion ケースを追加

カバレッジ差分:
1. 前回値 `15.34%` -> 今回値 `15.95%`（差分 `+0.61%`）
2. 変化理由: `SpaceToDepth` Builtin を direct registry へ追加（到達可能 Builtin 数 `25 -> 26`）。

### Step A5: 属性・shape 正規化の強化
1. axis/perm/pads などの属性を direct 受理形式へ事前正規化する。
2. 定数入力必須の箇所に対して constant-folding を限定導入する。
3. rank 制約違反を事前検知し、代替変換可否を分岐する。

対象ファイル:
1. `onnx2tf/tflite_builder/preprocess/rules/normalize_attrs.py`（新規）
2. `onnx2tf/tflite_builder/preprocess/rules/constant_fold.py`（新規）
3. `onnx2tf/tflite_builder/op_registry.py`

完了条件:
1. `requires_constant_input` / `unsupported_attribute_value` の失敗率が低下する。
2. 既存成功ケースに回帰がない。

#### Step A5 実施結果（2026-02-11）
実装内容:
1. `onnx2tf/tflite_builder/preprocess/rules/normalize_attrs.py` を新規追加し、属性/shape 正規化ルールを実装。
2. 正規化対象:
3. `Transpose` の `perm` 属性を定数入力へ正規化（input[1] 生成）
4. `Reduce*` / `Squeeze` / `Unsqueeze` の `axes` 属性を定数入力へ正規化
5. `Gather` / `Softmax` / `LpNormalization` の負axisを rank 基準で正規化
6. `Conv` / `Pool` の `pads`（長さ2）を `[t,l,b,r]` 形式へ正規化
7. `Softmax(axis != last)` は `Transpose -> Softmax(last axis) -> Transpose` へ事前変換
8. `onnx2tf/tflite_builder/preprocess/rules/constant_fold.py` を新規追加し、限定 constant-folding を実装。
9. fold 対象: `Identity`, `Cast`, `Unsqueeze`, `Squeeze`, `Concat`, `Reshape`, `Transpose`, `Gather`, `Shape`, `Add/Sub/Mul/Div`, `Neg`
10. `onnx2tf/tflite_builder/preprocess/rules/__init__.py` の default 順を更新:
11. `pattern_fusion_wave2 -> pseudo_ops_wave1 -> constant_fold_a5 -> normalize_attrs_a5`
12. `onnx2tf/tflite_builder/op_registry.py` を更新し、`Transpose` を attr-perm/暗黙perm でも受理可能化（min_inputs 1..2）

テスト:
1. `tests/test_tflite_builder_preprocess.py` に normalize/constant-fold の単体テストを追加
2. `tests/test_tflite_builder_direct.py` の operator smoke に A5 対象モデルを追加
3. 既存の Step A3/A4 ケースを含む選択実行で回帰なしを確認

効果（失敗率観点）:
1. `requires_constant_input` を起こしやすい `Transpose`/`Reduce axes` 入力を前段で定数化
2. `unsupported_attribute_value` を起こしやすい `Softmax axis != last` を前段で変換

カバレッジ差分:
1. 前回値 `15.95%` -> 今回値 `15.95%`（差分 `+0.00%`）
2. 変化なし理由: Step A5 は主に失敗回避/正規化強化で、Builtin 種別の新規追加はなし。

### Step A6: custom-op candidate 縮小フェーズ
1. custom candidate 一覧を再評価し、Builtin 化済み OP を候補から外す。
2. `schema_policy_matrix` の `custom_candidate` を段階的に減らす。
3. allowlist 運用時の診断（未許可/未対応）を明確化する。

対象ファイル:
1. `onnx2tf/tflite_builder/op_registry.py`
2. `onnx2tf/tflite_builder/lower_from_onnx2tf.py`
3. `README.md`

完了条件:
1. 同一モデル群で custom-op 降格率が低下する。
2. policy matrix の差分を定量報告できる。

#### Step A6 実施結果（2026-02-11）
実装内容:
1. `onnx2tf/tflite_builder/op_registry.py` に `Einsum` の builtin dispatch（`FULLY_CONNECTED`）を追加し、`ij,jk->ik` 型の rank-2 matmul パターンを受理するバリデータを実装。
2. builtin バリデーション失敗時に custom candidate へフォールバックできるよう `resolve_node_dispatch` を更新し、既存 custom-op 運用との互換性を維持。
3. `onnx2tf/tflite_builder/lower_from_onnx2tf.py` の `custom_op_policy` に診断キーを追加:
4. `candidate_count_excluding_builtin_supported`
5. `candidate_ops_now_builtin_supported`
6. `allowlist_builtin_supported_ops`
7. `allowlist_custom_candidate_ops`
8. `allowlist_unknown_ops`
9. `tests/test_tflite_builder_direct.py` を更新し、Einsum の custom 経路と builtin 優先経路の両方を検証。
10. `README.md` の flatbuffer_direct 対応表を更新（`Einsum`/`SpaceToDepth` の builtin 記載、summary 数値更新）。

効果（custom-op 縮小観点）:
1. `Einsum` は条件を満たす場合に custom ではなく builtin へ優先 dispatch される。
2. `custom_op_policy.candidate_count_excluding_builtin_supported` により、「候補だが既に builtin 化済み」の残件を定量把握可能になった（現状 `17 -> 16`）。

カバレッジ差分:
1. 前回値 `15.95%` -> 今回値 `16.56%`（差分 `+0.61%`）
2. 変化理由: `Einsum` の builtin 対応により、到達可能 Builtin 数が `26 -> 27` に増加。

### Step A7: テスト・回帰固定
1. backend matrix テストに「置換あり/なし」を追加する。
2. 失敗 reason_code のスナップショットテストを追加する。
3. OP coverage report のキー互換を固定する。

対象ファイル:
1. `tests/test_tflite_builder_direct.py`
2. `tests/test_tflite_builder_op_coverage.py`（必要なら新規）
3. `tests/test_model_convert.py`

完了条件:
1. 主要テストが CI で安定通過する。
2. direct 変換成功率の改善が数値で確認できる。

### Step A8: 文書化・運用導線
1. README に「TF経路との差」「direct 前処理で吸収される範囲」を明記する。
2. `FLATBUFFER_DIRECT_MIGRATION_GUIDE.md` に段階移行手順を追記する。
3. 既知制約と回避オプションを整理する。

対象ファイル:
1. `README.md`
2. `FLATBUFFER_DIRECT_MIGRATION_GUIDE.md`
3. `update-builder.md`（進捗反映）

完了条件:
1. 利用者が「なぜ失敗したか」「何を有効化すべきか」を自己解決できる。
2. 追加された最適化ルールの適用範囲が追跡可能。

## 測定指標（KPI）
1. `unsupported_onnx_op` 件数
2. `custom_op_candidate_disabled` 件数
3. `custom_candidate` -> `builtin_supported` への移行数
4. direct 成功率（代表モデルセット）
5. 変換後推論可否（`Interpreter.allocate_tensors()` / `invoke()`）

## カバレッジ定義と運用ルール
1. カバレッジ母数は LiteRT `schema.fbs` の `BuiltinOperator` から `CUSTOM` と `STABLEHLO_*` を除外した集合を使う。
2. 2026-02-11 時点の基準値は以下とする。
3. 母数（`CUSTOM` と `STABLEHLO_*` 除外）: `163`
4. 現状の ONNX -> flatbuffer_direct で到達可能な TFLite Builtin 数: `27`
5. カバレッジ: `16.56%`
6. 今後は修正を加えるたびに、同一条件でカバレッジを再計測して本ファイルへ反映する。
7. 各 Step 完了時に「前回値 -> 今回値（差分）」を必ず追記する。
8. カバレッジが変化しない修正でも「変化なし（理由）」を記録する。
9. `CUSTOM`/`STABLEHLO_*` を再び母数へ含めないことを運用ルールとして固定する。

## リスクと対策
1. 過剰置換による誤変換
- 対策: ルール単位で feature flag を持ち、段階有効化する。
2. shape 推論不整合
- 対策: 置換後 shape 検証を必須化し、失敗時はロールバックする。
3. メンテナンス負荷増大
- 対策: ルールを小粒化し、適用履歴とテストをセットで管理する。

## 直近の推奨実装順
1. Step A1
2. Step A2
3. Step A3
4. Step A7（最小テスト先行）
5. Step A4/A5/A6 を反復

## 進捗トラッキング（テンプレ）
1. `[x] Step A1 完了`
2. `[x] Step A2 完了`
3. `[x] Step A3 完了`
4. `[x] Step A4 完了`
5. `[x] Step A5 完了`
6. `[x] Step A6 完了`
7. `[ ] Step A7 完了`
8. `[ ] Step A8 完了`
