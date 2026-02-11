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
1. `[ ] Step A1 完了`
2. `[ ] Step A2 完了`
3. `[ ] Step A3 完了`
4. `[ ] Step A4 完了`
5. `[ ] Step A5 完了`
6. `[ ] Step A6 完了`
7. `[ ] Step A7 完了`
8. `[ ] Step A8 完了`
