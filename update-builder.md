# TFLite FlatBuffer Builder 直接生成: 作業分解

## 目的
ONNX -> TensorFlow -> TFLiteConverter の最終段を段階的に置き換え、TFLite FlatBuffer Builder を直接使って `.tflite` を生成できる経路を追加する。

## 方針
1. 既存経路 (`tf_converter`) をデフォルトで維持する。
2. 新経路 (`flatbuffer_direct`) をオプトインで追加する。
3. 最初は対応OPを限定し、未対応は明示的に失敗させるか既存経路へフォールバックする。
4. 互換性と検証を優先し、段階ごとにマージ可能な単位で進める。

## 運用ルール
1. コンテキストのコンパクションが発生した直後は、必ず `update-builder.md` を再読し、作業ルールと現在状況を把握してから再開する。
2. `flat-bXX` 固定のブランチ分割戦略は廃止し、`flatbuffer` を基準に必要に応じて自由に補助ブランチを作成してよい。
3. 作業ステップの状況に応じて `update-builder.md` を更新する。
4. 各ステップの完了後は `flatbuffer` ブランチ向けにプルリクエストを発行する。
5. プルリクエストの自動マージは許可する。
6. プルリクエストタイトルは `Step X:` で開始し、作業ステップが判別できる文言にする。
7. 巨大モデルを使う長時間テストは実行しない。
8. `main` ブランチは絶対に操作しない。変更は `flatbuffer` と補助ブランチでのみ行う。

## スコープ外（初期段階）
1. 全OP対応の一括実装
2. いきなりのINT8キャリブレーション完全互換
3. 既存の精度補正ロジックの全面置換

## マイルストーン
1. M1: バックエンド切替インターフェース導入（実装空でも可）
2. M2: `flatbuffer_direct` の最小生成器（Identity/Reshape/Addなど限定）
3. M3: Builtin中心の主要演算対応（Conv/Depthwise/Pool/FC/Activation）
4. M4: FP32/FP16 の実運用化
5. M5: 量子化対応の段階拡張

## 進捗状況（2026-02-07）
1. 完了:
- `update-builder.md` を `flat-b01` と `flatbuffer` の両ブランチに反映済み
- `flat-b01`: `a9d1451`
- `flatbuffer`: `608653c`
- Step 1 実装（`tflite_backend` 導入、CLI追加、`flatbuffer_direct` スタブ分岐）を実装完了

2. 検証済み:
- `python -m py_compile onnx2tf/onnx2tf.py onnx2tf/tflite_builder/__init__.py`
- `python -m onnx2tf.onnx2tf -h` に `--tflite_backend` が表示されること
- `tflite_backend='flatbuffer_direct'` で `NotImplementedError` が返ること
- `tflite_backend='tf_converter'` で従来どおり `Functional` が返ること

3. 未着手:
- Step 2 以降（Builder基盤、IR、主要OP対応、テスト拡張）

## 作業ステップ

### Step 0: 仕様固定と前提整理
1. 直接生成経路の最初の対応範囲を固定する（FP32/FP16、Builtinのみ、単一Subgraph）。
2. `flatbuffer_direct` で未対応の場合の動作を決める（`error` or `fallback`）。
3. 使用する schema バージョンを固定する（例: LiteRT tag `v2.1.2`）。

完了条件:
1. このファイルの方針に合意できる。
2. 対応/非対応の境界が明文化されている。

### Step 1: バックエンド切替インターフェース導入（第1段階）
1. `convert()` に `tflite_backend` 引数を追加する。
2. CLI に `--tflite_backend` (`tf_converter`, `flatbuffer_direct`) を追加する。
3. 既定値は `tf_converter` のままにする。
4. `flatbuffer_direct` 指定時は新モジュール入口を呼ぶ分岐を追加する。
5. この時点では `flatbuffer_direct` は「未実装エラー」でもよい。

対象ファイル:
1. `onnx2tf/onnx2tf.py`
2. `README.md`（オプション説明）

完了条件:
1. 既存引数を変えずに従来挙動が維持される。
2. `--tflite_backend flatbuffer_direct` で新経路へ入ることが確認できる。

### Step 2: Direct Builder 基盤モジュール作成
1. 新規モジュール `onnx2tf/tflite_builder/` を作る。
2. `schema_generated.py` のロードを共通化する。
3. `ModelT/SubGraphT/TensorT/OperatorT/BufferT` を生成する雛形を実装する。
4. `TFL3` 識別子で書き出す共通関数を作る。

対象ファイル:
1. `onnx2tf/tflite_builder/__init__.py`
2. `onnx2tf/tflite_builder/schema_loader.py`
3. `onnx2tf/tflite_builder/model_writer.py`

完了条件:
1. 最小ダミーモデルを `.tflite` として書き出せる。
2. `ai_edge_litert` の `Interpreter` でロードできる。

### Step 3: 中間表現（IR）最小版を導入
1. TF依存の `tf_layers_dict` と分離した `GraphIR` を定義する。
2. `TensorIR`, `OpIR`, `QuantParamIR` を定義する。
3. まずは `Input`, `Const`, `Add`, `Reshape`, `Identity` のみIR化する。
4. `flatbuffer_direct` は IR から `.tflite` を生成する。

対象ファイル:
1. `onnx2tf/tflite_builder/ir.py`
2. `onnx2tf/tflite_builder/lower_from_onnx2tf.py`（既存データ構造からの変換）

完了条件:
1. 小規模モデルで `flatbuffer_direct` 出力が可能。
2. 出力モデルの入出力shapeとdtypeが期待通り。

### Step 4: OperatorCode/Tensor/Buffer の正規構築
1. `OperatorCode` の重複排除テーブルを実装する。
2. 定数データを `Buffer` へ集約する。
3. Tensor index の整合性チェックを追加する。
4. `shape_signature`（動的次元 `-1`）を書き込む。

対象ファイル:
1. `onnx2tf/tflite_builder/opcodes.py`
2. `onnx2tf/tflite_builder/tensor_buffer_builder.py`

完了条件:
1. index不整合で壊れたtfliteが出ない。
2. 動的shapeを含む最小ケースでロード可能。

### Step 5: 主要Builtin OP対応（第1バッチ）
1. 対応順を固定する。
2. 第1バッチ: `ADD`, `MUL`, `SUB`, `DIV`, `RESHAPE`, `TRANSPOSE`, `CONCATENATION`, `LOGISTIC`, `SOFTMAX`
3. 各OPで必要な BuiltinOptions を埋める。
4. 未対応OPは理由つき例外を返す。

対象ファイル:
1. `onnx2tf/tflite_builder/op_builders/*.py`
2. `onnx2tf/tflite_builder/dispatcher.py`

完了条件:
1. 第1バッチ対応モデルが `flatbuffer_direct` で変換成功する。
2. 未対応は沈黙せずに明示失敗する。

### Step 6: Conv系対応（第2バッチ）
1. `CONV_2D`, `DEPTHWISE_CONV_2D`, `AVERAGE_POOL_2D`, `MAX_POOL_2D`, `FULLY_CONNECTED` を追加する。
2. Weight layout 変換を厳密化する。
3. Padding/Stride/Dilation のマッピング表を定義する。

対象ファイル:
1. `onnx2tf/tflite_builder/op_builders/conv.py`
2. `onnx2tf/tflite_builder/op_builders/pool.py`
3. `onnx2tf/tflite_builder/op_builders/fc.py`

完了条件:
1. CNN系代表モデルで `flatbuffer_direct` が動作する。
2. 形状不一致時に診断情報が出る。

### Step 7: I/O 名・SignatureDef・メタデータ整備
1. 既存の `rewrite_tflite_inout_opname` 後処理依存を減らす。
2. 生成時に ONNX I/O 名を直接反映する。
3. 必要なら SignatureDef を初期生成する。

対象ファイル:
1. `onnx2tf/tflite_builder/signature_builder.py`
2. `onnx2tf/utils/common_functions.py`（既存後処理の整理）

完了条件:
1. `-coion` なしでも入出力名が期待通りになる。
2. 既存後処理を使う場合も互換維持される。

### Step 8: テスト拡張
1. backend別テストを追加する（`tf_converter` と `flatbuffer_direct`）。
2. 変換可否だけでなく `Interpreter` 実行テストを追加する。
3. ONNX比較はまず最終出力一致（許容誤差あり）から始める。

対象ファイル:
1. `tests/test_model_convert.py`
2. `tests/test_tflite_builder_*.py`（新規）

完了条件:
1. CIで backend matrix が通る。
2. 変換成功だけでなく推論可能性も担保できる。

### Step 9: ドキュメントと移行ガイド
1. README に `--tflite_backend` と制約を明記する。
2. 対応OP一覧・既知制限・フォールバック方針を記載する。
3. バージョン固定された schema 運用方法を記載する。

対象ファイル:
1. `README.md`
2. `update-builder.md`（進捗追記）

完了条件:
1. 利用者が安全に `flatbuffer_direct` を試せる。
2. 問題切り分けに必要な情報がREADMEに揃っている。

## 受け入れ基準（全体）
1. 既定の `tf_converter` 経路に回帰がない。
2. `flatbuffer_direct` は対象モデルで `Interpreter` 実行可能。
3. 未対応ケースが明示的に失敗し、原因が追える。
4. schema はタグ固定で再現性を持つ。

## リスクと対策
1. schema 破壊的変更リスク:
対策: タグ固定 + `ONNX2TF_TFLITE_SCHEMA_TAG` で明示切替。
2. OP仕様差異による精度低下:
対策: OP単位のゴールデンテストと段階リリース。
3. 変換成功でも実行不可:
対策: 変換テストだけでなく `Interpreter.allocate_tensors()` を必須化。

## 進捗トラッキング（運用テンプレ）
1. `[x] Step 0 完了`
2. `[x] Step 1 完了`
3. `[ ] Step 2 完了`
4. `[ ] Step 3 完了`
5. `[ ] Step 4 完了`
6. `[ ] Step 5 完了`
7. `[ ] Step 6 完了`
8. `[ ] Step 7 完了`
9. `[ ] Step 8 完了`
10. `[ ] Step 9 完了`
