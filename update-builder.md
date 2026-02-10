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
9. 現在のフォルダ配下のファイル以外は操作しない。
10. 既存の `.onnx` ファイルを削除しない。
11. 作業開始時は、作業履歴チャット欄の「編集済みファイル」をまず一括で折りたたんだ状態で作業を開始する。

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

## 進捗状況（2026-02-10）
1. 完了:
- `update-builder.md` を `flat-b01` と `flatbuffer` の両ブランチに反映済み
- `flat-b01`: `a9d1451`
- `flatbuffer`: `608653c`
- Step 1 実装（`tflite_backend` 導入、CLI追加、`flatbuffer_direct` スタブ分岐）を実装完了
- Step 2 実装（`onnx2tf/tflite_builder/` 基盤作成、schema loader / model writer 実装）
- Step 3 実装（`GraphIR` 最小構成の導入、ONNX から IR への lowering 実装）
- Step 4 実装（OperatorCode 重複排除、Tensor/Buffer 構築、index整合チェック、`shape_signature` 出力）
- Step 5 実装（第1バッチ Builtin OP: `ADD` `SUB` `MUL` `DIV` `RESHAPE` `TRANSPOSE` `CONCATENATION` `LOGISTIC` `SOFTMAX`）
- Step 6 実装（第2バッチ Builtin OP: `CONV_2D` `DEPTHWISE_CONV_2D` `AVERAGE_POOL_2D` `MAX_POOL_2D` `FULLY_CONNECTED`）
- Step 7 実装（ONNX I/O 名を生成時に直接反映、SignatureDef を生成）
- Step 8 実装（`tests/test_tflite_builder_direct.py` 追加、backend matrix/Interpreter 実行テスト追加）
- Step 9 実装（README に backend 制約・対応OP・schemaタグ固定運用を追記）
- Step 10-12 実装（`flatbuffer_direct` に限定 dynamic range quantization を追加、`QuantParamIR`/quantization 書き込み対応、`-oiqt` は未対応のまま明示エラー）
- Step 13 実装（`-odrqt` 対象拡張: 定数入力を使う `ADD/SUB/MUL/DIV/CONCATENATION` に対して定数INT8化 + `DEQUANTIZE` 挿入を追加）
- Step 14 実装（`-odrqt` の `quant_type` 連携: kernel weight に `per-channel` / `per-tensor` を反映）
- Step 15 実装（`-oiqt` 最小対応: `*_integer_quant.tflite` と `*_full_integer_quant.tflite` を生成。`QUANTIZE/DEQUANTIZE` による限定I/O量子化を追加）
- Step 16 実装（dynamic range quantization 精度強化: per-channel 対象を定数演算へ拡張、しきい値制御を追加、percentile校正戦略を追加）
- Step 17 実装（`integer_quant_with_int16_act` / `full_integer_quant_with_int16_act` の direct builder 対応を追加）
- Step 18 実装（追加要件の仕様固定: 精度評価・1GB分割・全OP本格実装の境界条件とCLI方針を明文化）
- Step 19 実装（ONNX Runtime/TFLite 共通推論ラッパーと指標算出を追加し、`*_accuracy_report.json` を生成）
- Step 20 実装（評価用CLIオプション拡張、量子化モデル向け dequant/raw 比較モード、閾値超過時の失敗制御を追加）
- Step 21 実装（IR/Tensor/Bufferベースのサイズ見積りと依存関係を壊さない分割候補探索、1GB近傍収束ロジックを追加）
- Step 22 実装（`*_0001.tflite` 形式の分割出力、`*_split_manifest.json` 出力、各分割の `Interpreter.allocate_tensors()` 検証を追加）
- Step 23 実装（manifestに従う分割モデル逐次実行評価器を追加し、`*_split_accuracy_report.json` を出力。`unsplit_tflite` / `onnx` 比較と閾値失敗制御を追加）

2. 検証済み:
- `python -m py_compile onnx2tf/onnx2tf.py onnx2tf/tflite_builder/__init__.py`
- `python -m onnx2tf.onnx2tf -h` に `--tflite_backend` が表示されること
- `tflite_backend='flatbuffer_direct'` で `.tflite`（float32/float16）が生成されること
- `tflite_backend='tf_converter'` で従来どおり変換可能なこと
- `python -m py_compile onnx2tf/tflite_builder/*.py onnx2tf/tflite_builder/op_builders/*.py`
- 小規模 ONNX モデル（Add/Reshape/Conv/AveragePool/Gemm）で `flatbuffer_direct` 出力を生成し `Interpreter.allocate_tensors()` が通ること
- `pytest -q tests/test_tflite_builder_direct.py` が通過（18 passed）
- `-odrqt` 指定で `*_dynamic_range_quant.tflite` が生成され、Gemm小規模モデルで `Interpreter.allocate_tensors()` および `invoke()` が通ること
- `-odrqt` 指定で Add(constant) 小規模モデルも `Interpreter.invoke()` まで通ること
- `-odrqt` + `--quant_type per-channel/per-tensor` でFCモデルの量子化 scale 形状が切り替わること（テストで検証）
- `-oiqt` 指定で `*_integer_quant.tflite` と `*_full_integer_quant.tflite` が生成され、両方 `Interpreter.invoke()` まで通ること
- `-oiqt` 指定で `*_integer_quant_with_int16_act.tflite` と `*_full_integer_quant_with_int16_act.tflite` が生成され、両方 `Interpreter.invoke()` まで通ること
- `ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_METHOD=percentile` 指定時でも `-odrqt` 変換が通ること
- `ONNX2TF_FLATBUFFER_DIRECT_QUANT_MIN_ABS_MAX` による量子化しきい値制御で非量子化ケースが明示失敗すること
- `eval_with_onnx=True` 指定で `*_accuracy_report.json` が生成され、`max_abs/mean_abs/rmse/cosine_similarity` が出力されること
- `python -m onnx2tf.onnx2tf -h` に `--eval_with_onnx`, `--eval_num_samples`, `--eval_rtol`, `--eval_atol`, `--eval_fail_on_threshold`, `--eval_target_tflite`, `--eval_compare_mode` が表示されること
- `full_integer_quant` を評価対象にした `eval_compare_mode=dequant/raw` の双方で評価レポート出力できること
- `eval_fail_on_threshold=True` かつ閾値超過時に変換を失敗終了できること
- `auto_split_tflite_by_size=True` 指定で `*_split_plan.json` が生成され、推定サイズと分割候補点を出力できること
- `pytest -q tests/test_tflite_split_planner.py` が通過（4 passed）
- `auto_split_tflite_by_size=True` かつ split 必要時に `*_0001.tflite`, `*_0002.tflite`... および `*_split_manifest.json` が生成されること
- 分割出力された全 `*_nnnn.tflite` が `Interpreter.allocate_tensors()` を通過すること
- `eval_split_models=True` 指定で `*_split_accuracy_report.json` が生成され、分割モデルと参照（`unsplit_tflite`/`onnx`）の差分が定量化されること
- `eval_split_fail_on_threshold=True` または split評価器の `fail_on_threshold=True` 指定で閾値超過時に失敗終了できること

3. 未着手:
- 追加要件 Step 24-28（全OP本格実装）

## 拡張ステージ（M5-Stage1: Dynamic Range Quant 最小対応）
### Step 10: 拡張仕様固定（限定解禁）
1. `flatbuffer_direct` における量子化は段階解禁とし、まず `-odrqt` のみを対象にする。
2. 対応対象は weight-only INT8（dynamic range）とし、対象OPは `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED` に限定する。
3. `-oiqt` は引き続き未対応とし、明示例外で失敗させる。

完了条件:
1. README と実装の制約が一致している。
2. 非対応モードが暗黙フォールバックせず明示失敗する。

### Step 11: QuantParam IR / Tensor 書き込み拡張
1. `QuantParamIR` を導入し、scale/zero_point/min/max/quantized_dimension を保持できるようにする。
2. Tensor シリアライズ時に `QuantizationParametersT` を書き込む。
3. 既存 FP32/FP16 出力への回帰がないことを確認する。

完了条件:
1. INT8 weight tensor の quantization parameter が `.tflite` に反映される。
2. 既存テストが通過する。

### Step 12: Dynamic Range Quant 実装（第1段階）
1. `flatbuffer_direct` で `-odrqt` 指定時に dynamic range quantized `.tflite` を追加生成する。
2. 量子化は対象OPの weight tensor のみを対称INT8で実施する。
3. 対象weightが存在しない場合は明示例外を返す。

完了条件:
1. `*_dynamic_range_quant.tflite` が生成される。
2. 小規模モデルで `Interpreter.allocate_tensors()` が通る。

### Step 13: Dynamic Range Quant 対象拡張（定数演算）
1. `ADD`, `SUB`, `MUL`, `DIV`, `CONCATENATION` の定数入力を量子化対象に追加する。
2. 上記定数は INT8 化したうえで、演算直前に `DEQUANTIZE` を挿入して互換性を維持する。
3. 対応外のケースは引き続き明示失敗を維持する。

完了条件:
1. Add(constant) などの小規模モデルで `-odrqt` が成功し、推論可能である。
2. 既存 `-odrqt`（Conv/Depthwise/FC weight-only）が回帰しない。

### Step 14: Dynamic Range Quant の量子化モード切替
1. `flatbuffer_direct` に `quant_type` を連携し、`-odrqt` 時の kernel weight 量子化モードを切替可能にする。
2. `per-channel` は kernel weight に対して channel-wise scale を出力し、`per-tensor` は単一 scale を出力する。
3. 定数演算（Add/Sub/Mul/Div/Concat）側は互換性のため per-tensor 量子化のままとする。

完了条件:
1. `-odrqt --quant_type per-channel` と `per-tensor` の双方で変換・推論が可能。
2. FC系小規模モデルで scale 長が切り替わることをテストで確認できる。

### Step 15: Integer Quantization 最小対応
1. `flatbuffer_direct` で `-oiqt` 指定時に `*_integer_quant.tflite` と `*_full_integer_quant.tflite` を生成する。
2. `integer_quant` は既存 dynamic-range 相当の量子化済み重み経路を利用する。
3. `full_integer_quant` は量子化I/Oラッパー（`DEQUANTIZE`/`QUANTIZE`）を挿入して `input_quant_dtype` / `output_quant_dtype` を反映する。
4. `integer_quant_with_int16_act` 系はこの段階では対象外とする。

完了条件:
1. `-oiqt` 指定で integer/full-integer の2ファイルが生成される。
2. 小規模FCモデルで変換後に推論実行可能である。

### Step 16: Dynamic Range Quantization 精度強化
1. `per-channel` の適用対象を kernel weight 以外の定数演算入力にも拡張する。
2. 量子化しきい値制御（最小要素数、最小絶対値、scale floor）を導入する。
3. 校正戦略として `max` に加えて `percentile` を導入する。

完了条件:
1. percentile 校正モードで `-odrqt` が成功する。
2. しきい値設定で量子化対象を制御できる。

### Step 17: int16 Activation 系 direct builder 対応
1. `-oiqt` 時に `*_integer_quant_with_int16_act.tflite` を生成する。
2. `-oiqt` 時に `*_full_integer_quant_with_int16_act.tflite` を生成する。
3. 両ファイルが Interpreter 実行可能であることを確認する。

完了条件:
1. int16 activation 系2ファイルが生成される。
2. 小規模モデルで `Interpreter.invoke()` が通る。

## 追加要件ステージ（M6-M8）
### Step 18: 追加要件の仕様固定
1. ONNX と TFLite の精度比較指標（max_abs/mean_abs/RMSE/cosine）と判定閾値を固定する。
2. FlatBuffer 1GB 近傍分割の制約（上限、命名規則、manifest 形式）を固定する。
3. 全OP本格実装の対象範囲（対象opset、dtype、サブグラフ方針）を固定する。

完了条件:
1. 評価・分割・OP実装の境界条件が文書化されている。
2. CLI/README の設計方針が合意されている。

固定仕様（Step 18 確定値）:
1. 精度評価:
- 既定指標: `max_abs`, `mean_abs`, `rmse`, `cosine_similarity`
- 既定閾値（float系）: `max_abs<=1e-4`, `mean_abs<=1e-5`, `rmse<=1e-5`, `cosine>=0.9999`
- 既定閾値（quant系）: `max_abs<=5e-2`, `mean_abs<=1e-2`, `rmse<=2e-2`, `cosine>=0.98`
- 入力データ方針: `-cind` 指定を最優先、未指定時は固定seed疑似乱数（seed=0）で生成
- 既定サンプル数: 10
- レポート出力: `{model_name}_accuracy_report.json`

2. 1GB近傍分割:
- ハード上限: `1_073_741_824` bytes（1GiB）
- 既定目標サイズ: `1_060_000_000` bytes（安全余白込み）
- ファイル命名: `{output_file_name}_0001.tflite`, `{output_file_name}_0002.tflite`, ...
- manifest: `{output_file_name}_split_manifest.json`
- manifest 既定項目: `schema_version`, `base_model`, `target_max_bytes`, `partitions`, `edges`
- 分割単位: 依存関係を壊さない連続ノード範囲（既存ONNX分割ロジックを骨格流用）

3. 全OP本格実装:
- 対象opset: ONNX opset 13-18 を第1目標、以降は上位opsetを段階追随
- 対象dtype（第1目標）: `float32`, `float16`, `int8`, `uint8`, `int16`, `int32`, `int64`, `bool`
- サブグラフ方針: まず単一SubGraphで完成度を上げ、複数SubGraphは後続段階で対応
- Builtin優先、表現不能演算は Custom OP 方針へ明示分岐
- 未対応は沈黙せず理由付き例外 + 機械可読レポートに出力

4. CLI設計方針（追加予定）:
- 精度評価: `--eval_with_onnx`, `--eval_num_samples`, `--eval_rtol`, `--eval_atol`, `--eval_fail_on_threshold`, `--eval_target_tflite`, `--eval_compare_mode`
- 分割: `--auto_split_tflite_by_size`, `--tflite_split_max_bytes`, `--tflite_split_target_bytes`
- OPカバレッジ: `--report_op_coverage`

### Step 19: 精度評価基盤（最終出力比較）
1. ONNX Runtime と TFLite Interpreter の共通推論ラッパーを実装する。
2. 同一入力で両者出力を比較し、指標を算出する。
3. `*_accuracy_report.json` を生成する。

完了条件:
1. 小規模モデルで評価レポートを出力できる。
2. 指標計算の再現性が確保できる。

### Step 20: 精度評価運用化（CLI/CI）
1. 評価用 CLI オプションを追加する（サンプル数、rtol/atol、fail-on-threshold）。
2. 量子化モデル向けの比較モード（dequant比較）を追加する。
3. 閾値超過時の終了コード制御を追加する。

完了条件:
1. CI で評価ゲートとして使える。
2. 失敗時の診断情報が JSON とログに出る。

### Step 21: 1GB分割のサイズ見積り基盤
1. IR/Tensor/Buffer から FlatBuffer サイズ推定器を実装する。
2. 依存関係を壊さない分割候補点を探索する。
3. 1GB 近傍への収束ロジックを実装する。

完了条件:
1. 分割前に見積り結果を出力できる。
2. 不正な分割案を事前に排除できる。

### Step 22: 自動分割出力実装
1. `*_0001.tflite`, `*_0002.tflite` 形式で分割出力する。
2. パーティション間 I/O を `*_split_manifest.json` に保存する。
3. 各分割ファイルのロード可能性を検証する。

完了条件:
1. 1GB 超見込み時に自動で分割される。
2. 全分割ファイルで `Interpreter.allocate_tensors()` が通る。

### Step 23: 分割後精度評価
1. manifest に従って分割モデルを逐次実行する評価器を実装する。
2. 非分割版または ONNX と終端出力を比較する。
3. 分割起因差分をレポート化する。

完了条件:
1. 分割前後の精度差が定量化できる。
2. 許容閾値超過時に失敗として扱える。

### Step 24: 全OP本格実装の基盤整備
1. OPディスパッチを登録テーブル化し、検証フック（shape/dtype/attr）を共通化する。
2. 未対応理由の機械可読レポートを生成できるようにする。
3. ONNX op schema と対応状況の突合を自動化する。

完了条件:
1. 対応率を自動集計できる。
2. 未対応OPの優先順位付けが可能になる。

### Step 25: 全OP実装 Wave 1（高頻度OP）
1. 実モデル出現頻度が高い未対応OPから優先実装する。
2. OP単体変換テストと推論比較テストを追加する。
3. 互換性影響の大きい属性差分を先に吸収する。

完了条件:
1. 主要公開モデルで未対応率が大幅に低下する。
2. 回帰テストで既存対応OPの退行がない。

### Step 26: 全OP実装 Wave 2（複雑演算）
1. Reduce/Norm/Broadcast複合/Index 系を拡張する。
2. dtype 境界ケース（int/bool/mixed precision）を網羅する。
3. 量子化経路での互換性を合わせる。

完了条件:
1. 複雑演算を含む代表モデル群が変換可能になる。
2. 精度評価で許容範囲を満たす。

### Step 27: 全OP実装 Wave 3（難所）
1. ControlFlow/Sequence/特殊演算を段階対応する。
2. Builtin で表現不能な演算の Custom OP 方針を確立する。
3. 失敗時のフォールバック/診断パスを整備する。

完了条件:
1. 難所演算の変換可否が明示される。
2. Custom OP 方針で運用可能な形になる。

### Step 28: 収束・安定化
1. 対象opset範囲で未対応ゼロまたは方針化済みにする。
2. 分割・量子化・精度評価を含む統合回帰を確立する。
3. README と移行ガイドを最終更新する。

完了条件:
1. 追加要件3点（精度評価/1GB分割/全OP実装）が運用可能になる。
2. 利用者向けドキュメントが自己完結する。

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
3. `[x] Step 2 完了`
4. `[x] Step 3 完了`
5. `[x] Step 4 完了`
6. `[x] Step 5 完了`
7. `[x] Step 6 完了`
8. `[x] Step 7 完了`
9. `[x] Step 8 完了`
10. `[x] Step 9 完了`
11. `[x] Step 10 完了`
12. `[x] Step 11 完了`
13. `[x] Step 12 完了`
14. `[x] Step 13 完了`
15. `[x] Step 14 完了`
16. `[x] Step 15 完了`
17. `[x] Step 16 完了`
18. `[x] Step 17 完了`
19. `[x] Step 18 完了`
20. `[x] Step 19 完了`
21. `[x] Step 20 完了`
22. `[x] Step 21 完了`
23. `[x] Step 22 完了`
24. `[x] Step 23 完了`
25. `[ ] Step 24 完了`
26. `[ ] Step 25 完了`
27. `[ ] Step 26 完了`
28. `[ ] Step 27 完了`
29. `[ ] Step 28 完了`
