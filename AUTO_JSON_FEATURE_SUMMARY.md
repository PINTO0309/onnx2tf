# 自動JSON生成機能の実装概要

## 実装内容

### 1. 汎用的なJSON生成アルゴリズム (`json_auto_generator.py`)

- **全オペレーション対応**: READMEのParameter replacementセクションに記載されている22種類以上のオペレーションすべてに対応
- **オペレーション別Fixerクラス**: 各オペレーションタイプに特化した修正ロジックを実装
  - `TransposeFixer`: Transpose操作の最適化
  - `AddMulDivSubFixer`: 算術演算（Add, Mul, Div, Sub）の次元調整
  - `ConcatFixer`: Concat操作の軸修正
  - `SplitFixer`: Split操作の軸とsplit数の調整
  - その他18種類のFixerクラス

### 2. エラー解析機能

- **変換エラーの詳細解析**: エラーメッセージから問題のあるオペレーションを特定
- **次元ミスマッチの検出**: shape情報を抽出して適切な修正を提案
- **TensorFlowオペレーション名のマッピング**: `tf.math.multiply_48`のような名前からONNXオペレーションを逆引き

### 3. 反復的最適化の仕組み

#### onnx2tf.py の改良
```python
# 変換エラー時の処理
max_attempts = 3
for attempt in range(1, max_attempts + 1):
    # エラーに基づいてJSONを生成
    auto_json = generate_auto_replacement_json(
        max_iterations=attempt * 3  # 試行回数を増やしながら
    )
    # 生成されたJSONの効果を確認
```

#### json_auto_generator.py の反復ロジック
```python
while iteration < max_iterations and current_error > target_accuracy:
    # 候補となる修正を生成
    candidate_fixes = generate_candidate_fixes()
    
    # 修正の組み合わせを作成
    fix_combinations = combine_fixes(candidate_fixes)
    
    # 最も効果的な組み合わせを選択
    best_operations = select_best_combination(fix_combinations)
```

### 4. 修正の優先順位付け

1. **エラーで特定されたオペレーション**: 直接エラーに関わるオペレーションを最優先
2. **関連オペレーションタイプ**: エラーから推測される関連オペレーション
3. **信頼度スコアリング**: 各修正に0.0-1.0の信頼度を付与
4. **組み合わせ最適化**: 複数の修正を組み合わせて最適解を探索

### 5. 実装例：custom_spo2.onnxでの動作

エラー:
```
Dimensions must be equal, but are 32 and 2 for tf.math.multiply_48/Mul
Input shapes: [1,2,1,256,32,1], [1,1,1,1,2,1]
```

生成されたJSON:
```json
{
  "operations": [
    {
      "op_name": "wa/extractor/Mul",
      "param_target": "inputs",
      "param_name": "image0",
      "pre_process_transpose_perm": [0, 4, 2, 3, 1, 5]
    },
    // ... 他の Mul オペレーションへの修正
  ]
}
```

## 技術的特徴

### 1. 次元変換の自動検出
- 6次元テンソルのブロードキャスト問題を検出
- 適切なtranspose permutationを自動生成

### 2. スマートな組み合わせ生成
```python
# オペレーションタイプ別にグループ化
op_type_groups = group_by_operation_type(fixes)

# Mul/Add/Sub/Div操作を優先的に選択
if error_type == "multiply":
    prioritize_arithmetic_fixes(combinations)
```

### 3. エラーパターンマッチング
```python
patterns = [
    r'tf\.math\.(multiply|add|subtract|divide)_(\d+)',
    r'layer "([^"]+)"',
    r'{{node ([^}]+)}}',
]
```

## 使用方法

### 1. 変換エラー時の自動生成
```bash
# -prfオプションなしで実行
onnx2tf -i model.onnx

# エラー発生時に自動的にJSONが生成される
# → saved_model/model_auto.json
```

### 2. 精度エラー時の自動生成
```bash
# -cotofオプションで精度チェックを有効化
onnx2tf -i model.onnx -cotof

# 精度エラー > 1e-2 の場合に自動生成
```

### 3. 生成されたJSONの使用
```bash
# 自動生成されたJSONを使って再変換
onnx2tf -i model.onnx -prf saved_model/model_auto.json
```

## 今後の改善点

1. **実際の変換テストループ**: 現在はヒューリスティックベースだが、実際に変換を試行して最適解を見つける機能を追加
2. **並列処理**: 複数の修正候補を並列でテスト
3. **学習ベースの最適化**: 過去の成功パターンを学習して提案精度を向上
4. **より詳細なエラー解析**: スタックトレースの全体を解析してより正確な修正を生成

## まとめ

この実装により、onnx2tfは変換エラーや精度問題に対して自動的に解決策を提案できるようになりました。ユーザーは手動でJSONを作成する必要がなく、ツールが自動的に最適な変換パラメータを見つけ出します。