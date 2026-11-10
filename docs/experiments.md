# 分析・追加実験の進め方

実験の入口は `scripts/`、再利用する処理は `src/experiments/` に置いています。
スクリプトをimportしても学習・ダウンロード・集計は始まりません。
追加・変更した分析コードのコメントと説明は日本語です。
モデル本体と既存の公開Hydra設定は引き続き利用します。

## 最初にコマンドを確認する

```bash
uv run python scripts/run_toys_pipeline.py \
  --config configs/local/toys.json \
  --output-dir artifacts/toys_trial_01 --dry-run
```

`--dry-run` はファイルを書き込まず、4ステージの実行コマンドを表示します。
未生成の埋め込みがある場合、R-KMeansの `max_steps=-1` と
`max_epochs` で終了条件を表します。実行時は埋め込み件数を確認し、
`ceil(件数 / バッチサイズ) × エポック数` を更新上限に設定します。
表示したコマンドを個別に実行して、各ステージだけを調査することもできます。

## 条件を変えて実験する

`configs/local/toys.json` をコピーして編集し、実験ごとに出力先を分けます。
`embedding_model` と `embedding_dim` は組み合わせて変更してください。
`data_dir` を変えれば他のカテゴリにも適用できます。件数は埋め込みから取得します。

```bash
uv run python scripts/run_toys_pipeline.py \
  --config configs/local/toys.json \
  --output-dir artifacts/toys_lr_0003 \
  --set 04_tiger:optim.optimizer.lr=0.0003
```

繰り返し使う上書きはJSONの `overrides` に保存します。

```json
{
  "04_tiger": ["optim.optimizer.lr=0.0003", "trainer.accumulate_grad_batches=8"]
}
```

JSONの `seed` がnullなら各公開設定のシードを維持します。
整数なら全ステージに同じ値を渡します。0も有効です。
形状やステージ間の受け渡しを変える値は、JSONの対応する項目を編集します。
`--set` は学習率などの詳細調整用です。

## デバッガーでTIGERを追う

```bash
uv run python -m pdb -m src.train \
  experiment=tiger_train_flat +debug=tiger \
  data_dir=data/amazon_data/toys \
  semantic_id_path=artifacts/toys_original_config/02_semantic_ids/pickle/merged_predictions_tensor.pt \
  num_hierarchies=4 \
  hydra.run.dir=artifacts/tiger_debug_01
```

`+debug=tiger` はTIGER用の短時間確認設定です。DDP・再試行・データローダーの
別プロセスを外し、学習・検証・テストをそれぞれ最大2バッチに制限します。
学習の拡張上限は8、勾配蓄積は1です。異常勾配の検出も有効になります。
この結果は本実験の精度比較には使いません。

IDEでも同じ引数で `src.train` モジュールを実行できます。
`src/train.py` の `trainer.fit` 前後、データのcollate関数、
モデルの `training_step` にブレークポイントを置くと入力から損失まで追跡できます。

## Pythonから処理を呼ぶ

```python
from dataclasses import replace
from src.experiments.config import ExperimentConfig
from src.experiments.pipeline import ExperimentRunner, build_stages, build_command

config = replace(ExperimentConfig(), output_dir="artifacts/toys_python_01", seed=42)
stages = build_stages(config)
command = build_command(config, stages[0])  # 実行せずコマンドを確認する
# ExperimentRunner(config).run()  # 4ステージを実行する
```

学習そのものを同じPythonプロセスで呼ぶ場合はHydra設定を合成します。

```python
from hydra import compose, initialize_config_dir
from src.experiments.common import ROOT
from src.train import train  # 独自の設定リゾルバーも登録する

with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
    cfg = compose(config_name="train", overrides=[
        "experiment=tiger_train_flat", "+debug=tiger",
        "data_dir=data/amazon_data/toys",
        "semantic_id_path=artifacts/toys_original_config/02_semantic_ids/pickle/merged_predictions_tensor.pt",
        "num_hierarchies=4",
        "paths.output_dir=artifacts/tiger_python_01",
        "paths.profile_dir=artifacts/tiger_python_01/profile_output",
        f"paths.work_dir={ROOT}",
    ])
metrics, objects = train(cfg)
model = objects["model"]
trainer = objects["trainer"]
```

Pythonからの `train(cfg)` は再試行ランチャーを経由しません。
CLIのHydra実行環境がないため、上記のように出力先とプロファイル先を指定します。
戻り値には指標のコピーとモデル・DataModule・Trainer・コールバック・ロガーが含まれます。
`inference(cfg)` も調査用のオブジェクト辞書を返し、予測は設定済みのwriterに保存します。

## 中間結果と失敗箇所を確認する

出力先には次のファイルを保存します。

| ファイル | 用途 |
|---|---|
| `experiment.json` | 今回の実験条件 |
| `status.json` | 実行中・完了・失敗のステージと例外 |
| 各ステージの `command.json` | 実際に実行したコマンド |
| 各ステージの `console.log` | 直近の標準出力と例外の詳細 |
| `_attempts/<ステージ>.<時刻>/` | 失敗した試行のログと途中成果物 |
| 各ステージの `.hydra/` | Hydraが保存する設定 |
| `embedding_checks.json` | 埋め込みの形状・型・有限値の確認 |
| `semantic_id_checks.json` | SIDの形状と一意性の確認 |
| `04_tiger/csv/version_*/metrics.csv` | 学習・検証・テストの指標 |

同じコマンドを再実行すると、条件と成果物が一致する完了ステージを再利用します。
失敗ステージは途中成果物を `_attempts/` に退避して最初から再実行します。
途中のモデル重みからの再開とは異なります。
設定を変えた場合や旧形式の成果物がある場合は、新しい `--output-dir` が必要です。
入力データやソースコードを同じパスで変更した場合も、出力先を分けてください。

## ログから結果を集計する

```bash
uv run python scripts/summarize_toys_run.py --run-dir artifacts/toys_lr_0003
```

`summary.json` と日本語の `results.md` を生成します。
未完了の実験も集計できます。欠測指標は0に置き換えません。
複数のCSVバージョンがある場合は最新の数値版だけを使います。
過去版を確認するときは `--version version_0` を指定します。
元設定の単独TIGER実行ディレクトリも指定できます。

## 元設定での学習とデータ確認

従来の起動方法も利用できます。

```bash
uv run python scripts/start_toys_training.py --dry-run
uv run python scripts/start_toys_training.py --foreground
uv run python scripts/inspect_amazon_data.py --data-dir data/amazon_data
uv run python scripts/download_embedding_model.py --model google/flan-t5-xl
```

元設定の学習は無限反復ローダーを利用します。
`max_epochs` はデータの周回数を表しません。上限更新数や早期終了条件も確認してください。
`--foreground` を省略すると従来どおり監視プロセスを切り離します。
学習率などは `--override optim.optimizer.lr=0.0003` で変更できます。

## コードの配置と検証

| モジュール | 責務 |
|---|---|
| `config.py` | 実験条件と入力値の検査 |
| `pipeline.py` | ステージ定義、コマンド生成、実行・再開 |
| `validation.py` | 埋め込み・SIDの検査 |
| `reporting.py` | CSVの読み取り、指標抽出、レポート作成 |
| `inspection.py` | データセット件数の確認 |
| `background.py` | 元設定の学習開始と監視 |
| `common.py` | パス、JSON、子プロセス環境 |

```bash
uv run python -m pytest -q tests
uv run ruff check src/experiments scripts src/train.py src/inference.py tests
```
