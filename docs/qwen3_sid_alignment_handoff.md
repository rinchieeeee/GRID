# Qwen3・Amazon Toys の SID–Text alignment 実験 引き継ぎ

2026-11-11 時点。作業場所は `/home/d_hayashi/github/GRID`。ブランチは
`experiment/qwen3-sid-text-alignment`（`feature/setting-for-uv` から作成、HEAD `2cf4b49`）。
実装と設定はローカルの未コミット差分にある。別セッションでも同じ作業ツリーを開き、最初に
`git status --short --branch` で確認する。`artifacts/` は Git の追跡対象外。

## Qwen2.5のbaseline実行

2026-11-11に、Qwen2.5-0.5Bの推薦タスクのみの学習・全件評価を完了した。設定は
[`qwen25_toys_baseline.json`](../configs/local/qwen25_toys_baseline.json)。
全494,707,456パラメータを更新し、micro batch32×勾配蓄積4で実効推薦batch128、
AdamW・学習率3e-4・warmup600 step・cosine decayを使う。
上限20,000 step、全validationのRecall@5を200 stepごとに測り、改善なし4評価で停止する。
推薦lossは回答4 tokenのCEを例内平均し、128例で平均する。seed42、beam幅10、prefix制約あり、評価batch16。

入口は `python scripts/run_qwen_sid.py train --config configs/local/qwen25_toys_baseline.json`。
再開は同じコマンドに `--resume` を付ける。停止後は `train` を `evaluate` に置き換え、
選択したbest checkpointで全validation・testを評価する。
保存先は `artifacts/qwen25_toys_alignment_ablation/without_alignment/`。
`train.log` に100 stepごとの推薦CE・学習率・メモリ、`metrics.jsonl` に評価stepごとの指標、
`best/` に選択checkpoint、`last/` に再開用状態を保存する。
推薦planはモデル非依存のため `artifacts/qwen3_toys_data/` に共有保存し、後続条件も同じ系列を使う。

4,200 stepで改善なし4評価に達して終了し、3,400 stepのcheckpointを選択した。
validation Recall@5は0.0404904183、test Recall@5は0.0286935916。各19,412件で評価した。
全指標・学習条件・停止情報は
[`summary.md`](../artifacts/qwen25_toys_alignment_ablation/without_alignment/summary.md) と
同ディレクトリの `summary.json`、`results.json` に保存した。

## Qwen2.5のalignmentあり学習・test比較（完了）

2026-11-11にSID–Text joint学習と全件評価を完了した。設定は
[`qwen25_toys_with_alignment.json`](../configs/local/qwen25_toys_with_alignment.json)。
baselineから保存先・condition・alignment_weight（0→1）のみ変更した。
3,200 stepで早期終了し、bestは2,400 step。test Recall@5は0.0308572017で、
baselineに対して相対7.54%改善した。Hit@1は低下した。
保存先は `artifacts/qwen25_toys_alignment_ablation/with_alignment/`。
初期値と共通推薦planのSHA256、モデルrevision等の一致は `condition_check.json` に保存。

TIGERを含む全test指標、学習量の違い、結果ファイルの場所は
[最終比較結果](qwen25_toys_test_results.md)を参照。
Qwen2.5は各条件を同じ早期終了設定で個別実行したため、実更新数は一致しない。
以下の「未実施」「未決定」はQwen3本学習に関する過去の状態であり、Qwen2.5は上記のとおり完了済み。

## 現在地と次に決めること

- データ準備、Qwen3の取得、実験コードの実装、RTX 3090での全重み更新の確認、速度計測まで完了した。
  Qwen2.5のbaseline・alignmentあり学習・結果比較は上記のとおり完了。Qwen3のalignmentあり本学習は未実施。
  共通 `recommendation_plan.json` は20,000更新をカバーする11 epoch分を生成済み。
- 直近の依頼は、既存token embedding固定の速度効果をLoRAあり／なしで比較することだった。
  この計測は完了。**本学習の更新方式は未決定**。現在のJSONは20,000 optimizer stepを上限とする初期設定。
- 当初の本学習方針は論文に合わせた全重み更新。現行のA/B実行入口は、既存tokenとSIDの
  embeddingを含む全重み更新で動く。LoRAの構成関数はあるが、学習・評価の入口への組込みは未実施。
  既存embedding固定の行分割実装は計測用で、同じ方式を本学習に選ぶ場合は実装が必要。
- 目的は、同じQwen3でSID–Text alignmentを加えると推薦精度が上がるかを測ること。
  A/Bで更新方式と実更新数を揃える。元TIGERは参考値として併記する。
  実験設計の背景と論文の参照箇所は [計画書](qwen3_sid_alignment_plan.md)。
- 分析コードの作業方針は [AGENTS.md](../AGENTS.md)。日本語の説明、設定ファイル、
  import可能な小さい処理、pdbで追える前景実行を維持する。バッチ選定専用の `benchmark.py` と
  一時計測スクリプトは役目を終えたため削除済み。評価指標、loss、学習履歴、事前確認コードは保持している。

## A/Bの学習条件と評価

| 項目 | A `without_alignment` | B `with_alignment` |
|---|---|---|
| 推薦 | SID履歴から次商品SIDを生成 | Aと同じ推薦例・順序・更新数 |
| 補助タスク | なし | メタデータ全文→SID、SID→メタデータ全文 |
| 1更新あたりの例数 | 推薦128件 | 推薦128件＋各方向128件 |
| loss | `CE_rec` | `CE_rec + 0.5 × (CE_text_to_sid + CE_sid_to_text)` |
| 初期値 | Qwen3-0.6B、seed 42 | Aと同じ初期パラメータ |

- 推薦バッチの末尾が128件未満でも使い、Bの補助例数も同じ件数に合わせる。補助例は更新ごとに
  推薦系列と独立した乱数で復元抽出する。別々のmicro batchに分けて勾配を足すjoint SFT。
- 各例のCEを回答token数で平均してからタスク内で平均する。promptとpaddingはmaskし、
  SID回答は4 token、テキスト回答は末尾EOSも教師にする。Qwen3のthinkingは無効。
  `lm_head` は教師のある位置だけに適用する。
- 既定設定はmicro batch上限16、勾配蓄積8、gradient checkpointing無効。micro batchごとの上限は
  padding込み3,072 tokenと回答1,024 token。FP32の重みとfused AdamW、BF16 autocast、
  学習率 `3e-4`、weight decay 0、cosine decay、warmup 3%、勾配norm上限1。
- 全カタログの有効なSIDを対象に、prefix制約あり、beam幅10、固定長4 tokenで生成する。
  勾配蓄積後の `optimizer.step()` を1 stepと数え、`eval_steps`（既定1,000）ごとと
  `max_steps`（既定20,000）で全validation 19,412件を評価する。Recall@5が厳密に改善したstepの
  checkpointを保存する。同値は改善なしに数える。学習率のスケジュールは `max_steps` を使う。
  単独実行では改善なしが連続 `early_stopping_patience` 回（既定10回）に達した時点で停止する。
  A/B実行では同じ評価stepで両条件がこの基準を満たした時点で同時に停止する。
  選択後に両条件のtest全19,412件でHit@1、Recall@5/10、NDCG@5/10を計算する。
  `comparison.py` は初期値、推薦plan、計画更新数、実更新数、推薦例の累積件数を照合して差分を出す。

## データとSID

- 元データは `data/amazon_data/toys/` のTFRecord。商品11,924件、training・evaluation・testingは
  それぞれ19,412履歴。元の分割を使い、評価入力が対応する履歴の直近20商品以内のsuffixであることを検査済み。
- 既存SIDは `artifacts/toys_original_config/02_semantic_ids/pickle/merged_predictions_tensor.pt`。
  形状は `[4, 11924]`、全商品で4要素の組が一意。先頭3階層は0〜255、第4階層は衝突解消用で観測値0〜12。
  SHA256は `756b4308b442685a46838b7858fee45f4f1104bd868f33af535438dd9410450a`。
- 元のSID生成はTFRecordの `text` 全文（Title、Brand、Categories、Price）をFLAN-T5-XL encoderに入力し、
  最大128 tokenで平均プーリングしたembeddingをR-KMeansで離散化した。元設定は
  `artifacts/toys_rkmeans_10epochs/01_embeddings/.hydra/config.yaml`。
  alignmentは同じ `text` フィールドをそのまま使う。テキストのユニーク数は11,898件。
  商品単位で11,328件をalignment学習、596件を保留にし、両方向で同じ分割を使う。
  カタログはtransductive。推薦の保留インタラクションは学習に使わない。
- QwenのSID tokenは `<sid_0_0>` 形式の4階層×256種。各tokenが1 IDになることを検査する。
  tokenizerは151,669から152,693 tokenに拡張。新規1,024行は既存の未使用padding行に割り当たる分も含め、
  seed 42で各1,024次元をFP32の `Normal(0, 0.02)` で初期化する。R-KMeansのcentroidはSIDの割当てに用い、
  Qwenの埋め込みにはこの乱数初期値を使う。入力embeddingと `lm_head` は同じ重みを共有する。
  学習済みtokenizerから再読込する際はSID行を再初期化しない。実装は [model.py](../src/experiments/qwen_sid/model.py)。
- 推薦例は元GRIDのcollateに合わせ、32履歴ごとに全連続部分列から最大512例を抽出する。
  上限超過時は `randint` の抽出と重複indexの除去を使う。各部分列の末尾30商品から、最大29商品を入力、
  最後の商品を正解とする。有限のepochで全履歴を候補生成に回すため、元実装で落ちる最後の小バッチも使う。
  サンプリング後の推薦例に全ユーザーが残る保証はない。epochごとに再shuffle・再抽出し、A/Bで同じplanを読む。
- 1 epoch目は推薦251,869例、1,968 optimizer更新。元TIGERのsource batch32×蓄積16で換算すると、
  19,412入力履歴は約38 TIGER step相当。これは処理した元履歴件数での換算。

## GPUで確認したこと

RTX 3090 24 GiBで全パラメータ更新のprobeに成功。596,825,088パラメータに勾配が入り、
既存語彙行、SID行、Transformerの更新を確認した。現行設定の3更新でピーク確保18.84 GiB。
`probe` は本学習に先立つ更新・メモリ確認用で、計測専用ファイルの削除対象には含めていない。

下表は最終の固定有無比較。2更新のウォームアップ後、1 epochの1,968更新から均等に選んだ8更新を測定し、
学習のみの時間を外挿した。validation・checkpoint I/Oは別途かかる。

| Micro batch上限 | LoRA | 既存embedding | A 秒/更新（推定/epoch） | B 秒/更新（推定/epoch） | B ピーク確保 |
|---:|---|---|---:|---:|---:|
| 16 | なし | 更新 | 2.459（1時間21分） | 6.103（3時間20分） | 19.46 GiB |
| 16 | なし | 固定 | 2.412（1時間19分） | 5.912（3時間14分） | 18.00 GiB |
| 16 | あり | 更新 | 2.572（1時間24分） | 6.324（3時間27分） | 14.51 GiB |
| 16 | あり | 固定 | 2.510（1時間22分） | 6.131（3時間21分） | 15.21 GiB |
| 32 | なし | 固定 | 未計測 | OOM | — |
| 32 | あり | 固定 | 未計測 | 5.871（3時間13分） | 22.64 GiB |

- 同じmicro batch 16での固定による時間短縮はAで約2%、Bで約3%。
  Bのピーク割当量はLoRAなしで18.46→16.73 GiB、ありで12.87→11.14 GiB。
  固定時は既存行とSID行を別パラメータにし、入力と出力の両方で共有した。
  既存行がビット単位で不変で、SID行とTransformer本体またはadapterが更新されることを確認した。
- LoRAはrank 16、alpha 32、dropout 0。attentionのq/k/v/oとMLPのgate/up/downに適用。
  既存embedding更新ありの先行計測では、Bの全重み更新・micro batch 16が6.040秒/更新、
  LoRA・32が5.887秒/更新でピーク確保23.01 GiB、LoRA・64はOOM。micro batch 32ではtoken上限も拡大した。
- 既存embedding更新ありのLoRA保存では `save_embedding_layers=True` が必要。小型モデルで保存・再読込を検証済み。
  adapterを再読込して追加学習する場合は共有embeddingの `requires_grad` も有効化する。
  固定モードを採用する場合は行の固定、重み共有、checkpointの保存・再開・生成まで確認する。
- 全重み更新・既存embedding更新では、A/Bを各1 epoch進める学習時間は合計約4時間41分。
  beam幅16のvalidationは、初期の256件サンプルからの外挿で各約24分。
- 最終比較の記録は [固定有無の計測結果](../artifacts/qwen3_toys_alignment_ablation/sid_only_embedding_check/results.md) と
  同ディレクトリの `results.json`。先行比較は
  [既存embedding更新ありの計測結果](../artifacts/qwen3_toys_alignment_ablation/lora_full_embedding_check/results.md)。

### step基準に変更した後の所要時間

RTX 3090で現行の全重み更新を各条件10回実行した。最初の2更新でウォームアップし、1 epochの
範囲から均等に選んだ8更新の平均を使った。1更新は推薦128例。Bには各方向128例の補助タスクも含む。
評価バッチ8、beam幅10で、全体から分散して選んだvalidation 1,024件を評価した。
Aは45.53秒、Bは45.90秒で、19,412件に比例外挿すると約14.4〜14.5分。

| 条件 | 1更新の実測（データ整形込み） | 最初の評価開始まで（1,000更新） | 全件validationの推定 | 評価終了までの合計 |
|---|---:|---:|---:|---:|
| A 推薦のみ | 2.50秒 | 約42分 | 約14.4分 | 約56分 |
| B alignmentあり | 6.13秒 | 約1時間42分 | 約14.5分 | 約1時間57分 |

A/Bランナーは1 GPUで順に実行するため、現在の設定で両条件の初回評価完了までの合計は約2時間53分。
共通plan作成、probe、モデル読込、
checkpoint保存は別途かかる。

同じ推薦例・補助例・評価例でQwen2.5-0.5Bも測定した。既定のmicro batch16ではAが1.61秒/更新、
Bが4.02秒/更新、全件validationの推定は約8.8分。micro batch32・勾配蓄積4にし、
padding token上限を6,144、回答token上限を2,048にするとAは1.50秒、Bは3.60秒になった。
この設定は最長実例での3更新probeも通過した。Qwen2.5の評価batchは8〜128を測り、16が最速で、
1,024件26.82秒、全件推定約8.5分だった。各モデル標準のchat templateを使用した。
モデルrevision、各更新の時間、メモリ、他のbatch設定は
[Qwen2.5との実測比較](../artifacts/qwen3_toys_alignment_ablation/qwen25_speed_check/results.md) に保存した。

## 元TIGERの参考値

保存先は `artifacts/toys_original_100epochs/20261110_000649/`。全4,600更新、選択checkpointは
`checkpoints/checkpoint_epoch=000_step=003600.ckpt`、約13.13Mパラメータ。
beam幅10、prefix制約なしのtest結果を `csv/version_0/metrics.csv` の末尾で確認できる。
validationは1,600学習バッチごと（蓄積16で100 optimizer stepごと）で、Recall@5を監視する
patienceは10評価。Qwen3の初期評価間隔は全件評価の所要時間を考慮して1,000 optimizer stepとした。

| Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---:|---:|---:|---:|
| 0.042396456 | 0.062332578 | 0.027543532 | 0.033975791 |

モデル構造、学習量、デコード条件がQwenと異なる。alignmentの効果は、共通条件のQwen A/Bの差で見る。

## 実装と再開手順

| ファイル | 主な処理 |
|---|---|
| [qwen3_toys.json](../configs/local/qwen3_toys.json) | パス、seed、学習・評価条件 |
| [data.py](../src/experiments/qwen_sid/data.py) | TFRecordとSIDの検証、共通推薦plan、タスク例と回答mask |
| [model.py](../src/experiments/qwen_sid/model.py) | SID追加・初期化、LoRA構成、回答位置のCE |
| [optimization.py](../src/experiments/qwen_sid/optimization.py) | 補助例の抽出、micro batch分割、勾配加算 |
| [probe.py](../src/experiments/qwen_sid/probe.py) | 実データの長い例による全重み更新・メモリ確認 |
| [training.py](../src/experiments/qwen_sid/training.py) | 全重み更新の学習ループ、validation、checkpoint保存・再開 |
| [evaluation.py](../src/experiments/qwen_sid/evaluation.py) | GPUでのSID prefix制約、beam生成、ランキング指標 |
| [comparison.py](../src/experiments/qwen_sid/comparison.py) | A/Bの同条件検証、比較結果の出力 |
| [test_qwen_sid.py](../tests/test_qwen_sid.py) | 元GRIDと同じ抽出、lossと勾配、prefix mask、更新・再読込のテスト |

ローカルモデルは `artifacts/models/Qwen3-0.6B/`、revisionは
`c1899de289a04d12100db370d81485cdf75e47ca`。学習ログ用のrevisionはこのディレクトリ内の
`.cache/huggingface/download/config.json.metadata` から取得する。`pyproject.toml` と `uv.lock` で
Transformersを4.47.0から4.51.3へ更新済み。PEFTは0.15.2。

本学習の更新方式とstep設定を確定したら、リポジトリ直下から実行する。
全重み更新の入口は以下。A/Bランナーは `max_steps` をカバーする共通planを作成し、probe後に各条件を
次の評価stepまで交互に進める。同時停止後に各best checkpointの評価とA/B比較を行う。

```bash
uv run python scripts/run_qwen_sid_ablation.py --config configs/local/qwen3_toys.json
```

段階ごとの入口は [run_qwen_sid.py](../scripts/run_qwen_sid.py)。`prepare` で生成するのはmanifestと
タスク例まで。`train` を単独で呼ぶ場合は、事前に `data.prepare_recommendation_plan(config)` も実行する。
`train --config <条件JSON> --until-step 100` で100更新まで進めて保存でき、`python -m pdb` 経由でも実行できる。
指定stepが評価間隔の途中なら、保存して中断し、次の評価は元の間隔に従う。単独実行の再開は同じ設定で
`--resume` を付ける。A/B実行の再開は同じランナーを再実行する。ランナーは
`last/progress.json` の完了stepを読み、未完了の条件から進める。初回checkpoint前の中断なら初期値からやり直す。
checkpointは評価stepまたは指定した中断stepで保存し、推薦系列内の次の位置と改善なし回数も復元する。
学習条件を変更する場合は新しい `output_dir` を使う。seed・サンプリング条件が変わる場合や
既存planで `max_steps` に届かない場合は、別の `prepared_dir` にplanを生成する。

| 保存先 | 内容 |
|---|---|
| `artifacts/qwen3_toys_data/` | 準備済み `manifest.json`、`sids.json`、`training_histories.json`、split別JSONL。推薦planと各epochの系列JSONLは本学習準備で生成 |
| `artifacts/qwen3_toys_alignment_ablation/` | 既存の `probe.json`、速度計測。開始後は `status.json`、設定、各stageのログ、`comparison.md` と `comparison.json` |
| 上記の `without_alignment/` と `with_alignment/` | `experiment.json`、`status.json`、step別の `metrics.jsonl`、`best/`、再開用 `last/`、直近validation予測、最終の `evaluation_*` と `testing_*`、`results.json` |

コードの変更後は `uv run pytest -q`、対象ファイルの `uv run ruff check`、`git diff --check` で確認する。
