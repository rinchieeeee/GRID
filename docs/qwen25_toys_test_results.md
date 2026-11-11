# Toys：TIGER型推薦モデルとQwen2.5のtest評価結果

実験日：2026-11-10〜11。保存済みの最終test評価を集計した。

## 結果

Qwen2.5-0.5BはSID–Text alignmentの追加でRecall@5/10・NDCG@5/10が改善したが、Hit@1は低下した。TIGER型モデルは、共通して記録された4指標すべてでQwenの両条件を上回った。

数値は0〜1の値。太字は各行の最大値。TIGERのHit@1は未記録。

| test指標 | TIGER型 | Qwen alignmentなし | Qwen alignmentあり | Qwen内の相対変化 |
|---|---:|---:|---:|---:|
| hit@1 | 未記録 | **0.010715** | 0.009891 | -7.69% |
| recall@5 | **0.042396** | 0.028694 | 0.030857 | +7.54% |
| recall@10 | **0.062333** | 0.041109 | 0.044509 | +8.27% |
| ndcg@5 | **0.027544** | 0.019908 | 0.020664 | +3.80% |
| ndcg@10 | **0.033976** | 0.023889 | 0.025050 | +4.86% |

相対変化は `(alignmentあり / alignmentなし − 1) × 100`。Recall@5は0.028694から0.030857へ、絶対差で約0.002164、相対で7.54%改善した。alignmentありのQwenはTIGERに対し、Recall@5で27.22%、NDCG@5で24.98%低い。

## 学習とcheckpoint選択

| 項目 | TIGER型 | Qwen alignmentなし | Qwen alignmentあり |
|---|---:|---:|---:|
| パラメータ数 | 13,128,064 | 494,707,456 | 494,707,456 |
| 終了optimizer step | 4,600 | 4,200 | 3,200 |
| testに使ったbest step | 3,600 | 3,400 | 2,400 |
| 累積推薦例数 | — | 537,515 | 409,565 |
| best validation Recall@5 | 0.052648 | 0.040490 | 0.041057 |

各モデルはvalidation Recall@5でbest checkpointを選択し、選択後にtestを評価した。TIGERのtest結果はCSVの最終行（step=4,600）にあるが、評価に使われたcheckpointはstep=3,600である。

Qwenの両条件は全重み更新、seed42、実効推薦batch128（micro batch32×勾配蓄積4）、AdamW、学習率3e-4、weight decay0、warmup600 step、cosine decay、上限20,000 step。200 stepごとに全validationを評価し、改善なし4回で早期終了した。設定差分は保存先・condition・alignment_weight（0→1）のみ。初期パラメータ、推薦plan、モデルrevisionの一致を検証済み。

alignmentありでは推薦128例に対し、メタデータ全文→SIDとSID→メタデータ全文を各128例、独立に復元抽出して同じ更新へ加える。各例の回答tokenのCEを平均し、タスク内でも平均したうえで、`CE_rec + 0.5 × (CE_text_to_sid + CE_sid_to_text)` を最小化する。商品単位のalignment学習分割は11,328件、保留は596件。

## 比較条件と解釈の限界

- 同じAmazon Toys分割と既存4階層SIDを使用。Qwenのtestは19,412件、カタログ11,924商品。Qwen両条件のtestユーザー・正解・順序の一致を確認済み。
- beam幅は全条件10。TIGERはprefix制約なし、Qwenは有効SIDのprefix制約あり。
- Qwenの2条件は同じ上限・評価間隔・早期終了設定だが、実更新数と推薦例の累積件数は異なる。同じ更新数に固定した比較ではない。
- TIGERとQwenはモデル構造、推薦例のサンプリング、1更新の処理量、学習量、早期終了条件が異なる。TIGERとの精度差をalignmentだけの効果と解釈しない。
- 単一seedの結果で、統計的有意性や再現性は未検証。

## 結果・予測・設定の保存先

以下はリポジトリルートからの相対パス。`artifacts/` はGit管理対象外で、ローカルの評価結果・モデル・予測はpushされない。このMarkdownと実験コード・設定をGitに保存する。

| 条件 | 集計結果 | 各評価例の予測 | 設定 |
|---|---|---|---|
| TIGER型 | [metrics.csv](../artifacts/toys_original_100epochs/20261110_000649/csv/version_0/metrics.csv) 最終行の`test/*`列 | この比較では未確認 | [hparams.yaml](../artifacts/toys_original_100epochs/20261110_000649/csv/version_0/hparams.yaml) |
| Qwen alignmentなし | [testing_metrics.json](../artifacts/qwen25_toys_alignment_ablation/without_alignment/testing_metrics.json) | [testing_predictions.jsonl](../artifacts/qwen25_toys_alignment_ablation/without_alignment/testing_predictions.jsonl) | [qwen25_toys_baseline.json](../configs/local/qwen25_toys_baseline.json) |
| Qwen alignmentあり | [testing_metrics.json](../artifacts/qwen25_toys_alignment_ablation/with_alignment/testing_metrics.json) | [testing_predictions.jsonl](../artifacts/qwen25_toys_alignment_ablation/with_alignment/testing_predictions.jsonl) | [qwen25_toys_with_alignment.json](../configs/local/qwen25_toys_with_alignment.json) |

Qwenの `results.json` は `evaluation` キーにvalidation、`testing` キーにtestを保存する。予測JSONLの各行には `user_id`・`target`・`predictions` を保存する。

## Qwenの実行・再開

データ準備と共通推薦planの作成は[引き継ぎ資料](qwen3_sid_alignment_handoff.md)を参照。以下はリポジトリルートで前景実行する。既存の出力先に新規学習を重ねると上書き防止で停止する。

```bash
# 新規のalignmentあり学習（今回の実験は完了済み）
uv run python scripts/run_qwen_sid.py train --config configs/local/qwen25_toys_with_alignment.json
# checkpointから再開（早期終了済みなら追加更新しない）
uv run python scripts/run_qwen_sid.py train --config configs/local/qwen25_toys_with_alignment.json --resume
# best checkpointでvalidationとtestを全件評価
uv run python scripts/run_qwen_sid.py evaluate --config configs/local/qwen25_toys_with_alignment.json
```

今回のQwen比較は各条件を個別実行した。`run_qwen_sid_ablation.py` は両条件を同じ実step数まで進める別の実行方式であり、今回の個別早期終了の結果に対して厳密同stepの `compare_runs` を適用しない。
