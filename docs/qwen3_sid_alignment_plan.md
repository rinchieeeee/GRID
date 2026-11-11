# Qwen3によるSID–Text alignmentの対照実験

更新日: 2026-11-11。ブランチ: `experiment/qwen3-sid-text-alignment`。
目的は、同じQwen3にSID–Text alignmentを加えることで推薦精度が改善するかを調べること。

Qwen2.5-0.5Bでの推薦のみ／alignmentありの学習・test評価は完了した。
TIGERを含む比較結果は[最終比較結果](qwen25_toys_test_results.md)を参照。
以下はQwen3向けの計画であり、Qwen2.5の実更新数・評価間隔とは異なる。
本学習は未開始。更新方式ごとの速度計測、実装状況、再開手順は [引き継ぎ資料](qwen3_sid_alignment_handoff.md) に記録した。

## 比較する条件

| 項目 | A | B |
|---|---|---|
| 主タスク | SID履歴→次商品SID | 同じ |
| 補助タスク | なし | 商品メタデータ→SID、SID→商品メタデータ |
| loss | 推薦CE | 推薦CE＋λ/2×（両方向のCEの和） |
| λ | 0 | 1 |
| 初期値・SID・推薦例・順序・更新数 | 共通 | 共通 |
| 評価 | prefix制約あり、beam幅10 | 同じ |

3タスクの例を単純に混ぜて推薦例を減らす方式ではない。
推薦128例につき1更新し、Bだけ同じ更新に各方向128例の補助lossを加算する。
各例のCEは回答長で平均する。promptとpaddingは教師から除く。
SID回答は4桁のみを教師とし、メタデータ回答ではEOSも教師にする。
初期値全体と推薦系列planのSHA256を比較し、A/Bの一致を検証する。

## データと推薦例の作り方

- データは `data/amazon_data/toys/`。商品11,924件、各splitに19,412ユーザー履歴。
- 既存SIDは `artifacts/toys_original_config/02_semantic_ids/pickle/merged_predictions_tensor.pt`。
  `[4, 11924]` で全商品が一意。3階層の0〜255と衝突解消の0〜12からなる。
- SHA256: `756b4308b442685a46838b7858fee45f4f1104bd868f33af535438dd9410450a`。
- 元embedding作成時と同じ商品名・ブランド・カテゴリ・価格の全文を、加工せず使用する。
  全文が一致する商品は一部あり、11,924商品に対して異なるテキストは11,898件。
- alignment用に商品単位で5%（596商品）を保留し、両方向とも同じ分割にする。
  学習には残り11,328商品の両方向を利用する。カタログは既存SID同様transductive。
- 推薦のtraining/evaluation/testingは既存分割を維持する。
  評価系列の入力は元データで直近20商品に制限されており、そのsuffix関係を全件検査する。
- 元GRID同様、32ユーザーの履歴中の全連続部分列から最大512例を抽出する。
  元collateの抽出順・randintによる抽出・重複indexの除去と一致することをテスト済み。
  各部分列は末尾30商品を残し、末尾商品を正解、直前までの最大29商品を入力にする。
- 元GRIDとの意図的な差として、毎周全履歴を巡回するため最後の小さいsource batchも使用する。
  すべてのユーザーを候補生成に使うが、サンプリング後の例に全ユーザーが必ず残るわけではない。
- 1 epoch目の推薦例は251,869件、実効batch128では1,968更新。
  毎epoch再shuffle・再抽出した系列を共通ファイルへ保存し、両条件で同じものを読む。
  epochごとの推薦例数は多少変わる。全連続部分列をすべて学習する方式ではない。
- testの正解は学習・学習条件選択に使用しない。

## モデルと速度確認

モデルは `Qwen/Qwen3-0.6B`、revisionはダウンロードメタデータに記録する。
元の言語token IDを維持し、4階層×256のSID tokenを追加する。
第4階層の未使用コードも確保するが、prefix制約付き評価では有効なSIDだけを許可する。
全SIDのembeddingは同じ乱数分布で初期化し、条件間で初期値を一致させる。
Transformersは4.51.3へ更新済み。thinkingは無効にする。

- フル更新では既存embedding・追加SID・Transformerの全パラメータを更新する。
- LoRAの確認でも、**既存tokenと追加SIDを含むembedding全行を更新する**。
  Transformerはattentionのq/k/v/oとMLPのgate/up/downにrank16・alpha32のLoRAを適用する。
  入力embeddingとlm_headの共有を維持する。LoRA dropoutは0。
- LoRA adapter保存時には `save_embedding_layers=True` を指定し、更新した全embeddingも保存する。
  adapter再読込後に学習を再開する場合はembeddingのrequires_gradも有効にする必要がある。
- 既存embedding固定の速度は、後続の独立した比較でLoRAあり／なしの両方を測定した。
  固定方式は計測まで検証済み。現行の本学習入口は全重み更新に対応する。
- GPUはRTX 3090（24 GiB）。重みとAdamWはFP32、計算はBF16 autocast。
- フル更新で選定した設定はmicro batch16、勾配蓄積8、checkpointing無効。
  長いテキストにはpadding/回答token数の上限でmicro batchを小分けし、実効batchとlossの重みを維持する。
- 速度測定だけのコードはリポジトリに残さない。測定値はignoredなartifactsに保存する。
  評価・loss・学習履歴・データ検証に必要なコードは残す。

## 本学習の初期設定

設定は `configs/local/qwen3_toys.json`。optimizerはAdamW、学習率3e-4、weight decay0、
cosine decay、warmup3%、gradient clipping1。seed42。
学習上限は初期設定で20,000 optimizer step（従来の10 epoch案と同程度）。
勾配蓄積後の `optimizer.step()` を1 stepと数え、各1,000 stepと学習上限で全validationを評価する。
各条件のRecall@5が厳密に改善したstepをbest checkpointに採用する。
単独実行では改善なしが連続10評価に達した時点で停止する。A/B実行では同じ評価stepで両条件が
この基準を満たした時点で同時に停止し、各条件のbestでtestを評価する。学習率のスケジュールは上限stepを使う。
epochは共通推薦系列を再抽出する単位。速度計測は完了し、本学習の更新方式は未決定。

元TIGERのsource batch32×勾配蓄積16では、延べ履歴件数で1 epochが約38 optimizer stepに相当する。
元TIGERは無限ローダーなので、これは表示上のepochではなく入力履歴件数の換算。
前回TIGERは1,600学習バッチごと（勾配蓄積16で100更新ごと）にRecall@5を評価し、
改善なし10評価で早期終了する設定で、実績は4,600更新だった。
Qwen3の初期評価間隔は全件評価の時間を考慮して1,000更新とした。

## 評価と比較

- 全カタログを対象とし、test全19,412件でHit@1、Recall@5/10、NDCG@5/10を計算する。
- prefix制約あり・beam幅10・長さ正規化なし、固定4桁を生成する。
  prefixのマスクはGPUで一括適用し、通常のtrieマスクと一致することをテストする。
- testは学習条件の決定とvalidationによるcheckpoint選択後に評価する。
- 単一seedの探索結果として扱う。Bには補助タスク分の追加計算量がある。
- 前回TIGERの参考値: Recall@5=0.042396、Recall@10=0.062333、
  NDCG@5=0.027544、NDCG@10=0.033976。
  元結果はbeam幅10・prefix制約なしで、構造・規模・学習量も異なる。

## コードとデバッグ

| ファイル | 役割 |
|---|---|
| `src/experiments/qwen_sid/data.py` | 商品・履歴の読取、部分列抽出、共通推薦plan、回答mask |
| `model.py` | SID追加、全更新・LoRA構成、回答位置だけのCE |
| `optimization.py` | 補助タスクの抽出、メモリ上限内のmicro batch、勾配加算 |
| `probe.py` | 最長実例での更新可否・勾配・メモリ検査 |
| `training.py` | 全更新によるA/B学習、checkpoint、再開、学習履歴 |
| `evaluation.py` | 制約付き生成、ランキング指標、全件評価 |
| `comparison.py` | 比較条件の一致検査とA/Bレポート |

現状、LoRAは構成・loss・保存再読込・速度の検証まで。本学習ループへの組込みは更新方式決定後に行う。
各関数をPythonから呼べる。importでは実行しない。前景の入口は以下。

```bash
uv run python scripts/run_qwen_sid.py prepare
uv run python scripts/run_qwen_sid.py probe
# 本学習を開始する場合の入口（現在は未実行）
uv run python scripts/run_qwen_sid_ablation.py
# 例: pdbで特定条件の学習を追う
uv run python -m pdb scripts/run_qwen_sid.py train --config <条件JSON>
```

出典: [MiniOneRec §3.3 / §4 / Appendix A](https://arxiv.org/pdf/2510.24431)。
モデル・既存SID・Toys分割・選択タスクが異なるため、論文の厳密な再現ではない。
