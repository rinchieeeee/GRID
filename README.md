# Generative Recommendation with Semantic IDs (GRID)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/config-hydra-89b8cd)](https://hydra.cc/)
[![Lightning](https://img.shields.io/badge/pytorch-lightning-792ee5)](https://lightning.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.22224-b31b1b.svg)](https://arxiv.org/abs/2507.22224)


**GRID** (Generative Recommendation with Semantic IDs) is a state-of-the-art framework for generative recommendation systems using semantic IDs, developed by a group of scientists and engineers from [Snap Research](https://research.snap.com/team/user-modeling-and-personalization.html). This project implements novel approaches for learning semantic IDs from text embedding and generating recommendations through transformer-based generative models.

## 🚀 Overview

GRID facilitates generative recommendation three overarching steps:

- **Embedding Generation with LLMs**: Converting item text into embeddings using any LLMs available on Huggingface. 
- **Semantic ID Learning**: Converting item embedding into hierarchical semantic IDs using Residual Quantization techniques such as RQ-KMeans, RQ-VAE, RVQ. 
- **Generative Recommendations**: Using transformer architectures to generate recommendation sequences as semantic ID tokens. 


## 📦 Installation

### Prerequisites
- Python 3.12.5 managed by pyenv
- uv
- Linux x86_64 and an NVIDIA GPU with a CUDA 12.8 compatible driver

### Setup Environment

```bash
git clone https://github.com/snap-research/GRID.git
cd GRID

pyenv install -s 3.12.5
pyenv local 3.12.5

# Reproduce the environment from uv.lock using the pyenv interpreter
uv sync --locked --python "$(pyenv which python)"

# Check dependency consistency and run commands
uv pip check
uv run python -m src.train --help
```

Dependencies are managed in `pyproject.toml` and pinned in `uv.lock`.
The PyTorch family and FBGEMM use the official CUDA 12.8 wheel index.
This setup matches the local CUDA Toolkit 12.8 and RTX 3090.
Use `uv run` to execute commands, or `source .venv/bin/activate` to activate
this environment in your shell.

## 🎯 Quick Start

### 1. Data Preparation

Prepare your dataset in the expected format:
```
data/
├── train/       # training sequence of user history 
├── validation/  # validation sequence of user history 
├── test/        # testing sequence of user history 
└── items/       # text of all items in the dataset
```

We provide pre-processed Amazon data explored in the [P5 paper](https://arxiv.org/abs/2203.13366) [4]. The data can be downloaded from this [google drive link](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing).

### 2. Embedding Generation with LLMs

Generate embeddings from LLMs, which later will be transformed into semantic IDs. 

```bash
python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty # avaiable data includes 'beauty', 'sports', and 'toys'
```

### 3. Train and Generate Semantic IDs

Learn semantic ID centroids for embeddings generated in step 2:

```bash
python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<output_path_from_step_2>/merged_predictions_tensor.pt \ # this can be found in the log dirs in step2
    embedding_dim=2048 \ # the model dimension of the LLMs you use in step 2. 2048 for flan-t5-xl as used in this example.
    num_hierarchies=3 \  # we train 3 codebooks
    codebook_width=256 \ # each codebook has 256 rows of centroids  
```

Generate SIDs:

```bash
python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<output_path_from_step_2>/merged_predictions_tensor.pt \ 
    embedding_dim=2048 \ 
    num_hierarchies=3 \  
    codebook_width=256 \ 
    ckpt_path=<the_checkpoint_you_just_get_above> # this can be found in the log dir for training SIDs
```


### 4. Train Generative Recommendation Model with Semantic IDs

Train the recommendation model using the learned semantic IDs:

```bash
python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/beauty \ 
    semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 # Please note that we add 1 for num_hierarchies because in the previous step we appended one additional digit to de-duplicate the semantic IDs we generate.
```

### 4. Generate Recommendations

Run inference to generate recommendations:

```bash
python -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/beauty \ 
    semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
    ckpt_path=<the_checkpoint_you_just_get_above> \ # this can be found in the log dir for training GR models
    num_hierarchies=4 \ # Please note that we add 1 for num_hierarchies because in the previous step we appended one additional digit to de-duplicate the semantic IDs we generate.
```

## Supported Models:

### Semantic ID:

1. Residual K-means proposed in One-Rec [2]
2. Residual Vector Quantization
3. Residual Quantization with Variational Autoencoder [3]

### Generative Recommendation:

1. TIGER [1]

## 📚 Citation

If you use GRID in your research, please cite:

```bibtex
@inproceedings{grid,
  title     = {Generative Recommendation with Semantic IDs: A Practitioner's Handbook},
  author    = {Ju, Clark Mingxuan and Collins, Liam and Neves, Leonardo and Kumar, Bhuvesh and Wang, Louis Yufeng and Zhao, Tong and Shah, Neil},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2025}
}
```

## 🤝 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Configuration management by [Hydra](https://hydra.cc/)
- Inspired by recent advances in generative AI and recommendation systems
- Part of this repo is built on top of https://github.com/ashleve/lightning-hydra-template

## 📞 Contact

For questions and support:
- Create an issue on GitHub
- Contact the development team: Clark Mingxuan Ju (mju@snap.com), Liam Collins (lcollins2@snap.com), Bhuvesh Kumar (bhuvesh@snap.com) and Leonardo Neves (lneves@snap.com).

## Bibliography 

[1] Rajput, Shashank, et al. "Recommender systems with generative retrieval." Advances in Neural Information Processing Systems 36 (2023): 10299-10315.

[2] Deng, Jiaxin, et al. "Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment." arXiv preprint arXiv:2502.18965 (2025).

[3] Lee, Doyup, et al. "Autoregressive image generation using residual quantization." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[4] Geng, Shijie, et al. "Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5)." Proceedings of the 16th ACM conference on recommender systems. 2022.

## 分析コードの使い方

デバッグ、追加実験、Pythonからの個別実行、ログ集計の手順は
[分析・追加実験ガイド](docs/experiments.md) を参照してください。
実験条件は `configs/local/toys.json` に集約しています。
既存の成果物がある場合は、新しい出力先を指定して実行します。

```bash
uv run python scripts/run_toys_pipeline.py --output-dir artifacts/toys_trial_01 --dry-run
```

## Local Toys experiment

`scripts/inspect_amazon_data.py` counts records in the extracted Amazon datasets.
Toys is the smallest category: 11,924 items and 19,412 user histories in each
of training, evaluation and testing. The source archive is the Google Drive
link in Data Preparation; extract its `amazon_data/` directory under `data/`.

```bash
uv run python scripts/inspect_amazon_data.py
uv run python scripts/run_toys_pipeline.py
```

The pipeline generates FLAN-T5-XL embeddings, trains a three-level Residual
K-means with 256 codes per level for 90 epochs (approximately 30 passes per
level), generates unique four-level Semantic IDs, and trains TIGER for exactly
10 epochs with early stopping disabled. The finite training loader visits each
record once per epoch. Training augmentation samples up to 32 examples per
batch, and uses BF16 mixed precision on one GPU. Validation runs after each
TIGER epoch; the best validation Recall@5 checkpoint is used for the final test.

Outputs, commands, configuration snapshots, console logs, metrics and
checkpoints are under `artifacts/toys_rkmeans_10epochs/`. Run
`uv run python scripts/summarize_toys_run.py` to refresh `results.md` and
`summary.json` with the validation history and final test results. `status.json` tracks
each stage. A completed stage has a `.complete` marker and is skipped on
subsequent runs. Downloaded data and experiment artifacts are ignored by Git.

The local trial changes the original experiment settings: training augmentation
is capped at 32 examples rather than 512, precision is BF16 mixed rather than
FP32, and validation/test batch sizes are 128 rather than 8. The source history
batch size remains 32 and gradient accumulation remains 16. Thus the optimizer
sees up to 512 augmented examples per update, compared with up to 8,192 under
the original cap. These settings affect training and this trial is not a strict
reproduction of the original experiment.

### Original configuration runtime check

The original `rkmeans_train_flat` and `tiger_train_flat` settings were also
verified on the RTX 3090. R-KMeans completed its original 30 optimizer steps.
TIGER ran with FP32, augmentation cap 512, gradient accumulation 16, and eight
data-loading processes: 64 batches, four optimizer updates and full validation
passed. DDP initialized successfully with one GPU and `world_size=1`. Peak
PyTorch reserved memory was 12.91 GiB. This was a short runtime check, not a
complete 320,000-step run; a diagnostic callback stopped it after 64 batches.
Artifacts and configuration snapshots are in `artifacts/toys_original_config/`.

### 元設定による早期終了付き学習

```bash
uv run python scripts/start_toys_training.py
```

元の `tiger_train_flat` を使い、`trainer.max_epochs=100` のみを変更して
独立プロセスで学習を開始します。読み込みバッチ32、拡張上限512、FP32、
勾配蓄積16、ワーカー8、単一GPUでのDDPを維持します。検証は1,600バッチ
（100回のパラメータ更新）ごとに実行し、検証 Recall@5 が10回連続で
改善しなければ早期終了します。

元の学習ローダーは無限反復するため、データ1周でエポックが終了しません。
通常は epoch 0 のまま早期終了または320,000更新の上限まで進み、
`max_epochs=100` は「データ100周」の制限にはなりません。

この実行は `toys_original_config` に保存した元設定の30ステップで学習した
Residual K-means と、そのチェックポイントから生成したSIDを使用します。
前節の動作確認が済んでおり、データとSIDが同じ相対パスに存在することが
必要です。実行先と監視プロセスIDは
`artifacts/toys_original_100epochs/latest_run.json`、各実行の状態は
`status.json`、ログは `console.log` に保存されます。

2026年11月10日のToys実行は約4時間43分で正常終了しました。4,600更新で
早期終了し、検証 Recall@5 が最良だった3,600更新時点のモデルで
テスト履歴19,412件を評価しました。

| テスト指標 | 結果 |
|---|---:|
| Recall@5 | 0.042396 |
| Recall@10 | 0.062333 |
| NDCG@5 | 0.027544 |
| NDCG@10 | 0.033976 |

推薦モデルは13,128,064パラメータです。この実行は公開configに従った
単一シードの結果であり、論文の実験条件を厳密に再現したものではありません。
