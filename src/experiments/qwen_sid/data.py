"""TFRecordから商品・履歴を読み、3タスクの回答付きデータを作る。"""

import gzip
import hashlib
import json
import math
import random
import struct
from bisect import bisect_right
from pathlib import Path

import torch

from src.experiments.common import write_json


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(folder: Path):
    """tf.dataを起動せずExampleを読む。embeddingなど不要な列は返さない。"""
    from tensorflow.core.example.example_pb2 import Example

    paths = sorted(folder.glob("*.tfrecord.gz"))
    if not paths:
        raise FileNotFoundError(folder)
    for path in paths:
        with gzip.open(path, "rb") as stream:
            while header := stream.read(12):
                if len(header) != 12:
                    raise ValueError(f"不完全なTFRecordヘッダ: {path}")
                size = struct.unpack("<Q", header[:8])[0]
                record = stream.read(size)
                if len(record) != size or len(stream.read(4)) != 4:
                    raise ValueError(f"不完全なTFRecord: {path}")
                feature = Example.FromString(record).features.feature
                if "id" in feature:
                    yield {
                        "item_id": int(feature["id"].int64_list.value[0]),
                        "text": feature["text"].bytes_list.value[0].decode("utf-8"),
                    }
                else:
                    yield {
                        "user_id": int(feature["user_id"].int64_list.value[0]),
                        "items": list(feature["sequence_data"].int64_list.value),
                    }


def sid_text(codes: list[int]) -> str:
    return "".join(f"<sid_{level}_{code}>" for level, code in enumerate(codes))


def recommendation_example(row: dict, sids: list[list[int]], history_items: int):
    items = row["items"]
    if len(items) < 2:
        raise ValueError("推薦履歴には入力と正解の2商品以上が必要です")
    history = items[:-1][-history_items:]
    return {
        "task": "recommendation",
        "user_id": row["user_id"],
        "item_id": items[-1],
        "prompt": "Recommend the next item from this chronological purchase history. "
        "Respond only with its semantic ID.\nHistory: "
        + " ".join(sid_text(sids[item]) for item in history),
        "response": sid_text(sids[items[-1]]),
    }


def alignment_examples(item: dict, codes: list[int]):
    """商品メタデータは元の文字列を加工せず全文使う。"""
    sid = sid_text(codes)
    return [
        {
            "task": "text_to_sid",
            "item_id": item["item_id"],
            "prompt": "Identify the item described by this metadata. "
            "Respond only with its semantic ID.\nMetadata: " + item["text"],
            "response": sid,
        },
        {
            "task": "sid_to_text",
            "item_id": item["item_id"],
            "prompt": "Provide the full item metadata (title, brand, categories, "
            "and price) for this semantic ID.\nSemantic ID: " + sid,
            "response": item["text"],
        },
    ]


def sample_grid_subsequences(rows: list[dict], cap: int, generator=None):
    """元collateと同じ抽出順・randint抽出・重複index除去で部分列を返す。

    元実装は全候補を列挙して選択indexを照合する。ここでは選択indexから
    (ユーザー、終端、始端)を逆算し、長い履歴でも不要な候補を展開しない。
    """
    ends = []
    total = 0
    for row in rows:
        n = len(row["items"])
        total += n * (n - 1) // 2
        ends.append(total)
    if total > cap:
        selected = sorted(
            set(torch.randint(total, (cap,), generator=generator).tolist())
        )
    else:
        selected = range(total)
    result = []
    for index in selected:
        row_index = bisect_right(ends, index)
        local = index - (ends[row_index - 1] if row_index else 0)
        end = (1 + math.isqrt(1 + 8 * local)) // 2 + 1
        start = local - (end - 1) * (end - 2) // 2
        row = rows[row_index]
        result.append({"user_id": row["user_id"], "items": row["items"][start:end]})
    return result


def training_sequences(histories, config, epoch):
    """source batch32・上限512・SID120トークンの元GRID条件で推薦例を作る。"""
    rows = list(histories)
    random.Random(config["seed"] + epoch).shuffle(rows)
    generator = torch.Generator().manual_seed(config["seed"] + epoch)
    size = config["source_history_batch_size"]
    examples = []
    # 全ユーザーを毎周利用するため、元GRIDのdrop_lastだけは無効にする。
    for start in range(0, len(rows), size):
        subsequences = sample_grid_subsequences(
            rows[start : start + size], config["augmentation_cap"], generator
        )
        examples.extend(
            {
                "user_id": row["user_id"],
                "items": row["items"][-(config["history_items"] + 1) :],
            }
            for row in subsequences
        )
    # 推薦順序もalignmentのサンプリングとは独立に固定する。
    random.Random(config["seed"] + 10000 + epoch).shuffle(examples)
    return examples


def training_recommendations(histories, sids, config, epoch):
    return [
        recommendation_example(row, sids, config["history_items"])
        for row in training_sequences(histories, config, epoch)
    ]


def recommendation_plan_inputs(config: dict) -> dict:
    return {
        "seed": config["seed"],
        "source_manifest_sha256": file_hash(
            Path(config["prepared_dir"]) / "manifest.json"
        ),
        "sampling": {
            key: config[key]
            for key in (
                "history_items",
                "source_history_batch_size",
                "augmentation_cap",
            )
        },
    }


def load_recommendation_plan(config: dict) -> dict:
    out = Path(config["prepared_dir"])
    plan = json.loads((out / "recommendation_plan.json").read_text())
    inputs = recommendation_plan_inputs(config)
    if any(plan.get(key) != value for key, value in inputs.items()):
        raise ValueError("共通推薦planの入力manifestまたは抽出条件が変わっています")
    batch_size = config["micro_batch_size"] * config["gradient_accumulation_steps"]
    steps = sum(math.ceil(epoch["examples"] / batch_size) for epoch in plan["epochs"])
    if steps < config["max_steps"]:
        raise ValueError(
            "共通推薦planの更新数が不足しています。別のprepared_dirで生成してください"
        )
    return plan


def prepare_recommendation_plan(config: dict) -> dict:
    """max_stepsまでの更新に必要な推薦系列を、epochごとに共通保存する。"""
    out = Path(config["prepared_dir"])
    if (out / "recommendation_plan.json").exists():
        return load_recommendation_plan(config)
    histories = json.loads((out / "training_histories.json").read_text())
    batch_size = config["micro_batch_size"] * config["gradient_accumulation_steps"]
    epochs = []
    steps = 0
    while steps < config["max_steps"]:
        epoch = len(epochs)
        rows = training_sequences(histories, config, epoch)
        if not rows:
            raise ValueError("推薦例が空です")
        path = out / f"recommendation_epoch_{epoch:02d}.jsonl"
        with path.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        epochs.append(
            {"path": str(path), "examples": len(rows), "sha256": file_hash(path)}
        )
        steps += math.ceil(len(rows) / batch_size)
        print(
            f"prepared epoch={epoch + 1} recommendation_examples={len(rows)}",
            flush=True,
        )
    plan = {"epochs": epochs, **recommendation_plan_inputs(config)}
    write_json(out / "recommendation_plan.json", plan)
    return plan


def validate_histories(splits: dict, item_count: int) -> dict:
    """同一ユーザーのtrain→validation→testの末尾追加関係を全件検査する。"""
    indexed = {}
    for name, rows in splits.items():
        indexed[name] = {row["user_id"]: row["items"] for row in rows}
        if len(indexed[name]) != len(rows):
            raise ValueError(f"ユーザー重複: {name}")
        for row in rows:
            if len(row["items"]) < 2:
                raise ValueError(f"短すぎる履歴: {name}")
            if any(not 0 <= item < item_count for item in row["items"]):
                raise ValueError(f"範囲外の商品ID: {name}")
    train, val, test = (indexed[n] for n in ("training", "evaluation", "testing"))
    if train.keys() != val.keys() or train.keys() != test.keys():
        raise ValueError("split間のユーザー集合が異なります")
    # 公開データの評価履歴は直近20商品に制限されている。
    # trainが長い場合は完全prefixでなく、既知履歴のsuffixと一致する。
    for earlier, later in ((train, val), (val, test)):
        for user, previous in earlier.items():
            history = later[user][:-1]
            if previous != history and not (
                len(history) == 20 and previous[-20:] == history
            ):
                raise ValueError("split間で履歴のprefix/suffix関係が崩れています")
    return {name: len(rows) for name, rows in splits.items()}


def prepare_data(config: dict) -> dict:
    folder = Path(config["data_dir"])
    sid_path = Path(config["sid_path"])
    sid_tensor = torch.load(sid_path, map_location="cpu", weights_only=True)
    items = sorted(read_records(folder / "items"), key=lambda row: row["item_id"])
    if [row["item_id"] for row in items] != list(range(len(items))):
        raise ValueError("商品IDの欠損・重複があります")
    if sid_tensor.shape != (4, len(items)) or sid_tensor.is_floating_point():
        raise ValueError("SIDの形状または型が不正です")
    if (sid_tensor < 0).any() or len(torch.unique(sid_tensor.T, dim=0)) != len(items):
        raise ValueError("SIDに負値・重複があります")
    if any(not row["text"].strip() for row in items):
        raise ValueError("空の商品メタデータがあります")
    sids = sid_tensor.T.tolist()
    splits = {
        name: list(read_records(folder / name))
        for name in ("training", "evaluation", "testing")
    }
    counts = validate_histories(splits, len(items))
    shuffled = list(range(len(items)))
    random.Random(config["seed"]).shuffle(shuffled)
    val_count = max(1, int(len(items) * config["alignment_validation_fraction"]))
    heldout = set(shuffled[:val_count])
    alignment = {"training": [], "evaluation": []}
    for item in items:
        split = "evaluation" if item["item_id"] in heldout else "training"
        alignment[split].extend(alignment_examples(item, sids[item["item_id"]]))
    out = Path(config["prepared_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "training_histories.json", splits["training"])
    for split, rows in splits.items():
        examples = [
            recommendation_example(row, sids, config["history_items"]) for row in rows
        ]
        if split in alignment:
            examples.extend(alignment[split])
        with (out / f"{split}.jsonl").open("w") as stream:
            for row in examples:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(out / "sids.json", sids)
    source_files = sorted(folder.glob("*/*.tfrecord.gz"))
    manifest = {
        "config": config,
        "sid_sha256": file_hash(sid_path),
        "source_sha256": {str(p): file_hash(p) for p in source_files},
        "prepared_sha256": {
            p.name: file_hash(p)
            for p in [
                *out.glob("*.jsonl"),
                out / "sids.json",
                out / "training_histories.json",
            ]
        },
        "items": len(items),
        "history_counts": counts,
        "unique_metadata": len({row["text"] for row in items}),
        "alignment_train_items": len(items) - val_count,
        "alignment_validation_items": val_count,
        "catalog_policy": "transductive; metadata holdout by item; no heldout behavior",
        "sid_tokens": [
            f"<sid_{level}_{code}>"
            for level in range(4)
            for code in range(config["sid_codebook_width"])
        ],
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def read_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def encode_example(tokenizer, row: dict, max_length: int) -> dict:
    """回答境界で明示的に分割し、promptとpaddingを教師から除く。"""
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    answer = tokenizer.encode(row["response"], add_special_tokens=False)
    # SIDは元GRIDと同じ4桁を教師とする。文章回答のみEOSも教師にする。
    if row["task"] == "sid_to_text":
        answer.append(tokenizer.eos_token_id)
    if len(prefix) + len(answer) > max_length:
        raise ValueError(
            f"最大長超過（全文を保持するため切り詰めません）: {row['task']}"
        )
    return {
        "input_ids": prefix + answer,
        "labels": [-100] * len(prefix) + answer,
        "prompt_ids": prefix,
        "task": row["task"],
        "item_id": row["item_id"],
    }


def collate_examples(rows: list[dict], pad_id: int, device="cpu", pad_to=None):
    length = max(len(row["input_ids"]) for row in rows)
    length = max(length, pad_to or 0)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"])
        labels[i, :n] = torch.tensor(row["labels"])
        mask[i, :n] = 1
    return {
        "input_ids": ids.to(device),
        "attention_mask": mask.to(device),
        "labels": labels.to(device),
    }
