"""推薦lossとalignment補助lossの勾配を、小さいバッチで加算する。"""

import random

import torch

from src.experiments.qwen_sid.data import collate_examples
from src.experiments.qwen_sid.model import response_losses


def auxiliary_groups(alignment: dict, count: int, seed: int, step: int, weight: float):
    """推薦用乱数と独立に、補助タスクの例を再現可能に抽出する。"""
    if weight == 0:
        return []
    rng = random.Random(seed + 1_000_000 + step)
    return [
        (task, rng.choices(alignment[task], k=count), weight / 2)
        for task in ("text_to_sid", "sid_to_text")
    ]


def micro_batches(rows, config):
    """長い回答だけのバッチでもOOMを避け、例数・lossの重みは変えない。"""
    batch, longest, responses = [], 0, 0
    for row in rows:
        n = len(row["input_ids"])
        answer_length = sum(label != -100 for label in row["labels"])
        exceeds = (
            len(batch) >= config["micro_batch_size"]
            or max(longest, n) * (len(batch) + 1)
            > config.get("max_padded_tokens_per_micro_batch", float("inf"))
            or responses + answer_length
            > config.get("max_response_tokens_per_micro_batch", float("inf"))
        )
        if batch and exceeds:
            yield batch
            batch, longest, responses = [], 0, 0
        batch.append(row)
        longest = max(longest, n)
        responses += answer_length
    if batch:
        yield batch


def backward_groups(model, tokenizer, groups, config):
    """各タスクを独立に平均して勾配加算する。GPUにはmicro batchだけを置く。"""
    task_losses = {}
    for task, rows, weight in groups:
        total = 0.0
        for selected in micro_batches(rows, config):
            batch = collate_examples(
                selected,
                tokenizer.pad_token_id,
                "cuda",
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = response_losses(model, batch)
                loss = losses.sum() * (weight / len(rows))
            if not torch.isfinite(loss):
                raise ValueError("非有限loss")
            loss.backward()
            total += float(losses.detach().sum())
        task_losses[task] = total / len(rows)
    return task_losses
