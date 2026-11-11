"""本学習前に、実データの長い例で全重み更新とGPUメモリを確認する。"""

import gc
import time
from pathlib import Path

import torch

from src.experiments.common import write_json
from src.experiments.qwen_sid.data import encode_example, read_examples
from src.experiments.qwen_sid.model import load_model, parameter_digest
from src.experiments.qwen_sid.optimization import backward_groups


def run_probe(config: dict, manifest: dict) -> dict:
    model, tokenizer, original_count = load_model(
        config["model_path"], manifest["sid_tokens"], config["seed"]
    )
    initial_digest = parameter_digest(model)
    if not config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_disable()
    encoded = [
        encode_example(tokenizer, row, config["max_length"])
        for row in read_examples(Path(config["prepared_dir"]) / "training.jsonl")
    ]
    tasks = ("recommendation", "text_to_sid", "sid_to_text")
    longest = {
        task: max(
            (r for r in encoded if r["task"] == task), key=lambda r: len(r["input_ids"])
        )
        for task in tasks
    }
    size = config["micro_batch_size"]
    groups = [
        (task, [longest[task]] * size, 1.0 if task == "recommendation" else 0.5)
        for task in tasks
    ]
    model.to("cuda").train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fused=True,
    )
    embed = model.get_input_embeddings().weight
    old_id = tokenizer.encode("Title", add_special_tokens=False)[0]
    sid_id = tokenizer.convert_tokens_to_ids(manifest["sid_tokens"][0])
    layer = model.model.layers[0].self_attn.q_proj.weight
    before = [
        embed[old_id].detach().clone(),
        embed[sid_id].detach().clone(),
        layer.detach().clone(),
    ]
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    losses, all_have_grad = [], True
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        losses.append(backward_groups(model, tokenizer, groups, config))
        all_have_grad &= all(p.grad is not None for p in model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["max_grad_norm"]
        )
        if not torch.isfinite(norm):
            raise ValueError("非有限の勾配")
        optimizer.step()
    torch.cuda.synchronize()
    updated = [
        not torch.equal(a, b)
        for a, b in zip(before, [embed[old_id], embed[sid_id], layer], strict=True)
    ]
    result = {
        "passed": all_have_grad and all(updated),
        "mode": "full_parameters",
        "initial_parameter_sha256": initial_digest,
        "frozen_parameters": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "all_parameter_tensors_received_gradients": all_have_grad,
        "old_embedding_updated": updated[0],
        "sid_embedding_updated": updated[1],
        "transformer_updated": updated[2],
        "losses": losses,
        "optimizer_steps": 3,
        "micro_batch_size": size,
        "max_actual_length": max(len(r["input_ids"]) for r in encoded),
        "original_tokenizer_length": original_count,
        "tokenizer_length": len(tokenizer),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "elapsed_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(),
        "config": config,
    }
    write_json(Path(config["output_dir"]) / "probe.json", result)
    if not result["passed"]:
        raise ValueError("全重み更新の確認に失敗しました")
    del model, optimizer, groups
    gc.collect()
    torch.cuda.empty_cache()
    return result
