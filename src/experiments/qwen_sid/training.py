"""共通の推薦系列で、alignmentの有無だけを変えて全重みを学習する。"""

import fcntl
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

from src.experiments.common import write_json
from src.experiments.qwen_sid.data import (
    encode_example,
    file_hash,
    load_recommendation_plan,
    read_examples,
    recommendation_example,
)
from src.experiments.qwen_sid.evaluation import evaluate_recommendations
from src.experiments.qwen_sid.model import load_model, parameter_digest
from src.experiments.qwen_sid.optimization import auxiliary_groups, backward_groups


def validate_step_config(config: dict):
    for key in ("max_steps", "eval_steps", "early_stopping_patience"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key}には正の整数を指定してください")
    log_steps = config.get("log_steps", 10)
    if type(log_steps) is not int or log_steps < 1:
        raise ValueError("log_stepsには正の整数を指定してください")


def validation_steps(config: dict):
    """定期間隔に加え、上限が間隔の倍数でなければ上限でも評価する。"""
    yield from range(config["eval_steps"], config["max_steps"], config["eval_steps"])
    yield config["max_steps"]


def save_checkpoint(model, tokenizer, optimizer, scheduler, out, progress, record):
    """全ファイルの保存後にlastを切り替え、bestと評価履歴も一緒に公開する。"""
    last = out / "last"
    previous = last.resolve()
    stage = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=out))
    pending = out / ".last.next"
    try:
        model.save_pretrained(stage)
        tokenizer.save_pretrained(stage)
        if record and record["best"]:
            model_files = list(stage.iterdir())
            best = stage / "best"
            best.mkdir()
            for path in model_files:
                os.link(path, best / path.name)
            write_json(best / "selection.json", record)
        elif (last / "best").exists():
            shutil.copytree(last / "best", stage / "best", copy_function=os.link)
        history = stage / "metrics.jsonl"
        if (last / history.name).exists():
            shutil.copyfile(last / history.name, history)
        if record:
            with history.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            (out / ".validation_predictions.next").replace(
                stage / "validation_predictions.jsonl"
            )
        elif (last / "validation_predictions.jsonl").exists():
            os.link(
                last / "validation_predictions.jsonl",
                stage / "validation_predictions.jsonl",
            )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
            stage / "training_state.pt",
        )
        write_json(stage / "progress.json", progress)
        for name in ("best", "metrics.jsonl", "validation_predictions.jsonl"):
            alias = out / name
            if not alias.is_symlink():
                alias.symlink_to(Path("last") / name)
        pending.unlink(missing_ok=True)
        pending.symlink_to(stage.name, target_is_directory=True)
        pending.replace(last)
    finally:
        pending.unlink(missing_ok=True)
        if last.resolve() != stage:
            shutil.rmtree(stage, ignore_errors=True)
    if previous.parent == out and previous.name.startswith(".checkpoint-"):
        shutil.rmtree(previous, ignore_errors=True)


def load_training_data(tokenizer, config):
    """推薦系列は共通planから読む。補助タスクは独立した候補集合にする。"""
    data = Path(config["prepared_dir"])
    manifest = json.loads((data / "manifest.json").read_text())
    for name, digest in manifest["prepared_sha256"].items():
        if file_hash(data / name) != digest:
            raise ValueError(f"準備済みデータが変更されています: {name}")
    plan = load_recommendation_plan(config)
    alignment = {task: [] for task in ("text_to_sid", "sid_to_text")}
    for row in read_examples(data / "training.jsonl"):
        if row["task"] in alignment:
            alignment[row["task"]].append(
                encode_example(tokenizer, row, config["max_length"])
            )
    sids = json.loads((data / "sids.json").read_text())
    validation = [
        r
        for r in read_examples(data / "evaluation.jsonl")
        if r["task"] == "recommendation"
    ]
    return plan, alignment, sids, validation


def train(config: dict, resume=False, until_step=None, *, stop_on_plateau=True):
    """推薦128例ごとに1更新。Bのみ各方向128例の補助lossを加える。

    loss_A = mean(rec)
    loss_B = mean(rec) + lambda/2 * (mean(text_to_sid) + mean(sid_to_text))
    until_stepで中断・再開しても評価間隔とLR scheduleはmax_steps基準で共通。
    """
    validate_step_config(config)
    if until_step is not None and (
        type(until_step) is not int or not 1 <= until_step <= config["max_steps"]
    ):
        raise ValueError("until_stepは1からmax_stepsまでの整数で指定してください")

    out = Path(config["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(config, out, resume, until_step, stop_on_plateau)


def _train(config, out, resume, until_step, stop_on_plateau):
    import transformers

    experiment_path = out / "experiment.json"
    if experiment_path.exists() and not resume:
        raise FileExistsError(f"既存実験は上書きしません: {out}")
    previous = json.loads(experiment_path.read_text()) if resume else None
    if previous and previous["config"] != config:
        raise ValueError("再開時の設定が変更されています")
    if previous and previous["stop_on_plateau"] != stop_on_plateau:
        raise ValueError("再開時のearly stopping方式が変更されています")

    last = out / "last"
    saved = json.loads((last / "progress.json").read_text()) if last.exists() else None
    if saved and not previous:
        raise ValueError("再開用checkpointに対応するexperiment.jsonがありません")
    progress = saved or {
        "step": 0,
        "epoch_index": 0,
        "next_example": 0,
        "best_validation_recall_at_5": -1.0,
        "best_step": None,
        "evaluations_without_improvement": 0,
        "cumulative_recommendations": 0,
        "interval_examples": 0,
        "interval_loss_sums": {},
        "interval_training_seconds": 0.0,
        "stop_reason": None,
    }
    stop_step = until_step or config["max_steps"]
    prepared = Path(config["prepared_dir"])
    plan_digest = file_hash(prepared / "recommendation_plan.json")
    if previous and previous["recommendation_plan_sha256"] != plan_digest:
        raise ValueError("再開時の共通推薦planが変更されています")
    for folder in out.glob(".checkpoint-*"):
        if folder != last.resolve():
            shutil.rmtree(folder, ignore_errors=True)
    if progress["stop_reason"] or progress["step"] >= stop_step:
        return progress

    manifest = json.loads((prepared / "manifest.json").read_text())
    path = str(last) if saved else config["model_path"]
    model, tokenizer, _ = load_model(path, manifest["sid_tokens"], config["seed"])
    if not config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_disable()
    initial_digest = (
        previous["initial_parameter_sha256"] if saved else parameter_digest(model)
    )
    if previous and previous["initial_parameter_sha256"] != initial_digest:
        raise ValueError("再開時の初期パラメータが変更されています")
    training_data = load_training_data(tokenizer, config)
    effective_batch = config["micro_batch_size"] * config["gradient_accumulation_steps"]
    total_steps = config["max_steps"]
    model.to("cuda").train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fused=True,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * config["warmup_ratio"]), total_steps
    )
    if saved:
        state = torch.load(
            out / "last" / "training_state.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        del state
    metadata = {
        "config": config,
        "initial_parameter_sha256": initial_digest,
        "recommendation_plan_sha256": plan_digest,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "effective_recommendation_batch": effective_batch,
        "planned_updates": total_steps,
        "selection": "validation recall@5",
        "stop_on_plateau": stop_on_plateau,
        "model_revision": Path(
            config["model_path"], ".cache/huggingface/download/config.json.metadata"
        )
        .read_text()
        .splitlines()[0],
    }
    write_json(experiment_path, metadata)
    try:
        return train_updates(
            model,
            tokenizer,
            optimizer,
            scheduler,
            training_data,
            config,
            out,
            progress,
            stop_step,
            stop_on_plateau,
        )
    except BaseException as error:
        write_json(
            out / "status.json",
            {"state": "failed", "step": progress["step"], "error": repr(error)},
        )
        raise


def update_parameters(model, tokenizer, optimizer, scheduler, groups, config):
    """全タスクの勾配を蓄積し、パラメータと学習率を1回更新する。"""
    optimizer.zero_grad(set_to_none=True)
    losses = backward_groups(model, tokenizer, groups, config)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
    if not torch.isfinite(norm):
        raise ValueError("非有限gradient norm")
    optimizer.step()
    scheduler.step()
    return losses


def validate_step(model, tokenizer, validation, sids, config, output, progress):
    """全validationでRecall@5を測り、bestと改善なし回数を更新する。"""
    started = time.monotonic()
    metrics = evaluate_recommendations(
        model, tokenizer, validation, sids, config, output
    )
    score = metrics["recall@5"]
    if not math.isfinite(score):
        raise ValueError("validation Recall@5が非有限です")
    improved = score > progress["best_validation_recall_at_5"]
    if improved:
        progress["best_validation_recall_at_5"] = score
        progress["best_step"] = progress["step"]
        progress["evaluations_without_improvement"] = 0
    else:
        progress["evaluations_without_improvement"] += 1
    return {
        "validation": metrics,
        "best": improved,
        "best_step": progress["best_step"],
        "evaluations_without_improvement": progress["evaluations_without_improvement"],
        "validation_seconds": time.monotonic() - started,
    }


def train_updates(
    model,
    tokenizer,
    optimizer,
    scheduler,
    training_data,
    config,
    out,
    progress,
    stop_step,
    stop_on_plateau,
):
    """推薦planの途中から再開し、optimizer更新直後だけ停止を判定する。"""
    plan, alignment, sids, validation = training_data
    effective_batch = config["micro_batch_size"] * config["gradient_accumulation_steps"]
    total_steps = config["max_steps"]
    patience = config["early_stopping_patience"]
    first_epoch = progress["epoch_index"]
    next_example = progress["next_example"]
    started = time.monotonic()
    interval_started = started
    sums = Counter(progress["interval_loss_sums"])
    interval_examples = progress["interval_examples"]
    interval_seconds = progress["interval_training_seconds"]
    for epoch in range(first_epoch, len(plan["epochs"])):
        epoch_plan = plan["epochs"][epoch]
        source = Path(epoch_plan["path"])
        if file_hash(source) != epoch_plan["sha256"]:
            raise ValueError("共通推薦例が変更されています")
        sequences = read_examples(source)
        if len(sequences) != epoch_plan["examples"]:
            raise ValueError("共通推薦planの件数が系列ファイルと異なります")
        print(f"epoch={epoch + 1} recommendation_examples={len(sequences)}", flush=True)
        for start in range(next_example, len(sequences), effective_batch):
            rows = [
                encode_example(
                    tokenizer,
                    recommendation_example(row, sids, config["history_items"]),
                    config["max_length"],
                )
                for row in sequences[start : start + effective_batch]
            ]
            groups = [("recommendation", rows, 1.0)] + auxiliary_groups(
                alignment,
                len(rows),
                config["seed"],
                progress["step"],
                config["alignment_weight"],
            )
            losses = update_parameters(
                model, tokenizer, optimizer, scheduler, groups, config
            )
            progress["step"] += 1
            step = progress["step"]
            used = start + len(rows)
            progress["epoch_index"] = epoch + (used == len(sequences))
            progress["next_example"] = 0 if used == len(sequences) else used
            progress["cumulative_recommendations"] += len(rows)
            interval_examples += len(rows)
            for task, loss in losses.items():
                sums[task] += loss * len(rows)
            position = {
                "epoch": epoch + 1,
                "step": step,
                "planned_updates": total_steps,
                "cumulative_recommendations": progress["cumulative_recommendations"],
                "epoch_recommendations_used": used,
                "epoch_recommendations_total": len(sequences),
            }
            if step % config.get("log_steps", 10) == 0:
                status = {
                    **position,
                    "state": "training",
                    "train_loss": {
                        task: loss / interval_examples for task, loss in sums.items()
                    },
                    "elapsed_seconds": time.monotonic() - started,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                }
                write_json(out / "status.json", status)
                print(json.dumps(status), flush=True)

            should_evaluate = step % config["eval_steps"] == 0 or step == total_steps
            if not should_evaluate and step != stop_step:
                continue
            optimizer.zero_grad(set_to_none=True)
            interval_seconds += time.monotonic() - interval_started
            record = None
            if should_evaluate:
                write_json(out / "status.json", {**position, "state": "validating"})
                selection = validate_step(
                    model,
                    tokenizer,
                    validation,
                    sids,
                    config,
                    out / ".validation_predictions.next",
                    progress,
                )
                record = {
                    **position,
                    **selection,
                    "training_seconds": interval_seconds,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "recommendation_examples": interval_examples,
                    "train_loss": {
                        task: loss / interval_examples for task, loss in sums.items()
                    },
                }
                sums.clear()
                interval_examples = 0
                interval_seconds = 0.0
            if step == total_steps:
                progress["stop_reason"] = "max_steps"
            elif (
                should_evaluate
                and stop_on_plateau
                and progress["evaluations_without_improvement"] >= patience
            ):
                progress["stop_reason"] = "early_stopping"
            progress.update(
                interval_examples=interval_examples,
                interval_loss_sums=dict(sums),
                interval_training_seconds=interval_seconds,
            )
            save_checkpoint(
                model, tokenizer, optimizer, scheduler, out, progress, record
            )
            if record:
                print(json.dumps(record), flush=True)
            if progress["stop_reason"] or step == stop_step:
                state = "trained" if progress["stop_reason"] else "step_complete"
                write_json(
                    out / "status.json",
                    {**position, **progress, "state": state},
                )
                return progress
            model.train()
            interval_started = time.monotonic()
        next_example = 0
    raise ValueError("共通推薦planを学習上限まで読み取れませんでした")
