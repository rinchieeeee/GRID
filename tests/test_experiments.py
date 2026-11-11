"""実験条件の分離、再開、検証、ログ集計の回帰を確認する。"""

import gzip
import importlib
import struct
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.experiments.common import ROOT, read_json, write_json
from src.experiments.config import STAGES, ExperimentConfig
from src.experiments.inspection import count_file
from src.experiments.pipeline import (
    ExperimentRunner,
    Stage,
    build_command,
    build_stages,
)
from src.experiments.reporting import collect_metrics, render_report, summarize_run
from src.experiments.validation import validate_embeddings, validate_semantic_ids


@pytest.mark.parametrize(
    "field,value",
    [
        ("rkmeans_epochs", 0),
        ("embedding_dim", -1),
        ("seed", "42"),
        ("overrides", {"typo": []}),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        replace(ExperimentConfig(), **{field: value})


def test_config_round_trip(tmp_path):
    config = replace(ExperimentConfig(), output_dir=str(tmp_path), seed=0)
    path = tmp_path / "config.json"
    write_json(path, config.to_dict())
    assert ExperimentConfig.load(path) == config


def test_commands_compose_with_existing_hydra_configs(tmp_path):
    importlib.import_module("src.utils.custom_hydra_resolvers")
    config = replace(ExperimentConfig(), output_dir=str(tmp_path), seed=0)
    for stage in build_stages(config, 11924):
        with initialize_config_dir(
            version_base="1.3", config_dir=str(ROOT / "configs")
        ):
            cfg = compose(
                config_name="train" if stage.module == "src.train" else "inference",
                overrides=build_command(config, stage)[3:],
            )
        assert cfg.seed == 0
        assert cfg.trainer.devices == 1
        if stage.name == "02_rkmeans":
            assert cfg.trainer.max_steps == 1080
        if stage.name == "04_tiger":
            assert cfg.num_hierarchies == 4
            assert cfg.trainer.max_epochs == 10


def test_debug_config_composes():
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=tiger_train_flat",
                "+debug=tiger",
                "data_dir=/tmp/data",
                "semantic_id_path=/tmp/sid.pt",
                "num_hierarchies=4",
            ],
        )
    assert cfg.should_skip_retry
    assert cfg.callbacks.restart_job is None
    assert cfg.trainer.max_steps == 2
    assert cfg.data_loading.train_dataloader_config.dataloader.num_workers == 0


def test_stage_override_wins_once():
    config = ExperimentConfig(overrides={STAGES[3]: ["trainer.max_epochs=3"]})
    command = build_command(config, build_stages(config)[3])
    assert "++trainer.max_epochs=3" in command
    assert "++trainer.max_epochs=10" not in command


def test_runner_records_failure_and_does_not_mark_complete(tmp_path, monkeypatch):
    runner = ExperimentRunner(replace(ExperimentConfig(), output_dir=str(tmp_path)))
    stage = Stage(
        STAGES[0],
        "src.inference",
        ("experiment=sem_embeds_inference_flat",),
        tmp_path / STAGES[0] / "output.pt",
    )

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(2, args[0])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_stage(stage)
    assert (
        read_json(tmp_path / "status.json")["stages"][stage.name]["state"] == "failed"
    )
    assert not (tmp_path / stage.name / ".complete").exists()


def test_runner_reuses_identical_stage_only(tmp_path, monkeypatch):
    config = replace(ExperimentConfig(), output_dir=str(tmp_path))
    runner = ExperimentRunner(config)
    stage = Stage(
        STAGES[0],
        "src.inference",
        ("experiment=sem_embeds_inference_flat",),
        tmp_path / STAGES[0] / "output.pt",
    )
    calls = []

    def complete(*args, **kwargs):
        calls.append(args)
        stage.artifact.touch()

    monkeypatch.setattr(subprocess, "run", complete)
    runner.run_stage(stage)
    runner.run_stage(stage)
    assert len(calls) == 1
    changed = ExperimentRunner(replace(config, precision="32-true"))
    with pytest.raises(ValueError, match="不一致"):
        changed.run_stage(stage)


def test_pipeline_generates_checks_and_resumes(tmp_path, monkeypatch):
    config = replace(ExperimentConfig(), output_dir=str(tmp_path), embedding_dim=2)
    calls = []

    def complete(command, **kwargs):
        directory = Path(
            next(x.split("=", 1)[1] for x in command if x.startswith("hydra.run.dir="))
        )
        calls.append(directory.name)
        if directory.name == STAGES[0]:
            (directory / "pickle").mkdir()
            torch.save(
                torch.ones(3, 2), directory / "pickle/merged_predictions_tensor.pt"
            )
        elif directory.name == STAGES[2]:
            (directory / "pickle").mkdir()
            torch.save(
                torch.tensor([[0, 1, 2]] * 4),
                directory / "pickle/merged_predictions_tensor.pt",
            )
        else:
            (directory / "checkpoints").mkdir()
            (directory / "checkpoints/last.ckpt").touch()

    monkeypatch.setattr(subprocess, "run", complete)
    assert ExperimentRunner(config).run()["state"] == "completed"
    assert ExperimentRunner(config).run()["state"] == "completed"
    assert calls == list(STAGES)
    assert read_json(tmp_path / "embedding_checks.json")["shape"] == [3, 2]
    with pytest.raises(ValueError, match="異なる実験設定"):
        ExperimentRunner(replace(config, seed=12)).run()


def test_invalid_embeddings_and_duplicate_ids(tmp_path):
    emb = tmp_path / "emb.pt"
    torch.save(torch.tensor([[float("nan"), 0.0]]), emb)
    with pytest.raises(ValueError, match="NaN"):
        validate_embeddings(emb, 2)
    ids = tmp_path / "ids.pt"
    torch.save(torch.zeros(4, 3, dtype=torch.long), ids)
    with pytest.raises(ValueError, match="重複"):
        validate_semantic_ids(ids, 4, 3)


def test_record_count_and_truncation(tmp_path):
    path = tmp_path / "data.tfrecord.gz"
    record = struct.pack("<Q", 3) + b"0000" + b"abc" + b"0000"
    with gzip.open(path, "wb") as stream:
        stream.write(record * 2)
    assert count_file(path) == 2
    with gzip.open(path, "wb") as stream:
        stream.write(record[:-1])
    with pytest.raises(ValueError, match="途中"):
        count_file(path)


def test_latest_metrics_and_missing_values(tmp_path):
    for version, value in ((2, 0.2), (10, 0.1)):
        directory = tmp_path / f"04_tiger/csv/version_{version}"
        directory.mkdir(parents=True)
        (directory / "metrics.csv").write_text(
            f"epoch,step,val/recall@5,val/loss\n0,10,{value},\n", encoding="utf-8"
        )
    summary = summarize_run(tmp_path)
    assert summary["best_validation"]["val/recall@5"] == 0.1
    assert "—" in render_report(summary)
    assert "val/loss" not in summary["validation_history"][0]
    assert (
        summarize_run(tmp_path, "version_2")["best_validation"]["val/recall@5"] == 0.2
    )
    assert not collect_metrics([{"val/loss": "nan"}], "val/")


def test_retry_archives_partial_predictions(tmp_path, monkeypatch):
    config = replace(ExperimentConfig(), output_dir=str(tmp_path))
    stage = Stage(
        STAGES[0],
        "src.inference",
        ("experiment=sem_embeds_inference_flat",),
        tmp_path / STAGES[0] / "output.pt",
    )
    partial = tmp_path / STAGES[0] / "pickle/partial.pkl"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"previous attempt")

    def complete(*args, **kwargs):
        assert not partial.exists()
        stage.artifact.touch()

    monkeypatch.setattr(subprocess, "run", complete)
    ExperimentRunner(config).run_stage(stage)
    archived = list((tmp_path / "_attempts").glob("*/pickle/partial.pkl"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == b"previous attempt"
