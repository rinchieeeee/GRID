"""Pythonからの分析実行と、初期化失敗時の例外を確認する。"""

import importlib
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from src.utils import launcher_utils


def test_training_returns_independent_metrics_and_objects(monkeypatch):
    module = importlib.import_module("src.train")
    trainer = Mock()
    trainer.callback_metrics = {"train/loss": 1.0}
    trainer.checkpoint_callback.best_model_path = "best.ckpt"

    def evaluate(**kwargs):
        trainer.callback_metrics.clear()
        trainer.callback_metrics["test/recall@5"] = 0.2

    trainer.test.side_effect = evaluate
    cfg = OmegaConf.create({"train": True, "test": True, "ckpt_path": None})
    modules = SimpleNamespace(
        cfg=cfg,
        trainer=trainer,
        model=object(),
        datamodule=object(),
        callbacks=[],
        loggers=[],
    )

    @contextmanager
    def launch(config):
        yield modules

    monkeypatch.setattr(module, "pipeline_launcher", launch)
    metrics, objects = module.train(cfg)
    assert metrics == {"train/loss": 1.0, "test/recall@5": 0.2}
    assert objects["model"] is modules.model
    assert trainer.test.call_args.kwargs["ckpt_path"] == "best.ckpt"


def test_initialization_error_is_preserved(monkeypatch):
    def fail(cfg):
        raise ValueError("初期化時の原因")

    monkeypatch.setattr(launcher_utils, "initialize_pipeline_modules", fail)
    with pytest.raises(ValueError, match="初期化時の原因"):
        with launcher_utils.pipeline_launcher(OmegaConf.create({})):
            pytest.fail("初期化に失敗したので実行されない")


def test_cleanup_error_does_not_hide_training_error(monkeypatch):
    modules = SimpleNamespace(loggers=[], trainer=object())
    monkeypatch.setattr(
        launcher_utils, "initialize_pipeline_modules", lambda cfg: modules
    )
    monkeypatch.setattr(
        launcher_utils,
        "finalize_loggers",
        Mock(side_effect=RuntimeError("終了時のエラー")),
    )
    with pytest.raises(ValueError, match="学習時の原因"):
        with launcher_utils.pipeline_launcher(OmegaConf.create({})):
            raise ValueError("学習時の原因")


def test_seed_zero_is_applied(monkeypatch):
    seed = Mock()
    monkeypatch.setattr(launcher_utils.L, "seed_everything", seed)
    monkeypatch.setattr(
        launcher_utils.hydra.utils, "instantiate", Mock(return_value=object())
    )
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "ckpt_path": None,
            "data_loading": {"datamodule": {"_target_": "dummy.Data"}},
            "model": {"_target_": "dummy.Model"},
            "trainer": {"_target_": "dummy.Trainer"},
        }
    )
    launcher_utils.initialize_pipeline_modules(cfg)
    seed.assert_called_once_with(0, workers=True)


def test_script_imports_do_not_execute_jobs(monkeypatch):
    import os
    import subprocess

    from huggingface_hub import snapshot_download

    download = Mock(spec=snapshot_download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    process = Mock(side_effect=AssertionError("import時には実行しない"))
    monkeypatch.setattr(subprocess, "Popen", process)
    cwd, environment = os.getcwd(), dict(os.environ)
    for name in (
        "run_toys_pipeline",
        "start_toys_training",
        "summarize_toys_run",
        "inspect_amazon_data",
        "download_embedding_model",
        "runtime_probe",
    ):
        importlib.import_module(f"scripts.{name}")
    assert os.getcwd() == cwd
    assert dict(os.environ) == environment
    download.assert_not_called()
    process.assert_not_called()


def test_python_config_resolves_without_hydra_runtime(tmp_path):
    from hydra import compose, initialize_config_dir

    from src.experiments.common import ROOT

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=tiger_train_flat",
                "+debug=tiger",
                "data_dir=data/amazon_data/toys",
                "semantic_id_path=/tmp/sid.pt",
                "num_hierarchies=4",
                f"paths.output_dir={tmp_path}",
                f"paths.profile_dir={tmp_path}/profile_output",
                f"paths.work_dir={ROOT}",
            ],
        )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["paths"]["output_dir"] == str(tmp_path)
