"""Hydra設定またはPythonから学習とテストを実行する。"""

# rootutilsによるパス設定後にローカルモジュールを読み込む。
# ruff: noqa: E402

from typing import Any

import hydra
import rootutils
import torch
from omegaconf import DictConfig

# ローカルモジュールの読み込み前にパスと環境変数を設定する。
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras
from src.utils import custom_hydra_resolvers as _resolvers  # noqa: F401
from src.utils.launcher_utils import pipeline_launcher
from src.utils.restart_job import LocalJobLauncher

command_line_logger = RankedLogger(__name__, rank_zero_only=True)


def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """学習・評価指標と、調査に使うモデル・ローダー・Trainerを返す。"""
    torch.set_float32_matmul_precision("medium")
    with pipeline_launcher(cfg) as modules:
        if cfg.get("train"):
            command_line_logger.info("学習を開始します")
            modules.trainer.fit(
                model=modules.model,
                datamodule=modules.datamodule,
                ckpt_path=modules.cfg.get("ckpt_path"),
            )
        # callback_metricsはテスト時に更新されるため、学習直後にコピーする。
        train_metrics = dict(modules.trainer.callback_metrics)
        test_metrics = {}
        if cfg.get("test"):
            callback = getattr(modules.trainer, "checkpoint_callback", None)
            ckpt_path = getattr(callback, "best_model_path", None) or None
            if ckpt_path is None:
                command_line_logger.warning(
                    "最良チェックポイントがないため現在の重みで評価します"
                )
            modules.trainer.test(
                model=modules.model, datamodule=modules.datamodule, ckpt_path=ckpt_path
            )
            test_metrics = dict(modules.trainer.callback_metrics)
            command_line_logger.info(f"評価チェックポイント: {ckpt_path}")
        metrics = {**train_metrics, **test_metrics}
        command_line_logger.info(f"指標: {metrics}")
        objects = {
            "cfg": modules.cfg,
            "model": modules.model,
            "datamodule": modules.datamodule,
            "trainer": modules.trainer,
            "callbacks": modules.callbacks,
            "loggers": modules.loggers,
        }
        return metrics, objects


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """CLIでは既存の再試行付き実行を維持する。"""
    extras(cfg)
    LocalJobLauncher(cfg=cfg).launch(function_to_run=train)


if __name__ == "__main__":
    main()
