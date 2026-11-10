"""Hydra設定またはPythonから推論を実行する。"""

# rootutilsによるパス設定後にローカルモジュールを読み込む。
# ruff: noqa: E402

from typing import Any

import hydra
import rootutils
from omegaconf import DictConfig

# 実行ディレクトリによらずローカルモジュールを参照できるようにする。
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras
from src.utils import custom_hydra_resolvers as _resolvers  # noqa: F401
from src.utils.launcher_utils import pipeline_launcher

command_line_logger = RankedLogger(__name__, rank_zero_only=True)


def inference(cfg: DictConfig) -> dict[str, Any]:
    """推論結果は設定済みコールバックで保存し、調査用オブジェクトを返す。"""
    with pipeline_launcher(cfg) as modules:
        ckpt_path = modules.cfg.get("ckpt_path")
        command_line_logger.info(f"推論を開始します。チェックポイント: {ckpt_path}")
        modules.trainer.predict(
            model=modules.model,
            datamodule=modules.datamodule,
            ckpt_path=ckpt_path,
            return_predictions=False,
        )
        return {
            "cfg": modules.cfg,
            "model": modules.model,
            "datamodule": modules.datamodule,
            "trainer": modules.trainer,
            "callbacks": modules.callbacks,
            "loggers": modules.loggers,
        }


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    """設定を表示して推論を実行する。"""
    extras(cfg)
    inference(cfg)


if __name__ == "__main__":
    main()
