from contextlib import contextmanager
from dataclasses import dataclass

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)
from src.utils.file_utils import (
    get_last_modified_file,
    has_no_extension,
    list_subfolders,
)
from src.utils.logging_utils import finalize_loggers
from src.utils.restart_job_utils import get_attribute_from_metadata_file
from src.utils.utils import has_class_object_inside_list

command_line_logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class PipelineModules:
    cfg: DictConfig
    datamodule: LightningDataModule
    model: LightningModule
    # Lightningの引数名に合わせて複数形を使う。
    callbacks: list[Callback]
    loggers: list[Logger]
    trainer: Trainer


def update_cfg_with_most_recent_checkpoint_path(cfg: DictConfig) -> DictConfig:
    """再試行時や明示された探索先から、最新のチェックポイントを設定する。"""

    ckpt_path = cfg.get("ckpt_path", None)

    if (
        ckpt_path is not None
        and has_no_extension(ckpt_path)
        and cfg.get("should_retrieve_latest_ckpt_path", False)
    ):
        # 探索先の直下に時刻名のチェックポイントフォルダーが並ぶ構成を想定する。
        checkpoint_folders = list_subfolders(ckpt_path)
        if len(checkpoint_folders) > 0:
            # 時刻名の降順で最新の保存先を選ぶ。
            checkpoint_folders.sort(reverse=True)
            latest_ckpt_folder = checkpoint_folders[0]
            last_modified = get_last_modified_file(
                folder_path=latest_ckpt_folder, suffix="*.ckpt"
            )
            if len(last_modified) > 0:
                ckpt_path = last_modified
                command_line_logger.info(
                    f"最新のチェックポイントから再開します: {ckpt_path}"
                )

    # 再試行では、初期指定よりも同じ実験の直近チェックポイントを優先する。
    if (
        cfg.get("callbacks")
        and cfg.callbacks.get("model_checkpoint")
        and cfg.callbacks.get("restart_job")
        and get_attribute_from_metadata_file(
            f"{cfg.callbacks.restart_job.metadata_dir}/restart_metadata.json",
            "current_run",
        )
        > 0
    ):
        checkpoint_folder = cfg.callbacks.model_checkpoint.dirpath
        # 同じ保存先にあるckptファイルを更新日時で比較する。
        last_modified = get_last_modified_file(
            folder_path=checkpoint_folder, suffix="*.ckpt"
        )
        if len(last_modified) > 0:
            ckpt_path = last_modified
            command_line_logger.info(
                f"最新のチェックポイントから再開します: {ckpt_path}"
            )

    cfg.ckpt_path = ckpt_path
    return cfg


def initialize_pipeline_modules(
    cfg: DictConfig,
) -> PipelineModules:
    """設定からデータ・モデル・コールバック・ロガー・Trainerを構築する。"""
    # PyTorch・NumPy・標準randomのシードを揃える。0も有効な値として扱う。
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    command_line_logger.info(
        f"Instantiating datamodule <{cfg.data_loading.datamodule._target_}>"
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data_loading.datamodule
    )

    command_line_logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    command_line_logger.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    command_line_logger.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    command_line_logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    cfg = update_cfg_with_most_recent_checkpoint_path(cfg)

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        # 設定で明示しない限り、対応するコールバックがある機能だけを有効にする。
        enable_checkpointing=cfg.trainer.get(
            "enable_checkpointing",
            has_class_object_inside_list(callbacks, ModelCheckpoint),
        ),
        enable_model_summary=cfg.trainer.get(
            "enable_model_summary",
            has_class_object_inside_list(callbacks, ModelSummary),
        ),
        logger=loggers,
    )

    pipeline_modules = PipelineModules(
        cfg=cfg,
        datamodule=datamodule,
        model=model,
        callbacks=callbacks,
        loggers=loggers,
        trainer=trainer,
    )

    return pipeline_modules


@contextmanager
def pipeline_launcher(cfg: DictConfig):
    """実行資源を用意し、失敗時も元の例外を保ってロガーを終了する。"""
    modules = None
    failed = False
    try:
        modules = initialize_pipeline_modules(cfg)
        if modules.loggers:
            log_hyperparameters(cfg, modules.model, modules.trainer)
        yield modules
    except BaseException:
        failed = True
        raise
    finally:
        if modules is not None:
            try:
                finalize_loggers(modules.trainer)
            except Exception:
                # 後処理の失敗で学習・推論の元の例外を隠さない。
                if not failed:
                    raise
                command_line_logger.exception(
                    "例外発生後のロガー終了処理にも失敗しました"
                )
