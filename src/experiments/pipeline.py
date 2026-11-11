"""実験を4ステージに分割し、コマンド生成と実行を分離する。"""

import argparse
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from src.experiments.common import (
    ROOT,
    experiment_environment,
    read_json,
    write_json,
)
from src.experiments.config import STAGES, ExperimentConfig
from src.experiments.validation import validate_embeddings, validate_semantic_ids


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    overrides: tuple[str, ...]
    artifact: Path


def loader_overrides(prefix: str) -> list[str]:
    """デバッグ時にデータ処理を同一プロセスで追跡できるようにする。"""
    return [
        f"{prefix}.num_workers=0",
        f"{prefix}.timeout=0",
        f"{prefix}.persistent_workers=false",
    ]


def build_stages(
    config: ExperimentConfig, item_count: int | None = None
) -> list[Stage]:
    """副作用なしで各ステージの設定を構築する。"""
    out = config.output_path
    emb = out / STAGES[0] / "pickle/merged_predictions_tensor.pt"
    ckpt = out / STAGES[1] / "checkpoints/last.ckpt"
    sid = out / STAGES[2] / "pickle/merged_predictions_tensor.pt"
    data = f"data_dir={config.data_path}"
    quantization = [
        data,
        f"embedding_path={emb}",
        f"embedding_dim={config.embedding_dim}",
        f"num_hierarchies={config.num_hierarchies}",
        f"codebook_width={config.codebook_width}",
    ]
    train_loader = "data_loading.datamodule.train_dataloader_config"
    predict_loader = "data_loading.datamodule.predict_dataloader_config"
    # 有限ローダーのエポック数で終了する。件数が分かれば元と同じ更新上限も記録する。
    steps = (
        math.ceil(item_count / config.rkmeans_batch_size) * config.rkmeans_epochs
        if item_count is not None
        else -1
    )
    rkmeans = [
        "experiment=rkmeans_train_flat",
        *quantization,
        f"trainer.max_epochs={config.rkmeans_epochs}",
        f"trainer.max_steps={steps}",
        "trainer.limit_val_batches=0",
        "trainer.log_every_n_steps=1",
        "model.verbose=true",
        f"{train_loader}.batch_size_per_device={config.rkmeans_batch_size}",
        f"{train_loader}.drop_last=false",
        f"{train_loader}.dataset_class._target_="
        "src.data.loading.components.dataloading.CachedFiniteSequenceIterable",
        *loader_overrides(train_loader),
        "callbacks.model_checkpoint.every_n_train_steps=null",
        "callbacks.model_checkpoint.every_n_epochs=1",
        "callbacks.model_checkpoint.save_on_train_epoch_end=true",
        "callbacks.model_checkpoint.save_last=true",
    ]
    tiger = [
        "experiment=tiger_train_flat",
        data,
        f"semantic_id_path={sid}",
        f"num_hierarchies={config.num_hierarchies + 1}",
        f"trainer.max_epochs={config.recommendation_epochs}",
        "trainer.max_steps=-1",
        "trainer.val_check_interval=1.0",
        "trainer.log_every_n_steps=10",
        "callbacks.early_stopping=null",
        "callbacks.model_checkpoint.save_last=true",
        "data_loading.train_dataloader_config.dataloader.collate_fn.max_batch_size="
        f"{config.augmentation_batch_size}",
        "data_loading.train_dataloader_config.dataloader.drop_last=false",
        "data_loading.train_dataloader_config.dataloader.assign_all_files_per_worker=false",
        "data_loading.train_dataloader_config.dataloader.dataset_class._target_="
        "src.data.loading.components.dataloading.FiniteSequenceIterable",
    ]
    for split in ("train", "val", "test"):
        prefix = f"data_loading.{split}_dataloader_config.dataloader"
        tiger.extend(loader_overrides(prefix))
        if split != "train":
            tiger.append(
                f"{prefix}.batch_size_per_device={config.evaluation_batch_size}"
            )
    return [
        Stage(
            STAGES[0],
            "src.inference",
            tuple(
                [
                    "experiment=sem_embeds_inference_flat",
                    data,
                    f"embedding_model={config.embedding_model}",
                    *loader_overrides(predict_loader),
                ]
            ),
            emb,
        ),
        Stage(STAGES[1], "src.train", tuple(rkmeans), ckpt),
        Stage(
            STAGES[2],
            "src.inference",
            tuple(
                [
                    "experiment=rkmeans_inference_flat",
                    *quantization,
                    f"ckpt_path={ckpt}",
                    "callbacks.bq_writer=null",
                    *loader_overrides(predict_loader),
                ]
            ),
            sid,
        ),
        Stage(
            STAGES[3],
            "src.train",
            tuple(tiger),
            out / STAGES[3] / "checkpoints/last.ckpt",
        ),
    ]


def build_command(config: ExperimentConfig, stage: Stage) -> list[str]:
    """最後の上書きを優先し、同じキーをHydraに重複して渡さない。"""
    overrides = [
        *stage.overrides,
        "trainer.devices=1",
        "trainer.strategy=auto",
        f"trainer.precision={config.precision}",
        "extras.print_config=false",
    ]
    if stage.module == "src.train":
        overrides.append("should_skip_retry=true")
    if config.seed is not None:
        overrides.append(f"seed={config.seed}")
    overrides.extend(config.overrides.get(stage.name, []))
    overrides.append(f"hydra.run.dir={config.output_path / stage.name}")
    values = {}
    for override in overrides:
        key, value = override.split("=", 1)
        values[key.lstrip("+")] = value
    return [
        sys.executable,
        "-m",
        stage.module,
        *[
            f"{key}={value}"
            if key == "experiment" or key.startswith("hydra.")
            else f"++{key}={value}"
            for key, value in values.items()
        ],
    ]


class ExperimentRunner:
    """状態をインスタンス内に保持し、importやコンストラクタでは実行しない。"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output = config.output_path
        self.status = read_json(self.output / "status.json", {"stages": {}})

    def save_status(self) -> None:
        write_json(self.output / "status.json", self.status)

    def run_stage(self, stage: Stage) -> Path:
        """同一条件で正常終了したステージだけを再利用する。"""
        self.status["current_stage"] = stage.name
        self.status["stages"].setdefault(stage.name, {})
        command = build_command(self.config, stage)
        directory = self.output / stage.name
        marker = directory / ".complete"
        if marker.exists():
            previous = read_json(directory / "command.json")
            if previous != command or not stage.artifact.is_file():
                raise ValueError(
                    f"{stage.name}: 保存済み条件または成果物が不一致です。"
                    "別のoutput_dirを指定してください。"
                )
            self.status["current_stage"] = stage.name
            self.status["stages"].setdefault(stage.name, {})["state"] = "completed"
            self.save_status()
            print(f"完了済み: {stage.name}", flush=True)
            return stage.artifact
        # 途中の予測や再開メタデータを次の試行に混ぜず、調査用に丸ごと保存する。
        if directory.exists() and any(directory.iterdir()):
            attempts = self.output / "_attempts"
            attempts.mkdir(parents=True, exist_ok=True)
            directory.replace(attempts / f"{stage.name}.{time.time_ns()}")
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "console.log"
        write_json(directory / "command.json", command)
        record = {"state": "running", "started_at": time.time(), "command": command}
        self.status.update(current_stage=stage.name)
        self.status["stages"][stage.name] = record
        self.save_status()
        print(f"開始: {stage.name}（ログ: {log_path}）", flush=True)
        started = time.monotonic()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=experiment_environment(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not stage.artifact.is_file():
                raise FileNotFoundError(f"成果物が生成されていません: {stage.artifact}")
        except BaseException as exc:
            record.update(state="failed", error=repr(exc))
            raise
        else:
            record.update(state="completed")
            marker.touch()
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            self.save_status()
        return stage.artifact

    def run(self) -> dict:
        """埋め込み→量子化→SID→推薦学習を実行する。"""
        config_path = self.output / "experiment.json"
        saved = read_json(config_path)
        if saved is not None and saved != self.config.to_dict():
            raise ValueError(
                "出力先に異なる実験設定があります。別のoutput_dirを指定してください。"
            )
        # 旧スクリプトの完了マーカーだけでは実験条件を保証できない。
        if saved is None and any(self.output.glob("*/.complete")):
            raise ValueError(
                "旧形式の成果物があります。新しいoutput_dirを指定してください。"
            )
        write_json(config_path, self.config.to_dict())
        self.status.update(
            state="running",
            dataset=self.config.dataset,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        self.status.pop("error", None)
        self.save_status()
        try:
            stages = build_stages(self.config)
            embeddings = self.run_stage(stages[0])
            checks = validate_embeddings(embeddings, self.config.embedding_dim)
            write_json(self.output / "embedding_checks.json", checks)
            stages = build_stages(self.config, checks["shape"][0])
            self.run_stage(stages[1])
            ids = self.run_stage(stages[2])
            checks = validate_semantic_ids(
                ids, self.config.num_hierarchies + 1, checks["shape"][0]
            )
            write_json(self.output / "semantic_id_checks.json", checks)
            self.run_stage(stages[3])
            self.status["state"] = "completed"
        except BaseException as exc:
            self.status.update(state="failed", error=repr(exc))
            current = self.status.get("current_stage")
            if current:
                self.status["stages"][current].update(state="failed", error=repr(exc))
                (self.output / current / ".complete").unlink(missing_ok=True)
            raise
        finally:
            self.save_status()
        return self.status


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="埋め込み・SID・推薦モデルの実験を実行する"
    )
    parser.add_argument("--config", type=Path, help="実験設定のJSONファイル")
    parser.add_argument("--output-dir", help="成果物の保存先（実験ごとに分ける）")
    parser.add_argument(
        "--dry-run", action="store_true", help="コマンド表示のみ。書き込みなし"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="STAGE:KEY=VALUE",
        help="ステージ別のHydra上書き。複数指定可",
    )
    args = parser.parse_args(argv)
    config = ExperimentConfig.load(args.config)
    overrides = {key: list(values) for key, values in config.overrides.items()}
    for value in args.set:
        stage, separator, override = value.partition(":")
        if not separator or stage not in STAGES or "=" not in override:
            parser.error("--set は STAGE:KEY=VALUE 形式で指定してください")
        overrides.setdefault(stage, []).append(override)
    config = replace(
        config, overrides=overrides, output_dir=args.output_dir or config.output_dir
    )
    if args.dry_run:
        # 保存済み埋め込みがあれば正確な更新上限を表示する。
        embedding = build_stages(config)[0].artifact
        count = (
            validate_embeddings(embedding, config.embedding_dim)["shape"][0]
            if embedding.exists()
            else None
        )
        for stage in build_stages(config, count):
            print(shlex.join(build_command(config, stage)))
        return
    ExperimentRunner(config).run()


if __name__ == "__main__":
    main()
