"""A/Bを次のvalidation stepまで交互に進める前景実行入口。"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_stage(root: Path, name: str, arguments: list[str]):
    from src.experiments.common import experiment_environment, write_json

    command = [sys.executable, "scripts/run_qwen_sid.py", *arguments]
    write_json(
        root / "status.json",
        {"state": "running", "stage": name, "pid": os.getpid(), "command": command},
    )
    with (root / f"{name}.log").open("a") as log:
        subprocess.run(
            command,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=experiment_environment(),
        )


def train_variants(root, variants, config):
    """両条件が同じstepでpatienceに達した場合に、A/Bを同時に止める。"""
    from src.experiments.common import read_json
    from src.experiments.qwen_sid.training import validation_steps

    progress_paths = {name: root / name / "last" / "progress.json" for name in variants}
    progress = {
        name: read_json(path, {"step": 0}) for name, path in progress_paths.items()
    }
    patience = config["early_stopping_patience"]
    for step in validation_steps(config):
        for name, path in variants.items():
            if progress[name]["step"] >= step:
                continue
            arguments = [
                "train",
                "--config",
                str(path),
                "--until-step",
                str(step),
                "--defer-early-stopping",
            ]
            if (root / name / "experiment.json").exists():
                arguments.append("--resume")
            run_stage(root, f"{name}_training", arguments)
            progress[name] = read_json(progress_paths[name])
            if progress[name]["step"] != step:
                raise ValueError(f"{name}の学習が予定のstep {step}まで進みませんでした")
        metadata = [read_json(root / name / "experiment.json") for name in variants]
        for key in ("initial_parameter_sha256", "recommendation_plan_sha256"):
            if len({entry[key] for entry in metadata}) != 1:
                raise ValueError(f"A/Bの比較条件が一致しません: {key}")
        if step < config["max_steps"] and all(
            state["step"] == step
            and state["evaluations_without_improvement"] >= patience
            for state in progress.values()
        ):
            return step, "early_stopping"
    return config["max_steps"], "max_steps"


def main(argv=None):
    from src.experiments.common import write_json
    from src.experiments.qwen_sid.comparison import compare_runs
    from src.experiments.qwen_sid.data import prepare_data, prepare_recommendation_plan
    from src.experiments.qwen_sid.training import validate_step_config

    parser = argparse.ArgumentParser(description="Qwen SID alignmentの対照実験")
    parser.add_argument("--config", default="configs/local/qwen3_toys.json")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    validate_step_config(config)
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_config = root / "config.json"
        if run_config.exists() and json.loads(run_config.read_text()) != config:
            raise ValueError("同じ出力先の実験条件が変更されています")
        write_json(run_config, config)
        started = time.time()
        try:
            plan_path = Path(config["prepared_dir"]) / "recommendation_plan.json"
            if not plan_path.exists():
                prepare_data(config)
            prepare_recommendation_plan(config)
            probe_path = root / "probe.json"
            if (
                not probe_path.exists()
                or json.loads(probe_path.read_text())["config"] != config
            ):
                run_stage(root, "probe", ["probe", "--config", str(run_config)])
            if not json.loads(probe_path.read_text())["passed"]:
                raise ValueError("全重み更新の事前確認が失敗しています")
            variants = {}
            for name, weight in (("without_alignment", 0.0), ("with_alignment", 1.0)):
                variant = {
                    **config,
                    "condition": name,
                    "alignment_weight": weight,
                    "output_dir": str(root / name),
                }
                path = root / f"config_{name}.json"
                write_json(path, variant)
                variants[name] = path
            steps, stop_reason = train_variants(root, variants, config)
            for name, path in variants.items():
                run_stage(
                    root, f"{name}_evaluation", ["evaluate", "--config", str(path)]
                )
            compare_runs(root)
            write_json(
                root / "status.json",
                {
                    "state": "completed",
                    "step": steps,
                    "training_stop_reason": stop_reason,
                    "pid": os.getpid(),
                    "elapsed_seconds": time.time() - started,
                },
            )
        except BaseException as error:
            write_json(root / "status.json", {"state": "failed", "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
