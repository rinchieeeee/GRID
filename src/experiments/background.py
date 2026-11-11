"""元設定を使うTIGER学習の起動と監視を分離する。"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from src.experiments.common import (
    ROOT,
    experiment_environment,
    read_json,
    resolve_path,
    write_json,
)

DEFAULT_BASE = "artifacts/toys_original_100epochs"
DEFAULT_SID = (
    "artifacts/toys_original_config/02_semantic_ids/pickle/merged_predictions_tensor.pt"
)


def training_command(
    out: Path, data_dir: Path, sid: Path, max_epochs: int, overrides: list[str]
) -> list[str]:
    """元設定を基準に、明示した条件だけを変更する。"""
    return [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        "experiment=tiger_train_flat",
        f"data_dir={data_dir}",
        f"semantic_id_path={sid}",
        "num_hierarchies=4",
        f"trainer.max_epochs={max_epochs}",
        *overrides,
        f"hydra.run.dir={out}",
    ]


def monitor(out: Path, command: list[str]) -> int:
    """子プロセスの終了コードを保持して状態を記録する。"""
    out.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "starting",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "monitor_pid": os.getpid(),
        "command": command,
        "output_dir": str(out),
    }
    write_json(out / "command.json", command)
    write_json(out / "status.json", status)
    started = time.monotonic()
    try:
        with (out / "console.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=experiment_environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            status.update(state="running", training_launcher_pid=process.pid)
            write_json(out / "status.json", status)
            code = process.wait()
        status.update(state="completed" if code == 0 else "failed", exit_code=code)
        return code
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        raise
    finally:
        status.update(
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(out / "status.json", status)


def check_previous_run(base: Path) -> None:
    previous = read_json(base / "latest_run.json")
    if not previous:
        return
    state_path = Path(previous["output_dir"]) / "status.json"
    state = read_json(state_path, {})
    if state.get("state") in (None, "starting", "running"):
        try:
            os.kill(previous["monitor_pid"], 0)
        except ProcessLookupError:
            return
        raise RuntimeError(f"既に学習が実行されています: {state_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="元設定のTIGER学習を起動する")
    parser.add_argument("--base-dir", default=DEFAULT_BASE)
    parser.add_argument("--data-dir", default="data/amazon_data/toys")
    parser.add_argument("--semantic-id-path", default=DEFAULT_SID)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument(
        "--override", action="append", default=[], help="Hydraのkey=value"
    )
    parser.add_argument(
        "--foreground", action="store_true", help="監視プロセスを切り離さない"
    )
    parser.add_argument("--dry-run", action="store_true")
    # 旧スクリプトの内部呼び出しも維持する。
    parser.add_argument("--monitor", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.monitor:
        out = resolve_path(args.monitor)
        command = read_json(out / "command.json") or training_command(
            out,
            resolve_path(args.data_dir),
            resolve_path(args.semantic_id_path),
            args.max_epochs,
            args.override,
        )
        raise SystemExit(monitor(out, command))
    base = resolve_path(args.base_dir)
    out = base / time.strftime("%Y%m%d_%H%M%S")
    command = training_command(
        out,
        resolve_path(args.data_dir),
        resolve_path(args.semantic_id_path),
        args.max_epochs,
        args.override,
    )
    if args.dry_run:
        import shlex

        print(shlex.join(command))
        return
    if not resolve_path(args.semantic_id_path).is_file():
        raise FileNotFoundError(resolve_path(args.semantic_id_path))
    check_previous_run(base)
    out.mkdir(parents=True)
    if args.foreground:
        write_json(
            base / "latest_run.json",
            {"monitor_pid": os.getpid(), "output_dir": str(out)},
        )
        raise SystemExit(monitor(out, command))
    write_json(out / "command.json", command)
    with (out / "monitor.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "src.experiments.background",
                "--monitor",
                str(out),
            ],
            cwd=ROOT,
            env=experiment_environment(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    reference = {"monitor_pid": process.pid, "output_dir": str(out)}
    write_json(base / "latest_run.json", reference)
    print(reference)


if __name__ == "__main__":
    main()
