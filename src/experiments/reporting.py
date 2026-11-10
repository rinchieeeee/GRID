"""モデルを読み込まず、CSVログから検証履歴とテスト結果を集計する。"""

import argparse
import csv
import math
from pathlib import Path

from src.experiments.common import ROOT, read_json, resolve_path, write_json
from src.experiments.config import ExperimentConfig


def read_metrics(run_dir: Path, version: str | None = None) -> list[dict[str, str]]:
    """再試行間の指標混在を避け、指定版または最新の数値版を読む。"""
    csv_root = run_dir / "04_tiger/csv"
    if not csv_root.exists():
        csv_root = run_dir / "csv"
    if version is None:
        versions = [
            p
            for p in csv_root.glob("version_*")
            if p.name.removeprefix("version_").isdigit()
        ]
        if not versions:
            return []
        directory = max(versions, key=lambda p: int(p.name.removeprefix("version_")))
    else:
        directory = csv_root / version
    metrics = directory / "metrics.csv"
    with metrics.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def collect_metrics(rows: list[dict[str, str]], prefix: str) -> list[dict[str, float]]:
    """欠測値を0に置換せず、観測された指標だけを返す。"""
    collected = []
    for row in rows:
        values = {
            key: float(value)
            for key, value in row.items()
            if value and (key.startswith(prefix) or key in ("epoch", "step"))
        }
        values = {key: value for key, value in values.items() if math.isfinite(value)}
        if any(key.startswith(prefix) for key in values):
            collected.append(values)
    return collected


def summarize_run(run_dir: Path, version: str | None = None) -> dict:
    """集計結果を返す純粋な読み取り処理。未完了の実験にも利用できる。"""
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    rows = read_metrics(run_dir, version)
    validation = collect_metrics(rows, "val/")
    testing = collect_metrics(rows, "test/")
    candidates = [
        row
        for row in validation
        if math.isfinite(row.get("val/recall@5", float("nan")))
    ]
    checkpoint_dir = (
        run_dir / "04_tiger/checkpoints"
        if (run_dir / "04_tiger").exists()
        else run_dir / "checkpoints"
    )
    return {
        "config": read_json(run_dir / "experiment.json"),
        "training_checks": read_json(run_dir / "training_checks.json"),
        "strict_original_config_reproduction": False,
        "status": read_json(run_dir / "status.json"),
        "datasets": read_json(ROOT / "data/dataset_stats.json", {}),
        "semantic_ids": read_json(run_dir / "semantic_id_checks.json"),
        "validation_history": validation,
        "best_validation": max(
            candidates, key=lambda row: row["val/recall@5"], default=None
        ),
        "test": testing[-1] if testing else None,
        "best_checkpoints": [
            str(path) for path in sorted(checkpoint_dir.glob("checkpoint*"))
        ],
    }


def format_metric(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "—"


def render_report(summary: dict) -> str:
    """実際の設定と観測値を使って日本語の結果レポートを構築する。"""
    lines = ["# 実験結果", "", "公開設定の厳密な再現実験ではありません。", ""]
    if summary["config"]:
        lines.extend(["## 実験設定", "", "```json"])
        import json

        lines.extend(
            [json.dumps(summary["config"], ensure_ascii=False, indent=2), "```", ""]
        )
    lines.extend(
        [
            "## 検証履歴",
            "",
            "欠測指標は「—」で表示します。",
            "",
            "| Epoch | Step | Loss | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    keys = ("val/loss", "val/recall@5", "val/recall@10", "val/ndcg@5", "val/ndcg@10")
    for row in summary["validation_history"]:
        epoch = str(int(row["epoch"]) + 1) if "epoch" in row else "—"
        step = str(int(row["step"])) if "step" in row else "—"
        lines.append(
            "| "
            + " | ".join([epoch, step, *[format_metric(row.get(key)) for key in keys]])
            + " |"
        )
    lines.extend(["", "## テスト結果", ""])
    if summary["test"]:
        lines.extend(
            f"- {key}: {format_metric(value)}" for key, value in summary["test"].items()
        )
    else:
        lines.append("テスト結果はまだ記録されていません。")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="実験ログを集計する")
    parser.add_argument("--run-dir", default=ExperimentConfig().output_dir)
    parser.add_argument("--version", help="例: version_0。省略時は最新の数値版")
    args = parser.parse_args(argv)
    run_dir = resolve_path(args.run_dir)
    summary = summarize_run(run_dir, args.version)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "results.md").write_text(render_report(summary), encoding="utf-8")
    print(f"集計結果: {run_dir / 'results.md'}")


if __name__ == "__main__":
    main()
