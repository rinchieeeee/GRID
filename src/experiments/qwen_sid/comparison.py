"""同一初期値・同一推薦例のA/B結果を確認し、比較表を保存する。"""

import json
from pathlib import Path

from src.experiments.common import write_json


def compare_runs(root: Path):
    runs = {}
    for name in ("without_alignment", "with_alignment"):
        folder = root / name
        runs[name] = {
            "experiment": json.loads((folder / "experiment.json").read_text()),
            "results": json.loads((folder / "results.json").read_text()),
            "history": [
                json.loads(line)
                for line in (folder / "metrics.jsonl").read_text().splitlines()
            ],
        }
    a, b = runs["without_alignment"], runs["with_alignment"]
    for key in (
        "initial_parameter_sha256",
        "recommendation_plan_sha256",
        "planned_updates",
    ):
        if a["experiment"][key] != b["experiment"][key]:
            raise ValueError(f"比較条件が異なります: {key}")
    for key in ("step", "cumulative_recommendations"):
        if a["history"][-1][key] != b["history"][-1][key]:
            raise ValueError(f"学習量が異なります: {key}")
    a_test, b_test = a["results"]["testing"], b["results"]["testing"]
    for key in ("beam_width", "check_valid_prefix"):
        if a_test[key] != b_test[key]:
            raise ValueError(f"評価条件が異なります: {key}")
    metrics = ("hit@1", "recall@5", "recall@10", "ndcg@5", "ndcg@10")
    table = {
        key: {
            "without_alignment": a_test[key],
            "with_alignment": b_test[key],
            "difference": b_test[key] - a_test[key],
        }
        for key in metrics
    }
    write_json(root / "comparison.json", {"runs": runs, "test_comparison": table})
    prefix = "あり" if a_test["check_valid_prefix"] else "なし"
    lines = [
        "# SID–Text alignment のA/B比較",
        "",
        "Qwen3-0.6B、全重み更新、共通の初期値・推薦例・更新回数。",
        f"評価はprefix制約{prefix}・beam幅{a_test['beam_width']}。"
        "checkpointはvalidation Recall@5で選択。",
        "",
        "| test指標 | 推薦のみ | alignmentあり | 差（あり−なし） |",
        "|---|---:|---:|---:|",
    ]
    for key, row in table.items():
        lines.append(
            f"| {key} | {row['without_alignment']:.6f} | "
            f"{row['with_alignment']:.6f} | {row['difference']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "単一seedの結果。alignment側は補助タスク分の計算量が増える。",
            "前回TIGERは構造・規模・学習量・デコード条件が異なる参考値。",
            "前回TIGER: Recall@5=0.042396、Recall@10=0.062333、",
            "NDCG@5=0.027544、NDCG@10=0.033976（beam幅10・prefix制約なし）。",
        ]
    )
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return table
