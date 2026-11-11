"""Qwen SID実験の前景実行入口。処理本体はPythonからも個別に呼び出せる。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="QwenのSID joint SFT実験")
    parser.add_argument("stage", choices=["prepare", "probe", "train", "evaluate"])
    parser.add_argument("--config", default="configs/local/qwen3_toys.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--until-step", type=int, help="このoptimizer更新まで進めて保存"
    )
    parser.add_argument(
        "--defer-early-stopping",
        action="store_true",
        help="A/Bランナー用。改善なしの回数を記録し、停止判定をランナーに委ねる",
    )
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    if args.stage == "prepare":
        from src.experiments.qwen_sid.data import prepare_data

        result = prepare_data(config)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("source_sha256", "sid_tokens")
                },
                indent=2,
            )
        )
    elif args.stage == "probe":
        from src.experiments.qwen_sid.probe import run_probe

        manifest = json.loads(
            (Path(config["prepared_dir"]) / "manifest.json").read_text()
        )
        print(json.dumps(run_probe(config, manifest), indent=2))
    elif args.stage == "train":
        from src.experiments.qwen_sid.training import train

        train(
            config,
            resume=args.resume,
            until_step=args.until_step,
            stop_on_plateau=not args.defer_early_stopping,
        )
    else:
        from src.experiments.qwen_sid.evaluation import evaluate_checkpoint

        print(json.dumps(evaluate_checkpoint(config), indent=2))


if __name__ == "__main__":
    main()
