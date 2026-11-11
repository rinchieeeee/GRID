"""埋め込みモデルを取得する。import時にはダウンロードしない。"""

import argparse


def download_model(model: str = "google/flan-t5-xl", max_workers: int = 4) -> str:
    """推論に必要なモデル重みとトークナイザーだけを取得する。"""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model,
        allow_patterns=[
            "config.json",
            "*.safetensors",
            "*.safetensors.index.json",
            "*token*",
            "spiece.model",
        ],
        max_workers=max_workers,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="埋め込みモデルを取得する")
    parser.add_argument("--model", default="google/flan-t5-xl")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    print(download_model(args.model, args.max_workers))


if __name__ == "__main__":
    main()
