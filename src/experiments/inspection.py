"""TensorFlowやモデルを読み込まずTFRecordの件数を調べる。"""

import argparse
import gzip
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.experiments.common import resolve_path, write_json


def count_file(path: Path) -> int:
    """長さとレコード境界を確認する。CRC値そのものの検証は行わない。"""
    count = 0
    with gzip.open(path, "rb") as stream:
        while header := stream.read(12):
            if len(header) != 12:
                raise ValueError(f"TFRecordのヘッダーが途中で切れています: {path}")
            length = struct.unpack("<Q", header[:8])[0]
            # 破損した長さフィールドによる巨大メモリ確保を避ける。
            remaining = length + 4
            while remaining:
                chunk = stream.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError(f"TFRecordが途中で切れています: {path}")
                remaining -= len(chunk)
            count += 1
    return count


def inspect_datasets(data_dir: Path, workers: int = 8) -> dict:
    """カテゴリ・split別に件数、ファイル数、圧縮後サイズを返す。"""
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    result = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for category in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            splits = {}
            for split in sorted(p for p in category.iterdir() if p.is_dir()):
                files = sorted(split.glob("*.tfrecord.gz"))
                splits[split.name] = {
                    "records": sum(pool.map(count_file, files)),
                    "files": len(files),
                    "bytes": sum(path.stat().st_size for path in files),
                }
            result[category.name] = splits
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Amazonデータの件数を確認する")
    parser.add_argument("--data-dir", default="data/amazon_data")
    parser.add_argument("--output", default="data/dataset_stats.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    result = inspect_datasets(resolve_path(args.data_dir), args.workers)
    write_json(resolve_path(args.output), result)
    for category, counts in result.items():
        print(category, counts)


if __name__ == "__main__":
    main()
