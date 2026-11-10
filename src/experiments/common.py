"""実験用のパスとファイル操作。import時には環境を変更しない。"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """相対パスを実行ディレクトリによらずリポジトリ基準で解決する。"""
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path, default: Any = None) -> Any:
    """存在しない補助ファイルだけを省略し、壊れたJSONは例外にする。"""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, value: Any) -> None:
    """監視中にも部分的なJSONが見えないように置換する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def experiment_environment() -> dict[str, str]:
    """子プロセスだけに適用する環境を返し、呼び出し元の環境は維持する。"""
    return {
        **os.environ,
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_XET": "1",
        "PYTHONUNBUFFERED": "1",
        "HYDRA_FULL_ERROR": "1",
    }
