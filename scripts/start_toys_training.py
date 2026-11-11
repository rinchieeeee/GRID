"""既存コマンドとの互換性を保つ実行入口。処理本体はsrc.experimentsに置く。"""

import sys
from pathlib import Path

# 直接実行でも、パッケージをインストールせずに利用できるようにする。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.background import main, monitor  # noqa: E402, F401

if __name__ == "__main__":
    main()
