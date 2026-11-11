"""追加実験で変更する条件を一か所に集める。"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.experiments.common import read_json, resolve_path

STAGES = ("01_embeddings", "02_rkmeans", "03_semantic_ids", "04_tiger")


@dataclass(frozen=True)
class ExperimentConfig:
    """Toysの既存実験を既定値とし、他カテゴリにも同じ処理を適用する。"""

    dataset: str = "toys"
    data_dir: str = "data/amazon_data/toys"
    output_dir: str = "artifacts/toys_rkmeans_10epochs"
    embedding_model: str = "google/flan-t5-xl"
    embedding_dim: int = 2048
    num_hierarchies: int = 3
    codebook_width: int = 256
    rkmeans_epochs: int = 90
    rkmeans_batch_size: int = 1024
    recommendation_epochs: int = 10
    augmentation_batch_size: int = 32
    evaluation_batch_size: int = 128
    precision: str = "bf16-mixed"
    seed: int | None = None
    # Hydraの詳細設定はステージ単位で上書きする。
    overrides: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "embedding_dim",
            "num_hierarchies",
            "codebook_width",
            "rkmeans_epochs",
            "rkmeans_batch_size",
            "recommendation_epochs",
            "augmentation_batch_size",
            "evaluation_batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} は正の整数にしてください: {value!r}")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed は整数またはnullにしてください")
        if not isinstance(self.overrides, dict) or set(self.overrides) - set(STAGES):
            raise ValueError(f"overrides のキーは {STAGES} から指定してください")
        for stage, values in self.overrides.items():
            if not isinstance(values, list) or any(
                not isinstance(value, str) or "=" not in value for value in values
            ):
                raise ValueError(f"{stage} の上書きは key=value のリストにしてください")
        for name in (
            "dataset",
            "data_dir",
            "output_dir",
            "embedding_model",
            "precision",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} は空でない文字列にしてください")

    @property
    def data_path(self) -> Path:
        return resolve_path(self.data_dir)

    @property
    def output_path(self) -> Path:
        return resolve_path(self.output_dir)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path | None = None) -> "ExperimentConfig":
        if path is None:
            return cls()
        if not path.is_file():
            raise FileNotFoundError(path)
        return cls(**read_json(path))
