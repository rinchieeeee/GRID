"""ステージ境界でテンソルを検証し、異常の発生箇所を限定する。"""

from pathlib import Path


def validate_embeddings(path: Path, embedding_dim: int) -> dict:
    """件数をデータから取得し、形状と有限値を確認する。"""
    import torch

    values = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError(f"埋め込みは2次元テンソルである必要があります: {path}")
    if values.shape[0] == 0 or values.shape[1] != embedding_dim:
        raise ValueError(f"埋め込み形状が不正です: {tuple(values.shape)}")
    if not torch.isfinite(values).all():
        raise ValueError(f"埋め込みにNaNまたはInfがあります: {path}")
    return {"shape": list(values.shape), "dtype": str(values.dtype), "finite": True}


def validate_semantic_ids(path: Path, levels: int, item_count: int) -> dict:
    """重複除去用の最終桁を含むSIDの形状と一意性を検証する。"""
    import torch

    values = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(values, torch.Tensor) or tuple(values.shape) != (
        levels,
        item_count,
    ):
        raise ValueError(
            f"SIDは ({levels}, {item_count}) のテンソルにしてください: {path}"
        )
    if values.is_floating_point() or values.is_complex() or values.dtype == torch.bool:
        raise ValueError("SIDは整数テンソルである必要があります")
    if (values < 0).any():
        raise ValueError("SIDに負の値があります")
    unique_count = torch.unique(values.T, dim=0).shape[0]
    if unique_count != item_count:
        raise ValueError(f"SIDが重複しています: {unique_count}/{item_count}")
    return {"shape": list(values.shape), "unique_items": unique_count}
