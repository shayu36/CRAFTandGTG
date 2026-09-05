"""GraphGPS 风格 Laplacian positional encoding。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .spectral_lap_pe import LaplacianEigenpairs


class LapPEEncoder(nn.Module):
    """用 DeepSet 编码 ``[eigenvector value, eigenvalue]`` 频率集合。"""

    def __init__(
        self,
        num_eigenvectors: int,
        pe_dim: int,
        *,
        encoder: str = "DeepSet",
        pooling: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_eigenvectors <= 0 or pe_dim <= 0:
            raise ValueError("严格模式: LapPE num_eigenvectors/pe_dim 必须为正")
        if encoder.lower() != "deepset":
            if encoder.lower() == "signnet":
                raise NotImplementedError("第一版 GraphGPS LapPE 不实现 SignNet")
            raise ValueError(f"严格模式: 不支持 LapPE encoder={encoder!r}")
        if pooling not in {"mean", "sum"}:
            raise ValueError("严格模式: LapPE pooling 只能为 mean/sum")
        if not 0 <= dropout <= 1:
            raise ValueError("严格模式: LapPE dropout 必须在 [0,1]")
        self.num_eigenvectors = int(num_eigenvectors)
        self.pe_dim = int(pe_dim)
        self.pooling = pooling
        self.frequency_mlp = nn.Sequential(
            nn.Linear(2, pe_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pe_dim, pe_dim),
        )
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(pe_dim),
            nn.Linear(pe_dim, pe_dim),
            nn.GELU(),
        )

    def forward(self, eigenpairs: LaplacianEigenpairs) -> torch.Tensor:
        eigvals, eigvecs, mask = eigenpairs.eigvals, eigenpairs.eigvecs, eigenpairs.mask
        if eigvecs.ndim != 2 or eigvecs.shape[1] != self.num_eigenvectors:
            raise ValueError(
                f"严格模式: LapPE eigvecs 应为 [V,{self.num_eigenvectors}]，"
                f"实得 {tuple(eigvecs.shape)}"
            )
        if eigvals.shape != (eigvecs.shape[0], self.num_eigenvectors, 1):
            raise ValueError("严格模式: LapPE eigvals/eigvecs shape 不一致")
        if mask.dtype != torch.bool or mask.shape != (self.num_eigenvectors,):
            raise ValueError("严格模式: LapPE mask shape/dtype 错误")
        if not torch.isfinite(eigvals).all() or not torch.isfinite(eigvecs).all():
            raise ValueError("严格模式: LapPE 输入含 NaN/Inf")

        signed_vectors = eigvecs
        if self.training:
            signs = torch.randint(
                0, 2, (self.num_eigenvectors,), device=eigvecs.device, dtype=torch.long
            ).to(eigvecs.dtype)
            signs = signs.mul_(2).sub_(1)
            signed_vectors = eigvecs * signs.unsqueeze(0)
        frequency_input = torch.cat([signed_vectors.unsqueeze(-1), eigvals], dim=-1)
        frequency_h = self.frequency_mlp(frequency_input)
        expanded_mask = mask.view(1, -1, 1).to(frequency_h.device)
        frequency_h = frequency_h * expanded_mask
        pooled = frequency_h.sum(dim=1)
        if self.pooling == "mean":
            pooled = pooled / mask.sum().clamp_min(1).to(pooled.dtype)
        result = self.output_mlp(pooled)
        if not torch.isfinite(result).all():
            raise FloatingPointError("严格模式: LapPE encoder 输出含 NaN/Inf")
        return result


class FeatureLapPEInit(nn.Module):
    """独立投影一层原始特征，并和该层 LapPE 融合。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_eigenvectors: int,
        pe_dim: int,
        *,
        encoder: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        self.lappe = LapPEEncoder(
            num_eigenvectors,
            pe_dim,
            encoder=encoder,
            pooling="mean",
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + pe_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, eigenpairs: LaplacianEigenpairs) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"严格模式: 层输入应为 [V,{self.input_dim}]，实得 {tuple(x.shape)}"
            )
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("严格模式: 层输入必须为有限浮点 Tensor")
        feature_h = self.feature_projection(x)
        pe_h = self.lappe(eigenpairs)
        if feature_h.shape[0] != pe_h.shape[0]:
            raise ValueError("严格模式: 节点特征与 LapPE 节点数不一致")
        return self.fusion(torch.cat([feature_h, pe_h], dim=-1))
