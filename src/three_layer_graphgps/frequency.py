"""Explicit spectral low/high decomposition for the unified Stage 2 graph."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence
import warnings

import torch
from torch import nn

from .spectral_lap_pe import LaplacianEigenpairs


@dataclass(frozen=True)
class FrequencyComponents:
    """Mixed joint GraphGPS features and their exact spectral decomposition."""

    mixed: torch.Tensor
    low: torch.Tensor
    high: torch.Tensor
    coefficients: torch.Tensor
    num_low_modes: int
    cutoff_eigenvalue: float


@dataclass(frozen=True)
class LowFrequencyTransferInputs:
    """Static-only inputs reserved for source/target low-frequency alignment.

    This object deliberately contains no high-frequency features and no dynamic
    target values.  Each source city owns exactly ``1 / S`` total probability
    mass, irrespective of its number of Regions.
    """

    source_low: torch.Tensor
    target_low: torch.Tensor
    source_marginals: torch.Tensor
    target_marginals: torch.Tensor
    source_city_sizes: tuple[int, ...]
    cost_metric: str = "cosine"


class SpectralFeatureDecoupler(nn.Module):
    """Project GraphGPS hidden states onto low Laplacian modes.

    The implementation evaluates ``basis.T @ hidden`` followed by
    ``basis @ coefficients``.  Its largest projection intermediates are
    therefore node-by-feature and mode-by-feature tensors; no node-by-node
    projector is materialized.
    """

    def __init__(
        self,
        *,
        orthogonality_tolerance: float = 1e-3,
        reconstruction_tolerance: float = 1e-5,
    ) -> None:
        super().__init__()
        if not isinstance(orthogonality_tolerance, (int, float)) or orthogonality_tolerance <= 0:
            raise ValueError("orthogonality_tolerance must be a positive number")
        if not isinstance(reconstruction_tolerance, (int, float)) or reconstruction_tolerance <= 0:
            raise ValueError("reconstruction_tolerance must be a positive number")
        self.orthogonality_tolerance = float(orthogonality_tolerance)
        self.reconstruction_tolerance = float(reconstruction_tolerance)
        self.last_elapsed_seconds = 0.0
        self.last_estimated_intermediate_bytes = 0

    @staticmethod
    def _validate_inputs(
        hidden: torch.Tensor,
        eigenpairs: LaplacianEigenpairs,
        num_low_modes: int,
    ) -> None:
        if not isinstance(num_low_modes, int) or isinstance(num_low_modes, bool) or num_low_modes <= 0:
            raise ValueError("num_low_modes must be a positive integer")
        if hidden.ndim != 2 or hidden.shape[0] == 0 or hidden.shape[1] == 0:
            raise ValueError(f"hidden must be non-empty [num_nodes, hidden_dim], got {tuple(hidden.shape)}")
        if not hidden.is_floating_point():
            raise TypeError(f"hidden must use a floating dtype, got {hidden.dtype}")
        if eigenpairs.eigvecs.ndim != 2:
            raise ValueError(f"eigvecs must be [num_nodes, k], got {tuple(eigenpairs.eigvecs.shape)}")
        if eigenpairs.eigvals.ndim != 3 or eigenpairs.eigvals.shape[2] != 1:
            raise ValueError(f"eigvals must be [num_nodes, k, 1], got {tuple(eigenpairs.eigvals.shape)}")
        if eigenpairs.mask.ndim != 1:
            raise ValueError(f"mask must be [k], got {tuple(eigenpairs.mask.shape)}")
        num_nodes, num_modes = eigenpairs.eigvecs.shape
        if hidden.shape[0] != num_nodes:
            raise ValueError(
                f"hidden/eigvecs node mismatch: {hidden.shape[0]} != {num_nodes}"
            )
        if eigenpairs.eigvals.shape != (num_nodes, num_modes, 1):
            raise ValueError(
                "eigvals/eigvecs shape mismatch: "
                f"{tuple(eigenpairs.eigvals.shape)} vs {(num_nodes, num_modes, 1)}"
            )
        if eigenpairs.mask.shape[0] != num_modes:
            raise ValueError(f"mask length {eigenpairs.mask.shape[0]} != eigenpair width {num_modes}")
        if eigenpairs.mask.dtype != torch.bool:
            raise TypeError(f"mask must have dtype torch.bool, got {eigenpairs.mask.dtype}")
        if not eigenpairs.eigvecs.is_floating_point() or not eigenpairs.eigvals.is_floating_point():
            raise TypeError("eigvecs and eigvals must use floating dtypes")
        if eigenpairs.eigvecs.dtype != hidden.dtype or eigenpairs.eigvals.dtype != hidden.dtype:
            raise TypeError(
                "hidden, eigvecs, and eigvals must have the same dtype; got "
                f"{hidden.dtype}, {eigenpairs.eigvecs.dtype}, {eigenpairs.eigvals.dtype}"
            )
        if (
            hidden.device != eigenpairs.eigvecs.device
            or hidden.device != eigenpairs.eigvals.device
            or hidden.device != eigenpairs.mask.device
        ):
            raise ValueError("hidden, eigvecs, eigvals, and mask must be on the same device")
        if not torch.isfinite(hidden).all():
            raise ValueError("hidden contains NaN or Inf")
        if not torch.isfinite(eigenpairs.eigvecs).all():
            raise ValueError("eigvecs contains NaN or Inf")
        if not torch.isfinite(eigenpairs.eigvals).all():
            raise ValueError("eigvals contains NaN or Inf")

    def forward(
        self,
        hidden: torch.Tensor,
        eigenpairs: LaplacianEigenpairs,
        num_low_modes: int,
    ) -> FrequencyComponents:
        self._validate_inputs(hidden, eigenpairs, num_low_modes)
        started = time.perf_counter()

        valid_indices = torch.nonzero(eigenpairs.mask, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError("Laplacian eigenpairs contain no valid modes")

        reference_values = eigenpairs.eigvals[0, :, 0]
        if not torch.allclose(
            eigenpairs.eigvals[:, eigenpairs.mask, 0],
            reference_values[eigenpairs.mask].unsqueeze(0).expand(hidden.shape[0], -1),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("eigvals must repeat the same ordered spectrum for every node")

        valid_values = reference_values[valid_indices]
        order = torch.argsort(valid_values)
        valid_indices = valid_indices[order]
        valid_values = valid_values[order]

        actual_modes = min(num_low_modes, int(valid_indices.numel()))
        if actual_modes < num_low_modes:
            warnings.warn(
                f"requested {num_low_modes} low modes but only {actual_modes} valid eigenpairs exist; "
                "using all valid modes",
                RuntimeWarning,
                stacklevel=2,
            )

        selected_indices = valid_indices[:actual_modes]
        selected_values = valid_values[:actual_modes]
        basis = eigenpairs.eigvecs[:, selected_indices]

        gram = basis.transpose(0, 1) @ basis
        identity = torch.eye(actual_modes, dtype=hidden.dtype, device=hidden.device)
        orthogonality_error = float(torch.max(torch.abs(gram - identity)).item())
        if orthogonality_error > self.orthogonality_tolerance:
            raise ValueError(
                "selected Laplacian eigenvectors are not sufficiently orthonormal: "
                f"max_error={orthogonality_error:.6g}, "
                f"tolerance={self.orthogonality_tolerance:.6g}"
            )

        coefficients = basis.transpose(0, 1) @ hidden
        low = basis @ coefficients
        high = hidden - low

        if not torch.isfinite(coefficients).all() or not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("spectral feature decomposition produced NaN or Inf")

        reconstruction = low + high
        denominator = torch.linalg.vector_norm(hidden).clamp_min(torch.finfo(hidden.dtype).eps)
        reconstruction_error = torch.linalg.vector_norm(hidden - reconstruction) / denominator
        if float(reconstruction_error.item()) > self.reconstruction_tolerance:
            raise ValueError(
                "spectral reconstruction error exceeds tolerance: "
                f"error={float(reconstruction_error.item()):.6g}, "
                f"tolerance={self.reconstruction_tolerance:.6g}"
            )

        residual_error = torch.linalg.vector_norm(basis.transpose(0, 1) @ high) / denominator
        if float(residual_error.item()) > self.orthogonality_tolerance:
            raise ValueError(
                "high-frequency residual is not orthogonal to selected low modes: "
                f"relative_error={float(residual_error.item()):.6g}, "
                f"tolerance={self.orthogonality_tolerance:.6g}"
            )

        element_size = hidden.element_size()
        self.last_estimated_intermediate_bytes = int(
            (basis.numel() + coefficients.numel() + low.numel() + high.numel()) * element_size
        )
        self.last_elapsed_seconds = time.perf_counter() - started
        return FrequencyComponents(
            mixed=hidden,
            low=low,
            high=high,
            coefficients=coefficients,
            num_low_modes=actual_modes,
            cutoff_eigenvalue=float(selected_values[-1].item()),
        )


def build_low_frequency_transfer_inputs(
    source_low_by_city: Sequence[torch.Tensor],
    target_low: torch.Tensor,
) -> LowFrequencyTransferInputs:
    """Build the static low-frequency interface reserved for cosine-cost CCA.

    No loss is computed here.  The returned marginals encode equal total mass
    for every source city, ready for the existing CRAFT Wasserstein routine once
    a real target static graph is available.
    """

    if not source_low_by_city:
        raise ValueError("at least one source city low-frequency tensor is required")
    tensors = tuple(source_low_by_city)
    if target_low.ndim != 2 or target_low.shape[0] == 0:
        raise ValueError("target_low must be a non-empty [num_regions, hidden_dim] tensor")
    if not target_low.is_floating_point() or not torch.isfinite(target_low).all():
        raise ValueError("target_low must be a finite floating tensor")
    feature_dim = target_low.shape[1]
    device = target_low.device
    dtype = target_low.dtype
    sizes: list[int] = []
    for city_index, tensor in enumerate(tensors):
        if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] != feature_dim:
            raise ValueError(
                f"source_low_by_city[{city_index}] must be non-empty [N,{feature_dim}], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.device != device or tensor.dtype != dtype:
            raise ValueError("all source and target low-frequency tensors must share dtype and device")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"source_low_by_city[{city_index}] contains NaN or Inf")
        sizes.append(int(tensor.shape[0]))

    num_cities = len(tensors)
    source_marginals = torch.cat(
        [torch.full((size,), 1.0 / (num_cities * size), dtype=dtype, device=device) for size in sizes],
        dim=0,
    )
    target_marginals = torch.full(
        (target_low.shape[0],),
        1.0 / target_low.shape[0],
        dtype=dtype,
        device=device,
    )
    return LowFrequencyTransferInputs(
        source_low=torch.cat(tensors, dim=0),
        target_low=target_low,
        source_marginals=source_marginals,
        target_marginals=target_marginals,
        source_city_sizes=tuple(sizes),
    )
