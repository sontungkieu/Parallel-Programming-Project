"""MPI Sinkhorn-Knopp solver package."""

from .sinkhorn import (
    generate_cost_matrix,
    sinkhorn_knopp_dense,
    sinkhorn_knopp_distributed,
)

__all__ = [
    "generate_cost_matrix",
    "sinkhorn_knopp_dense",
    "sinkhorn_knopp_distributed",
]
