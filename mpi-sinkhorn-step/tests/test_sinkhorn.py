from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from mpi_sinkhorn.sinkhorn import (
    generate_cost_matrix,
    sinkhorn_knopp_dense,
    sinkhorn_knopp_distributed,
)


def test_generate_cost_matrix_is_deterministic() -> None:
    first = generate_cost_matrix(8, 7, mode="random", seed=42)
    second = generate_cost_matrix(8, 7, mode="random", seed=42)
    different = generate_cost_matrix(8, 7, mode="random", seed=43)

    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, different)


@pytest.mark.parametrize("mode", ["random", "squared_distance_1d", "block"])
def test_generate_cost_matrix_modes(mode: str) -> None:
    cost = generate_cost_matrix(10, 12, mode=mode, seed=0)
    assert cost.shape == (10, 12)
    assert np.all(np.isfinite(cost))
    assert np.all(cost >= 0.0)


def test_dense_sinkhorn_matches_marginals() -> None:
    cost = generate_cost_matrix(16, 12, mode="squared_distance_1d", seed=0)
    a = np.full(16, 1.0 / 16)
    b = np.full(12, 1.0 / 12)

    result = sinkhorn_knopp_dense(
        cost,
        a=a,
        b=b,
        epsilon=0.5,
        max_iters=200,
        tol=1e-9,
        check_every=5,
        return_transport=True,
    )

    transport = result["transport"]
    assert transport is not None
    assert result["row_error"] < 1e-7
    assert result["col_error"] < 1e-7
    np.testing.assert_allclose(transport.sum(axis=1), a, atol=1e-7)
    np.testing.assert_allclose(transport.sum(axis=0), b, atol=1e-7)


def test_distributed_comm_self_matches_dense() -> None:
    cost = generate_cost_matrix(20, 18, mode="block", seed=0)
    a = np.full(20, 1.0 / 20)
    b = np.full(18, 1.0 / 18)

    dense = sinkhorn_knopp_dense(
        cost,
        a=a,
        b=b,
        epsilon=0.5,
        max_iters=200,
        tol=1e-9,
        check_every=5,
    )
    distributed = sinkhorn_knopp_distributed(
        MPI.COMM_SELF,
        local_cost=cost,
        local_a=a,
        b=b,
        epsilon=0.5,
        max_iters=200,
        tol=1e-9,
        check_every=5,
    )

    assert distributed["row_error"] < 1e-7
    assert distributed["col_error"] < 1e-7
    assert abs(distributed["transport_objective"] - dense["transport_objective"]) < 1e-9


def test_dense_rejects_invalid_marginals() -> None:
    cost = generate_cost_matrix(4, 4)
    a = np.array([0.25, 0.25, 0.25, 0.0])

    with pytest.raises(ValueError, match="positive"):
        sinkhorn_knopp_dense(cost, a=a)
