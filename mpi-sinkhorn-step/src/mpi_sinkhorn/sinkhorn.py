"""Dense and MPI row-partitioned Sinkhorn-Knopp solvers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from mpi4py import MPI

TINY = 1e-12


def generate_cost_matrix(
    rows: int,
    cols: int,
    mode: str = "random",
    seed: int = 0,
) -> np.ndarray:
    """Generate a deterministic cost matrix shared by sequential and MPI runners."""
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.random((rows, cols), dtype=np.float64)

    if mode == "squared_distance_1d":
        x = np.linspace(0.0, 1.0, rows, dtype=np.float64)
        y = np.linspace(0.0, 1.0, cols, dtype=np.float64)
        return (x[:, None] - y[None, :]) ** 2

    if mode == "block":
        row_groups = np.floor(np.linspace(0.0, 4.0, rows, endpoint=False)).astype(int)
        col_groups = np.floor(np.linspace(0.0, 4.0, cols, endpoint=False)).astype(int)
        mismatch = (row_groups[:, None] != col_groups[None, :]).astype(np.float64)
        x = np.linspace(0.0, 1.0, rows, dtype=np.float64)
        y = np.linspace(0.0, 1.0, cols, dtype=np.float64)
        return mismatch + 0.05 * (x[:, None] - y[None, :]) ** 2

    raise ValueError(f"unsupported cost mode: {mode}")


def _validate_cost(cost: np.ndarray, name: str = "cost") -> np.ndarray:
    array = np.asarray(cost, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[1] <= 0:
        raise ValueError(f"{name} must have at least one column")
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} entries must be finite")
    return array


def _default_marginal(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.full(length, 1.0 / length, dtype=np.float64)


def _validate_marginal(
    marginal: np.ndarray | None,
    length: int,
    name: str,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    if marginal is None:
        return _default_marginal(length)

    array = np.asarray(marginal, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} entries must be finite")
    if length == 0 and allow_empty:
        return array
    if np.any(array <= 0.0):
        raise ValueError(f"{name} entries must be positive")
    return array


def _validate_marginal_mass(a: np.ndarray, b: np.ndarray) -> None:
    a_sum = float(np.sum(a))
    b_sum = float(np.sum(b))
    if a_sum <= 0.0 or b_sum <= 0.0:
        raise ValueError("marginal masses must be positive")
    if not np.isclose(a_sum, b_sum, rtol=1e-10, atol=1e-12):
        raise ValueError(f"marginal masses must match, got {a_sum} and {b_sum}")


def sinkhorn_knopp_dense(
    cost: np.ndarray,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    epsilon: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-6,
    check_every: int = 10,
    return_transport: bool = False,
) -> dict[str, Any]:
    """Solve entropy-regularized OT using dense Sinkhorn-Knopp scaling."""
    cost = _validate_cost(cost)
    rows, cols = cost.shape
    if rows <= 0:
        raise ValueError("cost must have at least one row")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if check_every <= 0:
        raise ValueError("check_every must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")

    a_vec = _validate_marginal(a, rows, "a")
    b_vec = _validate_marginal(b, cols, "b")
    _validate_marginal_mass(a_vec, b_vec)

    started = time.perf_counter()
    kernel = np.exp(-(cost - float(cost.min())) / epsilon)
    u = np.ones_like(a_vec)
    v = np.ones_like(b_vec)
    row_error = float("inf")
    col_error = float("inf")
    iterations = 0

    for iteration in range(1, max_iters + 1):
        u = a_vec / (kernel @ v + TINY)
        v = b_vec / (kernel.T @ u + TINY)
        iterations = iteration

        if iteration % check_every == 0 or iteration == max_iters:
            transport_check = (u[:, None] * kernel) * v[None, :]
            row_error = float(np.abs(transport_check.sum(axis=1) - a_vec).sum())
            col_error = float(np.abs(transport_check.sum(axis=0) - b_vec).sum())
            if row_error < tol and col_error < tol:
                break

    transport = (u[:, None] * kernel) * v[None, :]
    row_error = float(np.abs(transport.sum(axis=1) - a_vec).sum())
    col_error = float(np.abs(transport.sum(axis=0) - b_vec).sum())
    transport_objective = float(np.sum(transport * cost))
    runtime_sec = time.perf_counter() - started

    return {
        "transport": transport if return_transport else None,
        "u": u,
        "v": v,
        "iterations": iterations,
        "runtime_sec": runtime_sec,
        "row_error": row_error,
        "col_error": col_error,
        "transport_objective": transport_objective,
    }


def sinkhorn_knopp_distributed(
    comm: Any,
    local_cost: np.ndarray,
    local_a: np.ndarray,
    b: np.ndarray,
    epsilon: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-6,
    check_every: int = 10,
    return_local_transport: bool = False,
) -> dict[str, Any]:
    """Solve entropy-regularized OT with row-partitioned MPI communication."""
    local_cost = _validate_cost(local_cost, "local_cost")
    local_rows, cols = local_cost.shape
    local_a_vec = _validate_marginal(local_a, local_rows, "local_a", allow_empty=True)
    b_vec = _validate_marginal(b, cols, "b")

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if check_every <= 0:
        raise ValueError("check_every must be positive")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")

    local_a_mass = np.array(float(np.sum(local_a_vec)), dtype=np.float64)
    global_a_mass = np.array(0.0, dtype=np.float64)
    comm.Allreduce(local_a_mass, global_a_mass, op=MPI.SUM)
    b_mass = float(np.sum(b_vec))
    if float(global_a_mass) <= 0.0 or b_mass <= 0.0:
        raise ValueError("marginal masses must be positive")
    if not np.isclose(float(global_a_mass), b_mass, rtol=1e-10, atol=1e-12):
        raise ValueError(
            f"marginal masses must match, got {float(global_a_mass)} and {b_mass}"
        )

    local_min = np.array(
        float(local_cost.min()) if local_cost.size else np.inf,
        dtype=np.float64,
    )
    global_min = np.array(0.0, dtype=np.float64)
    comm.Allreduce(local_min, global_min, op=MPI.MIN)

    started = time.perf_counter()
    local_kernel = np.exp(-(local_cost - float(global_min)) / epsilon)
    local_u = np.ones(local_rows, dtype=np.float64)
    v = np.ones_like(b_vec)
    row_error = float("inf")
    col_error = float("inf")
    transport_objective = float("inf")
    iterations = 0

    for iteration in range(1, max_iters + 1):
        local_u = local_a_vec / (local_kernel @ v + TINY)
        local_kernel_t_u = local_kernel.T @ local_u
        global_kernel_t_u = np.empty_like(local_kernel_t_u)
        comm.Allreduce(local_kernel_t_u, global_kernel_t_u, op=MPI.SUM)
        v = b_vec / (global_kernel_t_u + TINY)
        iterations = iteration

        if iteration % check_every == 0 or iteration == max_iters:
            local_transport = (local_u[:, None] * local_kernel) * v[None, :]
            local_row_error = np.array(
                float(np.abs(local_transport.sum(axis=1) - local_a_vec).sum()),
                dtype=np.float64,
            )
            row_error_sum = np.array(0.0, dtype=np.float64)
            comm.Allreduce(local_row_error, row_error_sum, op=MPI.SUM)

            local_col_mass = local_transport.sum(axis=0)
            global_col_mass = np.empty_like(local_col_mass)
            comm.Allreduce(local_col_mass, global_col_mass, op=MPI.SUM)

            local_objective = np.array(
                float(np.sum(local_transport * local_cost)),
                dtype=np.float64,
            )
            objective_sum = np.array(0.0, dtype=np.float64)
            comm.Allreduce(local_objective, objective_sum, op=MPI.SUM)

            row_error = float(row_error_sum)
            col_error = float(np.abs(global_col_mass - b_vec).sum())
            transport_objective = float(objective_sum)
            if row_error < tol and col_error < tol:
                break

    local_transport = (local_u[:, None] * local_kernel) * v[None, :]
    local_row_error = np.array(
        float(np.abs(local_transport.sum(axis=1) - local_a_vec).sum()),
        dtype=np.float64,
    )
    row_error_sum = np.array(0.0, dtype=np.float64)
    comm.Allreduce(local_row_error, row_error_sum, op=MPI.SUM)

    local_col_mass = local_transport.sum(axis=0)
    global_col_mass = np.empty_like(local_col_mass)
    comm.Allreduce(local_col_mass, global_col_mass, op=MPI.SUM)

    local_objective = np.array(float(np.sum(local_transport * local_cost)), dtype=np.float64)
    objective_sum = np.array(0.0, dtype=np.float64)
    comm.Allreduce(local_objective, objective_sum, op=MPI.SUM)

    row_error = float(row_error_sum)
    col_error = float(np.abs(global_col_mass - b_vec).sum())
    transport_objective = float(objective_sum)
    runtime_sec = time.perf_counter() - started

    return {
        "local_transport": local_transport if return_local_transport else None,
        "local_u": local_u,
        "v": v,
        "iterations": iterations,
        "runtime_sec": runtime_sec,
        "row_error": row_error,
        "col_error": col_error,
        "transport_objective": transport_objective,
    }
