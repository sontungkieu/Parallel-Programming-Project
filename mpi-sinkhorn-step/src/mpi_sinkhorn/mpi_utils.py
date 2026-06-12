"""Small MPI and JSON helpers used by the solver runners."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import numpy as np


def split_indices(n_items: int, size: int) -> list[tuple[int, int]]:
    """Return balanced half-open index ranges for `size` ranks."""
    if n_items < 0:
        raise ValueError("n_items must be non-negative")
    if size <= 0:
        raise ValueError("size must be positive")

    base = n_items // size
    extra = n_items % size
    ranges: list[tuple[int, int]] = []
    start = 0
    for rank in range(size):
        width = base + (1 if rank < extra else 0)
        end = start + width
        ranges.append((start, end))
        start = end
    return ranges


def get_rank_info(comm: Any) -> tuple[int, int, str]:
    return comm.Get_rank(), comm.Get_size(), socket.gethostname()


def gather_hostnames(comm: Any) -> list[str] | None:
    """Gather one hostname per rank on root."""
    hostname = socket.gethostname()
    return comm.gather(hostname, root=0)


def scatter_row_blocks(comm: Any, matrix: np.ndarray | None) -> np.ndarray:
    """Scatter row blocks of a 2D matrix from rank 0 to all ranks."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if matrix is None:
            raise ValueError("rank 0 must provide a matrix to scatter")
        if matrix.ndim != 2:
            raise ValueError("matrix must be 2D")
        ranges = split_indices(matrix.shape[0], size)
        chunks = [np.ascontiguousarray(matrix[start:end]) for start, end in ranges]
    else:
        chunks = None

    return comm.scatter(chunks, root=0)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
