from __future__ import annotations

import argparse

import numpy as np
from mpi4py import MPI

from mpi_sinkhorn.mpi_utils import gather_hostnames, save_json, scatter_row_blocks, split_indices
from mpi_sinkhorn.sinkhorn import generate_cost_matrix, sinkhorn_knopp_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MPI row-partitioned Sinkhorn-Knopp solver.")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--cost-mode", default="random", choices=["random", "squared_distance_1d", "block"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--return-local-transport", action="store_true")
    return parser.parse_args()


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    args = parse_args()

    if rank == 0:
        cost = generate_cost_matrix(args.rows, args.cols, mode=args.cost_mode, seed=args.seed)
    else:
        cost = None

    local_cost = scatter_row_blocks(comm, cost)
    start, end = split_indices(args.rows, size)[rank]
    local_a = np.full(end - start, 1.0 / args.rows, dtype=np.float64)
    b = np.full(args.cols, 1.0 / args.cols, dtype=np.float64)

    result = sinkhorn_knopp_distributed(
        comm,
        local_cost=local_cost,
        local_a=local_a,
        b=b,
        epsilon=args.epsilon,
        max_iters=args.max_iters,
        tol=args.tol,
        check_every=args.check_every,
        return_local_transport=args.return_local_transport,
    )
    hostnames = gather_hostnames(comm)

    if rank == 0:
        output = {
            "algorithm": "sinkhorn_mpi",
            "rows": args.rows,
            "cols": args.cols,
            "cost_mode": args.cost_mode,
            "seed": args.seed,
            "epsilon": args.epsilon,
            "max_iters": args.max_iters,
            "tol": args.tol,
            "check_every": args.check_every,
            "num_processes": size,
            "runtime_sec": result["runtime_sec"],
            "iterations": result["iterations"],
            "row_error": result["row_error"],
            "col_error": result["col_error"],
            "transport_objective": result["transport_objective"],
            "hostnames": hostnames or [],
        }
        save_json(args.output, output)
        print(
            "MPI Sinkhorn complete: "
            f"runtime={output['runtime_sec']:.6f}s, "
            f"iterations={output['iterations']}, "
            f"row_error={output['row_error']:.3e}, "
            f"col_error={output['col_error']:.3e}, "
            f"objective={output['transport_objective']:.12f}, "
            f"processes={size}",
            flush=True,
        )


if __name__ == "__main__":
    main()
