from __future__ import annotations

import argparse
import socket

from mpi_sinkhorn.mpi_utils import save_json
from mpi_sinkhorn.sinkhorn import generate_cost_matrix, sinkhorn_knopp_dense


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense Sinkhorn-Knopp solver.")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--cost-mode", default="random", choices=["random", "squared_distance_1d", "block"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--return-transport", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cost = generate_cost_matrix(args.rows, args.cols, mode=args.cost_mode, seed=args.seed)
    result = sinkhorn_knopp_dense(
        cost,
        epsilon=args.epsilon,
        max_iters=args.max_iters,
        tol=args.tol,
        check_every=args.check_every,
        return_transport=args.return_transport,
    )

    output = {
        "algorithm": "sinkhorn_dense",
        "rows": args.rows,
        "cols": args.cols,
        "cost_mode": args.cost_mode,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "max_iters": args.max_iters,
        "tol": args.tol,
        "check_every": args.check_every,
        "num_processes": 1,
        "runtime_sec": result["runtime_sec"],
        "iterations": result["iterations"],
        "row_error": result["row_error"],
        "col_error": result["col_error"],
        "transport_objective": result["transport_objective"],
        "hostnames": [socket.gethostname()],
    }
    save_json(args.output, output)
    print(
        "Dense Sinkhorn complete: "
        f"runtime={output['runtime_sec']:.6f}s, "
        f"iterations={output['iterations']}, "
        f"row_error={output['row_error']:.3e}, "
        f"col_error={output['col_error']:.3e}, "
        f"objective={output['transport_objective']:.12f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
