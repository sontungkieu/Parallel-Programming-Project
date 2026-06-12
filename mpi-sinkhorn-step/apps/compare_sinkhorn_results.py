from __future__ import annotations

import argparse

from mpi_sinkhorn.mpi_utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dense and MPI Sinkhorn result JSON files.")
    parser.add_argument("--sequential", required=True)
    parser.add_argument("--parallel", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequential = load_json(args.sequential)
    parallel = load_json(args.parallel)

    sequential_runtime = float(sequential["runtime_sec"])
    parallel_runtime = float(parallel["runtime_sec"])
    num_processes = int(parallel["num_processes"])
    speedup = sequential_runtime / parallel_runtime if parallel_runtime > 0.0 else float("inf")
    efficiency = speedup / num_processes if num_processes > 0 else float("nan")
    objective_diff = abs(
        float(sequential["transport_objective"]) - float(parallel["transport_objective"])
    )

    print(f"Sequential runtime: {sequential_runtime:.6f} sec")
    print(f"MPI runtime: {parallel_runtime:.6f} sec")
    print(f"Speedup: {speedup:.4f}x")
    print(f"Efficiency: {efficiency:.4f}")
    print(f"Sequential row error: {float(sequential['row_error']):.6e}")
    print(f"Sequential col error: {float(sequential['col_error']):.6e}")
    print(f"MPI row error: {float(parallel['row_error']):.6e}")
    print(f"MPI col error: {float(parallel['col_error']):.6e}")
    print(f"Sequential objective: {float(sequential['transport_objective']):.12f}")
    print(f"MPI objective: {float(parallel['transport_objective']):.12f}")
    print(f"Objective difference: {objective_diff:.6e}")


if __name__ == "__main__":
    main()
