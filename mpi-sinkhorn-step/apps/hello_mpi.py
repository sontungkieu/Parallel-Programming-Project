from __future__ import annotations

import socket

from mpi4py import MPI


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Hello from rank {rank}/{size} on {socket.gethostname()}", flush=True)


if __name__ == "__main__":
    main()
