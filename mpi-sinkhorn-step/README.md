# MPI Sinkhorn Step MVP

Solver-only MVP for a Parallel and Distributed Programming project.

The implemented core is a real Sinkhorn-Knopp solver for entropy-regularized optimal transport:

```text
K = exp(-C / epsilon)
u = a / (K v)
v = b / (K^T u)
P = diag(u) K diag(v)
```

The MPI version partitions the cost matrix by rows. Each rank computes local matrix-vector work, then uses `MPI.Allreduce` for the global `K.T @ u` update and convergence metrics.

For detailed VM setup, SSH connection, hostfile, and cluster run commands, read:

- [Python MPI cluster runbook](PYTHON_CLUSTER_RUNBOOK.md)

## Install

On every machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If `python3 -m venv` is unavailable on Ubuntu, install it first:

```bash
sudo apt install -y python3-venv
```

For development and tests:

```bash
pip install -e ".[dev]"
```

## Local Smoke

```bash
python apps/sequential_sinkhorn_runner.py \
  --rows 128 \
  --cols 128 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 1e-6 \
  --output results/smoke_seq.json

mpirun -np 3 python apps/mpi_sinkhorn_runner.py \
  --rows 128 \
  --cols 128 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 1e-6 \
  --output results/smoke_mpi_np3.json

python apps/compare_sinkhorn_results.py \
  --sequential results/smoke_seq.json \
  --parallel results/smoke_mpi_np3.json
```

## Cluster Smoke

```bash
mpirun -np 6 --hostfile hosts python apps/hello_mpi.py

mpirun -np 6 --hostfile hosts python apps/mpi_sinkhorn_runner.py \
  --rows 128 \
  --cols 128 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --output results/smoke_cluster_np6.json
```

## Demo Preset

```bash
python apps/sequential_sinkhorn_runner.py \
  --rows 1000 \
  --cols 1000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.1 \
  --max-iters 100 \
  --output results/demo_sinkhorn_seq.json

mpirun -np 6 --hostfile hosts python apps/mpi_sinkhorn_runner.py \
  --rows 1000 \
  --cols 1000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.1 \
  --max-iters 100 \
  --output results/demo_sinkhorn_mpi_np6.json

python apps/compare_sinkhorn_results.py \
  --sequential results/demo_sinkhorn_seq.json \
  --parallel results/demo_sinkhorn_mpi_np6.json
```

## Test

```bash
pytest
```

If the package is not installed yet, local checks can also be run with:

```bash
PYTHONPATH=src python3 -m pytest -q
```

## Output Metrics

Both sequential and MPI runners write JSON with:

- `runtime_sec`
- `iterations`
- `row_error`
- `col_error`
- `transport_objective`
- `num_processes`
- `hostnames`
