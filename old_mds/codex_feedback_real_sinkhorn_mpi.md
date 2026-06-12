# Codex Feedback — Narrow Real-Sinkhorn MPI MVP First

## 0. Purpose

This file is feedback for Codex/local agent after reviewing the current project handoff and the follow-up plan.

The current direction is good, but the implementation scope must be narrowed first. Do not try to implement the whole optimizer, benchmark suite, plotting, and report at once. The first deliverable must be a correct and runnable MPI row-partitioned Sinkhorn-Knopp solver.

Project direction:

```text
Distributed Entropy-Regularized Optimal Transport using MPI,
applied later to a Sinkhorn-Step-inspired optimizer.
```

The most important point:

```text
Do not implement only softmin and call it Sinkhorn.
Softmin is only a fallback/baseline.
The core project must include a real Sinkhorn-Knopp solver.
```

---

## 1. Main Feedback

The original handoff is detailed, but too broad for a first implementation pass. Codex should not start by generating all files and all features. That will likely produce many incomplete modules.

Instead, implement a solver-only MVP first:

```text
1. hello_mpi.py
2. dense Sinkhorn-Knopp solver
3. MPI row-partitioned Sinkhorn-Knopp solver
4. correctness comparison
5. speedup/efficiency benchmark
```

Only after this works, implement the optimizer application.

The solver-only demo is already a valid Parallel Computing project because it contains:

```text
- distributed matrix-vector multiplication
- repeated MPI Allreduce communication
- correctness metrics
- runtime benchmark
- speedup and efficiency analysis
```

The optimizer should be treated as an application layer, not the first thing to build.

---

## 2. MVP Priority

Implement these files first:

```text
mpi-sinkhorn-step/
├── pyproject.toml
├── requirements.txt
├── src/mpi_sinkhorn/__init__.py
├── src/mpi_sinkhorn/sinkhorn.py
├── src/mpi_sinkhorn/mpi_utils.py
├── apps/hello_mpi.py
├── apps/sequential_sinkhorn_runner.py
├── apps/mpi_sinkhorn_runner.py
├── apps/compare_sinkhorn_results.py
└── tests/test_sinkhorn.py
```

Do not implement these until the solver-only MVP passes:

```text
src/mpi_sinkhorn/objectives.py
src/mpi_sinkhorn/directions.py
src/mpi_sinkhorn/optimizer.py
apps/sequential_runner.py
apps/mpi_multiseed_runner.py
apps/mpi_distributed_probe_runner.py
apps/compare_results.py
src/mpi_sinkhorn/plotting.py
apps/plot_results.py
```

---

## 3. Deterministic Correctness Requirement

Sequential and MPI runners must solve the exact same OT problem.

Given the same:

```text
--rows
--cols
--cost-mode
--seed
--epsilon
--max-iters
--tol
```

the generated cost matrix must be identical.

Implement a shared deterministic cost generator:

```python
def generate_cost_matrix(rows: int, cols: int, mode: str = "random", seed: int = 0) -> np.ndarray:
    ...
```

Supported modes:

```text
random
squared_distance_1d
block
```

Recommended behavior:

```text
MVP:
    Rank 0 generates the full cost matrix and scatters row blocks.

Extension:
    Each rank deterministically generates only its own local rows using global row indices.
```

Do not let each rank independently call random generation without coordination. That makes the sequential-vs-MPI comparison meaningless.

---

## 4. Dense Sinkhorn Requirements

Implement this API in `src/mpi_sinkhorn/sinkhorn.py`:

```python
def sinkhorn_knopp_dense(
    cost: np.ndarray,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    epsilon: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-6,
    check_every: int = 10,
    return_transport: bool = False,
) -> dict:
    ...
```

It solves:

```text
min_P <C, P> + epsilon * sum_ij P_ij (log P_ij - 1)
subject to:
    P 1 = a
    P^T 1 = b
    P_ij >= 0
```

Use Sinkhorn-Knopp scaling:

```text
K = exp(-C / epsilon)
u = a / (K v)
v = b / (K^T u)
P = diag(u) K diag(v)
```

Numerical stabilization:

```python
K = np.exp(-(cost - cost.min()) / epsilon)
```

Add a tiny constant in denominators:

```python
tiny = 1e-12
```

Return at least:

```python
{
    "transport": P if return_transport else None,
    "u": u,
    "v": v,
    "iterations": iterations,
    "runtime_sec": runtime_sec,
    "row_error": row_error,
    "col_error": col_error,
    "transport_objective": transport_objective,
}
```

Validation required:

```text
- cost is 2D
- all cost entries are finite
- epsilon > 0
- a and b are positive
- sum(a) and sum(b) match
```

Default marginals should be uniform if `a` or `b` is not provided.

---

## 5. MPI Row-Partitioned Sinkhorn Requirements

Implement this API:

```python
def sinkhorn_knopp_distributed(
    comm,
    local_cost: np.ndarray,
    local_a: np.ndarray,
    b: np.ndarray,
    epsilon: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-6,
    check_every: int = 10,
    return_local_transport: bool = False,
) -> dict:
    ...
```

MPI decomposition:

```text
Rank r owns rows [start_r, end_r) of C.
Rank r stores:
    local_cost = C[start_r:end_r, :]
    local_a = a[start_r:end_r]

All ranks store:
    b
    v
```

Every iteration:

```text
1. local_Kv = local_K @ v
2. local_u = local_a / (local_Kv + tiny)
3. local_KTu = local_K.T @ local_u
4. global_KTu = Allreduce(SUM, local_KTu)
5. v = b / (global_KTu + tiny)
```

Important stabilization detail:

```text
Dense solver uses cost.min().
Distributed solver must use global min across all local_cost blocks.
```

So distributed solver should do:

```python
local_min = np.array(local_cost.min(), dtype=float)
global_min = np.array(0.0, dtype=float)
comm.Allreduce(local_min, global_min, op=MPI.MIN)
local_K = np.exp(-(local_cost - float(global_min)) / epsilon)
```

---

## 6. Correct Distributed Error Computation

Do not compute marginal errors locally and call them global errors.

At convergence check:

```python
local_P = (local_u[:, None] * local_K) * v[None, :]
```

Row error:

```python
local_row_error = np.abs(local_P.sum(axis=1) - local_a).sum()
row_error = comm.allreduce(local_row_error, op=MPI.SUM)
```

Column error:

```python
local_col_mass = local_P.sum(axis=0)
global_col_mass = np.empty_like(local_col_mass)
comm.Allreduce(local_col_mass, global_col_mass, op=MPI.SUM)
col_error = np.abs(global_col_mass - b).sum()
```

Transport objective:

```python
local_objective = np.sum(local_P * local_cost)
transport_objective = comm.allreduce(local_objective, op=MPI.SUM)
```

These three metrics are required:

```text
row_error
col_error
transport_objective
```

They allow us to compare sequential and MPI results.

---

## 7. Do Not Gather Full Transport Matrix for Large Runs

For small correctness tests, gathering full `P` is acceptable:

```text
16x16
64x64
128x128
```

For benchmark/demo sizes larger than `512x512`, do not gather full `P` by default.

For large runs, only compute scalar metrics:

```text
runtime_sec
iterations
row_error
col_error
transport_objective
```

This avoids unnecessary memory and network overhead.

---

## 8. Runner Requirements

### 8.1 `apps/sequential_sinkhorn_runner.py`

Required behavior:

```text
- parse CLI arguments
- generate deterministic cost matrix
- run dense Sinkhorn
- save JSON result
```

Required args:

```text
--rows
--cols
--cost-mode
--seed
--epsilon
--max-iters
--tol
--check-every
--output
```

### 8.2 `apps/mpi_sinkhorn_runner.py`

Required behavior:

```text
- parse CLI args on rank 0
- broadcast config to all ranks
- rank 0 generates full deterministic cost matrix for MVP
- scatter row blocks to ranks
- construct local_a on each rank
- all ranks run sinkhorn_knopp_distributed
- rank 0 saves JSON result
```

Implementation detail:

```text
Use row partitioning helpers in mpi_utils.py.
For MVP, Python-level scatter/gather is acceptable.
Optimization with Scatterv/Gatherv can be added later.
```

### 8.3 `apps/compare_sinkhorn_results.py`

Required behavior:

```text
- load sequential JSON
- load MPI JSON
- print runtime comparison
- print speedup
- print efficiency
- print row/column errors
- print objective difference
```

Speedup:

```python
speedup = sequential_runtime / mpi_runtime
```

Efficiency:

```python
efficiency = speedup / num_processes
```

---

## 9. JSON Output Schema

All solver runners should save JSON in the same shape:

```json
{
  "algorithm": "sinkhorn_dense_or_mpi",
  "rows": 1000,
  "cols": 1000,
  "cost_mode": "random",
  "seed": 0,
  "epsilon": 0.1,
  "max_iters": 200,
  "tol": 1e-6,
  "check_every": 10,
  "num_processes": 1,
  "runtime_sec": 0.0,
  "iterations": 0,
  "row_error": 0.0,
  "col_error": 0.0,
  "transport_objective": 0.0,
  "hostnames": []
}
```

Do not invent different JSON shapes per runner.

---

## 10. Packaging Requirement

Add `pyproject.toml` so the package can be installed on all workers with:

```bash
pip install -e .
```

This avoids errors like:

```text
ModuleNotFoundError: No module named 'mpi_sinkhorn'
```

Minimal `pyproject.toml`:

```toml
[project]
name = "mpi-sinkhorn-step"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

---

## 11. Smoke Commands First

Make these commands pass before attempting large benchmarks or optimizer code.

Sequential smoke:

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
```

MPI smoke on local machine:

```bash
mpirun -np 3 python apps/mpi_sinkhorn_runner.py \
  --rows 128 \
  --cols 128 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 1e-6 \
  --output results/smoke_mpi_np3.json
```

Compare:

```bash
python apps/compare_sinkhorn_results.py \
  --sequential results/smoke_seq.json \
  --parallel results/smoke_mpi_np3.json
```

Then cluster smoke:

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

---

## 12. Demo Presets

Use three levels.

### Smoke

```text
rows = 128
cols = 128
max_iters = 50
epsilon = 0.5
```

Purpose:

```text
fast correctness check
```

### Demo

```text
rows = 1000
cols = 1000
max_iters = 100
epsilon = 0.5 or 0.1
```

Purpose:

```text
class presentation/offline demo
```

### Report

```text
rows = 3000
cols = 3000
max_iters = 200
epsilon = 0.5 or 0.1
```

Purpose:

```text
benchmark tables and speedup plots
```

---

## 13. Optimizer Comes After Solver MVP

Only after solver-only MVP works, implement:

```text
src/mpi_sinkhorn/objectives.py
src/mpi_sinkhorn/directions.py
src/mpi_sinkhorn/optimizer.py
apps/sequential_runner.py
apps/mpi_multiseed_runner.py
apps/mpi_distributed_probe_runner.py
apps/compare_results.py
```

Optimizer modes:

```text
--weight-mode softmin
--weight-mode sinkhorn
```

For `sinkhorn` mode:

```text
C[i, j] = f(x_i + probe_radius * d_j)
P = Sinkhorn(C, a, b)
W[i, j] = P[i, j] / sum_j P[i, j]
update_dir[i] = sum_j W[i, j] * D[j]
```

Fallback strategy:

```text
If optimizer sinkhorn mode fails:
    demo solver-only MPI Sinkhorn + optimizer softmin mode.

If distributed probe optimizer fails:
    demo mpi_multiseed_runner.py.
```

---

## 14. Minimal Definition of Done for MVP

The MVP is done when:

```text
- mpirun hello_mpi.py works locally.
- mpirun hello_mpi.py works on 3-machine cluster.
- sequential_sinkhorn_runner.py saves valid JSON.
- mpi_sinkhorn_runner.py saves valid JSON.
- compare_sinkhorn_results.py shows similar objective and marginal errors.
- MPI with np=1 matches dense solver closely.
- MPI with np=3 runs successfully.
- MPI with np=6 runs successfully on the cluster.
- Results include runtime, speedup, efficiency, row_error, col_error, and transport_objective.
```

After this, continue to the optimizer and plotting/report features.

---

## 15. Final Advice for Codex

Implementation priority:

```text
Correctness first.
Then cluster execution.
Then benchmark.
Then optimizer.
Then plots/report polish.
```

Avoid over-engineering. The project grade will benefit more from a reliable 3-machine MPI demo and a clearly explained distributed Sinkhorn algorithm than from many unfinished advanced features.
