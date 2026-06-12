# MPI Parallel Computing Project — Distributed Sinkhorn OT and Sinkhorn-Step Optimizer

## 0. Context

This project is for a Parallel Computing course. The course requirement is:

- Form a group of 4 people.
- Set up an MPI cluster with at least 3 physical machines.
- Each physical Windows/Mac/Linux machine should run at most one Ubuntu VM if virtualization is used.
- Ubuntu VMs should be connected through a bridged network or the same LAN/hotspot.
- Implement a parallel algorithm on the cluster.
- Demo can be offline.
- A report must be submitted.
- The code must contain at least 250 lines per person, approximately 1000 lines for a group of 4.
- Grading depends on topic interest, parallelization quality, whether the demo runs, report quality, and whether all members understand the code.

The project selected here is:

> **Distributed Entropy-Regularized Optimal Transport using MPI, applied to a Sinkhorn-Step-inspired optimizer**

The project is inspired by `anindex/ssax`, a JAX implementation of Sinkhorn Step optimizer. However, for course reliability, the MPI project should implement a simplified, self-contained version using `mpi4py`, `numpy`, and optionally `scipy/matplotlib`, rather than depending heavily on JAX internals.

The key idea is to implement a real Sinkhorn-Knopp solver for entropy-regularized optimal transport, parallelize its repeated matrix-vector work with MPI, and then use that solver as the weighting mechanism inside a gradient-free optimizer.

---

## 1. Final Goal

Build a working MPI cluster on 3 physical machines and run a distributed Sinkhorn optimal transport solver that demonstrates speedup compared with a sequential baseline. Then use the solver inside a Sinkhorn-Step-inspired black-box optimizer.

The final demo should show:

1. MPI cluster test across 3 physical machines.
2. Sequential Sinkhorn-Knopp solver baseline.
3. MPI row-partitioned Sinkhorn-Knopp solver.
4. Sequential optimizer baseline with `softmin` and `sinkhorn` weighting modes.
5. MPI distributed optimizer using Sinkhorn weighting.
6. Runtime comparison.
7. Speedup and parallel efficiency.
8. Convergence plots and result tables.
9. Explanation of how the Sinkhorn algorithm and optimizer are parallelized.

---

## 2. Recommended Project Scope

### 2.1 Main Algorithm

Implement a simplified Sinkhorn-Step-style gradient-free optimizer.

At every iteration:

1. Maintain a population of candidate points `X`, shape `(num_points, dim)`.
2. Generate probe directions `D`, shape `(num_dirs, dim)`.
3. For each candidate `x_i` and direction `d_j`, evaluate:

   ```text
   f(x_i + probe_radius * d_j)
   ```

4. Use the evaluated costs to compute a weighted direction for each candidate.
5. Move candidates toward lower-cost directions.
6. Track best solution found.

The real Sinkhorn Step uses entropy-regularized optimal transport. For this project, implement two modes:

- `softmin` mode: easier, robust, course-friendly fallback.
- `sinkhorn` mode: required main mode using a real Sinkhorn-Knopp matrix normalization routine.

The MPI parallelization has two layers:

1. Real Sinkhorn OT solver:

   ```text
   distributed matrix-vector products inside Sinkhorn iterations
   ```

2. Optimizer:

   ```text
   cost evaluation over num_points * num_dirs probe points
   ```

The solver layer is more important academically because it uses repeated distributed communication, not just independent embarrassingly parallel jobs.

### 2.2 Real Sinkhorn Implementation Plan

This section is the most important planning note for anyone continuing from this handoff. Do not implement only `softmin` and claim it is real Sinkhorn. `softmin` is useful as a fallback, but the main algorithm should include a real Sinkhorn-Knopp optimal transport solver.

#### 2.2.1 Mathematical target

Solve the entropy-regularized optimal transport problem:

```text
min_P <C, P> + epsilon * sum_ij P_ij (log P_ij - 1)
subject to:
    P 1 = a
    P^T 1 = b
    P_ij >= 0
```

Where:

- `C` is the cost matrix, shape `(n, m)`.
- `a` is the source marginal distribution, shape `(n,)`.
- `b` is the target marginal distribution, shape `(m,)`.
- `sum(a) = sum(b) = 1`.
- `epsilon` controls entropy regularization.
- `P` is the transport plan, shape `(n, m)`.

Sinkhorn-Knopp solves this through matrix scaling:

```text
K = exp(-C / epsilon)
u = a / (K v)
v = b / (K^T u)
P = diag(u) K diag(v)
```

The implementation should report convergence using marginal errors:

```text
row_error = ||P 1 - a||_1
col_error = ||P^T 1 - b||_1
```

#### 2.2.2 Sequential solver first

Implement a dense NumPy solver before writing MPI code.

Required behavior:

- Accept `cost`, optional `a`, optional `b`, `epsilon`, `max_iters`, `tol`.
- Default to uniform marginals if `a` or `b` is not provided.
- Return `transport`, `u`, `v`, final errors, number of iterations, runtime.
- Validate shape, positivity, finite values, and matching total mass.
- Stop early when both row and column errors are below `tol`.

Recommended API:

```python
def sinkhorn_knopp_dense(
    cost: np.ndarray,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    epsilon: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-6,
    check_every: int = 10,
) -> dict:
    ...
```

Numerical note:

- Start with the standard `K = exp(-cost / epsilon)` implementation.
- Stabilize by subtracting `cost.min()` before exponentiation if needed.
- Add a small constant such as `1e-12` to denominators.
- Keep log-domain Sinkhorn as an optional extension only if underflow becomes a problem.

#### 2.2.3 MPI row-partitioned solver

Use row partitioning because it is simple to explain and maps well to MPI:

```text
Rank r owns rows [start_r, end_r) of C.
Rank r stores:
    local_cost = C[start_r:end_r, :]
    local_a = a[start_r:end_r]
All ranks store:
    b
    v
```

At each Sinkhorn iteration:

```text
1. Each rank computes local_Kv = K_local @ v.
2. Each rank updates local_u = local_a / local_Kv.
3. Each rank computes local_KTu = K_local.T @ local_u.
4. All ranks call Allreduce(sum) to form global_KTu.
5. All ranks update v = b / global_KTu.
6. Every check_every iterations, compute marginal errors with Allreduce.
```

This is a strong MPI story because every Sinkhorn iteration performs distributed matrix-vector multiplication and global reduction.

Recommended API:

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
) -> dict:
    ...
```

Return on every rank:

```python
{
    "local_transport": local_P,
    "local_u": local_u,
    "v": v,
    "iterations": iters,
    "row_error": row_error,
    "col_error": col_error,
    "runtime_sec": runtime,
}
```

Rank 0 may additionally gather `local_transport` for correctness tests and plotting, but large benchmark runs should avoid gathering the full matrix unless needed.

#### 2.2.4 Correctness tests

Before integrating with the optimizer, verify the solver itself:

- Sequential `P` has row sums close to `a`.
- Sequential `P` has column sums close to `b`.
- MPI `local_P` gathered on rank 0 is close to sequential `P`.
- MPI final objective value is close to sequential final objective value.
- Results are stable across `np=1`, `np=3`, `np=6`.

Suggested test matrix sizes:

```text
small: 16 x 16
medium: 256 x 256
large: 1000 x 1000 or larger for benchmarks
```

#### 2.2.5 Optimizer integration

After the solver works, use it inside the optimizer.

At each optimizer iteration:

```text
C[i, j] = f(x_i + probe_radius * d_j)
```

Then compute a transport plan:

```text
a = uniform distribution over candidate points
b = uniform distribution over directions
P = Sinkhorn(C, a, b)
```

For candidate updates, convert the transport plan into per-candidate conditional direction weights:

```text
W[i, j] = P[i, j] / sum_j P[i, j]
update_dir[i] = sum_j W[i, j] * D[j]
X[i] = X[i] + step_radius * update_dir[i]
```

This gives a real Sinkhorn-based weighting rule while keeping the optimizer understandable. The marginal `b` also encourages the optimizer not to collapse all candidates onto the same direction too early.

The optimizer should support:

```text
--weight-mode softmin
--weight-mode sinkhorn
```

Use `softmin` as a backup demo if the full Sinkhorn mode has numerical issues during presentation.

#### 2.2.6 Implementation order

Implement a solver-only MVP first. Do not start by generating all optimizer, plotting, benchmark, and report modules because that will likely produce many incomplete files.

MVP file list:

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

MVP order:

1. Package skeleton with `pyproject.toml`, `requirements.txt`, and `pip install -e .` support.
2. `apps/hello_mpi.py`.
3. Deterministic cost generator.
4. `sinkhorn_knopp_dense` with tests.
5. `apps/sequential_sinkhorn_runner.py`.
6. `sinkhorn_knopp_distributed` with tests on `mpirun -np 1`, `np=3`, and `np=6`.
7. `apps/mpi_sinkhorn_runner.py`.
8. `apps/compare_sinkhorn_results.py`.
9. Solver-only benchmark summary.

After the MVP passes:

1. Sequential optimizer with `softmin` fallback.
2. Sequential optimizer with `sinkhorn` weighting.
3. MPI multiseed fallback optimizer.
4. MPI distributed-probe optimizer with `sinkhorn` weighting.
5. Plotting, report figures, README polish.

The solver-only demo should be treated as the primary scientific deliverable. The optimizer is the application layer.

### 2.3 Optimization Problems

After the Sinkhorn solver is implemented, add benchmark objective functions for the optimizer application:

Implement benchmark objective functions:

- Ackley
- Rastrigin
- Rosenbrock
- Styblinski-Tang
- Sphere
- Griewank

Use at least 3 objective functions in the final report.

Recommended final benchmark:

| Difficulty | Objective | Dimension | Points | Iterations |
|---|---:|---:|---:|---:|
| Easy | Ackley | 2 | 1000 | 30 |
| Medium | Rastrigin | 10 | 3000 | 50 |
| Hard | Ackley | 20 | 5000 | 80 |

---

## 3. Why This Is a Good Parallel Computing Project

The project is suitable because:

1. The objective function is black-box and expensive when evaluated many times.
2. Each probe evaluation is independent.
3. MPI can divide probe evaluations across ranks.
4. The program uses real distributed-memory communication primitives.
5. The project can measure speedup, efficiency, load balance, and communication overhead.
6. The project can be demonstrated on 3 physical machines.

Parallel pattern:

```text
Master rank:
    generate candidate points X
    generate directions D
    split probe jobs
    broadcast metadata
    gather costs
    update candidates
    log best result

Worker ranks:
    receive assigned probe jobs
    evaluate objective values
    return local cost results
```

MPI primitives to use:

- `MPI.COMM_WORLD.Get_rank()`
- `MPI.COMM_WORLD.Get_size()`
- `comm.bcast()`
- `comm.scatter()` or `comm.Scatterv()`
- `comm.gather()` or `comm.Gatherv()`
- `comm.reduce()` or `comm.allreduce()`
- `comm.Barrier()`

---

## 4. Repository Structure to Create

Create a new repository or project directory:

```text
mpi-sinkhorn-step/
├── pyproject.toml
├── README.md
├── requirements.txt
├── hosts.example
├── scripts/
│   ├── setup_ubuntu.sh
│   ├── test_mpi.sh
│   ├── run_sequential.sh
│   ├── run_mpi_local.sh
│   ├── run_mpi_cluster.sh
│   └── collect_results.sh
├── src/
│   └── mpi_sinkhorn/
│       ├── __init__.py
│       ├── objectives.py
│       ├── directions.py
│       ├── sinkhorn.py
│       ├── optimizer.py
│       ├── mpi_utils.py
│       ├── benchmark.py
│       ├── plotting.py
│       └── logging_utils.py
├── apps/
│   ├── sequential_runner.py
│   ├── mpi_multiseed_runner.py
│   ├── mpi_distributed_probe_runner.py
│   └── plot_results.py
├── configs/
│   ├── ackley_2d.json
│   ├── rastrigin_10d.json
│   └── ackley_20d.json
├── results/
│   └── .gitkeep
├── report/
│   ├── figures/
│   ├── tables/
│   └── report_outline.md
└── tests/
    ├── test_objectives.py
    ├── test_directions.py
    ├── test_sinkhorn.py
    └── test_optimizer.py
```

---

## 5. Python Dependencies

Add `pyproject.toml` so every worker can install the package consistently:

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

Install locally on every machine:

```bash
pip install -e .
```

`requirements.txt`:

```text
numpy>=1.24
mpi4py>=3.1
matplotlib>=3.7
pandas>=2.0
pytest>=7.0
tqdm>=4.65
```

Optional:

```text
scipy>=1.10
```

Avoid using JAX for the first working version. JAX can be added later as an optional acceleration mode, but the core demo should run on CPU reliably.

---

## 6. MPI Cluster Setup — 3 Physical Machines

Assume there are 3 machines:

```text
master  = 192.168.43.101
worker1 = 192.168.43.102
worker2 = 192.168.43.103
```

Each machine should run Ubuntu directly or one Ubuntu VM with bridged adapter.

### 6.1 Install packages on all machines

Run on every Ubuntu machine:

```bash
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev openssh-server python3-pip python3-venv git rsync htop
```

### 6.2 Create same user on all machines

Recommended username:

```text
mpiuser
```

If the current user is different, that is fine, but it must be consistent across machines or SSH config must handle usernames.

### 6.3 Enable SSH server

On all machines:

```bash
sudo systemctl enable ssh
sudo systemctl start ssh
sudo systemctl status ssh
```

### 6.4 Check IP addresses

On each machine:

```bash
ip addr
hostname -I
```

Make sure every machine can ping the others:

```bash
ping 192.168.43.101
ping 192.168.43.102
ping 192.168.43.103
```

### 6.5 SSH key setup from master

On master:

```bash
ssh-keygen -t ed25519 -C "mpi-cluster" -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id mpiuser@192.168.43.101
ssh-copy-id mpiuser@192.168.43.102
ssh-copy-id mpiuser@192.168.43.103
```

Test:

```bash
ssh mpiuser@192.168.43.102 hostname
ssh mpiuser@192.168.43.103 hostname
```

### 6.6 Create hostfile

`hosts`:

```text
192.168.43.101 slots=2
192.168.43.102 slots=2
192.168.43.103 slots=2
```

If each VM has 4 CPU cores, use:

```text
192.168.43.101 slots=4
192.168.43.102 slots=4
192.168.43.103 slots=4
```

### 6.7 Test MPI cluster

On master:

```bash
mpirun -np 3 --hostfile hosts hostname
```

Expected output should contain hostnames from all 3 machines.

More explicit:

```bash
mpirun -np 6 --hostfile hosts --map-by slot hostname
```

### 6.8 Fix common OpenMPI root issue

Do not run as root. If unavoidable for quick test only:

```bash
mpirun --allow-run-as-root -np 3 --hostfile hosts hostname
```

But for final demo, use normal user.

### 6.9 Fix firewall issues

For local lab network only, temporarily disable firewall:

```bash
sudo ufw disable
```

Or allow SSH:

```bash
sudo ufw allow ssh
sudo ufw allow from 192.168.43.0/24
```

---

## 7. Implementation Details

Important ordering note: this section describes all modules, but it is not the implementation order. Follow the solver-only MVP order in Section 2.2.6 and Section 17 first. `objectives.py`, `directions.py`, optimizer runners, plotting, and report automation come after the Sinkhorn solver MVP passes.

## 7.1 `objectives.py`

Implement benchmark objective functions.

Required API:

```python
def evaluate_objective(name: str, x: np.ndarray) -> np.ndarray:
    """
    x shape: (..., dim)
    return shape: (...,)
    """
```

Functions:

```python
def sphere(x):
    return np.sum(x ** 2, axis=-1)


def rastrigin(x):
    A = 10.0
    d = x.shape[-1]
    return A * d + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=-1)


def ackley(x):
    d = x.shape[-1]
    a = 20.0
    b = 0.2
    c = 2 * np.pi
    sum_sq = np.sum(x ** 2, axis=-1)
    sum_cos = np.sum(np.cos(c * x), axis=-1)
    return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.e


def rosenbrock(x):
    return np.sum(100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2, axis=-1)


def styblinski_tang(x):
    return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=-1)
```

Also implement bounds:

```python
OBJECTIVE_BOUNDS = {
    "sphere": (-5.0, 5.0),
    "rastrigin": (-5.12, 5.12),
    "ackley": (-32.768, 32.768),
    "rosenbrock": (-5.0, 10.0),
    "styblinski_tang": (-5.0, 5.0),
}
```

---

## 7.2 `directions.py`

Generate probe directions.

Required API:

```python
def make_directions(dim: int, mode: str = "orthoplex", rng=None) -> np.ndarray:
    """
    return D shape: (num_dirs, dim)
    """
```

Modes:

1. `orthoplex`: `+e_i` and `-e_i`, total `2 * dim` directions.
2. `random`: random Gaussian normalized directions.
3. `hypercube`: random subset of `{−1, +1}^dim`, normalized.

For the course demo, `orthoplex` is easiest to explain.

---

## 7.3 `sinkhorn.py`

Implement the required Sinkhorn-Knopp routines. This module is the mathematical core of the project, not an optional helper.

Sequential and MPI runners must solve the exact same OT problem when given the same CLI arguments. Implement a shared deterministic cost generator:

```python
def generate_cost_matrix(
    rows: int,
    cols: int,
    mode: str = "random",
    seed: int = 0,
) -> np.ndarray:
    ...
```

Required cost modes:

```text
random
squared_distance_1d
block
```

MVP behavior:

```text
Rank 0 generates the full cost matrix and scatters row blocks.
```

Extension:

```text
Each rank deterministically generates only its own local rows using global row indices.
```

Required API:

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
    """
    cost shape: (n, m)
    return transport plan, scalings, errors, iteration count, runtime.
    """
```

Core dense version:

```python
K = np.exp(-(cost - cost.min()) / epsilon)
u = np.ones_like(a)
v = np.ones_like(b)

for it in range(max_iters):
    u = a / (K @ v + 1e-12)
    v = b / (K.T @ u + 1e-12)

    if it % check_every == 0:
        P = (u[:, None] * K) * v[None, :]
        row_error = np.abs(P.sum(axis=1) - a).sum()
        col_error = np.abs(P.sum(axis=0) - b).sum()
        if row_error < tol and col_error < tol:
            break
```

MPI distributed API:

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
    """
    local_cost shape: (local_n, m)
    local_a shape: (local_n,)
    b shape: (m,)
    """
```

Distributed loop:

```python
local_cost_min = np.array(local_cost.min(), dtype=float)
global_cost_min = np.array(0.0, dtype=float)
comm.Allreduce(local_cost_min, global_cost_min, op=MPI.MIN)

local_K = np.exp(-(local_cost - global_cost_min) / epsilon)
v = np.ones_like(b)

for it in range(max_iters):
    local_u = local_a / (local_K @ v + 1e-12)
    local_KTu = local_K.T @ local_u
    global_KTu = np.empty_like(local_KTu)
    comm.Allreduce(local_KTu, global_KTu, op=MPI.SUM)
    v = b / (global_KTu + 1e-12)
```

Correct distributed convergence metrics:

```python
local_P = (local_u[:, None] * local_K) * v[None, :]

local_row_error = np.abs(local_P.sum(axis=1) - local_a).sum()
row_error = comm.allreduce(local_row_error, op=MPI.SUM)

local_col_mass = local_P.sum(axis=0)
global_col_mass = np.empty_like(local_col_mass)
comm.Allreduce(local_col_mass, global_col_mass, op=MPI.SUM)
col_error = np.abs(global_col_mass - b).sum()

local_objective = np.sum(local_P * local_cost)
transport_objective = comm.allreduce(local_objective, op=MPI.SUM)
```

Do not gather the full transport matrix for benchmark or demo sizes larger than `512x512` by default. Only gather full `P` for small correctness tests such as `16x16`, `64x64`, or `128x128`.

All solver runners should emit the same JSON shape:

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

Also keep `softmin_weights` as fallback:

```python
def softmin_weights(cost: np.ndarray, temperature: float) -> np.ndarray:
    z = -(cost - cost.min(axis=1, keepdims=True)) / temperature
    w = np.exp(z)
    return w / (w.sum(axis=1, keepdims=True) + 1e-12)
```

For final reporting, clearly separate:

- `softmin_weights`: fallback baseline, not real Sinkhorn OT.
- `sinkhorn_knopp_dense`: sequential real Sinkhorn baseline.
- `sinkhorn_knopp_distributed`: MPI real Sinkhorn implementation.

---

## 7.4 `optimizer.py`

Implement sequential optimizer.

Required API:

```python
class SinkhornLikeOptimizer:
    def __init__(
        self,
        objective: str,
        dim: int,
        num_points: int,
        max_iters: int,
        probe_radius: float,
        step_radius: float,
        temperature: float,
        sinkhorn_epsilon: float,
        weight_mode: str,
        direction_mode: str,
        seed: int,
    ):
        ...

    def run(self) -> dict:
        ...
```

Each iteration:

```python
D = make_directions(dim, mode=direction_mode)
for t in range(max_iters):
    probes = X[:, None, :] + probe_radius * D[None, :, :]
    costs = evaluate_objective(objective, probes.reshape(-1, dim))
    costs = costs.reshape(num_points, num_dirs)
    weights = make_weights(costs, mode=weight_mode)
    update_dirs = weights @ D
    X = X + step_radius * update_dirs
    X = np.clip(X, low, high)
    current_cost = evaluate_objective(objective, X)
    log best
```

Weighting behavior:

```text
weight_mode = softmin:
    weights = softmin_weights(costs, temperature)

weight_mode = sinkhorn:
    P = sinkhorn_knopp_dense(costs, epsilon=sinkhorn_epsilon)["transport"]
    weights = P / (P.sum(axis=1, keepdims=True) + 1e-12)
```

The `sinkhorn` mode is the main algorithmic mode. The `softmin` mode remains useful for debugging, fallback demo, and ablation comparison.

Return:

```python
{
    "objective": objective,
    "dim": dim,
    "num_points": num_points,
    "max_iters": max_iters,
    "best_x": best_x.tolist(),
    "best_cost": float(best_cost),
    "history": history,
    "runtime_sec": runtime,
}
```

---

## 7.5 `mpi_utils.py`

Implement helper functions:

```python
def split_indices(n_items: int, size: int) -> list[tuple[int, int]]:
    """Return start/end ranges for each rank."""


def get_rank_info(comm):
    return comm.Get_rank(), comm.Get_size(), socket.gethostname()


def save_json(path: str, data: dict):
    ...


def load_config(path: str) -> dict:
    ...
```

---

## 7.6 `mpi_distributed_probe_runner.py`

This is the optimizer application MPI version. The primary real Sinkhorn MPI demo is `mpi_sinkhorn_runner.py`; this runner shows how the Sinkhorn solver can be used inside a black-box optimizer.

High-level algorithm:

```text
All ranks start.
Rank 0 initializes candidate population X.
Rank 0 creates directions D.
Rank 0 broadcasts config, X, D.
At each iteration:
    Rank 0 creates all probe points or metadata.
    All ranks receive X and D.
    Each rank evaluates a slice of candidate indices.
    Rank local computes local_costs for assigned candidates.
    If weight_mode == softmin:
        Rank 0 gathers all local_costs.
        Rank 0 computes softmin weights.
    If weight_mode == sinkhorn:
        Ranks run sinkhorn_knopp_distributed on local_costs.
        Each rank computes local update directions from local transport rows.
        Rank 0 gathers local update directions.
    Rank 0 updates X.
    Rank 0 evaluates current candidate costs and logs best.
Rank 0 saves final result JSON.
```

Simpler split:

```text
split by candidate points, not by individual probe points
```

If `num_points = 3000` and `size = 6`:

```text
rank 0: candidates 0-499
rank 1: candidates 500-999
rank 2: candidates 1000-1499
rank 3: candidates 1500-1999
rank 4: candidates 2000-2499
rank 5: candidates 2500-2999
```

Each rank evaluates:

```python
local_X = X[start:end]
local_probes = local_X[:, None, :] + probe_radius * D[None, :, :]
local_costs = evaluate_objective(objective, local_probes.reshape(-1, dim))
local_costs = local_costs.reshape(end - start, num_dirs)
```

For `softmin` mode, gather:

```python
all_costs = comm.gather(local_costs, root=0)
if rank == 0:
    costs = np.concatenate(all_costs, axis=0)
```

For `sinkhorn` mode, avoid gathering the full cost matrix just to compute weights. Each rank should call the distributed Sinkhorn solver:

```python
local_a = np.full(end - start, 1.0 / num_points)
b = np.full(num_dirs, 1.0 / num_dirs)
sinkhorn_result = sinkhorn_knopp_distributed(
    comm,
    local_cost=local_costs,
    local_a=local_a,
    b=b,
    epsilon=sinkhorn_epsilon,
    max_iters=sinkhorn_iters,
    tol=sinkhorn_tol,
)
local_P = sinkhorn_result["local_transport"]
local_weights = local_P / (local_P.sum(axis=1, keepdims=True) + 1e-12)
local_update_dirs = local_weights @ D
```

Then rank 0 gathers `local_update_dirs` and updates `X`.

At the start of each iteration:

```python
X = comm.bcast(X, root=0)
D = comm.bcast(D, root=0)
```

This optimizer still centralizes candidate updates on rank 0, which is acceptable for the course project. The stronger distributed-memory contribution is inside the Sinkhorn solver's repeated `Allreduce`.

---

## 7.7 `mpi_multiseed_runner.py`

Implement a second MPI mode where each rank runs independent seeds.

This mode is easier and robust.

Algorithm:

```text
Rank 0 creates list of seeds.
Scatter seed chunks to ranks.
Each rank runs sequential optimizer for its assigned seeds.
Each rank returns best result.
Rank 0 selects global best.
```

This mode should be used as backup demo if distributed-probe mode has issues.

Command:

```bash
mpirun -np 6 --hostfile hosts python apps/mpi_multiseed_runner.py \
  --objective ackley \
  --dim 10 \
  --num-points 1000 \
  --max-iters 50 \
  --num-seeds 24 \
  --output results/mpi_multiseed_ackley10d.json
```

---

## 7.8 `benchmark.py`

Implement automated benchmarking.

Benchmark cases:

```text
sequential_np1
mpi_np3
mpi_np6
mpi_np9 or mpi_np12 if CPU cores allow
```

Metrics:

```text
runtime_sec
best_cost
speedup = sequential_runtime / parallel_runtime
efficiency = speedup / num_processes
num_points
dim
max_iters
objective
hostname list
```

Write results to CSV:

```text
results/benchmark_summary.csv
```

---

## 7.9 `plotting.py`

Generate plots:

1. Convergence curve:

   ```text
   iteration vs best_cost
   ```

2. Runtime bar chart:

   ```text
   number of processes vs runtime
   ```

3. Speedup chart:

   ```text
   number of processes vs speedup
   include ideal speedup line
   ```

4. Efficiency chart:

   ```text
   number of processes vs efficiency
   ```

5. Optional 2D landscape plot:

   For Ackley 2D, plot contour and best point.

---

## 8. CLI Requirements

Every app should support `--help`.

### 8.1 Solver-only runner requirements

`apps/sequential_sinkhorn_runner.py`:

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

`apps/mpi_sinkhorn_runner.py`:

```text
- parse CLI args on rank 0
- broadcast config to all ranks
- rank 0 generates full deterministic cost matrix for MVP
- scatter row blocks to ranks
- construct local_a on each rank
- all ranks run sinkhorn_knopp_distributed
- rank 0 saves JSON result
```

`apps/compare_sinkhorn_results.py`:

```text
- load sequential JSON
- load MPI JSON
- print runtime comparison
- print speedup
- print efficiency
- print row/column errors
- print objective difference
```

Speedup and efficiency:

```python
speedup = sequential_runtime / mpi_runtime
efficiency = speedup / num_processes
```

Example CLI for sequential Sinkhorn solver:

```bash
python apps/sequential_sinkhorn_runner.py \
  --rows 1000 \
  --cols 1000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.1 \
  --max-iters 200 \
  --tol 1e-6 \
  --check-every 10 \
  --output results/sequential_sinkhorn_1000x1000.json
```

Example CLI for MPI Sinkhorn solver:

```bash
mpirun -np 6 --hostfile hosts python apps/mpi_sinkhorn_runner.py \
  --rows 1000 \
  --cols 1000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.1 \
  --max-iters 200 \
  --tol 1e-6 \
  --check-every 10 \
  --output results/mpi_sinkhorn_1000x1000_np6.json
```

Example CLI for sequential optimizer:

```bash
python apps/sequential_runner.py \
  --objective ackley \
  --dim 10 \
  --num-points 3000 \
  --max-iters 50 \
  --probe-radius 0.2 \
  --step-radius 0.05 \
  --temperature 0.1 \
  --sinkhorn-epsilon 0.1 \
  --weight-mode sinkhorn \
  --seed 0 \
  --output results/sequential_ackley10d.json
```

Example CLI for MPI distributed probe:

```bash
mpirun -np 6 --hostfile hosts python apps/mpi_distributed_probe_runner.py \
  --objective ackley \
  --dim 10 \
  --num-points 3000 \
  --max-iters 50 \
  --probe-radius 0.2 \
  --step-radius 0.05 \
  --temperature 0.1 \
  --sinkhorn-epsilon 0.1 \
  --weight-mode sinkhorn \
  --seed 0 \
  --output results/mpi_probe_ackley10d_np6.json
```

Example CLI for plotting:

```bash
python apps/plot_results.py \
  --input results/benchmark_summary.csv \
  --output-dir report/figures
```

---

## 9. Demo Script

Final demo should follow this order.

### 9.1 Show physical cluster

```bash
cat hosts
mpirun -np 6 --hostfile hosts hostname
```

### 9.2 Run hello MPI

Create `apps/hello_mpi.py`:

```python
from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"Hello from rank {rank}/{size} on {socket.gethostname()}")
```

Run:

```bash
mpirun -np 6 --hostfile hosts python apps/hello_mpi.py
```

### 9.3 Run sequential Sinkhorn baseline

```bash
python apps/sequential_sinkhorn_runner.py \
  --rows 1000 \
  --cols 1000 \
  --cost-mode random \
  --epsilon 0.1 \
  --max-iters 200 \
  --output results/demo_sinkhorn_seq.json
```

### 9.4 Run MPI Sinkhorn solver

```bash
mpirun -np 6 --hostfile hosts python apps/mpi_sinkhorn_runner.py \
  --rows 1000 \
  --cols 1000 \
  --cost-mode random \
  --epsilon 0.1 \
  --max-iters 200 \
  --output results/demo_sinkhorn_mpi_np6.json
```

### 9.5 Compare Sinkhorn solver results

```bash
python apps/compare_sinkhorn_results.py \
  --sequential results/demo_sinkhorn_seq.json \
  --parallel results/demo_sinkhorn_mpi_np6.json
```

The output should show sequential runtime, MPI runtime, speedup, efficiency, row marginal error, column marginal error, and difference from the sequential objective value.

### 9.6 Run sequential optimizer baseline

```bash
python apps/sequential_runner.py \
  --objective ackley \
  --dim 10 \
  --num-points 3000 \
  --max-iters 50 \
  --weight-mode sinkhorn \
  --output results/demo_seq.json
```

### 9.7 Run MPI distributed probe optimizer

```bash
mpirun -np 6 --hostfile hosts python apps/mpi_distributed_probe_runner.py \
  --objective ackley \
  --dim 10 \
  --num-points 3000 \
  --max-iters 50 \
  --weight-mode sinkhorn \
  --output results/demo_mpi_np6.json
```

### 9.8 Show optimizer result comparison

```bash
python apps/compare_results.py \
  --sequential results/demo_seq.json \
  --parallel results/demo_mpi_np6.json
```

Output should show:

```text
Sequential runtime: xx.xx sec
MPI runtime: yy.yy sec
Speedup: zz.zz x
Efficiency: ee.ee
Best sequential cost: ...
Best MPI cost: ...
```

---

## 10. Member Task Division

### Member 1 — Cluster and MPI Infrastructure

Responsibilities:

- Set up Ubuntu/OpenMPI/SSH on 3 machines.
- Create `hosts` file.
- Write setup scripts.
- Write `hello_mpi.py`.
- Write `mpi_utils.py`.
- Document cluster setup.
- Take screenshots/photos of cluster test.

Expected code contribution:

```text
scripts/setup_ubuntu.sh
scripts/test_mpi.sh
src/mpi_sinkhorn/mpi_utils.py
apps/hello_mpi.py
README cluster section
```

### Member 2 — Objective Functions and Sequential Baseline

Responsibilities:

- Implement benchmark objective functions.
- Implement direction generation.
- Implement sequential dense Sinkhorn-Knopp solver.
- Implement `sequential_sinkhorn_runner.py`.
- Implement sequential optimizer.
- Add tests for objectives, directions, Sinkhorn, and optimizer.

Expected code contribution:

```text
src/mpi_sinkhorn/objectives.py
src/mpi_sinkhorn/directions.py
src/mpi_sinkhorn/sinkhorn.py
src/mpi_sinkhorn/optimizer.py
apps/sequential_sinkhorn_runner.py
apps/sequential_runner.py
tests/test_objectives.py
tests/test_directions.py
tests/test_sinkhorn.py
```

### Member 3 — MPI Distributed Algorithm

Responsibilities:

- Implement row-partitioned MPI Sinkhorn-Knopp solver.
- Implement `mpi_sinkhorn_runner.py`.
- Implement `compare_sinkhorn_results.py`.
- Implement `mpi_distributed_probe_runner.py`.
- Implement `mpi_multiseed_runner.py` backup version.
- Handle row splitting, broadcast, gather, and Allreduce.
- Ensure output correctness.
- Compare sequential and MPI results.

Expected code contribution:

```text
apps/mpi_sinkhorn_runner.py
apps/compare_sinkhorn_results.py
apps/mpi_distributed_probe_runner.py
apps/mpi_multiseed_runner.py
apps/compare_results.py
src/mpi_sinkhorn/sinkhorn.py
```

### Member 4 — Benchmark, Visualization, Report

Responsibilities:

- Implement benchmark automation.
- Implement plotting.
- Generate tables and figures.
- Write report and presentation content.
- Analyze speedup and efficiency.

Expected code contribution:

```text
src/mpi_sinkhorn/benchmark.py
src/mpi_sinkhorn/plotting.py
apps/plot_results.py
report/report_outline.md
README result section
```

---

## 11. Milestones

## Milestone 1 — Minimal MPI Cluster Works

Deliverables:

- 3 physical machines connected.
- SSH passwordless login from master to workers.
- `mpirun hostname` works.
- `hello_mpi.py` works.

Success command:

```bash
mpirun -np 6 --hostfile hosts python apps/hello_mpi.py
```

## Milestone 2 — Sequential Sinkhorn Solver Works

Deliverables:

- Dense Sinkhorn-Knopp solver implemented.
- Uniform and custom marginals supported.
- Row and column marginal errors reported.
- JSON output saved.
- Unit tests validate marginal constraints.

Success command:

```bash
python apps/sequential_sinkhorn_runner.py --rows 256 --cols 256 --epsilon 0.1 --max-iters 200 --output results/test_sinkhorn_seq.json
```

## Milestone 3 — MPI Distributed Sinkhorn Solver Works

Deliverables:

- Cost matrix rows distributed across ranks.
- `Allreduce` used for the `K.T @ u` update.
- MPI result matches sequential result within tolerance.
- Runtime and marginal errors logged.

Success command:

```bash
mpirun -np 6 --hostfile hosts python apps/mpi_sinkhorn_runner.py --rows 1000 --cols 1000 --epsilon 0.1 --max-iters 200 --output results/test_sinkhorn_mpi_np6.json
```

## Milestone 4 — Sequential Optimizer Works

Deliverables:

- Objective functions implemented.
- Direction generation implemented.
- Sequential optimizer implemented.
- Both `softmin` and `sinkhorn` weighting modes supported.
- JSON output and convergence history saved.

Success command:

```bash
python apps/sequential_runner.py --objective ackley --dim 10 --num-points 1000 --max-iters 30 --weight-mode sinkhorn --output results/test_optimizer_seq.json
```

## Milestone 5 — MPI Optimizer Modes Work

Deliverables:

- MPI multiseed backup mode works.
- MPI distributed probe mode works.
- Sinkhorn weighting mode works in distributed-probe optimizer.
- Rank 0 logs runtime, history, and best result.

Success commands:

```bash
mpirun -np 6 --hostfile hosts python apps/mpi_multiseed_runner.py --objective ackley --dim 10 --num-points 1000 --max-iters 30 --num-seeds 12 --output results/test_multiseed.json

mpirun -np 6 --hostfile hosts python apps/mpi_distributed_probe_runner.py --objective ackley --dim 10 --num-points 3000 --max-iters 50 --weight-mode sinkhorn --output results/test_probe_np6.json
```

## Milestone 6 — Benchmark and Report Figures

Deliverables:

- Run sequential Sinkhorn, MPI Sinkhorn np=3, np=6, np=9/12 if possible.
- Run sequential optimizer and MPI optimizer benchmarks.
- Generate CSV summary.
- Generate convergence/speedup/efficiency plots.

Success command:

```bash
python apps/plot_results.py --input results/benchmark_summary.csv --output-dir report/figures
```

---

## 12. Report Outline

Use this report structure:

```text
1. Introduction
   - Motivation: entropy-regularized optimal transport and black-box optimization are computationally expensive.
   - Sinkhorn iterations require repeated matrix-vector operations that can be distributed.
   - We implement a real MPI Sinkhorn-Knopp solver and apply it to a Sinkhorn-Step-inspired optimizer.

2. Background
   - Optimal transport.
   - Entropy regularization.
   - Sinkhorn-Knopp matrix scaling.
   - Gradient-free optimization.
   - Population-based search.
   - Probe directions.
   - MPI distributed-memory model.

3. Sequential Sinkhorn Algorithm
   - Cost matrix C.
   - Marginal distributions a and b.
   - Kernel K = exp(-C / epsilon).
   - Scaling vectors u and v.
   - Transport plan P.
   - Marginal error convergence checks.

4. MPI Sinkhorn Parallelization
   - Row partitioning of C.
   - Local K @ v computation.
   - Local K.T @ u contribution.
   - MPI Allreduce for global column scaling update.
   - Communication cost per iteration.
   - Correctness comparison against sequential Sinkhorn.

5. Optimizer Application
   - Candidate population X.
   - Direction set D.
   - Probe evaluation.
   - Softmin fallback weighting.
   - Sinkhorn transport weighting.
   - Update rule.

6. MPI Optimizer Parallelization
   - Cluster architecture.
   - MPI rank roles.
   - Candidate partitioning.
   - Broadcast X and D.
   - Local objective evaluation.
   - Gather cost matrix.
   - Rank 0 update.
   - Communication cost discussion.

7. Implementation
   - Programming language: Python.
   - Libraries: mpi4py, numpy, matplotlib.
   - Repository structure.
   - Configuration and execution commands.

8. Experimental Setup
   - Hardware: 3 physical machines.
   - Network: same LAN/hotspot, Ubuntu VMs with bridged adapter.
   - Number of MPI processes.
   - Sinkhorn matrix sizes.
   - Objective functions.
   - Parameters.

9. Results
   - Sinkhorn marginal error table.
   - Sequential vs MPI Sinkhorn runtime.
   - Sinkhorn speedup and efficiency.
   - Convergence curves.
   - Optimizer runtime table.
   - Optimizer speedup chart.
   - Optimizer efficiency chart.
   - Best objective values.

10. Discussion
   - Why speedup is not perfectly linear.
   - Communication overhead.
   - Rank 0 update bottleneck.
   - Sinkhorn numerical stability.
   - Load balancing.
   - Limitations.

11. Conclusion
   - Summary of achieved cluster and algorithm.
   - Lessons learned.
   - Future work: log-domain Sinkhorn, JAX/GPU, sparse cost matrices, motion planning extension.
```

---

## 13. README Content Requirements

`README.md` should include:

1. Project title.
2. Group members.
3. Problem statement.
4. Algorithm overview.
5. Parallelization overview.
6. Cluster setup instructions.
7. Installation instructions.
8. How to run sequential baseline.
9. How to run MPI version.
10. How to reproduce benchmark results.
11. Result summary.
12. Known issues.

---

## 14. Important Engineering Decisions

### 14.1 Do not depend on Kaggle for final MPI demo

Kaggle may be used for additional single-node experiments only. It should not be counted as a physical MPI node.

### 14.2 Cloudflare Tunnel should not be used as MPI network

Cloudflare Tunnel may be used only to SSH into the master machine remotely. MPI communication should run on LAN/VPN directly.

### 14.3 Keep a backup demo

The final demo should have three MPI programs:

1. `mpi_sinkhorn_runner.py`: primary real Sinkhorn MPI solver demo.
2. `mpi_distributed_probe_runner.py`: optimizer application, more impressive but more moving parts.
3. `mpi_multiseed_runner.py`: easiest and most robust fallback.

If the optimizer has issues during demo, still show the real Sinkhorn MPI solver. If the solver benchmark setup has issues, run the multiseed optimizer as the final backup.

### 14.4 Avoid too-large configurations during live demo

For live/offline demo, use moderate parameters:

```text
Sinkhorn solver:
rows = 1000
cols = 1000
epsilon = 0.1
max_iters = 200
np = 6

Optimizer:
objective = ackley
dim = 10
num_points = 2000 or 3000
max_iters = 30 or 50
np = 6
```

For report, run larger experiments offline.

---

## 15. Potential Problems and Fixes

### Problem: `mpirun` asks for password

Fix:

```bash
ssh-copy-id mpiuser@worker_ip
```

### Problem: `mpirun hostname` only runs on master

Check `hosts` file and SSH access.

### Problem: MPI cannot connect to worker

Check firewall:

```bash
sudo ufw disable
```

Check network:

```bash
ping worker_ip
ssh worker_ip hostname
```

### Problem: Python module not found on workers

Install same environment on all machines:

```bash
python3 -m venv ~/mpi-env
source ~/mpi-env/bin/activate
pip install -r requirements.txt
```

Use same project path on all machines:

```text
/home/mpiuser/mpi-sinkhorn-step
```

Sync code from master:

```bash
rsync -av --delete ./ mpiuser@192.168.43.102:/home/mpiuser/mpi-sinkhorn-step/
rsync -av --delete ./ mpiuser@192.168.43.103:/home/mpiuser/mpi-sinkhorn-step/
```

### Problem: MPI uses wrong network interface

Try specifying TCP interface:

```bash
mpirun -np 6 --hostfile hosts --mca btl tcp,self --mca btl_tcp_if_include wlan0 python apps/hello_mpi.py
```

Replace `wlan0` with the correct interface from:

```bash
ip addr
```

### Problem: speedup is poor

Reasons:

- Workload too small.
- Network overhead dominates.
- Rank 0 bottleneck.
- Too many processes for available CPU cores.
- Python serialization overhead.

Fix:

- Increase `num_points`.
- Increase `dim`.
- Increase `max_iters`.
- Use fewer but meaningful MPI processes.
- Avoid printing inside loops.

### Problem: Sinkhorn underflows or returns NaN

Reasons:

- `epsilon` is too small.
- Cost values are too large.
- `exp(-cost / epsilon)` underflows to zero.
- Marginals contain zeros or do not sum to the same mass.

Fix:

- Start with `epsilon = 0.1` or `epsilon = 1.0`.
- Normalize or rescale the cost matrix.
- Compute `K = exp(-(C - C.min()) / epsilon)`.
- Add `1e-12` to denominators.
- Validate `a`, `b`, and `cost` before iterations.
- Keep log-domain Sinkhorn as an advanced extension if standard scaling is not stable enough.

---

## 16. Definition of Done

The solver-only MVP is considered complete when:

- `mpirun hello_mpi.py` works locally.
- `mpirun hello_mpi.py` works on the 3-machine cluster.
- `sequential_sinkhorn_runner.py` saves valid JSON.
- `mpi_sinkhorn_runner.py` saves valid JSON.
- `compare_sinkhorn_results.py` shows similar objective and marginal errors.
- MPI with `np=1` matches dense solver closely.
- MPI with `np=3` runs successfully.
- MPI with `np=6` runs successfully on the cluster.
- Results include runtime, speedup, efficiency, row_error, col_error, and transport_objective.

The project is considered complete when:

- The solver-only MVP is complete.
- 3 physical machines can run `mpirun hostname`.
- Sequential Sinkhorn solver runs and saves JSON result.
- MPI Sinkhorn solver runs and saves JSON result.
- MPI Sinkhorn result matches sequential result within tolerance.
- Sequential optimizer runs and saves JSON result.
- MPI multiseed runner runs and saves JSON result.
- MPI distributed probe runner runs and saves JSON result.
- Optimizer supports both `softmin` and `sinkhorn` weighting modes.
- Benchmark summary CSV is generated.
- At least 3 plots are generated, including Sinkhorn speedup or marginal error.
- Report draft is written.
- Every member can explain their part.
- The project has around 1000+ lines of original code, excluding external libraries.

---

## 17. Suggested First Commands for Codex

Codex should start by creating the project skeleton:

```bash
mkdir -p mpi-sinkhorn-step/{scripts,src/mpi_sinkhorn,apps,configs,results,report/figures,report/tables,tests}
touch mpi-sinkhorn-step/src/mpi_sinkhorn/__init__.py
touch mpi-sinkhorn-step/results/.gitkeep
```

Then implement in this order:

1. `pyproject.toml`
2. `requirements.txt`
3. `src/mpi_sinkhorn/sinkhorn.py`
4. `src/mpi_sinkhorn/mpi_utils.py`
5. `apps/hello_mpi.py`
6. `apps/sequential_sinkhorn_runner.py`
7. `apps/mpi_sinkhorn_runner.py`
8. `apps/compare_sinkhorn_results.py`
9. `tests/test_sinkhorn.py`

Only after the solver-only MVP passes, continue with:

1. `src/mpi_sinkhorn/objectives.py`
2. `src/mpi_sinkhorn/directions.py`
3. `src/mpi_sinkhorn/optimizer.py`
4. `apps/sequential_runner.py`
5. `apps/mpi_multiseed_runner.py`
6. `apps/mpi_distributed_probe_runner.py`
7. `apps/compare_results.py`
8. `src/mpi_sinkhorn/plotting.py`
9. `apps/plot_results.py`
10. `README.md`
11. `report/report_outline.md`

---

## 18. Minimum Viable Demo Command Set

Make these smoke commands pass before attempting large benchmarks or optimizer code.

```bash
# 1. Sequential smoke
python apps/sequential_sinkhorn_runner.py \
  --rows 128 \
  --cols 128 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 1e-6 \
  --output results/smoke_seq.json

# 2. MPI smoke on local machine
mpirun -np 3 python apps/mpi_sinkhorn_runner.py \
  --rows 128 \
  --cols 128 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 1e-6 \
  --output results/smoke_mpi_np3.json

# 3. Compare smoke
python apps/compare_sinkhorn_results.py \
  --sequential results/smoke_seq.json \
  --parallel results/smoke_mpi_np3.json

# 4. Check MPI cluster
mpirun -np 6 --hostfile hosts python apps/hello_mpi.py

# 5. Cluster smoke
mpirun -np 6 --hostfile hosts python apps/mpi_sinkhorn_runner.py \
  --rows 128 \
  --cols 128 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --output results/smoke_cluster_np6.json
```

After smoke passes, use these commands as the minimum demo target.

```bash
# 1. Run sequential Sinkhorn baseline
python apps/sequential_sinkhorn_runner.py \
  --rows 1000 \
  --cols 1000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.1 \
  --max-iters 100 \
  --output results/demo_sinkhorn_seq.json

# 2. Run MPI Sinkhorn
mpirun -np 6 --hostfile hosts python apps/mpi_sinkhorn_runner.py \
  --rows 1000 \
  --cols 1000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.1 \
  --max-iters 100 \
  --output results/demo_sinkhorn_mpi_np6.json

# 3. Compare Sinkhorn
python apps/compare_sinkhorn_results.py \
  --sequential results/demo_sinkhorn_seq.json \
  --parallel results/demo_sinkhorn_mpi_np6.json

# 4. Run MPI optimizer application after solver MVP passes
mpirun -np 6 --hostfile hosts python apps/mpi_distributed_probe_runner.py \
  --objective ackley \
  --dim 10 \
  --num-points 3000 \
  --max-iters 50 \
  --weight-mode sinkhorn \
  --output results/demo_mpi_np6.json
```

---

## 19. Optional Advanced Extension

After the basic version works, add one or more of the following:

1. Log-domain Sinkhorn for stronger numerical stability.
2. JAX acceleration for objective evaluation.
3. MPI `Scatterv/Gatherv` instead of Python `scatter/gather`.
4. Load balancing for heterogeneous machines.
5. Fault-tolerant result saving.
6. Sparse cost matrices for larger OT problems.
7. Motion planning extension using 2D obstacle map.
8. Kaggle single-node GPU benchmark as an appendix only.

Do not attempt these before the basic MPI demo works.

---

## 20. Final Advice

Priority order:

```text
1. Correctness first.
2. Make local hello_mpi, sequential Sinkhorn, and MPI Sinkhorn smoke tests pass.
3. Make 3-machine MPI cluster execution work.
4. Benchmark solver-only Sinkhorn speedup and efficiency.
5. Make sequential optimizer work with softmin and Sinkhorn modes.
6. Make simple MPI multiseed backup work.
7. Make distributed probe optimizer work.
8. Generate plots and polish report/demo.
```

Avoid over-engineering. The main grading value comes from a working cluster, a real distributed Sinkhorn algorithm, clear MPI communication, reliable demo, and understandable report.
