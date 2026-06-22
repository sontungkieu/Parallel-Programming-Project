# Parallel-Programming-Project
2025.2 IT4130E Parallel and Distributed Programming Project

## Branches

| Branch | Purpose | Use when |
|---|---|---|
| `main` | Python MPI Sinkhorn MVP | You need the stable Python implementation, package structure, tests, and basic cluster demo. |
| `codex/cpp-benchmark` | C++ MPI implementation, communication optimizations, benchmark tables, and charts | You need the faster C++ demo, fixed-iteration benchmark results, and final performance charts. |

Recommended workflow:

```bash
git checkout main
# Read the Python runbook for baseline setup and cluster smoke tests.

git checkout codex/cpp-benchmark
# Read the C++ runbook for optimized benchmarks and final demo charts.
```

For the final live demo, use Ubuntu VMs with bridged networking instead of WSL.
The VM setup is documented in the runbooks below.

## Implementation

- [MPI Sinkhorn Step MVP](mpi-sinkhorn-step/README.md)
- [Python MPI cluster runbook](mpi-sinkhorn-step/PYTHON_CLUSTER_RUNBOOK.md)
- [Implementation summary](mpi_sinkhorn_mvp_implementation_report.md)

## C++ Benchmark Branch

The C++ benchmark branch is not merged into `main`. Check it out explicitly:

```bash
git fetch origin
git checkout codex/cpp-benchmark
```

Important files on `codex/cpp-benchmark`:

- [C++ cluster runbook](https://github.com/sontungkieu/Parallel-Programming-Project/blob/codex/cpp-benchmark/mpi-sinkhorn-step/CPP_CLUSTER_RUNBOOK.md)
- [C++ Sinkhorn source](https://github.com/sontungkieu/Parallel-Programming-Project/blob/codex/cpp-benchmark/mpi-sinkhorn-step/cpp/sinkhorn_cpp.cpp)
- [Benchmark findings](https://github.com/sontungkieu/Parallel-Programming-Project/blob/codex/cpp-benchmark/mpi-sinkhorn-step/results/bench_cpp_vm2_sameband_variants_fixed50/findings.md)
- [Benchmark charts](https://github.com/sontungkieu/Parallel-Programming-Project/blob/codex/cpp-benchmark/mpi-sinkhorn-step/results/bench_cpp_vm2_sameband_variants_fixed50/charts.md)

## Reports

- [Archived MPI project topic report](old_mds/mpi_project_topic_report.md)
- [Archived MPI SSAX project handoff](old_mds/mpi_ssax_project_handoff.md)
- [Archived real Sinkhorn plan](old_mds/plan_next_mpi_ssax_real_sinkhorn.md)
- [Archived real Sinkhorn MPI feedback](old_mds/codex_feedback_real_sinkhorn_mpi.md)
- [Archived milestones](old_mds/milestones.md)
