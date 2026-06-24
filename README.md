# Parallel-Programming-Project

2025.2 IT4130E Parallel and Distributed Programming Project.

This repository contains the Sinkhorn optimal transport MPI project, including the
original Python/mpi4py prototype, the final C++/MPI implementation, benchmark
scripts, experiment outputs, report, and presentation slides.

## Branches

| Branch | Purpose |
|---|---|
| `main` | Integrated project branch with Python prototype, C++ MPI solver, dynamic scheduling support, benchmark outputs, report, and slides. |
| `cpp-benchmark` | Historical C++ benchmark/report branch. Merged into `main`. |
| `dynamic-scheduling` | Historical dynamic scheduling branch. Merged into `main`. |

## Implementation

- [MPI Sinkhorn Step MVP](mpi-sinkhorn-step/README.md)
- [Python MPI cluster runbook](mpi-sinkhorn-step/PYTHON_CLUSTER_RUNBOOK.md)
- [C++ MPI cluster runbook](mpi-sinkhorn-step/CPP_CLUSTER_RUNBOOK.md)
- [C++ Sinkhorn source](mpi-sinkhorn-step/cpp/sinkhorn_cpp.cpp)
- [Implementation summary](mpi_sinkhorn_mvp_implementation_report.md)

## Final Report and Slides

- [Final report PDF](mpi-sinkhorn-step/report/main.pdf)
- [Final report LaTeX source](mpi-sinkhorn-step/report/main.tex)
- [Presentation slides PDF](mpi-sinkhorn-step/report/slides.pdf)
- [Presentation slides LaTeX source](mpi-sinkhorn-step/report/slides.tex)
- [Slide speaker script](mpi-sinkhorn-step/report/slides_script.md)

## Benchmark Results

Key result folders:

- [Router 3-node report suite](mpi-sinkhorn-step/results/bench_cpp_router3_report_suite_safe)
- [Router 3-node supplemental suite](mpi-sinkhorn-step/results/bench_cpp_router3_supplemental_after_ram)
- [Wi-Fi 3-node report suite](mpi-sinkhorn-step/results/bench_cpp_wifi3_25_26_190_report_suite)
- [Wi-Fi 3-node supplemental suite](mpi-sinkhorn-step/results/bench_cpp_wifi3_25_26_190_supplemental)
- [Dynamic scheduling router suite](mpi-sinkhorn-step/results/bench_cpp_router3_dynamic_fixed50_reps4)

## Archived Notes

- [Archived MPI project topic report](old_mds/mpi_project_topic_report.md)
- [Archived MPI SSAX project handoff](old_mds/mpi_ssax_project_handoff.md)
- [Archived real Sinkhorn plan](old_mds/plan_next_mpi_ssax_real_sinkhorn.md)
- [Archived real Sinkhorn MPI feedback](old_mds/codex_feedback_real_sinkhorn_mpi.md)
- [Archived milestones](old_mds/milestones.md)
