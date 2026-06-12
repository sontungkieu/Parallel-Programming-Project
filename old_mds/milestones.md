# Milestones

## MPI SSAX Real Sinkhorn Plan

Trạng thái: solver-only MVP implemented locally, cluster verification pending

Mục tiêu scope:

- Chốt hướng real Sinkhorn-Knopp là thuật toán MPI chính.
- Giữ `softmin` làm baseline/fallback, không xem là real Sinkhorn.
- Dùng optimizer như tầng ứng dụng của distributed Sinkhorn solver.

Milestones:

1. Minimal local MPI works. Done: `mpirun -np 3 apps/hello_mpi.py` passed.
2. Solver-only MVP works locally. Done: sequential Sinkhorn, MPI Sinkhorn `np=1`, `np=3`, local `np=6 --oversubscribe`, and comparison passed.
3. Solver-only MVP works on 3-machine cluster with MPI `np=6`. Pending: requires real `hosts` file and machines.
4. Solver-only benchmark records runtime, speedup, efficiency, row_error, col_error, and transport_objective. Partially done: smoke comparison reports all metrics; large benchmark pending.
5. Sequential optimizer works with `softmin` and `sinkhorn` modes. Pending after solver MVP cluster verification.
6. MPI optimizer modes work. Pending.
7. Benchmark plots and report figures are complete. Pending.

Ghi chú:

- `mpi_ssax_project_handoff.md` đã được cập nhật để real Sinkhorn không còn là optional extension.
- `plan_next_mpi_ssax_real_sinkhorn.md` là active plan hiện tại cho hướng triển khai tiếp theo.
- `codex_feedback_real_sinkhorn_mpi.md` đã được dùng để thu hẹp MVP: ưu tiên solver-only trước optimizer/plot/report.
- `mpi-sinkhorn-step/` đã được tạo với solver-only MVP và local smoke đã pass.
