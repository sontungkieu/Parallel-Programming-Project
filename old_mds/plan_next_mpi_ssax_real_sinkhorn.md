# Plan Next: MPI SSAX Real Sinkhorn

## 0. Trạng thái triển khai hiện tại

Đã triển khai solver-only MVP trong `mpi-sinkhorn-step/`:

- Package skeleton với `pyproject.toml`, `requirements.txt`, `src/mpi_sinkhorn`.
- Deterministic cost generator: `random`, `squared_distance_1d`, `block`.
- Dense Sinkhorn-Knopp solver.
- MPI row-partitioned Sinkhorn-Knopp solver dùng `Allreduce`.
- Runner: `hello_mpi.py`, `sequential_sinkhorn_runner.py`, `mpi_sinkhorn_runner.py`, `compare_sinkhorn_results.py`.
- Unit tests cho generator, dense solver và distributed solver trên `MPI.COMM_SELF`.
- README cho MVP và lệnh smoke/demo.

Đã verify local:

```text
PYTHONPATH=src python3 -m pytest -q
7 passed

PYTHONPATH=src python3 -m compileall -q src apps tests
pass

PYTHONPATH=src mpirun -np 3 python3 apps/hello_mpi.py
pass

Sequential smoke 128x128:
row_error = 1.283e-10
col_error = 1.283e-10
objective = 0.347356000544

MPI smoke np=1:
objective difference vs dense = 0

MPI smoke np=3:
objective difference vs dense = 0

MPI smoke np=6 local with --oversubscribe:
objective = 0.347356000544
```

Pending:

- Cluster smoke với `--hostfile hosts` trên 3 máy thật.
- Benchmark lớn hơn cho speedup/efficiency.
- Optimizer application sau khi cluster MVP được xác nhận.

## 1. Mục tiêu chốt

Hướng triển khai nên được chốt là:

```text
Distributed Entropy-Regularized Optimal Transport using MPI,
applied to a Sinkhorn-Step-inspired optimizer.
```

Điểm quan trọng: không chỉ dùng `softmin` rồi gọi là Sinkhorn. `softmin` chỉ là baseline/fallback. Phần chính phải có solver Sinkhorn-Knopp thật, chạy được ở bản tuần tự và bản MPI.

## 2. Vì sao real Sinkhorn phù hợp

Real Sinkhorn phù hợp với môn Parallel and Distributed Programming vì:

- Thuật toán có nền toán rõ: entropy-regularized optimal transport.
- Có vòng lặp tính toán lặp lại, không chỉ chia job độc lập.
- MPI được dùng đúng bản chất distributed-memory qua `Allreduce`.
- Có correctness metric rõ: sai số marginal hàng/cột.
- Có benchmark speedup rõ: sequential Sinkhorn vs MPI Sinkhorn.
- Optimizer chỉ là tầng ứng dụng thêm, không phải thứ duy nhất để bảo vệ project.

## 3. Scope bắt buộc và fallback

Scope bắt buộc nhưng phải chia làm hai tầng. Tầng 1 là MVP bắt buộc làm trước:

1. Sequential dense Sinkhorn-Knopp solver.
2. MPI row-partitioned Sinkhorn-Knopp solver.
3. So sánh correctness giữa sequential và MPI.
4. Benchmark runtime, speedup, efficiency.

Chỉ sau khi solver-only MVP pass mới làm tầng 2:

5. Sequential optimizer có `softmin` và `sinkhorn` modes.
6. MPI optimizer dùng distributed probe evaluation và Sinkhorn weighting.

Không triển khai các file optimizer, plotting và report automation trước khi MVP solver-only chạy đúng:

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

Fallback:

- Nếu optimizer lỗi lúc demo, vẫn demo được `mpi_sinkhorn_runner.py`.
- Nếu Sinkhorn mode trong optimizer chưa ổn, dùng `softmin` để demo optimizer và dùng solver-only để chứng minh real Sinkhorn.
- Nếu distributed probe optimizer có vấn đề, dùng `mpi_multiseed_runner.py`.

## 4. Thuật toán real Sinkhorn cần làm

Bài toán:

```text
min_P <C, P> + epsilon * sum_ij P_ij (log P_ij - 1)
subject to:
    P 1 = a
    P^T 1 = b
    P_ij >= 0
```

Trong đó:

- `C`: cost matrix, shape `(n, m)`.
- `a`: source marginal, shape `(n,)`.
- `b`: target marginal, shape `(m,)`.
- `sum(a) = sum(b) = 1`.
- `epsilon`: entropy regularization.
- `P`: transport plan.

Sinkhorn-Knopp:

```text
K = exp(-C / epsilon)
u = a / (K v)
v = b / (K^T u)
P = diag(u) K diag(v)
```

Kiểm tra hội tụ:

```text
row_error = ||P 1 - a||_1
col_error = ||P^T 1 - b||_1
```

## 5. MPI decomposition

Chia theo hàng là dễ triển khai và dễ giải thích nhất.

```text
Rank r giữ:
    local_cost = C[start_r:end_r, :]
    local_a = a[start_r:end_r]

Mọi rank giữ:
    b
    v
```

Mỗi vòng Sinkhorn:

```text
1. local_Kv = K_local @ v
2. local_u = local_a / local_Kv
3. local_KTu = K_local.T @ local_u
4. global_KTu = Allreduce(sum, local_KTu)
5. v = b / global_KTu
6. Định kỳ tính row_error và col_error bằng Allreduce
```

Đây là phần quan trọng nhất để trình bày MPI: mỗi iteration có local compute và global communication.

## 6. Thứ tự triển khai

### Phase 1: Skeleton và tiện ích

Tạo cấu trúc:

```text
mpi-sinkhorn-step/
├── src/mpi_sinkhorn/
├── apps/
├── tests/
├── scripts/
├── configs/
├── results/
└── report/
```

File cần có sớm:

- `pyproject.toml`
- `requirements.txt`
- `src/mpi_sinkhorn/__init__.py`
- `src/mpi_sinkhorn/mpi_utils.py`
- `apps/hello_mpi.py`

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

`pyproject.toml` là bắt buộc để worker cài package bằng:

```bash
pip install -e .
```

### Phase 2: Sequential Sinkhorn

Implement:

- `src/mpi_sinkhorn/sinkhorn.py`
- deterministic cost generator.
- `sinkhorn_knopp_dense(...)`
- `apps/sequential_sinkhorn_runner.py`
- `tests/test_sinkhorn.py`

Sequential và MPI runner phải giải cùng một OT problem khi nhận cùng:

```text
--rows
--cols
--cost-mode
--seed
--epsilon
--max-iters
--tol
```

Implement shared cost generator:

```python
def generate_cost_matrix(rows: int, cols: int, mode: str = "random", seed: int = 0) -> np.ndarray:
    ...
```

Cost modes cần hỗ trợ:

```text
random
squared_distance_1d
block
```

MVP: rank 0 generate full cost matrix rồi scatter row blocks. Extension sau này mới để mỗi rank tự generate local rows bằng global row index.

API khuyến nghị:

```python
def sinkhorn_knopp_dense(
    cost,
    a=None,
    b=None,
    epsilon=0.1,
    max_iters=200,
    tol=1e-6,
    check_every=10,
    return_transport=False,
) -> dict:
    ...
```

Điều kiện pass:

- Row sums gần `a`.
- Column sums gần `b`.
- Không NaN/Inf.
- Có JSON result chứa runtime, iterations, row_error, col_error.
- Có `transport_objective`.
- `return_transport=False` mặc định để tránh lưu/gather matrix lớn.

### Phase 3: MPI Sinkhorn

Implement:

- `sinkhorn_knopp_distributed(...)`
- `apps/mpi_sinkhorn_runner.py`
- `apps/compare_sinkhorn_results.py`

API khuyến nghị:

```python
def sinkhorn_knopp_distributed(
    comm,
    local_cost,
    local_a,
    b,
    epsilon=0.1,
    max_iters=200,
    tol=1e-6,
    check_every=10,
    return_local_transport=False,
) -> dict:
    ...
```

Distributed solver phải dùng global min để khớp dense solver:

```python
local_min = np.array(local_cost.min(), dtype=float)
global_min = np.array(0.0, dtype=float)
comm.Allreduce(local_min, global_min, op=MPI.MIN)
local_K = np.exp(-(local_cost - float(global_min)) / epsilon)
```

Không được tính error cục bộ rồi gọi là global error:

```text
row_error = Allreduce(sum, local row error)
col_error = error after Allreduce(sum, local column mass)
transport_objective = Allreduce(sum, sum(local_P * local_cost))
```

Không gather full `P` cho benchmark/demo lớn hơn `512x512` mặc định. Chỉ gather full `P` cho test nhỏ như `16x16`, `64x64`, `128x128`.

Điều kiện pass:

- `mpirun -np 1` chạy giống sequential.
- `mpirun -np 3` chạy được.
- `mpirun -np 6` chạy được.
- Kết quả MPI gần sequential trong tolerance.
- Có speedup khi matrix đủ lớn.
- JSON schema giống sequential runner.

JSON output schema chung:

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

### Phase 4: Optimizer tuần tự

Implement:

- `src/mpi_sinkhorn/objectives.py`
- `src/mpi_sinkhorn/directions.py`
- `src/mpi_sinkhorn/optimizer.py`
- `apps/sequential_runner.py`

Optimizer cần có:

```text
--weight-mode softmin
--weight-mode sinkhorn
```

Với `sinkhorn` mode:

```text
C[i, j] = f(x_i + probe_radius * d_j)
P = Sinkhorn(C, a, b)
W[i, j] = P[i, j] / sum_j P[i, j]
update_dir[i] = sum_j W[i, j] * D[j]
```

### Phase 5: MPI optimizer

Implement:

- `apps/mpi_multiseed_runner.py`
- `apps/mpi_distributed_probe_runner.py`
- `apps/compare_results.py`

Trong distributed probe:

- Mỗi rank giữ một slice candidate.
- Mỗi rank tính `local_costs`.
- Nếu `weight_mode=softmin`, có thể gather cost về rank 0.
- Nếu `weight_mode=sinkhorn`, gọi `sinkhorn_knopp_distributed` trên `local_costs`.
- Mỗi rank tính `local_update_dirs`.
- Rank 0 gather update directions và cập nhật `X`.

### Phase 6: Benchmark và report

Cần benchmark:

- Sequential Sinkhorn.
- MPI Sinkhorn `np=3`, `np=6`, thêm `np=9/12` nếu đủ core.
- Sequential optimizer.
- MPI optimizer.

Metrics:

- Runtime.
- Speedup.
- Efficiency.
- Row marginal error.
- Column marginal error.
- Best objective cost.
- Convergence history.

Plot tối thiểu:

- Sinkhorn runtime bar chart.
- Sinkhorn speedup chart.
- Optimizer convergence curve.
- Efficiency chart nếu kịp.

## 7. Demo tối thiểu

Smoke local phải pass trước demo lớn:

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

Sau đó mới chạy cluster smoke:

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

Demo preset:

```text
Smoke:
    rows = 128
    cols = 128
    max_iters = 50
    epsilon = 0.5

Demo:
    rows = 1000
    cols = 1000
    max_iters = 100
    epsilon = 0.5 or 0.1

Report:
    rows = 3000
    cols = 3000
    max_iters = 200
    epsilon = 0.5 or 0.1
```

Demo tối thiểu sau khi smoke pass:

```bash
mpirun -np 6 --hostfile hosts python apps/hello_mpi.py

python apps/sequential_sinkhorn_runner.py \
  --rows 1000 \
  --cols 1000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.1 \
  --max-iters 200 \
  --output results/demo_sinkhorn_seq.json

mpirun -np 6 --hostfile hosts python apps/mpi_sinkhorn_runner.py \
  --rows 1000 \
  --cols 1000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.1 \
  --max-iters 200 \
  --output results/demo_sinkhorn_mpi_np6.json

python apps/compare_sinkhorn_results.py \
  --sequential results/demo_sinkhorn_seq.json \
  --parallel results/demo_sinkhorn_mpi_np6.json

mpirun -np 6 --hostfile hosts python apps/mpi_distributed_probe_runner.py \
  --objective ackley \
  --dim 10 \
  --num-points 3000 \
  --max-iters 50 \
  --weight-mode sinkhorn \
  --output results/demo_mpi_np6.json
```

## 8. Rủi ro và cách xử lý

### Sinkhorn bị NaN hoặc không hội tụ

Cách xử lý:

- Tăng `epsilon`, ví dụ `0.1` hoặc `1.0`.
- Rescale cost matrix.
- Dùng `K = exp(-(C - C.min()) / epsilon)`.
- Thêm `1e-12` ở mẫu số.
- Validate `a`, `b`, `cost`.

### Speedup xấu

Cách xử lý:

- Tăng kích thước matrix Sinkhorn.
- Giảm print/log trong vòng lặp.
- So sánh nhiều mức `np`.
- Tách thời gian setup và thời gian iteration.
- Không gather full transport plan trong benchmark lớn nếu không cần.

### Optimizer khó giải thích

Cách xử lý:

- Trình bày solver Sinkhorn là thuật toán chính.
- Trình bày optimizer là ứng dụng của transport plan để chọn hướng cập nhật.
- So sánh `softmin` vs `sinkhorn` như ablation.

## 9. Definition of Done

MVP solver-only được xem là xong khi:

- `mpirun hello_mpi.py` chạy local.
- `mpirun hello_mpi.py` chạy trên cluster 3 máy.
- Sequential Sinkhorn chạy và lưu JSON.
- MPI Sinkhorn chạy và lưu JSON.
- `compare_sinkhorn_results.py` cho thấy objective và marginal errors gần nhau.
- MPI với `np=1` gần dense solver.
- MPI với `np=3` chạy thành công.
- MPI với `np=6` chạy thành công trên cluster.
- Kết quả có runtime, speedup, efficiency, row_error, col_error, transport_objective.

Project đầy đủ sau MVP được xem là đủ scope khi:

- Tất cả điều kiện MVP solver-only đã pass.
- Sequential optimizer chạy được với `softmin` và `sinkhorn`.
- MPI optimizer chạy được ít nhất một mode chính và một fallback.
- Có benchmark summary CSV.
- Có tối thiểu 3 biểu đồ.
- README có lệnh setup/run/reproduce.
- Report giải thích rõ Sinkhorn, MPI parallelization, kết quả và giới hạn.
