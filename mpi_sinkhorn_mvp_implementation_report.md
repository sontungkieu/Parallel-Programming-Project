# MPI Sinkhorn MVP Implementation Report

## 1. Mục tiêu đã triển khai

Đã triển khai MVP solver-only cho hướng:

```text
Distributed Entropy-Regularized Optimal Transport using MPI,
applied later to a Sinkhorn-Step-inspired optimizer.
```

MVP này tập trung vào real Sinkhorn-Knopp solver trước, chưa triển khai optimizer application. Lý do là solver-only đã đủ thể hiện bài toán song song hóa MPI:

- Chia ma trận cost theo hàng cho nhiều rank.
- Mỗi rank tính toán local matrix-vector work.
- Dùng `MPI.Allreduce` để cập nhật global column scaling.
- Có correctness metrics rõ ràng: `row_error`, `col_error`, `transport_objective`.
- Có runner để so sánh sequential vs MPI runtime, speedup và efficiency.

## 2. Cấu trúc đã tạo

Đã tạo project con:

```text
mpi-sinkhorn-step/
├── README.md
├── pyproject.toml
├── requirements.txt
├── apps/
│   ├── compare_sinkhorn_results.py
│   ├── hello_mpi.py
│   ├── mpi_sinkhorn_runner.py
│   └── sequential_sinkhorn_runner.py
├── results/
│   └── .gitkeep
├── src/
│   └── mpi_sinkhorn/
│       ├── __init__.py
│       ├── mpi_utils.py
│       └── sinkhorn.py
└── tests/
    └── test_sinkhorn.py
```

Root repo cũng được cập nhật:

- `README.md`: thêm link tới implementation, report này và tài liệu đã archive.
- `.gitignore`: ignore output JSON trong `mpi-sinkhorn-step/results/` và metadata `*:Zone.Identifier`.
- `old_mds/`: chứa các file markdown planning/report cũ.

## 3. Core algorithm

File chính:

```text
mpi-sinkhorn-step/src/mpi_sinkhorn/sinkhorn.py
```

Đã implement:

- `generate_cost_matrix(...)`
- `sinkhorn_knopp_dense(...)`
- `sinkhorn_knopp_distributed(...)`

Dense Sinkhorn giải:

```text
min_P <C, P> + epsilon * sum_ij P_ij (log P_ij - 1)
subject to:
    P 1 = a
    P^T 1 = b
    P_ij >= 0
```

Với scaling:

```text
K = exp(-(C - C.min()) / epsilon)
u = a / (K v)
v = b / (K^T u)
P = diag(u) K diag(v)
```

MPI version chia theo hàng:

```text
Rank r giữ:
    local_cost = C[start_r:end_r, :]
    local_a = a[start_r:end_r]

Mọi rank giữ:
    b
    v
```

Mỗi iteration:

```text
local_u = local_a / (local_K @ v)
local_KTu = local_K.T @ local_u
global_KTu = Allreduce(SUM, local_KTu)
v = b / global_KTu
```

MPI solver dùng global min qua `Allreduce(MIN)` để stabilization giống dense solver.

## 4. Deterministic cost generator

Sequential và MPI runner dùng cùng generator:

```python
generate_cost_matrix(rows, cols, mode, seed)
```

Modes đã hỗ trợ:

- `random`
- `squared_distance_1d`
- `block`

MVP hiện tại để rank 0 tạo full cost matrix rồi scatter row blocks. Đây là cách đơn giản và dễ kiểm chứng correctness. Extension sau có thể cho mỗi rank tự generate local rows theo global row index.

## 5. Runners đã triển khai

### `hello_mpi.py`

In rank, size và hostname để kiểm tra MPI launch:

```bash
mpirun -np 3 python apps/hello_mpi.py
```

### `sequential_sinkhorn_runner.py`

Chạy dense Sinkhorn và lưu JSON:

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

### `mpi_sinkhorn_runner.py`

Chạy MPI row-partitioned Sinkhorn:

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

### `compare_sinkhorn_results.py`

So sánh sequential JSON và MPI JSON:

```bash
python apps/compare_sinkhorn_results.py \
  --sequential results/smoke_seq.json \
  --parallel results/smoke_mpi_np3.json
```

Output gồm:

- Sequential runtime.
- MPI runtime.
- Speedup.
- Efficiency.
- Row/column errors.
- Transport objective.
- Objective difference.

## 6. JSON schema

Sequential và MPI runners dùng cùng schema:

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

## 7. Tests đã thêm

File:

```text
mpi-sinkhorn-step/tests/test_sinkhorn.py
```

Test coverage hiện có:

- Cost generator deterministic với cùng seed.
- Các mode cost matrix trả shape và finite values hợp lệ.
- Dense Sinkhorn khớp marginal constraints.
- Distributed solver trên `MPI.COMM_SELF` khớp dense objective.
- Dense solver reject invalid marginal.

## 8. Verification đã chạy

Trong `mpi-sinkhorn-step/`, đã chạy:

```bash
PYTHONPATH=src python3 -m pytest -q
```

Kết quả:

```text
7 passed
```

Đã chạy compile check:

```bash
PYTHONPATH=src python3 -m compileall -q src apps tests
```

Kết quả: pass.

Đã chạy sequential smoke:

```text
rows = 128
cols = 128
cost_mode = random
seed = 0
epsilon = 0.5
max_iters = 50
```

Kết quả:

```text
row_error = 1.283344e-10
col_error = 1.283343e-10
transport_objective = 0.347356000544
```

Đã chạy MPI smoke local:

```bash
PYTHONPATH=src mpirun -np 3 python3 apps/mpi_sinkhorn_runner.py ...
```

Kết quả compare gần nhất:

```text
Sequential runtime: 0.001404 sec
MPI runtime: 0.000718 sec
Speedup: 1.9561x
Efficiency: 0.6520
Sequential row error: 1.283344e-10
Sequential col error: 1.283343e-10
MPI row error: 1.283344e-10
MPI col error: 1.283344e-10
Sequential objective: 0.347356000544
MPI objective: 0.347356000544
Objective difference: 0.000000e+00
```

Đã chạy thêm local `np=6` với `--oversubscribe` trước đó để xác nhận logic nhiều rank hoạt động. Cluster thật với `--hostfile hosts` chưa chạy vì môi trường hiện tại không có 3 máy/hostfile.

## 9. Ghi chú môi trường

`pip install -e .` bị chặn trên system Python do PEP 668. Thử tạo venv thì môi trường hiện tại thiếu `python3-venv`.

Vì vậy verification local được chạy bằng:

```bash
PYTHONPATH=src ...
```

README của project con đã ghi rõ nếu Ubuntu thiếu venv thì cài:

```bash
sudo apt install -y python3-venv
```

Trên máy demo/worker, khuyến nghị dùng:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 10. Những việc chưa làm

Chưa triển khai optimizer application:

- `objectives.py`
- `directions.py`
- `optimizer.py`
- `sequential_runner.py`
- `mpi_multiseed_runner.py`
- `mpi_distributed_probe_runner.py`
- plotting/report automation

Lý do: theo feedback và plan mới, solver-only MVP phải pass trước khi mở rộng sang optimizer.

Chưa chạy cluster 3 máy thật:

- Cần `hosts` file.
- Cần cùng environment/package trên các worker.
- Cần kiểm tra SSH passwordless và `mpirun hostname`.

## 11. Bước tiếp theo

1. Tạo/cập nhật `hosts` cho 3 máy thật.
2. Cài OpenMPI, Python env và package trên mọi worker.
3. Chạy:

   ```bash
   mpirun -np 6 --hostfile hosts python apps/hello_mpi.py
   ```

4. Chạy cluster smoke:

   ```bash
   mpirun -np 6 --hostfile hosts python apps/mpi_sinkhorn_runner.py \
     --rows 128 \
     --cols 128 \
     --cost-mode random \
     --seed 0 \
     --epsilon 0.5 \
     --max-iters 50 \
     --output results/smoke_cluster_np6.json
   ```

5. Chạy benchmark demo/report với matrix lớn hơn.
6. Sau khi cluster MVP ổn định, triển khai optimizer application.
