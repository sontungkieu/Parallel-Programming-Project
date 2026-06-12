# Báo Cáo Khảo Sát Topic MPI Cluster

Ngày khảo sát: 2026-05-12

## 1. Bối Cảnh Đề Bài

Môn học yêu cầu nhóm 4 người thiết lập một MPI cluster gồm ít nhất 3 máy vật lý. Có thể dùng 3 máy Windows/macOS, mỗi máy cài 1 Ubuntu VM với Bridge Adapter để các máy Ubuntu cùng nằm trong một mạng LAN. Demo được phép chạy offline. Nhóm phải nộp report và chương trình tối thiểu khoảng 1000 dòng code cho nhóm 4 người.

Tiêu chí chấm chính:

- Chủ đề có độ thú vị và độ khó phù hợp.
- Cách bài toán được song song hóa.
- Demo chạy được trên MPI cluster.
- Report trình bày rõ ràng.
- Thành viên hiểu code của nhóm.

Mục tiêu khảo sát là duyệt các repo công khai tại <https://github.com/anindex> để chọn hướng triển khai phù hợp.

## 2. Tiêu Chí Đánh Giá Repo

Các repo được đánh giá theo các tiêu chí sau:

- Phù hợp với MPI trên nhiều máy CPU, không phụ thuộc quá nặng vào GPU.
- Có bài toán đủ khó để đạt điểm tốt, không chỉ là demo toy.
- Có khả năng demo offline trên 3 Ubuntu VM.
- Có thể chia việc hợp lý cho 4 thành viên.
- Có cơ sở lý thuyết đủ rõ để viết report.
- Có thể đo speedup, efficiency, scalability, correctness.
- Tránh dependency quá nặng như ROS/Gazebo/Isaac/large GPU stack nếu không thật sự cần.

## 3. Nhóm Repo Đáng Chú Ý

### 3.1. Graph Search Và Path Planning

Repo liên quan:

- <https://github.com/anindex/aac>
- <https://github.com/anindex/PathPlanningSimulation>
- <https://github.com/anindex/Processing-path_planning>
- <https://github.com/anindex/PathPlanning-ROS-Rviz>

Nhận xét:

- Đây là nhóm phù hợp nhất với MPI cluster CPU.
- Bài toán tìm đường trên graph/grid/road network có thể song song hóa tự nhiên.
- Dễ tạo demo offline: nhập bản đồ hoặc graph lớn, chạy Dijkstra/A*/ALT, so sánh bản tuần tự và MPI.
- Dễ viết report vì có lý thuyết rõ: shortest path, heuristic admissibility, landmark-based A*, distributed workload, speedup.

Điểm mạnh:

- Ít dependency.
- Chạy tốt trên VM.
- Dễ chia việc.
- Dễ chứng minh correctness bằng path cost.
- Dễ đo hiệu năng bằng thời gian chạy, số node expanded, speedup.

Rủi ro:

- Nếu chỉ làm A* cơ bản thì độ khó chưa cao.
- Cần thêm yếu tố nâng cấp như ALT landmark, batch queries, distributed preprocessing hoặc distributed graph partitioning.

### 3.2. Motion Planning Và Optimal Transport

Repo liên quan:

- <https://github.com/anindex/gtmp>
- <https://github.com/anindex/mpot>
- <https://github.com/anindex/motion_planning_baselines>
- <https://github.com/anindex/stoch_gpmp>
- <https://github.com/anindex/torch_robotics>
- <https://github.com/anindex/kinax>

Nhận xét:

- Chủ đề rất hay: motion planning, trajectory optimization, collision checking.
- `gtmp` và `mpot` có tính song song cao, nhưng repo gốc thiên về JAX/PyTorch/GPU vectorization.
- Nếu dùng trực tiếp repo gốc trên VM CPU sẽ khó ổn định và có thể không thể hiện đúng MPI.

Hướng phù hợp nếu chọn nhóm này:

- Không copy nguyên repo GPU.
- Tự viết bản MPI đơn giản hóa: nhiều rank cùng sample trajectory, check collision, tính cost, chọn trajectory tốt nhất bằng `MPI_Reduce`.
- Demo trên môi trường 2D/3D đơn giản thay vì robot arm nặng.

Điểm mạnh:

- Chủ đề ấn tượng.
- Dễ trực quan hóa đường đi.
- Có thể nâng độ khó bằng RRT/PRM/MPPI hoặc multi-start trajectory optimization.

Rủi ro:

- Nếu làm quá tham vọng sẽ khó hoàn thành.
- Collision checking và visualization có thể tốn thời gian.

### 3.3. Sinkhorn / Optimal Transport Solver

Repo liên quan:

- <https://github.com/anindex/ssax>
- <https://github.com/anindex/torch_sinkhorn>
- <https://github.com/anindex/fgw>

Nhận xét:

- Đây là nhóm có nền toán tốt: optimal transport, Sinkhorn iteration, matrix computation.
- Có thể song song hóa theo block matrix hoặc batch problem.
- Phù hợp để đo hiệu năng tính toán.

Điểm mạnh:

- Thuật toán có độ khó tốt.
- MPI phù hợp cho chia ma trận, broadcast/reduce vector scaling.
- Có thể viết report toán học khá mạnh.

Rủi ro:

- Demo có thể khô, khó hấp dẫn nếu không có visualization.
- Cần hiểu numerical stability, convergence, regularization.

### 3.4. Numerical Physics / Monte Carlo Simulation

Repo liên quan:

- <https://github.com/anindex/penrose_process>
- <https://github.com/anindex/warpax>

Nhận xét:

- Chủ đề lạ và có thể gây ấn tượng: mô phỏng vật lý, parameter sweep, Monte Carlo.
- Song song hóa dễ làm bằng cách chia tập tham số hoặc chia trajectory simulation cho nhiều rank.

Điểm mạnh:

- Rất phù hợp với batch simulation.
- Demo có thể tạo biểu đồ kết quả trước và chạy live một phần.

Rủi ro:

- Nếu chỉ chia parameter sweep đơn giản, giảng viên có thể xem là song song hóa chưa sâu.
- Lý thuyết vật lý có thể khó giải thích trong report.

### 3.5. Robotics / Localization / ROS

Repo liên quan:

- <https://github.com/anindex/krp_localization>
- <https://github.com/anindex/lgp>
- <https://github.com/anindex/navigation-processing-base>
- <https://github.com/anindex/youbot_navigation>

Nhận xét:

- Chủ đề robotics hấp dẫn nhưng dependency thường nặng.
- ROS, Gazebo, dataset và visualization có thể làm demo MPI cluster thiếu ổn định.

Kết luận:

- Không nên chọn làm hướng chính nếu thời gian hạn chế.
- Có thể dùng ý tưởng localization/path planning, nhưng nên viết lại bản MPI nhẹ.

## 4. Xếp Hạng Hướng Triển Khai

| Hạng | Hướng | Repo tham khảo | Mức phù hợp |
|---|---|---|---|
| 1 | Distributed Landmark-Aided A* Search | `aac`, `PathPlanningSimulation` | Rất phù hợp |
| 2 | MPI Sampling-Based Motion Planning | `gtmp`, `mpot`, `motion_planning_baselines` | Phù hợp nếu scope gọn |
| 3 | MPI Sinkhorn / Optimal Transport | `ssax`, `torch_sinkhorn`, `fgw` | Phù hợp về thuật toán, demo cần chăm |
| 4 | MPI Monte Carlo Physics Simulation | `penrose_process`, `warpax` | Hay nhưng dễ thành parameter sweep |
| 5 | MPI Robotics Localization | `krp_localization`, `lgp` | Không khuyến nghị do setup nặng |

## 5. Đề Xuất Chốt

Đề xuất chọn:

**Distributed Landmark-Aided A* Search on MPI Cluster**

Tên tiếng Việt:

**Tìm Đường Song Song Trên Cụm MPI Bằng A* Và Landmark Heuristic**

Lý do chọn:

- Chạy tốt trên 3 Ubuntu VM, không cần GPU.
- Có thể demo offline chắc chắn.
- Có thể tăng độ khó bằng ALT landmark, graph lớn, batch routing và distributed preprocessing.
- Có thể viết đủ code cho nhóm 4 người.
- Có kết quả định lượng rõ: path cost, node expansion, runtime, speedup, efficiency.
- Dễ chia thành các module độc lập.

Lưu ý về `aac`: repo này có nội dung liên quan landmark heuristic và A*, nhưng phần license ghi code chỉ dành cho academic peer-review/reproducibility và không cấp quyền derivative/commercial. Vì vậy nhóm chỉ nên dùng repo này làm nguồn ý tưởng học thuật, không copy code trực tiếp.

## 6. Thiết Kế Thuật Toán Đề Xuất

### 6.1. Input

Hệ thống nhận một trong các dạng input:

- Grid map 2D có obstacle.
- Weighted graph sinh ngẫu nhiên.
- Road-like graph tự tạo bằng lưới lớn có trọng số.

Mỗi truy vấn gồm:

- Start node.
- Goal node.
- Thuật toán cần chạy: Dijkstra, A*, ALT, MPI-ALT.

### 6.2. Baseline Tuần Tự

Cài đặt các baseline:

- Dijkstra.
- A* với heuristic Manhattan/Euclidean.
- ALT A* với landmark heuristic.

ALT heuristic:

```text
h(v, t) = max over landmarks L of |dist(L, t) - dist(L, v)|
```

Trong đó:

- `v` là node hiện tại.
- `t` là goal.
- `dist(L, x)` là khoảng cách ngắn nhất từ landmark `L` đến node `x`.

Heuristic này admissible nhờ triangle inequality, nên A* vẫn trả về đường đi tối ưu.

### 6.3. MPI Parallelization

Có ba mức song song hóa:

1. **Distributed landmark preprocessing**
   - Chia danh sách landmarks cho các MPI rank.
   - Mỗi rank chạy Dijkstra từ landmark của mình.
   - Rank 0 gather bảng khoảng cách hoặc lưu phân tán tùy mode.

2. **Parallel batch queries**
   - Chia nhiều cặp start-goal cho các rank.
   - Mỗi rank chạy A*/ALT cho tập query của mình.
   - Rank 0 tổng hợp runtime, path cost, số node expanded.

3. **Optional advanced mode: distributed search**
   - Chia graph theo partition.
   - Mỗi rank quản lý frontier cục bộ.
   - Dùng message passing để trao đổi boundary nodes.
   - Mode này khó hơn, có thể làm nếu còn thời gian.

### 6.4. Metrics

Các chỉ số cần report:

- Total runtime.
- Preprocessing time.
- Query time trung bình.
- Số node expanded.
- Path cost.
- Speedup: `T1 / Tp`.
- Efficiency: `Speedup / p`.
- Correctness: path cost của MPI bằng baseline tuần tự.

## 7. Demo Đề Xuất

Demo offline trên 3 máy Ubuntu VM:

```bash
mpirun -np 3 --hostfile hosts.txt ./mpi_path_planner \
  --map maps/large_grid.txt \
  --landmarks 32 \
  --queries data/queries.txt \
  --algorithm mpi-alt
```

Kịch bản demo:

1. Ping/SSH giữa 3 Ubuntu VM để chứng minh cluster hoạt động.
2. Chạy `mpirun hostname` để chứng minh MPI chạy trên nhiều máy.
3. Chạy baseline tuần tự trên 1 process.
4. Chạy MPI với 3 process hoặc nhiều process hơn nếu mỗi máy có nhiều core.
5. In bảng kết quả:
   - Runtime.
   - Speedup.
   - Node expanded.
   - Path cost check.
6. Mở visualization kết quả đường đi trên grid.

## 8. Phân Chia Công Việc 4 Người

| Thành viên | Công việc chính |
|---|---|
| Người 1 | Setup MPI cluster, hostfile, SSH, script chạy demo |
| Người 2 | Cài graph/grid parser, Dijkstra, A* baseline |
| Người 3 | Cài ALT landmark heuristic và preprocessing |
| Người 4 | Cài MPI distribution, benchmark, visualization, report |

Tất cả thành viên cần hiểu pipeline tổng thể:

```text
map/graph -> landmark preprocessing -> A*/ALT query -> MPI aggregation -> benchmark + visualization
```

## 9. Cấu Trúc Repo Nên Có

```text
.
├── src/
│   ├── graph/
│   ├── algorithms/
│   ├── mpi/
│   ├── benchmark/
│   └── visualization/
├── maps/
├── data/
├── scripts/
├── results/
├── report/
└── README.md
```

Nếu viết bằng C/C++:

- Dùng OpenMPI hoặc MPICH.
- Build bằng CMake hoặc Makefile.

Nếu viết bằng Python:

- Dùng `mpi4py`.
- Cần đảm bảo dependency nhẹ và cài được trên cả 3 VM.

Khuyến nghị kỹ thuật:

- Nếu nhóm tự tin C/C++: chọn C++ + MPI để đúng chất môn học hơn.
- Nếu nhóm cần demo nhanh: chọn Python + mpi4py, nhưng cần tối ưu code đủ tốt và trình bày rõ MPI communication.

## 10. Rủi Ro Và Cách Giảm Rủi Ro

| Rủi ro | Cách xử lý |
|---|---|
| MPI cluster lỗi mạng | Chuẩn bị script test SSH, `mpirun hostname`, static IP/hosts |
| Demo quá chậm | Có nhiều size map: small, medium, large |
| Speedup không đẹp | Tách preprocessing và batch query để tăng workload |
| A* cơ bản bị xem là dễ | Thêm ALT landmarks, batch routing, benchmark nhiều query |
| Thành viên không hiểu code | Chia module rõ và yêu cầu mỗi người demo phần mình |
| Visualization lỗi | Luôn có output text/table làm fallback |

## 11. Kết Luận

Trong các repo đã khảo sát, hướng **Distributed Landmark-Aided A* Search** là lựa chọn cân bằng nhất giữa độ khó, khả năng demo, khả năng chạy trên MPI cluster thật và khả năng viết report tốt.

Nhóm có thể lấy cảm hứng từ các repo của `anindex` về A*/ALT/path planning/motion planning, nhưng nên tự triển khai lại hệ thống MPI để thể hiện rõ năng lực parallel programming.

Ưu tiên tiếp theo:

1. Chốt topic với nhóm.
2. Tạo skeleton repo.
3. Cài baseline Dijkstra/A*.
4. Thêm ALT heuristic.
5. Thêm MPI preprocessing và batch query.
6. Viết benchmark + visualization.
7. Chuẩn bị report và demo offline trên 3 máy.
