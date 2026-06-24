# C++ Sinkhorn bench_cpp_wifi3_25_26_190_supplemental Supplemental Experiments

Cluster: `.25,.26,.190` over `wifi3` with `3` slots per node; wifi3 run after master restart; hostfile uses three slots per node to allow np=8.

This supplement covers the experiments still missing after the first report suite:
- Extra process counts at `N=8000`: `np=8`.
- `2N` speedup at `N=16000`: sequential plus `np=1 2 4 8`.
- Load-balance extraction at the report workload `N=8000`, `iters=2000`, `np=3`.
- Larger-size probe at `N=14000 16000 18000`, `np=3`.

Note: all completed runs are summarized; failures, if any, are listed in failures.txt.

## Extra Process Counts At N=8000

| np | count | mpi avg s | speedup vs seq8000 | efficiency | avg imbalance % | max compute s | max comm s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3 | 6.32121 | 1.40501x | 0.175626 | 0.357022 | 3.76518 | 5.12026 |

## 2N Speedup At N=16000

| np | count | mpi avg s | speedup vs seq | efficiency | avg imbalance % | max compute s | max comm s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 36.5253 | 1.43698x | 1.43698 | 0 | 36.5205 | 0.000700621 |
| 2 | 1 | 33.7498 | 1.55515x | 0.777576 | 0.0143008 | 29.3109 | 12.8036 |
| 4 | 1 | 14.6415 | 3.58475x | 0.896187 | 0.123528 | 12.0344 | 7.27999 |
| 8 | 1 | 16.7208 | 3.13897x | 0.392371 | 0.147843 | 8.11303 | 12.4976 |

## Larger Size Probe

| size | np | runtime s | avg imbalance % | max compute s | max comm s |
|---:|---:|---:|---:|---:|---:|
| 14000 | 3 | 12.3478 | 0.0871651 | 11.0464 | 5.20693 |
| 16000 | 3 | 16.6515 | 0.0124782 | 13.8773 | 6.46043 |
| 18000 | 3 | 20.9624 | 0.0774842 | 17.8508 | 8.36847 |

## Workload Load-Balance Extraction

| rank | local rows | total s | compute s | comm s | overhead s |
|---:|---:|---:|---:|---:|---:|
| 0 | 2667 | 164.39 | 125.182 | 39.1892 | 0.0191654 |
| 1 | 2667 | 164.394 | 103.973 | 60.4055 | 0.0149273 |
| 2 | 2666 | 164.393 | 83.0847 | 81.2827 | 0.0254255 |

Raw per-run data is in `per_run.csv`.
The extracted workload balance rows are in `workload_balance_by_rank.csv`.
