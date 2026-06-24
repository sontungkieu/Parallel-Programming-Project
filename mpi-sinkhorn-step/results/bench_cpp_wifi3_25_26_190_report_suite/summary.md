# C++ Sinkhorn bench_cpp_wifi3_25_26_190_report_suite Report Experiment Suite

Cluster: `.25,.26,.190` over `wifi3`. Hostfile uses `2` slots per node.

Experiments covered:
- Baseline size sweep: `np=3`, sizes `2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000`, fixed 50 iterations, 4 reps.
- Process sweep: `np=1 2 3 4 6`, fixed 50 iterations, 3 reps.
- Input-size calibration: sizes `8000 10000 12000`, fixed 50 iterations, 1 rep.
- Workload calibration: `np=3`, iteration counts `200 500 1000 2000`, 1 rep.
- Load balance and compute/communication breakdown from per-rank metrics.

Safety note: wifi3 report suite uses the same 12000 dense-size cap as the router suite to keep memory pressure comparable.

## Baseline Size Sweep

| size | seq avg s | mpi np3 avg s | speedup | efficiency | avg imbalance % | max obj diff |
|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 0.636966 | 0.570606 | 1.1163x | 0.372099 | 0.331756 | 3.234e-15 |
| 3000 | 1.3339 | 1.05016 | 1.27019x | 0.423395 | 0.396367 | 1.874e-15 |
| 4000 | 2.47114 | 1.41275 | 1.74918x | 0.583059 | 0.471138 | 5.385e-15 |
| 5000 | 3.55243 | 2.15936 | 1.64513x | 0.548376 | 0.285271 | 7.869e-15 |
| 6000 | 5.40062 | 2.61078 | 2.06859x | 0.68953 | 0.175844 | 9.270e-15 |
| 7000 | 6.76466 | 3.34253 | 2.02382x | 0.674605 | 0.155098 | 2.255e-14 |
| 8000 | 8.88136 | 4.16984 | 2.1299x | 0.709968 | 0.115555 | 2.963e-14 |
| 9000 | 11.2251 | 5.01064 | 2.24025x | 0.746751 | 0.0751978 | 1.292e-14 |
| 10000 | 13.4114 | 6.85194 | 1.95731x | 0.652437 | 0.0771146 | 3.900e-15 |
| 11000 | 15.5787 | 7.37273 | 2.11302x | 0.70434 | 0.0813858 | 1.550e-14 |
| 12000 | 19.6726 | 8.25547 | 2.38297x | 0.794325 | 0.0526258 | 6.448e-14 |

## Process Sweep

| np | mpi avg s | speedup vs seq | efficiency | avg imbalance % | max compute s | max comm s | max overhead s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.66667 | 0.918761x | 0.918761 | 0 | 9.66457 | 0.00046574 | 0.00163728 |
| 2 | 5.35549 | 1.65837x | 0.829183 | 0.0481883 | 4.40363 | 1.02081 | 0.00068048 |
| 3 | 3.9899 | 2.22596x | 0.741986 | 0.190927 | 2.92141 | 1.54172 | 0.000831599 |
| 4 | 3.43779 | 2.58345x | 0.645862 | 0.512304 | 2.23143 | 1.6661 | 0.000968417 |
| 6 | 3.45042 | 2.57399x | 0.428998 | 0.452468 | 1.62285 | 2.11972 | 0.00113335 |

## Input-Size Calibration

| size | max iters | np | mpi runtime s | avg imbalance % |
|---:|---:|---:|---:|---:|
| 8000 | 50 | 3 | 3.76002 | 0.0591341 |
| 10000 | 50 | 3 | 5.70956 | 0.244336 |
| 12000 | 50 | 3 | 8.95444 | 0.0245445 |

## Workload Calibration

| size | max iters | np | mpi runtime s | avg imbalance % |
|---:|---:|---:|---:|---:|
| 8000 | 200 | 3 | 14.1758 | 0.0514099 |
| 8000 | 500 | 3 | 39.8636 | 0.0107385 |
| 8000 | 1000 | 3 | 73.7464 | 0.00341213 |
| 8000 | 2000 | 3 | 164.394 | 0.00200105 |

## Load Balance Snapshot

| experiment | size | max iters | np | count | runtime s | avg imbalance % | max compute s | max comm s | max overhead s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| load_balance | 8000 | 50 | 3 | 3 | 4.48517 | 0.208688 | 3.35805 | 2.19716 | 0.000778485 |

Raw per-run data is in `per_run.csv`.
Per-rank load-balance rows are in `load_balance_by_rank.csv`.
