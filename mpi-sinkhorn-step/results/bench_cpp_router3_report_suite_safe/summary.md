# C++ Sinkhorn Router3 Report Experiment Suite

Cluster: `192.168.1.25/.26/.75` over router Ethernet. Hostfile uses the safe two-slot-per-node profile.

Experiments covered:
- Baseline size sweep: `np=3`, sizes `2000..12000`, fixed 50 iterations, 4 reps.
- Process sweep: several `np` values up to the safe slots limit, fixed 50 iterations, 3 reps.
- Input-size calibration: safe dense sizes up to `12000`, fixed 50 iterations, 1 rep.
- Workload calibration: `np=3`, fixed size with longer iteration counts, 1 rep.
- Load balance and compute/communication breakdown from per-rank metrics.

Safety note: larger dense-size and 12-process attempts were stopped after node `.75` reported a kernel soft lockup. This suite caps memory pressure and uses longer iteration counts for the 2-3 minute workload target.

## Baseline Size Sweep

| size | seq avg s | mpi np3 avg s | speedup | efficiency | avg imbalance % | max obj diff |
|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 0.636966 | 0.64895 | 0.981533x | 0.327178 | 1.33262 | 3.234e-15 |
| 3000 | 1.3339 | 1.01798 | 1.31034x | 0.436781 | 0.309365 | 1.874e-15 |
| 4000 | 2.47114 | 1.45141 | 1.70258x | 0.567525 | 0.250058 | 5.385e-15 |
| 5000 | 3.55243 | 1.91979 | 1.85042x | 0.616808 | 0.139915 | 7.869e-15 |
| 6000 | 5.40062 | 2.51674 | 2.14588x | 0.715294 | 0.0858178 | 9.270e-15 |
| 7000 | 6.76466 | 3.58734 | 1.8857x | 0.628567 | 0.102194 | 2.255e-14 |
| 8000 | 8.88136 | 4.53933 | 1.95653x | 0.652178 | 0.0571388 | 2.963e-14 |
| 9000 | 11.2251 | 5.85563 | 1.91697x | 0.638991 | 0.0295534 | 1.292e-14 |
| 10000 | 13.4114 | 7.60183 | 1.76423x | 0.588077 | 0.0574794 | 3.900e-15 |
| 11000 | 15.5787 | 8.20865 | 1.89784x | 0.632615 | 0.0144139 | 1.550e-14 |
| 12000 | 19.6726 | 9.13863 | 2.15268x | 0.717561 | 0.0375727 | 6.448e-14 |

## Process Sweep

| np | mpi avg s | speedup vs seq | efficiency | avg imbalance % | max compute s | max comm s | max overhead s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.1052 | 0.878894x | 0.878894 | 0 | 10.1041 | 0.000384133 | 0.000640281 |
| 2 | 6.82146 | 1.30197x | 0.650987 | 0.03824 | 6.03563 | 1.84212 | 0.00108205 |
| 3 | 4.58633 | 1.93649x | 0.645495 | 0.0566401 | 3.94414 | 2.3933 | 0.000774148 |
| 4 | 4.14948 | 2.14036x | 0.535089 | 0.15077 | 3.04601 | 2.52303 | 0.00078577 |
| 6 | 99.2255 | 0.0895068x | 0.0149178 | 0.143182 | 1.96407 | 98.0342 | 0.000925466 |

## Input-Size Calibration

| size | max iters | np | mpi runtime s | avg imbalance % |
|---:|---:|---:|---:|---:|
| 8000 | 50 | 3 | 4.61667 | 0.102829 |
| 10000 | 50 | 3 | 8.24162 | 0.0632706 |
| 12000 | 50 | 3 | 11.2458 | 0.0804827 |

## Workload Calibration

| size | max iters | np | mpi runtime s | avg imbalance % |
|---:|---:|---:|---:|---:|
| 8000 | 200 | 3 | 16.5055 | 0.028812 |
| 8000 | 500 | 3 | 43.1399 | 0.00350065 |
| 8000 | 1000 | 3 | 72.9962 | 0.00252331 |
| 8000 | 2000 | 3 | 139.209 | 0.00266036 |

## Load Balance Snapshot

| experiment | size | max iters | np | count | runtime s | avg imbalance % | max compute s | max comm s | max overhead s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| load_balance | 8000 | 50 | 6 | 3 | 64.7333 | 1.70597 | 2.01869 | 63.4943 | 0.0019779 |

Raw per-run data is in `per_run.csv`.
Per-rank load-balance rows are in `load_balance_by_rank.csv`.
