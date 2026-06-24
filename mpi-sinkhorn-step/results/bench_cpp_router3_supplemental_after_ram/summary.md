# C++ Sinkhorn Router3 Supplemental Experiments

Cluster: `192.168.1.25/.26/.75` over router Ethernet after RAM upgrade and 6GB swap per node.

This supplement covers the router experiments still missing after the first report suite:
- Extra process counts at `N=8000`: `np=8`; retained `np=12` partial runs are summarized when present.
- `2N` speedup at `N=16000`: sequential plus `np=1,2,4,8`.
- Load-balance extraction at the report workload `N=8000`, `iters=2000`, `np=3`.
- Larger-size probe at `N=14000,16000,18000`, `np=3`.

Note: the retained `np=12` run at `N=8000` is partial because the third repetition made the master VM stop accepting SSH; the table keeps the completed JSON files and exposes `count`.

## Extra Process Counts At N=8000

| np | count | mpi avg s | speedup vs seq8000 | efficiency | avg imbalance % | max compute s | max comm s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3 | 189.597 | 0.0468434x | 0.00585542 | 33.4111 | 1.82151 | 188.665 |
| 12 | 2 | 96.9558 | 0.0916021x | 0.00763351 | 0.208159 | 2.43436 | 96.3444 |

## 2N Speedup At N=16000

| np | count | mpi avg s | speedup vs seq | efficiency | avg imbalance % | max compute s | max comm s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 30.2328 | 1.5293x | 1.5293 | 0 | 30.2314 | 0.000574091 |
| 2 | 1 | 24.4677 | 1.88964x | 0.944818 | 0.0900565 | 21.3959 | 7.67919 |
| 4 | 1 | 13.6649 | 3.38348x | 0.845871 | 0.0267388 | 8.9342 | 5.64827 |
| 8 | 1 | 576.744 | 0.0801657x | 0.0100207 | 0.00723543 | 5.68515 | 573.111 |

## Larger Size Probe

| size | np | runtime s | avg imbalance % | max compute s | max comm s |
|---:|---:|---:|---:|---:|---:|
| 14000 | 3 | 14.4021 | 0.0118601 | 13.3899 | 8.38496 |
| 16000 | 3 | 16.5969 | 0.0207988 | 15.5616 | 6.98701 |
| 18000 | 3 | 20.6886 | 0.0246299 | 19.4567 | 8.51065 |

## Workload Load-Balance Extraction

| rank | local rows | total s | compute s | comm s | overhead s |
|---:|---:|---:|---:|---:|---:|
| 0 | 2667 | 139.208 | 116.224 | 22.9657 | 0.0185206 |
| 1 | 2667 | 139.209 | 85.3807 | 53.8159 | 0.0120134 |
| 2 | 2666 | 139.205 | 68.0776 | 71.1172 | 0.0101601 |

Raw per-run data is in `per_run.csv`.
The extracted workload balance rows are in `workload_balance_by_rank.csv`.
