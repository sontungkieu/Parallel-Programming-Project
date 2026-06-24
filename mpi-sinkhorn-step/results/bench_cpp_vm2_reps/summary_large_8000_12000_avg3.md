# C++ VM 2-node large benchmark, 3-run averages

Command parameters: random cost, seed 0, epsilon 0.5, max_iters 200, tol 1e-6, check_every 10.

| size | seq avg (s) | MPI np=2 avg (s) | speedup | efficiency | max objective diff | iterations |
|---:|---:|---:|---:|---:|---:|---:|
| 8000 | 2.124563 | 1.465780 | 1.4494x | 0.7247 | 1.052e-13 | 10 |
| 9000 | 2.723570 | 1.929634 | 1.4114x | 0.7057 | 1.088e-13 | 10 |
| 10000 | 3.387961 | 2.166852 | 1.5635x | 0.7818 | 8.316e-14 | 10 |
| 11000 | 3.992483 | 2.805868 | 1.4229x | 0.7115 | 2.102e-13 | 10 |
| 12000 | 5.692592 | 3.259240 | 1.7466x | 0.8733 | 8.810e-14 | 10 |
