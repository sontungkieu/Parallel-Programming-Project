# Dynamic Router3 Run Manifest

Cluster: `.25/.26/.75`, MPI `np=3`, fixed 50 Sinkhorn iterations, chunk rows `256`.

| variant | partition mode | sizes | reps | weights | purpose |
|---|---|---|---:|---|---|
| seq | sequential | 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 | 4 | n/a | Fresh sequential baseline on master |
| contiguous | contiguous | 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 | 4 | n/a | Backward-compatible MPI baseline |
| weighted_equal | weighted-chunk | 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 | 4 | equal | Chunk scheduler without hetero weighting |
| weighted_manual_211 | weighted-chunk | 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 | 4 | 2,1,1 | Explicit CLI rank weights coverage |
| weighted_adaptive | weighted-chunk | 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 | 4 | learned across reps | Adaptive weight feedback coverage |
| runtime_queue | runtime-queue | 2000 4000 | 2 | equal | Experimental queue overhead/correctness smoke |
