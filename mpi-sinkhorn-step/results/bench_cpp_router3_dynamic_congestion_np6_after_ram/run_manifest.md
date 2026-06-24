# Dynamic Router3 Run Manifest

Cluster: `.25/.26/.75`, MPI `np=6`, fixed 50 Sinkhorn iterations, chunk rows `256`.

| variant | partition mode | sizes | reps | weights | purpose |
|---|---|---|---:|---|---|
| seq | sequential | 8000 | 3 | n/a | Fresh sequential baseline on master |
| contiguous | contiguous | 8000 | 3 | n/a | Backward-compatible MPI baseline |
| weighted_equal | weighted-chunk | 8000 | 3 | equal | Chunk scheduler without hetero weighting |
| weighted_manual_211 | weighted-chunk | 8000 | 3 | 2,1,1,1,1,1 | Explicit CLI rank weights coverage |
| weighted_adaptive | weighted-chunk | 8000 | 3 | learned across reps | Adaptive weight feedback coverage |
| runtime_queue | runtime-queue | 2000 4000 | 0 | equal | Experimental queue overhead/correctness smoke |
