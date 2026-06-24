# Router3 Supplemental Manifest

| group | config | purpose |
|---|---|---|
| process_extra | N=8000, np=8, reps=3, iters=50 | Add higher process counts missing from the first process sweep |
| speedup2n | N=16000, seq + np=1 2 4 8, reps=1, iters=50 | Speedup at 2N |
| workload_balance_extract | N=8000, np=3, iters=2000 | Extract per-rank load-balance table from previous suite |
| size_probe | N=14000 16000 18000, np=3, reps=1, iters=50 | Check larger dense sizes after RAM upgrade |
