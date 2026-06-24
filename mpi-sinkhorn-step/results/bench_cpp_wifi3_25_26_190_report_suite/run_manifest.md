# bench_cpp_wifi3_25_26_190_report_suite Manifest

| experiment | config | purpose |
|---|---|---|
| baseline | np=3, sizes=2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000, reps=4, iters=50 | Runtime vs input size on wifi3 |
| process_sweep | size=8000, np=1 2 3 4 6, reps=3, iters=50 | Speedup/efficiency vs process count |
| input_calibration | np=3, sizes=8000 10000 12000, reps=1, iters=50 | Larger-size calibration within RAM limits |
| workload_calibration | np=3, size=8000, iters=200 500 1000 2000, reps=1 | 2-3 minute workload calibration without unsafe dense RAM growth |
| load_balance | np=3, size=8000, reps=3, iters=50 | Per-rank granularity/load-balance snapshot |
