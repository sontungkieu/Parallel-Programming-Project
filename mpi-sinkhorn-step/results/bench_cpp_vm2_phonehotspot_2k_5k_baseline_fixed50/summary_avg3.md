# C++ Baseline Benchmark - Phone Hotspot, 2 VMs, Fixed 50 Iterations

- Master: `10.81.215.186`
- Worker: `10.81.215.74`
- MPI mode: `double_baseline`, `np=2`, `--map-by node`
- Sizes: 2000, 3000, 4000, 5000
- Repetitions: 3
- Iterations: fixed 50, `tol=0`, `check_every=50`

| Size | Seq avg (s) | MPI avg (s) | Speedup avg | Efficiency | MPI payload avg (MB) | Obj diff avg |
|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 0.712903 | 0.681481 | 1.0510 | 0.5255 | 0.763 | 1.819e-14 |
| 3000 | 1.407982 | 1.269109 | 1.1218 | 0.5609 | 1.144 | 3.845e-14 |
| 4000 | 2.457496 | 2.192344 | 1.1542 | 0.5771 | 1.526 | 1.101e-14 |
| 5000 | 4.230090 | 2.638800 | 1.6000 | 0.8000 | 1.907 | 9.030e-14 |
