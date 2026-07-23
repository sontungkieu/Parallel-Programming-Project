# 4-machine Wi-Fi/hotspot C++ MPI, n=12000, reps=3

- Hosts: `10.110.77.186`, `10.110.77.74`, `10.110.77.242`, `10.110.77.195`
- MPI: `np=4`, `--map-by node`, 1 process per machine, `comm-mode=double`
- Workload: `rows=cols=12000`, `max-iters=50`, `tol=0`, `seed=0`, `cost-mode=random`, default contiguous partition
- Sequential baseline reused for rough speedup: previous fixed-50 n=12000 reps 2-4 avg = 18.778839 s

| rep | runtime_sec | wall_clock_sec | speedup_vs_seq_last3 | objective | row_error | col_error | suggested weights |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 25.774130 | 32.12 | 0.729 | 0.099999182361 | 1.200e-08 | 1.200e-08 | `0.5336644834,1.127125018,0.930099085,1.409111414` |
| 2 | 20.459867 | 25.55 | 0.918 | 0.099999182361 | 1.200e-08 | 1.200e-08 | `0.6083170875,1.213762918,0.8378931405,1.340026854` |
| 3 | 19.650477 | 26.20 | 0.956 | 0.099999182361 | 1.200e-08 | 1.200e-08 | `0.5998658274,1.198116785,0.901852663,1.300164724` |

Aggregate:
- runtime avg = 21.961491 s, min = 19.650477 s, max = 25.774130 s, stdev = 3.326550 s
- wall-clock avg = 27.96 s, min = 25.55 s, max = 32.12 s
- rough speedup vs reused sequential last3 avg = 0.855x
- efficiency vs 4 processes = 21.4%

Smoke hostnames:
```text
Warning: Permanently added '10.110.77.242' (ED25519) to the list of known hosts.
Warning: Permanently added '10.110.77.195' (ED25519) to the list of known hosts.
Warning: Permanently added '10.110.77.74' (ED25519) to the list of known hosts.
vmubuntu
quan
Parallel
VMubuntu
```
