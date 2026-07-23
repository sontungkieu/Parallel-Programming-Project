# Plugged-in rerun: 4-machine hotspot fixed-50 plus master sequential

- Hosts: `10.110.77.186`, `10.110.77.74`, `10.110.77.242`, `10.110.77.195`
- Workload: `N=12000`, `max-iters=50`, `seed=0`, `epsilon=0.1`, `comm-mode=double`
- Note: user reported machine was plugged in before this run.

| kind | rep | runtime_sec | wall_clock_sec | speedup_vs_seq_avg | objective |
|---|---:|---:|---:|---:|---:|
| seq_master | 1 | 42.605688 | 50.58 | 1.000 | 0.099999182361 |
| seq_master | 2 | 29.069422 | 31.37 | 1.000 | 0.099999182361 |
| seq_master | 3 | 30.235317 | 32.55 | 1.000 | 0.099999182361 |
| mpi4 | 1 | 14.398397 | 28.26 | 2.359 | 0.099999182361 |
| mpi4 | 2 | 14.574736 | 18.32 | 2.331 | 0.099999182361 |
| mpi4 | 3 | 15.639178 | 17.95 | 2.172 | 0.099999182361 |

Aggregate:
- master sequential avg3 = `33.970142 s`
- MPI 4-machine avg3 = `14.870770 s`
- speedup vs current master avg3 = `2.284x`
- efficiency vs 4 processes = `0.571`
- MPI min/max = `14.398397 / 15.639178 s`

MPI suggested weights by rep:
- rep 1: `0.6215809375,1.095844416,0.8196113925,1.462963254`
- rep 2: `0.6435622093,1.097760816,0.822926847,1.435750128`
- rep 3: `0.6358747523,1.086533122,0.8209257886,1.456666337`
